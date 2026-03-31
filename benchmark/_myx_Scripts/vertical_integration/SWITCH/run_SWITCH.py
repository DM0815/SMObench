#!/usr/bin/env python3
import os
import re
import sys
import time
import argparse
from pathlib import Path
from itertools import chain

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import networkx as nx
import matplotlib.pyplot as plt

# Project paths -----------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

switch_path = os.path.join(project_root, "Methods/SWITCH")
sys.path.append(switch_path)

import switch as sw  # noqa: E402
from Utils.SMOBench_clustering import universal_clustering  # noqa: E402


# Utility functions -------------------------------------------------------------
def parse_dataset_info(args: argparse.Namespace):
    """Infer dataset / subset names either from explicit argument or RNA path."""
    if getattr(args, "dataset", None):
        parts = args.dataset.strip("/").split("/")
        if len(parts) == 2:
            return parts[0], parts[1]
        if len(parts) == 1:
            return parts[0], "Unknown"

    match = re.search(r'SMOBench[_-]Data/([^/]+)/([^/]+)/([^/]+)/adata_RNA\.h5ad', args.RNA_path)
    if match:
        return match.group(2), match.group(3)
    return "Unknown", "Unknown"


def ensure_spatial_coords(adata: sc.AnnData, label: str, fallback: sc.AnnData = None):
    """Guarantee spatial coordinates exist, generating from UMAP if missing."""
    if "spatial" in adata.obsm:
        return
    if fallback is not None and "spatial" in fallback.obsm:
        print(f"Copying spatial coordinates for {label} from reference dataset.")
        adata.obsm["spatial"] = fallback.obsm["spatial"].copy()
        return

    print(f"Warning: {label} lacks spatial coordinates. Generating pseudo coordinates with UMAP.")
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    adata.obsm["spatial"] = adata.obsm["X_umap"].copy()


def annotate_genes_if_needed(rna: sc.AnnData, gtf_path: str, gtf_by: str, required: bool):
    """Run gene annotation when genomic coordinates are missing (required for ATAC)."""
    if not required:
        return

    required_cols = {"chrom", "chromStart", "chromEnd"}
    if required_cols.issubset(rna.var.columns):
        return
    if not gtf_path:
        missing = ", ".join(sorted(required_cols - set(rna.var.columns)))
        raise ValueError(
            f"RNA AnnData lacks genomic annotation columns ({missing}). "
            "Please provide --gtf_path to enable SWITCH guidance construction."
        )
    print(f"Annotating RNA genes using GTF: {gtf_path}")
    sw.pp.get_gene_annotation(rna, gtf=gtf_path, gtf_by=gtf_by, drop_na=True)


def preprocess_rna(rna: sc.AnnData, hv_top: int):
    """Standard RNA preprocessing following SWITCH tutorial."""
    rna.layers["counts"] = rna.X.copy()
    sc.pp.highly_variable_genes(rna, n_top_genes=min(hv_top, rna.n_vars), flavor="seurat_v3")
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.scale(rna)


def preprocess_atac(atac: sc.AnnData, hv_top: int, min_cells: int = 20):
    """Prepare ATAC AnnData by parsing peaks and selecting informative features."""
    print("Preprocessing ATAC modality ...")
    split = atac.var_names.str.split(r"[:-]")
    atac.var["chrom"] = split.str[0]
    atac.var["chromStart"] = split.str[1].astype(int)
    atac.var["chromEnd"] = split.str[2].astype(int)
    sc.pp.filter_genes(atac, min_cells=min_cells)
    # Save raw counts BEFORE normalization — SWITCH encoder does its own (x*1e4/l).log1p()
    from scipy import sparse
    X_raw = atac.X.toarray() if sparse.issparse(atac.X) else np.asarray(atac.X)
    X_raw = np.nan_to_num(X_raw, nan=0.0).astype(np.float32)
    X_raw = np.clip(X_raw, 0, None)
    # Detect if data is pre-normalized (non-integer) — flag for prob_model selection
    is_integer = np.allclose(X_raw, np.round(X_raw))
    if not is_integer:
        # Convert pre-normalized ATAC to pseudo-integer counts for NB compatibility.
        # Normal prob_model causes NaN during cycle-consistency (decoder generates
        # extreme fake values → encoder normalize → NaN propagation).
        # Strategy: normalize_total(1e4) → round to nearest int → use NB prob_model.
        print(f"  ATAC data appears pre-normalized (non-integer). Converting to pseudo-counts for NB.")
        from scipy import sparse as _sp2
        # First normalize to 1e4 total per cell
        _X = atac.X.toarray() if _sp2.issparse(atac.X) else np.asarray(atac.X, dtype=np.float64)
        _X = np.clip(_X, 0, None)
        _row_sums = _X.sum(axis=1, keepdims=True)
        _row_sums[_row_sums == 0] = 1.0
        _X = _X / _row_sums * 1e4
        # Round to integer pseudo-counts, ensure at least 0
        _X = np.round(_X).astype(np.float32)
        _X = np.clip(_X, 0, None)
        atac.layers["counts"] = _X
        is_integer = True  # now it IS integer
        print(f"  Pseudo-counts: range [{_X.min():.0f}, {_X.max():.0f}], "
              f"lib_size range [{_X.sum(1).min():.0f}, {_X.sum(1).max():.0f}]")
    else:
        atac.layers["counts"] = X_raw
    atac.uns["_is_integer_counts"] = is_integer

    sc.pp.log1p(atac)
    sc.pp.highly_variable_genes(atac, n_top_genes=min(hv_top, atac.n_vars), flavor="seurat_v3")
    sc.pp.scale(atac)


def preprocess_adt(adt: sc.AnnData):
    """Prepare ADT for SWITCH — save raw counts, mark all as HVG.

    SWITCH's encoder internally normalizes via (x * 1e4 / libsize).log1p(),
    so we must feed it raw (non-negative) counts, NOT scaled data.
    Adds pseudocount of 1 to prevent zero library sizes (division by zero).
    Replaces both X and layers["counts"] for robustness.
    """
    print("Preprocessing ADT modality ...")
    from scipy import sparse
    X = adt.X
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    # Handle NaN values BEFORE clip (np.clip preserves NaN!)
    n_nan = np.isnan(X).sum()
    if n_nan > 0:
        print(f"  WARNING: ADT data contains {n_nan} NaN values — replacing with 0.")
        X = np.nan_to_num(X, nan=0.0)
    X = np.clip(X, 0, None) + 1  # pseudocount to prevent zero library size
    adt.layers["counts"] = X.copy()
    adt.X = X  # Also replace X for robustness
    print(f"  ADT shape: {adt.shape}, lib_size range: [{X.sum(axis=1).min():.1f}, {X.sum(axis=1).max():.1f}]")
    adt.var["highly_variable"] = True


def _strip_protein_prefix(name: str) -> str:
    """Strip species prefixes from ADT protein names.

    Handles patterns like 'Mouse-CD4', 'mouse_CD8a', 'Ms-Hu-CD11b',
    'mouse_rat_human_CD27', 'Rat_IgG2b', 'Human-CD8', etc.
    """
    import re
    # Remove leading species tokens (Mouse, Human, Rat, Ms, Hu) separated by - or _
    stripped = re.sub(
        r'^(?:(?:mouse|human|rat|ms|hu)[_-])+',
        '', name, flags=re.IGNORECASE,
    )
    return stripped if stripped else name


def build_protein_guidance(rna: sc.AnnData, protein: sc.AnnData) -> nx.MultiDiGraph:
    """Construct a simple RNA-protein guidance graph based on shared gene symbols."""
    print("Building RNA-protein guidance graph ...")
    graph = nx.MultiDiGraph()
    rna_names = {name.upper(): name for name in rna.var_names}
    protein_names = protein.var_names

    nodes = set(rna.var_names).union(protein_names)
    for item in nodes:
        graph.add_edge(item, item, weight=1.0, sign=1, type="loop")

    matched = 0
    for prot in protein_names:
        # Try direct match first, then stripped match
        gene = rna_names.get(prot.upper())
        if gene is None:
            stripped = _strip_protein_prefix(prot)
            gene = rna_names.get(stripped.upper())
        if gene is None:
            continue
        graph.add_edge(gene, prot, weight=1.0, sign=1, type="fwd")
        graph.add_edge(prot, gene, weight=1.0, sign=1, type="rev")
        matched += 1

    if matched == 0:
        raise ValueError("No overlapping features between RNA genes and ADT markers; guidance graph is empty.")
    print(f"Matched {matched} markers between RNA and ADT.")
    return graph


def sanitize_var_columns(adata: sc.AnnData):
    """Ensure categorical columns used by SWITCH are stored as plain strings."""
    for col in ["gene_ids", "feature_types", "genome"]:
        if col in adata.var.columns:
            adata.var[col] = adata.var[col].astype(str)


def format_duration(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs}s"


# Main pipeline -----------------------------------------------------------------
def main(args: argparse.Namespace):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Running SWITCH on device: {device}")

    # Load datasets ----------------------------------------------------------------
    rna = sc.read_h5ad(args.RNA_path)
    rna.var_names_make_unique()
    sanitize_var_columns(rna)

    if args.ADT_path and args.ATAC_path:
        raise ValueError("Please provide either --ADT_path or --ATAC_path, not both.")
    if not args.ADT_path and not args.ATAC_path:
        raise ValueError("One of --ADT_path or --ATAC_path must be supplied.")

    if args.ADT_path:
        other = sc.read_h5ad(args.ADT_path)
        modality = "ADT"
    else:
        other = sc.read_h5ad(args.ATAC_path)
        modality = "ATAC"

    other.var_names_make_unique()
    sanitize_var_columns(other)

    # Align cells ------------------------------------------------------------------
    common_obs = rna.obs_names.intersection(other.obs_names)
    if common_obs.empty:
        raise ValueError("No overlapping cells between RNA and the secondary modality.")
    rna = rna[common_obs].copy()
    other = other[common_obs].copy()
    print(f"Aligned {rna.n_obs} common cells.")

    # Spatial coordinates ----------------------------------------------------------
    ensure_spatial_coords(rna, "RNA")
    ensure_spatial_coords(other, modality, fallback=rna)

    # Gene annotation for RNA ------------------------------------------------------
    gtf_path = args.gtf_path.strip()
    annotate_genes_if_needed(rna, gtf_path, args.gtf_by, required=(modality == "ATAC"))

    # Preprocessing ----------------------------------------------------------------
    preprocess_rna(rna, args.rna_hv_genes)

    if modality == "ATAC":
        preprocess_atac(other, args.other_hv_features, args.min_peak_cells)
    else:
        preprocess_adt(other)

    # Guidance graph ---------------------------------------------------------------
    # NOTE: Must build guidance BEFORE setup_data, because
    # rna_anchored_guidance_graph(propagate_highly_variable=True) overwrites
    # other.var["highly_variable"] in-place.  setup_data stores features from
    # that column, so it must see the propagated version.
    if modality == "ATAC":
        print("Constructing RNA-anchored guidance graph for ATAC ...")
        guidance = sw.pp.rna_anchored_guidance_graph(
            rna,
            other,
            promoter_len=args.promoter_len,
            extend_range=args.extend_range,
        )
    else:
        guidance = build_protein_guidance(rna, other)

    guidance_hvf = guidance.subgraph(chain(
        rna.var.query("highly_variable").index,
        other.var.query("highly_variable").index
    )).copy()

    # Ensure ALL HV features are in the guidance graph (add self-loops for missing ones)
    rna_hvf = set(rna.var.query("highly_variable").index)
    other_hvf = set(other.var.query("highly_variable").index)
    all_hvf = rna_hvf.union(other_hvf)
    missing = all_hvf - set(guidance_hvf.nodes)
    if missing:
        for feat in missing:
            guidance_hvf.add_edge(feat, feat, weight=1.0, sign=1, type="loop")
        print(f"Added {len(missing)} missing features to guidance graph as self-loops.")

    if guidance_hvf.number_of_edges() == 0:
        raise ValueError("Filtered guidance graph is empty. Please review preprocessing or HVG selection.")

    # SWITCH data setup (AFTER guidance graph, so propagated HV is captured) -------
    sw.pp.setup_data(
        rna,
        prob_model="NB",
        use_highly_variable=True,
        use_layer="counts",
    )

    # Always use NB prob_model. Pre-normalized data has already been converted
    # to pseudo-integer counts in preprocess_atac, so NB is safe.
    # (Normal prob_model causes NaN during cycle-consistency training.)
    sw.pp.setup_data(
        other,
        prob_model="NB",
        use_highly_variable=True,
        use_layer="counts",
    )

    # Auto-compute radius cutoff targeting ~10 neighbors (following SWITCH tutorials)
    from sklearn.neighbors import NearestNeighbors
    coords = rna.obsm["spatial"]
    nn = NearestNeighbors(n_neighbors=11).fit(coords)  # 10 neighbors + self
    dists, _ = nn.kneighbors(coords)
    auto_radius = float(np.median(dists[:, -1]) * 1.1)  # median 10th-NN distance * 1.1
    spatial_cutoff = args.spatial_radius if args.spatial_radius != 1.0 else auto_radius
    print(f"Spatial graph: using cutoff={spatial_cutoff:.2f} (auto={auto_radius:.2f})")
    sw.pp.cal_spatial_net(rna, cutoff=spatial_cutoff, model="Radius")
    sw.pp.cal_spatial_net(other, cutoff=spatial_cutoff, model="Radius")

    # Validate graph (warnings only) ----------------------------------------------
    try:
        sw.pp.check_graph(guidance_hvf, [rna, other], verbose=True, attr="warn", sym="warn", loop="warn")
    except Exception as exc:
        print(f"Warning: guidance graph check raised an exception: {exc}")

    # Monkey-patch: clamp zero library sizes to prevent NaN in normalize -----------
    from switch.model import GATEncoder as _GATEncoder
    _orig_normalize = _GATEncoder.normalize

    def _safe_normalize(self, x, l):
        l = l.clamp(min=1.0)  # prevent division by zero for cells with no HVG expression
        return (x * (self.TOTAL_COUNT / l)).log1p()
    _GATEncoder.normalize = _safe_normalize
    print("Patched GATEncoder.normalize with library-size clamping (min=1.0)")

    # SWITCH model ----------------------------------------------------------------
    key_other = modality.lower()
    adata_dict = {"rna": rna, key_other: other}

    print("Initializing SWITCH model ...")
    model = sw.SWITCH(
        adatas=adata_dict,
        vertices=sorted(guidance_hvf.nodes),
        latent_dim=args.latent_dim,
        h_dim=args.hidden_dim,
        h_depth_enc=args.encoder_depth,
        dropout=args.dropout,
        conv_layer="GAT",
        seed=args.seed,
        device=device,
    )

    model.compile(
        lam_graph=args.lam_graph,
        lam_align=args.lam_align,
        lam_cycle=args.lam_cycle,
        lam_adv=args.lam_adv,
        lam_kl=args.lam_kl,
        vae_lr=args.vae_lr,
        dsc_lr=args.dsc_lr if args.dsc_lr > 0 else None,
    )

    start = time.time()
    print("\n========== Pretraining ==========")
    model.pretrain(
        adatas=adata_dict,
        graph=guidance_hvf,
        max_epochs=args.pretrain_epochs,
        dsc_k=args.pretrain_dsc_k,
    )

    print("\n========== Training ==========")
    model.train(
        adatas=adata_dict,
        graph=guidance_hvf,
        max_epochs=args.train_epochs,
        dsc_k=args.train_dsc_k,
    )
    total_time = time.time() - start
    print(f"SWITCH training finished in {format_duration(total_time)}")

    # Embeddings ------------------------------------------------------------------
    rna_embedding = model.encode_data("rna", rna)
    other_embedding = model.encode_data(key_other, other)

    adata_result = rna.copy()
    adata_result.obsm["SWITCH"] = np.asarray(rna_embedding)
    adata_result.obsm[f"SWITCH_{modality.lower()}"] = np.asarray(other_embedding)
    adata_result.uns["integration_type"] = "vertical"
    adata_result.uns["train_time"] = total_time

    # Dataset metadata ------------------------------------------------------------
    dataset_name, subset_name = parse_dataset_info(args)
    print(f"Detected dataset: {dataset_name}, subset: {subset_name}")

    # Plotting --------------------------------------------------------------------
    plot_base_dir = Path("Results/plot/vertical_integration")
    method_name = args.method if args.method else "SWITCH"
    plot_dir = plot_base_dir / method_name / dataset_name / subset_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"Plots will be saved to: {plot_dir}")

    sc.pp.neighbors(adata_result, use_rep="SWITCH", n_neighbors=30)
    sc.tl.umap(adata_result)

    tools = ["mclust", "louvain", "leiden", "kmeans"]
    for tool in tools:
        adata_result = universal_clustering(
            adata_result,
            n_clusters=args.cluster_nums,
            used_obsm="SWITCH",
            method=tool,
            key=tool,
            use_pca=False
        )

        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
        sc.pl.umap(adata_result, color=tool, ax=axes[0], title=f"{method_name}-{tool}", s=20, show=False)
        sc.pl.embedding(adata_result, basis="spatial", color=tool, ax=axes[1],
                        title=f"{method_name}-{tool}", s=20, show=False)
        plt.tight_layout(w_pad=0.3)
        fig.savefig(plot_dir / f"clustering_{tool}_umap_spatial.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Remove temporary search columns from clustering
    tmp_cols = [col for col in adata_result.obs.columns if col.startswith("tmp_search")]
    for col in tmp_cols:
        del adata_result.obs[col]

    # Save integrated AnnData -----------------------------------------------------
    save_dir = Path(args.save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    adata_result.write(args.save_path)
    print(adata_result)
    print(f"Saved SWITCH integration result to {args.save_path}")


if __name__ == "__main__":
    os.environ["R_HOME"] = "/home/zhenghong/miniconda3/envs/smobench/lib/R"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    parser = argparse.ArgumentParser(description="Run SWITCH vertical integration")
    parser.add_argument("--data_type", type=str, default="10x", help="Dataset type annotation for logging purposes")
    parser.add_argument("--RNA_path", type=str, required=True, help="Path to RNA AnnData (.h5ad)")
    parser.add_argument("--ADT_path", type=str, default="", help="Path to ADT AnnData (.h5ad)")
    parser.add_argument("--ATAC_path", type=str, default="", help="Path to ATAC AnnData (.h5ad)")
    parser.add_argument("--save_path", type=str, required=True, help="Output h5ad path")
    parser.add_argument("--dataset", type=str, default="", help="Optional dataset/subset descriptor (e.g., Human_Tonsils/S1)")
    parser.add_argument("--cluster_nums", type=int, required=True, help="Target cluster number for evaluation")
    parser.add_argument("--method", type=str, default="SWITCH", help="Method name used for plotting/output folders")
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device, e.g. cuda:0 or cpu")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")

    # Annotation & preprocessing hyper-parameters
    parser.add_argument("--gtf_path", type=str, default="", help="Path to gene annotation GTF (required if RNA lacks genomic coordinates)")
    parser.add_argument("--gtf_by", type=str, default="gene_name", help="Field within GTF attributes for gene annotation")
    parser.add_argument("--rna_hv_genes", type=int, default=2000, help="Number of highly variable genes for RNA")
    parser.add_argument("--other_hv_features", type=int, default=2000, help="Number of highly variable features for ATAC")
    parser.add_argument("--min_peak_cells", type=int, default=20, help="Minimum ATAC peak coverage when filtering")
    parser.add_argument("--promoter_len", type=int, default=2000, help="Promoter length for guidance graph (bp)")
    parser.add_argument("--extend_range", type=int, default=0, help="Extend range for guidance graph (bp)")
    parser.add_argument("--spatial_radius", type=float, default=1.0, help="(deprecated) Radius cutoff for spatial neighbor graph")
    parser.add_argument("--spatial_k", type=int, default=10, help="Number of nearest neighbors for spatial graph (KNN)")

    # Model hyper-parameters
    parser.add_argument("--latent_dim", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--encoder_depth", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lam_graph", type=float, default=0.35)
    parser.add_argument("--lam_align", type=float, default=0.3)
    parser.add_argument("--lam_cycle", type=float, default=1.0)
    parser.add_argument("--lam_adv", type=float, default=0.02)
    parser.add_argument("--lam_kl", type=float, default=1.0)
    parser.add_argument("--vae_lr", type=float, default=2e-4)
    parser.add_argument("--dsc_lr", type=float, default=5e-5)
    parser.add_argument("--pretrain_epochs", type=int, default=1500)
    parser.add_argument("--pretrain_dsc_k", type=int, default=4)
    parser.add_argument("--train_epochs", type=int, default=800)
    parser.add_argument("--train_dsc_k", type=int, default=12)

    args = parser.parse_args()
    main(args)
