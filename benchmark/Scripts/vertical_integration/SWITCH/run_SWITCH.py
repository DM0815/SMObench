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
    sc.pp.normalize_total(atac, target_sum=1e4)
    sc.pp.log1p(atac)
    sc.pp.highly_variable_genes(atac, n_top_genes=min(hv_top, atac.n_vars), flavor="seurat_v3")
    sc.pp.scale(atac)


def preprocess_adt(adt: sc.AnnData):
    """Simple CLR-like preprocessing for ADT counts."""
    print("Preprocessing ADT modality ...")
    sc.pp.normalize_total(adt, target_sum=1e4)
    sc.pp.log1p(adt)
    sc.pp.scale(adt)
    adt.var["highly_variable"] = True


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
        gene = rna_names.get(prot.upper())
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

    # SWITCH data setup ------------------------------------------------------------
    sw.pp.setup_data(
        rna,
        prob_model="NB",
        use_highly_variable=True,
        use_layer="counts",
    )

    sw.pp.setup_data(
        other,
        prob_model="NB",
        use_highly_variable=True,
    )

    # Guidance graph ---------------------------------------------------------------
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

    if guidance_hvf.number_of_edges() == 0:
        raise ValueError("Filtered guidance graph is empty. Please review preprocessing or HVG selection.")

    sw.pp.cal_spatial_net(rna, cutoff=args.spatial_radius, model="Radius")
    sw.pp.cal_spatial_net(other, cutoff=args.spatial_radius, model="Radius")

    # Validate graph (warnings only) ----------------------------------------------
    try:
        sw.pp.check_graph(guidance_hvf, [rna, other], verbose=True, attr="warn", sym="warn", loop="warn")
    except Exception as exc:
        print(f"Warning: guidance graph check raised an exception: {exc}")

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
    parser.add_argument("--spatial_radius", type=float, default=1.0, help="Radius cutoff for spatial neighbor graph")

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
