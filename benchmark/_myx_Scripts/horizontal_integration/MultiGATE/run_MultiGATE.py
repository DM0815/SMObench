#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import logging
import warnings
import re

import numpy as np
import pandas as pd

if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

if "CONDA_PREFIX" in os.environ:
    lib_path = os.path.join(os.environ["CONDA_PREFIX"], "lib")
    os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

import scanpy as sc
import matplotlib.pyplot as plt

# Reduce TensorFlow verbosity before importing MultiGATE
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# === Project path setup ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

multi_gate_root = os.path.join(project_root, "Methods/MultiGATE")
sys.path.append(multi_gate_root)

import tensorflow.compat.v1 as _tf1

import MultiGATE  # noqa: E402
from Utils.SMOBench_clustering import universal_clustering  # noqa: E402

# --- Monkey-patch build_session for allow_soft_placement ---
from MultiGATE.MultiGATE import MultiGATE as _MultiGATEClass

_orig_build_session = _MultiGATEClass.build_session

def _patched_build_session(self, gpu=True):
    if self.config is not None:
        config = self.config
    else:
        config = _tf1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.allow_soft_placement = True
    if gpu == False:
        config.intra_op_parallelism_threads = 0
        config.inter_op_parallelism_threads = 0
    self.session = _tf1.Session(config=config)
    self.session.run([_tf1.global_variables_initializer(), _tf1.local_variables_initializer()])

_MultiGATEClass.build_session = _patched_build_session

# --- Monkey-patch train_MultiGATE for gene_peak_Net propagation ---
# Cal_gene_peak_Net_new reassigns its local `rna` variable, so gene_peak_Net
# may only be written to adata2.uns. Ensure adata1 has it before training.
_orig_train_MultiGATE = MultiGATE.train_MultiGATE

def _patched_train_MultiGATE(adata1, adata2, *args, **kwargs):
    if 'gene_peak_Net' not in adata1.uns:
        if 'gene_peak_Net' in adata2.uns:
            adata1.uns['gene_peak_Net'] = adata2.uns['gene_peak_Net']
            logging.info("Patched: propagated gene_peak_Net from adata2 to adata1 (%d edges)",
                         len(adata1.uns['gene_peak_Net']))
        else:
            logging.error("gene_peak_Net not found in EITHER adata1 or adata2!")
    return _orig_train_MultiGATE(adata1, adata2, *args, **kwargs)

MultiGATE.train_MultiGATE = _patched_train_MultiGATE
logging.basicConfig(level=logging.INFO)
logging.info("Patched: train_MultiGATE with gene_peak_Net propagation fix")


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_dataset_info(args):
    """Infer dataset and subset names from arguments."""
    if args.dataset:
        parts = args.dataset.strip("/").split("/")
        if len(parts) == 2:
            return parts[0], parts[1]
        if len(parts) == 1:
            return parts[0], "Unknown"

    # Try to infer from RNA path
    match = re.search(r"Dataset/([^/]+)/([^/]+)/([^/]+)/adata_RNA\.h5ad", args.RNA_path)
    if match:
        return match.group(2), match.group(3)
    return "Unknown", "Unknown"


def ensure_spatial_coordinates(adata, label):
    """Ensure AnnData object has spatial coordinates, otherwise fallback to UMAP embedding."""
    if "spatial" in adata.obsm:
        return adata

    logging.warning("%s lacks spatial coordinates. Using UMAP embedding as pseudo-spatial.", label)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    adata.obsm["spatial"] = adata.obsm["X_umap"].copy()
    return adata


def preprocess_rna(adata, n_top_genes=3000, min_cells=10):
    """Standard RNA preprocessing pipeline."""
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=min(n_top_genes, adata.n_vars))
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    return adata


def preprocess_protein(adata):
    """Simple CLR + scaling for ADT modality."""
    # Centered log-ratio per cell
    data = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
    data = np.asarray(data, dtype=np.float64)
    with np.errstate(divide="ignore"):
        clr = np.log1p(data)
    clr_mean = clr.mean(axis=1, keepdims=True)
    clr -= clr_mean
    adata.layers["clr"] = clr
    adata.X = clr
    sc.pp.scale(adata, max_value=10)
    return adata


def preprocess_atac(adata, n_top_features=30000):
    """Basic preprocessing for ATAC modality."""
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=min(n_top_features, adata.n_vars),
    )
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    return adata


def build_gene_protein_network(adata_rna, adata_protein, csv_path=None):
    """Attach gene-protein mapping to AnnData objects."""
    if csv_path and os.path.exists(csv_path):
        logging.info("Loading gene-protein mapping from %s", csv_path)
        df = pd.read_csv(csv_path)
        if "Gene" not in df.columns:
            if "gene" in df.columns:
                df = df.rename(columns={"gene": "Gene"})
            elif "Gene" not in df.columns:
                raise ValueError("CSV file must contain a 'Gene' column.")
        if "Protein" not in df.columns:
            if "QueryName" in df.columns:
                df = df.rename(columns={"QueryName": "Protein"})
            else:
                raise ValueError("CSV file must contain 'Protein' or 'QueryName'.")

        required_cols = {"Gene", "Protein"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV file must contain columns {required_cols}.")

        df = df[list(required_cols)].dropna()
        df = df[
            df["Gene"].isin(adata_rna.var_names)
            & df["Protein"].isin(adata_protein.var_names)
        ]
        df = df.drop_duplicates()

        if df.empty:
            logging.warning("Filtered gene-protein mapping is empty. Falling back to automatic mapping.")
        else:
            df = df.rename(columns={"Protein": "Peak"})
            adata_rna.uns["gene_peak_Net"] = df.copy()
            adata_protein.uns["gene_peak_Net"] = df.copy()
            return df.shape[0], "csv"

    # Fallback to heuristic matching
    logging.info("Constructing gene-protein network via heuristic matching.")
    MultiGATE.Cal_gene_protein_Net(adata_rna, adata_protein, verbose=True)
    gp_df = adata_rna.uns.get("gene_peak_Net", pd.DataFrame())
    return getattr(gp_df, "shape", (0,))[0] if isinstance(gp_df, pd.DataFrame) else 0, "auto"


def build_gene_peak_network(adata_rna, adata_atac, gtf_path, max_distance):
    """Construct gene-peak relationships required for ATAC integration."""
    if not os.path.exists(gtf_path):
        raise FileNotFoundError(f"GTF file not found: {gtf_path}")
    logging.info("Constructing gene-peak network using %s (max distance: %d)", gtf_path, max_distance)
    MultiGATE.Cal_gene_peak_Net_new(
        adata_rna,
        adata_atac,
        range=max_distance,
        file=gtf_path,
        verbose=True,
    )
    # Debug: check where gene_peak_Net ended up
    in_rna = 'gene_peak_Net' in adata_rna.uns
    in_atac = 'gene_peak_Net' in adata_atac.uns
    logging.info("After Cal_gene_peak_Net_new: gene_peak_Net in rna=%s, in atac=%s", in_rna, in_atac)

    # Cal_gene_peak_Net_new reassigns its local `rna` variable, so gene_peak_Net
    # may only be written to adata_atac.uns.  Ensure both have it.
    if in_atac and not in_rna:
        adata_rna.uns['gene_peak_Net'] = adata_atac.uns['gene_peak_Net']
        logging.info("Copied gene_peak_Net from atac to rna (%d edges)", len(adata_rna.uns['gene_peak_Net']))
    elif in_rna and not in_atac:
        adata_atac.uns['gene_peak_Net'] = adata_rna.uns['gene_peak_Net']
        logging.info("Copied gene_peak_Net from rna to atac (%d edges)", len(adata_atac.uns['gene_peak_Net']))
    elif not in_rna and not in_atac:
        raise RuntimeError("gene_peak_Net not found in either adata after Cal_gene_peak_Net_new!")


def run_clustering_and_plots(adata, cluster_nums, dataset_name, subset_name, plot_root, used_obsm="MultiGATE_clip_all"):
    """Run clustering with multiple algorithms and generate spatial/UMAP plots."""
    os.makedirs(plot_root, exist_ok=True)

    # Ensure neighbors/UMAP available once
    if used_obsm not in adata.obsm:
        raise KeyError(f"Embedding '{used_obsm}' not found in adata.obsm.")

    sc.pp.neighbors(adata, use_rep=used_obsm, n_neighbors=30)
    if "X_umap" not in adata.obsm:
        sc.tl.umap(adata)

    cluster_methods = ["mclust", "louvain", "leiden", "kmeans"]
    for method in cluster_methods:
        logging.info("Clustering with %s (k=%d).", method, cluster_nums)
        adata = universal_clustering(
            adata,
            n_clusters=cluster_nums,
            used_obsm=used_obsm,
            method=method,
            key=method,
            use_pca=False,
        )

        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
        sc.pl.umap(adata, color=method, ax=axes[0], title=f"MultiGATE-{method}", show=False, s=20)
        sc.pl.embedding(
            adata,
            basis="spatial",
            color=method,
            ax=axes[1],
            title=f"MultiGATE-{method}",
            show=False,
            s=20,
        )
        plt.tight_layout(w_pad=0.3)
        outfile = os.path.join(plot_root, f"clustering_{method}_umap_spatial.png")
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return adata


def main(args):
    setup_logger()
    warnings.filterwarnings("ignore")
    sc.settings.set_figure_params(dpi=100, facecolor="white")

    if args.seed is not None:
        np.random.seed(args.seed)

    logging.info("Loading RNA data from %s", args.RNA_path)
    adata_rna = sc.read_h5ad(args.RNA_path)
    adata_rna.var_names_make_unique()

    modality = None
    modality_name = None

    if args.ADT_path:
        logging.info("Loading ADT data from %s", args.ADT_path)
        adata_other = sc.read_h5ad(args.ADT_path)
        modality = "protein"
        modality_name = "Proteome"
    elif args.ATAC_path:
        logging.info("Loading ATAC data from %s", args.ATAC_path)
        adata_other = sc.read_h5ad(args.ATAC_path)
        modality = "ATAC_RNA"
        modality_name = "Epigenome"
    else:
        raise ValueError("You must provide either --ADT_path or --ATAC_path.")

    adata_other.var_names_make_unique()

    # Align cells
    common_cells = adata_rna.obs_names.intersection(adata_other.obs_names)
    if len(common_cells) == 0:
        raise ValueError("No overlapping cells between RNA and secondary modality.")
    logging.info("Common cells: %d (RNA total: %d, %s total: %d)", len(common_cells), adata_rna.n_obs, modality_name, adata_other.n_obs)
    adata_rna = adata_rna[common_cells].copy()
    adata_other = adata_other[common_cells].copy()

    # Ensure spatial coordinates
    adata_rna = ensure_spatial_coordinates(adata_rna, "RNA")
    adata_other = ensure_spatial_coordinates(adata_other, modality_name)
    adata_rna.obsm["spatial"][:, 1] *= -1
    adata_other.obsm["spatial"][:, 1] *= -1

    # Preprocessing
    logging.info("Preprocessing RNA modality...")
    adata_rna = preprocess_rna(adata_rna, n_top_genes=args.n_top_genes, min_cells=args.min_cells)

    if modality == "protein":
        logging.info("Preprocessing ADT modality...")
        adata_other = preprocess_protein(adata_other)
    else:
        logging.info("Preprocessing ATAC modality...")
        adata_other = preprocess_atac(adata_other, n_top_features=args.n_top_features)

    # Spatial graphs
    rna_radius = args.rna_radius or (40 if modality == "protein" else 100)
    other_radius = args.other_radius or rna_radius
    logging.info("Constructing spatial graph for RNA (rad_cutoff=%s)", rna_radius)
    MultiGATE.Cal_Spatial_Net(adata_rna, rad_cutoff=rna_radius)
    MultiGATE.Stats_Spatial_Net(adata_rna)

    logging.info("Constructing spatial graph for %s (rad_cutoff=%s)", modality_name, other_radius)
    MultiGATE.Cal_Spatial_Net(adata_other, rad_cutoff=other_radius)
    MultiGATE.Stats_Spatial_Net(adata_other)

    # Gene-protein or gene-peak relationships
    if modality == "protein":
        edges, source = build_gene_protein_network(adata_rna, adata_other, csv_path=args.gene_protein_csv)
        logging.info("Gene-protein edges: %s (source: %s)", edges, source)
    else:
        # Fix: strip "_merged_*" suffix from ATAC peak names so that
        # Cal_gene_peak_Net_new can parse "chr:start-end" correctly
        import re as _re
        cleaned = adata_other.var_names.map(lambda x: _re.sub(r'_merged_\d+$', '', x))
        if not cleaned.equals(adata_other.var_names):
            logging.info("Stripped _merged_* suffix from %d ATAC peak names", (cleaned != adata_other.var_names).sum())
            adata_other.var_names = cleaned
            adata_other.var_names_make_unique()
        build_gene_peak_network(adata_rna, adata_other, gtf_path=args.gtf_path, max_distance=args.gene_peak_max_distance)

    # Train MultiGATE
    logging.info("Training MultiGATE (%s integration)...", modality_name)
    start_time = time.time()
    adata_rna, adata_other = MultiGATE.train_MultiGATE(
        adata_rna,
        adata_other,
        n_epochs=args.n_epochs,
        temp=args.temperature,
        type=modality,
        save_attention=args.save_attention,
        protein_value=args.protein_edge_weight,
    )
    train_time = time.time() - start_time
    logging.info("Training completed in %.2f seconds.", train_time)

    # Prepare output AnnData (RNA view with integrated embeddings)
    adata_result = adata_rna
    adata_result.uns["method"] = "MultiGATE"
    adata_result.uns["train_time"] = train_time
    adata_result.uns["integration_task"] = "RNA+ADT" if modality == "protein" else "RNA+ATAC"
    adata_result.uns["MultiGATE_params"] = {
        "n_epochs": args.n_epochs,
        "temperature": args.temperature,
        "type": modality,
        "rna_radius": rna_radius,
        "other_radius": other_radius,
        "gene_peak_max_distance": args.gene_peak_max_distance if modality != "protein" else None,
    }

    dataset_name, subset_name = parse_dataset_info(args)
    logging.info("Dataset detected as %s / %s", dataset_name, subset_name)

    plot_dir = os.path.join("Results/plot/MultiGATE", dataset_name, subset_name)
    adata_result = run_clustering_and_plots(
        adata_result,
        cluster_nums=args.cluster_nums,
        dataset_name=dataset_name,
        subset_name=subset_name,
        plot_root=plot_dir,
        used_obsm="MultiGATE_clip_all",
    )

    # Persist outputs
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    adata_result.write(args.save_path)
    logging.info("Saved integrated AnnData to %s", args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MultiGATE vertical integration.")
    parser.add_argument("--RNA_path", type=str, required=True, help="Path to RNA h5ad.")
    parser.add_argument("--ADT_path", type=str, default="", help="Path to ADT h5ad (for RNA+ADT).")
    parser.add_argument("--ATAC_path", type=str, default="", help="Path to ATAC h5ad (for RNA+ATAC).")
    parser.add_argument("--save_path", type=str, required=True, help="Output path for integrated h5ad.")
    parser.add_argument("--cluster_nums", type=int, required=True, help="Target number of clusters.")
    parser.add_argument("--dataset", type=str, default="", help="Optional dataset identifier 'Dataset/Sub'.")
    parser.add_argument("--n_epochs", type=int, default=1000, help="Training epochs for MultiGATE.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature parameter for MultiGATE.")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed.")
    parser.add_argument("--n_top_genes", type=int, default=3000, help="Number of RNA HVGs.")
    parser.add_argument("--n_top_features", type=int, default=30000, help="Number of ATAC HV features.")
    parser.add_argument("--min_cells", type=int, default=10, help="Minimum cells per gene for RNA filtering.")
    parser.add_argument("--rna_radius", type=float, default=None, help="Spatial radius for RNA graph.")
    parser.add_argument("--other_radius", type=float, default=None, help="Spatial radius for secondary modality.")
    parser.add_argument("--gene_protein_csv", type=str, default="", help="Optional CSV mapping proteins to genes.")
    parser.add_argument("--gtf_path", type=str, default=os.path.join(
        multi_gate_root, "docs", "source", "data_tutorial", "human", "gencode.v25.chr_patch_hapl_scaff.annotation.gtf.gz"
    ), help="Path to GTF file for gene-peak construction.")
    parser.add_argument("--gene_peak_max_distance", type=int, default=150000, help="Max distance (bp) for gene-peak pairing.")
    parser.add_argument("--protein_edge_weight", type=float, default=0.001, help="Edge weight used for protein networks.")
    parser.add_argument("--save_attention", action="store_true", help="Store attention matrices from MultiGATE.")

    args = parser.parse_args()
    main(args)
