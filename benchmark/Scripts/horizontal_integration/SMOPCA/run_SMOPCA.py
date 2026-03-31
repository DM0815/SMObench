# -*- coding: utf-8 -*-
"""
Run SMOPCA for horizontal integration (RNA + ADT or RNA + ATAC)
Adapted from SpatialGlue pipeline for consistency within SMOBench framework.
"""

import os
import sys
import time
import torch
import logging
import warnings
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# === Import project root and SMOPCA module ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

smopca_path = os.path.join(project_root, "Methods/SMOPCA/src")
sys.path.append(smopca_path)

import model
import utils
from Methods.SpatialGlue.preprocess import fix_seed
from Utils.SMOBench_clustering import universal_clustering

# === Suppress logging clutter ===
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------
# Helper function: parse dataset info
# ---------------------------------------------------------------------
def parse_dataset_info(args):
    """Extract dataset_name and subset_name from fusion paths."""
    if hasattr(args, 'dataset') and args.dataset:
        return args.dataset, "fusion"

    if "HLN_Fusion" in args.RNA_path:
        return "HLN", "fusion"
    elif "HT_Fusion" in args.RNA_path:
        return "HT", "fusion"
    elif "Mouse_Thymus" in args.RNA_path:
        return "Mouse_Thymus", "fusion"
    elif "Mouse_Spleen" in args.RNA_path:
        return "Mouse_Spleen", "fusion"
    elif "Mouse_Brain" in args.RNA_path:
        return "Mouse_Brain", "fusion"
    else:
        return "Unknown", "fusion"


# ---------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    fix_seed(args.seed)

    # === Load Fusion Data ===
    print("Loading RNA and secondary modality (ADT/ATAC) fusion data...")
    adata_rna = sc.read_h5ad(args.RNA_path)
    adata_rna.var_names_make_unique()

    if args.ADT_path:
        adata_other = sc.read_h5ad(args.ADT_path)
        modality = "ADT"
        modality_name = "Proteome"
    elif args.ATAC_path:
        adata_other = sc.read_h5ad(args.ATAC_path)
        modality = "ATAC"
        modality_name = "Epigenome"
    else:
        raise ValueError("Please provide either --ADT_path or --ATAC_path.")

    adata_other.var_names_make_unique()
    print(f"Processing horizontal integration: RNA + {modality_name} fusion data...")

    # === Ensure common cells ===
    common_cells = adata_rna.obs_names.intersection(adata_other.obs_names)
    print(f"Common cells: {len(common_cells)}")
    adata_rna = adata_rna[common_cells].copy()
    adata_other = adata_other[common_cells].copy()

    # === Add pseudo spatial coordinates if missing ===
    for adata, name in [(adata_rna, "RNA"), (adata_other, modality_name)]:
        if "spatial" not in adata.obsm.keys():
            print(f"Warning: No spatial coordinates found in {name}. Generating pseudo-spatial coordinates...")
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
            adata.obsm["spatial"] = adata.obsm["X_umap"].copy()
            print(f"Generated pseudo-spatial coordinates for {name}")

    # === Preprocess features ===
    print("Preprocessing RNA and secondary modality features...")
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.highly_variable_genes(adata_rna, n_top_genes=3000)
    adata_rna = adata_rna[:, adata_rna.var["highly_variable"]].copy()

    sc.pp.scale(adata_other)  # scale both ADT and ATAC

    X1 = adata_rna.X.A if hasattr(adata_rna.X, "A") else adata_rna.X
    X2 = adata_other.X.A if hasattr(adata_other.X, "A") else adata_other.X
    pos = adata_rna.obsm["spatial"]

    print(f"Input feature shapes: RNA {X1.shape}, {modality_name} {X2.shape}, pos {pos.shape}")

    # === Initialize and Train SMOPCA ===
    print("Training SMOPCA for horizontal integration...")
    smopca = model.SMOPCA(
        Y_list=[X1.T, X2.T],
        Z_dim=args.Z_dim,
        pos=pos,
        intercept=False,
        omics_weight=False
    )

    start_time = time.time()
    smopca.estimateParams(
        sigma_init_list=(1, 1),
        tol_sigma=2e-5,
        sigma_xtol_list=(1e-6, 1e-6),
        gamma_init=1,
        estimate_gamma=True
    )
    z = smopca.calculatePosterior()
    train_time = time.time() - start_time
    print(f"SMOPCA training completed in {train_time:.2f}s")

    # === Build AnnData with embeddings ===
    adata = sc.AnnData(z)
    try:
        adata.obs = adata_rna.obs.copy()
        adata.obs_names = adata_rna.obs_names.copy()
    except Exception:
        if adata.shape[0] == adata_rna.shape[0]:
            adata.obs_names = adata_rna.obs_names.copy()
        else:
            adata.obs_names = [f"cell_{i}" for i in range(adata.shape[0])]

    adata.obsm["SMOPCA"] = np.asarray(z)
    adata.obsm["spatial"] = pos
    adata.uns.update({
        "train_time": train_time,
        "integration_type": "horizontal",
        "method": "SMOPCA"
    })
    adata.var_names = [f"SMOPCA_{i}" for i in range(adata.shape[1])]

    # === Clustering ===
    print("Running clustering methods on SMOPCA embeddings...")
    tools = ["mclust", "louvain", "leiden", "kmeans"]
    for tool in tools:
        print(f"  Running {tool} clustering...")
        adata = universal_clustering(
            adata,
            n_clusters=args.cluster_nums,
            used_obsm="SMOPCA",
            method=tool,
            key=tool,
            use_pca=False
        )
    print("All clustering methods completed")

    # === Save Results ===
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    adata.write(args.save_path)
    print(f"Results saved to {args.save_path}")

    # === Visualization (optional) ===
    try:
        sc.pp.neighbors(adata, n_neighbors=30, use_rep="X")
        sc.tl.umap(adata)
        sc.pl.umap(adata, color=["kmeans"], title="SMOPCA clustering", show=False)
        plt_path = args.save_path.replace(".h5ad", "_umap.png")
        plt.savefig(plt_path, dpi=300, bbox_inches="tight")
        print(f"UMAP saved to {plt_path}")
    except Exception as e:
        print("Warning: Visualization skipped due to error:", e)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    os.environ['R_HOME'] = '/home/zhenghong/miniconda3/envs/smobench/lib/R'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    print("Starting SMOPCA horizontal integration...")

    parser = argparse.ArgumentParser(description="Run SMOPCA horizontal integration")
    parser.add_argument("--data_type", type=str, default="fusion", help="Data type (e.g., fusion, RNA, ADT, ATAC)")
    parser.add_argument("--method", type=str, default="SMOPCA", help="Method name")
    parser.add_argument("--RNA_path", type=str, required=True, help="Path to RNA fusion adata (.h5ad)")
    parser.add_argument("--ADT_path", type=str, default="", help="Path to ADT fusion adata (.h5ad)")
    parser.add_argument("--ATAC_path", type=str, default="", help="Path to ATAC fusion adata (.h5ad)")
    parser.add_argument("--save_path", type=str, required=True, help="Output path to save integrated AnnData (.h5ad)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Computation device")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    parser.add_argument("--cluster_nums", type=int, default=7, help="Number of clusters for KMeans/Leiden")
    parser.add_argument("--Z_dim", type=int, default=20, help="Latent embedding dimension")
    parser.add_argument("--dataset", type=str, default="", help="Dataset name for tracking")
    args = parser.parse_args()

    main(args)
