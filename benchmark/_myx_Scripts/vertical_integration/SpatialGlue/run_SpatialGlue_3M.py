#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run SpatialGlue-3M triple-modality integration (RNA + ADT + ATAC)

Uses the official SpatialGlue_3M package from:
https://github.com/JinmiaoChenLab/SpatialGlue_3M
"""

import os
import sys
import re
import time
import argparse
import torch
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

# === Path setup ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

# SpatialGlue_3M package (cloned from GitHub)
spatialglue_3m_repo = os.path.join(project_root, "Methods/SpatialGlue/SpatialGlue_3M_repo")
sys.path.append(spatialglue_3m_repo)

from SpatialGlue_3M.SpatialGlue_pyG import Train_SpatialGlue
from SpatialGlue_3M.preprocess import (
    construct_neighbor_graph, fix_seed, pca, clr_normalize_each_cell, lsi
)
from Utils.SMOBench_clustering import universal_clustering


def parse_dataset_info(args):
    if hasattr(args, 'dataset') and args.dataset:
        parts = args.dataset.strip('/').split('/')
        if len(parts) == 2:
            return parts[0], parts[1]
        elif len(parts) == 1:
            return parts[0], "Unknown"
    match = re.search(r'Dataset/.+?/([^/]+)/([^/]+)/adata_RNA\.h5ad', args.RNA_path)
    if match:
        return match.group(1), match.group(2)
    return "Unknown", "Unknown"


def main(args):
    print("=== Starting SpatialGlue-3M triple-modality integration ===")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    fix_seed(args.seed)

    # === Load all three modalities ===
    print("Loading RNA + ADT + ATAC data...")
    adata_rna = sc.read_h5ad(args.RNA_path)
    adata_adt = sc.read_h5ad(args.ADT_path)
    adata_atac = sc.read_h5ad(args.ATAC_path)
    for a in [adata_rna, adata_adt, adata_atac]:
        a.var_names_make_unique()

    # === RNA preprocessing ===
    sc.pp.filter_genes(adata_rna, min_cells=10)
    sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.scale(adata_rna)
    adata_rna_high = adata_rna[:, adata_rna.var["highly_variable"]]
    n_comps = 30
    adata_rna.obsm["feat"] = pca(adata_rna_high, n_comps=n_comps)
    print(f"RNA: {adata_rna.shape}, PCA{n_comps}")

    # === ADT preprocessing ===
    adata_adt = clr_normalize_each_cell(adata_adt)
    sc.pp.scale(adata_adt)
    n_comps_adt = min(n_comps, adata_adt.X.shape[1])
    adata_adt.obsm["feat"] = pca(adata_adt, n_comps=n_comps_adt)
    print(f"ADT: {adata_adt.shape}, PCA{n_comps_adt}")

    # === ATAC preprocessing ===
    sc.pp.highly_variable_genes(adata_atac, flavor="seurat_v3", n_top_genes=3000)
    lsi(adata_atac, use_highly_variable=False, n_components=n_comps)
    adata_atac.obsm["feat"] = adata_atac.obsm["X_lsi"].copy()
    print(f"ATAC: {adata_atac.shape}, LSI{adata_atac.obsm['feat'].shape[1]}")

    # === Align feature dimensions ===
    dims = [a.obsm["feat"].shape[1] for a in [adata_rna, adata_adt, adata_atac]]
    target_dim = min(dims)
    print(f"Aligning feature dimensions to {target_dim} (RNA={dims[0]}, ADT={dims[1]}, ATAC={dims[2]})")
    for a in [adata_rna, adata_adt, adata_atac]:
        a.obsm["feat"] = a.obsm["feat"][:, :target_dim]

    # === Construct 3M neighbor graph ===
    data = construct_neighbor_graph(adata_rna, adata_adt, adata_atac)

    # === Train SpatialGlue-3M ===
    print("Training SpatialGlue-3M model...")
    model = Train_SpatialGlue(data, datatype='Triplet', device=device, epochs=200)
    start_time = time.time()
    output = model.train()
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f}s")

    # === Build result AnnData ===
    adata = adata_rna.copy()
    adata.obsm["SpatialGlue"] = output["SpatialGlue"].copy()
    adata.uns["train_time"] = train_time

    # === Clean embeddings ===
    emb = adata.obsm["SpatialGlue"]
    if np.any(~np.isfinite(emb)):
        print("Warning: Found NaN/Inf in embedding, cleaning...")
        emb[np.isnan(emb)] = 0
        emb[np.isinf(emb)] = np.sign(emb[np.isinf(emb)]) * 1e10
        adata.obsm["SpatialGlue"] = emb

    # === Clustering and visualization ===
    dataset_name, subset_name = parse_dataset_info(args)
    print(f"Dataset: {dataset_name}, subset: {subset_name}")

    sc.pp.neighbors(adata, use_rep="SpatialGlue", n_neighbors=30)
    sc.tl.umap(adata)

    method_name = "SpatialGlue_3M"
    plot_dir = os.path.join("Results/plot", method_name, dataset_name, subset_name)
    os.makedirs(plot_dir, exist_ok=True)

    tools = ["mclust", "louvain", "leiden", "kmeans"]
    for tool in tools:
        adata = universal_clustering(
            adata, n_clusters=args.cluster_nums,
            used_obsm="SpatialGlue", method=tool,
            key=tool, use_pca=False,
        )
        if "spatial" in adata.obsm.keys():
            adata.obsm["spatial"][:, 1] = -adata.obsm["spatial"][:, 1]
        fig, axs = plt.subplots(1, 2, figsize=(7, 3))
        sc.pl.umap(adata, color=tool, ax=axs[0], title=f"{method_name}-{tool}", s=20, show=False)
        sc.pl.embedding(adata, basis="spatial", color=tool, ax=axs[1], title=f"{method_name}-{tool}", s=20, show=False)
        plt.tight_layout(w_pad=0.3)
        plt.savefig(os.path.join(plot_dir, f"clustering_{tool}_umap_spatial.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # === Save ===
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    adata.write(args.save_path)
    print(f"Saved integrated AnnData to: {args.save_path}")


if __name__ == "__main__":
    os.environ['R_HOME'] = '/home/zhenghong/miniconda3/envs/smobench/lib/R'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser(description="Run SpatialGlue 3M integration (RNA+ADT+ATAC)")
    parser.add_argument("--data_type", type=str, default="simulation")
    parser.add_argument("--RNA_path", type=str, required=True)
    parser.add_argument("--ADT_path", type=str, required=True)
    parser.add_argument("--ATAC_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--cluster_nums", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--dataset", type=str, default="")
    args = parser.parse_args()

    main(args)
