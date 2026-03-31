#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SpaMI Horizontal Integration Script
Author: SMOBench Team
Description:
    Performs horizontal integration using SpaMI on fusion datasets
    (e.g., RNA+ADT or RNA+ATAC) for batch effect removal and modality alignment.
"""

import os
import sys
import re
import time
import argparse
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import episcanpy
from anndata import AnnData


# === Path Setup ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

spami_path = os.path.join(project_root, "Methods/SpaMI/SpaMI")
sys.path.append(spami_path)

# === Import Required Functions ===
from utils import fix_seed
from preprocess import tfidf, construct_adj, add_contrastive_label
from main import train
from Utils.SMOBench_clustering import universal_clustering


def parse_dataset_info(args):
    """Extract dataset name and subset for saving results"""
    if hasattr(args, 'dataset') and args.dataset:
        return args.dataset, "fusion"

    if "HLN_Fusion" in args.RNA_path:
        return "HLN", "fusion"
    elif "HT_Fusion" in args.RNA_path:
        return "HT", "fusion"
    elif "ME_S1_Fusion" in args.RNA_path:
        return "MISAR_S1", "fusion"
    elif "ME_S2_Fusion" in args.RNA_path:
        return "MISAR_S2", "fusion"
    elif "Mouse_Thymus_Fusion" in args.RNA_path:
        return "Mouse_Thymus", "fusion"
    elif "Mouse_Spleen_Fusion" in args.RNA_path:
        return "Mouse_Spleen", "fusion"
    elif "Mouse_Brain_Fusion" in args.RNA_path:
        return "Mouse_Brain", "fusion"

    return "Unknown", "fusion"


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    fix_seed(args.seed)

    # === Load Fusion Data ===
    print("Loading fusion data...")
    adata_omics1 = sc.read_h5ad(args.RNA_path)
    adata_omics1.var_names_make_unique()

    if args.ADT_path:
        adata_omics2 = sc.read_h5ad(args.ADT_path)
        modality = "ADT"
    elif args.ATAC_path:
        adata_omics2 = sc.read_h5ad(args.ATAC_path)
        modality = "ATAC"
    else:
        raise ValueError("Either ADT_path or ATAC_path must be provided.")

    adata_omics2.var_names_make_unique()
    print(f"Performing horizontal integration: RNA + {modality}")

    # === Check and align cells ===
    common_obs = adata_omics1.obs_names.intersection(adata_omics2.obs_names)
    adata_omics1 = adata_omics1[common_obs].copy()
    adata_omics2 = adata_omics2[common_obs].copy()
    print(f"Common cells retained: {len(common_obs)}")

    # === Add pseudo spatial coordinates if missing ===
    for i, (adata, name) in enumerate([(adata_omics1, 'RNA'), (adata_omics2, modality)]):
        if 'spatial' not in adata.obsm.keys():
            print(f"Warning: No spatial coordinates in {name}, generating pseudo coordinates...")
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
            adata.obsm['spatial'] = adata.obsm['X_umap']

    # === Preprocess RNA ===
    sc.pp.filter_genes(adata_omics1, min_cells=10)
    sc.pp.normalize_total(adata_omics1, target_sum=1e4)
    sc.pp.log1p(adata_omics1)
    sc.pp.highly_variable_genes(adata_omics1, n_top_genes=1800)
    adata_omics1 = adata_omics1[:, adata_omics1.var['highly_variable']]
    sc.pp.scale(adata_omics1)
    from sklearn.decomposition import PCA
    feat_omics1 = PCA(n_components=50).fit_transform(adata_omics1.X.toarray() if hasattr(adata_omics1.X, "toarray") else adata_omics1.X)

    # === Preprocess second modality ===
    if modality == "ADT":
        sc.pp.scale(adata_omics2)
        # feat_omics2 = PCA(n_components=50).fit_transform(adata_omics2.X.toarray() if hasattr(adata_omics2.X, "toarray") else adata_omics2.X)
        # 自动限制 PCA 维度不超过特征数或样本数
        X_omics2 = adata_omics2.X.toarray() if hasattr(adata_omics2.X, "toarray") else adata_omics2.X
        n_comps_omics2 = min(50, X_omics2.shape[0], X_omics2.shape[1])
        print(f"Performing PCA on omics2 with n_comps = {n_comps_omics2} (samples={X_omics2.shape[0]}, features={X_omics2.shape[1]})")
        feat_omics2 = PCA(n_components=n_comps_omics2).fit_transform(X_omics2)

    elif modality == "ATAC":
        # === ATAC preprocessing ===
        import episcanpy.api as epi
        epi.pp.binarize(adata_omics2)
        epi.pp.filter_features(adata_omics2, min_cells=np.ceil(0.005 * adata_omics2.shape[0]))

        count_mat = adata_omics2.X.copy()

        # === TF-IDF normalization ===
        X = tfidf(count_mat)
        X_norm = np.log1p((X / np.maximum(X.sum(axis=1, keepdims=True), 1e-9)) * 1e4)

        # === LSI (SVD) ===
        from sklearn.utils.extmath import randomized_svd
        n_comps_lsi = min(51, X_norm.shape[0], X_norm.shape[1])
        print(f"Performing LSI on ATAC with n_comps = {n_comps_lsi} (samples={X_norm.shape[0]}, features={X_norm.shape[1]})")

        U, S, Vt = randomized_svd(X_norm, n_components=n_comps_lsi)
        X_lsi = U[:, 1:]  # skip first component
        feat_omics2 = X_lsi

    # === Build graph data for SpaMI ===
    adj_omics1, graph_neigh_omics1 = construct_adj(adata_omics1, n_neighbors=3)
    label_CSL_omics1 = add_contrastive_label(adata_omics1)
    omics1_data = {
        'feat': feat_omics1,
        'adj': adj_omics1,
        'graph_neigh': graph_neigh_omics1,
        'label_CSL': label_CSL_omics1,
    }

    adj_omics2, graph_neigh_omics2 = construct_adj(adata_omics2, n_neighbors=3)
    label_CSL_omics2 = add_contrastive_label(adata_omics2)
    omics2_data = {
        'feat': feat_omics2,
        'adj': adj_omics2,
        'graph_neigh': graph_neigh_omics2,
        'label_CSL': label_CSL_omics2,
    }

    # === Train SpaMI ===
    print("Training SpaMI for horizontal integration...")
    start_time = time.time()
    emb = train(omics1_data, omics2_data, args.dataset, out_dim=20)
    end_time = time.time()

    print("Training time:", round(end_time - start_time, 2), "seconds")

    adata = adata_omics1.copy()
    adata.obsm['SpaMI'] = emb
    adata.obsm['spatial'] = adata_omics1.obsm['spatial']
    adata.uns['integration_type'] = 'horizontal'
    adata.uns['train_time'] = end_time - start_time

    # === Parse Dataset Info ===
    dataset_name, subset_name = parse_dataset_info(args)
    print(f"Detected dataset: {dataset_name}, subset: {subset_name}")

    # === Clustering and UMAP Generation ===
    tools = ['mclust', 'louvain', 'leiden', 'kmeans']
    
    # === Generate UMAP coordinates (store in adata, no plotting) ===
    print("Generating UMAP coordinates...")
    sc.pp.neighbors(adata, use_rep='SpaMI', n_neighbors=30)
    sc.tl.umap(adata)
    print("UMAP coordinates generated and stored in adata.obsm['X_umap']")
    
    # === Run clustering methods ===
    print("Running clustering methods...")
    for tool in tools:
        print(f"  Running {tool} clustering...")
        adata = universal_clustering(
            adata,
            n_clusters=args.cluster_nums,
            used_obsm='SpaMI',
            method=tool,
            key=tool,
            use_pca=False
        )
    
    print("All clustering methods completed")

    # === Save AnnData ===
    save_dir = os.path.dirname(args.save_path)
    os.makedirs(save_dir, exist_ok=True)
    adata.write(args.save_path)
    print(adata)
    print('Saving horizontal integration results to...', args.save_path)


if __name__ == "__main__":
    os.environ['R_HOME'] = '/home/zhenghong/miniconda3/envs/smobench/lib/R'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    parser = argparse.ArgumentParser(description="Run SpaMI horizontal integration")
    parser.add_argument('--RNA_path', type=str, required=True, help='Path to RNA fusion adata')
    parser.add_argument('--ADT_path', type=str, default='', help='Path to ADT fusion adata')
    parser.add_argument('--ATAC_path', type=str, default='', help='Path to ATAC fusion adata')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save integrated adata')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device (cuda:0 or cpu)')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--cluster_nums', type=int, help='Number of clusters')
    args = parser.parse_args()

    main(args)
