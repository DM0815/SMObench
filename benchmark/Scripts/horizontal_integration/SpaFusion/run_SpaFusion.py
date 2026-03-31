#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import time
import torch
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path

# ===========================
# Setup Paths
# ===========================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

# Add SpaFusion to path
spafusion_path = os.path.join(project_root, "Methods/SpaFusion")
sys.path.append(spafusion_path)

# ===========================
# Import SpaFusion modules
# ===========================
from main import (
    pre_train,
    train,
    setup_seed,
    norm_adj,
    adjacent_matrix_preprocessing,
    load_data
)
from high_order_matrix import process_adjacency_matrix
from Utils.SMOBench_clustering import universal_clustering


# ============================================================
# Helper: Dataset name parsing
# ============================================================
def parse_dataset_info(args):
    """Extract dataset_name and subset_name from fusion paths."""
    if hasattr(args, 'dataset') and args.dataset:
        return args.dataset, "fusion"

    # Auto-parse from RNA_path
    name_map = {
        "HLN_Fusion": "HLN",
        "HT_Fusion": "HT",
        "ME_S1_Fusion": "MISAR_S1",
        "ME_S2_Fusion": "MISAR_S2",
        "Mouse_Thymus_Fusion": "Mouse_Thymus",
        "Mouse_Spleen_Fusion": "Mouse_Spleen",
        "Mouse_Brain_Fusion": "Mouse_Brain",
    }
    for key, val in name_map.items():
        if key in args.RNA_path:
            return val, "fusion"

    return "Unknown", "fusion"


# ============================================================
# Main function
# ============================================================
def main(args):

    # --------------------- Device setup ---------------------
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Starting SpaFusion horizontal integration on device:", device)

    # --------------------- Random seed ---------------------
    setup_seed(args.seed)

    # --------------------- Load data ---------------------
    adata_rna = sc.read_h5ad(args.RNA_path)
    adata_rna.var_names_make_unique()

    if not args.ADT_path:
        raise ValueError("SpaFusion requires both RNA and ADT (Protein) data!")

    adata_protein = sc.read_h5ad(args.ADT_path)
    adata_protein.var_names_make_unique()

    # Use existing labels if available
    if 'Spatial_Label' in adata_rna.obs.columns:
        label = adata_rna.obs['Spatial_Label'].values
        n_clusters = len(np.unique(label))
    else:
        label = None
        n_clusters = args.cluster_nums

    # --------------------- Preprocessing & Graphs ---------------------
    adata_rna, adata_protein = load_data(
        adata_omics1=adata_rna,
        view1="RNA",
        adata_omics2=adata_protein,
        view2="Protein",
        n_neighbors=args.spatial_k,
        k=args.adj_k
    )

    # Feature matrices
    data1 = adata_rna.obsm['feat'].copy()
    data2 = adata_protein.obsm['feat'].copy()

    # Construct adjacency matrices
    adj_path = Path("./pre_adj") / args.dataset
    adj_path.mkdir(parents=True, exist_ok=True)

    adj = adjacent_matrix_preprocessing(adata_rna, adata_protein, adj_path)

    # Normalize adjacency matrices
    feature_adj1 = norm_adj(adj['adj_feature_omics1'])
    feature_adj2 = norm_adj(adj['adj_feature_omics2'])
    spatial_adj1 = norm_adj(adj['adj_spatial_omics1'])
    spatial_adj2 = norm_adj(adj['adj_spatial_omics2'])

    # High-order adjacency matrices
    Mt1 = norm_adj(process_adjacency_matrix(feature_adj1, adj_path / f"{args.dataset}_Mt1.npy"))
    Mt2 = norm_adj(process_adjacency_matrix(feature_adj2, adj_path / f"{args.dataset}_Mt2.npy"))

    # Convert to torch tensors
    data1 = torch.tensor(data1, dtype=torch.float32).to(device)
    data2 = torch.tensor(data2, dtype=torch.float32).to(device)
    feature_adj1 = torch.tensor(feature_adj1, dtype=torch.float32).to(device)
    feature_adj2 = torch.tensor(feature_adj2, dtype=torch.float32).to(device)
    spatial_adj1 = torch.tensor(spatial_adj1, dtype=torch.float32).to(device)
    spatial_adj2 = torch.tensor(spatial_adj2, dtype=torch.float32).to(device)
    Mt1 = torch.tensor(Mt1, dtype=torch.float32).to(device)
    Mt2 = torch.tensor(Mt2, dtype=torch.float32).to(device)

    print("Adjacency tensor shape:", spatial_adj2.shape)

    # --------------------- Pretraining ---------------------
    print("\n========== Pretraining SpaFusion ==========")
    start_time_pretrain = time.time()

    emb_combination, emb_RNA, emb_ADT = pre_train(
        x1=data1,
        x2=data2,
        spatial_adj1=spatial_adj1,
        feature_adj1=feature_adj1,
        spatial_adj2=spatial_adj2,
        feature_adj2=feature_adj2,
        Mt1=Mt1,
        Mt2=Mt2,
        y=label,
        n_clusters=n_clusters,
        num_epoch=args.pretrain_epoch,
        device=device,
        weight_list=args.weight_list,
        lr=args.lr,
        dataset_name=args.dataset
    )

    pretrain_time = time.time() - start_time_pretrain
    print("SpaFusion Pretrain time:", pretrain_time)

    # --------------------- Training ---------------------
    print("\n========== Training SpaFusion ==========")
    start_time_train = time.time()

    for i in range(1):
        print(f"Training round {i}")
        emb_combination, emb_RNA, emb_ADT = train(
            x1=data1,
            x2=data2,
            spatial_adj1=spatial_adj1,
            feature_adj1=feature_adj1,
            spatial_adj2=spatial_adj2,
            feature_adj2=feature_adj2,
            y=label,
            n_clusters=n_clusters,
            Mt1=Mt1,
            Mt2=Mt2,
            num_epoch=args.train_epoch,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            device=device,
            seed=args.seed,
            weight_list=args.weight_list,
            lr=args.lr,
            num=i,
            spatial_K=args.spatial_k,
            adj_K=args.adj_k,
            dataset_name=args.dataset
        )

    train_time = time.time() - start_time_train
    print("SpaFusion Train time:", train_time)

    # --------------------- Build result AnnData ---------------------
    print("\n========== Building result AnnData ==========")
    adata = adata_rna.copy()
    adata.obsm['SpaFusion'] = emb_combination.detach().cpu().numpy()
    adata.obsm['emb_latent_omics1'] = emb_RNA.detach().cpu().numpy()
    adata.obsm['emb_latent_omics2'] = emb_ADT.detach().cpu().numpy()
    adata.uns.update({
        'train_time': train_time,
        'pretrain_time': pretrain_time,
        'integration_type': 'horizontal'
    })

    # --------------------- UMAP ---------------------
    print("\nGenerating UMAP coordinates...")
    sc.pp.neighbors(adata, use_rep='SpaFusion', n_neighbors=30)
    sc.tl.umap(adata)
    print("UMAP coordinates stored in adata.obsm['X_umap']")

    # --------------------- Clustering ---------------------
    print("\nRunning clustering methods...")
    tools = ['mclust', 'louvain', 'leiden', 'kmeans']
    for tool in tools:
        print(f"Running {tool} clustering...")
        adata = universal_clustering(
            adata,
            n_clusters=args.cluster_nums,
            used_obsm='SpaFusion',
            method=tool,
            key=tool,
            use_pca=False
        )
    print("All clustering methods completed.")

    # --------------------- Save results ---------------------
    save_dir = os.path.dirname(args.save_path)
    os.makedirs(save_dir, exist_ok=True)
    adata.write(args.save_path)

    print(adata)
    print("SpaFusion horizontal integration results saved to:", args.save_path)


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    # Environment variables for R & threading
    os.environ['R_HOME'] = '/home/zhenghong/miniconda3/envs/smobench/lib/R'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    print("Starting SpaFusion horizontal integration...")

    parser = argparse.ArgumentParser(description="Run SpaFusion horizontal integration for RNA+ADT")
    parser.add_argument('--RNA_path', type=str, required=True, help='Path to RNA AnnData (.h5ad)')
    parser.add_argument('--ADT_path', type=str, required=True, help='Path to ADT/Protein AnnData (.h5ad)')
    parser.add_argument('--dataset', type=str, default='D1', help='Dataset name')
    parser.add_argument('--save_path', type=str, required=True, help='Output path to save integrated AnnData')

    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device, e.g., cuda:0 or cpu')

    parser.add_argument('--spatial_k', type=int, default=9, help='Number of spatial neighbors')
    parser.add_argument('--adj_k', type=int, default=20, help='Number of adjacency neighbors')

    parser.add_argument('--pretrain_epoch', type=int, default=100, help='Number of pretraining epochs')
    parser.add_argument('--train_epoch', type=int, default=100, help='Number of training epochs')

    parser.add_argument('--lambda1', type=float, default=1.0, help='Lambda1 for clustering loss')
    parser.add_argument('--lambda2', type=float, default=0.1, help='Lambda2 for dense loss')
    parser.add_argument('--weight_list', type=list, default=[1, 1, 1, 1, 1, 1], help='Weights for reconstruction loss')

    parser.add_argument('--cluster_nums', type=int, default=None, help='Number of clusters for clustering')
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for SpaFusion training")

    args = parser.parse_args()
    main(args)
