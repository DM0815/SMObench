#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run SpaBalance-3M triple-modality integration (RNA + ADT + ATAC)
Author: William Huang, 2025
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

# === 路径设置 ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

# ✅ 使用 SpaBalance_3M 版本
spabalance_path = os.path.join(project_root, "Methods/SpaBalance/SpaBalance_3M")
sys.path.append(spabalance_path)

# === 导入 3M 版本核心模块 ===
from preprocess import fix_seed, clr_normalize_each_cell, pca, lsi, construct_neighbor_graph
from Train_model_3M import Train_SpaBalance_3M   # ✅ 注意此处为 Train_model_3M
from Utils.SMOBench_clustering import universal_clustering

# === 工具函数 ===
def parse_dataset_info(args):
    """从 RNA_path 或 save_path 中提取 dataset_name 和 subset_name"""
    match = re.search(r'Dataset/.+?/([^/]+)/([^/]+)/adata_RNA\.h5ad', args.RNA_path)
    if match:
        return match.group(1), match.group(2)
    return "Unknown", "Unknown"


def preprocess_rna(adata):
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    adata_high = adata[:, adata.var["highly_variable"]]
    adata.obsm["feat"] = pca(adata_high, n_comps=30)
    return adata


def preprocess_adt(adata):
    adata = clr_normalize_each_cell(adata)
    sc.pp.scale(adata)
    n_comps = min(30, adata.X.shape[1])
    adata.obsm["feat"] = pca(adata, n_comps=n_comps)
    return adata


def preprocess_atac(adata):
    if "X_lsi" not in adata.obsm.keys():
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
        lsi(adata, use_highly_variable=False, n_components=30)
    adata.obsm["feat"] = adata.obsm["X_lsi"].copy()
    return adata


def align_modalities(adatas):
    """确保三个模态的特征维度一致"""
    dims = [a.obsm["feat"].shape[1] for a in adatas]
    target_dim = min(dims)
    print(f"[Dimension Alignment] Aligning to {target_dim}-dim space")
    for adata in adatas:
        adata.obsm["feat"] = adata.obsm["feat"][:, :target_dim]
    return adatas


# === 主函数 ===
def main(args):
    print("=== Starting SpaBalance-3M triple-modality integration ===")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    fix_seed(args.seed)

    # === 加载数据 ===
    adata_rna = sc.read_h5ad(args.RNA_path)
    adata_adt = sc.read_h5ad(args.ADT_path)
    adata_atac = sc.read_h5ad(args.ATAC_path)
    for a in [adata_rna, adata_adt, adata_atac]:
        a.var_names_make_unique()

    # === 各模态预处理 ===
    adata_rna = preprocess_rna(adata_rna)
    adata_adt = preprocess_adt(adata_adt)
    adata_atac = preprocess_atac(adata_atac)

    # === 对齐维度 ===
    adata_rna, adata_adt, adata_atac = align_modalities([adata_rna, adata_adt, adata_atac])

    # === 构建多模态邻接图 ===
    data = construct_neighbor_graph(adata_rna, adata_adt, adata_atac)

    # === 训练模型 ===
    model = Train_SpaBalance_3M(data, datatype=args.data_type, device=device)
    start_time = time.time()
    output = model.train()
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f}s")

    # === 保存结果 ===
    adata = adata_rna.copy()
    adata.obsm["SpaBalance"] = output["SpaBalance"].copy()
    adata.uns["train_time"] = train_time

    # === 聚类与可视化 ===
    dataset_name, subset_name = parse_dataset_info(args)
    sc.pp.neighbors(adata, use_rep="SpaBalance", n_neighbors=30)
    sc.tl.umap(adata)

    plot_dir = os.path.join("Results/plot/vertical_integration/SpaBalance_3M_triple", dataset_name, subset_name)
    os.makedirs(plot_dir, exist_ok=True)

    tools = ["mclust", "louvain", "leiden", "kmeans"]
    for tool in tools:
        adata = universal_clustering(
            adata, n_clusters=args.cluster_nums,
            used_obsm="SpaBalance", method=tool,
            key=tool, use_pca=False,
        )
        if "spatial" in adata.obsm.keys():
            adata.obsm["spatial"][:, 1] = -adata.obsm["spatial"][:, 1]
        fig, axs = plt.subplots(1, 2, figsize=(7, 3))
        sc.pl.umap(adata, color=tool, ax=axs[0], title=f"SpaBalance-3M-{tool}", s=20, show=False)
        sc.pl.embedding(adata, basis="spatial", color=tool, ax=axs[1], title=f"SpaBalance-3M-{tool}", s=20, show=False)
        plt.tight_layout(w_pad=0.3)
        plt.savefig(os.path.join(plot_dir, f"clustering_{tool}_umap_spatial.png"), dpi=300, bbox_inches="tight")
        plt.close()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    adata.write(args.save_path)
    print(f"Saved integrated AnnData to: {args.save_path}")


# === 主入口 ===
if __name__ == "__main__":
    os.environ['R_HOME'] = '/home/zhenghong/miniconda3/envs/smobench/lib/R'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser(description="Run SpaBalance-3M triple-modality integration (RNA+ADT+ATAC)")
    parser.add_argument("--data_type", type=str, required=True, help="Data type, e.g. simulation, realdata")
    parser.add_argument("--RNA_path", type=str, required=True)
    parser.add_argument("--ADT_path", type=str, required=True)
    parser.add_argument("--ATAC_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--cluster_nums", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args()

    main(args)
