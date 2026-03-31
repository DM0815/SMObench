#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

# 添加 SpaBalance 方法路径
spabalance_path = os.path.join(project_root, "Methods/SpaBalance/SpaBalance")
sys.path.append(spabalance_path)

from preprocess import fix_seed, clr_normalize_each_cell, pca, lsi, construct_neighbor_graph
from Train_model import Train_SpaBalance
from Utils.SMOBench_clustering import universal_clustering


def parse_dataset_info(args):
    """从 RNA_path 或 save_path 中提取 dataset_name 和 subset_name"""
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
    print("Starting SpaBalance vertical integration...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === 读取数据 ===
    adata_omics1 = sc.read_h5ad(args.RNA_path)
    adata_omics1.var_names_make_unique()

    if args.ADT_path:
        modality = "ADT"
        adata_omics2 = sc.read_h5ad(args.ADT_path)
    elif args.ATAC_path:
        modality = "ATAC"
        adata_omics2 = sc.read_h5ad(args.ATAC_path)
    else:
        raise ValueError("Either --ADT_path or --ATAC_path must be provided.")
    adata_omics2.var_names_make_unique()

    # === 固定随机种子 ===
    fix_seed(args.seed)

        # === RNA preprocessing ===
    sc.pp.filter_genes(adata_omics1, min_cells=10)
    sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata_omics1, target_sum=1e4)
    sc.pp.log1p(adata_omics1)
    sc.pp.scale(adata_omics1)

    adata_omics1_high = adata_omics1[:, adata_omics1.var["highly_variable"]]
    n_comps = 30
    adata_omics1.obsm["feat"] = pca(adata_omics1_high, n_comps=n_comps)

    # === ADT preprocessing ===
    if modality == "ADT":
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        n_comps_adt = min(n_comps, adata_omics2.X.shape[1])  # 不超过特征数
        print(f"Performing PCA on ADT with n_comps = {n_comps_adt}")
        adata_omics2.obsm["feat"] = pca(adata_omics2, n_comps=n_comps_adt)

    # === ATAC preprocessing ===
    elif modality == "ATAC":
        if "X_lsi" not in adata_omics2.obsm.keys():
            sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
            lsi(adata_omics2, use_highly_variable=False, n_components=n_comps)
        adata_omics2.obsm["feat"] = adata_omics2.obsm["X_lsi"].copy()

    # === RNA preprocessing（用于 RNA+RNA 场景，确保有PCA特征）===
    elif modality == "RNA":
        if "feat" not in adata_omics2.obsm.keys():
            sc.pp.normalize_total(adata_omics2)
            sc.pp.log1p(adata_omics2)
            sc.pp.scale(adata_omics2)
            n_comps_rna = min(n_comps, adata_omics2.X.shape[1])
            print(f"Performing PCA on RNA with n_comps = {n_comps_rna}")
            adata_omics2.obsm["feat"] = pca(adata_omics2, n_comps=n_comps_rna)

    # === 对齐 RNA 和 另一组学（ADT / ATAC / RNA）的特征维度 ===
    dim1 = adata_omics1.obsm["feat"].shape[1]
    dim2 = adata_omics2.obsm["feat"].shape[1]
    if dim1 != dim2:
        target_dim = min(dim1, dim2)
        print(f"[Dimension Alignment] RNA dim = {dim1}, {modality} dim = {dim2} → align to {target_dim}")
        adata_omics1.obsm["feat"] = adata_omics1.obsm["feat"][:, :target_dim]
        adata_omics2.obsm["feat"] = adata_omics2.obsm["feat"][:, :target_dim]
    else:
        print(f"[Dimension Alignment] Both modalities already have {dim1} dims")


    # === 构建图结构 ===
    data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=args.data_type)

    # === 训练模型 ===
    model = Train_SpaBalance(data, datatype=args.data_type, device=device)
    start_time = time.time()
    output = model.train()
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f}s")

    # === 结果整合 ===
    adata = adata_omics1.copy()
    adata.obsm["SpaBalance"] = output["SpaBalance"].copy()
    adata.uns["train_time"] = train_time

    # === 检查 NaN/Inf ===
    emb = adata.obsm["SpaBalance"]
    if np.any(~np.isfinite(emb)):
        print("Warning: Found NaN/Inf in embedding, cleaning...")
        emb[np.isnan(emb)] = 0
        emb[np.isinf(emb)] = np.sign(emb[np.isinf(emb)]) * 1e10
        adata.obsm["SpaBalance"] = emb

    # === 解析数据集名称 ===
    dataset_name, subset_name = parse_dataset_info(args)
    print(f"Detected dataset: {dataset_name}, subset: {subset_name}")

    # === 聚类 & 可视化 ===
    sc.pp.neighbors(adata, use_rep="SpaBalance", n_neighbors=30)
    sc.tl.umap(adata)

    tools = ["mclust", "louvain", "leiden", "kmeans"]
    plot_dir = os.path.join("Results/plot/SpaBalance", dataset_name, subset_name)
    os.makedirs(plot_dir, exist_ok=True)

    for tool in tools:
        adata = universal_clustering(
            adata,
            n_clusters=args.cluster_nums,
            used_obsm="SpaBalance",
            method=tool,
            key=tool,
            use_pca=False,
        )
        # flip y-axis for spatial
        if "spatial" in adata.obsm.keys():
            adata.obsm["spatial"][:, 1] = -adata.obsm["spatial"][:, 1]

        fig, axs = plt.subplots(1, 2, figsize=(7, 3))
        sc.pl.umap(adata, color=tool, ax=axs[0], title=f"SpaBalance-{tool}", s=20, show=False)
        sc.pl.embedding(adata, basis="spatial", color=tool, ax=axs[1], title=f"SpaBalance-{tool}", s=20, show=False)
        plt.tight_layout(w_pad=0.3)
        plt.savefig(os.path.join(plot_dir, f"clustering_{tool}_umap_spatial.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # === 保存结果 ===
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    adata.write(args.save_path)
    print(f"Saved integrated AnnData to: {args.save_path}")


if __name__ == "__main__":
    os.environ['R_HOME'] = '/home/zhenghong/miniconda3/envs/smobench/lib/R'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser(description="Run SpaBalance vertical integration")
    parser.add_argument("--data_type", type=str, required=True, help="Data type, e.g. 10x, SPOTS, MISAR")
    parser.add_argument("--RNA_path", type=str, required=True)
    parser.add_argument("--ADT_path", type=str, default="")
    parser.add_argument("--ATAC_path", type=str, default="")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--method", type=str, default="SpaBalance")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--cluster_nums", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=2022)

    args = parser.parse_args()
    main(args)
