#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import sys
import time
import torch
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn
import argparse
import scipy
import matplotlib.pyplot as plt
from anndata import AnnData

# === 路径设置 ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

spami_path = os.path.join(project_root, "Methods/SpaMI/SpaMI")
sys.path.append(spami_path)

from preprocess import tfidf, construct_adj, add_contrastive_label
from utils import fix_seed
from main import train
from Utils.SMOBench_clustering import universal_clustering


def detect_modality(modality_path: str) -> str:
    """根据路径名或变量特征判断是 ATAC 还是 ADT"""
    path_lower = modality_path.lower()
    if "atac" in path_lower or "peak" in path_lower:
        return "ATAC"
    elif "adt" in path_lower or "protein" in path_lower:
        return "ADT"
    else:
        # fallback：通过变量数量判断
        try:
            adata = sc.read_h5ad(modality_path)
            if adata.shape[1] < 300:
                return "ADT"
            else:
                return "ATAC"
        except Exception:
            return "ATAC"


def preprocess_rna(adata):
    """标准化 RNA 预处理"""
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    adata = adata[:, adata.var['highly_variable']]
    X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
    X_pca = sklearn.decomposition.PCA(n_components=50).fit_transform(X)
    return X_pca, adata


def preprocess_atac(adata):
    """ATAC 模态的 TF-IDF + SVD 预处理"""
    import episcanpy
    episcanpy.pp.binarize(adata)
    episcanpy.pp.filter_features(adata, min_cells=np.ceil(0.005 * adata.shape[0]))
    count_mat = adata.X.copy()
    X = tfidf(count_mat)
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    U, _, _ = sklearn.utils.extmath.randomized_svd(X_norm, n_components=51)
    X_lsi = (U - U.mean(1, keepdims=True)) / U.std(1, ddof=1, keepdims=True)
    return X_lsi[:, 1:], adata


def preprocess_adt(adata):
    """ADT 模态（protein）预处理，使用 Seurat CLR"""
    sc.pp.filter_genes(adata, min_cells=50)

    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x)) if len(x) > 0 else 1
        return np.log1p(x / exp)

    X = adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X)
    adata.X = np.apply_along_axis(seurat_clr, 1, X)
    return adata.X, adata

def parse_dataset_info(args):
    """
    从 RNA_path 或 save_path 中提取 dataset_name 和 subset_name
    支持两种模式：
    1. 手动指定 --dataset Human_Lymph_Nodes/A1
    2. 自动从路径中提取
    """
    if hasattr(args, 'dataset') and args.dataset:
        parts = args.dataset.strip('/').split('/')
        if len(parts) == 2:
            return parts[0], parts[1]
        elif len(parts) == 1:
            return parts[0], "Unknown"
    
    # 自动解析 RNA_path
    match = re.search(r'SMOBench_Data/([^/]+)/([^/]+)/adata_RNA\.h5ad', args.RNA_path)
    if match:
        return match.group(1), match.group(2)
    return "Unknown", "Unknown"


def main(args):
    fix_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # === 加载数据 ===
    print("Loading data...")
    adata_omics1 = sc.read_h5ad(args.RNA_path)
    adata_omics1.var_names_make_unique()

    if args.ATAC_path and os.path.exists(args.ATAC_path):
        modality_path = args.ATAC_path
    elif args.ADT_path and os.path.exists(args.ADT_path):
        modality_path = args.ADT_path
    else:
        raise ValueError("You must provide either --ATAC_path or --ADT_path")

    adata_omics2 = sc.read_h5ad(modality_path)
    adata_omics2.var_names_make_unique()

    modality = detect_modality(modality_path)
    print(f"Detected modality: {modality}")

    # === 预处理 RNA ===
    feat_omics1, adata_omics1 = preprocess_rna(adata_omics1)

    # === 预处理第二模态 ===
    if modality == "ATAC":
        feat_omics2, adata_omics2 = preprocess_atac(adata_omics2)
    elif modality == "ADT":
        feat_omics2, adata_omics2 = preprocess_adt(adata_omics2)
    else:
        raise ValueError("Unsupported modality type.")

    # === 构图 & contrastive label ===
    adj_omics1, graph_neigh_omics1 = construct_adj(adata_omics1, n_neighbors=3)
    adj_omics2, graph_neigh_omics2 = construct_adj(adata_omics2, n_neighbors=3)
    label_CSL_omics1 = add_contrastive_label(adata_omics1)
    label_CSL_omics2 = add_contrastive_label(adata_omics2)

    omics1_data = {
        'feat': feat_omics1,
        'adj': adj_omics1,
        'graph_neigh': graph_neigh_omics1,
        'label_CSL': label_CSL_omics1,
    }
    omics2_data = {
        'feat': feat_omics2,
        'adj': adj_omics2,
        'graph_neigh': graph_neigh_omics2,
        'label_CSL': label_CSL_omics2,
    }

    # === 模型训练 ===
    print("Training SpaMI...")
    start = time.time()
    embedding=train(omics1_data, omics2_data, args.data_type, out_dim=20)
    train_time = time.time() - start
    print(f"Training done in {train_time:.2f} s")

    
    adata = adata_omics1.copy()
    adata.uns['train_time'] = train_time
    adata.obsm['SpaMI'] = embedding


   # === 解析数据集信息 ===
    dataset_name, subset_name = parse_dataset_info(args)
    print(f"Detected dataset: {dataset_name}, subset: {subset_name}")

    # === 图像保存路径 ===
    plot_base_dir = "Results/plot/vertical_integration/SpaMI"
    method_name = args.method if args.method else "SpaMI"
    plot_dir = os.path.join(plot_base_dir, method_name, dataset_name, subset_name)
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Plot images will be saved to: {plot_dir}")

    # === 计算UMAP (只计算一次) ===
    sc.pp.neighbors(adata, use_rep='SpaMI', n_neighbors=30)
    sc.tl.umap(adata)

    # === 聚类与可视化 ===
    tools = ['mclust', 'louvain', 'leiden', 'kmeans']
    for tool in tools:
        adata = universal_clustering(
            adata,
            n_clusters=args.cluster_nums,
            used_obsm='SpaMI',
            method=tool,
            key=tool,
            use_pca=False
        )
        adata.obsm['spatial'][:, 1] = -1 * adata.obsm['spatial'][:, 1]

        fig, ax_list = plt.subplots(1, 2, figsize=(7, 3))

        sc.pl.umap(adata, color=tool, ax=ax_list[0], title=f'{method_name}-{tool}', s=20, show=False)
        sc.pl.embedding(adata, basis='spatial', color=tool, ax=ax_list[1], title=f'{method_name}-{tool}', s=20, show=False)

        plt.tight_layout(w_pad=0.3)
        plt.savefig(
            os.path.join(plot_dir, f'clustering_{tool}_umap_spatial.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    # === 清理临时变量 ===
    # Remove temporary clustering results from universal_clustering
    temp_cols = [col for col in adata.obs.columns if 'tmp_search' in col]
    for col in temp_cols:
        del adata.obs[col]
        if col in adata.uns:
            del adata.uns[col]
    
    # === 保存 AnnData ===
    save_dir = os.path.dirname(args.save_path)
    os.makedirs(save_dir, exist_ok=True)
    adata.write(args.save_path)
    print(adata)
    print('Saving results to...', args.save_path)


if __name__ == "__main__":
    # Match SMOBench standard scripts: use smobench conda environment R installation
    os.environ['R_HOME'] = '/home/zhenghong/miniconda3/envs/smobench/lib/R'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser(description="Run SpaMI vertical integration (RNA+ATAC / RNA+ADT)")
    parser.add_argument('--data_type', type=str, default='10x', help='Data type, e.g. 10x, SPOTS, MISAR')
    parser.add_argument('--RNA_path', type=str, required=True, help='Path to RNA adata')
    parser.add_argument('--ADT_path', type=str, default='', help='Path to ADT adata')
    parser.add_argument('--ATAC_path', type=str, default='', help='Path to ATAC adata')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save integrated adata')
    parser.add_argument('--dataset', type=str, default='', help='Dataset name')
    parser.add_argument('--cluster_nums', type=int, required=True, help='Number of clusters')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--method', type=str, default='SpatialGlue', help='Method name for plotting')

    args = parser.parse_args()
    main(args)
