#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import torch
import argparse
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import re
import logging
# Configure basic logging to track progress during heavy computations
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# === 项目路径设置 ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

smopca_path = os.path.join(project_root, "_myx_Methods/SMOPCA/src")
sys.path.append(smopca_path)

from Methods.SpatialGlue.preprocess import fix_seed
from Utils.SMOBench_clustering import universal_clustering
import model
import utils


# === 自动解析数据集信息 ===
def parse_dataset_info(args):
    if hasattr(args, 'dataset') and args.dataset:
        parts = args.dataset.strip('/').split('/')
        if len(parts) == 2:
            return parts[0], parts[1]
        elif len(parts) == 1:
            return parts[0], "Unknown"
    match = re.search(r'SMOBench_Data/([^/]+)/([^/]+)/adata_RNA\.h5ad', args.RNA_path)
    if match:
        return match.group(1), match.group(2)
    return "Unknown", "Unknown"


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    fix_seed(args.seed)

    # === 读取数据 ===
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
        raise ValueError("You must specify either --ADT_path or --ATAC_path.")

    adata_other.var_names_make_unique()
    print(f"Performing SMOPCA vertical integration of RNA + {modality_name}")

    # === 匹配细胞 ===
    common_cells = adata_rna.obs_names.intersection(adata_other.obs_names)
    print(f"Common cells found: {len(common_cells)}")
    adata_rna = adata_rna[common_cells].copy()
    adata_other = adata_other[common_cells].copy()

    # === 预处理 ===
    print("Preprocessing data...")
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.highly_variable_genes(adata_rna, n_top_genes=3000)
    adata_rna = adata_rna[:, adata_rna.var["highly_variable"]].copy()

    sc.pp.scale(adata_rna)
    sc.pp.scale(adata_other)

    X1 = adata_rna.X.A if hasattr(adata_rna.X, "A") else adata_rna.X
    X2 = adata_other.X.A if hasattr(adata_other.X, "A") else adata_other.X

    # === 确保存在空间坐标 ===
    if "spatial" not in adata_rna.obsm:
        print("No spatial coordinates found, using pseudo coordinates from UMAP.")
        sc.pp.neighbors(adata_rna)
        sc.tl.umap(adata_rna)
        adata_rna.obsm["spatial"] = adata_rna.obsm["X_umap"]

    pos = adata_rna.obsm["spatial"]

    # === 训练SMOPCA ===
    print("Training SMOPCA...")
    # Ensure latent dimension does not exceed any modality's feature count
    max_latent_dim = min(args.Z_dim, X1.shape[1], X2.shape[1])
    if max_latent_dim < args.Z_dim:
        logging.info(f"Adjusting Z_dim from {args.Z_dim} to {max_latent_dim} to match modality feature dimensions.")

    smopca = model.SMOPCA(Y_list=[X1.T, X2.T], Z_dim=max_latent_dim, pos=pos,
                          intercept=False, omics_weight=False)

    start_time = time.time()
    smopca.estimateParams(
        sigma_init_list=(1, 1),
        tol_sigma=2e-5,
        sigma_xtol_list=(1e-6, 1e-6),
        gamma_init=1,
        estimate_gamma=True
    )
    z = smopca.calculatePosterior()
    end_time = time.time()
    train_time = end_time - start_time
    print(f"Training completed in {train_time:.2f}s")

    # === 构建 AnnData ===
    adata = sc.AnnData(z)
    adata.obs = adata_rna.obs.copy()
    adata.obs_names = adata_rna.obs_names.copy()
    adata.var_names = [f"SMOPCA_{i}" for i in range(adata.shape[1])]
    adata.obsm["SMOPCA"] = np.asarray(z)
    adata.obsm["spatial"] = pos
    adata.uns["train_time"] = train_time
    adata.uns["method"] = "SMOPCA"

    # === 解析数据集信息 ===
    dataset_name, subset_name = parse_dataset_info(args)
    print(f"Detected dataset: {dataset_name}, subset: {subset_name}")

    # === 聚类与可视化 ===
    tools = ['mclust', 'louvain', 'leiden', 'kmeans']
    for tool in tools:
        adata = universal_clustering(
            adata,
            n_clusters=args.cluster_nums,
            used_obsm='SMOPCA',
            method=tool,
            key=tool,
            use_pca=False
        )

        adata.obsm['spatial'][:, 1] = -1 * adata.obsm['spatial'][:, 1]
        fig, ax_list = plt.subplots(1, 2, figsize=(7, 3))
        sc.pp.neighbors(adata, use_rep='SMOPCA', n_neighbors=30)
        sc.tl.umap(adata)
        sc.pl.umap(adata, color=tool, ax=ax_list[0], title=f'SMOPCA-{tool}', s=20, show=False)
        sc.pl.embedding(adata, basis='spatial', color=tool, ax=ax_list[1], title=f'SMOPCA-{tool}', s=20, show=False)
        plt.tight_layout(w_pad=0.3)

        plot_dir = os.path.join("Results/plot/SMOPCA", dataset_name, subset_name)
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f'clustering_{tool}_umap_spatial.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # === 保存结果 ===
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    adata.write(args.save_path)
    print(f"Saved integrated data to {args.save_path}")


if __name__ == "__main__":
    os.environ['R_HOME'] = '/home/zhenghong/miniconda3/envs/smobench/lib/R'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    parser = argparse.ArgumentParser(description="Run SMOPCA vertical integration")
    parser.add_argument("--data_type", type=str, default="10x", help="Data type, e.g. 10x, SPOTS, MISAR")
    parser.add_argument("--RNA_path", type=str, required=True)
    parser.add_argument("--ADT_path", type=str, default="")
    parser.add_argument("--ATAC_path", type=str, default="")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--Z_dim", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cluster_nums", type=int, required=True)
    parser.add_argument("--dataset", type=str, default="")
    args = parser.parse_args()

    main(args)
