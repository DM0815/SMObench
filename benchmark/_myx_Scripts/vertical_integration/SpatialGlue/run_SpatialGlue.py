import os
import torch
import pandas as pd
import scanpy as sc
import argparse
import time
import sys
import re
import matplotlib.pyplot as plt

# 将项目根目录加入模块搜索路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

# Add SpatialGlue to path
spatialglue_path = os.path.join(project_root, "Methods/SpatialGlue")
sys.path.append(spatialglue_path)

from preprocess import fix_seed
from preprocess import clr_normalize_each_cell, pca, lsi
from preprocess import construct_neighbor_graph
from SpatialGlue_pyG import Train_SpatialGlue
from Utils.SMOBench_clustering import universal_clustering


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
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    # === 读取数据 ===
    adata_omics1 = sc.read_h5ad(args.RNA_path)
    adata_omics1.var_names_make_unique()

    if args.ADT_path:
        adata_omics2 = sc.read_h5ad(args.ADT_path)
        modality = 'ADT'
        adata_omics2.var_names_make_unique()
    elif args.ATAC_path:
        adata_omics2 = sc.read_h5ad(args.ATAC_path)
        modality = 'ATAC'
        adata_omics2.var_names_make_unique()
    else:
        raise ValueError("Either ADT_path or ATAC_path must be provided.")

    # === 固定种子 ===
    fix_seed(args.seed)

    # === 预处理 RNA ===
    sc.pp.filter_genes(adata_omics1, min_cells=10)
    sc.pp.normalize_total(adata_omics1, target_sum=1e4)
    sc.pp.log1p(adata_omics1)
    sc.pp.highly_variable_genes(adata_omics1, n_top_genes=3000)
    sc.pp.scale(adata_omics1)

    if args.data_type == 'SPOTS':
        n_comps = 20
    elif args.data_type == 'Stereo-CITE-seq':
        n_comps = 10
    else:
        n_comps = 30

    adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
    adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=n_comps)

    # === 预处理第二组学 ===
    if modality == 'ADT':
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=n_comps)
    elif modality == 'ATAC':
        if 'X_lsi' not in adata_omics2.obsm.keys():
            sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
            lsi(adata_omics2, use_highly_variable=False, n_components=n_comps)
        adata_omics2.obsm['feat'] = adata_omics2.obsm['X_lsi'].copy()

    # === 构建图 ===
    data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=args.data_type)

    # === 训练模型 ===
    model = Train_SpatialGlue(data, datatype=args.data_type, device=device)
    
    start_time = time.time()
    output = model.train()
    end_time = time.time()
    train_time = end_time - start_time
    print('training time:', train_time)

    # === 构建结果 AnnData ===
    adata = adata_omics1.copy()
    adata.obsm['emb_latent_omics1'] = output['emb_latent_omics1'].copy()
    adata.obsm['emb_latent_omics2'] = output['emb_latent_omics2'].copy()
    adata.obsm['SpatialGlue'] = output['SpatialGlue'].copy()
    adata.uns['train_time'] = train_time
    
    # === Clean embeddings for clustering ===
    import numpy as np
    embeddings_clean = adata.obsm['SpatialGlue'].copy()
    
    # Check for and handle infinite/NaN values
    if np.any(~np.isfinite(embeddings_clean)):
        print("Warning: Found infinite or NaN values in embeddings. Cleaning...")
        embeddings_clean[np.isinf(embeddings_clean)] = np.sign(embeddings_clean[np.isinf(embeddings_clean)]) * 1e10
        embeddings_clean[np.isnan(embeddings_clean)] = 0
        adata.obsm['SpatialGlue'] = embeddings_clean
        print(f"Cleaned embeddings shape: {embeddings_clean.shape}")
    else:
        print("Embeddings are clean (no inf/NaN values)")

    # === 解析数据集信息 ===
    dataset_name, subset_name = parse_dataset_info(args)
    print(f"Detected dataset: {dataset_name}, subset: {subset_name}")

    # === 图像保存路径 ===
    plot_base_dir = "Results/plot"
    method_name = args.method if args.method else "SpatialGlue"
    plot_dir = os.path.join(plot_base_dir, method_name, dataset_name, subset_name)
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Plot images will be saved to: {plot_dir}")

    # === 计算UMAP (只计算一次) ===
    sc.pp.neighbors(adata, use_rep='SpatialGlue', n_neighbors=30)
    sc.tl.umap(adata)

    # === 聚类与可视化 ===
    tools = ['mclust', 'louvain', 'leiden', 'kmeans']
    for tool in tools:
        adata = universal_clustering(
            adata,
            n_clusters=args.cluster_nums,
            used_obsm='SpatialGlue',
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
    # the location of R, which is required for the 'mclust' algorithm. Please replace the path below with local R installation path
    os.environ['R_HOME'] = '/home/zhenghong/miniconda3/envs/smobench/lib/R'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    print("Starting...")
    parser = argparse.ArgumentParser(description='Run SpatialGlue integration')
    parser.add_argument('--data_type', type=str, default='10x', help='Data type, e.g. 10x, SPOTS, MISAR')
    parser.add_argument('--RNA_path', type=str, required=True, help='Path to RNA adata')
    parser.add_argument('--ADT_path', type=str, default='', help='Path to ADT adata')
    parser.add_argument('--ATAC_path', type=str, default='', help='Path to ATAC adata')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save integrated adata')
    parser.add_argument('--seed', type=int, default=2022, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, e.g. cuda:0 or cpu')

    parser.add_argument('--method', type=str, default='SpatialGlue', help='Method name for plotting')
    parser.add_argument('--dataset', type=str, default='', help='Dataset name, e.g. Human_Lymph_Nodes/A1. If not provided, auto-extracted from paths.')

    parser.add_argument('--cluster_nums', type=int, help='cluster_nums')

    args = parser.parse_args()
    main(args)