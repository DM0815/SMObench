import os
import torch
import pandas as pd
import scanpy as sc
import argparse
import time
import sys
import re
import matplotlib.pyplot as plt

# Add project root directory to module search path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

# Add COSMOS to path
cosmos_path = os.path.join(project_root, "Methods/COSMOS")
sys.path.append(cosmos_path)

from COSMOS import cosmos
from Utils.SMOBench_clustering import universal_clustering


def parse_dataset_info(args):
    """
    Extract dataset_name and subset_name from RNA_path or save_path
    Support two modes:
    1. Manual specification --dataset Human_Lymph_Nodes/A1
    2. Auto extraction from paths
    """
    if hasattr(args, 'dataset') and args.dataset:
        parts = args.dataset.strip('/').split('/')
        if len(parts) == 2:
            return parts[0], parts[1]
        elif len(parts) == 1:
            return parts[0], "Unknown"
    
    # Auto parse RNA_path
    match = re.search(r'Dataset/([^/]+)/([^/]+)/([^/]+)/adata_RNA\.h5ad', args.RNA_path)
    if match:
        return match.group(2), match.group(3)
    return "Unknown", "Unknown"


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    
    # === Load Data ===
    adata_rna = sc.read_h5ad(args.RNA_path)
    adata_rna.var_names_make_unique()

    if args.ADT_path:
        adata_adt = sc.read_h5ad(args.ADT_path)
        modality = 'ADT'
        modality_name = 'Proteome'
        adata_adt.var_names_make_unique()
        second_adata = adata_adt
    elif args.ATAC_path:
        adata_atac = sc.read_h5ad(args.ATAC_path)
        modality = 'ATAC'
        modality_name = 'Epigenome'
        adata_atac.var_names_make_unique()
        second_adata = adata_atac
    else:
        raise ValueError("Either ADT_path or ATAC_path must be provided.")

    print(f"Processing {args.data_type}: RNA + {modality_name} integration...")

    # === Ensure both datasets have same cells ===
    common_obs = adata_rna.obs_names.intersection(second_adata.obs_names)
    print(f"Common cells: {len(common_obs)}, RNA cells: {adata_rna.n_obs}, {modality_name} cells: {second_adata.n_obs}")
    adata_rna = adata_rna[common_obs].copy()
    second_adata = second_adata[common_obs].copy()
    print(f"After intersection - RNA: {adata_rna.n_obs}, {modality_name}: {second_adata.n_obs}")

    # === Initialize COSMOS model ===
    print("Initializing COSMOS model...")
    cosmos_model = cosmos.Cosmos(adata1=adata_rna, adata2=second_adata)
    
    # === Preprocessing ===
    print("Preprocessing data...")
    cosmos_model.preprocessing_data(
        do_norm=True,
        do_log=True,
        n_top_genes=3000,
        do_pca=False,
        n_neighbors=10
    )
    
    # === Train COSMOS model ===
    print("Training COSMOS model...")
    start_time = time.time()
    
    # Extract GPU ID from device string
    gpu_id = 0
    if 'cuda:' in args.device:
        gpu_id = int(args.device.split(':')[1])
    
    embedding = cosmos_model.train(
        spatial_regularization_strength=0.01,
        z_dim=50,
        lr=1e-3,
        wnn_epoch=500,
        total_epoch=1000,
        max_patience_bef=10,
        max_patience_aft=30,
        min_stop=200,
        random_seed=args.seed,
        gpu=gpu_id,
        regularization_acceleration=True,
        edge_subset_sz=1000000
    )
    
    end_time = time.time()
    train_time = end_time - start_time
    print('Training time:', train_time)

    # === Build Result AnnData ===
    # Use the first dataset (RNA) as the base
    adata = adata_rna.copy()
    adata.obsm['COSMOS'] = embedding
    adata.uns['train_time'] = train_time
    
    # === Clean embeddings for clustering ===
    import numpy as np
    embeddings_clean = adata.obsm['COSMOS'].copy()
    
    # Check for and handle infinite/NaN values
    if np.any(~np.isfinite(embeddings_clean)):
        print("Warning: Found infinite or NaN values in embeddings. Cleaning...")
        embeddings_clean[np.isinf(embeddings_clean)] = np.sign(embeddings_clean[np.isinf(embeddings_clean)]) * 1e10
        embeddings_clean[np.isnan(embeddings_clean)] = 0
        adata.obsm['COSMOS'] = embeddings_clean
        print(f"Cleaned embeddings shape: {embeddings_clean.shape}")
    else:
        print("Embeddings are clean (no inf/NaN values)")
    
    # === Parse Dataset Info ===
    dataset_name, subset_name = parse_dataset_info(args)
    print(f"Detected dataset: {dataset_name}, subset: {subset_name}")

    # === Plot Save Path ===
    plot_base_dir = "Results/plot"
    method_name = args.method if args.method else "COSMOS"
    plot_dir = os.path.join(plot_base_dir, method_name, dataset_name, subset_name)
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Plot images will be saved to: {plot_dir}")

    # === Clustering and Visualization ===
    tools = ['mclust', 'louvain', 'leiden', 'kmeans']
    
    # === Generate UMAP for visualization (only once) ===
    sc.pp.neighbors(adata, use_rep='COSMOS', n_neighbors=30)
    sc.tl.umap(adata)
    
    for tool in tools:
        adata = universal_clustering(
            adata,
            n_clusters=args.cluster_nums,
            used_obsm='COSMOS',
            method=tool,
            key=tool,
            use_pca=False
        )
        
        # Flip spatial coordinates for visualization if available
        if 'spatial' in adata.obsm.keys():
            adata.obsm['spatial'][:, 1] = -1 * adata.obsm['spatial'][:, 1]

        fig, ax_list = plt.subplots(1, 2, figsize=(7, 3))

        # Plot UMAP and spatial
        sc.pl.umap(adata, color=tool, ax=ax_list[0], title=f'{method_name}-{tool}', s=20, show=False)
        if 'spatial' in adata.obsm.keys():
            sc.pl.embedding(adata, basis='spatial', color=tool, ax=ax_list[1], title=f'{method_name}-{tool}', s=20, show=False)
        else:
            # If no spatial coordinates, plot UMAP again
            sc.pl.umap(adata, color=tool, ax=ax_list[1], title=f'{method_name}-{tool} (no spatial)', s=20, show=False)

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
    
    # === Save AnnData ===
    save_dir = os.path.dirname(args.save_path)
    os.makedirs(save_dir, exist_ok=True)
    adata.write(args.save_path)
    print(adata)
    print('Saving results to...', args.save_path)


if __name__ == "__main__":
    # Set environment variables for R and threading
    os.environ['R_HOME'] = '/home/zhenghong/miniconda3/envs/smobench/lib/R'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    print("Starting COSMOS integration...")
    parser = argparse.ArgumentParser(description='Run COSMOS integration')
    parser.add_argument('--data_type', type=str, default='10x', help='Data type, e.g. 10x, SPOTS, MISAR, simulation')
    parser.add_argument('--RNA_path', type=str, required=True, help='Path to RNA adata')
    parser.add_argument('--ADT_path', type=str, default='', help='Path to ADT adata')
    parser.add_argument('--ATAC_path', type=str, default='', help='Path to ATAC adata')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save integrated adata')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, e.g. cuda:0 or cpu')

    parser.add_argument('--method', type=str, default='COSMOS', help='Method name for plotting')
    parser.add_argument('--dataset', type=str, default='', help='Dataset name, e.g. Human_Lymph_Nodes/A1. If not provided, auto-extracted from paths.')

    parser.add_argument('--cluster_nums', type=int, help='Number of clusters')

    args = parser.parse_args()
    main(args)