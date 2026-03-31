import os
import torch
import pandas as pd
import scanpy as sc
import argparse
import time
import sys
import re
import matplotlib.pyplot as plt

# Set CUDA environment for SpaMosaic
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Add project root directory to module search path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

# Add SpaMosaic to path
spamosaic_path = os.path.join(project_root, "Methods/SpaMosaic")
sys.path.append(spamosaic_path)

import spamosaic
from spamosaic.framework import SpaMosaic
import spamosaic.utils as utls
from spamosaic.preprocessing import RNA_preprocess, ADT_preprocess, Epigenome_preprocess
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
        modality = 'adt'
        adata_adt.var_names_make_unique()
        input_dict = {
            'rna': [adata_rna],
            'adt': [adata_adt]
        }
    elif args.ATAC_path:
        adata_atac = sc.read_h5ad(args.ATAC_path)
        modality = 'atac'
        adata_atac.var_names_make_unique()
        input_dict = {
            'rna': [adata_rna],
            'atac': [adata_atac]
        }
    else:
        raise ValueError("Either ADT_path or ATAC_path must be provided.")

    # === Set batch key for SpaMosaic ===
    # SpaMosaic requires a batch key for preprocessing
    # For vertical integration, all data come from the same section/batch
    for key in input_dict:
        for adata in input_dict[key]:
            if adata is not None and 'src' not in adata.obs.columns:
                adata.obs['src'] = 'batch0'  # Use consistent batch name for vertical integration

    # === Preprocessing ===
    input_key = 'dimred_bc'
    
    # RNA preprocessing
    RNA_preprocess(
        input_dict['rna'], 
        batch_corr=True, 
        favor='scanpy', 
        n_hvg=5000, 
        batch_key='src', 
        key=input_key
    )
    
    # Secondary modality preprocessing
    if modality == 'adt':
        ADT_preprocess(
            input_dict['adt'], 
            batch_corr=True, 
            batch_key='src', 
            key=input_key
        )
    elif modality == 'atac':
        Epigenome_preprocess(
            input_dict['atac'], 
            batch_corr=True, 
            batch_key='src', 
            key=input_key
        )

    # === Train SpaMosaic Model ===
    print("Initializing SpaMosaic model...")
    # For vertical integration (single section, multi-modal), we don't need inter_knn_base and w_g
    model = SpaMosaic(
        modBatch_dict=input_dict, 
        input_key=input_key,
        batch_key='src', 
        intra_knns=10,
        seed=args.seed,
        device=args.device
    )

    print("Training SpaMosaic model...")
    start_time = time.time()
    model.train(net='wlgcn', lr=0.01, T=0.01, n_epochs=100)
    end_time = time.time()
    train_time = end_time - start_time
    print('Training time:', train_time)

    # === Get Embeddings ===
    print("Inferring embeddings...")
    ad_embs = model.infer_emb(input_dict, emb_key='emb', final_latent_key='merged_emb')
    
    # For vertical integration, ad_embs contains one element (the integrated section)
    adata = ad_embs[0].copy()
    
    # Store embeddings with consistent naming
    adata.obsm['SpaMosaic'] = adata.obsm['merged_emb'].copy()
    adata.uns['train_time'] = train_time
    
    # === Clean embeddings for clustering ===
    import numpy as np
    embeddings_clean = adata.obsm['SpaMosaic'].copy()
    
    # Check for and handle infinite/NaN values
    if np.any(~np.isfinite(embeddings_clean)):
        print("Warning: Found infinite or NaN values in embeddings. Cleaning...")
        embeddings_clean[np.isinf(embeddings_clean)] = np.sign(embeddings_clean[np.isinf(embeddings_clean)]) * 1e10
        embeddings_clean[np.isnan(embeddings_clean)] = 0
        adata.obsm['SpaMosaic'] = embeddings_clean
        print(f"Cleaned embeddings shape: {embeddings_clean.shape}")
    else:
        print("Embeddings are clean (no inf/NaN values)")
    
    # Get UMAP embeddings
    ad_mosaic = sc.concat(ad_embs)
    ad_mosaic = utls.get_umap(ad_mosaic, use_reps=['merged_emb'])
    adata.obsm['merged_emb_umap'] = ad_mosaic.obsm['merged_emb_umap'].copy()

    # === Parse Dataset Info ===
    dataset_name, subset_name = parse_dataset_info(args)
    print(f"Detected dataset: {dataset_name}, subset: {subset_name}")

    # === Plot Save Path ===
    plot_base_dir = "Results/plot"
    method_name = args.method if args.method else "SpaMosaic"
    plot_dir = os.path.join(plot_base_dir, method_name, dataset_name, subset_name)
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Plot images will be saved to: {plot_dir}")

    # === Clustering and Visualization ===
    tools = ['mclust', 'louvain', 'leiden', 'kmeans']
    
    # === Generate UMAP for visualization (only once) ===
    sc.pp.neighbors(adata, use_rep='SpaMosaic', n_neighbors=30)
    sc.tl.umap(adata)
    
    for tool in tools:
        adata = universal_clustering(
            adata,
            n_clusters=args.cluster_nums,
            used_obsm='SpaMosaic',
            method=tool,
            key=tool,
            use_pca=False
        )
        
        # Flip spatial coordinates for visualization
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

    print("Starting SpaMosaic integration...")
    parser = argparse.ArgumentParser(description='Run SpaMosaic integration')
    parser.add_argument('--data_type', type=str, default='10x', help='Data type, e.g. 10x, SPOTS, MISAR')
    parser.add_argument('--RNA_path', type=str, required=True, help='Path to RNA adata')
    parser.add_argument('--ADT_path', type=str, default='', help='Path to ADT adata')
    parser.add_argument('--ATAC_path', type=str, default='', help='Path to ATAC adata')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save integrated adata')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, e.g. cuda:0 or cpu')

    parser.add_argument('--method', type=str, default='SpaMosaic', help='Method name for plotting')
    parser.add_argument('--dataset', type=str, default='', help='Dataset name, e.g. Human_Lymph_Nodes/A1. If not provided, auto-extracted from paths.')

    parser.add_argument('--cluster_nums', type=int, help='Number of clusters')

    args = parser.parse_args()
    main(args)