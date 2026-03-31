import os
import torch
import pandas as pd
import scanpy as sc
import argparse
import time
import sys
import re
import matplotlib.pyplot as plt
import numpy as np

# Add project root directory to module search path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

# Add SpaMV to path
spamv_path = os.path.join(project_root, "Methods/SpaMV")
sys.path.append(spamv_path)

from SpaMV.spamv import SpaMV
from SpaMV.utils import preprocess_dc
from Utils.SMOBench_clustering import universal_clustering


def parse_dataset_info(args):
    """Extract dataset_name and subset_name from fusion paths"""
    if hasattr(args, 'dataset') and args.dataset:
        return args.dataset, "fusion"
    
    # Auto parse from RNA_path
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
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    
    # === Load Fusion Data ===
    adata_rna = sc.read_h5ad(args.RNA_path)  # RNA fusion data
    adata_rna.var_names_make_unique()

    if args.ADT_path:
        adata_adt = sc.read_h5ad(args.ADT_path)  # ADT fusion data
        modality = 'ADT'
        modality_name = 'Proteome'
        adata_adt.var_names_make_unique()
        datasets = [adata_rna, adata_adt]
        omics_names = ['Transcriptome', modality_name]
        # For protein datasets, use scaling
        scale = True
    elif args.ATAC_path:
        adata_atac = sc.read_h5ad(args.ATAC_path)  # ATAC fusion data
        modality = 'ATAC'
        modality_name = 'Epigenome'
        adata_atac.var_names_make_unique()
        datasets = [adata_rna, adata_atac]
        omics_names = ['Transcriptome', modality_name]
        # For epigenome datasets, no scaling
        scale = False
    else:
        raise ValueError("Either ADT_path or ATAC_path must be provided.")

    print(f"Processing horizontal integration: {omics_names[0]} + {omics_names[1]} fusion data...")

    # === Check for batch information in fusion data ===
    print("Checking batch information for horizontal integration...")
    for i, adata in enumerate(datasets):
        if 'batch' in adata.obs.columns:
            print(f"Batch distribution in {omics_names[i]}: {adata.obs['batch'].value_counts()}")
            print("Horizontal integration will address batch effects between these batches")
        else:
            print(f"Warning: No 'batch' column found in {omics_names[i]} fusion data. Adding default batch labels.")
            # Add default batch information if not present
            n_cells = adata.n_obs
            adata.obs['batch'] = ['batch_1'] * (n_cells // 2) + ['batch_2'] * (n_cells - n_cells // 2)

    # === Ensure both datasets have same cells ===
    common_obs = datasets[0].obs_names.intersection(datasets[1].obs_names)
    print(f"Common cells: {len(common_obs)}, RNA cells: {datasets[0].n_obs}, {modality_name} cells: {datasets[1].n_obs}")
    datasets[0] = datasets[0][common_obs].copy()
    datasets[1] = datasets[1][common_obs].copy()
    print(f"After intersection - RNA: {datasets[0].n_obs}, {modality_name}: {datasets[1].n_obs}")

    # === Add spatial coordinates for horizontal integration if not present ===
    for i, adata in enumerate(datasets):
        if 'spatial' not in adata.obsm.keys():
            print(f"Warning: No spatial coordinates found in {omics_names[i]}. Generating pseudo-spatial coordinates for horizontal integration...")
            # Generate pseudo-spatial coordinates based on UMAP
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
            adata.obsm['spatial'] = adata.obsm['X_umap'].copy()
            print(f"Generated pseudo-spatial coordinates for {omics_names[i]}")

    # === Preprocessing using SpaMV's built-in function (adapted for horizontal integration) ===
    print("Preprocessing datasets using SpaMV preprocess_dc for horizontal integration...")
    datasets = preprocess_dc(
        datasets, 
        omics_names, 
        prune=True, 
        min_cells=10,
        min_genes=200, 
        hvg=True, 
        n_top_genes=3000,  # Keep more genes for horizontal integration
        normalization=True,
        target_sum=1e4, 
        log1p=True, 
        scale=scale
    )

    # === Initialize and Train SpaMV Model (horizontal integration specific) ===
    print("Initializing SpaMV model for horizontal integration...")
    model = SpaMV(
        adatas=datasets, 
        interpretable=False,  # Use non-interpretable mode for batch effect removal
        device=device,
        random_seed=args.seed,
        max_epochs_stage1=500,  # Increased epochs for horizontal integration
        max_epochs_stage2=500,  # Increased epochs for horizontal integration
        early_stopping=True,
        patience=250  # Increased patience for horizontal integration
    )

    print("Training SpaMV model for horizontal integration...")
    start_time = time.time()
    model.train()
    end_time = time.time()
    train_time = end_time - start_time
    print('Horizontal integration training time:', train_time)

    # === Get Embeddings ===
    print("Getting embeddings from horizontal integration...")
    embeddings = model.get_embedding()
    
    # === Build Result AnnData ===
    # Use the first dataset (RNA) as the base
    adata = datasets[0].copy()
    adata.obsm['SpaMV'] = embeddings
    adata.uns['train_time'] = train_time
    adata.uns['integration_type'] = 'horizontal'
    
    # === Clean embeddings for clustering ===
    embeddings_clean = adata.obsm['SpaMV'].copy()
    
    # Check for and handle infinite/NaN values
    if np.any(~np.isfinite(embeddings_clean)):
        print("Warning: Found infinite or NaN values in embeddings. Cleaning...")
        # Replace inf with large finite values
        embeddings_clean[np.isinf(embeddings_clean)] = np.sign(embeddings_clean[np.isinf(embeddings_clean)]) * 1e10
        # Replace NaN with zeros
        embeddings_clean[np.isnan(embeddings_clean)] = 0
        adata.obsm['SpaMV'] = embeddings_clean
        print(f"Cleaned embeddings shape: {embeddings_clean.shape}")
    else:
        print("Embeddings are clean (no inf/NaN values)")
    
    # === Parse Dataset Info ===
    dataset_name, subset_name = parse_dataset_info(args)
    print(f"Detected dataset: {dataset_name}, subset: {subset_name}")

    # === Plot Save Path ===
    plot_base_dir = "Results/plot"
    method_name = args.method if args.method else "SpaMV"
    plot_dir = os.path.join(plot_base_dir, "horizontal_integration", method_name, dataset_name, subset_name)
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Plot images will be saved to: {plot_dir}")

    # === Clustering and Visualization ===
    tools = ['mclust', 'louvain', 'leiden', 'kmeans']
    
    # === Generate UMAP for visualization (only once) ===
    sc.pp.neighbors(adata, use_rep='SpaMV', n_neighbors=30)
    sc.tl.umap(adata)
    
    for tool in tools:
        adata = universal_clustering(
            adata,
            n_clusters=args.cluster_nums,
            used_obsm='SpaMV',
            method=tool,
            key=tool,
            use_pca=False
        )
        
        # Flip spatial coordinates for visualization if available
        if 'spatial' in adata.obsm.keys():
            adata.obsm['spatial'][:, 1] = -1 * adata.obsm['spatial'][:, 1]

        fig, ax_list = plt.subplots(1, 3, figsize=(10, 3))

        # Plot UMAP, spatial, and batch
        sc.pl.umap(adata, color=tool, ax=ax_list[0], title=f'{method_name}-{tool}', s=20, show=False)
        if 'spatial' in adata.obsm.keys():
            sc.pl.embedding(adata, basis='spatial', color=tool, ax=ax_list[1], title=f'{method_name}-{tool}', s=20, show=False)
        else:
            sc.pl.umap(adata, color=tool, ax=ax_list[1], title=f'{method_name}-{tool} (no spatial)', s=20, show=False)
        
        # Plot batch distribution for horizontal integration evaluation
        if 'batch' in adata.obs.columns:
            sc.pl.umap(adata, color='batch', ax=ax_list[2], title=f'{method_name} batch', s=20, show=False)
        else:
            sc.pl.umap(adata, color=tool, ax=ax_list[2], title=f'{method_name}-{tool} (no batch)', s=20, show=False)

        plt.tight_layout(w_pad=0.3)
        plt.savefig(
            os.path.join(plot_dir, f'clustering_{tool}_umap_spatial_batch.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    # === Save AnnData ===
    save_dir = os.path.dirname(args.save_path)
    os.makedirs(save_dir, exist_ok=True)
    adata.write(args.save_path)
    print(adata)
    print('Saving horizontal integration results to...', args.save_path)


if __name__ == "__main__":
    # Set environment variables for R and threading
    os.environ['R_HOME'] = '/home/zhenghong/miniconda3/envs/smobench/lib/R'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    print("Starting SpaMV horizontal integration...")
    parser = argparse.ArgumentParser(description='Run SpaMV horizontal integration')
    parser.add_argument('--data_type', type=str, default='fusion', help='Data type for horizontal integration')
    parser.add_argument('--RNA_path', type=str, required=True, help='Path to RNA fusion adata')
    parser.add_argument('--ADT_path', type=str, default='', help='Path to ADT fusion adata')
    parser.add_argument('--ATAC_path', type=str, default='', help='Path to ATAC fusion adata')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save integrated adata')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, e.g. cuda:0 or cpu')

    parser.add_argument('--method', type=str, default='SpaMV', help='Method name for plotting')
    parser.add_argument('--dataset', type=str, default='', help='Dataset name for horizontal integration')

    parser.add_argument('--cluster_nums', type=int, help='Number of clusters')

    args = parser.parse_args()
    main(args)