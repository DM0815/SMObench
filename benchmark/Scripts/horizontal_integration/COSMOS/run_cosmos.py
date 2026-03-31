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

# Add COSMOS to path
cosmos_path = os.path.join(project_root, "Methods/COSMOS")
sys.path.append(cosmos_path)

from COSMOS import cosmos
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
        second_adata = adata_adt
    elif args.ATAC_path:
        adata_atac = sc.read_h5ad(args.ATAC_path)  # ATAC fusion data
        modality = 'ATAC'
        modality_name = 'Epigenome'
        adata_atac.var_names_make_unique()
        second_adata = adata_atac
    else:
        raise ValueError("Either ADT_path or ATAC_path must be provided.")

    print(f"Processing horizontal integration: RNA + {modality_name} fusion data...")

    # === Check for batch information in fusion data ===
    print("Checking batch information for horizontal integration...")
    if 'batch' in adata_rna.obs.columns:
        print(f"Batch distribution in RNA: {adata_rna.obs['batch'].value_counts()}")
        print("Horizontal integration will address batch effects between these batches")
    else:
        print("Warning: No 'batch' column found in RNA fusion data. Adding default batch labels.")
        # Add default batch information if not present
        n_cells = adata_rna.n_obs
        adata_rna.obs['batch'] = ['batch_1'] * (n_cells // 2) + ['batch_2'] * (n_cells - n_cells // 2)
    
    if 'batch' in second_adata.obs.columns:
        print(f"Batch distribution in {modality_name}: {second_adata.obs['batch'].value_counts()}")
    else:
        print(f"Warning: No 'batch' column found in {modality_name} fusion data. Adding default batch labels.")
        second_adata.obs['batch'] = adata_rna.obs['batch'].copy()

    # === Ensure both datasets have same cells ===
    common_obs = adata_rna.obs_names.intersection(second_adata.obs_names)
    print(f"Common cells: {len(common_obs)}, RNA cells: {adata_rna.n_obs}, {modality_name} cells: {second_adata.n_obs}")
    adata_rna = adata_rna[common_obs].copy()
    second_adata = second_adata[common_obs].copy()
    print(f"After intersection - RNA: {adata_rna.n_obs}, {modality_name}: {second_adata.n_obs}")

    # === Add spatial coordinates for horizontal integration if not present ===
    for i, (adata, name) in enumerate([(adata_rna, 'RNA'), (second_adata, modality_name)]):
        if 'spatial' not in adata.obsm.keys():
            print(f"Warning: No spatial coordinates found in {name}. Generating pseudo-spatial coordinates for horizontal integration...")
            # Generate pseudo-spatial coordinates based on UMAP
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
            adata.obsm['spatial'] = adata.obsm['X_umap'].copy()
            print(f"Generated pseudo-spatial coordinates for {name}")

    # === Initialize COSMOS model for horizontal integration ===
    print("Initializing COSMOS model for horizontal integration...")
    cosmos_model = cosmos.Cosmos(adata1=adata_rna, adata2=second_adata)
    
    # === Preprocessing (adapted for horizontal integration) ===
    print("Preprocessing data for horizontal integration...")
    cosmos_model.preprocessing_data(
        do_norm=True,
        do_log=True,
        n_top_genes=3000,  # Keep standard gene selection
        do_pca=False,
        n_neighbors=15  # Increased neighbors for better batch integration
    )
    
    # === Train COSMOS model (horizontal integration specific) ===
    print("Training COSMOS model for horizontal integration...")
    start_time = time.time()
    
    # Extract GPU ID from device string
    gpu_id = 0
    if 'cuda:' in args.device:
        gpu_id = int(args.device.split(':')[1])
    
    embedding = cosmos_model.train(
        spatial_regularization_strength=0.005,  # Reduced for horizontal integration
        z_dim=50,
        lr=5e-4,  # Reduced learning rate for stable horizontal integration
        wnn_epoch=600,  # Increased epochs for batch effect removal
        total_epoch=1200,  # Increased total epochs for horizontal integration
        max_patience_bef=15,  # Increased patience
        max_patience_aft=40,  # Increased patience for convergence
        min_stop=300,  # Increased minimum stopping
        random_seed=args.seed,
        gpu=gpu_id,
        regularization_acceleration=True,
        edge_subset_sz=1000000
    )
    
    end_time = time.time()
    train_time = end_time - start_time
    print('Horizontal integration training time:', train_time)

    # === Build Result AnnData ===
    # Use the first dataset (RNA) as the base
    adata = adata_rna.copy()
    adata.obsm['COSMOS'] = embedding
    adata.uns['train_time'] = train_time
    adata.uns['integration_type'] = 'horizontal'
    
    # === Clean embeddings for clustering ===
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

    # === Clustering and UMAP Generation ===
    tools = ['mclust', 'louvain', 'leiden', 'kmeans']
    
    # === Generate UMAP coordinates (store in adata, no plotting) ===
    print("Generating UMAP coordinates...")
    sc.pp.neighbors(adata, use_rep='COSMOS', n_neighbors=30)
    sc.tl.umap(adata)
    print("UMAP coordinates generated and stored in adata.obsm['X_umap']")
    
    # === Run clustering methods ===
    print("Running clustering methods...")
    for tool in tools:
        print(f"  Running {tool} clustering...")
        adata = universal_clustering(
            adata,
            n_clusters=args.cluster_nums,
            used_obsm='COSMOS',
            method=tool,
            key=tool,
            use_pca=False
        )
    
    print("All clustering methods completed")

    # === Clean temporary variables ===
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
    print('Saving horizontal integration results to...', args.save_path)


if __name__ == "__main__":
    # Set environment variables for R and threading
    os.environ['R_HOME'] = '/home/zhenghong/miniconda3/envs/smobench/lib/R'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    print("Starting COSMOS horizontal integration...")
    parser = argparse.ArgumentParser(description='Run COSMOS horizontal integration')
    parser.add_argument('--data_type', type=str, default='fusion', help='Data type for horizontal integration')
    parser.add_argument('--RNA_path', type=str, required=True, help='Path to RNA fusion adata')
    parser.add_argument('--ADT_path', type=str, default='', help='Path to ADT fusion adata')
    parser.add_argument('--ATAC_path', type=str, default='', help='Path to ATAC fusion adata')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save integrated adata')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, e.g. cuda:0 or cpu')

    parser.add_argument('--method', type=str, default='COSMOS', help='Method name for plotting')
    parser.add_argument('--dataset', type=str, default='', help='Dataset name for horizontal integration')

    parser.add_argument('--cluster_nums', type=int, help='Number of clusters')

    args = parser.parse_args()
    main(args)