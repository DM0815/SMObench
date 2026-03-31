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

# Add SpatialGlue to path
spatialglue_path = os.path.join(project_root, "Methods/SpatialGlue")
sys.path.append(spatialglue_path)

from preprocess import fix_seed
from preprocess import clr_normalize_each_cell, pca, lsi
from preprocess import construct_neighbor_graph
from SpatialGlue_pyG import Train_SpatialGlue
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
    adata_omics1 = sc.read_h5ad(args.RNA_path)  # RNA fusion data
    adata_omics1.var_names_make_unique()

    if args.ADT_path:
        adata_omics2 = sc.read_h5ad(args.ADT_path)  # ADT fusion data
        modality = 'ADT'
        modality_name = 'Proteome'
        adata_omics2.var_names_make_unique()
    elif args.ATAC_path:
        adata_omics2 = sc.read_h5ad(args.ATAC_path)  # ATAC fusion data
        modality = 'ATAC'
        modality_name = 'Epigenome'
        adata_omics2.var_names_make_unique()
    else:
        raise ValueError("Either ADT_path or ATAC_path must be provided.")

    print(f"Processing horizontal integration: RNA + {modality_name} fusion data...")

    # === Check for batch information in fusion data ===
    print("Checking batch information for horizontal integration...")
    if 'batch' in adata_omics1.obs.columns:
        print(f"Batch distribution in RNA: {adata_omics1.obs['batch'].value_counts()}")
        print("Horizontal integration will address batch effects between these batches")
    else:
        print("Warning: No 'batch' column found in RNA fusion data. Adding default batch labels.")
        n_cells = adata_omics1.n_obs
        adata_omics1.obs['batch'] = ['batch_1'] * (n_cells // 2) + ['batch_2'] * (n_cells - n_cells // 2)
    
    if 'batch' in adata_omics2.obs.columns:
        print(f"Batch distribution in {modality_name}: {adata_omics2.obs['batch'].value_counts()}")
    else:
        print(f"Warning: No 'batch' column found in {modality_name} fusion data. Adding default batch labels.")
        adata_omics2.obs['batch'] = adata_omics1.obs['batch'].copy()

    # === Ensure both datasets have same cells ===
    common_obs = adata_omics1.obs_names.intersection(adata_omics2.obs_names)
    print(f"Common cells: {len(common_obs)}, RNA cells: {adata_omics1.n_obs}, {modality_name} cells: {adata_omics2.n_obs}")
    adata_omics1 = adata_omics1[common_obs].copy()
    adata_omics2 = adata_omics2[common_obs].copy()
    print(f"After intersection - RNA: {adata_omics1.n_obs}, {modality_name}: {adata_omics2.n_obs}")

    # === Add spatial coordinates for horizontal integration if not present ===
    for i, (adata, name) in enumerate([(adata_omics1, 'RNA'), (adata_omics2, modality_name)]):
        if 'spatial' not in adata.obsm.keys():
            print(f"Warning: No spatial coordinates found in {name}. Generating pseudo-spatial coordinates for horizontal integration...")
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
            adata.obsm['spatial'] = adata.obsm['X_umap'].copy()
            print(f"Generated pseudo-spatial coordinates for {name}")

    # === Fix seed ===
    fix_seed(args.seed)

    # === RNA preprocessing (adapted for horizontal integration) ===
    print("SpatialGlue preprocessing for horizontal integration...")
    sc.pp.filter_genes(adata_omics1, min_cells=10)
    sc.pp.normalize_total(adata_omics1, target_sum=1e4)
    sc.pp.log1p(adata_omics1)
    sc.pp.highly_variable_genes(adata_omics1, n_top_genes=3000)
    sc.pp.scale(adata_omics1)

    # Determine n_comps based on data type (adapted for horizontal integration)
    if modality == 'ADT':
        n_comps = 40  # Increased for horizontal integration
    else:  # ATAC
        n_comps = 50  # Increased for horizontal integration

    adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
    adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=n_comps)

    # === Second modality preprocessing (adapted for horizontal integration) ===
    if modality == 'ADT':
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=n_comps)
    elif modality == 'ATAC':
        if 'X_lsi' not in adata_omics2.obsm.keys():
            sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
            lsi(adata_omics2, use_highly_variable=False, n_components=n_comps)
        adata_omics2.obsm['feat'] = adata_omics2.obsm['X_lsi'].copy()

    # === Construct neighbor graph for horizontal integration ===
    print("Constructing neighbor graph for horizontal integration...")
    data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype='fusion')

    # === Train SpatialGlue model for horizontal integration ===
    print("Training SpatialGlue model for horizontal integration...")
    start_time = time.time()
    
    datatype_flag = 'MISAR' if modality == 'ATAC' else 'fusion'
    target_epochs = 1200 if modality == 'ATAC' else 800

    model = Train_SpatialGlue(
        data,
        datatype=datatype_flag,
        device=device,
        learning_rate=1e-4,
        epochs=target_epochs
    )
    
    output = model.train()
    
    end_time = time.time()
    train_time = end_time - start_time
    print('Horizontal integration training time:', train_time)

    # === Build Result AnnData ===
    adata = adata_omics1.copy()
    adata.obsm['emb_latent_omics1'] = output['emb_latent_omics1'].copy()
    adata.obsm['emb_latent_omics2'] = output['emb_latent_omics2'].copy()
    adata.obsm['SpatialGlue'] = output['SpatialGlue'].copy()
    adata.uns['train_time'] = train_time
    adata.uns['integration_type'] = 'horizontal'
    
    # === Parse Dataset Info ===
    dataset_name, subset_name = parse_dataset_info(args)
    print(f"Detected dataset: {dataset_name}, subset: {subset_name}")

    # === Clustering and UMAP Generation ===
    tools = ['mclust', 'kmeans']
    
    # === Generate UMAP coordinates (store in adata, no plotting) ===
    print("Generating UMAP coordinates...")
    sc.pp.neighbors(adata, use_rep='SpatialGlue', n_neighbors=30)
    sc.tl.umap(adata)
    print("UMAP coordinates generated and stored in adata.obsm['X_umap']")
    
    # === Run clustering methods ===
    print("Running clustering methods...")
    for tool in tools:
        print(f"  Running {tool} clustering...")
        adata = universal_clustering(
            adata,
            n_clusters=args.cluster_nums,
            used_obsm='SpatialGlue',
            method=tool,
            key=tool,
            use_pca=False
        )
    
    print("All clustering methods completed")

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

    print("Starting SpatialGlue horizontal integration...")
    parser = argparse.ArgumentParser(description='Run SpatialGlue horizontal integration')
    parser.add_argument('--data_type', type=str, default='fusion', help='Data type for horizontal integration')
    parser.add_argument('--RNA_path', type=str, required=True, help='Path to RNA fusion adata')
    parser.add_argument('--ADT_path', type=str, default='', help='Path to ADT fusion adata')
    parser.add_argument('--ATAC_path', type=str, default='', help='Path to ATAC fusion adata')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save integrated adata')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, e.g. cuda:0 or cpu')

    parser.add_argument('--method', type=str, default='SpatialGlue', help='Method name for plotting')
    parser.add_argument('--dataset', type=str, default='', help='Dataset name for horizontal integration')

    parser.add_argument('--cluster_nums', type=int, help='Number of clusters')

    args = parser.parse_args()
    main(args)
