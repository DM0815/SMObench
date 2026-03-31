import os
import torch
import pandas as pd
import scanpy as sc
import argparse
import time
import json
from datetime import datetime
import sys
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

# Add project root directory to module search path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

# Add PRAGA to path
praga_path = os.path.join(project_root, "Methods/PRAGA")
sys.path.append(praga_path)

from PRAGA.Train_model import Train
from PRAGA.preprocess import construct_neighbor_graph, pca, clr_normalize_each_cell, lsi, fix_seed
from Utils.SMOBench_clustering import universal_clustering

# Maximum number of cells PRAGA can handle on a single 40GB GPU.
# PRAGA stores 8+ dense N x N adjacency matrices on GPU; for N=10000 each
# matrix is ~0.37 GiB (10000^2 * 4 bytes), totaling ~3 GiB, well within budget.
MAX_CELLS = 10000


def stratified_subsample(adata_rna, adata_mod2, max_cells, seed=2024):
    """Stratified subsampling by batch label, keeping proportions.

    Returns
    -------
    adata_rna_sub, adata_mod2_sub : subsampled AnnData objects (copies)
    kept_idx : np.ndarray of integer indices into the original obs axis
    """
    rng = np.random.RandomState(seed)
    n_total = adata_rna.n_obs
    batch_key = 'batch'

    if batch_key in adata_rna.obs.columns:
        batches = adata_rna.obs[batch_key].values
    else:
        batches = np.array(['all'] * n_total)

    unique_batches = np.unique(batches)
    kept = []
    for b in unique_batches:
        idx_b = np.where(batches == b)[0]
        n_keep = max(1, int(round(len(idx_b) * max_cells / n_total)))
        n_keep = min(n_keep, len(idx_b))
        chosen = rng.choice(idx_b, size=n_keep, replace=False)
        kept.append(chosen)

    kept_idx = np.sort(np.concatenate(kept))
    # Ensure we don't exceed max_cells (rounding may push slightly over)
    if len(kept_idx) > max_cells:
        kept_idx = rng.choice(kept_idx, size=max_cells, replace=False)
        kept_idx = np.sort(kept_idx)

    print(f"[Subsample] {n_total} -> {len(kept_idx)} cells (max_cells={max_cells})")
    for b in unique_batches:
        n_orig = np.sum(batches == b)
        n_kept = np.sum(batches[kept_idx] == b)
        print(f"  batch '{b}': {n_orig} -> {n_kept}")

    adata_rna_sub = adata_rna[kept_idx].copy()
    adata_mod2_sub = adata_mod2[kept_idx].copy()
    return adata_rna_sub, adata_mod2_sub, kept_idx


def knn_propagate_embeddings(emb_sub, features_sub, features_all, kept_idx, n_total, k=15):
    """Propagate embeddings from subsampled cells to all cells via KNN regression.

    Parameters
    ----------
    emb_sub : np.ndarray, shape (n_sub, dim_emb)
        Embeddings for the subsampled cells.
    features_sub : np.ndarray, shape (n_sub, dim_feat)
        Feature matrix for subsampled cells (used as KNN input space).
    features_all : np.ndarray, shape (n_total, dim_feat)
        Feature matrix for ALL cells.
    kept_idx : np.ndarray of int
        Indices of subsampled cells in the original ordering.
    n_total : int
        Total number of cells.
    k : int
        Number of neighbors for KNN regression.

    Returns
    -------
    emb_full : np.ndarray, shape (n_total, dim_emb)
    """
    dim_emb = emb_sub.shape[1]
    emb_full = np.zeros((n_total, dim_emb), dtype=np.float32)

    # Place known embeddings
    emb_full[kept_idx] = emb_sub

    # Identify held-out cells
    all_idx = np.arange(n_total)
    held_mask = np.ones(n_total, dtype=bool)
    held_mask[kept_idx] = False
    held_idx = all_idx[held_mask]

    if len(held_idx) == 0:
        return emb_full

    print(f"[KNN propagate] Predicting embeddings for {len(held_idx)} held-out cells using k={k} ...")
    knn = KNeighborsRegressor(n_neighbors=k, weights='distance', metric='cosine', n_jobs=-1)
    knn.fit(features_sub, emb_sub)
    emb_held = knn.predict(features_all[held_idx])
    emb_full[held_idx] = emb_held
    print(f"[KNN propagate] Done.")

    return emb_full


class Args:
    def __init__(self, datatype):
        # Basic parameters
        self.device = 'cuda:0'
        self.seed = 2024
        self.feat_n_comps = 200
        self.n_neighbors = 3
        self.KNN_k = 20
        
        # Horizontal integration specific parameters (enhanced for batch effect removal)
        if datatype == 'fusion_RNA_ADT':
            self.RNA_weight = 1
            self.ADT_weight = 4  # Slightly increased for better balance
            self.cl_weight = 2   # Increased for better clustering
            self.alpha = 0.85    # Slightly reduced for more flexibility
            self.tau = 1.2       # Increased for batch integration
            self.init_k = 8      # Increased for better representation
        elif datatype == 'fusion_RNA_ATAC':
            self.RNA_weight = 1
            self.ADT_weight = 2   # Balanced for ATAC
            self.cl_weight = 3    # Increased for batch effect removal
            self.alpha = 0.85     # Slightly reduced
            self.tau = 1.5        # Increased for ATAC batch integration
            self.init_k = 12      # Increased for complex batch structure
        else:
            # Default horizontal integration parameters
            self.RNA_weight = 1
            self.ADT_weight = 3
            self.cl_weight = 2
            self.alpha = 0.85
            self.tau = 1.2
            self.init_k = 8


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
    total_start_time = time.time()
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
        data_type = 'fusion_RNA_ADT'
    elif args.ATAC_path:
        adata_atac = sc.read_h5ad(args.ATAC_path)  # ATAC fusion data
        modality = 'ATAC'
        modality_name = 'Epigenome'
        adata_atac.var_names_make_unique()
        second_adata = adata_atac
        data_type = 'fusion_RNA_ATAC'
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
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
            adata.obsm['spatial'] = adata.obsm['X_umap'].copy()
            print(f"Generated pseudo-spatial coordinates for {name}")

    # === PRAGA Preprocessing (adapted for horizontal integration) ===
    print("PRAGA preprocessing for horizontal integration...")
    fix_seed(args.seed)
    
    # Initialize PRAGA args with horizontal integration parameters
    praga_args = Args(data_type)
    praga_args.device = args.device
    praga_args.seed = args.seed
    
    print(f"Using horizontal integration parameters: RNA_weight={praga_args.RNA_weight}, "
          f"ADT_weight={praga_args.ADT_weight}, cl_weight={praga_args.cl_weight}, "
          f"alpha={praga_args.alpha}, tau={praga_args.tau}")

    # RNA preprocessing
    sc.pp.filter_genes(adata_rna, min_cells=3)
    sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.scale(adata_rna)
    
    # Extract highly variable genes
    adata_rna_high = adata_rna[:, adata_rna.var['highly_variable']]
    adata_rna.obsm['feat'] = pca(adata_rna_high, n_comps=praga_args.feat_n_comps)

    # Second modality preprocessing
    if modality == 'ADT':
        # Protein preprocessing for horizontal integration
        second_adata = clr_normalize_each_cell(second_adata)
        sc.pp.scale(second_adata)
        second_adata.obsm['feat'] = pca(second_adata, n_comps=min(praga_args.feat_n_comps, second_adata.n_vars-1))
    else:
        # ATAC preprocessing for horizontal integration
        if 'X_lsi' not in second_adata.obsm.keys():
            sc.pp.highly_variable_genes(second_adata, flavor="seurat_v3", n_top_genes=3000)
            lsi(second_adata, use_highly_variable=False, n_components=51)
        second_adata.obsm['feat'] = second_adata.obsm['X_lsi'].copy()

    # Determine PRAGA datatype based on modality
    if modality == 'ADT':
        praga_datatype = 'Stereo-CITE-seq'
    elif modality == 'ATAC':
        praga_datatype = 'Spatial-epigenome-transcriptome'
    else:
        raise ValueError(f"Unknown modality for PRAGA: {modality}")
    
    print(f"Using PRAGA datatype: {praga_datatype}")

    # === Subsample if dataset is too large for PRAGA's dense N x N GPU matrices ===
    n_total = adata_rna.n_obs
    need_subsample = n_total > MAX_CELLS
    if need_subsample:
        print(f"[OOM guard] Dataset has {n_total} cells (> MAX_CELLS={MAX_CELLS}). "
              f"Subsampling before PRAGA training, will KNN-propagate embeddings afterwards.")
        # Save full-dataset references and feature matrices for later KNN propagation
        adata_rna_full = adata_rna
        second_adata_full = second_adata
        feat_rna_full = adata_rna.obsm['feat'].copy()       # (n_total, dim_feat_rna)
        feat_mod2_full = second_adata.obsm['feat'].copy()    # (n_total, dim_feat_mod2)
        # Concatenate both modality features for a richer KNN input space
        feat_combined_full = np.concatenate([feat_rna_full, feat_mod2_full], axis=1)

        adata_rna_sub, second_adata_sub, kept_idx = stratified_subsample(
            adata_rna, second_adata, max_cells=MAX_CELLS, seed=args.seed
        )
        # Use subsampled data for PRAGA
        adata_rna_train = adata_rna_sub
        second_adata_train = second_adata_sub
    else:
        adata_rna_train = adata_rna
        second_adata_train = second_adata

    # Construct neighbor graphs for both modalities
    print("Constructing neighbor graphs for horizontal integration...")
    data_dict = construct_neighbor_graph(adata_rna_train, second_adata_train, datatype=praga_datatype, n_neighbors=6, Arg=praga_args)
    adata_rna_train = data_dict['adata_omics1']
    second_adata_train = data_dict['adata_omics2']

    # === Train PRAGA model for horizontal integration ===
    print("Training PRAGA model for horizontal integration...")
    start_time = time.time()

    trainer = Train(
        data=data_dict,
        datatype=praga_datatype,
        device=args.device,
        random_seed=args.seed,
        dim_input=3000,
        dim_output=64,
        Arg=praga_args
    )

    output = trainer.train()
    latent_embeddings_train = output['PRAGA']

    end_time = time.time()
    train_time = end_time - start_time
    print('Horizontal integration training time:', train_time)

    # === Propagate embeddings to held-out cells if subsampled ===
    if need_subsample:
        print(f"[OOM guard] Propagating embeddings from {len(kept_idx)} subsampled cells to all {n_total} cells...")
        feat_combined_sub = feat_combined_full[kept_idx]
        latent_embeddings = knn_propagate_embeddings(
            emb_sub=latent_embeddings_train,
            features_sub=feat_combined_sub,
            features_all=feat_combined_full,
            kept_idx=kept_idx,
            n_total=n_total,
            k=15
        )
        # Use the full adata_rna (not the subsampled one) for downstream
        adata_rna = adata_rna_full
    else:
        latent_embeddings = latent_embeddings_train

    # === Build Result AnnData ===
    adata = adata_rna.copy()
    adata.obsm['PRAGA'] = latent_embeddings
    adata.uns['train_time'] = train_time
    adata.uns['integration_type'] = 'horizontal'
    if need_subsample:
        adata.uns['subsampled'] = True
        adata.uns['subsample_n'] = len(kept_idx)
        adata.uns['subsample_max_cells'] = MAX_CELLS
    
    # === Parse Dataset Info ===
    dataset_name, subset_name = parse_dataset_info(args)
    print(f"Detected dataset: {dataset_name}, subset: {subset_name}")

    # === Clustering and UMAP Generation ===
    tools = ['mclust', 'louvain', 'leiden', 'kmeans']
    
    # === Generate UMAP coordinates (store in adata, no plotting) ===
    print("Generating UMAP coordinates...")
    sc.pp.neighbors(adata, use_rep='PRAGA', n_neighbors=30)
    sc.tl.umap(adata)
    print("UMAP coordinates generated and stored in adata.obsm['X_umap']")
    
    # === Run clustering methods ===
    print("Running clustering methods...")
    for tool in tools:
        print(f"  Running {tool} clustering...")
        adata = universal_clustering(
            adata,
            n_clusters=args.cluster_nums,
            used_obsm='PRAGA',
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

    # === Save timing info ===
    total_time = time.time() - total_start_time
    dataset_name, _ = parse_dataset_info(args)
    modality_str = "RNA_ADT" if args.ADT_path else "RNA_ATAC"
    timing_info = {
        "method": "PRAGA",
        "dataset": dataset_name,
        "integration_type": "horizontal",
        "modality": modality_str,
        "n_cells": adata.n_obs,
        "embedding_shape": list(adata.obsm["PRAGA"].shape),
        "training_time_s": round(train_time, 2),
        "total_time_s": round(total_time, 2),
        "device": str(device),
        "seed": args.seed,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    timing_path = args.save_path.replace(".h5ad", "_timing_info.json")
    with open(timing_path, "w") as f:
        json.dump(timing_info, f, indent=2)
    print(f"Timing info saved to: {timing_path}")


if __name__ == "__main__":
    # Set environment variables for R and threading
    os.environ['R_HOME'] = '/home/users/nus/e1724738/miniconda3/envs/_Proj1_1/lib/R'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    print("Starting PRAGA horizontal integration...")
    parser = argparse.ArgumentParser(description='Run PRAGA horizontal integration')
    parser.add_argument('--data_type', type=str, default='fusion', help='Data type for horizontal integration')
    parser.add_argument('--RNA_path', type=str, required=True, help='Path to RNA fusion adata')
    parser.add_argument('--ADT_path', type=str, default='', help='Path to ADT fusion adata')
    parser.add_argument('--ATAC_path', type=str, default='', help='Path to ATAC fusion adata')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save integrated adata')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, e.g. cuda:0 or cpu')

    parser.add_argument('--method', type=str, default='PRAGA', help='Method name for plotting')
    parser.add_argument('--dataset', type=str, default='', help='Dataset name for horizontal integration')

    parser.add_argument('--cluster_nums', type=int, help='Number of clusters')

    args = parser.parse_args()
    main(args)