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

# Add PRAGA to path
praga_path = os.path.join(project_root, "Methods/PRAGA")
sys.path.append(praga_path)

from PRAGA.Train_model import Train
from PRAGA.preprocess import construct_neighbor_graph, pca, clr_normalize_each_cell, lsi, fix_seed
from Utils.SMOBench_clustering import universal_clustering


class Args:
    def __init__(self, datatype):
        # Basic parameters
        self.device = 'cuda:0'
        self.seed = 2024
        self.feat_n_comps = 200
        self.n_neighbors = 3
        self.KNN_k = 20
        
        # Data type specific parameters
        if datatype == '10x':
            self.RNA_weight = 1
            self.ADT_weight = 5
            self.cl_weight = 1
            self.alpha = 0.9
            self.tau = 1
            self.init_k = 6
        elif datatype == 'SPOTS':
            self.RNA_weight = 1
            self.ADT_weight = 3
            self.cl_weight = 1
            self.alpha = 0.9
            self.tau = 1
            self.init_k = 10
        elif datatype == 'Stereo-CITE-seq':
            self.RNA_weight = 1
            self.ADT_weight = 3
            self.cl_weight = 1
            self.alpha = 0.9
            self.tau = 1
            self.init_k = 10
        elif datatype == 'Spatial-epigenome-transcriptome':
            self.RNA_weight = 1
            self.ADT_weight = 3
            self.cl_weight = 2
            self.alpha = 0.9
            self.tau = 1
            self.init_k = 14
        elif datatype == 'MISAR':
            self.RNA_weight = 1
            self.ADT_weight = 1
            self.cl_weight = 5
            self.alpha = 0.9
            self.tau = 1
            self.init_k = 16
        elif datatype == 'simulation':
            # For 3M simulation data
            self.RNA_weight = 5
            self.ADT_weight = 1
            self.cl_weight = 5
            self.alpha = 0.9
            self.tau = 1
            self.init_k = 5
        else:
            # Default parameters
            self.RNA_weight = 1
            self.ADT_weight = 3
            self.cl_weight = 1
            self.alpha = 0.9
            self.tau = 1
            self.init_k = 6


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


def preprocess_data(adata_rna, adata_second, data_type, args_praga, modality_name="Second"):
    """Preprocess data following PRAGA's requirements"""
    print("Preprocessing data...")
    
    # Set random seed
    fix_seed(args_praga.seed)
    
    # Basic preprocessing
    adata_rna.var_names_make_unique()
    adata_second.var_names_make_unique()
    
    # Ensure both datasets have the same cells (intersect)
    common_obs = adata_rna.obs_names.intersection(adata_second.obs_names)
    print(f"Common cells: {len(common_obs)}, RNA cells: {adata_rna.n_obs}, {modality_name} cells: {adata_second.n_obs}")
    adata_rna = adata_rna[common_obs].copy()
    adata_second = adata_second[common_obs].copy()
    print(f"After intersection - RNA: {adata_rna.n_obs}, {modality_name}: {adata_second.n_obs}")
    
    # Gene filtering and normalization for RNA (without cell filtering yet)
    sc.pp.filter_genes(adata_rna, min_cells=3)
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    
    # Find highly variable genes
    sc.pp.highly_variable_genes(adata_rna, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata_rna.raw = adata_rna
    adata_rna = adata_rna[:, adata_rna.var.highly_variable]
    
    # Preprocess second modality based on type (without cell filtering)
    if data_type in ['10x', 'SPOTS', 'Stereo-CITE-seq']:
        # Protein data preprocessing
        sc.pp.filter_genes(adata_second, min_cells=3)
        clr_normalize_each_cell(adata_second)
    else:
        # ATAC/chromatin data preprocessing  
        sc.pp.filter_genes(adata_second, min_cells=3)
        sc.pp.normalize_total(adata_second, target_sum=1e4)
        sc.pp.log1p(adata_second)
        lsi(adata_second, n_components=50)
        adata_second.obsm['feat'] = adata_second.obsm['X_lsi']
    
    # Final cell filtering step - ensure both datasets remain synchronized
    # Filter cells based on gene count for RNA only, then apply to both
    cell_filter = (adata_rna.X > 0).sum(axis=1).A1 >= 200  # min 200 genes per cell
    adata_rna = adata_rna[cell_filter].copy()
    adata_second = adata_second[cell_filter].copy()
    print(f"After final cell filtering - RNA: {adata_rna.n_obs}, {modality_name}: {adata_second.n_obs}")
    
    # PCA for RNA data
    # Adjust n_comps to not exceed min(n_samples, n_features)
    max_comps_rna = min(adata_rna.n_obs, adata_rna.n_vars) - 1
    n_comps_rna = min(args_praga.feat_n_comps, max_comps_rna)
    print(f"Using {n_comps_rna} components for RNA PCA (max possible: {max_comps_rna})")
    adata_rna.obsm['feat'] = pca(adata_rna, n_comps=n_comps_rna)
    
    # PCA for protein data if not already processed
    if 'feat' not in adata_second.obsm:
        # Adjust n_comps to not exceed min(n_samples, n_features)
        max_comps = min(adata_second.n_obs, adata_second.n_vars) - 1
        n_comps = min(args_praga.feat_n_comps, max_comps)
        print(f"Using {n_comps} components for {modality_name} PCA (max possible: {max_comps})")
        adata_second.obsm['feat'] = pca(adata_second, n_comps=n_comps)
    
    print(f"RNA data shape after preprocessing: {adata_rna.shape}")
    print(f"Second modality data shape after preprocessing: {adata_second.shape}")
    
    return adata_rna, adata_second


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    
    # Initialize PRAGA arguments
    args_praga = Args(args.data_type)
    args_praga.device = args.device
    args_praga.seed = args.seed
    
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

    # === Preprocessing ===
    adata_rna, second_adata = preprocess_data(adata_rna, second_adata, args.data_type, args_praga, modality_name)
    
    # === Construct neighbor graphs ===
    data = construct_neighbor_graph(adata_rna, second_adata, 
                                  datatype=args.data_type, 
                                  n_neighbors=args_praga.n_neighbors, 
                                  Arg=args_praga)
    
    # === Train PRAGA model ===
    print("Training PRAGA model...")
    start_time = time.time()
    
    # Use regular Train class for all data types
    model = Train(data, args.data_type, device, args.seed, Arg=args_praga)
    
    # Train the model
    output = model.train()
    end_time = time.time()
    train_time = end_time - start_time
    print('Training time:', train_time)

    # === Build Result AnnData ===
    # Use the first dataset (RNA) as the base
    adata = data['adata_omics1'].copy()
    adata.obsm['PRAGA'] = output['PRAGA']
    adata.uns['train_time'] = train_time
    
    # === Clean embeddings for clustering ===
    import numpy as np
    embeddings_clean = adata.obsm['PRAGA'].copy()
    
    # Check for and handle infinite/NaN values
    if np.any(~np.isfinite(embeddings_clean)):
        print("Warning: Found infinite or NaN values in embeddings. Cleaning...")
        embeddings_clean[np.isinf(embeddings_clean)] = np.sign(embeddings_clean[np.isinf(embeddings_clean)]) * 1e10
        embeddings_clean[np.isnan(embeddings_clean)] = 0
        adata.obsm['PRAGA'] = embeddings_clean
        print(f"Cleaned embeddings shape: {embeddings_clean.shape}")
    else:
        print("Embeddings are clean (no inf/NaN values)")
    
    # === Parse Dataset Info ===
    dataset_name, subset_name = parse_dataset_info(args)
    print(f"Detected dataset: {dataset_name}, subset: {subset_name}")

    # === Plot Save Path ===
    plot_base_dir = "Results/plot"
    method_name = args.method if args.method else "PRAGA"
    plot_dir = os.path.join(plot_base_dir, method_name, dataset_name, subset_name)
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Plot images will be saved to: {plot_dir}")

    # === Clustering and Visualization ===
    # Use PRAGA's original clustering approach 
    from PRAGA.utils import clustering
    
    # PRAGA original mclust clustering
    clustering(adata, key='PRAGA', add_key='PRAGA', n_clusters=args.cluster_nums, method='mclust', use_pca=True)
    adata.obs['mclust'] = adata.obs['PRAGA']
    
    # Additional clustering methods for comparison using universal clustering
    tools = ['louvain', 'leiden', 'kmeans']
    
    # === Generate UMAP for visualization (only once) ===
    sc.pp.neighbors(adata, use_rep='PRAGA', n_neighbors=30)
    sc.tl.umap(adata)
    
    for tool in tools:
        adata = universal_clustering(
            adata,
            n_clusters=args.cluster_nums,
            used_obsm='PRAGA',
            method=tool,
            key=tool,
            use_pca=False
        )
    
    # Include mclust in the tools list for plotting
    tools = ['mclust', 'louvain', 'leiden', 'kmeans']
        
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

    print("Starting PRAGA integration...")
    parser = argparse.ArgumentParser(description='Run PRAGA integration')
    parser.add_argument('--data_type', type=str, default='10x', help='Data type, e.g. 10x, SPOTS, MISAR, simulation')
    parser.add_argument('--RNA_path', type=str, required=True, help='Path to RNA adata')
    parser.add_argument('--ADT_path', type=str, default='', help='Path to ADT adata')
    parser.add_argument('--ATAC_path', type=str, default='', help='Path to ATAC adata')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save integrated adata')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, e.g. cuda:0 or cpu')

    parser.add_argument('--method', type=str, default='PRAGA', help='Method name for plotting')
    parser.add_argument('--dataset', type=str, default='', help='Dataset name, e.g. Human_Lymph_Nodes/A1. If not provided, auto-extracted from paths.')

    parser.add_argument('--cluster_nums', type=int, help='Number of clusters')

    args = parser.parse_args()
    main(args)