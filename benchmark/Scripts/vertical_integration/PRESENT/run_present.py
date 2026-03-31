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

# Add PRESENT to path
present_path = os.path.join(project_root, "Methods/PRESENT")
sys.path.append(present_path)

from PRESENT import PRESENT_function
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
    
    # Extract GPU ID from device string
    device_id = 0
    if 'cuda:' in args.device:
        device_id = int(args.device.split(':')[1])
    
    # === Load Data ===
    adata_rna = sc.read_h5ad(args.RNA_path)
    adata_rna.var_names_make_unique()

    adata_adt = None
    adata_atac = None
    modality = None
    modality_name = None

    if args.ADT_path:
        adata_adt = sc.read_h5ad(args.ADT_path)
        modality = 'ADT'
        modality_name = 'Proteome'
        adata_adt.var_names_make_unique()
    elif args.ATAC_path:
        adata_atac = sc.read_h5ad(args.ATAC_path)
        modality = 'ATAC'
        modality_name = 'Epigenome'
        adata_atac.var_names_make_unique()
    else:
        raise ValueError("Either ADT_path or ATAC_path must be provided.")

    print(f"Processing {args.data_type}: RNA + {modality_name} integration...")

    # === Ensure both datasets have same cells ===
    if adata_adt is not None:
        common_obs = adata_rna.obs_names.intersection(adata_adt.obs_names)
        print(f"Common cells: {len(common_obs)}, RNA cells: {adata_rna.n_obs}, {modality_name} cells: {adata_adt.n_obs}")
        adata_rna = adata_rna[common_obs].copy()
        adata_adt = adata_adt[common_obs].copy()
        print(f"After intersection - RNA: {adata_rna.n_obs}, {modality_name}: {adata_adt.n_obs}")
    else:
        common_obs = adata_rna.obs_names.intersection(adata_atac.obs_names)
        print(f"Common cells: {len(common_obs)}, RNA cells: {adata_rna.n_obs}, {modality_name} cells: {adata_atac.n_obs}")
        adata_rna = adata_rna[common_obs].copy()
        adata_atac = adata_atac[common_obs].copy()
        print(f"After intersection - RNA: {adata_rna.n_obs}, {modality_name}: {adata_atac.n_obs}")

    # === Run PRESENT integration ===
    print("Running PRESENT integration...")
    start_time = time.time()
    
    # Run PRESENT_function with the loaded data
    adata_integrated = PRESENT_function(
        spatial_key="spatial",
        batch_key=None,  # No batch correction for single sample
        adata_rna=adata_rna,
        adata_atac=adata_atac,
        adata_adt=adata_adt,
        rdata_rna=None,  # No reference data
        rdata_rna_anno=None,
        rdata_atac=None,
        rdata_atac_anno=None,
        rdata_adt=None,
        rdata_adt_anno=None,
        gene_min_cells=1,
        peak_min_cells_fraction=0.03,
        protein_min_cells=1,
        num_hvg=3000,
        nclusters=args.cluster_nums,
        d_lat=50,
        k_neighbors=6,
        intra_neighbors=6,
        inter_neighbors=6,
        epochs=100,
        lr=1e-3,
        batch_size=320,
        device="cuda",
        device_id=device_id
    )
    
    end_time = time.time()
    train_time = end_time - start_time
    print('Training time:', train_time)

    # === Build Result AnnData ===
    # Use the integrated result
    adata = adata_integrated.copy()
    adata.obsm['PRESENT'] = adata_integrated.obsm['embeddings']
    adata.uns['train_time'] = train_time
    
    # === Clean embeddings for clustering ===
    import numpy as np
    embeddings_clean = adata.obsm['PRESENT'].copy()
    
    # Check for and handle infinite/NaN values
    if np.any(~np.isfinite(embeddings_clean)):
        print("Warning: Found infinite or NaN values in embeddings. Cleaning...")
        embeddings_clean[np.isinf(embeddings_clean)] = np.sign(embeddings_clean[np.isinf(embeddings_clean)]) * 1e10
        embeddings_clean[np.isnan(embeddings_clean)] = 0
        adata.obsm['PRESENT'] = embeddings_clean
        print(f"Cleaned embeddings shape: {embeddings_clean.shape}")
    else:
        print("Embeddings are clean (no inf/NaN values)")
    
    # === Parse Dataset Info ===
    dataset_name, subset_name = parse_dataset_info(args)
    print(f"Detected dataset: {dataset_name}, subset: {subset_name}")

    # === Plot Save Path ===
    plot_base_dir = "Results/plot"
    method_name = args.method if args.method else "PRESENT"
    plot_dir = os.path.join(plot_base_dir, method_name, dataset_name, subset_name)
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Plot images will be saved to: {plot_dir}")

    # === Clustering and Visualization ===
    tools = ['mclust', 'louvain', 'leiden', 'kmeans']
    
    # === Generate UMAP for visualization (only once) ===
    sc.pp.neighbors(adata, use_rep='PRESENT', n_neighbors=30)
    sc.tl.umap(adata)
    
    for tool in tools:
        adata = universal_clustering(
            adata,
            n_clusters=args.cluster_nums,
            used_obsm='PRESENT',
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

    print("Starting PRESENT integration...")
    parser = argparse.ArgumentParser(description='Run PRESENT integration')
    parser.add_argument('--data_type', type=str, default='10x', help='Data type, e.g. 10x, SPOTS, MISAR, simulation')
    parser.add_argument('--RNA_path', type=str, required=True, help='Path to RNA adata')
    parser.add_argument('--ADT_path', type=str, default='', help='Path to ADT adata')
    parser.add_argument('--ATAC_path', type=str, default='', help='Path to ATAC adata')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save integrated adata')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, e.g. cuda:0 or cpu')

    parser.add_argument('--method', type=str, default='PRESENT', help='Method name for plotting')
    parser.add_argument('--dataset', type=str, default='', help='Dataset name, e.g. Human_Lymph_Nodes/A1. If not provided, auto-extracted from paths.')

    parser.add_argument('--cluster_nums', type=int, help='Number of clusters')

    args = parser.parse_args()
    main(args)