#!/usr/bin/env python3
"""
SpaMultiVAE horizontal integration script for SMOBench
Adapted for fusion datasets and batch effect removal
Supports only RNA+ADT spatial multi-omics integration
"""

import os
import sys
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import h5py
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture

# Add project paths
project_root = '/home/zhenghong/SMOBench-CLEAN'
methods_path = os.path.join(project_root, 'Methods')
sys.path.append(project_root)
sys.path.append(methods_path)

# Import SpaMultiVAE modules
from spaMultiVAE.spaMultiVAE import SPAMULTIVAE
from spaMultiVAE.preprocess import normalize, geneSelection
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
    elif "Mouse_Thymus_Fusion" in args.RNA_path:
        return "Mouse_Thymus", "fusion"
    elif "Mouse_Spleen_Fusion" in args.RNA_path:
        return "Mouse_Spleen", "fusion"
    
    return "Unknown", "fusion"


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SpaMultiVAE horizontal integration')
    
    # Required arguments
    parser.add_argument('--data_type', type=str, default='fusion', help='Data type for horizontal integration')
    parser.add_argument('--RNA_path', type=str, required=True, help='RNA fusion data path')
    parser.add_argument('--ADT_path', type=str, required=True, help='ADT fusion data path')
    parser.add_argument('--save_path', type=str, required=True, help='Output save path')
    parser.add_argument('--cluster_nums', type=int, required=True, help='Number of clusters')
    parser.add_argument('--method', type=str, default='SpaMultiVAE', help='Method name')
    parser.add_argument('--dataset', type=str, default='', help='Dataset name')
    
    # Optional arguments - ATAC check (SpaMultiVAE doesn't support ATAC)
    parser.add_argument('--ATAC_path', type=str, default=None, help='ATAC path (not supported)')
    
    # Enhanced SpaMultiVAE parameters for horizontal integration
    parser.add_argument('--select_genes', type=int, default=0, help='Number of genes to select (0=all)')
    parser.add_argument('--select_proteins', type=int, default=0, help='Number of proteins to select (0=all)')
    parser.add_argument('--batch_size', default="auto", help='Batch size')
    parser.add_argument('--maxiter', type=int, default=6000, help='Max iterations (increased for horizontal)')
    parser.add_argument('--train_size', type=float, default=0.95, help='Training proportion')
    parser.add_argument('--patience', type=int, default=300, help='Early stopping patience (increased)')
    parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate (reduced for stability)')
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='Weight decay (reduced)')
    parser.add_argument('--gene_noise', type=float, default=0.1, help='Gene noise level (added for regularization)')
    parser.add_argument('--protein_noise', type=float, default=0.05, help='Protein noise level (added)')
    parser.add_argument('--dropoutE', type=float, default=0.1, help='Encoder dropout (added)')
    parser.add_argument('--dropoutD', type=float, default=0.05, help='Decoder dropout (added)')
    parser.add_argument('--encoder_layers', nargs='+', default=[256, 128], type=int, help='Encoder layers (enhanced)')
    parser.add_argument('--GP_dim', type=int, default=3, help='GP dimensions (increased)')
    parser.add_argument('--Normal_dim', type=int, default=25, help='Normal dimensions (increased)')
    parser.add_argument('--gene_decoder_layers', nargs='+', default=[256, 128], type=int, help='Gene decoder layers (enhanced)')
    parser.add_argument('--protein_decoder_layers', nargs='+', default=[128], type=int, help='Protein decoder layers')
    parser.add_argument('--dynamicVAE', action='store_true', default=True, help='Use dynamic VAE (enabled for horizontal)')
    parser.add_argument('--init_beta', type=float, default=8, help='Initial beta (reduced)')
    parser.add_argument('--min_beta', type=float, default=2, help='Min beta (reduced)')
    parser.add_argument('--max_beta', type=float, default=20, help='Max beta (reduced)')
    parser.add_argument('--KL_loss', type=float, default=0.02, help='KL loss target (slightly reduced)')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples')
    parser.add_argument('--fix_inducing_points', action='store_true', default=True, help='Fix inducing points')
    parser.add_argument('--grid_inducing_points', action='store_true', default=True, help='Use grid inducing points')
    parser.add_argument('--inducing_point_steps', type=int, default=25, help='Inducing point steps (increased)')
    parser.add_argument('--inducing_point_nums', type=int, default=None, help='Number of inducing points')
    parser.add_argument('--fixed_gp_params', action='store_true', default=False, help='Fix GP params')
    parser.add_argument('--loc_range', type=float, default=25.0, help='Location range (increased)')
    parser.add_argument('--kernel_scale', type=float, default=25.0, help='Kernel scale (increased)')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--random_seed', type=int, default=2024, help='Random seed')
    
    return parser.parse_args()


def main():
    """Main function for horizontal integration"""
    args = parse_args()
    
    # Check for ATAC path and reject (SpaMultiVAE only supports RNA+ADT)
    if args.ATAC_path:
        print("Error: SpaMultiVAE does not support RNA+ATAC integration.")
        print("   This method only supports RNA+ADT data for horizontal integration.")
        sys.exit(1)
    
    # Set random seed
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    print("="*60)
    print(f"SpaMultiVAE Horizontal Integration")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print("="*60)
    
    # === Load Fusion Data ===
    print("📂 Loading fusion datasets...")
    rna_adata = sc.read_h5ad(args.RNA_path)
    adt_adata = sc.read_h5ad(args.ADT_path)
    
    print(f"RNA data shape: {rna_adata.shape}")
    print(f"ADT data shape: {adt_adata.shape}")
    
    # === Check for batch information in fusion data ===
    print("🔍 Checking batch information for horizontal integration...")
    if 'batch' in rna_adata.obs.columns:
        print(f"Batch distribution in RNA: {rna_adata.obs['batch'].value_counts()}")
        print("Horizontal integration will address batch effects between these batches")
    else:
        print("Warning: No 'batch' column found in RNA fusion data. Adding default batch labels.")
        n_cells = rna_adata.n_obs
        rna_adata.obs['batch'] = ['batch_1'] * (n_cells // 2) + ['batch_2'] * (n_cells - n_cells // 2)
    
    if 'batch' in adt_adata.obs.columns:
        print(f"Batch distribution in ADT: {adt_adata.obs['batch'].value_counts()}")
    else:
        print("Warning: No 'batch' column found in ADT fusion data. Adding default batch labels.")
        adt_adata.obs['batch'] = rna_adata.obs['batch'].copy()

    # === Ensure both datasets have same cells ===
    common_obs = rna_adata.obs_names.intersection(adt_adata.obs_names)
    print(f"Common cells: {len(common_obs)}, RNA cells: {rna_adata.n_obs}, ADT cells: {adt_adata.n_obs}")
    rna_adata = rna_adata[common_obs].copy()
    adt_adata = adt_adata[common_obs].copy()
    print(f"After intersection - RNA: {rna_adata.n_obs}, ADT: {adt_adata.n_obs}")

    # === Add spatial coordinates for horizontal integration if not present ===
    for adata, name in [(rna_adata, 'RNA'), (adt_adata, 'ADT')]:
        if 'spatial' not in adata.obsm.keys():
            print(f"Warning: No spatial coordinates found in {name}. Generating pseudo-spatial coordinates...")
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
            adata.obsm['spatial'] = adata.obsm['X_umap'].copy()
            print(f"Generated pseudo-spatial coordinates for {name}")
    
    # === Preprocessing (adapted for horizontal integration) ===
    print("🔧 Preprocessing for horizontal integration...")
    
    # Normalize RNA data
    rna_adata = normalize(rna_adata, copy=True, target_sum=1e4, log1p=True)
    
    # Normalize ADT data
    adt_adata = normalize(adt_adata, copy=True, target_sum=1e4, log1p=True)
    
    # Gene and protein selection for horizontal integration
    if args.select_genes > 0:
        rna_adata = geneSelection(rna_adata, n_top_genes=args.select_genes, copy=True)
    else:
        # Default gene selection for horizontal integration
        sc.pp.highly_variable_genes(rna_adata, n_top_genes=3000, flavor="seurat_v3")
        rna_adata = rna_adata[:, rna_adata.var['highly_variable']].copy()
    
    print(f"Selected {rna_adata.n_vars} genes for horizontal integration")
    print(f"Using {adt_adata.n_vars} proteins for horizontal integration")
    
    # Extract expression matrices
    gene_exp = rna_adata.X.toarray() if hasattr(rna_adata.X, 'toarray') else rna_adata.X
    protein_exp = adt_adata.X.toarray() if hasattr(adt_adata.X, 'toarray') else adt_adata.X
    
    # Extract spatial coordinates
    if 'spatial' in rna_adata.obsm:
        spatial_coords = rna_adata.obsm['spatial']
    else:
        print("Warning: Using default spatial coordinates")
        spatial_coords = np.random.randn(rna_adata.n_obs, 2) * 10
    
    print(f"Gene expression shape: {gene_exp.shape}")
    print(f"Protein expression shape: {protein_exp.shape}")
    print(f"Spatial coordinates shape: {spatial_coords.shape}")
    
    # === Train SpaMultiVAE model for horizontal integration ===
    print("Training SpaMultiVAE for horizontal integration...")
    start_time = time.time()
    
    # Initialize model with enhanced parameters for horizontal integration
    model = SPAMULTIVAE(
        gene_exp=gene_exp,
        protein_exp=protein_exp,
        spatial_coords=spatial_coords,
        device=args.device,
        random_seed=args.random_seed,
        
        # Enhanced architecture for horizontal integration
        encoder_layers=args.encoder_layers,
        GP_dim=args.GP_dim,
        Normal_dim=args.Normal_dim,
        gene_decoder_layers=args.gene_decoder_layers,
        protein_decoder_layers=args.protein_decoder_layers,
        
        # Regularization for batch effect removal
        gene_noise=args.gene_noise,
        protein_noise=args.protein_noise,
        dropoutE=args.dropoutE,
        dropoutD=args.dropoutD,
        
        # Dynamic VAE parameters for stability
        dynamicVAE=args.dynamicVAE,
        init_beta=args.init_beta,
        min_beta=args.min_beta,
        max_beta=args.max_beta,
        KL_loss=args.KL_loss,
        
        # Spatial GP parameters (enhanced for horizontal integration)
        fix_inducing_points=args.fix_inducing_points,
        grid_inducing_points=args.grid_inducing_points,
        inducing_point_steps=args.inducing_point_steps,
        inducing_point_nums=args.inducing_point_nums,
        fixed_gp_params=args.fixed_gp_params,
        loc_range=args.loc_range,
        kernel_scale=args.kernel_scale,
        
        # Training parameters
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_samples=args.num_samples
    )
    
    # Train the model with enhanced parameters for horizontal integration
    model.train(
        batch_size=args.batch_size,
        maxiter=args.maxiter,
        train_size=args.train_size,
        patience=args.patience
    )
    
    end_time = time.time()
    train_time = end_time - start_time
    print(f'Horizontal integration training time: {train_time:.2f} seconds')
    
    # === Extract embeddings ===
    print("📊 Extracting embeddings...")
    embeddings = model.get_latent_representation()
    
    print(f"Latent representation shape: {embeddings.shape}")
    
    # === Build Result AnnData ===
    adata = rna_adata.copy()
    adata.obsm['SpaMultiVAE'] = embeddings
    adata.uns['train_time'] = train_time
    adata.uns['integration_type'] = 'horizontal'
    
    # === Parse Dataset Info ===
    dataset_name, subset_name = parse_dataset_info(args)
    print(f"Detected dataset: {dataset_name}, subset: {subset_name}")

    # === Clustering and UMAP Generation ===
    tools = ['mclust', 'louvain', 'leiden', 'kmeans']
    
    # === Generate UMAP coordinates (store in adata, no plotting) ===
    print("🗺️ Generating UMAP coordinates...")
    sc.pp.neighbors(adata, use_rep='SpaMultiVAE', n_neighbors=30)
    sc.tl.umap(adata)
    print("UMAP coordinates generated and stored in adata.obsm['X_umap']")
    
    # === Run clustering methods ===
    print("🎯 Running clustering methods...")
    for tool in tools:
        print(f"  Running {tool} clustering...")
        adata = universal_clustering(
            adata,
            n_clusters=args.cluster_nums,
            used_obsm='SpaMultiVAE',
            method=tool,
            key=tool,
            use_pca=False
        )
    
    print("All clustering methods completed")
    
    # === Save results ===
    print("💾 Saving results...")
    save_dir = os.path.dirname(args.save_path)
    os.makedirs(save_dir, exist_ok=True)
    adata.write(args.save_path)
    
    print(f"Results saved to: {args.save_path}")
    print(f"Final adata shape: {adata.shape}")
    print(f"Obsm keys: {list(adata.obsm.keys())}")
    print(f"Integration completed successfully!")


if __name__ == "__main__":
    # Set environment variables for R and threading
    os.environ['R_HOME'] = '/home/zhenghong/miniconda3/envs/smobench/lib/R'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    main()