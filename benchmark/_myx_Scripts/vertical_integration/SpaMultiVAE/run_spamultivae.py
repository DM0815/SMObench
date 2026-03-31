#!/usr/bin/env python3
"""
SpaMultiVAE integration script for SMOBench
Follows SpatialGlue style and official tutorial approach
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

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SpaMultiVAE for spatial multi-omics integration')
    
    # Required arguments
    parser.add_argument('--data_type', type=str, required=True, help='Data type')
    parser.add_argument('--RNA_path', type=str, required=True, help='RNA data path')
    parser.add_argument('--ADT_path', type=str, required=True, help='ADT data path')
    parser.add_argument('--save_path', type=str, required=True, help='Output save path')
    parser.add_argument('--cluster_nums', type=int, required=True, help='Number of clusters')
    parser.add_argument('--method', type=str, default='SpaMultiVAE', help='Method name')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    
    # Optional arguments - ATAC check
    parser.add_argument('--ATAC_path', type=str, default=None, help='ATAC path (not supported)')
    
    # SpaMultiVAE parameters
    parser.add_argument('--select_genes', type=int, default=0, help='Number of genes to select (0=all)')
    parser.add_argument('--select_proteins', type=int, default=0, help='Number of proteins to select (0=all)')
    parser.add_argument('--batch_size', default="auto", help='Batch size')
    parser.add_argument('--maxiter', type=int, default=5000, help='Max iterations')
    parser.add_argument('--train_size', type=float, default=0.95, help='Training proportion')
    parser.add_argument('--patience', type=int, default=200, help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay')
    parser.add_argument('--gene_noise', type=float, default=0, help='Gene noise level')
    parser.add_argument('--protein_noise', type=float, default=0, help='Protein noise level')
    parser.add_argument('--dropoutE', type=float, default=0, help='Encoder dropout')
    parser.add_argument('--dropoutD', type=float, default=0, help='Decoder dropout')
    parser.add_argument('--encoder_layers', nargs='+', default=[128, 64], type=int, help='Encoder layers')
    parser.add_argument('--GP_dim', type=int, default=2, help='GP dimensions')
    parser.add_argument('--Normal_dim', type=int, default=18, help='Normal dimensions')
    parser.add_argument('--gene_decoder_layers', nargs='+', default=[128], type=int, help='Gene decoder layers')
    parser.add_argument('--protein_decoder_layers', nargs='+', default=[128], type=int, help='Protein decoder layers')
    parser.add_argument('--dynamicVAE', action='store_true', default=False, help='Use dynamic VAE')
    parser.add_argument('--init_beta', type=float, default=10, help='Initial beta')
    parser.add_argument('--min_beta', type=float, default=4, help='Min beta')
    parser.add_argument('--max_beta', type=float, default=25, help='Max beta')
    parser.add_argument('--KL_loss', type=float, default=0.025, help='KL loss target')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples')
    parser.add_argument('--fix_inducing_points', action='store_true', default=True, help='Fix inducing points')
    parser.add_argument('--grid_inducing_points', action='store_true', default=True, help='Use grid inducing points')
    parser.add_argument('--inducing_point_steps', type=int, default=19, help='Inducing point steps')
    parser.add_argument('--inducing_point_nums', type=int, default=None, help='Number of inducing points')
    parser.add_argument('--fixed_gp_params', action='store_true', default=False, help='Fix GP params')
    parser.add_argument('--loc_range', type=float, default=20.0, help='Location range')
    parser.add_argument('--kernel_scale', type=float, default=20.0, help='Kernel scale')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--random_seed', type=int, default=2024, help='Random seed')
    
    return parser.parse_args()

def main():
    """Main function following tutorial style"""
    args = parse_args()
    
    # Check for ATAC path and reject
    if args.ATAC_path:
        print("Error: SpaMultiVAE does not support RNA+ATAC integration.")
        print("   This method only supports RNA+ADT data.")
        sys.exit(1)
    
    # Set random seed
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    print("="*60)
    print("🔬 SpaMultiVAE Integration - Tutorial Style")
    print("="*60)
    print(f"📊 Dataset: {args.dataset}")
    print(f"🎯 Target clusters: {args.cluster_nums}")
    print(f"🖥️  Device: {args.device}")
    print(f"⚙️  Dynamic VAE: {args.dynamicVAE}")
    
    start_time = time.time()
    
    try:
        # === 1. Load data (following tutorial) ===
        print("\n📂 Loading RNA and ADT data...")
        adata_rna = sc.read_h5ad(args.RNA_path)
        adata_adt = sc.read_h5ad(args.ADT_path)
        adata_rna.var_names_make_unique()
        adata_adt.var_names_make_unique()
        
        # Check cell consistency
        if adata_rna.n_obs != adata_adt.n_obs:
            raise ValueError(f"Cell count mismatch: RNA {adata_rna.n_obs} vs ADT {adata_adt.n_obs}")
        
        print(f"   Loaded {adata_rna.n_obs} cells")
        print(f"   RNA: {adata_rna.n_vars} genes")
        print(f"   ADT: {adata_adt.n_vars} proteins")
        
        # === 2. Prepare data matrices (following tutorial) ===
        print("\n🔧 Preparing data matrices...")
        
        # Get expression matrices
        x1 = adata_rna.X.toarray() if hasattr(adata_rna.X, 'toarray') else adata_rna.X
        x2 = adata_adt.X.toarray() if hasattr(adata_adt.X, 'toarray') else adata_adt.X
        x1 = x1.astype('float64')
        x2 = x2.astype('float64')
        
        # Get spatial coordinates
        if 'spatial' in adata_rna.obsm:
            loc = adata_rna.obsm['spatial'].astype('float64')
        else:
            print("   No spatial coordinates found, creating dummy coordinates")
            n_cells = adata_rna.n_obs
            loc = np.random.rand(n_cells, 2) * 100
            loc = loc.astype('float64')
        
        print(f"   📐 Data shapes - Genes: {x1.shape}, Proteins: {x2.shape}, Coords: {loc.shape}")
        
        # === 3. Parameter settings (following tutorial) ===
        print("\n⚙️  Setting parameters...")
        
        # Auto batch size
        if args.batch_size == "auto":
            if x1.shape[0] <= 1024:
                args.batch_size = 128
            elif x1.shape[0] <= 2048:
                args.batch_size = 256
            else:
                args.batch_size = 512
        else:
            args.batch_size = int(args.batch_size)
        
        print(f"   📦 Batch size: {args.batch_size}")
        
        # Gene selection
        if args.select_genes > 0:
            print(f"   🧬 Selecting top {args.select_genes} genes...")
            important_genes = geneSelection(x1, n=args.select_genes, plot=False)
            x1 = x1[:, important_genes]
        
        # Protein selection
        if args.select_proteins > 0:
            print(f"   🧬 Selecting top {args.select_proteins} proteins...")
            important_proteins = geneSelection(x2, n=args.select_proteins, plot=False)
            x2 = x2[:, important_proteins]
        
        # Scale spatial coordinates
        print("   📐 Scaling spatial coordinates...")
        scaler = MinMaxScaler()
        loc = scaler.fit_transform(loc) * args.loc_range
        
        print(f"   Final shapes - Genes: {x1.shape}, Proteins: {x2.shape}, Coords: {loc.shape}")
        
        # === 4. Setup inducing points (following tutorial) ===
        print("\n🎯 Setting up inducing points...")
        if args.grid_inducing_points:
            eps = 1e-5
            initial_inducing_points = np.mgrid[
                0:(1+eps):(1./args.inducing_point_steps), 
                0:(1+eps):(1./args.inducing_point_steps)
            ].reshape(2, -1).T * args.loc_range
            print(f"   📊 Grid inducing points: {initial_inducing_points.shape}")
        else:
            from sklearn.cluster import KMeans
            loc_kmeans = KMeans(n_clusters=args.inducing_point_nums, n_init=100).fit(loc)
            initial_inducing_points = loc_kmeans.cluster_centers_
            print(f"   📊 K-means inducing points: {initial_inducing_points.shape}")
        
        # === 5. Data preprocessing (following tutorial) ===
        print("\n🔧 Preprocessing data...")
        
        # Process gene data
        print("   🧬 Processing gene data...")
        adata1 = sc.AnnData(x1, dtype="float64")
        adata1 = normalize(adata1, size_factors=True, normalize_input=True, logtrans_input=True)
        
        # Process protein data
        print("   🧬 Processing protein data...")
        adata2 = sc.AnnData(x2, dtype="float64")
        adata2 = normalize(adata2, size_factors=False, normalize_input=True, logtrans_input=True)
        
        adata2_no_scale = sc.AnnData(x2, dtype="float64")
        adata2_no_scale = normalize(adata2_no_scale, size_factors=False, normalize_input=False, logtrans_input=True)
        
        # === 6. Fit GMM for protein background (following tutorial) ===
        print("   🎲 Fitting GMM for protein background...")
        gm = GaussianMixture(n_components=2, covariance_type="diag", n_init=20).fit(adata2_no_scale.X)
        back_idx = np.argmin(gm.means_, axis=0)
        protein_log_back_mean = np.log(np.expm1(gm.means_[back_idx, np.arange(adata2_no_scale.n_vars)]))
        protein_log_back_scale = np.sqrt(gm.covariances_[back_idx, np.arange(adata2_no_scale.n_vars)])
        print(f"   📊 Protein background mean shape: {protein_log_back_mean.shape}")
        
        # === 7. Initialize model (following tutorial) ===
        print("\n🏗️ Initializing SpaMultiVAE model...")
        model = SPAMULTIVAE(
            gene_dim=adata1.n_vars,
            protein_dim=adata2.n_vars,
            GP_dim=args.GP_dim,
            Normal_dim=args.Normal_dim,
            encoder_layers=args.encoder_layers,
            gene_decoder_layers=args.gene_decoder_layers,
            protein_decoder_layers=args.protein_decoder_layers,
            gene_noise=args.gene_noise,
            protein_noise=args.protein_noise,
            encoder_dropout=args.dropoutE,
            decoder_dropout=args.dropoutD,
            fixed_inducing_points=args.fix_inducing_points,
            initial_inducing_points=initial_inducing_points,
            fixed_gp_params=args.fixed_gp_params,
            kernel_scale=args.kernel_scale,
            N_train=adata1.n_obs,
            KL_loss=args.KL_loss,
            dynamicVAE=args.dynamicVAE,
            init_beta=args.init_beta,
            min_beta=args.min_beta,
            max_beta=args.max_beta,
            protein_back_mean=protein_log_back_mean,
            protein_back_scale=protein_log_back_scale,
            dtype=torch.float64,
            device=args.device
        )
        
        print("   Model initialized successfully")
        
        # === 8. Train model (following tutorial) ===
        print("\nTraining SpaMultiVAE...")
        t0 = time.time()
        
        model.train_model(
            pos=loc,
            gene_ncounts=adata1.X,
            gene_raw_counts=adata1.raw.X,
            gene_size_factors=adata1.obs.size_factors,
            protein_ncounts=adata2.X,
            protein_raw_counts=adata2.raw.X,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            train_size=args.train_size,
            maxiter=args.maxiter,
            patience=args.patience,
            save_model=False,  # Don't save intermediate model files
            model_weights=None
        )
        
        train_time = time.time() - t0
        print(f"   Training completed in {train_time:.2f} seconds")
        
        # === 9. Extract embeddings (following tutorial) ===
        print("\n🧮 Extracting latent embeddings...")
        final_latent = model.batching_latent_samples(
            X=loc, 
            gene_Y=adata1.X, 
            protein_Y=adata2.X, 
            batch_size=args.batch_size
        )
        print(f"   📊 Latent embedding shape: {final_latent.shape}")
        
        # === 10. Create integrated AnnData (SMOBench style) ===
        print("\n🔗 Creating integrated AnnData...")
        adata_integrated = adata_rna.copy()
        
        # Store integration results
        adata_integrated.obsm['SpaMultiVAE'] = final_latent
        adata_integrated.uns['train_time'] = train_time
        adata_integrated.uns['method'] = args.method
        
        # === 11. Compute UMAP ===
        print("\n📍 Computing UMAP...")
        sc.pp.neighbors(adata_integrated, use_rep='SpaMultiVAE', n_neighbors=15)
        sc.tl.umap(adata_integrated)
        
        # === 12. Clustering (SMOBench style - replace tutorial's KMeans) ===
        print(f"\n🎯 Running clustering with {args.cluster_nums} clusters...")
        clustering_methods = ['mclust', 'leiden', 'louvain', 'kmeans']
        
        for method in clustering_methods:
            try:
                print(f"   🔬 {method} clustering...")
                adata_integrated = universal_clustering(
                    adata_integrated,
                    n_clusters=args.cluster_nums,
                    used_obsm='SpaMultiVAE',
                    method=method,
                    key=method,
                    use_pca=False
                )
                print(f"      {method} completed")
            except Exception as e:
                print(f"      {method} failed: {e}")
                # Create placeholder
                adata_integrated.obs[method] = pd.Categorical(['0'] * adata_integrated.n_obs)
        
        # === 13. Save results ===
        print(f"\n💾 Saving results to {args.save_path}...")
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        adata_integrated.write(args.save_path)
        
        total_time = time.time() - start_time
        
        # === 14. Summary ===
        print("\n" + "="*60)
        print("SpaMultiVAE Integration Completed!")
        print("="*60)
        print(f"📁 Results saved: {args.save_path}")
        print(f"📊 Data shape: {adata_integrated.shape}")
        print(f"🧮 Embedding shape: {adata_integrated.obsm['SpaMultiVAE'].shape}")
        print(f"⏱️ Training time: {train_time:.2f} seconds")
        print(f"⏱️ Total time: {total_time:.2f} seconds")
        print()
        
        # Show clustering results
        for method in clustering_methods:
            if method in adata_integrated.obs.columns:
                n_clusters_found = len(adata_integrated.obs[method].unique())
                print(f"🎯 {method}: {n_clusters_found} clusters")
        
        print("="*60)
        
    except Exception as e:
        print(f"\nError during SpaMultiVAE integration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()