#!/usr/bin/env python3
"""
SpaMultiVAE horizontal integration script for SMOBench
Adapted for fusion datasets and batch effect removal
Supports only RNA+ADT spatial multi-omics integration
"""

import os
import sys
import time
import json
from datetime import datetime
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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
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
    total_start_time = time.time()

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
    
    # === Preprocessing following official SpaMultiVAE API ===
    print("Preprocessing for SpaMultiVAE...")

    # Get raw count matrices
    gene_raw = rna_adata.X.toarray().astype('float64') if hasattr(rna_adata.X, 'toarray') else np.array(rna_adata.X, dtype='float64')
    protein_raw = adt_adata.X.toarray().astype('float64') if hasattr(adt_adata.X, 'toarray') else np.array(adt_adata.X, dtype='float64')

    # Clean data: replace NaN/Inf, clamp negatives (log1p needs x >= 0)
    gene_raw = np.nan_to_num(gene_raw, nan=0.0, posinf=0.0, neginf=0.0)
    gene_raw = np.maximum(gene_raw, 0)
    protein_raw = np.nan_to_num(protein_raw, nan=0.0, posinf=0.0, neginf=0.0)
    protein_raw = np.maximum(protein_raw, 0)
    gene_mask = gene_raw.sum(axis=0) > 0
    gene_raw = gene_raw[:, gene_mask]
    protein_mask = protein_raw.sum(axis=0) > 0
    protein_raw = protein_raw[:, protein_mask]

    # Gene selection
    if args.select_genes > 0:
        importantGenes = geneSelection(gene_raw, n=args.select_genes, plot=False)
        gene_raw = gene_raw[:, importantGenes]

    # Extract and scale spatial coordinates
    if 'spatial' in rna_adata.obsm:
        loc = rna_adata.obsm['spatial'].astype('float64')
    else:
        sc.pp.neighbors(rna_adata)
        sc.tl.umap(rna_adata)
        loc = rna_adata.obsm['X_umap'].astype('float64')
    from sklearn.preprocessing import MinMaxScaler
    loc = MinMaxScaler().fit_transform(loc) * args.loc_range

    print(f"Gene matrix: {gene_raw.shape}, Protein matrix: {protein_raw.shape}, Locations: {loc.shape}")

    # Normalize RNA (size_factors + log1p)
    # filter_min_counts=False: we already pre-filtered zero genes/proteins above,
    # and filter_cells in normalize() would break alignment with loc and between modalities
    adata1 = sc.AnnData(gene_raw.copy(), dtype="float64")
    adata1 = normalize(adata1, filter_min_counts=False, size_factors=True, normalize_input=True, logtrans_input=True)

    # Normalize protein (no size_factors, with log1p)
    # Must copy protein_raw because normalize modifies X in place via log1p
    adata2 = sc.AnnData(protein_raw.copy(), dtype="float64")
    adata2 = normalize(adata2, filter_min_counts=False, size_factors=False, normalize_input=True, logtrans_input=True)

    # Protein background prior via GMM (separate copy to avoid double log1p)
    adata2_no_scale = sc.AnnData(protein_raw.copy(), dtype="float64")
    adata2_no_scale = normalize(adata2_no_scale, filter_min_counts=False, size_factors=False, normalize_input=False, logtrans_input=True)
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=2, covariance_type="diag", n_init=20).fit(adata2_no_scale.X)
    back_idx = np.argmin(gm.means_, axis=0)
    protein_log_back_mean = np.log(np.expm1(gm.means_[back_idx, np.arange(adata2_no_scale.n_vars)]))
    protein_log_back_scale = np.sqrt(gm.covariances_[back_idx, np.arange(adata2_no_scale.n_vars)])

    # Generate inducing points
    if args.grid_inducing_points and args.inducing_point_steps is not None:
        eps = 1e-5
        initial_inducing_points = np.mgrid[0:(1+eps):(1./args.inducing_point_steps), 0:(1+eps):(1./args.inducing_point_steps)].reshape(2, -1).T * args.loc_range
    else:
        from sklearn.cluster import KMeans
        n_ip = args.inducing_point_nums if args.inducing_point_nums else min(50, gene_raw.shape[0] // 10)
        initial_inducing_points = KMeans(n_clusters=n_ip, n_init=100).fit(loc).cluster_centers_

    # Auto batch size
    if args.batch_size == "auto":
        if gene_raw.shape[0] <= 1024:
            batch_size = 128
        elif gene_raw.shape[0] <= 2048:
            batch_size = 256
        else:
            batch_size = 512
    else:
        batch_size = int(args.batch_size)

    # === Train SpaMultiVAE ===
    print("Training SpaMultiVAE...")
    start_time = time.time()

    model = SPAMULTIVAE(
        gene_dim=adata1.n_vars, protein_dim=adata2.n_vars,
        GP_dim=args.GP_dim, Normal_dim=args.Normal_dim,
        encoder_layers=args.encoder_layers,
        gene_decoder_layers=args.gene_decoder_layers,
        protein_decoder_layers=args.protein_decoder_layers,
        gene_noise=args.gene_noise, protein_noise=args.protein_noise,
        encoder_dropout=args.dropoutE, decoder_dropout=args.dropoutD,
        fixed_inducing_points=args.fix_inducing_points,
        initial_inducing_points=initial_inducing_points,
        fixed_gp_params=args.fixed_gp_params,
        kernel_scale=args.kernel_scale,
        N_train=adata1.n_obs, KL_loss=args.KL_loss,
        dynamicVAE=args.dynamicVAE,
        init_beta=args.init_beta, min_beta=args.min_beta, max_beta=args.max_beta,
        protein_back_mean=protein_log_back_mean,
        protein_back_scale=protein_log_back_scale,
        dtype=torch.float32, device=args.device
    )

    # === Fix 4: NaN-aware Cholesky + gradient clipping to prevent divergence ===
    _orig_cholesky = torch.linalg.cholesky
    def _safe_cholesky(A, **kwargs):
        # Step 1: Detect and clean NaN/Inf in input matrix
        has_nan = torch.isnan(A).any() or torch.isinf(A).any()
        if has_nan:
            print(f"[WARN] NaN/Inf detected in Cholesky input, cleaning matrix (shape={A.shape})")
            A = A.clone()
            A[torch.isnan(A)] = 0.0
            A[torch.isinf(A)] = 0.0
            # Ensure positive diagonal
            diag = torch.diagonal(A, dim1=-2, dim2=-1)
            mask = diag <= 0
            if mask.any():
                A.diagonal(dim1=-2, dim2=-1)[mask] = 1.0
        # Step 2: Try progressive jitter
        for jit in [0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
            try:
                if jit > 0:
                    A_jit = A + jit * torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
                else:
                    A_jit = A
                return _orig_cholesky(A_jit, **kwargs)
            except torch._C._LinAlgError:
                continue
        # Step 3: Ultimate fallback - Cholesky of identity (safe, always works)
        print(f"[WARN] All Cholesky attempts failed, returning identity factor")
        n = A.shape[-1]
        return _orig_cholesky(torch.eye(n, device=A.device, dtype=A.dtype) * 1.001)
    torch.linalg.cholesky = _safe_cholesky
    model.svgp.jitter = 1e-3

    # Gradient hooks: clean NaN + clip to prevent parameter divergence
    def _clean_grad(grad):
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            grad = grad.clone()
            grad[torch.isnan(grad)] = 0.0
            grad[torch.isinf(grad)] = 0.0
        return torch.clamp(grad, -1.0, 1.0)
    for p in model.parameters():
        if p.requires_grad:
            p.register_hook(_clean_grad)

    # === Fix 6: Clamp Normal distribution scale to prevent float32 underflow/NaN ===
    from torch.distributions import Normal as _OrigNormal
    _orig_normal_init = _OrigNormal.__init__
    def _safe_normal_init(self, loc, scale, validate_args=None):
        scale = torch.nan_to_num(scale, nan=1e-6, posinf=1e6, neginf=1e-6)
        scale = torch.clamp(scale, min=1e-6)
        loc = torch.nan_to_num(loc, nan=0.0, posinf=1e6, neginf=-1e6)
        return _orig_normal_init(self, loc, scale, validate_args=validate_args)
    _OrigNormal.__init__ = _safe_normal_init

    # === Fix 7: Clamp PID _Kp_fun exp() to prevent math.exp overflow ===
    from math import exp as _math_exp
    def _safe_Kp_fun(self_pid, Err, scale=1):
        Err = max(min(float(Err), 500), -500)
        return 1.0 / (1.0 + float(scale) * _math_exp(Err))
    model.PID._Kp_fun = lambda Err, scale=1: _safe_Kp_fun(None, Err, scale)

    # === Fix 8: Use unique temp file for EarlyStopping checkpoint to avoid race ===
    import tempfile
    _tmpf = tempfile.NamedTemporaryFile(suffix='.pt', prefix='spamvae_', delete=False)
    _tmp_model_path = _tmpf.name
    _tmpf.close()

    model.train_model(
        pos=loc, gene_ncounts=adata1.X, gene_raw_counts=adata1.raw.X,
        gene_size_factors=adata1.obs.size_factors,
        protein_ncounts=adata2.X, protein_raw_counts=adata2.raw.X,
        lr=args.lr, weight_decay=args.weight_decay,
        batch_size=batch_size, num_samples=args.num_samples,
        train_size=args.train_size, maxiter=args.maxiter,
        patience=args.patience, save_model=False,
        model_weights=_tmp_model_path
    )

    # Clean up temp checkpoint
    if os.path.exists(_tmp_model_path):
        os.remove(_tmp_model_path)

    end_time = time.time()
    train_time = end_time - start_time
    print(f'Training time: {train_time:.2f} seconds')

    # === Extract embeddings ===
    print("Extracting embeddings...")
    embeddings = model.batching_latent_samples(
        X=loc, gene_Y=adata1.X, protein_Y=adata2.X, batch_size=batch_size
    )
    
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

    # === Save timing info ===
    total_time = time.time() - total_start_time
    dataset_name, _ = parse_dataset_info(args)
    timing_info = {
        "method": "SpaMultiVAE",
        "dataset": dataset_name,
        "integration_type": "horizontal",
        "modality": "RNA_ADT",
        "n_cells": adata.n_obs,
        "embedding_shape": list(adata.obsm["SpaMultiVAE"].shape),
        "training_time_s": round(train_time, 2),
        "total_time_s": round(total_time, 2),
        "device": args.device,
        "seed": args.random_seed,
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
    
    main()