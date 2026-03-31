#!/usr/bin/env python3
"""
Generate improved 3M simulation data (RNA + ADT + ATAC) with well-separated clusters.

The original 3M simulation data from SpatialGlue has silhouette < 0 for all
modalities, making cluster evaluation meaningless. This script generates new
synthetic data with controlled cluster separation.

Design:
  1. Shared latent space via make_blobs (5 clusters, well-separated)
  2. Linear projections to RNA (1000 features), ADT (100), ATAC (1000)
  3. Modality-specific noise and sparsity patterns
  4. Grid-based spatial coordinates with spatial domain structure
  5. Output: 3 h5ad files matching original format

Usage:
    python generate_3m_simulation.py --root /path/to/SMOBench-CLEAN
    python generate_3m_simulation.py --root /path/to/SMOBench-CLEAN --check_only
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse
from sklearn.datasets import make_blobs
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


def generate_spatial_coordinates(n_cells, n_clusters, labels, seed=42):
    """Generate grid-based spatial coordinates with spatial domain structure.

    Cells from the same cluster are placed in spatially contiguous regions,
    mimicking real tissue spatial domains.
    """
    rng = np.random.RandomState(seed)

    # Create a grid large enough for all cells
    grid_side = int(np.ceil(np.sqrt(n_cells))) + 2

    # Assign cluster centers to different spatial regions
    region_centers = []
    for i in range(n_clusters):
        angle = 2 * np.pi * i / n_clusters
        cx = grid_side / 2 + grid_side * 0.3 * np.cos(angle)
        cy = grid_side / 2 + grid_side * 0.3 * np.sin(angle)
        region_centers.append((cx, cy))

    coords = np.zeros((n_cells, 2))
    for c in range(n_clusters):
        mask = labels == c
        n_c = mask.sum()
        cx, cy = region_centers[c]
        # Scatter cells around cluster center with some spread
        spread = grid_side * 0.12
        coords[mask, 0] = cx + rng.randn(n_c) * spread
        coords[mask, 1] = cy + rng.randn(n_c) * spread

    return coords.astype(np.float32)


def generate_rna_like(latent, n_features=1000, sparsity=0.7, seed=42):
    """Generate RNA-seq-like count data from latent space.

    - Non-negative counts
    - High sparsity (~70%)
    - Log-normal-like distribution
    """
    rng = np.random.RandomState(seed)
    n_cells = latent.shape[0]
    d_latent = latent.shape[1]

    # Random projection
    W = rng.randn(d_latent, n_features) * 0.2
    bias = rng.exponential(1.0, n_features)

    # Generate rates (positive)
    rates = np.exp(latent @ W * 0.25 + bias)

    # Poisson sampling
    counts = rng.poisson(rates)

    # Add sparsity via dropout
    dropout_mask = rng.random((n_cells, n_features)) < sparsity
    counts[dropout_mask] = 0

    return counts.astype(np.float32)


def generate_adt_like(latent, n_features=100, seed=42):
    """Generate ADT/protein-like count data from latent space.

    - Non-negative
    - Higher counts than RNA
    - Lower sparsity (~40%)
    """
    rng = np.random.RandomState(seed)
    n_cells = latent.shape[0]
    d_latent = latent.shape[1]

    W = rng.randn(d_latent, n_features) * 0.3
    bias = rng.exponential(3.0, n_features)

    rates = np.exp(latent @ W * 0.15 + np.log(bias + 1))
    counts = rng.poisson(rates * 3)

    # Lower sparsity
    dropout_mask = rng.random((n_cells, n_features)) < 0.4
    counts[dropout_mask] = 0

    return counts.astype(np.float32)


def generate_atac_like(latent, n_features=1000, sparsity=0.7, seed=42):
    """Generate ATAC-seq-like accessibility data from latent space.

    - Non-negative integers
    - High sparsity (~70%)
    - Lower max counts than RNA
    """
    rng = np.random.RandomState(seed)
    n_cells = latent.shape[0]
    d_latent = latent.shape[1]

    W = rng.randn(d_latent, n_features) * 0.2
    bias = rng.exponential(0.8, n_features)

    rates = np.exp(latent @ W * 0.25 + bias)
    counts = rng.poisson(rates)

    dropout_mask = rng.random((n_cells, n_features)) < sparsity
    counts[dropout_mask] = 0

    return counts.astype(np.float32)


def create_adata(X, spatial_coords, labels, modality_name, gene_prefix):
    """Create AnnData matching the original 3M simulation format."""
    adata = ad.AnnData(X=X)

    # obs
    adata.obs['Spatial_Label'] = pd.Categorical([str(l + 1) for l in labels])
    adata.obs_names = [f"cell_{i}" for i in range(X.shape[0])]

    # var
    adata.var_names = [f"{gene_prefix}_{i}" for i in range(X.shape[1])]

    # obsm - match original format
    adata.obsm['spatial'] = spatial_coords

    # uns
    adata.uns['log1p'] = {'base': None}

    return adata


def validate_data(adatas, labels):
    """Check that generated data meets quality criteria."""
    print("\n--- Data Validation ---")

    results = {}
    for mod_name, adata in adatas.items():
        X = adata.X
        if sparse.issparse(X):
            X = X.toarray()

        # Basic stats
        sparsity = np.mean(X == 0)
        print(f"  {mod_name}: shape={X.shape}, min={X.min():.1f}, max={X.max():.1f}, "
              f"sparsity={sparsity:.3f}")

        # Silhouette on PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        n_pcs = min(50, X.shape[1], X.shape[0] - 1)
        pca = PCA(n_components=n_pcs)
        X_pca = pca.fit_transform(X_scaled)

        sil = silhouette_score(X_pca, labels, sample_size=min(1000, X.shape[0]))
        results[mod_name] = sil
        print(f"  {mod_name} silhouette (PCA{n_pcs}): {sil:.4f}")

    # Concatenated
    all_X = []
    for adata in adatas.values():
        X = adata.X
        if sparse.issparse(X):
            X = X.toarray()
        scaler = StandardScaler()
        all_X.append(scaler.fit_transform(X))
    X_concat = np.hstack(all_X)
    n_pcs = min(50, X_concat.shape[1])
    pca = PCA(n_components=n_pcs)
    X_pca = pca.fit_transform(X_concat)
    sil_concat = silhouette_score(X_pca, labels)
    results['concatenated'] = sil_concat
    print(f"  concatenated silhouette (PCA{n_pcs}): {sil_concat:.4f}")

    # KMeans ARI
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    km = KMeans(n_clusters=len(np.unique(labels)), n_init=10, random_state=42)
    pred = km.fit_predict(X_pca)
    ari = adjusted_rand_score(labels, pred)
    print(f"  KMeans ARI (concatenated): {ari:.4f}")

    concat_sil_ok = sil_concat > 0.01
    difficulty_ok = ari < 0.95
    print(f"\n  Concatenated silhouette > 0.01: {concat_sil_ok} ({sil_concat:.4f})")
    print(f"  KMeans ARI: {ari:.4f} (target: 0.5 < ARI < 0.95)")
    print(f"  Difficulty OK (ARI < 0.95): {difficulty_ok}")

    return concat_sil_ok and ari > 0.5 and difficulty_ok


def main():
    parser = argparse.ArgumentParser(description='Generate improved 3M simulation data')
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--n_cells', type=int, default=1296, help='Number of cells (match original)')
    parser.add_argument('--n_clusters', type=int, default=5)
    parser.add_argument('--cluster_std', type=float, default=2.0,
                        help='Cluster spread in latent space (higher=harder)')
    parser.add_argument('--center_box', type=float, nargs=2, default=[-5.0, 5.0],
                        help='Range for cluster centers (smaller=harder)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--check_only', action='store_true',
                        help='Only validate existing data, do not generate')
    parser.add_argument('--no_backup', action='store_true')
    parser.add_argument('--output_name', type=str, default='3M_Simulation_v2',
                        help='Output directory name under Dataset/withGT/')
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    output_dir = os.path.join(root, 'Dataset', 'withGT', args.output_name)
    backup_dir = None  # No backup needed when using new directory name

    if args.check_only:
        print(f"=== Checking existing 3M data in {output_dir} ===")
        labels = None
        adatas = {}
        for mod in ['RNA', 'ADT', 'ATAC']:
            path = os.path.join(output_dir, f'adata_{mod}.h5ad')
            adata = sc.read_h5ad(path)
            adatas[mod] = adata
            if labels is None and 'Spatial_Label' in adata.obs.columns:
                le = LabelEncoder()
                labels = le.fit_transform(adata.obs['Spatial_Label'].astype(str))
        validate_data(adatas, labels)
        return

    print("=== Generating improved 3M simulation data ===")
    print(f"  n_cells={args.n_cells}, n_clusters={args.n_clusters}, "
          f"cluster_std={args.cluster_std}, seed={args.seed}")

    # Step 1: Generate shared latent space
    # Use fewer latent dims (5) so clusters overlap more through projection
    latent, labels = make_blobs(
        n_samples=args.n_cells, n_features=5, centers=args.n_clusters,
        cluster_std=args.cluster_std, center_box=tuple(args.center_box),
        random_state=args.seed
    )

    print(f"  Latent space: {latent.shape}")
    print(f"  Cluster sizes: {np.bincount(labels)}")

    # Step 2: Generate spatial coordinates
    spatial = generate_spatial_coordinates(
        args.n_cells, args.n_clusters, labels, seed=args.seed
    )

    # Step 3: Generate modality data
    X_rna = generate_rna_like(latent, n_features=1000, sparsity=0.7, seed=args.seed)
    X_adt = generate_adt_like(latent, n_features=100, seed=args.seed + 1)
    X_atac = generate_atac_like(latent, n_features=1000, sparsity=0.7, seed=args.seed + 2)

    # Step 4: Create AnnData objects
    adata_rna = create_adata(X_rna, spatial, labels, 'RNA', 'Gene')
    adata_adt = create_adata(X_adt, spatial, labels, 'ADT', 'Protein')
    adata_atac = create_adata(X_atac, spatial, labels, 'ATAC', 'Peak')

    adatas = {'RNA': adata_rna, 'ADT': adata_adt, 'ATAC': adata_atac}

    # Step 5: Validate
    passed = validate_data(adatas, labels)

    if not passed:
        # Auto-adjust: if too easy (ARI=1), increase std; if too hard, decrease std
        for retry_std in [2.5, 3.0, 1.5, 1.0]:
            print(f"\nRetrying with cluster_std={retry_std}...")
            latent2, labels2 = make_blobs(
                n_samples=args.n_cells, n_features=5, centers=args.n_clusters,
                cluster_std=retry_std, center_box=tuple(args.center_box),
                random_state=args.seed
            )
            spatial2 = generate_spatial_coordinates(
                args.n_cells, args.n_clusters, labels2, seed=args.seed
            )
            X_rna2 = generate_rna_like(latent2, n_features=1000, sparsity=0.7, seed=args.seed)
            X_adt2 = generate_adt_like(latent2, n_features=100, seed=args.seed + 1)
            X_atac2 = generate_atac_like(latent2, n_features=1000, sparsity=0.7, seed=args.seed + 2)

            adata_rna = create_adata(X_rna2, spatial2, labels2, 'RNA', 'Gene')
            adata_adt = create_adata(X_adt2, spatial2, labels2, 'ADT', 'Protein')
            adata_atac = create_adata(X_atac2, spatial2, labels2, 'ATAC', 'Peak')
            adatas = {'RNA': adata_rna, 'ADT': adata_adt, 'ATAC': adata_atac}
            labels = labels2

            passed = validate_data(adatas, labels2)
            if passed:
                break

    # Step 6: Save to new directory
    os.makedirs(output_dir, exist_ok=True)

    for mod_name, adata in adatas.items():
        path = os.path.join(output_dir, f'adata_{mod_name}.h5ad')
        adata.write(path)
        print(f"  Saved {path}")

    print(f"\nDone! New 3M simulation data saved to {output_dir}")
    if passed:
        print("Data quality: PASS")
    else:
        print("WARNING: Data quality below target. Review parameters.")


if __name__ == '__main__':
    main()
