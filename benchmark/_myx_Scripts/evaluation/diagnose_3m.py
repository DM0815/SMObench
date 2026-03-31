#!/usr/bin/env python3
"""
SMOBench 3M (Three-Modality) Diagnostic Pipeline
==================================================
Diagnoses why 3M evaluation yields ARI~0, NMI~0 for all methods.

Steps:
  1. Data quality check - are the 3M ground truth clusters separable?
  2. Method output check - are embeddings valid (NaN, constant, etc.)?
  3. Clustering parameter sweep - is the resolution/k mismatch the cause?
  4. Decision tree - automated recommendation

Usage:
    python diagnose_3m.py --root /path/to/SMOBench-CLEAN
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

METHODS_3M = ['PRAGA_3M', 'SpaBalance_3M', 'SpatialGlue_3M']

METHOD_EMBEDDING_KEYS = {
    'PRAGA_3M':       ['PRAGA_3M', 'PRAGA', 'X_integrated', 'X_emb'],
    'SpaBalance_3M':  ['SpaBalance_3M', 'SpaBalance', 'X_integrated', 'X_emb'],
    'SpatialGlue_3M': ['SpatialGlue_3M', 'SpatialGlue', 'X_integrated', 'X_emb'],
}

GT_DIR = 'Dataset/withGT/3M_Simulation'

RESOLUTIONS = np.round(np.arange(0.1, 2.05, 0.1), 1).tolist()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dense(X):
    if sparse.issparse(X):
        return np.asarray(X.todense())
    return np.asarray(X)


def parse_args():
    parser = argparse.ArgumentParser(description='3M Diagnostic Pipeline')
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--methods', nargs='+', default=None)
    parser.add_argument('--skip_plot', action='store_true')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Step 1: Data Quality Check
# ---------------------------------------------------------------------------

def step1_data_quality(root_dir, output_dir):
    """Check if 3M simulation data has separable clusters."""
    print("\n" + "=" * 70)
    print("STEP 1: Data Quality Check")
    print("=" * 70)

    gt_base = os.path.join(root_dir, GT_DIR)
    results = {}

    # Load all modalities
    modalities = {}
    gt_labels = None
    gt_col = None

    for mod in ['RNA', 'ADT', 'ATAC']:
        path = os.path.join(gt_base, f'adata_{mod}.h5ad')
        if not os.path.isfile(path):
            print(f"  WARNING: {path} not found")
            continue

        adata = sc.read_h5ad(path)
        X = _to_dense(adata.X)
        modalities[mod] = {'adata': adata, 'X': X}
        print(f"  {mod}: shape={adata.shape}, "
              f"min={X.min():.3f}, max={X.max():.3f}, "
              f"sparsity={np.mean(X == 0):.3f}")

        # Extract GT labels
        if gt_labels is None:
            for col in ['Spatial_Label', 'Ground Truth', 'cell_type', 'cluster',
                        'domain', 'label', 'celltype']:
                if col in adata.obs.columns:
                    gt_labels = adata.obs[col].values
                    gt_col = col
                    break

        # Also check all obs columns for potential labels
        if gt_labels is None:
            print(f"  {mod} obs columns: {list(adata.obs.columns)}")

    if gt_labels is None:
        print("  ERROR: No ground truth labels found!")
        print("  Checking all obs columns across modalities...")
        for mod_name, mod_data in modalities.items():
            adata = mod_data['adata']
            print(f"    {mod_name}: {list(adata.obs.columns)}")
            # Check for any column with limited unique values
            for col in adata.obs.columns:
                n_unique = adata.obs[col].nunique()
                if 2 <= n_unique <= 50:
                    print(f"      Candidate label col: {col} ({n_unique} unique values)")
                    if gt_labels is None:
                        gt_labels = adata.obs[col].values
                        gt_col = col

    if gt_labels is not None:
        le = LabelEncoder()
        gt_labels_int = le.fit_transform(gt_labels.astype(str))
        n_clusters = len(le.classes_)
        print(f"\n  GT labels from '{gt_col}': {n_clusters} clusters")
        print(f"  Cluster sizes: {np.bincount(gt_labels_int)}")
        results['gt_col'] = gt_col
        results['n_clusters'] = n_clusters
        results['cluster_sizes'] = np.bincount(gt_labels_int).tolist()
    else:
        print("  CRITICAL: No suitable label column found!")
        results['gt_col'] = None
        results['n_clusters'] = 0
        gt_labels_int = None

    # Silhouette score per modality (on PCA-reduced data)
    sil_scores = {}
    for mod_name, mod_data in modalities.items():
        X = mod_data['X']
        # PCA reduce
        n_pcs = min(50, X.shape[1], X.shape[0] - 1)
        if n_pcs < 2:
            continue
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=n_pcs)
        X_pca = pca.fit_transform(X_scaled)

        if gt_labels_int is not None and len(np.unique(gt_labels_int)) > 1:
            sil = silhouette_score(X_pca, gt_labels_int, sample_size=min(1000, X.shape[0]))
            sil_scores[mod_name] = sil
            print(f"  Silhouette ({mod_name}, PCA{n_pcs}): {sil:.4f}")
        else:
            print(f"  Silhouette ({mod_name}): N/A (no labels)")

    # Concatenated modality silhouette
    if len(modalities) > 1 and gt_labels_int is not None:
        all_X = []
        min_cells = min(m['X'].shape[0] for m in modalities.values())
        for mod_data in modalities.values():
            X_mod = mod_data['X'][:min_cells]
            scaler = StandardScaler()
            all_X.append(scaler.fit_transform(X_mod))
        X_concat = np.hstack(all_X)
        n_pcs = min(50, X_concat.shape[1], X_concat.shape[0] - 1)
        pca = PCA(n_components=n_pcs)
        X_pca = pca.fit_transform(X_concat)
        sil_concat = silhouette_score(X_pca, gt_labels_int[:min_cells],
                                       sample_size=min(1000, min_cells))
        sil_scores['concatenated'] = sil_concat
        print(f"  Silhouette (concatenated, PCA{n_pcs}): {sil_concat:.4f}")

    results['silhouette_scores'] = sil_scores
    results['data_separable'] = any(v > 0.1 for v in sil_scores.values())
    print(f"\n  Data separable (any silhouette > 0.1): {results['data_separable']}")

    # UMAP visualisation
    if not args_global.skip_plot and gt_labels_int is not None:
        try:
            _plot_umap_data_quality(modalities, gt_labels_int, gt_col, output_dir)
        except Exception as e:
            print(f"  Plot error: {e}")

    return results, gt_labels_int, modalities


# ---------------------------------------------------------------------------
# Step 2: Method Output Check
# ---------------------------------------------------------------------------

def step2_method_output(root_dir, methods, gt_labels_int, output_dir):
    """Check embedding quality for each method."""
    print("\n" + "=" * 70)
    print("STEP 2: Method Output Check")
    print("=" * 70)

    results = {}

    for method in methods:
        print(f"\n  --- {method} ---")
        result_dir = os.path.join(root_dir, '_myx_Results', 'adata',
                                   'vertical_integration', method)
        if not os.path.isdir(result_dir):
            print(f"  SKIP: directory not found")
            results[method] = {'status': 'not_found'}
            continue

        h5ad_files = sorted(Path(result_dir).rglob('*.h5ad'))
        if not h5ad_files:
            print(f"  SKIP: no h5ad files")
            results[method] = {'status': 'no_h5ad'}
            continue

        adata = sc.read_h5ad(str(h5ad_files[0]))
        print(f"  File: {h5ad_files[0].name}")
        print(f"  Shape: {adata.shape}")
        print(f"  obsm keys: {list(adata.obsm.keys())}")
        print(f"  obs columns: {list(adata.obs.columns)}")

        # Find embedding
        embedding = None
        emb_key = None
        for key in METHOD_EMBEDDING_KEYS.get(method, [method, 'X_integrated', 'X_emb']):
            if key in adata.obsm:
                embedding = np.asarray(adata.obsm[key])
                emb_key = key
                break

        if embedding is None:
            print(f"  ERROR: no embedding found")
            results[method] = {'status': 'no_embedding', 'obsm_keys': list(adata.obsm.keys())}
            continue

        # Embedding diagnostics
        n_nan = np.sum(np.isnan(embedding))
        n_inf = np.sum(np.isinf(embedding))
        n_const_cols = np.sum(np.std(embedding, axis=0) < 1e-10)
        var_per_dim = np.var(embedding, axis=0)

        info = {
            'status': 'ok',
            'emb_key': emb_key,
            'emb_shape': embedding.shape,
            'n_nan': int(n_nan),
            'n_inf': int(n_inf),
            'n_constant_cols': int(n_const_cols),
            'emb_min': float(embedding.min()),
            'emb_max': float(embedding.max()),
            'emb_mean': float(embedding.mean()),
            'emb_std': float(embedding.std()),
            'var_range': (float(var_per_dim.min()), float(var_per_dim.max())),
        }

        print(f"  Embedding: {emb_key} → {embedding.shape}")
        print(f"  NaN={n_nan}, Inf={n_inf}, ConstCols={n_const_cols}/{embedding.shape[1]}")
        print(f"  Range: [{embedding.min():.4f}, {embedding.max():.4f}], "
              f"Mean={embedding.mean():.4f}, Std={embedding.std():.4f}")

        if n_nan > 0 or n_inf > 0:
            info['problem'] = 'numerical_issues'
        elif n_const_cols > embedding.shape[1] * 0.5:
            info['problem'] = 'too_many_constant_dims'
        else:
            info['problem'] = 'none'

        # Check existing clustering columns
        clustering_cols = [c for c in adata.obs.columns
                          if c in ['leiden', 'louvain', 'kmeans', 'mclust']]
        if clustering_cols:
            print(f"  Clustering columns: {clustering_cols}")
            for col in clustering_cols:
                vals = adata.obs[col]
                if hasattr(vals, 'cat'):
                    n_clusters_pred = vals.cat.categories.size
                else:
                    n_clusters_pred = len(np.unique(vals.dropna()))
                print(f"    {col}: {n_clusters_pred} clusters")
                info[f'n_clusters_{col}'] = n_clusters_pred

        # Silhouette on embedding
        if gt_labels_int is not None:
            min_n = min(len(gt_labels_int), embedding.shape[0])
            if min_n > 10 and len(np.unique(gt_labels_int[:min_n])) > 1:
                sil = silhouette_score(embedding[:min_n], gt_labels_int[:min_n],
                                       sample_size=min(1000, min_n))
                info['silhouette_embedding'] = float(sil)
                print(f"  Silhouette (embedding vs GT): {sil:.4f}")

        results[method] = info

    return results


# ---------------------------------------------------------------------------
# Step 3: Clustering Parameter Sweep
# ---------------------------------------------------------------------------

def step3_clustering_sweep(root_dir, methods, gt_labels_int, output_dir):
    """Sweep resolution parameter and compute ARI/NMI at each."""
    print("\n" + "=" * 70)
    print("STEP 3: Clustering Parameter Sweep")
    print("=" * 70)

    if gt_labels_int is None:
        print("  SKIP: no GT labels available")
        return {}

    n_gt_clusters = len(np.unique(gt_labels_int))
    print(f"  GT clusters: {n_gt_clusters}")

    all_rows = []

    for method in methods:
        print(f"\n  --- {method} ---")
        result_dir = os.path.join(root_dir, '_myx_Results', 'adata',
                                   'vertical_integration', method)
        h5ad_files = sorted(Path(result_dir).rglob('*.h5ad'))
        if not h5ad_files:
            continue

        adata = sc.read_h5ad(str(h5ad_files[0]))

        # Find embedding
        embedding = None
        for key in METHOD_EMBEDDING_KEYS.get(method, [method, 'X_integrated', 'X_emb']):
            if key in adata.obsm:
                embedding = np.asarray(adata.obsm[key])
                break
        if embedding is None:
            continue

        min_n = min(len(gt_labels_int), embedding.shape[0])
        gt_aligned = gt_labels_int[:min_n]
        emb_aligned = embedding[:min_n]

        # Build neighbor graph
        adata_tmp = sc.AnnData(X=emb_aligned)
        adata_tmp.obsm['X_emb'] = emb_aligned
        sc.pp.neighbors(adata_tmp, use_rep='X_emb', n_neighbors=15)

        best_ari = -1
        best_res = 0

        for res in RESOLUTIONS:
            try:
                sc.tl.leiden(adata_tmp, resolution=res, key_added='leiden_sweep')
                labels_pred = adata_tmp.obs['leiden_sweep'].cat.codes.values
                n_pred = len(np.unique(labels_pred))

                ari = adjusted_rand_score(gt_aligned, labels_pred)
                nmi = normalized_mutual_info_score(gt_aligned, labels_pred)

                all_rows.append({
                    'Method': method,
                    'Resolution': res,
                    'N_Clusters_Pred': n_pred,
                    'N_Clusters_GT': n_gt_clusters,
                    'ARI': ari,
                    'NMI': nmi,
                })

                if ari > best_ari:
                    best_ari = ari
                    best_res = res

                if res in [0.1, 0.5, 1.0, 1.5, 2.0]:
                    print(f"    res={res:.1f}: k_pred={n_pred:3d}, "
                          f"ARI={ari:.4f}, NMI={nmi:.4f}")
            except Exception as e:
                print(f"    res={res:.1f}: ERROR {e}")

        # Also try KMeans with exact k
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_gt_clusters, n_init=10, random_state=42)
        km_labels = km.fit_predict(emb_aligned)
        km_ari = adjusted_rand_score(gt_aligned, km_labels)
        km_nmi = normalized_mutual_info_score(gt_aligned, km_labels)
        all_rows.append({
            'Method': method,
            'Resolution': -1,  # sentinel for KMeans
            'N_Clusters_Pred': n_gt_clusters,
            'N_Clusters_GT': n_gt_clusters,
            'ARI': km_ari,
            'NMI': km_nmi,
        })

        print(f"    KMeans(k={n_gt_clusters}): ARI={km_ari:.4f}, NMI={km_nmi:.4f}")
        print(f"    Best Leiden: res={best_res:.1f}, ARI={best_ari:.4f}")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df.to_csv(os.path.join(output_dir, 'clustering_sweep.csv'), index=False)
        print(f"\n  Saved clustering_sweep.csv ({len(df)} rows)")

    return df


# ---------------------------------------------------------------------------
# Step 4: Decision Tree
# ---------------------------------------------------------------------------

def step4_decision(data_results, method_results, sweep_df, output_dir):
    """Automated decision tree for diagnosis."""
    print("\n" + "=" * 70)
    print("STEP 4: Diagnosis & Recommendation")
    print("=" * 70)

    decisions = []

    # Check data quality
    if not data_results.get('data_separable', False):
        decisions.append({
            'Issue': 'Data not separable',
            'Severity': 'HIGH',
            'Recommendation': 'Clusters in 3M simulation data overlap in feature space. '
                              'Report as limitation of synthetic data, not method failure.',
            'Action': 'Add discussion paragraph about synthetic data limitations.'
        })
    else:
        decisions.append({
            'Issue': 'Data separable',
            'Severity': 'OK',
            'Recommendation': 'Raw data has separable clusters. Problem is downstream.',
            'Action': 'Continue to method/clustering diagnosis.'
        })

    # Check method outputs
    for method, info in method_results.items():
        if info.get('status') != 'ok':
            decisions.append({
                'Issue': f'{method}: {info.get("status", "unknown")}',
                'Severity': 'HIGH',
                'Recommendation': f'Method output has issues: {info}',
                'Action': f'Rerun {method} or check logs.'
            })
        elif info.get('problem') != 'none':
            decisions.append({
                'Issue': f'{method}: {info["problem"]}',
                'Severity': 'MEDIUM',
                'Recommendation': f'Embedding quality issue: {info["problem"]}',
                'Action': 'Check method parameters or rerun.'
            })

    # Check clustering sweep
    if sweep_df is not None and not sweep_df.empty:
        for method in sweep_df['Method'].unique():
            sub = sweep_df[sweep_df['Method'] == method]
            best = sub.loc[sub['ARI'].idxmax()]

            if best['ARI'] > 0.3:
                decisions.append({
                    'Issue': f'{method}: resolution mismatch',
                    'Severity': 'MEDIUM',
                    'Recommendation': f'Best ARI={best["ARI"]:.4f} at resolution='
                                      f'{best["Resolution"]:.1f} (or KMeans). '
                                      f'Default resolution was wrong.',
                    'Action': f'Re-evaluate with optimal resolution={best["Resolution"]:.1f}.'
                })
            else:
                decisions.append({
                    'Issue': f'{method}: embedding cannot separate clusters',
                    'Severity': 'HIGH',
                    'Recommendation': f'Even with optimal clustering, best ARI={best["ARI"]:.4f}. '
                                      f'Embedding quality is poor.',
                    'Action': 'Check if method correctly handled 3-modality input.'
                })

    # Save decisions
    df_decisions = pd.DataFrame(decisions)
    df_decisions.to_csv(os.path.join(output_dir, 'diagnosis_decisions.csv'), index=False)

    print("\nDecisions:")
    for _, row in df_decisions.iterrows():
        print(f"  [{row['Severity']:6s}] {row['Issue']}")
        print(f"          -> {row['Recommendation']}")
        print(f"          Action: {row['Action']}")

    return df_decisions


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_umap_data_quality(modalities, gt_labels_int, gt_col, output_dir):
    """UMAP visualisation of raw data quality."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import umap

    n_mods = len(modalities) + 1  # +1 for concatenated
    fig, axes = plt.subplots(1, n_mods, figsize=(5 * n_mods, 4.5))
    if n_mods == 1:
        axes = [axes]

    min_cells = min(m['X'].shape[0] for m in modalities.values())
    gt_sub = gt_labels_int[:min_cells]

    idx = 0
    for mod_name, mod_data in modalities.items():
        X = mod_data['X'][:min_cells]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        n_pcs = min(50, X.shape[1], X.shape[0] - 1)
        pca = PCA(n_components=n_pcs)
        X_pca = pca.fit_transform(X_scaled)

        reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap = reducer.fit_transform(X_pca)

        scatter = axes[idx].scatter(X_umap[:, 0], X_umap[:, 1],
                                     c=gt_sub, cmap='tab10', s=3, alpha=0.6)
        axes[idx].set_title(f'{mod_name} (raw)', fontsize=11)
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
        idx += 1

    # Concatenated
    all_X = []
    for mod_data in modalities.values():
        X_mod = mod_data['X'][:min_cells]
        scaler = StandardScaler()
        all_X.append(scaler.fit_transform(X_mod))
    X_concat = np.hstack(all_X)
    n_pcs = min(50, X_concat.shape[1], X_concat.shape[0] - 1)
    pca = PCA(n_components=n_pcs)
    X_pca = pca.fit_transform(X_concat)
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_pca)

    scatter = axes[idx].scatter(X_umap[:, 0], X_umap[:, 1],
                                 c=gt_sub, cmap='tab10', s=3, alpha=0.6)
    axes[idx].set_title('Concatenated (raw)', fontsize=11)
    axes[idx].set_xticks([])
    axes[idx].set_yticks([])

    fig.suptitle(f'3M Simulation - Raw Data Quality (GT: {gt_col})',
                 fontsize=13, fontweight='bold')
    plt.colorbar(scatter, ax=axes[-1], label='Cluster')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'umap_data_quality.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"  Saved umap_data_quality.png")


def _plot_embedding_umap(root_dir, methods, gt_labels_int, output_dir):
    """UMAP visualisation of method embeddings."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import umap

    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4.5))
    if n_methods == 1:
        axes = [axes]

    for idx, method in enumerate(methods):
        result_dir = os.path.join(root_dir, '_myx_Results', 'adata',
                                   'vertical_integration', method)
        h5ad_files = sorted(Path(result_dir).rglob('*.h5ad'))
        if not h5ad_files:
            axes[idx].set_title(f'{method}\n(no data)')
            continue

        adata = sc.read_h5ad(str(h5ad_files[0]))
        embedding = None
        for key in METHOD_EMBEDDING_KEYS.get(method, [method, 'X_integrated', 'X_emb']):
            if key in adata.obsm:
                embedding = np.asarray(adata.obsm[key])
                break

        if embedding is None:
            axes[idx].set_title(f'{method}\n(no embedding)')
            continue

        min_n = min(len(gt_labels_int), embedding.shape[0])
        emb = embedding[:min_n]
        gt = gt_labels_int[:min_n]

        reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap = reducer.fit_transform(emb)

        scatter = axes[idx].scatter(X_umap[:, 0], X_umap[:, 1],
                                     c=gt, cmap='tab10', s=3, alpha=0.6)
        axes[idx].set_title(f'{method}\nemb: {emb.shape}', fontsize=11)
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])

    fig.suptitle('3M Simulation - Method Embeddings (colored by GT)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'umap_embeddings.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"  Saved umap_embeddings.png")


def _plot_resolution_sweep(sweep_df, output_dir):
    """Plot resolution sweep results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    methods = sweep_df['Method'].unique()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for method in methods:
        sub = sweep_df[(sweep_df['Method'] == method) & (sweep_df['Resolution'] > 0)]
        axes[0].plot(sub['Resolution'], sub['ARI'], 'o-', label=method, markersize=4)
        axes[1].plot(sub['Resolution'], sub['NMI'], 'o-', label=method, markersize=4)

    axes[0].set_xlabel('Resolution')
    axes[0].set_ylabel('ARI')
    axes[0].set_title('ARI vs Resolution')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Resolution')
    axes[1].set_ylabel('NMI')
    axes[1].set_title('NMI vs Resolution')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'resolution_sweep.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"  Saved resolution_sweep.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

args_global = None  # Will be set in main()


def main():
    global args_global
    args = parse_args()
    args_global = args
    root = os.path.abspath(args.root)

    methods = args.methods or METHODS_3M

    output_dir = os.path.join(root, '_myx_Results', 'evaluation', '3m', 'diagnostics')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Root:    {root}")
    print(f"Methods: {methods}")
    print(f"Output:  {output_dir}")

    total_start = time.time()

    # Step 1: Data quality
    data_results, gt_labels_int, modalities = step1_data_quality(root, output_dir)

    # Step 2: Method output check
    method_results = step2_method_output(root, methods, gt_labels_int, output_dir)

    # Step 3: Clustering sweep
    sweep_df = step3_clustering_sweep(root, methods, gt_labels_int, output_dir)

    # Step 4: Decision
    decisions = step4_decision(data_results, method_results, sweep_df, output_dir)

    # Plots
    if not args.skip_plot and gt_labels_int is not None:
        try:
            _plot_embedding_umap(root, methods, gt_labels_int, output_dir)
        except Exception as e:
            print(f"  Embedding UMAP plot error: {e}")

        if sweep_df is not None and not sweep_df.empty:
            try:
                _plot_resolution_sweep(sweep_df, output_dir)
            except Exception as e:
                print(f"  Resolution sweep plot error: {e}")

    # Save comprehensive report
    report = {
        'data_quality': data_results,
        'method_outputs': method_results,
        'n_sweep_rows': len(sweep_df) if sweep_df is not None and not sweep_df.empty else 0,
        'n_decisions': len(decisions),
    }

    import json
    with open(os.path.join(output_dir, 'diagnostic_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)

    elapsed = time.time() - total_start
    print(f"\n3M Diagnostic complete in {elapsed/60:.1f} min")
    print(f"Results in: {output_dir}")


if __name__ == '__main__':
    main()
