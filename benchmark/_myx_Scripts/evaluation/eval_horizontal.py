#!/usr/bin/env python3
"""
SMOBench Horizontal Integration Evaluation
Computes SC (Spatial Coherence) + BioC (Biological Conservation) + BER (Batch Effect Removal)
for all 13 methods × 7 datasets × 4 clustering methods.

Usage:
    python eval_horizontal.py --root /path/to/SMOBench-CLEAN
    python eval_horizontal.py --root /path/to/SMOBench-CLEAN --methods CANDIES COSMOS
    python eval_horizontal.py --root /path/to/SMOBench-CLEAN --test
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALL_METHODS = [
    'CANDIES', 'COSMOS', 'MISO', 'PRAGA', 'PRESENT',
    'SMOPCA', 'SpaBalance', 'SpaFusion', 'SpaMI', 'SpaMosaic',
    'SpaMultiVAE', 'SpaMV', 'SpatialGlue',
]

WITHGT_DATASETS = ['Human_Lymph_Nodes', 'Human_Tonsils', 'Mouse_Embryos_S1', 'Mouse_Embryos_S2']
WOGT_DATASETS = ['Mouse_Thymus', 'Mouse_Spleen', 'Mouse_Brain']
ALL_DATASETS = WITHGT_DATASETS + WOGT_DATASETS

CLUSTERING_METHODS = ['leiden', 'louvain', 'kmeans', 'mclust']

# Method → supported datasets for horizontal integration
METHOD_DATASET_COMPAT = {
    'SpatialGlue': ALL_DATASETS,
    'SpaMosaic':   ALL_DATASETS,
    'PRESENT':     ALL_DATASETS,
    'COSMOS':      ALL_DATASETS,
    'MISO':        ALL_DATASETS,
    'SpaMV':       ALL_DATASETS,
    'CANDIES':     ALL_DATASETS,
    'SpaBalance':  ALL_DATASETS,
    'PRAGA':       ALL_DATASETS,
    'SpaMI':       ALL_DATASETS,
    'SMOPCA':      ALL_DATASETS,
    'SpaMultiVAE': ['Human_Lymph_Nodes', 'Human_Tonsils', 'Mouse_Thymus', 'Mouse_Spleen'],
    'SpaFusion':   ['Human_Lymph_Nodes', 'Human_Tonsils', 'Mouse_Spleen', 'Mouse_Thymus'],
}

DATASET_GT_INFO = {
    'Human_Lymph_Nodes': {'type': 'RNA_ADT',  'gt_dir': 'Human_Lymph_Nodes'},
    'Human_Tonsils':     {'type': 'RNA_ADT',  'gt_dir': 'Human_Tonsils'},
    'Mouse_Embryos_S1':  {'type': 'RNA_ATAC', 'gt_dir': 'Mouse_Embryos_S1'},
    'Mouse_Embryos_S2':  {'type': 'RNA_ATAC', 'gt_dir': 'Mouse_Embryos_S2'},
}

DATASET_TYPES = {
    'Human_Lymph_Nodes': 'RNA_ADT', 'Human_Tonsils': 'RNA_ADT',
    'Mouse_Embryos_S1': 'RNA_ATAC', 'Mouse_Embryos_S2': 'RNA_ATAC',
    'Mouse_Thymus': 'RNA_ADT', 'Mouse_Spleen': 'RNA_ADT',
    'Mouse_Brain': 'RNA_ATAC',
}

# Result directories may use short or full names
DATASET_DIR_ALIASES = {
    'Human_Lymph_Nodes': ['Human_Lymph_Nodes', 'HLN'],
    'Human_Tonsils':     ['Human_Tonsils', 'HT'],
    'Mouse_Embryos_S1':  ['Mouse_Embryos_S1', 'MISAR_S1'],
    'Mouse_Embryos_S2':  ['Mouse_Embryos_S2', 'MISAR_S2'],
    'Mouse_Thymus':      ['Mouse_Thymus'],
    'Mouse_Spleen':      ['Mouse_Spleen'],
    'Mouse_Brain':       ['Mouse_Brain'],
}


def parse_args():
    parser = argparse.ArgumentParser(description='SMOBench Horizontal Integration Evaluation')
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory of SMOBench-CLEAN')
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Subset of methods to evaluate (default: all 13)')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Subset of datasets to evaluate (default: all 7)')
    parser.add_argument('--clustering', nargs='+', default=None,
                        help='Subset of clustering methods (default: all 4)')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: evaluate only 1 method × 1 dataset')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Setup: import Eval core functions
# ---------------------------------------------------------------------------

def setup_eval_imports(root_dir):
    """Add Eval/ to sys.path and import core functions."""
    eval_dir = os.path.join(root_dir, 'Eval')
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)

    from src.demo import (eval_horizontal_integration,
                          save_evaluation_results)
    try:
        from src.clustering import knn_adj_matrix
    except ImportError:
        from src.clustering_simple import knn_adj_matrix

    return eval_horizontal_integration, save_evaluation_results, knn_adj_matrix


# ---------------------------------------------------------------------------
# Ground truth & batch label loading
# ---------------------------------------------------------------------------

def load_horizontal_ground_truth(root_dir, dataset_name):
    """Load concatenated GT labels for horizontal integration (all slices combined)."""
    if dataset_name not in DATASET_GT_INFO:
        return None

    info = DATASET_GT_INFO[dataset_name]
    dataset_path = os.path.join(root_dir, 'Dataset', 'withGT',
                                info['type'], info['gt_dir'])
    if not os.path.isdir(dataset_path):
        return None

    all_labels = []
    for slice_dir in sorted(os.listdir(dataset_path)):
        gt_path = os.path.join(dataset_path, slice_dir, 'adata_RNA.h5ad')
        if not os.path.isfile(gt_path):
            continue
        try:
            adata_gt = sc.read_h5ad(gt_path)
            if 'Spatial_Label' in adata_gt.obs.columns:
                all_labels.extend(adata_gt.obs['Spatial_Label'].values)
        except Exception as e:
            print(f"    [GT] Failed to load {gt_path}: {e}")

    return np.array(all_labels) if all_labels else None


def get_batch_labels(adata):
    """Extract batch labels from horizontal integration h5ad."""
    for col in ['batch', 'slice', 'sample', 'orig.ident', 'batch_id', 'slice_id']:
        if col in adata.obs.columns:
            vals = adata.obs[col].values
            if hasattr(vals, 'categories'):
                vals = vals.astype(str)
            return vals

    # Fallback: assign based on cell index position
    n = adata.n_obs
    n_batches = min(4, max(2, n // 1000))
    batch_size = n // n_batches
    labels = []
    for i in range(n_batches):
        end = (i + 1) * batch_size if i < n_batches - 1 else n
        labels.extend([f'batch_{i}'] * (end - i * batch_size))
    return np.array(labels[:n])


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate_all(root_dir, methods, datasets, clustering_methods, test_mode=False):
    """Run horizontal evaluation for all method × dataset × clustering combinations."""

    eval_func, save_func, knn_adj = setup_eval_imports(root_dir)

    input_dir = os.path.join(root_dir, '_myx_Results', 'adata', 'horizontal_integration')
    output_dir = os.path.join(root_dir, '_myx_Results', 'evaluation', 'horizontal')
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    total_start = time.time()

    for method in methods:
        method_dir = os.path.join(input_dir, method)
        if not os.path.isdir(method_dir):
            print(f"[SKIP] {method}: directory not found")
            continue

        # Determine compatible datasets for this method
        compat = METHOD_DATASET_COMPAT.get(method, ALL_DATASETS)

        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print(f"{'='*60}")

        for dataset in datasets:
            if dataset not in compat:
                print(f"  [{dataset}] SKIP (unsupported)")
                continue

            # Try aliases for directory name (e.g. HLN → Human_Lymph_Nodes)
            dataset_dir = None
            for alias in DATASET_DIR_ALIASES.get(dataset, [dataset]):
                candidate = os.path.join(method_dir, alias)
                if os.path.isdir(candidate):
                    dataset_dir = candidate
                    break
            if dataset_dir is None:
                print(f"  [{dataset}] SKIP (no results)")
                continue

            # Find h5ad file (horizontal integration produces 1 file per dataset)
            h5ad_files = sorted(Path(dataset_dir).rglob('*.h5ad'))
            if not h5ad_files:
                print(f"  [{dataset}] SKIP (no h5ad files)")
                continue

            h5ad_path = h5ad_files[0]
            print(f"  [{dataset}] Loading {h5ad_path.name}")

            try:
                adata = sc.read_h5ad(str(h5ad_path))
            except Exception as e:
                print(f"  [{dataset}] ERROR loading: {e}")
                continue

            # Get embeddings
            embeddings = None
            for key in [method, 'X_integrated', 'X_emb', 'embeddings']:
                if key in adata.obsm:
                    embeddings = adata.obsm[key]
                    break
            if embeddings is None:
                print(f"  [{dataset}] SKIP (no embeddings)")
                continue

            # Spatial coordinates
            spatial_coords = None
            for key in ['spatial', 'X_spatial']:
                if key in adata.obsm:
                    spatial_coords = adata.obsm[key]
                    break
            if spatial_coords is None:
                print(f"  [{dataset}] WARN: no spatial coords, using first 2 dims")
                spatial_coords = embeddings[:, :2]

            # Ground truth
            y_GT = load_horizontal_ground_truth(root_dir, dataset)
            if y_GT is not None:
                y_GT = np.asarray(y_GT, dtype=int)

            # Batch labels
            batch_labels = get_batch_labels(adata)

            # Align lengths
            lengths = [embeddings.shape[0], spatial_coords.shape[0], len(batch_labels)]
            if y_GT is not None:
                lengths.append(len(y_GT))
            min_len = min(lengths)
            max_len = max(lengths)
            if min_len != max_len:
                print(f"  [{dataset}] WARN: length mismatch ({min_len}~{max_len}), truncating")
                embeddings = embeddings[:min_len]
                spatial_coords = spatial_coords[:min_len]
                batch_labels = batch_labels[:min_len]
                if y_GT is not None:
                    y_GT = y_GT[:min_len]

            # Adjacency matrix
            adj_matrix = knn_adj(embeddings)

            # Per-clustering evaluation
            for clust in clustering_methods:
                if clust not in adata.obs.columns:
                    continue

                y_pred_raw = adata.obs[clust]
                if hasattr(y_pred_raw, 'cat'):
                    y_pred = y_pred_raw.cat.codes.values
                elif y_pred_raw.dtype == 'object':
                    uniq = sorted(set(y_pred_raw))
                    label_map = {l: i for i, l in enumerate(uniq)}
                    y_pred = np.array([label_map[l] for l in y_pred_raw])
                else:
                    y_pred = y_pred_raw.values
                y_pred = np.asarray(y_pred[:min_len], dtype=int)

                metrics = eval_func(
                    embeddings=embeddings,
                    adj_matrix=adj_matrix,
                    y_pred=y_pred,
                    y_GT=y_GT,
                    spatial_coords=spatial_coords,
                    batch_labels=batch_labels,
                    method_name=method,
                    dataset_name=dataset,
                    slice_name='horizontal',
                    clustering_method=clust,
                )

                # Save per-file CSV
                method_out = os.path.join(output_dir, method)
                save_func(
                    metrics_dict=metrics,
                    output_dir=method_out,
                    method_name=method,
                    dataset_name=dataset,
                    slice_name='horizontal',
                    clustering_method=clust,
                    has_gt=(y_GT is not None),
                )

                all_results.append(metrics)

            has_gt_str = 'withGT' if y_GT is not None else 'woGT'
            n_batch = len(np.unique(batch_labels))
            print(f"  [{dataset}] OK — {adata.n_obs} cells, {has_gt_str}, {n_batch} batches")

            if test_mode:
                break
        if test_mode:
            break

    elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Horizontal evaluation complete: {len(all_results)} results in {elapsed/60:.1f} min")
    print(f"Per-file CSVs saved to: {output_dir}")
    return all_results, output_dir


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _process_horizontal_by_directory(output_dir):
    """Parse horizontal per-file CSVs using 'horizontal' keyword in filename.

    Filenames: {Method}_{Dataset}_horizontal_{clustering}_{gt}.csv
    inside directory: output_dir / {Method} /
    """
    from collections import defaultdict

    HORIZ_SC_METRICS = ['Moran Index']
    HORIZ_BER_METRICS = ['kBET', 'KNN_connectivity', 'bASW', 'iLISI', 'PCR']
    HORIZ_BIOC_WITHGT = ['ARI', 'NMI', 'asw_celltype', 'graph_clisi']
    HORIZ_BIOC_WOGT = ['Davies-Bouldin Index', 'Silhouette Coefficient', 'Calinski-Harabaz Index']

    results_by_clustering = {c: [] for c in CLUSTERING_METHODS}

    for method_dir in sorted(Path(output_dir).iterdir()):
        if not method_dir.is_dir():
            continue
        method_name = method_dir.name

        for csv_path in sorted(method_dir.glob('*.csv')):
            try:
                fname = csv_path.stem
                parts = fname.split('_')
                if 'horizontal' not in parts:
                    continue

                h_idx = parts.index('horizontal')
                dataset_name = '_'.join(parts[1:h_idx])
                clustering = parts[h_idx + 1]
                gt_status = parts[-1]

                if clustering not in results_by_clustering:
                    continue

                df = pd.read_csv(csv_path)

                if 'Metric' in df.columns and 'Value' in df.columns:
                    metrics_dict = dict(zip(df['Metric'], df['Value']))
                else:
                    metrics_dict = df.iloc[0].to_dict() if len(df) else {}

                metrics_dict.update({
                    'Method': method_name,
                    'Dataset': dataset_name,
                    'Clustering': clustering,
                    'GT_Available': (gt_status == 'withGT'),
                    'Dataset_Type': DATASET_TYPES.get(dataset_name, 'Unknown'),
                })
                results_by_clustering[clustering].append(metrics_dict)
            except Exception as e:
                print(f"  Warning: error parsing {csv_path}: {e}")
                continue

    return results_by_clustering


def _aggregate_horizontal(results_list):
    """Aggregate horizontal results to dataset level."""
    from collections import defaultdict

    HORIZ_SC_METRICS = ['Moran Index']
    HORIZ_BER_METRICS = ['kBET', 'KNN_connectivity', 'bASW', 'iLISI', 'PCR']
    HORIZ_BIOC_WITHGT = ['ARI', 'NMI', 'asw_celltype', 'graph_clisi']
    HORIZ_BIOC_WOGT = ['Davies-Bouldin Index', 'Silhouette Coefficient', 'Calinski-Harabaz Index']

    if not results_list:
        return pd.DataFrame()

    grouped = defaultdict(list)
    for r in results_list:
        grouped[(r['Method'], r['Dataset'])].append(r)

    rows = []
    for (method, dataset), slices in grouped.items():
        has_gt = slices[0].get('GT_Available', False)
        agg = {
            'Method': method, 'Dataset': dataset,
            'Dataset_Type': DATASET_TYPES.get(dataset, 'Unknown'),
            'Clustering': slices[0].get('Clustering', 'leiden'),
            'GT_Available': has_gt,
            'Num_Slices': len(slices),
        }
        all_metrics = HORIZ_SC_METRICS + HORIZ_BER_METRICS + (HORIZ_BIOC_WITHGT if has_gt else HORIZ_BIOC_WOGT)
        for m in all_metrics:
            vals = [s.get(m, np.nan) for s in slices]
            vals = [v for v in vals if pd.notna(v)]
            agg[m] = np.nanmean(vals) if vals else np.nan
        rows.append(agg)

    return pd.DataFrame(rows)


def _score_horizontal(df):
    """Calculate SC, BVC, BER, Final scores for horizontal."""
    from generate_final_results import normalize_metric_value

    HORIZ_SC_METRICS = ['Moran Index']
    HORIZ_BER_METRICS = ['kBET', 'KNN_connectivity', 'bASW', 'iLISI', 'PCR']
    HORIZ_BIOC_WITHGT = ['ARI', 'NMI', 'asw_celltype', 'graph_clisi']
    HORIZ_BIOC_WOGT = ['Davies-Bouldin Index', 'Silhouette Coefficient', 'Calinski-Harabaz Index']
    LOWER_IS_BETTER = ['Davies-Bouldin Index']

    df = df.copy()
    for (clust, has_gt), sub in df.groupby(['Clustering', 'GT_Available']):
        mask = (df['Clustering'] == clust) & (df['GT_Available'] == has_gt)
        if len(sub) < 2:
            continue

        # SC
        for m in HORIZ_SC_METRICS:
            if m in sub.columns:
                df.loc[mask, 'SC_Score'] = df.loc[mask, m]

        # BioC
        bioc_metrics = HORIZ_BIOC_WITHGT if has_gt else HORIZ_BIOC_WOGT
        bioc_cols = []
        for m in bioc_metrics:
            if m in sub.columns:
                if m in ['Davies-Bouldin Index', 'Calinski-Harabaz Index']:
                    all_vals = sub[m].dropna().tolist()
                    if all_vals:
                        df.loc[mask, f'{m}_normalized'] = [
                            normalize_metric_value(v, m, all_vals) for v in df.loc[mask, m]
                        ]
                        bioc_cols.append(f'{m}_normalized')
                else:
                    bioc_cols.append(m)
        if bioc_cols:
            df.loc[mask, 'BVC_Score'] = df.loc[mask, bioc_cols].mean(axis=1)

        # BER
        ber_cols = [m for m in HORIZ_BER_METRICS if m in sub.columns]
        if ber_cols:
            df.loc[mask, 'BER_Score'] = df.loc[mask, ber_cols].mean(axis=1)

        # Final
        score_cols = [c for c in ['SC_Score', 'BVC_Score', 'BER_Score'] if c in df.columns]
        if score_cols:
            df.loc[mask, 'Final_Score'] = df.loc[mask, score_cols].mean(axis=1)

    return df


def _create_horizontal_tables(df, methods_order):
    """Create grouped summary tables with full dataset names."""
    GROUPS = OrderedDict([
        ("RNA_ADT_withGT", ["Human_Lymph_Nodes", "Human_Tonsils"]),
        ("RNA_ADT_woGT",   ["Mouse_Thymus", "Mouse_Spleen"]),
        ("RNA_ATAC_withGT", ["Mouse_Embryos_S1", "Mouse_Embryos_S2"]),
        ("RNA_ATAC_woGT",  ["Mouse_Brain"]),
    ])
    ALL_DS = [d for g in GROUPS.values() for d in g]

    tables = OrderedDict()
    if df.empty:
        for gn, ds in GROUPS.items():
            tables[gn] = pd.DataFrame(index=methods_order, columns=ds + ["Average"], dtype=float)
        tables["Comprehensive"] = pd.DataFrame(index=methods_order, columns=ALL_DS + ["Average"], dtype=float)
        return tables

    present_methods = sorted(set(df['Method'].unique()) | set(methods_order))

    def build(sub, datasets):
        if sub.empty:
            pv = pd.DataFrame(index=present_methods, columns=datasets, dtype=float)
        else:
            pv = sub.pivot_table(index='Method', columns='Dataset', values='Final_Score', aggfunc='first')
            pv = pv.reindex(index=present_methods).reindex(columns=datasets)
        pv['Average'] = pv.mean(axis=1, skipna=True)
        return pv.round(3)

    for gn, ds in GROUPS.items():
        tables[gn] = build(df[df['Dataset'].isin(ds)], ds)
    tables['Comprehensive'] = build(df[df['Dataset'].isin(ALL_DS)], ALL_DS)
    return tables


def aggregate_results(output_dir, root_dir):
    """Read per-file CSVs and generate summary tables (custom parser for full dataset names)."""
    eval_dir = os.path.join(root_dir, 'Eval')
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)

    summary_dir = os.path.join(root_dir, '_myx_Results', 'evaluation', 'summary')
    os.makedirs(summary_dir, exist_ok=True)

    results_by_clustering = _process_horizontal_by_directory(output_dir)

    for clust_method in CLUSTERING_METHODS:
        if not results_by_clustering.get(clust_method):
            continue

        print(f"\nAggregating {clust_method} results...")
        agg_df = _aggregate_horizontal(results_by_clustering[clust_method])
        if agg_df.empty:
            continue

        scored_df = _score_horizontal(agg_df)
        group_tables = _create_horizontal_tables(scored_df, ALL_METHODS)

        scored_df.to_csv(os.path.join(summary_dir, f'horizontal_detailed_{clust_method}.csv'), index=False)
        for group_name, table in group_tables.items():
            if not table.empty:
                table.to_csv(os.path.join(summary_dir, f'horizontal_{group_name}_{clust_method}.csv'))

        print(f"  {clust_method}: {len(scored_df)} entries → summary saved")

    print(f"\nSummary tables saved to: {summary_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    root = os.path.abspath(args.root)

    methods = args.methods or ALL_METHODS
    datasets = args.datasets or ALL_DATASETS
    clustering = args.clustering or CLUSTERING_METHODS

    print(f"Root:       {root}")
    print(f"Methods:    {methods}")
    print(f"Datasets:   {datasets}")
    print(f"Clustering: {clustering}")
    print(f"Test mode:  {args.test}")

    # Step 1: Compute metrics
    all_results, output_dir = evaluate_all(
        root_dir=root,
        methods=methods,
        datasets=datasets,
        clustering_methods=clustering,
        test_mode=args.test,
    )

    # Step 2: Aggregate into summary tables
    if all_results and not args.test:
        aggregate_results(output_dir, root)

    print("\nDone.")


if __name__ == '__main__':
    main()
