#!/usr/bin/env python3
"""
SMOBench Vertical Integration Evaluation
Computes SC (Spatial Coherence) + BioC (Biological Conservation) metrics
for all 15 methods × 7 datasets × 4 clustering methods.

Usage:
    python eval_vertical.py --root /path/to/SMOBench-CLEAN
    python eval_vertical.py --root /path/to/SMOBench-CLEAN --methods CANDIES COSMOS
    python eval_vertical.py --root /path/to/SMOBench-CLEAN --test
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALL_METHODS = [
    'CANDIES', 'COSMOS', 'MISO', 'MultiGATE', 'PRAGA', 'PRESENT',
    'SMOPCA', 'SpaBalance', 'SpaFusion', 'SpaMI', 'SpaMosaic',
    'SpaMultiVAE', 'SpaMV', 'SpatialGlue', 'SWITCH',
]

# Use full names consistent with Dataset/ directory
WITHGT_DATASETS = ['Human_Lymph_Nodes', 'Human_Tonsils', 'Mouse_Embryos_S1', 'Mouse_Embryos_S2']
WOGT_DATASETS = ['Mouse_Thymus', 'Mouse_Spleen', 'Mouse_Brain']
ALL_DATASETS = WITHGT_DATASETS + WOGT_DATASETS

CLUSTERING_METHODS = ['leiden', 'louvain', 'kmeans', 'mclust']

# Methods that only support RNA_ADT (no ATAC datasets)
METHOD_DATASET_SKIP = {
    'SpaMultiVAE': {'Mouse_Embryos_S1', 'Mouse_Embryos_S2', 'Mouse_Brain'},
    'SpaFusion':   {'Mouse_Embryos_S1', 'Mouse_Embryos_S2', 'Mouse_Brain'},
}

# Dataset full name → GT directory info
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

# Vertical integration results use short directory names
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
    parser = argparse.ArgumentParser(description='SMOBench Vertical Integration Evaluation')
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory of SMOBench-CLEAN')
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Subset of methods to evaluate (default: all)')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Subset of datasets to evaluate (default: all)')
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

    from src.demo import (eval_vertical_integration,
                          save_evaluation_results)
    try:
        from src.clustering import knn_adj_matrix
    except ImportError:
        from src.clustering_simple import knn_adj_matrix

    return eval_vertical_integration, save_evaluation_results, knn_adj_matrix


# ---------------------------------------------------------------------------
# Ground truth loading (fixed paths)
# ---------------------------------------------------------------------------

def load_ground_truth(root_dir, dataset_name, slice_name):
    """Load GT labels from Dataset/withGT/."""
    if dataset_name not in DATASET_GT_INFO:
        return None

    info = DATASET_GT_INFO[dataset_name]
    gt_path = os.path.join(root_dir, 'Dataset', 'withGT',
                           info['type'], info['gt_dir'],
                           slice_name, 'adata_RNA.h5ad')
    try:
        adata_gt = sc.read_h5ad(gt_path)
        if 'Spatial_Label' in adata_gt.obs.columns:
            return adata_gt.obs['Spatial_Label'].values
    except Exception as e:
        print(f"    [GT] Failed to load {gt_path}: {e}")
    return None


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate_all(root_dir, methods, datasets, clustering_methods, test_mode=False):
    """Run evaluation for all method × dataset × clustering combinations."""

    eval_func, save_func, knn_adj = setup_eval_imports(root_dir)

    input_dir = os.path.join(root_dir, '_myx_Results', 'adata', 'vertical_integration')
    output_dir = os.path.join(root_dir, '_myx_Results', 'evaluation', 'vertical')
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    total_start = time.time()

    for method in methods:
        method_dir = os.path.join(input_dir, method)
        if not os.path.isdir(method_dir):
            print(f"[SKIP] {method}: directory not found")
            continue

        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print(f"{'='*60}")

        for dataset in datasets:
            # Check compatibility
            if method in METHOD_DATASET_SKIP and dataset in METHOD_DATASET_SKIP[method]:
                print(f"  [{dataset}] SKIP (unsupported modality)")
                continue

            # Try aliases for directory name (vertical results may use short names)
            dataset_dir = None
            for alias in DATASET_DIR_ALIASES.get(dataset, [dataset]):
                candidate = os.path.join(method_dir, alias)
                if os.path.isdir(candidate):
                    dataset_dir = candidate
                    break
            if dataset_dir is None:
                print(f"  [{dataset}] SKIP (no results)")
                continue

            # Find all h5ad files (slices)
            h5ad_files = sorted(Path(dataset_dir).rglob('*.h5ad'))
            if not h5ad_files:
                print(f"  [{dataset}] SKIP (no h5ad files)")
                continue

            print(f"  [{dataset}] {len(h5ad_files)} slices")

            # Determine the actual directory name used
            actual_dir_name = os.path.basename(dataset_dir)

            for h5ad_path in h5ad_files:
                # Determine slice name from directory or filename
                if h5ad_path.parent.name != actual_dir_name:
                    slice_name = h5ad_path.parent.name
                else:
                    slice_name = h5ad_path.stem.split('_')[-1]

                try:
                    adata = sc.read_h5ad(str(h5ad_path))
                except Exception as e:
                    print(f"    [{slice_name}] ERROR loading: {e}")
                    continue

                # Get embeddings
                embeddings = None
                for key in [method, 'X_integrated', 'X_emb', 'embeddings']:
                    if key in adata.obsm:
                        embeddings = adata.obsm[key]
                        break
                if embeddings is None:
                    print(f"    [{slice_name}] SKIP (no embeddings)")
                    continue

                # Spatial coordinates
                if 'spatial' not in adata.obsm:
                    print(f"    [{slice_name}] SKIP (no spatial coords)")
                    continue
                spatial_coords = adata.obsm['spatial']

                # Adjacency matrix
                adj_matrix = knn_adj(embeddings)

                # Ground truth
                y_GT = load_ground_truth(root_dir, dataset, slice_name)
                if y_GT is not None:
                    y_GT = np.asarray(y_GT, dtype=int)
                    # Align lengths
                    min_len = min(len(y_GT), embeddings.shape[0])
                    if len(y_GT) != embeddings.shape[0]:
                        y_GT = y_GT[:min_len]
                        embeddings = embeddings[:min_len]
                        spatial_coords = spatial_coords[:min_len]
                        adj_matrix = adj_matrix[:min_len, :min_len]

                # Per-clustering evaluation
                for clust in clustering_methods:
                    if clust not in adata.obs.columns:
                        continue

                    y_pred_raw = adata.obs[clust]
                    if hasattr(y_pred_raw, 'cat'):
                        y_pred = y_pred_raw.cat.codes.values
                    else:
                        y_pred = y_pred_raw.values
                    y_pred = np.asarray(y_pred, dtype=int)

                    if y_GT is not None:
                        y_pred = y_pred[:len(y_GT)]

                    metrics = eval_func(
                        embeddings=embeddings,
                        adj_matrix=adj_matrix,
                        y_pred=y_pred,
                        y_GT=y_GT,
                        spatial_coords=spatial_coords,
                        method_name=method,
                        dataset_name=dataset,
                        slice_name=slice_name,
                        clustering_method=clust,
                    )

                    # Save per-file CSV
                    method_out = os.path.join(output_dir, method, dataset)
                    save_func(
                        metrics_dict=metrics,
                        output_dir=method_out,
                        method_name=method,
                        dataset_name=dataset,
                        slice_name=slice_name,
                        clustering_method=clust,
                        has_gt=(y_GT is not None),
                    )

                    all_results.append(metrics)

                print(f"    [{slice_name}] OK ({len(clustering_methods)} clusterings)")

            if test_mode:
                break  # 1 dataset in test mode
        if test_mode:
            break  # 1 method in test mode

    elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Vertical evaluation complete: {len(all_results)} results in {elapsed/60:.1f} min")
    print(f"Per-file CSVs saved to: {output_dir}")
    return all_results, output_dir


# ---------------------------------------------------------------------------
# Aggregation: generate summary tables
# ---------------------------------------------------------------------------

def _process_results_by_directory(output_dir):
    """Parse per-file CSVs using directory structure (not filename) for robust dataset names.

    Directory layout: output_dir / {Method} / {Dataset} / {filename}.csv
    This avoids Eval's filename parser which doesn't handle multi-word dataset names.
    """
    import glob

    SC_METRICS = ['Moran Index', 'Geary C']
    BIOC_METRICS_WITHGT = ['ARI', 'NMI', 'asw_celltype', 'graph_clisi']
    BIOC_METRICS_WOGT = ['Davies-Bouldin Index', 'Silhouette Coefficient', 'Calinski-Harabaz Index']

    results_by_clustering = {c: [] for c in CLUSTERING_METHODS}

    for method_dir in sorted(Path(output_dir).iterdir()):
        if not method_dir.is_dir():
            continue
        method_name = method_dir.name

        for dataset_dir in sorted(method_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset_name = dataset_dir.name

            for csv_path in sorted(dataset_dir.glob('*.csv')):
                try:
                    fname = csv_path.stem  # e.g. CANDIES_Human_Lymph_Nodes_A1_leiden_withGT
                    # Parse GT status and clustering from the end of filename
                    parts = fname.split('_')
                    gt_status = parts[-1]   # withGT or woGT
                    clustering = parts[-2]  # leiden, louvain, kmeans, mclust

                    if clustering not in results_by_clustering:
                        continue

                    # Slice name: strip method, dataset, clustering, gt from filename
                    # The slice is everything between dataset_name and clustering
                    # Find where dataset ends in the filename
                    prefix = f"{method_name}_{dataset_name}_"
                    suffix = f"_{clustering}_{gt_status}"
                    if fname.startswith(prefix) and fname.endswith(suffix):
                        slice_name = fname[len(prefix):-len(suffix)]
                    else:
                        slice_name = 'unknown'

                    df = pd.read_csv(csv_path)
                    metrics_dict = dict(zip(df['Metric'], df['Value']))
                    metrics_dict.update({
                        'Method': method_name,
                        'Dataset': dataset_name,
                        'Slice': slice_name,
                        'Clustering': clustering,
                        'GT_Available': (gt_status == 'withGT'),
                        'Dataset_Type': DATASET_TYPES.get(dataset_name, 'Unknown'),
                    })
                    results_by_clustering[clustering].append(metrics_dict)
                except Exception as e:
                    print(f"  Warning: error parsing {csv_path}: {e}")
                    continue

    return results_by_clustering


def aggregate_results(output_dir, root_dir):
    """Read per-file CSVs and generate summary tables."""
    # Import scoring helpers from Eval (these don't depend on filename parsing)
    eval_dir = os.path.join(root_dir, 'Eval')
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    from generate_final_results import (
        aggregate_by_dataset,
        calculate_normalized_scores,
        create_summary_tables,
    )

    summary_dir = os.path.join(root_dir, '_myx_Results', 'evaluation', 'summary')
    os.makedirs(summary_dir, exist_ok=True)

    # Use our own directory-based parser instead of Eval's filename parser
    results_by_clustering = _process_results_by_directory(output_dir)

    for clust_method in CLUSTERING_METHODS:
        if clust_method not in results_by_clustering or not results_by_clustering[clust_method]:
            continue

        print(f"\nAggregating {clust_method} results...")
        agg_df = aggregate_by_dataset(results_by_clustering[clust_method])
        if agg_df.empty:
            continue

        # Fix: Eval's aggregate_by_dataset overwrites Dataset_Type with its own
        # short-name dict (HLN/HT/MISAR_S1), so full names become 'Unknown'.
        # Re-apply our Dataset_Type mapping here.
        agg_df['Dataset_Type'] = agg_df['Dataset'].map(DATASET_TYPES).fillna('Unknown')

        scored_df = calculate_normalized_scores(agg_df)
        rna_adt_tbl, rna_atac_tbl, comp_tbl = create_summary_tables(scored_df, clust_method)

        # Save
        scored_df.to_csv(os.path.join(summary_dir, f'vertical_detailed_{clust_method}.csv'), index=False)
        if not comp_tbl.empty:
            comp_tbl.to_csv(os.path.join(summary_dir, f'vertical_summary_{clust_method}.csv'))
        if not rna_adt_tbl.empty:
            rna_adt_tbl.to_csv(os.path.join(summary_dir, f'vertical_RNA_ADT_{clust_method}.csv'))
        if not rna_atac_tbl.empty:
            rna_atac_tbl.to_csv(os.path.join(summary_dir, f'vertical_RNA_ATAC_{clust_method}.csv'))

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
