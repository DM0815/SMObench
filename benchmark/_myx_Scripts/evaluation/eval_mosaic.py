#!/usr/bin/env python3
"""
SMOBench Mosaic Integration Evaluation
Computes SC + BioC + BER for SpaMosaic mosaic results (7 datasets × 2 scenarios).

Usage:
    python eval_mosaic.py --root /path/to/SMOBench-CLEAN
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

DATASETS = {
    'HLN':          {'full': 'Human_Lymph_Nodes', 'type': 'RNA_ADT',  'gt': True},
    'HT':           {'full': 'Human_Tonsils',     'type': 'RNA_ADT',  'gt': True},
    'Mouse_Spleen': {'full': 'Mouse_Spleen',      'type': 'RNA_ADT',  'gt': False},
    'Mouse_Thymus': {'full': 'Mouse_Thymus',       'type': 'RNA_ADT',  'gt': False},
    'MISAR_S1':     {'full': 'Mouse_Embryos_S1',  'type': 'RNA_ATAC', 'gt': True},
    'MISAR_S2':     {'full': 'Mouse_Embryos_S2',  'type': 'RNA_ATAC', 'gt': True},
    'Mouse_Brain':  {'full': 'Mouse_Brain',        'type': 'RNA_ATAC', 'gt': False},
}

SCENARIOS = ['without_rna', 'without_second']
CLUSTERING_METHODS = ['leiden', 'louvain', 'kmeans']


def setup_eval_imports(root_dir):
    eval_dir = os.path.join(root_dir, 'Eval')
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    from src.demo import eval_horizontal_integration, save_evaluation_results
    try:
        from src.clustering import knn_adj_matrix
    except ImportError:
        from src.clustering_simple import knn_adj_matrix
    return eval_horizontal_integration, save_evaluation_results, knn_adj_matrix


def load_ground_truth(root_dir, ds_key):
    """Load concatenated GT labels from all slices."""
    info = DATASETS[ds_key]
    if not info['gt']:
        return None
    gt_type = 'withGT' if info['gt'] else 'woGT'
    dataset_path = os.path.join(root_dir, 'Dataset', gt_type, info['type'], info['full'])
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
            print(f"    [GT] Failed: {e}")
    return np.array(all_labels) if all_labels else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--clustering', type=str, default='leiden')
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    eval_func, save_func, knn_adj = setup_eval_imports(root)

    mosaic_dir = os.path.join(root, '_myx_Results', 'adata', 'mosaic_integration', 'SpaMosaic')
    output_dir = os.path.join(root, '_myx_Results', 'evaluation', 'mosaic')
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    for ds_key, ds_info in DATASETS.items():
        y_GT = load_ground_truth(root, ds_key)

        for scenario in SCENARIOS:
            h5ad_path = os.path.join(mosaic_dir, ds_key, scenario,
                                      f'SpaMosaic_{ds_key}_{scenario}.h5ad')
            if not os.path.exists(h5ad_path):
                print(f"[SKIP] {ds_key}/{scenario}: file not found")
                continue

            print(f"\n--- {ds_key} / {scenario} ---")
            adata = sc.read_h5ad(h5ad_path)

            # Embeddings
            embeddings = adata.obsm.get('SpaMosaic')
            if embeddings is None:
                print(f"  SKIP: no SpaMosaic embeddings")
                continue

            # Spatial coords
            spatial_coords = adata.obsm.get('spatial')
            if spatial_coords is None:
                print(f"  WARN: no spatial, using embedding[:, :2]")
                spatial_coords = embeddings[:, :2]

            # Batch labels
            batch_labels = adata.obs.get('batch', adata.obs.get('src'))
            if batch_labels is not None:
                batch_labels = np.array(batch_labels.astype(str))
            else:
                batch_labels = np.array(['batch0'] * adata.n_obs)

            # Align GT length
            gt = None
            if y_GT is not None:
                if len(y_GT) == adata.n_obs:
                    gt = np.asarray(y_GT, dtype=int)
                else:
                    print(f"  WARN: GT length mismatch ({len(y_GT)} vs {adata.n_obs}), skip GT")

            adj_matrix = knn_adj(embeddings)

            for clust in CLUSTERING_METHODS:
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
                y_pred = np.asarray(y_pred, dtype=int)

                metrics = eval_func(
                    embeddings=embeddings,
                    adj_matrix=adj_matrix,
                    y_pred=y_pred,
                    y_GT=gt,
                    spatial_coords=spatial_coords,
                    batch_labels=batch_labels,
                    method_name='SpaMosaic',
                    dataset_name=ds_info['full'],
                    slice_name=scenario,
                    clustering_method=clust,
                )

                # Add mosaic-specific info
                metrics['Scenario'] = scenario
                metrics['Dataset_Short'] = ds_key

                save_func(
                    metrics_dict=metrics,
                    output_dir=os.path.join(output_dir, ds_key),
                    method_name='SpaMosaic',
                    dataset_name=ds_info['full'],
                    slice_name=scenario,
                    clustering_method=clust,
                    has_gt=(gt is not None),
                )

                all_results.append(metrics)
                n_clust = len(np.unique(y_pred))
                gt_str = 'withGT' if gt is not None else 'woGT'
                print(f"  {clust}: {n_clust} clusters, {gt_str}")

    # Save combined CSV
    if all_results:
        df = pd.DataFrame(all_results)
        out_csv = os.path.join(output_dir, 'mosaic_evaluation_results.csv')
        df.to_csv(out_csv, index=False)
        print(f"\nSaved {len(df)} rows → {out_csv}")

        # Summary by scenario × dataset (leiden only)
        if args.clustering in df.columns or 'clustering_method' in df.columns:
            clust_col = 'clustering_method' if 'clustering_method' in df.columns else 'Clustering'
            df_leiden = df[df[clust_col] == 'leiden'] if clust_col in df.columns else df
        else:
            df_leiden = df

        print("\n=== Summary (leiden) ===")
        for scenario in SCENARIOS:
            df_sc = df_leiden[df_leiden['Scenario'] == scenario]
            if df_sc.empty:
                continue
            print(f"\n{scenario}:")
            score_cols = [c for c in df_sc.columns if 'Score' in c or c in ['Moran_Index', 'ARI', 'NMI']]
            if score_cols:
                print(df_sc[['Dataset_Short'] + score_cols].to_string(index=False))


if __name__ == '__main__':
    main()
