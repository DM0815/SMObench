"""Backfill metrics for methods that have embeddings but no metrics in h5ad.

Reads each adata_integrated.h5ad, finds methods with embeddings but missing
metrics, runs evaluate, and saves metrics back to adata.uns.
"""
import sys
sys.path.insert(0, "../../src")

import os
import glob
import numpy as np
import anndata as ad

from smobench.plot.summary import _discover_methods, _auto_n_clusters, _auto_label_key, _auto_gt
from smobench.clustering import cluster
from smobench.metrics.evaluate import evaluate

RESULTS_ROOT = "../results/vertical"


def backfill(h5ad_path, dataset_name, slice_name):
    """Backfill metrics for one h5ad file."""
    print(f"\n{'─'*60}")
    print(f"  {dataset_name}/{slice_name}: {h5ad_path}")
    print(f"{'─'*60}")

    adata = ad.read_h5ad(h5ad_path)
    methods = _discover_methods(adata)
    if not methods:
        print("  No embeddings found, skipping")
        return

    n_clusters = _auto_n_clusters(dataset_name, slice_name)
    label_key = _auto_label_key(adata)
    has_gt = _auto_gt(dataset_name)
    if has_gt is None:
        has_gt = label_key is not None

    print(f"  Methods: {methods}")
    print(f"  n_clusters={n_clusters}, label_key={label_key}, has_gt={has_gt}")

    CLUSTERING = ["leiden", "louvain", "kmeans", "mclust"]

    updated = False
    for method in methods:
        metrics_key = f"{method}_metrics"

        # Check if metrics already exist (nested format: {clustering: {metric: value}})
        existing = adata.uns.get(metrics_key, {})
        if isinstance(existing, dict) and any(isinstance(v, dict) for v in existing.values()):
            print(f"  {method}: metrics already exist, skipping")
            continue

        # Run all clustering methods and evaluate
        metrics_dict = {}
        for clust in CLUSTERING:
            clust_key = f"{method}_{clust}"
            if clust_key not in adata.obs.columns:
                try:
                    cluster(adata, method=clust, n_clusters=n_clusters,
                            embedding_key=method, key_added=clust_key)
                except Exception as e:
                    print(f"  {method}/{clust}: cluster error — {e}")
                    continue

            try:
                scores = evaluate(
                    adata, embedding_key=method, cluster_key=clust_key,
                    label_key=label_key, has_gt=has_gt,
                )
                metrics_dict[clust] = scores
            except Exception as e:
                print(f"  {method}/{clust}: eval error — {e}")
                continue

        if not metrics_dict:
            print(f"  {method}: all clustering failed, skipping")
            continue

        # Save nested dict to uns (matches run_all_methods format)
        adata.uns[metrics_key] = metrics_dict
        first_scores = next(iter(metrics_dict.values()))
        print(f"  {method}: OK ({len(metrics_dict)} clusterings) — {', '.join(f'{k}={v:.4f}' for k, v in first_scores.items() if isinstance(v, float))}")
        updated = True

    if updated:
        adata.write_h5ad(h5ad_path)
        print(f"  Saved: {h5ad_path}")
    else:
        print("  Nothing to update")


def main():
    pattern = os.path.join(RESULTS_ROOT, "*", "*", "adata_integrated.h5ad")
    h5ad_files = sorted(glob.glob(pattern))
    print(f"Found {len(h5ad_files)} h5ad files")

    for path in h5ad_files:
        parts = path.split(os.sep)
        # .../vertical/Dataset/Slice/adata_integrated.h5ad
        slice_name = parts[-2]
        dataset_name = parts[-3]
        try:
            backfill(path, dataset_name, slice_name)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
