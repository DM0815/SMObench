"""Vertical integration pipeline: cross-modality, same batch."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from anndata import AnnData

from smobench.pipeline.benchmark import BenchmarkResult


def run_vertical(
    dataset: str,
    slice_name: str,
    method_name: str,
    clustering: list[str],
    device: str = "cuda:0",
    seed: int = 42,
    data_root: str | None = None,
    save_integrated: str | None = None,
) -> list[dict]:
    """Run vertical integration for one method on one slice.

    Returns list of records (one per clustering method).
    """
    from smobench.data import load_dataset, DATASET_REGISTRY
    from smobench.methods import get_method
    from smobench.clustering import cluster
    from smobench.metrics.evaluate import evaluate
    from smobench._constants import get_n_clusters

    ds_info = DATASET_REGISTRY[dataset]
    label_key = "Spatial_Label" if ds_info["gt"] else None
    n_clusters = get_n_clusters(dataset, slice_name)
    modality = "ADT" if "ADT" in ds_info["modality"] else "ATAC"

    # Load data
    adata_rna, adata_mod2 = load_dataset(dataset, slice_name, data_root)

    # Integrate (with automatic environment isolation)
    from smobench.pipeline._isolation import subprocess_integrate

    t0 = time.time()
    embedding, kept_indices = subprocess_integrate(
        method_name, adata_rna, adata_mod2,
        device=device, seed=seed, modality=modality,
    )
    runtime = time.time() - t0
    print(f"  [{method_name}] Integration done in {runtime:.1f}s "
          f"(embedding shape: {embedding.shape})")

    # If method filtered cells, subset adata to kept cells only
    if embedding.shape[0] != adata_rna.n_obs:
        if kept_indices is not None:
            print(f"  [{method_name}] Method kept {len(kept_indices)}/{adata_rna.n_obs} cells. "
                  f"Subsetting adata for metric computation.")
            adata_rna = adata_rna[kept_indices].copy()
        else:
            print(f"  [{method_name}] WARNING: embedding has {embedding.shape[0]} cells "
                  f"but adata has {adata_rna.n_obs}. No kept_indices provided — "
                  f"using first {embedding.shape[0]} cells.")
            adata_rna = adata_rna[:embedding.shape[0]].copy()
    adata_rna.obsm[method_name] = embedding

    # Save to integrated adata if requested
    if save_integrated:
        from smobench.io import save_embedding
        save_embedding(adata_rna, method_name, embedding, save_integrated, train_time=runtime)

    # Cluster and evaluate
    records = []
    for clust_method in clustering:
        clust_key = f"{method_name}_{clust_method}"
        try:
            cluster(adata_rna, method=clust_method, n_clusters=n_clusters,
                    embedding_key=method_name, key_added=clust_key)

            scores = evaluate(
                adata_rna, embedding_key=method_name, cluster_key=clust_key,
                label_key=label_key, has_gt=ds_info["gt"],
            )

            records.append({
                "Task": "vertical",
                "Dataset": dataset,
                "Slice": slice_name,
                "Method": method_name,
                "Clustering": clust_method,
                "Modality": ds_info["modality"],
                "GT": ds_info["gt"],
                "Runtime": runtime,
                **scores,
            })
        except Exception as e:
            print(f"    {clust_method} failed: {e}")

    return records
