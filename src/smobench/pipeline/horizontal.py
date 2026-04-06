"""Horizontal integration pipeline: same modality, cross-batch."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from anndata import AnnData

from smobench.pipeline.benchmark import BenchmarkResult


def run_horizontal(
    dataset: str,
    method_name: str,
    clustering: list[str],
    device: str = "cuda:0",
    seed: int = 42,
    data_root: str | None = None,
    save_integrated: str | None = None,
) -> list[dict]:
    """Run horizontal integration for one method on one dataset.

    Horizontal integration merges multiple slices (batches) of the same modality.
    Fusion data (pre-merged) is loaded from fusionWithGT/fusionWoGT directories.
    """
    from smobench.data import DATASET_REGISTRY
    from smobench.data.registry import _DEFAULT_ROOT
    from smobench.methods import get_method
    from smobench.clustering import cluster
    from smobench.metrics.evaluate import evaluate

    import scanpy as sc
    from pathlib import Path

    from smobench._constants import get_n_clusters

    ds_info = DATASET_REGISTRY[dataset]
    label_key = "Spatial_Label" if ds_info["gt"] else None
    # For horizontal, use first slice's n_clusters as representative
    n_clusters = get_n_clusters(dataset, ds_info["slices"][0]) if isinstance(ds_info["n_clusters"], dict) else ds_info["n_clusters"]
    modality = ds_info["modality"]
    mod2_name = "ADT" if "ADT" in modality else "ATAC"

    # Load fusion data
    root = Path(data_root or _DEFAULT_ROOT)
    fusion_dir = "fusionWithGT" if ds_info["gt"] else "fusionWoGT"
    fusion_base = root / fusion_dir / modality

    rna_path = fusion_base / f"{dataset}_Fusion_RNA.h5ad"
    mod2_path = fusion_base / f"{dataset}_Fusion_{mod2_name}.h5ad"

    if not rna_path.exists():
        raise FileNotFoundError(f"Fusion RNA not found: {rna_path}")
    if not mod2_path.exists():
        raise FileNotFoundError(f"Fusion {mod2_name} not found: {mod2_path}")

    adata_rna = sc.read_h5ad(str(rna_path))
    adata_mod2 = sc.read_h5ad(str(mod2_path))

    # Verify batch column exists
    if "batch" not in adata_rna.obs.columns:
        raise ValueError(f"No 'batch' column in fusion data. Re-run generate_fusion_data.py.")

    # Integrate (with automatic environment isolation)
    from smobench.pipeline._isolation import subprocess_integrate

    t0 = time.time()
    embedding, kept_indices = subprocess_integrate(
        method_name, adata_rna, adata_mod2,
        device=device, seed=seed, modality=mod2_name,
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

    # Cluster and evaluate
    records = []
    for clust_method in clustering:
        clust_key = f"{method_name}_{clust_method}"
        try:
            cluster(adata_rna, method=clust_method, n_clusters=n_clusters,
                    embedding_key=method_name, key_added=clust_key)

            scores = evaluate(
                adata_rna, embedding_key=method_name, cluster_key=clust_key,
                label_key=label_key, batch_key="batch", has_gt=ds_info["gt"],
            )

            records.append({
                "Task": "horizontal",
                "Dataset": dataset,
                "Method": method_name,
                "Clustering": clust_method,
                "Modality": modality,
                "GT": ds_info["gt"],
                "Runtime": runtime,
                **scores,
            })
        except Exception as e:
            print(f"    {clust_method} failed: {e}")

    return records
