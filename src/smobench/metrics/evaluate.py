"""Master evaluation functions."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from anndata import AnnData


def evaluate(
    adata: AnnData,
    embedding_key: str,
    cluster_key: str,
    label_key: Optional[str] = None,
    batch_key: Optional[str] = None,
    has_gt: Optional[bool] = None,
    n_neighbors: Optional[int] = None,
) -> dict:
    """Compute all applicable metrics for an integration result.

    Parameters
    ----------
    adata : AnnData
        Integrated data with embedding in obsm and clustering in obs.
    embedding_key : str
        Key in adata.obsm for the integration embedding.
    cluster_key : str
        Key in adata.obs for clustering labels.
    label_key : str, optional
        Key in adata.obs for ground truth labels. If provided, BioC metrics are computed.
    batch_key : str, optional
        Key in adata.obs for batch labels. If provided, BER metrics are computed.
    has_gt : bool, optional
        Whether dataset has ground truth. Auto-detected from label_key if not set.
    n_neighbors : int
        Number of neighbors for graph-based metrics.

    Returns
    -------
    dict
        Metric name → value.
    """
    from smobench.metrics import spatial_coherence, bio_conservation, bio_quality, batch_effect
    from smobench._constants import N_NEIGHBORS, GT_LABEL_KEY, BATCH_LABEL_KEY

    if n_neighbors is None:
        n_neighbors = N_NEIGHBORS

    if has_gt is None:
        has_gt = label_key is not None and label_key in adata.obs.columns

    results = {}

    # SC metrics (always computed)
    results["Moran_I"] = spatial_coherence.morans_i(adata, embedding_key, n_neighbors)

    # BioC metrics (withGT)
    if has_gt and label_key:
        gt_labels = adata.obs[label_key].values
        pred_labels = adata.obs[cluster_key].values
        results["ARI"] = bio_conservation.ari(gt_labels, pred_labels)
        results["NMI"] = bio_conservation.nmi(gt_labels, pred_labels)
        results["cASW"] = bio_conservation.asw_celltype(adata, embedding_key, label_key)
        results["cLISI"] = bio_conservation.graph_clisi(adata, embedding_key, label_key, n_neighbors)

    # BVC metrics (woGT or always as supplement)
    if not has_gt:
        results["Silhouette"] = bio_quality.silhouette(adata, embedding_key, cluster_key)
        results["DBI"] = bio_quality.davies_bouldin(adata, embedding_key, cluster_key)
        results["CHI"] = bio_quality.calinski_harabasz(adata, embedding_key, cluster_key)

    # BER metrics (only when batch_key exists)
    if batch_key and batch_key in adata.obs.columns:
        n_batches = len(adata.obs[batch_key].unique())
        if n_batches >= 2:
            results["kBET"] = batch_effect.kbet(adata, embedding_key, batch_key, n_neighbors)
            results["bASW"] = batch_effect.asw_batch(adata, embedding_key, batch_key)
            results["iLISI"] = batch_effect.graph_ilisi(adata, embedding_key, batch_key, n_neighbors)
            results["KNN_conn"] = batch_effect.knn_connectivity(adata, embedding_key, batch_key, n_neighbors)
            results["PCR"] = batch_effect.pcr(adata, embedding_key, batch_key)

    # Composite scores
    results["SC_Score"] = results.get("Moran_I", 0.0)

    if has_gt:
        bioc_keys = ["ARI", "NMI", "cASW", "cLISI"]
        bioc_vals = [results[k] for k in bioc_keys if k in results]
        results["BioC_Score"] = float(np.mean(bioc_vals)) if bioc_vals else 0.0
    else:
        # Normalize DBI (invert) and CHI for woGT
        results["BVC_Score"] = results.get("Silhouette", 0.0)  # simplified

    if batch_key and batch_key in adata.obs.columns:
        ber_keys = ["kBET", "bASW", "iLISI", "KNN_conn", "PCR"]
        ber_vals = [results[k] for k in ber_keys if k in results]
        results["BER_Score"] = float(np.mean(ber_vals)) if ber_vals else 0.0

    return results


def fast(adata: AnnData, embedding_key: str, cluster_key: str, **kwargs) -> dict:
    """Fast evaluation: SC + Silhouette only."""
    from smobench.metrics import spatial_coherence, bio_quality

    return {
        "Moran_I": spatial_coherence.morans_i(adata, embedding_key),
        "Silhouette": bio_quality.silhouette(adata, embedding_key, cluster_key),
    }


def standard(adata: AnnData, embedding_key: str, cluster_key: str, **kwargs) -> dict:
    """Standard evaluation: SC + BioC/BVC."""
    return evaluate(adata, embedding_key, cluster_key, **kwargs)


def all_metrics(adata: AnnData, embedding_key: str, cluster_key: str, **kwargs) -> dict:
    """Full evaluation: all metrics including BER."""
    return evaluate(adata, embedding_key, cluster_key, **kwargs)
