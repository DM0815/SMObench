"""Biological/Cluster Quality metrics (woGT): Silhouette, DBI, CHI."""

from __future__ import annotations

import numpy as np
from anndata import AnnData


def silhouette(adata: AnnData, embedding_key: str, cluster_key: str) -> float:
    """Silhouette Coefficient. Range [-1, 1], higher is better."""
    from sklearn.metrics import silhouette_score

    labels = adata.obs[cluster_key].values
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(silhouette_score(adata.obsm[embedding_key], labels))


def davies_bouldin(adata: AnnData, embedding_key: str, cluster_key: str) -> float:
    """Davies-Bouldin Index. Range [0, +inf), LOWER is better."""
    from sklearn.metrics import davies_bouldin_score

    labels = adata.obs[cluster_key].values
    if len(np.unique(labels)) < 2:
        return float("inf")
    return float(davies_bouldin_score(adata.obsm[embedding_key], labels))


def calinski_harabasz(adata: AnnData, embedding_key: str, cluster_key: str) -> float:
    """Calinski-Harabasz Index. Range [0, +inf), HIGHER is better."""
    from sklearn.metrics import calinski_harabasz_score

    labels = adata.obs[cluster_key].values
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(calinski_harabasz_score(adata.obsm[embedding_key], labels))
