"""Spatial Coherence metrics: Moran's I and Geary's C."""

from __future__ import annotations

import numpy as np
from anndata import AnnData


def morans_i(adata: AnnData, embedding_key: str, n_neighbors: int = 20) -> float:
    """Compute Moran's I for spatial autocorrelation of embedding.

    Higher values indicate stronger spatial coherence.

    Returns
    -------
    float
        Moran's I statistic, range roughly [-1, 1].
    """
    import scanpy as sc
    from scipy.sparse import issparse

    adata_tmp = adata.copy()

    # Build spatial neighbor graph
    if "spatial" not in adata_tmp.obsm:
        raise ValueError("adata.obsm['spatial'] required for spatial coherence metrics")

    sc.pp.neighbors(adata_tmp, use_rep=embedding_key, n_neighbors=n_neighbors)
    W = adata_tmp.obsp["connectivities"]

    embeddings = adata_tmp.obsm[embedding_key]
    if issparse(embeddings):
        embeddings = embeddings.toarray()

    n = embeddings.shape[0]
    # Use first PC of embedding for Moran's I
    from sklearn.decomposition import PCA
    if embeddings.shape[1] > 1:
        x = PCA(n_components=1, random_state=42).fit_transform(embeddings).ravel()
    else:
        x = embeddings.ravel()

    x_bar = x.mean()
    z = x - x_bar

    if issparse(W):
        numerator = float(W.multiply(np.outer(z, z)).sum())
    else:
        numerator = float((W * np.outer(z, z)).sum())

    denominator = float((z ** 2).sum())
    W_sum = float(W.sum())

    if denominator == 0 or W_sum == 0:
        return 0.0

    I = (n / W_sum) * (numerator / denominator)
    return float(I)


def gearys_c(adata: AnnData, embedding_key: str, n_neighbors: int = 20) -> float:
    """Compute Geary's C for spatial autocorrelation.

    Lower values indicate stronger spatial coherence (inverse of Moran's I).

    Returns
    -------
    float
        Geary's C statistic, range [0, +inf). 0 = perfect, 1 = no autocorrelation.
    """
    import scanpy as sc
    from scipy.sparse import issparse

    adata_tmp = adata.copy()
    sc.pp.neighbors(adata_tmp, use_rep=embedding_key, n_neighbors=n_neighbors)
    W = adata_tmp.obsp["connectivities"]

    embeddings = adata_tmp.obsm[embedding_key]
    if issparse(embeddings):
        embeddings = embeddings.toarray()

    n = embeddings.shape[0]
    from sklearn.decomposition import PCA
    if embeddings.shape[1] > 1:
        x = PCA(n_components=1, random_state=42).fit_transform(embeddings).ravel()
    else:
        x = embeddings.ravel()

    x_bar = x.mean()
    denominator = float(((x - x_bar) ** 2).sum())
    W_sum = float(W.sum())

    if denominator == 0 or W_sum == 0:
        return 1.0

    # Geary's C = sum_ij w_ij (x_i - x_j)^2 / (2 * W * sum (x - xbar)^2) * (n-1)
    if issparse(W):
        rows, cols = W.nonzero()
        diff_sq = (x[rows] - x[cols]) ** 2
        data = np.array(W[rows, cols]).ravel()
        numerator = float((data * diff_sq).sum())
    else:
        diff_matrix = np.subtract.outer(x, x) ** 2
        numerator = float((W * diff_matrix).sum())

    C = ((n - 1) / (2 * W_sum)) * (numerator / denominator)
    return float(C)
