"""Batch Effect Removal metrics: kBET, bASW, iLISI, KNN connectivity, PCR."""

from __future__ import annotations

import numpy as np
from anndata import AnnData


def kbet(
    adata: AnnData, embedding_key: str, batch_key: str,
    n_neighbors: int = 20, alpha: float = 0.05,
) -> float:
    """kBET acceptance rate (chi-square test). Range [0, 1], higher = better mixing."""
    from sklearn.neighbors import NearestNeighbors
    from scipy.stats import chi2

    embeddings = adata.obsm[embedding_key]
    batch_labels = adata.obs[batch_key].values
    unique_batches = np.unique(batch_labels)
    n_batches = len(unique_batches)

    if n_batches < 2:
        return 1.0

    batch_map = {b: i for i, b in enumerate(unique_batches)}
    batch_num = np.array([batch_map[b] for b in batch_labels])

    global_props = np.bincount(batch_num, minlength=n_batches) / len(batch_num)

    knn = min(n_neighbors, len(embeddings) - 1)
    nbrs = NearestNeighbors(n_neighbors=knn + 1).fit(embeddings)
    _, indices = nbrs.kneighbors()
    neighbor_indices = indices[:, 1:]

    reject_count = 0
    df = n_batches - 1

    for neighbors in neighbor_indices:
        observed = np.bincount(batch_num[neighbors], minlength=n_batches).astype(float)
        expected = global_props * len(neighbors)
        valid = expected > 0
        if valid.sum() < 2:
            continue
        chi2_stat = np.sum((observed[valid] - expected[valid]) ** 2 / expected[valid])
        p_value = 1.0 - chi2.cdf(chi2_stat, df=df)
        if p_value < alpha:
            reject_count += 1

    return 1.0 - (reject_count / len(neighbor_indices))


def asw_batch(adata: AnnData, embedding_key: str, batch_key: str) -> float:
    """Batch ASW. Rescaled so that higher = better batch mixing."""
    from sklearn.metrics import silhouette_score

    labels = adata.obs[batch_key].values
    if len(np.unique(labels)) < 2:
        return 1.0
    score = silhouette_score(adata.obsm[embedding_key], labels)
    # Invert: low silhouette for batch = good mixing
    return float(1 - (score + 1) / 2)


def graph_ilisi(
    adata: AnnData, embedding_key: str, batch_key: str, n_neighbors: int = 20,
) -> float:
    """Graph-based integration LISI. Rescaled to [0, 1], higher = better mixing."""
    from sklearn.neighbors import NearestNeighbors

    embeddings = adata.obsm[embedding_key]
    batch_labels = adata.obs[batch_key].values
    unique_batches = np.unique(batch_labels)
    n_batches = len(unique_batches)

    if n_batches < 2:
        return 1.0

    batch_map = {b: i for i, b in enumerate(unique_batches)}
    batch_num = np.array([batch_map[b] for b in batch_labels])

    knn = min(n_neighbors, len(embeddings) - 1)
    nbrs = NearestNeighbors(n_neighbors=knn + 1).fit(embeddings)
    _, indices = nbrs.kneighbors()
    neighbor_indices = indices[:, 1:]

    lisi_values = []
    for neighbors in neighbor_indices:
        freqs = np.bincount(batch_num[neighbors], minlength=n_batches) / len(neighbors)
        freqs = freqs[freqs > 0]
        simpson = float(np.sum(freqs ** 2))
        lisi = 1.0 / simpson if simpson > 0 else n_batches
        lisi_values.append(lisi)

    mean_lisi = np.mean(lisi_values)
    # Normalize to [0, 1]: iLISI = (mean_lisi - 1) / (n_batches - 1)
    score = (mean_lisi - 1) / (n_batches - 1) if n_batches > 1 else 1.0
    return float(np.clip(score, 0, 1))


def knn_connectivity(
    adata: AnnData, embedding_key: str, batch_key: str, n_neighbors: int = 20,
) -> float:
    """KNN graph connectivity across batches. Range [0, 1], higher = better."""
    import scanpy as sc
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    adata_tmp = adata.copy()
    sc.pp.neighbors(adata_tmp, use_rep=embedding_key, n_neighbors=n_neighbors)
    G = adata_tmp.obsp["connectivities"]

    labels = adata_tmp.obs[batch_key].values
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return 1.0

    scores = []
    for label in unique_labels:
        mask = labels == label
        subgraph = G[mask][:, mask]
        n_components, _ = connected_components(subgraph, directed=False)
        n_cells = mask.sum()
        scores.append(1.0 - (n_components - 1) / max(n_cells - 1, 1))

    return float(np.mean(scores))


def pcr(adata: AnnData, embedding_key: str, batch_key: str) -> float:
    """Principal Component Regression for batch effect. Range [0, 1], higher = less batch effect."""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import LabelEncoder

    embeddings = adata.obsm[embedding_key]
    batch = LabelEncoder().fit_transform(adata.obs[batch_key].values)

    from sklearn.decomposition import PCA
    n_comps = min(50, embeddings.shape[1], embeddings.shape[0] - 1)
    pcs = PCA(n_components=n_comps).fit_transform(embeddings)

    # R² of batch predicting PCs
    model = LinearRegression()
    model.fit(batch.reshape(-1, 1), pcs)
    r2 = model.score(batch.reshape(-1, 1), pcs)

    # Invert: low R² = batch explains little variance = good
    return float(1 - r2)
