"""Biological Conservation metrics (withGT): ARI, NMI, cASW, cLISI."""

from __future__ import annotations

import numpy as np
from anndata import AnnData


def ari(labels_true, labels_pred) -> float:
    """Adjusted Rand Index. Range [-1, 1], higher is better."""
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(labels_true, labels_pred))


def nmi(labels_true, labels_pred) -> float:
    """Normalized Mutual Information. Range [0, 1], higher is better."""
    from sklearn.metrics import normalized_mutual_info_score
    return float(normalized_mutual_info_score(labels_true, labels_pred))


def asw_celltype(adata: AnnData, embedding_key: str, label_key: str) -> float:
    """Average Silhouette Width for cell types. Rescaled to [0, 1]."""
    from sklearn.metrics import silhouette_score

    embeddings = adata.obsm[embedding_key]
    labels = adata.obs[label_key].values

    if len(np.unique(labels)) < 2:
        return 0.0

    score = silhouette_score(embeddings, labels)
    return float((score + 1) / 2)  # rescale from [-1,1] to [0,1]


def graph_clisi(
    adata: AnnData, embedding_key: str, label_key: str, n_neighbors: int = 20
) -> float:
    """Graph-based cell-type LISI. Rescaled to [0, 1], higher = better separation."""
    import scanpy as sc

    adata_tmp = adata.copy()
    sc.pp.neighbors(adata_tmp, use_rep=embedding_key, n_neighbors=n_neighbors)

    labels = adata_tmp.obs[label_key].values
    n_labels = len(np.unique(labels))

    if n_labels < 2:
        return 0.0

    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(adata_tmp.obsm[embedding_key])
    _, indices = nbrs.kneighbors()

    label_map = {l: i for i, l in enumerate(np.unique(labels))}
    labels_num = np.array([label_map[l] for l in labels])

    lisi_values = []
    for i in range(len(labels)):
        neighbor_labels = labels_num[indices[i]]
        freqs = np.bincount(neighbor_labels, minlength=n_labels) / len(neighbor_labels)
        freqs = freqs[freqs > 0]
        simpson = float(np.sum(freqs ** 2))
        lisi = 1.0 / simpson if simpson > 0 else n_labels
        lisi_values.append(lisi)

    mean_lisi = np.mean(lisi_values)
    # Normalize: cLISI=1 means perfect separation, higher raw LISI = more mixing
    # For cell types, we want LOW mixing, so score = (n_labels - mean_lisi) / (n_labels - 1)
    score = (n_labels - mean_lisi) / (n_labels - 1) if n_labels > 1 else 1.0
    return float(np.clip(score, 0, 1))
