"""Cross-Modal Global Topology Consistency (CM-GTC).

CM-GTC measures how well joint embeddings preserve topological structure
from each input modality — evaluating integration *process quality*
rather than clustering *result quality*.

TODO: Port cm_gtc_v2.py implementation here.
"""

from __future__ import annotations

import numpy as np
from anndata import AnnData


def cmgtc(
    adata: AnnData,
    embedding_key: str,
    rna_matrix: np.ndarray,
    mod2_matrix: np.ndarray,
    n_neighbors: int = 20,
) -> float:
    """Compute CM-GTC score.

    Parameters
    ----------
    adata : AnnData
        Integrated data.
    embedding_key : str
        Key in adata.obsm for joint embedding.
    rna_matrix : np.ndarray
        Preprocessed RNA feature matrix (n_cells × n_features).
    mod2_matrix : np.ndarray
        Preprocessed secondary modality matrix.
    n_neighbors : int
        Number of neighbors for KNN graph.

    Returns
    -------
    float
        CM-GTC score in [0, 1]. Higher = better topology preservation.
    """
    from sklearn.neighbors import NearestNeighbors

    joint_emb = adata.obsm[embedding_key]
    n = joint_emb.shape[0]

    # Build KNN graphs for each modality and joint embedding
    knn_joint = _knn_graph(joint_emb, n_neighbors)
    knn_rna = _knn_graph(rna_matrix, n_neighbors)
    knn_mod2 = _knn_graph(mod2_matrix, n_neighbors)

    # Compute topology consistency: overlap of KNN neighborhoods
    consistency_rna = _neighbor_overlap(knn_joint, knn_rna)
    consistency_mod2 = _neighbor_overlap(knn_joint, knn_mod2)

    # Average across modalities
    score = (consistency_rna + consistency_mod2) / 2
    return float(score)


def _knn_graph(X: np.ndarray, k: int) -> np.ndarray:
    """Build KNN index matrix (n_cells × k)."""
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, indices = nbrs.kneighbors(X)
    return indices[:, 1:]  # exclude self


def _neighbor_overlap(knn_a: np.ndarray, knn_b: np.ndarray) -> float:
    """Compute average neighbor overlap between two KNN graphs."""
    overlaps = []
    for i in range(len(knn_a)):
        set_a = set(knn_a[i])
        set_b = set(knn_b[i])
        overlap = len(set_a & set_b) / len(set_a | set_b)
        overlaps.append(overlap)
    return float(np.mean(overlaps))
