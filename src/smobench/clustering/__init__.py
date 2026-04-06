"""
Unified clustering module.

Supports: leiden, louvain, kmeans, mclust.
Default resolution=1.0 for graph-based methods, no search.
"""

from __future__ import annotations

import numpy as np
import scanpy as sc
from anndata import AnnData

from smobench._constants import (
    CLUSTERING_METHODS as SUPPORTED_METHODS,
    DEFAULT_CLUSTERING,
    DEFAULT_RESOLUTION,
    N_NEIGHBORS,
    RANDOM_SEED,
)


def cluster(
    adata: AnnData,
    method: str = "leiden",
    n_clusters: int = 10,
    embedding_key: str = "X_emb",
    key_added: str | None = None,
    resolution: float = DEFAULT_RESOLUTION,
    n_neighbors: int = N_NEIGHBORS,
    random_state: int = RANDOM_SEED,
) -> AnnData:
    """Cluster cells on embedding.

    Parameters
    ----------
    resolution : float
        Resolution for leiden/louvain. Default 1.0.
    n_clusters : int
        Target clusters for kmeans/mclust. Ignored by leiden/louvain.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown '{method}'. Choose from: {SUPPORTED_METHODS}")

    if key_added is None:
        key_added = method

    if method in ("leiden", "louvain"):
        sc.pp.neighbors(adata, use_rep=embedding_key, n_neighbors=n_neighbors,
                        random_state=random_state)
        if method == "leiden":
            sc.tl.leiden(adata, resolution=resolution, random_state=random_state, key_added=key_added)
        else:
            _louvain_igraph(adata, resolution=resolution, random_state=random_state, key_added=key_added)

    elif method == "kmeans":
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        adata.obs[key_added] = km.fit_predict(adata.obsm[embedding_key]).astype(str)

    elif method == "mclust":
        adata = _mclust_R(adata, n_clusters, embedding_key, random_state)
        adata.obs[key_added] = adata.obs["mclust"].astype(str)

    adata.obs[key_added] = adata.obs[key_added].astype("category")
    return adata


def cluster_all(
    adata: AnnData,
    n_clusters: int,
    embedding_key: str,
    methods: list[str] | None = None,
    **kwargs,
) -> AnnData:
    """Run multiple clustering methods. Results: obs['{embedding_key}_{method}']."""
    if methods is None:
        methods = DEFAULT_CLUSTERING

    for method in methods:
        key = f"{embedding_key}_{method}"
        adata = cluster(adata, method=method, n_clusters=n_clusters,
                        embedding_key=embedding_key, key_added=key, **kwargs)

    return adata


def _louvain_igraph(adata, resolution=1.0, random_state=0, key_added="louvain"):
    """Louvain clustering via igraph's community_multilevel."""
    import igraph as ig
    import pandas as pd

    adjacency = adata.obsp["connectivities"]
    sources, targets = adjacency.nonzero()
    weights = np.array(adjacency[sources, targets]).flatten()
    g = ig.Graph(n=adjacency.shape[0], edges=list(zip(sources, targets)),
                 directed=False)
    g.es["weight"] = weights

    part = g.community_multilevel(weights="weight", resolution=resolution)
    labels = pd.Categorical([str(x) for x in part.membership])
    adata.obs[key_added] = labels


def _mclust_R(adata, n_clusters, embedding_key, random_seed):
    """Run mclust via R subprocess."""
    import os
    import tempfile
    import subprocess
    import pandas as pd
    from sklearn.decomposition import PCA

    emb = np.ascontiguousarray(adata.obsm[embedding_key], dtype=np.float64)

    if np.any(~np.isfinite(emb)):
        raise ValueError("Embedding contains NaN/Inf.")

    max_dims = min(emb.shape[1], 20, emb.shape[0] - 1)
    if max_dims < emb.shape[1]:
        emb = PCA(n_components=max_dims, random_state=random_seed).fit_transform(emb)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "emb.csv")
        result_path = os.path.join(tmpdir, "labels.csv")
        script_path = os.path.join(tmpdir, "mclust.R")

        np.savetxt(data_path, emb, delimiter=",")

        with open(script_path, 'w') as f:
            f.write(f"""
library(mclust)
set.seed({random_seed})
data <- as.matrix(read.csv("{data_path}", header=FALSE))
res <- Mclust(data, G={n_clusters}, verbose=FALSE)
if (is.null(res)) stop("Mclust returned NULL")
write.csv(res$classification, "{result_path}", row.names=FALSE)
""")

        clean_env = {k: v for k, v in os.environ.items()
                     if not k.startswith('CUDA') and k != 'PYTORCH_CUDA_ALLOC_CONF'}

        proc = subprocess.run(
            ["Rscript", script_path],
            capture_output=True, text=True, timeout=600, env=clean_env,
        )

        if proc.returncode != 0 or not os.path.exists(result_path):
            raise RuntimeError(f"mclust failed (rc={proc.returncode}): {proc.stderr.strip()[:200]}")

        labels = pd.read_csv(result_path).iloc[:, 0].values
        adata.obs['mclust'] = pd.Categorical(labels.astype(int))

    return adata
