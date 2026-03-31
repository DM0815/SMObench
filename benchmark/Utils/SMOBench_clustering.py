import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

def universal_clustering(adata, n_clusters, used_obsm, method='kmeans', key='clusters', 
                         use_pca=False, n_comps=20, start=0.1, end=3.0, increment=0.01,
                         random_state=2024):
    """
    Universal clustering function supporting multiple clustering methods for SMOBench
    
    Parameters:
        adata: AnnData object containing integration embeddings
        n_clusters: Target number of clusters
        used_obsm: Key for embedding in adata.obsm (e.g., 'merged_emb', 'spatial_emb')
        method: Clustering method ('kmeans', 'mclust', 'leiden', 'louvain')
        key: Output key for storing clusters in adata.obs
        use_pca: Whether to apply PCA preprocessing to embeddings
        n_comps: Number of PCA components (only used if use_pca=True)
        start: Starting resolution for leiden/louvain search
        end: Ending resolution for leiden/louvain search  
        increment: Resolution search step size
        random_state: Random seed for reproducibility
    
    Returns:
        adata: Updated AnnData object with clustering results in adata.obs[key]
    """
    
    # Validate inputs
    if used_obsm not in adata.obsm.keys():
        raise ValueError(f"Embedding key '{used_obsm}' not found in adata.obsm. Available keys: {list(adata.obsm.keys())}")
    
    if method not in ['kmeans', 'mclust', 'leiden', 'louvain']:
        raise ValueError(f"Unsupported method '{method}'. Choose from: kmeans, mclust, leiden, louvain")
    
    # Set random seeds for reproducibility
    np.random.seed(random_state)
    
    # Apply PCA preprocessing if requested
    if use_pca:
        print(f"Applying PCA preprocessing: {adata.obsm[used_obsm].shape[1]} -> {n_comps} dimensions")
        pca = PCA(n_components=n_comps, random_state=random_state)
        adata.obsm[f'{used_obsm}_pca'] = pca.fit_transform(adata.obsm[used_obsm])
        embedding_key = f'{used_obsm}_pca'
    else:
        embedding_key = used_obsm
    
    # Perform clustering based on method
    if method == 'kmeans':
        print(f"Running K-means clustering with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(adata.obsm[embedding_key])
        adata.obs[key] = cluster_labels.astype(str)
        
    elif method == 'mclust':
        print(f"Running mclust clustering with {n_clusters} clusters...", flush=True)
        try:
            adata = _mclust_R(adata, num_cluster=n_clusters, used_obsm=embedding_key, random_seed=random_state)
            print("[DEBUG] _mclust_R returned, setting obs key...", flush=True)
            adata.obs[key] = adata.obs['mclust'].astype(str)
            print("[DEBUG] obs key set successfully", flush=True)
        except Exception as e:
            print(f"mclust failed: {e}", flush=True)
            print("Falling back to K-means clustering...", flush=True)
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(adata.obsm[embedding_key])
            adata.obs[key] = cluster_labels.astype(str)
    
    elif method in ['leiden', 'louvain']:
        import sys
        print(f"[DEBUG] Entering {method} clustering section", flush=True)
        sys.stdout.flush()
        print(f"Running {method} clustering with {n_clusters} clusters...", flush=True)

        # Build neighborhood graph
        print(f"[DEBUG] About to call sc.pp.neighbors with n_neighbors=50...", flush=True)
        sys.stdout.flush()
        sc.pp.neighbors(adata, use_rep=embedding_key, n_neighbors=50, random_state=random_state)
        print(f"[DEBUG] sc.pp.neighbors completed", flush=True)
        sys.stdout.flush()

        # Search for optimal resolution
        print(f"[DEBUG] Starting resolution search...", flush=True)
        sys.stdout.flush()
        resolution = _search_resolution(
            adata,
            n_clusters=n_clusters,
            method=method,
            start=start,
            end=end,
            increment=increment,
            random_state=random_state
        )
        print(f"[DEBUG] Found resolution: {resolution}", flush=True)
        sys.stdout.flush()

        # Apply clustering with found resolution
        if method == 'leiden':
            print(f"[DEBUG] Calling sc.tl.leiden...", flush=True)
            sys.stdout.flush()
            sc.tl.leiden(adata, resolution=resolution, random_state=random_state, key_added=key)
        else:  # louvain
            print(f"[DEBUG] Calling sc.tl.louvain...", flush=True)
            sys.stdout.flush()
            sc.tl.louvain(adata, resolution=resolution, random_state=random_state, key_added=key)
        
        # Convert to string for consistency
        adata.obs[key] = adata.obs[key].astype(str)
    
    import sys
    print(f"[DEBUG] About to print clustering complete...", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    n_found = adata.obs[key].nunique()
    print(f"Clustering complete. Found {n_found} clusters (target: {n_clusters})", flush=True)
    print(f"[DEBUG] universal_clustering returning for method={method}", flush=True)
    sys.stdout.flush()
    return adata


def _mclust_R(adata, num_cluster, used_obsm='emb', random_seed=2024):
    """
    R mclust clustering via subprocess to avoid segfaults crashing the main process.
    Falls back to writing data to a temp file, running R in a subprocess, reading results back.
    """
    import tempfile, subprocess, os

    embedding_data = adata.obsm[used_obsm].copy()
    n_samples, n_dims = embedding_data.shape

    embedding_data = np.ascontiguousarray(embedding_data, dtype=np.float64)

    if np.any(~np.isfinite(embedding_data)):
        print("Warning: embedding contains NaN/Inf values, replacing with 0")
        embedding_data = np.nan_to_num(embedding_data, nan=0.0, posinf=0.0, neginf=0.0)

    # Cap dimensions to avoid mclust internal errors
    max_dims = min(n_dims, 20, n_samples - 1)
    if max_dims < n_dims:
        print(f"Reducing embedding dimensions for mclust: {n_dims} -> {max_dims}")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=max_dims, random_state=random_seed)
        embedding_data = pca.fit_transform(embedding_data)

    print(f"mclust input: {embedding_data.shape[0]} samples x {embedding_data.shape[1]} dims, G={num_cluster}", flush=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "emb.csv")
        result_path = os.path.join(tmpdir, "labels.csv")

        np.savetxt(data_path, embedding_data, delimiter=",")

        r_script = f"""
library(mclust)
set.seed({random_seed})
data <- as.matrix(read.csv("{data_path}", header=FALSE))
res <- tryCatch(
    Mclust(data, G={num_cluster}, verbose=FALSE),
    error=function(e) {{ cat("MCLUST_ERROR:", conditionMessage(e), "\\n"); NULL }}
)
if (is.null(res)) {{
    cat("MCLUST_FAILED\\n")
    q(status=1)
}}
write.csv(res$classification, "{result_path}", row.names=FALSE)
cat("MCLUST_OK\\n")
"""
        r_script_path = os.path.join(tmpdir, "run_mclust.R")
        with open(r_script_path, 'w') as f:
            f.write(r_script)

        # Use clean environment for R subprocess to avoid CUDA/PyTorch fork issues
        clean_env = {k: v for k, v in os.environ.items()
                     if not k.startswith('CUDA') and k != 'PYTORCH_CUDA_ALLOC_CONF'}
        proc = subprocess.run(
            ["Rscript", r_script_path],
            capture_output=True, text=True, timeout=600,
            env=clean_env
        )

        print(f"mclust R stdout: {proc.stdout.strip()}", flush=True)
        if proc.stderr.strip():
            # Filter out mclust banner
            err_lines = [l for l in proc.stderr.strip().split('\n')
                        if 'mclust' not in l.lower() and 'citation' not in l.lower() and l.strip()]
            if err_lines:
                print(f"mclust R stderr: {'; '.join(err_lines[:3])}", flush=True)

        if proc.returncode != 0 or not os.path.exists(result_path):
            raise RuntimeError(f"mclust subprocess failed (rc={proc.returncode})")

        print("[DEBUG] Reading mclust labels from CSV...", flush=True)
        labels_df = pd.read_csv(result_path)
        mclust_labels = labels_df.iloc[:, 0].values
        print(f"[DEBUG] Got {len(mclust_labels)} labels, unique: {np.unique(mclust_labels)}", flush=True)

    print("[DEBUG] Exited tempdir context, assigning to adata.obs...", flush=True)
    adata.obs['mclust'] = pd.Categorical(mclust_labels.astype(int))
    print("[DEBUG] _mclust_R returning successfully", flush=True)
    return adata


def _search_resolution(adata, n_clusters, method='leiden', start=0.1, end=3.0, increment=0.01, random_state=2024):
    """
    Search for optimal resolution to achieve target cluster number
    """
    import sys
    print(f"[DEBUG] _search_resolution called: method={method}, n_clusters={n_clusters}", flush=True)
    sys.stdout.flush()
    print(f"Searching resolution for {method} clustering (target: {n_clusters} clusters)...", flush=True)

    # Search from high to low resolution for more stable results
    resolutions = sorted(np.arange(start, end, increment), reverse=True)
    print(f"[DEBUG] Will try {len(resolutions)} resolutions from {resolutions[0]:.3f} to {resolutions[-1]:.3f}", flush=True)
    sys.stdout.flush()

    for i, resolution in enumerate(resolutions):
        if i == 0:
            print(f"[DEBUG] First iteration: resolution={resolution:.3f}, calling sc.tl.{method}...", flush=True)
            sys.stdout.flush()
        # Apply clustering with current resolution
        if method == 'leiden':
            sc.tl.leiden(adata, resolution=resolution, random_state=random_state, key_added='tmp_search')
            n_found = adata.obs['tmp_search'].nunique()
        else:  # louvain
            sc.tl.louvain(adata, resolution=resolution, random_state=random_state, key_added='tmp_search')
            n_found = adata.obs['tmp_search'].nunique()
        if i == 0:
            print(f"[DEBUG] First sc.tl.{method} call succeeded: {n_found} clusters", flush=True)
            sys.stdout.flush()
        
        print(f"Resolution {resolution:.3f}: {n_found} clusters")
        
        if n_found == n_clusters:
            print(f"Found optimal resolution: {resolution:.3f}")
            return resolution
        
        # Clean up temporary results
        if 'tmp_search' in adata.obs.columns:
            del adata.obs['tmp_search']
    
    # If exact match not found, return resolution closest to target
    print(f"Exact cluster number not found. Using resolution {resolution:.3f} with {n_found} clusters")
    return resolution


def batch_clustering(adata, n_clusters, used_obsm, methods=['leiden', 'louvain', 'kmeans', 'mclust'], 
                     prefix='', **kwargs):
    """
    Apply multiple clustering methods to the same embedding
    
    Parameters:
        adata: AnnData object
        n_clusters: Target cluster number
        used_obsm: Embedding key in adata.obsm  
        methods: List of clustering methods to apply
        prefix: Prefix for output keys (e.g., 'SpatialGlue_')
        **kwargs: Additional arguments passed to universal_clustering
    
    Returns:
        adata: AnnData with clustering results for all methods
    """
    
    for method in methods:
        key = f"{prefix}{method}" if prefix else method
        print(f"\n=== Applying {method} clustering ===")
        
        try:
            adata = universal_clustering(
                adata, 
                n_clusters=n_clusters,
                used_obsm=used_obsm, 
                method=method,
                key=key,
                **kwargs
            )
            print(f"{method} clustering completed -> adata.obs['{key}']")
        except Exception as e:
            print(f"{method} clustering failed: {e}")
    
    return adata


def evaluate_clustering_quality(adata, cluster_key, embedding_key=None):
    """
    Calculate clustering quality metrics
    
    Parameters:
        adata: AnnData object with clustering results
        cluster_key: Key for cluster labels in adata.obs
        embedding_key: Key for embedding in adata.obsm (optional)
    
    Returns:
        dict: Dictionary of quality metrics
    """
    
    if cluster_key not in adata.obs.columns:
        raise ValueError(f"Cluster key '{cluster_key}' not found in adata.obs")
    
    # Basic cluster statistics
    cluster_labels = adata.obs[cluster_key]
    n_clusters = cluster_labels.nunique()
    cluster_sizes = cluster_labels.value_counts().sort_index()
    
    metrics = {
        'n_clusters': n_clusters,
        'min_cluster_size': cluster_sizes.min(),
        'max_cluster_size': cluster_sizes.max(),
        'mean_cluster_size': cluster_sizes.mean(),
        'std_cluster_size': cluster_sizes.std()
    }
    
    # Silhouette score if embedding provided
    if embedding_key and embedding_key in adata.obsm.keys():
        try:
            silhouette = silhouette_score(adata.obsm[embedding_key], cluster_labels.astype(int))
            metrics['silhouette_score'] = silhouette
        except Exception as e:
            print(f"Could not calculate silhouette score: {e}")
            metrics['silhouette_score'] = np.nan
    
    return metrics


# Convenience functions for specific use cases
def spatialglue_clustering(adata, n_clusters, **kwargs):
    """Clustering wrapper for SpatialGlue results"""
    return batch_clustering(adata, n_clusters, used_obsm='spatial_emb', prefix='SpatialGlue_', **kwargs)

def spamosaic_clustering(adata, n_clusters, **kwargs):
    """Clustering wrapper for SpaMosaic results"""  
    return batch_clustering(adata, n_clusters, used_obsm='merged_emb', prefix='SpaMosaic_', **kwargs)

def praga_clustering(adata, n_clusters, **kwargs):
    """Clustering wrapper for PRAGA results"""
    return batch_clustering(adata, n_clusters, used_obsm='PRAGA_emb', prefix='PRAGA_', **kwargs)
