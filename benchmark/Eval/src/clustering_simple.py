"""
Simple clustering utilities as fallback when communities library is not available
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def knn_adj_matrix(embeddings, n_neighbors=15):
    """
    Construct k-nearest neighbors adjacency matrix
    
    Parameters:
    -----------
    embeddings : np.ndarray
        Cell embeddings, shape (n_cells, n_features)
    n_neighbors : int
        Number of nearest neighbors
        
    Returns:
    --------
    adj_matrix : scipy.sparse.csr_matrix
        Adjacency matrix
    """
    
    # Ensure we don't request more neighbors than available samples
    n_samples = embeddings.shape[0]
    n_neighbors = min(n_neighbors, n_samples - 1)
    
    if n_neighbors <= 0:
        # Return empty sparse matrix if no neighbors possible
        return csr_matrix((n_samples, n_samples))
    
    # Fit k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(embeddings)  # +1 for self
    distances, indices = nbrs.kneighbors(embeddings)
    
    # Create adjacency matrix
    row_ind = []
    col_ind = []
    data = []
    
    for i in range(n_samples):
        for j in range(1, len(indices[i])):  # Skip self (index 0)
            neighbor_idx = indices[i][j]
            distance = distances[i][j]
            
            # Use inverse distance as weight (add small epsilon to avoid division by zero)
            weight = 1.0 / (distance + 1e-8)
            
            row_ind.append(i)
            col_ind.append(neighbor_idx)
            data.append(weight)
    
    # Create sparse matrix
    adj_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(n_samples, n_samples))
    
    # Make symmetric
    adj_matrix = adj_matrix + adj_matrix.T
    
    # Normalize
    adj_matrix.data = adj_matrix.data / adj_matrix.data.max()
    
    return adj_matrix