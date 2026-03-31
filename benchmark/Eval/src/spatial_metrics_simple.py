"""
Simplified spatial metrics implementations as fallback when pysal is not available
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def Moran_Geary(coordinates, labels):
    """
    Simplified implementation of Moran's I and Geary's C without pysal dependency
    
    Parameters:
    -----------
    coordinates : array-like, shape (n_samples, 2)
        Spatial coordinates of samples
    labels : array-like, shape (n_samples,)
        Cluster labels for each sample
        
    Returns:
    --------
    moran_result : object with .I attribute
        Moran's I statistic
    geary_result : object with .C attribute  
        Geary's C statistic
    """
    
    # Convert coordinates to numpy array
    coordinates = np.asarray(coordinates)
    
    # Handle labels conversion carefully
    try:
        import pandas as pd
        # Check if it's a pandas categorical BEFORE converting to array
        if isinstance(labels, pd.Categorical) or (hasattr(labels, 'dtype') and str(labels.dtype).startswith('category')):
            # Convert pandas categorical to numeric codes
            if isinstance(labels, pd.Categorical):
                numeric_labels = labels.codes.astype(float)
            else:
                numeric_labels = pd.Categorical(labels).codes.astype(float)
        else:
            # Try to convert to array first
            labels_array = np.asarray(labels)
            if not np.issubdtype(labels_array.dtype, np.number):
                # String/object labels - manual conversion
                unique_labels = np.unique(labels_array)
                label_to_num = {label: i for i, label in enumerate(unique_labels)}
                numeric_labels = np.array([label_to_num[label] for label in labels_array], dtype=float)
            else:
                # Already numeric
                numeric_labels = labels_array.astype(float)
    except Exception as e:
        # Fallback: manual string to numeric conversion
        try:
            # Convert each element manually
            labels_list = list(labels)
            unique_labels = list(set(labels_list))
            label_to_num = {label: i for i, label in enumerate(unique_labels)}
            numeric_labels = np.array([label_to_num[label] for label in labels_list], dtype=float)
        except:
            # Ultimate fallback: just use range indices
            numeric_labels = np.arange(len(labels), dtype=float)
    
    # Build spatial weights matrix using k-nearest neighbors
    n_neighbors = min(3, len(coordinates) - 1)
    if n_neighbors <= 0:
        # Fallback values when insufficient data
        class SimpleResult:
            def __init__(self, value):
                self.I = value if hasattr(self, 'I') else None
                self.C = value if hasattr(self, 'C') else None
        
        moran_result = SimpleResult(0.0)
        moran_result.I = 0.0
        geary_result = SimpleResult(1.0) 
        geary_result.C = 1.0
        return moran_result, geary_result
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coordinates)
    _, indices = nbrs.kneighbors(coordinates)
    
    # Create weights matrix
    n = len(coordinates)
    weights = np.zeros((n, n))
    
    for i in range(n):
        neighbors = indices[i, 1:]  # Skip self
        for j in neighbors:
            weights[i, j] = 1.0
    
    # Normalize weights 
    row_sums = weights.sum(axis=1)
    weights = weights / (row_sums[:, np.newaxis] + 1e-10)
    
    # Calculate Moran's I
    y = numeric_labels.astype(float)
    n = len(y)
    y_mean = np.mean(y)
    y_centered = y - y_mean
    
    # Numerator: sum of cross products weighted by spatial weights
    numerator = 0
    for i in range(n):
        for j in range(n):
            numerator += weights[i, j] * y_centered[i] * y_centered[j]
    
    # Denominator: sum of squared deviations
    denominator = np.sum(y_centered ** 2)
    
    # Total sum of weights
    W = np.sum(weights)
    
    if denominator > 0 and W > 0:
        moran_i = (n / W) * (numerator / denominator)
    else:
        moran_i = 0.0
    
    # Calculate Geary's C
    numerator_geary = 0
    for i in range(n):
        for j in range(n):
            numerator_geary += weights[i, j] * (y[i] - y[j]) ** 2
    
    if denominator > 0 and W > 0:
        geary_c = ((n - 1) / (2 * W)) * (numerator_geary / denominator)
    else:
        geary_c = 1.0
    
    # Create result objects with .I and .C attributes
    class MoranResult:
        def __init__(self, value):
            self.I = value
    
    class GearyResult:
        def __init__(self, value):
            self.C = value
    
    return MoranResult(moran_i), GearyResult(geary_c)


def moran_i_simple(coordinates, values, k=3):
    """
    Simple Moran's I calculation
    """
    try:
        moran_result, _ = Moran_Geary(coordinates, values)
        return moran_result.I
    except:
        return 0.0


def geary_c_simple(coordinates, values, k=3):
    """
    Simple Geary's C calculation  
    """
    try:
        _, geary_result = Moran_Geary(coordinates, values)
        return geary_result.C
    except:
        return 1.0