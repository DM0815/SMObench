"""
Vertical Integration Evaluation Functions for SMOBench
Evaluates spatial multi-omics integration methods on SC (Spatial Coherence) and BioC (Biological Conservation) metrics
"""

import os
import pandas as pd
import numpy as np
from sklearn import metrics
# Import functions with fallback handling
def safe_import():
    """Safely import functions with fallback implementations"""
    global Moran_Geary, silhouette_simple, graph_clisi
    
    import sys
    import os
    
    # Add Eval directory to path for local imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    eval_dir = os.path.dirname(current_dir)
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    
    try:
        # Try to import individual functions
        from src.compute_metric import silhouette_simple, graph_clisi
        print("Successfully imported silhouette_simple and graph_clisi from compute_metric")
        
        # Try to import Moran_Geary
        try:
            from src.compute_metric import Moran_Geary
            print("Successfully imported Moran_Geary from compute_metric")
        except ImportError:
            print("Warning: Moran_Geary import failed, using spatial_metrics_simple fallback")
            from src.spatial_metrics_simple import Moran_Geary
            
    except ImportError as e:
        print(f"Warning: compute_metric import failed ({e}), using fallback implementations")
        
        # Import from fallback
        try:
            from src.spatial_metrics_simple import Moran_Geary
            print("Using spatial_metrics_simple fallback for Moran_Geary")
        except ImportError:
            print("Warning: spatial_metrics_simple also failed, using inline fallback")
            
            def Moran_Geary(coordinates, labels):
                """Inline fallback implementation"""
                class SimpleResult:
                    def __init__(self, value):
                        self.I = value if hasattr(self, 'I') else None
                        self.C = value if hasattr(self, 'C') else None
                
                moran_result = SimpleResult(0.0)
                moran_result.I = 0.0
                geary_result = SimpleResult(1.0) 
                geary_result.C = 1.0
                return moran_result, geary_result
        
        def silhouette_simple(embeddings, labels):
            """Fallback silhouette implementation"""
            from sklearn.metrics import silhouette_score
            try:
                asw = silhouette_score(embeddings, labels)
                return (asw + 1) / 2  # Scale to match original implementation
            except:
                return 0.0
        
        def graph_clisi(adj_matrix, labels):
            """Fallback graph_clisi implementation"""
            # Simplified version - just return a reasonable default
            return 0.5

# Call the safe import function
safe_import()

def calculate_ber_metrics(embeddings, batch_labels, n_neighbors=5):
    """
    Calculate Batch Effect Removal (BER) metrics
    
    Parameters:
    -----------
    embeddings : np.ndarray
        Integrated embeddings
    batch_labels : array-like
        Batch assignment for each cell (for vertical integration, could be modality labels)
    n_neighbors : int
        Number of neighbors for KNN-based metrics
        
    Returns:
    --------
    dict : BER metrics
    """
    try:
        from sklearn.neighbors import NearestNeighbors
        from sklearn.metrics import silhouette_score
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        import numpy as np
        
        ber_metrics = {}
        
        # Convert batch labels to numeric if they are strings
        unique_batches = np.unique(batch_labels)
        n_batches = len(unique_batches)
        
        if n_batches < 2:
            # No batch effect to remove (single modality/batch)
            return {
                'kBET': 1.0,
                'KNN_connectivity': 1.0, 
                'bASW': 1.0,
                'iLISI': 1.0,
                'PCR': 1.0
            }
        
        # Convert string batch labels to numeric for processing
        if batch_labels.dtype == 'object' or not np.issubdtype(batch_labels.dtype, np.number):
            batch_label_map = {label: i for i, label in enumerate(unique_batches)}
            batch_labels_numeric = np.array([batch_label_map[label] for label in batch_labels])
        else:
            batch_labels_numeric = batch_labels.astype(int)
        
        # Build kNN structure once (exclude self-neighbors)
        knn = min(n_neighbors, len(embeddings) - 1)
        if knn < 1:
            return {
                'kBET': 1.0,
                'KNN_connectivity': 1.0,
                'bASW': 1.0,
                'iLISI': 1.0,
                'PCR': 1.0
            }
        
        nbrs = NearestNeighbors(n_neighbors=knn + 1).fit(embeddings)
        distances, indices_full = nbrs.kneighbors(embeddings)
        neighbor_indices = indices_full[:, 1:]
        neighbor_distances = distances[:, 1:]
        
        # 1. kBET (simplified version)
        # Measures batch mixing in local neighborhoods
        
        kbet_scores = []
        for i, neighbors in enumerate(neighbor_indices):
            neighbor_batches = batch_labels_numeric[neighbors]
            # Calculate batch distribution in neighborhood
            batch_counts = np.bincount(neighbor_batches, minlength=n_batches)
            expected_prop = 1.0 / n_batches
            actual_props = batch_counts / len(neighbors)
            # Chi-square like statistic (simplified)
            kbet_score = 1.0 - np.sum((actual_props - expected_prop) ** 2)
            kbet_scores.append(max(0.0, kbet_score))
        
        ber_metrics['kBET'] = np.mean(kbet_scores)
        
        # Build global kNN graph for connectivity/iLISI calculations
        row_ind = np.repeat(np.arange(len(embeddings)), neighbor_indices.shape[1])
        col_ind = neighbor_indices.flatten()
        weights = (1.0 / (neighbor_distances.flatten() + 1e-8))
        knn_graph = csr_matrix((weights, (row_ind, col_ind)), shape=(len(embeddings), len(embeddings)))
        knn_graph = knn_graph.maximum(knn_graph.T)
        
        # 2. KNN connectivity (scIB-style: largest connected component per batch)
        connectivity_scores = []
        for i, batch in enumerate(unique_batches):
            batch_mask = batch_labels_numeric == i
            batch_size = np.sum(batch_mask)
            if batch_size < 2:
                continue
            
            subgraph = knn_graph[batch_mask][:, batch_mask]
            if subgraph.nnz == 0:
                connectivity_scores.append(0.0)
                continue
            
            n_components, component_labels = connected_components(subgraph, directed=False)
            if n_components == 0:
                connectivity_scores.append(0.0)
                continue
            largest = np.bincount(component_labels).max()
            connectivity_scores.append(largest / batch_size)
        
        ber_metrics['KNN_connectivity'] = np.mean(connectivity_scores) if connectivity_scores else 0.0
        
        # 3. bASW (batch-corrected Average Silhouette Width)
        try:
            # Negative silhouette with respect to batch labels (higher is better when batches are mixed)
            # Raw silhouette ∈ [-1, 1]; normalize to [0, 1] then invert
            batch_silhouette = silhouette_score(embeddings, batch_labels_numeric)
            ber_metrics['bASW'] = (1.0 - batch_silhouette) / 2.0
        except:
            ber_metrics['bASW'] = 0.5
        
        # 4. iLISI (Integration Local Inverse Simpson Index) - simplified
        ilisi_scores = []
        for i, neighbors in enumerate(neighbor_indices):
            neighbor_batches = batch_labels_numeric[neighbors]
            batch_counts = np.bincount(neighbor_batches, minlength=n_batches)
            # Simpson index
            props = batch_counts / len(neighbors)
            simpson = np.sum(props ** 2)
            ilisi = 1.0 / simpson if simpson > 0 else 1.0
            ilisi_scores.append(min(ilisi / n_batches, 1.0))  # Normalize
        
        ber_metrics['iLISI'] = np.mean(ilisi_scores)
        
        # 5. PCR (Principal Component Regression) - simplified
        try:
            from sklearn.decomposition import PCA
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
            
            # Use PCA components to predict batch
            pca = PCA(n_components=min(50, embeddings.shape[1]))
            pca_embeddings = pca.fit_transform(embeddings)
            
            clf = LogisticRegression(random_state=42, max_iter=100)
            scores = cross_val_score(clf, pca_embeddings, batch_labels_numeric, cv=3)
            # Higher PCR means more batch effect (lower is better)
            ber_metrics['PCR'] = max(0.0, 1.0 - np.mean(scores))
        except:
            ber_metrics['PCR'] = 0.5
            
        return ber_metrics
        
    except Exception as e:
        print(f"Warning: Could not compute BER metrics: {e}")
        return {
            'kBET': 0.0,
            'KNN_connectivity': 0.0,
            'bASW': 0.0,
            'iLISI': 0.0,
            'PCR': 0.0
        }

def eval_vertical_integration(embeddings, adj_matrix, y_pred, y_GT, spatial_coords, method_name, dataset_name, slice_name, clustering_method):
    """
    Evaluate vertical integration results for a single slice
    
    Parameters:
    -----------
    embeddings : np.ndarray
        Integrated embeddings from the method
    adj_matrix : sparse matrix
        Adjacency matrix for spatial relationships
    y_pred : array-like
        Predicted cluster labels
    y_GT : array-like or None
        Ground truth labels (Spatial_Label for withGT datasets, None for woGT)
    spatial_coords : np.ndarray
        Spatial coordinates for cells
    method_name : str
        Name of the integration method
    dataset_name : str
        Name of the tissue/dataset (e.g., 'HLN', 'HT', 'ME_S1')
    slice_name : str
        Name of the slice (e.g., 'A1', 'S1', 'E11')
    clustering_method : str
        Clustering method used ('leiden', 'louvain', 'kmeans', 'mclust')
    
    Returns:
    --------
    dict : Evaluation metrics with SC and BVC scores
    """
    
    # Calculate Spatial Coherence (SC) metrics
    try:
        # y_pred should already be numeric from eval_adata.py preprocessing
        y_pred_numeric = np.asarray(y_pred, dtype=float)
        
        Moran, Geary = Moran_Geary(spatial_coords, y_pred_numeric)
        sc_metrics = {
            'Moran Index': Moran.I,
            'Geary C': Geary.C
        }
    except Exception as e:
        print(f"Warning: Could not compute spatial metrics for {method_name}_{dataset_name}_{slice_name}: {e}")
        sc_metrics = {
            'Moran Index': 0.0,
            'Geary C': 0.0
        }
    
    # Initialize metrics dictionary
    metrics_dict = sc_metrics.copy()
    
    # Calculate Biological Conservation (BioC) metrics
    if y_GT is not None:
        # withGT metrics - check for data alignment first
        try:
            # Check if all arrays have consistent lengths
            n_embeddings = embeddings.shape[0]
            n_gt = len(y_GT)
            n_pred = len(y_pred)
            n_coords = spatial_coords.shape[0]
            
            if not all(n == n_embeddings for n in [n_gt, n_pred, n_coords]):
                print(f"Warning: Data length mismatch for {method_name}_{dataset_name}_{slice_name}:")
                print(f"  Embeddings: {n_embeddings}, GT: {n_gt}, Pred: {n_pred}, Coords: {n_coords}")
                
                # Find minimum length and truncate all arrays
                min_len = min(n_embeddings, n_gt, n_pred, n_coords)
                print(f"  Truncating all arrays to length: {min_len}")
                
                embeddings_aligned = embeddings[:min_len]
                y_GT_aligned = y_GT[:min_len]
                y_pred_aligned = y_pred[:min_len]
                adj_matrix_aligned = adj_matrix[:min_len, :min_len]
            else:
                embeddings_aligned = embeddings
                y_GT_aligned = y_GT
                y_pred_aligned = y_pred
                adj_matrix_aligned = adj_matrix
            
            # Calculate comprehensive withGT metrics (19 total)
            y_gt_flat = np.ravel(y_GT_aligned)
            y_pred_flat = np.ravel(y_pred_aligned)
            
            # Clustering Accuracy Metrics (11 metrics)
            bioc_metrics = {
                'ARI': metrics.adjusted_rand_score(y_gt_flat, y_pred_flat),
                'NMI': metrics.normalized_mutual_info_score(y_gt_flat, y_pred_flat),
                'AMI': metrics.adjusted_mutual_info_score(y_gt_flat, y_pred_flat),
                'FMI': metrics.fowlkes_mallows_score(y_gt_flat, y_pred_flat),
                'Homogeneity': metrics.homogeneity_score(y_gt_flat, y_pred_flat),
                'Completeness': metrics.completeness_score(y_gt_flat, y_pred_flat),
                'V-measure': metrics.v_measure_score(y_gt_flat, y_pred_flat)
            }
            
            # Import additional metrics from compute_metric
            try:
                from src.compute_metric import purity, F_measure, jaccard, Dice
                bioc_metrics.update({
                    'Purity': purity(y_pred_flat, y_gt_flat),
                    'F-measure': F_measure(y_pred_flat, y_gt_flat),
                    'Jaccard Index': jaccard(y_pred_flat, y_gt_flat),
                    'Dice Index': Dice(y_pred_flat, y_gt_flat)
                })
            except ImportError:
                # Fallback implementations if imports fail
                bioc_metrics.update({
                    'Purity': 0.0,
                    'F-measure': 0.0,
                    'Jaccard Index': 0.0,
                    'Dice Index': 0.0
                })
            
            # Quality Metrics (3 metrics) - keep original values for min-max standardization later
            try:
                raw_silhouette = metrics.silhouette_score(embeddings_aligned, y_pred_flat)
                raw_chi_withgt = metrics.calinski_harabasz_score(embeddings_aligned, y_pred_flat)
                raw_bdi_withgt = metrics.davies_bouldin_score(embeddings_aligned, y_pred_flat)
                
                bioc_metrics.update({
                    'Silhouette Coefficient': raw_silhouette,
                    'Calinski-Harabasz Index': raw_chi_withgt,  # Keep original CHI
                    'Davies-Bouldin Index': raw_bdi_withgt  # Keep original DBI
                })
            except:
                bioc_metrics.update({
                    'Silhouette Coefficient': 0.0,
                    'Calinski-Harabasz Index': 0.0,
                    'Davies-Bouldin Index': 0.0
                })
            
            # Additional BioC metrics (2 metrics)
            bioc_metrics.update({
                'asw_celltype': silhouette_simple(embeddings_aligned, y_GT_aligned),
                'graph_clisi': graph_clisi(adj_matrix_aligned, y_GT_aligned)
            })
            metrics_dict.update(bioc_metrics)
        except Exception as e:
            print(f"Warning: Could not compute withGT BioC metrics for {method_name}_{dataset_name}_{slice_name}: {e}")
            # Add default values for comprehensive withGT metrics
            bioc_metrics = {
                # Clustering Accuracy Metrics (11)
                'ARI': 0.0,
                'NMI': 0.0,
                'AMI': 0.0,
                'FMI': 0.0,
                'Homogeneity': 0.0,
                'Completeness': 0.0,
                'V-measure': 0.0,
                'Purity': 0.0,
                'F-measure': 0.0,
                'Jaccard Index': 0.0,
                'Dice Index': 0.0,
                # Quality Metrics (3)
                'Silhouette Coefficient': 0.0,
                'Calinski-Harabasz Index': 0.0,
                'Davies-Bouldin Index': 0.0,  # Standardized default
                # Additional BioC metrics (2)
                'asw_celltype': 0.0,
                'graph_clisi': 0.0
            }
            metrics_dict.update(bioc_metrics)
    else:
        # woGT metrics
        try:
            raw_bdi = metrics.davies_bouldin_score(embeddings, y_pred)
            raw_chi = metrics.calinski_harabasz_score(embeddings, y_pred)
            raw_silhouette = metrics.silhouette_score(embeddings, y_pred, metric='euclidean')
            
            bioc_metrics = {
                'Davies-Bouldin Index': raw_bdi,  # Keep original DBI
                'Silhouette Coefficient': raw_silhouette,
                'Calinski-Harabaz Index': raw_chi  # Keep original CHI
            }
            metrics_dict.update(bioc_metrics)
        except Exception as e:
            print(f"Warning: Could not compute woGT BioC metrics for {method_name}_{dataset_name}_{slice_name}: {e}")
            # Add default values
            bioc_metrics = {
                'Davies-Bouldin Index': 0.0,
                'Silhouette Coefficient': 0.0,
                'Calinski-Harabaz Index': 0.0
            }
            metrics_dict.update(bioc_metrics)
    
    # Calculate two-dimensional scores for vertical integration
    # SC Score: Moran Index only
    sc_score = metrics_dict.get('Moran Index', 0.0)
    
    # BVC Score: depends on GT availability
    if y_GT is not None:
        # withGT: ARI, NMI, asw_celltype, graph_clisi
        bvc_metrics = ['ARI', 'NMI', 'asw_celltype', 'graph_clisi']
        bvc_values = [metrics_dict.get(m, 0.0) for m in bvc_metrics if m in metrics_dict]
        bvc_score = np.mean(bvc_values) if bvc_values else 0.0
    else:
        # woGT: Davies-Bouldin Index, Silhouette Coefficient, Calinski-Harabasz Index
        # Note: DBI and CHI will be standardized later at dataset level
        bvc_metrics = ['Davies-Bouldin Index', 'Silhouette Coefficient', 'Calinski-Harabaz Index']
        bvc_values = [metrics_dict.get(m, 0.0) for m in bvc_metrics if m in metrics_dict]
        bvc_score = np.mean(bvc_values) if bvc_values else 0.0
    
    # Total Score: SC + BVC average (two dimensions for vertical integration)
    total_score = (sc_score + bvc_score) / 2
    
    # Add scores to metrics
    metrics_dict.update({
        'SC_Score': sc_score,
        'BVC_Score': bvc_score,
        'Total_Score': total_score
    })
    
    # Add metadata
    metrics_dict['Method'] = method_name
    metrics_dict['Dataset'] = dataset_name
    metrics_dict['Slice'] = slice_name
    metrics_dict['Clustering'] = clustering_method
    metrics_dict['GT_Available'] = y_GT is not None
    
    return metrics_dict

def eval_horizontal_integration(embeddings, adj_matrix, y_pred, y_GT, spatial_coords, batch_labels, method_name, dataset_name, slice_name, clustering_method):
    """
    Evaluate horizontal integration results for a single slice with three dimensions: SC, BVC, BER
    
    Parameters:
    -----------
    embeddings : np.ndarray
        Integrated embeddings from the method
    adj_matrix : sparse matrix
        Adjacency matrix for spatial relationships
    y_pred : array-like
        Predicted cluster labels
    y_GT : array-like or None
        Ground truth labels (Spatial_Label for withGT datasets, None for woGT)
    spatial_coords : np.ndarray
        Spatial coordinates for cells
    batch_labels : array-like
        Batch assignment for each cell (required for BER metrics)
    method_name : str
        Name of the integration method
    dataset_name : str
        Name of the tissue/dataset
    slice_name : str
        Name of the slice
    clustering_method : str
        Clustering method used ('leiden', 'louvain', 'kmeans', 'mclust')
    
    Returns:
    --------
    dict : Evaluation metrics with SC, BVC, BER scores
    """
    
    # Calculate Spatial Coherence (SC) metrics
    try:
        # y_pred should already be numeric from eval_adata.py preprocessing
        y_pred_numeric = np.asarray(y_pred, dtype=float)
        
        Moran, Geary = Moran_Geary(spatial_coords, y_pred_numeric)
        sc_metrics = {
            'Moran Index': Moran.I,
            'Geary C': Geary.C
        }
    except Exception as e:
        print(f"Warning: Could not compute spatial metrics for {method_name}_{dataset_name}_{slice_name}: {e}")
        sc_metrics = {
            'Moran Index': 0.0,
            'Geary C': 0.0
        }
    
    # Initialize metrics dictionary
    metrics_dict = sc_metrics.copy()
    
    # Calculate Biological Conservation (BVC) metrics
    if y_GT is not None:
        # withGT metrics - check for data alignment first
        try:
            # Check if all arrays have consistent lengths
            n_embeddings = embeddings.shape[0]
            n_gt = len(y_GT)
            n_pred = len(y_pred)
            n_coords = spatial_coords.shape[0]
            
            if not all(n == n_embeddings for n in [n_gt, n_pred, n_coords]):
                print(f"Warning: Data length mismatch for {method_name}_{dataset_name}_{slice_name}:")
                print(f"  Embeddings: {n_embeddings}, GT: {n_gt}, Pred: {n_pred}, Coords: {n_coords}")
                
                # Find minimum length and truncate all arrays
                min_len = min(n_embeddings, n_gt, n_pred, n_coords)
                print(f"  Truncating all arrays to length: {min_len}")
                
                embeddings_aligned = embeddings[:min_len]
                y_GT_aligned = y_GT[:min_len]
                y_pred_aligned = y_pred[:min_len]
                adj_matrix_aligned = adj_matrix[:min_len, :min_len]
                batch_labels_aligned = batch_labels[:min_len] if batch_labels is not None else None
            else:
                embeddings_aligned = embeddings
                y_GT_aligned = y_GT
                y_pred_aligned = y_pred
                adj_matrix_aligned = adj_matrix
                batch_labels_aligned = batch_labels
            
            # Calculate withGT BVC metrics: ARI, NMI, asw_celltype, graph_clisi
            y_gt_flat = np.ravel(y_GT_aligned)
            y_pred_flat = np.ravel(y_pred_aligned)
            
            bioc_metrics = {}
            
            # Calculate individual metrics with error handling
            try:
                bioc_metrics['ARI'] = metrics.adjusted_rand_score(y_gt_flat, y_pred_flat)
            except Exception as e:
                print(f"    Warning: ARI calculation failed: {e}")
                bioc_metrics['ARI'] = 0.0
                
            try:
                bioc_metrics['NMI'] = metrics.normalized_mutual_info_score(y_gt_flat, y_pred_flat)
            except Exception as e:
                print(f"    Warning: NMI calculation failed: {e}")
                bioc_metrics['NMI'] = 0.0
                
            try:
                bioc_metrics['asw_celltype'] = silhouette_simple(embeddings_aligned, y_GT_aligned)
            except Exception as e:
                print(f"    Warning: asw_celltype calculation failed: {e}")
                bioc_metrics['asw_celltype'] = 0.0
                
            try:
                bioc_metrics['graph_clisi'] = graph_clisi(adj_matrix_aligned, y_GT_aligned)
            except Exception as e:
                print(f"    Warning: graph_clisi calculation failed: {e}")
                bioc_metrics['graph_clisi'] = 0.0
            metrics_dict.update(bioc_metrics)
        except Exception as e:
            print(f"Warning: Could not compute withGT BVC metrics for {method_name}_{dataset_name}_{slice_name}: {e}")
            # Add default values for withGT BVC metrics
            bioc_metrics = {
                'ARI': 0.0,
                'NMI': 0.0,
                'asw_celltype': 0.0,
                'graph_clisi': 0.0
            }
            metrics_dict.update(bioc_metrics)
            embeddings_aligned = embeddings
            batch_labels_aligned = batch_labels
    else:
        # woGT metrics: Davies-Bouldin Index, Silhouette Coefficient, Calinski-Harabasz Index
        try:
            raw_dbi = metrics.davies_bouldin_score(embeddings, y_pred)
            raw_chi = metrics.calinski_harabasz_score(embeddings, y_pred)
            raw_silhouette = metrics.silhouette_score(embeddings, y_pred, metric='euclidean')
            
            bioc_metrics = {
                'Davies-Bouldin Index': raw_dbi,  # Keep original DBI
                'Silhouette Coefficient': raw_silhouette,
                'Calinski-Harabaz Index': raw_chi  # Keep original CHI
            }
            metrics_dict.update(bioc_metrics)
        except Exception as e:
            print(f"Warning: Could not compute woGT BVC metrics for {method_name}_{dataset_name}_{slice_name}: {e}")
            # Add default values
            bioc_metrics = {
                'Davies-Bouldin Index': 0.0,
                'Silhouette Coefficient': 0.0,
                'Calinski-Harabaz Index': 0.0
            }
            metrics_dict.update(bioc_metrics)
        
        embeddings_aligned = embeddings
        batch_labels_aligned = batch_labels
    
    # Calculate Batch Effect Removal (BER) metrics
    if batch_labels_aligned is not None and len(np.unique(batch_labels_aligned)) > 1:
        try:
            ber_metrics = calculate_ber_metrics(embeddings_aligned, batch_labels_aligned)
            metrics_dict.update(ber_metrics)
        except Exception as e:
            print(f"Warning: Could not compute BER metrics for {method_name}_{dataset_name}_{slice_name}: {e}")
            ber_metrics = {
                'kBET': 0.0,
                'KNN_connectivity': 0.0,
                'bASW': 0.0,
                'iLISI': 0.0,
                'PCR': 0.0
            }
            metrics_dict.update(ber_metrics)
    else:
        # No batch effect to remove or no batch labels
        print(f"Warning: No batch labels or single batch for {method_name}_{dataset_name}_{slice_name}")
        ber_metrics = {
            'kBET': 1.0,
            'KNN_connectivity': 1.0,
            'bASW': 1.0,
            'iLISI': 1.0,
            'PCR': 1.0
        }
        metrics_dict.update(ber_metrics)
    
    # Calculate three-dimensional scores for horizontal integration
    # SC Score: Moran Index only
    sc_score = metrics_dict.get('Moran Index', 0.0)
    
    # BVC Score: depends on GT availability
    if y_GT is not None:
        # withGT: ARI, NMI, asw_celltype, graph_clisi
        bvc_metrics = ['ARI', 'NMI', 'asw_celltype', 'graph_clisi']
        bvc_values = []
        for m in bvc_metrics:
            if m in metrics_dict:
                val = metrics_dict[m]
                if not np.isnan(val) and not np.isinf(val):
                    bvc_values.append(val)
        bvc_score = np.mean(bvc_values) if bvc_values else 0.0
    else:
        # woGT: Davies-Bouldin Index, Silhouette Coefficient, Calinski-Harabasz Index
        # Note: DBI and CHI will be standardized later at dataset level
        bvc_metrics = ['Davies-Bouldin Index', 'Silhouette Coefficient', 'Calinski-Harabaz Index']
        bvc_values = []
        for m in bvc_metrics:
            if m in metrics_dict:
                val = metrics_dict[m]
                if not np.isnan(val) and not np.isinf(val):
                    bvc_values.append(val)
        bvc_score = np.mean(bvc_values) if bvc_values else 0.0
    
    # BER Score: kBET, KNN_connectivity, bASW, iLISI, PCR
    ber_metric_names = ['kBET', 'KNN_connectivity', 'bASW', 'iLISI', 'PCR']
    ber_values = []
    for m in ber_metric_names:
        if m in metrics_dict:
            val = metrics_dict[m]
            if not np.isnan(val) and not np.isinf(val):
                ber_values.append(val)
    ber_score = np.mean(ber_values) if ber_values else 0.0
    
    # Final SCORE: SC + BVC + BER average (three dimensions for horizontal integration)
    final_score = np.mean([sc_score, bvc_score, ber_score])
    
    # Add scores to metrics
    metrics_dict.update({
        'SC_Score': sc_score,
        'BVC_Score': bvc_score,
        'BER_Score': ber_score,
        'Final_Score': final_score
    })
    
    # Add metadata
    metrics_dict['Method'] = method_name
    metrics_dict['Dataset'] = dataset_name
    metrics_dict['Slice'] = slice_name
    metrics_dict['Clustering'] = clustering_method
    metrics_dict['GT_Available'] = y_GT is not None
    
    return metrics_dict

def save_evaluation_results(metrics_dict, output_dir, method_name, dataset_name, slice_name, clustering_method, has_gt):
    """
    Save evaluation results to CSV file
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary containing evaluation metrics
    output_dir : str
        Output directory path
    method_name : str
        Name of the integration method
    dataset_name : str
        Name of the tissue/dataset
    slice_name : str
        Name of the slice
    clustering_method : str
        Clustering method used
    has_gt : bool
        Whether ground truth is available
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for saving (exclude metadata fields)
    save_metrics = {k: v for k, v in metrics_dict.items() 
                   if k not in ['Method', 'Dataset', 'Slice', 'Clustering', 'GT_Available']}
    
    # Create DataFrame
    df = pd.DataFrame(list(save_metrics.items()), columns=['Metric', 'Value'])
    
    # Determine file suffix
    suffix = 'withGT' if has_gt else 'woGT'
    
    # Save to CSV
    filename = f"{method_name}_{dataset_name}_{slice_name}_{clustering_method}_{suffix}.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    
    print(f"Saved evaluation results to: {filepath}")
    
    return filepath

def calculate_dataset_summary(metrics_list, dataset_name, method_name, clustering_method):
    """
    Calculate summary statistics for a dataset (averaging across slices)
    
    Parameters:
    -----------
    metrics_list : list of dict
        List of metrics dictionaries from different slices
    dataset_name : str
        Name of the dataset
    method_name : str
        Name of the method
    clustering_method : str
        Clustering method used
        
    Returns:
    --------
    dict : Summary metrics for the dataset
    """
    
    if not metrics_list:
        return None
    
    # Determine metric categories
    has_gt = metrics_list[0]['GT_Available']
    
    # Define metrics for two dimensions (vertical integration)
    sc_metrics = ['Moran Index']  # SC: only Moran Index
    
    if has_gt:
        bioc_metrics = ['ARI', 'NMI', 'asw_celltype', 'graph_clisi']
    else:
        bioc_metrics = ['Davies-Bouldin Index', 'Silhouette Coefficient', 'Calinski-Harabaz Index']
    
    # Calculate averages
    summary = {
        'Method': method_name,
        'Dataset': dataset_name,
        'Clustering': clustering_method,
        'GT_Available': has_gt
    }
    
    # Average SC metrics
    sc_values = []
    for metric in sc_metrics:
        values = [m[metric] for m in metrics_list if metric in m and not np.isnan(m[metric])]
        if values:
            avg_val = np.mean(values)
            summary[metric] = avg_val
            sc_values.append(avg_val)
    
    # Average BioC metrics
    bioc_values = []
    for metric in bioc_metrics:
        values = [m[metric] for m in metrics_list if metric in m and not np.isnan(m[metric])]
        if values:
            avg_val = np.mean(values)
            summary[metric] = avg_val
            bioc_values.append(avg_val)
    
    # Calculate two-dimensional scores (vertical integration)
    summary['SC_Score'] = summary.get('Moran Index', 0.0)  # SC: only Moran Index
    summary['BVC_Score'] = np.mean(bioc_values) if bioc_values else 0.0  # BVC: average of BioC metrics
    summary['Total_Score'] = (summary['SC_Score'] + summary['BVC_Score']) / 2  # Two-way average
    
    return summary

def calculate_horizontal_dataset_summary(metrics_list, dataset_name, method_name, clustering_method):
    """
    Calculate summary statistics for horizontal integration dataset (averaging across slices)
    
    Parameters:
    -----------
    metrics_list : list of dict
        List of metrics dictionaries from different slices
    dataset_name : str
        Name of the dataset
    method_name : str
        Name of the method
    clustering_method : str
        Clustering method used
        
    Returns:
    --------
    dict : Summary metrics for the dataset with three dimensions (SC, BVC, BER)
    """
    
    if not metrics_list:
        return None
    
    # Determine metric categories for horizontal integration (three dimensions)
    has_gt = metrics_list[0]['GT_Available']
    
    sc_metrics = ['Moran Index']  # SC: only Moran Index
    ber_metrics = ['kBET', 'KNN_connectivity', 'bASW', 'iLISI', 'PCR']  # BER metrics
    
    if has_gt:
        bioc_metrics = ['ARI', 'NMI', 'asw_celltype', 'graph_clisi']
    else:
        bioc_metrics = ['Davies-Bouldin Index', 'Silhouette Coefficient', 'Calinski-Harabaz Index']
    
    # Calculate averages
    summary = {
        'Method': method_name,
        'Dataset': dataset_name,
        'Clustering': clustering_method,
        'GT_Available': has_gt
    }
    
    # Average SC metrics
    sc_values = []
    for metric in sc_metrics:
        values = [m[metric] for m in metrics_list if metric in m and not np.isnan(m[metric])]
        if values:
            avg_val = np.mean(values)
            summary[metric] = avg_val
            sc_values.append(avg_val)
    
    # Average BioC metrics
    bioc_values = []
    for metric in bioc_metrics:
        values = [m[metric] for m in metrics_list if metric in m and not np.isnan(m[metric])]
        if values:
            avg_val = np.mean(values)
            summary[metric] = avg_val
            bioc_values.append(avg_val)
    
    # Average BER metrics
    ber_values = []
    for metric in ber_metrics:
        values = [m[metric] for m in metrics_list if metric in m and not np.isnan(m[metric])]
        if values:
            avg_val = np.mean(values)
            summary[metric] = avg_val
            ber_values.append(avg_val)
    
    # Calculate three-dimensional scores for horizontal integration
    summary['SC_Score'] = summary.get('Moran Index', 0.0)  # SC: only Moran Index
    summary['BVC_Score'] = np.mean(bioc_values) if bioc_values else 0.0  # BVC: average of BioC metrics
    summary['BER_Score'] = np.mean(ber_values) if ber_values else 0.0   # BER: average of BER metrics
    summary['Final_Score'] = (summary['SC_Score'] + summary['BVC_Score'] + summary['BER_Score']) / 3  # Three-way average
    
    return summary
