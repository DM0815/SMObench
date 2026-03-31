"""
Batch Evaluation of Vertical Integration Results for SMOBench
Processes all integration results and generates comprehensive evaluation metrics

Usage:
    python eval_adata.py [--test]
    
    --test : Run in test mode (evaluate only first 5 files)
"""

import os
import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.demo import eval_vertical_integration, save_evaluation_results, calculate_dataset_summary
# Try to import from original clustering, fall back to simplified version
try:
    from src.clustering import knn_adj_matrix
    print("Using original clustering functions")
except ImportError as e:
    print(f"Warning: src.clustering not available ({e}), using simple implementation")
    from src.clustering_simple import knn_adj_matrix

# Dataset classification based on Dataset/withGT and Dataset/woGT directory structure
WITHGT_DATASETS = ['HLN', 'HT', 'MISAR_S1', 'MISAR_S2']  # From Dataset/withGT
WOGT_DATASETS = ['Mouse_Thymus', 'Mouse_Spleen', 'Mouse_Brain']  # From Dataset/woGT

# Dataset type mapping for GT loading
WITHGT_DATASET_TYPES = {
    'HLN': 'RNA_ADT',           # Human_Lymph_Nodes
    'HT': 'RNA_ADT',            # Human_Tonsils  
    'MISAR_S1': 'RNA_ATAC',     # Mouse_Embryos_S1
    'MISAR_S2': 'RNA_ATAC'      # Mouse_Embryos_S2
}

# Integration methods
METHODS = [
    'SMOPCA',
    # Legacy methods already evaluated (left commented to avoid re-processing):
    # 'CANDIES',
    # 'COSMOS',
    # 'PRAGA',
    # 'PRESENT',
    # 'SpaMV',
    # 'SpaMosaic',
    # 'SpatialGlue',
    # 'SpaMultiVAE',
    # 'SpaBalance',
    # 'SpaMI',
    # 'SpaFusion',
]

# Method-dataset compatibility (used to skip unsupported combinations)
METHOD_DATASET_COMPATIBILITY = {
    'SpaMultiVAE': ['HLN', 'HT', 'Mouse_Thymus', 'Mouse_Spleen'],  # Only RNA_ADT datasets (withGT + woGT)
    'SpaFusion': ['HLN', 'HT', 'Mouse_Thymus', 'Mouse_Spleen'],    # SpaFusion currently supports RNA+ADT datasets
    'SMOPCA': [
        'HLN',
        'HT',
        'MISAR_S1',
        'MISAR_S2',
        'Mouse_Brain',
        'Mouse_Spleen',
        'Mouse_Thymus',
    ],
}

# Clustering methods
CLUSTERING_METHODS = ['leiden', 'louvain', 'kmeans', 'mclust']

def get_ground_truth_labels(dataset_name, slice_name):
    """
    Load ground truth labels from original dataset
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset (e.g., 'HLN', 'HT')
    slice_name : str
        Name of the slice (e.g., 'A1', 'S1')
        
    Returns:
    --------
    np.ndarray or None : Ground truth labels if available
    """
    if dataset_name not in WITHGT_DATASETS:
        return None
    
    # Map dataset names to original paths
    dataset_mapping = {
        'HLN': 'Human_Lymph_Nodes',
        'HT': 'Human_Tonsils', 
        'MISAR_S1': 'Mouse_Embryos_S1',
        'MISAR_S2': 'Mouse_Embryos_S2'
    }
    
    # Map slice names for mouse embryos
    if dataset_name.startswith('MISAR'):
        slice_mapping = {'E11': 'E11', 'E13': 'E13', 'E15': 'E15', 'E18': 'E18'}
    else:
        slice_mapping = {'A1': 'A1', 'D1': 'D1', 'S1': 'S1', 'S2': 'S2', 'S3': 'S3'}
    
    if slice_name not in slice_mapping:
        print(f"Warning: Unknown slice {slice_name} for dataset {dataset_name}")
        return None
    
    # Determine data type
    if dataset_name in ['HLN', 'HT']:
        data_type = 'RNA_ADT'
    else:
        data_type = 'RNA_ATAC'
    
    # Construct path to original dataset
    original_dataset = dataset_mapping.get(dataset_name, dataset_name)
    gt_path = f"/home/zhenghong/SMOBench-CLEAN/Dataset/withGT/{data_type}/{original_dataset}/{slice_name}/adata_RNA.h5ad"
    
    try:
        adata_gt = sc.read_h5ad(gt_path)
        # Ground truth is stored in 'Spatial_Label' column
        if 'Spatial_Label' in adata_gt.obs.columns:
            return adata_gt.obs['Spatial_Label'].values
        else:
            print(f"Warning: No 'Spatial_Label' found in {gt_path}")
            return None
    except Exception as e:
        print(f"Warning: Could not load ground truth from {gt_path}: {e}")
        return None

def process_single_result(result_path, method_name, dataset_name, slice_name, output_dir):
    """
    Process a single integration result file
    
    Parameters:
    -----------
    result_path : str
        Path to the AnnData file with integration results
    method_name : str
        Name of the integration method
    dataset_name : str
        Name of the dataset
    slice_name : str
        Name of the slice
    output_dir : str
        Output directory for saving results
        
    Returns:
    --------
    list : List of evaluation results for each clustering method
    """
    
    print(f"Processing: {method_name} - {dataset_name} - {slice_name}")
    
    try:
        # Load integration results
        adata = sc.read_h5ad(result_path)
        
        # Get embeddings (try different possible keys)
        embedding_keys = [method_name, 'X_integrated', 'X_emb', 'embeddings']
        embeddings = None
        
        for key in embedding_keys:
            if key in adata.obsm:
                embeddings = adata.obsm[key]
                break
        
        if embeddings is None:
            print(f"Warning: No embeddings found for {method_name} in {result_path}")
            return []
        
        # Get spatial coordinates
        if 'spatial' in adata.obsm:
            spatial_coords = adata.obsm['spatial']
        else:
            print(f"Warning: No spatial coordinates found in {result_path}")
            return []
        
        # Create adjacency matrix
        adj_matrix = knn_adj_matrix(embeddings)
        
        # Get ground truth if available - load from original dataset for consistency
        # This ensures all methods are evaluated fairly regardless of whether they preserve Spatial_Label
        if dataset_name in WITHGT_DATASETS:
            y_GT = get_ground_truth_labels(dataset_name, slice_name)
            if y_GT is not None:
                # Ensure it's a numeric numpy array
                y_GT = np.asarray(y_GT, dtype=int)
        else:
            y_GT = None
        
        # Process each clustering method
        results = []
        for clustering_method in CLUSTERING_METHODS:
            if clustering_method in adata.obs.columns:
                y_pred_raw = adata.obs[clustering_method]
                
                # Convert to numeric if categorical
                if hasattr(y_pred_raw, 'cat'):
                    y_pred = y_pred_raw.cat.codes.values
                elif str(y_pred_raw.dtype).startswith('category'):
                    # Handle pandas categorical
                    import pandas as pd
                    y_pred = pd.Categorical(y_pred_raw).codes
                else:
                    y_pred = y_pred_raw.values
                    
                # Ensure it's a numeric numpy array
                y_pred = np.asarray(y_pred, dtype=int)
                
                # Evaluate
                metrics_dict = eval_vertical_integration(
                    embeddings=embeddings,
                    adj_matrix=adj_matrix, 
                    y_pred=y_pred,
                    y_GT=y_GT,
                    spatial_coords=spatial_coords,
                    method_name=method_name,
                    dataset_name=dataset_name,
                    slice_name=slice_name,
                    clustering_method=clustering_method
                )
                
                # Save individual result
                save_evaluation_results(
                    metrics_dict=metrics_dict,
                    output_dir=output_dir,
                    method_name=method_name,
                    dataset_name=dataset_name,
                    slice_name=slice_name,
                    clustering_method=clustering_method,
                    has_gt=(y_GT is not None)
                )
                
                results.append(metrics_dict)
            else:
                print(f"Warning: Clustering method {clustering_method} not found in {result_path}")
        
        return results
        
    except Exception as e:
        print(f"Error processing {result_path}: {e}")
        return []

def evaluate_all_vertical_integration(test_mode=False):
    """
    Evaluate all vertical integration results
    
    Parameters:
    -----------
    test_mode : bool
        If True, process only first 5 files for testing
    """
    
    print("="*80)
    print("SMOBench Vertical Integration Evaluation")
    print("="*80)
    
    # Setup directories
    results_base_dir = "/home/zhenghong/SMOBench-CLEAN/Results/adata/vertical_integration"
    output_base_dir = "/home/zhenghong/SMOBench-CLEAN/Results/evaluation/vertical_integration"
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Track all results for summary
    all_results = []
    files_processed = 0
    
    if test_mode:
        print("RUNNING IN TEST MODE - Processing only first 5 files")
    
    # Process each method
    for method_name in METHODS:
        method_dir = os.path.join(results_base_dir, method_name)
        if not os.path.exists(method_dir):
            print(f"Warning: Method directory not found: {method_dir}")
            continue
        
        print(f"\n--- Processing Method: {method_name} ---")
        
        # Process each dataset
        for dataset_name in os.listdir(method_dir):
            dataset_path = os.path.join(method_dir, dataset_name)
            if not os.path.isdir(dataset_path):
                continue
            
            # Check method-dataset compatibility
            if method_name in METHOD_DATASET_COMPATIBILITY:
                if dataset_name not in METHOD_DATASET_COMPATIBILITY[method_name]:
                    print(f"  Skipping {dataset_name} (not supported by {method_name})")
                    continue
            
            print(f"\n  Dataset: {dataset_name}")
            
            # Create output directory for this method-dataset combination
            output_dir = os.path.join(output_base_dir, method_name, dataset_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Process each slice
            dataset_results = []
            for slice_name in os.listdir(dataset_path):
                slice_path = os.path.join(dataset_path, slice_name)
                
                # Find the result file
                if os.path.isdir(slice_path):
                    # Look for .h5ad file in slice directory
                    h5ad_files = list(Path(slice_path).glob("*.h5ad"))
                    if h5ad_files:
                        result_path = str(h5ad_files[0])
                    else:
                        print(f"    Warning: No .h5ad file found in {slice_path}")
                        continue
                elif slice_path.endswith('.h5ad'):
                    result_path = slice_path
                    slice_name = Path(slice_path).stem.split('_')[-1]  # Extract slice name from filename
                else:
                    continue
                
                # Process this result
                slice_results = process_single_result(
                    result_path=result_path,
                    method_name=method_name,
                    dataset_name=dataset_name,
                    slice_name=slice_name,
                    output_dir=output_dir
                )
                
                dataset_results.extend(slice_results)
                all_results.extend(slice_results)
                files_processed += 1
                
                # Test mode: stop after 5 files
                if test_mode and files_processed >= 5:
                    print(f"\nTest mode: Stopping after processing {files_processed} files")
                    break
            
            # Early exit for test mode
            if test_mode and files_processed >= 5:
                break
        
        # Early exit for test mode
        if test_mode and files_processed >= 5:
            break
            
            # Calculate dataset summary for each clustering method
            if dataset_results:
                for clustering_method in CLUSTERING_METHODS:
                    method_results = [r for r in dataset_results if r['Clustering'] == clustering_method]
                    if method_results:
                        summary = calculate_dataset_summary(
                            metrics_list=method_results,
                            dataset_name=dataset_name,
                            method_name=method_name,
                            clustering_method=clustering_method
                        )
                        
                        if summary:
                            # Save dataset summary
                            summary_df = pd.DataFrame([summary])
                            summary_path = os.path.join(output_dir, f"{method_name}_{dataset_name}_{clustering_method}_summary.csv")
                            summary_df.to_csv(summary_path, index=False)
                            print(f"    Saved dataset summary: {summary_path}")
    
    print(f"\n--- Evaluation Complete ---")
    print(f"Total results processed: {len(all_results)}")
    print(f"Results saved to: {output_base_dir}")
    
    return all_results

if __name__ == "__main__":
    import sys
    
    # Check for test mode
    test_mode = '--test' in sys.argv
    
    # Run the evaluation
    results = evaluate_all_vertical_integration(test_mode=test_mode)
    
    print("\n" + "="*80)
    print("Evaluation completed! Check the Results/evaluation/vertical_integration directory.")
    if not test_mode:
        print("Next step: Run generate_final_results.py to create comprehensive summary tables.")
    else:
        print("Test mode completed successfully.")
    print("="*80)
