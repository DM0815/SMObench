"""
Generate Final Evaluation Results for SMOBench Vertical Integration
Creates comprehensive summary tables and rankings for all methods and datasets
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

# Dataset classification for RNA_ADT vs RNA_ATAC categorization
DATASET_TYPES = {
    'HLN_A1': 'RNA_ADT',
    'HLN_D1': 'RNA_ADT',
    'HLN': 'RNA_ADT',  # Aggregated HLN
    'HT_S1': 'RNA_ADT',
    'HT_S2': 'RNA_ADT', 
    'HT_S3': 'RNA_ADT',
    'HT': 'RNA_ADT',   # Aggregated HT
    'MISAR_S1': 'RNA_ATAC',
    'MISAR_S2': 'RNA_ATAC',
    'Mouse_Thymus': 'RNA_ADT',
    'Mouse_Spleen': 'RNA_ADT', 
    'Mouse_Brain': 'RNA_ATAC'
}

# Metrics categorization for score calculation
SC_METRICS = ['Moran Index', 'Geary C']

# BioC metrics vary by ground truth availability
BIOC_METRICS_WITHGT = ['ARI', 'NMI', 'asw_celltype', 'graph_clisi']
BIOC_METRICS_WOGT = ['Davies-Bouldin Index', 'Silhouette Coefficient', 'Calinski-Harabaz Index']

# Metrics where lower values are better
LOWER_IS_BETTER = ['Davies-Bouldin Index', 'Geary C']

def normalize_metric_value(value, metric_name, all_values_for_metric):
    """
    Normalize a metric value to 0-1 scale
    ONLY NORMALIZES DBI (Davies-Bouldin Index) and CHI (Calinski-Harabasz Index)
    Other metrics return original values
    
    Parameters:
    -----------
    value : float
        The metric value to normalize
    metric_name : str
        Name of the metric (for direction determination)
    all_values_for_metric : list
        All values for this metric across methods/datasets
        
    Returns:
    --------
    float : Normalized value between 0 and 1 for DBI/CHI, original value for others
    """
    
    # Only normalize DBI and CHI, return original values for all other metrics
    metrics_to_normalize = ['Davies-Bouldin Index', 'Calinski-Harabaz Index']
    
    if metric_name not in metrics_to_normalize:
        return value  # Return original value without normalization
    
    if len(all_values_for_metric) == 0 or np.isnan(value):
        return 0.5
    
    min_val = min(all_values_for_metric)
    max_val = max(all_values_for_metric)
    
    # Handle case where all values are the same
    if max_val == min_val:
        return 0.5
    
    if metric_name in LOWER_IS_BETTER:
        # For metrics where lower is better, invert the normalization
        normalized = (max_val - value) / (max_val - min_val)
    else:
        # For metrics where higher is better
        normalized = (value - min_val) / (max_val - min_val)
    
    return max(0.0, min(1.0, normalized))

def process_individual_results(results_dir):
    """
    Process individual slice results and aggregate by dataset
    
    Parameters:
    -----------
    results_dir : str
        Path to the evaluation results directory
        
    Returns:
    --------
    dict : Aggregated results by clustering method
    """
    
    print("Processing individual evaluation results...")
    
    # Find all individual result files
    withgt_files = glob.glob(os.path.join(results_dir, "**/*_withGT.csv"), recursive=True)
    wogt_files = glob.glob(os.path.join(results_dir, "**/*_woGT.csv"), recursive=True)
    
    all_files = withgt_files + wogt_files
    
    if not all_files:
        print(f"No evaluation result files found in {results_dir}")
        return {}
    
    print(f"Found {len(all_files)} evaluation result files")
    
    # Group results by clustering method
    results_by_clustering = {
        'leiden': [],
        'louvain': [],
        'kmeans': [],
        'mclust': []
    }
    
    # Process each file
    for file_path in all_files:
        try:
            # Parse filename to extract metadata
            filename = os.path.basename(file_path)
            parts = filename.replace('.csv', '').split('_')
            
            if len(parts) < 5:
                print(f"Warning: Cannot parse filename {filename}")
                continue
            
            method_name = parts[0]
            
            # Handle dataset names like MISAR_S1, MISAR_S2, HLN_A1, HT_S1, etc.
            if parts[1] in ['MISAR', 'HLN', 'HT'] and len(parts) >= 6:
                dataset_name = f"{parts[1]}_{parts[2]}"  # MISAR_S1, HLN_A1, HT_S1, etc.
                slice_name = parts[3]
                clustering_method = parts[4]
                gt_status = parts[5]
            elif parts[1] in ['Mouse'] and len(parts) >= 6:
                dataset_name = f"{parts[1]}_{parts[2]}"  # Mouse_Brain, Mouse_Spleen, Mouse_Thymus
                slice_name = parts[3]
                clustering_method = parts[4]
                gt_status = parts[5]
            else:
                # Standard format: METHOD_DATASET_SLICE_CLUSTERING_GT
                dataset_name = parts[1]
                slice_name = parts[2] 
                clustering_method = parts[3]
                gt_status = parts[4]  # 'withGT' or 'woGT'
            
            # Read the evaluation results
            df = pd.read_csv(file_path)
            
            # Convert to dictionary
            metrics_dict = dict(zip(df['Metric'], df['Value']))
            
            # Add metadata
            metrics_dict.update({
                'Method': method_name,
                'Dataset': dataset_name,
                'Slice': slice_name,
                'Clustering': clustering_method,
                'GT_Available': (gt_status == 'withGT'),
                'Dataset_Type': DATASET_TYPES.get(dataset_name, 'Unknown')
            })
            
            # Add to appropriate clustering group
            if clustering_method in results_by_clustering:
                results_by_clustering[clustering_method].append(metrics_dict)
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return results_by_clustering

def aggregate_by_dataset(results_list):
    """
    Aggregate slice-level results to dataset level
    
    Parameters:
    -----------
    results_list : list
        List of dictionaries containing slice-level results
        
    Returns:
    --------
    pd.DataFrame : Dataset-level aggregated results
    """
    
    if not results_list:
        return pd.DataFrame()
    
    # Group by method and dataset
    grouped_data = {}
    
    for result in results_list:
        key = (result['Method'], result['Dataset'])
        if key not in grouped_data:
            grouped_data[key] = []
        grouped_data[key].append(result)
    
    # Aggregate each group
    aggregated_results = []
    
    for (method, dataset), slice_results in grouped_data.items():
        if not slice_results:
            continue
        
        # Determine metrics to aggregate based on GT availability
        has_gt = slice_results[0]['GT_Available']
        dataset_type = DATASET_TYPES.get(dataset, 'Unknown')  # Look up aggregated dataset type
        clustering = slice_results[0]['Clustering']
        
        # Calculate averages for each metric
        aggregated = {
            'Method': method,
            'Dataset': dataset,
            'Dataset_Type': dataset_type,
            'Clustering': clustering,
            'GT_Available': has_gt,
            'Num_Slices': len(slice_results)
        }
        
        # Aggregate SC metrics
        for metric in SC_METRICS:
            values = [r.get(metric, np.nan) for r in slice_results if metric in r]
            if values and not all(np.isnan(values)):
                aggregated[metric] = np.nanmean(values)
            else:
                aggregated[metric] = np.nan
        
        # Aggregate BioC metrics based on GT availability
        if has_gt:
            bioc_metrics = BIOC_METRICS_WITHGT
        else:
            bioc_metrics = BIOC_METRICS_WOGT
        
        for metric in bioc_metrics:
            values = [r.get(metric, np.nan) for r in slice_results if metric in r]
            if values and not all(np.isnan(values)):
                aggregated[metric] = np.nanmean(values)
            else:
                aggregated[metric] = np.nan
        
        aggregated_results.append(aggregated)
    
    return pd.DataFrame(aggregated_results)

def calculate_normalized_scores(df):
    """
    Calculate normalized SC, BioC, and Total scores
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with aggregated dataset results
        
    Returns:
    --------
    pd.DataFrame : DataFrame with added normalized scores
    """
    
    if df.empty:
        return df
    
    df = df.copy()
    
    # Separate withGT and woGT datasets for normalization
    withgt_mask = df['GT_Available'] == True
    wogt_mask = df['GT_Available'] == False
    
    # Calculate scores for each group separately
    for mask, suffix in [(withgt_mask, 'withGT'), (wogt_mask, 'woGT')]:
        if not mask.any():
            continue
        
        subset_df = df[mask]
        
        # Determine metrics for this group
        if suffix == 'withGT':
            bioc_metrics = BIOC_METRICS_WITHGT
        else:
            bioc_metrics = BIOC_METRICS_WOGT
        
        # Process SC metrics (keep original values, no normalization for scoring)
        sc_scores = []
        for metric in SC_METRICS:
            if metric in subset_df.columns:
                # SC metrics don't need normalization, use original values
                sc_scores.append(metric)
        
        # Calculate SC score using only Moran Index
        if 'Moran Index' in subset_df.columns:
            df.loc[mask, 'SC_Score'] = df.loc[mask, 'Moran Index']
        
        # Process BioC metrics
        bioc_scores = []
        bioc_normalized_scores = []
        for metric in bioc_metrics:
            if metric in subset_df.columns:
                # Only normalize DBI and CHI
                if metric in ['Davies-Bouldin Index', 'Calinski-Harabaz Index']:
                    all_values = subset_df[metric].dropna().tolist()
                    if all_values:
                        normalized_values = [normalize_metric_value(val, metric, all_values) 
                                           for val in subset_df[metric]]
                        df.loc[mask, f'{metric}_normalized'] = normalized_values
                        bioc_normalized_scores.append(f'{metric}_normalized')
                else:
                    # Other BioC metrics use original values
                    bioc_scores.append(metric)
        
        # Calculate BioC score using mix of normalized (DBI/CHI) and original values
        all_bioc_scores = bioc_scores + bioc_normalized_scores
        if all_bioc_scores:
            df.loc[mask, 'BioC_Score'] = df.loc[mask, all_bioc_scores].mean(axis=1)
        
        # Calculate Total score
        if 'SC_Score' in df.columns and 'BioC_Score' in df.columns:
            df.loc[mask, 'Total_Score'] = (df.loc[mask, 'SC_Score'] + df.loc[mask, 'BioC_Score']) / 2
    
    return df

def create_summary_tables(df, clustering_method):
    """
    Create summary tables for a specific clustering method
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with normalized scores
    clustering_method : str
        Name of the clustering method
        
    Returns:
    --------
    tuple : (rna_adt_table, rna_atac_table, comprehensive_table) summary tables
    """
    
    # Filter for this clustering method
    method_df = df[df['Clustering'] == clustering_method].copy()
    
    if method_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Create pivot tables for RNA_ADT and RNA_ATAC
    rna_adt_df = method_df[method_df['Dataset_Type'] == 'RNA_ADT']
    rna_atac_df = method_df[method_df['Dataset_Type'] == 'RNA_ATAC']
    
    # Create summary tables
    def create_pivot_table(data_df, data_type):
        if data_df.empty:
            return pd.DataFrame()
        
        # Create pivot table with methods as rows and datasets as columns
        pivot_df = data_df.pivot_table(
            index='Method',
            columns='Dataset', 
            values='Total_Score',
            aggfunc='mean'
        ).round(3)
        
        # Add overall average
        pivot_df['Average'] = pivot_df.mean(axis=1).round(3)
        
        # Sort by average score
        pivot_df = pivot_df.sort_values('Average', ascending=False)
        
        return pivot_df
    
    rna_adt_table = create_pivot_table(rna_adt_df, 'RNA_ADT')
    rna_atac_table = create_pivot_table(rna_atac_df, 'RNA_ATAC')
    
    # Create comprehensive table with all 7 datasets
    comprehensive_table = create_pivot_table(method_df, 'All')
    
    return rna_adt_table, rna_atac_table, comprehensive_table

def save_results(results_dict, output_dir):
    """
    Save all results to files
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing all results organized by clustering method
    output_dir : str
        Output directory for saving results
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results for each clustering method
    for clustering_method, data in results_dict.items():
        if 'detailed_results' in data:
            detailed_path = os.path.join(output_dir, f"detailed_results_{clustering_method}.csv")
            data['detailed_results'].to_csv(detailed_path, index=False)
            print(f"Saved detailed results: {detailed_path}")
        
        if 'rna_adt_summary' in data and not data['rna_adt_summary'].empty:
            rna_adt_path = os.path.join(output_dir, f"summary_RNA_ADT_{clustering_method}.csv")
            data['rna_adt_summary'].to_csv(rna_adt_path)
            print(f"Saved RNA_ADT summary: {rna_adt_path}")
        
        if 'rna_atac_summary' in data and not data['rna_atac_summary'].empty:
            rna_atac_path = os.path.join(output_dir, f"summary_RNA_ATAC_{clustering_method}.csv")
            data['rna_atac_summary'].to_csv(rna_atac_path)
            print(f"Saved RNA_ATAC summary: {rna_atac_path}")
        
        if 'comprehensive_summary' in data and not data['comprehensive_summary'].empty:
            comprehensive_path = os.path.join(output_dir, f"summary_{clustering_method}.csv")
            data['comprehensive_summary'].to_csv(comprehensive_path)
            print(f"Saved comprehensive summary: {comprehensive_path}")

def generate_comprehensive_evaluation():
    """
    Generate comprehensive evaluation results for vertical integration
    """
    
    print("="*80)
    print("SMOBench Vertical Integration - Final Results Generation")
    print("="*80)
    
    # Setup directories
    results_dir = "/home/zhenghong/SMOBench-CLEAN/Results/evaluation/vertical_integration"
    output_dir = "/home/zhenghong/SMOBench-CLEAN/Results/evaluation/vertical_integration/final_results"
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        print("Please run eval_adata.py first to generate evaluation results.")
        return
    
    # Process individual results
    results_by_clustering = process_individual_results(results_dir)
    
    if not any(results_by_clustering.values()):
        print("Error: No evaluation results found.")
        return
    
    # Process each clustering method
    final_results = {}
    
    for clustering_method in ['leiden', 'louvain', 'kmeans', 'mclust']:
        if clustering_method not in results_by_clustering or not results_by_clustering[clustering_method]:
            print(f"Warning: No results found for {clustering_method}")
            continue
        
        print(f"\n--- Processing {clustering_method} results ---")
        
        # Aggregate by dataset
        aggregated_df = aggregate_by_dataset(results_by_clustering[clustering_method])
        
        if aggregated_df.empty:
            print(f"Warning: No aggregated results for {clustering_method}")
            continue
        
        # Calculate normalized scores
        scored_df = calculate_normalized_scores(aggregated_df)
        
        # Create summary tables
        rna_adt_table, rna_atac_table, comprehensive_table = create_summary_tables(scored_df, clustering_method)
        
        # Store results
        final_results[clustering_method] = {
            'detailed_results': scored_df,
            'rna_adt_summary': rna_adt_table,
            'rna_atac_summary': rna_atac_table,
            'comprehensive_summary': comprehensive_table
        }
        
        print(f"  Processed {len(scored_df)} dataset-level results")
        if not rna_adt_table.empty:
            print(f"  RNA_ADT summary: {rna_adt_table.shape[0]} methods × {rna_adt_table.shape[1]-1} datasets")
        if not rna_atac_table.empty:
            print(f"  RNA_ATAC summary: {rna_atac_table.shape[0]} methods × {rna_atac_table.shape[1]-1} datasets")
        if not comprehensive_table.empty:
            print(f"  Comprehensive summary: {comprehensive_table.shape[0]} methods × {comprehensive_table.shape[1]-1} datasets")
    
    # Save all results
    save_results(final_results, output_dir)
    
    # Display summary tables
    print("\n" + "="*80)
    print("FINAL SUMMARY TABLES")
    print("="*80)
    
    for clustering_method in ['leiden', 'louvain', 'kmeans', 'mclust']:
        if clustering_method not in final_results:
            continue
        
        print(f"\n{clustering_method.upper()} CLUSTERING RESULTS:")
        print("-" * 50)
        
        if not final_results[clustering_method]['comprehensive_summary'].empty:
            print(f"\nComprehensive Summary (All 7 Datasets):")
            print(final_results[clustering_method]['comprehensive_summary'].to_string())
        
        if not final_results[clustering_method]['rna_adt_summary'].empty:
            print(f"\nRNA + ADT Integration:")
            print(final_results[clustering_method]['rna_adt_summary'].to_string())
        
        if not final_results[clustering_method]['rna_atac_summary'].empty:
            print(f"\nRNA + ATAC Integration:")
            print(final_results[clustering_method]['rna_atac_summary'].to_string())
    
    print(f"\n" + "="*80)
    print("Results Generation Complete!")
    print(f"All results saved to: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    generate_comprehensive_evaluation()