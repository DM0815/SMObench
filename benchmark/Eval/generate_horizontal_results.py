#!/usr/bin/env python3
"""
Generate final results for horizontal integration evaluation
"""

import os
import glob
import pandas as pd
import numpy as np
import argparse
from collections import defaultdict, OrderedDict

# Import functions from vertical integration
from generate_final_results import (
    normalize_metric_value,
    save_results,
    DATASET_TYPES,
)

# Horizontal integration specific constants
HORIZONTAL_SC_METRICS = ['Moran Index']
HORIZONTAL_BER_METRICS = ['kBET', 'KNN_connectivity', 'bASW', 'iLISI', 'PCR']
HORIZONTAL_BIOC_METRICS_WITHGT = ['ARI', 'NMI', 'asw_celltype', 'graph_clisi']
HORIZONTAL_BIOC_METRICS_WOGT = ['Davies-Bouldin Index', 'Silhouette Coefficient', 'Calinski-Harabaz Index']

LOWER_IS_BETTER = ['Davies-Bouldin Index', 'Geary C']

METHOD_DATASET_COMPATIBILITY = {
    'SpatialGlue': ['HLN', 'HT', 'MISAR_S1', 'MISAR_S2', 'Mouse_Thymus', 'Mouse_Spleen', 'Mouse_Brain'],
    'SpaMosaic': ['HLN', 'HT', 'MISAR_S1', 'MISAR_S2', 'Mouse_Thymus', 'Mouse_Spleen', 'Mouse_Brain'],
    'PRESENT': ['HLN', 'HT', 'MISAR_S1', 'MISAR_S2', 'Mouse_Thymus', 'Mouse_Spleen', 'Mouse_Brain'],
    'COSMOS': ['HLN', 'HT', 'MISAR_S1', 'MISAR_S2', 'Mouse_Thymus', 'Mouse_Spleen', 'Mouse_Brain'],
    'SpaMV': ['HLN', 'HT', 'MISAR_S1', 'MISAR_S2', 'Mouse_Thymus', 'Mouse_Spleen'],
    'CANDIES': ['HLN', 'HT', 'MISAR_S1', 'MISAR_S2', 'Mouse_Spleen'],
    'PRAGA': ['HLN', 'HT', 'Mouse_Spleen'],
    'SpaMultiVAE': ['HLN', 'HT', 'Mouse_Thymus', 'Mouse_Spleen'],
    'SpaFusion': ['HLN', 'HT', 'Mouse_Thymus', 'Mouse_Spleen'],
    'SpaBalance': ['HLN', 'HT', 'Mouse_Thymus', 'Mouse_Spleen'],
    'SpaMI': ['HLN', 'HT', 'Mouse_Thymus', 'Mouse_Spleen'],
}

ALL_METHODS = sorted(METHOD_DATASET_COMPATIBILITY.keys())
DATASET_GROUPS = OrderedDict([
    ("RNA_ADT_withGT", ["HLN", "HT"]),
    ("RNA_ADT_woGT", ["Mouse_Thymus", "Mouse_Spleen"]),
    ("RNA_ATAC_withGT", ["MISAR_S1", "MISAR_S2"]),
    ("RNA_ATAC_woGT", ["Mouse_Brain"]),
])
DATASET_ORDER = [dataset for group in DATASET_GROUPS.values() for dataset in group]

def calculate_horizontal_normalized_scores(df):
    """
    Calculate normalized scores for horizontal integration with three dimensions
    
    Parameters:
    -----------
    df : pd.DataFrame
        Aggregated results DataFrame
    
    Returns:
    --------
    pd.DataFrame : DataFrame with normalized scores and dimension scores
    """
    df = df.copy()
    
    # Group by clustering and GT availability for normalization
    for (clustering, has_gt), subset_df in df.groupby(['Clustering', 'GT_Available']):
        mask = (df['Clustering'] == clustering) & (df['GT_Available'] == has_gt)
        
        if len(subset_df) < 2:
            continue  # Skip normalization if only one data point
        
        # Process SC metrics (no normalization needed - already in good range)
        for metric in HORIZONTAL_SC_METRICS:
            if metric in subset_df.columns:
                df.loc[mask, 'SC_Score'] = df.loc[mask, metric]
        
        # Process BioC metrics
        if has_gt:
            bioc_metrics = HORIZONTAL_BIOC_METRICS_WITHGT
        else:
            bioc_metrics = HORIZONTAL_BIOC_METRICS_WOGT
        
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
            df.loc[mask, 'BVC_Score'] = df.loc[mask, all_bioc_scores].mean(axis=1)
        
        # Process BER metrics (no normalization needed for most)
        ber_values = []
        for metric in HORIZONTAL_BER_METRICS:
            if metric in subset_df.columns:
                ber_values.append(metric)
        
        if ber_values:
            df.loc[mask, 'BER_Score'] = df.loc[mask, ber_values].mean(axis=1)
        
        # Calculate Final Score (three dimensions)
        score_columns = []
        if 'SC_Score' in df.columns:
            score_columns.append('SC_Score')
        if 'BVC_Score' in df.columns:
            score_columns.append('BVC_Score')
        if 'BER_Score' in df.columns:
            score_columns.append('BER_Score')
        
        if score_columns:
            df.loc[mask, 'Final_Score'] = df.loc[mask, score_columns].mean(axis=1)
    
    return df

def create_horizontal_summary_tables(df, methods_order):
    """
    Create grouped summary tables for horizontal integration results.
    Returns an OrderedDict mapping group names to summary DataFrames.
    """
    tables = OrderedDict()

    if df.empty:
        for group_name, datasets in DATASET_GROUPS.items():
            tables[group_name] = pd.DataFrame(index=methods_order, columns=datasets + ["Average"], dtype=float)
        tables["Comprehensive"] = pd.DataFrame(index=methods_order, columns=DATASET_ORDER + ["Average"], dtype=float)
        return tables

    df = df[df["Dataset"].isin(DATASET_ORDER)].copy()
    methods_present = sorted(set(df["Method"].unique()))
    full_methods = sorted(set(methods_order) | set(methods_present))

    def build_table(subset_df, datasets):
        if subset_df.empty:
            pivot = pd.DataFrame(index=full_methods, columns=datasets, dtype=float)
        else:
            pivot = subset_df.pivot_table(
                index="Method",
                columns="Dataset",
                values="Final_Score",
                aggfunc="first"
            )
            pivot = pivot.reindex(index=full_methods)
            pivot = pivot.reindex(columns=datasets)
        pivot["Average"] = pivot.mean(axis=1, skipna=True)
        return pivot.round(3)

    for group_name, datasets in DATASET_GROUPS.items():
        subset = df[df["Dataset"].isin(datasets)]
        tables[group_name] = build_table(subset, datasets)

    tables["Comprehensive"] = build_table(df, DATASET_ORDER)
    return tables

def process_horizontal_results(results_dir):
    """
    Process horizontal integration results
    
    Parameters:
    -----------
    results_dir : str
        Directory containing horizontal integration results
    
    Returns:
    --------
    dict : Results organized by clustering method
    """
    
    print(f"Processing horizontal integration results from: {results_dir}")
    
    # Find all result files
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
            
            if len(parts) < 4:
                print(f"Warning: Cannot parse filename {filename}")
                continue
            
            # Extract metadata from filename
            method_name = parts[0]
            
            if len(parts) < 4:
                print(f"Warning: Cannot parse filename {filename}")
                continue
            
            if "horizontal" not in parts:
                continue
            horizontal_idx = parts.index("horizontal")
            if horizontal_idx <= 1:
                print(f"Warning: Cannot determine dataset from {filename}")
                continue
            
            dataset_parts = parts[1:horizontal_idx]
            dataset_name = "_".join(dataset_parts)
            
            if 'horizontal' in parts:
                horizontal_idx = parts.index('horizontal')
                if len(parts) > horizontal_idx + 1:
                    clustering_method = parts[horizontal_idx + 1]
                    gt_status = parts[-1]  # withGT or woGT
                else:
                    continue
            else:
                continue
            
            if clustering_method not in results_by_clustering:
                continue
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            if df.empty:
                continue
            
            # Horizontal integration CSV files have Metric,Value format - need to transpose
            if 'Metric' in df.columns and 'Value' in df.columns:
                # Convert key-value pairs to single row dictionary
                metrics_dict = dict(zip(df['Metric'], df['Value']))
                
                # Add metadata
                metrics_dict.update({
                    'Method': method_name,
                    'Dataset': dataset_name,
                    'Clustering': clustering_method,
                    'GT_Available': gt_status == 'withGT',
                    'Slice': 'horizontal',  # For horizontal integration
                    'Dataset_Type': 'RNA_ADT' if dataset_name in ['HLN', 'HT', 'Mouse_Thymus', 'Mouse_Spleen'] else 'RNA_ATAC'
                })
                
                results_by_clustering[clustering_method].append(metrics_dict)
            else:
                # Fallback for other formats
                for _, row in df.iterrows():
                    result_dict = row.to_dict()
                    result_dict.update({
                        'Method': method_name,
                        'Dataset': dataset_name,
                        'Clustering': clustering_method,
                        'GT_Available': gt_status == 'withGT',
                        'Slice': 'horizontal',  # For horizontal integration
                        'Dataset_Type': 'RNA_ADT' if dataset_name in ['HLN', 'HT', 'Mouse_Thymus', 'Mouse_Spleen'] else 'RNA_ATAC'
                    })
                    
                    results_by_clustering[clustering_method].append(result_dict)
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Convert lists to DataFrames
    for clustering_method in results_by_clustering:
        if results_by_clustering[clustering_method]:
            results_by_clustering[clustering_method] = pd.DataFrame(results_by_clustering[clustering_method])
        else:
            results_by_clustering[clustering_method] = pd.DataFrame()
        
        print(f"  {clustering_method}: {len(results_by_clustering[clustering_method])} results")
    
    return results_by_clustering

def aggregate_horizontal_by_dataset(results_list):
    """
    Aggregate slice-level horizontal results to dataset level,
    preserving SC/BioC metrics and BER components.
    """
    if not results_list:
        return pd.DataFrame()

    grouped = defaultdict(list)
    for result in results_list:
        key = (result['Method'], result['Dataset'])
        grouped[key].append(result)

    aggregated_rows = []
    for (method, dataset), slices in grouped.items():
        if not slices:
            continue
        has_gt = slices[0].get('GT_Available', False)
        dataset_type = DATASET_TYPES.get(dataset, 'Unknown')
        clustering = slices[0].get('Clustering', 'leiden')

        aggregated = {
            'Method': method,
            'Dataset': dataset,
            'Dataset_Type': dataset_type,
            'Clustering': clustering,
            'GT_Available': has_gt,
            'Num_Slices': len(slices),
        }

        def add_metric(metric_name):
            values = [slice_result.get(metric_name, np.nan) for slice_result in slices]
            values = [v for v in values if pd.notna(v)]
            aggregated[metric_name] = np.nanmean(values) if values else np.nan

        for metric in HORIZONTAL_SC_METRICS:
            add_metric(metric)

        bioc_list = HORIZONTAL_BIOC_METRICS_WITHGT if has_gt else HORIZONTAL_BIOC_METRICS_WOGT
        for metric in bioc_list:
            add_metric(metric)

        for metric in HORIZONTAL_BER_METRICS:
            add_metric(metric)

        aggregated_rows.append(aggregated)

    return pd.DataFrame(aggregated_rows)

def save_horizontal_results(final_results, output_dir):
    """
    Save horizontal integration results to files
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    for clustering_method, results in final_results.items():
        if not results:
            continue
        
        clustering_dir = os.path.join(output_dir, clustering_method)
        os.makedirs(clustering_dir, exist_ok=True)
        
        # Save detailed results
        detailed_file = os.path.join(clustering_dir, f"detailed_results_{clustering_method}.csv")
        results['detailed_results'].to_csv(detailed_file, index=False)
        print(f"Saved detailed results: {detailed_file}")
        
        for group_name, table in results['group_tables'].items():
            file_path = os.path.join(clustering_dir, f"{group_name}_summary_{clustering_method}.csv")
            table.to_csv(file_path)
            print(f"Saved {group_name} summary: {file_path}")

def generate_horizontal_evaluation():
    """
    Generate comprehensive evaluation results for horizontal integration
    """
    
    print("="*80)
    print("SMOBench Horizontal Integration - Final Results Generation")
    print("="*80)
    
    # Setup directories
    results_dir = "/home/zhenghong/SMOBench-CLEAN/Results/evaluation/horizontal_integration"
    output_dir = "/home/zhenghong/SMOBench-CLEAN/Results/evaluation/horizontal_integration/final_results"
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        print("Please run eval_horizontal_integration.py first to generate evaluation results.")
        return
    
    # Process individual results
    results_by_clustering = process_horizontal_results(results_dir)
    
    if not any(not df.empty for df in results_by_clustering.values()):
        print("Error: No evaluation results found.")
        return
    
    # Process each clustering method
    final_results = {}
    
    for clustering_method in ['leiden', 'louvain', 'kmeans', 'mclust']:
        if clustering_method not in results_by_clustering or results_by_clustering[clustering_method].empty:
            print(f"Warning: No results found for {clustering_method}")
            continue
        
        print(f"\nProcessing {clustering_method} clustering results...")
        
        # Aggregate by dataset (horizontal integration typically has one result per dataset)
        # Convert DataFrame to list of dictionaries for aggregation
        results_list = results_by_clustering[clustering_method].to_dict('records')
        aggregated_df = aggregate_horizontal_by_dataset(results_list)
        
        if aggregated_df.empty:
            print(f"Warning: No aggregated results for {clustering_method}")
            continue
        
        # Calculate normalized scores for horizontal integration
        scored_df = calculate_horizontal_normalized_scores(aggregated_df)
        
        # Create summary tables
        group_tables = create_horizontal_summary_tables(scored_df, ALL_METHODS)
        
        # Store results
        final_results[clustering_method] = {
            'detailed_results': scored_df,
            'group_tables': group_tables,
        }
        
        print(f"  Processed {len(scored_df)} dataset-level results")
        for group_name, table in group_tables.items():
            if not table.empty:
                print(f"  {group_name}: {table.shape[0]} methods × {table.shape[1]-1} datasets")
    
    # Save all results
    save_horizontal_results(final_results, output_dir)
    
    # Display summary tables
    print("\n" + "="*80)
    print("HORIZONTAL INTEGRATION - FINAL SUMMARY TABLES")
    print("="*80)
    
    for clustering_method in ['leiden', 'louvain', 'kmeans', 'mclust']:
        if clustering_method not in final_results:
            continue
        
        print(f"\n{clustering_method.upper()} CLUSTERING RESULTS:")
        print("-" * 50)
        
        for group_name, table in final_results[clustering_method]['group_tables'].items():
            print(f"\n{group_name}:")
            if table.empty:
                print("No data available.")
            else:
                print(table.to_string())
    
    print(f"\n" + "="*80)
    print("Horizontal Integration Results Generation Complete!")
    print(f"All results saved to: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    generate_horizontal_evaluation()
