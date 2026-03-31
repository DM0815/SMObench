#!/usr/bin/env python3
"""
Script to create comprehensive Excel table of multimodal datasets and their label information
"""
import scanpy as sc
import pandas as pd
import os
import glob
from pathlib import Path

def analyze_multimodal_groups():
    """Analyze all h5ad files and group by dataset/experiment"""
    
    # Find all h5ad files
    h5ad_files = glob.glob("**/*.h5ad", recursive=True)
    h5ad_files.sort()
    
    # Group files by dataset
    datasets = {}
    
    for filepath in h5ad_files:
        path_parts = Path(filepath).parts
        
        # Extract dataset group and experiment
        if len(path_parts) >= 2:
            dataset_group = path_parts[0]
            if len(path_parts) >= 3:
                experiment = path_parts[1]
            else:
                experiment = "main"
            filename = Path(filepath).name
            
            # Extract modality from filename
            if "RNA" in filename:
                modality = "RNA"
            elif "ATAC" in filename:
                modality = "ATAC"
            elif "ADT" in filename:
                modality = "ADT"
            elif "peaks" in filename:
                modality = "Peaks"
            else:
                modality = "Unknown"
            
            # Create unique key for each experiment
            key = f"{dataset_group}_{experiment}"
            
            if key not in datasets:
                datasets[key] = {
                    'dataset_group': dataset_group,
                    'experiment': experiment,
                    'modalities': {},
                    'files': []
                }
            
            datasets[key]['files'].append(filepath)
            datasets[key]['modalities'][modality] = filepath
    
    return datasets

def get_file_label_info(filepath):
    """Get detailed label information from a single h5ad file"""
    try:
        adata = sc.read_h5ad(filepath)
        
        # Common label column names to look for
        label_keywords = ['label', 'cluster', 'celltype', 'cell_type', 'annotation', 
                         'group', 'class', 'type', 'leiden', 'louvain']
        
        label_columns = []
        for col in adata.obs.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in label_keywords):
                label_columns.append(col)
        
        label_details = {}
        for col in label_columns:
            unique_vals = list(adata.obs[col].unique())
            unique_vals.sort()  # Sort for consistency
            label_details[col] = {
                'unique_count': len(unique_vals),
                'values': unique_vals
            }
        
        return {
            'success': True,
            'n_cells': adata.n_obs,
            'n_features': adata.n_vars,
            'all_obs_columns': list(adata.obs.columns),
            'label_columns': label_columns,
            'label_details': label_details
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def create_excel_table():
    """Create comprehensive Excel table with all dataset information"""
    
    datasets = analyze_multimodal_groups()
    
    # Prepare data for Excel
    excel_data = []
    
    for key, dataset_info in datasets.items():
        dataset_group = dataset_info['dataset_group']
        experiment = dataset_info['experiment']
        modalities = dataset_info['modalities']
        
        # Get label information for each modality
        modality_labels = {}
        common_labels = None
        
        for modality, filepath in modalities.items():
            print(f"Processing {dataset_group}/{experiment} - {modality}")
            label_info = get_file_label_info(filepath)
            modality_labels[modality] = label_info
            
            if label_info['success'] and label_info['label_columns']:
                current_labels = set(label_info['label_columns'])
                if common_labels is None:
                    common_labels = current_labels
                else:
                    common_labels = common_labels.intersection(current_labels)
        
        # Determine if labels are consistent across modalities
        has_consistent_labels = common_labels is not None and len(common_labels) > 0
        
        # Create row for this dataset group/experiment
        row = {
            'Dataset_Group': dataset_group,
            'Experiment': experiment,
            'Dataset_Key': key,
            'Has_Labels': any(info.get('success', False) and info.get('label_columns', []) 
                            for info in modality_labels.values()),
            'Consistent_Labels_Across_Modalities': has_consistent_labels,
            'Common_Label_Columns': ', '.join(sorted(common_labels)) if common_labels else '',
            'Total_Modalities': len(modalities),
            'Available_Modalities': ', '.join(sorted(modalities.keys())),
        }
        
        # Add detailed information for each modality
        for modality in ['RNA', 'ATAC', 'ADT', 'Peaks']:
            if modality in modalities:
                info = modality_labels[modality]
                if info['success']:
                    row[f'{modality}_File'] = modalities[modality]
                    row[f'{modality}_Cells'] = info['n_cells']
                    row[f'{modality}_Features'] = info['n_features']
                    row[f'{modality}_Has_Labels'] = len(info['label_columns']) > 0
                    row[f'{modality}_Label_Columns'] = ', '.join(info['label_columns'])
                    
                    # Add details for each label column
                    for col in info['label_columns']:
                        if col in info['label_details']:
                            details = info['label_details'][col]
                            row[f'{modality}_{col}_Count'] = details['unique_count']
                            row[f'{modality}_{col}_Values'] = ', '.join(map(str, details['values'][:10]))  # First 10 values
                            if len(details['values']) > 10:
                                row[f'{modality}_{col}_Values'] += ', ...'
                else:
                    row[f'{modality}_File'] = modalities[modality]
                    row[f'{modality}_Error'] = info['error']
            else:
                row[f'{modality}_File'] = 'Not Available'
        
        excel_data.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(excel_data)
    
    # Sort by dataset group and experiment
    df = df.sort_values(['Dataset_Group', 'Experiment'])
    
    return df

def main():
    """Main function to create and save Excel table"""
    print("Analyzing multimodal datasets...")
    
    df = create_excel_table()
    
    # Save to Excel
    output_file = "multimodal_labels_analysis.xlsx"
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Main summary sheet
        df.to_excel(writer, sheet_name='Multimodal_Labels_Summary', index=False)
    
    print(f"\nExcel table saved to: {output_file}")
    print(f"Total datasets analyzed: {len(df)}")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"Datasets with labels: {df['Has_Labels'].sum()}")
    print(f"Datasets without labels: {(~df['Has_Labels']).sum()}")
    print(f"Datasets with consistent labels across modalities: {df['Consistent_Labels_Across_Modalities'].sum()}")
    
    # Show overview
    print(f"\nDataset groups overview:")
    group_summary = df.groupby('Dataset_Group').agg({
        'Has_Labels': 'sum',
        'Total_Modalities': 'mean',
        'Experiment': 'count'
    }).round(1)
    group_summary.columns = ['Experiments_with_Labels', 'Avg_Modalities', 'Total_Experiments']
    print(group_summary)

if __name__ == "__main__":
    main()