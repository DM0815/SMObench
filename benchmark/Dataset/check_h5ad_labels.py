#!/usr/bin/env python3
"""
Script to check all h5ad files for label information
"""
import scanpy as sc
import pandas as pd
import os
import glob

def check_h5ad_labels(filepath):
    """Check a single h5ad file for label information"""
    try:
        adata = sc.read_h5ad(filepath)
        
        result = {
            'file': filepath,
            'n_obs': adata.n_obs,
            'n_vars': adata.n_vars,
            'obs_columns': list(adata.obs.columns),
            'has_potential_labels': False,
            'potential_label_columns': []
        }
        
        # Common label column names to look for
        label_keywords = ['label', 'cluster', 'celltype', 'cell_type', 'annotation', 
                         'group', 'class', 'type', 'leiden', 'louvain']
        
        for col in adata.obs.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in label_keywords):
                result['potential_label_columns'].append(col)
                result['has_potential_labels'] = True
        
        # Check for unique values in potential label columns
        if result['potential_label_columns']:
            for col in result['potential_label_columns']:
                unique_vals = adata.obs[col].nunique()
                print(f"  {col}: {unique_vals} unique values")
                if unique_vals < 50:  # Show actual values if not too many
                    print(f"    Values: {list(adata.obs[col].unique())}")
        
        return result
        
    except Exception as e:
        return {
            'file': filepath,
            'error': str(e)
        }

def main():
    # Find all h5ad files
    h5ad_files = glob.glob("**/*.h5ad", recursive=True)
    h5ad_files.sort()
    
    print(f"Found {len(h5ad_files)} h5ad files\n")
    
    results = []
    
    for filepath in h5ad_files:
        print(f"Checking: {filepath}")
        result = check_h5ad_labels(filepath)
        results.append(result)
        
        if 'error' in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Shape: {result['n_obs']} cells × {result['n_vars']} features")
            print(f"  Obs columns: {result['obs_columns']}")
            print(f"  Has potential labels: {result['has_potential_labels']}")
            if result['potential_label_columns']:
                print(f"  Potential label columns: {result['potential_label_columns']}")
        print()
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    files_with_labels = [r for r in results if r.get('has_potential_labels', False)]
    files_without_labels = [r for r in results if not r.get('has_potential_labels', False) and 'error' not in r]
    files_with_errors = [r for r in results if 'error' in r]
    
    print(f"Files with potential labels: {len(files_with_labels)}")
    print(f"Files without labels: {len(files_without_labels)}")
    print(f"Files with errors: {len(files_with_errors)}")
    
    if files_with_labels:
        print("\nFiles WITH potential labels:")
        for r in files_with_labels:
            print(f"  {r['file']}: {r['potential_label_columns']}")
    
    if files_without_labels:
        print("\nFiles WITHOUT labels:")
        for r in files_without_labels:
            print(f"  {r['file']}")
    
    if files_with_errors:
        print("\nFiles with errors:")
        for r in files_with_errors:
            print(f"  {r['file']}: {r['error']}")

if __name__ == "__main__":
    main()