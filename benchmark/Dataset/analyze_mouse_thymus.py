#!/usr/bin/env python3
"""
Script to analyze all attributes in Mouse_Thymus h5ad files
"""
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_h5ad_attributes(filepath):
    """Analyze all attributes of an h5ad file"""
    try:
        print(f"\n{'='*60}")
        print(f"Analyzing: {filepath}")
        print(f"{'='*60}")
        
        adata = sc.read_h5ad(filepath)
        
        print(f"Shape: {adata.n_obs} cells × {adata.n_vars} features")
        
        # Basic info
        print(f"\nBasic Information:")
        print(f"- Observations (cells): {adata.n_obs}")
        print(f"- Variables (features): {adata.n_vars}")
        
        # Observation metadata (.obs)
        print(f"\nObservation metadata (.obs):")
        print(f"- Number of columns: {len(adata.obs.columns)}")
        if len(adata.obs.columns) > 0:
            print(f"- Column names: {list(adata.obs.columns)}")
            for col in adata.obs.columns:
                print(f"  * {col}:")
                print(f"    - Data type: {adata.obs[col].dtype}")
                print(f"    - Non-null count: {adata.obs[col].count()}")
                if adata.obs[col].dtype == 'object':
                    unique_vals = adata.obs[col].unique()
                    print(f"    - Unique values: {len(unique_vals)}")
                    if len(unique_vals) <= 20:
                        print(f"    - Values: {list(unique_vals)}")
                    else:
                        print(f"    - Sample values: {list(unique_vals[:10])}...")
                elif pd.api.types.is_numeric_dtype(adata.obs[col]):
                    print(f"    - Min: {adata.obs[col].min()}")
                    print(f"    - Max: {adata.obs[col].max()}")
                    print(f"    - Mean: {adata.obs[col].mean():.3f}")
        else:
            print("- No observation metadata found")
        
        # Variable metadata (.var)
        print(f"\nVariable metadata (.var):")
        print(f"- Number of columns: {len(adata.var.columns)}")
        if len(adata.var.columns) > 0:
            print(f"- Column names: {list(adata.var.columns)}")
            for col in adata.var.columns:
                print(f"  * {col}:")
                print(f"    - Data type: {adata.var[col].dtype}")
                print(f"    - Non-null count: {adata.var[col].count()}")
                if adata.var[col].dtype == 'object':
                    unique_vals = adata.var[col].unique()
                    print(f"    - Unique values: {len(unique_vals)}")
                    if len(unique_vals) <= 20:
                        print(f"    - Values: {list(unique_vals)}")
                    else:
                        print(f"    - Sample values: {list(unique_vals[:10])}...")
        else:
            print("- No variable metadata found")
        
        # Expression matrix (.X)
        print(f"\nExpression matrix (.X):")
        print(f"- Type: {type(adata.X)}")
        print(f"- Shape: {adata.X.shape}")
        print(f"- Data type: {adata.X.dtype}")
        
        # Check if sparse
        if hasattr(adata.X, 'format'):
            print(f"- Sparse format: {adata.X.format}")
            print(f"- Non-zero elements: {adata.X.nnz}")
            print(f"- Sparsity: {(1 - adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1])) * 100:.2f}%")
        
        # Statistics about expression values
        if hasattr(adata.X, 'data'):
            # Sparse matrix
            data = adata.X.data
        else:
            # Dense matrix
            data = adata.X.flatten()
        
        print(f"- Min value: {np.min(data)}")
        print(f"- Max value: {np.max(data)}")
        print(f"- Mean value: {np.mean(data):.6f}")
        print(f"- Std value: {np.std(data):.6f}")
        
        # Unstructured metadata (.uns)
        print(f"\nUnstructured metadata (.uns):")
        if adata.uns:
            print(f"- Keys: {list(adata.uns.keys())}")
            for key in adata.uns.keys():
                val = adata.uns[key]
                print(f"  * {key}: {type(val)} - {val}")
        else:
            print("- No unstructured metadata found")
        
        # Observation pairwise annotations (.obsp)
        print(f"\nObservation pairwise annotations (.obsp):")
        if adata.obsp:
            print(f"- Keys: {list(adata.obsp.keys())}")
            for key in adata.obsp.keys():
                print(f"  * {key}: {adata.obsp[key].shape} - {type(adata.obsp[key])}")
        else:
            print("- No observation pairwise annotations found")
        
        # Variable pairwise annotations (.varp)
        print(f"\nVariable pairwise annotations (.varp):")
        if adata.varp:
            print(f"- Keys: {list(adata.varp.keys())}")
            for key in adata.varp.keys():
                print(f"  * {key}: {adata.varp[key].shape} - {type(adata.varp[key])}")
        else:
            print("- No variable pairwise annotations found")
        
        # Multidimensional observations (.obsm)
        print(f"\nMultidimensional observations (.obsm):")
        if adata.obsm:
            print(f"- Keys: {list(adata.obsm.keys())}")
            for key in adata.obsm.keys():
                print(f"  * {key}: {adata.obsm[key].shape} - {type(adata.obsm[key])}")
        else:
            print("- No multidimensional observations found")
        
        # Multidimensional variables (.varm)
        print(f"\nMultidimensional variables (.varm):")
        if adata.varm:
            print(f"- Keys: {list(adata.varm.keys())}")
            for key in adata.varm.keys():
                print(f"  * {key}: {adata.varm[key].shape} - {type(adata.varm[key])}")
        else:
            print("- No multidimensional variables found")
        
        # Layers
        print(f"\nLayers:")
        if adata.layers:
            print(f"- Keys: {list(adata.layers.keys())}")
            for key in adata.layers.keys():
                layer_data = adata.layers[key]
                print(f"  * {key}: {layer_data.shape} - {type(layer_data)}")
                if hasattr(layer_data, 'format'):
                    print(f"    - Sparse format: {layer_data.format}")
                    print(f"    - Non-zero elements: {layer_data.nnz}")
        else:
            print("- No layers found")
        
        return True
        
    except Exception as e:
        print(f"ERROR reading {filepath}: {e}")
        return False

def main():
    """Main function to analyze all Mouse_Thymus h5ad files"""
    
    # Mouse_Thymus files from the target directory
    thymus_files = [
        "/home/users/nus/e1724738/_main/_private/NUS/_Proj1/SMOBench_Data/Mouse_Thymus/Mouse_Thymus1/adata_ADT.h5ad",
        "/home/users/nus/e1724738/_main/_private/NUS/_Proj1/SMOBench_Data/Mouse_Thymus/Mouse_Thymus1/adata_RNA.h5ad",
        "/home/users/nus/e1724738/_main/_private/NUS/_Proj1/SMOBench_Data/Mouse_Thymus/Mouse_Thymus2/adata_ADT.h5ad",
        "/home/users/nus/e1724738/_main/_private/NUS/_Proj1/SMOBench_Data/Mouse_Thymus/Mouse_Thymus2/adata_RNA.h5ad",
        "/home/users/nus/e1724738/_main/_private/NUS/_Proj1/SMOBench_Data/Mouse_Thymus/Mouse_Thymus3/adata_ADT.h5ad",
        "/home/users/nus/e1724738/_main/_private/NUS/_Proj1/SMOBench_Data/Mouse_Thymus/Mouse_Thymus3/adata_RNA.h5ad",
        "/home/users/nus/e1724738/_main/_private/NUS/_Proj1/SMOBench_Data/Mouse_Thymus/Mouse_Thymus4/adata_ADT.h5ad",
        "/home/users/nus/e1724738/_main/_private/NUS/_Proj1/SMOBench_Data/Mouse_Thymus/Mouse_Thymus4/adata_RNA.h5ad"
    ]
    
    print("Analyzing Mouse_Thymus h5ad files...")
    print(f"Total files to analyze: {len(thymus_files)}")
    
    successful = 0
    failed = 0
    
    for filepath in thymus_files:
        success = analyze_h5ad_attributes(filepath)
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully analyzed: {successful} files")
    print(f"Failed to analyze: {failed} files")

if __name__ == "__main__":
    main()