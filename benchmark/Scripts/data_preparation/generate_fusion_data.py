#!/usr/bin/env python3
"""
Generate tissue-specific modality fusion datasets for horizontal integration
"""

import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import optimized peak fusion utilities
from peak_fusion_utils import fuse_atac_data_optimized

# NEW: robust row-wise merger that preserves obs_names and obsm['spatial']
def _merge_filtered_adatas(filtered_adatas, dataset_names):
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import scanpy as sc

    X_list, obs_list, spatial_list = [], [], []
    for i, ad in enumerate(filtered_adatas):
        ad = ad.copy()
        # stable global ID: "<batch>:<barcode>"
        ad.obs.index = [f"{dataset_names[i]}:{x}" for x in ad.obs_names]
        # record
        X_list.append(ad.X)
        obs_df = ad.obs.copy()
        obs_df["batch"] = dataset_names[i]
        obs_list.append(obs_df)
        # collect spatial if present
        if "spatial" in ad.obsm:
            coords = np.asarray(ad.obsm["spatial"], dtype=float)
            spatial_list.append(coords)
        else:
            spatial_list.append(None)
        # writeback
        filtered_adatas[i] = ad

    # stack X
    merged_X = sp.vstack(X_list).tocsr() if hasattr(X_list[0], "tocsr") else np.vstack(X_list)
    # IMPORTANT: do NOT ignore_index -> keep the IDs we just set
    merged_obs = pd.concat(obs_list, axis=0)

    # use the first .var as template (all filtered share same features & order)
    merged_var = filtered_adatas[0].var.copy()
    for col in ["gene_ids", "feature_types", "genome"]:
        if col in merged_var.columns:
            merged_var[col] = merged_var[col].astype(str)

    merged = sc.AnnData(X=merged_X, obs=merged_obs, var=merged_var)

    # stitch spatial if every part has it
    if all(s is not None for s in spatial_list):
        merged.obsm["spatial"] = np.vstack(spatial_list)

    merged.obs_names_make_unique()
    merged.var_names_make_unique()
    return merged

def _sanity_check(adata, name=""):
    assert adata.X.shape[0] == adata.n_obs and adata.X.shape[1] == adata.n_vars, f"{name}: X shape mismatch"
    if "spatial" in adata.obsm:
        s = adata.obsm["spatial"]
        assert s.shape[0] == adata.n_obs and s.shape[1] == 2, f"{name}: spatial shape error"

def fuse_rna_data(adatas, dataset_names):
    """Fuse RNA data from multiple datasets"""
    print(f"  Fusing RNA data...")
    
    # Find common genes across datasets
    common_genes = set(adatas[0].var_names)
    for adata in adatas[1:]:
        common_genes = common_genes.intersection(set(adata.var_names))
    
    common_genes = sorted(list(common_genes))
    print(f"    Common genes: {len(common_genes)}")
    
    if len(common_genes) == 0:
        print(f"    No common genes found")
        return None
    
    # Filter common genes and merge datasets
    filtered_adatas = []
    for i, adata in enumerate(adatas):
        try:
            filtered_adata = adata[:, common_genes].copy()
            # Ensure unique observation and variable names
            filtered_adata.obs_names_make_unique()
            filtered_adata.var_names_make_unique()
            # OPTIONAL: align critical var columns across datasets
            for col in ["gene_ids", "feature_types", "genome"]:
                if col not in filtered_adata.var.columns:
                    filtered_adata.var[col] = pd.Series(index=filtered_adata.var_names, dtype="object")
            # Validate data integrity
            assert filtered_adata.shape[0] > 0 and filtered_adata.shape[1] > 0
            assert hasattr(filtered_adata.X, 'shape')
            filtered_adatas.append(filtered_adata)
        except Exception as e:
            print(f"    Dataset {i} filtering failed: {e}")
            return None
    
    try:
        merged_adata = _merge_filtered_adatas(filtered_adatas, dataset_names)
    except Exception as e:
        print(f"    Manual merge failed: {e}")
        return None
    print(f"    Merged: {merged_adata.n_obs} cells, {merged_adata.n_vars} genes")
    return merged_adata

def fuse_adt_data(adatas, dataset_names):
    """Fuse ADT protein data from multiple datasets"""
    print(f"  Fusing ADT data...")
    
    # First attempt direct protein matching
    common_proteins = set(adatas[0].var_names)
    for adata in adatas[1:]:
        common_proteins = common_proteins.intersection(set(adata.var_names))
    
    print(f"    Direct match common proteins: {len(common_proteins)}")
    
    # If no common proteins, try name normalization
    if len(common_proteins) == 0:
        print(f"    Attempting protein name normalization...")
        import re
        
        def normalize_protein_name(name):
            """Normalize protein names"""
            # Convert to lowercase
            name = name.lower()
            # Remove species prefixes
            name = re.sub(r'^(mouse|human|rat)[-_]', '', name)
            # Standardize separators
            name = re.sub(r'[-_]', '', name)
            # Remove common suffixes and spaces
            name = re.sub(r'\s+', '', name)
            return name
        
        # Create normalization mapping for each dataset
        normalized_mappings = []
        normalized_sets = []
        
        for i, adata in enumerate(adatas):
            mapping = {}
            normalized_set = set()
            for protein in adata.var_names:
                normalized = normalize_protein_name(protein)
                mapping[normalized] = protein
                normalized_set.add(normalized)
            normalized_mappings.append(mapping)
            normalized_sets.append(normalized_set)
        
        # Find common proteins after normalization
        common_normalized = normalized_sets[0]
        for normalized_set in normalized_sets[1:]:
            common_normalized = common_normalized.intersection(normalized_set)
        
        print(f"    Normalized common proteins: {len(common_normalized)}")
        
        if len(common_normalized) == 0:
            print(f"    No common proteins after normalization")
            return None
        
        # Filter data using normalized common proteins
        filtered_adatas = []
        for i, adata in enumerate(adatas):
            # Map back to original protein names
            selected_proteins = []
            for norm_protein in sorted(common_normalized):
                if norm_protein in normalized_mappings[i]:
                    selected_proteins.append(normalized_mappings[i][norm_protein])
            
            if len(selected_proteins) > 0:
                try:
                    filtered_adata = adata[:, selected_proteins].copy()
                    # Rename to normalized names
                    new_var_names = [normalize_protein_name(p) for p in selected_proteins]
                    filtered_adata.var_names = new_var_names
                    filtered_adata.obs_names_make_unique()
                    filtered_adata.var_names_make_unique()
                    # OPTIONAL: align critical var columns across datasets
                    for col in ["gene_ids", "feature_types", "genome"]:
                        if col not in filtered_adata.var.columns:
                            filtered_adata.var[col] = pd.Series(index=filtered_adata.var_names, dtype="object")
                    # Validate data integrity
                    assert filtered_adata.shape[0] > 0 and filtered_adata.shape[1] > 0
                    assert hasattr(filtered_adata.X, 'shape')
                    filtered_adatas.append(filtered_adata)
                except Exception as e:
                    print(f"    Dataset {i} filtering failed: {e}")
                    return None
        
        common_proteins = sorted(list(common_normalized))
        print(f"    Using normalized protein names: {len(common_proteins)} proteins")
    else:
        common_proteins = sorted(list(common_proteins))
        # Filter common proteins and merge
        filtered_adatas = []
        for i, adata in enumerate(adatas):
            try:
                filtered_adata = adata[:, common_proteins].copy()
                # Ensure unique names
                filtered_adata.obs_names_make_unique()
                filtered_adata.var_names_make_unique()
                # OPTIONAL: align critical var columns across datasets
                for col in ["gene_ids", "feature_types", "genome"]:
                    if col not in filtered_adata.var.columns:
                        filtered_adata.var[col] = pd.Series(index=filtered_adata.var_names, dtype="object")
                # Validate data integrity
                assert filtered_adata.shape[0] > 0 and filtered_adata.shape[1] > 0
                assert hasattr(filtered_adata.X, 'shape')
                filtered_adatas.append(filtered_adata)
            except Exception as e:
                print(f"    Dataset {i} filtering failed: {e}")
                return None
    
    if len(filtered_adatas) != len(adatas):
        print(f"    Some datasets failed filtering")
        return None
    
    # Manually merge data
    try:
        merged_adata = _merge_filtered_adatas(filtered_adatas, dataset_names)
    except Exception as e:
        print(f"    Manual merge failed: {e}")
        return None
    print(f"    Merged: {merged_adata.n_obs} cells, {merged_adata.n_vars} proteins")
    return merged_adata

def fuse_atac_data(adatas, dataset_names):
    """Fuse ATAC data using common peak strategy"""
    print(f"  Fusing ATAC data using common peaks...")
    
    # Find common peaks across datasets
    print(f"    Original peak counts: {[adata.n_vars for adata in adatas]}")
    
    common_peaks = set(adatas[0].var_names)
    for adata in adatas[1:]:
        common_peaks = common_peaks.intersection(set(adata.var_names))
    
    common_peaks = sorted(list(common_peaks))
    print(f"    Common peaks: {len(common_peaks)}")
    
    if len(common_peaks) == 0:
        print(f"    No common peaks found")
        return None
    
    # Filter common peaks and merge
    filtered_adatas = []
    for i, adata in enumerate(adatas):
        try:
            filtered_adata = adata[:, common_peaks].copy()
            # Ensure unique names
            filtered_adata.obs_names_make_unique()
            filtered_adata.var_names_make_unique()
            # Validate data integrity
            assert filtered_adata.shape[0] > 0 and filtered_adata.shape[1] > 0
            assert hasattr(filtered_adata.X, 'shape')
            filtered_adatas.append(filtered_adata)
            print(f"    {dataset_names[i]}: {filtered_adata.shape}")
        except Exception as e:
            print(f"    Dataset {i} filtering failed: {e}")
            return None
    
    # Manually merge data
    try:
        # Merge expression matrices
        import scipy.sparse as sp
        X_list = []
        obs_list = []
        
        for i, adata in enumerate(filtered_adatas):
            X_list.append(adata.X)
            # Add batch information
            obs_df = adata.obs.copy()
            obs_df['batch'] = dataset_names[i]
            obs_list.append(obs_df)
        
        # Vertically concatenate matrices
        merged_X = sp.vstack(X_list)
        # Convert to CSR format for h5ad compatibility
        if hasattr(merged_X, 'tocsr'):
            merged_X = merged_X.tocsr()
        merged_obs = pd.concat(obs_list, ignore_index=True)
        
        # Create merged AnnData object
        merged_adata = sc.AnnData(
            X=merged_X,
            obs=merged_obs,
            var=filtered_adatas[0].var.copy()
        )
        
    except Exception as e:
        print(f"    Manual merge failed: {e}")
        return None
    
    print(f"    Final merge: {merged_adata.n_obs} cells, {merged_adata.n_vars} peaks")
    return merged_adata

def generate_all_fusion():
    """Generate all tissue modality fusion datasets"""
    print("Generating tissue-specific modality fusion datasets")
    print("=" * 60)
    
    base_path = Path("/home/zhenghong/SMOBench-CLEAN/Dataset")
    
    # Define all data groups
    fusion_groups = {
        # Datasets with ground truth
        'fusionWithGT': {
            'HLN': {
                'data_type': 'RNA_ADT',
                'datasets': [
                    ('Human_Lymph_Nodes/A1', 'A1'),
                    ('Human_Lymph_Nodes/D1', 'D1')
                ]
            },
            'HT': {
                'data_type': 'RNA_ADT',
                'datasets': [
                    ('Human_Tonsils/S1', 'S1'),
                    ('Human_Tonsils/S2', 'S2'),
                    ('Human_Tonsils/S3', 'S3')
                ]
            },
            'ME_S1': {
                'data_type': 'RNA_ATAC',
                'datasets': [
                    ('Mouse_Embryos_S1/E11', 'E11'),
                    ('Mouse_Embryos_S1/E13', 'E13'),
                    ('Mouse_Embryos_S1/E15', 'E15'),
                    ('Mouse_Embryos_S1/E18', 'E18')
                ]
            },
            'ME_S2': {
                'data_type': 'RNA_ATAC',
                'datasets': [
                    ('Mouse_Embryos_S2/E11', 'E11'),
                    ('Mouse_Embryos_S2/E13', 'E13'),
                    ('Mouse_Embryos_S2/E15', 'E15'),
                    ('Mouse_Embryos_S2/E18', 'E18')
                ]
            }
        },
        # Datasets without ground truth
        'fusionWoGT': {
            'Mouse_Thymus': {
                'data_type': 'RNA_ADT',
                'datasets': [
                    ('Mouse_Thymus/Mouse_Thymus1', 'Thymus1'),
                    ('Mouse_Thymus/Mouse_Thymus2', 'Thymus2'),
                    ('Mouse_Thymus/Mouse_Thymus3', 'Thymus3'),
                    ('Mouse_Thymus/Mouse_Thymus4', 'Thymus4')
                ]
            },
            'Mouse_Spleen': {
                'data_type': 'RNA_ADT',
                'datasets': [
                    ('Mouse_Spleen/Mouse_Spleen1', 'Spleen1'),
                    ('Mouse_Spleen/Mouse_Spleen2', 'Spleen2')
                ]
            },
            'Mouse_Brain': {
                'data_type': 'RNA_ATAC',
                'datasets': [
                    ('Mouse_Brain/Mouse_Brain_ATAC', 'ATAC'),
                    ('Mouse_Brain/Mouse_Brain_H3K4me3', 'H3K4me3'),
                    ('Mouse_Brain/Mouse_Brain_H3K27ac', 'H3K27ac'),
                    ('Mouse_Brain/Mouse_Brain_H3K27me3', 'H3K27me3')
                ]
            }
        }
    }
    
    generated_files = []
    
    for fusion_type, groups in fusion_groups.items():
        print(f"\nGenerating {fusion_type} datasets")
        
        for group_name, group_info in groups.items():
            print(f"\nProcessing tissue: {group_name}")
            print(f"Data type: {group_info['data_type']}")
            
            # Determine modalities
            if group_info['data_type'] == 'RNA_ADT':
                modalities = ['RNA', 'ADT']
            elif group_info['data_type'] == 'RNA_ATAC':
                modalities = ['RNA', 'ATAC']
            else:
                print(f"  Unsupported data type: {group_info['data_type']}")
                continue
            
            # Generate fusion data for each modality
            for modality in modalities:
                print(f"\n  Processing modality: {modality}")
                
                # Load modality data from all datasets
                adatas = []
                dataset_names = []
                
                for dataset_path, dataset_id in group_info['datasets']:
                    # Build file paths
                    if fusion_type == 'fusionWithGT':
                        base_dir = base_path / "withGT"
                    else:
                        base_dir = base_path / "woGT"
                    
                    # Handle special file naming rules
                    if modality == 'ATAC' and fusion_type == 'fusionWoGT' and 'Mouse_Brain' in dataset_path:
                        file_path = base_dir / group_info['data_type'] / dataset_path / "adata_peaks_normalized.h5ad"
                    else:
                        file_path = base_dir / group_info['data_type'] / dataset_path / f"adata_{modality}.h5ad"
                    
                    if file_path.exists():
                        try:
                            adata = sc.read_h5ad(file_path)
                            # Ensure unique gene names
                            adata.var_names_make_unique()
                            adata.obs['batch'] = f'{group_name}_{dataset_id}'
                            adatas.append(adata)
                            dataset_names.append(f'{group_name}_{dataset_id}')
                            print(f"    {dataset_id}: {adata.n_obs} cells, {adata.n_vars} features")
                        except Exception as e:
                            print(f"    {dataset_id}: loading failed - {e}")
                    else:
                        print(f"    {dataset_id}: file not found - {file_path}")
                
                if len(adatas) < 2:
                    print(f"    {modality} insufficient data for fusion (requires ≥2 datasets)")
                    continue
                
                # Select fusion strategy by modality type
                if modality == 'RNA':
                    merged_adata = fuse_rna_data(adatas, dataset_names)
                elif modality == 'ADT':
                    merged_adata = fuse_adt_data(adatas, dataset_names)
                elif modality == 'ATAC':
                    merged_adata = fuse_atac_data_optimized(adatas, dataset_names)
                else:
                    print(f"    Unsupported modality: {modality}")
                    continue
                
                if merged_adata is not None:
                    # Save fusion file
                    output_dir = base_path / fusion_type / group_info['data_type']
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_file = output_dir / f"{group_name}_Fusion_{modality}.h5ad"
                    _sanity_check(merged_adata, name=f"{group_name}_{modality}")
                    merged_adata.write(output_file)
                    print(f"    Saved: {output_file}")
                    generated_files.append(f"{fusion_type}/{group_info['data_type']}/{group_name}_Fusion_{modality}.h5ad")
                else:
                    print(f"    {modality} fusion failed")
    
    # Generate summary report
    print(f"\nGenerating summary report")
    print("=" * 60)
    
    withgt_count = len([f for f in generated_files if f.startswith('fusionWithGT')])
    wogt_count = len([f for f in generated_files if f.startswith('fusionWoGT')])
    
    print(f"fusionWithGT: {withgt_count} files")
    print(f"fusionWoGT: {wogt_count} files")
    print(f"Total: {len(generated_files)} files")
    
    print(f"\nGenerated files:")
    for file_path in sorted(generated_files):
        print(f"  {file_path}")
    
    print(f"\nModality fusion dataset generation completed!")

if __name__ == "__main__":
    generate_all_fusion()
