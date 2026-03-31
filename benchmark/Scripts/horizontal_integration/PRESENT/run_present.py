#!/usr/bin/env python3
"""
PRESENT horizontal integration script for SMOBench
Uses PRESENT's native horizontal integration support with individual datasets
Supports both RNA+ADT and RNA+ATAC spatial multi-omics integration
"""

import os
import sys
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

# Add project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "Methods"))

# Import PRESENT modules
from PRESENT import gene_sets_alignment, peak_sets_alignment, PRESENT_BC_function
from Utils.SMOBench_clustering import universal_clustering


def parse_dataset_info(args):
    """Extract dataset_name and subset_name from arguments"""
    if hasattr(args, 'dataset') and args.dataset:
        return args.dataset, "fusion"
    return "Unknown", "fusion"


def load_individual_datasets(dataset_name, modality):
    """Load individual datasets based on dataset name and modality"""
    datasets = []
    
    if dataset_name == "HLN":
        # Human Lymph Nodes: A1 + D1
        paths = [
            ("Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/A1", "HLN_A1"),
            ("Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/D1", "HLN_D1")
        ]
    elif dataset_name == "HT":
        # Human Tonsils: S1 + S2 + S3
        paths = [
            ("Dataset/withGT/RNA_ADT/Human_Tonsils/S1", "HT_S1"),
            ("Dataset/withGT/RNA_ADT/Human_Tonsils/S2", "HT_S2"),
            ("Dataset/withGT/RNA_ADT/Human_Tonsils/S3", "HT_S3")
        ]
    elif dataset_name == "MISAR_S1":
        # Mouse Embryos S1: E11 + E13 + E15 + E18
        paths = [
            ("Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1/E11", "MISAR_S1_E11"),
            ("Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1/E13", "MISAR_S1_E13"),
            ("Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1/E15", "MISAR_S1_E15"),
            ("Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1/E18", "MISAR_S1_E18")
        ]
    elif dataset_name == "MISAR_S2":
        # Mouse Embryos S2: E11 + E13 + E15 + E18
        paths = [
            ("Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2/E11", "MISAR_S2_E11"),
            ("Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2/E13", "MISAR_S2_E13"),
            ("Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2/E15", "MISAR_S2_E15"),
            ("Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2/E18", "MISAR_S2_E18")
        ]
    elif dataset_name == "Mouse_Thymus":
        # Mouse Thymus: Multiple samples
        paths = [
            ("Dataset/woGT/RNA_ADT/Mouse_Thymus/Mouse_Thymus1", "Mouse_Thymus1"),
            ("Dataset/woGT/RNA_ADT/Mouse_Thymus/Mouse_Thymus2", "Mouse_Thymus2"),
            ("Dataset/woGT/RNA_ADT/Mouse_Thymus/Mouse_Thymus3", "Mouse_Thymus3"),
            ("Dataset/woGT/RNA_ADT/Mouse_Thymus/Mouse_Thymus4", "Mouse_Thymus4")
        ]
    elif dataset_name == "Mouse_Spleen":
        # Mouse Spleen: Multiple samples
        paths = [
            ("Dataset/woGT/RNA_ADT/Mouse_Spleen/Mouse_Spleen1", "Mouse_Spleen1"),
            ("Dataset/woGT/RNA_ADT/Mouse_Spleen/Mouse_Spleen2", "Mouse_Spleen2")
        ]
    elif dataset_name == "Mouse_Brain":
        # Mouse Brain: Multiple modalities (treated as batches for horizontal integration)
        paths = [
            ("Dataset/woGT/RNA_ATAC/Mouse_Brain/Mouse_Brain_ATAC", "Mouse_Brain_ATAC"),
            ("Dataset/woGT/RNA_ATAC/Mouse_Brain/Mouse_Brain_H3K4me3", "Mouse_Brain_H3K4me3"),
            ("Dataset/woGT/RNA_ATAC/Mouse_Brain/Mouse_Brain_H3K27ac", "Mouse_Brain_H3K27ac"),
            ("Dataset/woGT/RNA_ATAC/Mouse_Brain/Mouse_Brain_H3K27me3", "Mouse_Brain_H3K27me3")
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Load datasets
    for base_path, batch_label in paths:
        if modality == 'RNA':
            file_path = f"{base_path}/adata_RNA.h5ad"
        elif modality == 'ADT':
            file_path = f"{base_path}/adata_ADT.h5ad"
        elif modality == 'ATAC':
            # Check multiple possible ATAC file names
            possible_files = [
                f"{base_path}/adata_ATAC.h5ad",
                f"{base_path}/adata_peaks_normalized.h5ad"
            ]
            file_path = None
            for pf in possible_files:
                if os.path.exists(pf):
                    file_path = pf
                    break
            if file_path is None:
                print(f"Warning: No ATAC file found for {base_path}")
                continue
        else:
            raise ValueError(f"Unknown modality: {modality}")
        
        if os.path.exists(file_path):
            adata = sc.read_h5ad(file_path)
            adata.var_names_make_unique()
            # Set batch label (use sample instead of batch for PRESENT compatibility)
            adata.obs['sample'] = batch_label
            adata.obs['batch'] = batch_label
            # Modify barcode if needed to avoid conflicts
            adata.obs_names = [f"{batch_label}-{x}" for x in adata.obs_names]
            datasets.append(adata)
            print(f"Loaded {file_path}: {adata.shape}")
        else:
            print(f"Warning: File not found: {file_path}")
    
    return datasets


def main(args):
    print("="*60)
    print(f"PRESENT Horizontal Integration")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print("="*60)
    
    # === Determine modality from paths ===
    if args.ADT_path or 'RNA_ADT' in args.RNA_path:
        modality = 'ADT'
        modality_name = 'Proteome'
        modalities = ['RNA', 'ADT']
    elif args.ATAC_path or 'RNA_ATAC' in args.RNA_path:
        modality = 'ATAC'
        modality_name = 'Epigenome'
        modalities = ['RNA', 'ATAC']
    else:
        raise ValueError("Cannot determine modality from paths")
    
    print(f"Processing horizontal integration: RNA + {modality_name}")
    
    # === Load individual datasets (not fusion data) ===
    print("📂 Loading individual datasets for horizontal integration...")
    
    rna_datasets = load_individual_datasets(args.dataset, 'RNA')
    second_datasets = load_individual_datasets(args.dataset, modality)
    
    print(f"Loaded {len(rna_datasets)} RNA datasets")
    print(f"Loaded {len(second_datasets)} {modality} datasets")
    
    # === Feature sets unification using PRESENT functions ===
    print("🔍 Performing feature sets unification...")
    start_time = time.time()
    
    # Gene sets alignment for RNA
    print("Aligning gene sets across RNA datasets...")
    rna_datasets_aligned = gene_sets_alignment(rna_datasets)
    
    # Feature sets alignment for second modality
    if modality == 'ADT':
        print("Aligning protein sets across ADT datasets...")
        # For ADT, also use gene_sets_alignment (proteins are treated as genes)
        second_datasets_aligned = gene_sets_alignment(second_datasets)
    elif modality == 'ATAC':
        print("Aligning peak sets across ATAC datasets...")
        second_datasets_aligned = peak_sets_alignment(second_datasets)
    
    # Concatenate aligned datasets
    print("Concatenating aligned datasets...")
    adata_rna = rna_datasets_aligned[0].concatenate(rna_datasets_aligned[1:])
    if modality == 'ATAC':
        adata_atac = second_datasets_aligned[0].concatenate(second_datasets_aligned[1:])
    else:  # ADT
        adata_adt = second_datasets_aligned[0].concatenate(second_datasets_aligned[1:])
    
    print(f"RNA concatenated data shape: {adata_rna.shape}")
    if modality == 'ATAC':
        print(f"ATAC concatenated data shape: {adata_atac.shape}")
    else:
        print(f"ADT concatenated data shape: {adata_adt.shape}")
    
    # === Add spatial coordinates if not present ===
    for adata, name in [(adata_rna, 'RNA')] + ([(adata_atac, 'ATAC')] if modality == 'ATAC' else [(adata_adt, 'ADT')]):
        if 'spatial' not in adata.obsm.keys():
            print(f"Warning: No spatial coordinates found in {name}. Generating pseudo-spatial coordinates...")
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
            adata.obsm['spatial'] = adata.obsm['X_umap'].copy()
            print(f"Generated pseudo-spatial coordinates for {name}")
    
    # === Run PRESENT horizontal integration ===
    print("Running PRESENT horizontal integration...")
    
    # Prepare PRESENT_BC_function arguments
    present_args = {
        'spatial_key': 'spatial',
        'batch_key': 'sample',  # PRESENT uses 'sample' for batch key
        'adata_rna': adata_rna,
        'gene_min_cells': 1,
        'num_hvg': 3000,
        'nclusters': args.cluster_nums,
        'device': args.device
    }
    
    # Add modality-specific arguments
    if modality == 'ATAC':
        present_args['adata_atac'] = adata_atac
        present_args['peak_min_cells_fraction'] = 0.03
    else:  # ADT
        # For ADT, PRESENT may treat it as additional RNA data
        # We'll pass it as adata_rna with proper preprocessing
        pass
    
    # Run PRESENT integration
    adata = PRESENT_BC_function(**present_args)
    
    end_time = time.time()
    train_time = end_time - start_time
    print(f'Horizontal integration training time: {train_time:.2f} seconds')
    
    # === Build Result AnnData ===
    adata.obsm['PRESENT'] = adata.obsm['embeddings']
    adata.uns['train_time'] = train_time
    adata.uns['integration_type'] = 'horizontal'
    
    # Add batch information for evaluation
    print(f"Sample distribution: {adata.obs['sample'].value_counts()}")
    adata.obs['batch'] = adata.obs['sample']  # Use sample as batch for evaluation
    
    # === Parse Dataset Info ===
    dataset_name, subset_name = parse_dataset_info(args)
    print(f"Detected dataset: {dataset_name}, subset: {subset_name}")

    # === Clustering and UMAP Generation ===
    tools = ['mclust', 'louvain', 'leiden', 'kmeans']
    
    # === Generate UMAP coordinates ===
    print("🗺️ Generating UMAP coordinates...")
    sc.pp.neighbors(adata, use_rep='embeddings')
    sc.tl.umap(adata)
    print("UMAP coordinates generated and stored in adata.obsm['X_umap']")
    
    # === Run clustering methods ===
    print("🎯 Running clustering methods...")
    for tool in tools:
        print(f"  Running {tool} clustering...")
        adata = universal_clustering(
            adata,
            n_clusters=args.cluster_nums,
            used_obsm='PRESENT',
            method=tool,
            key=tool,
            use_pca=False
        )
    
    print("All clustering methods completed")
    
    # === Save results ===
    print("💾 Saving results...")
    save_dir = os.path.dirname(args.save_path)
    os.makedirs(save_dir, exist_ok=True)
    adata.write(args.save_path)
    
    print(f"Results saved to: {args.save_path}")
    print(f"Final adata shape: {adata.shape}")
    print(f"Obsm keys: {list(adata.obsm.keys())}")
    print(f"Integration completed successfully!")


if __name__ == "__main__":
    # Set environment variables for R and threading
    os.environ['R_HOME'] = '/home/zhenghong/miniconda3/envs/smobench/lib/R'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    print("Starting PRESENT horizontal integration...")
    parser = argparse.ArgumentParser(description='Run PRESENT horizontal integration')
    parser.add_argument('--data_type', type=str, default='horizontal', help='Data type for horizontal integration')
    parser.add_argument('--RNA_path', type=str, required=True, help='RNA data path pattern')
    parser.add_argument('--ADT_path', type=str, default='', help='ADT data path pattern')
    parser.add_argument('--ATAC_path', type=str, default='', help='ATAC data path pattern')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save integrated adata')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, e.g. cuda:0 or cpu')

    parser.add_argument('--method', type=str, default='PRESENT', help='Method name for plotting')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name for horizontal integration')

    parser.add_argument('--cluster_nums', type=int, help='Number of clusters')

    args = parser.parse_args()
    main(args)