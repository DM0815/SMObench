import os
import scanpy as sc
import argparse
import time
import json
from datetime import datetime
import sys
import re
import numpy as np

# Set CUDA workspace config for CuBLAS operations
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Add project root directory to module search path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

# Add SpaMosaic to path
spamosaic_path = os.path.join(project_root, "Methods/SpaMosaic")
sys.path.append(spamosaic_path)

import spamosaic
from spamosaic.framework import SpaMosaic
import spamosaic.utils as utls
from spamosaic.preprocessing import RNA_preprocess, ADT_preprocess, Epigenome_preprocess
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
            # Set batch label
            adata.obs['src'] = batch_label
            # Modify barcode if needed to avoid conflicts
            adata.obs_names = [f"{batch_label}-{x}" for x in adata.obs_names]
            datasets.append(adata)
            print(f"Loaded {file_path}: {adata.shape}")
        else:
            print(f"Warning: File not found: {file_path}")
    
    return datasets


def main(args):
    total_start_time = time.time()
    print("="*60)
    print(f"SpaMosaic Horizontal Integration")
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

    # === Determine if fusion mode (subsampled data with batch labels) ===
    fusion_mode = (args.RNA_path and os.path.isfile(args.RNA_path) and
                   args.RNA_path.endswith('.h5ad') and
                   args.dataset.startswith('scalability_'))

    if fusion_mode:
        # Fusion mode: load subsampled h5ad files directly, split by batch
        print("Loading fusion data in scalability mode...")
        adata_rna_full = sc.read_h5ad(args.RNA_path)
        adata_rna_full.var_names_make_unique()
        # Ensure float32 for numba/scanpy compatibility
        if hasattr(adata_rna_full.X, 'dtype') and adata_rna_full.X.dtype != np.float32:
            adata_rna_full.X = adata_rna_full.X.astype(np.float32)
        other_path = args.ADT_path if modality == 'ADT' else args.ATAC_path
        adata_other_full = sc.read_h5ad(other_path)
        adata_other_full.var_names_make_unique()
        if hasattr(adata_other_full.X, 'dtype') and adata_other_full.X.dtype != np.float32:
            adata_other_full.X = adata_other_full.X.astype(np.float32)

        # Copy spatial coordinates from RNA to other modality if missing
        # (fusion ATAC data lacks obsm['spatial'] but shares cells with RNA)
        if 'spatial' in adata_rna_full.obsm and 'spatial' not in adata_other_full.obsm:
            adata_other_full.obsm['spatial'] = adata_rna_full.obsm['spatial'].copy()
            print("Copied spatial coordinates from RNA to second modality")

        # Split by batch to create per-batch datasets
        if 'batch' not in adata_rna_full.obs.columns:
            n = adata_rna_full.n_obs
            adata_rna_full.obs['batch'] = (['batch_1'] * (n // 2) +
                                           ['batch_2'] * (n - n // 2))
            adata_other_full.obs['batch'] = adata_rna_full.obs['batch'].copy()

        rna_datasets = []
        second_datasets = []
        for batch_label in adata_rna_full.obs['batch'].unique():
            mask = adata_rna_full.obs['batch'] == batch_label
            rna_sub = adata_rna_full[mask].copy()
            other_sub = adata_other_full[mask].copy()
            rna_sub.obs['src'] = batch_label
            other_sub.obs['src'] = batch_label
            rna_sub.obs_names = [f"{batch_label}-{x}" for x in rna_sub.obs_names]
            other_sub.obs_names = [f"{batch_label}-{x}" for x in other_sub.obs_names]
            rna_datasets.append(rna_sub)
            second_datasets.append(other_sub)
            print(f"  Batch '{batch_label}': {rna_sub.n_obs} cells")
    else:
        # Standard mode: load individual datasets
        print("Loading individual datasets for horizontal integration...")
        rna_datasets = load_individual_datasets(args.dataset, 'RNA')
        second_datasets = load_individual_datasets(args.dataset, modality)

    print(f"Loaded {len(rna_datasets)} RNA datasets")
    print(f"Loaded {len(second_datasets)} {modality} datasets")

    # === Find shared features ===
    print("Finding shared features across datasets...")

    # Find shared genes across RNA datasets
    shared_genes = rna_datasets[0].var_names
    for adata in rna_datasets[1:]:
        shared_genes = shared_genes.intersection(adata.var_names)
    print(f"Shared genes: {len(shared_genes)}")

    # Find shared features for second modality
    shared_features_2nd = second_datasets[0].var_names
    for adata in second_datasets[1:]:
        shared_features_2nd = shared_features_2nd.intersection(adata.var_names)
    print(f"Shared {modality} features: {len(shared_features_2nd)}")

    # Filter datasets to shared features
    rna_datasets = [adata[:, shared_genes].copy() for adata in rna_datasets]
    second_datasets = [adata[:, shared_features_2nd].copy() for adata in second_datasets]

    # === Prepare input dictionary for SpaMosaic ===
    print("Preparing input for SpaMosaic horizontal integration...")

    input_dict = {
        'rna': rna_datasets
    }

    if modality == 'ADT':
        input_dict['protein'] = second_datasets
    elif modality == 'ATAC':
        input_dict['epigenome'] = second_datasets

    input_key = 'dimred_bc'
    
    # === Preprocessing ===
    print("Preprocessing for horizontal integration...")
    start_time = time.time()

    # Pre-filter: remove genes with <1 cell expressing to avoid loess singularity
    for ad in input_dict['rna']:
        sc.pp.filter_genes(ad, min_cells=1)

    # RNA preprocessing with batch correction
    print("Processing RNA modality...")
    RNA_preprocess(
        input_dict['rna'],
        batch_corr=True,  # Enable batch correction for horizontal integration
        n_hvg=3000,       # Increased for better representation
        batch_key='src', 
        key=input_key
    )
    
    # Second modality preprocessing
    if modality == 'ADT':
        print("Processing ADT modality...")
        ADT_preprocess(
            input_dict['protein'], 
            batch_corr=True,  # Enable batch correction
            batch_key='src', 
            key=input_key
        )
    elif modality == 'ATAC':
        print("Processing ATAC modality...")
        Epigenome_preprocess(
            input_dict['epigenome'], 
            batch_corr=True,  # Enable batch correction
            n_peak=5000,      # Highly variable peaks
            batch_key='src', 
            key=input_key
        )
    
    # === Training SpaMosaic model ===
    print("Training SpaMosaic for horizontal integration...")
    
    # Enhanced parameters for horizontal integration
    model = SpaMosaic(
        modBatch_dict=input_dict, 
        input_key=input_key,
        batch_key='src', 
        intra_knns=15,        # Increased for better spatial connectivity
        inter_knn_base=15,    # Increased for better batch integration
        w_g=0.9,             # Increased spatial weight
        smooth_input=True,    # Enable input smoothing for horizontal integration
        smooth_L=2,          # LGCN layers for smoothing
        inter_auto_knn=True,  # Auto balance kNN for different batch sizes
        rmv_outlier=True,     # Remove MNN outliers
        contamination=0.1,    # 10% outlier removal
        seed=args.seed,
        device=args.device
    )
    
    # Training with enhanced parameters
    if modality == 'ADT':
        w_rec_g = 0.1  # Low value for protein modality
    else:  # ATAC
        w_rec_g = 1.0  # High value for epigenome modality
    
    model.train(
        net='wlgcn', 
        lr=0.001, 
        n_epochs=100,  # Increased epochs for horizontal integration
        w_rec_g=w_rec_g
    )
    
    end_time = time.time()
    train_time = end_time - start_time
    print(f'Horizontal integration training time: {train_time:.2f} seconds')
    
    # === Inference ===
    print("📊 Inferring embeddings...")
    
    ad_embs = model.infer_emb(input_dict, emb_key='emb', final_latent_key='merged_emb')
    ad_mosaic = sc.concat(ad_embs)
    ad_mosaic = utls.get_umap(ad_mosaic, use_reps=['merged_emb'])
    
    # === Build Result AnnData ===
    adata = ad_mosaic.copy()
    adata.obsm['SpaMosaic'] = adata.obsm['merged_emb']
    adata.uns['train_time'] = train_time
    adata.uns['integration_type'] = 'horizontal'
    
    # Add batch information for evaluation
    print(f"Batch distribution: {adata.obs['src'].value_counts()}")
    adata.obs['batch'] = adata.obs['src']  # Use src as batch for evaluation
    
    # === Parse Dataset Info ===
    dataset_name, subset_name = parse_dataset_info(args)
    print(f"Detected dataset: {dataset_name}, subset: {subset_name}")

    # === Clustering and UMAP Generation ===
    # tools = ['mclust', 'louvain', 'leiden', 'kmeans']
    tools = ['mclust', 'kmeans']
    
    # === Generate UMAP coordinates (already done by SpaMosaic) ===
    print("🗺️ UMAP coordinates already generated by SpaMosaic")
    adata.obsm['X_umap'] = adata.obsm['merged_emb_umap']
    
    # === Run clustering methods ===
    print("🎯 Running clustering methods...")
    for tool in tools:
        print(f"  Running {tool} clustering...")
        adata = universal_clustering(
            adata,
            n_clusters=args.cluster_nums,
            used_obsm='SpaMosaic',
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

    # === Save timing info ===
    total_time = time.time() - total_start_time
    dataset_name, _ = parse_dataset_info(args)
    modality_str = "RNA_ADT" if args.ADT_path else "RNA_ATAC"
    timing_info = {
        "method": "SpaMosaic",
        "dataset": dataset_name,
        "integration_type": "horizontal",
        "modality": modality_str,
        "n_cells": adata.n_obs,
        "embedding_shape": list(adata.obsm["SpaMosaic"].shape),
        "training_time_s": round(train_time, 2),
        "total_time_s": round(total_time, 2),
        "device": args.device,
        "seed": args.seed,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    timing_path = args.save_path.replace(".h5ad", "_timing_info.json")
    with open(timing_path, "w") as f:
        json.dump(timing_info, f, indent=2)
    print(f"Timing info saved to: {timing_path}")


if __name__ == "__main__":
    # Set environment variables for R and threading
    os.environ['R_HOME'] = '/home/users/nus/e1724738/miniconda3/envs/_Proj1_1/lib/R'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    print("Starting SpaMosaic horizontal integration...")
    parser = argparse.ArgumentParser(description='Run SpaMosaic horizontal integration')
    parser.add_argument('--data_type', type=str, default='horizontal', help='Data type for horizontal integration')
    parser.add_argument('--RNA_path', type=str, required=True, help='RNA data path pattern')
    parser.add_argument('--ADT_path', type=str, default='', help='ADT data path pattern')
    parser.add_argument('--ATAC_path', type=str, default='', help='ATAC data path pattern')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save integrated adata')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, e.g. cuda:0 or cpu')

    parser.add_argument('--method', type=str, default='SpaMosaic', help='Method name for plotting')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name for horizontal integration')

    parser.add_argument('--cluster_nums', type=int, help='Number of clusters')

    args = parser.parse_args()
    main(args)