#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run SpaMV-3M triple-modality integration (RNA + ADT + ATAC)

SpaMV accepts List[AnnData] of any length. For 3M, we pass
datasets=[adata_rna, adata_adt, adata_atac].
"""

import os
import sys
import re
import time
import argparse
import torch
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

# === Path setup ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

spamv_path = os.path.join(project_root, "Methods/SpaMV")
sys.path.append(spamv_path)

from SpaMV.spamv import SpaMV
from SpaMV.utils import preprocess_dc
from Utils.SMOBench_clustering import universal_clustering


def parse_dataset_info(args):
    if hasattr(args, 'dataset') and args.dataset:
        parts = args.dataset.strip('/').split('/')
        if len(parts) == 2:
            return parts[0], parts[1]
        elif len(parts) == 1:
            return parts[0], "Unknown"
    match = re.search(r'Dataset/([^/]+)/([^/]+)/([^/]+)/adata_RNA\.h5ad', args.RNA_path)
    if match:
        return match.group(2), match.group(3)
    return "Unknown", "Unknown"


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # === Load all three modalities ===
    print("Loading RNA + ADT + ATAC data...")
    adata_rna = sc.read_h5ad(args.RNA_path)
    adata_adt = sc.read_h5ad(args.ADT_path)
    adata_atac = sc.read_h5ad(args.ATAC_path)
    for a in [adata_rna, adata_adt, adata_atac]:
        a.var_names_make_unique()

    datasets = [adata_rna, adata_adt, adata_atac]
    omics_names = ['Transcriptome', 'Proteome', 'Epigenome']
    print(f"Processing 3M integration: {' + '.join(omics_names)}")

    # === Preprocessing using SpaMV's built-in function ===
    print("Preprocessing datasets using SpaMV preprocess_dc...")
    datasets = preprocess_dc(
        datasets,
        omics_names,
        prune=True,
        min_cells=10,
        min_genes=200,
        hvg=True,
        n_top_genes=3000,
        normalization=True,
        target_sum=1e4,
        log1p=True,
        scale=True  # ADT present, use scaling
    )

    # === Initialize and Train SpaMV Model ===
    print("Initializing SpaMV model...")
    model = SpaMV(
        adatas=datasets,
        interpretable=False,
        device=device,
        random_seed=args.seed,
        max_epochs_stage1=400,
        max_epochs_stage2=400,
        early_stopping=True,
        patience=200
    )

    print("Training SpaMV model...")
    start_time = time.time()
    model.train()
    train_time = time.time() - start_time
    print(f'Training time: {train_time:.2f}s')

    # === Get Embeddings ===
    print("Getting embeddings...")
    embeddings = model.get_embedding()

    # === Build Result AnnData ===
    adata = datasets[0].copy()
    adata.obsm['SpaMV'] = embeddings
    adata.uns['train_time'] = train_time

    # === Clean embeddings ===
    embeddings_clean = adata.obsm['SpaMV'].copy()
    if np.any(~np.isfinite(embeddings_clean)):
        print("Warning: Found infinite or NaN values in embeddings. Cleaning...")
        embeddings_clean[np.isinf(embeddings_clean)] = np.sign(embeddings_clean[np.isinf(embeddings_clean)]) * 1e10
        embeddings_clean[np.isnan(embeddings_clean)] = 0
        adata.obsm['SpaMV'] = embeddings_clean
    else:
        print("Embeddings are clean (no inf/NaN values)")

    # === Parse Dataset Info ===
    dataset_name, subset_name = parse_dataset_info(args)
    print(f"Detected dataset: {dataset_name}, subset: {subset_name}")

    # === Plot Save Path ===
    method_name = "SpaMV_3M"
    plot_dir = os.path.join("Results/plot", method_name, dataset_name, subset_name)
    os.makedirs(plot_dir, exist_ok=True)

    # === Clustering and Visualization ===
    tools = ['mclust', 'louvain', 'leiden', 'kmeans']

    sc.pp.neighbors(adata, use_rep='SpaMV', n_neighbors=30)
    sc.tl.umap(adata)

    for tool in tools:
        adata = universal_clustering(
            adata,
            n_clusters=args.cluster_nums,
            used_obsm='SpaMV',
            method=tool,
            key=tool,
            use_pca=False
        )

        if 'spatial' in adata.obsm.keys():
            adata.obsm['spatial'][:, 1] = -1 * adata.obsm['spatial'][:, 1]

        fig, ax_list = plt.subplots(1, 2, figsize=(7, 3))
        sc.pl.umap(adata, color=tool, ax=ax_list[0], title=f'{method_name}-{tool}', s=20, show=False)
        if 'spatial' in adata.obsm.keys():
            sc.pl.embedding(adata, basis='spatial', color=tool, ax=ax_list[1], title=f'{method_name}-{tool}', s=20, show=False)
        else:
            sc.pl.umap(adata, color=tool, ax=ax_list[1], title=f'{method_name}-{tool} (no spatial)', s=20, show=False)

        plt.tight_layout(w_pad=0.3)
        plt.savefig(os.path.join(plot_dir, f'clustering_{tool}_umap_spatial.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # === Clean cross-version anndata incompatible fields ===
    if 'log1p' in adata.uns and 'base' in adata.uns['log1p'] and adata.uns['log1p']['base'] is None:
        del adata.uns['log1p']['base']

    # === Save AnnData ===
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    adata.write(args.save_path)
    print(adata)
    print(f'Saved results to {args.save_path}')


if __name__ == "__main__":
    os.environ['R_HOME'] = '/home/users/nus/e1724738/miniconda3/envs/_Proj1_1/lib/R'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    print("Starting SpaMV-3M integration...")
    parser = argparse.ArgumentParser(description='Run SpaMV 3M integration (RNA+ADT+ATAC)')
    parser.add_argument('--data_type', type=str, default='simulation')
    parser.add_argument('--RNA_path', type=str, required=True)
    parser.add_argument('--ADT_path', type=str, required=True)
    parser.add_argument('--ATAC_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--method', type=str, default='SpaMV_3M')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--cluster_nums', type=int, required=True)

    args = parser.parse_args()
    main(args)
