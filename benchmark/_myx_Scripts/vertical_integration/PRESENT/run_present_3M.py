#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run PRESENT-3M triple-modality integration (RNA + ADT + ATAC)

PRESENT_function already supports adata_rna, adata_atac, adata_adt simultaneously.
This script simply loads all three and passes them in.
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

present_path = os.path.join(project_root, "Methods/PRESENT")
sys.path.append(present_path)

from PRESENT import PRESENT_function
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

    device_id = 0
    if 'cuda:' in args.device:
        device_id = int(args.device.split(':')[1])

    # === Load all three modalities ===
    print("Loading RNA + ADT + ATAC data...")
    adata_rna = sc.read_h5ad(args.RNA_path)
    adata_adt = sc.read_h5ad(args.ADT_path)
    adata_atac = sc.read_h5ad(args.ATAC_path)
    for a in [adata_rna, adata_adt, adata_atac]:
        a.var_names_make_unique()

    # === Align cells across all three modalities ===
    common_obs = adata_rna.obs_names.intersection(adata_adt.obs_names).intersection(adata_atac.obs_names)
    print(f"Common cells: {len(common_obs)} (RNA={adata_rna.n_obs}, ADT={adata_adt.n_obs}, ATAC={adata_atac.n_obs})")
    adata_rna = adata_rna[common_obs].copy()
    adata_adt = adata_adt[common_obs].copy()
    adata_atac = adata_atac[common_obs].copy()

    # === Run PRESENT integration with all three modalities ===
    print("Running PRESENT 3M integration (RNA + ADT + ATAC)...")
    start_time = time.time()

    adata_integrated = PRESENT_function(
        spatial_key="spatial",
        batch_key=None,
        adata_rna=adata_rna,
        adata_atac=adata_atac,
        adata_adt=adata_adt,
        rdata_rna=None,
        rdata_rna_anno=None,
        rdata_atac=None,
        rdata_atac_anno=None,
        rdata_adt=None,
        rdata_adt_anno=None,
        gene_min_cells=1,
        peak_min_cells_fraction=0.03,
        protein_min_cells=1,
        num_hvg=3000,
        nclusters=args.cluster_nums,
        d_lat=50,
        k_neighbors=6,
        intra_neighbors=6,
        inter_neighbors=6,
        epochs=300,
        lr=1e-3,
        batch_size=320,
        device="cuda",
        device_id=device_id
    )

    train_time = time.time() - start_time
    print(f'Training time: {train_time:.2f}s')

    # === Build Result AnnData ===
    adata = adata_integrated.copy()
    adata.obsm['PRESENT'] = adata_integrated.obsm['embeddings']
    adata.uns['train_time'] = train_time

    # === Clean embeddings ===
    embeddings_clean = adata.obsm['PRESENT'].copy()
    if np.any(~np.isfinite(embeddings_clean)):
        print("Warning: Found infinite or NaN values in embeddings. Cleaning...")
        embeddings_clean[np.isinf(embeddings_clean)] = np.sign(embeddings_clean[np.isinf(embeddings_clean)]) * 1e10
        embeddings_clean[np.isnan(embeddings_clean)] = 0
        adata.obsm['PRESENT'] = embeddings_clean
    else:
        print("Embeddings are clean (no inf/NaN values)")

    # === Parse Dataset Info ===
    dataset_name, subset_name = parse_dataset_info(args)
    print(f"Detected dataset: {dataset_name}, subset: {subset_name}")

    # === Plot Save Path ===
    method_name = "PRESENT_3M"
    plot_dir = os.path.join("Results/plot", method_name, dataset_name, subset_name)
    os.makedirs(plot_dir, exist_ok=True)

    # === Clustering and Visualization ===
    tools = ['mclust', 'louvain', 'leiden', 'kmeans']

    sc.pp.neighbors(adata, use_rep='PRESENT', n_neighbors=30)
    sc.tl.umap(adata)

    for tool in tools:
        adata = universal_clustering(
            adata,
            n_clusters=args.cluster_nums,
            used_obsm='PRESENT',
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

    # === Clean temp columns ===
    temp_cols = [col for col in adata.obs.columns if 'tmp_search' in col]
    for col in temp_cols:
        del adata.obs[col]
        if col in adata.uns:
            del adata.uns[col]

    # === Save AnnData ===
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    adata.write(args.save_path)
    print(adata)
    print(f'Saved results to {args.save_path}')


if __name__ == "__main__":
    os.environ['R_HOME'] = '/home/zhenghong/miniconda3/envs/smobench/lib/R'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    print("Starting PRESENT-3M integration...")
    parser = argparse.ArgumentParser(description='Run PRESENT 3M integration (RNA+ADT+ATAC)')
    parser.add_argument('--data_type', type=str, default='simulation')
    parser.add_argument('--RNA_path', type=str, required=True)
    parser.add_argument('--ADT_path', type=str, required=True)
    parser.add_argument('--ATAC_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--method', type=str, default='PRESENT_3M')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--cluster_nums', type=int, required=True)

    args = parser.parse_args()
    main(args)
