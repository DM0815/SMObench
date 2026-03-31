#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run PRAGA-3M triple-modality integration (RNA + ADT + ATAC)

PRAGA has dedicated 3M modules: model_3M.py, preprocess_3M.py, Train_model_3M.py
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

# PRAGA 3M uses `from ..clustering_utils import ...` in optimal_clustering.py,
# requiring Methods/PRAGA/ to be a proper package (not just a sys.path entry).
# We add Methods/ to sys.path so PRAGA becomes a sub-package and .. resolves correctly.
methods_path = os.path.join(project_root, "Methods")
praga_path = os.path.join(project_root, "Methods/PRAGA")

# Ensure __init__.py exists for package resolution
for d in [praga_path, os.path.join(praga_path, "PRAGA")]:
    init_f = os.path.join(d, "__init__.py")
    if not os.path.exists(init_f):
        open(init_f, 'w').close()

sys.path.insert(0, methods_path)

from PRAGA.PRAGA.Train_model_3M import Train_3M
from PRAGA.PRAGA.preprocess_3M import construct_neighbor_graph
from PRAGA.PRAGA.preprocess import pca, clr_normalize_each_cell, lsi, fix_seed
from Utils.SMOBench_clustering import universal_clustering


class Args3M:
    """PRAGA hyperparameters for 3M simulation data."""
    def __init__(self):
        self.device = 'cuda:0'
        self.seed = 2024
        self.feat_n_comps = 200
        self.n_neighbors = 3
        self.KNN_k = 20
        self.RNA_weight = 5
        self.ADT_weight = 1
        self.ATAC_weight = 1
        self.cl_weight = 5
        self.alpha = 0.9
        self.tau = 1
        self.init_k = 5


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

    args_praga = Args3M()
    args_praga.device = args.device
    args_praga.seed = args.seed
    fix_seed(args.seed)

    # === Load all three modalities ===
    print("Loading RNA + ADT + ATAC data...")
    adata_rna = sc.read_h5ad(args.RNA_path)
    adata_adt = sc.read_h5ad(args.ADT_path)
    adata_atac = sc.read_h5ad(args.ATAC_path)
    for a in [adata_rna, adata_adt, adata_atac]:
        a.var_names_make_unique()

    # === RNA preprocessing ===
    sc.pp.filter_genes(adata_rna, min_cells=3)
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.highly_variable_genes(adata_rna, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata_rna.raw = adata_rna
    adata_rna = adata_rna[:, adata_rna.var.highly_variable]

    max_comps_rna = min(adata_rna.n_obs, adata_rna.n_vars) - 1
    n_comps_rna = min(args_praga.feat_n_comps, max_comps_rna)
    adata_rna.obsm['feat'] = pca(adata_rna, n_comps=n_comps_rna)
    print(f"RNA: {adata_rna.shape}, PCA{n_comps_rna}")

    # === ADT preprocessing ===
    sc.pp.filter_genes(adata_adt, min_cells=3)
    clr_normalize_each_cell(adata_adt)
    max_comps_adt = min(adata_adt.n_obs, adata_adt.n_vars) - 1
    n_comps_adt = min(args_praga.feat_n_comps, max_comps_adt)
    adata_adt.obsm['feat'] = pca(adata_adt, n_comps=n_comps_adt)
    print(f"ADT: {adata_adt.shape}, PCA{n_comps_adt}")

    # === ATAC preprocessing ===
    sc.pp.filter_genes(adata_atac, min_cells=3)
    sc.pp.normalize_total(adata_atac, target_sum=1e4)
    sc.pp.log1p(adata_atac)
    lsi(adata_atac, n_components=50)
    adata_atac.obsm['feat'] = adata_atac.obsm['X_lsi']
    print(f"ATAC: {adata_atac.shape}, LSI50")

    # === Construct 3M neighbor graph ===
    data = construct_neighbor_graph(adata_rna, adata_adt, adata_atac,
                                    n_neighbors=args_praga.n_neighbors)

    # === Train PRAGA-3M ===
    print("Training PRAGA-3M model...")
    start_time = time.time()
    model = Train_3M(data, datatype='Triplet', device=device,
                     random_seed=args.seed, Arg=args_praga)
    output = model.train()
    train_time = time.time() - start_time
    print(f'Training time: {train_time:.2f}s')

    # === Build Result AnnData ===
    adata = data['adata_omics1'].copy()
    adata.obsm['PRAGA'] = output['PRAGA']
    adata.uns['train_time'] = train_time

    # === Clean embeddings ===
    embeddings_clean = adata.obsm['PRAGA'].copy()
    if np.any(~np.isfinite(embeddings_clean)):
        print("Warning: Found NaN/Inf in embeddings. Cleaning...")
        embeddings_clean[np.isinf(embeddings_clean)] = np.sign(embeddings_clean[np.isinf(embeddings_clean)]) * 1e10
        embeddings_clean[np.isnan(embeddings_clean)] = 0
        adata.obsm['PRAGA'] = embeddings_clean
    else:
        print("Embeddings are clean")

    # === Dataset info ===
    dataset_name, subset_name = parse_dataset_info(args)
    print(f"Dataset: {dataset_name}, subset: {subset_name}")

    # === Clustering ===
    sc.pp.neighbors(adata, use_rep='PRAGA', n_neighbors=30)
    sc.tl.umap(adata)

    for tool in ['mclust', 'louvain', 'leiden', 'kmeans']:
        adata = universal_clustering(
            adata, n_clusters=args.cluster_nums,
            used_obsm='PRAGA', method=tool,
            key=tool, use_pca=False
        )

    method_name = "PRAGA_3M"
    plot_dir = os.path.join("Results/plot", method_name, dataset_name, subset_name)
    os.makedirs(plot_dir, exist_ok=True)

    for tool in ['mclust', 'louvain', 'leiden', 'kmeans']:
        if 'spatial' in adata.obsm.keys():
            adata.obsm['spatial'][:, 1] = -1 * adata.obsm['spatial'][:, 1]
        fig, axs = plt.subplots(1, 2, figsize=(7, 3))
        sc.pl.umap(adata, color=tool, ax=axs[0], title=f'{method_name}-{tool}', s=20, show=False)
        if 'spatial' in adata.obsm.keys():
            sc.pl.embedding(adata, basis='spatial', color=tool, ax=axs[1], title=f'{method_name}-{tool}', s=20, show=False)
        plt.tight_layout(w_pad=0.3)
        plt.savefig(os.path.join(plot_dir, f'clustering_{tool}_umap_spatial.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Clean temp columns
    temp_cols = [col for col in adata.obs.columns if 'tmp_search' in col]
    for col in temp_cols:
        del adata.obs[col]
        if col in adata.uns:
            del adata.uns[col]

    # === Save ===
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    adata.write(args.save_path)
    print(f'Saved results to {args.save_path}')


if __name__ == "__main__":
    os.environ['R_HOME'] = '/home/users/nus/e1724738/miniconda3/envs/_Proj1_1/lib/R'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    print("Starting PRAGA-3M integration...")
    parser = argparse.ArgumentParser(description='Run PRAGA 3M integration (RNA+ADT+ATAC)')
    parser.add_argument('--data_type', type=str, default='simulation')
    parser.add_argument('--RNA_path', type=str, required=True)
    parser.add_argument('--ADT_path', type=str, required=True)
    parser.add_argument('--ATAC_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--method', type=str, default='PRAGA_3M')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--cluster_nums', type=int, required=True)

    args = parser.parse_args()
    main(args)
