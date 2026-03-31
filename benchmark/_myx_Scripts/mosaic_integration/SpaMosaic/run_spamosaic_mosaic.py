#!/usr/bin/env python3
"""
SpaMosaic Mosaic Integration — Missing Modality Evaluation

For each dataset with multiple batches:
  - Bridge batches (first N-1): have ALL modalities
  - Query batch (last 1): missing one modality

Two scenarios:
  - without_rna:    query batch has only ADT/ATAC (RNA = None)
  - without_second: query batch has only RNA (ADT/ATAC = None)

Usage:
    python run_spamosaic_mosaic.py --dataset HLN --scenario without_rna \
        --save_path .../SpaMosaic_HLN_without_rna.h5ad --cluster_nums 10
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import scanpy as sc
from datetime import datetime

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['R_HOME'] = '/home/users/nus/e1724738/miniconda3/envs/_Proj1_1/lib/R'
os.environ['PATH'] = '/home/users/nus/e1724738/miniconda3/envs/_Proj1_1/bin:' + os.environ.get('PATH', '')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "Methods/SpaMosaic"))

from spamosaic.framework import SpaMosaic
import spamosaic.utils as utls
from spamosaic.preprocessing import RNA_preprocess, ADT_preprocess, Epigenome_preprocess
from Utils.SMOBench_clustering import universal_clustering

# ── Dataset Configuration ──────────────────────────────────────────────────
DATASET_CONFIG = {
    'HLN': {
        'second_mod': 'ADT',
        'cluster_nums': 10,
        'batches': [
            ('Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/A1', 'HLN_A1'),
            ('Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/D1', 'HLN_D1'),
        ],
    },
    'HT': {
        'second_mod': 'ADT',
        'cluster_nums': 5,
        'batches': [
            ('Dataset/withGT/RNA_ADT/Human_Tonsils/S1', 'HT_S1'),
            ('Dataset/withGT/RNA_ADT/Human_Tonsils/S2', 'HT_S2'),
            ('Dataset/withGT/RNA_ADT/Human_Tonsils/S3', 'HT_S3'),
        ],
    },
    'Mouse_Spleen': {
        'second_mod': 'ADT',
        'cluster_nums': 5,
        'batches': [
            ('Dataset/woGT/RNA_ADT/Mouse_Spleen/Mouse_Spleen1', 'Mouse_Spleen1'),
            ('Dataset/woGT/RNA_ADT/Mouse_Spleen/Mouse_Spleen2', 'Mouse_Spleen2'),
        ],
    },
    'Mouse_Thymus': {
        'second_mod': 'ADT',
        'cluster_nums': 8,
        'batches': [
            ('Dataset/woGT/RNA_ADT/Mouse_Thymus/Mouse_Thymus1', 'Mouse_Thymus1'),
            ('Dataset/woGT/RNA_ADT/Mouse_Thymus/Mouse_Thymus2', 'Mouse_Thymus2'),
            ('Dataset/woGT/RNA_ADT/Mouse_Thymus/Mouse_Thymus3', 'Mouse_Thymus3'),
            ('Dataset/woGT/RNA_ADT/Mouse_Thymus/Mouse_Thymus4', 'Mouse_Thymus4'),
        ],
    },
    'MISAR_S1': {
        'second_mod': 'ATAC',
        'cluster_nums': 12,
        'batches': [
            ('Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1/E11', 'MISAR_S1_E11'),
            ('Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1/E13', 'MISAR_S1_E13'),
            ('Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1/E15', 'MISAR_S1_E15'),
            ('Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1/E18', 'MISAR_S1_E18'),
        ],
    },
    'MISAR_S2': {
        'second_mod': 'ATAC',
        'cluster_nums': 14,
        'batches': [
            ('Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2/E11', 'MISAR_S2_E11'),
            ('Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2/E13', 'MISAR_S2_E13'),
            ('Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2/E15', 'MISAR_S2_E15'),
            ('Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2/E18', 'MISAR_S2_E18'),
        ],
    },
    'Mouse_Brain': {
        'second_mod': 'ATAC',
        'cluster_nums': 18,
        'batches': [
            ('Dataset/woGT/RNA_ATAC/Mouse_Brain/Mouse_Brain_ATAC', 'Mouse_Brain_ATAC'),
            ('Dataset/woGT/RNA_ATAC/Mouse_Brain/Mouse_Brain_H3K4me3', 'Mouse_Brain_H3K4me3'),
            ('Dataset/woGT/RNA_ATAC/Mouse_Brain/Mouse_Brain_H3K27ac', 'Mouse_Brain_H3K27ac'),
            ('Dataset/woGT/RNA_ATAC/Mouse_Brain/Mouse_Brain_H3K27me3', 'Mouse_Brain_H3K27me3'),
        ],
    },
}


def load_batch(base_path, modality, batch_label):
    """Load one batch's AnnData for a given modality."""
    if modality == 'RNA':
        fp = os.path.join(base_path, 'adata_RNA.h5ad')
    elif modality == 'ADT':
        fp = os.path.join(base_path, 'adata_ADT.h5ad')
    elif modality == 'ATAC':
        for name in ['adata_ATAC.h5ad', 'adata_peaks_normalized.h5ad']:
            fp = os.path.join(base_path, name)
            if os.path.exists(fp):
                break
    else:
        raise ValueError(f"Unknown modality: {modality}")

    if not os.path.exists(fp):
        raise FileNotFoundError(f"File not found: {fp}")

    ad_raw = sc.read_h5ad(fp)
    ad_raw.var_names_make_unique()
    # Rebuild clean AnnData to avoid 1D obsm entries crashing sc.concat
    import scipy.sparse as sp
    X = ad_raw.X
    if hasattr(X, 'dtype') and X.dtype != np.float32:
        X = X.astype(np.float32)
    ad = sc.AnnData(X=X, obs=ad_raw.obs.copy(), var=ad_raw.var.copy())
    if 'spatial' in ad_raw.obsm:
        ad.obsm['spatial'] = ad_raw.obsm['spatial'].copy()
    del ad_raw
    ad.obs['src'] = batch_label
    ad.obs_names = [f"{batch_label}-{x}" for x in ad.obs_names]
    return ad


def normalize_feature_name(name):
    """Normalize ADT/ATAC feature names for cross-batch matching."""
    n = name.lower().replace('-', '_')
    # Remove species prefixes (order matters: longest first)
    for prefix in ['ms_hu_', 'mouse_rat_human_', 'mouse_human_', 'mouse_rat_',
                    'mouse_', 'human_', 'rat_']:
        if n.startswith(prefix):
            n = n[len(prefix):]
            break
    return n


def harmonize_var_names(ad_list):
    """Harmonize feature names across batches using normalized names.
    Renames all var_names to a canonical form so intersection works."""
    if not ad_list or all(ad is None for ad in ad_list):
        return ad_list

    for ad in ad_list:
        if ad is None:
            continue
        mapping = {}
        for name in ad.var_names:
            mapping[name] = normalize_feature_name(name)
        ad.var.index = [mapping[n] for n in ad.var_names]
        ad.var_names_make_unique()

    return ad_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(DATASET_CONFIG.keys()))
    parser.add_argument('--scenario', type=str, required=True,
                        choices=['without_rna', 'without_second'])
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--cluster_nums', type=int, default=None)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    second_mod = cfg['second_mod']
    cluster_nums = args.cluster_nums or cfg['cluster_nums']
    n_batches = len(cfg['batches'])

    print("=" * 60)
    print(f"SpaMosaic Mosaic Integration")
    print(f"Dataset: {args.dataset} ({n_batches} batches)")
    print(f"Scenario: {args.scenario}")
    print(f"Second modality: {second_mod}")
    print(f"Bridge batches: first {n_batches - 1}, Query batch: last 1")
    print("=" * 60)

    total_start = time.time()

    # ── Load all batches ──────────────────────────────────────────────────
    rna_list = []
    second_list = []
    for i, (base_path, label) in enumerate(cfg['batches']):
        full_path = os.path.join(project_root, base_path)

        is_query = (i == n_batches - 1)  # last batch is query

        if args.scenario == 'without_rna' and is_query:
            rna_list.append(None)
            print(f"  Batch {i} [{label}]: RNA=None (query), {second_mod}=loaded")
        else:
            ad_rna = load_batch(full_path, 'RNA', label)
            rna_list.append(ad_rna)
            print(f"  Batch {i} [{label}]: RNA={ad_rna.n_obs} cells", end="")

        if args.scenario == 'without_second' and is_query:
            second_list.append(None)
            print(f", {second_mod}=None (query)")
        else:
            ad_sec = load_batch(full_path, second_mod, label)
            second_list.append(ad_sec)
            if args.scenario == 'without_rna' and is_query:
                print(f"  → {second_mod}={ad_sec.n_obs} cells")
            else:
                print(f", {second_mod}={ad_sec.n_obs} cells")

    # ── Harmonize feature names (ADT only; ATAC peaks use genomic coords) ──
    if second_mod == 'ADT':
        harmonize_var_names(second_list)

    # ── Filter genes, then find shared features ──────────────────────────
    # filter_genes BEFORE intersection to avoid per-batch mismatch
    for ad in rna_list:
        if ad is not None:
            sc.pp.filter_genes(ad, min_cells=1)
    for ad in second_list:
        if ad is not None:
            sc.pp.filter_genes(ad, min_cells=1)

    rna_valid = [ad for ad in rna_list if ad is not None]
    sec_valid = [ad for ad in second_list if ad is not None]

    if rna_valid:
        shared_genes = rna_valid[0].var_names
        for ad in rna_valid[1:]:
            shared_genes = shared_genes.intersection(ad.var_names)
        print(f"Shared RNA genes: {len(shared_genes)}")
        for i, ad in enumerate(rna_list):
            if ad is not None:
                rna_list[i] = ad[:, shared_genes].copy()

    if sec_valid:
        shared_feats = sec_valid[0].var_names
        for ad in sec_valid[1:]:
            shared_feats = shared_feats.intersection(ad.var_names)
        print(f"Shared {second_mod} features: {len(shared_feats)}")
        if len(shared_feats) > 0:
            for i, ad in enumerate(second_list):
                if ad is not None:
                    second_list[i] = ad[:, shared_feats].copy()
        else:
            # ATAC peaks are called independently per sample — no shared features
            # Each batch keeps its own features; Epigenome_preprocess handles this
            print(f"  → Skipping intersection (each batch keeps own {second_mod} features)")

    # ── Build modBatch_dict ───────────────────────────────────────────────
    second_key = 'protein' if second_mod == 'ADT' else 'epigenome'
    input_dict = {
        'rna': rna_list,
        second_key: second_list,
    }

    input_key = 'dimred_bc'

    # ── Preprocessing (only non-None batches) ─────────────────────────────
    print("Preprocessing...")
    rna_for_pp = [ad for ad in input_dict['rna'] if ad is not None]
    RNA_preprocess(rna_for_pp, batch_corr=True, n_hvg=3000,
                   batch_key='src', key=input_key)

    sec_for_pp = [ad for ad in input_dict[second_key] if ad is not None]
    if second_mod == 'ADT':
        ADT_preprocess(sec_for_pp, batch_corr=True, batch_key='src', key=input_key)
    else:
        # Check if ATAC batches share features
        if len(sec_for_pp) > 1:
            _shared = sec_for_pp[0].var_names
            for _ad in sec_for_pp[1:]:
                _shared = _shared.intersection(_ad.var_names)
        else:
            _shared = sec_for_pp[0].var_names if sec_for_pp else []
        if len(_shared) > 0:
            Epigenome_preprocess(sec_for_pp, batch_corr=True, n_peak=5000,
                                 batch_key='src', key=input_key)
        else:
            # ATAC peaks differ per batch — preprocess each independently
            print("  ATAC peaks not shared → independent LSI per batch")
            from spamosaic.preprocessing import lsiTransformer
            n_comps = 50
            for ad in sec_for_pp:
                n_peak = min(5000, ad.n_vars)
                import scanpy as _sc
                _sc.pp.highly_variable_genes(ad, flavor='seurat_v3',
                                              n_top_genes=n_peak)
                hvf = ad.var.query('highly_variable').index.to_numpy()
                transformer = lsiTransformer(
                    n_components=min(n_comps, len(hvf) - 1),
                    drop_first=True, log=True, norm=True,
                    z_score=True, tfidf=True, svd=True, pcaAlgo='arpack')
                ad.obsm[input_key] = transformer.fit_transform(ad[:, hvf]).values
                print(f"    {ad.obs['src'].iloc[0]}: {ad.n_vars} peaks → {ad.obsm[input_key].shape[1]}D")

    # ── Train SpaMosaic ───────────────────────────────────────────────────
    print("Training SpaMosaic model...")
    train_start = time.time()

    model = SpaMosaic(
        modBatch_dict=input_dict,
        input_key=input_key,
        batch_key='src',
        intra_knns=15,
        inter_knn_base=15,
        w_g=0.9,
        smooth_input=True,
        smooth_L=2,
        inter_auto_knn=True,
        rmv_outlier=True,
        contamination=0.1,
        seed=args.seed,
        device=args.device,
    )

    w_rec_g = 0.1 if second_mod == 'ADT' else 1.0
    model.train(net='wlgcn', lr=0.001, n_epochs=100, w_rec_g=w_rec_g)
    train_time = time.time() - train_start
    print(f"Training time: {train_time:.1f}s")

    # ── Inference ─────────────────────────────────────────────────────────
    print("Inferring embeddings...")
    ad_embs = model.infer_emb(input_dict, emb_key='emb', final_latent_key='merged_emb')
    ad_mosaic = sc.concat(ad_embs)
    ad_mosaic = utls.get_umap(ad_mosaic, use_reps=['merged_emb'])

    adata = ad_mosaic.copy()
    adata.obsm['SpaMosaic'] = adata.obsm['merged_emb']
    adata.obsm['X_umap'] = adata.obsm['merged_emb_umap']
    adata.uns['train_time'] = train_time
    adata.uns['integration_type'] = 'mosaic'
    adata.uns['scenario'] = args.scenario
    adata.obs['batch'] = adata.obs['src']

    # Clean embeddings
    emb = adata.obsm['SpaMosaic'].copy()
    if np.any(~np.isfinite(emb)):
        print("Warning: cleaning inf/NaN in embeddings")
        emb[np.isinf(emb)] = 0
        emb[np.isnan(emb)] = 0
        adata.obsm['SpaMosaic'] = emb

    # ── Clustering ────────────────────────────────────────────────────────
    print("Clustering...")
    for tool in ['leiden', 'louvain', 'kmeans', 'mclust']:
        print(f"  {tool}...")
        adata = universal_clustering(
            adata, n_clusters=cluster_nums,
            used_obsm='SpaMosaic', method=tool, key=tool, use_pca=False
        )

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    adata.write(args.save_path)
    print(f"Saved: {args.save_path}  shape={adata.shape}")

    total_time = time.time() - total_start
    timing = {
        'method': 'SpaMosaic',
        'dataset': args.dataset,
        'scenario': args.scenario,
        'n_cells': adata.n_obs,
        'n_batches': n_batches,
        'bridge_batches': n_batches - 1,
        'query_batch': cfg['batches'][-1][1],
        'training_time_s': round(train_time, 2),
        'total_time_s': round(total_time, 2),
        'device': args.device,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    timing_path = args.save_path.replace('.h5ad', '_timing.json')
    with open(timing_path, 'w') as f:
        json.dump(timing, f, indent=2)
    print(f"Done! Total time: {total_time:.1f}s")


if __name__ == '__main__':
    main()
