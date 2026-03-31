#!/usr/bin/env python3
"""
Compute per-spot CM-GTC scores for ED Fig 3 (spatial projection).

For each method × dataset × slice (vertical, withGT only):
  1. Load joint embedding from method h5ad
  2. Load original modality data (RNA + ADT/ATAC)
  3. Load spatial coordinates from original dataset
  4. Compute CM-GTC and extract per-spot consistency_cross_modal
  5. Save to CSV: [Method, Dataset, Slice, cell_idx, x, y, cmgtc_perspot]

Usage:
    python compute_cmgtc_perspot.py
    python compute_cmgtc_perspot.py --methods CANDIES MISO   # subset
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = '/data/projects/11003054/e1724738/_private/NUS/_Proj1/SMOBench-CLEAN'
RESULTS = os.path.join(ROOT, '_myx_Results')
VERT_ADATA = os.path.join(RESULTS, 'adata', 'vertical_integration')
OUTPUT_DIR = os.path.join(RESULTS, 'evaluation', 'cmgtc_perspot')

VERTICAL_METHODS = [
    'CANDIES', 'COSMOS', 'MISO', 'MultiGATE', 'PRAGA', 'PRESENT',
    'SMOPCA', 'SpaBalance', 'SpaFusion', 'SpaMI', 'SpaMosaic',
    'SpaMultiVAE', 'SpaMV', 'SpatialGlue', 'SWITCH',
]

WITHGT_DATASETS = {
    'Human_Lymph_Nodes': {'type': 'RNA_ADT', 'slices': ['A1', 'D1']},
    'Human_Tonsils':     {'type': 'RNA_ADT', 'slices': ['S1', 'S2', 'S3']},
    'Mouse_Embryos_S1':  {'type': 'RNA_ATAC', 'slices': ['E11', 'E13', 'E15', 'E18']},
    'Mouse_Embryos_S2':  {'type': 'RNA_ATAC', 'slices': ['E11', 'E13', 'E15', 'E18']},
}

# woGT datasets — CM-GTC is label-free, can compute per-spot without GT
WOGT_DATASETS = {
    'Mouse_Spleen':  {'type': 'RNA_ADT', 'slices': None},  # auto-detect
    'Mouse_Thymus':  {'type': 'RNA_ADT', 'slices': None},
    'Mouse_Brain':   {'type': 'RNA_ATAC', 'slices': None},
}

# Merge all datasets
ALL_DATASETS = {**WITHGT_DATASETS, **WOGT_DATASETS}

# Methods that cannot handle ATAC
METHOD_SKIP_ATAC = {'SpaMultiVAE', 'SpaFusion'}

CMGTC_PATH = '/home/users/nus/e1724738/_main/_private/NUS/_Proj1/storage/_2_metric'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dense(X):
    if sparse.issparse(X):
        return np.asarray(X.todense())
    return np.asarray(X)


def _preprocess_rna(adata):
    adata = adata.copy()
    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    n_hvg = min(3000, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor='seurat_v3')
    adata = adata[:, adata.var['highly_variable']].copy()
    return _to_dense(adata.X)


def _preprocess_adt(adata):
    adata = adata.copy()
    X = _to_dense(adata.X).astype(np.float64)
    X = np.clip(X, 0, None) + 1
    geo_mean = np.exp(np.mean(np.log(X), axis=1, keepdims=True))
    return (np.log(X / geo_mean)).astype(np.float32)


def _preprocess_atac(adata):
    from sklearn.decomposition import TruncatedSVD
    adata = adata.copy()
    if sparse.issparse(adata.X):
        adata.X = (adata.X > 0).astype(np.float32)
    else:
        adata.X = (np.asarray(adata.X) > 0).astype(np.float32)
    sc.pp.filter_genes(adata, min_cells=1)
    n_peak = min(5000, adata.n_vars)
    if adata.n_vars > n_peak:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_peak, flavor='seurat_v3')
        adata = adata[:, adata.var['highly_variable']].copy()
    X = _to_dense(adata.X)
    tf = X / (X.sum(axis=1, keepdims=True) + 1e-8)
    idf = np.log1p(X.shape[0] / (X.sum(axis=0, keepdims=True) + 1e-8))
    tfidf = (tf * idf).astype(np.float32)
    n_comps = min(51, tfidf.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comps, random_state=42)
    lsi = svd.fit_transform(tfidf)
    return lsi[:, 1:].astype(np.float32)


def load_spatial_coords(dataset, slice_name):
    """Load spatial coordinates from original dataset."""
    base = _resolve_slice_dir(dataset, slice_name)
    if base is None:
        return None
    rna_path = os.path.join(base, 'adata_RNA.h5ad')
    if not os.path.isfile(rna_path):
        return None
    adata = sc.read_h5ad(rna_path)
    # Try multiple spatial key names
    for key in ['X_spatial', 'spatial', 'X_umap']:
        if key in adata.obsm:
            coords = np.asarray(adata.obsm[key])
            if coords.shape[1] >= 2:
                return coords[:, :2]
    return None


def _resolve_slice_dir(dataset, slice_name):
    """Resolve slice directory, handling woGT name mismatches.
    e.g. 'Spleen1' → 'Mouse_Spleen1', 'ATAC' → 'Mouse_Brain_ATAC'."""
    info = ALL_DATASETS[dataset]
    for gt in ['withGT', 'woGT']:
        base = os.path.join(ROOT, 'Dataset', gt, info['type'], dataset)
        if not os.path.isdir(base):
            continue
        # Try exact match
        candidate = os.path.join(base, slice_name)
        if os.path.isdir(candidate):
            return candidate
        # Try prefix: 'Spleen1' → 'Mouse_Spleen1'
        candidate = os.path.join(base, f'{dataset}{slice_name}')
        if os.path.isdir(candidate):
            return candidate
        # Try prefix with underscore: 'ATAC' → 'Mouse_Brain_ATAC'
        candidate = os.path.join(base, f'{dataset}_{slice_name}')
        if os.path.isdir(candidate):
            return candidate
        # Fuzzy: any subdir ending with slice_name
        for d in sorted(os.listdir(base)):
            if d.endswith(slice_name) and os.path.isdir(os.path.join(base, d)):
                return os.path.join(base, d)
    return None


def load_modality_data(dataset, slice_name):
    """Load RNA + second modality for a slice."""
    info = ALL_DATASETS[dataset]
    base = _resolve_slice_dir(dataset, slice_name)
    if base is None:
        return {}
    rna_path = os.path.join(base, 'adata_RNA.h5ad')
    second_mod = 'ADT' if 'ADT' in info['type'] else 'ATAC'
    second_path = os.path.join(base, f'adata_{second_mod}.h5ad')
    if not os.path.isfile(second_path) and second_mod == 'ATAC':
        alt = os.path.join(base, 'adata_peaks_normalized.h5ad')
        if os.path.isfile(alt):
            second_path = alt

    modalities = {}
    if os.path.isfile(rna_path):
        modalities['rna'] = _preprocess_rna(sc.read_h5ad(rna_path))
    if os.path.isfile(second_path):
        ad_s = sc.read_h5ad(second_path)
        modalities[second_mod.lower()] = _preprocess_adt(ad_s) if second_mod == 'ADT' else _preprocess_atac(ad_s)
    return modalities


def get_embedding(adata, method_name):
    """Extract embedding from method output h5ad."""
    candidates = [method_name, method_name.lower(), f'{method_name}_emb',
                  'X_emb', 'X_integrated', 'latent', 'X_pca']
    for key in candidates:
        if key in adata.obsm:
            emb = adata.obsm[key]
            if sparse.issparse(emb):
                emb = emb.toarray()
            return np.asarray(emb)
    skip = {'X_spatial', 'spatial', 'X_umap'}
    for key in adata.obsm:
        if key not in skip:
            emb = adata.obsm[key]
            if sparse.issparse(emb):
                emb = emb.toarray()
            return np.asarray(emb)
    return None


def find_h5ad(method, dataset, slice_name):
    """Find method output h5ad for a given slice."""
    base = os.path.join(VERT_ADATA, method, dataset)
    if not os.path.isdir(base):
        return None
    # Try: METHOD/DATASET/SLICE/*.h5ad
    slice_dir = os.path.join(base, slice_name)
    if os.path.isdir(slice_dir):
        h5ads = sorted(Path(slice_dir).glob('*.h5ad'))
        if h5ads:
            return str(h5ads[0])
    # Try: METHOD/DATASET/*SLICE*.h5ad
    for f in sorted(Path(base).glob('*.h5ad')):
        if slice_name in f.stem:
            return str(f)
    # Try: METHOD/DATASET/*.h5ad (single file)
    h5ads = sorted(Path(base).glob('*.h5ad'))
    if len(h5ads) == 1:
        return str(h5ads[0])
    return None


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_all(methods=None):
    # Import CMGTC
    if CMGTC_PATH not in sys.path:
        sys.path.insert(0, CMGTC_PATH)
    from cm_gtc_v2 import CMGTC_v2 as CMGTC

    evaluator = CMGTC(
        similarity_metric='cosine',
        correlation_metric='spearman',
        aggregation_strategy='min',
        verbose=False,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    method_list = methods or VERTICAL_METHODS

    all_rows = []
    n_computed = 0
    n_skipped = 0

    for method in method_list:
        print(f"\n=== {method} ===")

        for dataset, info in ALL_DATASETS.items():
            # Skip ATAC-incompatible
            if method in METHOD_SKIP_ATAC and 'ATAC' in info['type']:
                continue

            # Auto-detect slices for woGT datasets
            slices = info['slices']
            if slices is None:
                # Detect from method adata directory
                method_ds_dir = os.path.join(VERT_ADATA, method, dataset)
                if os.path.isdir(method_ds_dir):
                    sub = sorted(os.listdir(method_ds_dir))
                    slices = [s for s in sub if os.path.isdir(os.path.join(method_ds_dir, s))]
                    if not slices:
                        # Single h5ad, use dataset name as slice
                        slices = [dataset]
                else:
                    continue

            for slice_name in slices:
                tag = f"{method}/{dataset}/{slice_name}"

                # 1) Find method h5ad
                h5ad_path = find_h5ad(method, dataset, slice_name)
                if h5ad_path is None:
                    print(f"  [{tag}] SKIP - no h5ad")
                    n_skipped += 1
                    continue

                # 2) Load embedding
                try:
                    adata = sc.read_h5ad(h5ad_path)
                except Exception as e:
                    print(f"  [{tag}] SKIP - read error: {e}")
                    n_skipped += 1
                    continue

                embedding = get_embedding(adata, method)
                if embedding is None:
                    print(f"  [{tag}] SKIP - no embedding")
                    n_skipped += 1
                    continue

                # 3) Load modality data
                modalities = load_modality_data(dataset, slice_name)
                if len(modalities) < 1:
                    print(f"  [{tag}] SKIP - no modality data")
                    n_skipped += 1
                    continue

                # 4) Load spatial coordinates
                coords = load_spatial_coords(dataset, slice_name)
                if coords is None:
                    print(f"  [{tag}] SKIP - no spatial coords")
                    n_skipped += 1
                    continue

                # 5) Align sizes
                n_embed = embedding.shape[0]
                min_n = min(n_embed, coords.shape[0],
                            *[v.shape[0] for v in modalities.values()])
                embedding = embedding[:min_n]
                coords = coords[:min_n]
                modalities = {k: v[:min_n] for k, v in modalities.items()}

                if min_n < 10:
                    print(f"  [{tag}] SKIP - too few cells ({min_n})")
                    n_skipped += 1
                    continue

                # 6) Compute CM-GTC with per-spot details (global + per-modality)
                try:
                    score, details = evaluator.compute_cm_gtc(embedding, modalities)
                    perspot = details['consistency_cross_modal']  # shape (N,) — min-aggregated
                except Exception as e:
                    print(f"  [{tag}] ERROR: {e}")
                    n_skipped += 1
                    continue

                # Per-modality CM-GTC (single-modality, no min-aggregation)
                permod_scores = {}
                for mod_name, mod_data in modalities.items():
                    try:
                        _, d = evaluator.compute_cm_gtc(embedding, {mod_name: mod_data})
                        permod_scores[mod_name] = d['consistency_cross_modal']
                    except Exception:
                        permod_scores[mod_name] = np.full(min_n, np.nan)

                # 7) Collect rows
                for i in range(min_n):
                    row = {
                        'Method': method,
                        'Dataset': dataset,
                        'Slice': slice_name,
                        'cell_idx': i,
                        'x': coords[i, 0],
                        'y': coords[i, 1],
                        'cmgtc_perspot': perspot[i],
                    }
                    for mod_name, mod_ps in permod_scores.items():
                        row[f'cmgtc_{mod_name}'] = mod_ps[i]
                    all_rows.append(row)

                n_computed += 1
                mod_info = ', '.join(f'{m}={s.mean():.3f}' for m, s in permod_scores.items())
                print(f"  [{tag}] CM-GTC={score:.4f}, {min_n} spots, "
                      f"per-spot range=[{perspot.min():.3f}, {perspot.max():.3f}], {mod_info}")

    # Save per-method CSV (safe for parallel jobs)
    if all_rows:
        df = pd.DataFrame(all_rows)
        methods_tag = '_'.join(sorted(set(df['Method'])))
        out_path = os.path.join(OUTPUT_DIR, f'cmgtc_perspot_{methods_tag}.csv')
        df.to_csv(out_path, index=False)
        print(f"\nSaved {len(df)} per-spot scores to {out_path}")

        # Also save per-method summary
        summary = df.groupby(['Method', 'Dataset', 'Slice']).agg(
            mean_cmgtc=('cmgtc_perspot', 'mean'),
            std_cmgtc=('cmgtc_perspot', 'std'),
            n_spots=('cmgtc_perspot', 'count'),
            pct_negative=('cmgtc_perspot', lambda x: (x < 0).mean() * 100),
        ).reset_index()
        summary_path = os.path.join(OUTPUT_DIR, f'cmgtc_perspot_summary_{methods_tag}.csv')
        summary.to_csv(summary_path, index=False)
        print(f"Saved summary to {summary_path}")

    print(f"\nDone: {n_computed} computed, {n_skipped} skipped")
    return all_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+', default=None)
    args = parser.parse_args()

    t0 = time.time()
    compute_all(args.methods)
    elapsed = time.time() - t0
    print(f"Total time: {elapsed/60:.1f} min")


if __name__ == '__main__':
    main()
