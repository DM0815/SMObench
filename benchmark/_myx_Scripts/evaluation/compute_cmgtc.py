#!/usr/bin/env python3
"""
SMOBench CM-GTC (Cross-Modal Global Topology Consistency) Evaluation
Computes CM-GTC scores for all methods × datasets on joint embeddings.

CM-GTC measures how well joint embeddings preserve topological structure
from each input modality — evaluating integration *process quality*
rather than clustering *result quality*.

Usage:
    python compute_cmgtc.py --root /path/to/SMOBench-CLEAN --task vertical
    python compute_cmgtc.py --root /path/to/SMOBench-CLEAN --task horizontal
    python compute_cmgtc.py --root /path/to/SMOBench-CLEAN --task both
    python compute_cmgtc.py --root /path/to/SMOBench-CLEAN --task vertical --methods CANDIES COSMOS --test
"""

import os
import sys
import argparse
import time
import json
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

VERTICAL_METHODS = [
    'CANDIES', 'COSMOS', 'MISO', 'MultiGATE', 'PRAGA', 'PRESENT',
    'SMOPCA', 'SpaBalance', 'SpaFusion', 'SpaMI', 'SpaMosaic',
    'SpaMultiVAE', 'SpaMV', 'SpatialGlue', 'SWITCH',
]

HORIZONTAL_METHODS = [
    'CANDIES', 'COSMOS', 'MISO', 'PRAGA', 'PRESENT',
    'SMOPCA', 'SpaBalance', 'SpaFusion', 'SpaMI', 'SpaMosaic',
    'SpaMultiVAE', 'SpaMV', 'SpatialGlue',
]

WITHGT_DATASETS = ['Human_Lymph_Nodes', 'Human_Tonsils', 'Mouse_Embryos_S1', 'Mouse_Embryos_S2']
WOGT_DATASETS = ['Mouse_Thymus', 'Mouse_Spleen', 'Mouse_Brain']
ALL_DATASETS = WITHGT_DATASETS + WOGT_DATASETS

# Dataset → modality type
DATASET_MODALITY = {
    'Human_Lymph_Nodes': 'RNA_ADT', 'Human_Tonsils': 'RNA_ADT',
    'Mouse_Embryos_S1': 'RNA_ATAC', 'Mouse_Embryos_S2': 'RNA_ATAC',
    'Mouse_Thymus': 'RNA_ADT', 'Mouse_Spleen': 'RNA_ADT',
    'Mouse_Brain': 'RNA_ATAC',
}

# Dataset → GT directory info (for loading original modality data)
DATASET_DIR_INFO = {
    'Human_Lymph_Nodes': {'gt': 'withGT', 'gt_dir': 'Human_Lymph_Nodes'},
    'Human_Tonsils':     {'gt': 'withGT', 'gt_dir': 'Human_Tonsils'},
    'Mouse_Embryos_S1':  {'gt': 'withGT', 'gt_dir': 'Mouse_Embryos_S1'},
    'Mouse_Embryos_S2':  {'gt': 'withGT', 'gt_dir': 'Mouse_Embryos_S2'},
    'Mouse_Thymus':      {'gt': 'woGT',   'gt_dir': None},
    'Mouse_Spleen':      {'gt': 'woGT',   'gt_dir': None},
    'Mouse_Brain':       {'gt': 'woGT',   'gt_dir': None},
}

# Methods that only support RNA_ADT
METHOD_SKIP_ATAC = {'SpaMultiVAE', 'SpaFusion'}

# Result directories may use short or full names
DATASET_DIR_ALIASES = {
    'Human_Lymph_Nodes': ['Human_Lymph_Nodes', 'HLN'],
    'Human_Tonsils':     ['Human_Tonsils', 'HT'],
    'Mouse_Embryos_S1':  ['Mouse_Embryos_S1', 'MISAR_S1'],
    'Mouse_Embryos_S2':  ['Mouse_Embryos_S2', 'MISAR_S2'],
    'Mouse_Thymus':      ['Mouse_Thymus'],
    'Mouse_Spleen':      ['Mouse_Spleen'],
    'Mouse_Brain':       ['Mouse_Brain'],
}


def parse_args():
    parser = argparse.ArgumentParser(description='SMOBench CM-GTC Evaluation')
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory of SMOBench-CLEAN')
    parser.add_argument('--cmgtc_path', type=str, default=None,
                        help='Path to cm_gtc.py (default: auto-detect from storage)')
    parser.add_argument('--task', choices=['vertical', 'horizontal', 'both'],
                        default='both', help='Which task to evaluate')
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Subset of methods to evaluate')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Subset of datasets to evaluate')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: 1 method × 1 dataset')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Import CM-GTC
# ---------------------------------------------------------------------------

def setup_cmgtc_import(root_dir, cmgtc_path=None):
    """Import CMGTC class from storage."""
    if cmgtc_path is None:
        # Auto-detect: look in common locations
        candidates = [
            os.path.join(os.path.dirname(root_dir), 'storage', '_2_metric'),
            '/home/users/nus/e1724738/_main/_private/NUS/_Proj1/storage/_2_metric',
        ]
        for cand in candidates:
            if os.path.isfile(os.path.join(cand, 'cm_gtc.py')):
                cmgtc_path = cand
                break

    if cmgtc_path is None:
        raise FileNotFoundError("Cannot find cm_gtc.py. Use --cmgtc_path to specify.")

    if cmgtc_path not in sys.path:
        sys.path.insert(0, cmgtc_path)

    from cm_gtc_v2 import CMGTC_v2 as CMGTC
    return CMGTC


# ---------------------------------------------------------------------------
# Modality data loading
# ---------------------------------------------------------------------------

def _to_dense(X):
    """Convert sparse matrix to dense numpy array."""
    if sparse.issparse(X):
        return np.asarray(X.todense())
    return np.asarray(X)


def _preprocess_rna(adata):
    """Standardized RNA preprocessing: normalize → log1p → 3000 HVGs → dense."""
    adata = adata.copy()
    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    n_hvg = min(3000, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor='seurat_v3')
    adata = adata[:, adata.var['highly_variable']].copy()
    return _to_dense(adata.X)


def _preprocess_adt(adata):
    """Standardized ADT preprocessing: CLR normalization → dense."""
    adata = adata.copy()
    # CLR: log(x / geometric_mean(x))
    X = _to_dense(adata.X).astype(np.float64)
    X = np.clip(X, 0, None)
    X += 1  # pseudocount
    geo_mean = np.exp(np.mean(np.log(X), axis=1, keepdims=True))
    X = np.log(X / geo_mean)
    return X.astype(np.float32)


def _preprocess_atac(adata):
    """Standardized ATAC preprocessing: binarize → filter top 5000 variable peaks → TF-IDF → LSI 50D."""
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
    return lsi[:, 1:].astype(np.float32)  # drop first component (read depth)


def _resolve_wogt_slice_dir(root_dir, dataset, slice_name):
    """Resolve woGT slice directory, handling name mismatches.

    Embedding results use short slice names (e.g. 'Thymus1', 'Spleen2', 'ATAC')
    but actual directories use full names (e.g. 'Mouse_Thymus1', 'Mouse_Brain_ATAC').
    """
    mod_type = DATASET_MODALITY[dataset]
    base = os.path.join(root_dir, 'Dataset', 'woGT', mod_type, dataset)

    # Try exact match first
    candidate = os.path.join(base, slice_name)
    if os.path.isdir(candidate):
        return candidate

    # Try prefixing with dataset name: 'Thymus1' → 'Mouse_Thymus1'
    candidate = os.path.join(base, f'{dataset}{slice_name}')
    if os.path.isdir(candidate):
        return candidate

    # Try prefixing with dataset name + underscore: 'ATAC' → 'Mouse_Brain_ATAC'
    candidate = os.path.join(base, f'{dataset}_{slice_name}')
    if os.path.isdir(candidate):
        return candidate

    # Fuzzy match: find subdirectory ending with slice_name
    if os.path.isdir(base):
        for d in sorted(os.listdir(base)):
            if d.endswith(slice_name) and os.path.isdir(os.path.join(base, d)):
                return os.path.join(base, d)

    return None


def _find_second_modality_file(base_dir, second_mod):
    """Find the second modality h5ad file, handling naming variants.

    Mouse_Brain uses 'adata_peaks_normalized.h5ad' instead of 'adata_ATAC.h5ad'.
    """
    # Standard name
    std_path = os.path.join(base_dir, f'adata_{second_mod}.h5ad')
    if os.path.isfile(std_path):
        return std_path

    # Alternative for ATAC: adata_peaks_normalized.h5ad
    if second_mod == 'ATAC':
        alt_path = os.path.join(base_dir, 'adata_peaks_normalized.h5ad')
        if os.path.isfile(alt_path):
            return alt_path

    return None


def load_modality_data_vertical(root_dir, dataset, slice_name):
    """Load original RNA + ADT/ATAC modality data for a vertical integration slice."""
    mod_type = DATASET_MODALITY[dataset]
    info = DATASET_DIR_INFO[dataset]

    if info['gt'] == 'withGT':
        base = os.path.join(root_dir, 'Dataset', 'withGT', mod_type,
                            info['gt_dir'], slice_name)
    else:
        base = _resolve_wogt_slice_dir(root_dir, dataset, slice_name)
        if base is None:
            # Fall back to dataset root (no subdirectories)
            base = os.path.join(root_dir, 'Dataset', 'woGT', mod_type, dataset)

    rna_path = os.path.join(base, 'adata_RNA.h5ad')
    second_mod = 'ADT' if 'ADT' in mod_type else 'ATAC'
    second_path = _find_second_modality_file(base, second_mod)

    modalities = {}
    if os.path.isfile(rna_path):
        adata_rna = sc.read_h5ad(rna_path)
        modalities['rna'] = _preprocess_rna(adata_rna)
    if second_path and os.path.isfile(second_path):
        adata_sec = sc.read_h5ad(second_path)
        if second_mod == 'ADT':
            modalities[second_mod.lower()] = _preprocess_adt(adata_sec)
        else:
            modalities[second_mod.lower()] = _preprocess_atac(adata_sec)

    return modalities


def load_modality_data_horizontal_per_slice(root_dir, dataset):
    """Load modality data per slice for horizontal integration.

    Returns a list of (slice_name, n_cells, modalities_dict) tuples.
    Different slices may have different feature counts, so we keep them separate
    and compute per-slice CM-GTC, then average.
    """
    mod_type = DATASET_MODALITY[dataset]
    info = DATASET_DIR_INFO[dataset]
    second_mod = 'ADT' if 'ADT' in mod_type else 'ATAC'

    slices = []  # List of (slice_name, n_cells, {mod_name: data_array})

    if info['gt'] == 'withGT':
        base = os.path.join(root_dir, 'Dataset', 'withGT', mod_type, info['gt_dir'])
        for slice_dir in sorted(os.listdir(base)):
            slice_path = os.path.join(base, slice_dir)
            if not os.path.isdir(slice_path):
                continue
            rna_path = os.path.join(slice_path, 'adata_RNA.h5ad')
            sec_path = _find_second_modality_file(slice_path, second_mod)
            mods = {}
            if os.path.isfile(rna_path):
                mods['rna'] = _preprocess_rna(sc.read_h5ad(rna_path))
            if sec_path:
                adata_sec = sc.read_h5ad(sec_path)
                if second_mod == 'ADT':
                    mods[second_mod.lower()] = _preprocess_adt(adata_sec)
                else:
                    mods[second_mod.lower()] = _preprocess_atac(adata_sec)
            if mods:
                n_cells = next(iter(mods.values())).shape[0]
                slices.append((slice_dir, n_cells, mods))
    else:
        base = os.path.join(root_dir, 'Dataset', 'woGT', mod_type, dataset)
        subdirs = sorted([d for d in os.listdir(base)
                         if os.path.isdir(os.path.join(base, d))])
        if subdirs:
            for sd in subdirs:
                sp = os.path.join(base, sd)
                rna_path = os.path.join(sp, 'adata_RNA.h5ad')
                sec_path = _find_second_modality_file(sp, second_mod)
                mods = {}
                if os.path.isfile(rna_path):
                    mods['rna'] = _preprocess_rna(sc.read_h5ad(rna_path))
                if sec_path:
                    ad_s = sc.read_h5ad(sec_path)
                    mods[second_mod.lower()] = _preprocess_adt(ad_s) if second_mod == 'ADT' else _preprocess_atac(ad_s)
                if mods:
                    n_cells = next(iter(mods.values())).shape[0]
                    slices.append((sd, n_cells, mods))
        else:
            rna_path = os.path.join(base, 'adata_RNA.h5ad')
            sec_path = _find_second_modality_file(base, second_mod)
            mods = {}
            if os.path.isfile(rna_path):
                mods['rna'] = _preprocess_rna(sc.read_h5ad(rna_path))
            if sec_path:
                ad_s = sc.read_h5ad(sec_path)
                mods[second_mod.lower()] = _preprocess_adt(ad_s) if second_mod == 'ADT' else _preprocess_atac(ad_s)
            if mods:
                n_cells = next(iter(mods.values())).shape[0]
                slices.append((dataset, n_cells, mods))

    # Also check fusion data if no slices found
    if not slices:
        fusion_rna = os.path.join(root_dir, 'Dataset', '_myx_fusionWoGT', mod_type,
                                  f'{dataset}_Fusion_RNA.h5ad')
        fusion_sec_path = os.path.join(root_dir, 'Dataset', '_myx_fusionWoGT', mod_type,
                                       f'{dataset}_Fusion_{second_mod}.h5ad')
        mods = {}
        if os.path.isfile(fusion_rna):
            mods['rna'] = _preprocess_rna(sc.read_h5ad(fusion_rna))
        if os.path.isfile(fusion_sec_path):
            ad_s = sc.read_h5ad(fusion_sec_path)
            mods[second_mod.lower()] = _preprocess_adt(ad_s) if second_mod == 'ADT' else _preprocess_atac(ad_s)
        if mods:
            n_cells = next(iter(mods.values())).shape[0]
            slices.append(('fusion', n_cells, mods))

    return slices


# ---------------------------------------------------------------------------
# Core CM-GTC computation
# ---------------------------------------------------------------------------

def compute_cmgtc_for_task(
    CMGTC_cls, root_dir, task, methods, datasets, test_mode=False
):
    """Compute CM-GTC for all method × dataset combinations in a task."""

    if task == 'vertical':
        input_dir = os.path.join(root_dir, '_myx_Results', 'adata', 'vertical_integration')
        method_list = methods or VERTICAL_METHODS
    else:
        input_dir = os.path.join(root_dir, '_myx_Results', 'adata', 'horizontal_integration')
        method_list = methods or HORIZONTAL_METHODS

    output_dir = os.path.join(root_dir, '_myx_Results', 'evaluation', 'cmgtc')
    os.makedirs(output_dir, exist_ok=True)

    dataset_list = datasets or ALL_DATASETS
    all_rows = []

    evaluator = CMGTC_cls(
        similarity_metric='cosine',
        correlation_metric='spearman',
        aggregation_strategy='min',
        verbose=False,
    )

    # Pre-load per-slice modality data for horizontal (cached across methods)
    horiz_modality_cache = {}
    if task == 'horizontal':
        for dataset in (datasets or ALL_DATASETS):
            try:
                slices_data = load_modality_data_horizontal_per_slice(root_dir, dataset)
                if slices_data:
                    horiz_modality_cache[dataset] = slices_data
                    total_cells = sum(s[1] for s in slices_data)
                    print(f"  [PRELOAD] {dataset}: {len(slices_data)} slices, {total_cells} cells")
            except Exception as e:
                print(f"  [PRELOAD] {dataset}: ERROR {e}")

    for method in method_list:
        method_dir = os.path.join(input_dir, method)
        if not os.path.isdir(method_dir):
            print(f"[SKIP] {method}: not found")
            continue

        print(f"\n--- {method} ({task}) ---")

        for dataset in dataset_list:
            # Skip ATAC-incompatible methods
            if method in METHOD_SKIP_ATAC and 'ATAC' in DATASET_MODALITY.get(dataset, ''):
                continue

            # Try aliases for directory name (e.g. HLN → Human_Lymph_Nodes)
            dataset_dir = None
            for alias in DATASET_DIR_ALIASES.get(dataset, [dataset]):
                candidate = os.path.join(method_dir, alias)
                if os.path.isdir(candidate):
                    dataset_dir = candidate
                    break
            if dataset_dir is None:
                continue

            h5ad_files = sorted(Path(dataset_dir).rglob('*.h5ad'))
            if not h5ad_files:
                continue

            for h5ad_path in h5ad_files:
                # Determine slice name
                if h5ad_path.parent.name != dataset:
                    slice_name = h5ad_path.parent.name
                else:
                    slice_name = h5ad_path.stem.split('_')[-1]

                try:
                    adata = sc.read_h5ad(str(h5ad_path))

                    # Get joint embedding
                    embedding = None
                    for key in [method, 'X_integrated', 'X_emb']:
                        if key in adata.obsm:
                            embedding = np.asarray(adata.obsm[key])
                            break
                    if embedding is None:
                        continue

                    if task == 'vertical':
                        # --- Vertical: per-slice, single modality load ---
                        modalities = load_modality_data_vertical(root_dir, dataset, slice_name)
                        if len(modalities) < 1:
                            print(f"  [{dataset}/{slice_name}] SKIP (no modality data)")
                            continue

                        n_embed = embedding.shape[0]
                        min_n = n_embed
                        for mod_data in modalities.values():
                            min_n = min(min_n, mod_data.shape[0])
                        if min_n < n_embed:
                            embedding = embedding[:min_n]
                        modalities = {k: v[:min_n] for k, v in modalities.items()}

                        score, details = evaluator.compute_cm_gtc(embedding, modalities)

                        row = {
                            'Method': method, 'Dataset': dataset,
                            'Slice': slice_name, 'Task': task,
                            'Modality_Type': DATASET_MODALITY.get(dataset, ''),
                            'N_Cells': min_n,
                            'Embedding_Dim': embedding.shape[1],
                            'CM_GTC': score,
                        }
                        # Per-modality CM-GTC (single-modality, no min-aggregation)
                        mod_parts = []
                        for mod_name, mod_data in modalities.items():
                            try:
                                ms, _ = evaluator.compute_cm_gtc(embedding, {mod_name: mod_data})
                                row[f'CM_GTC_{mod_name}'] = ms
                                mod_parts.append(f'{mod_name}={ms:.3f}')
                            except Exception:
                                row[f'CM_GTC_{mod_name}'] = np.nan
                        all_rows.append(row)
                        print(f"  [{dataset}/{slice_name}] CM-GTC = {score:.4f} ({', '.join(mod_parts)})")

                    else:
                        # --- Horizontal: per-slice CM-GTC, then average ---
                        if dataset not in horiz_modality_cache:
                            print(f"  [{dataset}/{slice_name}] SKIP (no modality data)")
                            continue

                        slices_data = horiz_modality_cache[dataset]
                        n_embed = embedding.shape[0]

                        # Partition embedding by slice cell counts
                        offset = 0
                        slice_scores = []
                        for sl_name, sl_ncells, sl_mods in slices_data:
                            end = offset + sl_ncells
                            if end > n_embed:
                                end = n_embed
                            if offset >= n_embed:
                                break

                            emb_slice = embedding[offset:end]
                            n_slice = emb_slice.shape[0]

                            # Align modality data
                            aligned_mods = {}
                            for mk, mv in sl_mods.items():
                                aligned_mods[mk] = mv[:n_slice]

                            if aligned_mods and n_slice >= 10:
                                try:
                                    s, d = evaluator.compute_cm_gtc(emb_slice, aligned_mods)
                                    slice_entry = {'global': s}
                                    for mk, mv in aligned_mods.items():
                                        try:
                                            ms, _ = evaluator.compute_cm_gtc(emb_slice, {mk: mv})
                                            slice_entry[mk] = ms
                                        except Exception:
                                            pass
                                    slice_scores.append(slice_entry)
                                except Exception:
                                    pass

                            offset = end

                        if slice_scores:
                            avg_score = np.mean([s['global'] for s in slice_scores])
                            row = {
                                'Method': method, 'Dataset': dataset,
                                'Slice': 'horizontal', 'Task': task,
                                'Modality_Type': DATASET_MODALITY.get(dataset, ''),
                                'N_Cells': n_embed,
                                'Embedding_Dim': embedding.shape[1],
                                'CM_GTC': avg_score,
                                'N_Slices_Scored': len(slice_scores),
                            }
                            # Per-modality averages
                            mod_parts = []
                            all_mod_keys = set()
                            for s in slice_scores:
                                all_mod_keys.update(k for k in s if k != 'global')
                            for mk in sorted(all_mod_keys):
                                vals = [s[mk] for s in slice_scores if mk in s]
                                if vals:
                                    row[f'CM_GTC_{mk}'] = np.mean(vals)
                                    mod_parts.append(f'{mk}={np.mean(vals):.3f}')
                            all_rows.append(row)
                            print(f"  [{dataset}/{slice_name}] CM-GTC = {avg_score:.4f} "
                                  f"({', '.join(mod_parts)}, avg of {len(slice_scores)} slices)")
                        else:
                            print(f"  [{dataset}/{slice_name}] SKIP (no valid slices for CM-GTC)")

                except Exception as e:
                    print(f"  [{dataset}/{slice_name}] ERROR: {e}")
                    continue

            if test_mode:
                break
        if test_mode:
            break

    # Save results
    if all_rows:
        df = pd.DataFrame(all_rows)
        out_path = os.path.join(output_dir, f'cmgtc_{task}.csv')
        df.to_csv(out_path, index=False)
        print(f"\nSaved {len(df)} CM-GTC results to: {out_path}")

        # Print summary
        summary = df.groupby('Method')['CM_GTC'].agg(['mean', 'std', 'count'])
        print(f"\n{task.upper()} CM-GTC Summary:")
        print(summary.sort_values('mean', ascending=False).to_string())

    return all_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    root = os.path.abspath(args.root)

    CMGTC_cls = setup_cmgtc_import(root, args.cmgtc_path)

    print(f"Root:    {root}")
    print(f"Task:    {args.task}")
    print(f"Test:    {args.test}")

    total_start = time.time()
    all_results = []

    if args.task in ('vertical', 'both'):
        results = compute_cmgtc_for_task(
            CMGTC_cls, root, 'vertical',
            args.methods, args.datasets, args.test
        )
        all_results.extend(results)

    if args.task in ('horizontal', 'both'):
        results = compute_cmgtc_for_task(
            CMGTC_cls, root, 'horizontal',
            args.methods, args.datasets, args.test
        )
        all_results.extend(results)

    elapsed = time.time() - total_start
    print(f"\nTotal: {len(all_results)} CM-GTC scores computed in {elapsed/60:.1f} min")


if __name__ == '__main__':
    main()
