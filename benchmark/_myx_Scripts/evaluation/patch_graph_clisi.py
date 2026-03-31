#!/usr/bin/env python3
"""
Patch graph_clisi: recompute real cLISI values for all withGT evaluation CSVs.

The original Eval/src/demo.py falls back to constant 0.5 because pysal import
fails in compute_metric.py. This script extracts the LISI algorithm and
recomputes graph_clisi from the h5ad embeddings + ground truth labels.

Also fixes SpatialGlue CM-GTC duplicates in cmgtc_horizontal.csv.

Usage:
    python patch_graph_clisi.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.neighbors import NearestNeighbors

ROOT = '/data/projects/11003054/e1724738/_private/NUS/_Proj1/SMOBench-CLEAN'
RESULTS = os.path.join(ROOT, '_myx_Results')

VERTICAL_ADATA = os.path.join(RESULTS, 'adata', 'vertical_integration')
HORIZONTAL_ADATA = os.path.join(RESULTS, 'adata', 'horizontal_integration')
VERTICAL_EVAL = os.path.join(RESULTS, 'evaluation', 'vertical')
HORIZONTAL_EVAL = os.path.join(RESULTS, 'evaluation', 'horizontal')

DATASET_GT_INFO = {
    'Human_Lymph_Nodes': {'type': 'RNA_ADT',  'gt_dir': 'Human_Lymph_Nodes',
                          'slices': ['A1', 'D1']},
    'Human_Tonsils':     {'type': 'RNA_ADT',  'gt_dir': 'Human_Tonsils',
                          'slices': ['S1', 'S2', 'S3']},
    'Mouse_Embryos_S1':  {'type': 'RNA_ATAC', 'gt_dir': 'Mouse_Embryos_S1',
                          'slices': ['E11', 'E13', 'E15', 'E18']},
    'Mouse_Embryos_S2':  {'type': 'RNA_ATAC', 'gt_dir': 'Mouse_Embryos_S2',
                          'slices': ['E11', 'E13', 'E15', 'E18']},
}

WITH_GT = set(DATASET_GT_INFO.keys())

K_NEIGHBORS = 15
PERPLEXITY = 30.0


def _hbeta(distances, beta):
    P = np.exp(-distances * beta)
    sum_P = max(np.sum(P), 1e-12)
    H = np.log(sum_P) + beta * np.sum(distances * P) / sum_P
    P = P / sum_P
    return H, P


def _perplexity_search(distances, target_perplexity=30.0, tol=1e-5, max_iter=200):
    target_H = np.log(target_perplexity)
    beta = 1.0
    beta_min = -np.inf
    beta_max = np.inf
    H, P = _hbeta(distances, beta)
    for _ in range(max_iter):
        H_diff = H - target_H
        if abs(H_diff) < tol:
            break
        if H_diff > 0:
            beta_min = beta
            beta = beta * 2.0 if beta_max == np.inf else (beta + beta_max) / 2.0
        else:
            beta_max = beta
            beta = beta / 2.0 if beta_min == -np.inf else (beta + beta_min) / 2.0
        H, P = _hbeta(distances, beta)
    return P


def compute_lisi_score(knn_indices, knn_distances, labels, perplexity=30.0):
    n_cells = knn_indices.shape[0]
    n_labels = len(np.unique(labels))
    lisi = np.zeros(n_cells)
    for i in range(n_cells):
        dists = knn_distances[i]
        P = _perplexity_search(dists, target_perplexity=perplexity)
        label_probs = np.zeros(n_labels)
        for j, idx in enumerate(knn_indices[i]):
            if idx < len(labels):
                label_probs[labels[idx]] += P[j]
        simpson = np.sum(label_probs ** 2)
        lisi[i] = 1.0 / max(simpson, 1e-12)
    return lisi


def compute_graph_clisi(emb, cell_labels, k=K_NEIGHBORS, perplexity=PERPLEXITY):
    unique_labels = np.unique(cell_labels)
    n_labels = len(unique_labels)
    if n_labels <= 1:
        return 1.0
    label_map = {l: i for i, l in enumerate(unique_labels)}
    labels_int = np.array([label_map[l] for l in cell_labels])
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=-1)
    nn.fit(emb)
    knn_distances, knn_indices = nn.kneighbors(emb)
    lisi_values = compute_lisi_score(knn_indices, knn_distances, labels_int,
                                      perplexity=min(perplexity, k - 1))
    median_lisi = np.median(lisi_values)
    scaled = (n_labels - median_lisi) / (n_labels - 1) if n_labels > 1 else 1.0
    return np.clip(scaled, 0, 1)


def get_embedding(adata, method_name):
    candidates = [method_name, method_name.lower(), f'{method_name}_emb',
                  'X_emb', 'latent', 'X_pca']
    for key in candidates:
        if key in adata.obsm:
            emb = adata.obsm[key]
            if issparse(emb):
                emb = emb.toarray()
            return emb
    skip = {'X_spatial', 'spatial', 'X_umap'}
    for key in adata.obsm:
        if key not in skip:
            emb = adata.obsm[key]
            if issparse(emb):
                emb = emb.toarray()
            return emb
    return None


def load_ground_truth(dataset, slice_name):
    """Load Spatial_Label from the original dataset."""
    if dataset not in DATASET_GT_INFO:
        return None
    info = DATASET_GT_INFO[dataset]
    gt_path = os.path.join(ROOT, 'Dataset', 'withGT', info['type'],
                           info['gt_dir'], slice_name, 'adata_RNA.h5ad')
    try:
        import scanpy as sc
        adata_gt = sc.read_h5ad(gt_path)
        if 'Spatial_Label' in adata_gt.obs.columns:
            return adata_gt.obs['Spatial_Label'].values
    except Exception as e:
        print(f"    [GT] Failed to load {gt_path}: {e}")
    return None


def load_ground_truth_horizontal(adata, dataset):
    """Load GT for horizontal integration: try h5ad obs first, then concat from slices."""
    # 1) Check if horizontal h5ad already has Spatial_Label
    if hasattr(adata, 'obs') and 'Spatial_Label' in adata.obs.columns:
        print(f"    [GT] Found Spatial_Label in h5ad obs")
        return adata.obs['Spatial_Label'].values
    # 2) Concatenate from all slices, match by obs_names
    if dataset not in DATASET_GT_INFO:
        return None
    info = DATASET_GT_INFO[dataset]
    slices = info.get('slices', [])
    if not slices:
        return None
    import scanpy as sc
    all_labels = {}
    for sl in slices:
        gt_path = os.path.join(ROOT, 'Dataset', 'withGT', info['type'],
                               info['gt_dir'], sl, 'adata_RNA.h5ad')
        try:
            adata_gt = sc.read_h5ad(gt_path)
            if 'Spatial_Label' in adata_gt.obs.columns:
                for idx, label in zip(adata_gt.obs_names, adata_gt.obs['Spatial_Label']):
                    all_labels[idx] = label
        except Exception as e:
            print(f"    [GT] Failed to load {gt_path}: {e}")
    if not all_labels:
        return None
    # Try to match by obs_names
    matched = []
    n_matched = 0
    for cell_id in adata.obs_names:
        if cell_id in all_labels:
            matched.append(all_labels[cell_id])
            n_matched += 1
        else:
            matched.append(None)
    if n_matched == 0:
        # obs_names don't match; fall back to positional concat
        print(f"    [GT] No obs_name match, trying positional concat...")
        concat_labels = []
        for sl in slices:
            gt_path = os.path.join(ROOT, 'Dataset', 'withGT', info['type'],
                                   info['gt_dir'], sl, 'adata_RNA.h5ad')
            try:
                adata_gt = sc.read_h5ad(gt_path)
                if 'Spatial_Label' in adata_gt.obs.columns:
                    concat_labels.extend(adata_gt.obs['Spatial_Label'].values.tolist())
            except:
                pass
        if len(concat_labels) == adata.n_obs:
            print(f"    [GT] Positional concat matched: {len(concat_labels)} cells")
            return np.array(concat_labels)
        elif abs(len(concat_labels) - adata.n_obs) <= 5:
            # Tolerate small mismatch (method may drop a few cells during integration)
            n_use = min(len(concat_labels), adata.n_obs)
            print(f"    [GT] Positional concat approx match: concat={len(concat_labels)}, "
                  f"h5ad={adata.n_obs}, using first {n_use}")
            return np.array(concat_labels[:n_use])
        else:
            print(f"    [GT] Length mismatch: concat={len(concat_labels)}, h5ad={adata.n_obs}")
            return None
    print(f"    [GT] Matched {n_matched}/{adata.n_obs} cells by obs_names")
    return np.array(matched)


_clisi_cache = {}


def _get_clisi(adata_path, method_name, dataset, slice_name):
    cache_key = (adata_path, dataset, slice_name)
    if cache_key in _clisi_cache:
        return _clisi_cache[cache_key]
    # Load embedding from method output
    try:
        import scanpy as sc
        adata = sc.read_h5ad(adata_path)
    except Exception as e:
        print(f"    [WARN] Cannot read {adata_path}: {e}")
        _clisi_cache[cache_key] = None
        return None
    emb = get_embedding(adata, method_name)
    if emb is None:
        print(f"    [WARN] No embedding in {adata_path}, keys: {list(adata.obsm.keys())}")
        _clisi_cache[cache_key] = None
        return None
    # Load ground truth from original dataset
    labels = load_ground_truth(dataset, slice_name)
    if labels is None:
        print(f"    [WARN] No GT for {dataset}/{slice_name}")
        _clisi_cache[cache_key] = None
        return None
    # Align lengths
    min_len = min(len(labels), emb.shape[0])
    if len(labels) != emb.shape[0]:
        print(f"    [WARN] Length mismatch: GT={len(labels)}, emb={emb.shape[0]}, using {min_len}")
        labels = labels[:min_len]
        emb = emb[:min_len]
    mask = pd.notna(labels)
    if mask.sum() < 10:
        print(f"    [WARN] Too few valid labels ({mask.sum()})")
        _clisi_cache[cache_key] = None
        return None
    emb_clean = emb[mask]
    labels_clean = np.array(labels[mask])
    print(f"    Computing cLISI: {emb_clean.shape[0]} cells, {len(np.unique(labels_clean))} types...")
    score = compute_graph_clisi(emb_clean, labels_clean)
    print(f"    cLISI = {score:.4f}")
    _clisi_cache[cache_key] = score
    return score


def _update_scores(df, new_clisi):
    """Recompute BVC_Score and Total_Score after patching graph_clisi."""
    def _get_val(metric):
        row = df[df['Metric'] == metric]
        return float(row['Value'].iloc[0]) if len(row) else None
    ari = _get_val('ARI')
    nmi = _get_val('NMI')
    asw = _get_val('asw_celltype')
    sc = _get_val('SC_Score')
    if all(v is not None for v in [ari, nmi, asw]):
        bvc = np.mean([ari, nmi, asw, new_clisi])
        df.loc[df['Metric'] == 'BVC_Score', 'Value'] = bvc
        if sc is not None:
            df.loc[df['Metric'] == 'Total_Score', 'Value'] = np.mean([sc, bvc])


def find_adata_file(base_dir, method, dataset, slice_name=None):
    ds_names = [dataset]
    for ds in ds_names:
        patterns = []
        if slice_name:
            patterns.append(os.path.join(base_dir, method, ds, slice_name, '*.h5ad'))
            patterns.append(os.path.join(base_dir, method, ds, f'*{slice_name}*.h5ad'))
        patterns.append(os.path.join(base_dir, method, ds, '*.h5ad'))
        # Also search one level deeper (e.g. METHOD/DATASET/fusion/*.h5ad)
        patterns.append(os.path.join(base_dir, method, ds, '*', '*.h5ad'))
        patterns.append(os.path.join(base_dir, method, f'*{ds}*.h5ad'))
        for pat in patterns:
            matches = glob.glob(pat)
            if matches:
                return matches[0]
    return None


def _extract_slice_from_csv(csv_basename, method, dataset):
    """Extract slice name from CSV filename like CANDIES_Human_Lymph_Nodes_A1_leiden_withGT.csv"""
    prefix = f"{method}_{dataset}_"
    if csv_basename.startswith(prefix):
        rest = csv_basename[len(prefix):]
        # rest = "A1_leiden_withGT.csv" -> slice = "A1"
        parts = rest.split('_')
        if len(parts) >= 2:
            return parts[0]
    return None


def patch_vertical():
    print("\n" + "=" * 60)
    print("Patching VERTICAL graph_clisi...")
    print("=" * 60)
    patched = 0
    skipped = 0
    for method_dir in sorted(os.listdir(VERTICAL_EVAL)):
        method_path = os.path.join(VERTICAL_EVAL, method_dir)
        if not os.path.isdir(method_path):
            continue
        for ds_dir in sorted(os.listdir(method_path)):
            ds_path = os.path.join(method_path, ds_dir)
            if not os.path.isdir(ds_path):
                continue
            if ds_dir not in WITH_GT:
                continue
            csv_files = glob.glob(os.path.join(ds_path, '*.csv'))
            if not csv_files:
                continue
            # Group CSVs by slice name
            slice_csvs = {}
            for cf in csv_files:
                sl = _extract_slice_from_csv(os.path.basename(cf), method_dir, ds_dir)
                if sl:
                    slice_csvs.setdefault(sl, []).append(cf)
                else:
                    slice_csvs.setdefault(None, []).append(cf)
            for sl, csvs in slice_csvs.items():
                if sl is None:
                    skipped += 1
                    continue
                adata_file = find_adata_file(VERTICAL_ADATA, method_dir, ds_dir, sl)
                if not adata_file:
                    print(f"  [SKIP] No h5ad for {method_dir}/{ds_dir}/{sl}")
                    skipped += 1
                    continue
                clisi = _get_clisi(adata_file, method_dir, ds_dir, sl)
                if clisi is None:
                    skipped += 1
                    continue
                for csv_file in csvs:
                    try:
                        df = pd.read_csv(csv_file)
                        if 'Metric' in df.columns and 'Value' in df.columns:
                            mask = df['Metric'] == 'graph_clisi'
                            if mask.any():
                                old_val = df.loc[mask, 'Value'].iloc[0]
                                df.loc[mask, 'Value'] = clisi
                                # Also update BVC_Score and Total_Score
                                _update_scores(df, clisi)
                                df.to_csv(csv_file, index=False)
                                print(f"  Patched {os.path.basename(csv_file)}: {old_val} -> {clisi:.4f}")
                                patched += 1
                        elif 'graph_clisi' in df.columns:
                            old_val = df['graph_clisi'].iloc[0] if len(df) > 0 else 'N/A'
                            df['graph_clisi'] = clisi
                            df.to_csv(csv_file, index=False)
                            patched += 1
                    except Exception as e:
                        print(f"  [ERROR] {csv_file}: {e}")
    print(f"\nVertical: patched {patched} CSVs, skipped {skipped}")


def _get_clisi_horizontal(adata_path, method_name, dataset):
    """Compute cLISI for horizontal integration (all slices merged)."""
    cache_key = (adata_path, dataset, '_horizontal')
    if cache_key in _clisi_cache:
        return _clisi_cache[cache_key]
    try:
        import scanpy as sc
        adata = sc.read_h5ad(adata_path)
    except Exception as e:
        print(f"    [WARN] Cannot read {adata_path}: {e}")
        _clisi_cache[cache_key] = None
        return None
    emb = get_embedding(adata, method_name)
    if emb is None:
        print(f"    [WARN] No embedding in {adata_path}, keys: {list(adata.obsm.keys())}")
        _clisi_cache[cache_key] = None
        return None
    labels = load_ground_truth_horizontal(adata, dataset)
    if labels is None:
        print(f"    [WARN] No GT for horizontal {dataset}")
        _clisi_cache[cache_key] = None
        return None
    # Handle None entries from partial obs_name matching
    mask = pd.notna(labels)
    if hasattr(labels[0], '__eq__'):
        mask = np.array([l is not None and pd.notna(l) for l in labels])
    if mask.sum() < 10:
        print(f"    [WARN] Too few valid labels ({mask.sum()})")
        _clisi_cache[cache_key] = None
        return None
    emb_clean = emb[mask]
    labels_clean = np.array([str(l) for l, m in zip(labels, mask) if m])
    print(f"    Computing cLISI: {emb_clean.shape[0]} cells, {len(np.unique(labels_clean))} types...")
    score = compute_graph_clisi(emb_clean, labels_clean)
    print(f"    cLISI = {score:.4f}")
    _clisi_cache[cache_key] = score
    return score


def _parse_horizontal_dataset(csv_basename, method):
    """Extract dataset from horizontal CSV filename like METHOD_DATASET_horizontal_clustering_withGT.csv"""
    prefix = f"{method}_"
    if not csv_basename.startswith(prefix):
        return None
    rest = csv_basename[len(prefix):]
    idx = rest.find('_horizontal_')
    if idx <= 0:
        return None
    return rest[:idx]


def patch_horizontal():
    print("\n" + "=" * 60)
    print("Patching HORIZONTAL graph_clisi...")
    print("=" * 60)
    patched = 0
    skipped = 0
    for method_dir in sorted(os.listdir(HORIZONTAL_EVAL)):
        method_path = os.path.join(HORIZONTAL_EVAL, method_dir)
        if not os.path.isdir(method_path):
            continue
        # Horizontal CSVs are flat under method_path (no dataset subdirs)
        csv_files = glob.glob(os.path.join(method_path, '*_withGT.csv'))
        if not csv_files:
            continue
        # Group CSVs by dataset
        dataset_csvs = {}
        for cf in csv_files:
            ds = _parse_horizontal_dataset(os.path.basename(cf), method_dir)
            if ds and ds in WITH_GT:
                dataset_csvs.setdefault(ds, []).append(cf)
        for dataset, csvs in sorted(dataset_csvs.items()):
            adata_file = find_adata_file(HORIZONTAL_ADATA, method_dir, dataset)
            if not adata_file:
                print(f"  [SKIP] No h5ad for {method_dir}/{dataset}")
                skipped += 1
                continue
            clisi = _get_clisi_horizontal(adata_file, method_dir, dataset)
            if clisi is None:
                skipped += 1
                continue
            for csv_file in csvs:
                try:
                    df = pd.read_csv(csv_file)
                    if 'Metric' in df.columns and 'Value' in df.columns:
                        mask = df['Metric'] == 'graph_clisi'
                        if mask.any():
                            old_val = df.loc[mask, 'Value'].iloc[0]
                            df.loc[mask, 'Value'] = clisi
                            _update_scores(df, clisi)
                            df.to_csv(csv_file, index=False)
                            print(f"  Patched {os.path.basename(csv_file)}: {old_val} -> {clisi:.4f}")
                            patched += 1
                except Exception as e:
                    print(f"  [ERROR] {csv_file}: {e}")
    print(f"\nHorizontal: patched {patched} CSVs, skipped {skipped}")


def fix_cmgtc_dup():
    print("\n" + "=" * 60)
    print("Fixing SpatialGlue CM-GTC duplicates...")
    print("=" * 60)
    csv_path = os.path.join(RESULTS, 'evaluation', 'cmgtc', 'cmgtc_horizontal.csv')
    if not os.path.isfile(csv_path):
        print(f"  [SKIP] {csv_path} not found")
        return
    df = pd.read_csv(csv_path)
    n_before = len(df)
    df = df.drop_duplicates(subset=['Method', 'Dataset'], keep='first')
    n_after = len(df)
    if n_before != n_after:
        df.to_csv(csv_path, index=False)
        print(f"  Removed {n_before - n_after} duplicates ({n_before} -> {n_after})")
    else:
        print(f"  No duplicates found ({n_before} rows)")


def main():
    import time
    t0 = time.time()
    fix_cmgtc_dup()
    patch_vertical()
    patch_horizontal()
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"All done in {elapsed / 60:.1f} minutes")


def main_horizontal_only():
    """Run only horizontal patch (vertical already done)."""
    import time
    t0 = time.time()
    patch_horizontal()
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Horizontal patch done in {elapsed / 60:.1f} minutes")


if __name__ == '__main__':
    import sys
    if '--horizontal-only' in sys.argv:
        main_horizontal_only()
    else:
        main()
