#!/usr/bin/env python3
"""
CM-GTC per-modality for a SINGLE dataset. For parallel execution.

Usage:
    python compute_cmgtc_mosaic_single.py --root . --dataset HLN
"""

import os, sys, argparse, numpy as np, pandas as pd, scanpy as sc
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')

def setup_cmgtc():
    p = '/home/users/nus/e1724738/_main/_private/NUS/_Proj1/storage/_2_metric'
    if p not in sys.path:
        sys.path.insert(0, p)
    from cm_gtc_v2 import CMGTC_v2 as CMGTC
    return CMGTC

DATASET_CONFIG = {
    'HLN':          {'full': 'Human_Lymph_Nodes', 'mod': 'RNA_ADT',  'gt': 'withGT',
                     'slices': ['A1', 'D1']},
    'HT':           {'full': 'Human_Tonsils',     'mod': 'RNA_ADT',  'gt': 'withGT',
                     'slices': ['S1', 'S2', 'S3']},
    'Mouse_Spleen': {'full': 'Mouse_Spleen',      'mod': 'RNA_ADT',  'gt': 'woGT',
                     'slices': ['Mouse_Spleen1', 'Mouse_Spleen2']},
    'Mouse_Thymus': {'full': 'Mouse_Thymus',       'mod': 'RNA_ADT',  'gt': 'woGT',
                     'slices': ['Mouse_Thymus1', 'Mouse_Thymus2', 'Mouse_Thymus3', 'Mouse_Thymus4']},
    'MISAR_S1':     {'full': 'Mouse_Embryos_S1',  'mod': 'RNA_ATAC', 'gt': 'withGT',
                     'slices': ['E11', 'E13', 'E15', 'E18']},
    'MISAR_S2':     {'full': 'Mouse_Embryos_S2',  'mod': 'RNA_ATAC', 'gt': 'withGT',
                     'slices': ['E11', 'E13', 'E15', 'E18']},
    'Mouse_Brain':  {'full': 'Mouse_Brain',        'mod': 'RNA_ATAC', 'gt': 'woGT',
                     'slices': ['Mouse_Brain_ATAC', 'Mouse_Brain_H3K4me3',
                                'Mouse_Brain_H3K27ac', 'Mouse_Brain_H3K27me3']},
}
SCENARIOS = ['without_rna', 'without_second']

def to_dense(X):
    return np.asarray(X.todense()) if sparse.issparse(X) else np.asarray(X)

def _preprocess_rna(adata):
    adata = adata.copy()
    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    n_hvg = min(3000, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor='seurat_v3')
    adata = adata[:, adata.var['highly_variable']].copy()
    return to_dense(adata.X)

def _preprocess_adt(adata):
    adata = adata.copy()
    X = to_dense(adata.X).astype(np.float64)
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
    X = to_dense(adata.X)
    tf = X / (X.sum(axis=1, keepdims=True) + 1e-8)
    idf = np.log1p(X.shape[0] / (X.sum(axis=0, keepdims=True) + 1e-8))
    tfidf = (tf * idf).astype(np.float32)
    n_comps = min(51, tfidf.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comps, random_state=42)
    lsi = svd.fit_transform(tfidf)
    return lsi[:, 1:].astype(np.float32)

def find_sec_file(d, mod):
    for n in [f'adata_{mod}.h5ad', 'adata_peaks_normalized.h5ad']:
        p = os.path.join(d, n)
        if os.path.isfile(p): return p
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    ds_key = args.dataset
    cfg = DATASET_CONFIG[ds_key]
    slices = cfg['slices']
    n_slices = len(slices)
    sec_mod = 'ADT' if 'ADT' in cfg['mod'] else 'ATAC'
    sec_key = sec_mod.lower()

    CMGTC = setup_cmgtc()
    evaluator = CMGTC(similarity_metric='cosine', correlation_metric='spearman',
                       aggregation_strategy='min', verbose=False)

    mosaic_dir = os.path.join(root, '_myx_Results', 'adata', 'mosaic_integration', 'SpaMosaic')
    out_dir = os.path.join(root, '_myx_Results', 'evaluation', 'mosaic')
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for scenario in SCENARIOS:
        h5ad = os.path.join(mosaic_dir, ds_key, scenario, f'SpaMosaic_{ds_key}_{scenario}.h5ad')
        if not os.path.isfile(h5ad):
            print(f"SKIP {ds_key}/{scenario}")
            continue

        print(f"\n=== {ds_key} / {scenario} ===")
        adata = sc.read_h5ad(h5ad)
        emb = np.asarray(adata.obsm.get('SpaMosaic', adata.obsm.get('merged_emb')))

        # batch counts
        counts = []
        if 'src' in adata.obs.columns:
            for b in adata.obs['src'].unique():
                counts.append(int((adata.obs['src'] == b).sum()))
        if len(counts) != n_slices:
            per = adata.n_obs // n_slices
            counts = [per] * n_slices
            counts[-1] = adata.n_obs - sum(counts[:-1])

        offset = 0
        for si, (sl, nc) in enumerate(zip(slices, counts)):
            end = min(offset + nc, emb.shape[0])
            emb_sl = emb[offset:end]
            n = emb_sl.shape[0]
            is_query = (si == n_slices - 1)
            role = 'query' if is_query else 'bridge'
            missing = ('rna' if scenario == 'without_rna' else sec_key) if is_query else 'none'

            # Load ALL modalities (including hidden)
            if cfg['gt'] == 'withGT':
                base = os.path.join(root, 'Dataset', 'withGT', cfg['mod'], cfg['full'], sl)
            else:
                base = os.path.join(root, 'Dataset', 'woGT', cfg['mod'], cfg['full'], sl)

            mods = {}
            rna_p = os.path.join(base, 'adata_RNA.h5ad')
            if os.path.isfile(rna_p):
                mods['rna'] = _preprocess_rna(sc.read_h5ad(rna_p))
            sec_p = find_sec_file(base, sec_mod)
            if sec_p:
                ad_s = sc.read_h5ad(sec_p)
                mods[sec_key] = _preprocess_adt(ad_s) if sec_mod == 'ADT' else _preprocess_atac(ad_s)

            if n < 10 or not mods:
                print(f"  {sl} ({role}): SKIP")
                offset = end
                continue

            row = {'Dataset': ds_key, 'Scenario': scenario, 'Slice': sl,
                   'Role': role, 'Missing_Mod': missing, 'N_Cells': n}

            for mk, mv in mods.items():
                try:
                    s, _ = evaluator.compute_cm_gtc(emb_sl, {mk: mv[:n]})
                    row[f'CM_GTC_{mk}'] = s
                    tag = ' ★HIDDEN' if (is_query and mk == missing) else ''
                    print(f"  {sl} ({role}) {mk}: {s:.4f}{tag}")
                except Exception as e:
                    row[f'CM_GTC_{mk}'] = np.nan
                    print(f"  {sl} ({role}) {mk}: ERR {e}")

            # global
            try:
                sg, _ = evaluator.compute_cm_gtc(emb_sl, {k: v[:n] for k, v in mods.items()})
                row['CM_GTC_global'] = sg
            except:
                row['CM_GTC_global'] = np.nan

            rows.append(row)
            offset = end

    if rows:
        df = pd.DataFrame(rows)
        out_csv = os.path.join(out_dir, f'mosaic_cmgtc_{ds_key}.csv')
        df.to_csv(out_csv, index=False)
        print(f"\nSaved: {out_csv}")
        print(df.to_string(index=False))

if __name__ == '__main__':
    main()
