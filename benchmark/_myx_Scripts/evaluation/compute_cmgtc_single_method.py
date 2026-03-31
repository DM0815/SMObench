#!/usr/bin/env python3
"""Compute vertical CM-GTC for a single method. Output to method-specific CSV."""
import os, sys, argparse, numpy as np, pandas as pd, scanpy as sc
from pathlib import Path
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/users/nus/e1724738/_main/_private/NUS/_Proj1/storage/_2_metric')
from cm_gtc_v2 import CMGTC_v2 as CMGTC

ALL_DATASETS = ['Human_Lymph_Nodes','Human_Tonsils','Mouse_Embryos_S1','Mouse_Embryos_S2',
                'Mouse_Thymus','Mouse_Spleen','Mouse_Brain']
DATASET_MODALITY = {
    'Human_Lymph_Nodes':'RNA_ADT','Human_Tonsils':'RNA_ADT',
    'Mouse_Embryos_S1':'RNA_ATAC','Mouse_Embryos_S2':'RNA_ATAC',
    'Mouse_Thymus':'RNA_ADT','Mouse_Spleen':'RNA_ADT','Mouse_Brain':'RNA_ATAC',
}
DATASET_DIR_INFO = {
    'Human_Lymph_Nodes':{'gt':'withGT'},'Human_Tonsils':{'gt':'withGT'},
    'Mouse_Embryos_S1':{'gt':'withGT'},'Mouse_Embryos_S2':{'gt':'withGT'},
    'Mouse_Thymus':{'gt':'woGT'},'Mouse_Spleen':{'gt':'woGT'},'Mouse_Brain':{'gt':'woGT'},
}
DATASET_DIR_ALIASES = {
    'Human_Lymph_Nodes':['Human_Lymph_Nodes','HLN'],'Human_Tonsils':['Human_Tonsils','HT'],
    'Mouse_Embryos_S1':['Mouse_Embryos_S1','MISAR_S1'],'Mouse_Embryos_S2':['Mouse_Embryos_S2','MISAR_S2'],
    'Mouse_Thymus':['Mouse_Thymus'],'Mouse_Spleen':['Mouse_Spleen'],'Mouse_Brain':['Mouse_Brain'],
}

def _to_dense(X):
    return np.asarray(X.todense()) if sparse.issparse(X) else np.asarray(X)

def _preprocess_rna(adata):
    adata = adata.copy()
    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    n = min(3000, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n, flavor='seurat_v3')
    return _to_dense(adata[:, adata.var['highly_variable']].X)

def _preprocess_adt(adata):
    X = _to_dense(adata.X).astype(np.float64)
    X = np.clip(X, 0, None) + 1
    return (np.log(X / np.exp(np.mean(np.log(X), axis=1, keepdims=True)))).astype(np.float32)

def _preprocess_atac(adata):
    from sklearn.decomposition import TruncatedSVD
    adata = adata.copy()
    adata.X = ((_to_dense(adata.X) if not sparse.issparse(adata.X) else np.asarray((adata.X > 0).todense())) > 0).astype(np.float32)
    sc.pp.filter_genes(adata, min_cells=1)
    n = min(5000, adata.n_vars)
    if adata.n_vars > n:
        sc.pp.highly_variable_genes(adata, n_top_genes=n, flavor='seurat_v3')
        adata = adata[:, adata.var['highly_variable']].copy()
    X = _to_dense(adata.X)
    tf = X / (X.sum(axis=1, keepdims=True) + 1e-8)
    idf = np.log1p(X.shape[0] / (X.sum(axis=0, keepdims=True) + 1e-8))
    tfidf = (tf * idf).astype(np.float32)
    nc = min(51, tfidf.shape[1] - 1)
    lsi = TruncatedSVD(n_components=nc, random_state=42).fit_transform(tfidf)
    return lsi[:, 1:].astype(np.float32)

def _resolve_dir(root, dataset, slice_name):
    info = DATASET_DIR_INFO[dataset]
    mod = DATASET_MODALITY[dataset]
    for gt in ['withGT', 'woGT']:
        base = os.path.join(root, 'Dataset', gt, mod, dataset)
        if not os.path.isdir(base): continue
        for cand in [slice_name, f'{dataset}{slice_name}', f'{dataset}_{slice_name}']:
            p = os.path.join(base, cand)
            if os.path.isdir(p): return p
        for d in sorted(os.listdir(base)):
            if d.endswith(slice_name) and os.path.isdir(os.path.join(base, d)):
                return os.path.join(base, d)
    return None

def _find_sec(d, mod):
    for n in [f'adata_{mod}.h5ad', 'adata_peaks_normalized.h5ad']:
        p = os.path.join(d, n)
        if os.path.isfile(p): return p
    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--method', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()

    root = os.path.abspath(args.root)
    method = args.method
    evaluator = CMGTC(similarity_metric='cosine', correlation_metric='spearman',
                       aggregation_strategy='min', verbose=False)
    input_dir = os.path.join(root, '_myx_Results', 'adata', 'vertical_integration')
    method_dir = os.path.join(input_dir, method)
    if not os.path.isdir(method_dir):
        print(f"SKIP: {method_dir} not found"); return

    rows = []
    for dataset in ALL_DATASETS:
        dd = None
        for alias in DATASET_DIR_ALIASES.get(dataset, [dataset]):
            c = os.path.join(method_dir, alias)
            if os.path.isdir(c): dd = c; break
        if not dd: continue

        for hp in sorted(Path(dd).rglob('*.h5ad')):
            sl = hp.parent.name if hp.parent.name != dataset else hp.stem.split('_')[-1]
            try:
                adata = sc.read_h5ad(str(hp))
                emb = None
                for k in [method, 'X_integrated', 'X_emb']:
                    if k in adata.obsm: emb = np.asarray(adata.obsm[k]); break
                if emb is None: continue

                base = _resolve_dir(root, dataset, sl)
                if not base: continue
                mod = DATASET_MODALITY[dataset]
                sec_mod = 'ADT' if 'ADT' in mod else 'ATAC'
                mods = {}
                rp = os.path.join(base, 'adata_RNA.h5ad')
                if os.path.isfile(rp): mods['rna'] = _preprocess_rna(sc.read_h5ad(rp))
                sp = _find_sec(base, sec_mod)
                if sp:
                    ad = sc.read_h5ad(sp)
                    mods[sec_mod.lower()] = _preprocess_adt(ad) if sec_mod=='ADT' else _preprocess_atac(ad)
                if not mods: continue

                mn = min(emb.shape[0], *[v.shape[0] for v in mods.values()])
                emb = emb[:mn]; mods = {k:v[:mn] for k,v in mods.items()}
                score, _ = evaluator.compute_cm_gtc(emb, mods)
                row = {'Method':method,'Dataset':dataset,'Slice':sl,'Task':'vertical',
                       'Modality_Type':mod,'N_Cells':mn,'Embedding_Dim':emb.shape[1],'CM_GTC':score}
                for mk,mv in mods.items():
                    try: ms,_=evaluator.compute_cm_gtc(emb,{mk:mv}); row[f'CM_GTC_{mk}']=ms
                    except: pass
                rows.append(row)
                print(f"  [{dataset}/{sl}] CM-GTC={score:.4f}")
            except Exception as e:
                print(f"  [{dataset}/{sl}] ERR: {e}")

    if rows:
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f"Saved {len(rows)} → {args.output}")

if __name__=='__main__': main()
