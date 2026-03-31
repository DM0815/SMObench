#!/usr/bin/env python3
"""
Compute per-modality CM-GTC: joint→RNA and joint→ADT/ATAC separately.
Outputs CSV with CM_GTC_RNA, CM_GTC_ADT/ATAC, and CM_GTC (min).

Usage:
    python compute_cmgtc_per_modality.py --root /path/to/SMOBench-CLEAN
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

DATASET_MODALITY = {
    'Human_Lymph_Nodes': 'RNA_ADT', 'Human_Tonsils': 'RNA_ADT',
    'Mouse_Spleen': 'RNA_ADT', 'Mouse_Thymus': 'RNA_ADT',
    'Mouse_Embryos_S1': 'RNA_ATAC', 'Mouse_Embryos_S2': 'RNA_ATAC',
    'Mouse_Brain': 'RNA_ATAC',
}

METHODS = [
    'CANDIES', 'COSMOS', 'MISO', 'MultiGATE', 'PRAGA', 'PRESENT',
    'SMOPCA', 'SpaBalance', 'SpaFusion', 'SpaMI', 'SpaMosaic',
    'SpaMultiVAE', 'SpaMV', 'SpatialGlue', 'SWITCH',
]


def compute_topology_consistency(embedding, modality_data, n_sample=500):
    """Compute Spearman rank correlation between joint and modality pairwise distances."""
    N = embedding.shape[0]
    if N > n_sample:
        idx = np.random.choice(N, n_sample, replace=False)
        embedding = embedding[idx]
        modality_data = modality_data[idx]

    N = embedding.shape[0]
    # Cosine similarity
    sim_joint = cosine_similarity(embedding)
    sim_mod = cosine_similarity(modality_data)

    # Per-spot rank correlation
    scores = []
    for i in range(N):
        sj = sim_joint[i].copy()
        sm = sim_mod[i].copy()
        sj[i] = np.nan
        sm[i] = np.nan
        mask = ~(np.isnan(sj) | np.isnan(sm))
        if mask.sum() < 10:
            continue
        r, _ = spearmanr(sj[mask], sm[mask])
        if not np.isnan(r):
            scores.append(r)

    return max(0, np.mean(scores)) if scores else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = os.path.abspath(args.root)

    rows = []

    for method in METHODS:
        for dataset, mod_type in DATASET_MODALITY.items():
            adata_dir = os.path.join(root, '_myx_Results/adata/vertical_integration',
                                     method, dataset)
            if not os.path.isdir(adata_dir):
                continue

            h5ads = sorted(Path(adata_dir).rglob('*.h5ad'))
            for h5ad_path in h5ads:
                slice_name = h5ad_path.parent.name if h5ad_path.parent.name != dataset else h5ad_path.stem.split('_')[-1]

                try:
                    adata = sc.read_h5ad(str(h5ad_path))

                    # Find embedding
                    emb_key = None
                    for k in [method, 'X_integrated', 'X_emb']:
                        if k in adata.obsm:
                            emb_key = k
                            break
                    if emb_key is None:
                        continue

                    embedding = adata.obsm[emb_key]

                    # Load original modality data
                    if mod_type == 'RNA_ADT':
                        mod_names = ['RNA', 'ADT']
                    else:
                        mod_names = ['RNA', 'ATAC']

                    gt_base = os.path.join(root, 'Dataset/withGT', mod_type, dataset, slice_name)
                    if not os.path.isdir(gt_base):
                        # Try without slice
                        gt_base = os.path.join(root, 'Dataset/withGT', mod_type, dataset)
                        sub_dirs = sorted(os.listdir(gt_base)) if os.path.isdir(gt_base) else []
                        if sub_dirs and slice_name not in sub_dirs:
                            gt_base = os.path.join(gt_base, sub_dirs[0])
                        else:
                            gt_base = os.path.join(gt_base, slice_name)

                    mod_scores = {}
                    for mod_name in mod_names:
                        mod_path = os.path.join(gt_base, f'adata_{mod_name}.h5ad')
                        if not os.path.isfile(mod_path):
                            continue
                        adata_mod = sc.read_h5ad(mod_path)

                        # Align cell counts
                        n = min(embedding.shape[0], adata_mod.n_obs)
                        emb_sub = embedding[:n]

                        # Get feature matrix
                        if hasattr(adata_mod.X, 'toarray'):
                            X_mod = adata_mod.X[:n].toarray().astype(np.float32)
                        else:
                            X_mod = np.array(adata_mod.X[:n], dtype=np.float32)

                        # Simple preprocessing
                        X_mod = np.log1p(X_mod)

                        score = compute_topology_consistency(emb_sub, X_mod)
                        mod_scores[mod_name] = score

                    if len(mod_scores) == 2:
                        cmgtc_min = min(mod_scores.values())
                        row = {
                            'Method': method, 'Dataset': dataset, 'Slice': slice_name,
                            'Modality_Type': mod_type, 'N_Cells': embedding.shape[0],
                        }
                        for mod_name, score in mod_scores.items():
                            row[f'CM_GTC_{mod_name}'] = score
                        row['CM_GTC_min'] = cmgtc_min
                        rows.append(row)
                        print(f"  {method}/{dataset}/{slice_name}: "
                              + " | ".join(f"{k}={v:.4f}" for k, v in mod_scores.items())
                              + f" | min={cmgtc_min:.4f}")

                except Exception as e:
                    print(f"  ERROR {method}/{dataset}/{slice_name}: {e}")

    df = pd.DataFrame(rows)
    out = os.path.join(root, '_myx_Results/evaluation/cmgtc_per_modality.csv')
    df.to_csv(out, index=False)
    print(f"\nSaved: {out} ({len(df)} rows)")

    # Summary
    print("\n=== Summary by modality type ===")
    for mod_type in ['RNA_ADT', 'RNA_ATAC']:
        sub = df[df['Modality_Type'] == mod_type]
        if sub.empty:
            continue
        print(f"\n{mod_type}:")
        for col in [c for c in sub.columns if c.startswith('CM_GTC')]:
            print(f"  {col:15s}: mean={sub[col].mean():.4f}, median={sub[col].median():.4f}")


if __name__ == '__main__':
    main()
