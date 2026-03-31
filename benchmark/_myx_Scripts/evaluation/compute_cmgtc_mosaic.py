#!/usr/bin/env python3
"""
CM-GTC Per-Modality Analysis for SpaMosaic Mosaic Integration

Core idea: After mosaic integration, the query batch never saw one modality.
We use the HIDDEN (ground-truth) modality data to measure whether SpaMosaic
successfully transferred topological information from bridge batches.

Output per (dataset, scenario, slice):
  - CM-GTC_RNA: topology preservation for RNA
  - CM-GTC_second: topology preservation for ADT/ATAC
  - role: bridge (had all modalities) or query (missing one)
  - missing_mod: which modality was hidden from SpaMosaic

Usage:
    python compute_cmgtc_mosaic.py --root /path/to/SMOBench-CLEAN
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')


def setup_cmgtc():
    candidates = [
        '/home/users/nus/e1724738/_main/_private/NUS/_Proj1/storage/_2_metric',
    ]
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, 'cm_gtc.py')):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            from cm_gtc import CMGTC
            return CMGTC
    raise FileNotFoundError("Cannot find cm_gtc.py")


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
    if sparse.issparse(X):
        return np.asarray(X.todense())
    return np.asarray(X)


def find_second_mod_file(base_dir, second_mod):
    for name in [f'adata_{second_mod}.h5ad', 'adata_peaks_normalized.h5ad']:
        p = os.path.join(base_dir, name)
        if os.path.isfile(p):
            return p
    return None


def load_all_modalities(root, ds_key, slice_name):
    """Load ALL modality data for a slice (including 'hidden' ones).
    Always loads both modalities regardless of mosaic scenario."""
    cfg = DATASET_CONFIG[ds_key]
    mod_type = cfg['mod']
    gt_type = cfg['gt']
    second_mod = 'ADT' if 'ADT' in mod_type else 'ATAC'

    if gt_type == 'withGT':
        base = os.path.join(root, 'Dataset', 'withGT', mod_type, cfg['full'], slice_name)
    else:
        base = os.path.join(root, 'Dataset', 'woGT', mod_type, cfg['full'], slice_name)

    mods = {}
    rna_path = os.path.join(base, 'adata_RNA.h5ad')
    if os.path.isfile(rna_path):
        mods['rna'] = to_dense(sc.read_h5ad(rna_path).X)

    sec_path = find_second_mod_file(base, second_mod)
    if sec_path:
        mods[second_mod.lower()] = to_dense(sc.read_h5ad(sec_path).X)

    return mods, second_mod.lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    CMGTC = setup_cmgtc()

    mosaic_dir = os.path.join(root, '_myx_Results', 'adata', 'mosaic_integration', 'SpaMosaic')
    output_dir = os.path.join(root, '_myx_Results', 'evaluation', 'mosaic')
    os.makedirs(output_dir, exist_ok=True)

    all_rows = []

    for ds_key, cfg in DATASET_CONFIG.items():
        slices = cfg['slices']
        n_slices = len(slices)
        second_mod_name = 'adt' if 'ADT' in cfg['mod'] else 'atac'

        for scenario in SCENARIOS:
            h5ad_path = os.path.join(mosaic_dir, ds_key, scenario,
                                      f'SpaMosaic_{ds_key}_{scenario}.h5ad')
            if not os.path.isfile(h5ad_path):
                print(f"[SKIP] {ds_key}/{scenario}")
                continue

            print(f"\n{'='*60}")
            print(f"{ds_key} / {scenario}")
            print(f"{'='*60}")

            adata = sc.read_h5ad(h5ad_path)
            embedding = np.asarray(adata.obsm.get('SpaMosaic', adata.obsm.get('merged_emb')))

            # Get per-batch cell counts
            batch_counts = []
            if 'src' in adata.obs.columns:
                for b in adata.obs['src'].unique():
                    batch_counts.append((adata.obs['src'] == b).sum())

            if len(batch_counts) != n_slices:
                cell_per = adata.n_obs // n_slices
                batch_counts = [cell_per] * n_slices
                batch_counts[-1] = adata.n_obs - sum(batch_counts[:-1])

            # Per-slice, per-modality CM-GTC
            offset = 0
            for si, (sl_name, sl_ncells) in enumerate(zip(slices, batch_counts)):
                end = min(offset + sl_ncells, embedding.shape[0])
                emb_slice = embedding[offset:end]
                n_slice = emb_slice.shape[0]

                is_query = (si == n_slices - 1)
                role = 'query' if is_query else 'bridge'

                if scenario == 'without_rna':
                    missing_mod = 'rna' if is_query else 'none'
                else:
                    missing_mod = second_mod_name if is_query else 'none'

                # Load ALL modalities (including hidden ones!)
                all_mods, sec_name = load_all_modalities(root, ds_key, sl_name)

                if n_slice < 10 or not all_mods:
                    print(f"  {sl_name} ({role}): SKIP")
                    offset = end
                    continue

                row = {
                    'Dataset': ds_key,
                    'Dataset_Full': cfg['full'],
                    'Scenario': scenario,
                    'Slice': sl_name,
                    'Role': role,
                    'Missing_Mod': missing_mod,
                    'N_Cells': n_slice,
                }

                # Compute CM-GTC separately for each modality
                evaluator = CMGTC(
                    similarity_metric='cosine',
                    correlation_metric='spearman',
                    aggregation_strategy='min',
                    verbose=False,
                )

                for mod_name, mod_data in all_mods.items():
                    aligned_data = mod_data[:n_slice]
                    try:
                        score, _ = evaluator.compute_cm_gtc(
                            emb_slice, {mod_name: aligned_data})
                        row[f'CM_GTC_{mod_name}'] = score

                        # Mark if this modality was hidden from SpaMosaic
                        was_hidden = (is_query and mod_name == missing_mod.replace('second', sec_name))
                        tag = ' ★ HIDDEN' if (is_query and mod_name == missing_mod) else ''
                        print(f"  {sl_name} ({role}) CM-GTC_{mod_name} = {score:.4f}{tag}")
                    except Exception as e:
                        row[f'CM_GTC_{mod_name}'] = np.nan
                        print(f"  {sl_name} ({role}) CM-GTC_{mod_name}: ERROR {e}")

                # Also compute min-aggregated global score
                try:
                    score_global, _ = evaluator.compute_cm_gtc(
                        emb_slice,
                        {k: v[:n_slice] for k, v in all_mods.items()})
                    row['CM_GTC_global'] = score_global
                except Exception:
                    row['CM_GTC_global'] = np.nan

                all_rows.append(row)
                offset = end

    if all_rows:
        df = pd.DataFrame(all_rows)

        # Save detailed per-slice results
        detail_csv = os.path.join(output_dir, 'mosaic_cmgtc_per_slice.csv')
        df.to_csv(detail_csv, index=False)
        print(f"\nDetailed: {detail_csv}")

        # Summary: average per (dataset, scenario), split by role
        print(f"\n{'='*70}")
        print("SUMMARY: Bridge vs Query CM-GTC (per-modality)")
        print(f"{'='*70}")

        for scenario in SCENARIOS:
            df_sc = df[df['Scenario'] == scenario]
            if df_sc.empty:
                continue
            print(f"\n--- {scenario} ---")
            mod_cols = [c for c in df.columns if c.startswith('CM_GTC_') and c != 'CM_GTC_global']
            for ds_key in DATASET_CONFIG:
                df_ds = df_sc[df_sc['Dataset'] == ds_key]
                if df_ds.empty:
                    continue
                bridge = df_ds[df_ds['Role'] == 'bridge']
                query = df_ds[df_ds['Role'] == 'query']
                parts = [f"  {ds_key}:"]
                for mc in mod_cols:
                    b_val = bridge[mc].mean() if not bridge.empty and mc in bridge else np.nan
                    q_val = query[mc].mean() if not query.empty and mc in query else np.nan
                    mod_label = mc.replace('CM_GTC_', '')
                    parts.append(f"    {mod_label}: bridge={b_val:.3f}, query={q_val:.3f}")
                print('\n'.join(parts))

        # Aggregated per (dataset, scenario) for bubble chart
        agg_rows = []
        for (ds, sc_name), grp in df.groupby(['Dataset', 'Scenario']):
            row = {'Dataset': ds, 'Scenario': sc_name}
            mod_cols = [c for c in df.columns if c.startswith('CM_GTC_') and c != 'CM_GTC_global']
            for mc in mod_cols:
                row[mc] = grp[mc].mean()
            row['CM_GTC_global'] = grp['CM_GTC_global'].mean()
            agg_rows.append(row)
        df_agg = pd.DataFrame(agg_rows)
        agg_csv = os.path.join(output_dir, 'mosaic_cmgtc.csv')
        df_agg.to_csv(agg_csv, index=False)
        print(f"\nAggregated: {agg_csv}")
        print(df_agg.to_string(index=False))


if __name__ == '__main__':
    main()
