#!/usr/bin/env python3
"""
Extended Data Fig 2: CM-GTC Weight Sensitivity Analysis
Sweeps CM-GTC weight from 0.05 to 0.50 and computes Kendall's tau
on method rankings to show ranking robustness.

Usage:
    python plot_edfig2_weight_sensitivity.py --root /path/to/SMOBench-CLEAN
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
import matplotlib.pyplot as plt
import sys as _sys
_sys.path.insert(0, __import__('os').path.dirname(__file__))
from style_config import apply_style, PAL13
apply_style()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--clustering', type=str, default='leiden')
    parser.add_argument('--dpi', type=int, default=300)
    return parser.parse_args()


# 2x2 dataset groups: modality × GT
DATASET_GROUPS = [
    ('RNA+ADT',  'withGT', ['Human_Lymph_Nodes', 'Human_Tonsils']),
    ('RNA+ADT',  'woGT',   ['Mouse_Thymus', 'Mouse_Spleen']),
    ('RNA+ATAC', 'withGT', ['Mouse_Embryos_S1', 'Mouse_Embryos_S2']),
    ('RNA+ATAC', 'woGT',   ['Mouse_Brain']),
]


WITHGT_DATASETS = ['Human_Lymph_Nodes', 'Human_Tonsils', 'Mouse_Embryos_S1', 'Mouse_Embryos_S2']


def load_data(root, task, clustering, gt_datasets=None):
    """Load final results with all metric columns, filtered by gt_datasets."""
    if gt_datasets is None:
        gt_datasets = WITHGT_DATASETS
    summary_dir = os.path.join(root, '_myx_Results', 'evaluation', 'summary')
    for prefix in ['final', 'detailed']:
        path = os.path.join(summary_dir, f'{task}_{prefix}_{clustering}.csv')
        if os.path.isfile(path):
            df = pd.read_csv(path)
            return df[df['Dataset'].isin(gt_datasets)].copy()
    return None


def compute_ranking_with_weight(df, task, cmgtc_weight):
    """Compute method rankings for a given CM-GTC weight."""
    method_scores = {}

    if task == 'vertical':
        remaining = 1.0 - cmgtc_weight
        w_sc = remaining * (0.2 / 0.6)
        w_bioc = remaining * (0.4 / 0.6)

        for method, grp in df.groupby('Method'):
            sc = grp.get('SC_Score', pd.Series(0)).mean()
            bioc = grp.get('BioC_Score', pd.Series(0)).mean()
            cmgtc = grp.get('CM_GTC', pd.Series(0)).mean()
            method_scores[method] = w_sc * sc + w_bioc * bioc + cmgtc_weight * cmgtc

    else:  # horizontal
        remaining = 1.0 - cmgtc_weight
        orig_sum = 0.15 + 0.3 + 0.3
        w_sc = remaining * (0.15 / orig_sum)
        w_bioc = remaining * (0.3 / orig_sum)
        w_ber = remaining * (0.3 / orig_sum)

        for method, grp in df.groupby('Method'):
            sc = grp.get('SC_Score', pd.Series(0)).mean()
            bioc = grp.get('BioC_Score', pd.Series(0)).mean()
            ber = grp.get('BER_Score', pd.Series(0)).mean()
            cmgtc = grp.get('CM_GTC', pd.Series(0)).mean()
            method_scores[method] = w_sc * sc + w_bioc * bioc + w_ber * ber + cmgtc_weight * cmgtc

    return pd.Series(method_scores).rank(ascending=False)


def sweep_weights(df, task, weight_range=None):
    """Sweep CM-GTC weight and compute ranking stability."""
    if weight_range is None:
        weight_range = np.linspace(0.0, 1.0, 21)

    default_w = 0.4 if task == 'vertical' else 0.25
    ref_ranking = compute_ranking_with_weight(df, task, default_w)

    results = []
    for w in weight_range:
        ranking = compute_ranking_with_weight(df, task, w)
        common = ref_ranking.index.intersection(ranking.index)
        if len(common) < 3:
            continue

        tau, p_tau = kendalltau(ref_ranking[common], ranking[common])
        rho, p_rho = spearmanr(ref_ranking[common], ranking[common])

        results.append({
            'CM_GTC_Weight': w,
            'Kendall_Tau': tau,
            'Kendall_P': p_tau,
            'Spearman_Rho': rho,
            'Spearman_P': p_rho,
        })

    return pd.DataFrame(results)


def plot_sensitivity(sweep_df, task, out_dir, dpi=300,
                     out_prefix='edfig2_weight_sensitivity', out_suffix=''):
    """Plot weight sensitivity curve."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(sweep_df['CM_GTC_Weight'], sweep_df['Kendall_Tau'],
            'o-', color='#E24A33', label="Kendall's τ", linewidth=2)
    ax.plot(sweep_df['CM_GTC_Weight'], sweep_df['Spearman_Rho'],
            's-', color='#348ABD', label="Spearman's ρ", linewidth=2)

    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='τ = 0.8 threshold')

    default_w = 0.4 if task == 'vertical' else 0.25
    ax.axvline(x=default_w, color='green', linestyle=':', alpha=0.7,
               label=f'Default weight ({default_w})')

    ax.set_xlabel('CM-GTC Weight', fontsize=12)
    ax.set_ylabel('Rank Correlation', fontsize=12)
    ax.set_title(f'{task.title()} Integration — CM-GTC Weight Sensitivity', fontsize=13)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f'{out_prefix}_{task}{out_suffix}.png')
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    out_dir = os.path.join(root, '_myx_Results', 'plots')
    os.makedirs(out_dir, exist_ok=True)

    for task in ['vertical', 'horizontal']:
        print(f"\n--- {task.title()} ---")

        for mod_label, gt_type, ds_list in DATASET_GROUPS:
            if len(ds_list) < 1:
                print(f"  {mod_label} {gt_type}: only {len(ds_list)} dataset, skipping")
                continue

            df = load_data(root, task, args.clustering, gt_datasets=ds_list)
            if df is None or df.empty:
                print(f"  {mod_label} {gt_type}: no data")
                continue

            sweep_df = sweep_weights(df, task)
            if sweep_df.empty:
                continue

            prefix = 'edfig2_weight_sensitivity' if gt_type == 'withGT' else 'supp_weight_sensitivity'
            mod_safe = mod_label.replace('+', '_')
            suffix = f'_{mod_safe}_{gt_type}'

            sweep_df.to_csv(os.path.join(out_dir,
                f'{prefix.replace("sensitivity","sweep")}_{task}{suffix}.csv'), index=False)
            plot_sensitivity(sweep_df, task, out_dir, args.dpi,
                             out_prefix=prefix, out_suffix=suffix)
            print(f"  [{mod_label} {gt_type}] Kendall's τ range: "
                  f"[{sweep_df['Kendall_Tau'].min():.3f}, {sweep_df['Kendall_Tau'].max():.3f}]")

    print("\nDone.")


if __name__ == '__main__':
    main()
