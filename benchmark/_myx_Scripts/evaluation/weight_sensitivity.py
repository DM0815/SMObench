#!/usr/bin/env python3
"""
SMOBench Supplementary: Weight Sensitivity Analysis
Sweeps CM-GTC weight from 0.1 to 0.5 and computes Kendall's tau on method rankings.
Proves that ranking is robust to weight choice.

Usage:
    python weight_sensitivity.py --root /path/to/SMOBench-CLEAN
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
import matplotlib.pyplot as plt


CLUSTERING_METHODS = ['leiden', 'louvain', 'kmeans', 'mclust']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--clustering', type=str, default='leiden')
    parser.add_argument('--dpi', type=int, default=300)
    return parser.parse_args()


WITHGT_DATASETS = ['Human_Lymph_Nodes', 'Human_Tonsils', 'Mouse_Embryos_S1', 'Mouse_Embryos_S2']

# 2x2 dataset groups: modality × GT
DATASET_GROUPS = [
    ('RNA+ADT',  'withGT', ['Human_Lymph_Nodes', 'Human_Tonsils']),
    ('RNA+ADT',  'woGT',   ['Mouse_Thymus', 'Mouse_Spleen']),
    ('RNA+ATAC', 'withGT', ['Mouse_Embryos_S1', 'Mouse_Embryos_S2']),
    ('RNA+ATAC', 'woGT',   ['Mouse_Brain']),
]


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
        # SMOBench_V = (1-cmgtc_weight)/2 * SC + (1-cmgtc_weight)/2 * BioC + cmgtc_weight * CM-GTC
        # But maintain SC:BioC ratio as 1:2
        remaining = 1.0 - cmgtc_weight
        w_sc = remaining * (0.2 / 0.6)  # Original ratio SC/(SC+BioC) = 0.2/0.6
        w_bioc = remaining * (0.4 / 0.6)

        for method, grp in df.groupby('Method'):
            sc = grp.get('SC_Score', pd.Series(0)).mean()
            bioc = grp.get('BioC_Score', pd.Series(0)).mean()
            cmgtc = grp.get('CM_GTC', pd.Series(0)).mean()
            method_scores[method] = w_sc * sc + w_bioc * bioc + cmgtc_weight * cmgtc

    else:  # horizontal
        # SMOBench_H: maintain SC:BioC:BER ratio, adjust CM-GTC weight
        remaining = 1.0 - cmgtc_weight
        orig_sum = 0.15 + 0.3 + 0.3  # original SC+BioC+BER = 0.75
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

    # Reference ranking at default weight
    default_w = 0.4 if task == 'vertical' else 0.25
    ref_ranking = compute_ranking_with_weight(df, task, default_w)

    results = []
    for w in weight_range:
        ranking = compute_ranking_with_weight(df, task, w)
        # Align methods
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


def plot_sensitivity(sweep_df, task, out_dir, dpi=300, out_suffix=''):
    """Plot weight sensitivity curve."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(sweep_df['CM_GTC_Weight'], sweep_df['Kendall_Tau'],
            'o-', color='#E24A33', label="Kendall's τ", linewidth=2)
    ax.plot(sweep_df['CM_GTC_Weight'], sweep_df['Spearman_Rho'],
            's-', color='#348ABD', label="Spearman's ρ", linewidth=2)

    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='τ = 0.8 threshold')

    # Mark default weight
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

    out_path = os.path.join(out_dir, f'supp_weight_sensitivity_{task}{out_suffix}.png')
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def orthogonality_analysis(df, task, out_dir, dpi=300, out_suffix=''):
    """Compute and plot CM-GTC vs traditional metrics correlation (orthogonality)."""
    if 'CM_GTC' not in df.columns:
        print("  No CM-GTC column, skipping orthogonality analysis")
        return

    method_avg = df.groupby('Method').mean(numeric_only=True)

    traditional_cols = ['SC_Score', 'BioC_Score']
    if task == 'horizontal':
        traditional_cols.append('BER_Score')

    available = [c for c in traditional_cols if c in method_avg.columns]
    if not available:
        return

    fig, axs = plt.subplots(1, len(available), figsize=(5*len(available), 4.5))
    if len(available) == 1:
        axs = [axs]

    for idx, col in enumerate(available):
        x = method_avg[col]
        y = method_avg['CM_GTC']
        rho, p = spearmanr(x, y)

        axs[idx].scatter(x, y, s=60, alpha=0.7, edgecolors='white')
        for method in method_avg.index:
            axs[idx].annotate(method, (x[method], y[method]),
                             fontsize=7, xytext=(3, 3), textcoords='offset points')

        axs[idx].set_xlabel(col.replace('_', ' '), fontsize=11)
        axs[idx].set_ylabel('CM-GTC', fontsize=11)
        axs[idx].set_title(f'ρ = {rho:.3f} (p = {p:.3f})', fontsize=10)
        axs[idx].grid(True, alpha=0.3)

    fig.suptitle(f'{task.title()} — CM-GTC Orthogonality Analysis', fontsize=13, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f'supp_cmgtc_orthogonality_{task}{out_suffix}.png')
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
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

            mod_safe = mod_label.replace('+', '_')
            suffix = f'_{mod_safe}_{gt_type}'

            # Weight sensitivity
            sweep_df = sweep_weights(df, task)
            if not sweep_df.empty:
                sweep_df.to_csv(os.path.join(out_dir,
                    f'supp_weight_sweep_{task}{suffix}.csv'), index=False)
                plot_sensitivity(sweep_df, task, out_dir, args.dpi,
                                 out_suffix=suffix)
                print(f"  [{mod_label} {gt_type}] τ range: "
                      f"[{sweep_df['Kendall_Tau'].min():.3f}, "
                      f"{sweep_df['Kendall_Tau'].max():.3f}]")

            # Orthogonality
            orthogonality_analysis(df, task, out_dir, args.dpi,
                                   out_suffix=suffix)

    print("\nDone.")


if __name__ == '__main__':
    main()
