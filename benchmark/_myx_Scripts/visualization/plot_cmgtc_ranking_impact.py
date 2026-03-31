#!/usr/bin/env python3
"""
CM-GTC Ranking Impact: Traditional vs SMOBench ranking comparison.

Shows how adding CM-GTC changes method rankings.
Left column: Traditional (SC + BioC [+ BER for horizontal])
Right column: SMOBench (+ CM-GTC)

Usage:
    python plot_cmgtc_ranking_impact.py --root /path/to/SMOBench-CLEAN
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys as _sys
_sys.path.insert(0, __import__('os').path.dirname(__file__))
from style_config import apply_style, PAL13
apply_style()

DATASET_GROUPS = [
    ('RNA+ADT',  'withGT', ['Human_Lymph_Nodes', 'Human_Tonsils']),
    ('RNA+ADT',  'woGT',   ['Mouse_Thymus', 'Mouse_Spleen']),
    ('RNA+ATAC', 'withGT', ['Mouse_Embryos_S1', 'Mouse_Embryos_S2']),
    ('RNA+ATAC', 'woGT',   ['Mouse_Brain']),
]

palette = PAL13 + ['#333333', '#777777']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--clustering', type=str, default='leiden')
    parser.add_argument('--dpi', type=int, default=300)
    return parser.parse_args()


def load_data(root, task, clustering):
    summary = os.path.join(root, '_myx_Results', 'evaluation', 'summary')
    for prefix in ['final', 'detailed']:
        path = os.path.join(summary, f'{task}_{prefix}_{clustering}.csv')
        if os.path.isfile(path):
            df = pd.read_csv(path)
            df.columns = [c.replace(' ', '_') for c in df.columns]
            if 'BVC_Score' in df.columns and 'BioC_Score' not in df.columns:
                df.rename(columns={'BVC_Score': 'BioC_Score'}, inplace=True)
            return df
    return None


def compute_rankings(df, task, datasets):
    """Compute Traditional and SMOBench rankings."""
    df_sub = df[df['Dataset'].isin(datasets)]
    if df_sub.empty:
        return None, None

    method_avg = df_sub.groupby('Method').mean(numeric_only=True)

    if task == 'vertical':
        # Traditional: SC + BioC only (weight ratio 1:2)
        trad = 0.333 * method_avg.get('SC_Score', 0) + 0.667 * method_avg.get('BioC_Score', 0)
        # SMOBench: 0.2*SC + 0.4*BioC + 0.4*CM-GTC
        bench = method_avg.get('SMOBench_V', trad)
        trad_label = 'Traditional\n(SC + BioC)'
        bench_label = 'SMOBench$_V$\n(+ CM-GTC)'
    else:
        # Traditional: SC + BioC + BER (weight ratio 1:2:2)
        trad = (0.2 * method_avg.get('SC_Score', 0) +
                0.4 * method_avg.get('BioC_Score', 0) +
                0.4 * method_avg.get('BER_Score', 0))
        bench = method_avg.get('SMOBench_H', trad)
        trad_label = 'Traditional\n(SC + BioC + BER)'
        bench_label = 'SMOBench$_H$\n(+ CM-GTC)'

    trad_rank = trad.rank(ascending=False).astype(int)
    bench_rank = bench.rank(ascending=False).astype(int)

    return (trad_rank, bench_rank, trad_label, bench_label)


def plot_impact(ax, trad_rank, bench_rank, trad_label, bench_label, title):
    """Draw ranking impact bump chart on axes."""
    methods = sorted(trad_rank.index, key=lambda m: trad_rank[m])
    n = len(methods)
    method_colors = {m: palette[i % len(palette)] for i, m in enumerate(methods)}

    x = [0, 1]

    for m in methods:
        r1 = trad_rank[m]
        r2 = bench_rank[m]
        shift = int(r1 - r2)  # positive = improved
        color = method_colors[m]

        lw = 2.5 if abs(shift) >= 2 else 1.5
        alpha = 1.0

        ax.plot(x, [r1, r2], 'o-', color=color, linewidth=lw, alpha=alpha,
                markersize=6, zorder=10)

        # Left label
        fw = 'bold' if abs(shift) >= 2 else 'normal'
        ax.text(-0.05, r1, m, fontsize=8, va='center', ha='right',
                color=color, fontweight=fw, clip_on=False)
        # Right label
        ax.text(1.05, r2, m, fontsize=8, va='center', ha='left',
                color=color, fontweight=fw, clip_on=False)

        # Shift annotation for |shift| >= 2
        if abs(shift) >= 2:
            mid_y = (r1 + r2) / 2
            sign = f'+{shift}' if shift > 0 else str(shift)
            bbox_color = '#FFCCCC' if shift < 0 else '#CCFFCC'
            ax.text(0.5, mid_y, sign, fontsize=7, va='center', ha='center',
                    color='#333333', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=bbox_color,
                              edgecolor='#999999', linewidth=0.5))

    # Kendall's tau
    common = trad_rank.index.intersection(bench_rank.index)
    tau, _ = kendalltau(trad_rank[common], bench_rank[common])

    ax.set_xticks(x)
    ax.set_xticklabels([trad_label, bench_label], fontsize=10)
    ax.set_ylabel('Rank', fontsize=11)
    ax.set_ylim(n + 0.5, 0.5)
    ax.set_yticks(range(1, n + 1))
    ax.set_xlim(-0.5, 1.5)

    for r in range(1, n + 1):
        ax.axhline(r, color='#EEEEEE', lw=0.4, zorder=0)

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.text(0.95, n + 0.2, f"Kendall's τ = {tau:.3f}",
            fontsize=9, ha='right', va='top', transform=ax.get_yaxis_transform())


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    out_dir = os.path.join(root, '_myx_Results', 'plots')
    os.makedirs(out_dir, exist_ok=True)

    for mod_label, gt_type, ds_list in DATASET_GROUPS:
        mod_safe = mod_label.replace('+', '_')
        prefix = 'j' if gt_type == 'withGT' else 'supp_j'
        suffix = f'_{mod_safe}_{gt_type}'

        df_v = load_data(root, 'vertical', args.clustering)
        df_h = load_data(root, 'horizontal', args.clustering)

        results_v = compute_rankings(df_v, 'vertical', ds_list) if df_v is not None else None
        results_h = compute_rankings(df_h, 'horizontal', ds_list) if df_h is not None else None

        has_v = results_v is not None
        has_h = results_h is not None
        n_panels = int(has_v) + int(has_h)
        if n_panels == 0:
            continue

        fig, axs = plt.subplots(1, n_panels, figsize=(8 * n_panels, 5.5))
        if n_panels == 1:
            axs = [axs]

        idx = 0
        if has_v:
            plot_impact(axs[idx], *results_v, 'Vertical Integration')
            idx += 1
        if has_h:
            plot_impact(axs[idx], *results_h, 'Horizontal Integration')

        plt.tight_layout(w_pad=4)

        out_path = os.path.join(out_dir, f'{prefix}_cmgtc_ranking_impact{suffix}.pdf')
        plt.savefig(out_path, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {out_path}")


if __name__ == '__main__':
    main()
