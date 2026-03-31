#!/usr/bin/env python3
"""
Extended Data Fig 1: CM-GTC Orthogonality Analysis (merged 3-panel)
3 panels: (a) CM-GTC vs SC, (b) CM-GTC vs BioC, (c) CM-GTC vs BER
Each panel overlays Vertical withGT, Horizontal withGT, and woGT groups
with different colors.

Usage:
    python plot_edfig1_cmgtc_orthogonality.py --root /path/to/SMOBench-CLEAN
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys as _sys
_sys.path.insert(0, __import__('os').path.dirname(__file__))
from style_config import apply_style, PAL13
apply_style()


WITHGT_DATASETS = ['Human_Lymph_Nodes', 'Human_Tonsils', 'Mouse_Embryos_S1', 'Mouse_Embryos_S2']
WOGT_DATASETS = ['Mouse_Thymus', 'Mouse_Spleen', 'Mouse_Brain']

# Colors matching reference figure
COLOR_VERT = '#E24A33'   # coral for vertical withGT
COLOR_HORIZ = '#348ABD'  # blue for horizontal withGT
COLOR_WOGT = '#E5AE38'   # gold for woGT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--clustering', type=str, default='leiden')
    parser.add_argument('--dpi', type=int, default=300)
    return parser.parse_args()


def load_data(root, task, clustering):
    """Load full final results (all datasets)."""
    summary_dir = os.path.join(root, '_myx_Results', 'evaluation', 'summary')
    for prefix in ['final', 'detailed']:
        path = os.path.join(summary_dir, f'{task}_{prefix}_{clustering}.csv')
        if os.path.isfile(path):
            df = pd.read_csv(path)
            df.columns = [c.replace(' ', '_') for c in df.columns]
            if 'BVC_Score' in df.columns and 'BioC_Score' not in df.columns:
                df.rename(columns={'BVC_Score': 'BioC_Score'}, inplace=True)
            return df
    return None


def get_method_avg(df, datasets):
    """Average metrics per method for a subset of datasets."""
    df_sub = df[df['Dataset'].isin(datasets)]
    if df_sub.empty:
        return None
    return df_sub.groupby('Method').mean(numeric_only=True)


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    out_dir = os.path.join(root, '_myx_Results', 'plots')
    os.makedirs(out_dir, exist_ok=True)

    df_v = load_data(root, 'vertical', args.clustering)
    df_h = load_data(root, 'horizontal', args.clustering)

    # Prepare method averages for each group
    groups = []
    if df_v is not None:
        avg = get_method_avg(df_v, WITHGT_DATASETS)
        if avg is not None:
            groups.append(('Vertical', COLOR_VERT, avg, True))
    if df_h is not None:
        avg = get_method_avg(df_h, WITHGT_DATASETS)
        if avg is not None:
            groups.append(('Horizontal', COLOR_HORIZ, avg, True))
    if df_v is not None:
        avg = get_method_avg(df_v, WOGT_DATASETS)
        if avg is not None:
            groups.append(('woGT', COLOR_WOGT, avg, False))

    # 3 panels: SC, BioC, BER
    panels = [
        ('a', 'CM-GTC vs SC', 'SC_Score', 'SC Score'),
        ('b', 'CM-GTC vs BioC', 'BioC_Score', 'BioC Score'),
        ('c', 'CM-GTC vs BER', 'BER_Score', 'BER Score'),
    ]

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))

    for idx, (label, title, col, xlabel) in enumerate(panels):
        ax = axs[idx]

        for group_name, color, avg, show_label in groups:
            if col not in avg.columns or 'CM_GTC' not in avg.columns:
                continue

            x = avg[col]
            y = avg['CM_GTC']

            # Skip if all NaN
            mask = x.notna() & y.notna()
            if mask.sum() < 3:
                continue

            rho, _ = spearmanr(x[mask], y[mask])

            ax.scatter(x[mask], y[mask], s=50, alpha=0.7, color=color,
                       edgecolors='white', linewidth=0.3,
                       label=f'{group_name} (r={rho:.2f})')

            # Only annotate BER panel (least crowded) with method names
            if col == 'BER_Score':
                for method in avg.index[mask]:
                    ax.annotate(method, (x[method], y[method]),
                                fontsize=6, xytext=(3, 3),
                                textcoords='offset points')

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('CM-GTC', fontsize=11)
        ax.set_title(f'{label} {title}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.2)

    plt.tight_layout()

    out_path = os.path.join(out_dir, 'edfig_merged_orthogonality.pdf')
    plt.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
