#!/usr/bin/env python3
"""
SMOBench Figure 4 (right): Mosaic 4-Subcase Analysis
Following paper[3] Figure 6 format:
- 4 subcases × (BVC vs BER scatter + RI bar chart)
- CM-GTC incorporated into RI calculation

Subcases:
  1: RNA & RNA+protein → RNA-ADT datasets, evaluate on protein
  2: RNA & RNA+ATAC   → RNA-ATAC datasets, evaluate on ATAC
  3: ATAC & RNA+ATAC  → RNA-ATAC datasets, evaluate on RNA
  4: RNA+protein & RNA+ATAC → cross-type summary

Usage:
    python plot_mosaic_subcase.py --root /path/to/SMOBench-CLEAN
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys as _sys
_sys.path.insert(0, __import__('os').path.dirname(__file__))
from style_config import apply_style, PAL13
apply_style()

import matplotlib.gridspec as gridspec


METHOD_ORDER = [
    'CANDIES', 'COSMOS', 'MISO', 'MultiGATE', 'PRAGA', 'PRESENT',
    'SMOPCA', 'SpaBalance', 'SpaFusion', 'SpaMI', 'SpaMosaic',
    'SpaMultiVAE', 'SpaMV', 'SpatialGlue', 'SWITCH',
]

METHOD_COLORS = {m: plt.cm.tab20(i/15) for i, m in enumerate(METHOD_ORDER)}

RNA_ADT_DATASETS = ['Human_Lymph_Nodes', 'Human_Tonsils', 'Mouse_Thymus', 'Mouse_Spleen']
RNA_ATAC_DATASETS = ['Mouse_Embryos_S1', 'Mouse_Embryos_S2', 'Mouse_Brain']

SUBCASE_CONFIG = {
    1: {'name': 'RNA & RNA+Protein',   'datasets': RNA_ADT_DATASETS,  'desc': 'RNA ref, ADT data'},
    2: {'name': 'RNA & RNA+ATAC',      'datasets': RNA_ATAC_DATASETS, 'desc': 'RNA ref, ATAC data'},
    3: {'name': 'ATAC & RNA+ATAC',     'datasets': RNA_ATAC_DATASETS, 'desc': 'ATAC ref, RNA data'},
    4: {'name': 'RNA+Protein & RNA+ATAC', 'datasets': RNA_ADT_DATASETS + RNA_ATAC_DATASETS, 'desc': 'Cross-type'},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--clustering', type=str, default='leiden')
    parser.add_argument('--dpi', type=int, default=300)
    return parser.parse_args()


def load_final_data(root, clustering):
    """Load both vertical and horizontal final results."""
    summary_dir = os.path.join(root, '_myx_Results', 'evaluation', 'summary')
    data = {}
    for task in ['vertical', 'horizontal']:
        for prefix in ['final', 'detailed']:
            path = os.path.join(summary_dir, f'{task}_{prefix}_{clustering}.csv')
            if os.path.isfile(path):
                data[task] = pd.read_csv(path)
                break
    return data


def compute_ri(df, metric_cols):
    """Compute Ranking Index (RI) — average rank across metrics (lower = better)."""
    ranks = pd.DataFrame()
    for col in metric_cols:
        if col in df.columns:
            ranks[col] = df[col].rank(ascending=False)
    if ranks.empty:
        return pd.Series(dtype=float)
    return ranks.mean(axis=1)


def plot_subcase(ax_scatter, ax_bar, df, subcase_id, config):
    """Plot BVC vs BER scatter + RI bar for one subcase."""
    datasets = config['datasets']
    df_sub = df[df['Dataset'].isin(datasets)] if 'Dataset' in df.columns else df

    if df_sub.empty:
        ax_scatter.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax_scatter.set_title(f'Subcase {subcase_id}: {config["name"]}')
        return

    # Aggregate per method
    score_cols = []
    for col in ['BioC_Score', 'BVC_Score', 'BER_Score', 'CM_GTC']:
        if col in df_sub.columns:
            score_cols.append(col)

    bioc_col = 'BVC_Score' if 'BVC_Score' in df_sub.columns else 'BioC_Score'
    ber_col = 'BER_Score'

    method_avg = df_sub.groupby('Method')[score_cols].mean().reset_index()

    # --- Scatter: BioC vs BER ---
    if bioc_col in method_avg.columns and ber_col in method_avg.columns:
        for _, row in method_avg.iterrows():
            method = row['Method']
            x = row[bioc_col]
            y = row.get(ber_col, 0)
            size = row.get('CM_GTC', 0.5) * 400 + 30
            color = METHOD_COLORS.get(method, 'gray')
            ax_scatter.scatter(x, y, s=size, c=[color], alpha=0.8,
                              edgecolors='white', linewidth=0.5, zorder=5)
            ax_scatter.annotate(method, (x, y), fontsize=6,
                               xytext=(3, 3), textcoords='offset points')

        ax_scatter.set_xlabel('BioC', fontsize=9)
        ax_scatter.set_ylabel('BER', fontsize=9)
    else:
        available = [c for c in score_cols if c != 'CM_GTC']
        if len(available) >= 2:
            for _, row in method_avg.iterrows():
                method = row['Method']
                color = METHOD_COLORS.get(method, 'gray')
                ax_scatter.scatter(row[available[0]], row[available[1]],
                                  s=100, c=[color], alpha=0.8, edgecolors='white')
                ax_scatter.annotate(method, (row[available[0]], row[available[1]]),
                                   fontsize=6, xytext=(3, 3), textcoords='offset points')
            ax_scatter.set_xlabel(available[0], fontsize=9)
            ax_scatter.set_ylabel(available[1], fontsize=9)

    ax_scatter.set_title(f'Subcase {subcase_id}: {config["name"]}', fontsize=10)
    ax_scatter.grid(True, alpha=0.3)

    # --- Bar: RI ---
    ri = compute_ri(method_avg.set_index('Method'), score_cols)
    if not ri.empty:
        ri_sorted = ri.sort_values()
        colors = [METHOD_COLORS.get(m, 'gray') for m in ri_sorted.index]
        ax_bar.barh(range(len(ri_sorted)), ri_sorted.values, color=colors, alpha=0.8)
        ax_bar.set_yticks(range(len(ri_sorted)))
        ax_bar.set_yticklabels(ri_sorted.index, fontsize=8)
        ax_bar.set_xlabel('RI (lower = better)', fontsize=9)
        ax_bar.set_title(f'Ranking Index', fontsize=10)
        ax_bar.invert_yaxis()


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    out_dir = os.path.join(root, '_myx_Results', 'plots')
    os.makedirs(out_dir, exist_ok=True)

    data = load_final_data(root, args.clustering)
    if not data:
        print("No final results found. Run evaluation first.")
        return

    # Combine vertical and horizontal data
    all_df = pd.concat(data.values(), ignore_index=True)
    print(f"Loaded {len(all_df)} rows from {list(data.keys())}")

    # Create 4-subcase figure (4 rows × 2 cols: scatter + bar)
    fig, axs = plt.subplots(4, 2, figsize=(14, 20))

    for sc_id, config in SUBCASE_CONFIG.items():
        row = sc_id - 1
        plot_subcase(axs[row, 0], axs[row, 1], all_df, sc_id, config)

    fig.suptitle(f'Mosaic Integration — 4-Subcase Analysis ({args.clustering})',
                 fontsize=14, y=1.01)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f'fig4_mosaic_4subcase_{args.clustering}.png')
    plt.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
