#!/usr/bin/env python3
"""
SMOBench Figure 2(a): Vertical Integration Heatmap
Methods × {Moran's I, ARI, NMI, ASW, cLISI, CM-GTC, SC_Score, BioC_Score, CM-GTC_Score, Total}

Usage:
    python plot_vertical_heatmap.py --root /path/to/SMOBench-CLEAN
    python plot_vertical_heatmap.py --root /path/to/SMOBench-CLEAN --clustering leiden
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Column groups for display order
METRIC_COLS_WITHGT = [
    'Moran_Index', 'ARI', 'NMI', 'asw_celltype', 'graph_clisi',
]
METRIC_COLS_WOGT = [
    'Moran_Index', 'Davies-Bouldin_Index_normalized',
    'Silhouette_Coefficient', 'Calinski-Harabaz_Index_normalized',
]
SCORE_COLS = ['SC_Score', 'BioC_Score', 'CM_GTC', 'SMOBench_V']

METHOD_ORDER = [
    'CANDIES', 'COSMOS', 'MISO', 'MultiGATE', 'PRAGA', 'PRESENT',
    'SMOPCA', 'SpaBalance', 'SpaFusion', 'SpaMI', 'SpaMosaic',
    'SpaMultiVAE', 'SpaMV', 'SpatialGlue', 'SWITCH',
]

# Main figure: withGT datasets only
RNA_ADT_WITHGT = ['Human_Lymph_Nodes', 'Human_Tonsils']
RNA_ATAC_WITHGT = ['Mouse_Embryos_S1', 'Mouse_Embryos_S2']

# Supplementary: woGT datasets
RNA_ADT_WOGT = ['Mouse_Thymus', 'Mouse_Spleen']
RNA_ATAC_WOGT = ['Mouse_Brain']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--clustering', type=str, default='leiden')
    parser.add_argument('--modality', choices=['RNA_ADT', 'RNA_ATAC', 'all'], default='all')
    parser.add_argument('--dpi', type=int, default=300)
    return parser.parse_args()


def load_data(root, clustering):
    """Load final scored CSV (with CM-GTC merged)."""
    summary_dir = os.path.join(root, '_myx_Results', 'evaluation', 'summary')

    # Try final file first (has CM-GTC), then detailed
    final_path = os.path.join(summary_dir, f'vertical_final_{clustering}.csv')
    if os.path.isfile(final_path):
        df = pd.read_csv(final_path)
    elif os.path.isfile(os.path.join(summary_dir, f'vertical_detailed_{clustering}.csv')):
        df = pd.read_csv(os.path.join(summary_dir, f'vertical_detailed_{clustering}.csv'))
    else:
        raise FileNotFoundError(f"No vertical results found in {summary_dir}")

    # Normalize column names: spaces → underscores
    df.columns = [c.replace(' ', '_') for c in df.columns]
    return df


def plot_heatmap(df, modality_type, clustering, out_dir, dpi=300,
                 datasets=None, gt_type='withGT'):
    """Plot heatmap for one modality × GT group.

    Parameters
    ----------
    datasets : list[str]
        Dataset names to include.
    gt_type : str
        'withGT' or 'woGT', controls which metric columns to use.
    """
    if datasets is None:
        datasets = []
    metric_cols = METRIC_COLS_WITHGT if gt_type == 'withGT' else METRIC_COLS_WOGT

    # Filter by datasets
    df_mod = df[df['Dataset'].isin(datasets)].copy()
    if df_mod.empty:
        print(f"  No data for {modality_type} {gt_type}")
        return

    # Average across datasets per method
    available_cols = [c for c in metric_cols + SCORE_COLS if c in df_mod.columns]
    method_avg = df_mod.groupby('Method')[available_cols].mean()

    # Reorder methods
    methods_present = [m for m in METHOD_ORDER if m in method_avg.index]
    method_avg = method_avg.loc[methods_present]

    # Sort by SMOBench_V (or Total) descending
    sort_col = 'SMOBench_V' if 'SMOBench_V' in method_avg.columns else available_cols[-1]
    method_avg = method_avg.sort_values(sort_col, ascending=False)

    # Create figure
    n_cols = len(available_cols)
    fig_w = max(8, n_cols * 0.9 + 2)
    fig_h = max(4, len(method_avg) * 0.4 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Normalize per column for coloring (0-1 range)
    norm_data = method_avg.copy()
    for col in norm_data.columns:
        cmin, cmax = norm_data[col].min(), norm_data[col].max()
        if cmax > cmin:
            norm_data[col] = (norm_data[col] - cmin) / (cmax - cmin)
        else:
            norm_data[col] = 0.5

    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    sns.heatmap(
        norm_data,
        annot=method_avg.round(3),
        fmt='',
        cmap=cmap,
        linewidths=0.5,
        linecolor='white',
        ax=ax,
        cbar_kws={'label': 'Normalized Score', 'shrink': 0.6},
        xticklabels=[c.replace('_', '\n') for c in available_cols],
    )

    # Add vertical separator before score columns
    n_metrics = len([c for c in available_cols if c not in SCORE_COLS])
    if n_metrics < n_cols:
        ax.axvline(x=n_metrics, color='black', linewidth=2)

    mod_label = modality_type.replace('_', '+')
    gt_label = f' ({gt_type})' if gt_type == 'woGT' else ''
    ax.set_title(f'Vertical Integration — {mod_label}{gt_label} ({clustering})',
                 fontsize=13, pad=10)
    ax.set_ylabel('Method', fontsize=11)
    ax.set_xlabel('')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    prefix = 'fig2a' if gt_type == 'withGT' else 'supp'
    out_base = os.path.join(out_dir,
                            f'{prefix}_vertical_heatmap_{mod_label}_{gt_type}_{clustering}')
    plt.savefig(out_base + '.pdf', bbox_inches='tight')
    plt.savefig(out_base + '.png', dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_base}.pdf")


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    out_dir = os.path.join(root, '_myx_Results', 'plots')
    os.makedirs(out_dir, exist_ok=True)

    df = load_data(root, args.clustering)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    # Main figure: withGT datasets only
    if args.modality in ('RNA_ADT', 'all'):
        print("\nPlotting RNA_ADT withGT heatmap (main figure)...")
        plot_heatmap(df, 'RNA_ADT', args.clustering, out_dir, args.dpi,
                     datasets=RNA_ADT_WITHGT, gt_type='withGT')
    if args.modality in ('RNA_ATAC', 'all'):
        print("\nPlotting RNA_ATAC withGT heatmap (main figure)...")
        plot_heatmap(df, 'RNA_ATAC', args.clustering, out_dir, args.dpi,
                     datasets=RNA_ATAC_WITHGT, gt_type='withGT')

    # Supplementary: woGT datasets
    if args.modality in ('RNA_ADT', 'all'):
        print("\nPlotting RNA_ADT woGT heatmap (supplementary)...")
        plot_heatmap(df, 'RNA_ADT', args.clustering, out_dir, args.dpi,
                     datasets=RNA_ADT_WOGT, gt_type='woGT')
    if args.modality in ('RNA_ATAC', 'all'):
        print("\nPlotting RNA_ATAC woGT heatmap (supplementary)...")
        plot_heatmap(df, 'RNA_ATAC', args.clustering, out_dir, args.dpi,
                     datasets=RNA_ATAC_WOGT, gt_type='woGT')


if __name__ == '__main__':
    main()
