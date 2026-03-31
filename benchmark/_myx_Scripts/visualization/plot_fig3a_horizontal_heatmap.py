#!/usr/bin/env python3
"""
SMOBench Figure 3(a): Horizontal Integration Heatmap
Methods × {SC metrics, BioC metrics, BER metrics, scores}

Main figure uses withGT datasets; supplementary uses woGT datasets.

Usage:
    python plot_fig3a_horizontal_heatmap.py --root /path/to/SMOBench-CLEAN
    python plot_fig3a_horizontal_heatmap.py --root /path/to/SMOBench-CLEAN --clustering leiden
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


METRIC_COLS_WITHGT = [
    'Moran_Index',
    'ARI', 'NMI', 'asw_celltype', 'graph_clisi',
    'kBET', 'KNN_connectivity', 'bASW', 'iLISI', 'PCR',
]
METRIC_COLS_WOGT = [
    'Moran_Index',
    'Davies-Bouldin_Index_normalized', 'Silhouette_Coefficient',
    'Calinski-Harabaz_Index_normalized',
    'kBET', 'KNN_connectivity', 'bASW', 'iLISI', 'PCR',
]
SCORE_COLS = ['SC_Score', 'BioC_Score', 'BER_Score', 'CM_GTC', 'SMOBench_H']

METHOD_ORDER = [
    'CANDIES', 'COSMOS', 'MISO', 'PRAGA', 'PRESENT',
    'SMOPCA', 'SpaBalance', 'SpaFusion', 'SpaMI', 'SpaMosaic',
    'SpaMultiVAE', 'SpaMV', 'SpatialGlue',
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
    parser.add_argument('--dpi', type=int, default=300)
    return parser.parse_args()


def load_data(root, clustering):
    summary_dir = os.path.join(root, '_myx_Results', 'evaluation', 'summary')
    final_path = os.path.join(summary_dir, f'horizontal_final_{clustering}.csv')
    if os.path.isfile(final_path):
        df = pd.read_csv(final_path)
    elif os.path.isfile(os.path.join(summary_dir, f'horizontal_detailed_{clustering}.csv')):
        df = pd.read_csv(os.path.join(summary_dir, f'horizontal_detailed_{clustering}.csv'))
    else:
        raise FileNotFoundError(f"No horizontal results in {summary_dir}")

    # Normalize column names: spaces → underscores
    df.columns = [c.replace(' ', '_') for c in df.columns]
    # Unify BVC_Score → BioC_Score
    if 'BVC_Score' in df.columns and 'BioC_Score' not in df.columns:
        df.rename(columns={'BVC_Score': 'BioC_Score'}, inplace=True)
    return df


def plot_heatmap(df_sub, modality_label, metric_cols, clustering, out_dir, dpi,
                 gt_type='withGT'):
    """Plot a single modality × GT heatmap."""
    available_cols = [c for c in metric_cols + SCORE_COLS if c in df_sub.columns]
    method_avg = df_sub.groupby('Method')[available_cols].mean()
    methods_present = [m for m in METHOD_ORDER if m in method_avg.index]
    method_avg = method_avg.loc[methods_present]

    sort_col = 'SMOBench_H' if 'SMOBench_H' in method_avg.columns else available_cols[-1]
    method_avg = method_avg.sort_values(sort_col, ascending=False)

    # Normalize
    norm_data = method_avg.copy()
    for col in norm_data.columns:
        cmin, cmax = norm_data[col].min(), norm_data[col].max()
        if cmax > cmin:
            norm_data[col] = (norm_data[col] - cmin) / (cmax - cmin)
        else:
            norm_data[col] = 0.5

    n_cols = len(available_cols)
    fig_w = max(10, n_cols * 0.8 + 2)
    fig_h = max(4, len(method_avg) * 0.4 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

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

    n_metrics = len([c for c in available_cols if c not in SCORE_COLS])
    if n_metrics < n_cols:
        ax.axvline(x=n_metrics, color='black', linewidth=2)

    gt_label = f' ({gt_type})' if gt_type == 'woGT' else ''
    ax.set_title(f'Horizontal Integration — {modality_label}{gt_label} ({clustering})',
                 fontsize=13, pad=10)
    ax.set_ylabel('Method', fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    prefix = 'fig3a' if gt_type == 'withGT' else 'supp'
    mod_tag = modality_label.replace('+', '_')
    out_path = os.path.join(out_dir,
                            f'{prefix}_horizontal_heatmap_{mod_tag}_{gt_type}_{clustering}')
    plt.savefig(out_path + '.pdf', bbox_inches='tight')
    plt.savefig(out_path + '.png', dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}.pdf")


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    out_dir = os.path.join(root, '_myx_Results', 'plots')
    os.makedirs(out_dir, exist_ok=True)

    df = load_data(root, args.clustering)
    print(f"Loaded {len(df)} rows")

    # Main figure: withGT datasets
    for mod_type, ds_list in [('RNA_ADT', RNA_ADT_WITHGT), ('RNA_ATAC', RNA_ATAC_WITHGT)]:
        df_sub = df[df['Dataset'].isin(ds_list)]
        if len(df_sub) > 0:
            label = mod_type.replace('_', '+')
            print(f"\n--- {label} withGT ({len(df_sub)} rows) ---")
            plot_heatmap(df_sub, label, METRIC_COLS_WITHGT, args.clustering,
                         out_dir, args.dpi, gt_type='withGT')

    # Supplementary: woGT datasets
    for mod_type, ds_list in [('RNA_ADT', RNA_ADT_WOGT), ('RNA_ATAC', RNA_ATAC_WOGT)]:
        df_sub = df[df['Dataset'].isin(ds_list)]
        if len(df_sub) > 0:
            label = mod_type.replace('_', '+')
            print(f"\n--- {label} woGT ({len(df_sub)} rows) ---")
            plot_heatmap(df_sub, label, METRIC_COLS_WOGT, args.clustering,
                         out_dir, args.dpi, gt_type='woGT')


if __name__ == '__main__':
    main()
