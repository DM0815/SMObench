#!/usr/bin/env python3
"""
SMOBench Figure 4(a): 3-Modality Integration Performance Heatmap
7 methods x performance metrics (3Mv2 evaluation).

Why this panel:
  The benchmark covers 3-modality integration as an extension scenario.
  A heatmap shows at a glance which method handles 3M best — even if
  all methods struggle (ARI/NMI near zero), that is itself informative.

Usage:
    python plot_fig4a_3m_heatmap.py --root /path/to/SMOBench-CLEAN
    python plot_fig4a_3m_heatmap.py --root /path/to/SMOBench-CLEAN --clustering leiden
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


METRIC_COLS = ['Moran_Index', 'ARI', 'NMI', 'asw_celltype', 'graph_clisi']
SCORE_COLS = ['SC_Score', 'BioC_Score', 'CM_GTC', 'SMOBench_V']

DISPLAY_LABELS = {
    'Moran_Index': "Moran's I",
    'ARI': 'ARI',
    'NMI': 'NMI',
    'asw_celltype': 'ASW',
    'graph_clisi': 'cLISI',
    'SC_Score': 'SC\nScore',
    'BioC_Score': 'BioC\nScore',
    'CM_GTC': 'CM-GTC',
    'SMOBench_V': 'SMOBench\n$_V$',
}

METHOD_DISPLAY = {
    'SpatialGlue_3Mv2': 'SpatialGlue',
    'SpaBalance_3Mv2': 'SpaBalance',
    'SMOPCA_3Mv2': 'SMOPCA',
    'MISO_3Mv2': 'MISO',
    'PRESENT_3Mv2': 'PRESENT',
    'SpaMV_3Mv2': 'SpaMV',
    'PRAGA_3Mv2': 'PRAGA',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--clustering', type=str, default='leiden')
    parser.add_argument('--dpi', type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    eval_dir = os.path.join(root, '_myx_Results', 'evaluation', '3m_v2')
    out_dir = os.path.join(root, '_myx_Results', 'plots')
    os.makedirs(out_dir, exist_ok=True)

    # Load main evaluation results
    eval_path = os.path.join(eval_dir, '3m_evaluation_results.csv')
    if not os.path.isfile(eval_path):
        print(f"Not found: {eval_path}")
        return
    df = pd.read_csv(eval_path)
    print(f"Loaded {len(df)} rows from 3m_evaluation_results.csv")

    # Rename columns with spaces to underscores
    col_rename = {
        'Moran Index': 'Moran_Index',
        'Geary C': 'Geary_C',
        'Silhouette Coefficient': 'Silhouette_Coefficient',
        'Calinski-Harabasz Index': 'CHI',
        'Davies-Bouldin Index': 'DBI',
        'Jaccard Index': 'Jaccard_Index',
        'Dice Index': 'Dice_Index',
        'BVC_Score': 'BioC_Score',
    }
    df.rename(columns=col_rename, inplace=True)

    # Filter to requested clustering
    if 'Clustering' in df.columns:
        df_clust = df[df['Clustering'] == args.clustering]
        if df_clust.empty:
            print(f"No rows for clustering={args.clustering}, using all")
            # Deduplicate by taking first per method
            df_clust = df.drop_duplicates(subset='Method', keep='first')
        df = df_clust

    # Load CM-GTC
    cmgtc_path = os.path.join(eval_dir, 'cmgtc_3m_combined.csv')
    if os.path.isfile(cmgtc_path):
        df_cmgtc = pd.read_csv(cmgtc_path)
        print(f"Loaded CM-GTC: {len(df_cmgtc)} rows")
        cmgtc_map = dict(zip(df_cmgtc['Method'], df_cmgtc['CM_GTC']))
        df['CM_GTC'] = df['Method'].map(cmgtc_map)
    else:
        print(f"Warning: CM-GTC file not found: {cmgtc_path}")
        df['CM_GTC'] = np.nan

    # Compute SMOBench_V composite score
    df['SMOBench_V'] = (0.2 * df['SC_Score'].fillna(0)
                        + 0.4 * df['BioC_Score'].fillna(0)
                        + 0.4 * df['CM_GTC'].fillna(0))

    # Select available columns
    all_cols = METRIC_COLS + SCORE_COLS
    available_cols = [c for c in all_cols if c in df.columns]

    # Build method x metric matrix
    method_data = df.groupby('Method')[available_cols].mean()
    # Sort by SMOBench_V descending
    if 'SMOBench_V' in method_data.columns:
        method_data = method_data.sort_values('SMOBench_V', ascending=False)

    if method_data.empty:
        print("No method data to plot")
        return

    print(f"\nMethods: {list(method_data.index)}")
    print(f"Columns: {available_cols}")
    print(method_data.round(3).to_string())

    # Normalize per column for coloring
    norm_data = method_data.copy()
    for col in norm_data.columns:
        cmin, cmax = norm_data[col].min(), norm_data[col].max()
        if cmax > cmin:
            norm_data[col] = (norm_data[col] - cmin) / (cmax - cmin)
        else:
            norm_data[col] = 0.5

    # Plot heatmap
    n_cols = len(available_cols)
    fig_w = max(6, n_cols * 0.9 + 2)
    fig_h = max(2.5, len(method_data) * 0.6 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    # Display labels for x-axis
    xlabels = [DISPLAY_LABELS.get(c, c) for c in available_cols]
    # Display labels for y-axis
    ylabels = [METHOD_DISPLAY.get(m, m) for m in method_data.index]

    sns.heatmap(
        norm_data,
        annot=method_data.round(3),
        fmt='',
        cmap=cmap,
        linewidths=0.5,
        linecolor='white',
        ax=ax,
        cbar_kws={'label': 'Normalized Score', 'shrink': 0.8},
        xticklabels=xlabels,
        yticklabels=ylabels,
    )

    # Vertical separator before score columns
    n_metrics = len([c for c in available_cols if c not in SCORE_COLS])
    if n_metrics < n_cols:
        ax.axvline(x=n_metrics, color='black', linewidth=2)

    ax.set_title(f'3-Modality Integration ({args.clustering})', fontsize=13, pad=10)
    ax.set_ylabel('')
    ax.set_xlabel('')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=11, rotation=0)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f'fig4a_3m_heatmap_{args.clustering}.png')
    plt.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
