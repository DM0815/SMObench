#!/usr/bin/env python3
"""
ED Fig 4: woGT datasets — SC_Score (unsupervised BioC) vs CM-GTC scatter.

For 3 woGT datasets (Mouse_Spleen, Mouse_Thymus, Mouse_Brain),
plot per-method SC_Score vs CM-GTC scatter, colored by method.

Usage:
    python plot_edfig4_wogt_cmgtc.py --root /path/to/SMOBench-CLEAN
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import sys as _sys
_sys.path.insert(0, __import__('os').path.dirname(__file__))
from style_config import apply_style, PAL13
apply_style()

import matplotlib.pyplot as plt

WOGT_DATASETS = ['Mouse_Thymus', 'Mouse_Spleen', 'Mouse_Brain']

VERTICAL_METHODS = [
    'CANDIES', 'COSMOS', 'MISO', 'MultiGATE', 'PRAGA', 'PRESENT',
    'SMOPCA', 'SpaBalance', 'SpaFusion', 'SpaMI', 'SpaMosaic',
    'SpaMultiVAE', 'SpaMV', 'SpatialGlue', 'SWITCH',
]

# Color palette for methods
METHOD_COLORS = {}
_cmap = plt.colormaps.get_cmap('tab20').resampled(15)
for i, m in enumerate(VERTICAL_METHODS):
    METHOD_COLORS[m] = _cmap(i)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--clustering', type=str, default='leiden')
    parser.add_argument('--dpi', type=int, default=300)
    return parser.parse_args()


def load_wogt_scores(root, clustering):
    """Load SC_Score from woGT evaluation CSVs (row-based Metric/Value format)."""
    eval_dir = os.path.join(root, '_myx_Results', 'evaluation', 'vertical')
    rows = []
    for method in VERTICAL_METHODS:
        for dataset in WOGT_DATASETS:
            ds_dir = os.path.join(eval_dir, method, dataset)
            if not os.path.isdir(ds_dir):
                continue
            import glob
            csvs = glob.glob(os.path.join(ds_dir, f'*_{clustering}_woGT.csv'))
            for csv_path in csvs:
                try:
                    df = pd.read_csv(csv_path)
                    if 'Metric' not in df.columns:
                        continue
                    # Extract slice name from filename
                    bn = os.path.basename(csv_path)
                    # e.g., CANDIES_Mouse_Thymus_Thymus1_leiden_woGT.csv
                    prefix = f"{method}_{dataset}_"
                    rest = bn[len(prefix):]
                    slice_name = rest.split(f'_{clustering}')[0]

                    sc_row = df[df['Metric'] == 'SC_Score']
                    sc_val = float(sc_row['Value'].iloc[0]) if len(sc_row) else np.nan
                    sil_row = df[df['Metric'] == 'Silhouette Coefficient']
                    sil_val = float(sil_row['Value'].iloc[0]) if len(sil_row) else np.nan
                    dbi_row = df[df['Metric'] == 'Davies-Bouldin Index']
                    dbi_val = float(dbi_row['Value'].iloc[0]) if len(dbi_row) else np.nan

                    rows.append({
                        'Method': method, 'Dataset': dataset,
                        'Slice': slice_name,
                        'SC_Score': sc_val,
                        'Silhouette': sil_val,
                        'DBI': dbi_val,
                    })
                except Exception:
                    pass
    return pd.DataFrame(rows)


def load_cmgtc(root):
    """Load CM-GTC scores from cmgtc_vertical.csv."""
    path = os.path.join(root, '_myx_Results', 'evaluation', 'cmgtc', 'cmgtc_vertical.csv')
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No CM-GTC results at {path}")
    df = pd.read_csv(path)
    return df[df['Dataset'].isin(WOGT_DATASETS)]


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    out_dir = os.path.join(root, '_myx_Results', 'plots')
    os.makedirs(out_dir, exist_ok=True)

    print("Loading woGT evaluation scores...")
    df_scores = load_wogt_scores(root, args.clustering)
    print(f"  {len(df_scores)} entries")

    print("Loading CM-GTC scores...")
    df_cmgtc = load_cmgtc(root)
    print(f"  {len(df_cmgtc)} entries")

    # Merge on Method + Dataset + Slice
    merged = pd.merge(df_scores, df_cmgtc[['Method', 'Dataset', 'Slice', 'CM_GTC']],
                       on=['Method', 'Dataset', 'Slice'], how='inner')
    print(f"  Merged: {len(merged)} entries")

    if merged.empty:
        print("ERROR: No merged data")
        return

    # Average per method per dataset
    avg = merged.groupby(['Method', 'Dataset']).agg(
        SC_Score=('SC_Score', 'mean'),
        CM_GTC=('CM_GTC', 'mean'),
    ).reset_index()

    # --- Plot 1: Per-dataset scatter (taller panels) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, dataset in enumerate(WOGT_DATASETS):
        ax = axes[i]
        df_ds = avg[avg['Dataset'] == dataset]
        for _, row in df_ds.iterrows():
            ax.scatter(row['SC_Score'], row['CM_GTC'],
                      color=METHOD_COLORS.get(row['Method'], 'gray'),
                      s=80, alpha=0.85, edgecolors='white', linewidths=0.5,
                      zorder=3)
            ax.annotate(
                row['Method'], (row['SC_Score'], row['CM_GTC']),
                fontsize=5.5, alpha=0.8,
                xytext=(5, 5), textcoords='offset points',
                clip_on=True)

        ax.set_xlabel("SC Score (Moran's I)", fontsize=11)
        ax.set_ylabel('CM-GTC', fontsize=11)
        title = dataset.replace('_', ' ')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        # Add some padding to axes limits
        xmin, xmax = df_ds['SC_Score'].min(), df_ds['SC_Score'].max()
        ymin, ymax = df_ds['CM_GTC'].min(), df_ds['CM_GTC'].max()
        xpad = (xmax - xmin) * 0.15
        ypad = (ymax - ymin) * 0.15
        ax.set_xlim(xmin - xpad, xmax + xpad * 2)
        ax.set_ylim(ymin - ypad, ymax + ypad * 2)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'edfig4_wogt_sc_vs_cmgtc_{args.clustering}.png')
    plt.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    # --- Plot 2: Combined scatter with dataset shapes ---
    fig, ax = plt.subplots(figsize=(7.5, 6))
    markers = {'Mouse_Thymus': 'o', 'Mouse_Spleen': 's', 'Mouse_Brain': '^'}

    texts = []
    for _, row in avg.iterrows():
        ax.scatter(row['SC_Score'], row['CM_GTC'],
                  color=METHOD_COLORS.get(row['Method'], 'gray'),
                  marker=markers.get(row['Dataset'], 'o'),
                  s=80, alpha=0.85, edgecolors='white', linewidths=0.5,
                  zorder=3)

    # Legend for methods (outside plot to save space)
    from matplotlib.lines import Line2D
    method_handles = [Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=METHOD_COLORS[m], markersize=7, label=m)
                      for m in VERTICAL_METHODS if m in avg['Method'].values]
    ds_handles = [Line2D([0], [0], marker=markers[d], color='gray',
                         markersize=8, label=d.replace('_', ' '), linestyle='None')
                  for d in WOGT_DATASETS]

    leg1 = ax.legend(handles=method_handles, loc='upper left', fontsize=7,
                     ncol=2, title='Method', title_fontsize=8,
                     framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=ds_handles, loc='lower right', fontsize=8, title='Dataset',
              title_fontsize=9)

    ax.set_xlabel("SC Score (Moran's I)", fontsize=12)
    ax.set_ylabel('CM-GTC', fontsize=12)
    ax.set_title(f'woGT Datasets: SC Score vs CM-GTC ({args.clustering})',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path2 = os.path.join(out_dir, f'edfig4_wogt_combined_{args.clustering}.png')
    plt.savefig(out_path2, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path2}")


if __name__ == '__main__':
    main()
