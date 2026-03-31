#!/usr/bin/env python3
"""
SMOBench Figure 2(b,c): SC vs BioC Scatter Plots
(b) RNA-ADT datasets: x=SC, y=BioC, bubble_size=CM-GTC, color=method
(c) RNA-ATAC datasets: same layout

Usage:
    python plot_sc_bioc_scatter.py --root /path/to/SMOBench-CLEAN
    python plot_sc_bioc_scatter.py --root /path/to/SMOBench-CLEAN --clustering leiden
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from adjustText import adjust_text

sys.path.insert(0, os.path.dirname(__file__))
from style_config import apply_style, COLORS
apply_style()

METHOD_ORDER = [
    'CANDIES', 'COSMOS', 'MISO', 'MultiGATE', 'PRAGA', 'PRESENT',
    'SMOPCA', 'SpaBalance', 'SpaFusion', 'SpaMI', 'SpaMosaic',
    'SpaMultiVAE', 'SpaMV', 'SpatialGlue', 'SWITCH',
]

# Distinct colors for up to 15 methods
METHOD_COLORS = {m: plt.cm.tab20(i/15) for i, m in enumerate(METHOD_ORDER)}

# Main figure: withGT datasets only (BioC sub-metrics differ for woGT)
RNA_ADT_DATASETS = ['Human_Lymph_Nodes', 'Human_Tonsils']
RNA_ATAC_DATASETS = ['Mouse_Embryos_S1', 'Mouse_Embryos_S2']

# Supplementary: woGT datasets
RNA_ADT_WOGT = ['Mouse_Thymus', 'Mouse_Spleen']
RNA_ATAC_WOGT = ['Mouse_Brain']

DATASET_TYPES = {
    'Human_Lymph_Nodes': 'RNA_ADT', 'Human_Tonsils': 'RNA_ADT',
    'Mouse_Embryos_S1': 'RNA_ATAC', 'Mouse_Embryos_S2': 'RNA_ATAC',
    'Mouse_Thymus': 'RNA_ADT', 'Mouse_Spleen': 'RNA_ADT',
    'Mouse_Brain': 'RNA_ATAC',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--clustering', type=str, default='leiden')
    parser.add_argument('--dpi', type=int, default=300)
    return parser.parse_args()


def load_data(root, clustering):
    summary_dir = os.path.join(root, '_myx_Results', 'evaluation', 'summary')
    for prefix in ['vertical_final', 'vertical_detailed']:
        path = os.path.join(summary_dir, f'{prefix}_{clustering}.csv')
        if os.path.isfile(path):
            return pd.read_csv(path)
    raise FileNotFoundError(f"No vertical results in {summary_dir}")


def plot_scatter(df, modality_type, clustering, out_dir, dpi=300,
                 datasets=None, filename_prefix='fig2'):
    """Plot SC vs BioC scatter, bubble size + color = CM-GTC (green→red)."""
    if datasets is None:
        datasets = RNA_ADT_DATASETS if modality_type == 'RNA_ADT' else RNA_ATAC_DATASETS
    df_mod = df[df['Dataset'].isin(datasets)].copy()
    if df_mod.empty:
        return

    sc_col = 'SC_Score' if 'SC_Score' in df_mod.columns else None
    bioc_col = 'BioC_Score' if 'BioC_Score' in df_mod.columns else None
    cmgtc_col = 'CM_GTC' if 'CM_GTC' in df_mod.columns else None

    if sc_col is None or bioc_col is None:
        print(f"  Missing SC_Score or BioC_Score columns")
        return

    agg_cols = [c for c in [sc_col, bioc_col, cmgtc_col] if c is not None]
    method_avg = df_mod.groupby('Method')[agg_cols].agg(['mean', 'sem']).reset_index()

    # Collect CM-GTC values for normalization
    cmgtc_values = []
    rows_data = []
    for _, row in method_avg.iterrows():
        method = row['Method'].values[0] if hasattr(row['Method'], 'values') else row[('Method', '')]
        x = row[(sc_col, 'mean')]
        y = row[(bioc_col, 'mean')]
        x_err = row[(sc_col, 'sem')]
        y_err = row[(bioc_col, 'sem')]
        cmgtc = row[(cmgtc_col, 'mean')] if cmgtc_col else 0
        cmgtc_values.append(cmgtc)
        rows_data.append((method, x, y, x_err, y_err, cmgtc))

    cmgtc_arr = np.array(cmgtc_values)
    cmin, cmax = cmgtc_arr.min(), cmgtc_arr.max()
    crange = cmax - cmin if cmax > cmin else 1.0

    # Sky blue(low) → golden(mid) → coral red(high), from 13-color palette
    from style_config import PAL13
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'cmgtc', [PAL13[12], PAL13[11], PAL13[8], PAL13[0]], N=256)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Median reference lines (very thin, gray)
    xs = [r[1] for r in rows_data]
    ys = [r[2] for r in rows_data]
    ax.axhline(np.median(ys), color='#CCCCCC', ls='--', lw=0.5, zorder=1)
    ax.axvline(np.median(xs), color='#CCCCCC', ls='--', lw=0.5, zorder=1)

    # Manual offsets per method (tuned per modality)
    # RNA+ADT crowded: PRAGA/SpaMI, PRESENT/SMOPCA, SWITCH/SpatialGlue
    # RNA+ATAC crowded: CANDIES/SpaMI, MISO/PRESENT/SMOPCA, SWITCH/SpaBalance/SpaMosaic
    OFFSETS_ADT = {
        'MultiGATE':   (10, -10),
        'CANDIES':     (10, 8),
        'COSMOS':      (10, -12),
        'SpatialGlue': (-70, 12),
        'SWITCH':      (-55, -10),
        'SpaBalance':  (-65, 10),
        'PRAGA':       (-55, -12),
        'SpaMI':       (10, 10),
        'SpaMosaic':   (10, 8),
        'SpaMV':       (10, -8),
        'PRESENT':     (-55, -12),
        'SMOPCA':      (10, 10),
        'MISO':        (10, -8),
        'SpaFusion':   (-60, 10),
        'SpaMultiVAE': (-70, 8),
    }
    OFFSETS_ATAC = {
        'SpaMI':       (10, 10),
        'CANDIES':     (-55, 10),
        'COSMOS':      (10, -10),
        'SpatialGlue': (10, -15),
        'SpaBalance':  (-60, 10),
        'SWITCH':      (10, -12),
        'SpaMosaic':   (-60, 15),
        'SpaMV':       (10, -10),
        'PRAGA':       (-50, 10),
        'MISO':        (-50, -8),
        'SMOPCA':      (10, 10),
        'PRESENT':     (10, -10),
        'MultiGATE':   (10, -8),
    }
    offsets = OFFSETS_ADT if modality_type == 'RNA_ADT' else OFFSETS_ATAC

    for method, x, y, x_err, y_err, cmgtc in rows_data:
        norm_val = (cmgtc - cmin) / crange
        color = cmap(norm_val)

        ax.errorbar(x, y, xerr=x_err, yerr=y_err,
                    fmt='none', ecolor=color, alpha=0.5, capsize=2,
                    linewidth=0.8, zorder=2)

        ax.scatter(x, y, s=100, c=[color], alpha=0.85,
                   edgecolors='white', linewidth=0.4, zorder=5)

        offset = offsets.get(method, (8, 6))
        needs_arrow = abs(offset[0]) > 20 or abs(offset[1]) > 20

        ax.annotate(method, (x, y),
                    xytext=offset, textcoords='offset points',
                    fontsize=8, color='#000000',
                    arrowprops=dict(arrowstyle='-', color='#999999', lw=0.5)
                    if needs_arrow else None)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=cmin, vmax=cmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02, aspect=25)
    cbar.set_label('CM-GTC', fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(0.4)

    ax.set_xlabel('SC Score (Spatial Coherence)', fontsize=12)
    ax.set_ylabel('BioC Score (Biological Conservation)', fontsize=12)

    panel = 'b' if modality_type == 'RNA_ADT' else 'c'
    mod_label = modality_type.replace('_', '+')
    ax.set_title(f'{mod_label}', fontsize=13, fontweight='bold',
                 color='#000000')

    plt.tight_layout()

    out_path = os.path.join(out_dir, f'{filename_prefix}{panel}_sc_bioc_{modality_type}_{clustering}.pdf')
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    out_dir = os.path.join(root, '_myx_Results', 'plots')
    os.makedirs(out_dir, exist_ok=True)

    df = load_data(root, args.clustering)
    print(f"Loaded {len(df)} rows")

    # withGT (main figure)
    print("\n[withGT] Plotting RNA_ADT scatter...")
    plot_scatter(df, 'RNA_ADT', args.clustering, out_dir, args.dpi,
                 datasets=RNA_ADT_DATASETS, filename_prefix='fig2')

    print("[withGT] Plotting RNA_ATAC scatter...")
    plot_scatter(df, 'RNA_ATAC', args.clustering, out_dir, args.dpi,
                 datasets=RNA_ATAC_DATASETS, filename_prefix='fig2')

    # woGT (supplementary)
    print("\n[woGT] Plotting RNA_ADT scatter...")
    plot_scatter(df, 'RNA_ADT', args.clustering, out_dir, args.dpi,
                 datasets=RNA_ADT_WOGT, filename_prefix='supp_')

    print("[woGT] Plotting RNA_ATAC scatter...")
    plot_scatter(df, 'RNA_ATAC', args.clustering, out_dir, args.dpi,
                 datasets=RNA_ATAC_WOGT, filename_prefix='supp_')


if __name__ == '__main__':
    main()
