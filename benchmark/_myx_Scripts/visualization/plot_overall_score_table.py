#!/usr/bin/env python3
"""
Overall Score Summary Table (dot-matrix style).
Rows = methods, Columns = V_ADT, V_ATAC, H_ADT, H_ATAC, Overall.
Sorted by Overall score. withGT only (main), woGT separate.

Usage:
    python plot_overall_score_table.py --root /path/to/SMOBench-CLEAN
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(__file__))
from style_config import apply_style, METRIC_COLORS
apply_style()

DATASET_GROUPS = {
    'V_RNA+ADT':  ('vertical',   ['Human_Lymph_Nodes', 'Human_Tonsils']),
    'V_RNA+ATAC': ('vertical',   ['Mouse_Embryos_S1', 'Mouse_Embryos_S2']),
    'H_RNA+ADT':  ('horizontal', ['Human_Lymph_Nodes', 'Human_Tonsils']),
    'H_RNA+ATAC': ('horizontal', ['Mouse_Embryos_S1', 'Mouse_Embryos_S2']),
}

WOGT_GROUPS = {
    'V_RNA+ADT':  ('vertical',   ['Mouse_Thymus', 'Mouse_Spleen']),
    'V_RNA+ATAC': ('vertical',   ['Mouse_Brain']),
    'H_RNA+ADT':  ('horizontal', ['Mouse_Thymus', 'Mouse_Spleen']),
    'H_RNA+ATAC': ('horizontal', ['Mouse_Brain']),
}

ALL_METHODS = [
    'CANDIES', 'COSMOS', 'MISO', 'MultiGATE', 'PRAGA', 'PRESENT',
    'SMOPCA', 'SpaBalance', 'SpaFusion', 'SpaMI', 'SpaMosaic',
    'SpaMultiVAE', 'SpaMV', 'SpatialGlue', 'SWITCH',
]

COLUMNS = ['V_RNA+ADT', 'V_RNA+ATAC', 'H_RNA+ADT', 'H_RNA+ATAC', 'Overall']

COL_LABELS = {
    'V_RNA+ADT': 'Vertical\nRNA+ADT',
    'V_RNA+ATAC': 'Vertical\nRNA+ATAC',
    'H_RNA+ADT': 'Horizontal\nRNA+ADT',
    'H_RNA+ATAC': 'Horizontal\nRNA+ATAC',
    'Overall': 'Overall',
}

GROUP_HEADERS = [
    ('Vertical Integration', ['V_RNA+ADT', 'V_RNA+ATAC']),
    ('Horizontal Integration', ['H_RNA+ADT', 'H_RNA+ATAC']),
    ('', ['Overall']),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--clustering', type=str, default='leiden')
    parser.add_argument('--dpi', type=int, default=300)
    return parser.parse_args()


def load_final(root, task, clustering):
    summary = os.path.join(root, '_myx_Results', 'evaluation', 'summary')
    for prefix in ['final', 'detailed']:
        path = os.path.join(summary, f'{task}_{prefix}_{clustering}.csv')
        if os.path.isfile(path):
            df = pd.read_csv(path)
            df.columns = [c.replace(' ', '_') for c in df.columns]
            return df
    return None


def build_score_table(root, clustering, groups):
    """Build method × scenario score table."""
    score_col_map = {'vertical': 'SMOBench_V', 'horizontal': 'SMOBench_H'}

    records = {m: {} for m in ALL_METHODS}

    for col_name, (task, datasets) in groups.items():
        df = load_final(root, task, clustering)
        if df is None:
            continue
        score_col = score_col_map[task]
        if score_col not in df.columns:
            continue
        df_sub = df[df['Dataset'].isin(datasets)]
        avg = df_sub.groupby('Method')[score_col].mean()
        for m in ALL_METHODS:
            if m in avg.index:
                records[m][col_name] = avg[m]

    table = pd.DataFrame(records).T
    # Overall = mean of available scores
    if len(table.columns) > 0:
        table['Overall'] = table.mean(axis=1)
    table = table.sort_values('Overall', ascending=False)
    # Only keep methods that have at least one score
    table = table.dropna(how='all')
    return table


def plot_score_table(table, out_path, title='', dpi=300):
    """Draw landscape dot-matrix: methods as columns, scenarios as rows.
    Style matched to plot_heatmap_dotmatrix.py."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    methods = list(table.index)
    rows = [c for c in COLUMNS if c in table.columns]
    n_methods = len(methods)
    n_rows = len(rows)

    # Colors matched to dotmatrix: SC=red, BioC=green, BER=purple, agg=blue-gray
    row_colors = {}
    for r in rows:
        if r.startswith('V_'):
            row_colors[r] = METRIC_COLORS['BioC']    # green #52a65e
        elif r.startswith('H_'):
            row_colors[r] = METRIC_COLORS['BER']     # purple #8c529a
        else:
            row_colors[r] = METRIC_COLORS['agg']     # blue-gray #496d87

    def make_cmap(base_color):
        return mcolors.LinearSegmentedColormap.from_list('c', ['#FAFAFA', base_color], N=256)

    # Layout (landscape) — matched to dotmatrix proportions
    col_width = 0.82
    row_height = 0.50
    left_margin = 1.8
    top_margin = 1.0

    fig_w = left_margin + n_methods * col_width + 0.5
    fig_h = top_margin + n_rows * row_height + 0.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis('off')

    # Method names along the top (rotated 45°)
    for ci, method in enumerate(methods):
        x = left_margin + ci * col_width + col_width / 2
        y = fig_h - top_margin + 0.1
        ax.text(x, y, method, fontsize=7.5, fontweight='bold', ha='left', va='bottom',
                rotation=45, color='#222222')

    # Separator line
    sep_y = fig_h - top_margin - 0.05
    ax.plot([left_margin - 0.2, left_margin + n_methods * col_width + 0.1],
            [sep_y, sep_y], color='#CCCCCC', linewidth=0.8)

    # Row labels on left + data — fontsize matched to dotmatrix (7.5 for labels, 6.5 for values)
    for ri, r in enumerate(rows):
        y = sep_y - 0.05 - (ri + 0.5) * row_height
        label = COL_LABELS.get(r, r).replace('\n', ' ')
        ax.text(left_margin - 0.15, y, label, fontsize=7.5, ha='right', va='center',
                color='#555555', fontweight='bold' if r == 'Overall' else 'normal')

        base_color = row_colors[r]
        cmap = make_cmap(base_color)

        # Normalize within this row
        row_vals = table[r].dropna() if r in table.columns else pd.Series()
        vmin, vmax = (row_vals.min(), row_vals.max()) if len(row_vals) > 0 else (0, 1)

        for ci, method in enumerate(methods):
            x = left_margin + ci * col_width + col_width / 2
            val = table.loc[method, r] if r in table.columns else np.nan

            if pd.isna(val):
                ax.text(x, y, '—', fontsize=6.5, ha='center', va='center', color='#CCCCCC')
                continue

            norm_val = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5

            if r == 'Overall':
                # Bar style — matched to dotmatrix aggregate bar
                bar_w = col_width * 0.80 * norm_val
                bar_h = row_height * 0.45
                rect = plt.Rectangle((x - col_width * 0.40, y - bar_h / 2),
                                     bar_w, bar_h, facecolor=cmap(norm_val),
                                     edgecolor='none', alpha=0.85)
                ax.add_patch(rect)
                ax.text(x, y, f'{val:.3f}', fontsize=6.5, ha='center', va='center',
                        color='#222222', fontweight='bold')
            else:
                # Dot style — radius matched to dotmatrix (max ~0.155)
                radius = 0.10 + 0.055 * norm_val
                circle = plt.Circle((x, y), radius, facecolor=cmap(norm_val),
                                    edgecolor='none', alpha=0.85)
                ax.add_patch(circle)
                ax.text(x, y, f'{val:.3f}', fontsize=6.5, ha='center', va='center',
                        color='#222222')

    # Bottom line
    bottom_y = sep_y - 0.05 - n_rows * row_height
    ax.plot([left_margin - 0.2, left_margin + n_methods * col_width + 0.1],
            [bottom_y, bottom_y], color='#000000', linewidth=1.0)

    plt.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white', pad_inches=0.08)
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    out_dir = os.path.join(root, '_myx_Results', 'plots')
    os.makedirs(out_dir, exist_ok=True)

    # withGT
    table_wgt = build_score_table(root, args.clustering, DATASET_GROUPS)
    if not table_wgt.empty:
        out = os.path.join(out_dir, 'overall_score_table_withGT.pdf')
        plot_score_table(table_wgt, out, dpi=args.dpi)

    # woGT
    table_wogt = build_score_table(root, args.clustering, WOGT_GROUPS)
    if not table_wogt.empty:
        out = os.path.join(out_dir, 'overall_score_table_woGT.pdf')
        plot_score_table(table_wogt, out, dpi=args.dpi)


if __name__ == '__main__':
    main()
