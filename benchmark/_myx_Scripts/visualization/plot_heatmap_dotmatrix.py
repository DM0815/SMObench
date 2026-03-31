#!/usr/bin/env python3
"""
SMOBench Dot-Matrix Heatmap — publication-quality, wide/flat layout.

Replaces all seaborn heatmaps (fig2a, fig3a, fig4a) with a dot-matrix style:
  - Colored circles (size + saturation ∝ value) for individual metrics
  - Blue horizontal bars for aggregate/composite scores
  - Grouped column headers with different colors per metric category
  - Wide, flat aspect ratio

Usage:
    python plot_heatmap_dotmatrix.py --root /path/to/SMOBench-CLEAN
    python plot_heatmap_dotmatrix.py --root /path/to/SMOBench-CLEAN --only fig2a
    python plot_heatmap_dotmatrix.py --root /path/to/SMOBench-CLEAN --only fig3a
    python plot_heatmap_dotmatrix.py --root /path/to/SMOBench-CLEAN --only fig4a
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
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors

# Import global style
sys.path.insert(0, os.path.dirname(__file__))
from style_config import apply_style, COLORS, METRIC_COLORS
apply_style()


# ── Colour palette (from style_config, TEAM-aligned) ───────────────────────
def make_cmap(base_color):
    """Create a white→base colormap."""
    return mcolors.LinearSegmentedColormap.from_list(
        'custom', ['#FAFAFA', base_color], N=256)

CMAP_SC   = make_cmap(METRIC_COLORS['SC'])
CMAP_BIOC = make_cmap(METRIC_COLORS['BioC'])
CMAP_BER  = make_cmap(METRIC_COLORS['BER'])
CMAP_CMGTC = make_cmap(METRIC_COLORS['CMGTC'])
CMAP_BAR  = make_cmap(METRIC_COLORS['agg'])


# ── Column definitions ─────────────────────────────────────────────────────
VERT_GROUPS = [
    {
        'title': 'Spatial\nCoherence',
        'cols': ['Moran_Index'],
        'cmap': CMAP_SC,
        'style': 'dot',
    },
    {
        'title': 'Biological\nConservation',
        'cols': ['ARI', 'NMI', 'asw_celltype', 'graph_clisi'],
        'cmap': CMAP_BIOC,
        'style': 'dot',
    },
    {
        'title': 'Aggregate\nScore',
        'cols': ['SC_Score', 'BioC_Score', 'CM_GTC', 'SMOBench_V'],
        'cmap': CMAP_BAR,
        'style': 'bar',
    },
]

HORIZ_GROUPS = [
    {
        'title': 'Spatial\nCoherence',
        'cols': ['Moran_Index'],
        'cmap': CMAP_SC,
        'style': 'dot',
    },
    {
        'title': 'Biological\nConservation',
        'cols': ['ARI', 'NMI', 'asw_celltype', 'graph_clisi'],
        'cmap': CMAP_BIOC,
        'style': 'dot',
    },
    {
        'title': 'Batch Effect\nRemoval',
        'cols': ['kBET', 'KNN_connectivity', 'bASW', 'iLISI', 'PCR'],
        'cmap': CMAP_BER,
        'style': 'dot',
    },
    {
        'title': 'Aggregate\nScore',
        'cols': ['SC_Score', 'BioC_Score', 'BER_Score', 'CM_GTC', 'SMOBench_H'],
        'cmap': CMAP_BAR,
        'style': 'bar',
    },
]

THREEM_GROUPS = [
    {
        'title': 'Spatial\nCoherence',
        'cols': ['Moran_Index'],
        'cmap': CMAP_SC,
        'style': 'dot',
    },
    {
        'title': 'Biological\nConservation',
        'cols': ['ARI', 'NMI', 'asw_celltype', 'graph_clisi'],
        'cmap': CMAP_BIOC,
        'style': 'dot',
    },
    {
        'title': 'Aggregate\nScore',
        'cols': ['SC_Score', 'BioC_Score', 'CM_GTC', 'SMOBench_V'],
        'cmap': CMAP_BAR,
        'style': 'bar',
    },
]

# Display labels
DISPLAY_LABELS = {
    'Moran_Index': "Moran's\nIndex",
    'asw_celltype': 'ASW\ncelltype',
    'graph_clisi': 'Graph\ncLISI',
    'KNN_connectivity': 'KNN\nconn.',
    'Davies-Bouldin_Index_normalized': 'DBI\n(norm)',
    'Silhouette_Coefficient': 'Silhouette',
    'Calinski-Harabaz_Index_normalized': 'CHI\n(norm)',
    'SC_Score': 'Spatial\nCoherence',
    'BioC_Score': 'Bio\nConserv.',
    'BER_Score': 'Batch\nRemoval',
    'CM_GTC': 'CM-GTC',
    'SMOBench_V': 'Total',
    'SMOBench_H': 'Total',
}


# ── Core drawing function ──────────────────────────────────────────────────
def plot_dotmatrix(df, groups, out_path, sort_by=None, dpi=300,
                   row_height=0.38, col_width=0.82, bar_width_factor=0.80,
                   dot_max_radius=0.155, fontsize_val=6.5, fontsize_label=7.5):
    """
    Draw a dot-matrix heatmap matching the reference template style.
    Font: Arial. No dot edges. Clean thin lines.
    """
    # ── Global font: Arial ──────────────────────────────────────────────
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    # Collect all columns that exist in df
    all_cols = []
    valid_groups = []
    for g in groups:
        existing = [c for c in g['cols'] if c in df.columns]
        if existing:
            valid_groups.append({**g, 'cols': existing})
            all_cols.extend(existing)

    if not all_cols:
        print(f"  No valid columns found, skipping")
        return

    # Sort methods
    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)

    methods = df.index.tolist()
    n_methods = len(methods)
    n_cols = len(all_cols)

    # Per-column min-max for coloring (0–1 normalization)
    col_min = df[all_cols].min()
    col_max = df[all_cols].max()
    col_range = col_max - col_min
    col_range[col_range == 0] = 1

    # ── Figure layout ───────────────────────────────────────────────────
    left_margin = 1.3
    right_margin = 0.1
    top_margin = 0.9
    bot_margin = 0.15
    gap = 0.25          # gap between metric groups

    total_gap = gap * (len(valid_groups) - 1)
    fig_w = left_margin + n_cols * col_width + total_gap + right_margin
    fig_h = top_margin + n_methods * row_height + bot_margin

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis('off')

    # ── Compute column x-positions ──────────────────────────────────────
    col_x = {}
    group_spans = []
    x_cursor = left_margin
    for gi, g in enumerate(valid_groups):
        x_start = x_cursor
        for ci, col in enumerate(g['cols']):
            col_x[col] = x_cursor + col_width / 2
            x_cursor += col_width
        x_end = x_cursor
        group_spans.append((x_start, x_end, g['title'], g['cmap']))
        if gi < len(valid_groups) - 1:
            x_cursor += gap

    # ── TOP solid line ─────────────────────────────────────────────────
    top_line_y = fig_h - 0.05
    ax.plot([left_margin - 0.3, x_cursor + 0.05], [top_line_y, top_line_y],
            color='#000000', linewidth=1.0, solid_capstyle='butt')

    # ── Draw group headers (bold, black, centered) ──────────────────────
    header_y = fig_h - 0.30
    for x_start, x_end, title, cmap in group_spans:
        cx = (x_start + x_end) / 2
        ax.text(cx, header_y, title, ha='center', va='center',
                fontsize=fontsize_label + 1, fontweight='bold',
                color='#000000', linespacing=1.05)

    # ── "Method" label in top-left ──────────────────────────────────────
    label_y = fig_h - top_margin + 0.18
    ax.text(left_margin - 0.12, label_y + 0.04, 'Method', ha='right', va='bottom',
            fontsize=fontsize_label - 1, color='#666666', fontstyle='italic')

    # ── Draw column labels ──────────────────────────────────────────────
    for col in all_cols:
        lbl = DISPLAY_LABELS.get(col, col)
        ax.text(col_x[col], label_y, lbl, ha='center', va='bottom',
                fontsize=fontsize_label - 1, color='#666666',
                fontweight='normal', linespacing=1.0)

    # ── Separator line below column labels (colored per group) ──────────
    sep_y = label_y - 0.06
    for x_start, x_end, title, cmap in group_spans:
        base_color = cmap(0.65)
        ax.plot([x_start, x_end], [sep_y, sep_y],
                color=base_color, linewidth=1.5, solid_capstyle='butt')

    # ── Draw rows ───────────────────────────────────────────────────────
    for ri, method in enumerate(methods):
        y = sep_y - 0.08 - (ri + 0.5) * row_height

        # Row separator line (thin gray, between rows)
        if ri > 0:
            line_y = y + row_height / 2
            ax.plot([left_margin - 0.2, x_cursor], [line_y, line_y],
                    color='#E8E8E8', linewidth=0.4, zorder=1)

        # Method name (bold, right-aligned)
        ax.text(left_margin - 0.12, y, method, ha='right', va='center',
                fontsize=fontsize_label, fontweight='bold', color='#000000')

        # Draw each cell
        for g in valid_groups:
            cmap = g['cmap']
            style = g['style']
            for col in g['cols']:
                cx = col_x[col]
                raw_val = df.loc[method, col]

                if pd.isna(raw_val):
                    ax.text(cx, y, '---', ha='center', va='center',
                            fontsize=fontsize_val - 1, color='#BBBBBB')
                    continue

                norm_val = (raw_val - col_min[col]) / col_range[col]
                norm_val = np.clip(norm_val, 0, 1)

                if style == 'dot':
                    # Circle: uniform max size, color depth only
                    radius = dot_max_radius
                    color = cmap(0.15 + 0.85 * norm_val)
                    circle = plt.Circle((cx, y), radius,
                                        facecolor=color, edgecolor='none',
                                        transform=ax.transData, zorder=3)
                    ax.add_patch(circle)
                    # Text color based on actual background luminance
                    r, g, b = color[:3]
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    text_color = 'white' if lum < 0.55 else '#000000'
                    ax.text(cx, y, f'{raw_val:.2f}', ha='center', va='center',
                            fontsize=fontsize_val, color=text_color, zorder=4,
                            fontweight='normal')

                elif style == 'bar':
                    bar_h = row_height * 0.50
                    bar_max_w = col_width * bar_width_factor
                    bar_w = bar_max_w * max(norm_val, 0.10)
                    color = cmap(0.35 + 0.65 * norm_val)

                    rect = FancyBboxPatch(
                        (cx - bar_max_w / 2, y - bar_h / 2),
                        bar_w, bar_h,
                        boxstyle="round,pad=0.02",
                        facecolor=color, edgecolor='none', zorder=3)
                    ax.add_patch(rect)
                    r, g, b = color[:3]
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    text_color = 'white' if lum < 0.55 else '#000000'
                    ax.text(cx, y, f'{raw_val:.2f}', ha='center', va='center',
                            fontsize=fontsize_val, color=text_color, zorder=4,
                            fontweight='normal')

    # ── BOTTOM solid line ──────────────────────────────────────────────
    last_y = sep_y - 0.08 - (n_methods) * row_height
    ax.plot([left_margin - 0.3, x_cursor + 0.05], [last_y, last_y],
            color='#000000', linewidth=1.0, solid_capstyle='butt')

    plt.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white',
                pad_inches=0.08)
    plt.close()
    print(f'  Saved: {out_path}')


# ── Data loading & figure-specific wrappers ─────────────────────────────────

# 2×2 dataset groups: modality × GT
DATASET_GROUPS = [
    ('RNA+ADT',  'withGT', ['Human_Lymph_Nodes', 'Human_Tonsils']),
    ('RNA+ADT',  'woGT',   ['Mouse_Thymus', 'Mouse_Spleen']),
    ('RNA+ATAC', 'withGT', ['Mouse_Embryos_S1', 'Mouse_Embryos_S2']),
    ('RNA+ATAC', 'woGT',   ['Mouse_Brain']),
]

# woGT uses different BioC sub-metrics
VERT_GROUPS_WOGT = [
    {
        'title': 'Spatial\nCoherence',
        'cols': ['Moran_Index'],
        'cmap': CMAP_SC,
        'style': 'dot',
    },
    {
        'title': 'Biological\nConservation',
        'cols': ['Davies-Bouldin_Index_normalized', 'Silhouette_Coefficient',
                 'Calinski-Harabaz_Index_normalized'],
        'cmap': CMAP_BIOC,
        'style': 'dot',
    },
    {
        'title': 'Aggregate\nScore',
        'cols': ['SC_Score', 'BioC_Score', 'CM_GTC', 'SMOBench_V'],
        'cmap': CMAP_BAR,
        'style': 'bar',
    },
]

HORIZ_GROUPS_WOGT = [
    {
        'title': 'Spatial\nCoherence',
        'cols': ['Moran_Index'],
        'cmap': CMAP_SC,
        'style': 'dot',
    },
    {
        'title': 'Biological\nConservation',
        'cols': ['Davies-Bouldin_Index_normalized', 'Silhouette_Coefficient',
                 'Calinski-Harabaz_Index_normalized'],
        'cmap': CMAP_BIOC,
        'style': 'dot',
    },
    {
        'title': 'Batch Effect\nRemoval',
        'cols': ['kBET', 'KNN_connectivity', 'bASW', 'iLISI', 'PCR'],
        'cmap': CMAP_BER,
        'style': 'dot',
    },
    {
        'title': 'Aggregate\nScore',
        'cols': ['SC_Score', 'BioC_Score', 'BER_Score', 'CM_GTC', 'SMOBench_H'],
        'cmap': CMAP_BAR,
        'style': 'bar',
    },
]

VERT_METHODS = [
    'CANDIES', 'COSMOS', 'MISO', 'MultiGATE', 'PRAGA', 'PRESENT',
    'SMOPCA', 'SpaBalance', 'SpaFusion', 'SpaMI', 'SpaMosaic',
    'SpaMultiVAE', 'SpaMV', 'SpatialGlue', 'SWITCH',
]
HORIZ_METHODS = [
    'CANDIES', 'COSMOS', 'MISO', 'PRAGA', 'PRESENT',
    'SMOPCA', 'SpaBalance', 'SpaFusion', 'SpaMI', 'SpaMosaic',
    'SpaMultiVAE', 'SpaMV', 'SpatialGlue',
]
THREEM_METHOD_MAP = {
    'SpatialGlue_3Mv2': 'SpatialGlue', 'SpaBalance_3Mv2': 'SpaBalance',
    'SMOPCA_3Mv2': 'SMOPCA', 'MISO_3Mv2': 'MISO',
    'PRESENT_3Mv2': 'PRESENT', 'SpaMV_3Mv2': 'SpaMV',
    'PRAGA_3Mv2': 'PRAGA',
}


def load_vertical(root, clustering='leiden'):
    summary = os.path.join(root, '_myx_Results', 'evaluation', 'summary')
    for name in [f'vertical_final_{clustering}.csv',
                 f'vertical_detailed_{clustering}.csv']:
        p = os.path.join(summary, name)
        if os.path.isfile(p):
            return pd.read_csv(p)
    raise FileNotFoundError(f"No vertical CSV in {summary}")


def load_horizontal(root, clustering='leiden'):
    summary = os.path.join(root, '_myx_Results', 'evaluation', 'summary')
    for name in [f'horizontal_final_{clustering}.csv',
                 f'horizontal_detailed_{clustering}.csv']:
        p = os.path.join(summary, name)
        if os.path.isfile(p):
            return pd.read_csv(p)
    raise FileNotFoundError(f"No horizontal CSV in {summary}")


def load_3m(root):
    base = os.path.join(root, '_myx_Results', 'evaluation', '3m_v2')
    p = os.path.join(base, '3m_evaluation_results.csv')
    if not os.path.isfile(p):
        raise FileNotFoundError(f"No 3M CSV at {p}")
    df = pd.read_csv(p)
    # Rename columns
    df.columns = [c.replace(' ', '_').replace("'", '') for c in df.columns]
    if 'Moran_Index' not in df.columns and 'Moran_index' in df.columns:
        df.rename(columns={'Moran_index': 'Moran_Index'}, inplace=True)
    return df


def prepare_vertical(root, clustering, datasets, groups=None):
    """Prepare averaged DataFrame for vertical heatmap."""
    if groups is None:
        groups = VERT_GROUPS
    df = load_vertical(root, clustering)
    df.columns = [c.replace(' ', '_').replace("'", '') for c in df.columns]
    if 'Moran_Index' not in df.columns and 'Moran_index' in df.columns:
        df.rename(columns={'Moran_index': 'Moran_Index'}, inplace=True)

    df = df[df['Dataset'].isin(datasets)]

    metric_cols = [c for g in groups for c in g['cols'] if c in df.columns]
    agg = df.groupby('Method')[metric_cols].mean()

    methods = [m for m in VERT_METHODS if m in agg.index]
    return agg.loc[methods]


def prepare_horizontal(root, clustering, datasets, groups=None):
    """Prepare averaged DataFrame for horizontal heatmap."""
    if groups is None:
        groups = HORIZ_GROUPS
    df = load_horizontal(root, clustering)
    df.columns = [c.replace(' ', '_').replace("'", '') for c in df.columns]
    if 'Moran_Index' not in df.columns and 'Moran_index' in df.columns:
        df.rename(columns={'Moran_index': 'Moran_Index'}, inplace=True)
    if 'BVC_Score' in df.columns and 'BioC_Score' not in df.columns:
        df.rename(columns={'BVC_Score': 'BioC_Score'}, inplace=True)

    df = df[df['Dataset'].isin(datasets)]

    metric_cols = [c for g in groups for c in g['cols'] if c in df.columns]
    agg = df.groupby('Method')[metric_cols].mean()
    methods = [m for m in HORIZ_METHODS if m in agg.index]
    return agg.loc[methods]


def prepare_3m(root):
    df = load_3m(root)
    # Merge CM-GTC from separate file if not in main CSV
    if 'CM_GTC' not in df.columns:
        cmgtc_path = os.path.join(root, '_myx_Results', 'evaluation', '3m_v2',
                                  'cmgtc_3m_combined.csv')
        if os.path.isfile(cmgtc_path):
            df_cmgtc = pd.read_csv(cmgtc_path)
            df_cmgtc.columns = [c.replace(' ', '_') for c in df_cmgtc.columns]
            df = df.merge(df_cmgtc[['Method', 'CM_GTC']],
                          on='Method', how='left')
            print(f"  Merged CM_GTC from {cmgtc_path}")

    # Unify BVC_Score → BioC_Score
    if 'BVC_Score' in df.columns and 'BioC_Score' not in df.columns:
        df.rename(columns={'BVC_Score': 'BioC_Score'}, inplace=True)

    metric_cols = [c for g in THREEM_GROUPS for c in g['cols'] if c in df.columns]
    agg = df.groupby('Method')[metric_cols].mean()
    # Rename methods
    agg.index = [THREEM_METHOD_MAP.get(m, m) for m in agg.index]
    # Compute SMOBench_V if missing
    if 'SMOBench_V' not in agg.columns:
        if all(c in agg.columns for c in ['SC_Score', 'BioC_Score', 'CM_GTC']):
            agg['SMOBench_V'] = 0.2 * agg['SC_Score'] + 0.4 * agg['BioC_Score'] + 0.4 * agg['CM_GTC']
    return agg


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--clustering', type=str, default='leiden')
    parser.add_argument('--only', type=str, default=None,
                        help='Only generate specific figure: fig2a, fig3a, fig4a')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    out_dir = os.path.join(root, '_myx_Results', 'plots')
    os.makedirs(out_dir, exist_ok=True)

    do_all = args.only is None

    # ── Fig 2a: Vertical heatmaps (2×2: modality × GT) ─────────────────
    if do_all or 'fig2a' in args.only:
        print('=== Fig 2a: Vertical Heatmap (dot-matrix) ===')
        for mod_label, gt_type, ds_list in DATASET_GROUPS:
            groups = VERT_GROUPS if gt_type == 'withGT' else VERT_GROUPS_WOGT
            prefix = 'fig2a' if gt_type == 'withGT' else 'supp'
            mod_safe = mod_label.replace('+', '_')
            try:
                df = prepare_vertical(root, args.clustering, ds_list, groups)
                out = os.path.join(out_dir,
                    f'{prefix}_vertical_heatmap_{mod_safe}_{gt_type}_{args.clustering}.pdf')
                plot_dotmatrix(df, groups, out, sort_by='SMOBench_V',
                               dpi=args.dpi)
            except Exception as e:
                print(f'  Error ({mod_label} {gt_type}): {e}')

    # ── Fig 3a: Horizontal heatmaps (2×2: modality × GT) ────────────────
    if do_all or 'fig3a' in args.only:
        print('\n=== Fig 3a: Horizontal Heatmap (dot-matrix) ===')
        for mod_label, gt_type, ds_list in DATASET_GROUPS:
            groups = HORIZ_GROUPS if gt_type == 'withGT' else HORIZ_GROUPS_WOGT
            prefix = 'fig3a' if gt_type == 'withGT' else 'supp'
            mod_safe = mod_label.replace('+', '_')
            try:
                df = prepare_horizontal(root, args.clustering, ds_list, groups)
                out = os.path.join(out_dir,
                    f'{prefix}_horizontal_heatmap_{mod_safe}_{gt_type}_{args.clustering}.pdf')
                plot_dotmatrix(df, groups, out, sort_by='SMOBench_H',
                               dpi=args.dpi)
            except Exception as e:
                print(f'  Error ({mod_label} {gt_type}): {e}')

    # ── Fig 4a: 3M heatmap ──────────────────────────────────────────────
    if do_all or 'fig4a' in args.only:
        print('\n=== Fig 4a: 3M Heatmap (dot-matrix) ===')
        try:
            df = prepare_3m(root)
            out = os.path.join(out_dir,
                f'fig4a_3m_heatmap_{args.clustering}.pdf')
            plot_dotmatrix(df, THREEM_GROUPS, out, sort_by='SMOBench_V',
                           dpi=args.dpi)
        except Exception as e:
            print(f'  Error: {e}')

    print('\nDone.')


if __name__ == '__main__':
    main()
