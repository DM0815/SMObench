#!/usr/bin/env python3
"""
SpaMosaic Mosaic Integration — Dot-Matrix Heatmap
Same visual style as plot_heatmap_dotmatrix.py (fig2a/fig3a).

Layout: 7 datasets (rows) × 2 scenario panels side by side.
Each panel: SC dot + BVC dot + BER dot + Final bar.

Usage:
    python plot_mosaic_bubble.py --root /path/to/SMOBench-CLEAN
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(__file__))
from style_config import apply_style, METRIC_COLORS
apply_style()

# ── Colormaps (same as plot_heatmap_dotmatrix) ────────────────────────────
def make_cmap(base_color):
    return mcolors.LinearSegmentedColormap.from_list('c', ['#FAFAFA', base_color], N=256)

CMAP_SC   = make_cmap(METRIC_COLORS['SC'])
CMAP_BIOC = make_cmap(METRIC_COLORS['BioC'])
CMAP_BER  = make_cmap(METRIC_COLORS['BER'])
CMAP_BAR  = make_cmap(METRIC_COLORS['agg'])

# ── Dataset config ────────────────────────────────────────────────────────
DATASETS_ORDER = [
    ('HLN',          'Human Lymph'),
    ('HT',           'Human Tonsils'),
    ('Mouse_Spleen', 'Mouse Spleen'),
    ('MISAR_S1',     'Mouse Embryos(S1)'),
    ('MISAR_S2',     'Mouse Embryos(S2)'),
    ('Mouse_Thymus', 'Mouse Thymus'),
    ('Mouse_Brain',  'Mouse Brain'),
]

WOGT_DATASETS = {'Mouse_Spleen', 'Mouse_Thymus', 'Mouse_Brain'}
SCENARIOS = ['without_rna', 'without_second']
SCENARIO_LABELS = ['Without RNA', 'Without ADT/ATAC']

CMAP_CMGTC = make_cmap(METRIC_COLORS['CMGTC'])

# Columns per scenario panel
PANEL_COLS = [
    {'key': 'SC_Score',    'label': 'Spatial\nCoherence',  'cmap': CMAP_SC,    'style': 'dot'},
    {'key': 'BVC_Score',   'label': 'Bio\nConservation',   'cmap': CMAP_BIOC,  'style': 'dot'},
    {'key': 'BER_Score',   'label': 'Batch Effect\nRemoval', 'cmap': CMAP_BER, 'style': 'dot'},
    {'key': 'CM_GTC',      'label': 'CM-GTC',              'cmap': CMAP_CMGTC, 'style': 'dot'},
    {'key': 'Final_Score', 'label': 'Final\nScore',        'cmap': CMAP_BAR,   'style': 'bar'},
]


def load_mosaic_results(root, clustering='leiden'):
    mosaic_eval_dir = os.path.join(root, '_myx_Results', 'evaluation', 'mosaic')
    rows = []
    for ds_key, ds_label in DATASETS_ORDER:
        ds_dir = os.path.join(mosaic_eval_dir, ds_key)
        if not os.path.isdir(ds_dir):
            continue
        for scenario in SCENARIOS:
            for gt in ['withGT', 'woGT']:
                matches = glob.glob(os.path.join(ds_dir,
                    f'SpaMosaic_*_{scenario}_{clustering}_{gt}.csv'))
                if matches:
                    df = pd.read_csv(matches[0])
                    vals = {}
                    for _, row in df.iterrows():
                        try:
                            vals[str(row['Metric'])] = float(row['Value'])
                        except (ValueError, TypeError):
                            vals[str(row['Metric'])] = row['Value']

                    # woGT: BVC = raw Silhouette only
                    if ds_key in WOGT_DATASETS:
                        sil = float(vals.get('Silhouette Coefficient', 0))
                        vals['BVC_Score'] = sil
                        sc_s = float(vals.get('SC_Score', 0))
                        ber_s = float(vals.get('BER_Score', 0))
                        vals['Final_Score'] = (sc_s + sil + ber_s) / 3

                    vals['Dataset'] = ds_key
                    vals['Scenario'] = scenario
                    rows.append(vals)
                    break
    df = pd.DataFrame(rows)

    # Merge CM-GTC from per-dataset CSVs
    cmgtc_files = glob.glob(os.path.join(mosaic_eval_dir, 'mosaic_cmgtc_*.csv'))
    if cmgtc_files:
        cmgtc_dfs = [pd.read_csv(f) for f in cmgtc_files]
        cmgtc_all = pd.concat(cmgtc_dfs)
        cmgtc_agg = cmgtc_all.groupby(['Dataset', 'Scenario'])['CM_GTC_global'].mean().reset_index()
        cmgtc_agg.rename(columns={'CM_GTC_global': 'CM_GTC'}, inplace=True)
        df = df.merge(cmgtc_agg, on=['Dataset', 'Scenario'], how='left')

    return df


def plot_mosaic_dotmatrix(df, out_path, dpi=300):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    n_datasets = len(DATASETS_ORDER)
    n_panel_cols = len(PANEL_COLS)
    n_scenarios = len(SCENARIOS)

    # Layout params (same as plot_heatmap_dotmatrix)
    row_height = 0.42
    col_width = 0.88
    dot_max_radius = 0.16
    fontsize_val = 7.0
    fontsize_label = 7.5
    left_margin = 1.6
    panel_gap = 0.6   # gap between two scenario panels
    right_margin = 0.15
    top_margin = 1.1
    bot_margin = 0.15

    total_w = left_margin + n_scenarios * n_panel_cols * col_width + panel_gap + right_margin
    total_h = top_margin + n_datasets * row_height + bot_margin

    fig, ax = plt.subplots(figsize=(total_w, total_h))
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis('off')

    # Compute column x positions
    col_positions = {}  # (scenario_idx, col_idx) → x_center
    for si in range(n_scenarios):
        x_base = left_margin + si * (n_panel_cols * col_width + panel_gap)
        for ci in range(n_panel_cols):
            col_positions[(si, ci)] = x_base + ci * col_width + col_width / 2

    # Compute global min/max per metric for color normalization
    metric_keys = [c['key'] for c in PANEL_COLS]
    col_min = {}
    col_max = {}
    for key in metric_keys:
        vals = df[key].dropna().astype(float) if key in df.columns else pd.Series()
        col_min[key] = vals.min() if len(vals) > 0 else 0
        col_max[key] = vals.max() if len(vals) > 0 else 1
        if col_max[key] == col_min[key]:
            col_max[key] = col_min[key] + 1

    # ── Top line ──────────────────────────────────────────────────────────
    top_y = total_h - 0.05
    x_end = left_margin + n_scenarios * n_panel_cols * col_width + panel_gap
    ax.plot([left_margin - 0.4, x_end + 0.05], [top_y, top_y],
            color='#000000', linewidth=1.0, solid_capstyle='butt')

    # ── Scenario titles ───────────────────────────────────────────────────
    for si, label in enumerate(SCENARIO_LABELS):
        x_base = left_margin + si * (n_panel_cols * col_width + panel_gap)
        cx = x_base + n_panel_cols * col_width / 2
        ax.text(cx, total_h - 0.25, label, ha='center', va='center',
                fontsize=fontsize_label + 2, fontweight='bold', color='#000000')

    # ── Column labels ─────────────────────────────────────────────────────
    label_y = total_h - top_margin + 0.20
    ax.text(left_margin - 0.15, label_y + 0.04, 'Dataset', ha='right', va='bottom',
            fontsize=fontsize_label - 1, color='#666666', fontstyle='italic')

    for si in range(n_scenarios):
        for ci, col_def in enumerate(PANEL_COLS):
            x = col_positions[(si, ci)]
            ax.text(x, label_y, col_def['label'], ha='center', va='bottom',
                    fontsize=fontsize_label - 1, color='#666666',
                    fontweight='normal', linespacing=1.0)

    # ── Colored separator lines under column labels ───────────────────────
    sep_y = label_y - 0.06
    for si in range(n_scenarios):
        x_base = left_margin + si * (n_panel_cols * col_width + panel_gap)
        for ci, col_def in enumerate(PANEL_COLS):
            x_start = x_base + ci * col_width
            x_end_col = x_start + col_width
            base_color = col_def['cmap'](0.65)
            ax.plot([x_start, x_end_col], [sep_y, sep_y],
                    color=base_color, linewidth=1.5, solid_capstyle='butt')

    # ── Vertical separator between panels ─────────────────────────────────
    sep_x = left_margin + n_panel_cols * col_width + panel_gap / 2
    y_top = sep_y + 0.2
    y_bot = sep_y - 0.08 - n_datasets * row_height + 0.1
    ax.plot([sep_x, sep_x], [y_bot, y_top],
            color='#000000', linewidth=1.0, solid_capstyle='butt')

    # ── Draw rows ─────────────────────────────────────────────────────────
    for ri, (ds_key, ds_label) in enumerate(DATASETS_ORDER):
        y = sep_y - 0.08 - (ri + 0.5) * row_height

        # Row separator
        if ri > 0:
            line_y = y + row_height / 2
            ax.plot([left_margin - 0.3, left_margin + n_scenarios * n_panel_cols * col_width + panel_gap],
                    [line_y, line_y], color='#E8E8E8', linewidth=0.4, zorder=1)

        # Dataset label
        ax.text(left_margin - 0.15, y, ds_label, ha='right', va='center',
                fontsize=fontsize_label, fontweight='bold', color='#000000')

        # Draw cells
        for si, scenario in enumerate(SCENARIOS):
            df_row = df[(df['Dataset'] == ds_key) & (df['Scenario'] == scenario)]
            if df_row.empty:
                for ci in range(n_panel_cols):
                    cx = col_positions[(si, ci)]
                    ax.text(cx, y, '---', ha='center', va='center',
                            fontsize=fontsize_val - 1, color='#BBBBBB')
                continue
            row_vals = df_row.iloc[0]

            for ci, col_def in enumerate(PANEL_COLS):
                cx = col_positions[(si, ci)]
                key = col_def['key']
                cmap = col_def['cmap']
                style = col_def['style']

                raw_val = row_vals.get(key, np.nan)
                if pd.isna(raw_val):
                    ax.text(cx, y, '---', ha='center', va='center',
                            fontsize=fontsize_val - 1, color='#BBBBBB')
                    continue

                raw_val = float(raw_val)
                rng = col_max[key] - col_min[key]
                norm_val = np.clip((raw_val - col_min[key]) / rng, 0, 1)

                if style == 'dot':
                    color = cmap(0.15 + 0.85 * norm_val)
                    circle = plt.Circle((cx, y), dot_max_radius,
                                        facecolor=color, edgecolor='none',
                                        transform=ax.transData, zorder=3)
                    ax.add_patch(circle)
                    r, g, b = color[:3]
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    tc = 'white' if lum < 0.55 else '#000000'
                    ax.text(cx, y, f'{raw_val:.2f}', ha='center', va='center',
                            fontsize=fontsize_val, color=tc, zorder=4)

                elif style == 'bar':
                    bar_h = row_height * 0.50
                    bar_max_w = col_width * 0.80
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
                    tc = 'white' if lum < 0.55 else '#000000'
                    ax.text(cx, y, f'{raw_val:.2f}', ha='center', va='center',
                            fontsize=fontsize_val, color=tc, zorder=4)

    # ── Bottom line ───────────────────────────────────────────────────────
    last_y = sep_y - 0.08 - n_datasets * row_height
    ax.plot([left_margin - 0.4, left_margin + n_scenarios * n_panel_cols * col_width + panel_gap + 0.05],
            [last_y, last_y], color='#000000', linewidth=1.0, solid_capstyle='butt')

    plt.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white', pad_inches=0.08)
    plt.savefig(out_path.replace('.pdf', '.png'), dpi=dpi, bbox_inches='tight',
                facecolor='white', pad_inches=0.08)
    plt.close()
    print(f'Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--clustering', type=str, default='leiden')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    df = load_mosaic_results(root, args.clustering)
    print(f"Loaded {len(df)} rows")
    cols = ['Dataset', 'Scenario', 'SC_Score', 'BVC_Score', 'BER_Score', 'Final_Score']
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))

    if args.output:
        out = args.output
    else:
        out = os.path.join(root, '_myx_Results', 'plots', f'mosaic_bubble_{args.clustering}.pdf')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plot_mosaic_dotmatrix(df, out)


if __name__ == '__main__':
    main()
