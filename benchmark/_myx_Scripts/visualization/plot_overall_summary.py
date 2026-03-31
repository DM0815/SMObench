#!/usr/bin/env python3
"""
SMOBench Overall Summary Figure: V + H scores per method.

Layout:
  Columns = methods (sorted by overall score), each with RNA_ADT + RNA_ATAC sub-columns
  Row 1: Vertical Integration (blue circles)
  Row 2: Horizontal Integration (orange circles)
  Bottom: Overall Score bar chart

Usage:
    python plot_overall_summary.py --root /path/to/SMOBench-CLEAN
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(__file__))
from style_config import apply_style
apply_style()

# Task row colors
COLOR_V = '#3274A1'     # steel blue — Vertical
COLOR_H = '#E1812C'     # orange — Horizontal
COLOR_BAR = '#3A7D6E'   # teal green — Overall Score bars


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--clustering', default='leiden')
    p.add_argument('--dpi', type=int, default=300)
    return p.parse_args()


def load_modality_scores(summary_dir, clustering):
    """Load V/H scores split by RNA_ADT and RNA_ATAC."""
    scores = {}
    files = {
        'V_ADT':  f'vertical_RNA_ADT_{clustering}.csv',
        'V_ATAC': f'vertical_RNA_ATAC_{clustering}.csv',
        'H_ADT':  f'horizontal_RNA_ADT_withGT_{clustering}.csv',
        'H_ATAC': f'horizontal_RNA_ATAC_withGT_{clustering}.csv',
    }
    for key, fname in files.items():
        path = os.path.join(summary_dir, fname)
        if os.path.isfile(path):
            df = pd.read_csv(path)
            scores[key] = dict(zip(df['Method'], df['Average']))
    return scores


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    summary_dir = os.path.join(root, '_myx_Results', 'evaluation', 'summary')
    out_dir = os.path.join(root, '_myx_Results', 'plots')
    os.makedirs(out_dir, exist_ok=True)

    scores = load_modality_scores(summary_dir, args.clustering)

    # Compute overall score per method: mean of available V/H scores
    overall = {}
    all_methods_set = set()
    for key in scores:
        all_methods_set.update(scores[key].keys())

    for method in all_methods_set:
        vals = []
        for key in ['V_ADT', 'V_ATAC', 'H_ADT', 'H_ATAC']:
            if key in scores and method in scores[key]:
                v = scores[key][method]
                if pd.notna(v):
                    vals.append(v)
        if vals:
            overall[method] = np.mean(vals)

    # Sort methods by overall score descending
    sorted_methods = sorted(overall.keys(), key=lambda m: overall[m], reverse=True)
    n_methods = len(sorted_methods)

    # --- Layout constants ---
    col_w = 1.3
    row_h = 0.85
    bar_h = 1.6
    header_h = 1.0
    n_rows = 2  # Vertical + Horizontal
    left_margin = 2.8

    fig_w = left_margin + col_w * n_methods + 0.5
    fig_h = header_h + row_h * n_rows + 0.4 + bar_h + 0.8

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, left_margin + col_w * n_methods + 0.3)
    ax.set_ylim(-bar_h - 0.8, header_h + row_h * n_rows + 0.8)
    ax.axis('off')
    ax.set_aspect('equal')

    # --- Column headers: method names + RNA_ADT / RNA_ATAC ---
    top_y = header_h + row_h * n_rows
    for i, method in enumerate(sorted_methods):
        cx = left_margin + i * col_w + col_w / 2
        ax.text(cx, top_y + 0.45, method, ha='center', va='bottom',
                fontsize=7.5, fontweight='bold')
        for j, mod_label in enumerate(['RNA\nADT', 'RNA\nATAC']):
            sx = cx - 0.22 + j * 0.44
            ax.text(sx, top_y + 0.05, mod_label, ha='center', va='top',
                    fontsize=5, color='#888888', linespacing=0.85)

        # Dashed separator between methods
        if i < n_methods - 1:
            sep_x = left_margin + (i + 1) * col_w
            ax.plot([sep_x, sep_x], [header_h - 0.2, top_y],
                    color='#E0E0E0', ls=':', lw=0.5, zorder=0)

    # --- Row labels ---
    row_labels = ['Vertical\nIntegration', 'Horizontal\nIntegration']
    row_colors = [COLOR_V, COLOR_H]
    row_keys = [('V_ADT', 'V_ATAC'), ('H_ADT', 'H_ATAC')]

    for r, (label, color) in enumerate(zip(row_labels, row_colors)):
        ry = header_h + row_h * (n_rows - 1 - r) + row_h / 2
        ax.text(left_margin - 0.3, ry, label, ha='right', va='center',
                fontsize=8.5, fontweight='bold', color='#333333')

    # Top/bottom borders
    ax.plot([left_margin - 0.15, left_margin + col_w * n_methods + 0.15],
            [top_y, top_y], color='black', lw=1.0, clip_on=False)
    bot_border_y = header_h - 0.05
    ax.plot([left_margin - 0.15, left_margin + col_w * n_methods + 0.15],
            [bot_border_y, bot_border_y], color='black', lw=1.0, clip_on=False)

    # Row separator
    mid_y = header_h + row_h * 0.5 + row_h / 2 - row_h / 2
    ax.plot([left_margin - 0.15, left_margin + col_w * n_methods + 0.15],
            [header_h + row_h / 2, header_h + row_h / 2],
            color='#E0E0E0', lw=0.4, zorder=0)

    # --- Collect all values for color normalization ---
    all_vals = []
    for d in scores.values():
        all_vals.extend([v for v in d.values() if pd.notna(v)])
    vmin = min(all_vals) if all_vals else 0
    vmax = max(all_vals) if all_vals else 1

    radius = 0.17

    # --- Draw circles ---
    for r, (color, (key_adt, key_atac)) in enumerate(zip(row_colors, row_keys)):
        ry = header_h + row_h * (n_rows - 1 - r) + row_h / 2

        for i, method in enumerate(sorted_methods):
            cx = left_margin + i * col_w + col_w / 2

            for j, key in enumerate([key_adt, key_atac]):
                sx = cx - 0.22 + j * 0.44
                val = scores.get(key, {}).get(method, None)

                if val is not None and pd.notna(val):
                    # Normalize for color intensity
                    norm = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                    norm = max(0.15, min(1.0, norm))

                    # Blend white → base color
                    r_c, g_c, b_c = mcolors.to_rgb(color)
                    face = (1.0 + (r_c - 1.0) * norm,
                            1.0 + (g_c - 1.0) * norm,
                            1.0 + (b_c - 1.0) * norm)

                    circle = plt.Circle((sx, ry), radius, facecolor=face,
                                        edgecolor='none', zorder=5)
                    ax.add_patch(circle)

                    lum = 0.299 * face[0] + 0.587 * face[1] + 0.114 * face[2]
                    txt_c = 'white' if lum < 0.55 else '#333333'
                    ax.text(sx, ry, f'{val:.2f}', ha='center', va='center',
                            fontsize=5.2, fontweight='bold', color=txt_c, zorder=6)

    # --- Score label ---
    bar_top_y = bot_border_y - 0.35
    ax.text(left_margin - 0.3, bar_top_y - bar_h / 2, 'Score',
            ha='right', va='center', fontsize=8.5, fontweight='bold', color='#333333')

    # Separator line above bars
    ax.plot([left_margin - 0.15, left_margin + col_w * n_methods + 0.15],
            [bar_top_y + 0.05, bar_top_y + 0.05], color='black', lw=1.0, clip_on=False)

    # --- Bar chart ---
    bar_max_h = bar_h - 0.4
    score_vmax = max(overall.values()) if overall else 1

    for i, method in enumerate(sorted_methods):
        cx = left_margin + i * col_w + col_w / 2
        val = overall.get(method, 0)
        h = (val / score_vmax) * bar_max_h if score_vmax > 0 else 0

        for j in range(2):
            sx = cx - 0.22 + j * 0.44
            bw = 0.3
            alpha = 0.9 if j == 0 else 0.65
            bar = FancyBboxPatch(
                (sx - bw / 2, bar_top_y - h), bw, h,
                boxstyle='round,pad=0.02', facecolor=COLOR_BAR,
                edgecolor='none', alpha=alpha, zorder=5)
            ax.add_patch(bar)

        # Score value
        ax.text(cx, bar_top_y - h - 0.12, f'{val:.3f}',
                ha='center', va='top', fontsize=5.2, fontweight='bold', color=COLOR_BAR)

    out_path = os.path.join(out_dir, f'fig_overall_summary_{args.clustering}.pdf')
    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
