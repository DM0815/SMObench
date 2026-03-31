#!/usr/bin/env python3
"""
Fig 5c: Method Selection Guide — clean dot-matrix style.
Filled dot = supported, hollow dot = partial, × = not supported.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

sys.path.insert(0, os.path.dirname(__file__))
from style_config import apply_style, PAL13
apply_style()

# Sorted by: 1) number of supported capabilities (desc), 2) alphabetical
# Computed from CAPS below
METHOD_ORDER = None  # will be computed in main()

# 2 = supported, 1 = partial, 0 = not supported
# Verified against actual data:
#   RNA+ATAC: SpaFusion/SpaMultiVAE = 0 (no ATAC results)
#   3M: MISO/PRAGA/PRESENT/SMOPCA/SpaBalance/SpaMV/SpatialGlue = 2 (7 methods)
#   Large(>5k): MultiGATE fails at 5k+, SMOPCA/SWITCH fail at ~38k
#   Horizontal: SWITCH/MultiGATE = 0 (no horizontal results)
#   MultiBatch: same as horizontal support
CAPS = {
    #                    RNA_ADT  RNA_ATAC  Large  MultiBatch  3M  Vert  Horiz
    'CANDIES':          [2, 2, 2, 2, 0, 2, 2],
    'COSMOS':           [2, 2, 2, 2, 0, 2, 2],
    'MISO':             [2, 2, 2, 2, 2, 2, 2],
    'MultiGATE':        [2, 2, 0, 0, 0, 2, 0],
    'PRAGA':            [2, 2, 2, 2, 2, 2, 2],
    'PRESENT':          [2, 2, 2, 2, 2, 2, 2],
    'SMOPCA':           [2, 2, 1, 2, 2, 2, 2],
    'SpaBalance':       [2, 2, 2, 2, 2, 2, 2],
    'SpaFusion':        [2, 0, 2, 1, 0, 2, 1],
    'SpaMI':            [2, 2, 2, 2, 0, 2, 2],
    'SpaMosaic':        [2, 2, 2, 2, 0, 2, 2],
    'SpaMultiVAE':      [2, 0, 2, 1, 0, 2, 1],
    'SpaMV':            [2, 2, 2, 2, 2, 2, 2],
    'SpatialGlue':      [2, 2, 2, 2, 2, 2, 2],
    'SWITCH':           [2, 2, 1, 0, 0, 2, 0],
}

CAP_NAMES = [
    'RNA+ADT', 'RNA+ATAC', 'Large data\n(>5k spots)',
    'Multi-batch', 'Three-modality', 'Vertical', 'Horizontal',
]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    out_dir = os.path.join(os.path.abspath(args.root), '_myx_Results', 'plots')
    os.makedirs(out_dir, exist_ok=True)

    # Sort: most capabilities first, then alphabetical
    methods = sorted(CAPS.keys(), key=lambda m: (-sum(CAPS[m]), m))
    n_m = len(methods)
    n_c = len(CAP_NAMES)

    # Transposed layout: methods=rows, capabilities=columns (tall narrow)
    cw, rh = 0.7, 0.55  # cell width, row height
    top_margin = 1.2
    left_margin = 1.8
    fig_w = left_margin + n_c * cw + 0.5
    fig_h = top_margin + n_m * rh + 1.0

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-0.2, left_margin + n_c * cw + 0.3)
    ax.set_ylim(-0.8, top_margin + n_m * rh + 0.3)
    ax.axis('off')

    # Colors matching dot-matrix heatmap style
    from style_config import METRIC_COLORS
    c_yes = METRIC_COLORS['BioC']    # green #52a65e — supported
    c_partial = METRIC_COLORS['CMGTC']  # tan #bb9369 — partial
    c_no = METRIC_COLORS['SC']       # coral #e75b58 — not supported

    radius = 0.20

    # Column headers (capabilities, rotated)
    for j, cap in enumerate(CAP_NAMES):
        x = left_margin + (j + 0.5) * cw
        y = top_margin + n_m * rh + 0.05
        ax.text(x, y, cap, ha='right', va='bottom', fontsize=8, fontweight='bold',
                rotation=45, color='#000000')

    # Row labels (method names)
    for i, m in enumerate(methods):
        y = top_margin + (n_m - 1 - i + 0.5) * rh
        ax.text(left_margin - 0.15, y, m, ha='right', va='center', fontsize=9,
                fontweight='bold', color='#000000')

    # Top and bottom lines
    y_top = top_margin + n_m * rh - 0.05
    y_bot = top_margin - 0.05
    ax.plot([left_margin - 0.1, left_margin + n_c * cw + 0.1], [y_top, y_top],
            color='#000000', lw=0.8)
    ax.plot([left_margin - 0.1, left_margin + n_c * cw + 0.1], [y_bot, y_bot],
            color='#000000', lw=0.8)

    # Light horizontal grid between rows
    for i in range(1, n_m):
        y = top_margin + i * rh - 0.05
        ax.plot([left_margin - 0.1, left_margin + n_c * cw + 0.1], [y, y],
                color='#E8E8E8', lw=0.4)

    # Draw cells (transposed: row=method, col=capability)
    for i, m in enumerate(methods):
        for j, cap in enumerate(CAP_NAMES):
            x = left_margin + (j + 0.5) * cw
            y = top_margin + (n_m - 1 - i + 0.5) * rh
            val = CAPS[m][j]

            if val == 2:
                circle = plt.Circle((x, y), radius, facecolor=c_yes,
                                    edgecolor='none', zorder=3)
                ax.add_patch(circle)
                ax.text(x, y, '✓', ha='center', va='center', fontsize=9,
                        color='white', fontweight='bold', zorder=4)
            elif val == 1:
                circle = plt.Circle((x, y), radius, facecolor='white',
                                    edgecolor=c_partial, linewidth=1.8, zorder=3)
                ax.add_patch(circle)
                ax.text(x, y, '○', ha='center', va='center', fontsize=8,
                        color=c_partial, fontweight='bold', zorder=4)
            else:
                ax.text(x, y, '×', ha='center', va='center', fontsize=11,
                        color=c_no, fontweight='bold', zorder=3)

    # Legend at bottom
    h1 = mlines.Line2D([], [], marker='o', color='white', markerfacecolor=c_yes,
                        markersize=10, label='Supported', linestyle='None')
    h2 = mlines.Line2D([], [], marker='o', color=c_partial, markerfacecolor='white',
                        markersize=10, markeredgewidth=1.5, label='Partial',
                        linestyle='None')
    h3 = mlines.Line2D([], [], marker='x', color=c_no,
                        markersize=9, markeredgewidth=2, label='Not supported',
                        linestyle='None')
    ax.legend(handles=[h1, h2, h3], loc='upper center',
              bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=9,
              frameon=False, handletextpad=0.3, columnspacing=1.5)

    out = os.path.join(out_dir, 'fig5c_method_selection_guide')
    fig.savefig(out + '.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(out + '.png', dpi=args.dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out}.pdf")


if __name__ == '__main__':
    main()
