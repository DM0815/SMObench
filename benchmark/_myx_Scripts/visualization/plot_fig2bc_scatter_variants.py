#!/usr/bin/env python3
"""
SMOBench Figure 2(b,c) — 5 scatter plot variants for SC × BioC × CM-GTC.

Generates 5 styles for both RNA+ADT and RNA+ATAC:
  1. Ternary plot (3 axes: SC, BioC, CM-GTC)
  2. Color-coded scatter (x=SC, y=BioC, color=CM-GTC viridis)
  3. Landscape scatter (x=SC, y=BioC, background=CM-GTC interpolation)
  4. Color + contour (x=SC, y=BioC, color=CM-GTC, contour=SMOBench_V)
  5. Side-by-side panels (left: SC vs BioC, right: CM-GTC bar chart)

Usage:
    python plot_fig2bc_scatter_variants.py --root /path/to/SMOBench-CLEAN
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy.interpolate import griddata

# ── Config ──────────────────────────────────────────────────────────────────
# Main figure: withGT datasets only (BioC sub-metrics differ for woGT)
RNA_ADT_DATASETS = ['Human_Lymph_Nodes', 'Human_Tonsils']
RNA_ATAC_DATASETS = ['Mouse_Embryos_S1', 'Mouse_Embryos_S2']

# Supplementary: woGT datasets
RNA_ADT_WOGT = ['Mouse_Thymus', 'Mouse_Spleen']
RNA_ATAC_WOGT = ['Mouse_Brain']

METHOD_MARKERS = {
    'CANDIES': 'o', 'COSMOS': 's', 'MISO': '^', 'MultiGATE': 'D',
    'PRAGA': 'v', 'PRESENT': 'P', 'SMOPCA': '*', 'SpaBalance': 'X',
    'SpaFusion': 'h', 'SpaMI': 'o', 'SpaMosaic': 's', 'SpaMultiVAE': '^',
    'SpaMV': 'D', 'SpatialGlue': 'v', 'SWITCH': 'P',
}


def load_data(root, clustering='leiden'):
    summary = os.path.join(root, '_myx_Results', 'evaluation', 'summary')
    for name in [f'vertical_final_{clustering}.csv',
                 f'vertical_detailed_{clustering}.csv']:
        p = os.path.join(summary, name)
        if os.path.isfile(p):
            return pd.read_csv(p)
    raise FileNotFoundError(f"No vertical CSV in {summary}")


def prepare(df, modality, datasets=None):
    if datasets is None:
        datasets = RNA_ADT_DATASETS if modality == 'RNA_ADT' else RNA_ATAC_DATASETS
    df = df[df['Dataset'].isin(datasets)].copy()
    df.columns = [c.replace(' ', '_').replace("'", '') for c in df.columns]
    if 'Moran_Index' not in df.columns and 'Moran_index' in df.columns:
        df.rename(columns={'Moran_index': 'Moran_Index'}, inplace=True)
    cols = ['SC_Score', 'BioC_Score', 'CM_GTC', 'SMOBench_V']
    agg = df.groupby('Method')[[c for c in cols if c in df.columns]].mean()
    return agg


# ── Variant 1: Ternary-like triangle plot ───────────────────────────────────
def plot_ternary(agg, title, out_path, dpi=300):
    """Pseudo-ternary: 3 axes on equilateral triangle coordinates."""
    fig, ax = plt.subplots(figsize=(7, 6))

    # Ternary coordinates: SC→bottom-left, BioC→bottom-right, CM-GTC→top
    def to_xy(sc, bioc, cmgtc):
        total = sc + bioc + cmgtc
        if total == 0:
            return 0.5, 0.33
        sc_n, bioc_n, cmgtc_n = sc/total, bioc/total, cmgtc/total
        x = 0.5 * (2 * bioc_n + cmgtc_n)
        y = (np.sqrt(3) / 2) * cmgtc_n
        return x, y

    # Draw triangle
    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, np.sqrt(3)/2, 0]
    ax.plot(tri_x, tri_y, 'k-', linewidth=1)

    # Axis labels
    ax.text(0, -0.05, 'SC', ha='center', fontsize=11, fontweight='bold')
    ax.text(1, -0.05, 'BioC', ha='center', fontsize=11, fontweight='bold')
    ax.text(0.5, np.sqrt(3)/2 + 0.04, 'CM-GTC', ha='center', fontsize=11, fontweight='bold')

    # Plot methods
    colors = plt.cm.tab20(np.linspace(0, 1, len(agg)))
    for i, (method, row) in enumerate(agg.iterrows()):
        sc_v = row.get('SC_Score', 0)
        bioc_v = row.get('BioC_Score', 0)
        cmgtc_v = row.get('CM_GTC', 0)
        x, y = to_xy(sc_v, bioc_v, cmgtc_v)
        ax.scatter(x, y, s=120, c=[colors[i]], edgecolor='white',
                   linewidth=0.8, zorder=5, marker=METHOD_MARKERS.get(method, 'o'))
        ax.annotate(method, (x, y), fontsize=6, ha='center', va='bottom',
                    xytext=(0, 6), textcoords='offset points')

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.12, np.sqrt(3)/2 + 0.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {out_path}')


# ── Variant 2: Color-coded scatter ──────────────────────────────────────────
def plot_color_scatter(agg, title, out_path, dpi=300):
    """x=SC, y=BioC, color=CM-GTC (viridis), size=fixed."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    sc_vals = agg['SC_Score'].values
    bioc_vals = agg['BioC_Score'].values
    cmgtc_vals = agg['CM_GTC'].values

    scatter = ax.scatter(sc_vals, bioc_vals, c=cmgtc_vals, cmap='RdYlGn',
                         s=140, edgecolor='#333333', linewidth=0.6, zorder=5)

    for method, row in agg.iterrows():
        ax.annotate(method, (row['SC_Score'], row['BioC_Score']),
                    fontsize=6.5, ha='center', va='bottom',
                    xytext=(0, 7), textcoords='offset points')

    # Median lines
    ax.axhline(bioc_vals.mean(), color='gray', ls='--', lw=0.7, alpha=0.5)
    ax.axvline(sc_vals.mean(), color='gray', ls='--', lw=0.7, alpha=0.5)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('CM-GTC', fontsize=10)

    ax.set_xlabel('Spatial Coherence (SC)', fontsize=11)
    ax.set_ylabel('Biological Conservation (BioC)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)

    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {out_path}')


# ── Variant 3: Landscape (background interpolation) ────────────────────────
def plot_landscape(agg, title, out_path, dpi=300):
    """x=SC, y=BioC, background heatmap=CM-GTC interpolation."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    sc_vals = agg['SC_Score'].values
    bioc_vals = agg['BioC_Score'].values
    cmgtc_vals = agg['CM_GTC'].values

    # Create background grid
    margin = 0.03
    xi = np.linspace(sc_vals.min() - margin, sc_vals.max() + margin, 100)
    yi = np.linspace(bioc_vals.min() - margin, bioc_vals.max() + margin, 100)
    xi, yi = np.meshgrid(xi, yi)

    try:
        zi = griddata((sc_vals, bioc_vals), cmgtc_vals, (xi, yi), method='cubic')
        zi = np.nan_to_num(zi, nan=np.nanmean(cmgtc_vals))
        ax.contourf(xi, yi, zi, levels=20, cmap='RdYlGn', alpha=0.35)
        cs = ax.contour(xi, yi, zi, levels=8, colors='gray', linewidths=0.4, alpha=0.5)
        ax.clabel(cs, inline=True, fontsize=6, fmt='%.2f')
    except Exception:
        pass

    ax.scatter(sc_vals, bioc_vals, c=cmgtc_vals, cmap='RdYlGn',
               s=140, edgecolor='black', linewidth=0.8, zorder=5)

    for method, row in agg.iterrows():
        ax.annotate(method, (row['SC_Score'], row['BioC_Score']),
                    fontsize=6.5, ha='center', va='bottom',
                    xytext=(0, 7), textcoords='offset points')

    ax.set_xlabel('Spatial Coherence (SC)', fontsize=11)
    ax.set_ylabel('Biological Conservation (BioC)', fontsize=11)
    ax.set_title(f'{title} — CM-GTC landscape', fontsize=11, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)

    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {out_path}')


# ── Variant 4: Color + iso-score contours ──────────────────────────────────
def plot_color_contour(agg, title, out_path, dpi=300):
    """x=SC, y=BioC, color=CM-GTC, dashed contours=equal SMOBench_V."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    sc_vals = agg['SC_Score'].values
    bioc_vals = agg['BioC_Score'].values
    cmgtc_vals = agg['CM_GTC'].values
    total_vals = agg['SMOBench_V'].values if 'SMOBench_V' in agg.columns else None

    # Draw iso-score lines (SMOBench_V = w1*SC + w2*BioC + w3*CMGTC)
    # Approximate: total ≈ 0.2*SC + 0.4*BioC + 0.4*CMGTC_mean
    if total_vals is not None:
        cmgtc_mean = cmgtc_vals.mean()
        margin = 0.03
        xi = np.linspace(sc_vals.min() - margin, sc_vals.max() + margin, 100)
        yi = np.linspace(bioc_vals.min() - margin, bioc_vals.max() + margin, 100)
        XI, YI = np.meshgrid(xi, yi)
        # Iso-score: total = 0.2*SC + 0.4*BioC + 0.4*CMGTC_mean
        ZI = 0.2 * XI + 0.4 * YI + 0.4 * cmgtc_mean
        levels = np.linspace(total_vals.min() - 0.02, total_vals.max() + 0.02, 8)
        cs = ax.contour(XI, YI, ZI, levels=levels, colors='#AAAAAA',
                        linestyles='--', linewidths=0.6, alpha=0.7)
        ax.clabel(cs, inline=True, fontsize=6, fmt='%.2f')

    scatter = ax.scatter(sc_vals, bioc_vals, c=cmgtc_vals, cmap='RdYlGn',
                         s=160, edgecolor='#333333', linewidth=0.8, zorder=5)

    for method, row in agg.iterrows():
        ax.annotate(method, (row['SC_Score'], row['BioC_Score']),
                    fontsize=6.5, ha='center', va='bottom',
                    xytext=(0, 8), textcoords='offset points')

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('CM-GTC', fontsize=10)

    ax.set_xlabel('Spatial Coherence (SC)', fontsize=11)
    ax.set_ylabel('Biological Conservation (BioC)', fontsize=11)
    ax.set_title(f'{title} — iso-SMOBench contours', fontsize=11, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)

    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {out_path}')


# ── Variant 5: Side-by-side (scatter + CM-GTC bar) ─────────────────────────
def plot_sidebyside(agg, title, out_path, dpi=300):
    """Left: SC vs BioC scatter; Right: CM-GTC horizontal bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={'width_ratios': [1.2, 1]})

    sc_vals = agg['SC_Score'].values
    bioc_vals = agg['BioC_Score'].values
    cmgtc_vals = agg['CM_GTC'].values

    # Left: SC vs BioC
    colors = plt.cm.tab20(np.linspace(0, 1, len(agg)))
    for i, (method, row) in enumerate(agg.iterrows()):
        ax1.scatter(row['SC_Score'], row['BioC_Score'], s=100, c=[colors[i]],
                    edgecolor='#333333', linewidth=0.5, zorder=5,
                    marker=METHOD_MARKERS.get(method, 'o'))
        ax1.annotate(method, (row['SC_Score'], row['BioC_Score']),
                     fontsize=6, ha='center', va='bottom',
                     xytext=(0, 6), textcoords='offset points')

    ax1.axhline(bioc_vals.mean(), color='gray', ls='--', lw=0.7, alpha=0.5)
    ax1.axvline(sc_vals.mean(), color='gray', ls='--', lw=0.7, alpha=0.5)
    ax1.set_xlabel('SC', fontsize=11)
    ax1.set_ylabel('BioC', fontsize=11)
    ax1.set_title('SC vs BioC', fontsize=11, fontweight='bold')
    ax1.spines[['top', 'right']].set_visible(False)

    # Right: CM-GTC bar chart
    sorted_agg = agg.sort_values('CM_GTC', ascending=True)
    bar_colors = plt.cm.RdYlGn(
        (sorted_agg['CM_GTC'] - sorted_agg['CM_GTC'].min()) /
        max(sorted_agg['CM_GTC'].max() - sorted_agg['CM_GTC'].min(), 0.01))
    ax2.barh(range(len(sorted_agg)), sorted_agg['CM_GTC'],
             color=bar_colors, edgecolor='#333333', linewidth=0.4)
    ax2.set_yticks(range(len(sorted_agg)))
    ax2.set_yticklabels(sorted_agg.index, fontsize=8)
    ax2.set_xlabel('CM-GTC', fontsize=11)
    ax2.set_title('CM-GTC Ranking', fontsize=11, fontweight='bold')
    ax2.spines[['top', 'right']].set_visible(False)
    # Add value labels
    for i, v in enumerate(sorted_agg['CM_GTC']):
        ax2.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=7)

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {out_path}')


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--clustering', type=str, default='leiden')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    out_dir = os.path.join(root, '_myx_Results', 'plots', 'scatter_variants')
    os.makedirs(out_dir, exist_ok=True)

    df = load_data(root, args.clustering)

    variants = [
        ('v1_ternary', plot_ternary),
        ('v2_color', plot_color_scatter),
        ('v3_landscape', plot_landscape),
        ('v4_contour', plot_color_contour),
        ('v5_sidebyside', plot_sidebyside),
    ]

    # withGT (main figure)
    for mod in ['RNA_ADT', 'RNA_ATAC']:
        print(f'\n=== {mod} (withGT) ===')
        ds = RNA_ADT_DATASETS if mod == 'RNA_ADT' else RNA_ATAC_DATASETS
        agg = prepare(df, mod, datasets=ds)
        title = f'Vertical Integration ({mod.replace("_", "+")})'

        for vname, vfunc in variants:
            out = os.path.join(out_dir, f'fig2bc_{mod}_{vname}.png')
            try:
                vfunc(agg, title, out, dpi=args.dpi)
            except Exception as e:
                print(f'  Error ({vname}): {e}')

    # woGT (supplementary)
    for mod in ['RNA_ADT', 'RNA_ATAC']:
        print(f'\n=== {mod} (woGT) ===')
        ds = RNA_ADT_WOGT if mod == 'RNA_ADT' else RNA_ATAC_WOGT
        agg = prepare(df, mod, datasets=ds)
        if agg.empty:
            print(f'  No woGT data for {mod}, skipping')
            continue
        title = f'Vertical Integration ({mod.replace("_", "+")}) — woGT'

        for vname, vfunc in variants:
            out = os.path.join(out_dir, f'fig2bc_{mod}_{vname}_woGT.png')
            try:
                vfunc(agg, title, out, dpi=args.dpi)
            except Exception as e:
                print(f'  Error ({vname}): {e}')

    print('\nDone. All variants saved to:', out_dir)


if __name__ == '__main__':
    main()
