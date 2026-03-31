#!/usr/bin/env python3
"""
ED Fig 3: Per-spot CM-GTC spatial visualization.

For each withGT dataset x slice, plot a grid of spatial scatter plots
(one per method), colored by per-spot CM-GTC score.

Usage:
    python plot_edfig3_cmgtc_spatial.py --root /path/to/SMOBench-CLEAN
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
import matplotlib.colors as mcolors

METHOD_ORDER = [
    'CANDIES', 'COSMOS', 'MISO', 'MultiGATE', 'PRAGA', 'PRESENT',
    'SMOPCA', 'SpaBalance', 'SpaFusion', 'SpaMI', 'SpaMosaic',
    'SpaMultiVAE', 'SpaMV', 'SpatialGlue', 'SWITCH',
]

DATASET_SLICES = {
    'Human_Lymph_Nodes': ['A1', 'D1'],
    'Human_Tonsils': ['S1', 'S2', 'S3'],
    'Mouse_Embryos_S1': ['E11', 'E13', 'E15', 'E18'],
    'Mouse_Embryos_S2': ['E11', 'E13', 'E15', 'E18'],
    # woGT datasets will be auto-detected from per-spot CSVs
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--slice', type=str, default=None)
    parser.add_argument('--dpi', type=int, default=300)
    return parser.parse_args()


def load_perspot_data(root):
    perspot_dir = os.path.join(root, '_myx_Results', 'evaluation', 'cmgtc_perspot')
    dfs = []
    for f in sorted(os.listdir(perspot_dir)):
        if f.startswith('cmgtc_perspot_') and not f.startswith('cmgtc_perspot_summary') and f.endswith('.csv'):
            dfs.append(pd.read_csv(os.path.join(perspot_dir, f)))
    if not dfs:
        raise FileNotFoundError(f"No per-spot CSV files in {perspot_dir}")
    return pd.concat(dfs, ignore_index=True)


def load_gt_spatial(root, dataset, slice_name):
    """Load GT spatial labels + coordinates for datasets with ground truth."""
    import scanpy as sc
    WITHGT = {
        'Human_Lymph_Nodes': 'RNA_ADT', 'Human_Tonsils': 'RNA_ADT',
        'Mouse_Embryos_S1': 'RNA_ATAC', 'Mouse_Embryos_S2': 'RNA_ATAC',
    }
    if dataset not in WITHGT:
        return None, None, None
    mod_type = WITHGT[dataset]
    gt_path = os.path.join(root, 'Dataset', 'withGT', mod_type, dataset, slice_name, 'adata_RNA.h5ad')
    if not os.path.isfile(gt_path):
        return None, None, None
    adata = sc.read_h5ad(gt_path)
    gt_col = None
    for c in ['Spatial_Label', 'Ground Truth', 'cell_type']:
        if c in adata.obs.columns:
            gt_col = c; break
    if gt_col is None or 'spatial' not in adata.obsm:
        return None, None, None
    return adata.obs[gt_col].values, adata.obsm['spatial'], gt_col


def plot_one_slice(df_slice, dataset, slice_name, out_dir, root=None, dpi=300):
    methods_present = [m for m in METHOD_ORDER if m in df_slice['Method'].unique()]
    n_methods = len(methods_present)
    if n_methods == 0:
        return

    DS_DISPLAY = {
        'Human_Lymph_Nodes': 'Human Lymph Nodes',
        'Human_Tonsils': 'Human Tonsils',
        'Mouse_Embryos_S1': 'Mouse Embryos S1',
        'Mouse_Embryos_S2': 'Mouse Embryos S2',
        'Mouse_Spleen': 'Mouse Spleen',
        'Mouse_Thymus': 'Mouse Thymus',
        'Mouse_Brain': 'Mouse Brain',
    }

    # Check if GT available
    gt_labels, gt_coords, gt_col = None, None, None
    if root:
        gt_labels, gt_coords, gt_col = load_gt_spatial(root, dataset, slice_name)
    has_gt = gt_labels is not None

    # Total panels: GT (if available) + methods
    n_panels = (1 if has_gt else 0) + n_methods
    # 15 methods (+GT=16) → 2×8; 13 methods (+GT=14) → 2×7; woGT no GT panel
    if n_panels <= 14:
        ncols = 7
    else:
        ncols = 8
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 3.2), squeeze=False)

    vmin = df_slice['cmgtc_perspot'].quantile(0.02)
    vmax = df_slice['cmgtc_perspot'].quantile(0.98)
    vmin = max(0, vmin)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'soft_red', ['#FEF9F8', '#FADBD8', '#F1948A', '#E74C3C', '#B03A2E'], N=256)

    tab10 = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
             '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',
             '#aec7e8','#ffbb78','#98df8a','#ff9896','#c5b0d5']

    panel_idx = 0

    # GT panel (first position)
    if has_gt:
        row, col = divmod(panel_idx, ncols)
        ax = axes[row][col]
        domains = sorted(set(gt_labels))
        for i, d in enumerate(domains):
            mask = gt_labels == d
            ax.scatter(gt_coords[mask, 0], gt_coords[mask, 1],
                       s=max(5, min(20, 15000 / len(gt_labels))),
                       alpha=0.85, color=tab10[i % len(tab10)], edgecolors='none')
        ax.set_title('GT', fontsize=9, fontweight='bold', pad=3)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        panel_idx += 1

    # Method panels
    for method in methods_present:
        row, col = divmod(panel_idx, ncols)
        ax = axes[row][col]
        df_m = df_slice[df_slice['Method'] == method]
        n_cells = len(df_m)
        pt_size = max(5, min(20, 15000 / n_cells))
        ax.scatter(df_m['x'], df_m['y'], c=df_m['cmgtc_perspot'],
                   cmap=cmap, norm=norm, s=pt_size, alpha=0.85, edgecolors='none')
        mean_val = df_m['cmgtc_perspot'].mean()
        ax.set_title(f'{method}\n(mean = {mean_val:.3f})', fontsize=9, pad=3)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        panel_idx += 1

    # Hide unused
    for idx in range(panel_idx, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.subplots_adjust(right=0.91)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Per-spot CM-GTC', fontsize=10)

    ds_display = DS_DISPLAY.get(dataset, dataset.replace('_', ' '))
    fig.suptitle(f'{ds_display} ({slice_name})', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 0.91, 0.98])

    out_path = os.path.join(out_dir, f'edfig3_cmgtc_spatial_{dataset}_{slice_name}.pdf')
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    out_dir = os.path.join(root, '_myx_Results', 'plots')
    os.makedirs(out_dir, exist_ok=True)

    print("Loading per-spot data...")
    df = load_perspot_data(root)
    print(f"Loaded {len(df)} rows, {df['Method'].nunique()} methods")

    if args.dataset:
        datasets = {args.dataset: DATASET_SLICES.get(args.dataset, [])}
    else:
        # Auto-detect all datasets and slices from the data
        datasets = {}
        for (ds, sl), _ in df.groupby(['Dataset', 'Slice']):
            datasets.setdefault(ds, []).append(sl)
        for ds in datasets:
            datasets[ds] = sorted(set(datasets[ds]))

    for dataset, slices in datasets.items():
        slice_list = [args.slice] if args.slice else slices
        for sl in slice_list:
            print(f"\nPlotting {dataset}/{sl}...")
            df_slice = df[(df['Dataset'] == dataset) & (df['Slice'] == sl)]
            if df_slice.empty:
                print(f"  No data")
                continue
            plot_one_slice(df_slice, dataset, sl, out_dir, root=root, dpi=args.dpi)


if __name__ == '__main__':
    main()
