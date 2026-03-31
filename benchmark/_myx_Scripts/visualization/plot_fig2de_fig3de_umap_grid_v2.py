#!/usr/bin/env python3
"""
SMOBench Figure 2(d,e) / 3(e,f): Multi-method UMAP + Spatial Comparison — v2 GRID layout.

v2 changes vs v1:
  - Methods arranged in a GRID (default 3 per row) instead of single column
  - Produces a figure with aspect ratio ~1.0 instead of extremely tall ~0.16
  - Each method cell: UMAP (left) + Spatial (right)

Usage:
    python plot_fig2de_fig3de_umap_grid_v2.py --root /path/to/SMOBench-CLEAN \
        --task vertical --dataset Human_Tonsils --clustering leiden
    python plot_fig2de_fig3de_umap_grid_v2.py --root /path/to/SMOBench-CLEAN \
        --task horizontal --dataset Human_Lymph_Nodes --clustering leiden
"""

import os
import sys
import argparse
import math
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path
import anndata as ad
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings('ignore')

# Import global style + palette
sys.path.insert(0, os.path.dirname(__file__))
from style_config import apply_style, PAL13
apply_style()

sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=300, facecolor='white')

# Use scanpy default palette (tab20)


VERTICAL_METHODS = [
    'CANDIES', 'COSMOS', 'MISO', 'MultiGATE', 'PRAGA', 'PRESENT',
    'SMOPCA', 'SpaBalance', 'SpaFusion', 'SpaMI', 'SpaMosaic',
    'SpaMultiVAE', 'SpaMV', 'SpatialGlue', 'SWITCH',
]

HORIZONTAL_METHODS = [
    'CANDIES', 'COSMOS', 'MISO', 'PRAGA', 'PRESENT',
    'SMOPCA', 'SpaBalance', 'SpaFusion', 'SpaMI', 'SpaMosaic',
    'SpaMultiVAE', 'SpaMV', 'SpatialGlue',
]

DATASET_GT_INFO = {
    'Human_Lymph_Nodes': {'type': 'RNA_ADT',  'gt_dir': 'Human_Lymph_Nodes'},
    'Human_Tonsils':     {'type': 'RNA_ADT',  'gt_dir': 'Human_Tonsils'},
    'Mouse_Embryos_S1':  {'type': 'RNA_ATAC', 'gt_dir': 'Mouse_Embryos_S1'},
    'Mouse_Embryos_S2':  {'type': 'RNA_ATAC', 'gt_dir': 'Mouse_Embryos_S2'},
}

# woGT datasets: no ground-truth labels, spatial coords recoverable from Dataset/woGT/
DATASET_WOGT_INFO = {
    'Mouse_Thymus': {'type': 'RNA_ADT',  'gt_base': 'woGT'},
    'Mouse_Spleen': {'type': 'RNA_ADT',  'gt_base': 'woGT'},
    'Mouse_Brain':  {'type': 'RNA_ATAC', 'gt_base': 'woGT'},
}

# Mapping: adata slice name -> Dataset/woGT sub-directory name
WOGT_SLICE_DIR_MAP = {
    'Mouse_Thymus': {f'Thymus{i}': f'Mouse_Thymus{i}' for i in range(1, 5)},
    'Mouse_Spleen': {f'Spleen{i}': f'Mouse_Spleen{i}' for i in range(1, 3)},
    'Mouse_Brain':  {
        'ATAC': 'Mouse_Brain_ATAC',
        'H3K27ac': 'Mouse_Brain_H3K27ac',
        'H3K27me3': 'Mouse_Brain_H3K27me3',
        'H3K4me3': 'Mouse_Brain_H3K4me3',
    },
}

DATASET_DIR_ALIASES = {
    'Human_Lymph_Nodes': ['Human_Lymph_Nodes', 'HLN'],
    'Human_Tonsils':     ['Human_Tonsils', 'HT'],
    'Mouse_Embryos_S1':  ['Mouse_Embryos_S1', 'MISAR_S1'],
    'Mouse_Embryos_S2':  ['Mouse_Embryos_S2', 'MISAR_S2'],
    'Mouse_Thymus':      ['Mouse_Thymus'],
    'Mouse_Spleen':      ['Mouse_Spleen'],
    'Mouse_Brain':       ['Mouse_Brain'],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--task', choices=['vertical', 'horizontal'], default='vertical')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--slice', type=str, default=None,
                        help='Specific slice (for vertical). If None, uses first available.')
    parser.add_argument('--clustering', type=str, default='leiden')
    parser.add_argument('--plot_type', choices=['umap', 'spatial', 'both'], default='both')
    parser.add_argument('--methods', nargs='+', default=None)
    parser.add_argument('--methods_per_row', type=int, default=4,
                        help='Number of methods per row in the grid (default: 4)')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--point_size', type=float, default=30)
    parser.add_argument('--cell_w', type=float, default=3.0,
                        help='Width per subplot cell in inches')
    parser.add_argument('--cell_h', type=float, default=2.8,
                        help='Height per subplot cell in inches')
    parser.add_argument('--show_ari', action='store_true', default=True,
                        help='Show ARI/NMI annotation per method')
    parser.add_argument('--show_legend', action='store_true', default=True,
                        help='Show cell-type color legend at top')
    return parser.parse_args()


def load_scores(root, task, dataset, clustering):
    """Load SMOBench scores from evaluation CSV. Returns {method: score}."""
    summary = os.path.join(root, '_myx_Results', 'evaluation', 'summary')
    score_col = 'SMOBench_V' if task == 'vertical' else 'SMOBench_H'
    for name in [f'{task}_final_{clustering}.csv',
                 f'{task}_detailed_{clustering}.csv']:
        p = os.path.join(summary, name)
        if os.path.isfile(p):
            df = pd.read_csv(p)
            df.columns = [c.replace(' ', '_').replace("'", '') for c in df.columns]
            if score_col not in df.columns:
                continue
            # Filter to this dataset, average across slices
            df_ds = df[df['Dataset'] == dataset]
            if df_ds.empty:
                continue
            scores = df_ds.groupby('Method')[score_col].mean().to_dict()
            return scores, score_col
    return {}, score_col


def find_slice(root, task, method, dataset):
    """Find first available slice h5ad for a method-dataset pair."""
    if task == 'vertical':
        base = os.path.join(root, '_myx_Results', 'adata', 'vertical_integration')
    else:
        base = os.path.join(root, '_myx_Results', 'adata', 'horizontal_integration')

    dataset_dir = None
    for alias in DATASET_DIR_ALIASES.get(dataset, [dataset]):
        candidate = os.path.join(base, method, alias)
        if os.path.isdir(candidate):
            dataset_dir = candidate
            break
    if dataset_dir is None:
        return None, None

    h5ad_files = sorted(Path(dataset_dir).rglob('*.h5ad'))
    if not h5ad_files:
        return None, None

    path = h5ad_files[0]
    if path.parent.name != dataset:
        slice_name = path.parent.name
    else:
        slice_name = path.stem.split('_')[-1]

    return str(path), slice_name


def load_and_prepare(h5ad_path, method, clustering, point_size=8,
                     root=None, task=None, dataset=None,
                     gt_coord_set=None):
    """Load adata, compute UMAP if needed. Fix corrupted spatial coords."""
    adata = sc.read_h5ad(h5ad_path)

    emb_key = None
    for key in [method, 'X_integrated', 'X_emb']:
        if key in adata.obsm:
            emb_key = key
            break

    if emb_key and 'X_umap' not in adata.obsm:
        try:
            sc.pp.neighbors(adata, use_rep=emb_key, n_neighbors=15)
            sc.tl.umap(adata)
        except Exception:
            pass

    # Check spatial validity: overlap-based when GT coords available
    spatial_valid = False
    y_pre_negated = False
    if 'spatial' in adata.obsm:
        sp = adata.obsm['spatial']
        x_range = sp[:, 0].max() - sp[:, 0].min()
        y_range = sp[:, 1].max() - sp[:, 1].min()
        m_range = max(x_range, y_range)

        if gt_coord_set and m_range > 1:
            n_check = min(500, sp.shape[0])
            idx = np.linspace(0, sp.shape[0] - 1, n_check, dtype=int)
            # Check original sign
            n_match = sum(
                1 for i in idx
                if (round(float(sp[i, 0]), 1), round(float(sp[i, 1]), 1))
                in gt_coord_set)
            if n_match / n_check > 0.5:
                spatial_valid = True
            else:
                # Check negated y (MultiGATE stores y pre-negated)
                n_match_neg = sum(
                    1 for i in idx
                    if (round(float(sp[i, 0]), 1),
                        round(-float(sp[i, 1]), 1))
                    in gt_coord_set)
                if n_match_neg / n_check > 0.5:
                    spatial_valid = True
                    y_pre_negated = True

        if not spatial_valid:
            # Fallback: range-based (for cases without GT reference)
            spatial_valid = (x_range > 100 or y_range > 100)

    if not spatial_valid and root and dataset and dataset in DATASET_GT_INFO:
        # Recover spatial from GT slices
        info = DATASET_GT_INFO[dataset]
        gt_base = os.path.join(root, 'Dataset', 'withGT', info['type'], info['gt_dir'])
        if os.path.isdir(gt_base):
            gt_slices = sorted(os.listdir(gt_base))
            # Try 1: concat all slices (for methods that process all slices together)
            gt_spatials = []
            gt_total = 0
            for s in gt_slices:
                p = os.path.join(gt_base, s, 'adata_RNA.h5ad')
                if os.path.isfile(p):
                    gt = sc.read_h5ad(p)
                    if 'spatial' in gt.obsm:
                        gt_spatials.append((gt.obsm['spatial'], gt.n_obs))
                        gt_total += gt.n_obs
            if gt_total == adata.n_obs and gt_spatials:
                adata.obsm['spatial'] = np.vstack([s[0] for s in gt_spatials])
                spatial_valid = True
            elif gt_spatials and gt_total > 0 and abs(gt_total - adata.n_obs) / gt_total < 0.01:
                # Try 2: approximate match (within 1% tolerance, e.g. SpaMV drops a few cells)
                all_sp = np.vstack([s[0] for s in gt_spatials])
                adata.obsm['spatial'] = all_sp[:adata.n_obs]
                spatial_valid = True
            else:
                # Try 3: match single slice by cell count (for methods like SpaMultiVAE)
                for sp_arr, sp_n in gt_spatials:
                    if sp_n == adata.n_obs:
                        adata.obsm['spatial'] = sp_arr
                        spatial_valid = True
                        break

    if not spatial_valid and root and dataset and dataset in DATASET_WOGT_INFO:
        # Recover spatial from woGT Dataset slices
        info = DATASET_WOGT_INFO[dataset]
        wogt_base = os.path.join(root, 'Dataset', info['gt_base'], info['type'], dataset)
        slice_dir_map = WOGT_SLICE_DIR_MAP.get(dataset, {})
        if os.path.isdir(wogt_base):
            # Collect spatial from all woGT slices (in mapped directory order)
            wogt_spatials = []
            wogt_total = 0
            for slice_dir_name in sorted(os.listdir(wogt_base)):
                p = os.path.join(wogt_base, slice_dir_name, 'adata_RNA.h5ad')
                if os.path.isfile(p):
                    gt = sc.read_h5ad(p)
                    if 'spatial' in gt.obsm:
                        wogt_spatials.append((gt.obsm['spatial'], gt.n_obs))
                        wogt_total += gt.n_obs
            if wogt_total == adata.n_obs and wogt_spatials:
                adata.obsm['spatial'] = np.vstack([s[0] for s in wogt_spatials])
                spatial_valid = True
            elif wogt_spatials and wogt_total > 0 and abs(wogt_total - adata.n_obs) / wogt_total < 0.01:
                all_sp = np.vstack([s[0] for s in wogt_spatials])
                adata.obsm['spatial'] = all_sp[:adata.n_obs]
                spatial_valid = True
            else:
                # Try single slice match
                for sp_arr, sp_n in wogt_spatials:
                    if sp_n == adata.n_obs:
                        adata.obsm['spatial'] = sp_arr
                        spatial_valid = True
                        break

    if 'spatial' in adata.obsm and spatial_valid:
        sp = adata.obsm['spatial'].copy()
        if y_pre_negated:
            # y already negated (e.g., MultiGATE) — use as-is
            adata.obsm['spatial_flip'] = sp
        else:
            sp[:, 1] = -sp[:, 1]
            adata.obsm['spatial_flip'] = sp
    elif 'spatial' in adata.obsm:
        del adata.obsm['spatial']

    return adata


def force_palette(adata, color_key):
    """Force tab10 colors for consistent GT/method coloring.
    Skips if colors already set (e.g., by hungarian_recolor)."""
    if color_key and color_key in adata.obs.columns:
        color_uns_key = f'{color_key}_colors'
        if color_uns_key in adata.uns:
            return  # Don't overwrite Hungarian-matched colors
        n_cats = adata.obs[color_key].nunique()
        tab10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        adata.uns[color_uns_key] = np.array(tab10[:n_cats])


def hungarian_recolor(adata_method, adata_gt, cluster_key, gt_key):
    """Reorder method cluster colors to match GT cell type colors via Hungarian matching."""
    if cluster_key not in adata_method.obs.columns or gt_key not in adata_gt.obs.columns:
        return

    # Ensure GT palette is set
    force_palette(adata_gt, gt_key)
    gt_color_key = f'{gt_key}_colors'
    gt_cats = adata_gt.obs[gt_key].cat.categories.tolist()
    gt_colors = list(adata_gt.uns.get(gt_color_key, []))
    if not gt_colors:
        return
    n_g = len(gt_cats)

    method_cats = adata_method.obs[cluster_key].cat.categories.tolist()
    n_m = len(method_cats)

    # --- Match cells between method and GT ---
    # Try 1: obs_names intersection
    common = adata_method.obs_names.intersection(adata_gt.obs_names)
    if len(common) >= max(10, adata_method.n_obs * 0.3):
        method_labels = adata_method.obs.loc[common, cluster_key]
        gt_labels = adata_gt.obs.loc[common, gt_key]
    elif adata_method.n_obs == adata_gt.n_obs:
        # Try 2: positional matching (same number of cells, assume same order)
        method_labels = adata_method.obs[cluster_key].values
        gt_labels = adata_gt.obs[gt_key].values
    else:
        # Cannot match
        return

    # Build confusion matrix
    confusion = np.zeros((n_m, n_g))
    for i, mc in enumerate(method_cats):
        mask_m = (method_labels == mc) if hasattr(method_labels, '__eq__') else (np.array(method_labels) == mc)
        for j, gc in enumerate(gt_cats):
            mask_g = (gt_labels == gc) if hasattr(gt_labels, '__eq__') else (np.array(gt_labels) == gc)
            confusion[i, j] = (mask_m & mask_g).sum()

    # Hungarian algorithm: maximize overlap (minimize negative)
    row_ind, col_ind = linear_sum_assignment(-confusion)

    # Assign GT colors to matched method clusters
    tab10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    method_colors = [None] * n_m
    for mi, gi in zip(row_ind, col_ind):
        if gi < len(gt_colors):
            method_colors[mi] = gt_colors[gi]

    # Fill unmatched clusters with remaining tab10 colors
    used = set(c for c in method_colors if c is not None)
    remaining = [c for c in tab10 if c not in used]
    rem_idx = 0
    for i in range(n_m):
        if method_colors[i] is None:
            method_colors[i] = remaining[rem_idx] if rem_idx < len(remaining) else '#888888'
            rem_idx += 1

    adata_method.uns[f'{cluster_key}_colors'] = np.array(method_colors)


def plot_method_cell(adata, method_name, clustering, plot_type, point_size,
                     ax_umap=None, ax_spatial=None, is_gt=False, gt_col=None,
                     ari=None, nmi=None):
    """Plot UMAP and/or spatial for a single method into provided axes."""
    if is_gt:
        color_key = gt_col
        label = 'GT'
    else:
        color_key = clustering if clustering in adata.obs.columns else None
        label = method_name

    # Force PAL13 colors
    force_palette(adata, color_key)

    if ax_umap is not None:
        if 'X_umap' in adata.obsm and color_key:
            sc.pl.umap(adata, color=color_key, ax=ax_umap,
                       title=label, s=point_size,
                       show=False, frameon=False, legend_loc='none')
        else:
            ax_umap.set_title(f'{label}', fontsize=8)
            ax_umap.text(0.5, 0.5, 'N/A', ha='center', va='center',
                         fontsize=8, transform=ax_umap.transAxes)
        ax_umap.axis('off')
        # Clip scatter points to axes boundary & remove residual legends
        for c in ax_umap.collections:
            c.set_clip_on(True)
        if ax_umap.get_legend():
            ax_umap.get_legend().remove()
        # Annotate SMOBench score
        if ari is not None and not is_gt:
            ax_umap.text(0.98, 0.02, f'{ari:.3f}', transform=ax_umap.transAxes,
                         ha='right', va='bottom', fontsize=6.5, fontweight='bold',
                         color='#D35400',
                         bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                   alpha=0.85, edgecolor='#D35400', linewidth=0.6))

    if ax_spatial is not None:
        basis = 'spatial_flip' if 'spatial_flip' in adata.obsm else 'spatial'
        if basis in adata.obsm and color_key:
            sc.pl.embedding(adata, basis=basis, color=color_key,
                            ax=ax_spatial, title='',
                            s=point_size, show=False, frameon=False,
                            legend_loc='none')
        else:
            ax_spatial.set_title('', fontsize=8)
            ax_spatial.text(0.5, 0.5, 'N/A', ha='center', va='center',
                            fontsize=8, transform=ax_spatial.transAxes)
        ax_spatial.axis('off')
        # Clip scatter points to axes boundary & remove residual legends
        for c in ax_spatial.collections:
            c.set_clip_on(True)
        if ax_spatial.get_legend():
            ax_spatial.get_legend().remove()


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    out_dir = os.path.join(root, '_myx_Results', 'plots')
    os.makedirs(out_dir, exist_ok=True)

    methods = args.methods or (VERTICAL_METHODS if args.task == 'vertical'
                               else HORIZONTAL_METHODS)

    # Subplot columns per method (UMAP, spatial, or both)
    sub_cols = 0
    if args.plot_type in ('umap', 'both'):
        sub_cols += 1
    if args.plot_type in ('spatial', 'both'):
        sub_cols += 1

    # Collect available methods
    method_data = []
    target_slice = args.slice
    for method in methods:
        h5ad_path, slice_name = find_slice(root, args.task, method, args.dataset)
        if h5ad_path is None:
            continue
        if target_slice and slice_name != target_slice:
            base = os.path.join(root, '_myx_Results', 'adata',
                                f'{args.task}_integration', method, args.dataset)
            for p in sorted(Path(base).rglob('*.h5ad')):
                if target_slice in str(p):
                    h5ad_path = str(p)
                    slice_name = target_slice
                    break
        if target_slice is None:
            target_slice = slice_name
        method_data.append((method, h5ad_path, slice_name))

    if not method_data:
        print(f"No results found for {args.dataset}")
        return

    # --- Determine if dataset has GT ---
    has_gt = args.dataset in DATASET_GT_INFO

    # --- GRID LAYOUT ---
    # Total entries = GT (if available) + N methods
    if has_gt:
        entries = [('__GT__', None, None)] + method_data
    else:
        entries = list(method_data)  # no GT slot for woGT datasets
    n_entries = len(entries)
    mpr = args.methods_per_row  # methods per row
    n_grid_rows = math.ceil(n_entries / mpr)
    n_grid_cols = min(n_entries, mpr)

    # Total subplot grid: each method cell = 1 row × sub_cols columns
    total_cols = n_grid_cols * sub_cols
    total_rows = n_grid_rows

    fig_w = args.cell_w * total_cols
    fig_h = args.cell_h * total_rows
    fig, axs = plt.subplots(total_rows, total_cols,
                            figsize=(fig_w, fig_h),
                            squeeze=False)

    # --- Load and plot GT ---
    gt_plotted = False
    adata_gt = None
    gt_col = None
    if has_gt:
        info = DATASET_GT_INFO[args.dataset]
        gt_base = os.path.join(root, 'Dataset', 'withGT', info['type'], info['gt_dir'])

        if args.task == 'horizontal' and os.path.isdir(gt_base):
            # HORIZONTAL: load ALL GT slices and concatenate
            gt_adatas = []
            gt_dirs = sorted(os.listdir(gt_base))
            for gd in gt_dirs:
                candidate = os.path.join(gt_base, gd, 'adata_RNA.h5ad')
                if os.path.isfile(candidate):
                    gt_adatas.append(sc.read_h5ad(candidate))

            if gt_adatas:
                # Find GT column from first slice
                for col in ['Spatial_Label', 'Ground Truth', 'cell_type']:
                    if col in gt_adatas[0].obs.columns:
                        gt_col = col
                        break

                if gt_col:
                    if len(gt_adatas) == 1:
                        adata_gt = gt_adatas[0]
                    else:
                        # Manual concat: avoids ad.concat issues with obsm shapes
                        import scipy.sparse as sp_sparse
                        # Make var_names unique first (some datasets have duplicates)
                        for a in gt_adatas:
                            a.var_names_make_unique()
                        # Find common genes
                        common_vars = gt_adatas[0].var_names
                        for a in gt_adatas[1:]:
                            common_vars = common_vars.intersection(a.var_names)
                        common_vars = sorted(common_vars)
                        # Stack expression, obs, spatial
                        X_blocks, obs_frames, spatial_blocks = [], [], []
                        for a in gt_adatas:
                            a_sub = a[:, common_vars]
                            X_blocks.append(a_sub.X)
                            obs_frames.append(a.obs[[gt_col]])
                            if 'spatial' in a.obsm:
                                spatial_blocks.append(a.obsm['spatial'])
                        if sp_sparse.issparse(X_blocks[0]):
                            combined_X = sp_sparse.vstack(X_blocks)
                        else:
                            combined_X = np.vstack(X_blocks)
                        combined_obs = pd.concat(obs_frames, ignore_index=True)
                        adata_gt = ad.AnnData(X=combined_X, obs=combined_obs)
                        adata_gt.var_names = common_vars
                        adata_gt.obs_names_make_unique()
                        if spatial_blocks:
                            adata_gt.obsm['spatial'] = np.vstack(spatial_blocks)

                    # Flip spatial
                    if 'spatial' in adata_gt.obsm:
                        sp = adata_gt.obsm['spatial'].copy()
                        sp[:, 1] = -sp[:, 1]
                        adata_gt.obsm['spatial_flip'] = sp

                    # Compute UMAP on concatenated GT
                    try:
                        import scipy.sparse as _sps
                        if _sps.issparse(adata_gt.X):
                            adata_gt.X = adata_gt.X.astype('float32')
                        elif not np.issubdtype(adata_gt.X.dtype, np.floating):
                            adata_gt.X = adata_gt.X.astype('float32')
                        sc.tl.pca(adata_gt)
                        sc.pp.neighbors(adata_gt, use_rep='X_pca', n_neighbors=15)
                        sc.tl.umap(adata_gt)
                    except Exception as e:
                        print(f"  Warning: GT UMAP failed: {e}")

                    print(f"  GT loaded: {adata_gt.n_obs} cells from {len(gt_adatas)} slices")
        else:
            # VERTICAL: load single slice (original logic)
            gt_path = None
            if target_slice:
                candidate = os.path.join(gt_base, target_slice, 'adata_RNA.h5ad')
                if os.path.isfile(candidate):
                    gt_path = candidate
            if gt_path is None and os.path.isdir(gt_base):
                gt_dirs = sorted(os.listdir(gt_base))
                for gd in gt_dirs:
                    candidate = os.path.join(gt_base, gd, 'adata_RNA.h5ad')
                    if os.path.isfile(candidate):
                        gt_path = candidate
                        break

            if gt_path and os.path.isfile(gt_path):
                adata_gt = sc.read_h5ad(gt_path)
                for col in ['Spatial_Label', 'Ground Truth', 'cell_type']:
                    if col in adata_gt.obs.columns:
                        gt_col = col
                        break
                if gt_col:
                    if 'spatial' in adata_gt.obsm:
                        sp = adata_gt.obsm['spatial'].copy()
                        sp[:, 1] = -sp[:, 1]
                        adata_gt.obsm['spatial_flip'] = sp
                    try:
                        import scipy.sparse as _sps
                        if _sps.issparse(adata_gt.X):
                            adata_gt.X = adata_gt.X.astype('float32')
                        elif not np.issubdtype(adata_gt.X.dtype, np.floating):
                            adata_gt.X = adata_gt.X.astype('float32')
                        sc.tl.pca(adata_gt)
                        sc.pp.neighbors(adata_gt, use_rep='X_pca', n_neighbors=15)
                        sc.tl.umap(adata_gt)
                    except Exception as e:
                        print(f"  Warning: GT UMAP failed for {args.dataset}/{args.slice}: {e}")

        # Plot GT if loaded
        if adata_gt is not None and gt_col:
            ax_u = axs[0, 0] if args.plot_type in ('umap', 'both') else None
            ax_s = axs[0, sub_cols-1] if args.plot_type in ('spatial', 'both') else None
            plot_method_cell(adata_gt, 'GT', args.clustering, args.plot_type,
                             args.point_size, ax_u, ax_s,
                             is_gt=True, gt_col=gt_col)
            gt_plotted = True

    if has_gt and not gt_plotted:
        for c in range(sub_cols):
            axs[0, c].text(0.5, 0.5, 'No GT', ha='center', va='center', fontsize=9)
            axs[0, c].axis('off')

    # --- Build GT spatial coord set for overlap-based validity check ---
    _gt_coord_set = set()
    _gt_base_dir = None
    if args.dataset in DATASET_GT_INFO:
        _info = DATASET_GT_INFO[args.dataset]
        _gt_base_dir = os.path.join(root, 'Dataset', 'withGT',
                                    _info['type'], _info['gt_dir'])
    elif args.dataset in DATASET_WOGT_INFO:
        _info = DATASET_WOGT_INFO[args.dataset]
        _gt_base_dir = os.path.join(root, 'Dataset', _info['gt_base'],
                                    _info['type'], args.dataset)
    if _gt_base_dir and os.path.isdir(_gt_base_dir):
        for _sd in sorted(os.listdir(_gt_base_dir)):
            _p = os.path.join(_gt_base_dir, _sd, 'adata_RNA.h5ad')
            if os.path.isfile(_p):
                _gt = sc.read_h5ad(_p)
                if 'spatial' in _gt.obsm:
                    for _c in np.round(_gt.obsm['spatial'], 1):
                        _gt_coord_set.add((float(_c[0]), float(_c[1])))

    # --- Load scores for annotation ---
    scores, score_label = {}, ''
    if args.show_ari:
        scores, score_label = load_scores(root, args.task, args.dataset, args.clustering)
        if scores:
            print(f"  Loaded {len(scores)} {score_label} scores for {args.dataset}")

    # --- Plot methods in grid ---
    # For woGT datasets, methods start at position 0 (no GT slot)
    method_offset = 1 if has_gt else 0
    for idx, (method, h5ad_path, slice_name) in enumerate(method_data):
        pos = idx + method_offset
        grid_r = pos // mpr
        grid_c = pos % mpr
        col_start = grid_c * sub_cols

        try:
            adata = load_and_prepare(h5ad_path, method, args.clustering, args.point_size,
                                        root=root, task=args.task, dataset=args.dataset,
                                        gt_coord_set=_gt_coord_set)

            # Hungarian matching: align cluster colors with GT
            if adata_gt is not None and gt_col:
                hungarian_recolor(adata, adata_gt, args.clustering, gt_col)

            # Look up SMOBench score
            method_score = scores.get(method)

            ax_u = axs[grid_r, col_start] if args.plot_type in ('umap', 'both') else None
            ax_s = axs[grid_r, col_start + sub_cols - 1] if args.plot_type in ('spatial', 'both') else None
            plot_method_cell(adata, method, args.clustering, args.plot_type,
                             args.point_size, ax_u, ax_s,
                             ari=method_score, nmi=None)
        except Exception as e:
            for c in range(sub_cols):
                axs[grid_r, col_start + c].set_title(f'{method} — Error', fontsize=7)
            print(f"  [{method}] Error: {e}")

    # --- Add cell-type legend at top ---
    if args.show_legend and adata_gt is not None and gt_col:
        try:
            categories = adata_gt.obs[gt_col].cat.categories.tolist()
            color_key = f'{gt_col}_colors'
            if color_key in adata_gt.uns:
                colors = adata_gt.uns[color_key]
            else:
                colors = sc.pl.palettes.default_20[:len(categories)]
            handles = [mpatches.Patch(facecolor=c, edgecolor='#666666',
                                      linewidth=0.5, label=cat)
                       for cat, c in zip(categories, colors)]
            fig.legend(handles=handles, loc='upper center',
                       bbox_to_anchor=(0.5, 1.0),
                       ncol=min(len(categories), 8),
                       fontsize=7, frameon=True, fancybox=True,
                       edgecolor='#CCCCCC', title='Cell Type',
                       title_fontsize=8)
        except Exception as e:
            print(f"  Legend error: {e}")

    # --- Hide unused cells ---
    for pos in range(n_entries, n_grid_rows * mpr):
        grid_r = pos // mpr
        grid_c = pos % mpr
        col_start = grid_c * sub_cols
        for c in range(sub_cols):
            if col_start + c < total_cols:
                axs[grid_r, col_start + c].axis('off')

    plt.subplots_adjust(hspace=0.25, wspace=0.05)

    tag = 'fig2de' if args.task == 'vertical' else 'fig3ef'
    base_name = f'{tag}_umap_spatial_{args.dataset}_{args.clustering}'
    out_pdf = os.path.join(out_dir, base_name + '.pdf')
    plt.savefig(out_pdf, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out_pdf}")


if __name__ == '__main__':
    main()
