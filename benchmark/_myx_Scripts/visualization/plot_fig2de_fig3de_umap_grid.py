#!/usr/bin/env python3
"""
SMOBench Figure 2(d,e) / 3(e,f): Multi-method UMAP + Spatial Comparison
Generates a multi-panel figure: GT + all methods, each showing UMAP and/or spatial domain.

Usage:
    python plot_umap_comparison.py --root /path/to/SMOBench-CLEAN \
        --task vertical --dataset HT --clustering leiden
    python plot_umap_comparison.py --root /path/to/SMOBench-CLEAN \
        --task horizontal --dataset HT --clustering leiden --color_by batch
"""

import os
import argparse
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=300, facecolor='white')


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

# Result directories may use short or full names
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
    parser.add_argument('--color_by', choices=['cluster', 'batch', 'both'], default='cluster')
    parser.add_argument('--plot_type', choices=['umap', 'spatial', 'both'], default='both')
    parser.add_argument('--methods', nargs='+', default=None)
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--point_size', type=float, default=10)
    return parser.parse_args()


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


def load_and_prepare(h5ad_path, method, clustering, point_size=10):
    """Load adata, compute UMAP if needed."""
    adata = sc.read_h5ad(h5ad_path)

    # Find embedding
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

    # Flip spatial Y for consistent display
    if 'spatial' in adata.obsm:
        sp = adata.obsm['spatial'].copy()
        sp[:, 1] = -sp[:, 1]
        adata.obsm['spatial_flip'] = sp

    return adata


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    out_dir = os.path.join(root, '_myx_Results', 'plots')
    os.makedirs(out_dir, exist_ok=True)

    methods = args.methods or (VERTICAL_METHODS if args.task == 'vertical' else HORIZONTAL_METHODS)

    # Determine columns per method
    n_plot_cols = 0
    if args.plot_type in ('umap', 'both'):
        n_plot_cols += 1
    if args.plot_type in ('spatial', 'both'):
        n_plot_cols += 1

    # Collect available methods
    method_data = []
    target_slice = args.slice
    for method in methods:
        h5ad_path, slice_name = find_slice(root, args.task, method, args.dataset)
        if h5ad_path is None:
            continue
        if target_slice and slice_name != target_slice:
            # Find specific slice
            h5ad_path2, _ = None, None
            base = os.path.join(root, '_myx_Results', 'adata',
                                f'{args.task}_integration', method, args.dataset)
            for p in sorted(Path(base).rglob('*.h5ad')):
                if target_slice in str(p):
                    h5ad_path2 = str(p)
                    break
            if h5ad_path2:
                h5ad_path = h5ad_path2
                slice_name = target_slice

        if target_slice is None:
            target_slice = slice_name  # Use first found slice for all

        method_data.append((method, h5ad_path, slice_name))

    if not method_data:
        print(f"No results found for {args.dataset}")
        return

    n_methods = len(method_data) + 1  # +1 for GT
    ncols = n_plot_cols
    nrows = n_methods

    fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    if ncols == 1:
        axs = axs.reshape(-1, 1)
    if nrows == 1:
        axs = axs.reshape(1, -1)

    # --- Row 0: Ground Truth ---
    gt_row_plotted = False
    if args.dataset in DATASET_GT_INFO:
        info = DATASET_GT_INFO[args.dataset]
        gt_base = os.path.join(root, 'Dataset', 'withGT', info['type'],
                                info['gt_dir'])
        if target_slice:
            gt_path = os.path.join(gt_base, target_slice, 'adata_RNA.h5ad')
        else:
            gt_dirs = sorted(os.listdir(gt_base))
            gt_path = os.path.join(gt_base, gt_dirs[0], 'adata_RNA.h5ad') if gt_dirs else None

        if gt_path and os.path.isfile(gt_path):
            adata_gt = sc.read_h5ad(gt_path)
            gt_col = None
            for col in ['Spatial_Label', 'Ground Truth', 'cell_type']:
                if col in adata_gt.obs.columns:
                    gt_col = col
                    break

            if gt_col and 'spatial' in adata_gt.obsm:
                sp = adata_gt.obsm['spatial'].copy()
                sp[:, 1] = -sp[:, 1]
                adata_gt.obsm['spatial_flip'] = sp

                col_idx = 0
                if args.plot_type in ('umap', 'both'):
                    try:
                        sc.pp.neighbors(adata_gt, n_neighbors=15)
                        sc.tl.umap(adata_gt)
                        sc.pl.umap(adata_gt, color=gt_col, ax=axs[0, col_idx],
                                   title='GT (UMAP)', s=args.point_size, show=False, frameon=False)
                    except Exception:
                        axs[0, col_idx].set_title('GT (UMAP) — N/A')
                    col_idx += 1
                if args.plot_type in ('spatial', 'both'):
                    sc.pl.embedding(adata_gt, basis='spatial_flip', color=gt_col,
                                    ax=axs[0, col_idx], title='GT (Spatial)',
                                    s=args.point_size, show=False, frameon=False)
                gt_row_plotted = True

    if not gt_row_plotted:
        for c in range(ncols):
            axs[0, c].text(0.5, 0.5, 'No GT', ha='center', va='center', fontsize=12)
            axs[0, c].set_title('Ground Truth')

    # --- Rows 1+: Methods ---
    for row_idx, (method, h5ad_path, slice_name) in enumerate(method_data, start=1):
        try:
            adata = load_and_prepare(h5ad_path, method, args.clustering, args.point_size)
            color_key = args.clustering if args.clustering in adata.obs.columns else None

            col_idx = 0
            if args.plot_type in ('umap', 'both'):
                if 'X_umap' in adata.obsm and color_key:
                    sc.pl.umap(adata, color=color_key, ax=axs[row_idx, col_idx],
                               title=f'{method} (UMAP)', s=args.point_size,
                               show=False, frameon=False)
                else:
                    axs[row_idx, col_idx].set_title(f'{method} (UMAP) — N/A')
                col_idx += 1

            if args.plot_type in ('spatial', 'both'):
                if 'spatial_flip' in adata.obsm and color_key:
                    sc.pl.embedding(adata, basis='spatial_flip', color=color_key,
                                    ax=axs[row_idx, col_idx], title=f'{method} (Spatial)',
                                    s=args.point_size, show=False, frameon=False)
                else:
                    axs[row_idx, col_idx].set_title(f'{method} (Spatial) — N/A')

        except Exception as e:
            for c in range(ncols):
                axs[row_idx, c].set_title(f'{method} — Error')
            print(f"  [{method}] Error: {e}")

    fig.suptitle(f'{args.dataset} — {args.task.title()} Integration ({target_slice or ""})',
                 fontsize=14, y=1.01)
    plt.tight_layout()

    tag = f'fig2de' if args.task == 'vertical' else f'fig3ef'
    out_path = os.path.join(out_dir, f'{tag}_umap_spatial_{args.dataset}_{args.clustering}.png')
    plt.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
