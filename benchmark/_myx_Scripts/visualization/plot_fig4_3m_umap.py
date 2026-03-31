#!/usr/bin/env python3
"""
SMOBench Figure 4 (left b,c): 3M Integration UMAP + Spatial
Generates cluster UMAP and spatial plots for 7 three-modality methods (3Mv2).

Usage:
    python plot_3m_umap.py --root /path/to/SMOBench-CLEAN
    python plot_3m_umap.py --root /path/to/SMOBench-CLEAN --clustering mclust
"""

import os
import argparse
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import sys as _sys
_sys.path.insert(0, __import__('os').path.dirname(__file__))
from style_config import apply_style, PAL13
apply_style()

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=300, facecolor='white')

METHODS_3M = [
    'SpatialGlue_3Mv2', 'SpaBalance_3Mv2', 'SMOPCA_3Mv2',
    'MISO_3Mv2', 'PRESENT_3Mv2', 'SpaMV_3Mv2', 'PRAGA_3Mv2',
]

METHOD_EMBEDDING_KEYS = {
    'SpatialGlue_3Mv2': ['SpatialGlue_3Mv2', 'SpatialGlue', 'X_integrated', 'X_emb'],
    'SpaBalance_3Mv2':  ['SpaBalance_3Mv2', 'SpaBalance', 'X_integrated', 'X_emb'],
    'SMOPCA_3Mv2':      ['SMOPCA_3Mv2', 'SMOPCA', 'X_integrated', 'X_emb'],
    'MISO_3Mv2':        ['MISO_3Mv2', 'MISO', 'X_integrated', 'X_emb'],
    'PRESENT_3Mv2':     ['PRESENT_3Mv2', 'PRESENT', 'X_integrated', 'X_emb'],
    'SpaMV_3Mv2':       ['SpaMV_3Mv2', 'SpaMV', 'X_integrated', 'X_emb'],
    'PRAGA_3Mv2':       ['PRAGA_3Mv2', 'PRAGA', 'X_integrated', 'X_emb'],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--clustering', type=str, default='leiden')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--point_size', type=float, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    out_dir = os.path.join(root, '_myx_Results', 'plots')
    os.makedirs(out_dir, exist_ok=True)

    # Load GT for reference
    gt_path = os.path.join(root, 'Dataset', 'withGT', '3M_Simulation_v2', 'adata_RNA.h5ad')
    gt_col = None
    adata_gt = None
    if os.path.isfile(gt_path):
        adata_gt = sc.read_h5ad(gt_path)
        for col in ['Spatial_Label', 'Ground Truth', 'cell_type']:
            if col in adata_gt.obs.columns:
                gt_col = col
                break

    # Layout: 1 GT row + 7 methods, 2 cols (UMAP, Spatial)
    n_rows = 1 + len(METHODS_3M)
    fig, axs = plt.subplots(n_rows, 2, figsize=(10, 3.2 * n_rows))

    # --- Row 0: Ground Truth ---
    if adata_gt is not None and gt_col:
        if 'spatial' in adata_gt.obsm:
            sp = adata_gt.obsm['spatial'].copy()
            sp[:, 1] = -sp[:, 1]
            adata_gt.obsm['spatial_flip'] = sp

        try:
            sc.pp.neighbors(adata_gt, n_neighbors=15)
            sc.tl.umap(adata_gt)
            sc.pl.umap(adata_gt, color=gt_col, ax=axs[0, 0],
                       title='Ground Truth (UMAP)', s=args.point_size,
                       show=False, frameon=False)
        except Exception:
            axs[0, 0].set_title('GT UMAP — N/A')

        if 'spatial_flip' in adata_gt.obsm:
            sc.pl.embedding(adata_gt, basis='spatial_flip', color=gt_col,
                            ax=axs[0, 1], title='Ground Truth (Spatial)',
                            s=args.point_size, show=False, frameon=False)
    else:
        axs[0, 0].text(0.5, 0.5, 'No GT', ha='center', va='center')
        axs[0, 0].set_title('Ground Truth')
        axs[0, 1].set_visible(False)

    # Display name mapping (strip _3Mv2 suffix)
    method_display = {m: m.replace('_3Mv2', '') for m in METHODS_3M}

    # --- Rows 1-7: Methods ---
    for idx, method in enumerate(METHODS_3M):
        row = idx + 1
        result_dir = os.path.join(root, '_myx_Results', 'adata',
                                   'vertical_integration', method)
        h5ad_files = sorted(Path(result_dir).rglob('*.h5ad'))
        if not h5ad_files:
            axs[row, 0].set_title(f'{method} — No data')
            axs[row, 1].set_visible(False)
            continue

        try:
            adata = sc.read_h5ad(str(h5ad_files[0]))

            # Find embedding
            emb_key = None
            for key in METHOD_EMBEDDING_KEYS.get(method, []):
                if key in adata.obsm:
                    emb_key = key
                    break

            # Compute UMAP
            if emb_key and 'X_umap' not in adata.obsm:
                sc.pp.neighbors(adata, use_rep=emb_key, n_neighbors=15)
                sc.tl.umap(adata)

            # Plot UMAP
            color_key = args.clustering if args.clustering in adata.obs.columns else None
            if color_key is None:
                # Find any clustering
                for c in ['leiden', 'mclust', 'louvain', 'kmeans']:
                    if c in adata.obs.columns:
                        color_key = c
                        break

            display_name = method_display.get(method, method)
            if 'X_umap' in adata.obsm and color_key:
                sc.pl.umap(adata, color=color_key, ax=axs[row, 0],
                           title=f'{display_name} ({color_key})',
                           s=args.point_size, show=False, frameon=False)
            else:
                axs[row, 0].set_title(f'{display_name} — No UMAP')

            # Plot spatial
            if 'spatial' in adata.obsm and color_key:
                sp = adata.obsm['spatial'].copy()
                sp[:, 1] = -sp[:, 1]
                adata.obsm['spatial_flip'] = sp
                sc.pl.embedding(adata, basis='spatial_flip', color=color_key,
                                ax=axs[row, 1], title=f'{display_name} (Spatial)',
                                s=args.point_size, show=False, frameon=False)
            else:
                axs[row, 1].set_title(f'{display_name} — No spatial')

        except Exception as e:
            axs[row, 0].set_title(f'{method} — Error')
            axs[row, 1].set_title(f'{method} — Error')
            print(f"[{method}] Error: {e}")

    fig.suptitle('3M Integration — UMAP + Spatial Domain', fontsize=14, y=1.01)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f'fig4_3m_umap_spatial_{args.clustering}.png')
    plt.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
