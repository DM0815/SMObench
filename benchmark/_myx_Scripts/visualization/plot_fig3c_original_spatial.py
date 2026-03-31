#!/usr/bin/env python3
"""
SMOBench Figure 3(c): Original Spatial Data (BEFORE Integration)
Shows each slice's spatial coordinates colored by ground truth cell type (withGT)
or by spatial structure (woGT), to illustrate what the raw data looks like
before any horizontal integration method is applied.

Why this panel:
  Benchmark readers need to see the "input" — separate slices with their own
  spatial domains — to appreciate what integration methods are trying to solve.

Usage:
    python plot_fig3c_original_spatial.py --root /path/to/SMOBench-CLEAN --dataset Human_Tonsils
"""

import os
import argparse
import scanpy as sc
import matplotlib.pyplot as plt
import sys as _sys
_sys.path.insert(0, __import__('os').path.dirname(__file__))
from style_config import apply_style, PAL13
apply_style()

import warnings
warnings.filterwarnings('ignore')

sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=300, facecolor='white')

# Dataset → (gt_type, modality_dir)
DATASET_INFO = {
    'Human_Lymph_Nodes': ('withGT', 'RNA_ADT'),
    'Human_Tonsils':     ('withGT', 'RNA_ADT'),
    'Mouse_Embryos_S1':  ('withGT', 'RNA_ATAC'),
    'Mouse_Embryos_S2':  ('withGT', 'RNA_ATAC'),
    'Mouse_Thymus':      ('woGT',   'RNA_ADT'),
    'Mouse_Spleen':      ('woGT',   'RNA_ADT'),
    'Mouse_Brain':       ('woGT',   'RNA_ATAC'),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--point_size', type=float, default=15)
    return parser.parse_args()


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    out_dir = os.path.join(root, '_myx_Results', 'plots')
    os.makedirs(out_dir, exist_ok=True)

    if args.dataset not in DATASET_INFO:
        print(f"Unknown dataset: {args.dataset}")
        print(f"Available: {list(DATASET_INFO.keys())}")
        return

    gt_type, modality = DATASET_INFO[args.dataset]
    base_dir = os.path.join(root, 'Dataset', gt_type, modality, args.dataset)

    if not os.path.isdir(base_dir):
        print(f"Dataset directory not found: {base_dir}")
        return

    slices = sorted([s for s in os.listdir(base_dir)
                     if os.path.isdir(os.path.join(base_dir, s))])

    if not slices:
        print(f"No slices found in {base_dir}")
        return

    print(f"Dataset: {args.dataset} ({gt_type})")
    print(f"Slices: {slices}")

    n = len(slices)
    fig, axs = plt.subplots(1, n, figsize=(4 * n + 1, 4))
    if n == 1:
        axs = [axs]

    for idx, sl in enumerate(slices):
        h5ad_path = os.path.join(base_dir, sl, 'adata_RNA.h5ad')
        if not os.path.isfile(h5ad_path):
            axs[idx].set_title(f'{sl}\n(Missing)')
            axs[idx].axis('off')
            print(f"  [{sl}] adata_RNA.h5ad not found")
            continue

        adata = sc.read_h5ad(h5ad_path)
        print(f"  [{sl}] Loaded {adata.n_obs} cells")

        # Flip spatial Y for consistent display
        if 'spatial' not in adata.obsm:
            axs[idx].set_title(f'{sl}\n(No spatial)')
            axs[idx].axis('off')
            continue

        sp = adata.obsm['spatial'].copy()
        sp[:, 1] = -sp[:, 1]
        adata.obsm['spatial_flip'] = sp

        # Find color column
        color_col = None
        if gt_type == 'withGT':
            for col in ['Spatial_Label', 'Ground Truth', 'cell_type', 'celltype']:
                if col in adata.obs.columns:
                    color_col = col
                    break

        if color_col:
            sc.pl.embedding(adata, basis='spatial_flip', color=color_col,
                            ax=axs[idx], title=sl, s=args.point_size,
                            show=False, frameon=False)
        else:
            # woGT: plot spatial positions colored uniformly
            axs[idx].scatter(sp[:, 0], sp[:, 1], s=args.point_size,
                             c='steelblue', alpha=0.6, edgecolors='none')
            axs[idx].set_title(sl, fontsize=11)
            axs[idx].set_aspect('equal')
            axs[idx].axis('off')

    fig.suptitle(f'{args.dataset} — Original Spatial Data (Before Integration)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f'fig3c_original_spatial_{args.dataset}.png')
    plt.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
