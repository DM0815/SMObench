#!/usr/bin/env python3
"""
SMOBench Supp S9: Horizontal Integration — Batch UMAP + Per-slice Spatial.

Layout:
  UMAP section (2 rows × 8 cols): batch-colored UMAP, Original first
  Spatial section (4 rows × 8 cols): per-slice spatial domain (leiden-colored),
    Original repeated at each block start as reference

Usage:
    python plot_fig3de_batch_umap.py --root /path/to/SMOBench-CLEAN --dataset Human_Lymph_Nodes
"""

import os
import argparse
import math
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp_sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys as _sys
_sys.path.insert(0, __import__('os').path.dirname(__file__))
from style_config import apply_style, PAL13
apply_style()

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=300, facecolor='white')


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

# woGT datasets: no ground-truth labels, spatial coords from Dataset/woGT/
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

NCOLS = 7  # methods per row

# Spatial_Label number → region name mapping (from SpatialGlue paper Fig 2e, ED Fig 3d)
SPATIAL_LABEL_MAP = {
    'Human_Lymph_Nodes': {
        'A1': {
            '0': 'Cortex', '1': 'Medulla sinuses', '2': 'Follicle',
            '3': 'Medulla cords', '4': 'Pericapsular adipose',
            '5': 'Hilum', '6': 'Capsule', '7': 'Medulla vessels',
            '8': 'Subcapsular sinus', '9': 'Trabeculae',
        },
        'D1': {
            '1': 'Adipose Tissue', '2': 'B cell Follicle', '3': 'Capsule',
            '4': 'Connective Tissue', '5': 'Cortex', '6': 'Endothelial',
            '7': 'Exclude', '8': 'Marginal Sinus', '9': 'Medullary Chords',
            '10': 'Medullary Sinus', '11': 'Paracortex',
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--clustering', type=str, default='leiden')
    parser.add_argument('--methods', nargs='+', default=None)
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--point_size', type=float, default=8)
    return parser.parse_args()


def find_batch_col(adata):
    for col in ['batch', 'Batch', 'slice', 'section', 'sample', 'src']:
        if col in adata.obs.columns:
            return col
    return None


def load_original(root, dataset, clustering, ref_batch_labels=None):
    """Load unintegrated data (GT or woGT), compute UMAP + leiden."""
    # Determine data source: withGT or woGT
    if dataset in DATASET_GT_INFO:
        info = DATASET_GT_INFO[dataset]
        data_base = os.path.join(root, 'Dataset', 'withGT', info['type'], info['gt_dir'])
        is_wogt = False
    elif dataset in DATASET_WOGT_INFO:
        info = DATASET_WOGT_INFO[dataset]
        data_base = os.path.join(root, 'Dataset', info['gt_base'], info['type'], dataset)
        is_wogt = True
    else:
        return None, []

    if not os.path.isdir(data_base):
        return None, []

    slice_dir_map = WOGT_SLICE_DIR_MAP.get(dataset, {}) if is_wogt else {}
    # For woGT, build reverse map: dir_name -> adata_slice_name
    reverse_dir_map = {v: k for k, v in slice_dir_map.items()} if slice_dir_map else {}

    raw_adatas = []
    slice_names = []
    for gd in sorted(os.listdir(data_base)):
        p = os.path.join(data_base, gd, 'adata_RNA.h5ad')
        if os.path.isfile(p):
            a = sc.read_h5ad(p)
            a.var_names_make_unique()
            # Use adata-compatible slice name for batch label
            if is_wogt and gd in reverse_dir_map:
                batch_name = reverse_dir_map[gd]
            else:
                batch_name = gd
            a.obs['batch'] = batch_name
            # Replace numeric Spatial_Label with region names (withGT only)
            if not is_wogt and 'Spatial_Label' in a.obs.columns and dataset in SPATIAL_LABEL_MAP:
                label_map = SPATIAL_LABEL_MAP[dataset].get(gd, {})
                if label_map:
                    a.obs['Spatial_Label'] = a.obs['Spatial_Label'].astype(str).map(
                        lambda x: label_map.get(x, x))
            raw_adatas.append(a)
            slice_names.append(batch_name)

    if not raw_adatas:
        return None, []

    if len(raw_adatas) == 1:
        adata = raw_adatas[0]
    else:
        for a in raw_adatas:
            a.var_names_make_unique()
        common_vars = raw_adatas[0].var_names
        for a in raw_adatas[1:]:
            common_vars = common_vars.intersection(a.var_names)
        common_vars = sorted(common_vars)

        X_blocks, obs_frames, spatial_blocks = [], [], []
        for a in raw_adatas:
            X_blocks.append(a[:, common_vars].X)
            batch_obs = a.obs[['batch']].copy()
            # Carry GT label if available (withGT only)
            if not is_wogt:
                for col in ['Spatial_Label', 'Ground Truth', 'cell_type']:
                    if col in a.obs.columns:
                        batch_obs[col] = a.obs[col].values
                        break
            obs_frames.append(batch_obs)
            if 'spatial' in a.obsm:
                spatial_blocks.append(a.obsm['spatial'])
        if sp_sparse.issparse(X_blocks[0]):
            combined_X = sp_sparse.vstack(X_blocks)
        else:
            combined_X = np.vstack(X_blocks)
        combined_obs = pd.concat(obs_frames, ignore_index=True)
        adata = ad.AnnData(X=combined_X, obs=combined_obs)
        adata.var_names = common_vars
        adata.obs_names_make_unique()
        if spatial_blocks:
            adata.obsm['spatial'] = np.vstack(spatial_blocks)

    # Rename batch labels to match method naming
    if ref_batch_labels is not None and len(ref_batch_labels) == len(slice_names):
        ref_sorted = sorted(ref_batch_labels)
        name_map = {sn: rl for sn, rl in zip(sorted(slice_names), ref_sorted)}
        adata.obs['batch'] = adata.obs['batch'].map(name_map)

    # Flip spatial
    if 'spatial' in adata.obsm:
        sp = adata.obsm['spatial'].copy()
        sp[:, 1] = -sp[:, 1]
        adata.obsm['spatial_flip'] = sp

    # log1p + PCA + UMAP + leiden (no normalize to preserve batch effects)
    try:
        sc.pp.log1p(adata)
        sc.tl.pca(adata, n_comps=30)
        sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=15)
        sc.tl.umap(adata)
        sc.tl.leiden(adata, key_added=clustering)
    except Exception as e:
        print(f"  [Original] Warning: {e}")

    print(f"  Original loaded: {adata.n_obs} cells from {len(raw_adatas)} slices")
    return adata, slice_names


def recover_spatial(adata, root, dataset):
    """Recover spatial coords for method adata from GT or woGT slices."""
    # --- Load GT reference spatial per slice ---
    if dataset in DATASET_GT_INFO:
        info = DATASET_GT_INFO[dataset]
        data_base = os.path.join(root, 'Dataset', 'withGT', info['type'], info['gt_dir'])
    elif dataset in DATASET_WOGT_INFO:
        info = DATASET_WOGT_INFO[dataset]
        data_base = os.path.join(root, 'Dataset', info['gt_base'], info['type'], dataset)
    else:
        if 'spatial' in adata.obsm:
            sp = adata.obsm['spatial'].copy()
            sp[:, 1] = -sp[:, 1]
            adata.obsm['spatial_flip'] = sp
        return

    if not os.path.isdir(data_base):
        if 'spatial' in adata.obsm:
            sp = adata.obsm['spatial'].copy()
            sp[:, 1] = -sp[:, 1]
            adata.obsm['spatial_flip'] = sp
        return

    ref_spatials = []
    for s in sorted(os.listdir(data_base)):
        p = os.path.join(data_base, s, 'adata_RNA.h5ad')
        if os.path.isfile(p):
            ref = sc.read_h5ad(p)
            if 'spatial' in ref.obsm:
                ref_spatials.append((ref.obsm['spatial'], ref.n_obs))

    if not ref_spatials:
        if 'spatial' in adata.obsm:
            sp = adata.obsm['spatial'].copy()
            sp[:, 1] = -sp[:, 1]
            adata.obsm['spatial_flip'] = sp
        return

    batch_col = find_batch_col(adata) or 'batch'
    has_batch = batch_col in adata.obs.columns

    # --- Check if method spatial coords match GT (overlap-based) ---
    gt_coord_set = set()
    for sp_ref, _ in ref_spatials:
        for c in np.round(sp_ref, 1):
            gt_coord_set.add((float(c[0]), float(c[1])))

    has_valid_spatial = False
    if 'spatial' in adata.obsm:
        sp = adata.obsm['spatial']
        m_range = max(sp[:, 0].max() - sp[:, 0].min(),
                      sp[:, 1].max() - sp[:, 1].min())
        if m_range > 1:
            n_check = min(500, sp.shape[0])
            idx = np.linspace(0, sp.shape[0] - 1, n_check, dtype=int)
            n_match = sum(
                1 for i in idx
                if (round(float(sp[i, 0]), 1), round(float(sp[i, 1]), 1))
                in gt_coord_set)
            has_valid_spatial = (n_match / n_check) > 0.5

    if has_valid_spatial:
        # Spatial coords are real (match GT) — keep them, flip y
        sp2 = sp.copy()
        sp2[:, 1] = -sp2[:, 1]
        adata.obsm['spatial_flip'] = sp2

        if has_batch:
            batches = sorted(adata.obs[batch_col].unique())
            if len(batches) == len(ref_spatials):
                per_batch_counts = [
                    (adata.obs[batch_col] == b).sum() for b in batches]
                gt_counts = [n for _, n in ref_spatials]
                per_batch_ok = all(
                    abs(mc - gc) / max(gc, 1) < 0.05
                    for mc, gc in zip(per_batch_counts, gt_counts))
                if per_batch_ok:
                    print(f"    -> spatial valid, per-batch OK")
                    return
                # Wrong batch counts + slices spatially separated → centroid
                if m_range > 100:
                    gt_centroids = np.array(
                        [s.mean(axis=0) for s, _ in ref_spatials])
                    dists = np.zeros((sp.shape[0], len(gt_centroids)))
                    for i, c in enumerate(gt_centroids):
                        dists[:, i] = np.sqrt(
                            ((sp - c) ** 2).sum(axis=1))
                    assignments = dists.argmin(axis=1)
                    adata.obs['_spatial_batch'] = pd.Categorical(
                        [str(a) for a in assignments],
                        categories=[str(i) for i in range(
                            len(ref_spatials))])
                    print(f"    -> spatial valid, _spatial_batch by centroid "
                          f"({dict(zip(*np.unique(assignments, return_counts=True)))})")
                    return
                # Overlapping slices — can't fix batch, keep method labels
                print(f"    -> spatial valid, batch counts differ "
                      f"(keeping method labels)")
                return

        print(f"    -> spatial coords valid (overlap {n_match}/{n_check})")
        return

    # --- Spatial not valid: recover from GT reference ---
    if has_batch:
        batches = sorted(adata.obs[batch_col].unique())
        if len(batches) == len(ref_spatials):
            new_spatial = np.zeros((adata.n_obs, 2))
            for bi, batch_name in enumerate(batches):
                batch_indices = np.where(adata.obs[batch_col] == batch_name)[0]
                gt_sp, gt_n = ref_spatials[bi]
                n_assign = min(len(batch_indices), gt_n)
                new_spatial[batch_indices[:n_assign]] = gt_sp[:n_assign]
                if len(batch_indices) > gt_n:
                    # Cycle excess cells through GT coords (avoid pile-up)
                    excess = len(batch_indices) - gt_n
                    new_spatial[batch_indices[gt_n:]] = gt_sp[np.arange(excess) % gt_n]
            adata.obsm['spatial'] = new_spatial
            sp2 = new_spatial.copy()
            sp2[:, 1] = -sp2[:, 1]
            adata.obsm['spatial_flip'] = sp2
            print(f"    -> per-batch spatial recovery done")
            return

    # Fallback: total count match
    ref_total = sum(n for _, n in ref_spatials)
    if ref_total == adata.n_obs:
        adata.obsm['spatial'] = np.vstack([s[0] for s in ref_spatials])
    elif abs(ref_total - adata.n_obs) / max(ref_total, 1) < 0.02:
        all_sp = np.vstack([s[0] for s in ref_spatials])
        adata.obsm['spatial'] = all_sp[:adata.n_obs]

    if 'spatial' in adata.obsm:
        sp = adata.obsm['spatial'].copy()
        sp[:, 1] = -sp[:, 1]
        adata.obsm['spatial_flip'] = sp


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    out_dir = os.path.join(root, '_myx_Results', 'plots')
    os.makedirs(out_dir, exist_ok=True)

    methods = args.methods or HORIZONTAL_METHODS
    base_dir = os.path.join(root, '_myx_Results', 'adata', 'horizontal_integration')

    # Collect available methods
    available = []
    for method in methods:
        h5ad_files = sorted(Path(os.path.join(base_dir, method, args.dataset)).rglob('*.h5ad'))
        if h5ad_files:
            available.append((method, str(h5ad_files[0])))

    if not available:
        print(f"No horizontal results for {args.dataset}")
        return

    # Get reference batch labels
    ref_batch_labels = None
    try:
        first_adata = sc.read_h5ad(available[0][1])
        bc = find_batch_col(first_adata)
        if bc:
            ref_batch_labels = sorted(first_adata.obs[bc].unique().tolist())
        del first_adata
    except Exception:
        pass

    # withGT vs woGT: skip Original for woGT (no GT labels)
    has_gt = args.dataset in DATASET_GT_INFO
    show_original = has_gt

    # Load original (only for withGT)
    adata_orig, slice_names = (None, [])
    if show_original:
        adata_orig, slice_names = load_original(root, args.dataset, args.clustering, ref_batch_labels)

    # Get slice/batch labels for spatial splitting
    batch_labels = ref_batch_labels or slice_names
    n_slices = len(batch_labels)
    if not slice_names:
        slice_names = batch_labels  # woGT: use batch labels as slice names

    # --- Preload method adatas ---
    method_adatas = {}
    for method, h5ad_path in available:
        try:
            adata = sc.read_h5ad(h5ad_path)
            emb_key = None
            for key in [method, 'X_integrated', 'X_emb']:
                if key in adata.obsm:
                    emb_key = key
                    break
            if emb_key and 'X_umap' not in adata.obsm:
                sc.pp.neighbors(adata, use_rep=emb_key, n_neighbors=15)
                sc.tl.umap(adata)
            recover_spatial(adata, root, args.dataset)
            method_adatas[method] = adata
        except Exception as e:
            print(f"  [{method}] Error: {e}")

    # --- Layout ---
    n_methods = len(available)
    if show_original:
        entries_umap = ['Original'] + [m for m, _ in available]
    else:
        entries_umap = [m for m, _ in available]  # woGT: no Original
    umap_rows = math.ceil(len(entries_umap) / NCOLS)

    # Spatial: block 0 has Original + first (NCOLS-1) methods (withGT),
    #          or NCOLS methods (woGT, no Original).
    if show_original:
        methods_block0 = NCOLS - 1  # first block: Original + 6 methods = 7
    else:
        methods_block0 = NCOLS  # first block: 7 methods (no Original)
    remaining_after_block0 = max(0, n_methods - methods_block0)
    n_blocks = 1 + math.ceil(remaining_after_block0 / NCOLS) if remaining_after_block0 > 0 else 1
    spatial_rows = n_blocks * n_slices

    total_rows = umap_rows + spatial_rows
    fig, axs = plt.subplots(total_rows, NCOLS,
                            figsize=(3.0 * NCOLS, 2.8 * total_rows),
                            squeeze=False)

    # === Load scores for annotation ===
    scores_h = {}
    ranking_path = os.path.join(root, '_myx_Results', 'evaluation', 'summary',
                                 f'horizontal_ranking_{args.clustering}.csv')
    if os.path.isfile(ranking_path):
        df_rank = pd.read_csv(ranking_path)
        if 'Average' in df_rank.columns:
            scores_h = dict(zip(df_rank['Method'], df_rank['Average']))

    # === UMAP SECTION (rows 0 to umap_rows-1) ===
    for idx, entry in enumerate(entries_umap):
        r = idx // NCOLS
        c = idx % NCOLS
        ax = axs[r, c]

        if entry == 'Original':
            if adata_orig is not None and 'X_umap' in adata_orig.obsm:
                sc.pl.umap(adata_orig, color='batch', ax=ax,
                           title='Original', s=args.point_size,
                           show=False, frameon=False, legend_loc='none')
            else:
                ax.set_title('Original', fontsize=9)
                ax.axis('off')
        else:
            score_val = scores_h.get(entry)
            title_str = f'{entry}\n({score_val:.3f})' if score_val else entry
            if entry in method_adatas:
                adata = method_adatas[entry]
                batch_col = find_batch_col(adata)
                if 'X_umap' in adata.obsm and batch_col:
                    sc.pl.umap(adata, color=batch_col, ax=ax,
                               title=title_str, s=args.point_size,
                               show=False, frameon=False, legend_loc='none')
                else:
                    ax.set_title(title_str, fontsize=9)
                    ax.axis('off')
            else:
                ax.set_title(title_str, fontsize=9)
                ax.axis('off')

    # Hide empty UMAP cells
    for idx in range(len(entries_umap), umap_rows * NCOLS):
        axs[idx // NCOLS, idx % NCOLS].axis('off')

    # === Prepare per-slice GT color palettes for Original spatial ===
    has_gt = args.dataset in DATASET_GT_INFO
    gt_color_col = None
    gt_palettes = {}  # {slice_idx: {category: color}}
    if has_gt and adata_orig is not None:
        for col in ['Spatial_Label', 'Ground Truth', 'cell_type']:
            if col in adata_orig.obs.columns:
                gt_color_col = col
                break
    if gt_color_col and adata_orig is not None:
        tab20 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                 '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
                 '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5']
        batch_col_orig = find_batch_col(adata_orig) or 'batch'
        if batch_col_orig in adata_orig.obs.columns:
            sorted_batches = sorted(adata_orig.obs[batch_col_orig].unique().tolist())
            for si, bn in enumerate(sorted_batches):
                mask = adata_orig.obs[batch_col_orig] == bn
                slice_cats = sorted(adata_orig.obs.loc[mask, gt_color_col].unique().tolist())
                gt_palettes[si] = {cat: tab20[i % len(tab20)] for i, cat in enumerate(slice_cats)}

    # === Build per-method batch-to-position mapping ===
    # Map each method's sorted batch labels to slice positions (0, 1, ...)
    def get_slice_mask(adata_src, slice_idx):
        """Get mask for slice_idx-th batch, regardless of batch naming.
        Uses _spatial_batch if available (for methods with misaligned batch labels)."""
        if '_spatial_batch' in adata_src.obs.columns:
            batch_col_name = '_spatial_batch'
        else:
            batch_col_name = find_batch_col(adata_src) or 'batch'
        if batch_col_name not in adata_src.obs.columns:
            return None
        sorted_batches = sorted(adata_src.obs[batch_col_name].unique().tolist())
        if slice_idx >= len(sorted_batches):
            return None
        return adata_src.obs[batch_col_name] == sorted_batches[slice_idx]

    # === Per-slice Hungarian matching for method spatial plots ===
    from scipy.optimize import linear_sum_assignment

    def hungarian_recolor_slice(adata_slice, gt_slice, cluster_key, gt_key, palette):
        """Recolor method clusters to match GT regions for one slice."""
        if cluster_key not in adata_slice.obs.columns or gt_key not in gt_slice.obs.columns:
            return
        if not palette:
            return

        method_cats = sorted(adata_slice.obs[cluster_key].unique().tolist(),
                             key=lambda x: int(x) if str(x).isdigit() else x)
        gt_cats = sorted(gt_slice.obs[gt_key].unique().tolist())
        n_m, n_g = len(method_cats), len(gt_cats)

        # Positional matching (same cell order assumed)
        if adata_slice.n_obs != gt_slice.n_obs:
            return
        m_labels = adata_slice.obs[cluster_key].values
        g_labels = gt_slice.obs[gt_key].values

        # Build confusion matrix
        confusion = np.zeros((n_m, n_g))
        for i, mc in enumerate(method_cats):
            mask_m = (m_labels == mc)
            for j, gc in enumerate(gt_cats):
                mask_g = (g_labels == gc)
                confusion[i, j] = (mask_m & mask_g).sum()

        row_ind, col_ind = linear_sum_assignment(-confusion)

        # Assign GT region colors to matched clusters
        new_colors = {}
        new_labels = {}
        for mi, gi in zip(row_ind, col_ind):
            gt_name = gt_cats[gi]
            new_colors[method_cats[mi]] = palette.get(gt_name, '#888888')
            new_labels[method_cats[mi]] = gt_name

        # Set palette
        all_cats = sorted(new_labels.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
        adata_slice.uns[f'{cluster_key}_colors'] = np.array(
            [new_colors.get(c, '#888888') for c in all_cats])

    # === SPATIAL SECTION (rows umap_rows to end) ===
    for block_idx in range(n_blocks):
        if block_idx == 0:
            block_methods = [m for m, _ in available[:methods_block0]]
            block_entries = (['Original'] + block_methods) if show_original else block_methods
        else:
            start = methods_block0 + (block_idx - 1) * NCOLS
            block_methods = [m for m, _ in available[start:start + NCOLS]]
            block_entries = block_methods  # no Original repeat

        for slice_idx in range(n_slices):
            row = umap_rows + block_idx * n_slices + slice_idx
            slice_label = slice_names[slice_idx] if slice_idx < len(slice_names) else f'S{slice_idx}'

            for col_idx, entry in enumerate(block_entries):
                if col_idx >= NCOLS:
                    break
                ax = axs[row, col_idx]

                if entry == 'Original':
                    adata_src = adata_orig
                    color_col = gt_color_col or args.clustering
                else:
                    adata_src = method_adatas.get(entry)
                    color_col = args.clustering if adata_src is not None and args.clustering in adata_src.obs.columns else None

                if adata_src is None or color_col is None:
                    ax.set_title(f'{entry}\n({slice_label})', fontsize=7)
                    ax.axis('off')
                    continue

                # Filter to this slice by position (not by name)
                mask = get_slice_mask(adata_src, slice_idx)
                if mask is None or mask.sum() == 0:
                    ax.set_title(f'{entry}\n({slice_label})', fontsize=7)
                    ax.axis('off')
                    continue
                adata_slice = adata_src[mask].copy()

                # Set per-slice palette for Original GT labels
                if entry == 'Original' and gt_color_col and gt_color_col in adata_slice.obs.columns:
                    palette = gt_palettes.get(slice_idx, {})
                    if palette:
                        cats = sorted(palette.keys())
                        adata_slice.obs[gt_color_col] = pd.Categorical(
                            adata_slice.obs[gt_color_col], categories=cats)
                        adata_slice.uns[f'{gt_color_col}_colors'] = [
                            palette[c] for c in cats]

                # Hungarian matching for method spatial plots
                if entry != 'Original' and gt_color_col and adata_orig is not None:
                    gt_mask = get_slice_mask(adata_orig, slice_idx)
                    if gt_mask is not None and gt_mask.sum() > 0:
                        gt_slice = adata_orig[gt_mask].copy()
                        palette = gt_palettes.get(slice_idx, {})
                        hungarian_recolor_slice(adata_slice, gt_slice, color_col, gt_color_col, palette)

                # Plot spatial
                basis = 'spatial_flip' if 'spatial_flip' in adata_slice.obsm else 'spatial'
                if basis in adata_slice.obsm:
                    sc.pl.embedding(adata_slice, basis=basis, color=color_col,
                                    ax=ax, s=args.point_size * 3,
                                    show=False, frameon=False, legend_loc='none',
                                    title=f'{entry}\n({slice_label})')
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                            fontsize=8, transform=ax.transAxes)
                    ax.set_title(f'{entry}\n({slice_label})', fontsize=7)
                ax.axis('off')

            # Hide remaining empty cells in this row
            for col_idx in range(len(block_entries), NCOLS):
                axs[row, col_idx].axis('off')

    plt.subplots_adjust(hspace=0.35, wspace=0.05)

    # === Add legends ===
    legend_handles = []

    # Batch legend
    if batch_labels:
        batch_colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, bl in enumerate(batch_labels):
            legend_handles.append(mpatches.Patch(
                facecolor=batch_colors_list[i % len(batch_colors_list)], label=f'Batch: {bl}'))

    # Per-slice GT region legends
    for si, sn in enumerate(slice_names):
        palette = gt_palettes.get(si, {})
        if palette:
            legend_handles.append(mpatches.Patch(facecolor='none', edgecolor='none', label=''))
            legend_handles.append(mpatches.Patch(facecolor='none', edgecolor='none', label=f'— {sn} —'))
            for cat, color in palette.items():
                legend_handles.append(mpatches.Patch(facecolor=color, label=cat))

    if legend_handles:
        fig.legend(handles=legend_handles, loc='center right',
                   bbox_to_anchor=(1.12, 0.5), fontsize=6,
                   frameon=True, fancybox=True, edgecolor='#CCCCCC',
                   title='Legend', title_fontsize=7, ncol=1)

    out_pdf = os.path.join(out_dir, f'fig3ef_batch_umap_{args.dataset}_{args.clustering}.pdf')
    plt.savefig(out_pdf, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out_pdf}")


if __name__ == '__main__':
    main()
