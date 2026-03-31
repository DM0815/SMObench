#!/usr/bin/env python3
"""
Compose individual plots into publication-ready composite figures (PDF) — v2.

v2 changes vs v1:
  - Auto-computes height_ratios from actual image aspect ratios → no whitespace
  - Fig2/3: heatmaps side-by-side, compact layout
  - EDFig1/2: taller (fixes too-wide issue, ratio 2.3→~1.5)
  - EDFig4: side-by-side instead of stacked (fixes ratio 1.3→~1.8)
  - Supp all-datasets: 4×2 grid instead of 7×1 column (fixes ratio 0.2→~0.9)
  - Outputs to both _myx_Results/figures_composed/ and optionally picture/ archive

Usage:
    python compose_figures_v2.py --root /path/to/SMOBench-CLEAN
    python compose_figures_v2.py --root /path/to/SMOBench-CLEAN --fig2_dataset Human_Tonsils
"""

import os
import argparse
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import numpy as np


ALL_DATASETS = [
    'Human_Lymph_Nodes', 'Human_Tonsils',
    'Mouse_Embryos_S1', 'Mouse_Embryos_S2',
    'Mouse_Spleen', 'Mouse_Thymus', 'Mouse_Brain',
]


def img_hw_ratio(img_path):
    """Return height/width ratio of image. Returns 1.0 if missing."""
    if os.path.isfile(img_path):
        img = mpimg.imread(img_path)
        h, w = img.shape[:2]
        return h / w
    return 1.0


def add_panel_label(ax, label, x=-0.02, y=1.02, fontsize=18):
    """Add bold panel label (a, b, c...) to axes."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='bottom', ha='right')


def autocrop(img, bg_thresh=250):
    """Trim white borders from image array. bg_thresh: pixel value threshold for 'white'."""
    if img.ndim == 3 and img.shape[2] >= 3:
        if img.dtype == np.float32 or img.dtype == np.float64:
            gray = np.mean(img[:, :, :3], axis=2)
            mask = gray < (bg_thresh / 255.0)
        else:
            gray = np.mean(img[:, :, :3].astype(float), axis=2)
            mask = gray < bg_thresh
    else:
        mask = img < bg_thresh if img.max() > 1 else img < (bg_thresh / 255.0)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return img
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # Add tiny margin (2px) to avoid cutting into content
    pad = 2
    rmin = max(0, rmin - pad)
    rmax = min(img.shape[0] - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(img.shape[1] - 1, cmax + pad)
    return img[rmin:rmax+1, cmin:cmax+1]


def place_img(ax, img_path):
    """Display image in axes: autocrop white borders, stretch to fill."""
    if not os.path.isfile(img_path):
        ax.text(0.5, 0.5, f'Missing:\n{os.path.basename(img_path)}',
                ha='center', va='center', fontsize=8, transform=ax.transAxes)
        ax.set_facecolor('#f0f0f0')
    else:
        img = mpimg.imread(img_path)
        img = autocrop(img)
        ax.imshow(img)
    ax.axis('off')


# ---------------------------------------------------------------------------
# Fig 2: Vertical Integration — v2 compact layout
# ---------------------------------------------------------------------------
def compose_fig2(plots_dir, out_dir, dataset='Human_Lymph_Nodes'):
    """
    Fig 2 v2: Vertical Integration
      Row 0: (a) Heatmap RNA+ADT (left) | Heatmap RNA+ATAC (right)
      Row 1: (b) SC vs BioC ADT (left) | (c) SC vs BioC ATAC (right)
      Row 2: (d,e) UMAP + Spatial grid — full width
    """
    p = lambda f: os.path.join(plots_dir, f)

    path_adt  = p('fig2a_vertical_heatmap_RNA_ADT_leiden.png')
    path_atac = p('fig2a_vertical_heatmap_RNA_ATAC_leiden.png')
    path_sc1  = p('fig2b_sc_bioc_RNA_ADT_leiden.png')
    path_sc2  = p('fig2c_sc_bioc_RNA_ATAC_leiden.png')
    path_umap = p(f'fig2de_umap_spatial_{dataset}_leiden.png')

    # Compute proportional height_ratios based on actual images
    # Side-by-side panels (half width): effective row height = (fw/2) * ratio
    # Full-width panels: effective row height = fw * ratio
    r0 = max(img_hw_ratio(path_adt), img_hw_ratio(path_atac))    # ~1.1
    r1 = max(img_hw_ratio(path_sc1), img_hw_ratio(path_sc2))     # ~0.75
    r2 = img_hw_ratio(path_umap)                                  # ~0.47 (v2 grid) or ~6.2 (v1)

    # Cap the UMAP panel ratio to prevent extremely tall figures
    r2 = min(r2, 1.5)

    fig_w = 16
    # Adjust ratios for half-width vs full-width: row height = width_factor * ratio
    # Row 0,1: half-width → height = (fw/2)*r;  Row 2: full-width → height = fw*r
    h0 = (fig_w / 2) * r0
    h1 = (fig_w / 2) * r1
    h2 = fig_w * r2
    height_ratios = [h0, h1, h2]
    fig_h = h0 + h1 + h2

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(3, 2, hspace=0, wspace=0,
                          height_ratios=height_ratios)

    # (a) Heatmaps side by side
    ax_a1 = fig.add_subplot(gs[0, 0])
    place_img(ax_a1, path_adt)
    add_panel_label(ax_a1, 'a')

    ax_a2 = fig.add_subplot(gs[0, 1])
    place_img(ax_a2, path_atac)

    # (b,c) Scatters side by side
    ax_b = fig.add_subplot(gs[1, 0])
    place_img(ax_b, path_sc1)
    add_panel_label(ax_b, 'b')

    ax_c = fig.add_subplot(gs[1, 1])
    place_img(ax_c, path_sc2)
    add_panel_label(ax_c, 'c')

    # (d,e) UMAP + spatial grid — full width
    ax_de = fig.add_subplot(gs[2, :])
    place_img(ax_de, path_umap)
    add_panel_label(ax_de, 'd')

    out = os.path.join(out_dir, f'Fig2_vertical_{dataset}.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out}')


# ---------------------------------------------------------------------------
# Fig 3: Horizontal Integration — v2 compact layout
# ---------------------------------------------------------------------------
def compose_fig3(plots_dir, out_dir, dataset='Human_Lymph_Nodes'):
    """
    Fig 3 v2: Horizontal Integration — compact 3-row layout
      Row 0: (a) Horizontal heatmap — full width
      Row 1: (b) BER scatter ADT | BER scatter ATAC | (c) Original spatial
      Row 2: (d,e) Batch UMAP (left) | (f) Spatial domain grid (right)
    """
    p = lambda f: os.path.join(plots_dir, f)

    path_heat  = p('fig3a_horizontal_heatmap_leiden.png')
    path_ber1  = p('fig3b_ber_bvc_RNA_ADT_leiden.png')
    path_ber2  = p('fig3c_ber_bvc_RNA_ATAC_leiden.png')
    path_orig  = p(f'fig3c_original_spatial_{dataset}.png')
    path_batch = p(f'fig3ef_batch_umap_{dataset}_leiden.png')
    path_spat  = p(f'fig3ef_umap_spatial_{dataset}_leiden.png')

    r_heat  = img_hw_ratio(path_heat)     # ~0.69
    r_ber   = max(img_hw_ratio(path_ber1), img_hw_ratio(path_ber2))
    r_orig  = img_hw_ratio(path_orig)     # ~0.46
    r_batch = img_hw_ratio(path_batch)    # ~1.07
    r_spat  = min(img_hw_ratio(path_spat), 1.5)

    fig_w = 16
    # Row 0: full width heatmap
    h0 = (fig_w) * r_heat
    # Row 1: 3 panels (each 1/3 width)
    h1 = (fig_w / 3) * max(r_ber, r_orig)
    # Row 2: 2 panels side by side (batch + spatial)
    h2 = (fig_w / 2) * max(r_batch, r_spat)
    height_ratios = [h0, h1, h2]
    fig_h = h0 + h1 + h2

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(3, 6, hspace=0, wspace=0,
                          height_ratios=height_ratios)

    # (a) Heatmap full width (span all 6 cols)
    ax_a = fig.add_subplot(gs[0, :])
    place_img(ax_a, path_heat)
    add_panel_label(ax_a, 'a')

    # Row 1: (b) scatter ADT | scatter ATAC | (c) original spatial
    ax_b1 = fig.add_subplot(gs[1, 0:2])
    place_img(ax_b1, path_ber1)
    add_panel_label(ax_b1, 'b')

    ax_b2 = fig.add_subplot(gs[1, 2:4])
    place_img(ax_b2, path_ber2)

    ax_c = fig.add_subplot(gs[1, 4:6])
    place_img(ax_c, path_orig)
    add_panel_label(ax_c, 'c')

    # Row 2: (d,e) batch UMAP | (f) spatial domain
    ax_de = fig.add_subplot(gs[2, 0:3])
    place_img(ax_de, path_batch)
    add_panel_label(ax_de, 'd')

    ax_f = fig.add_subplot(gs[2, 3:6])
    place_img(ax_f, path_spat)
    add_panel_label(ax_f, 'f')

    out = os.path.join(out_dir, f'Fig3_horizontal_{dataset}.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out}')


# ---------------------------------------------------------------------------
# Fig 4: 3M + Mosaic — same as v1 (already OK, ratio ~1.1)
# ---------------------------------------------------------------------------
def compose_fig4(plots_dir, out_dir):
    """Fig 4: 3-Modality + Mosaic (left/right layout)."""
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(2, 2, hspace=0, wspace=0,
                          height_ratios=[1, 2.5], width_ratios=[1, 1.2])

    ax_a = fig.add_subplot(gs[0, 0])
    place_img(ax_a, os.path.join(plots_dir, 'fig4a_3m_heatmap_leiden.png'))
    add_panel_label(ax_a, 'a')

    ax_bc = fig.add_subplot(gs[1, 0])
    place_img(ax_bc, os.path.join(plots_dir, 'fig4_3m_umap_spatial_leiden.png'))
    add_panel_label(ax_bc, 'b')

    ax_right = fig.add_subplot(gs[:, 1])
    place_img(ax_right, os.path.join(plots_dir, 'fig4_mosaic_4subcase_leiden.png'))
    add_panel_label(ax_right, 'c')

    out = os.path.join(out_dir, 'Fig4_3m_mosaic.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out}')


# ---------------------------------------------------------------------------
# Fig 5: Robustness & Scalability — same as v1 (already OK, ratio ~0.8)
# ---------------------------------------------------------------------------
def compose_fig5(plots_dir, out_dir):
    """Fig 5: Robustness & Scalability."""
    fig = plt.figure(figsize=(16, 20))
    gs = fig.add_gridspec(3, 2, hspace=0, wspace=0,
                          height_ratios=[3, 3, 2.5])

    ax_a1 = fig.add_subplot(gs[0, :])
    place_img(ax_a1, os.path.join(plots_dir, 'fig5a_radar_grid_vertical.png'))
    add_panel_label(ax_a1, 'a')

    ax_a2 = fig.add_subplot(gs[1, :])
    place_img(ax_a2, os.path.join(plots_dir, 'fig5a_radar_grid_horizontal.png'))

    ax_b = fig.add_subplot(gs[2, 0])
    place_img(ax_b, os.path.join(plots_dir, 'fig5b_scalability_time_memory.png'))
    add_panel_label(ax_b, 'b')

    ax_c = fig.add_subplot(gs[2, 1])
    place_img(ax_c, os.path.join(plots_dir, 'fig5c_method_selection_guide.png'))
    add_panel_label(ax_c, 'c')

    out = os.path.join(out_dir, 'Fig5_robustness_scalability.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out}')


# ---------------------------------------------------------------------------
# Extended Data Figures — v2 improved aspect ratios
# ---------------------------------------------------------------------------
def compose_edfig1(plots_dir, out_dir):
    """ED Fig 1: CM-GTC orthogonality — v2 vertical stack (ratio 3.9→~1.0)."""
    p = lambda f: os.path.join(plots_dir, f)
    path_a = p('edfig1_cmgtc_orthogonality_vertical.png')
    path_b = p('edfig1_cmgtc_orthogonality_horizontal.png')

    r_a = img_hw_ratio(path_a)
    r_b = img_hw_ratio(path_b)

    fig_w = 14
    fig_h = fig_w * (r_a + r_b)  # sum of heights + spacing

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(2, 1, hspace=0, height_ratios=[r_a, r_b])

    ax_a = fig.add_subplot(gs[0])
    place_img(ax_a, path_a)
    add_panel_label(ax_a, 'a')

    ax_b = fig.add_subplot(gs[1])
    place_img(ax_b, path_b)
    add_panel_label(ax_b, 'b')

    out = os.path.join(out_dir, 'EDFig1_cmgtc_orthogonality.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out}')


def compose_edfig2(plots_dir, out_dir):
    """ED Fig 2: Weight sensitivity — v2 vertical stack (ratio 3.1→~1.0)."""
    p = lambda f: os.path.join(plots_dir, f)
    path_a = p('edfig2_weight_sensitivity_vertical.png')
    path_b = p('edfig2_weight_sensitivity_horizontal.png')

    r_a = img_hw_ratio(path_a)
    r_b = img_hw_ratio(path_b)

    fig_w = 14
    fig_h = fig_w * (r_a + r_b)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(2, 1, hspace=0, height_ratios=[r_a, r_b])

    ax_a = fig.add_subplot(gs[0])
    place_img(ax_a, path_a)
    add_panel_label(ax_a, 'a')

    ax_b = fig.add_subplot(gs[1])
    place_img(ax_b, path_b)
    add_panel_label(ax_b, 'b')

    out = os.path.join(out_dir, 'EDFig2_weight_sensitivity.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out}')


def compose_edfig3(plots_dir, out_dir):
    """ED Fig 3: Per-spot CM-GTC spatial — grid of representative slices."""
    slices = [
        'Human_Lymph_Nodes_A1', 'Human_Lymph_Nodes_D1',
        'Human_Tonsils_S1', 'Human_Tonsils_S2',
        'Mouse_Embryos_S1_E11', 'Mouse_Embryos_S1_E15',
        'Mouse_Embryos_S2_E11', 'Mouse_Embryos_S2_E15',
    ]
    ncols = 2
    nrows = math.ceil(len(slices) / ncols)

    fig = plt.figure(figsize=(16, nrows * 5))
    gs = fig.add_gridspec(nrows, ncols, hspace=0, wspace=0)

    for idx, sl in enumerate(slices):
        r, c = divmod(idx, ncols)
        ax = fig.add_subplot(gs[r, c])
        place_img(ax, os.path.join(plots_dir, f'edfig3_cmgtc_spatial_{sl}.png'))
        if idx == 0:
            add_panel_label(ax, 'a')

    out = os.path.join(out_dir, 'EDFig3_cmgtc_spatial.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out}')


def compose_edfig4(plots_dir, out_dir):
    """ED Fig 4: woGT SC vs CM-GTC — v2 vertical stack (ratio 4.0→~0.9)."""
    p = lambda f: os.path.join(plots_dir, f)
    path_a = p('edfig4_wogt_sc_vs_cmgtc_leiden.png')
    path_b = p('edfig4_wogt_combined_leiden.png')

    r_a = img_hw_ratio(path_a)   # ~0.33 (wide)
    r_b = img_hw_ratio(path_b)   # ~0.80 (squarish)

    fig_w = 14
    fig_h = fig_w * (r_a + r_b)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(2, 1, hspace=0, height_ratios=[r_a, r_b])

    ax_a = fig.add_subplot(gs[0])
    place_img(ax_a, path_a)
    add_panel_label(ax_a, 'a')

    ax_b = fig.add_subplot(gs[1])
    place_img(ax_b, path_b)
    add_panel_label(ax_b, 'b')

    out = os.path.join(out_dir, 'EDFig4_wogt_cmgtc.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out}')


# ---------------------------------------------------------------------------
# Supplementary: All-dataset grids — v2 multi-column layout
# ---------------------------------------------------------------------------
def compose_supp_all_datasets(plots_dir, out_dir, prefix, filename_template,
                              title, out_name, ncols_grid=2):
    """
    Supplementary: 7-dataset grid — v2 uses ncols_grid columns (default 2)
    instead of single column, producing a much more compact figure.
    """
    n = len(ALL_DATASETS)
    nrows_grid = math.ceil(n / ncols_grid)

    cell_w = 16.0 / ncols_grid
    # Estimate cell height from first available image
    first_path = os.path.join(plots_dir, filename_template.format(ds=ALL_DATASETS[0]))
    cell_h = cell_w * img_hw_ratio(first_path)
    cell_h = min(cell_h, 12)  # cap per-cell height

    fig_w = 16
    fig_h = nrows_grid * cell_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(nrows_grid, ncols_grid, hspace=0, wspace=0)

    for idx, ds in enumerate(ALL_DATASETS):
        r = idx // ncols_grid
        c = idx % ncols_grid
        ax = fig.add_subplot(gs[r, c])
        path = os.path.join(plots_dir, filename_template.format(ds=ds))
        place_img(ax, path)
        ax.set_title(ds.replace('_', ' '), fontsize=10, pad=4)

    # Hide unused cells
    for idx in range(n, nrows_grid * ncols_grid):
        r = idx // ncols_grid
        c = idx % ncols_grid
        fig.add_subplot(gs[r, c]).axis('off')

    fig.suptitle(title, fontsize=13, y=1.005)

    out = os.path.join(out_dir, out_name)
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out}')


def compose_supp_fig2de_all(plots_dir, out_dir):
    compose_supp_all_datasets(
        plots_dir, out_dir, 'fig2de',
        'fig2de_umap_spatial_{ds}_leiden.png',
        'Supplementary: Vertical Integration UMAP + Spatial (All Datasets)',
        'Supp_Fig2de_all_datasets.pdf',
        ncols_grid=2,
    )

def compose_supp_fig3c_all(plots_dir, out_dir):
    compose_supp_all_datasets(
        plots_dir, out_dir, 'fig3c',
        'fig3c_original_spatial_{ds}.png',
        'Supplementary: Original Spatial Data Before Integration (All Datasets)',
        'Supp_Fig3c_original_spatial_all.pdf',
        ncols_grid=2,
    )

def compose_supp_fig3de_all(plots_dir, out_dir):
    compose_supp_all_datasets(
        plots_dir, out_dir, 'fig3de',
        'fig3ef_batch_umap_{ds}_leiden.png',
        'Supplementary: Horizontal Batch vs Cluster UMAP (All Datasets)',
        'Supp_Fig3de_batch_umap_all.pdf',
        ncols_grid=2,
    )

def compose_supp_fig3f_all(plots_dir, out_dir):
    compose_supp_all_datasets(
        plots_dir, out_dir, 'fig3f',
        'fig3ef_umap_spatial_{ds}_leiden.png',
        'Supplementary: Horizontal Spatial Domain Recognition (All Datasets)',
        'Supp_Fig3f_horizontal_spatial_all.pdf',
        ncols_grid=2,
    )

def compose_supp_kendall(plots_dir, out_dir):
    """Supp: Kendall tau clustering sensitivity — v2 vertical stack."""
    p = lambda f: os.path.join(plots_dir, f)
    path_a = p('suppfig3_clustering_tau_vertical.png')
    path_b = p('suppfig3_clustering_tau_horizontal.png')

    r_a = img_hw_ratio(path_a)
    r_b = img_hw_ratio(path_b)

    fig_w = 12
    fig_h = fig_w * (r_a + r_b)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(2, 1, hspace=0, height_ratios=[r_a, r_b])

    ax_a = fig.add_subplot(gs[0])
    place_img(ax_a, path_a)
    add_panel_label(ax_a, 'a')

    ax_b = fig.add_subplot(gs[1])
    place_img(ax_b, path_b)
    add_panel_label(ax_b, 'b')

    out = os.path.join(out_dir, 'Supp_clustering_sensitivity_kendall.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Compose publication figures from individual plots (v2).')
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--fig2_dataset', type=str, default=None)
    parser.add_argument('--fig3_dataset', type=str, default=None)
    parser.add_argument('--all_datasets', action='store_true')
    parser.add_argument('--skip_supp', action='store_true')
    parser.add_argument('--only', type=str, default=None,
                        help='Only compose specific figure (fig2,fig3,fig4,fig5,ed1-4,supp)')
    args = parser.parse_args()

    plots_dir = os.path.join(args.root, '_myx_Results', 'plots')
    out_dir = os.path.join(args.root, '_myx_Results', 'figures_composed')
    os.makedirs(out_dir, exist_ok=True)

    if args.all_datasets:
        fig2_datasets = ALL_DATASETS
        fig3_datasets = ALL_DATASETS
    else:
        fig2_datasets = [args.fig2_dataset] if args.fig2_dataset else ALL_DATASETS
        fig3_datasets = [args.fig3_dataset] if args.fig3_dataset else ALL_DATASETS

    only = args.only

    # --- Main Figures ---
    if only is None or 'fig2' in only:
        print('=== Fig 2: Vertical Integration (v2) ===')
        for ds in fig2_datasets:
            compose_fig2(plots_dir, out_dir, dataset=ds)

    if only is None or 'fig3' in only:
        print('\n=== Fig 3: Horizontal Integration (v2) ===')
        for ds in fig3_datasets:
            compose_fig3(plots_dir, out_dir, dataset=ds)

    if only is None or 'fig4' in only:
        print('\n=== Fig 4: 3M + Mosaic ===')
        compose_fig4(plots_dir, out_dir)

    if only is None or 'fig5' in only:
        print('\n=== Fig 5: Robustness & Scalability ===')
        compose_fig5(plots_dir, out_dir)

    # --- ED Figures ---
    if only is None or 'ed1' in only:
        print('\n=== ED Fig 1 (v2) ===')
        compose_edfig1(plots_dir, out_dir)
    if only is None or 'ed2' in only:
        print('\n=== ED Fig 2 (v2) ===')
        compose_edfig2(plots_dir, out_dir)
    if only is None or 'ed3' in only:
        print('\n=== ED Fig 3 ===')
        compose_edfig3(plots_dir, out_dir)
    if only is None or 'ed4' in only:
        print('\n=== ED Fig 4 (v2) ===')
        compose_edfig4(plots_dir, out_dir)

    # --- Supplementary ---
    if not args.skip_supp and (only is None or 'supp' in only):
        print('\n=== Supplementary Figures (v2) ===')
        compose_supp_fig2de_all(plots_dir, out_dir)
        compose_supp_fig3c_all(plots_dir, out_dir)
        compose_supp_fig3de_all(plots_dir, out_dir)
        compose_supp_fig3f_all(plots_dir, out_dir)
        compose_supp_kendall(plots_dir, out_dir)

    print(f'\n=== All composed figures saved to: {out_dir} ===')


if __name__ == '__main__':
    main()
