#!/usr/bin/env python3
"""
Compose individual plots into publication-ready composite figures (PDF).
Each main figure and ED/supplementary figure → one PDF with panel labels.

Supports --fig2_dataset and --fig3_dataset to generate per-dataset variants.
By default, generates all 7 dataset variants so user can pick the main figure
and send others to supplementary.

Usage:
    python compose_figures.py --root /path/to/SMOBench-CLEAN
    python compose_figures.py --root /path/to/SMOBench-CLEAN --fig2_dataset Human_Tonsils --fig3_dataset Human_Tonsils
    python compose_figures.py --root /path/to/SMOBench-CLEAN --all_datasets
"""

import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


ALL_DATASETS = [
    'Human_Lymph_Nodes', 'Human_Tonsils',
    'Mouse_Embryos_S1', 'Mouse_Embryos_S2',
    'Mouse_Spleen', 'Mouse_Thymus', 'Mouse_Brain',
]


def add_panel_label(ax, label, x=-0.02, y=1.02, fontsize=20):
    """Add bold panel label (a, b, c...) to axes."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='bottom', ha='right')


def place_img(ax, img_path):
    """Display image in axes, hiding ticks."""
    if not os.path.isfile(img_path):
        ax.text(0.5, 0.5, f'Missing:\n{os.path.basename(img_path)}',
                ha='center', va='center', fontsize=8, transform=ax.transAxes)
        ax.set_facecolor('#f0f0f0')
    else:
        img = mpimg.imread(img_path)
        ax.imshow(img)
    ax.axis('off')


# ---------------------------------------------------------------------------
# Fig 2: Vertical Integration (5 panels: a, b, c, d, e)
# ---------------------------------------------------------------------------
def compose_fig2(plots_dir, out_dir, dataset='Human_Lymph_Nodes'):
    """
    Fig 2: Vertical Integration
      Row 0: (a) Heatmap RNA+ADT — full width
      Row 1: (a cont.) Heatmap RNA+ATAC — full width
      Row 2: (b) SC vs BioC ADT | (c) SC vs BioC ATAC
      Row 3: (d,e) UMAP + Spatial for chosen dataset — full width
    """
    fig = plt.figure(figsize=(16, 28))
    gs = fig.add_gridspec(4, 2, hspace=0.12, wspace=0.08,
                          height_ratios=[3, 3, 2.2, 4])

    # (a) Heatmap RNA+ADT — full width
    ax_a = fig.add_subplot(gs[0, :])
    place_img(ax_a, os.path.join(plots_dir, 'fig2a_vertical_heatmap_RNA+ADT_leiden.png'))
    add_panel_label(ax_a, 'a')

    # (a continued) Heatmap RNA+ATAC — full width
    ax_a2 = fig.add_subplot(gs[1, :])
    place_img(ax_a2, os.path.join(plots_dir, 'fig2a_vertical_heatmap_RNA+ATAC_leiden.png'))

    # (b) SC vs BioC scatter RNA+ADT
    ax_b = fig.add_subplot(gs[2, 0])
    place_img(ax_b, os.path.join(plots_dir, 'fig2b_sc_bioc_RNA_ADT_leiden.png'))
    add_panel_label(ax_b, 'b')

    # (c) SC vs BioC scatter RNA+ATAC
    ax_c = fig.add_subplot(gs[2, 1])
    place_img(ax_c, os.path.join(plots_dir, 'fig2c_sc_bioc_RNA_ATAC_leiden.png'))
    add_panel_label(ax_c, 'c')

    # (d,e) UMAP + spatial for chosen dataset
    ax_de = fig.add_subplot(gs[3, :])
    place_img(ax_de, os.path.join(plots_dir,
              f'fig2de_umap_spatial_{dataset}_leiden.png'))
    add_panel_label(ax_de, 'd')

    out = os.path.join(out_dir, f'Fig2_vertical_{dataset}.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out}')


# ---------------------------------------------------------------------------
# Fig 3: Horizontal Integration (6 panels: a, b, c, d, e, f)
# ---------------------------------------------------------------------------
def compose_fig3(plots_dir, out_dir, dataset='Human_Lymph_Nodes'):
    """
    Fig 3: Horizontal Integration
      Row 0: (a) Horizontal heatmap — full width
      Row 1: (b) BER vs BVC ADT | BER vs BVC ATAC
      Row 2: (c) Original spatial BEFORE integration — full width
      Row 3: (d,e) Batch + cluster UMAP AFTER integration — full width
      Row 4: (f) Spatial domain recognition AFTER integration — full width
    """
    fig = plt.figure(figsize=(16, 34))
    gs = fig.add_gridspec(5, 2, hspace=0.10, wspace=0.08,
                          height_ratios=[3, 2.2, 2.5, 4, 4])

    # (a) Horizontal heatmap — full width
    ax_a = fig.add_subplot(gs[0, :])
    place_img(ax_a, os.path.join(plots_dir, 'fig3a_horizontal_heatmap_leiden.png'))
    add_panel_label(ax_a, 'a')

    # (b) BER vs BVC scatter — ADT left, ATAC right
    ax_b1 = fig.add_subplot(gs[1, 0])
    place_img(ax_b1, os.path.join(plots_dir, 'fig3b_ber_bvc_RNA_ADT_leiden.png'))
    add_panel_label(ax_b1, 'b')

    ax_b2 = fig.add_subplot(gs[1, 1])
    place_img(ax_b2, os.path.join(plots_dir, 'fig3c_ber_bvc_RNA_ATAC_leiden.png'))

    # (c) Original spatial BEFORE integration — full width
    ax_c = fig.add_subplot(gs[2, :])
    place_img(ax_c, os.path.join(plots_dir,
              f'fig3c_original_spatial_{dataset}.png'))
    add_panel_label(ax_c, 'c')

    # (d,e) Batch + cluster UMAP AFTER integration — full width
    ax_de = fig.add_subplot(gs[3, :])
    place_img(ax_de, os.path.join(plots_dir,
              f'fig3ef_batch_umap_{dataset}_leiden.png'))
    add_panel_label(ax_de, 'd')

    # (f) Horizontal spatial domain recognition AFTER integration — full width
    ax_f = fig.add_subplot(gs[4, :])
    place_img(ax_f, os.path.join(plots_dir,
              f'fig3ef_umap_spatial_{dataset}_leiden.png'))
    add_panel_label(ax_f, 'f')

    out = os.path.join(out_dir, f'Fig3_horizontal_{dataset}.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out}')


# ---------------------------------------------------------------------------
# Fig 4: 3M + Mosaic (left/right layout)
# ---------------------------------------------------------------------------
def compose_fig4(plots_dir, out_dir):
    """
    Fig 4: 3-Modality + Mosaic
      Left column: (a) 3M heatmap on top, (b,c) 3M UMAP+spatial below
      Right column: mosaic 4-subcase
    """
    fig = plt.figure(figsize=(18, 16))
    # Left: 2 rows, Right: 1 row spanning both
    gs = fig.add_gridspec(2, 2, hspace=0.12, wspace=0.10,
                          height_ratios=[1, 2.5], width_ratios=[1, 1.2])

    # (a) 3M heatmap — top left
    ax_a = fig.add_subplot(gs[0, 0])
    place_img(ax_a, os.path.join(plots_dir, 'fig4a_3m_heatmap_leiden.png'))
    add_panel_label(ax_a, 'a')

    # (b,c) 3M UMAP + spatial — bottom left
    ax_bc = fig.add_subplot(gs[1, 0])
    place_img(ax_bc, os.path.join(plots_dir, 'fig4_3m_umap_spatial_leiden.png'))
    add_panel_label(ax_bc, 'b')

    # Right: Mosaic 4-subcase — spans both rows
    ax_right = fig.add_subplot(gs[:, 1])
    place_img(ax_right, os.path.join(plots_dir, 'fig4_mosaic_4subcase_leiden.png'))
    add_panel_label(ax_right, 'd')

    out = os.path.join(out_dir, 'Fig4_3m_mosaic.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out}')


# ---------------------------------------------------------------------------
# Fig 5: Robustness & Scalability (3 panels)
# ---------------------------------------------------------------------------
def compose_fig5(plots_dir, out_dir):
    """
    Fig 5: Robustness & Scalability
      Layout: 2 rows × 2 columns
        Left col (~65%): (a) bump charts top, (b) scalability bottom
        Right col (~35%): (c) method guide full height
    """
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(2, 2, hspace=0.15, wspace=0.08,
                          width_ratios=[6.5, 3.5], height_ratios=[1, 1])

    # (a) Bump charts — top-left, split into vertical + horizontal side by side
    ax_a = fig.add_subplot(gs[0, 0])
    # Create a merged image of vertical + horizontal bump charts
    from PIL import Image
    img_v = Image.open(os.path.join(plots_dir, 'fig5a_radar_grid_vertical.png'))
    img_h = Image.open(os.path.join(plots_dir, 'fig5a_radar_grid_horizontal.png'))
    # Side by side
    total_w = img_v.width + img_h.width
    max_h = max(img_v.height, img_h.height)
    merged = Image.new('RGB', (total_w, max_h), 'white')
    merged.paste(img_v, (0, 0))
    merged.paste(img_h, (img_v.width, 0))
    import numpy as _np
    ax_a.imshow(_np.array(merged))
    ax_a.axis('off')
    add_panel_label(ax_a, 'a')

    # (b) Scalability — bottom-left
    ax_b = fig.add_subplot(gs[1, 0])
    place_img(ax_b, os.path.join(plots_dir, 'fig5b_scalability_time_memory.png'))
    add_panel_label(ax_b, 'b')

    # (c) Method selection guide — right column, full height
    ax_c = fig.add_subplot(gs[:, 1])
    place_img(ax_c, os.path.join(plots_dir, 'fig5c_method_selection_guide.png'))
    add_panel_label(ax_c, 'c')

    out = os.path.join(out_dir, 'Fig5_robustness_scalability.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out}')


# ---------------------------------------------------------------------------
# Extended Data Figures
# ---------------------------------------------------------------------------
def compose_edfig1(plots_dir, out_dir):
    """ED Fig 1: CM-GTC orthogonality — vertical + horizontal."""
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 2, wspace=0.08)

    ax_a = fig.add_subplot(gs[0, 0])
    place_img(ax_a, os.path.join(plots_dir, 'edfig1_cmgtc_orthogonality_vertical.png'))
    add_panel_label(ax_a, 'a')

    ax_b = fig.add_subplot(gs[0, 1])
    place_img(ax_b, os.path.join(plots_dir, 'edfig1_cmgtc_orthogonality_horizontal.png'))
    add_panel_label(ax_b, 'b')

    out = os.path.join(out_dir, 'EDFig1_cmgtc_orthogonality.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out}')


def compose_edfig2(plots_dir, out_dir):
    """ED Fig 2: Weight sensitivity — vertical + horizontal."""
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 2, wspace=0.08)

    ax_a = fig.add_subplot(gs[0, 0])
    place_img(ax_a, os.path.join(plots_dir, 'edfig2_weight_sensitivity_vertical.png'))
    add_panel_label(ax_a, 'a')

    ax_b = fig.add_subplot(gs[0, 1])
    place_img(ax_b, os.path.join(plots_dir, 'edfig2_weight_sensitivity_horizontal.png'))
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
    n = len(slices)
    ncols = 2
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(16, nrows * 5))
    gs = fig.add_gridspec(nrows, ncols, hspace=0.08, wspace=0.05)

    for idx, sl in enumerate(slices):
        r, c = divmod(idx, ncols)
        ax = fig.add_subplot(gs[r, c])
        path = os.path.join(plots_dir, f'edfig3_cmgtc_spatial_{sl}.png')
        place_img(ax, path)
        if idx == 0:
            add_panel_label(ax, 'a')

    out = os.path.join(out_dir, 'EDFig3_cmgtc_spatial.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out}')


def compose_edfig4(plots_dir, out_dir):
    """ED Fig 4: woGT SC vs CM-GTC — scatter + combined."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 1, hspace=0.12)

    ax_a = fig.add_subplot(gs[0])
    place_img(ax_a, os.path.join(plots_dir, 'edfig4_wogt_sc_vs_cmgtc_leiden.png'))
    add_panel_label(ax_a, 'a')

    ax_b = fig.add_subplot(gs[1])
    place_img(ax_b, os.path.join(plots_dir, 'edfig4_wogt_combined_leiden.png'))
    add_panel_label(ax_b, 'b')

    out = os.path.join(out_dir, 'EDFig4_wogt_cmgtc.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out}')


# ---------------------------------------------------------------------------
# Supplementary: All-dataset grids
# ---------------------------------------------------------------------------
def compose_supp_all_datasets(plots_dir, out_dir, prefix, filename_template,
                              title, out_name):
    """Generic supplementary: 7-dataset grid (7 rows × 1 col)."""
    n = len(ALL_DATASETS)
    fig = plt.figure(figsize=(16, 5 * n))
    gs = fig.add_gridspec(n, 1, hspace=0.06)

    for idx, ds in enumerate(ALL_DATASETS):
        ax = fig.add_subplot(gs[idx])
        path = os.path.join(plots_dir, filename_template.format(ds=ds))
        place_img(ax, path)
        ax.set_ylabel(ds.replace('_', ' '), fontsize=11, rotation=90, labelpad=10)

    fig.suptitle(title, fontsize=14, y=1.005)
    plt.tight_layout()

    out = os.path.join(out_dir, out_name)
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out}')


def compose_supp_fig2de_all(plots_dir, out_dir):
    """Supp: All 7 datasets' vertical UMAP+spatial."""
    compose_supp_all_datasets(
        plots_dir, out_dir, 'fig2de',
        'fig2de_umap_spatial_{ds}_leiden.png',
        'Supplementary: Vertical Integration UMAP + Spatial (All Datasets)',
        'Supp_Fig2de_all_datasets.pdf',
    )


def compose_supp_fig3c_all(plots_dir, out_dir):
    """Supp: All 7 datasets' original spatial."""
    compose_supp_all_datasets(
        plots_dir, out_dir, 'fig3c',
        'fig3c_original_spatial_{ds}.png',
        'Supplementary: Original Spatial Data Before Integration (All Datasets)',
        'Supp_Fig3c_original_spatial_all.pdf',
    )


def compose_supp_fig3de_all(plots_dir, out_dir):
    """Supp: All 7 datasets' horizontal batch UMAP."""
    compose_supp_all_datasets(
        plots_dir, out_dir, 'fig3de',
        'fig3ef_batch_umap_{ds}_leiden.png',
        'Supplementary: Horizontal Batch vs Cluster UMAP (All Datasets)',
        'Supp_Fig3de_batch_umap_all.pdf',
    )


def compose_supp_fig3f_all(plots_dir, out_dir):
    """Supp: All 7 datasets' horizontal spatial domain recognition."""
    compose_supp_all_datasets(
        plots_dir, out_dir, 'fig3f',
        'fig3ef_umap_spatial_{ds}_leiden.png',
        'Supplementary: Horizontal Spatial Domain Recognition (All Datasets)',
        'Supp_Fig3f_horizontal_spatial_all.pdf',
    )


def compose_supp_kendall(plots_dir, out_dir):
    """Supp: Kendall tau clustering sensitivity heatmaps."""
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, wspace=0.10)

    ax_a = fig.add_subplot(gs[0, 0])
    place_img(ax_a, os.path.join(plots_dir, 'suppfig3_clustering_tau_vertical.png'))
    add_panel_label(ax_a, 'a')

    ax_b = fig.add_subplot(gs[0, 1])
    place_img(ax_b, os.path.join(plots_dir, 'suppfig3_clustering_tau_horizontal.png'))
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
        description='Compose publication figures from individual plots.')
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--fig2_dataset', type=str, default=None,
                        help='Dataset for Fig 2 d/e (default: generate all)')
    parser.add_argument('--fig3_dataset', type=str, default=None,
                        help='Dataset for Fig 3 c/d/e/f (default: generate all)')
    parser.add_argument('--all_datasets', action='store_true',
                        help='Generate Fig 2 and Fig 3 for all 7 datasets')
    parser.add_argument('--skip_supp', action='store_true',
                        help='Skip supplementary figure compositions')
    args = parser.parse_args()

    plots_dir = os.path.join(args.root, '_myx_Results', 'plots')
    out_dir = os.path.join(args.root, '_myx_Results', 'figures_composed')
    os.makedirs(out_dir, exist_ok=True)

    # --- Determine which datasets to compose ---
    if args.all_datasets:
        fig2_datasets = ALL_DATASETS
        fig3_datasets = ALL_DATASETS
    else:
        fig2_datasets = [args.fig2_dataset] if args.fig2_dataset else ALL_DATASETS
        fig3_datasets = [args.fig3_dataset] if args.fig3_dataset else ALL_DATASETS

    # --- Main Figures ---
    print('=== Composing Main Figures ===')

    print('\n--- Fig 2: Vertical Integration ---')
    for ds in fig2_datasets:
        compose_fig2(plots_dir, out_dir, dataset=ds)

    print('\n--- Fig 3: Horizontal Integration ---')
    for ds in fig3_datasets:
        compose_fig3(plots_dir, out_dir, dataset=ds)

    print('\n--- Fig 4: 3M + Mosaic ---')
    compose_fig4(plots_dir, out_dir)

    print('\n--- Fig 5: Robustness & Scalability ---')
    compose_fig5(plots_dir, out_dir)

    # --- ED Figures ---
    print('\n=== Composing ED Figures ===')
    compose_edfig1(plots_dir, out_dir)
    compose_edfig2(plots_dir, out_dir)
    compose_edfig3(plots_dir, out_dir)
    compose_edfig4(plots_dir, out_dir)

    # --- Supplementary Figures ---
    if not args.skip_supp:
        print('\n=== Composing Supplementary Figures ===')
        compose_supp_fig2de_all(plots_dir, out_dir)
        compose_supp_fig3c_all(plots_dir, out_dir)
        compose_supp_fig3de_all(plots_dir, out_dir)
        compose_supp_fig3f_all(plots_dir, out_dir)
        compose_supp_kendall(plots_dir, out_dir)

    print(f'\n=== All composed figures saved to: {out_dir} ===')


if __name__ == '__main__':
    main()
