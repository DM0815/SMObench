"""
Mouse Thymus1 slice comparison with per-panel legends.

This script loads the Mouse Thymus1 vertical-integration results for every
integration method and plots them in a single figure. Each subplot includes its
own legend that maps the Leiden-derived thymus annotations to colors.
"""

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

# Avoid numba caching issues when running via CLI
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "Results" / "adata" / "vertical_integration"
OUTPUT_DIR = BASE_DIR / "Results" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "mouse_thymus1_slice_comparison_legends.png"

# Integration methods to render (method, dataset, slice, filename)
METHODS = [
    ("CANDIES", "Mouse_Thymus", "Thymus1", "CANDIES_Mouse_Thymus_Thymus1.h5ad"),
    ("SpaMI", "Mouse_Thymus", "Thymus1", "SpaMI_MT_Thymus1.h5ad"),
    ("COSMOS", "Mouse_Thymus", "Thymus1", "COSMOS_Mouse_Thymus_Thymus1.h5ad"),
    ("SpatialGlue", "Mouse_Thymus", "Thymus1", "SpatialGlue_Mouse_Thymus_Thymus1.h5ad"),
    ("SpaMosaic", "Mouse_Thymus", "Thymus1", "SpaMosaic_Mouse_Thymus_Thymus1.h5ad"),
    ("SpaBalance", "Mouse_Thymus", "Thymus1", "SpaBalance_MT_Thymus1.h5ad"),
    ("PRAGA", "Mouse_Thymus", "Thymus1", "PRAGA_Mouse_Thymus_Thymus1.h5ad"),
    ("PRESENT", "Mouse_Thymus", "Thymus1", "PRESENT_Mouse_Thymus_Thymus1.h5ad"),
    ("SpaMV", "Mouse_Thymus", "Thymus1", "SpaMV_Mouse_Thymus_Thymus1.h5ad"),
    ("SpaFusion", "Mouse_Thymus", "Thymus1", "SpaFusion_MT_Thymus1.h5ad"),
    ("SpaMultiVAE", "Mouse_Thymus", "Thymus1", "SpaMultiVAE_Mouse_Thymus_Thymus1.h5ad"),
]

# Annotation mapping and palette
ANNOTATION_MAP = {
    1: "1-Medulla (SP T, mTEC, DC)",
    2: "2-Corticomedullary Junction (CMJ)",
    3: "3-Inner cortex region 1 (DN T, DP T, cTEC)",
    4: "4-Middle cortex region 2 (DN T, DP T, cTEC)",
    5: "5-Outer cortex region 3 (DN T, DP T, cTEC)",
    6: "6-Connective tissue capsule (fibroblast)",
    7: "7-Subcapsular zone (DN T)",
    8: "8-Connective tissue capsule (fibroblast, RBC, myeloid)",
}
ANNOT_LABELS = [ANNOTATION_MAP[k] for k in sorted(ANNOTATION_MAP)]

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]

COLOR_MAP = {label: color for label, color in zip(ANNOT_LABELS, PALETTE)}
UNKNOWN_COLOR = "#c0c0c0"


def load_method_adata(method_info):
    """Load AnnData for a method and attach annotation labels."""
    method, dataset, subset, filename = method_info
    file_path = RESULTS_DIR / method / dataset / subset / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file for {method}: {file_path}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = sc.read_h5ad(file_path)

    if "spatial" not in adata.obsm:
        raise KeyError(f"'spatial' coordinates missing in {file_path}")
    if "leiden" not in adata.obs:
        raise KeyError(f"'leiden' clustering missing in {file_path}")

    coords = np.asarray(adata.obsm["spatial"]).copy()
    if coords.shape[1] >= 2:
        coords[:, 1] = -coords[:, 1]
    adata.obsm["spatial"] = coords

    clusters = pd.to_numeric(adata.obs["leiden"].astype(str), errors="coerce")
    clusters = clusters.fillna(-1).astype(int)
    if clusters.min() == 0:
        clusters = clusters + 1

    labels = clusters.map(ANNOTATION_MAP)
    fallback = clusters.astype(str)
    labels = labels.fillna(fallback)

    ordered_labels = [ANNOTATION_MAP[i] for i in sorted(ANNOTATION_MAP)]
    adata.obs["Thymus_annotation"] = pd.Categorical(labels, categories=ordered_labels, ordered=True)
    adata.uns["Thymus_annotation_colors"] = [COLOR_MAP.get(lbl, UNKNOWN_COLOR) for lbl in ordered_labels]
    return adata


def plot_slices():
    n_methods = len(METHODS)
    n_rows, n_cols = 4, 3  # 12 slots, last one unused
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5.0, n_rows * 5.0))
    axes = axes.flatten()

    for idx, method_info in enumerate(METHODS):
        ax = axes[idx]
        adata = load_method_adata(method_info)
        coords = np.asarray(adata.obsm["spatial"])
        labels = adata.obs["Thymus_annotation"]
        present_labels = [lbl for lbl in labels.cat.categories if (labels == lbl).any()]

        for label in present_labels:
            mask = labels == label
            color = COLOR_MAP.get(label, UNKNOWN_COLOR)
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=color,
                s=10,
                linewidths=0,
                label=label,
            )

        ax.set_title(method_info[0], fontsize=21)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_frame_on(False)

        legend = ax.legend(
            loc="upper right",
            fontsize=8,
            markerscale=1.4,
            frameon=False,
            bbox_to_anchor=(1.0, 1.0),
            borderaxespad=0.2,
        )
        if legend is not None:
            for handle in legend.legend_handles:
                handle.set_alpha(1.0)

    # Hide any unused axes
    for idx in range(n_methods, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved slice comparison figure with per-panel legends to {OUTPUT_PATH}")


if __name__ == "__main__":
    plot_slices()
