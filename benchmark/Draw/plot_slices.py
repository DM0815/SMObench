"""
Slice comparison plot for Mouse Thymus1 vertical integration results.

This script loads the Mouse Thymus1 outputs from all integration methods,
maps Leiden clustering results to annotated thymus regions, and draws a
side-by-side slice comparison (2 rows × 6 columns; final panel used for legend).
"""

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

# Reduce numba-related issues when loading Scanpy objects
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "Results" / "adata" / "vertical_integration"
OUTPUT_DIR = BASE_DIR / "Results" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "mouse_thymus1_slice_comparison.png"

# Methods to plot (overall order determines subplot placement)
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
    ("SMOPCA", "Mouse_Thymus", "Thymus1", "SMOPCA_MT_Thymus1.h5ad"),
]

# Annotation mapping (following tutorial convention)
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

# Color palette for annotations (consistent across panels)
PALETTE = [ "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", ]
COLOR_MAP = {label: color for label, color in zip(ANNOT_LABELS, PALETTE)}
UNKNOWN_COLOR = "#c0c0c0"


def load_method_slice(method_info):
    """Load spatial coordinates and annotated labels for a single method."""
    method, dataset, slice_name, filename = method_info
    file_path = RESULTS_DIR / method / dataset / slice_name / filename
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
    # Align orientation with tutorial convention
    if coords.shape[1] >= 2:
        coords[:, 1] = -coords[:, 1]

    clusters = pd.Series(pd.to_numeric(adata.obs["leiden"].astype(str), errors="coerce"), index=adata.obs_names)
    if clusters.isna().any():
        raise ValueError(f"Non-numeric Leiden labels encountered in {file_path}")

    if clusters.min() == 0:
        clusters = clusters + 1

    labels = clusters.astype(int).map(ANNOTATION_MAP)
    # Fallback to generic cluster label if mapping missing
    labels = labels.fillna(clusters.astype(int).map(lambda x: f"Cluster {x}"))

    colors = labels.map(lambda lbl: COLOR_MAP.get(lbl, UNKNOWN_COLOR))

    # Preserve label categories for consistent legend entries
    label_categories = pd.Categorical(labels, categories=ANNOT_LABELS + sorted(set(labels) - set(ANNOT_LABELS)))

    return coords, label_categories, colors


def plot_slices():
    fig, axes = plt.subplots(2, 6, figsize=(24, 10))
    fig.suptitle("Mouse Thymus1 Slice Comparison (Leiden clusters)", fontsize=20, y=0.98)

    for idx, method_info in enumerate(METHODS):
        row = idx // 6
        col = idx % 6
        ax = axes[row, col]

        coords, labels, colors = load_method_slice(method_info)
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=6, linewidths=0)
        ax.set_title(method_info[0], fontsize=21)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_frame_on(False)

    legend_ax = fig.add_axes([0.88, 0.05, 0.05, 0.35])
    legend_ax.axis("off")
    for idx in range(len(ANNOT_LABELS)):
        legend_ax.scatter(
            0,
            idx,
            s=220,
            color=COLOR_MAP.get(ANNOT_LABELS[idx], UNKNOWN_COLOR),
            linewidths=0,
        )
        legend_ax.text(
            0.35,
            idx,
            str(idx + 1),
            fontsize=14,
            ha="left",
            va="center",
        )
    legend_ax.set_xlim(-0.6, 0.6)
    legend_ax.set_ylim(-0.5, len(ANNOT_LABELS) - 0.5)

    plt.subplots_adjust(wspace=0.02, hspace=0.08, right=0.86, bottom=0.06)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved slice comparison figure to {OUTPUT_PATH}")


if __name__ == "__main__":
    plot_slices()
