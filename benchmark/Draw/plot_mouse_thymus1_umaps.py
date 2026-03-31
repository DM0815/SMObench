"""
UMAP comparison for Mouse Thymus1 across 12 vertical integration methods.

Generates a 2x6 grid of UMAP plots, one per method, colored by Leiden clusters
when available (falls back to other clustering columns if necessary). The script
reuses the integrated embeddings stored in each AnnData file; if UMAP
coordinates are missing, they are computed on the fly from the detected
integration embedding.

Usage:
    conda run -n smobench bash -c "
      export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH;
      export NUMBA_DISABLE_JIT=1;
      python Draw/plot_mouse_thymus1_umaps.py"
"""

import os
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "Results" / "adata" / "vertical_integration"
OUTPUT_DIR = BASE_DIR / "Results" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "mouse_thymus1_umap_comparison.png"

# Method definitions match slice plotting script
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

CANDIDATE_EMBED_KEYS = [
    "SMOPCA",
    "CANDIES",
    "SpaMI",
    "COSMOS",
    "SpatialGlue",
    "SpaMosaic",
    "SpaBalance",
    "PRAGA",
    "PRESENT",
    "SpaMV",
    "SpaFusion",
    "SpaMultiVAE",
    "SpaMV",
    "X_emb",
    "X_integrated",
    "embeddings",
]

CLUSTER_COLUMNS = ["leiden", "mclust", "kmeans", "louvain"]


def infer_embedding_key(adata: sc.AnnData, method_name: str) -> Optional[str]:
    # Prefer method-specific key
    if method_name in adata.obsm_keys():
        return method_name
    for key in CANDIDATE_EMBED_KEYS:
        if key in adata.obsm_keys():
            return key
    return None


def ensure_umap(adata: sc.AnnData, embedding_key: str) -> None:
    if "X_umap" in adata.obsm:
        return
    sc.pp.neighbors(adata, use_rep=embedding_key, n_neighbors=30)
    sc.tl.umap(adata, random_state=0)


def choose_color_key(adata: sc.AnnData) -> Optional[str]:
    for key in CLUSTER_COLUMNS:
        if key in adata.obs:
            return key
    return None


def load_method_data(method_info):
    method, dataset, subset, filename = method_info
    file_path = RESULTS_DIR / method / dataset / subset / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file for {method}: {file_path}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = sc.read_h5ad(file_path)

    embedding_key = infer_embedding_key(adata, method)
    if embedding_key is None:
        raise KeyError(f"Cannot determine embedding key for {method} ({file_path})")

    adata = adata.copy()
    ensure_umap(adata, embedding_key)

    color_key = choose_color_key(adata)
    if color_key is None:
        raise KeyError(f"No clustering column found for {method} ({file_path})")

    adata.obs[color_key] = adata.obs[color_key].astype(str)
    return adata, color_key


def main():
    fig, axes = plt.subplots(2, 6, figsize=(24, 10))
    fig.suptitle("Mouse Thymus1 UMAP Comparison", fontsize=20, y=0.98)

    for idx, method_info in enumerate(METHODS):
        row, col = divmod(idx, 6)
        ax = axes[row, col]
        method_name = method_info[0]

        try:
            adata, color_key = load_method_data(method_info)
        except Exception as exc:
            ax.set_axis_off()
            ax.set_title(f"{method_name}\n{exc}", fontsize=10, color="red")
            continue

        sc.pl.umap(
            adata,
            color=color_key,
            ax=ax,
            show=False,
            title=f"{method_name}",
            legend_loc=None,
            size=18,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(left=False, bottom=False)

    plt.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved UMAP comparison figure to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

