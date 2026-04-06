"""UMAP and spatial embedding visualizations."""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
from anndata import AnnData

from smobench.plot.style import apply_style


def umap_spatial(
    adata: AnnData,
    color: str = "leiden",
    embedding_key: str | None = None,
    title: str | None = None,
    figsize: tuple = (10, 4),
    save: str | None = None,
):
    """Plot UMAP and spatial embedding side by side."""
    apply_style()

    # Compute UMAP if needed
    if "X_umap" not in adata.obsm:
        if embedding_key:
            sc.pp.neighbors(adata, use_rep=embedding_key, n_neighbors=20)
        sc.tl.umap(adata)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    sc.pl.umap(adata, color=color, ax=axes[0], show=False, s=15,
               title=f"{title or ''} UMAP" if title else "UMAP")

    if "spatial" in adata.obsm:
        sc.pl.embedding(adata, basis="spatial", color=color, ax=axes[1],
                        show=False, s=15,
                        title=f"{title or ''} Spatial" if title else "Spatial")
    else:
        axes[1].text(0.5, 0.5, "No spatial coordinates", ha="center", va="center")
        axes[1].set_title("Spatial")

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.show()
    return fig


def method_comparison_grid(
    adata: AnnData,
    methods: list[str],
    color: str = "leiden",
    plot_type: str = "umap",
    n_cols: int = 4,
    figsize_per_panel: tuple = (3, 3),
    save: str | None = None,
):
    """Plot grid of UMAP/spatial for multiple methods.

    Parameters
    ----------
    adata : AnnData
        Integrated adata with multiple method embeddings in obsm and
        clustering results in obs (e.g., 'SpatialGlue_leiden').
    methods : list[str]
        Method names to compare.
    color : str
        Clustering method suffix (e.g., 'leiden' → uses '{method}_leiden').
    """
    apply_style()

    n_methods = len(methods)
    n_rows = (n_methods + n_cols - 1) // n_cols
    figsize = (figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, method in enumerate(methods):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        clust_key = f"{method}_{color}"
        if clust_key not in adata.obs:
            ax.text(0.5, 0.5, f"{method}\n(no {color})", ha="center", va="center")
            ax.set_title(method)
            continue

        if plot_type == "umap" and f"{method}_umap" in adata.obsm:
            adata.obsm["X_umap"] = adata.obsm[f"{method}_umap"]
            sc.pl.umap(adata, color=clust_key, ax=ax, show=False, s=10, title=method)
        elif "spatial" in adata.obsm:
            sc.pl.embedding(adata, basis="spatial", color=clust_key, ax=ax,
                            show=False, s=10, title=method)
        else:
            ax.text(0.5, 0.5, method, ha="center", va="center")

    # Hide empty axes
    for idx in range(n_methods, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis("off")

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.show()
    return fig
