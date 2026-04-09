"""UMAP and spatial embedding visualizations."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
from anndata import AnnData

from smobench.plot.style import apply_style, get_method_color


def _discover_methods(adata: AnnData) -> list[str]:
    """Discover method embeddings in adata.obsm."""
    skip = {"spatial", "X_umap", "X_pca", "X_umap_orig", "feat"}
    completed = list(adata.uns.get("methods_completed", []))
    if completed:
        return [m for m in completed if m in adata.obsm]
    return [k for k in adata.obsm.keys() if k not in skip and not k.startswith("X_")]


def umap_spatial(
    adata: AnnData,
    color: str = "leiden",
    embedding_key: str | None = None,
    title: str | None = None,
    figsize: tuple = (10, 4),
    save: str | None = None,
):
    """Plot UMAP and spatial embedding side by side for a single method."""
    apply_style()

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
        fig.savefig(save, dpi=300, bbox_inches="tight")
    plt.show()
    return fig


def method_comparison_grid(
    adata: AnnData,
    methods: list[str] | None = None,
    color: str | None = None,
    plot_type: str = "both",
    clustering: str = "leiden",
    n_clusters: int = 10,
    n_cols: int = 4,
    figsize_per_panel: tuple = (3, 3),
    save: str | None = None,
):
    """Plot grid comparing methods via UMAP and/or spatial coordinates.

    Parameters
    ----------
    adata : AnnData
        Integrated adata with method embeddings in obsm.
    methods : list[str], optional
        Methods to compare. None = auto-discover from obsm.
    color : str, optional
        Column in obs to color by. None = auto-detect (ground truth > clustering).
    plot_type : str
        ``"umap"``, ``"spatial"``, or ``"both"`` (UMAP row + spatial row).
    clustering : str
        Clustering method for fallback coloring.
    n_clusters : int
        For clustering if needed.
    n_cols : int
        Columns in grid.
    """
    from smobench.clustering import cluster as do_cluster

    apply_style()

    if methods is None:
        methods = _discover_methods(adata)
    if not methods:
        print("No method embeddings found.")
        return None

    # Auto-detect color: ground truth > per-method clustering
    gt_color = None
    for candidate in ["Spatial_Label", "cell_type", "celltype", "label", "ground_truth"]:
        if candidate in adata.obs.columns:
            gt_color = candidate
            break

    has_spatial = "spatial" in adata.obsm
    if plot_type == "spatial" and not has_spatial:
        print("No spatial coordinates, falling back to UMAP.")
        plot_type = "umap"
    if plot_type == "both" and not has_spatial:
        plot_type = "umap"

    show_umap = plot_type in ("umap", "both")
    show_spatial = plot_type in ("spatial", "both") and has_spatial
    n_view_rows = (1 if show_umap else 0) + (1 if show_spatial else 0)

    n_methods = len(methods)
    actual_cols = min(n_cols, n_methods)
    method_rows = (n_methods + actual_cols - 1) // actual_cols
    total_rows = method_rows * n_view_rows

    figsize = (figsize_per_panel[0] * actual_cols, figsize_per_panel[1] * total_rows)
    fig, axes = plt.subplots(total_rows, actual_cols, figsize=figsize, squeeze=False)

    for idx, method in enumerate(methods):
        m_row = idx // actual_cols
        m_col = idx % actual_cols

        # Determine color for this method
        use_color = gt_color
        if use_color is None:
            clust_key = f"{method}_{clustering}"
            if clust_key not in adata.obs.columns:
                try:
                    do_cluster(adata, method=clustering, n_clusters=n_clusters,
                               embedding_key=method, key_added=clust_key)
                except Exception:
                    pass
            use_color = clust_key if clust_key in adata.obs.columns else None

        # Compute UMAP
        if show_umap:
            try:
                sc.pp.neighbors(adata, use_rep=method, n_neighbors=20)
                sc.tl.umap(adata)
                umap_ok = True
            except Exception as e:
                umap_ok = False

        # Draw UMAP row
        if show_umap:
            base_row = m_row * n_view_rows
            ax = axes[base_row, m_col]
            if umap_ok and use_color and use_color in adata.obs.columns:
                sc.pl.umap(adata, color=use_color, ax=ax, show=False, s=8,
                           title=method, legend_loc="none")
            elif umap_ok:
                sc.pl.umap(adata, ax=ax, show=False, s=8, title=method)
            else:
                ax.text(0.5, 0.5, f"{method}\nUMAP failed", ha="center",
                        va="center", transform=ax.transAxes)
                ax.set_title(method, fontsize=10)
                ax.axis("off")

        # Draw spatial row
        if show_spatial:
            sp_row = m_row * n_view_rows + (1 if show_umap else 0)
            ax = axes[sp_row, m_col]
            if use_color and use_color in adata.obs.columns:
                sc.pl.embedding(adata, basis="spatial", color=use_color, ax=ax,
                                show=False, s=8, title=f"{method} (spatial)",
                                legend_loc="none")
            else:
                sc.pl.embedding(adata, basis="spatial", ax=ax, show=False, s=8,
                                title=f"{method} (spatial)")

    # Hide empty axes
    for r in range(total_rows):
        for c in range(actual_cols):
            method_idx = (r // n_view_rows) * actual_cols + c
            if method_idx >= n_methods:
                axes[r, c].axis("off")

    # Row labels
    if n_view_rows == 2 and method_rows == 1:
        axes[0, 0].set_ylabel("UMAP", fontsize=12, fontweight="bold")
        axes[1, 0].set_ylabel("Spatial", fontsize=12, fontweight="bold")

    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
        if save.endswith(".pdf"):
            fig.savefig(save.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.show()
    return fig
