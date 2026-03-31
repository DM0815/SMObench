#!/usr/bin/env python3
"""Plot SpaBalance 3M spatial and UMAP views."""

import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H5AD_PATH = ROOT / "Results" / "adata" / "vertical_integration" / "SpaBalance_3M" / "SpaBalance_3M_adata_3M.h5ad"


def plot_spatial(adata, keys, out_path, size=40):
    adata = adata.copy()
    adata.obsm["spatial_plot"] = adata.obsm["spatial"].copy()
    adata.obsm["spatial_plot"][:, 1] *= -1
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 4))
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        sc.pl.embedding(
            adata,
            basis="spatial_plot",
            color=key,
            ax=ax,
            show=False,
            s=size,
            title=f"SpaBalance-3M {key}",
        )
    plt.tight_layout()
    plt.savefig(ROOT / out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_umap(adata, keys, out_path, size=40):
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 4))
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        sc.pl.umap(
            adata,
            color=key,
            ax=ax,
            show=False,
            s=size,
            title=f"SpaBalance-3M {key}",
        )
    plt.tight_layout()
    plt.savefig(ROOT / out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main():
    adata = sc.read_h5ad(H5AD_PATH)
    clusters = [c for c in ["mclust", "louvain", "leiden", "kmeans"] if c in adata.obs]
    if "spatial" not in adata.obsm:
        raise ValueError("AnnData lacks spatial coordinates (.obsm['spatial']).")

    plot_spatial(adata, clusters, "spabalance_3m_spatial.png", size=40)

    if "X_umap" not in adata.obsm:
        sc.pp.neighbors(adata, use_rep="SpaBalance", n_neighbors=30)
        sc.tl.umap(adata)
    plot_umap(adata, clusters, "spabalance_3m_umap.png", size=40)


if __name__ == "__main__":
    main()
