"""Heatmap and dot-matrix plots for benchmark results."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from smobench.plot.style import apply_style, METRIC_COLORS


def heatmap(
    df: pd.DataFrame,
    score_col: str = "BioC_Score",
    row: str = "Method",
    col: str = "Dataset",
    title: str | None = None,
    cmap: str = "RdYlGn",
    figsize: tuple | None = None,
    save: str | None = None,
    **kwargs,
):
    """Plot method × dataset performance heatmap.

    Parameters
    ----------
    df : DataFrame
        Benchmark results with Method, Dataset, and score columns.
    score_col : str
        Column to plot.
    row, col : str
        Columns for rows and columns of heatmap.
    """
    apply_style()

    pivot = df.pivot_table(index=row, columns=col, values=score_col, aggfunc="mean")
    pivot = pivot.sort_values(pivot.columns.tolist(), ascending=False)

    if figsize is None:
        figsize = (max(8, len(pivot.columns) * 1.5), max(4, len(pivot) * 0.5))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap=cmap, ax=ax,
        linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8, "label": score_col},
        **kwargs,
    )
    ax.set_title(title or f"SMObench: {score_col}", fontweight="bold")
    ax.set_ylabel("")
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
        print(f"Saved: {save}")
    plt.show()
    return fig


def dot_matrix(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    row: str = "Method",
    figsize: tuple | None = None,
    save: str | None = None,
):
    """Plot dot-matrix heatmap (like scib-style summary).

    Each cell shows a colored dot sized by score value.
    """
    apply_style()

    if metrics is None:
        metrics = [c for c in df.columns if c.endswith("_Score") or
                   c in ("ARI", "NMI", "Moran_I", "Silhouette", "kBET")]

    # Aggregate by method
    summary = df.groupby(row)[metrics].mean()
    summary = summary.sort_values(metrics[0], ascending=False)

    n_methods = len(summary)
    n_metrics = len(metrics)

    if figsize is None:
        figsize = (max(6, n_metrics * 1.2), max(4, n_methods * 0.5))

    fig, ax = plt.subplots(figsize=figsize)

    for i, method in enumerate(summary.index):
        for j, metric in enumerate(metrics):
            val = summary.loc[method, metric]
            if pd.isna(val):
                continue

            # Color by metric group
            color = _metric_color(metric)
            size = max(20, val * 300)  # scale dot size

            ax.scatter(j, n_methods - 1 - i, s=size, c=color, alpha=0.8, edgecolors="gray", linewidth=0.3)
            ax.text(j, n_methods - 1 - i, f"{val:.2f}", ha="center", va="center", fontsize=7)

    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_methods))
    ax.set_yticklabels(reversed(summary.index.tolist()), fontsize=10)
    ax.set_xlim(-0.5, n_metrics - 0.5)
    ax.set_ylim(-0.5, n_methods - 0.5)
    ax.set_title("SMObench: Method Comparison", fontweight="bold")

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
        print(f"Saved: {save}")
    plt.show()
    return fig


def _metric_color(metric_name: str) -> str:
    """Get color for a metric based on its group."""
    if "SC" in metric_name or "Moran" in metric_name:
        return METRIC_COLORS["SC"]
    elif "BioC" in metric_name or metric_name in ("ARI", "NMI", "cASW", "cLISI"):
        return METRIC_COLORS["BioC"]
    elif "BER" in metric_name or metric_name in ("kBET", "bASW", "iLISI", "KNN_conn", "PCR"):
        return METRIC_COLORS["BER"]
    elif "CMGTC" in metric_name:
        return METRIC_COLORS["CMGTC"]
    else:
        return METRIC_COLORS["agg"]
