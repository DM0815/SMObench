"""Scalability and runtime visualization."""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from smobench.plot.style import apply_style, get_method_color


def runtime_bar(
    df: pd.DataFrame,
    method_col: str = "Method",
    time_col: str = "Runtime",
    title: str = "Method Runtime Comparison",
    figsize: tuple = (10, 5),
    save: str | None = None,
):
    """Bar chart of method runtimes."""
    apply_style()

    summary = df.groupby(method_col)[time_col].median().sort_values()

    fig, ax = plt.subplots(figsize=figsize)
    colors = [get_method_color(m, i)
              for i, m in enumerate(summary.index)]
    bars = ax.barh(range(len(summary)), summary.values, color=colors, edgecolor="gray", linewidth=0.3)

    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels(summary.index)
    ax.set_xlabel("Runtime (seconds)")
    ax.set_title(title, fontweight="bold")

    # Add value labels
    for bar, val in zip(bars, summary.values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}s", va="center", fontsize=8)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.show()
    return fig


def scalability_curve(
    df: pd.DataFrame,
    x_col: str = "n_cells",
    y_col: str = "Runtime",
    method_col: str = "Method",
    title: str = "Scalability",
    figsize: tuple = (8, 5),
    save: str | None = None,
):
    """Line plot of runtime vs dataset size."""
    apply_style()

    fig, ax = plt.subplots(figsize=figsize)

    methods = df[method_col].unique()
    for i, method in enumerate(sorted(methods)):
        subset = df[df[method_col] == method].sort_values(x_col)
        color = get_method_color(method, i)
        ax.plot(subset[x_col], subset[y_col], "o-", label=method,
                color=color, linewidth=1.5, markersize=4)

    ax.set_xlabel("Number of cells")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title(title, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.show()
    return fig
