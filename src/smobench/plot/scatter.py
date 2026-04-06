"""Scatter plots for benchmark analysis."""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from smobench.plot.style import apply_style, get_method_color


def sc_vs_bioc(
    df: pd.DataFrame,
    x: str = "SC_Score",
    y: str = "BioC_Score",
    color_by: str = "Method",
    title: str | None = None,
    figsize: tuple = (8, 6),
    save: str | None = None,
):
    """Plot SC vs BioC scatter for method comparison."""
    apply_style()

    summary = df.groupby(color_by)[[x, y]].mean()
    methods = summary.index.tolist()

    fig, ax = plt.subplots(figsize=figsize)

    for i, method in enumerate(methods):
        color = get_method_color(method, i)
        ax.scatter(summary.loc[method, x], summary.loc[method, y],
                   s=100, c=color, zorder=3, label=method)
        ax.annotate(method, (summary.loc[method, x], summary.loc[method, y]),
                    fontsize=8, xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title or f"SMObench: {x} vs {y}", fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.show()
    return fig


def ber_vs_bvc(
    df: pd.DataFrame,
    figsize: tuple = (8, 6),
    save: str | None = None,
):
    """Plot BER vs BVC scatter for horizontal integration."""
    return sc_vs_bioc(df, x="BER_Score", y="BVC_Score",
                      title="BER vs BVC Trade-off", figsize=figsize, save=save)
