"""Radar charts and bump charts for method comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from smobench.plot.style import apply_style, get_method_color


def radar(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    methods: list[str] | None = None,
    title: str = "SMObench: Method Radar",
    figsize: tuple = (8, 8),
    save: str | None = None,
):
    """Plot radar chart comparing methods across metrics."""
    apply_style()

    if metrics is None:
        metrics = [c for c in df.columns if c.endswith("_Score")]
    if methods is None:
        methods = df["Method"].unique().tolist()

    summary = df.groupby("Method")[metrics].mean()
    summary = summary.loc[[m for m in methods if m in summary.index]]

    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    for i, method in enumerate(summary.index):
        values = summary.loc[method].tolist()
        values += values[:1]
        color = get_method_color(method, i)
        ax.plot(angles, values, 'o-', linewidth=1.5, label=method, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_title(title, fontweight="bold", pad=20)
    ax.legend(bbox_to_anchor=(1.15, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.show()
    return fig


def bump_chart(
    results_by_clustering: dict[str, pd.DataFrame],
    score_col: str = "BioC_Score",
    title: str = "Rank Stability Across Clustering Methods",
    figsize: tuple = (10, 6),
    save: str | None = None,
):
    """Plot bump chart showing method rank stability across clustering algorithms.

    Parameters
    ----------
    results_by_clustering : dict
        {clustering_name: DataFrame} where each DataFrame has Method and score columns.
    """
    apply_style()

    rankings = {}
    for clust_name, df in results_by_clustering.items():
        summary = df.groupby("Method")[score_col].mean().sort_values(ascending=False)
        rankings[clust_name] = {m: rank + 1 for rank, m in enumerate(summary.index)}

    rank_df = pd.DataFrame(rankings)
    methods = rank_df.index.tolist()
    clustering_names = list(results_by_clustering.keys())

    fig, ax = plt.subplots(figsize=figsize)

    for i, method in enumerate(methods):
        ranks = [rank_df.loc[method, c] for c in clustering_names]
        color = get_method_color(method, i)
        ax.plot(range(len(clustering_names)), ranks, 'o-', color=color,
                label=method, linewidth=1.5, markersize=6)

    ax.set_xticks(range(len(clustering_names)))
    ax.set_xticklabels(clustering_names)
    ax.set_ylabel("Rank")
    ax.set_title(title, fontweight="bold")
    ax.invert_yaxis()
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.show()
    return fig
