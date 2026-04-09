"""Flexible comparison plots: cross-dataset, cross-modality, single-method profiles.

Examples
--------
>>> from smobench.plot.comparison import grouped_bar, method_profile, modality_comparison
>>>
>>> # Compare all methods: ADT vs ATAC
>>> grouped_bar(df, x="Method", y="BioC_Score", hue="Modality")
>>>
>>> # Single method performance across datasets
>>> method_profile(df, method="SpatialGlue", metrics=["ARI", "NMI", "Moran_I"])
>>>
>>> # Modality breakdown: grouped bar per metric
>>> modality_comparison(df, metrics=["SC_Score", "BioC_Score"], group_col="Modality")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from smobench.plot.style import apply_style, get_method_color


# ── Grouped bar chart ───────────────────────────────────────────

def grouped_bar(
    df: pd.DataFrame,
    x: str = "Method",
    y: str = "BioC_Score",
    hue: str = "Dataset",
    title: str | None = None,
    figsize: tuple | None = None,
    palette: dict | list | None = None,
    ylabel: str | None = None,
    sort_by_mean: bool = True,
    horizontal: bool = False,
    bar_width: float = 0.8,
    show_values: bool = True,
    save: str | None = None,
):
    """Flexible grouped bar chart.

    Parameters
    ----------
    df : DataFrame
        Must contain columns ``x``, ``y``, ``hue``.
    x : str
        Category axis (e.g. "Method", "Dataset").
    y : str
        Value axis (metric column).
    hue : str
        Grouping variable (e.g. "Modality", "Dataset", "Method").
    sort_by_mean : bool
        Sort x-axis categories by mean y value (descending).
    horizontal : bool
        If True, draw horizontal bars.
    """
    apply_style()

    pivot = df.pivot_table(index=x, columns=hue, values=y, aggfunc="mean")
    if sort_by_mean:
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    categories = pivot.index.tolist()
    groups = pivot.columns.tolist()
    n_cat = len(categories)
    n_grp = len(groups)

    if figsize is None:
        if horizontal:
            figsize = (max(8, n_grp * 2), max(4, n_cat * 0.6))
        else:
            figsize = (max(8, n_cat * 1.2), 6)

    fig, ax = plt.subplots(figsize=figsize)

    positions = np.arange(n_cat)
    width = bar_width / n_grp

    # Colors
    if palette is None:
        cmap = plt.cm.Set2
        colors = [cmap(i / max(n_grp - 1, 1)) for i in range(n_grp)]
    elif isinstance(palette, dict):
        colors = [palette.get(g, '#888888') for g in groups]
    else:
        colors = list(palette)

    for i, grp in enumerate(groups):
        vals = pivot[grp].values
        offset = (i - (n_grp - 1) / 2) * width
        pos = positions + offset
        mask = ~np.isnan(vals)

        if horizontal:
            bars = ax.barh(pos[mask], vals[mask], height=width * 0.9,
                           color=colors[i % len(colors)], label=grp, zorder=3)
            if show_values:
                for bar, v in zip(bars, vals[mask]):
                    ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                            f'{v:.3f}', va='center', fontsize=7)
        else:
            bars = ax.bar(pos[mask], vals[mask], width=width * 0.9,
                          color=colors[i % len(colors)], label=grp, zorder=3)
            if show_values:
                for bar, v in zip(bars, vals[mask]):
                    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                            f'{v:.3f}', ha='center', va='bottom', fontsize=7)

    if horizontal:
        ax.set_yticks(positions)
        ax.set_yticklabels(categories)
        ax.set_xlabel(ylabel or y)
        ax.invert_yaxis()
    else:
        ax.set_xticks(positions)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.set_ylabel(ylabel or y)

    ax.set_title(title or f"{y} by {x} (grouped by {hue})", fontweight="bold")
    ax.legend(title=hue, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(axis='x' if horizontal else 'y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
        print(f"Saved: {save}")
    plt.show()
    return fig


# ── Method profile ──────────────────────────────────────────────

def method_profile(
    df: pd.DataFrame,
    method: str | list[str] | None = None,
    metrics: list[str] | None = None,
    group_by: str = "Dataset",
    plot_type: str = "bar",
    title: str | None = None,
    figsize: tuple | None = None,
    save: str | None = None,
):
    """Show one or more methods' performance across datasets/slices.

    Parameters
    ----------
    df : DataFrame
        Full results DataFrame.
    method : str or list
        Method name(s). If None, uses all methods.
    metrics : list[str]
        Which metrics to show. If None, auto-selects composite scores.
    group_by : str
        How to group x-axis: "Dataset", "Slice", "Dataset_Slice".
    plot_type : str
        "bar" for grouped bar, "radar" for radar, "line" for line plot.
    """
    apply_style()

    sub = df.copy()
    if method is not None:
        if isinstance(method, str):
            method = [method]
        sub = sub[sub["Method"].isin(method)]

    if sub.empty:
        print(f"No data for method(s): {method}")
        return None

    if metrics is None:
        candidates = ["SC_Score", "BioC_Score", "BVC_Score", "BER_Score", "Total"]
        metrics = [m for m in candidates if m in sub.columns and sub[m].notna().any()]

    # Build group label
    if group_by == "Dataset_Slice":
        sub = sub.copy()
        sub["_group"] = sub["Dataset"].astype(str) + "/" + sub["Slice"].astype(str)
        group_col = "_group"
    else:
        group_col = group_by

    if plot_type == "radar":
        return _profile_radar(sub, method, metrics, group_col, title, figsize, save)
    elif plot_type == "line":
        return _profile_line(sub, method, metrics, group_col, title, figsize, save)
    else:
        return _profile_bar(sub, method, metrics, group_col, title, figsize, save)


def _profile_bar(df, methods, metrics, group_col, title, figsize, save):
    """Grouped bar: x=metric, hue=group (dataset/slice)."""
    if len(methods) == 1:
        method_name = methods[0]
        sub = df[df["Method"] == method_name]
        pivot = sub.groupby(group_col)[metrics].mean()

        groups = pivot.index.tolist()
        n_metrics = len(metrics)
        n_groups = len(groups)

        if figsize is None:
            figsize = (max(8, n_metrics * 1.5), 6)

        fig, ax = plt.subplots(figsize=figsize)
        positions = np.arange(n_metrics)
        width = 0.8 / n_groups
        cmap = plt.cm.Set2

        for i, grp in enumerate(groups):
            vals = pivot.loc[grp, metrics].values
            offset = (i - (n_groups - 1) / 2) * width
            bars = ax.bar(positions + offset, vals, width=width * 0.9,
                          color=cmap(i / max(n_groups - 1, 1)), label=str(grp), zorder=3)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                            f'{v:.3f}', ha='center', va='bottom', fontsize=7)

        ax.set_xticks(positions)
        ax.set_xticklabels(metrics, rotation=30, ha='right')
        ax.set_ylabel("Score")
        ax.set_title(title or f"{method_name}: Performance by {group_col}", fontweight="bold")
        ax.legend(title=group_col, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    else:
        # Multiple methods: x=group, hue=method, subplots per metric
        n_metrics = len(metrics)
        if figsize is None:
            figsize = (max(6, n_metrics * 5), 5)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize, sharey=False)
        if n_metrics == 1:
            axes = [axes]

        for mi, metric in enumerate(metrics):
            ax = axes[mi]
            pivot = df.pivot_table(index=group_col, columns="Method", values=metric, aggfunc="mean")
            method_list = [m for m in methods if m in pivot.columns]
            groups = pivot.index.tolist()
            n_grp = len(groups)
            n_m = len(method_list)
            positions = np.arange(n_grp)
            width = 0.8 / n_m

            for i, m in enumerate(method_list):
                color = get_method_color(m, i)
                vals = pivot[m].values
                offset = (i - (n_m - 1) / 2) * width
                ax.bar(positions + offset, vals, width=width * 0.9,
                       color=color, label=m, zorder=3)

            ax.set_xticks(positions)
            ax.set_xticklabels(groups, rotation=45, ha='right', fontsize=8)
            ax.set_title(metric, fontweight="bold")
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            if mi == 0:
                ax.set_ylabel("Score")
            if mi == n_metrics - 1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

        fig.suptitle(title or f"Method Comparison by {group_col}", fontweight="bold", y=1.02)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
        print(f"Saved: {save}")
    plt.show()
    return fig


def _profile_radar(df, methods, metrics, group_col, title, figsize, save):
    """Radar: one polygon per group, axes = metrics."""
    groups_in_data = df[group_col].unique().tolist()

    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    if figsize is None:
        figsize = (8, 8)

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    cmap = plt.cm.Set2

    for method_name in methods:
        sub = df[df["Method"] == method_name]
        for gi, grp in enumerate(groups_in_data):
            grp_data = sub[sub[group_col] == grp]
            if grp_data.empty:
                continue
            vals = grp_data[metrics].mean().tolist()
            vals += vals[:1]
            label = f"{method_name} ({grp})" if len(methods) > 1 else str(grp)
            color = cmap(gi / max(len(groups_in_data) - 1, 1))
            ax.plot(angles, vals, 'o-', linewidth=1.5, label=label, color=color)
            ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    default_title = f"{methods[0]}: Performance by {group_col}" if len(methods) == 1 else f"Method Profile by {group_col}"
    ax.set_title(title or default_title, fontweight="bold", pad=20)
    ax.legend(bbox_to_anchor=(1.15, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
        print(f"Saved: {save}")
    plt.show()
    return fig


def _profile_line(df, methods, metrics, group_col, title, figsize, save):
    """Line plot: x=group, y=metric, one subplot per metric."""
    groups = sorted(df[group_col].unique())
    n_metrics = len(metrics)

    if figsize is None:
        figsize = (max(8, len(groups) * 1.5), max(4, n_metrics * 2.5))

    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)
    if n_metrics == 1:
        axes = [axes]

    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        for i, m in enumerate(methods):
            sub = df[df["Method"] == m]
            vals = sub.groupby(group_col)[metric].mean()
            vals = vals.reindex(groups)
            color = get_method_color(m, i)
            ax.plot(range(len(groups)), vals.values, 'o-', color=color,
                    label=m, linewidth=1.5, markersize=5)
        ax.set_ylabel(metric)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        if mi == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    axes[-1].set_xticks(range(len(groups)))
    axes[-1].set_xticklabels(groups, rotation=45, ha='right', fontsize=8)
    fig.suptitle(title or f"Performance Trend by {group_col}", fontweight="bold")
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
        print(f"Saved: {save}")
    plt.show()
    return fig


# ── Modality comparison ─────────────────────────────────────────

def modality_comparison(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    methods: list[str] | None = None,
    group_col: str = "Modality",
    title: str | None = None,
    figsize: tuple | None = None,
    save: str | None = None,
):
    """Compare method performance across modalities (or any grouping).

    Produces a faceted bar chart: one subplot per metric, x=method, hue=group_col.

    Parameters
    ----------
    df : DataFrame
        Must have columns: Method, ``group_col``, and metric columns.
    metrics : list[str]
        Metrics to compare. If None, auto-selects available composite scores.
    methods : list[str]
        Which methods to include. If None, uses all.
    group_col : str
        Column to compare across (e.g. "Modality", "Dataset").
    """
    apply_style()

    if metrics is None:
        candidates = ["SC_Score", "BioC_Score", "BVC_Score", "BER_Score", "Total"]
        metrics = [m for m in candidates if m in df.columns and df[m].notna().any()]

    if methods is not None:
        df = df[df["Method"].isin(methods)]

    if group_col not in df.columns:
        print(f"Column '{group_col}' not found. Available: {list(df.columns)}")
        return None

    groups = sorted(df[group_col].unique())
    method_list = df.groupby("Method")[metrics[0]].mean().sort_values(ascending=False).index.tolist()
    n_metrics = len(metrics)
    n_methods = len(method_list)
    n_groups = len(groups)

    if figsize is None:
        figsize = (max(8, n_methods * 0.8 * n_metrics), 5)

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize, sharey=False)
    if n_metrics == 1:
        axes = [axes]

    cmap = plt.cm.Set2
    colors = [cmap(i / max(n_groups - 1, 1)) for i in range(n_groups)]

    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        pivot = df.pivot_table(index="Method", columns=group_col, values=metric, aggfunc="mean")
        pivot = pivot.reindex(method_list)

        positions = np.arange(n_methods)
        width = 0.8 / n_groups

        for gi, grp in enumerate(groups):
            if grp not in pivot.columns:
                continue
            vals = pivot[grp].values
            offset = (gi - (n_groups - 1) / 2) * width
            mask = ~np.isnan(vals)
            ax.bar(positions[mask] + offset, vals[mask], width=width * 0.9,
                   color=colors[gi], label=str(grp) if mi == 0 else None, zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(method_list, rotation=60, ha='right', fontsize=8)
        ax.set_title(metric, fontweight="bold")
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        if mi == 0:
            ax.set_ylabel("Score")

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title=group_col, loc="upper right",
               bbox_to_anchor=(1.0, 1.0), fontsize=9)

    fig.suptitle(title or f"Method Performance by {group_col}", fontweight="bold", y=1.03)
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
        print(f"Saved: {save}")
    plt.show()
    return fig
