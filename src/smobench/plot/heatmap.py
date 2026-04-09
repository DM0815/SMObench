"""Heatmap and dot-matrix plots for benchmark results.

Two styles:
  - ``heatmap()``: seaborn heatmap (quick overview)
  - ``dot_matrix()``: publication-quality dot-matrix with colored circles
    and horizontal bars for aggregate scores (matches SMObench paper style)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

from smobench.plot.style import apply_style, METRIC_COLORS


# ── Column group presets ────────────────────────────────────────

def _make_cmap(base_color):
    """Create a white→base colormap."""
    return mcolors.LinearSegmentedColormap.from_list(
        'custom', ['#FAFAFA', base_color], N=256)

CMAP_SC   = _make_cmap(METRIC_COLORS['SC'])
CMAP_BIOC = _make_cmap(METRIC_COLORS['BioC'])
CMAP_BVC  = _make_cmap(METRIC_COLORS.get('BVC', METRIC_COLORS['BioC']))
CMAP_BER  = _make_cmap(METRIC_COLORS['BER'])
CMAP_CMGTC = _make_cmap(METRIC_COLORS['CMGTC'])
CMAP_AGG  = _make_cmap(METRIC_COLORS['agg'])

# withGT column groups
GROUPS_WITHGT = [
    {
        'title': 'Spatial\nCoherence',
        'cols': ['Moran_I'],
        'cmap': CMAP_SC,
        'style': 'dot',
    },
    {
        'title': 'Biological\nConservation',
        'cols': ['ARI', 'NMI', 'cASW', 'cLISI'],
        'cmap': CMAP_BIOC,
        'style': 'dot',
    },
    {
        'title': 'Aggregate\nScore',
        'cols': ['SC_Score', 'BioC_Score', 'CMGTC', 'Total'],
        'cmap': CMAP_AGG,
        'style': 'bar',
    },
]

# woGT column groups
GROUPS_WOGT = [
    {
        'title': 'Spatial\nCoherence',
        'cols': ['Moran_I'],
        'cmap': CMAP_SC,
        'style': 'dot',
    },
    {
        'title': 'Biological\nValidation',
        'cols': ['Silhouette', 'DBI_norm', 'CHI_norm'],
        'cmap': CMAP_BVC,
        'style': 'dot',
    },
    {
        'title': 'Aggregate\nScore',
        'cols': ['SC_Score', 'BVC_Score', 'CMGTC', 'Total'],
        'cmap': CMAP_AGG,
        'style': 'bar',
    },
]

# withGT + batch (horizontal)
GROUPS_WITHGT_BATCH = [
    {
        'title': 'Spatial\nCoherence',
        'cols': ['Moran_I'],
        'cmap': CMAP_SC,
        'style': 'dot',
    },
    {
        'title': 'Biological\nConservation',
        'cols': ['ARI', 'NMI', 'cASW', 'cLISI'],
        'cmap': CMAP_BIOC,
        'style': 'dot',
    },
    {
        'title': 'Batch Effect\nRemoval',
        'cols': ['kBET', 'bASW', 'iLISI', 'KNN_conn', 'PCR'],
        'cmap': CMAP_BER,
        'style': 'dot',
    },
    {
        'title': 'Aggregate\nScore',
        'cols': ['SC_Score', 'BioC_Score', 'BER_Score', 'CMGTC', 'Total'],
        'cmap': CMAP_AGG,
        'style': 'bar',
    },
]

# woGT + batch (horizontal)
GROUPS_WOGT_BATCH = [
    {
        'title': 'Spatial\nCoherence',
        'cols': ['Moran_I'],
        'cmap': CMAP_SC,
        'style': 'dot',
    },
    {
        'title': 'Biological\nValidation',
        'cols': ['Silhouette', 'DBI_norm', 'CHI_norm'],
        'cmap': CMAP_BVC,
        'style': 'dot',
    },
    {
        'title': 'Batch Effect\nRemoval',
        'cols': ['kBET', 'bASW', 'iLISI', 'KNN_conn', 'PCR'],
        'cmap': CMAP_BER,
        'style': 'dot',
    },
    {
        'title': 'Aggregate\nScore',
        'cols': ['SC_Score', 'BVC_Score', 'BER_Score', 'CMGTC', 'Total'],
        'cmap': CMAP_AGG,
        'style': 'bar',
    },
]

# Display labels for metrics
DISPLAY_LABELS = {
    'Moran_I': "Moran's I",
    'cASW': 'cASW',
    'cLISI': 'cLISI',
    'KNN_conn': 'KNN\nconn.',
    'SC_Score': 'SC',
    'BioC_Score': 'BioC',
    'BVC_Score': 'BVC',
    'BER_Score': 'BER',
    'CMGTC': 'CM-GTC',
    'Total': 'Total',
    'DBI_norm': 'DBI\n(norm)',
    'CHI_norm': 'CHI\n(norm)',
}


# ── Seaborn heatmap (quick overview) ────────────────────────────

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
    """Plot method × dataset performance heatmap (seaborn)."""
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
    return fig


# ── Publication-quality dot-matrix ──────────────────────────────

def dot_matrix(
    df: pd.DataFrame,
    groups: list[dict] | None = None,
    sort_by: str | None = "Total",
    row: str = "Method",
    title: str | None = None,
    save: str | None = None,
    dpi: int = 300,
    row_height: float = 0.38,
    col_width: float = 0.82,
    bar_width_factor: float = 0.80,
    dot_max_radius: float = 0.155,
    fontsize_val: float = 6.5,
    fontsize_label: float = 7.5,
    show: bool = True,
):
    """Publication-quality dot-matrix heatmap.

    Parameters
    ----------
    df : DataFrame
        Must have ``row`` column (default "Method") + metric columns.
        If multiple rows per method, they are averaged.
    groups : list of dict, optional
        Column group definitions. Each dict has keys:
        ``title``, ``cols`` (list of column names), ``cmap``, ``style`` ("dot"/"bar").
        If None, auto-selects based on available columns (withGT or woGT).
    sort_by : str
        Column to sort methods by (descending). None = keep original order.
    save : str, optional
        Save path (PDF/PNG/SVG).
    """
    apply_style()

    # Aggregate by method
    metric_candidates = [c for c in df.columns if c not in (row, "Dataset", "Slice",
                         "GT", "Clustering", "Runtime", "Task", "Modality")]
    summary = df.groupby(row)[metric_candidates].mean()

    # Auto-select groups
    if groups is None:
        if "ARI" in summary.columns:
            groups = GROUPS_WITHGT
        else:
            groups = GROUPS_WOGT

    # Filter to columns that exist
    valid_groups = []
    all_cols = []
    for g in groups:
        existing = [c for c in g['cols'] if c in summary.columns]
        if existing:
            valid_groups.append({**g, 'cols': existing})
            all_cols.extend(existing)

    if not all_cols:
        print("  No valid metric columns found, skipping dot_matrix")
        return None

    # Sort
    if sort_by and sort_by in summary.columns:
        summary = summary.sort_values(sort_by, ascending=False)

    methods = summary.index.tolist()
    n_methods = len(methods)
    n_cols = len(all_cols)

    # Per-column normalization
    col_min = summary[all_cols].min()
    col_max = summary[all_cols].max()
    col_range = col_max - col_min
    col_range[col_range == 0] = 1

    # Figure layout
    left_margin = 1.3
    right_margin = 0.1
    top_margin = 0.9
    bot_margin = 0.15
    gap = 0.25

    total_gap = gap * max(0, len(valid_groups) - 1)
    fig_w = left_margin + n_cols * col_width + total_gap + right_margin
    fig_h = top_margin + n_methods * row_height + bot_margin

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis('off')

    # Column x-positions
    col_x = {}
    group_spans = []
    x_cursor = left_margin
    for gi, g in enumerate(valid_groups):
        x_start = x_cursor
        for col_name in g['cols']:
            col_x[col_name] = x_cursor + col_width / 2
            x_cursor += col_width
        x_end = x_cursor
        group_spans.append((x_start, x_end, g['title'], g['cmap']))
        if gi < len(valid_groups) - 1:
            x_cursor += gap

    # Top line
    top_line_y = fig_h - 0.05
    ax.plot([left_margin - 0.3, x_cursor + 0.05], [top_line_y, top_line_y],
            color='#000000', linewidth=1.0, solid_capstyle='butt')

    # Group headers
    header_y = fig_h - 0.30
    for x_start, x_end, group_title, cmap in group_spans:
        cx = (x_start + x_end) / 2
        ax.text(cx, header_y, group_title, ha='center', va='center',
                fontsize=fontsize_label + 1, fontweight='bold',
                color='#000000', linespacing=1.05)

    # Title
    if title:
        ax.text(fig_w / 2, fig_h + 0.1, title, ha='center', va='bottom',
                fontsize=fontsize_label + 2, fontweight='bold', color='#000000')

    # "Method" label
    label_y = fig_h - top_margin + 0.18
    ax.text(left_margin - 0.12, label_y + 0.04, 'Method', ha='right', va='bottom',
            fontsize=fontsize_label - 1, color='#666666', fontstyle='italic')

    # Column labels
    for col_name in all_cols:
        lbl = DISPLAY_LABELS.get(col_name, col_name)
        ax.text(col_x[col_name], label_y, lbl, ha='center', va='bottom',
                fontsize=fontsize_label - 1, color='#666666',
                fontweight='normal', linespacing=1.0)

    # Separator line below column labels (colored per group)
    sep_y = label_y - 0.06
    for x_start, x_end, _, cmap in group_spans:
        base_color = cmap(0.65)
        ax.plot([x_start, x_end], [sep_y, sep_y],
                color=base_color, linewidth=1.5, solid_capstyle='butt')

    # Draw rows
    for ri, method in enumerate(methods):
        y = sep_y - 0.08 - (ri + 0.5) * row_height

        # Row separator
        if ri > 0:
            line_y = y + row_height / 2
            ax.plot([left_margin - 0.2, x_cursor], [line_y, line_y],
                    color='#E8E8E8', linewidth=0.4, zorder=1)

        # Method name
        ax.text(left_margin - 0.12, y, method, ha='right', va='center',
                fontsize=fontsize_label, fontweight='bold', color='#000000')

        # Draw cells
        for g in valid_groups:
            cmap = g['cmap']
            style = g['style']
            for col_name in g['cols']:
                cx = col_x[col_name]
                raw_val = summary.loc[method, col_name]

                if pd.isna(raw_val):
                    ax.text(cx, y, '---', ha='center', va='center',
                            fontsize=fontsize_val - 1, color='#BBBBBB')
                    continue

                norm_val = (raw_val - col_min[col_name]) / col_range[col_name]
                norm_val = np.clip(norm_val, 0, 1)

                if style == 'dot':
                    radius = dot_max_radius
                    color = cmap(0.15 + 0.85 * norm_val)
                    circle = plt.Circle((cx, y), radius,
                                        facecolor=color, edgecolor='none',
                                        transform=ax.transData, zorder=3)
                    ax.add_patch(circle)
                    r_c, g_c, b_c = color[:3]
                    lum = 0.299 * r_c + 0.587 * g_c + 0.114 * b_c
                    text_color = 'white' if lum < 0.55 else '#000000'
                    ax.text(cx, y, f'{raw_val:.2f}', ha='center', va='center',
                            fontsize=fontsize_val, color=text_color, zorder=4)

                elif style == 'bar':
                    bar_h = row_height * 0.50
                    bar_max_w = col_width * bar_width_factor
                    bar_w = bar_max_w * max(norm_val, 0.10)
                    color = cmap(0.35 + 0.65 * norm_val)

                    rect = FancyBboxPatch(
                        (cx - bar_max_w / 2, y - bar_h / 2),
                        bar_w, bar_h,
                        boxstyle="round,pad=0.02",
                        facecolor=color, edgecolor='none', zorder=3)
                    ax.add_patch(rect)
                    r_c, g_c, b_c = color[:3]
                    lum = 0.299 * r_c + 0.587 * g_c + 0.114 * b_c
                    text_color = 'white' if lum < 0.55 else '#000000'
                    ax.text(cx, y, f'{raw_val:.2f}', ha='center', va='center',
                            fontsize=fontsize_val, color=text_color, zorder=4)

    # Bottom line
    last_y = sep_y - 0.08 - n_methods * row_height
    ax.plot([left_margin - 0.3, x_cursor + 0.05], [last_y, last_y],
            color='#000000', linewidth=1.0, solid_capstyle='butt')

    if save:
        plt.savefig(save, dpi=dpi, bbox_inches='tight', facecolor='white',
                    pad_inches=0.08)
        print(f'  Saved: {save}')

    if show:
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
