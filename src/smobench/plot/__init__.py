"""
Visualization module for SMObench.

Usage:
    import smobench

    # From BenchmarkResult
    results = smobench.benchmark(...)
    results.plot.heatmap()
    results.plot.scatter()
    results.plot.radar()

    # Standalone functions
    smobench.plot.heatmap(df, score_col="BioC_Score")
    smobench.plot.dot_matrix(df)
    smobench.plot.sc_vs_bioc(df)
    smobench.plot.radar(df)
    smobench.plot.umap_spatial(adata)
    smobench.plot.runtime_bar(df)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from smobench.plot.style import apply_style, METHOD_COLORS, METRIC_COLORS, PAL15
from smobench.plot.heatmap import heatmap, dot_matrix
from smobench.plot.scatter import sc_vs_bioc, ber_vs_bvc
from smobench.plot.comparison import grouped_bar, method_profile, modality_comparison
from smobench.plot.radar import radar, bump_chart
from smobench.plot.umap import umap_spatial, method_comparison_grid
from smobench.plot.scalability import runtime_bar, scalability_curve
from smobench.plot.summary import plot_summary, evaluate_adata, plot_from_results

if TYPE_CHECKING:
    from smobench.pipeline.benchmark import BenchmarkResult


class ResultPlotter:
    """Accessor for plotting from BenchmarkResult."""

    def __init__(self, result: "BenchmarkResult"):
        self._result = result
        self._df = result.to_dataframe()

    def heatmap(self, score_col: str = "BioC_Score", save: str | None = None, **kwargs):
        return heatmap(self._df, score_col=score_col, save=save, **kwargs)

    def dot_matrix(self, save: str | None = None, **kwargs):
        return dot_matrix(self._df, save=save, **kwargs)

    def scatter(self, x: str = "SC_Score", y: str = "BioC_Score", save: str | None = None, **kwargs):
        return sc_vs_bioc(self._df, x=x, y=y, save=save, **kwargs)

    def radar(self, save: str | None = None, **kwargs):
        return radar(self._df, save=save, **kwargs)

    def runtime(self, save: str | None = None, **kwargs):
        return runtime_bar(self._df, save=save, **kwargs)

    def bump(self, score_col: str = "BioC_Score", save: str | None = None, **kwargs):
        by_clust = {c: g for c, g in self._df.groupby("Clustering")}
        return bump_chart(by_clust, score_col=score_col, save=save, **kwargs)

    def grouped_bar(self, x="Method", y="BioC_Score", hue="Dataset", save=None, **kwargs):
        return grouped_bar(self._df, x=x, y=y, hue=hue, save=save, **kwargs)

    def method_profile(self, method=None, metrics=None, group_by="Dataset", save=None, **kwargs):
        return method_profile(self._df, method=method, metrics=metrics, group_by=group_by, save=save, **kwargs)

    def modality_comparison(self, metrics=None, group_col="Modality", save=None, **kwargs):
        return modality_comparison(self._df, metrics=metrics, group_col=group_col, save=save, **kwargs)


__all__ = [
    "apply_style", "METHOD_COLORS", "METRIC_COLORS", "PAL15",
    "heatmap", "dot_matrix",
    "sc_vs_bioc", "ber_vs_bvc",
    "grouped_bar", "method_profile", "modality_comparison",
    "radar", "bump_chart",
    "umap_spatial", "method_comparison_grid",
    "runtime_bar", "scalability_curve",
    "ResultPlotter",
    "plot_summary", "evaluate_adata",
]
