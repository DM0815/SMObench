"""Summary comparison plots from integrated h5ad files.

Core API: pass one or more AnnData (or h5ad paths), auto-discover methods,
auto-split withGT / woGT, evaluate, and plot comparisons.

Usage::

    import smobench.plot as splot

    # Single dataset directory
    splot.plot_summary("results/vertical/Human_Lymph_Nodes/A1/", save_dir="figures/")

    # Multiple datasets — auto-splits GT / non-GT
    splot.plot_summary([
        "results/vertical/Human_Lymph_Nodes/A1/",
        "results/vertical/Mouse_Spleen/Mouse_Spleen1/",
    ], save_dir="figures/")

    # AnnData directly
    splot.plot_summary(adata, dataset_name="Human_Lymph_Nodes", slice_name="A1")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData


def _discover_methods(adata: AnnData) -> list[str]:
    """Discover method embeddings stored in adata.obsm."""
    skip = {"spatial", "X_umap", "X_pca", "X_umap_orig", "feat"}
    completed = list(adata.uns.get("methods_completed", []))
    if completed:
        return [m for m in completed if m in adata.obsm]
    return [k for k in adata.obsm.keys() if k not in skip and not k.startswith("X_")]


def _auto_n_clusters(dataset_name: str, slice_name: str, fallback: int = 10) -> int:
    """Look up n_clusters from _constants, fallback if not found."""
    try:
        from smobench._constants import get_n_clusters
        return get_n_clusters(dataset_name, slice_name)
    except (KeyError, ImportError):
        return fallback


def _auto_label_key(adata: AnnData) -> str | None:
    """Auto-detect ground truth label column."""
    for candidate in ["Spatial_Label", "cell_type", "celltype", "label", "ground_truth"]:
        if candidate in adata.obs.columns:
            return candidate
    return None


def _auto_gt(dataset_name: str) -> bool | None:
    """Check if dataset has ground truth from constants."""
    try:
        from smobench._constants import DATASETS
        ds = DATASETS.get(dataset_name, {})
        return ds.get("gt", None)
    except ImportError:
        return None


def evaluate_adata(
    adata: AnnData,
    clustering: str = "leiden",
    n_clusters: int | None = None,
    label_key: str | None = None,
    dataset_name: str = "",
    slice_name: str = "",
) -> pd.DataFrame:
    """Evaluate all methods in a single adata, return DataFrame.

    Parameters
    ----------
    adata : AnnData
        Integrated adata with method embeddings in ``obsm``.
    clustering : str
        Clustering method.
    n_clusters : int, optional
        Number of clusters. None = auto-detect from dataset config.
    label_key : str, optional
        Ground truth label column. None = auto-detect.
    dataset_name, slice_name : str
        Used for auto-detecting n_clusters and GT status.
    """
    from smobench.clustering import cluster
    from smobench.metrics.evaluate import evaluate

    methods = _discover_methods(adata)
    if not methods:
        raise ValueError("No method embeddings found in adata.obsm")

    # Auto-detect parameters
    if n_clusters is None:
        n_clusters = _auto_n_clusters(dataset_name, slice_name)
    if label_key is None:
        label_key = _auto_label_key(adata)

    has_gt = _auto_gt(dataset_name)
    if has_gt is None:
        has_gt = label_key is not None

    records = []
    for method in methods:
        clust_key = f"{method}_{clustering}"
        if clust_key not in adata.obs.columns:
            try:
                cluster(adata, method=clustering, n_clusters=n_clusters,
                        embedding_key=method, key_added=clust_key)
            except Exception as e:
                print(f"  {method}: cluster error — {e}")
                continue

        try:
            scores = evaluate(
                adata, embedding_key=method, cluster_key=clust_key,
                label_key=label_key, has_gt=has_gt,
            )
        except Exception as e:
            print(f"  {method}: eval error — {e}")
            continue

        runtime = adata.uns.get(f"{method}_train_time", np.nan)
        records.append({
            "Method": method,
            "Dataset": dataset_name,
            "Slice": slice_name,
            "GT": has_gt,
            "Clustering": clustering,
            "Runtime": round(float(runtime), 1) if np.isfinite(float(runtime)) else np.nan,
            **scores,
        })

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────
# Internal: collect data from various input types
# ──────────────────────────────────────────────────────────

def _collect(
    data: Union[AnnData, str, Path, list],
    clustering: str,
    label_key: str | None,
    dataset_name: str,
    slice_name: str,
) -> tuple[list[pd.DataFrame], list[tuple[AnnData, str, str]]]:
    """Collect evaluation DataFrames and (adata, ds, sl) tuples for UMAP."""
    dfs = []
    adatas = []  # (adata, dataset_name, slice_name)

    items = data if isinstance(data, list) else [data]

    for item in items:
        if isinstance(item, AnnData):
            ds = dataset_name
            sl = slice_name
            df = evaluate_adata(item, clustering=clustering, label_key=label_key,
                                dataset_name=ds, slice_name=sl)
            dfs.append(df)
            adatas.append((item, ds, sl))

        elif isinstance(item, (str, Path)):
            p = Path(item)
            if p.is_file() and p.suffix == ".h5ad":
                adata = sc.read_h5ad(str(p))
                ds = dataset_name or p.parent.parent.name
                sl = slice_name or p.parent.name
                df = evaluate_adata(adata, clustering=clustering, label_key=label_key,
                                    dataset_name=ds, slice_name=sl)
                dfs.append(df)
                adatas.append((adata, ds, sl))

            elif p.is_dir():
                # Load all h5ad files, track which methods we've seen
                ds = dataset_name or p.parent.name
                sl = slice_name or p.name
                seen_methods = set()
                all_h5ad = sorted(p.glob("*_integrated.h5ad"))

                # Load merged file first (has multiple methods)
                merged = p / "adata_integrated.h5ad"
                if merged.is_file():
                    adata = sc.read_h5ad(str(merged))
                    df = evaluate_adata(adata, clustering=clustering, label_key=label_key,
                                        dataset_name=ds, slice_name=sl)
                    dfs.append(df)
                    adatas.append((adata, ds, sl))
                    seen_methods.update(_discover_methods(adata))

                # Then load per-method files for methods not in merged
                for h5ad in all_h5ad:
                    if h5ad.name == "adata_integrated.h5ad":
                        continue
                    method_name = h5ad.stem.replace("_integrated", "")
                    if method_name in seen_methods:
                        continue
                    adata = sc.read_h5ad(str(h5ad))
                    df = evaluate_adata(adata, clustering=clustering, label_key=label_key,
                                        dataset_name=ds, slice_name=sl)
                    dfs.append(df)
                    adatas.append((adata, ds, sl))
                    seen_methods.update(_discover_methods(adata))
            else:
                raise FileNotFoundError(f"Not found: {p}")
        else:
            raise TypeError(f"Unsupported type: {type(item)}")

    return dfs, adatas


# ──────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────

def _save_fig(fig, save_dir: str | None, name: str):
    """Save figure as PDF + PNG."""
    if save_dir and fig is not None:
        for ext in ("pdf", "png"):
            path = os.path.join(save_dir, f"{name}.{ext}")
            fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {name}.pdf / .png")


def _plot_ranking(df, score_col, save_dir, suffix=""):
    """Horizontal bar chart of mean score per method."""
    import matplotlib.pyplot as plt
    from smobench.plot.style import apply_style, get_method_color

    apply_style()
    ranking = df.groupby("Method")[score_col].mean().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(3, len(ranking) * 0.4)))
    colors = [get_method_color(m, i) for i, m in enumerate(ranking.index)]
    bars = ax.barh(range(len(ranking)), ranking.values, color=colors,
                   edgecolor="gray", linewidth=0.3)
    ax.set_yticks(range(len(ranking)))
    ax.set_yticklabels(ranking.index)
    ax.set_xlabel(score_col)
    ax.set_title(f"Method Ranking by {score_col}", fontweight="bold")

    for bar, val in zip(bars, ranking.values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    _save_fig(fig, save_dir, f"ranking{suffix}")
    plt.show()
    return fig


def _make_title(task_label: str, is_gt: bool, datasets: list[str]) -> str:
    """Build a descriptive plot title."""
    gt_str = "with Ground Truth" if is_gt else "without Ground Truth"
    if len(datasets) == 1:
        return f"{task_label} — {datasets[0]} ({gt_str})"
    return f"{task_label} ({gt_str})"


def _plot_group(df, group_name, score_col, metric_list, plots, save_dir,
                has_batch=False, is_gt=True, task_label=""):
    """Plot a GT or non-GT group."""
    from smobench.plot.heatmap import (heatmap, dot_matrix,
                                        GROUPS_WITHGT, GROUPS_WOGT,
                                        GROUPS_WITHGT_BATCH, GROUPS_WOGT_BATCH)
    from smobench.plot.scatter import sc_vs_bioc
    from smobench.plot.radar import radar
    from smobench.plot.scalability import runtime_bar

    suffix = f"_{group_name}"
    datasets = df["Dataset"].unique().tolist()
    n_datasets = df[["Dataset", "Slice"]].drop_duplicates().shape[0]
    title = _make_title(task_label or group_name, is_gt, datasets)

    # Normalize DBI/CHI via cross-method min-max (matching original scripts)
    if not is_gt:
        df = df.copy()
        if "DBI" in df.columns:
            # DBI: lower is better → (max - val) / (max - min)
            dbi = df["DBI"].dropna()
            if len(dbi) > 0 and dbi.max() != dbi.min():
                df["DBI_norm"] = (dbi.max() - df["DBI"]) / (dbi.max() - dbi.min())
            else:
                df["DBI_norm"] = 0.5
        if "CHI" in df.columns:
            # CHI: higher is better → (val - min) / (max - min)
            chi = df["CHI"].dropna()
            if len(chi) > 0 and chi.max() != chi.min():
                df["CHI_norm"] = (df["CHI"] - chi.min()) / (chi.max() - chi.min())
            else:
                df["CHI_norm"] = 0.5

    print(f"\n{'─'*50}")
    print(f"  {title}: {df['Method'].nunique()} methods × {n_datasets} slices")
    print(f"{'─'*50}")

    # Heatmap: only useful with >1 dataset/slice
    if "heatmap" in plots and score_col in df.columns and n_datasets > 1:
        df = df.copy()
        df["Dataset_Slice"] = df["Dataset"] + "/" + df["Slice"]
        fig = heatmap(df, score_col=score_col, col="Dataset_Slice",
                      title=f"{title}: {score_col}")
        _save_fig(fig, save_dir, f"heatmap{suffix}")

    # Dot matrix (publication-quality)
    if "dot_matrix" in plots:
        if has_batch:
            groups = GROUPS_WITHGT_BATCH if is_gt else GROUPS_WOGT_BATCH
        else:
            groups = GROUPS_WITHGT if is_gt else GROUPS_WOGT
        fig = dot_matrix(df, groups=groups, title=title,
                         save=os.path.join(save_dir, f"dot_matrix{suffix}.pdf") if save_dir else None,
                         show=True)

    # Radar
    if "radar" in plots:
        avail_metrics = [m for m in metric_list if m in df.columns and df[m].notna().any()]
        if len(avail_metrics) < 3:
            # Fall back to sub-metrics
            if is_gt:
                fallback = ["ARI", "NMI", "cASW", "cLISI", "Moran_I"]
            else:
                fallback = ["Silhouette", "DBI", "CHI", "Moran_I"]
            avail_metrics = [m for m in fallback if m in df.columns and df[m].notna().any()]
        if len(avail_metrics) >= 3:
            fig = radar(df, metrics=avail_metrics)
            _save_fig(fig, save_dir, f"radar{suffix}")

    # Scatter (withGT only: SC vs BioC)
    if "scatter" in plots and "SC_Score" in df.columns and "BioC_Score" in df.columns:
        if df["BioC_Score"].notna().any():
            fig = sc_vs_bioc(df)
            _save_fig(fig, save_dir, f"scatter{suffix}")

    # Ranking
    if "rank" in plots and score_col in df.columns:
        _plot_ranking(df, score_col, save_dir, suffix)


# ──────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────

def plot_summary(
    data: Union[AnnData, str, Path, list],
    clustering: str = "leiden",
    label_key: str | None = None,
    save_dir: str | None = None,
    plots: list[str] | None = None,
    dataset_name: str = "",
    slice_name: str = "",
) -> pd.DataFrame:
    """Evaluate and plot method comparison from h5ad data.

    Automatically splits results into withGT and woGT groups, each getting
    appropriate metrics and plots.

    Parameters
    ----------
    data : AnnData, str, Path, or list
        - ``AnnData``: single integrated adata
        - ``str/Path``: h5ad file or directory with ``*_integrated.h5ad``
        - ``list``: list of any of the above
    clustering : str
        Clustering method.
    label_key : str, optional
        Ground truth label column. Auto-detected if None.
    save_dir : str, optional
        Save figures here (PDF + PNG).
    plots : list[str], optional
        Which plots. None = all. Options: ``heatmap``, ``dot_matrix``,
        ``radar``, ``scatter``, ``runtime``, ``rank``, ``umap_grid``,
        ``spatial_grid``.
    dataset_name, slice_name : str
        Metadata for single-adata input.

    Returns
    -------
    pd.DataFrame
        Evaluation results.
    """
    from smobench.plot.scalability import runtime_bar
    from smobench.plot.umap import method_comparison_grid

    ALL_PLOTS = {"heatmap", "dot_matrix", "radar", "scatter", "runtime",
                 "rank", "umap_grid", "spatial_grid"}
    if plots is None:
        plots = list(ALL_PLOTS)
    invalid = set(plots) - ALL_PLOTS
    if invalid:
        raise ValueError(f"Unknown plots: {invalid}. Choose from: {ALL_PLOTS}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # 1. Collect evaluations + adata references
    print(f"\nEvaluating ({clustering} clustering) ...")
    dfs, adatas = _collect(data, clustering, label_key, dataset_name, slice_name)

    if not dfs:
        print("No results.")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    n_methods = df["Method"].nunique()
    n_slices = df[["Dataset", "Slice"]].drop_duplicates().shape[0]
    print(f"\n{n_methods} methods × {n_slices} slices → {len(df)} records")

    # 2. Split by GT
    has_gt_col = "GT" in df.columns
    if has_gt_col:
        df_gt = df[df["GT"] == True]
        df_nogt = df[df["GT"] == False]
    else:
        df_gt = df
        df_nogt = pd.DataFrame()

    # 3. Plot withGT group
    if not df_gt.empty:
        gt_metrics = ["SC_Score", "BioC_Score", "BER_Score"]
        _plot_group(df_gt, "withGT", "BioC_Score", gt_metrics, plots, save_dir,
                    is_gt=True, task_label="Vertical Integration")

    # 4. Plot woGT group
    if not df_nogt.empty:
        nogt_metrics = ["SC_Score", "BVC_Score", "BER_Score"]
        _plot_group(df_nogt, "woGT", "BVC_Score", nogt_metrics, plots, save_dir,
                    is_gt=False, task_label="Vertical Integration")

    # 5. Shared plots (all data)
    if "runtime" in plots and "Runtime" in df.columns and df["Runtime"].notna().any():
        print(f"\n{'─'*50}\n  Runtime (all methods)\n{'─'*50}")
        fig = runtime_bar(df)
        _save_fig(fig, save_dir, "runtime")

    # 6. UMAP / Spatial grids — merge adatas by (dataset, slice)
    want_umap = "umap_grid" in plots
    want_spatial = "spatial_grid" in plots
    if (want_umap or want_spatial) and adatas:
        # Group adatas by (dataset, slice), merge obsm into first adata
        from collections import OrderedDict
        groups: dict[tuple[str, str], AnnData] = OrderedDict()
        for adata, ds, sl in adatas:
            key = (ds, sl)
            if key not in groups:
                groups[key] = adata.copy()
            else:
                base = groups[key]
                n_obs = base.n_obs
                for k in adata.obsm.keys():
                    if k not in base.obsm:
                        val = adata.obsm[k]
                        if val.shape[0] == n_obs:
                            base.obsm[k] = val
                        else:
                            print(f"  Skipping {k} obsm: shape mismatch ({val.shape[0]} vs {n_obs})")
                for col in adata.obs.columns:
                    if col not in base.obs.columns and len(adata.obs) == n_obs:
                        base.obs[col] = adata.obs[col].values
                for uk in adata.uns:
                    if uk not in base.uns:
                        base.uns[uk] = adata.uns[uk]

        # Update methods_completed to reflect all merged obsm keys
        skip = {"spatial", "X_umap", "X_pca", "X_umap_orig", "feat"}
        for key, base in groups.items():
            all_methods = [k for k in base.obsm.keys()
                           if k not in skip and not k.startswith("X_")]
            base.uns["methods_completed"] = all_methods

        if want_umap and want_spatial:
            pt = "both"
        elif want_spatial:
            pt = "spatial"
        else:
            pt = "umap"

        for (ds, sl), merged_adata in groups.items():
            tag = f"{ds}_{sl}" if ds else "data"
            print(f"\n{'─'*50}\n  UMAP/Spatial grid: {tag}\n{'─'*50}")

            nc = _auto_n_clusters(ds, sl)
            fig = method_comparison_grid(
                merged_adata, plot_type=pt, clustering=clustering, n_clusters=nc,
                save=os.path.join(save_dir, f"grid_{tag}.pdf") if save_dir else None,
            )

    if save_dir:
        print(f"\nAll figures saved to {save_dir}/")

    return df


def plot_from_results(
    results_dir: str | Path,
    task: str = "vertical",
    clustering: str = "leiden",
    save_dir: str | None = None,
    plots: list[str] | None = None,
) -> pd.DataFrame:
    """Plot dot-matrix comparison directly from h5ad results directory.

    Loads metrics from h5ad files (via ``load_results``), splits by GT,
    and generates publication-quality dot-matrix tables.

    Parameters
    ----------
    results_dir : str or Path
        Root results directory (e.g. ``tutorials/results``).
    task : str
        ``"vertical"``, ``"horizontal"``, ``"3m"``, ``"image"``, or ``"all"``.
    clustering : str
        Which clustering results to use (filters by Clustering column).
    save_dir : str, optional
        Directory to save figures.
    plots : list[str], optional
        Which plots to generate. Default: ``["dot_matrix", "rank"]``.

    Returns
    -------
    pd.DataFrame
        Loaded results.
    """
    from smobench.io import load_results
    from smobench.plot.heatmap import dot_matrix, GROUPS_WITHGT, GROUPS_WOGT, GROUPS_WITHGT_BATCH, GROUPS_WOGT_BATCH

    if plots is None:
        plots = ["dot_matrix", "rank"]

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    df = load_results(results_dir, task=task)
    if df.empty:
        print("No results found.")
        return df

    # Filter by clustering
    if "Clustering" in df.columns:
        df = df[df["Clustering"] == clustering]

    # Determine GT status from dataset config
    from smobench._constants import DATASETS
    df["GT"] = df["Dataset"].map(lambda d: DATASETS.get(d, {}).get("gt", False))

    # Determine if task has batch metrics
    has_batch = task in ("horizontal", "mosaic")

    # Split by GT
    df_gt = df[df["GT"] == True]
    df_nogt = df[df["GT"] == False]

    # Pretty titles
    task_labels = {
        "vertical": "Vertical Integration",
        "horizontal": "Horizontal Integration",
        "mosaic": "Mosaic Integration",
        "image": "Image Integration",
        "3m": "Three-Modality Integration",
    }
    task_label = task_labels.get(task, task.title())

    if not df_gt.empty:
        _plot_group(df_gt, f"{task}_withGT", "BioC_Score",
                    ["SC_Score", "BioC_Score", "BER_Score"],
                    plots, save_dir, has_batch=has_batch,
                    is_gt=True, task_label=task_label)

    if not df_nogt.empty:
        _plot_group(df_nogt, f"{task}_woGT", "BVC_Score",
                    ["SC_Score", "BVC_Score", "BER_Score"],
                    plots, save_dir, has_batch=has_batch,
                    is_gt=False, task_label=task_label)

    # Runtime (shared)
    if "runtime" in plots and "Runtime" in df.columns and df["Runtime"].notna().any():
        from smobench.plot.scalability import runtime_bar
        fig = runtime_bar(df)
        _save_fig(fig, save_dir, f"runtime_{task}")

    return df
