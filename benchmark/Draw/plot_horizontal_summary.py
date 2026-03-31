"""
Horizontal integration summary plots (leiden clustering only).

This script reads evaluation outputs for horizontal integration and generates
heatmap-style summary tables similar to the vertical integration visualization.
Two figures are produced: RNA_ADT datasets and RNA_ATAC datasets.

Usage:
    conda run -n smobench bash -c "
      export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH;
      export NUMBA_DISABLE_JIT=1;
      python Draw/plot_horizontal_summary.py"
"""

from __future__ import annotations

import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure root path and Eval utilities are importable
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR / "Eval") not in sys.path:
    sys.path.append(str(ROOT_DIR / "Eval"))

from generate_final_results import normalize_metric_value  # type: ignore

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

EVAL_BASE = ROOT_DIR / "Results" / "evaluation" / "horizontal_integration"
PLOT_DIR = ROOT_DIR / "Results" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset metadata
DATASET_TYPES = {
    "HLN": "RNA_ADT",
    "HT": "RNA_ADT",
    "Mouse_Thymus": "RNA_ADT",
    "Mouse_Spleen": "RNA_ADT",
    "MISAR_S1": "RNA_ATAC",
    "MISAR_S2": "RNA_ATAC",
    "Mouse_Brain": "RNA_ATAC",
}

WITH_GT = {"HLN", "HT", "MISAR_S1", "MISAR_S2"}

# Methods compatibility (only datasets listed will be considered)
METHOD_DATASET_COMPATIBILITY = {
    "SpatialGlue": list(DATASET_TYPES.keys()),
    "SpaMosaic": list(DATASET_TYPES.keys()),
    "PRESENT": list(DATASET_TYPES.keys()),
    "COSMOS": list(DATASET_TYPES.keys()),
    "SpaMV": ["HLN", "HT", "MISAR_S1", "MISAR_S2", "Mouse_Thymus", "Mouse_Spleen"],
    "CANDIES": ["HLN", "HT", "MISAR_S1", "MISAR_S2", "Mouse_Spleen"],
    "PRAGA": ["HLN", "HT", "Mouse_Spleen"],
    "SpaMultiVAE": ["HLN", "HT", "Mouse_Thymus", "Mouse_Spleen"],
    "SpaFusion": ["HLN", "HT", "Mouse_Spleen"],
    "SpaBalance": ["HLN", "HT", "MISAR_S1", "MISAR_S2", "Mouse_Spleen"],
    "SpaMI": ["HLN", "HT", "Mouse_Spleen", "Mouse_Thymus"],
    "SMOPCA": list(DATASET_TYPES.keys()),
}

# Small metrics
SC_METRICS = ["Moran Index"]
BVC_METRICS_WITH_GT = ["ARI", "NMI", "asw_celltype", "graph_clisi"]
BVC_METRICS_WO_GT = ["Davies-Bouldin Index", "Silhouette Coefficient", "Calinski-Harabaz Index"]
BER_METRICS = ["kBET", "KNN_connectivity", "bASW", "iLISI", "PCR"]


def discover_methods() -> list[str]:
    methods = []
    for entry in EVAL_BASE.iterdir():
        if entry.is_dir() and entry.name != "final_results":
            # ensure there exists at least one leiden result
            has_leiden = any("horizontal_leiden" in f.name for f in entry.glob("*horizontal_leiden_*.csv"))
            if has_leiden:
                methods.append(entry.name)
    return sorted(methods)


def load_metrics(method: str, dataset: str) -> dict[str, float] | None:
    suffix = "withGT" if dataset in WITH_GT else "woGT"
    file_name = f"{method}_{dataset}_horizontal_leiden_{suffix}.csv"
    file_path = EVAL_BASE / method / file_name
    if not file_path.exists():
        return None
    df = pd.read_csv(file_path)
    if {"Metric", "Value"}.issubset(df.columns):
        return dict(zip(df["Metric"], df["Value"]))
    # fallback to row-based format
    return df.iloc[0].to_dict()


def collect_method_data(methods: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    data: dict[str, dict[str, dict[str, float]]] = {}
    for method in methods:
        datasets_allowed = METHOD_DATASET_COMPATIBILITY.get(method, list(DATASET_TYPES.keys()))
        method_results: dict[str, dict[str, float]] = {}
        for dataset in datasets_allowed:
            metrics = load_metrics(method, dataset)
            if metrics:
                method_results[dataset] = metrics
        if method_results:
            data[method] = method_results
    return data


def aggregate_by_dataset_type(
    method_data: dict[str, dict[str, dict[str, float]]],
    dataset_names: list[str],
) -> pd.DataFrame:
    rows = []
    for method, dataset_metrics in method_data.items():
        row: dict[str, float | str] = {"Method": method}
        sc_values: list[float] = []
        bvc_components_with_gt: defaultdict[str, list[float]] = defaultdict(list)
        bvc_components_wo_gt: defaultdict[str, list[float]] = defaultdict(list)
        ber_components: defaultdict[str, list[float]] = defaultdict(list)
        sc_scores: list[float] = []
        bvc_scores_ds: list[float] = []
        ber_scores_ds: list[float] = []
        final_scores_ds: list[float] = []

        for dataset in dataset_names:
            metrics = dataset_metrics.get(dataset)
            if not metrics:
                continue

            # SC metrics
            for metric in SC_METRICS:
                value = metrics.get(metric)
                if value is not None and not np.isnan(value):
                    sc_values.append(value)

            # BVC metrics
            if dataset in WITH_GT:
                for metric in BVC_METRICS_WITH_GT:
                    value = metrics.get(metric)
                    if value is not None and not np.isnan(value):
                        bvc_components_with_gt[metric].append(value)
            else:
                for metric in BVC_METRICS_WO_GT:
                    value = metrics.get(metric)
                    if value is not None and not np.isnan(value):
                        bvc_components_wo_gt[metric].append(value)

            # BER metrics
            for metric in BER_METRICS:
                value = metrics.get(metric)
                if value is not None and not np.isnan(value):
                    ber_components[metric].append(value)

            # Dataset-level scores
            for metric, store in [
                ("SC_Score", sc_scores),
                ("BVC_Score", bvc_scores_ds),
                ("BER_Score", ber_scores_ds),
                ("Final_Score", final_scores_ds),
            ]:
                value = metrics.get(metric)
                if value is not None and not np.isnan(value):
                    store.append(value)

        if not (sc_values or bvc_components_with_gt or bvc_components_wo_gt or ber_components):
            continue

        # Averaged small metrics
        if sc_values:
            row["Moran Index"] = float(np.mean(sc_values))

        for metric, values in bvc_components_with_gt.items():
            row[metric] = float(np.mean(values))
        for metric, values in bvc_components_wo_gt.items():
            row[metric] = float(np.mean(values))

        for metric, values in ber_components.items():
            row[metric] = float(np.mean(values))

        # Aggregate scores (averaged across datasets)
        if sc_values:
            row["SC_Score"] = float(np.mean(sc_values))
        if bvc_scores_ds:
            row["BVC_Score"] = float(np.mean(bvc_scores_ds))
        if ber_scores_ds:
            row["BER_Score"] = float(np.mean(ber_scores_ds))
        if final_scores_ds:
            row["Final_Score"] = float(np.mean(final_scores_ds))

        rows.append(row)

    df = pd.DataFrame(rows).set_index("Method")

    if df.empty:
        return df

    # Normalize DBI and CHI
    if "Davies-Bouldin Index" in df.columns:
        valid = df["Davies-Bouldin Index"].dropna().tolist()
        if valid:
            df["DBI (norm)"] = df["Davies-Bouldin Index"].apply(
                lambda v: normalize_metric_value(v, "Davies-Bouldin Index", valid) if pd.notna(v) else np.nan
            )
        df.drop(columns=["Davies-Bouldin Index"], inplace=True)

    if "Calinski-Harabaz Index" in df.columns:
        valid = df["Calinski-Harabaz Index"].dropna().tolist()
        if valid:
            df["CHI (norm)"] = df["Calinski-Harabaz Index"].apply(
                lambda v: normalize_metric_value(v, "Calinski-Harabaz Index", valid) if pd.notna(v) else np.nan
            )
        df.drop(columns=["Calinski-Harabaz Index"], inplace=True)

    # Recompute BVC_Score using available components
    bvc_component_cols = [col for col in df.columns if col in BVC_METRICS_WITH_GT or col in ["Silhouette Coefficient", "DBI (norm)", "CHI (norm)"]]
    if bvc_component_cols:
        df["BVC_Score"] = df[bvc_component_cols].mean(axis=1, skipna=True)

    # BER score from components if missing
    ber_component_cols = [col for col in df.columns if col in BER_METRICS]
    if ber_component_cols:
        df["BER_Score"] = df[ber_component_cols].mean(axis=1, skipna=True)

    # SC score equals Moran Index
    if "Moran Index" in df.columns:
        df["SC_Score"] = df["Moran Index"]

    # Final score: mean of three scores
    score_cols = ["SC_Score", "BVC_Score", "BER_Score"]
    df["Final_Score"] = df[score_cols].mean(axis=1, skipna=True)

    # Sort by final score
    if "Final_Score" in df.columns:
        df.sort_values("Final_Score", ascending=False, inplace=True)

    return df


def normalize_for_plot(values: pd.DataFrame) -> np.ndarray:
    arr = values.to_numpy(dtype=float)
    norm_arr = np.full_like(arr, np.nan)
    for col_idx in range(arr.shape[1]):
        col = arr[:, col_idx]
        mask = ~np.isnan(col)
        if not mask.any():
            continue
        col_valid = col[mask]
        # ensure we normalize starting from zero regardless of observed minimum
        clipped = np.clip(col_valid, a_min=0.0, a_max=None)
        vmax = clipped.max()
        if np.isclose(vmax, 0):
            norm_arr[mask, col_idx] = 0.0
        else:
            norm_arr[mask, col_idx] = clipped / vmax
    return norm_arr


def plot_summary_table(df: pd.DataFrame, dataset_type: str) -> None:
    if df.empty:
        print(f"No data available for {dataset_type}, skipping plot.")
        return

    column_groups = [
        ("SC", [col for col in ["Moran Index"] if col in df.columns]),
        ("BioC (GT)", [c for c in BVC_METRICS_WITH_GT if c in df.columns]),
        ("BioC (woGT)", [c for c in ["DBI (norm)", "Silhouette Coefficient", "CHI (norm)"] if c in df.columns]),
        ("BER", [c for c in BER_METRICS if c in df.columns]),
        ("Aggregate", [c for c in ["SC_Score", "BVC_Score", "BER_Score", "Final_Score"] if c in df.columns]),
    ]

    columns = [col for _, cols in column_groups for col in cols]
    df = df[columns]

    norm_values = normalize_for_plot(df)
    masked = np.ma.masked_invalid(norm_values)

    fig_height = max(4, 0.6 * len(df.index))
    fig_width = max(10, 0.7 * len(columns))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    import matplotlib as mpl
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "custom_heatmap",
        ["#b3cde3", "#8c96c6", "#ffeda0", "#f03b20"],
        N=256,
    )
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    # Overlay text values
    for i, method in enumerate(df.index):
        for j, col in enumerate(columns):
            value = df.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="black", fontsize=9)
            else:
                ax.text(j, i, "-", ha="center", va="center", color="#555555", fontsize=9)

    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=11)

    # Draw grid lines
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(df.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Group labels
    x_positions = []
    for label, cols in column_groups:
        if not cols:
            continue
        start = columns.index(cols[0])
        end = columns.index(cols[-1])
        x_positions.append((label, (start + end) / 2.0, len(cols)))
    for label, xpos, width in x_positions:
        ax.text(xpos, -0.9, label, ha="center", va="center", fontsize=12, fontweight="bold")

    ax.set_title(f"Horizontal Integration Summary ({dataset_type})", fontsize=14, pad=20)
    plt.tight_layout()

    output_path = PLOT_DIR / f"horizontal_summary_{dataset_type}.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved horizontal summary figure: {output_path}")


def main():
    methods = discover_methods()
    if not methods:
        print("No horizontal integration evaluation results found.")
        return

    method_data = collect_method_data(methods)

    dataset_groups = {
        "RNA_ADT": ["HLN", "HT", "Mouse_Thymus", "Mouse_Spleen"],
        "RNA_ATAC": ["MISAR_S1", "MISAR_S2", "Mouse_Brain"],
    }

    for dataset_type, datasets in dataset_groups.items():
        df = aggregate_by_dataset_type(method_data, datasets)
        plot_summary_table(df, dataset_type)


if __name__ == "__main__":
    main()
