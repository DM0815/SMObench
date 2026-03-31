"""
Vertical integration summary plots (leiden clustering only).

This script aggregates evaluation metrics from
Results/evaluation/vertical_integration/final_results/detailed_results_leiden.csv
and produces heatmap-style summary tables, mirroring the horizontal integration
visualizations while accounting for the vertical integration metric set.

Usage:
    conda run -n smobench bash -c "
      export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH;
      export NUMBA_DISABLE_JIT=1;
      python Draw/plot_vertical_summary.py"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure root/Eval utilities are importable
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR / "Eval") not in sys.path:
    sys.path.append(str(ROOT_DIR / "Eval"))

from generate_final_results import normalize_metric_value  # type: ignore

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

DETAILED_RESULTS_PATH = (
    ROOT_DIR
    / "Results"
    / "evaluation"
    / "vertical_integration"
    / "final_results"
    / "detailed_results_leiden.csv"
)
PLOT_DIR = ROOT_DIR / "Results" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset metadata
DATASET_TYPES: Dict[str, str] = {
    "HLN": "RNA_ADT",
    "HT": "RNA_ADT",
    "Mouse_Thymus": "RNA_ADT",
    "Mouse_Spleen": "RNA_ADT",
    "MISAR_S1": "RNA_ATAC",
    "MISAR_S2": "RNA_ATAC",
    "Mouse_Brain": "RNA_ATAC",
}

WITH_GT = {"HLN", "HT", "MISAR_S1", "MISAR_S2"}

# Metric groups
SC_METRICS = ["Moran Index"]
BIOC_METRICS_WITH_GT = ["ARI", "NMI", "asw_celltype", "graph_clisi"]
BIOC_METRICS_WO_GT = ["Silhouette Coefficient", "DBI (norm)", "CHI (norm)"]
AGGREGATE_COLS = ["SC_Score", "BioC_Score", "Final_Score"]


def load_detailed_results() -> pd.DataFrame:
    if not DETAILED_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Detailed results file not found: {DETAILED_RESULTS_PATH}. "
            "Please run Eval/eval_vertical_integration.py first."
        )

    df = pd.read_csv(DETAILED_RESULTS_PATH)
    if df.empty:
        raise ValueError("Detailed results file is empty.")

    # Ensure we only use leiden clustering records
    df = df[df["Clustering"].str.lower() == "leiden"].copy()
    if df.empty:
        raise ValueError("No leiden clustering entries found in detailed results.")

    return df


def normalize_column(values: pd.Series, metric_name: str) -> pd.Series:
    valid = values.dropna().tolist()
    if not valid:
        return pd.Series(np.nan, index=values.index)
    return values.apply(
        lambda v: normalize_metric_value(v, metric_name, valid) if pd.notna(v) else np.nan
    )


def aggregate_by_dataset_type(df: pd.DataFrame, datasets: Iterable[str]) -> pd.DataFrame:
    subset = df[df["Dataset"].isin(datasets)].copy()
    if subset.empty:
        return pd.DataFrame(columns=["Method"])

    # Prepare normalized variants for DBI / CHI
    subset["DBI (norm)"] = normalize_column(subset["Davies-Bouldin Index"], "Davies-Bouldin Index")
    subset["CHI (norm)"] = normalize_column(subset["Calinski-Harabaz Index"], "Calinski-Harabaz Index")

    rows: List[Dict[str, float]] = []
    for method, method_df in subset.groupby("Method"):
        row: Dict[str, float] = {"Method": method}

        # Spatial consistency metrics
        for metric in SC_METRICS:
            values = method_df[metric].dropna()
            if not values.empty:
                row[metric] = float(values.mean())

        # BioC metrics (with GT)
        gt_df = method_df[method_df["GT_Available"]]
        for metric in BIOC_METRICS_WITH_GT:
            if metric in gt_df:
                values = gt_df[metric].dropna()
                if not values.empty:
                    row[metric] = float(values.mean())

        # BioC metrics (without GT)
        wo_gt_df = method_df[~method_df["GT_Available"]]
        for metric in BIOC_METRICS_WO_GT:
            if metric in wo_gt_df:
                values = wo_gt_df[metric].dropna()
                if not values.empty:
                    row[metric] = float(values.mean())

        # Aggregate scores
        sc_scores = method_df["SC_Score"].dropna()
        if not sc_scores.empty:
            row["SC_Score"] = float(sc_scores.mean())

        bioc_scores = method_df["BioC_Score"].dropna()
        if not bioc_scores.empty:
            row["BioC_Score"] = float(bioc_scores.mean())

        total_scores = method_df["Total_Score"].dropna()
        if not total_scores.empty:
            row["Final_Score"] = float(total_scores.mean())

        if len(row) > 1:  # ensure we captured at least one metric
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["Method"])

    result = pd.DataFrame(rows).set_index("Method")
    if "Final_Score" in result.columns:
        result.sort_values("Final_Score", ascending=False, inplace=True)
    return result


def normalize_for_plot(values: pd.DataFrame) -> np.ndarray:
    arr = values.to_numpy(dtype=float)
    norm_arr = np.full_like(arr, np.nan)
    for col_idx in range(arr.shape[1]):
        col = arr[:, col_idx]
        mask = ~np.isnan(col)
        if not mask.any():
            continue
        col_valid = col[mask]
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
        ("SC", [col for col in SC_METRICS if col in df.columns]),
        ("BioC (GT)", [col for col in BIOC_METRICS_WITH_GT if col in df.columns]),
        ("BioC (woGT)", [col for col in BIOC_METRICS_WO_GT if col in df.columns]),
        ("Aggregate", [col for col in AGGREGATE_COLS if col in df.columns]),
    ]

    columns = [col for _, cols in column_groups for col in cols]
    df = df[columns]

    norm_values = normalize_for_plot(df)
    masked = np.ma.masked_invalid(norm_values)

    fig_height = max(4, 0.6 * len(df.index))
    fig_width = max(10, 0.7 * len(columns))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "vertical_heatmap",
        ["#b3cde3", "#8c96c6", "#ffeda0", "#f03b20"],
        N=256,
    )
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=1)

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

    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(df.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    x_positions = []
    for label, cols in column_groups:
        if not cols:
            continue
        start = columns.index(cols[0])
        end = columns.index(cols[-1])
        x_positions.append((label, (start + end) / 2.0))
    for label, xpos in x_positions:
        ax.text(xpos, -0.9, label, ha="center", va="center", fontsize=12, fontweight="bold")

    ax.set_title(f"Vertical Integration Summary ({dataset_type})", fontsize=14, pad=20)
    plt.tight_layout()

    output_path = PLOT_DIR / f"vertical_summary_{dataset_type}.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved vertical summary figure: {output_path}")


def main() -> None:
    df = load_detailed_results()

    dataset_groups = {
        "RNA_ADT": ["HLN", "HT", "Mouse_Spleen", "Mouse_Thymus"],
        "RNA_ATAC": ["MISAR_S1", "MISAR_S2", "Mouse_Brain"],
    }

    for dataset_type, datasets in dataset_groups.items():
        summary_df = aggregate_by_dataset_type(df, datasets)
        if summary_df.empty:
            print(f"No summary generated for {dataset_type} datasets.")
            continue
        plot_summary_table(summary_df, dataset_type)


if __name__ == "__main__":
    main()
