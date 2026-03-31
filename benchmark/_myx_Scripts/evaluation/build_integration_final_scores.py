#!/usr/bin/env python3
"""
Build integration final-score summary split by modality (RNA_ADT vs RNA_ATAC).

Outputs:
  Results/summary_table/integration_final_scores.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WITH_GT_DATASETS = {
    "Human_Lymph_Nodes": "RNA_ADT",
    "Human_Tonsils": "RNA_ADT",
    "Mouse_Embryos_S1": "RNA_ATAC",
    "Mouse_Embryos_S2": "RNA_ATAC",
}


def average_or_nan(series: pd.Series) -> float:
    if series.dropna().empty:
        return np.nan
    return float(series.mean())


def load_vertical_split(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = root / "Results" / "evaluation" / "vertical_integration" / "final_results" / "detailed_results_leiden.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = df[(df["GT_Available"]) & (df["Dataset"].isin(WITH_GT_DATASETS.keys()))]
    df["Dataset_Type"] = df["Dataset"].map(WITH_GT_DATASETS)
    split = (
        df.groupby(["Method", "Dataset_Type"], as_index=False)["Total_Score"]
        .agg(Vertical_Final="mean")
    )
    overall = df.groupby("Method", as_index=False)["Total_Score"].agg(Vertical_Overall="mean")
    return split, overall


def load_horizontal_split(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = root / "Results" / "evaluation" / "horizontal_integration" / "final_results" / "leiden" / "detailed_results_leiden.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = df[(df["GT_Available"]) & (df["Dataset"].isin(WITH_GT_DATASETS.keys()))]
    df["Dataset_Type"] = df["Dataset"].map(WITH_GT_DATASETS)
    split = (
        df.groupby(["Method", "Dataset_Type"], as_index=False)["Final_Score"]
        .agg(Horizontal_Final="mean")
    )
    overall = df.groupby("Method", as_index=False)["Final_Score"].agg(Horizontal_Overall="mean")
    return split, overall


def load_mosaic(root: Path) -> pd.DataFrame:
    dir_path = root / "Results" / "evaluation" / "mosaic_integration"
    files = {
        "RNA_ADT": dir_path / "spamosaic_without_ADT_ATAC_scores.csv",
        "RNA_ATAC": dir_path / "spamosaic_without_RNA_scores.csv",
    }
    rows = []
    for dtype, file in files.items():
        if not file.exists():
            continue
        df = pd.read_csv(file)
        finals = df["Final"].dropna()
        if finals.empty:
            continue
        rows.append({"Method": "SpaMosaic", "Dataset_Type": dtype, "Mosaic_Final": float(finals.mean())})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Build integration final score summary table.")
    parser.add_argument(
        "--root",
        type=str,
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root directory.",
    )
    args = parser.parse_args()
    root = Path(args.root)

    vertical_split, vertical_overall = load_vertical_split(root)
    horizontal_split, horizontal_overall = load_horizontal_split(root)
    mosaic_df = load_mosaic(root)

    combined = pd.merge(vertical_split, horizontal_split, how="outer", on=["Method", "Dataset_Type"])
    combined = pd.merge(combined, mosaic_df, how="outer", on=["Method", "Dataset_Type"])

    if not combined.empty:
        all_methods = combined["Method"].drop_duplicates()
        all_types = pd.Series(["RNA_ADT", "RNA_ATAC"])
        extended = (
            all_methods.to_frame(name="Method")
            .assign(key=1)
            .merge(all_types.to_frame(name="Dataset_Type").assign(key=1), on="key")
            .drop(columns="key")
        )
        combined = extended.merge(combined, how="left", on=["Method", "Dataset_Type"])

    combined["Method_Type"] = combined["Method"] + "_" + combined["Dataset_Type"]

    combined = combined.merge(vertical_overall, how="left", on="Method").merge(horizontal_overall, how="left", on="Method")
    combined["Final_Score"] = combined[["Vertical_Overall", "Horizontal_Overall"]].mean(axis=1, skipna=True)
    combined["Overall"] = combined[["Vertical_Final", "Horizontal_Final"]].mean(axis=1, skipna=True)

    method_rank = (
        combined[["Method", "Final_Score"]]
        .drop_duplicates()
        .sort_values("Final_Score", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    method_rank["Rank"] = np.where(
        method_rank["Final_Score"].notna(),
        np.arange(1, len(method_rank) + 1),
        np.nan,
    )

    combined = combined.merge(method_rank[["Method", "Rank"]], on="Method", how="left")
    combined = combined.sort_values(["Rank", "Method", "Dataset_Type"], na_position="last")

    output_dir = root / "Results" / "summary_table"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "integration_final_scores.csv"
    combined.to_csv(output_path, index=False)
    print(f"Wrote integration final scores to {output_path}")


if __name__ == "__main__":
    main()
