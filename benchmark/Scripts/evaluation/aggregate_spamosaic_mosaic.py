#!/usr/bin/env python3
"""
Aggregate SpaMosaic mosaic-integration metrics from mosaic-results.xlsx.

The workbook has two sheets:
  - "wo ADT or ATAC"
  - "without RNA"

Each sheet contains per-subset metrics. This script groups subsets by dataset
(e.g., HLN aggregates A1/D1) and computes the mean for each metric, while
collecting the subset names. Two CSV files are written under
Results/evaluation/mosaic_integration/.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WORKBOOK = ROOT / "mosaic-results.xlsx"
OUTPUT_DIR = ROOT / "Results" / "evaluation" / "mosaic_integration"

COLUMN_MAP = {
    "Unnamed: 0": "Dataset",
    "Unnamed: 1": "Subset",
    "Moran Index": "Moran Index",
    "ARI": "ARI",
    "NMI": "NMI",
    "asw_celltype": "asw_celltype",
    "graph_clisi": "graph_clisi",
    "Silhouette Coefficient": "Silhouette Coefficient",
    "Calinski-Harabaz Index": "Calinski-Harabaz Index",
    "Davies-Bouldin Index": "Davies-Bouldin Index",
    "kBET": "kBET",
    "KNN_connectivity": "KNN_connectivity",
    "bASW": "bASW",
    "iLISI": "iLISI",
    "PCR": "PCR",
}


def aggregate_sheet(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COLUMN_MAP)
    if "Dataset" not in df or "Subset" not in df:
        raise ValueError("Sheet does not contain expected Dataset/Subset columns.")

    df["Dataset"] = df["Dataset"].ffill()
    df = df.dropna(subset=["Dataset", "Subset"])

    def assign_dataset_group(row):
        dataset = str(row["Dataset"])
        subset = str(row["Subset"])
        if dataset.upper() == "MISAR":
            if "_S1" in subset:
                return "MISAR_S1"
            if "_S2" in subset:
                return "MISAR_S2"
        return dataset

    df["Dataset_Group"] = df.apply(assign_dataset_group, axis=1)

    metrics = [c for c in df.columns if c not in {"Dataset", "Dataset_Group", "Subset"}]

    agg_dict: Dict[str, List[str] | str] = {
        "Subset": lambda s: ",".join(str(x) for x in pd.unique(s.dropna()))
    }
    for metric in metrics:
        agg_dict[metric] = "mean"

    grouped = (
        df.groupby("Dataset_Group")
        .agg(agg_dict)
        .reset_index()
        .rename(columns={"Dataset_Group": "Dataset", "Subset": "Subsets"})
    )
    return grouped


def main():
    parser = argparse.ArgumentParser(description="Aggregate SpaMosaic mosaic results into dataset-level CSVs.")
    parser.add_argument(
        "--workbook",
        type=str,
        default=str(WORKBOOK),
        help="Path to mosaic-results.xlsx",
    )
    args = parser.parse_args()

    workbook = Path(args.workbook)
    if not workbook.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook}")

    xls = pd.ExcelFile(workbook)
    sheets = {
        "wo ADT or ATAC": "spamosaic_without_ADT_ATAC.csv",
        "without RNA": "spamosaic_without_RNA.csv",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for sheet, filename in sheets.items():
        if sheet not in xls.sheet_names:
            print(f"[Skip] Sheet '{sheet}' not found in workbook.")
            continue
        df = pd.read_excel(xls, sheet_name=sheet)
        agg_df = aggregate_sheet(df)
        output_path = OUTPUT_DIR / filename
        agg_df.to_csv(output_path, index=False)
        print(f"[OK] Wrote {output_path}")


if __name__ == "__main__":
    main()
