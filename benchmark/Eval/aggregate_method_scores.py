#!/usr/bin/env python3
"""
Aggregate method-level scores for vertical and horizontal integration.

For every method, compute the average Spatial Coherence / Bio Conservation /
Final Score (vertical) and Spatial Coherence / Bio Conservation / Batch Effect
Removal / Final Score (horizontal) across all datasets it supports.

Output: Results/summary_table/method_level_scores.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def safe_mean(series: pd.Series) -> float | None:
    values = series.dropna()
    return values.mean() if len(values) else None


def aggregate_vertical(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["Method", "SC", "BioC", "Final"])
    return (
        df.groupby("Method")
        .agg(
            SC=("SC_Score", safe_mean),
            BioC=("BioC_Score", safe_mean),
            Final=("Total_Score", safe_mean),
        )
        .reset_index()
    )


def aggregate_horizontal(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["Method", "SC", "BioC", "BER", "Final"])
    groups = df.groupby("Method")
    rows = []
    for method, group in groups:
        sc = safe_mean(group["SC_Score"])
        bio = safe_mean(group["BVC_Score"])
        final = safe_mean(group["Final_Score"])
        # Derive BER from definition Final = mean(SC, BioC, BER)
        ber = None
        if sc is not None and bio is not None and final is not None:
            ber = 3 * final - sc - bio
        rows.append({"Method": method, "SC": sc, "BioC": bio, "BER": ber, "Final": final})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="SMOBench root directory (default: script parent).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: Results/summary_table/method_level_scores.csv).",
    )
    args = parser.parse_args()

    root = args.root
    vertical_path = root / "Results" / "evaluation" / "vertical_integration" / "final_results" / "detailed_results_leiden.csv"
    horizontal_path = root / "Results" / "evaluation" / "horizontal_integration" / "final_results" / "leiden" / "detailed_results_leiden.csv"

    if not vertical_path.exists():
        raise FileNotFoundError(f"Vertical detailed results not found: {vertical_path}")
    if not horizontal_path.exists():
        raise FileNotFoundError(f"Horizontal detailed results not found: {horizontal_path}")

    print(f"Loading vertical results from {vertical_path}")
    vert_df = aggregate_vertical(vertical_path)
    print(f"Loaded {len(vert_df)} methods (vertical).")

    print(f"Loading horizontal results from {horizontal_path}")
    hori_df = aggregate_horizontal(horizontal_path)
    print(f"Loaded {len(hori_df)} methods (horizontal).")

    combined = pd.merge(
        vert_df,
        hori_df,
        on="Method",
        how="outer",
        suffixes=("_Vertical", "_Horizontal"),
    ).sort_values("Method")

    # Normalize column names and preserve order
    def get_col(df, name, out_name):
        return df[name] if name in df else None

    combined = combined.rename(
        columns={
            "SC_Vertical": "SC_Vertical",
            "BioC_Vertical": "BioC_Vertical",
            "Final_Vertical": "Final_Vertical",
            "SC_Horizontal": "SC_Horizontal",
            "BioC_Horizontal": "BioC_Horizontal",
            "BER": "BER_Horizontal",
            "BER_Horizontal": "BER_Horizontal",
            "Final_Horizontal": "Final_Horizontal",
        }
    )

    for col in [
        "SC_Vertical",
        "BioC_Vertical",
        "Final_Vertical",
        "SC_Horizontal",
        "BioC_Horizontal",
        "BER_Horizontal",
        "Final_Horizontal",
    ]:
        if col not in combined.columns:
            combined[col] = None

    output_path = (
        args.output
        if args.output
        else root / "Results" / "summary_table" / "method_level_scores.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Saved aggregate method scores to: {output_path}")


if __name__ == "__main__":
    main()
