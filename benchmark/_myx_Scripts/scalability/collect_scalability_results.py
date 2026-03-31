#!/usr/bin/env python3
"""
Collect all scalability results and generate summary CSV + plots.
Produces figures similar to Extended Data Fig. 5/6 in [3] Hu et al.
"""

import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "_myx_Results/scalability/results")
OUT_DIR = os.path.join(PROJECT_ROOT, "_myx_Results/scalability")


def collect_results(results_dir):
    """Collect all scalability JSON results into a DataFrame."""
    records = []
    for jf in glob.glob(os.path.join(results_dir, "**/*_scalability.json"), recursive=True):
        with open(jf) as f:
            data = json.load(f)
        records.append(data)

    if not records:
        print("No results found!")
        return None

    df = pd.DataFrame(records)
    df = df[df["success"] == True]  # only successful runs
    print(f"Collected {len(df)} successful results from {len(records)} total")
    return df


def compute_summary(df):
    """Compute mean and std across repeats for time and memory."""
    time_col = "training_time_s" if "training_time_s" in df.columns else "wall_time_s"

    agg_dict = {
        "time_mean": (time_col, "mean"),
        "time_std": (time_col, "std"),
        "n_runs": (time_col, "count"),
    }

    # Add memory columns if available
    if "peak_gpu_memory_mb" in df.columns:
        agg_dict["gpu_mem_mean"] = ("peak_gpu_memory_mb", "mean")
        agg_dict["gpu_mem_std"] = ("peak_gpu_memory_mb", "std")
    if "peak_system_rss_mb" in df.columns:
        agg_dict["sys_mem_mean"] = ("peak_system_rss_mb", "mean")
        agg_dict["sys_mem_std"] = ("peak_system_rss_mb", "std")

    summary = df.groupby(["method", "modality", "n_cells"]).agg(**agg_dict).reset_index()

    return summary


def plot_scalability(summary, modality, out_path):
    """Plot time vs cell_count for all methods (one modality)."""
    sub = summary[summary["modality"] == modality].copy()
    if sub.empty:
        print(f"No data for {modality}")
        return

    methods = sorted(sub["method"].unique())

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for method in methods:
        mdata = sub[sub["method"] == method].sort_values("n_cells")
        ax.errorbar(
            mdata["n_cells"], mdata["time_mean"],
            yerr=mdata["time_std"],
            marker='o', label=method, capsize=3, linewidth=1.5
        )

    ax.set_xlabel("Number of cells", fontsize=12)
    ax.set_ylabel("Computational time (seconds)", fontsize=12)
    ax.set_title(f"Scalability Analysis - {modality}", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {out_path}")


def plot_memory(summary, modality, mean_col, std_col, ylabel, out_path):
    """Plot memory vs cell_count for all methods (one modality)."""
    sub = summary[summary["modality"] == modality].copy()
    if sub.empty or mean_col not in sub.columns:
        return

    methods = sorted(sub["method"].unique())
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for method in methods:
        mdata = sub[sub["method"] == method].sort_values("n_cells")
        ax.errorbar(
            mdata["n_cells"], mdata[mean_col],
            yerr=mdata[std_col],
            marker='o', label=method, capsize=3, linewidth=1.5
        )

    ax.set_xlabel("Number of cells", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"Memory Scalability - {modality}", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Memory plot saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    args = parser.parse_args()

    # Collect
    df = collect_results(args.results_dir)
    if df is None:
        return

    # Save raw
    raw_csv = os.path.join(args.out_dir, "scalability_raw.csv")
    df.to_csv(raw_csv, index=False)
    print(f"Raw results saved: {raw_csv}")

    # Summary
    summary = compute_summary(df)
    summary_csv = os.path.join(args.out_dir, "scalability_summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"Summary saved: {summary_csv}")

    # Print table
    print("\n" + "=" * 80)
    print("SCALABILITY SUMMARY")
    print("=" * 80)

    for modality in ["RNA_ADT", "RNA_ATAC"]:
        sub = summary[summary["modality"] == modality]
        if sub.empty:
            continue
        print(f"\n--- {modality} ---")
        pivot = sub.pivot_table(index="method", columns="n_cells", values="time_mean")
        print(pivot.round(2).to_string())

    # Plots - Time
    for modality in ["RNA_ADT", "RNA_ATAC"]:
        plot_path = os.path.join(args.out_dir, f"scalability_time_{modality}.png")
        plot_scalability(summary, modality, plot_path)

    # Plots - GPU Memory
    if "gpu_mem_mean" in summary.columns:
        for modality in ["RNA_ADT", "RNA_ATAC"]:
            plot_path = os.path.join(args.out_dir, f"scalability_gpu_memory_{modality}.png")
            plot_memory(summary, modality, "gpu_mem_mean", "gpu_mem_std",
                       "Peak GPU Memory (MB)", plot_path)

    # Plots - System Memory
    if "sys_mem_mean" in summary.columns:
        for modality in ["RNA_ADT", "RNA_ATAC"]:
            plot_path = os.path.join(args.out_dir, f"scalability_sys_memory_{modality}.png")
            plot_memory(summary, modality, "sys_mem_mean", "sys_mem_std",
                       "Peak System RSS (MB)", plot_path)


if __name__ == "__main__":
    main()
