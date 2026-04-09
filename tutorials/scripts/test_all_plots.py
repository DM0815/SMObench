"""Test all plot functions with full backfilled data."""
import sys
sys.path.insert(0, "../../src")

import os
import matplotlib
matplotlib.use("Agg")

from smobench.io import load_results
from smobench.plot.summary import plot_from_results
from smobench.plot.comparison import grouped_bar, method_profile, modality_comparison
from smobench.plot.heatmap import heatmap
from smobench.plot.radar import radar
from smobench.plot.scatter import sc_vs_bioc
from smobench.plot.scalability import runtime_bar

SAVE = "../figures_full"
os.makedirs(SAVE, exist_ok=True)

# Load all data
df = load_results("../results", task="vertical")
df = df[df["Clustering"] == "leiden"] if "Clustering" in df.columns else df
print(f"\nLoaded: {len(df)} records, {df['Method'].nunique()} methods, {df['Dataset'].nunique()} datasets")

# Add GT column
from smobench._constants import DATASETS
df["GT"] = df["Dataset"].map(lambda d: DATASETS.get(d, {}).get("gt", False))
df_gt = df[df["GT"] == True]
df_nogt = df[df["GT"] == False]

print(f"withGT: {len(df_gt)} records ({df_gt['Method'].nunique()} methods)")
print(f"woGT:   {len(df_nogt)} records ({df_nogt['Method'].nunique()} methods)")

# ── 1. plot_from_results (dot_matrix + rank) ──
print("\n" + "=" * 60)
print("  1. plot_from_results (dot_matrix + rank)")
print("=" * 60)
plot_from_results(
    "../results", task="vertical", clustering="leiden",
    save_dir=os.path.join(SAVE, "summary"),
    plots=["dot_matrix", "rank"],
)

# ── 2. Heatmap: BioC_Score across methods × datasets ──
print("\n" + "=" * 60)
print("  2. Heatmap (BioC_Score, withGT)")
print("=" * 60)
# Combine Dataset+Slice for column
df_gt2 = df_gt.copy()
df_gt2["Dataset_Slice"] = df_gt2["Dataset"] + "/" + df_gt2["Slice"]
fig = heatmap(df_gt2, score_col="BioC_Score", col="Dataset_Slice",
              title="Vertical Integration (with Ground Truth): BioC_Score",
              save=os.path.join(SAVE, "heatmap_bioc_gt.png"))

# ── 3. Radar (withGT, all methods) ──
print("\n" + "=" * 60)
print("  3. Radar (withGT)")
print("=" * 60)
fig = radar(df_gt, metrics=["ARI", "NMI", "cASW", "cLISI", "Moran_I"],
            title="Vertical (with GT): Method Radar",
            save=os.path.join(SAVE, "radar_gt.png"))

# ── 4. Scatter (SC vs BioC) ──
print("\n" + "=" * 60)
print("  4. Scatter (SC vs BioC, withGT)")
print("=" * 60)
fig = sc_vs_bioc(df_gt, title="SC_Score vs BioC_Score (with GT)",
                 save=os.path.join(SAVE, "scatter_sc_bioc.png"))

# ── 5. Runtime bar ──
print("\n" + "=" * 60)
print("  5. Runtime bar")
print("=" * 60)
fig = runtime_bar(df, save=os.path.join(SAVE, "runtime.png"))

# ── 6. grouped_bar: Method × Dataset ──
print("\n" + "=" * 60)
print("  6. grouped_bar (BioC_Score by Method × Dataset, withGT)")
print("=" * 60)
fig = grouped_bar(df_gt, x="Method", y="BioC_Score", hue="Dataset",
                  title="BioC_Score: Method × Dataset (with GT)",
                  save=os.path.join(SAVE, "grouped_bar_bioc_dataset.png"))

# ── 7. grouped_bar: woGT, BVC by Dataset ──
print("\n" + "=" * 60)
print("  7. grouped_bar (BVC_Score, woGT)")
print("=" * 60)
fig = grouped_bar(df_nogt, x="Method", y="BVC_Score", hue="Dataset",
                  title="BVC_Score: Method × Dataset (without GT)",
                  save=os.path.join(SAVE, "grouped_bar_bvc_nogt.png"))

# ── 8. method_profile: single method bar ──
print("\n" + "=" * 60)
print("  8. method_profile (SpatialGlue, bar, withGT)")
print("=" * 60)
fig = method_profile(df_gt, method="SpatialGlue",
                     metrics=["ARI", "NMI", "cASW", "cLISI", "Moran_I"],
                     group_by="Dataset",
                     title="SpatialGlue: Performance across Datasets",
                     save=os.path.join(SAVE, "profile_spatialglue_bar.png"))

# ── 9. method_profile: multi-method line ──
print("\n" + "=" * 60)
print("  9. method_profile (top5, line, woGT)")
print("=" * 60)
top5 = df_nogt.groupby("Method")["BVC_Score"].mean().nlargest(5).index.tolist()
print(f"  Top 5: {top5}")
fig = method_profile(df_nogt, method=top5,
                     metrics=["SC_Score", "BVC_Score"],
                     group_by="Dataset_Slice",
                     plot_type="line",
                     title="Top 5 Methods: Trend (without GT)",
                     save=os.path.join(SAVE, "profile_top5_line.png"))

# ── 10. method_profile: radar ──
print("\n" + "=" * 60)
print("  10. method_profile (SpatialGlue, radar, across datasets)")
print("=" * 60)
fig = method_profile(df_gt, method="SpatialGlue",
                     metrics=["ARI", "NMI", "cASW", "cLISI", "Moran_I"],
                     group_by="Dataset",
                     plot_type="radar",
                     title="SpatialGlue: Cross-Dataset Radar",
                     save=os.path.join(SAVE, "profile_spatialglue_radar.png"))

# ── 11. modality_comparison (by Dataset) ──
print("\n" + "=" * 60)
print("  11. modality_comparison (by Dataset, woGT)")
print("=" * 60)
fig = modality_comparison(df_nogt,
                          metrics=["SC_Score", "BVC_Score"],
                          group_col="Dataset",
                          title="Performance by Dataset (without GT)",
                          save=os.path.join(SAVE, "modality_by_dataset.png"))

# ── 12. modality_comparison (by Dataset, withGT) ──
print("\n" + "=" * 60)
print("  12. modality_comparison (by Dataset, withGT)")
print("=" * 60)
fig = modality_comparison(df_gt,
                          metrics=["SC_Score", "BioC_Score"],
                          group_col="Dataset",
                          title="Performance by Dataset (with GT)",
                          save=os.path.join(SAVE, "modality_by_dataset_gt.png"))

print("\n" + "=" * 60)
print(f"  All done! {len(os.listdir(SAVE))} figures in {SAVE}/")
print("=" * 60)
