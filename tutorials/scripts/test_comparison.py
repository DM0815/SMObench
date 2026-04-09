"""Test comparison plots: grouped_bar, method_profile, modality_comparison."""
import sys
sys.path.insert(0, "../../src")

import os
import matplotlib
matplotlib.use("Agg")

from smobench.plot.summary import plot_summary

SAVE_DIR = "../figures_test_comparison"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── 1. Load data: withGT (Tonsils S1) ──
print("=" * 60)
print("  Loading data: Human_Tonsils/S1 (withGT)")
print("=" * 60)
df_gt = plot_summary(
    "../results/vertical/Human_Tonsils/S1/",
    clustering="leiden",
    save_dir=None,
    plots=[],  # no plots, just get the DataFrame
)
print(f"Records: {len(df_gt)}, Methods: {sorted(df_gt['Method'].unique())}")
print(df_gt[['Method', 'Dataset', 'ARI', 'NMI', 'SC_Score', 'BioC_Score']].to_string())

# ── 2. Load data: woGT (Spleen + Thymus, multi-dataset) ──
print("\n" + "=" * 60)
print("  Loading data: Spleen + Thymus (woGT)")
print("=" * 60)
df_nogt = plot_summary(
    [
        "../results/vertical/Mouse_Spleen/Mouse_Spleen1/",
        "../results/vertical/Mouse_Spleen/Mouse_Spleen2/",
        "../results/vertical/Mouse_Thymus/Mouse_Thymus1/",
        "../results/vertical/Mouse_Thymus/Mouse_Thymus2/",
    ],
    clustering="leiden",
    save_dir=None,
    plots=[],
)
print(f"Records: {len(df_nogt)}, Methods: {sorted(df_nogt['Method'].unique())}")

# ── 3. Test grouped_bar ──
print("\n" + "=" * 60)
print("  Test: grouped_bar")
print("=" * 60)
from smobench.plot.comparison import grouped_bar, method_profile, modality_comparison

# All methods across datasets (woGT data has multiple datasets)
fig = grouped_bar(
    df_nogt, x="Method", y="BVC_Score", hue="Dataset",
    title="BVC_Score: Method x Dataset",
    save=os.path.join(SAVE_DIR, "grouped_bar_method_dataset.png"),
)

# ── 4. Test method_profile (single method, bar) ──
print("\n" + "=" * 60)
print("  Test: method_profile (SpatialGlue, bar)")
print("=" * 60)
fig = method_profile(
    df_gt, method="SpatialGlue",
    metrics=["ARI", "NMI", "cASW", "cLISI", "Moran_I"],
    group_by="Dataset",
    plot_type="bar",
    title="SpatialGlue: Metric Profile",
    save=os.path.join(SAVE_DIR, "profile_spatialglue_bar.png"),
)

# ── 5. Test method_profile (multiple methods, line) ──
print("\n" + "=" * 60)
print("  Test: method_profile (multi-method, line)")
print("=" * 60)
top_methods = df_nogt.groupby("Method")["BVC_Score"].mean().nlargest(5).index.tolist()
print(f"Top 5 methods: {top_methods}")
fig = method_profile(
    df_nogt, method=top_methods,
    metrics=["SC_Score", "BVC_Score"],
    group_by="Dataset_Slice",
    plot_type="line",
    title="Top 5 Methods: Trend across Slices",
    save=os.path.join(SAVE_DIR, "profile_multi_line.png"),
)

# ── 6. Test method_profile (radar) ──
print("\n" + "=" * 60)
print("  Test: method_profile (SpatialGlue, radar, woGT)")
print("=" * 60)
fig = method_profile(
    df_nogt, method="SpatialGlue",
    metrics=["Silhouette", "SC_Score", "BVC_Score"],
    group_by="Dataset",
    plot_type="radar",
    title="SpatialGlue: Cross-Dataset Radar",
    save=os.path.join(SAVE_DIR, "profile_spatialglue_radar.png"),
)

# ── 7. Test modality_comparison (use Dataset as proxy for modality) ──
print("\n" + "=" * 60)
print("  Test: modality_comparison (by Dataset)")
print("=" * 60)
fig = modality_comparison(
    df_nogt,
    metrics=["SC_Score", "BVC_Score"],
    group_col="Dataset",
    title="Performance by Dataset",
    save=os.path.join(SAVE_DIR, "modality_by_dataset.png"),
)

print("\n" + "=" * 60)
print(f"  All done! Figures in {SAVE_DIR}/")
print("=" * 60)
