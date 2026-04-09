"""Full plotting test: dot_matrix + heatmap + radar + scatter + ranking + runtime.

Uses Human_Tonsils/S1 (4326 cells, 11 methods, withGT).
"""
import sys
sys.path.insert(0, "../../src")

import matplotlib
matplotlib.use("Agg")

from smobench.plot.summary import plot_summary

# Single dataset — title should show dataset name
print("=" * 60)
print("  Test 1: Single dataset (Human_Tonsils/S1)")
print("=" * 60)
df = plot_summary(
    "../results/vertical/Human_Tonsils/S1/",
    clustering="leiden",
    save_dir="../figures_test_full/single/",
    plots=["dot_matrix", "heatmap", "radar", "scatter", "rank", "runtime"],
)
print(f"\nRecords: {len(df)}, Methods: {sorted(df['Method'].unique())}")

# Multi-dataset withGT — aggregated comparison
print("\n" + "=" * 60)
print("  Test 2: Multi-dataset withGT (Tonsils + Lymph Nodes)")
print("=" * 60)
df2 = plot_summary(
    [
        "../results/vertical/Human_Tonsils/S1/",
        "../results/vertical/Human_Tonsils/S2/",
        "../results/vertical/Human_Lymph_Nodes/A1/",
        "../results/vertical/Human_Lymph_Nodes/D1/",
    ],
    clustering="leiden",
    save_dir="../figures_test_full/multi_gt/",
    plots=["dot_matrix", "heatmap", "rank"],
)
print(f"\nRecords: {len(df2)}, Methods: {sorted(df2['Method'].unique())}")

# Multi-dataset woGT
print("\n" + "=" * 60)
print("  Test 3: Multi-dataset woGT (Spleen + Thymus)")
print("=" * 60)
df3 = plot_summary(
    [
        "../results/vertical/Mouse_Spleen/Mouse_Spleen1/",
        "../results/vertical/Mouse_Spleen/Mouse_Spleen2/",
        "../results/vertical/Mouse_Thymus/Mouse_Thymus1/",
        "../results/vertical/Mouse_Thymus/Mouse_Thymus2/",
    ],
    clustering="leiden",
    save_dir="../figures_test_full/multi_nogt/",
    plots=["dot_matrix", "heatmap", "rank"],
)
print(f"\nRecords: {len(df3)}, Methods: {sorted(df3['Method'].unique())}")

print("\n" + "=" * 60)
print("  All tests done! Check figures_test_full/")
print("=" * 60)
