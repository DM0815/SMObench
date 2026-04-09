"""
Plot method comparison from saved h5ad files.

Usage:
    python plot_comparison.py
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import smobench.plot as splot

# Human_Lymph_Nodes A1 — 14 methods, has ground truth
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "vertical", "Human_Lymph_Nodes", "A1")
SAVE_DIR = os.path.join(os.path.dirname(__file__), "figures")

df = splot.plot_summary(
    RESULTS_DIR,
    clustering="leiden",
    save_dir=SAVE_DIR,
    dataset_name="Human_Lymph_Nodes",
    slice_name="A1",
    # Skip umap/spatial grids for now (slow), can enable with:
    # plots=["heatmap", "dot_matrix", "radar", "scatter", "runtime", "rank", "umap_grid", "spatial_grid"],
    plots=["dot_matrix", "radar", "scatter", "runtime", "rank", "umap_grid", "spatial_grid"],
)

print("\n=== Summary ===")
score_cols = [c for c in ["Method", "ARI", "NMI", "Moran_I", "BioC_Score", "SC_Score", "Runtime"] if c in df.columns]
print(df[score_cols].to_string(index=False))
