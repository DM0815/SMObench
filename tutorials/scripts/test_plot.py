"""Quick test: dot_matrix plot from existing h5ad results."""
import sys
sys.path.insert(0, "../src")

import scanpy as sc
from smobench.plot.summary import plot_summary

# Use a small withGT dataset
adata = sc.read_h5ad("results/vertical/Human_Tonsils/S1/adata_integrated.h5ad")

print("obsm keys:", list(adata.obsm.keys()))
print("Shape:", adata.shape)

# Only generate dot_matrix + ranking (skip UMAP/spatial for speed)
df = plot_summary(
    adata,
    clustering="leiden",
    dataset_name="Human_Tonsils",
    slice_name="S1",
    save_dir="figures_test/",
    plots=["dot_matrix", "rank"],
)

print("\nResult DataFrame:")
print(df.to_string())
