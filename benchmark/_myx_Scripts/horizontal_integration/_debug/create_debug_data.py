"""
Subsample fusion h5ad files to ~300 cells for quick debug runs.
Ensures paired RNA/ADT (or RNA/ATAC) files share the same cell subset.
"""
import scanpy as sc
import numpy as np
import os, sys

N_CELLS = 300
SEED = 42

ROOT = "/home/users/nus/e1724738/_main/_private/NUS/_Proj1/SMOBench-CLEAN"
OUT = os.path.join(ROOT, "Dataset/_myx_fusion_debug")

# Source files: (subdir, tissue, modalities)
PAIRS = [
    # woGT ADT — for SMOPCA, CANDIES, PRAGA, SpaBalance, SpaFusion
    ("_myx_fusionWoGT/RNA_ADT", "Mouse_Thymus", ["RNA", "ADT"]),
    # withGT ATAC — for SMOPCA, SpaMI
    ("_myx_fusionWithGT/RNA_ATAC", "Mouse_Embryos_S1", ["RNA", "ATAC"]),
    # woGT ATAC — for SpaMV, SpaMI (Mouse_Brain)
    ("_myx_fusionWoGT/RNA_ATAC", "Mouse_Brain", ["RNA", "ATAC"]),
]

np.random.seed(SEED)

for subdir, tissue, mods in PAIRS:
    src_dir = os.path.join(ROOT, "Dataset", subdir)
    # Determine output subdirectory (preserve woGT/withGT and RNA_ADT/RNA_ATAC structure)
    out_dir = os.path.join(OUT, subdir.split("/")[-1])
    os.makedirs(out_dir, exist_ok=True)

    # Load first modality to get cell count
    first_file = os.path.join(src_dir, f"{tissue}_Fusion_{mods[0]}.h5ad")
    adata0 = sc.read_h5ad(first_file)
    n_total = adata0.n_obs
    n_sample = min(N_CELLS, n_total)

    # Random cell indices (shared across paired files)
    idx = np.sort(np.random.choice(n_total, n_sample, replace=False))
    print(f"{tissue}: {n_total} -> {n_sample} cells")

    for mod in mods:
        src = os.path.join(src_dir, f"{tissue}_Fusion_{mod}.h5ad")
        dst = os.path.join(out_dir, f"{tissue}_Fusion_{mod}.h5ad")
        adata = sc.read_h5ad(src)
        adata_sub = adata[idx].copy()
        adata_sub.write_h5ad(dst)
        print(f"  {mod}: {src} -> {dst} ({adata_sub.shape})")

print("\nDone. Debug data at:", OUT)
