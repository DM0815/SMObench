"""
Create debug datasets (~500 cells) for all integration types:
  - Vertical: subsample from withGT/woGT per-slice data
  - Horizontal: subsample from fusion data
  - Mosaic: 3M_Simulation has only 1296 cells, use directly (no subsample)
"""
import scanpy as sc
import numpy as np
import os
import sys

N_CELLS = 500
SEED = 42

ROOT = "/home/users/nus/e1724738/_main/_private/NUS/_Proj1/SMOBench-CLEAN"
OUT = os.path.join(ROOT, "Dataset/_myx_debug_all")

np.random.seed(SEED)


def subsample_h5ad(src_path, dst_path, idx):
    adata = sc.read_h5ad(src_path)
    adata_sub = adata[idx].copy()
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    adata_sub.write_h5ad(dst_path)
    print(f"  {src_path} -> {dst_path} ({adata_sub.shape})")
    return adata_sub


def subsample_pair(src_dir, dst_dir, files, n_cells=N_CELLS):
    """Subsample paired h5ad files to share the same cell indices."""
    first = sc.read_h5ad(os.path.join(src_dir, files[0]))
    n_total = first.n_obs
    n_sample = min(n_cells, n_total)
    idx = np.sort(np.random.choice(n_total, n_sample, replace=False))
    print(f"  {n_total} -> {n_sample} cells")
    for f in files:
        subsample_h5ad(os.path.join(src_dir, f), os.path.join(dst_dir, f), idx)


# ===========================================================================
# 1. Vertical integration debug data
# ===========================================================================
print("=" * 60)
print("1. Vertical integration debug data")
print("=" * 60)

VERTICAL_PAIRS = [
    # (src_subdir_under_Dataset, dst_subdir, list_of_files)
    # ADT withGT: HLN/A1
    ("withGT/RNA_ADT/Human_Lymph_Nodes/A1",
     "vertical/withGT/RNA_ADT/Human_Lymph_Nodes/A1",
     ["adata_RNA.h5ad", "adata_ADT.h5ad"]),
    # ATAC withGT: Mouse_Embryos_S1/E11
    ("withGT/RNA_ATAC/Mouse_Embryos_S1/E11",
     "vertical/withGT/RNA_ATAC/Mouse_Embryos_S1/E11",
     ["adata_RNA.h5ad", "adata_ATAC.h5ad"]),
    # ADT woGT: Mouse_Thymus/Mouse_Thymus1
    ("woGT/RNA_ADT/Mouse_Thymus/Mouse_Thymus1",
     "vertical/woGT/RNA_ADT/Mouse_Thymus/Mouse_Thymus1",
     ["adata_RNA.h5ad", "adata_ADT.h5ad"]),
    # ATAC woGT: Mouse_Brain/Mouse_Brain_ATAC
    ("woGT/RNA_ATAC/Mouse_Brain/Mouse_Brain_ATAC",
     "vertical/woGT/RNA_ATAC/Mouse_Brain/Mouse_Brain_ATAC",
     ["adata_RNA.h5ad", "adata_ATAC.h5ad"]),
]

for src_sub, dst_sub, files in VERTICAL_PAIRS:
    src_dir = os.path.join(ROOT, "Dataset", src_sub)
    dst_dir = os.path.join(OUT, dst_sub)
    print(f"\n{src_sub}:")
    subsample_pair(src_dir, dst_dir, files)

# ===========================================================================
# 2. Horizontal integration debug data (fusion files)
# ===========================================================================
print("\n" + "=" * 60)
print("2. Horizontal integration debug data")
print("=" * 60)

HORIZONTAL_PAIRS = [
    # (src_subdir, tissue, modalities)
    # ADT woGT: Mouse_Thymus
    ("_myx_fusionWoGT/RNA_ADT", "Mouse_Thymus", ["RNA", "ADT"]),
    # ADT withGT: Human_Lymph_Nodes
    ("_myx_fusionWithGT/RNA_ADT", "Human_Lymph_Nodes", ["RNA", "ADT"]),
    # ATAC withGT: Mouse_Embryos_S1
    ("_myx_fusionWithGT/RNA_ATAC", "Mouse_Embryos_S1", ["RNA", "ATAC"]),
    # ATAC woGT: Mouse_Brain
    ("_myx_fusionWoGT/RNA_ATAC", "Mouse_Brain", ["RNA", "ATAC"]),
]

for src_sub, tissue, mods in HORIZONTAL_PAIRS:
    src_dir = os.path.join(ROOT, "Dataset", src_sub)
    dst_dir = os.path.join(OUT, "horizontal", src_sub.split("/")[-1])
    print(f"\n{tissue} ({src_sub}):")
    files = [f"{tissue}_Fusion_{m}.h5ad" for m in mods]
    subsample_pair(src_dir, dst_dir, files)

# ===========================================================================
# 3. Mosaic (3M) — just symlink, data is small enough (1296 cells)
# ===========================================================================
print("\n" + "=" * 60)
print("3. Mosaic (3M_Simulation) — copying as-is (1296 cells)")
print("=" * 60)

mosaic_src = os.path.join(ROOT, "Dataset/withGT/3M_Simulation")
mosaic_dst = os.path.join(OUT, "mosaic/3M_Simulation")
os.makedirs(mosaic_dst, exist_ok=True)
for fname in ["adata_RNA.h5ad", "adata_ADT.h5ad", "adata_ATAC.h5ad"]:
    src = os.path.join(mosaic_src, fname)
    dst = os.path.join(mosaic_dst, fname)
    if not os.path.exists(dst):
        adata = sc.read_h5ad(src)
        adata.write_h5ad(dst)
        print(f"  Copied {fname} ({adata.shape})")
    else:
        print(f"  {fname} already exists, skipping")

print(f"\nDone. Debug data at: {OUT}")
