#!/usr/bin/env python3
"""
Prepare subsampled datasets for scalability analysis.
Following the approach in [3] (Hu et al., Nature Methods 2024):
  - Pick one large dataset per modality type
  - Randomly subsample different cell counts
  - Repeat sampling 5 times for robustness

RNA_ADT:  Mouse_Thymus Fusion (17,824 cells, largest ADT dataset)
RNA_ATAC: Mouse_Brain Fusion  (37,885 cells, largest ATAC dataset)
"""

import os
import argparse
import numpy as np
import scanpy as sc

# === Configuration ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATASET_DIR = os.path.join(PROJECT_ROOT, "Dataset")

DATASETS = {
    "RNA_ADT": {
        "name": "Mouse_Thymus",
        "RNA_path": os.path.join(DATASET_DIR, "_myx_fusionWoGT/RNA_ADT/Mouse_Thymus_Fusion_RNA.h5ad"),
        "other_path": os.path.join(DATASET_DIR, "_myx_fusionWoGT/RNA_ADT/Mouse_Thymus_Fusion_ADT.h5ad"),
        "total_cells": 17824,
        "cell_counts": [1000, 2500, 5000, 10000, 17824],
    },
    "RNA_ATAC": {
        "name": "Mouse_Brain",
        "RNA_path": os.path.join(DATASET_DIR, "_myx_fusionWoGT/RNA_ATAC/Mouse_Brain_Fusion_RNA.h5ad"),
        "other_path": os.path.join(DATASET_DIR, "_myx_fusionWoGT/RNA_ATAC/Mouse_Brain_Fusion_ATAC.h5ad"),
        "total_cells": 37885,
        "cell_counts": [1000, 2500, 5000, 10000, 20000, 37885],
    },
}

N_REPEATS = 5


def subsample_and_save(adata_rna, adata_other, n_cells, seed, out_dir, modality, dataset_name):
    """Subsample both modalities to n_cells and save."""
    np.random.seed(seed)

    total = adata_rna.n_obs
    if n_cells >= total:
        # Use full dataset
        adata_rna_sub = adata_rna.copy()
        adata_other_sub = adata_other.copy()
        n_cells = total  # actual count
    else:
        idx = np.random.choice(total, size=n_cells, replace=False)
        idx.sort()
        adata_rna_sub = adata_rna[idx].copy()
        adata_other_sub = adata_other[idx].copy()

    # Save
    os.makedirs(out_dir, exist_ok=True)
    rna_out = os.path.join(out_dir, f"{dataset_name}_RNA_{n_cells}cells_rep{seed}.h5ad")
    other_out = os.path.join(out_dir, f"{dataset_name}_{modality.split('_')[1]}_{n_cells}cells_rep{seed}.h5ad")

    adata_rna_sub.write(rna_out)
    adata_other_sub.write(other_out)
    print(f"  Saved: {n_cells} cells, rep {seed} -> {out_dir}")
    return rna_out, other_out


def main():
    parser = argparse.ArgumentParser(description="Prepare subsampled datasets for scalability analysis")
    parser.add_argument("--modality", type=str, choices=["RNA_ADT", "RNA_ATAC", "both"], default="both")
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "_myx_Results/scalability/subsampled_data"))
    parser.add_argument("--n_repeats", type=int, default=N_REPEATS)
    args = parser.parse_args()

    modalities = ["RNA_ADT", "RNA_ATAC"] if args.modality == "both" else [args.modality]

    for modality in modalities:
        cfg = DATASETS[modality]
        print(f"\n{'='*60}")
        print(f"Modality: {modality}, Dataset: {cfg['name']}")
        print(f"Loading data...")

        adata_rna = sc.read_h5ad(cfg["RNA_path"])
        adata_other = sc.read_h5ad(cfg["other_path"])
        print(f"  RNA: {adata_rna.shape}, Other: {adata_other.shape}")

        # Ensure same cell order
        common = adata_rna.obs_names.intersection(adata_other.obs_names)
        adata_rna = adata_rna[common].copy()
        adata_other = adata_other[common].copy()
        print(f"  Common cells: {len(common)}")

        out_dir = os.path.join(args.out_dir, modality)

        for n_cells in cfg["cell_counts"]:
            for rep in range(1, args.n_repeats + 1):
                seed = rep * 1000 + n_cells  # deterministic seed per (n_cells, rep)
                subsample_and_save(
                    adata_rna, adata_other, n_cells, seed,
                    out_dir, modality, cfg["name"]
                )

    print(f"\nDone! Subsampled data saved to: {args.out_dir}")

    # Print summary
    total_files = 0
    for modality in modalities:
        cfg = DATASETS[modality]
        n_points = len(cfg["cell_counts"])
        n = n_points * args.n_repeats * 2  # 2 files per (rna + other)
        total_files += n
        print(f"  {modality}: {n_points} cell counts x {args.n_repeats} repeats = {n} files")
    print(f"  Total: {total_files} files")


if __name__ == "__main__":
    main()
