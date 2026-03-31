#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run MISO-3M triple-modality integration (RNA + ADT + ATAC)

MISO accepts a features list of any length. For 3M, we pass
features=[rna_feat, adt_feat, atac_feat]. The model automatically
generates all 3-choose-2 combinations.
"""

import argparse
import os
import sys
import time

import numpy as np
import scanpy as sc
import torch

# === Path setup ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

miso_root = os.path.join(project_root, "Methods", "MISO")
if miso_root not in sys.path:
    sys.path.append(miso_root)

try:
    from miso import Miso
    from miso.utils import preprocess, set_random_seed
except ImportError as exc:
    raise ImportError(f"Failed to import MISO package: {exc}") from exc

from Utils.SMOBench_clustering import batch_clustering


def parse_dataset_info(rna_path: str, save_path: str):
    parts = rna_path.split(os.sep)
    try:
        idx = parts.index("Dataset")
        return parts[idx + 2], parts[idx + 3]
    except (ValueError, IndexError):
        pass
    parts = save_path.split(os.sep)
    try:
        idx = parts.index("Results")
        return parts[idx + 4], parts[idx + 5]
    except (ValueError, IndexError):
        return "Unknown", "Unknown"


def run_miso(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_random_seed(args.seed)

    # === Load all three modalities ===
    print("Loading RNA + ADT + ATAC data...")
    adata_rna = sc.read_h5ad(args.RNA_path)
    adata_adt = sc.read_h5ad(args.ADT_path)
    adata_atac = sc.read_h5ad(args.ATAC_path)
    for a in [adata_rna, adata_adt, adata_atac]:
        a.var_names_make_unique()

    # === Align cells across all three modalities ===
    common = adata_rna.obs_names.intersection(adata_adt.obs_names).intersection(adata_atac.obs_names)
    if len(common) == 0:
        raise ValueError("No shared cells across three modalities.")
    print(f"Common cells: {len(common)} (RNA={adata_rna.n_obs}, ADT={adata_adt.n_obs}, ATAC={adata_atac.n_obs})")

    adata_rna = adata_rna[common].copy()
    adata_adt = adata_adt[common].copy()
    adata_atac = adata_atac[common].copy()

    # === Feature extraction via MISO preprocess ===
    print("Preprocessing features...")
    rna_feat = preprocess(adata_rna, modality="rna")
    adt_feat = preprocess(adata_adt, modality="protein")
    atac_feat = preprocess(adata_atac, modality="atac")

    features = [
        rna_feat.astype(np.float32),
        adt_feat.astype(np.float32),
        atac_feat.astype(np.float32),
    ]
    print(f"Feature shapes: RNA={features[0].shape}, ADT={features[1].shape}, ATAC={features[2].shape}")

    # === Train MISO ===
    print(f"Training MISO model for 3M (seed={args.seed})...")
    model = Miso(features, ind_views="all", combs="all", sparse=False, device=device)

    start_time = time.time()
    model.train()
    train_time = time.time() - start_time
    emb = model.emb.astype(np.float32)
    print(f"Training completed in {train_time:.2f}s, embedding shape: {emb.shape}")

    # === Build result AnnData ===
    adata_result = adata_rna.copy()
    adata_result.obsm["MISO"] = emb
    adata_result.uns["train_time"] = train_time
    adata_result.uns["method"] = "MISO_3M"

    # === Clustering ===
    print("Running clustering (mclust / louvain / leiden / kmeans)...")
    adata_result = batch_clustering(
        adata_result,
        n_clusters=args.cluster_nums,
        used_obsm="MISO",
        methods=["mclust", "louvain", "leiden", "kmeans"],
        prefix="",
        use_pca=False,
        random_state=args.seed,
    )

    dataset_name, subset_name = parse_dataset_info(args.RNA_path, args.save_path)
    print(f"Dataset: {dataset_name}, Subset: {subset_name}")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    adata_result.write_h5ad(args.save_path)
    print(f"Saved integrated AnnData to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MISO 3M integration (RNA+ADT+ATAC)")
    parser.add_argument("--data_type", type=str, default="simulation")
    parser.add_argument("--RNA_path", type=str, required=True)
    parser.add_argument("--ADT_path", type=str, required=True)
    parser.add_argument("--ATAC_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--cluster_nums", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_miso(args)
