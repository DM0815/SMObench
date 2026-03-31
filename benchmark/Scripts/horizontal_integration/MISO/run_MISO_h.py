import argparse
import os
import sys
import time

import numpy as np
import scanpy as sc
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

miso_root = os.path.join(project_root, "Methods", "miso")
if miso_root not in sys.path:
    sys.path.append(miso_root)

from miso import Miso  # type: ignore
from miso.utils import preprocess, set_random_seed  # type: ignore
from Utils.SMOBench_clustering import universal_clustering


def parse_dataset_name(path: str) -> str:
    parts = path.split(os.sep)
    try:
        idx = parts.index("Dataset")
        return parts[idx + 2]
    except (ValueError, IndexError):
        return "Unknown"


def load_fusion(adata_path: str) -> sc.AnnData:
    adata = sc.read_h5ad(adata_path)
    adata.var_names_make_unique()
    return adata


def ensure_spatial(adata: sc.AnnData, label: str, seed: int = 42):
    if "spatial" in adata.obsm:
        return
    print(f"[Warning] No spatial coordinates for {label}; creating pseudo-spatial via UMAP.")
    sc.pp.neighbors(adata, use_rep=None)
    sc.tl.umap(adata, random_state=seed)
    adata.obsm["spatial"] = adata.obsm["X_umap"].copy()


def prepare_features(adata: sc.AnnData, modality: str) -> np.ndarray:
    if modality == "RNA":
        return preprocess(adata, modality="rna").astype(np.float32)
    if modality == "ADT":
        return preprocess(adata, modality="protein").astype(np.float32)
    if modality == "ATAC":
        return preprocess(adata, modality="atac").astype(np.float32)
    raise ValueError(f"Unsupported modality: {modality}")


def run_horizontal(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_random_seed(args.seed)

    if args.ADT_path:
        modality = "ADT"
        modal_path = args.ADT_path
    elif args.ATAC_path:
        modality = "ATAC"
        modal_path = args.ATAC_path
    else:
        raise ValueError("Either --ADT_path or --ATAC_path must be provided.")

    adata_rna = load_fusion(args.RNA_path)
    adata_other = load_fusion(modal_path)

    ensure_spatial(adata_rna, "RNA", args.seed)
    ensure_spatial(adata_other, modality, args.seed)

    if "batch" not in adata_rna.obs:
        n = adata_rna.n_obs
        adata_rna.obs["batch"] = ["batch_1"] * (n // 2) + ["batch_2"] * (n - n // 2)
    if "batch" not in adata_other.obs:
        adata_other.obs["batch"] = adata_rna.obs["batch"].copy()

    common = adata_rna.obs_names.intersection(adata_other.obs_names)
    if len(common) == 0:
        raise ValueError("No overlapping cells between RNA and secondary modality.")
    adata_rna = adata_rna[common].copy()
    adata_other = adata_other[common].copy()

    print("Preparing MISO inputs...")
    feat_rna = prepare_features(adata_rna, "RNA")
    feat_other = prepare_features(adata_other, modality)

    model = Miso([feat_rna, feat_other], ind_views="all", combs="all", sparse=False, device=device)
    start = time.time()
    model.train()
    train_time = time.time() - start
    print(f"MISO horizontal training time: {train_time:.2f}s")

    adata = adata_rna.copy()
    adata.obsm["MISO"] = model.emb.astype(np.float32)
    adata.uns["train_time"] = train_time
    adata.uns["integration_type"] = "horizontal"

    print("Running clustering (mclust & kmeans)...")
    for method in ["mclust", "kmeans"]:
        adata = universal_clustering(
            adata,
            n_clusters=args.cluster_nums,
            used_obsm="MISO",
            method=method,
            key=method,
            use_pca=False,
            random_state=args.seed,
        )

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    adata.write_h5ad(args.save_path)
    print(f"Saved horizontal integration result to {args.save_path}")


def build_argparser():
    parser = argparse.ArgumentParser(description="Run MISO horizontal integration on fusion datasets.")
    parser.add_argument("--RNA_path", type=str, required=True, help="Path to fusion RNA AnnData.")
    parser.add_argument("--ADT_path", type=str, default=None, help="Path to fusion ADT AnnData.")
    parser.add_argument("--ATAC_path", type=str, default=None, help="Path to fusion ATAC AnnData.")
    parser.add_argument("--save_path", type=str, required=True, help="Destination for output h5ad.")
    parser.add_argument("--cluster_nums", type=int, required=True, help="Target number of clusters.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device, e.g., cuda:0 or cpu.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_horizontal(args)
