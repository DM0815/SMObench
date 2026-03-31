import argparse
import os
import sys
import time

import numpy as np
import scanpy as sc
import torch

# Add project root directory to module search path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

# Add MISO sources
miso_root = os.path.join(project_root, "Methods", "miso")
if miso_root not in sys.path:
    sys.path.append(miso_root)

try:
    from miso import Miso
    from miso.utils import preprocess, set_random_seed
except ImportError as exc:
    raise ImportError(f"Failed to import MISO package: {exc}. Please ensure Methods/miso is accessible.") from exc

from Utils.SMOBench_clustering import batch_clustering


def parse_dataset_info(rna_path: str, save_path: str) -> tuple[str, str]:
    """Infer dataset/subset names from paths."""
    # Try to parse from RNA path
    parts = rna_path.split(os.sep)
    try:
        idx = parts.index("Dataset")
        dataset = parts[idx + 2]
        subset = parts[idx + 3]
        return dataset, subset
    except (ValueError, IndexError):
        pass

    # Fallback to save path
    parts = save_path.split(os.sep)
    try:
        idx = parts.index("Results")
        dataset = parts[idx + 4]
        subset = parts[idx + 5]
        return dataset, subset
    except (ValueError, IndexError):
        return "Unknown", "Unknown"


def load_and_align(rna_path: str, other_path: str, modality: str):
    """Load RNA + second modality AnnData objects and align cells."""
    adata_rna = sc.read_h5ad(rna_path)
    adata_other = sc.read_h5ad(other_path)
    adata_rna.var_names_make_unique()
    adata_other.var_names_make_unique()

    common = adata_rna.obs_names.intersection(adata_other.obs_names)
    if len(common) == 0:
        raise ValueError("No shared cells between RNA and secondary modality.")

    adata_rna = adata_rna[common].copy()
    adata_other = adata_other[common].copy()

    # Feature extraction via MISO preprocess utilities
    rna_feat = preprocess(adata_rna, modality="rna")
    if modality == "ADT":
        other_feat = preprocess(adata_other, modality="protein")
    elif modality == "ATAC":
        other_feat = preprocess(adata_other, modality="atac")
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    return adata_rna, rna_feat, other_feat


def run_miso(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_random_seed(args.seed)

    if args.ADT_path:
        modality = "ADT"
        other_path = args.ADT_path
        modality_name = "Proteome"
    elif args.ATAC_path:
        modality = "ATAC"
        other_path = args.ATAC_path
        modality_name = "Epigenome"
    else:
        raise ValueError("Either --ADT_path or --ATAC_path must be provided.")

    print(f"Loading data ({modality_name})...")
    adata_rna, rna_feat, other_feat = load_and_align(args.RNA_path, other_path, modality)
    features = [rna_feat.astype(np.float32), other_feat.astype(np.float32)]

    print(f"Training MISO model for {args.data_type} (seed={args.seed})...")
    model = Miso(features, ind_views="all", combs="all", sparse=False, device=device)

    start_time = time.time()
    model.train()
    train_time = time.time() - start_time
    emb = model.emb.astype(np.float32)
    print(f"Training completed in {train_time:.2f} seconds.")

    # Prepare AnnData container
    adata_result = adata_rna.copy()
    adata_result.obsm["MISO"] = emb
    adata_result.uns["train_time"] = train_time
    adata_result.uns["method"] = "MISO"

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


def build_argparser():
    parser = argparse.ArgumentParser(description="Run MISO vertical integration on a dataset.")
    parser.add_argument("--data_type", type=str, required=True, help="Dataset type label (for logging).")
    parser.add_argument("--RNA_path", type=str, required=True, help="Path to RNA AnnData (.h5ad).")
    parser.add_argument("--ADT_path", type=str, default=None, help="Path to ADT AnnData (if RNA+ADT).")
    parser.add_argument("--ATAC_path", type=str, default=None, help="Path to ATAC AnnData (if RNA+ATAC).")
    parser.add_argument("--save_path", type=str, required=True, help="Output path for integrated h5ad.")
    parser.add_argument("--cluster_nums", type=int, required=True, help="Target number of clusters.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Computation device (e.g., cuda:0 or cpu).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_miso(args)
