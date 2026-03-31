import os
import torch
import scanpy as sc
import argparse
import time
import json
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root directory to module search path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

# Add SpaBalance to path
spabalance_path = os.path.join(project_root, "Methods/SpaBalance/SpaBalance")
sys.path.append(spabalance_path)

from preprocess import fix_seed, clr_normalize_each_cell, pca, lsi, construct_neighbor_graph
from Train_model import Train_SpaBalance
from Utils.SMOBench_clustering import universal_clustering

# ---------------------------------------------------------------------------
# Monkey-patch: replace Multi_CrossAttention.forward to avoid O(N^2) memory.
#
# The original code builds a full [num_heads, N, N] attention matrix but only
# uses it via attention_weights.mean(dim=0).mean(dim=-1), which always equals
# 1/N (because softmax rows sum to 1).  This patch skips the huge matrix and
# uses the constant 1/N directly — mathematically equivalent, O(N) memory.
# ---------------------------------------------------------------------------
import torch.nn.functional as _F
from model import Multi_CrossAttention as _MCA

def _efficient_cross_attention_forward(self, emb1, emb2):
    num_cells = emb1.size(0)

    # Original aggregated_weights is always 1/N (see derivation above)
    aggregated_weights = torch.full(
        (num_cells,), 1.0 / num_cells, device=emb1.device, dtype=emb1.dtype
    )

    modality_weights = []
    for head in self.modality_attention_heads:
        modality_weight = head(aggregated_weights.unsqueeze(-1))
        modality_weights.append(modality_weight)

    modality_weights = torch.cat(modality_weights, dim=-1)
    modality_weights = _F.softmax(modality_weights, dim=-1)

    emb1_weighted = emb1 * modality_weights[:, 0:1]
    emb2_weighted = emb2 * modality_weights[:, 1:2]
    integrated_embeddings = emb1_weighted + emb2_weighted

    return integrated_embeddings, modality_weights

_MCA.forward = _efficient_cross_attention_forward
print("[patch] Multi_CrossAttention.forward patched: O(N^2)->O(N) memory")


def parse_dataset_info(args):
    """Extract dataset_name and subset_name from fusion paths"""
    if hasattr(args, 'dataset') and args.dataset:
        return args.dataset, "fusion"
    
    # Auto parse from path names
    if "HLN_Fusion" in args.RNA_path:
        return "HLN", "fusion"
    elif "HT_Fusion" in args.RNA_path:
        return "HT", "fusion"
    elif "ME_S1_Fusion" in args.RNA_path:
        return "MISAR_S1", "fusion"
    elif "ME_S2_Fusion" in args.RNA_path:
        return "MISAR_S2", "fusion"
    elif "Mouse_Thymus_Fusion" in args.RNA_path:
        return "Mouse_Thymus", "fusion"
    elif "Mouse_Spleen_Fusion" in args.RNA_path:
        return "Mouse_Spleen", "fusion"
    elif "Mouse_Brain_Fusion" in args.RNA_path:
        return "Mouse_Brain", "fusion"
    return "Unknown", "fusion"


def main(args):
    total_start_time = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # === Load Fusion Data ===
    adata_omics1 = sc.read_h5ad(args.RNA_path)
    adata_omics1.var_names_make_unique()

    if args.ADT_path:
        adata_omics2 = sc.read_h5ad(args.ADT_path)
        modality = "ADT"
        modality_name = "Proteome"
    elif args.ATAC_path:
        adata_omics2 = sc.read_h5ad(args.ATAC_path)
        modality = "ATAC"
        modality_name = "Epigenome"
    else:
        raise ValueError("Either ADT_path or ATAC_path must be provided.")
    adata_omics2.var_names_make_unique()

    print(f"Processing SpaBalance horizontal integration: RNA + {modality_name} fusion data...")

    # === Check for batch information ===
    for adata, name in [(adata_omics1, "RNA"), (adata_omics2, modality_name)]:
        if "batch" not in adata.obs.columns:
            print(f"Warning: No 'batch' column found in {name}, assigning default batch labels.")
            n_cells = adata.n_obs
            adata.obs["batch"] = ["batch_1"] * (n_cells // 2) + ["batch_2"] * (n_cells - n_cells // 2)

    # === Align cells ===
    common_obs = adata_omics1.obs_names.intersection(adata_omics2.obs_names)
    adata_omics1 = adata_omics1[common_obs].copy()
    adata_omics2 = adata_omics2[common_obs].copy()
    print(f"Aligned {len(common_obs)} common cells.")

    # === Spatial coordinates check ===
    for adata, name in [(adata_omics1, "RNA"), (adata_omics2, modality_name)]:
        if "spatial" not in adata.obsm.keys():
            print(f"Warning: No spatial coordinates in {name}, generating pseudo-spatial coords (UMAP)...")
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
            adata.obsm["spatial"] = adata.obsm["X_umap"].copy()

    # === Fix seed ===
    fix_seed(args.seed)

    # === RNA preprocessing ===
    sc.pp.filter_genes(adata_omics1, min_cells=10)
    sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata_omics1, target_sum=1e4)
    sc.pp.log1p(adata_omics1)
    sc.pp.scale(adata_omics1)

    adata_omics1_high = adata_omics1[:, adata_omics1.var["highly_variable"]]
    n_comps = 40
    adata_omics1.obsm["feat"] = pca(adata_omics1_high, n_comps=n_comps)

   # === ADT preprocessing ===
    if modality == "ADT":
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        n_comps_adt = min(n_comps, adata_omics2.X.shape[1])  # 不超过特征数
        print(f"Performing PCA on ADT with n_comps = {n_comps_adt}")
        adata_omics2.obsm["feat"] = pca(adata_omics2, n_comps=n_comps_adt)

    # === ATAC preprocessing ===
    elif modality == "ATAC":
        if "X_lsi" not in adata_omics2.obsm.keys():
            sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
            lsi(adata_omics2, use_highly_variable=False, n_components=n_comps)
        adata_omics2.obsm["feat"] = adata_omics2.obsm["X_lsi"].copy()

    # === RNA preprocessing（确保有PCA特征）===
    elif modality == "RNA":
        # 如果RNA没有feat，就执行PCA
        if "feat" not in adata_omics2.obsm.keys():
            sc.pp.normalize_total(adata_omics2)
            sc.pp.log1p(adata_omics2)
            sc.pp.scale(adata_omics2)
            n_comps_rna = min(n_comps, adata_omics2.X.shape[1])
            print(f"Performing PCA on RNA with n_comps = {n_comps_rna}")
            adata_omics2.obsm["feat"] = pca(adata_omics2, n_comps=n_comps_rna)

    # === 对齐 RNA 和 另一组学（ADT / ATAC）的特征维度 ===
    if adata_omics1.obsm["feat"].shape[1] != adata_omics2.obsm["feat"].shape[1]:
        target_dim = min(adata_omics1.obsm["feat"].shape[1], adata_omics2.obsm["feat"].shape[1])
        print(f"Aligning feature dims to {target_dim}")
        adata_omics1.obsm["feat"] = adata_omics1.obsm["feat"][:, :target_dim]
        adata_omics2.obsm["feat"] = adata_omics2.obsm["feat"][:, :target_dim]

    # === 构建邻接图 ===
    print("Constructing neighbor graph for SpaBalance horizontal integration...")
    data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype="fusion")

    # === Train SpaBalance model ===
    print("Training SpaBalance model for horizontal integration...")
    start_time = time.time()

    model = Train_SpaBalance(data, datatype="fusion", device=device)
    output = model.train()

    end_time = time.time()
    train_time = end_time - start_time
    print("Training completed. Time elapsed:", train_time)

    # === Save embeddings ===
    adata = adata_omics1.copy()
    adata.obsm["SpaBalance"] = output["SpaBalance"].copy()
    adata.uns["train_time"] = train_time
    adata.uns["integration_type"] = "horizontal"

    # === Clustering and visualization ===
    tools = ["mclust", "louvain", "leiden", "kmeans"]

    print("Generating neighbors and UMAP for SpaBalance embeddings...")
    sc.pp.neighbors(adata, use_rep="SpaBalance", n_neighbors=30)
    sc.tl.umap(adata)

    print("Running clustering algorithms...")
    for tool in tools:
        print(f"  Running {tool} clustering...")
        adata = universal_clustering(
            adata,
            n_clusters=args.cluster_nums,
            used_obsm="SpaBalance",
            method=tool,
            key=tool,
            use_pca=False,
        )

    # === Save Results ===
    save_dir = os.path.dirname(args.save_path)
    os.makedirs(save_dir, exist_ok=True)
    adata.write(args.save_path)
    print("Saved results to:", args.save_path)
    print("SpaBalance horizontal integration completed successfully.")

    # === Save Timing JSON ===
    total_time = time.time() - total_start_time
    dataset_name, _ = parse_dataset_info(args)
    modality_str = "RNA_ADT" if args.ADT_path else "RNA_ATAC"
    timing_info = {
        "method": "SpaBalance",
        "dataset": dataset_name,
        "integration_type": "horizontal",
        "modality": modality_str,
        "n_cells": adata.n_obs,
        "embedding_shape": list(adata.obsm["SpaBalance"].shape),
        "training_time_s": round(train_time, 2),
        "total_time_s": round(total_time, 2),
        "device": str(device),
        "seed": args.seed,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    timing_path = args.save_path.replace(".h5ad", "_timing_info.json")
    with open(timing_path, "w") as f:
        json.dump(timing_info, f, indent=2)
    print(f"Timing info saved to {timing_path}")


if __name__ == "__main__":
    os.environ['R_HOME'] = '/home/users/nus/e1724738/miniconda3/envs/_Proj1_1/lib/R'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser(description="Run SpaBalance horizontal integration")
    parser.add_argument("--RNA_path", type=str, required=True, help="Path to RNA fusion adata")
    parser.add_argument('--data_type', type=str, default='fusion', help='Data type for horizontal integration')
    parser.add_argument("--ADT_path", type=str, default="", help="Path to ADT fusion adata")
    parser.add_argument("--ATAC_path", type=str, default="", help="Path to ATAC fusion adata")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save integrated adata")
    parser.add_argument("--dataset", type=str, default="", help="Dataset name for horizontal integration")
    parser.add_argument("--cluster_nums", type=int, default=10, help="Number of clusters")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")

    args = parser.parse_args()
    main(args)
