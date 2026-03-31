# -*- coding: utf-8 -*-
"""
Run SMOPCA for horizontal integration (RNA + ADT or RNA + ATAC)
Adapted from SpatialGlue pipeline for consistency within SMOBench framework.
"""

import os
import sys
import time
import json
import torch
import logging
import warnings
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import scipy.sparse
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from scipy.linalg import eigh
from scipy.optimize import brentq

# === Import project root and SMOPCA module ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

smopca_path = os.path.join(project_root, "_myx_Methods/SMOPCA/src")
sys.path.append(smopca_path)

import model
import utils
from Methods.SpatialGlue.preprocess import fix_seed
from Utils.SMOBench_clustering import universal_clustering

# === Suppress logging clutter ===
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')


# =====================================================================
# Monkey-patch: SMOPCA.estimateParams with wider bound search + fallback
# =====================================================================
# Problem: The original estimateParams uses np.arange(0.1, 10.0, 0.1) to
# find a bracket for brentq. For some datasets the root lies below 0.1,
# so no bracket is found and brentq raises ValueError.
#
# Fix: Use np.logspace(-4, 2, 2000) for much wider coverage (1e-4 to 100),
# and if brentq still fails, fall back to minimize_scalar.
# =====================================================================

_original_estimateParams = model.SMOPCA.estimateParams


def _patched_estimateParams(self, iterations_gamma=10, iterations_sigma_W=20,
                            tol_gamma=1e-2, tol_sigma=1e-5,
                            estimate_gamma=False, gamma_init=1,
                            gamma_bound=(0.1, 5),
                            sigma_init_list=(), sigma_xtol_list=(),
                            gamma_tol=0.1):
    """Monkey-patched estimateParams with wider bound search and fallback."""
    assert len(sigma_init_list) == len(sigma_xtol_list) == self.modality_num
    logger = logging.getLogger(__name__)
    logger.info("start estimating parameters (patched), this will take a while...")

    gamma = gamma_init
    self.buildKernel(length_scale=gamma)

    for iter1 in range(iterations_gamma):
        bound_list = [None for _ in range(self.modality_num)]
        self.W_hat_list = []
        self.sigma_hat_sqr_list = []
        for modality in range(self.modality_num):
            Y = self.Y_list[modality]
            tr_YY_T = np.trace(Y @ Y.T)
            sigma_sqr = sigma_init_list[modality]
            sigma_hat_sqr = None
            W_hat = None
            logger.info(f"estimating sigma{modality + 1}")
            for iter2 in range(iterations_sigma_W):
                # estimate W_k
                D1 = np.diag(self.lbds * sigma_sqr / (self.lbds + sigma_sqr))
                P1 = Y @ self.U
                G = P1 @ D1 @ P1.T
                vals, vec = eigh(G)
                W_hat = vec[:, -self.d:]
                assert W_hat.shape == (self.m_list[modality], self.d)

                # estimate sigma_k
                def jac_sigma_sqr(_sigma_sqr):
                    part1 = self.m_list[modality] * self.n / _sigma_sqr
                    part2 = -np.sum(self.lbds / (self.lbds + _sigma_sqr)) * self.d / _sigma_sqr
                    D2 = np.diag((self.lbds * (2 * _sigma_sqr + self.lbds)) / (self.lbds + _sigma_sqr) ** 2)
                    P2 = W_hat.T @ Y @ self.U
                    part3 = (np.trace(P2 @ D2 @ P2.T) - tr_YY_T) / _sigma_sqr ** 2
                    jac = part1 + part2 + part3
                    logger.debug("jac{}({:.5f}) = {:.5f}".format(modality + 1, _sigma_sqr, jac))
                    return jac

                # === PATCHED: wider bound search using logspace ===
                if bound_list[modality] is None:
                    search_grid = np.logspace(-4, 2, 2000)  # 1e-4 to 100
                    lb = search_grid[0]
                    ub = search_grid[-1]
                    lb_res = -np.inf
                    ub_res = np.inf
                    found_bracket = False
                    for sigma in search_grid:
                        res = jac_sigma_sqr(sigma)
                        if res < 0:
                            lb = sigma
                            lb_res = res
                        else:
                            ub = sigma
                            ub_res = res
                            found_bracket = True
                            break

                    if found_bracket:
                        if abs(lb_res) < 1000:
                            lb = max(lb * 0.5, 1e-6)
                        if abs(ub_res) < 1000:
                            ub = ub * 2.0
                        bound_list[modality] = (lb, ub)
                        logger.info("sigma{} using bound: ({:.6f}, {:.6f})".format(modality + 1, lb, ub))
                    else:
                        # No bracket found at all; will use minimize_scalar fallback
                        bound_list[modality] = None
                        logger.warning("sigma{} no bracket found in logspace scan, will use minimize_scalar fallback".format(modality + 1))

                # === PATCHED: try brentq, fall back to minimize_scalar ===
                try:
                    if bound_list[modality] is not None:
                        sigma_hat_sqr = brentq(jac_sigma_sqr, bound_list[modality][0], bound_list[modality][1],
                                               xtol=sigma_xtol_list[modality])
                    else:
                        raise ValueError("No bracket available")
                except ValueError as e:
                    logger.warning("brentq failed for sigma{} ({}), falling back to minimize_scalar".format(
                        modality + 1, e))
                    ret = scipy.optimize.minimize_scalar(
                        lambda s: abs(jac_sigma_sqr(s)),
                        method="Bounded",
                        bounds=(1e-6, 100.0),
                        options={"xatol": sigma_xtol_list[modality]}
                    )
                    sigma_hat_sqr = ret.x
                    logger.info("minimize_scalar fallback: sigma{}hatsqr = {:.5f}".format(modality + 1, sigma_hat_sqr))

                logger.info("iter {} sigma{} done, sigma{}sqr = {:.5f}, sigma{}hatsqr = {:.5f}".format(
                    iter2, modality + 1, modality + 1, sigma_sqr, modality + 1, sigma_hat_sqr))

                if abs(sigma_sqr - sigma_hat_sqr) < tol_sigma:
                    logger.info(f"reach tolerance threshold, sigma{modality + 1} done!")
                    self.sigma_hat_sqr_list.append(sigma_hat_sqr)
                    self.W_hat_list.append(W_hat)
                    break
                sigma_sqr = sigma_hat_sqr
                if iter2 == iterations_sigma_W - 1:
                    logger.warning(f"reach end of iteration for sigma{modality + 1}!")
                    self.sigma_hat_sqr_list.append(sigma_hat_sqr)
                    self.W_hat_list.append(W_hat)

        if not estimate_gamma:
            break

        def f_gamma(g):
            from sklearn.gaussian_process.kernels import Matern
            matern_obj = Matern(length_scale=g, nu=self.nu)
            K = matern_obj(X=self.pos, Y=self.pos)
            lbds, U = eigh(K)
            val = 0
            for k in range(self.modality_num):
                if k == 0:
                    continue
                alpha_k = self.alpha_list[k]
                sigma_k_sqr = self.sigma_hat_sqr_list[k]
                W_k = self.W_hat_list[k]
                Y_k = self.Y_list[k]
                part1 = self.d * np.sum(np.log(1 + lbds / sigma_k_sqr))
                D = np.diag(lbds / (lbds + sigma_k_sqr))
                part2 = -np.trace(W_k.T @ Y_k @ U @ D @ U.T @ Y_k.T @ W_k) / sigma_k_sqr
                val += alpha_k * (part1 + part2)
            logger.debug("f_gamma({:.5f}) = {:.5f}".format(g, val))
            return val

        ret = scipy.optimize.minimize_scalar(f_gamma, method="Bounded", bounds=gamma_bound, tol=gamma_tol)
        gamma_hat = ret['x']
        logger.info("iter {} gamma minimize done, gamma = {:.5f}, gamma_hat = {:.5f}".format(iter1, gamma, gamma_hat))
        self.buildKernel(length_scale=gamma_hat)
        if abs(gamma - gamma_hat) < tol_gamma:
            self.gamma_hat = gamma_hat
            logger.info(f"reach tolerance threshold, gamma done!")
            break
        gamma = gamma_hat
        if iter1 == iterations_gamma - 1:
            self.gamma_hat = gamma_hat
            logger.warning(f"reach end of iteration for gamma!")
            break

    logger.info("estimation complete!")
    for modality, sigma_hat_sqr in enumerate(self.sigma_hat_sqr_list):
        logger.info("sigma{}hatsqr = {:.5f}".format(modality + 1, sigma_hat_sqr))
    if estimate_gamma:
        logger.info("gamma_hat = {:.5f}".format(self.gamma_hat))


model.SMOPCA.estimateParams = _patched_estimateParams
print("[PATCH] SMOPCA.estimateParams monkey-patched with wider bound search + minimize_scalar fallback")


# ---------------------------------------------------------------------
# Helper function: parse dataset info
# ---------------------------------------------------------------------
def parse_dataset_info(args):
    """Extract dataset_name and subset_name from fusion paths."""
    if hasattr(args, 'dataset') and args.dataset:
        return args.dataset, "fusion"

    if "HLN_Fusion" in args.RNA_path:
        return "HLN", "fusion"
    elif "HT_Fusion" in args.RNA_path:
        return "HT", "fusion"
    elif "Mouse_Thymus" in args.RNA_path:
        return "Mouse_Thymus", "fusion"
    elif "Mouse_Spleen" in args.RNA_path:
        return "Mouse_Spleen", "fusion"
    elif "Mouse_Brain" in args.RNA_path:
        return "Mouse_Brain", "fusion"
    else:
        return "Unknown", "fusion"


# ---------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------
def main(args):
    total_start_time = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    fix_seed(args.seed)

    # === Load Fusion Data ===
    print("Loading RNA and secondary modality (ADT/ATAC) fusion data...")
    adata_rna = sc.read_h5ad(args.RNA_path)
    adata_rna.var_names_make_unique()

    if args.ADT_path:
        adata_other = sc.read_h5ad(args.ADT_path)
        modality = "ADT"
        modality_name = "Proteome"
    elif args.ATAC_path:
        adata_other = sc.read_h5ad(args.ATAC_path)
        modality = "ATAC"
        modality_name = "Epigenome"
    else:
        raise ValueError("Please provide either --ADT_path or --ATAC_path.")

    adata_other.var_names_make_unique()
    print(f"Processing horizontal integration: RNA + {modality_name} fusion data...")

    # === Ensure common cells ===
    common_cells = adata_rna.obs_names.intersection(adata_other.obs_names)
    print(f"Common cells: {len(common_cells)}")
    adata_rna = adata_rna[common_cells].copy()
    adata_other = adata_other[common_cells].copy()

    # === Add pseudo spatial coordinates if missing ===
    for adata, name in [(adata_rna, "RNA"), (adata_other, modality_name)]:
        if "spatial" not in adata.obsm.keys():
            print(f"Warning: No spatial coordinates found in {name}. Generating pseudo-spatial coordinates...")
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
            adata.obsm["spatial"] = adata.obsm["X_umap"].copy()
            print(f"Generated pseudo-spatial coordinates for {name}")

    # === Preprocess features ===
    print("Preprocessing RNA and secondary modality features...")
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.highly_variable_genes(adata_rna, n_top_genes=3000)
    adata_rna = adata_rna[:, adata_rna.var["highly_variable"]].copy()

    sc.pp.scale(adata_other)  # scale both ADT and ATAC

    X1 = adata_rna.X.A if hasattr(adata_rna.X, "A") else adata_rna.X
    X2 = adata_other.X.A if hasattr(adata_other.X, "A") else adata_other.X
    pos = adata_rna.obsm["spatial"]

    print(f"Input feature shapes: RNA {X1.shape}, {modality_name} {X2.shape}, pos {pos.shape}")

    # =================================================================
    # Fix 2: Dimensionality reduction for massive ATAC feature counts
    # =================================================================
    # Problem: SMOPCA computes G = P1 @ D1 @ P1.T where P1 is (m, n)
    # with m = #features. For ATAC with 200K+ features, this creates a
    # (200K, 200K) dense matrix requiring hundreds of GB of memory,
    # causing segfaults.
    #
    # Fix: Use TruncatedSVD to reduce X2 (cells x features) to 3000
    # components when features exceed 5000, before transposing to Y.
    # =================================================================
    ATAC_DIM_THRESHOLD = 5000
    ATAC_DIM_TARGET = 3000
    if X2.shape[1] > ATAC_DIM_THRESHOLD:
        print(f"[DIM REDUCTION] {modality_name} has {X2.shape[1]} features (>{ATAC_DIM_THRESHOLD}), "
              f"reducing to {ATAC_DIM_TARGET} components via TruncatedSVD...")
        # TruncatedSVD works on both dense and sparse matrices.
        # X2 is (cells, features); we want to reduce features dimension.
        X2_for_svd = scipy.sparse.csr_matrix(X2) if not scipy.sparse.issparse(X2) else X2
        svd = TruncatedSVD(n_components=ATAC_DIM_TARGET, random_state=args.seed)
        X2 = svd.fit_transform(X2_for_svd)  # (cells, ATAC_DIM_TARGET)
        explained_var = svd.explained_variance_ratio_.sum()
        print(f"[DIM REDUCTION] TruncatedSVD done: {X2.shape}, explained variance ratio = {explained_var:.4f}")

    # === Initialize and Train SMOPCA ===
    print("Training SMOPCA for horizontal integration...")
    smopca = model.SMOPCA(
        Y_list=[X1.T, X2.T],
        Z_dim=args.Z_dim,
        pos=pos,
        intercept=False,
        omics_weight=False
    )

    start_time = time.time()
    smopca.estimateParams(
        sigma_init_list=(1, 1),
        tol_sigma=2e-5,
        sigma_xtol_list=(1e-6, 1e-6),
        gamma_init=1,
        estimate_gamma=True
    )
    z = smopca.calculatePosterior()
    train_time = time.time() - start_time
    print(f"SMOPCA training completed in {train_time:.2f}s")

    # === Build AnnData with embeddings ===
    adata = sc.AnnData(z)
    try:
        adata.obs = adata_rna.obs.copy()
        adata.obs_names = adata_rna.obs_names.copy()
    except Exception:
        if adata.shape[0] == adata_rna.shape[0]:
            adata.obs_names = adata_rna.obs_names.copy()
        else:
            adata.obs_names = [f"cell_{i}" for i in range(adata.shape[0])]

    adata.obsm["SMOPCA"] = np.asarray(z)
    adata.obsm["spatial"] = pos
    adata.uns.update({
        "train_time": train_time,
        "integration_type": "horizontal",
        "method": "SMOPCA"
    })
    adata.var_names = [f"SMOPCA_{i}" for i in range(adata.shape[1])]

    # === Clustering ===
    print("Running clustering methods on SMOPCA embeddings...")
    tools = ["mclust", "louvain", "leiden", "kmeans"]
    for tool in tools:
        print(f"  Running {tool} clustering...")
        adata = universal_clustering(
            adata,
            n_clusters=args.cluster_nums,
            used_obsm="SMOPCA",
            method=tool,
            key=tool,
            use_pca=False
        )
    print("All clustering methods completed")

    # === Save Results ===
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    adata.write(args.save_path)
    print(f"Results saved to {args.save_path}")

    # === Save Timing JSON ===
    total_time = time.time() - total_start_time
    dataset_name, _ = parse_dataset_info(args)
    modality_str = "RNA_ADT" if args.ADT_path else "RNA_ATAC"
    timing_info = {
        "method": "SMOPCA",
        "dataset": dataset_name,
        "integration_type": "horizontal",
        "modality": modality_str,
        "n_cells": adata.n_obs,
        "embedding_shape": list(adata.obsm["SMOPCA"].shape),
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

    # === Visualization (optional) ===
    try:
        sc.pp.neighbors(adata, n_neighbors=30, use_rep="X")
        sc.tl.umap(adata)
        sc.pl.umap(adata, color=["kmeans"], title="SMOPCA clustering", show=False)
        plt_path = args.save_path.replace(".h5ad", "_umap.png")
        plt.savefig(plt_path, dpi=300, bbox_inches="tight")
        print(f"UMAP saved to {plt_path}")
    except Exception as e:
        print("Warning: Visualization skipped due to error:", e)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    os.environ['R_HOME'] = '/home/users/nus/e1724738/miniconda3/envs/_Proj1_1/lib/R'
    os.environ['OMP_NUM_THREADS'] = '128'
    os.environ['MKL_NUM_THREADS'] = '128'
    os.environ['NUMEXPR_NUM_THREADS'] = '128'
    os.environ['OPENBLAS_NUM_THREADS'] = '128'

    print("Starting SMOPCA horizontal integration...")

    parser = argparse.ArgumentParser(description="Run SMOPCA horizontal integration")
    parser.add_argument("--data_type", type=str, default="fusion", help="Data type (e.g., fusion, RNA, ADT, ATAC)")
    parser.add_argument("--method", type=str, default="SMOPCA", help="Method name")
    parser.add_argument("--RNA_path", type=str, required=True, help="Path to RNA fusion adata (.h5ad)")
    parser.add_argument("--ADT_path", type=str, default="", help="Path to ADT fusion adata (.h5ad)")
    parser.add_argument("--ATAC_path", type=str, default="", help="Path to ATAC fusion adata (.h5ad)")
    parser.add_argument("--save_path", type=str, required=True, help="Output path to save integrated AnnData (.h5ad)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Computation device")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    parser.add_argument("--cluster_nums", type=int, default=7, help="Number of clusters for KMeans/Leiden")
    parser.add_argument("--Z_dim", type=int, default=20, help="Latent embedding dimension")
    parser.add_argument("--dataset", type=str, default="", help="Dataset name for tracking")
    args = parser.parse_args()

    main(args)
