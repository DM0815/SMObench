#!/usr/bin/env python3
"""
SMOPCA horizontal integration with CHECKPOINT support.
Each major stage saves results; on resume, completed stages are skipped.

Checkpoint stages:
  1. preprocess  — X1, X2 (after SVD), pos
  2. kernel      — lbds, U (eigendecomposition of Matern kernel)
  3. params      — W_hat_list, sigma_hat_sqr_list, gamma_hat (per gamma iteration)
  4. posterior   — z (joint embedding)
  5. clustering  — final adata with clustering labels

Usage:
    python run_SMOPCA_ckpt.py \
      --RNA_path ... --ATAC_path ... --save_path ... \
      --dataset Mouse_Brain --cluster_nums 18 \
      --ckpt_dir /path/to/checkpoints
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

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')


# =====================================================================
# Checkpoint helpers
# =====================================================================

def ckpt_path(ckpt_dir, name):
    return os.path.join(ckpt_dir, f'ckpt_{name}.npz')


def ckpt_exists(ckpt_dir, name):
    return os.path.isfile(ckpt_path(ckpt_dir, name))


def save_ckpt(ckpt_dir, name, **arrays):
    """Save numpy arrays to checkpoint."""
    path = ckpt_path(ckpt_dir, name)
    np.savez(path, **arrays)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  [CKPT] Saved {name} ({size_mb:.1f} MB): {path}")


def load_ckpt(ckpt_dir, name):
    """Load checkpoint, returns dict of arrays."""
    path = ckpt_path(ckpt_dir, name)
    data = dict(np.load(path, allow_pickle=True))
    print(f"  [CKPT] Loaded {name}: {path}")
    return data


# =====================================================================
# Monkey-patch: SMOPCA.estimateParams with wider bound search + fallback
# (same as run_SMOPCA_mt_tmp.py but with per-iteration checkpoint)
# =====================================================================

def _patched_estimateParams_ckpt(self, ckpt_dir,
                                  iterations_gamma=10, iterations_sigma_W=20,
                                  tol_gamma=1e-2, tol_sigma=1e-5,
                                  estimate_gamma=False, gamma_init=1,
                                  gamma_bound=(0.1, 5),
                                  sigma_init_list=(), sigma_xtol_list=(),
                                  gamma_tol=0.1):
    """estimateParams with checkpointing after each gamma iteration."""
    assert len(sigma_init_list) == len(sigma_xtol_list) == self.modality_num
    logger = logging.getLogger(__name__)
    logger.info("start estimating parameters (checkpoint-enabled)...")

    gamma = gamma_init

    # Check if kernel checkpoint exists (from buildKernel)
    if ckpt_exists(ckpt_dir, 'kernel'):
        data = load_ckpt(ckpt_dir, 'kernel')
        self.lbds = data['lbds']
        self.U = data['U']
        gamma = float(data['gamma'])
        logger.info(f"Kernel loaded from checkpoint (gamma={gamma:.5f})")
    else:
        self.buildKernel(length_scale=gamma)
        save_ckpt(ckpt_dir, 'kernel', lbds=self.lbds, U=self.U,
                  gamma=np.array(gamma))

    for iter1 in range(iterations_gamma):
        # Check if this gamma iteration was already completed
        iter_ckpt_name = f'params_gamma_iter{iter1}'
        if ckpt_exists(ckpt_dir, iter_ckpt_name):
            data = load_ckpt(ckpt_dir, iter_ckpt_name)
            self.W_hat_list = list(data['W_hat_list'])
            self.sigma_hat_sqr_list = list(data['sigma_hat_sqr_list'])
            if 'gamma_hat' in data:
                gamma_hat = float(data['gamma_hat'])
                gamma = gamma_hat
                # Load kernel for this gamma
                kernel_ckpt = f'kernel_gamma{iter1}'
                if ckpt_exists(ckpt_dir, kernel_ckpt):
                    kdata = load_ckpt(ckpt_dir, kernel_ckpt)
                    self.lbds = kdata['lbds']
                    self.U = kdata['U']
            logger.info(f"Gamma iteration {iter1} loaded from checkpoint")
            if not estimate_gamma:
                break
            continue

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
                    return part1 + part2 + part3

                # Wide bound search using logspace
                if bound_list[modality] is None:
                    search_grid = np.logspace(-4, 2, 2000)
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
                        bound_list[modality] = None
                        logger.warning("sigma{} no bracket found, will use minimize_scalar fallback".format(modality + 1))

                try:
                    if bound_list[modality] is not None:
                        sigma_hat_sqr = brentq(jac_sigma_sqr, bound_list[modality][0], bound_list[modality][1],
                                               xtol=sigma_xtol_list[modality])
                    else:
                        raise ValueError("No bracket available")
                except ValueError as e:
                    logger.warning("brentq failed for sigma{} ({}), falling back to minimize_scalar".format(modality + 1, e))
                    ret = scipy.optimize.minimize_scalar(
                        lambda s: abs(jac_sigma_sqr(s)),
                        method="Bounded", bounds=(1e-6, 100.0),
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

        # Save checkpoint for this gamma iteration
        ckpt_data = {
            'W_hat_list': np.array(self.W_hat_list, dtype=object),
            'sigma_hat_sqr_list': np.array(self.sigma_hat_sqr_list),
        }

        if not estimate_gamma:
            save_ckpt(ckpt_dir, iter_ckpt_name, **ckpt_data)
            break

        # Gamma optimization (each eval requires full eigh!)
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

        # Rebuild kernel with new gamma and save
        self.buildKernel(length_scale=gamma_hat)
        save_ckpt(ckpt_dir, f'kernel_gamma{iter1}',
                  lbds=self.lbds, U=self.U, gamma=np.array(gamma_hat))

        ckpt_data['gamma_hat'] = np.array(gamma_hat)
        save_ckpt(ckpt_dir, iter_ckpt_name, **ckpt_data)

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
    if estimate_gamma and hasattr(self, 'gamma_hat') and self.gamma_hat is not None:
        logger.info("gamma_hat = {:.5f}".format(self.gamma_hat))


# =====================================================================
# Main with checkpointing
# =====================================================================

def main(args):
    total_start_time = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    fix_seed(args.seed)

    ckpt_dir = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoint dir: {ckpt_dir}")

    # ==================================================================
    # Stage 1: Preprocess (load, normalize, SVD)
    # ==================================================================
    if ckpt_exists(ckpt_dir, 'preprocess'):
        print("\n[Stage 1] Loading preprocessed data from checkpoint...")
        data = load_ckpt(ckpt_dir, 'preprocess')
        X1 = data['X1']
        X2 = data['X2']
        pos = data['pos']
        # Also need obs metadata for final adata
        adata_rna = sc.read_h5ad(args.RNA_path)
        adata_rna.var_names_make_unique()
        if args.ATAC_path:
            adata_other = sc.read_h5ad(args.ATAC_path)
        else:
            adata_other = sc.read_h5ad(args.ADT_path)
        adata_other.var_names_make_unique()
        common_cells = adata_rna.obs_names.intersection(adata_other.obs_names)
        adata_rna = adata_rna[common_cells].copy()
        modality_name = "Epigenome" if args.ATAC_path else "Proteome"
        print(f"  X1: {X1.shape}, X2: {X2.shape}, pos: {pos.shape}")
    else:
        print("\n[Stage 1] Preprocessing data...")
        adata_rna = sc.read_h5ad(args.RNA_path)
        adata_rna.var_names_make_unique()

        if args.ADT_path:
            adata_other = sc.read_h5ad(args.ADT_path)
            modality_name = "Proteome"
        elif args.ATAC_path:
            adata_other = sc.read_h5ad(args.ATAC_path)
            modality_name = "Epigenome"
        else:
            raise ValueError("Please provide either --ADT_path or --ATAC_path.")
        adata_other.var_names_make_unique()

        common_cells = adata_rna.obs_names.intersection(adata_other.obs_names)
        print(f"  Common cells: {len(common_cells)}")
        adata_rna = adata_rna[common_cells].copy()
        adata_other = adata_other[common_cells].copy()

        # Pseudo spatial coords
        for adata, name in [(adata_rna, "RNA"), (adata_other, modality_name)]:
            if "spatial" not in adata.obsm.keys():
                print(f"  Warning: No spatial coordinates in {name}, generating pseudo-spatial...")
                sc.pp.neighbors(adata)
                sc.tl.umap(adata)
                adata.obsm["spatial"] = adata.obsm["X_umap"].copy()

        # Feature preprocessing
        sc.pp.normalize_total(adata_rna, target_sum=1e4)
        sc.pp.log1p(adata_rna)
        sc.pp.highly_variable_genes(adata_rna, n_top_genes=3000)
        adata_rna = adata_rna[:, adata_rna.var["highly_variable"]].copy()
        sc.pp.scale(adata_other)

        X1 = adata_rna.X.A if hasattr(adata_rna.X, "A") else adata_rna.X
        X2 = adata_other.X.A if hasattr(adata_other.X, "A") else adata_other.X
        pos = adata_rna.obsm["spatial"]

        print(f"  Input features: RNA {X1.shape}, {modality_name} {X2.shape}, pos {pos.shape}")

        # Dimensionality reduction for large feature counts
        ATAC_DIM_THRESHOLD = 5000
        ATAC_DIM_TARGET = 3000
        if X2.shape[1] > ATAC_DIM_THRESHOLD:
            print(f"  [DIM REDUCTION] {modality_name}: {X2.shape[1]} → {ATAC_DIM_TARGET} via TruncatedSVD...")
            t0 = time.time()
            X2_sparse = scipy.sparse.csr_matrix(X2) if not scipy.sparse.issparse(X2) else X2
            svd = TruncatedSVD(n_components=ATAC_DIM_TARGET, random_state=args.seed)
            X2 = svd.fit_transform(X2_sparse)
            print(f"  [DIM REDUCTION] Done in {time.time()-t0:.1f}s: {X2.shape}, "
                  f"explained var = {svd.explained_variance_ratio_.sum():.4f}")

        save_ckpt(ckpt_dir, 'preprocess', X1=X1, X2=X2, pos=pos)

    # ==================================================================
    # Stage 2: Initialize SMOPCA + build kernel (eigh is the bottleneck)
    # ==================================================================
    print(f"\n[Stage 2] Initializing SMOPCA model...")
    smopca = model.SMOPCA(
        Y_list=[X1.T, X2.T],
        Z_dim=args.Z_dim,
        pos=pos,
        intercept=False,
        omics_weight=False
    )

    # ==================================================================
    # Stage 3: Estimate parameters (with per-iteration checkpoints)
    # ==================================================================
    if ckpt_exists(ckpt_dir, 'posterior'):
        print("\n[Stage 3] Skipping estimateParams (posterior already computed)")
    else:
        print(f"\n[Stage 3] Estimating parameters (with checkpoints)...")
        t0 = time.time()
        _patched_estimateParams_ckpt(
            smopca, ckpt_dir,
            sigma_init_list=(1, 1),
            tol_sigma=2e-5,
            sigma_xtol_list=(1e-6, 1e-6),
            gamma_init=1,
            estimate_gamma=True
        )
        print(f"  estimateParams done in {time.time()-t0:.1f}s")

    # ==================================================================
    # Stage 4: Calculate posterior (joint embedding)
    # ==================================================================
    if ckpt_exists(ckpt_dir, 'posterior'):
        print("\n[Stage 4] Loading posterior from checkpoint...")
        data = load_ckpt(ckpt_dir, 'posterior')
        z = data['z']
    else:
        print("\n[Stage 4] Calculating posterior...")
        t0 = time.time()
        z = smopca.calculatePosterior()
        print(f"  Posterior done in {time.time()-t0:.1f}s, shape: {z.shape}")
        save_ckpt(ckpt_dir, 'posterior', z=z)

    # ==================================================================
    # Stage 5: Build adata + clustering + save
    # ==================================================================
    print(f"\n[Stage 5] Building AnnData and clustering...")
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
    adata.var_names = [f"SMOPCA_{i}" for i in range(adata.shape[1])]

    train_time = time.time() - total_start_time
    adata.uns.update({
        "train_time": train_time,
        "integration_type": "horizontal",
        "method": "SMOPCA"
    })

    print("  Running clustering...")
    for tool in ["mclust", "louvain", "leiden", "kmeans"]:
        print(f"    {tool}...")
        adata = universal_clustering(
            adata, n_clusters=args.cluster_nums,
            used_obsm="SMOPCA", method=tool, key=tool, use_pca=False
        )

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    adata.write(args.save_path)
    print(f"  Saved: {args.save_path}")

    # Timing
    total_time = time.time() - total_start_time
    timing_info = {
        "method": "SMOPCA", "dataset": args.dataset,
        "integration_type": "horizontal",
        "modality": "RNA_ATAC" if args.ATAC_path else "RNA_ADT",
        "n_cells": adata.n_obs,
        "embedding_shape": list(adata.obsm["SMOPCA"].shape),
        "total_time_s": round(total_time, 2),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    timing_path = args.save_path.replace(".h5ad", "_timing_info.json")
    with open(timing_path, "w") as f:
        json.dump(timing_info, f, indent=2)
    print(f"  Timing: {timing_path}")

    # UMAP
    try:
        sc.pp.neighbors(adata, n_neighbors=30, use_rep="X")
        sc.tl.umap(adata)
        sc.pl.umap(adata, color=["kmeans"], title="SMOPCA clustering", show=False)
        plt_path = args.save_path.replace(".h5ad", "_umap.png")
        plt.savefig(plt_path, dpi=300, bbox_inches="tight")
        print(f"  UMAP: {plt_path}")
    except Exception as e:
        print(f"  UMAP skipped: {e}")

    print(f"\nTotal time: {total_time:.1f}s ({total_time/3600:.2f}h)")


if __name__ == "__main__":
    os.environ['R_HOME'] = '/home/users/nus/e1724738/miniconda3/envs/_Proj1_1/lib/R'
    os.environ['OMP_NUM_THREADS'] = '128'
    os.environ['MKL_NUM_THREADS'] = '128'
    os.environ['NUMEXPR_NUM_THREADS'] = '128'
    os.environ['OPENBLAS_NUM_THREADS'] = '128'

    parser = argparse.ArgumentParser(description="SMOPCA with checkpointing")
    parser.add_argument("--data_type", type=str, default="fusion")
    parser.add_argument("--method", type=str, default="SMOPCA")
    parser.add_argument("--RNA_path", type=str, required=True)
    parser.add_argument("--ADT_path", type=str, default="")
    parser.add_argument("--ATAC_path", type=str, default="")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--cluster_nums", type=int, default=7)
    parser.add_argument("--Z_dim", type=int, default=20)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Directory to store/load checkpoints")
    args = parser.parse_args()

    main(args)
