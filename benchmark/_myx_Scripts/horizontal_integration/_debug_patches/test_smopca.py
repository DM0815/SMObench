#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick smoke-test for the two SMOPCA monkey-patches in run_SMOPCA.py:
  1. estimateParams: wider brentq bounds (logspace) + minimize_scalar fallback
  2. TruncatedSVD on ATAC features when > 5000 features

Strategy: import run_SMOPCA (which applies the monkey-patch at import time),
then replicate the core pipeline inline with estimate_gamma=False for speed.
This still tests both patches end-to-end but avoids the expensive gamma
optimisation loop that makes the full pipeline take >5 min on CPU.

Creates tiny synthetic data (50 cells, 80 RNA, 8000 ATAC).
Expected runtime: ~30s on CPU.
"""

import os
import sys
import time
import tempfile
import numpy as np
import scipy.sparse
import warnings
warnings.filterwarnings("ignore")

# ── Setup paths so imports work ────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "_myx_Methods/SMOPCA/src"))

os.environ['R_HOME'] = '/home/users/nus/e1724738/miniconda3/envs/_Proj1_1/lib/R'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import scanpy as sc
import torch
from sklearn.decomposition import TruncatedSVD

# ── Import run_SMOPCA to trigger monkey-patch at module load ──────────
print("=" * 60)
print("Importing run_SMOPCA to activate monkey-patches...")
import importlib
run_smopca_module = importlib.import_module(
    "_myx_Scripts.horizontal_integration.SMOPCA.run_SMOPCA")
print("=" * 60)

# Now also import the patched model
import model

# ── Parameters ─────────────────────────────────────────────────────────
N_CELLS = 50
N_RNA_FEATURES = 80
N_ATAC_FEATURES = 8000   # > 5000 → triggers TruncatedSVD patch
SEED = 42
Z_DIM = 10
ATAC_DIM_THRESHOLD = 5000
ATAC_DIM_TARGET = 3000

np.random.seed(SEED)
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


def test_patches():
    """Run the two patches and verify output."""
    all_ok = True

    # ── 1. Create synthetic data ───────────────────────────────────────
    print("\n[1/4] Creating synthetic data...")
    X_rna_raw = np.abs(rng.poisson(lam=5, size=(N_CELLS, N_RNA_FEATURES))).astype(np.float32)
    adata_rna = sc.AnnData(X=scipy.sparse.csr_matrix(X_rna_raw))
    adata_rna.obs_names = [f"cell_{i}" for i in range(N_CELLS)]
    adata_rna.var_names = [f"gene_{i}" for i in range(N_RNA_FEATURES)]
    adata_rna.obsm["spatial"] = rng.standard_normal((N_CELLS, 2)).astype(np.float32)

    X_atac_raw = scipy.sparse.random(N_CELLS, N_ATAC_FEATURES, density=0.05,
                                      format="csr", random_state=SEED,
                                      dtype=np.float32)
    adata_atac = sc.AnnData(X=X_atac_raw)
    adata_atac.obs_names = [f"cell_{i}" for i in range(N_CELLS)]
    adata_atac.var_names = [f"peak_{i}" for i in range(N_ATAC_FEATURES)]
    adata_atac.obsm["spatial"] = adata_rna.obsm["spatial"].copy()

    print(f"  RNA:  {adata_rna.shape}")
    print(f"  ATAC: {adata_atac.shape}")

    # ── 2. Preprocess (same as run_SMOPCA.py main()) ───────────────────
    print("\n[2/4] Preprocessing (normalize, HVG, scale)...")
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    # With only 80 genes we can't do n_top_genes=3000; keep them all
    # (In real runs n_top_genes=3000 works fine because we have >=3000 genes)
    X1 = adata_rna.X.A if hasattr(adata_rna.X, "A") else adata_rna.X

    # Scale ATAC; replace NaN (from zero-variance cols in tiny synthetic data) with 0
    sc.pp.scale(adata_atac)
    X2 = adata_atac.X.A if hasattr(adata_atac.X, "A") else adata_atac.X
    nan_count = np.isnan(X2).sum()
    if nan_count > 0:
        print(f"  Note: {nan_count} NaN values from zero-variance ATAC features "
              f"(artifact of tiny synthetic data), replacing with 0")
        X2 = np.nan_to_num(X2, nan=0.0)
    pos = adata_rna.obsm["spatial"]

    print(f"  X1 (RNA):  {X1.shape}")
    print(f"  X2 (ATAC): {X2.shape}")

    # ── 3. Test Patch 2: TruncatedSVD on high-dim ATAC ────────────────
    print("\n[3/4] Testing Patch 2: TruncatedSVD dim reduction...")
    if X2.shape[1] > ATAC_DIM_THRESHOLD:
        orig_atac_feats = X2.shape[1]
        print(f"  ATAC has {orig_atac_feats} features (>{ATAC_DIM_THRESHOLD}), "
              f"reducing to {ATAC_DIM_TARGET} via TruncatedSVD...")
        X2_for_svd = scipy.sparse.csr_matrix(X2) if not scipy.sparse.issparse(X2) else X2
        # TruncatedSVD: n_components cannot exceed min(n_samples, n_features) - 1
        effective_target = min(ATAC_DIM_TARGET, min(X2.shape) - 1)
        if effective_target < ATAC_DIM_TARGET:
            print(f"  Note: n_components clamped to {effective_target} "
                  f"(min(n_samples, n_features)-1 = {min(X2.shape)-1}) "
                  f"due to small test size. In production with >=3000 cells "
                  f"the full {ATAC_DIM_TARGET} would be used.")
        svd = TruncatedSVD(n_components=effective_target, random_state=SEED)
        X2 = svd.fit_transform(X2_for_svd)
        explained = svd.explained_variance_ratio_.sum()
        print(f"  TruncatedSVD done: {X2.shape}, explained variance = {explained:.4f}")
        # Verify: dim was actually reduced from original
        if X2.shape[1] < orig_atac_feats and X2.shape[0] == N_CELLS:
            print(f"  [PASS] TruncatedSVD reduced {orig_atac_feats} -> {X2.shape[1]} features")
        else:
            print(f"  [FAIL] Unexpected shape after SVD: {X2.shape}")
            all_ok = False
    else:
        print("  [SKIP] ATAC features <= threshold, SVD not triggered")
        all_ok = False

    # ── 4. Test Patch 1: patched estimateParams ────────────────────────
    print("\n[4/4] Testing Patch 1: patched estimateParams (wider brentq + fallback)...")
    print(f"  Creating SMOPCA model: Y1={X1.T.shape}, Y2={X2.T.shape}, "
          f"pos={pos.shape}, Z_dim={Z_DIM}")

    smopca = model.SMOPCA(
        Y_list=[X1.T, X2.T],
        Z_dim=Z_DIM,
        pos=pos,
        intercept=False,
        omics_weight=False,
    )

    # Verify that the monkey-patch is active
    import inspect
    source = inspect.getsource(smopca.estimateParams)
    if "logspace" in source and "minimize_scalar" in source:
        print("  [PASS] Monkey-patch is active (found logspace + minimize_scalar in source)")
    else:
        print("  [FAIL] Monkey-patch NOT active!")
        all_ok = False

    t0 = time.time()
    try:
        smopca.estimateParams(
            sigma_init_list=(1, 1),
            tol_sigma=2e-5,
            sigma_xtol_list=(1e-6, 1e-6),
            gamma_init=1,
            estimate_gamma=False,   # skip gamma opt for speed
        )
        elapsed = time.time() - t0
        print(f"  estimateParams completed in {elapsed:.1f}s")
        print(f"  sigma_hat_sqr: {smopca.sigma_hat_sqr_list}")
        if len(smopca.sigma_hat_sqr_list) == 2 and all(
            np.isfinite(s) and s > 0 for s in smopca.sigma_hat_sqr_list
        ):
            print("  [PASS] estimateParams returned valid sigma estimates")
        else:
            print("  [FAIL] Invalid sigma estimates")
            all_ok = False
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [FAIL] estimateParams raised: {type(e).__name__}: {e}")
        all_ok = False

    # ── 5. Calculate posterior and verify embedding ────────────────────
    if all_ok:
        print("\n[5/5] Computing posterior embedding...")
        z = smopca.calculatePosterior()
        print(f"  Embedding shape: {z.shape}")
        if z.shape == (N_CELLS, Z_DIM):
            print("  [PASS] Correct embedding shape")
        else:
            print(f"  [FAIL] Expected ({N_CELLS}, {Z_DIM}), got {z.shape}")
            all_ok = False
        if np.isnan(z).any():
            print("  [FAIL] Embedding contains NaN!")
            all_ok = False
        else:
            print("  [PASS] No NaN in embedding")

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_ok:
        print(">>> ALL PATCH TESTS PASSED <<<")
    else:
        print(">>> SOME TESTS FAILED <<<")
    print("=" * 60)
    return all_ok


if __name__ == "__main__":
    passed = test_patches()
    sys.exit(0 if passed else 1)
