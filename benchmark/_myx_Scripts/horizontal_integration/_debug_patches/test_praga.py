"""
Test the two PRAGA monkey-patches: stratified_subsample() and knn_propagate_embeddings().

Since PRAGA's Train loop requires CUDA AMP and is expensive, we test the patches
in isolation:
  1. stratified_subsample - verify proportional sampling, index correctness
  2. knn_propagate_embeddings - verify shape, placement of known embeddings,
     reasonable quality of KNN-predicted embeddings

Then we do a lightweight "integration smoke-test" that exercises the PRAGA
preprocessing pipeline (everything before trainer.train()) to verify the
subsampling wiring is plumbed correctly.
"""

import os
import sys
import numpy as np
import scipy.sparse as sp
import anndata as ad

# ── Set up paths ──────────────────────────────────────────────────────
PROJECT_ROOT = "/home/users/nus/e1724738/_main/_private/NUS/_Proj1/SMOBench-CLEAN"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "_myx_Scripts/horizontal_integration/PRAGA"))
sys.path.insert(0, PROJECT_ROOT)

# Import the functions under test directly from run_praga module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "run_praga",
    os.path.join(PROJECT_ROOT, "_myx_Scripts/horizontal_integration/PRAGA/run_praga.py"),
)
run_praga = importlib.util.module_from_spec(spec)

# We need torch imported before loading the module
import torch

spec.loader.exec_module(run_praga)

stratified_subsample = run_praga.stratified_subsample
knn_propagate_embeddings = run_praga.knn_propagate_embeddings
MAX_CELLS = run_praga.MAX_CELLS

SEED = 42
rng = np.random.RandomState(SEED)


# =====================================================================
# Helper: build synthetic AnnData pair
# =====================================================================
def make_synthetic(n_cells=500, n_rna=200, n_adt=30, batch_sizes=None):
    """Return (adata_rna, adata_adt) with shared obs_names."""
    if batch_sizes is None:
        batch_sizes = {"batch_A": n_cells * 6 // 10,
                       "batch_B": n_cells - n_cells * 6 // 10}
    assert sum(batch_sizes.values()) == n_cells

    cell_names = [f"cell_{i:05d}" for i in range(n_cells)]
    batch_labels = []
    for b, sz in batch_sizes.items():
        batch_labels.extend([b] * sz)

    spatial = rng.randn(n_cells, 2).astype(np.float32)

    rna_X = sp.random(n_cells, n_rna, density=0.3, format="csr",
                      random_state=rng, dtype=np.float32)
    rna_X.data = np.abs(rna_X.data) * 10

    adata_rna = ad.AnnData(
        X=rna_X,
        obs={"batch": batch_labels},
    )
    adata_rna.obs_names = cell_names
    adata_rna.var_names = [f"Gene_{i}" for i in range(n_rna)]
    adata_rna.obsm["spatial"] = spatial.copy()

    adt_X = np.abs(rng.randn(n_cells, n_adt).astype(np.float32)) * 5
    adata_adt = ad.AnnData(
        X=adt_X,
        obs={"batch": batch_labels},
    )
    adata_adt.obs_names = cell_names
    adata_adt.var_names = [f"Protein_{i}" for i in range(n_adt)]
    adata_adt.obsm["spatial"] = spatial.copy()

    return adata_rna, adata_adt


# =====================================================================
# TEST 1: stratified_subsample
# =====================================================================
def test_stratified_subsample():
    print("\n" + "=" * 72)
    print("TEST 1: stratified_subsample()")
    print("=" * 72)

    n_cells = 500
    max_cells = 100
    batch_sizes = {"batch_A": 300, "batch_B": 200}
    adata_rna, adata_adt = make_synthetic(n_cells=n_cells, batch_sizes=batch_sizes)

    rna_sub, adt_sub, kept_idx = stratified_subsample(
        adata_rna, adata_adt, max_cells=max_cells, seed=SEED
    )

    # 1a. Output shapes
    assert rna_sub.n_obs == adt_sub.n_obs, "RNA and ADT subsampled to different sizes"
    assert rna_sub.n_obs <= max_cells, f"Subsampled {rna_sub.n_obs} > max_cells={max_cells}"
    assert len(kept_idx) == rna_sub.n_obs
    print(f"  [OK] Subsampled to {rna_sub.n_obs} cells (<= {max_cells})")

    # 1b. kept_idx are valid and sorted
    assert np.all(kept_idx >= 0) and np.all(kept_idx < n_cells)
    assert np.all(np.diff(kept_idx) > 0), "kept_idx not sorted"
    print(f"  [OK] kept_idx valid and sorted")

    # 1c. Batch proportions preserved (roughly 60/40)
    sub_batches = rna_sub.obs["batch"].values
    n_a = np.sum(sub_batches == "batch_A")
    n_b = np.sum(sub_batches == "batch_B")
    ratio = n_a / (n_a + n_b)
    assert 0.45 < ratio < 0.75, f"Batch ratio {ratio:.2f} not close to 0.6"
    print(f"  [OK] Batch ratio: A={n_a}, B={n_b}, ratio={ratio:.2f} (expected ~0.6)")

    # 1d. Obs names match
    orig_names = np.array(adata_rna.obs_names)
    sub_names = np.array(rna_sub.obs_names)
    assert np.array_equal(sub_names, orig_names[kept_idx])
    print(f"  [OK] Obs names match original[kept_idx]")

    # 1e. No batch column -> should still work
    adata_rna_nobatch, adata_adt_nobatch = make_synthetic(n_cells=200, batch_sizes={"all": 200})
    del adata_rna_nobatch.obs["batch"]
    del adata_adt_nobatch.obs["batch"]
    rna_s2, adt_s2, idx2 = stratified_subsample(
        adata_rna_nobatch, adata_adt_nobatch, max_cells=50, seed=SEED
    )
    assert rna_s2.n_obs <= 50
    print(f"  [OK] Works without batch column ({rna_s2.n_obs} cells)")

    print("  --- TEST 1 PASSED ---")


# =====================================================================
# TEST 2: knn_propagate_embeddings
# =====================================================================
def test_knn_propagate_embeddings():
    print("\n" + "=" * 72)
    print("TEST 2: knn_propagate_embeddings()")
    print("=" * 72)

    n_total = 400
    n_sub = 100
    dim_feat = 50
    dim_emb = 16
    k = 10

    # Create feature space with 4 well-separated clusters.
    # Use purely directional separation (unit-norm cluster centres) so that
    # cosine distance in KNN can distinguish them easily.
    features_all = np.zeros((n_total, dim_feat), dtype=np.float32)
    true_emb = np.zeros((n_total, dim_emb), dtype=np.float32)
    for c in range(4):
        sl = slice(c * 100, (c + 1) * 100)
        # Random direction as cluster centre, then add small noise
        centre_feat = rng.randn(dim_feat).astype(np.float32)
        centre_feat /= np.linalg.norm(centre_feat) + 1e-8
        features_all[sl] = centre_feat[None, :] * 10 + rng.randn(100, dim_feat).astype(np.float32) * 0.3
        # Embedding: tight cluster with small variance
        centre_emb = rng.randn(dim_emb).astype(np.float32) * 3
        true_emb[sl] = centre_emb[None, :] + rng.randn(100, dim_emb).astype(np.float32) * 0.05

    # Subsample: take first 25 from each cluster
    kept_idx = np.sort(np.concatenate([np.arange(c*100, c*100+25) for c in range(4)]))
    assert len(kept_idx) == n_sub

    emb_sub = true_emb[kept_idx]
    features_sub = features_all[kept_idx]

    emb_full = knn_propagate_embeddings(
        emb_sub=emb_sub,
        features_sub=features_sub,
        features_all=features_all,
        kept_idx=kept_idx,
        n_total=n_total,
        k=k,
    )

    # 2a. Shape
    assert emb_full.shape == (n_total, dim_emb), f"Shape {emb_full.shape} != ({n_total}, {dim_emb})"
    print(f"  [OK] Output shape: {emb_full.shape}")

    # 2b. Known embeddings placed exactly
    assert np.allclose(emb_full[kept_idx], emb_sub, atol=1e-6), "Known embeddings not placed correctly"
    print(f"  [OK] Known embeddings placed exactly at kept_idx")

    # 2c. Held-out embeddings should be close to true (clusters are well-separated)
    held_mask = np.ones(n_total, dtype=bool)
    held_mask[kept_idx] = False
    mse = np.mean((emb_full[held_mask] - true_emb[held_mask]) ** 2)
    print(f"  [info] MSE of held-out predictions vs true: {mse:.4f}")
    # With well-separated clusters and k=10 KNN, MSE should be small
    assert mse < 1.0, f"MSE {mse:.4f} too high; KNN predictions seem wrong"
    print(f"  [OK] MSE < 1.0 -- predictions are reasonable")

    # 2d. Edge case: no held-out cells
    emb_full_all = knn_propagate_embeddings(
        emb_sub=true_emb,
        features_sub=features_all,
        features_all=features_all,
        kept_idx=np.arange(n_total),
        n_total=n_total,
        k=k,
    )
    assert np.allclose(emb_full_all, true_emb, atol=1e-6)
    print(f"  [OK] Edge case: all cells known -> returns exact embeddings")

    print("  --- TEST 2 PASSED ---")


# =====================================================================
# TEST 3: Integration smoke-test -- preprocessing + subsample wiring
# =====================================================================
def test_integration_wiring():
    """
    Simulate the main() flow up to the point where Train() would be called.
    We verify:
      - Preprocessing runs without error
      - Subsampling triggers when n > MAX_CELLS (we temporarily lower MAX_CELLS)
      - Data shapes are consistent after subsampling
    """
    print("\n" + "=" * 72)
    print("TEST 3: Integration wiring (preprocessing + subsample trigger)")
    print("=" * 72)

    import scanpy as sc
    from PRAGA.preprocess import pca, clr_normalize_each_cell, fix_seed

    n_cells = 200
    adata_rna, adata_adt = make_synthetic(n_cells=n_cells, n_rna=200, n_adt=30,
                                          batch_sizes={"bA": 120, "bB": 80})

    # Preprocess RNA (same as run_praga.main)
    fix_seed(SEED)
    sc.pp.filter_genes(adata_rna, min_cells=3)
    sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=min(3000, adata_rna.n_vars))
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.scale(adata_rna)
    adata_rna_high = adata_rna[:, adata_rna.var["highly_variable"]]
    n_comps = min(200, adata_rna_high.n_vars - 1, adata_rna_high.n_obs - 1)
    adata_rna.obsm["feat"] = pca(adata_rna_high, n_comps=n_comps)

    # Preprocess ADT
    adata_adt = clr_normalize_each_cell(adata_adt)
    sc.pp.scale(adata_adt)
    n_comps_adt = min(200, adata_adt.n_vars - 1)
    adata_adt.obsm["feat"] = pca(adata_adt, n_comps=n_comps_adt)

    print(f"  After preprocessing: RNA feat {adata_rna.obsm['feat'].shape}, ADT feat {adata_adt.obsm['feat'].shape}")

    # 3a. Subsample should NOT trigger when n_cells < MAX_CELLS
    need_sub = adata_rna.n_obs > MAX_CELLS
    assert not need_sub, f"Should not subsample {adata_rna.n_obs} cells (MAX_CELLS={MAX_CELLS})"
    print(f"  [OK] No subsampling needed for {adata_rna.n_obs} cells")

    # 3b. Subsample SHOULD trigger when we lower threshold
    temp_max = 80
    if adata_rna.n_obs > temp_max:
        feat_rna_full = adata_rna.obsm["feat"].copy()
        feat_adt_full = adata_adt.obsm["feat"].copy()
        feat_combined = np.concatenate([feat_rna_full, feat_adt_full], axis=1)

        rna_sub, adt_sub, kept_idx = stratified_subsample(
            adata_rna, adata_adt, max_cells=temp_max, seed=SEED
        )

        assert rna_sub.n_obs <= temp_max
        assert rna_sub.n_obs == adt_sub.n_obs
        assert "feat" in rna_sub.obsm and "feat" in adt_sub.obsm
        print(f"  [OK] Subsampling triggered: {adata_rna.n_obs} -> {rna_sub.n_obs}")

        # 3c. Simulate KNN propagation with fake embeddings
        dim_emb = 64
        fake_emb_sub = rng.randn(rna_sub.n_obs, dim_emb).astype(np.float32)
        feat_combined_sub = feat_combined[kept_idx]

        emb_full = knn_propagate_embeddings(
            emb_sub=fake_emb_sub,
            features_sub=feat_combined_sub,
            features_all=feat_combined,
            kept_idx=kept_idx,
            n_total=adata_rna.n_obs,
            k=15,
        )
        assert emb_full.shape == (adata_rna.n_obs, dim_emb)
        assert np.allclose(emb_full[kept_idx], fake_emb_sub, atol=1e-6)
        print(f"  [OK] KNN propagation: ({rna_sub.n_obs}, {dim_emb}) -> ({adata_rna.n_obs}, {dim_emb})")

    print("  --- TEST 3 PASSED ---")


# =====================================================================
# RUN ALL TESTS
# =====================================================================
if __name__ == "__main__":
    passed = 0
    failed = 0
    errors = []

    for test_fn in [test_stratified_subsample, test_knn_propagate_embeddings, test_integration_wiring]:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test_fn.__name__, e))
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 72)
    print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
    if errors:
        for name, err in errors:
            print(f"  FAIL: {name} -- {err}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
        sys.exit(0)
