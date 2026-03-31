#!/usr/bin/env python
"""
Test SpaFusion AMP monkey-patch in run_SpaFusion.py.

Tests:
1. Source-level verification that AMP patches exist
2. AMP autocast + GradScaler basic functionality (requires GPU)
3. Small end-to-end run with synthetic data (requires GPU)
"""

import os
import sys
import traceback
import numpy as np
import scipy.sparse as sp

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
os.chdir(PROJECT_ROOT)

import torch


def header(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def test_source_verification():
    """Test 1: Verify that run_SpaFusion.py contains AMP patches."""
    header("TEST 1: Source-Level Verification")
    try:
        src_path = os.path.join(
            PROJECT_ROOT,
            "_myx_Scripts/horizontal_integration/SpaFusion/run_SpaFusion.py"
        )
        with open(src_path, 'r') as f:
            src = f.read()

        checks = {
            "AMP autocast import": "torch.cuda.amp",
            "GradScaler": "GradScaler",
            "pre_train monkey-patch": "def _amp_pre_train(",
            "train monkey-patch": "def _amp_train(",
            "Patch applied to SpaFusion module": "spafusion_train.pre_train = _amp_pre_train",
        }

        all_found = True
        for name, pattern in checks.items():
            found = pattern in src
            status = "OK" if found else "MISSING"
            print(f"  [{status}] {name}")
            if not found:
                all_found = False

        if all_found:
            print("[PASS] All AMP patches verified in source.")
        else:
            print("[FAIL] Some patches missing from source!")
        return all_found
    except Exception as e:
        print(f"[FAIL] {e}")
        traceback.print_exc()
        return False


def test_amp_basic():
    """Test 2: AMP autocast + GradScaler basic functionality."""
    header("TEST 2: AMP Basic Functionality")
    if not torch.cuda.is_available():
        print("  [SKIP] No GPU available")
        return True  # not a failure, just skip

    try:
        device = torch.device('cuda:0')

        # Simple model
        model = torch.nn.Linear(64, 32).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.cuda.amp.GradScaler()

        x = torch.randn(100, 64, device=device)
        target = torch.randn(100, 32, device=device)

        # Forward with autocast
        with torch.cuda.amp.autocast():
            out = model(x)
            loss = torch.nn.functional.mse_loss(out, target)

        print(f"  Loss dtype under autocast: {loss.dtype}")
        assert loss.dtype == torch.float32, "Loss should be float32"

        # Backward with scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        print(f"  Loss value: {loss.item():.4f}")
        assert not torch.isnan(loss), "Loss is NaN!"

        print("[PASS] AMP autocast + GradScaler works correctly.")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        traceback.print_exc()
        return False


def test_spafusion_e2e():
    """Test 3: End-to-end run with small synthetic data on GPU."""
    header("TEST 3: SpaFusion E2E with Synthetic Data")
    if not torch.cuda.is_available():
        print("  [SKIP] No GPU available")
        return True

    try:
        import subprocess
        import tempfile
        import anndata as ad

        # Create small synthetic data
        n_cells = 200
        n_rna = 100
        n_adt = 20

        rng = np.random.default_rng(42)

        cell_names = [f"cell_{i}" for i in range(n_cells)]
        batch_labels = ["batch_A"] * 100 + ["batch_B"] * 100

        rna_X = sp.random(n_cells, n_rna, density=0.3, format="csr",
                          random_state=42, dtype=np.float32)
        rna_X.data = np.abs(rna_X.data) * 10

        adata_rna = ad.AnnData(X=rna_X, obs={"batch": batch_labels})
        adata_rna.obs_names = cell_names
        adata_rna.var_names = [f"Gene_{i}" for i in range(n_rna)]
        adata_rna.obsm["spatial"] = rng.standard_normal((n_cells, 2)).astype(np.float32)

        adt_X = np.abs(rng.standard_normal((n_cells, n_adt)).astype(np.float32)) * 5
        adata_adt = ad.AnnData(X=adt_X, obs={"batch": batch_labels})
        adata_adt.obs_names = cell_names
        adata_adt.var_names = [f"Protein_{i}" for i in range(n_adt)]
        adata_adt.obsm["spatial"] = adata_rna.obsm["spatial"].copy()

        tmpdir = tempfile.mkdtemp(prefix="spafusion_test_")
        rna_path = os.path.join(tmpdir, "test_rna.h5ad")
        adt_path = os.path.join(tmpdir, "test_adt.h5ad")
        save_path = os.path.join(tmpdir, "test_result.h5ad")

        adata_rna.write_h5ad(rna_path)
        adata_adt.write_h5ad(adt_path)

        print(f"  Saved test data: RNA {adata_rna.shape}, ADT {adata_adt.shape}")
        print(f"  Running SpaFusion with AMP patch...")

        cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "_myx_Scripts/horizontal_integration/SpaFusion/run_SpaFusion.py"),
            "--RNA_path", rna_path,
            "--ADT_path", adt_path,
            "--save_path", save_path,
            "--dataset", "test",
            "--cluster_nums", "3",
            "--device", "cuda:0",
            "--pretrain_epoch", "5",
            "--train_epoch", "5",
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=PROJECT_ROOT,
        )

        if result.returncode != 0:
            print(f"  STDOUT:\n{result.stdout[-2000:]}")
            print(f"  STDERR:\n{result.stderr[-2000:]}")
            print(f"[FAIL] SpaFusion exited with code {result.returncode}")
            return False

        if os.path.exists(save_path):
            result_adata = ad.read_h5ad(save_path)
            print(f"  Result: {result_adata.shape}")
            if "SpaFusion" in result_adata.obsm:
                emb = result_adata.obsm["SpaFusion"]
                print(f"  Embedding shape: {emb.shape}")
                if np.isnan(emb).any():
                    print(f"[FAIL] Embedding contains NaN!")
                    return False
                print(f"[PASS] SpaFusion E2E completed, embedding shape {emb.shape}")
                return True
            else:
                print(f"  obsm keys: {list(result_adata.obsm.keys())}")
                print(f"[FAIL] No 'SpaFusion' key in obsm")
                return False
        else:
            print(f"  STDOUT (last 2000 chars):\n{result.stdout[-2000:]}")
            print(f"[FAIL] Output file not created")
            return False

    except subprocess.TimeoutExpired:
        print(f"[FAIL] SpaFusion timed out (>300s)")
        return False
    except Exception as e:
        print(f"[FAIL] {e}")
        traceback.print_exc()
        return False


def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = {}
    results['source_verification'] = test_source_verification()
    results['amp_basic'] = test_amp_basic()
    results['spafusion_e2e'] = test_spafusion_e2e()

    header("SUMMARY")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: [{status}]")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll tests PASSED.")
    else:
        print("\nSome tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
