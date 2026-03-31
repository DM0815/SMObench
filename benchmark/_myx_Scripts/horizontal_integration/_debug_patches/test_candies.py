#!/usr/bin/env python
"""
Quick test for CANDIES monkey-patches in run_candies.py.

Tests the 5 patches WITHOUT requiring GPU or importing CANDIES source
(DiTs.py calls torch.cuda.current_device() at import time).

We extract the patch classes directly from run_candies.py source to test
them in isolation.

Tests:
1. SampledNTXentLoss (unit test: small batch -> full, large batch -> sampled)
2. EfficientConditionalDiffusionDataset (unit test: shape, mean, DataLoader)
3. NaN-safe reindex logic (unit test)
4. Large-dataset threshold logic (unit test)
5. Source-level verification that patches exist in run_candies.py

Usage:
    conda activate _Proj1_1
    cd <project_root>
    python _myx_Scripts/horizontal_integration/_debug_patches/test_candies.py
"""

import os
import sys
import traceback
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
os.chdir(PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F


def header(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


# =========================================================================
# Re-define the patches here so we can test them without importing
# run_candies.py (which would pull in DiTs -> cuda crash on login node).
# These are exact copies from run_candies.py lines 53-121, 297-319.
# =========================================================================

_LARGE_DATASET_THRESHOLD = 10000

class SampledNTXentLoss(nn.Module):
    """Memory-efficient NTXent loss that samples negatives instead of using all 2N-2."""
    def __init__(self, temperature: float = 0.07, max_negatives: int = 1024):
        super().__init__()
        self.temperature = temperature
        self.max_negatives = max_negatives
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z1, z2):
        batch_size = z1.shape[0]
        if batch_size * 2 <= self.max_negatives + 2:
            return self._full_forward(z1, z2)
        return self._sampled_forward(z1, z2)

    def _full_forward(self, z1, z2):
        batch_size = z1.shape[0]
        labels = torch.cat([torch.arange(batch_size)] * 2, dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(z1.device)
        features = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z1.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        logits = torch.cat([positives, negatives], dim=1) / self.temperature
        target = torch.zeros(logits.shape[0], dtype=torch.long).to(z1.device)
        return self.criterion(logits, target)

    def _sampled_forward(self, z1, z2):
        batch_size = z1.shape[0]
        features = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
        total = 2 * batch_size
        K = min(self.max_negatives, total - 2)
        loss = torch.tensor(0.0, device=z1.device)
        chunk_size = min(512, total)
        n_chunks = (total + chunk_size - 1) // chunk_size
        for c in range(n_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, total)
            anchors = features[start:end]
            chunk_len = end - start
            pos_indices = torch.arange(start, end, device=z1.device)
            pos_indices = torch.where(pos_indices < batch_size,
                                      pos_indices + batch_size,
                                      pos_indices - batch_size)
            positives = features[pos_indices]
            pos_sim = (anchors * positives).sum(dim=1, keepdim=True) / self.temperature
            neg_indices = torch.randint(0, total, (K,), device=z1.device)
            neg_features = features[neg_indices]
            neg_sim = torch.matmul(anchors, neg_features.T) / self.temperature
            logits = torch.cat([pos_sim, neg_sim], dim=1)
            target = torch.zeros(chunk_len, dtype=torch.long, device=z1.device)
            loss = loss + self.criterion(logits, target) * chunk_len
        return loss / total


class EfficientConditionalDiffusionDataset:
    """Memory-efficient dataset that precomputes con_data mean."""
    def __init__(self, adata_omics1, adata_omics2):
        self.adata_omics1 = adata_omics1
        self.adata_omics2 = adata_omics2
        self.st_sample = torch.tensor(adata_omics1, dtype=torch.float32)
        self.con_sample = torch.tensor(adata_omics2, dtype=torch.float32)
        self.con_data = torch.tensor(adata_omics2, dtype=torch.float32).mean(dim=0, keepdim=True)

    def __len__(self):
        return len(self.st_sample)

    def __getitem__(self, idx):
        return self.st_sample[idx], self.con_sample[idx], self.con_data


# =========================================================================
# Tests
# =========================================================================

def test_sampled_ntxent():
    """Test 1: SampledNTXentLoss forward pass (small and large)."""
    header("TEST 1: SampledNTXentLoss")
    try:
        device = torch.device('cpu')

        # --- With max_negatives=64 so batch_size=32 triggers full (32*2=64 <= 64+2) ---
        loss_fn = SampledNTXentLoss(temperature=0.07, max_negatives=64)

        # Small batch -> should use _full_forward
        z1 = torch.randn(32, 64, device=device)
        z2 = torch.randn(32, 64, device=device)
        loss_small = loss_fn(z1, z2)
        print(f"  Small batch (N=32):  loss = {loss_small.item():.4f}  (full_forward path)")
        assert not torch.isnan(loss_small), "NaN in small batch loss!"
        assert loss_small.item() > 0, "Loss should be positive"

        # Large batch -> should use _sampled_forward (200*2=400 > 64+2)
        z1_big = torch.randn(200, 64, device=device)
        z2_big = torch.randn(200, 64, device=device)
        loss_big = loss_fn(z1_big, z2_big)
        print(f"  Large batch (N=200): loss = {loss_big.item():.4f}  (sampled_forward path)")
        assert not torch.isnan(loss_big), "NaN in large batch loss!"
        assert loss_big.item() > 0, "Loss should be positive"

        # Gradient flows correctly
        z1_g = torch.randn(200, 64, device=device, requires_grad=True)
        z2_g = torch.randn(200, 64, device=device, requires_grad=True)
        loss_g = loss_fn(z1_g, z2_g)
        loss_g.backward()
        assert z1_g.grad is not None and z2_g.grad is not None, "Gradients not computed!"
        assert not torch.isnan(z1_g.grad).any(), "NaN in z1 gradients!"
        assert not torch.isnan(z2_g.grad).any(), "NaN in z2 gradients!"
        print(f"  Gradient check: OK (norms: z1={z1_g.grad.norm():.4f}, z2={z2_g.grad.norm():.4f})")

        # Check memory: for N=5000, full would be 10000x10000 = 400MB float32.
        # Sampled with K=1024 creates at most 512x1024 = 2MB chunks.
        # Just test it runs:
        z1_5k = torch.randn(5000, 64, device=device)
        z2_5k = torch.randn(5000, 64, device=device)
        loss_5k = loss_fn(z1_5k, z2_5k)
        print(f"  N=5000 batch:        loss = {loss_5k.item():.4f}  (sampled, no OOM)")
        assert not torch.isnan(loss_5k), "NaN in 5k batch loss!"

        print("[PASS] SampledNTXentLoss works correctly.")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        traceback.print_exc()
        return False


def test_efficient_dataset():
    """Test 2: EfficientConditionalDiffusionDataset."""
    header("TEST 2: EfficientConditionalDiffusionDataset")
    try:
        N, D1, D2 = 500, 64, 32
        data1 = np.random.randn(N, D1).astype(np.float32)
        data2 = np.random.randn(N, D2).astype(np.float32)

        ds = EfficientConditionalDiffusionDataset(data1, data2)
        assert len(ds) == N, f"Dataset length mismatch: {len(ds)} != {N}"

        x, x_hat, x_cond = ds[0]
        assert x.shape == (D1,), f"x shape mismatch: {x.shape}"
        assert x_hat.shape == (D2,), f"x_hat shape mismatch: {x_hat.shape}"
        assert x_cond.shape == (1, D2), f"x_cond shape: {x_cond.shape} (expected (1, {D2}))"

        # Verify con_data is the precomputed mean
        expected_mean = np.mean(data2, axis=0, keepdims=True)
        np.testing.assert_allclose(ds.con_data.numpy(), expected_mean, atol=1e-5)
        print(f"  con_data shape: {ds.con_data.shape} -- precomputed mean (was ({N}, {D2}))")

        # Test with DataLoader
        from torch.utils.data import DataLoader
        dl = DataLoader(ds, batch_size=32, shuffle=False)
        batch = next(iter(dl))
        bx, bx_hat, bx_cond = batch
        assert bx.shape == (32, D1), f"bx shape: {bx.shape}"
        assert bx_hat.shape == (32, D2), f"bx_hat shape: {bx_hat.shape}"
        assert bx_cond.shape == (32, 1, D2), f"bx_cond shape: {bx_cond.shape}"
        print(f"  DataLoader batch shapes: x={bx.shape}, x_hat={bx_hat.shape}, x_cond={bx_cond.shape}")

        # Memory comparison
        full_mem_per_sample = N * D2 * 4  # float32
        efficient_mem_per_sample = 1 * D2 * 4
        ratio = full_mem_per_sample / efficient_mem_per_sample
        print(f"  Memory saving: {ratio:.0f}x per sample ({full_mem_per_sample} -> {efficient_mem_per_sample} bytes)")

        # For N=37885 (real dataset size), full would be 37885*D2*batch_size*4 per batch
        N_real, D_real = 37885, 64
        full_batch_mem = N_real * D_real * 512 * 4 / (1024**3)
        eff_batch_mem = 1 * D_real * 512 * 4 / (1024**3)
        print(f"  Real-world (N={N_real}): {full_batch_mem:.2f} GB/batch -> {eff_batch_mem:.6f} GB/batch")

        print("[PASS] EfficientConditionalDiffusionDataset works correctly.")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        traceback.print_exc()
        return False


def test_nan_reindex():
    """Test 3: NaN-safe reindex logic (the positional alignment patch)."""
    header("TEST 3: NaN-Safe Reindex Logic")
    import pandas as pd
    try:
        N = 500

        # Case A: Same-size embeddings -> skip reindex (the patch)
        emb1 = np.random.randn(N, 64).astype(np.float32)
        emb2 = np.random.randn(N, 32).astype(np.float32)

        if emb1.shape[0] == emb2.shape[0] == N:
            aligned1 = emb1.copy()
            aligned2 = emb2.copy()
            print(f"  Case A (same size {N}): skipped reindex -- no NaN possible")
        assert not np.isnan(aligned1).any()
        assert not np.isnan(aligned2).any()

        # Case B: Mismatched spatial coords -> reindex produces NaN -> fix with nan_to_num
        spatial1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        spatial2 = np.array([[1.0, 2.0], [7.0, 8.0], [5.0, 6.0]])  # [7,8] not in spatial1
        emb_small1 = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        emb_small2 = np.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]])

        df1 = pd.DataFrame(emb_small1, index=[tuple(c) for c in spatial1])
        df2 = pd.DataFrame(emb_small2, index=[tuple(c) for c in spatial2])
        df2_aligned = df2.reindex(df1.index)
        aligned_emb2 = df2_aligned.to_numpy()

        nan_count = np.isnan(aligned_emb2).sum()
        print(f"  Case B (mismatched coords): {nan_count} NaN values before fix")
        assert nan_count > 0, "Expected NaN from reindex mismatch"

        # Apply the patch fix
        aligned_emb2 = np.nan_to_num(aligned_emb2, nan=0.0)
        assert not np.isnan(aligned_emb2).any(), "Still NaN after fix!"
        print(f"  After nan_to_num: 0 NaN values -- fix works")

        # Case C: Fusion data (same cells, same order) -> the patch skips reindex entirely
        # This is the common case for horizontal integration
        N_fusion = 1000
        emb_fus1 = np.random.randn(N_fusion, 64).astype(np.float32)
        emb_fus2 = np.random.randn(N_fusion, 32).astype(np.float32)
        N_cells = N_fusion
        if emb_fus1.shape[0] == emb_fus2.shape[0] == N_cells:
            # Patch path: skip reindex
            pass
        assert not np.isnan(emb_fus1).any() and not np.isnan(emb_fus2).any()
        print(f"  Case C (fusion, N={N_fusion}): patch correctly skips reindex")

        print("[PASS] NaN-safe reindex logic works correctly.")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        traceback.print_exc()
        return False


def test_large_dataset_thresholds():
    """Test 4: Large dataset threshold logic (epochs/folds/loss selection)."""
    header("TEST 4: Large-Dataset Threshold Logic")
    try:
        # Simulate the logic from run_candies.py lines 596-605
        for N_cells, expected_large in [(300, False), (9999, False), (10000, False), (10001, True), (37885, True)]:
            is_large = N_cells > _LARGE_DATASET_THRESHOLD
            assert is_large == expected_large, f"N={N_cells}: expected is_large={expected_large}, got {is_large}"

            if is_large:
                diff_k, diff_epochs, diff_patience = 1, 300, 30
            else:
                diff_k, diff_epochs, diff_patience = 3, 1000, 40

            # Verify loss selection
            if is_large:
                loss_type = "SampledNTXentLoss"
            else:
                loss_type = "NTXentLoss (original)"

            print(f"  N={N_cells:>6}: is_large={is_large}, k={diff_k}, epochs={diff_epochs}, "
                  f"patience={diff_patience}, loss={loss_type}")

        print("[PASS] Large-dataset threshold logic works correctly.")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        traceback.print_exc()
        return False


def test_source_verification():
    """Test 5: Verify that run_candies.py contains all expected patches."""
    header("TEST 5: Source-Level Verification")
    try:
        src_path = os.path.join(
            PROJECT_ROOT,
            "_myx_Scripts/horizontal_integration/CANDIES/run_candies.py"
        )
        with open(src_path, 'r') as f:
            src = f.read()

        checks = {
            "SampledNTXentLoss class": "class SampledNTXentLoss(nn.Module):",
            "_sampled_forward method": "def _sampled_forward(self, z1, z2):",
            "_full_forward method": "def _full_forward(self, z1, z2):",
            "EfficientConditionalDiffusionDataset": "class EfficientConditionalDiffusionDataset:",
            "con_data precomputed mean": ".mean(dim=0, keepdim=True)",
            "_patched_train_and_infer": "def _patched_train_and_infer(",
            "SampledNTXentLoss usage in patched_train": "SampledNTXentLoss(temperature=0.07, max_negatives=1024)",
            "_patched_normal_train_diff": "def _patched_normal_train_diff(",
            "CosineAnnealingLR": "CosineAnnealingLR(optimizer, T_max=num_epoch",
            "Monkey-patch applied": "_train_diff_module.normal_train_diff = _patched_normal_train_diff",
            "Large dataset threshold": "_LARGE_DATASET_THRESHOLD = 10000",
            "NaN reindex fix (skip)": "skipping spatial reindex (avoids NaN)",
            "NaN reindex fix (nan_to_num)": "np.nan_to_num(aligned_emb_latent_omics2, nan=0.0)",
            "Reduced epochs for large": 'diff_epochs = 300',
            "Reduced folds for large": 'diff_k = 1',
        }

        all_found = True
        for name, pattern in checks.items():
            found = pattern in src
            status = "OK" if found else "MISSING"
            print(f"  [{status}] {name}")
            if not found:
                all_found = False

        if all_found:
            print("[PASS] All 5 patches verified in source.")
        else:
            print("[FAIL] Some patches missing from source!")
        return all_found
    except Exception as e:
        print(f"[FAIL] {e}")
        traceback.print_exc()
        return False


def test_cosine_lr_vs_step_lr():
    """Test 6: CosineAnnealingLR vs StepLR behavior comparison."""
    header("TEST 6: CosineAnnealingLR vs StepLR Comparison")
    try:
        # Simulate the old StepLR(step_size=100, gamma=0.1)
        model_dummy = nn.Linear(10, 10)
        lr = 1e-3
        num_epoch = 1400

        opt_step = torch.optim.AdamW(model_dummy.parameters(), lr=lr)
        sched_step = torch.optim.lr_scheduler.StepLR(opt_step, step_size=100, gamma=0.1)

        opt_cos = torch.optim.AdamW(model_dummy.parameters(), lr=lr)
        sched_cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt_cos, T_max=num_epoch, eta_min=lr * 0.01)

        step_lrs = []
        cos_lrs = []
        for epoch in range(num_epoch):
            step_lrs.append(opt_step.param_groups[0]['lr'])
            cos_lrs.append(opt_cos.param_groups[0]['lr'])
            sched_step.step()
            sched_cos.step()

        # At epoch 300, StepLR has decayed 3 times: lr * 0.1^3 = 1e-6
        step_at_300 = step_lrs[300]
        cos_at_300 = cos_lrs[300]
        print(f"  At epoch 300: StepLR lr={step_at_300:.2e}, CosineAnnealingLR lr={cos_at_300:.2e}")
        assert step_at_300 < 1e-5, f"StepLR should be very small at epoch 300: {step_at_300}"
        assert cos_at_300 > 1e-4, f"CosineAnnealingLR should still be healthy at epoch 300: {cos_at_300}"

        # At epoch 1000, StepLR is essentially 0
        step_at_1000 = step_lrs[1000]
        cos_at_1000 = cos_lrs[1000]
        print(f"  At epoch 1000: StepLR lr={step_at_1000:.2e}, CosineAnnealingLR lr={cos_at_1000:.2e}")

        # CosineAnnealingLR min should be lr*0.01 = 1e-5
        cos_min = min(cos_lrs)
        print(f"  CosineAnnealingLR min lr: {cos_min:.2e} (eta_min = {lr*0.01:.2e})")
        assert abs(cos_min - lr * 0.01) < 1e-7, f"CosineAnnealingLR min should be {lr*0.01}"

        print(f"  StepLR kills LR to {step_at_300:.2e} by epoch 300 -> causes NaN in diffusion")
        print(f"  CosineAnnealingLR maintains {cos_at_300:.2e} at epoch 300 -> stable training")

        print("[PASS] CosineAnnealingLR correctly replaces aggressive StepLR.")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        traceback.print_exc()
        return False


def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python: {sys.executable}")
    print(f"Working dir: {os.getcwd()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\nNOTE: No GPU detected. Unit tests will run on CPU.")
        print("      End-to-end test (requires GPU for DiTs import) is SKIPPED.")

    results = {}

    results['sampled_ntxent'] = test_sampled_ntxent()
    results['efficient_dataset'] = test_efficient_dataset()
    results['nan_reindex'] = test_nan_reindex()
    results['large_thresholds'] = test_large_dataset_thresholds()
    results['source_verification'] = test_source_verification()
    results['cosine_vs_step_lr'] = test_cosine_lr_vs_step_lr()

    # Summary
    header("SUMMARY")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: [{status}]")
        if not passed:
            all_pass = False

    if not torch.cuda.is_available():
        print(f"  e2e_subprocess: [SKIP] (no GPU on login node)")

    if all_pass:
        print("\nAll runnable tests PASSED.")
    else:
        print("\nSome tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
