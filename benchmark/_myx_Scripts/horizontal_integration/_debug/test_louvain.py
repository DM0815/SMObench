#!/usr/bin/env python
"""Quick test: does sc.tl.louvain segfault independently?"""
import sys
import os
import numpy as np
import scanpy as sc
import anndata as ad

print("[1] Creating test AnnData...", flush=True)
np.random.seed(42)
n_cells = 300
n_dims = 64
X = np.random.randn(n_cells, n_dims).astype(np.float32)
adata = ad.AnnData(X)
adata.obsm['emb'] = X

print("[2] Computing neighbors...", flush=True)
sc.pp.neighbors(adata, use_rep='emb', n_neighbors=50)
print("[2] Done", flush=True)

print("[3] Testing sc.tl.leiden...", flush=True)
sys.stdout.flush()
try:
    sc.tl.leiden(adata, resolution=1.0, random_state=42, key_added='test_leiden')
    print(f"[3] leiden OK: {adata.obs['test_leiden'].nunique()} clusters", flush=True)
except Exception as e:
    print(f"[3] leiden ERROR: {e}", flush=True)

print("[4] Testing sc.tl.louvain...", flush=True)
sys.stdout.flush()
try:
    sc.tl.louvain(adata, resolution=1.0, random_state=42, key_added='test_louvain')
    print(f"[4] louvain OK: {adata.obs['test_louvain'].nunique()} clusters", flush=True)
except Exception as e:
    print(f"[4] louvain ERROR: {e}", flush=True)

print("[5] All tests passed!", flush=True)
