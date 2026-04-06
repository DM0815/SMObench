"""I/O utilities for integrated adata files.

Convention: one h5ad per dataset/slice, all methods write into the same file.

File structure:
    adata.obsm['{Method}']              → embedding (n_cells × n_dims)
    adata.obs['{Method}_{clustering}']  → clustering labels
    adata.uns['{Method}_train_time']    → training time (seconds)
    adata.uns['methods_completed']      → list of methods in file
    adata.uns['smobench_version']       → package version
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import scanpy as sc
from anndata import AnnData


def save_embedding(
    adata_rna: AnnData,
    method_name: str,
    embedding: np.ndarray,
    output_path: str | Path,
    train_time: float = 0.0,
) -> AnnData:
    """Save method embedding into a shared integrated h5ad.

    If output_path exists, appends to existing file.
    If not, creates new from adata_rna.
    """
    from smobench import __version__

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing or create new
    if output_path.exists():
        adata = sc.read_h5ad(str(output_path))
        if adata.n_obs != adata_rna.n_obs:
            raise ValueError(
                f"Cell count mismatch: existing={adata.n_obs}, new={adata_rna.n_obs}"
            )
    else:
        adata = adata_rna.copy()

    # Store embedding + metadata
    adata.obsm[method_name] = embedding
    adata.uns[f'{method_name}_train_time'] = train_time
    adata.uns['smobench_version'] = __version__

    # Track completed methods
    completed = list(adata.uns.get('methods_completed', []))
    if method_name not in completed:
        completed.append(method_name)
    adata.uns['methods_completed'] = completed

    # Save
    adata.write(str(output_path))
    return adata


def load_integrated(path: str | Path) -> AnnData:
    """Load an integrated adata file."""
    adata = sc.read_h5ad(str(path))
    methods = list(adata.uns.get('methods_completed', []))
    if methods:
        print(f"Loaded {path}: {len(methods)} methods ({', '.join(methods)})")
    return adata


def list_methods_in_file(path: str | Path) -> list[str]:
    """List methods stored in an integrated h5ad."""
    adata = sc.read_h5ad(str(path))
    return list(adata.uns.get('methods_completed', []))


def get_embedding(adata: AnnData, method_name: str) -> np.ndarray:
    """Get method embedding from integrated adata."""
    if method_name not in adata.obsm:
        available = [k for k in adata.obsm.keys() if k != 'spatial']
        raise KeyError(f"'{method_name}' not in obsm. Available: {available}")
    return adata.obsm[method_name]
