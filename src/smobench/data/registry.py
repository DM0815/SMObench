"""Dataset registry and loader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import scanpy as sc
from anndata import AnnData

from smobench._constants import (
    DATASETS, GT_LABEL_KEY,
    RNA_FILENAME, ADT_FILENAME, ATAC_FILENAME,
    FUSION_RNA_TEMPLATE, FUSION_MOD2_TEMPLATE,
)

# Default dataset root (can be overridden via env var)
def _find_default_root():
    env = os.environ.get("SMOBENCH_DATA_ROOT")
    if env:
        return env
    # Walk up from this file to find benchmark/Dataset
    p = Path(__file__).resolve()
    for parent in p.parents:
        candidate = parent / "benchmark" / "Dataset"
        if candidate.is_dir():
            return str(candidate)
    return str(p.parent)

_DEFAULT_ROOT = _find_default_root()

# Mutable registry: starts with built-in datasets, users can add more
DATASET_REGISTRY = dict(DATASETS)


def load_dataset(
    dataset_name: str,
    slice_name: str,
    data_root: Optional[str] = None,
) -> Tuple[AnnData, AnnData]:
    """Load a dataset slice, returning (adata_rna, adata_mod2)."""
    if dataset_name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise KeyError(f"Dataset '{dataset_name}' not found. Available: {available}")

    info = DATASET_REGISTRY[dataset_name]
    if slice_name not in info["slices"]:
        raise KeyError(f"Slice '{slice_name}' not in {dataset_name}. Available: {info['slices']}")

    root = Path(data_root or _DEFAULT_ROOT)
    base = root / info["path"] / slice_name

    rna_path = base / RNA_FILENAME
    if not rna_path.exists():
        raise FileNotFoundError(f"RNA file not found: {rna_path}")

    mod2_name = "ADT" if info["modality"] == "RNA_ADT" else "ATAC"
    mod2_filename = ADT_FILENAME if mod2_name == "ADT" else ATAC_FILENAME
    mod2_path = base / mod2_filename
    if not mod2_path.exists() and mod2_name == "ATAC":
        # Fallback: some datasets use adata_peaks_normalized.h5ad
        alt_path = base / "adata_peaks_normalized.h5ad"
        if alt_path.exists():
            mod2_path = alt_path
    if not mod2_path.exists():
        raise FileNotFoundError(f"{mod2_name} file not found: {mod2_path}")

    adata_rna = sc.read_h5ad(str(rna_path))
    adata_mod2 = sc.read_h5ad(str(mod2_path))

    # Attach metadata
    adata_rna.uns["dataset_name"] = dataset_name
    adata_rna.uns["slice_name"] = slice_name
    adata_rna.uns["modality"] = info["modality"]
    adata_rna.uns["has_gt"] = info["gt"]
    from smobench._constants import get_n_clusters
    adata_rna.uns["n_clusters"] = get_n_clusters(dataset_name, slice_name)

    return adata_rna, adata_mod2


def load_fusion(
    dataset_name: str,
    data_root: Optional[str] = None,
) -> Tuple[AnnData, AnnData]:
    """Load fusion (pre-merged) data for horizontal integration."""
    if dataset_name not in DATASET_REGISTRY:
        raise KeyError(f"Dataset '{dataset_name}' not found.")

    info = DATASET_REGISTRY[dataset_name]
    root = Path(data_root or _DEFAULT_ROOT)
    mod2_name = "ADT" if "ADT" in info["modality"] else "ATAC"

    fusion_dir = "fusionWithGT" if info["gt"] else "fusionWoGT"
    fusion_base = root / fusion_dir / info["modality"]

    rna_path = fusion_base / FUSION_RNA_TEMPLATE.format(dataset=dataset_name)
    mod2_path = fusion_base / FUSION_MOD2_TEMPLATE.format(dataset=dataset_name, modality=mod2_name)

    if not rna_path.exists():
        raise FileNotFoundError(f"Fusion RNA not found: {rna_path}")
    if not mod2_path.exists():
        raise FileNotFoundError(f"Fusion {mod2_name} not found: {mod2_path}")

    adata_rna = sc.read_h5ad(str(rna_path))
    adata_mod2 = sc.read_h5ad(str(mod2_path))
    return adata_rna, adata_mod2


def list_datasets(data_root: Optional[str] = None) -> pd.DataFrame:
    """List all registered datasets."""
    root = Path(data_root or _DEFAULT_ROOT)
    rows = []
    for name, info in sorted(DATASET_REGISTRY.items()):
        for s in info["slices"]:
            exists = (root / info["path"] / s / RNA_FILENAME).exists()
            rows.append({
                "Dataset": name,
                "Slice": s,
                "Modality": info["modality"],
                "GT": info["gt"],
                "Clusters": get_n_clusters(name, s),
                "Available": exists,
            })
    return pd.DataFrame(rows)


def register_dataset(
    name: str,
    path: str,
    modality: str,
    slices: list[str],
    has_gt: bool,
    n_clusters: int,
):
    """Register a custom dataset."""
    DATASET_REGISTRY[name] = {
        "modality": modality,
        "gt": has_gt,
        "slices": slices,
        "n_clusters": n_clusters,
        "path": path,
    }
