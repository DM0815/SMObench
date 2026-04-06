"""Dataset download utilities using pooch (lazy-loaded)."""

from __future__ import annotations

import os
from pathlib import Path

# Default cache directory
CACHE_DIR = os.environ.get("SMOBENCH_DATA_ROOT", "")

# Zenodo record URL (update with actual record ID after upload)
ZENODO_RECORD = "https://zenodo.org/records/XXXXXXX/files/"

# File registry: path -> sha256 hash
# Populated after uploading to Zenodo
_REGISTRY = {}


def _get_cache_dir() -> str:
    """Get cache directory, using pooch default if not set."""
    if CACHE_DIR:
        return CACHE_DIR
    try:
        import pooch
        return str(Path(pooch.os_cache("smobench")) / "datasets")
    except ImportError:
        return str(Path.home() / ".cache" / "smobench" / "datasets")


def get_data_home() -> Path:
    """Return the dataset cache directory."""
    return Path(_get_cache_dir())


def download_dataset(dataset_name: str, data_root: str | None = None) -> Path:
    """Download a dataset if not already cached.

    Parameters
    ----------
    dataset_name : str
        Dataset name (e.g., "Human_Tonsils").
    data_root : str, optional
        Override download directory.

    Returns
    -------
    Path
        Path to the downloaded dataset directory.
    """
    from smobench._constants import DATASETS

    if dataset_name not in DATASETS:
        raise KeyError(f"Unknown dataset: {dataset_name}")

    root = Path(data_root or _get_cache_dir())
    info = DATASETS[dataset_name]
    dataset_dir = root / info["path"]

    if dataset_dir.exists():
        return dataset_dir

    # If registry is populated, use pooch
    if _REGISTRY:
        import pooch

        fetcher = pooch.create(
            path=str(root),
            base_url=ZENODO_RECORD,
            registry=_REGISTRY,
        )
        for slice_name in info["slices"]:
            for fname in ["adata_RNA.h5ad", "adata_ADT.h5ad", "adata_ATAC.h5ad"]:
                key = f"{info['path']}/{slice_name}/{fname}"
                if key in _REGISTRY:
                    fetcher.fetch(key)
        return dataset_dir

    raise FileNotFoundError(
        f"Dataset '{dataset_name}' not found at {dataset_dir}.\n"
        f"Set SMOBENCH_DATA_ROOT to point to your data directory, or\n"
        f"download from: {ZENODO_RECORD}"
    )


def download_all(data_root: str | None = None):
    """Download all datasets."""
    from smobench._constants import DATASETS
    for name in DATASETS:
        try:
            download_dataset(name, data_root)
        except FileNotFoundError:
            print(f"[SKIP] {name}: not yet available for download")
