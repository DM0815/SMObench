"""
Dataset loading and management.

Usage:
    adata_rna, adata_mod2 = smobench.data.load_dataset("Human_Lymph_Nodes", "A1")
    smobench.data.list_datasets()
"""

from smobench.data.registry import (
    load_dataset,
    list_datasets,
    register_dataset,
    DATASET_REGISTRY,
)

from smobench.data.download import download_dataset, download_all

__all__ = [
    "load_dataset", "list_datasets", "register_dataset",
    "download_dataset", "download_all",
]
