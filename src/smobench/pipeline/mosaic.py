"""Mosaic integration pipeline: mixed modality and batch."""

from __future__ import annotations

import time

from smobench.pipeline.benchmark import BenchmarkResult


def run_mosaic(
    dataset: str,
    method_name: str,
    clustering: list[str],
    device: str = "cuda:0",
    seed: int = 42,
    data_root: str | None = None,
) -> list[dict]:
    """Run mosaic integration for one method on one dataset.

    Currently only SpaMosaic supports mosaic integration natively.
    """
    from smobench.methods import get_method

    method = get_method(method_name)
    if "mosaic" not in method.tasks:
        print(f"  SKIP: {method_name} does not support mosaic integration")
        return []

    # TODO: Implement mosaic data loading and evaluation
    # Mosaic integration requires special data preparation where
    # different slices have different modality combinations
    print(f"  Mosaic integration for {method_name}: not yet implemented in package")
    return []
