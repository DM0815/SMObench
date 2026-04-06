"""Basic tests for smobench package."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_imports():
    """Test all core modules can be imported."""
    from smobench.methods.base import BaseMethod
    from smobench.methods.registry import MethodRegistry, register_method, list_methods
    from smobench.data.registry import load_dataset, list_datasets, DATASET_REGISTRY
    from smobench.clustering import cluster, SUPPORTED_METHODS
    from smobench.metrics.evaluate import evaluate, fast
    from smobench.pipeline.benchmark import benchmark, BenchmarkResult
    from smobench.cli import app
    from smobench.io import save_embedding, load_integrated


def test_dataset_registry():
    """Test dataset registry has all 7 datasets."""
    from smobench.data.registry import DATASET_REGISTRY
    assert len(DATASET_REGISTRY) == 7
    assert "Human_Lymph_Nodes" in DATASET_REGISTRY
    assert "Mouse_Brain" in DATASET_REGISTRY


def test_list_datasets():
    """Test list_datasets returns 23 slices."""
    from smobench.data.registry import list_datasets
    df = list_datasets()
    assert len(df) == 23


def test_register_method_decorator():
    """Test @register_method decorator."""
    import numpy as np
    from smobench.methods.registry import register_method, MethodRegistry

    @register_method("TestMethod", tasks=["vertical"], requires_gpu=False)
    def test_func(adata_rna, adata_mod2, **kwargs):
        return np.zeros((10, 5))

    method = MethodRegistry.get("TestMethod")
    assert method.name == "TestMethod"
    assert method.tasks == ["vertical"]

    # Clean up
    del MethodRegistry._methods["TestMethod"]


def test_benchmark_result():
    """Test BenchmarkResult container."""
    from smobench.pipeline.benchmark import BenchmarkResult

    result = BenchmarkResult(records=[
        {"Method": "A", "Dataset": "D1", "ARI": 0.5, "BioC_Score": 0.6},
        {"Method": "B", "Dataset": "D1", "ARI": 0.7, "BioC_Score": 0.8},
    ])

    df = result.to_dataframe()
    assert len(df) == 2

    summary = result.summary()
    assert "BioC_Score" in summary.columns


def test_clustering_supported():
    """Test clustering methods."""
    from smobench.clustering import SUPPORTED_METHODS
    assert "leiden" in SUPPORTED_METHODS
    assert "kmeans" in SUPPORTED_METHODS


def test_metrics_functions_exist():
    """Test all metric functions are importable."""
    from smobench.metrics import (
        morans_i, gearys_c,
        ari, nmi, asw_celltype, graph_clisi,
        silhouette, davies_bouldin, calinski_harabasz,
        kbet, asw_batch, graph_ilisi, knn_connectivity, pcr,
        cmgtc,
    )


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
