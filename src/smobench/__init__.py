"""
SMObench: A Benchmark for Spatial Multi-Omics Integration Methods.

Quick start:
    >>> import smobench
    >>> results = smobench.benchmark("Human_Lymph_Nodes", methods="all", task="vertical")
    >>> results.plot.heatmap()

Modules:
    smobench.data       - Dataset loading and management
    smobench.methods    - Integration method registry and wrappers
    smobench.metrics    - Evaluation metrics (SC, BioC, BER, CM-GTC)
    smobench.clustering - Unified clustering (leiden, louvain, kmeans, mclust)
    smobench.pipeline   - Benchmark orchestration
    smobench.plot       - Visualization
"""

__version__ = "0.1.0"

from smobench import data
from smobench import methods
from smobench import metrics
from smobench import clustering
from smobench import pipeline
from smobench import plot

from smobench.methods import register_method, list_methods
from smobench.data import load_dataset, list_datasets
from smobench.pipeline import benchmark
from smobench import _constants as config
