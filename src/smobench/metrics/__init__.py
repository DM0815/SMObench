"""
Evaluation metrics for spatial multi-omics integration.

Metric dimensions:
    SC   - Spatial Coherence (Moran's I)
    BioC - Biological Conservation (ARI, NMI, cASW, cLISI) [withGT]
    BVC  - Biological/Cluster Quality (Silhouette, DBI, CHI) [woGT]
    BER  - Batch Effect Removal (kBET, bASW, iLISI, KNN conn, PCR)
    CMGTC - Cross-Modal Global Topology Consistency

Usage:
    scores = smobench.metrics.evaluate(adata, embedding_key="SpatialGlue",
                                        label_key="celltype")
    scores = smobench.metrics.fast(adata, ...)    # SC + Silhouette only
    scores = smobench.metrics.all(adata, ...)     # All metrics
"""

from smobench.metrics.evaluate import evaluate, fast, standard, all_metrics
from smobench.metrics.spatial_coherence import morans_i, gearys_c
from smobench.metrics.bio_conservation import ari, nmi, asw_celltype, graph_clisi
from smobench.metrics.bio_quality import silhouette, davies_bouldin, calinski_harabasz
from smobench.metrics.batch_effect import kbet, asw_batch, graph_ilisi, knn_connectivity, pcr
from smobench.metrics.cmgtc import cmgtc

__all__ = [
    "evaluate", "fast", "standard", "all_metrics",
    "morans_i", "gearys_c",
    "ari", "nmi", "asw_celltype", "graph_clisi",
    "silhouette", "davies_bouldin", "calinski_harabasz",
    "kbet", "asw_batch", "graph_ilisi", "knn_connectivity", "pcr",
]
