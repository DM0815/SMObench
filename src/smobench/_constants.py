"""
Centralized constants for SMObench.

ALL configurable values live here. No other module should hardcode
dataset names, method names, metric definitions, or clustering params.

Users can override at runtime:
    smobench.config.N_NEIGHBORS = 30
    smobench.config.CLUSTERING_METHODS = ["leiden", "kmeans"]
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────
# Clustering
# ──────────────────────────────────────────────────────────
CLUSTERING_METHODS = ["leiden", "louvain", "kmeans", "mclust"]
DEFAULT_CLUSTERING = ["leiden", "louvain", "kmeans", "mclust"]
N_NEIGHBORS = 20
RANDOM_SEED = 42
DEFAULT_RESOLUTION = 1.0       # Leiden/Louvain default (scanpy standard)
MATCH_CLUSTERS = False         # If True, search resolution to match n_clusters exactly

# ──────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────
SC_METRICS = ["Moran_I"]
BIOC_METRICS = ["ARI", "NMI", "cASW", "cLISI"]       # withGT
BVC_METRICS = ["Silhouette", "DBI", "CHI"]             # woGT
BER_METRICS = ["kBET", "bASW", "iLISI", "KNN_conn", "PCR"]
CMGTC_METRICS = ["CMGTC"]

# Which metrics are lower-is-better (need inversion for scoring)
LOWER_IS_BETTER = {"DBI", "Geary_C"}

# Ground truth label key in adata.obs
GT_LABEL_KEY = "Spatial_Label"

# Batch label key in adata.obs
BATCH_LABEL_KEY = "batch"

# ──────────────────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────────────────
DATASETS = {
    "Human_Lymph_Nodes": {
        "modality": "RNA_ADT", "gt": True,
        "data_type": "10x",
        "slices": ["A1", "D1"],
        "n_clusters": {"A1": 10, "D1": 11},
        "path": "withGT/RNA_ADT/Human_Lymph_Nodes",
    },
    "Human_Tonsils": {
        "modality": "RNA_ADT", "gt": True,
        "data_type": "10x",
        "slices": ["S1", "S2", "S3"],
        "n_clusters": {"S1": 4, "S2": 5, "S3": 5},
        "path": "withGT/RNA_ADT/Human_Tonsils",
    },
    "Mouse_Embryos_S1": {
        "modality": "RNA_ATAC", "gt": True,
        "data_type": "MISAR",
        "slices": ["E11", "E13", "E15", "E18"],
        "n_clusters": {"E11": 8, "E13": 12, "E15": 12, "E18": 14},
        "path": "withGT/RNA_ATAC/Mouse_Embryos_S1",
    },
    "Mouse_Embryos_S2": {
        "modality": "RNA_ATAC", "gt": True,
        "data_type": "MISAR",
        "slices": ["E11", "E13", "E15", "E18"],
        "n_clusters": {"E11": 13, "E13": 14, "E15": 15, "E18": 16},
        "path": "withGT/RNA_ATAC/Mouse_Embryos_S2",
    },
    "Mouse_Spleen": {
        "modality": "RNA_ADT", "gt": False,
        "data_type": "SPOTS",
        "slices": ["Mouse_Spleen1", "Mouse_Spleen2"],
        "n_clusters": 5,
        "path": "woGT/RNA_ADT/Mouse_Spleen",
    },
    "Mouse_Thymus": {
        "modality": "RNA_ADT", "gt": False,
        "data_type": "Stereo-CITE-seq",
        "slices": ["Mouse_Thymus1", "Mouse_Thymus2", "Mouse_Thymus3", "Mouse_Thymus4"],
        "n_clusters": 8,
        "path": "woGT/RNA_ADT/Mouse_Thymus",
    },
    "Mouse_Brain": {
        "modality": "RNA_ATAC", "gt": False,
        "data_type": "Spatial-epigenome-transcriptome",
        "slices": ["Mouse_Brain_ATAC", "Mouse_Brain_H3K27ac", "Mouse_Brain_H3K27me3", "Mouse_Brain_H3K4me3"],
        "n_clusters": 18,
        "path": "woGT/RNA_ATAC/Mouse_Brain",
    },
}


def get_n_clusters(dataset: str, slice_name: str) -> int:
    """Get n_clusters for a specific dataset/slice combination."""
    ds = DATASETS[dataset]
    nc = ds["n_clusters"]
    if isinstance(nc, dict):
        return nc[slice_name]
    return nc


# ──────────────────────────────────────────────────────────
# Methods (built-in module names for lazy loading)
# ──────────────────────────────────────────────────────────
BUILTIN_METHODS = [
    "spatialglue", "spamosaic", "praga", "cosmos", "present",
    "candies", "miso", "multigate", "smopca", "spabalance",
    "spafusion", "spami", "spamultivae", "spamv", "switch",
    "smart",
]

# Method → embedding key mapping (obsm key after integration)
METHOD_EMBEDDING_KEYS = {
    "SpatialGlue": "SpatialGlue",
    "SpaMosaic": "SpaMosaic",
    "PRAGA": "PRAGA",
    "COSMOS": "COSMOS",
    "PRESENT": "PRESENT",
    "CANDIES": "CANDIES",
    "MISO": "MISO",
    "MultiGATE": "MultiGATE_clip_all",
    "SMOPCA": "SMOPCA",
    "SpaBalance": "SpaBalance",
    "SpaFusion": "SpaFusion",
    "SpaMI": "SpaMI",
    "spaMultiVAE": "SpaMultiVAE",
    "SpaMV": "SpaMV",
    "SWITCH": "SWITCH",
    "SMART": "SMART",
}

# Method → unsupported dataset skip list (from original benchmark scripts)
# These combinations were explicitly skipped in the upstream benchmark.
# Key: method name. Value: set of dataset names to skip.
METHOD_DATASET_SKIP = {
    "SpaFusion": {"Mouse_Embryos_S1", "Mouse_Embryos_S2", "Mouse_Brain"},  # No ATAC support
    "spaMultiVAE": {"Mouse_Embryos_S1", "Mouse_Embryos_S2", "Mouse_Brain"},  # No ATAC support
}

# 3M (three-modality / mosaic) support
# "native" = method has dedicated 3M code from original authors
# "adapted" = we wrote adapter scripts to feed 3 modalities
METHOD_3M_SUPPORT = {
    "SpatialGlue": "native",     # SpatialGlue_3M/ module
    "SpaMosaic": "native",       # Designed for mosaic integration
    "PRAGA": "native",           # Train_model_3M, preprocess_3M
    "SpaBalance": "native",      # SpaBalance_3M/ module
    "MISO": "adapted",
    "PRESENT": "adapted",
    "SMOPCA": "adapted",
    "SpaMV": "adapted",
    # These do NOT have 3M scripts:
    # COSMOS, CANDIES, MultiGATE, SpaFusion, SpaMI, spaMultiVAE, SWITCH
}

# ──────────────────────────────────────────────────────────
# Tasks
# ──────────────────────────────────────────────────────────
TASKS = ["vertical", "horizontal", "mosaic"]

# Task → which metric dimensions are used
TASK_METRICS = {
    "vertical": ["SC", "BioC", "CMGTC"],       # BVC replaces BioC for woGT
    "horizontal": ["SC", "BVC", "BER", "CMGTC"],
    "mosaic": ["SC", "BVC", "BER", "CMGTC"],
}

# ──────────────────────────────────────────────────────────
# Scoring formulas
# ──────────────────────────────────────────────────────────
# All use simple mean. Weights can be customized here.
SCORE_WEIGHTS = {
    "vertical_withGT": {"SC": 1.0, "BioC": 1.0, "CMGTC": 1.0},
    "vertical_woGT": {"SC": 1.0, "BVC": 1.0, "CMGTC": 1.0},
    "horizontal_withGT": {"SC": 1.0, "BioC": 1.0, "BER": 1.0, "CMGTC": 1.0},
    "horizontal_woGT": {"SC": 1.0, "BVC": 1.0, "BER": 1.0, "CMGTC": 1.0},
    "mosaic": {"SC": 1.0, "BVC": 1.0, "BER": 1.0, "CMGTC": 1.0},
}

# ──────────────────────────────────────────────────────────
# File conventions
# ──────────────────────────────────────────────────────────
RNA_FILENAME = "adata_RNA.h5ad"
ADT_FILENAME = "adata_ADT.h5ad"
ATAC_FILENAME = "adata_ATAC.h5ad"
INTEGRATED_FILENAME = "adata_integrated.h5ad"

FUSION_RNA_TEMPLATE = "{dataset}_Fusion_RNA.h5ad"
FUSION_MOD2_TEMPLATE = "{dataset}_Fusion_{modality}.h5ad"
