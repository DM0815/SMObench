# SMObench Tutorial

## Installation

### Basic install (evaluation + data loading only)
```bash
pip install smobench
```

### Install with specific methods
```bash
# Install PyTorch first (match your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Then install method dependencies
pip install smobench[spatialglue]        # SpatialGlue only
pip install smobench[all-pyg]            # All PyTorch Geometric methods
pip install smobench[all-methods]        # Everything
```

### Available extras

| Extra | Methods covered |
|-------|----------------|
| `torch` | Base PyTorch (required for most methods) |
| `pyg` | torch-geometric + scatter + sparse |
| `dgl` | DGL (SpaMosaic) |
| `spatialglue` | SpatialGlue |
| `spamosaic` | SpaMosaic (needs pyg + dgl + hnswlib + annoy) |
| `praga` | PRAGA (needs pyg + scvi-tools) |
| `cosmos` | COSMOS (needs pyg + gudhi) |
| `candies` | CANDIES (needs pyg + einops + timm) |
| `miso` | MISO (needs torch + einops + opencv) |
| `spamv` | SpaMV (needs pyg + pyro-ppl) |
| `multigate` | MultiGATE (needs TensorFlow, conflicts with PyTorch) |
| `all-pyg` | All PyG-based methods |
| `all-methods` | All methods including DGL |

---

## Quick Start

### Python API

```python
import smobench

# List available methods and datasets
smobench.list_methods()
smobench.list_datasets()

# Load a dataset
adata_rna, adata_adt = smobench.load_dataset("Human_Tonsils", "S1")

# Run one method
from smobench.methods import get_method
sg = get_method("SpatialGlue")
embedding = sg.integrate(adata_rna, adata_adt, device="cuda:0")

# Run full benchmark (all methods, one dataset)
results = smobench.benchmark(
    dataset="Human_Tonsils",
    methods="all",
    task="vertical",
    device="cuda:0",
)
results.save("results.csv")
```

### Command Line

```bash
# Run benchmark
smobench run --dataset Human_Tonsils --methods SpatialGlue,PRAGA --task vertical

# Run all methods on all datasets
smobench run --dataset all --methods all --task vertical --output full_results.csv

# List available methods
smobench list methods

# List available datasets
smobench list datasets

# Evaluate existing results
smobench eval --input integrated.h5ad

# Generate plots
smobench plot --input results.csv --type heatmap
```

### YAML Config

```bash
smobench init --output my_config.yaml   # generate template
smobench run --config my_config.yaml    # run from config
```

---

## Adding Your Own Method

### Option 1: Decorator (simplest)

```python
import smobench
import numpy as np

@smobench.register_method("MyMethod", tasks=["vertical"], modalities=["RNA+ADT"])
def my_method(adata_rna, adata_mod2, device="cuda:0", seed=42, **kwargs):
    """Your integration logic here."""
    # Example: simple concatenation + PCA
    import scanpy as sc
    import anndata as ad

    combined = ad.concat([adata_rna, adata_mod2], axis=1)
    sc.pp.pca(combined, n_comps=30)
    return combined.obsm["X_pca"]  # shape: (n_cells, 30)

# Now it's available in the benchmark
results = smobench.benchmark(
    dataset="Human_Tonsils",
    methods=["MyMethod", "SpatialGlue"],
    task="vertical",
)
```

### Option 2: Class-based (more control)

```python
from smobench.methods import BaseMethod, MethodRegistry

class MyMethod(BaseMethod):
    name = "MyMethod"
    tasks = ["vertical", "horizontal"]
    modalities = ["RNA+ADT", "RNA+ATAC"]
    requires_gpu = True
    paper = "Author et al., Journal, 2025"
    url = "https://github.com/user/mymethod"

    def check_deps(self):
        try:
            import my_package
            return True
        except ImportError:
            return False

    def integrate(self, adata_rna, adata_mod2, device="cuda:0", seed=42, **kwargs):
        import my_package
        model = my_package.Model(device=device)
        embedding = model.fit_transform(adata_rna, adata_mod2)
        return embedding  # np.ndarray (n_cells, n_dims)

# Register it
MethodRegistry.register("MyMethod", method=MyMethod())
```

### Option 3: Entry points (from your own pip package)

In your package's `pyproject.toml`:
```toml
[project.entry-points."smobench.methods"]
MyMethod = "my_package.smobench_plugin:MyMethodClass"
```

Now anyone who installs your package gets it automatically in SMObench:
```bash
pip install my_package
smobench list methods  # MyMethod appears here
smobench run --methods MyMethod --dataset Human_Tonsils
```

---

## Adding Your Own Dataset

```python
import smobench

# Register a custom dataset
smobench.data.register_dataset(
    name="My_Tissue",
    path="/path/to/my/data",
    modality="RNA_ADT",        # or "RNA_ATAC"
    slices=["sample1", "sample2"],
    has_gt=True,
    n_clusters=8,
)

# Now use it in benchmarks
adata_rna, adata_adt = smobench.load_dataset("My_Tissue", "sample1")
results = smobench.benchmark(dataset="My_Tissue", methods="all")
```

**Expected directory structure:**
```
/path/to/my/data/
├── sample1/
│   ├── adata_RNA.h5ad     # RNA modality
│   └── adata_ADT.h5ad     # Secondary modality (ADT or ATAC)
└── sample2/
    ├── adata_RNA.h5ad
    └── adata_ADT.h5ad
```

**h5ad requirements:**
- `adata.X`: count matrix (can be sparse)
- `adata.obsm['spatial']`: spatial coordinates (n_cells, 2)
- `adata.obs['Spatial_Label']`: ground truth labels (optional, for withGT datasets)

---

## Evaluation Metrics

### Spatial Coherence (SC)
- **Moran's I**: Global spatial autocorrelation of cluster labels

### Biological Conservation (BioC) — requires ground truth
- **ARI**: Adjusted Rand Index
- **NMI**: Normalized Mutual Information
- **cASW**: Cell-type Average Silhouette Width
- **cLISI**: Cell-type LISI

### Biological Variance Conservation (BVC) — no ground truth needed
- **Silhouette**: Silhouette Coefficient
- **DBI**: Davies-Bouldin Index
- **CHI**: Calinski-Harabasz Index

### Batch Effect Removal (BER) — horizontal/mosaic tasks
- **kBET**: k-nearest neighbor Batch Effect Test
- **bASW**: Batch Average Silhouette Width
- **iLISI**: Integration LISI
- **KNN_conn**: kNN Graph Connectivity
- **PCR**: Principal Component Regression

### Scoring
```
Vertical (withGT):  SMObench_V = mean(SC, BioC, CMGTC)
Vertical (woGT):    SMObench_V = mean(SC, BVC, CMGTC)
Horizontal:         SMObench_H = mean(SC, BVC, BER, CMGTC)
```

---

## 15 Built-in Methods

| Method | Type | GPU | Key Deps | Install |
|--------|------|-----|----------|---------|
| SpatialGlue | GNN | Yes | torch-geometric | `pip install smobench[spatialglue]` |
| SpaMosaic | Mosaic | Yes | DGL + PyG | `pip install smobench[spamosaic]` |
| PRAGA | GNN | Yes | PyG + scvi | `pip install smobench[praga]` |
| COSMOS | Contrastive | Yes | PyG + gudhi | `pip install smobench[cosmos]` |
| PRESENT | GNN | No | PyG | `pip install smobench[present]` |
| CANDIES | Mosaic | Yes | PyG + einops | `pip install smobench[candies]` |
| MISO | Factorization | Yes | einops, opencv | `pip install smobench[miso]` |
| SMOPCA | PCA | No | (core only) | `pip install smobench` |
| SpaBalance | Mosaic | Yes | PyG | `pip install smobench[spabalance]` |
| SpaFusion | Other | Yes | PyG + munkres | `pip install smobench[spafusion]` |
| SpaMI | Contrastive | Yes | PyG + POT | `pip install smobench[spami]` |
| spaMultiVAE | VAE | Yes | torch | `pip install smobench[spamultivae]` |
| SpaMV | VAE | Yes | PyG + pyro | `pip install smobench[spamv]` |
| SWITCH | Other | Yes | PyG | `pip install smobench[switch]` |
| MultiGATE | GNN | Yes | TensorFlow | `pip install smobench[multigate]` |

---

## 7 Built-in Datasets (23 slices)

| Dataset | Modality | Slices | Clusters | Ground Truth |
|---------|----------|--------|----------|-------------|
| Human_Lymph_Nodes | RNA+ADT | A1, D1 | 10 | Yes |
| Human_Tonsils | RNA+ADT | S1, S2, S3 | 4 | Yes |
| Mouse_Embryos_S1 | RNA+ATAC | E11-E18 | 14 | Yes |
| Mouse_Embryos_S2 | RNA+ATAC | E11-E18 | 14 | Yes |
| Mouse_Spleen | RNA+ADT | Spleen1-2 | 5 | No |
| Mouse_Thymus | RNA+ADT | Thymus1-4 | 5 | No |
| Mouse_Brain | RNA+ATAC | 4 slices | 15 | No |

---

## Architecture

```
smobench/
├── __init__.py          # Top-level API
├── _constants.py        # All configurable values
├── cli.py               # Command-line interface
├── data/                # Dataset loading + download + registry
├── methods/
│   ├── base.py          # BaseMethod abstract class
│   ├── registry.py      # MethodRegistry + decorator + entry_points
│   ├── spatialglue.py   # Method wrapper (1 per method)
│   ├── ...
│   └── _vendor/         # Vendored method source code
│       ├── spatialglue/
│       ├── cosmos/
│       └── ...
├── metrics/             # SC, BioC, BVC, BER, CMGTC
├── clustering/          # Leiden, Louvain, K-means, Mclust
├── pipeline/            # Benchmark orchestration
├── plot/                # Visualization (heatmap, radar, scatter, UMAP)
└── io/                  # h5ad save/load utilities
```
