# SMObench: A Comprehensive Benchmark for Spatial Multi-Omics Integration

<p align="center">
  <img src="benchmark/Framework/FrameWork.jpg" width="800"/>
</p>

**SMObench** is a pip-installable Python package for benchmarking **16 spatial multi-omics integration methods** across **7 datasets** (23 slices) with **14+ evaluation metrics**. It supports vertical, horizontal, and mosaic integration tasks.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Option A: Singularity/Apptainer (Recommended)](#option-a-singularityapptainer-recommended)
  - [Option B: Conda/Pip (Direct Install)](#option-b-condapip-direct-install)
- [Prepare Datasets](#prepare-datasets)
- [Run the Benchmark](#run-the-benchmark)
- [Python API](#python-api)
- [Add Your Own Method](#add-your-own-method)
- [Methods](#methods)
- [Datasets](#datasets)
- [Evaluation Framework](#evaluation-framework)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Prerequisites

- **Linux** (Ubuntu 18.04+, CentOS 7+, etc.)
- **NVIDIA GPU** with CUDA ≥ 11.8 (for GPU methods; SMOPCA runs on CPU)
- **Python ≥ 3.9**
- **~15 GB disk space** for container image + datasets

Check your environment:

```bash
python3 --version         # ≥ 3.9
nvidia-smi                # Check GPU & CUDA driver version
```

---

## Installation

SMObench supports two installation modes. Choose based on your environment:

| | Singularity/Apptainer | Conda/Pip |
|---|---|---|
| **Best for** | Shared servers, HPC clusters, reproducibility | Personal workstations, quick testing |
| **Root needed?** | No (only admin installs Singularity once) | No |
| **Isolation** | Full container — no dependency conflicts | You manage the Python environment |
| **GPU support** | Built-in (CUDA 11.8 in container) | You install PyTorch + CUDA yourself |

### Option A: Singularity/Apptainer (Recommended)

This is the recommended approach. All 16 methods run inside a container, so you never deal with dependency conflicts.

#### A1. Check Singularity is available

```bash
# On HPC clusters, it's usually a module
module load singularity    # or: module load apptainer

# On regular servers, it may already be in PATH
singularity --version      # or: apptainer --version
# Expected: ≥ 3.7 (singularity) or ≥ 1.0 (apptainer)
```

> **Not installed?** Ask your admin (requires root to install, but not to use). See [Apptainer install docs](https://apptainer.org/docs/admin/main/installation.html). If you have root, it's just:
> ```bash
> # Ubuntu/Debian
> sudo apt install -y apptainer
> # CentOS/RHEL
> sudo yum install -y apptainer
> ```

#### A2. Clone and install SMObench

```bash
git clone https://github.com/your-org/SMObench.git
cd SMObench
pip install -e .    # Installs the lightweight orchestration layer (no PyTorch needed here)
```

#### A3. Build the container image

```bash
# Build the SIF image (~15-30 min, needs internet)
singularity build singularity/images/smobench_full.sif singularity/defs/smobench_full.def
```

> **Pre-built image:** If your lab provides a pre-built `smobench_full.sif`, just place it at `singularity/images/smobench_full.sif` and skip this step.

Verify:

```bash
singularity exec singularity/images/smobench_full.sif python -c "import torch; print(torch.__version__)"
# Expected: 2.1.0
```

#### A4. Set up per-method dependencies

Most methods work out of the box with the container. A few need extra packages, which are stored in lightweight **prefix directories** (`singularity/envs/{MethodName}/`).

These prefixes are included in the repository. If you need to rebuild one:

```bash
# Example: rebuild SMART prefix
pip install --target=singularity/envs/SMART muon harmony-pytorch scikit-misc

# Remove packages that the container already provides (to avoid conflicts)
cd singularity/envs/SMART
rm -rf torch* numpy* scipy* pandas* scikit-learn* anndata* scanpy* \
       matplotlib* h5py* setuptools* nvidia* cuda*
```

#### How it works

When you run a method, SMObench automatically:
1. Detects that dependencies aren't in your current Python
2. Finds the `.sif` image and per-method prefix
3. Runs `singularity exec --nv` with the right `PYTHONPATH`
4. Returns the embedding to your host process

You don't need to call `singularity` yourself — it's all transparent.

---

### Option B: Conda/Pip (Direct Install)

If you don't have Singularity, you can install everything directly. This is simpler but you need to manage CUDA/PyTorch compatibility yourself.

```bash
git clone https://github.com/your-org/SMObench.git
cd SMObench

# Create a conda environment
conda create -n smobench python=3.10
conda activate smobench

# Install PyTorch (match your CUDA version — see https://pytorch.org)
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install torch-geometric
pip install torch-geometric torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Install SMObench with all method dependencies
pip install -e ".[all-methods]"
```

> **Note:** Some methods (MultiGATE) need TensorFlow, and some (SMART) need `muon`. Install as needed:
> ```bash
> pip install muon harmony-pytorch    # for SMART
> pip install tensorflow              # for MultiGATE
> ```

---

## Prepare Datasets

SMObench expects `.h5ad` files in this directory layout:

```
Dataset/
├── withGT/                        # Datasets with ground truth labels
│   ├── RNA_ADT/
│   │   ├── Human_Lymph_Nodes/
│   │   │   ├── A1/
│   │   │   │   ├── adata_RNA.h5ad
│   │   │   │   └── adata_ADT.h5ad
│   │   │   └── D1/
│   │   └── Human_Tonsils/
│   │       └── S1/ S2/ S3/
│   └── RNA_ATAC/
│       ├── Mouse_Embryos_S1/
│       │   └── E11/ E13/ E15/ E18/
│       └── Mouse_Embryos_S2/
└── woGT/                          # Datasets without ground truth
    ├── RNA_ADT/
    │   ├── Mouse_Spleen/
    │   └── Mouse_Thymus/
    └── RNA_ATAC/
        └── Mouse_Brain/
```

Tell SMObench where your data is:

```bash
# Option 1: Environment variable (recommended)
export SMOBENCH_DATA_ROOT="/path/to/Dataset"

# Option 2: Pass as argument
python tutorials/run_all_methods.py --data-root /path/to/Dataset
```

---

## Run the Benchmark

### Quick start

```bash
# Make sure you're on a machine with a GPU
nvidia-smi

# If using Singularity on an HPC cluster
module load singularity

cd SMObench

# Run all 16 methods on all datasets
python tutorials/run_all_methods.py

# Run specific methods
python tutorials/run_all_methods.py --methods SpatialGlue PRAGA COSMOS SMART

# Run CPU-only method
python tutorials/run_all_methods.py --methods SMOPCA --device cpu
```

### HPC job scripts

<details>
<summary>PBS example</summary>

```bash
#!/bin/bash
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -q gpu
#PBS -N smobench

module load singularity
cd $PBS_O_WORKDIR/SMObench
python tutorials/run_all_methods.py
```

```bash
qsub run_benchmark.pbs
```

</details>

<details>
<summary>Slurm example</summary>

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

module load singularity
cd /path/to/SMObench
python tutorials/run_all_methods.py
```

```bash
sbatch run_benchmark.sh
```

</details>

### Results

Results are saved to `tutorials/results/`:

```
tutorials/results/
├── spatialglue_all_results.csv     # Per-method CSV
├── cosmos_all_results.csv
├── ...
├── all_methods_results.csv         # Combined results across all methods
└── vertical/                       # Integrated adata (.h5ad) files
    ├── Human_Lymph_Nodes/A1/
    │   └── SpatialGlue_integrated.h5ad
    └── ...
```

---

## Python API

```python
import smobench

# List available methods and datasets
smobench.list_methods()
smobench.list_datasets()

# Load a dataset
from smobench.data import load_dataset
adata_rna, adata_mod2 = load_dataset("Human_Lymph_Nodes", "A1",
                                      data_root="/path/to/Dataset")

# Run a single method (auto-detects Singularity or runs in-process)
from smobench.pipeline._isolation import subprocess_integrate
embedding, kept_indices = subprocess_integrate(
    "SpatialGlue", adata_rna, adata_mod2,
    device="cuda:0", seed=42, modality="ADT", data_type="10x"
)

# Cluster and evaluate
from smobench.clustering import cluster
from smobench.metrics.evaluate import evaluate
adata_rna.obsm["SpatialGlue"] = embedding
cluster(adata_rna, n_clusters=10, embedding_key="SpatialGlue",
        key_added="leiden_SpatialGlue")
scores = evaluate(adata_rna, embedding_key="SpatialGlue",
                  cluster_key="leiden_SpatialGlue",
                  n_clusters=10, has_ground_truth=True)
```

---

## Add Your Own Method

**Option 1: Decorator (simplest)**
```python
@smobench.register_method("MyMethod", tasks=["vertical", "horizontal"])
def my_method(adata_rna, adata_mod2, **kwargs):
    embedding = your_model(adata_rna, adata_mod2)
    return embedding  # np.ndarray (n_cells, n_dims)
```

**Option 2: Class-based (more control)**
```python
class MyMethod(smobench.methods.BaseMethod):
    name = "MyMethod"
    tasks = ["vertical", "horizontal"]
    modalities = ["RNA+ADT", "RNA+ATAC"]
    requires_gpu = True

    def integrate(self, adata_rna, adata_mod2, **kwargs):
        return embedding  # np.ndarray
```

**Option 3: Entry points (from your own package, no fork needed)**
```toml
# In your package's pyproject.toml
[project.entry-points."smobench.methods"]
MyMethod = "my_package:MyMethodClass"
```

After `pip install my-package`, SMObench auto-discovers it.

---

## Methods

| Method | Category | GPU | RNA+ADT | RNA+ATAC | Paper |
|--------|----------|-----|---------|----------|-------|
| SpatialGlue | GNN | Yes | ✓ | ✓ | Long et al., Nature Methods 2024 |
| SpaMosaic | Mosaic | Yes | ✓ | ✓ | - |
| PRAGA | GNN | Yes | ✓ | ✓ | - |
| COSMOS | Contrastive | Yes | ✓ | ✓* | Zhou et al., Nature Comms 2024 |
| PRESENT | GNN | Yes | ✓ | ✓ | - |
| CANDIES | Mosaic | Yes | ✓ | ✓ | - |
| MISO | Factorization | Yes | ✓ | ✓ | - |
| MultiGATE | GNN | Yes | ✓ | ✓ | - |
| SMOPCA | Factorization | No | ✓ | ✓ | - |
| SpaBalance | Mosaic | Yes | ✓ | ✓ | - |
| SpaFusion | Other | Yes | ✓ | ✗ | - |
| SpaMI | Contrastive | Yes | ✓ | ✓ | - |
| spaMultiVAE | VAE | Yes | ✓ | ✗ | - |
| SpaMV | VAE | Yes | ✓ | ✓ | - |
| SWITCH | Other | Yes | ✓ | ✓ | - |
| SMART | GNN | Yes | ✓ | ✓ | Huang et al., 2025 |

\* COSMOS may fail on very sparse datasets due to upstream HVG limitations.

---

## Datasets

| Dataset | Modality | Ground Truth | Slices | Clusters | Platform |
|---------|----------|:---:|--------|----------|----------|
| Human Lymph Nodes | RNA+ADT | ✓ | A1, D1 | 10, 11 | 10x Visium |
| Human Tonsils | RNA+ADT | ✓ | S1, S2, S3 | 4, 5, 5 | 10x Visium |
| Mouse Embryos S1 | RNA+ATAC | ✓ | E11, E13, E15, E18 | 8-14 | MISAR-seq |
| Mouse Embryos S2 | RNA+ATAC | ✓ | E11, E13, E15, E18 | 13-16 | MISAR-seq |
| Mouse Spleen | RNA+ADT | ✗ | Spleen1, Spleen2 | 5 | SPOTS |
| Mouse Thymus | RNA+ADT | ✗ | Thymus1-4 | 8 | Stereo-CITE-seq |
| Mouse Brain | RNA+ATAC | ✗ | ATAC, H3K27ac, H3K27me3, H3K4me3 | 18 | Spatial-epi-trans |

---

## Evaluation Framework

| Dimension | Metrics | When Used |
|-----------|---------|-----------|
| **SC** (Spatial Coherence) | Moran's I | All tasks |
| **BioC** (Bio Conservation) | ARI, NMI, cASW, cLISI | withGT datasets |
| **BVC** (Cluster Quality) | Silhouette, DBI, CHI | woGT datasets |
| **BER** (Batch Effect Removal) | kBET, bASW, iLISI, KNN conn, PCR | Horizontal/Mosaic |
| **CM-GTC** | Cross-Modal Graph Topology Consistency | All tasks |

**Scoring formulas:**
- Vertical (withGT): `Score = mean(SC, BioC, CMGTC)`
- Vertical (woGT): `Score = mean(SC, BVC, CMGTC)`
- Horizontal: `Score = mean(SC, BVC, BER, CMGTC)`

---

## Repository Structure

```
SMObench/
├── src/smobench/              # Python package
│   ├── data/                  # Dataset loading & registry
│   ├── methods/               # 16 method wrappers + plugin system
│   │   └── _vendor/           # Vendored upstream method source code
│   ├── metrics/               # SC, BioC, BVC, BER, CM-GTC
│   ├── clustering/            # Leiden, Louvain, K-means, Mclust
│   ├── pipeline/              # Subprocess isolation & orchestration
│   ├── _constants.py          # All configurable constants
│   ├── _env.py                # Environment resolution (SIF/conda/current)
│   └── _runner.py             # Subprocess entry point
├── singularity/
│   ├── defs/                  # Container definition files
│   ├── images/                # Built .sif images
│   └── envs/                  # Per-method pip prefix directories
├── tutorials/
│   └── run_all_methods.py     # Main benchmark script
├── pyproject.toml
└── README.md
```

---

## Troubleshooting

### "CUDA is not available"
SMObench **never** silently falls back to CPU. If you request `device="cuda:0"` but CUDA is unavailable, it raises `RuntimeError`.

**Fix:** Make sure you're on a machine with a GPU:
```bash
nvidia-smi   # Should show your GPU
```

### "Neither singularity nor apptainer found"
```bash
# HPC: load the module
module load singularity   # or: module load apptainer

# Regular server: check if installed
which singularity || which apptainer

# If not installed, use Option B (conda/pip) instead
```

### Method fails with "cholesky" or "not positive-definite"
This is a **method limitation**, not a bug. Some methods (e.g., spaMultiVAE) have numerical instability on certain datasets. SMObench reports this as `INTEGRATE_FAIL` — the method genuinely cannot handle that data.

### "Bin edges must be unique" (COSMOS)
COSMOS uses `cell_ranger` HVG flavor which can fail on very sparse data. SMObench applies a compatibility patch, but extremely sparse datasets may still fail.

### Slow first run
The first run triggers Numba JIT compilation. Subsequent runs are faster.

### Per-method prefix conflicts
If a method fails with version conflicts, clean and rebuild its prefix:
```bash
rm -rf singularity/envs/METHOD_NAME/*
pip install --target=singularity/envs/METHOD_NAME <packages>
# Remove packages the container already has (torch, numpy, scipy, etc.)
```

---

## Design Principles

- **Fail loud, never lie:** GPU requested but unavailable → `RuntimeError`, not silent CPU fallback.
- **Faithful reproduction:** Wrappers match the original GitHub implementation exactly. No extra "optimizations" or error suppression.
- **Method limitations exposed:** If a method can't handle certain data, it fails honestly rather than producing unreliable results.

---

## Citation

```bibtex
@article{smobench2025,
  title={SMObench: A Comprehensive Benchmark for Spatial Multi-Omics Integration},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
