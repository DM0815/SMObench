# SMObench Singularity Runner

Run all 15 spatial multi-omics integration methods in parallel using a single shared Singularity/Apptainer container with per-method isolated Python environments.

## Architecture

```
singularity/
├── images/
│   └── pytorch_2.1.2-cuda11.8-cudnn8-runtime.sif   # Shared base image (3.6G)
├── envs/
│   ├── base/              # Shared deps: scanpy, anndata, numpy, etc. (1.6G)
│   ├── SpatialGlue/       # torch-geometric, torch-scatter, torch-sparse
│   ├── SpaMosaic/         # torch-geometric + dgl
│   ├── PRAGA/             # torch-geometric + scvi-tools
│   ├── ...                # 15 method-specific envs
│   └── MultiGATE/         # tensorflow
├── setup_envs.sh          # Install all method dependencies
├── run_all.sh             # Run selected methods on one dataset
├── run_all_datasets.sh    # Run full benchmark (all methods x all datasets)
├── submit_setup.pbs       # PBS job: install dependencies
└── submit_run.pbs         # PBS job: full benchmark run
```

**Key design**: One SIF image + per-method `pip --target` directories, isolated via `PYTHONPATH`. No root/fakeroot needed.

## Quick Start

### 1. Load Singularity

```bash
module load singularity/4.3.1
```

### 2. Pull the base image (one-time, ~5 min)

```bash
cd /data/projects/51001003/dmeng/SMObench/singularity

export SINGULARITY_CACHEDIR=$PWD/.cache
export SINGULARITY_TMPDIR=$PWD/.tmp
mkdir -p .cache .tmp images

singularity pull --dir images docker://pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime
```

### 3. Install all method dependencies (one-time, ~30 min)

```bash
bash setup_envs.sh          # Install all 15 methods
bash setup_envs.sh base     # Or install shared deps only
bash setup_envs.sh SpatialGlue  # Or install one specific method
```

Or submit as a PBS job:

```bash
qsub submit_setup.pbs
```

### 4. Run a single dataset with all methods in parallel

```bash
# Run all 15 methods on Human_Tonsils S1 (4 at a time)
bash run_all.sh \
  --task vertical \
  --data_type 10x \
  --dataset Human_Tonsils \
  --slice S1 \
  --max_parallel 4 \
  --gpu 0

# Run only specific methods
bash run_all.sh \
  --task vertical \
  --data_type 10x \
  --dataset Human_Tonsils \
  --slice S1 \
  --methods "SpatialGlue,PRAGA,SpaMosaic"

# Preview commands without running (dry run)
bash run_all.sh \
  --task vertical \
  --data_type 10x \
  --dataset Human_Tonsils \
  --slice S1 \
  --dry_run
```

### 5. Run the full benchmark

```bash
# Interactively (all datasets x all methods)
bash run_all_datasets.sh 4 0    # max_parallel=4, gpu=0

# Or submit as PBS job
qsub submit_run.pbs
```

## Command Reference

### `run_all.sh` options

| Option | Description | Example |
|--------|-------------|---------|
| `--task` | Integration task | `vertical`, `horizontal`, `mosaic` |
| `--data_type` | Data platform type | `10x`, `MISAR`, `SPOTS`, `Stereo-CITE-seq`, `simulation` |
| `--dataset` | Dataset name | `Human_Tonsils`, `Human_Lymph_Nodes`, `Mouse_Embryos` |
| `--slice` | Slice ID | `S1`, `A1`, `MISAR_S1` |
| `--methods` | Comma-separated methods | `"SpatialGlue,PRAGA"` (default: all) |
| `--max_parallel` | Max concurrent jobs | `4` (default, adjust for GPU memory) |
| `--gpu` | GPU device ID | `0` (default) |
| `--cluster_nums` | Number of clusters | e.g., `4` |
| `--dry_run` | Print commands only | (flag, no value) |

### Data type to directory mapping

| `--data_type` | Modalities | Data directory |
|---------------|------------|----------------|
| `10x` | RNA + ADT | `Dataset/withGT/RNA_ADT/` |
| `SPOTS` | RNA + ADT | `Dataset/withGT/RNA_ADT/` |
| `Stereo-CITE-seq` | RNA + ADT | `Dataset/withGT/RNA_ADT/` |
| `MISAR` | RNA + ATAC | `Dataset/withGT/RNA_ATAC/` |
| `simulation` | RNA + ADT + ATAC | `Dataset/withGT/3M/` |

### Dataset / Slice combinations

**Vertical integration (RNA+ADT, 10x):**
- `Human_Lymph_Nodes`: A1, D1
- `Human_Tonsils`: S1, S2, S3

**Vertical integration (RNA+ATAC, MISAR):**
- `Mouse_Embryos`: MISAR_S1, MISAR_S2

**Horizontal integration (10x):**
- `Human_Tonsils`, `Mouse_Spleen`, `Mouse_Thymus`

**Mosaic integration (simulation):**
- `3M_Simulation`

## Methods (15 total)

| Method | Extra dependencies | Env size |
|--------|--------------------|----------|
| SpatialGlue | torch-geometric, torch-scatter/sparse | 402M |
| SpaMosaic | torch-geometric + dgl | 7.2G |
| PRAGA | torch-geometric + scvi-tools | 6.0G |
| COSMOS | torch-geometric + gudhi | 474M |
| PRESENT | torch-geometric + genomicranges | 365M |
| CANDIES | torch-geometric + einops, gudhi, gseapy | 746M |
| MISO | einops, opencv, scikit-image | 485M |
| SMOPCA | (base only) | ~0 |
| SpaBalance | torch-geometric | 362M |
| SpaFusion | torch-geometric | 362M |
| SpaMI | torch-geometric | 362M |
| spaMultiVAE | (base only) | ~0 |
| SpaMV | torch-geometric + pyro-ppl | 5.0G |
| SWITCH | torch-geometric + pynvml | 363M |
| MultiGATE | tensorflow | 2.6G |

**Total disk usage**: ~30G (image 3.6G + base 1.6G + method envs 26G)

## How it works

Each method runs inside the same Singularity container but with a different `PYTHONPATH`:

```bash
singularity exec --nv \
  --bind /path/to/benchmark:/bench \
  --bind /path/to/envs:/envs \
  pytorch_2.1.2-cuda11.8-cudnn8-runtime.sif \
  bash -c '
    export PYTHONPATH=/envs/SpatialGlue:/envs/base:$PYTHONPATH
    cd /bench
    python Scripts/vertical_integration/SpatialGlue/run_SpatialGlue.py \
      --data_type 10x \
      --RNA_path Dataset/withGT/RNA_ADT/Human_Tonsils/S1/adata_RNA.h5ad \
      --ADT_path Dataset/withGT/RNA_ADT/Human_Tonsils/S1/adata_ADT.h5ad \
      --save_path Results/adata/SpatialGlue/Human_Tonsils/S1/output.h5ad
  '
```

The `run_all.sh` script automates this for all methods with parallel job control.

## Logs

- Build/setup logs: `singularity/logs/setup/`
- Run logs: `singularity/logs/run/{METHOD}_{DATASET}_{SLICE}_{timestamp}.log`

## Troubleshooting

### NumPy version conflict
The base image ships with PyTorch 2.1.2 which requires numpy<2. The base env pins `numpy==1.26.4`. If you see numpy errors, ensure no method env contains its own numpy:
```bash
# Check
find envs/*/numpy -maxdepth 0 -type d 2>/dev/null
# Clean
for d in envs/*/numpy; do rm -rf "$d"; done
```

### R / mclust not available
The container does not include R. Mclust clustering will be skipped; Leiden, Louvain, and K-means still work. To add R, install via the host and it will be available through Singularity's host binding.

### pybedtools / hnswlib / annoy build failures
These packages require C/C++ compilation. If they fail to install, they can be skipped for most methods. Only SWITCH and SpaMosaic use them for optional features.

### GPU out of memory
Reduce `--max_parallel` (e.g., from 4 to 2) to limit concurrent GPU jobs:
```bash
bash run_all.sh --task vertical --data_type 10x --dataset Human_Tonsils --slice S1 --max_parallel 2
```

## Verified test run

```
SpatialGlue on Human_Tonsils S1:
  Device: cuda:0
  Training time: 6.0s
  Output: (4326, 64) embedding
  File: Results/adata/SpatialGlue/Human_Tonsils/S1/SpatialGlue_HT_S1.h5ad (599M)
```
