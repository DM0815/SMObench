# SMOBench Environment Setup Guide

## Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu/CentOS) with NVIDIA GPU support
- **Python**: Python 3.9 or higher
- **CUDA**: Compatible NVIDIA drivers and CUDA toolkit
- **Memory**: 32GB+ RAM recommended for large datasets
- **Storage**: 100GB+ free space for datasets and results

### Software Dependencies
- **Conda**: Miniconda or Anaconda package manager
- **Git**: Version control system
- **NVIDIA Drivers**: Latest stable drivers for your GPU

## Environment Setup

### 1. Conda Environment Creation

```bash
# Create the smobench conda environment
conda create -n smobench python=3.9

# Activate the environment
conda activate smobench
```

### 2. Core Package Installation

```bash
# Scientific computing stack
conda install numpy pandas matplotlib seaborn scikit-learn

# Single-cell and spatial omics
conda install -c bioconda scanpy anndata

# Deep learning framework
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# Additional omics packages
pip install squidpy spatialdata
```

### 3. Method-Specific Dependencies

#### SpatialGlue
```bash
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

#### SpaMosaic
```bash
pip install dgl-cuda11.8
pip install torch-geometric
pip install networkx
```

#### PRAGA
```bash
pip install torch-geometric
pip install sklearn
```

#### COSMOS & PRESENT
```bash
pip install torch-geometric
pip install scipy
pip install networkx
```

### 4. R Interface for mclust

```bash
# Install R and rpy2
conda install -c conda-forge r-base rpy2

# Start R and install mclust
R
> install.packages("mclust")
> quit()
```

### 5. Additional Utilities

```bash
# Visualization and analysis
pip install umap-learn
pip install plotly
pip install ipywidgets

# File handling
pip install h5py
pip install tables

# Progress bars and utilities  
pip install tqdm
pip install joblib
```

## GPU Configuration

### 1. Verify GPU Availability

```bash
# Check NVIDIA drivers
nvidia-smi

# Test CUDA in Python
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

### 2. GPU Memory Management

```bash
# Monitor GPU usage continuously
watch -n 1 nvidia-smi

# Set specific GPU for execution
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0 only
```

### 3. Common GPU Issues

**Out of Memory Errors**:
```python
# Clear GPU cache
import torch
torch.cuda.empty_cache()

# Reduce batch size in method parameters
# Use mixed precision training if supported
```

## Data Setup

### 1. Dataset Download and Organization

The datasets should already be organized in the following structure:
```
Dataset/
├── data_info/          # Dataset metadata files
├── withGT/            # Datasets with ground truth
│   ├── 3M_Simulation/
│   ├── RNA_ADT/
│   └── RNA_ATAC/
└── woGT/              # Datasets without ground truth
    ├── RNA_ADT/
    └── RNA_ATAC/
```

### 2. Verify Data Integrity

```bash
conda run -n smobench python Dataset/data_info/generate_data_info.py
```

## Development Environment

### 1. IDE Configuration

**Recommended**: VS Code with extensions:
- Python
- Jupyter
- GitLens
- Python Docstring Generator

**Jupyter Setup**:
```bash
conda install jupyter
python -m ipykernel install --user --name smobench --display-name "SMOBench"
```

### 2. Git Configuration

```bash
git config --global user.name "L1nzh"
git config --global user.email "your.email@domain.com"

# Repository setup (SMOBench-CLEAN)
git remote add origin https://github.com/L1nzh/SMOBench-CLEAN.git

# For push operations, use personal access token when prompted for password
# Token stored securely - do not commit sensitive credentials to repository
```

### 3. Conda Environment Management

**Standard Package Installation:**
```bash
# Activate environment first
conda activate smobench

# Install packages using conda (preferred)
conda install package_name

# Install using pip if not available in conda
pip install package_name
```

**Environment Modification (when conda environment has issues):**
```bash
# Activate environment first
conda activate smobench

# Use python -m pip for environment updates
python -m pip install package_name
python -m pip uninstall package_name

# For upgrading existing packages
python -m pip install --upgrade package_name

# Force reinstall if needed
python -m pip install --force-reinstall package_name
```

## Testing Installation

### 1. Basic Functionality Test

```python
# Test core packages
import numpy as np
import pandas as pd
import scanpy as sc
import torch

print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)  
print("Scanpy version:", sc.__version__)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

### 2. Data Loading Test

```python
# Test AnnData loading
import scanpy as sc

# Load a sample dataset
adata = sc.read_h5ad('Dataset/withGT/3M_Simulation/adata_RNA.h5ad')
print(f"Data shape: {adata.shape}")
print(f"Obs columns: {list(adata.obs.columns)}")
```

### 3. Method Import Test

```python
# Test method imports
try:
    from Methods.SpatialGlue import SpatialGlue_pyG
    print("✅ SpatialGlue import successful")
except ImportError as e:
    print("❌ SpatialGlue import failed:", e)

try:
    import rpy2.robjects as robjects
    robjects.r.library("mclust")
    print("✅ R mclust interface working")
except Exception as e:
    print("❌ R interface failed:", e)
```

## Troubleshooting

### Common Issues

#### 1. Conda Environment Issues
```bash
# If environment creation fails
conda clean --all
conda update conda

# If packages conflict
conda install --force-reinstall <package>
```

#### 2. CUDA/PyTorch Issues
```bash
# Reinstall PyTorch with correct CUDA version
conda uninstall pytorch
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### 3. R Interface Issues
```bash
# Reinstall rpy2
pip uninstall rpy2
conda install -c conda-forge rpy2

# Test R installation
R --version
```

#### 4. Memory Issues
```bash
# Increase swap space if needed
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Performance Optimization

#### 1. NumPy/SciPy Optimization
```bash
# Use optimized BLAS libraries
conda install -c conda-forge openblas
export OPENBLAS_NUM_THREADS=1  # Control threading
```

#### 2. Scanpy Configuration
```python
# Configure scanpy for better performance
import scanpy as sc
sc.settings.n_jobs = 8  # Use multiple cores
sc.settings.max_memory = 32  # GB
```

#### 3. PyTorch Configuration
```python
# Optimize PyTorch for your system
import torch
torch.set_num_threads(8)  # Control CPU threading
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
```

## Maintenance

### Regular Updates
```bash
# Update conda environment
conda update --all

# Update pip packages
pip list --outdated
pip install --upgrade <package>
```

### Environment Export
```bash
# Export environment for reproducibility
conda env export > environment.yml

# Create from exported environment
conda env create -f environment.yml
```

### Backup Important Files
- Environment configurations
- Custom scripts and modifications
- Processed datasets and results
- Documentation and notes

## Support and Resources

### Documentation
- [Scanpy Documentation](https://scanpy.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [SpatialGlue Paper](https://www.nature.com/articles/s41592-022-01467-z)

### Community Resources
- [Single Cell Best Practices](https://www.sc-best-practices.org/)
- [Spatial Omics Methods](https://github.com/topics/spatial-omics)

### Getting Help
- Check conda environment: `conda list`
- Verify GPU status: `nvidia-smi`
- Test imports systematically
- Consult method-specific documentation for troubleshooting