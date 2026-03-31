# Spatial Multi-Omics PCA

Spatial Multi-Omics PCA (SMOPCA) is a novel dimension reduction method to integrate multi-modal data and extract low-dimensional representations with preserved spatial dependencies among spots.

![fig1](./img/fig1.png)

## Installation

SMOPCA can be installed directly from PyPI using the following command:
```
pip install SMOPCA
```

## Run SMOPCA

1. Prepare input data
   - SMOPCA accepts gene expression and protein/atac data matrices. Each modality of the data is preprocessed and normalized  separately. Initially, genes and proteins with zero counts were filtered out. Subsequently, the count matrix was normalized based on library size, followed by log-transformation and scaling to achieve unit variance and zero mean. ATAC reads are mapped to gene regions and the peak matrix is collapsed into a gene activity matrix, adhering to the established protocol from the Satija lab. The gene activity matrix was preprocessed and normalized using the same method as applied to mRNA data. Finally, we recommend to save the data into a hdf5 file.
   - Note that SMOPCA takes input matrices with columns corresponding to cells or spots.
2. Specify model hyperparameters and Model training
   - the dimensionality of the latent factors (default 20)
   - the kernel type (default matern kernel)
   - For the rest of the parameters, see more in tutorials.
3. Downstream analysis
   - Visualization
   - Clustering analysis
   - Differential expression analysis
   - GSEA
   - Other tasks

## Datasets

Sample datasets are provided in ./data folder. The rest of the datasets used in this study are available at https://doi.org/10.5281/zenodo.15187362
