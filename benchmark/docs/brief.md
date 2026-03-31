# SMOBench: Spatial Multi-Omics Benchmarking Framework - Project Brief

## Overview
SMOBench is a comprehensive benchmarking framework for evaluating spatial multi-omics integration methods. The project systematically compares different computational approaches for integrating spatial transcriptomics with additional modalities (protein/ADT and chromatin accessibility/ATAC) across diverse biological systems.

## Research Objectives

### Core Goals
1. **Method Benchmarking**: Systematic evaluation of spatial multi-omics integration methods
2. **Task-Specific Analysis**: Compare performance across three integration paradigms:
   - **Vertical Integration**: Cross-modality, same batch (RNA + ADT/ATAC from same sample)
   - **Horizontal Integration**: Same modality, cross-batch (RNA from multiple samples)
   - **Mosaic Integration**: Mixed modality and batch integration
3. **Clustering Evaluation**: Assess downstream clustering performance with multiple algorithms
4. **Ground Truth Analysis**: Differentiate performance on datasets with/without reference annotations

### Key Research Questions
- Which integration methods perform best for different tissue types and modalities?
- How does spatial information improve multi-omics integration compared to non-spatial methods?
- What are the optimal clustering strategies for different integration approaches?
- How do methods perform when ground truth annotations are unavailable?

## Technical Scope

### Integration Methods
- **SpatialGlue**: Graph-based spatial multi-omics integration
- **SpaMosaic**: Hierarchical graph learning for spatial omics
- **PRAGA**: Adversarial training for multi-modal integration
- **COSMOS**: Cross-modality learning with optimal transport
- **PRESENT**: Spatial-aware multi-omics representation learning

### Datasets
**With Ground Truth (Supervised Evaluation)**:
- 3M_Simulation: Simulated multi-modal data with known cell types
- Human_Lymph_Nodes & Human_Tonsils: Clinical samples with spatial annotations
- Mouse_Embryos: Developmental time series with stage-specific patterns

**Without Ground Truth (Unsupervised Evaluation)**:
- Mouse_Spleen & Mouse_Thymus: Adult tissue samples
- Mouse_Brain: Neural tissue with multiple chromatin marks (H3K27ac, H3K27me3, H3K4me3, ATAC)

### Evaluation Framework
**Supervised Metrics** (with ground truth):
- ARI (Adjusted Rand Index): Clustering accuracy vs reference
- NMI (Normalized Mutual Information): Information preservation
- FMI (Fowlkes-Mallows Index): Precision-recall balance

**Unsupervised Metrics** (without ground truth):
- Silhouette Score: Cluster cohesion and separation
- Moran's I: Spatial autocorrelation of cluster assignments
- Spatial Consistency: Neighborhood cluster agreement

## Experimental Design

### Clustering Methods
- **Leiden/Louvain**: Graph-based community detection with resolution tuning
- **K-means**: Centroid-based clustering for baseline comparison
- **mclust**: Model-based clustering with Gaussian mixtures

### Standardization Protocol
1. **Method Standardization**: Normalize metrics within each dataset (min-max scaling)
2. **Cross-Dataset Aggregation**: Average performance across tissue types
3. **Batch Averaging**: For replicate samples (S1/S2), compute mean performance
4. **Final Ranking**: Aggregate scores across all evaluation criteria

### Computational Pipeline
1. **Data Preprocessing**: Load h5ad files, handle duplicate gene names
2. **Method Execution**: Run integration with standardized parameters
3. **Universal Clustering**: Apply all clustering methods to integration embeddings
4. **Metric Calculation**: Compute evaluation metrics for each method-clustering combination
5. **Result Aggregation**: Generate summary statistics and rankings

## Success Metrics

### Performance Benchmarking
- Comprehensive performance matrix across methods × datasets × clustering algorithms
- Statistical significance testing for method comparisons
- Robustness analysis across different hyperparameter settings

### Methodological Insights
- Identification of method-specific strengths and limitations
- Tissue-type and modality-specific performance patterns
- Computational efficiency and scalability analysis

### Reproducibility Standards
- Fixed random seeds for all stochastic processes
- Containerized execution environment (conda)
- Version-controlled code and documented parameters
- Standardized evaluation protocols

## Technical Infrastructure
- **Environment**: Python 3.9+ with conda package management
- **Computing**: GPU-accelerated processing for deep learning methods
- **Storage**: Efficient h5ad format for large-scale omics data
- **Visualization**: Comprehensive plotting for spatial and embedding visualizations
- **Documentation**: Complete methodology and reproducibility documentation