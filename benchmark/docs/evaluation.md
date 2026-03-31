# SMOBench Evaluation Framework

## Overview

SMOBench employs a comprehensive evaluation system to assess spatial multi-omics integration methods across three integration tasks: **vertical**, **horizontal**, and **mosaic**. The evaluation framework uses three categories of metrics to provide a holistic assessment of method performance.

## Evaluation Metric Categories

### 1. Spatial Coherence (SC)
Measures how well the integration preserves spatial relationships and neighborhood structure.

#### Metrics
- **Moran's I** [49]: Global spatial autocorrelation measure
  - Range: [-1, 1], Higher is better (positive indicates spatial clustering)
  - Formula: Measures whether similar values cluster spatially

- **Geary's C** [50]: Local spatial autocorrelation measure  
  - Range: [0, 2], Lower is better (values < 1 indicate positive autocorrelation)
  - Formula: Measures local spatial association

### 2. Biological Conservation (BioC)
Evaluates how well the integration preserves biological signal and enables accurate cell type identification.

#### Clustering Accuracy Metrics (With Ground Truth Only)
- **ARI** [51]: Adjusted Rand Index
  - Range: [-1, 1], Higher is better
  - Measures agreement between predicted and true clustering

- **NMI** [52]: Normalized Mutual Information
  - Range: [0, 1], Higher is better
  - Measures information shared between clusterings

- **AMI** [53]: Adjusted Mutual Information
  - Range: [-1, 1], Higher is better  
  - Chance-corrected version of mutual information

- **FMI** [54]: Fowlkes-Mallows Index
  - Range: [0, 1], Higher is better
  - Geometric mean of precision and recall for clustering

- **Purity** [55]: Cluster purity measure
  - Range: [0, 1], Higher is better
  - Fraction of samples in dominant class per cluster

- **Homogeneity**: All samples in cluster belong to same class
  - Range: [0, 1], Higher is better

- **Completeness**: All samples from same class in same cluster
  - Range: [0, 1], Higher is better

- **V-measure** [56]: Harmonic mean of homogeneity and completeness
  - Range: [0, 1], Higher is better

- **Jaccard Index** [57]: Intersection over union of cluster assignments
  - Range: [0, 1], Higher is better

- **Dice Index** [58]: Sørensen-Dice coefficient
  - Range: [0, 1], Higher is better

- **F-measure** [59]: Harmonic mean of precision and recall
  - Range: [0, 1], Higher is better

#### Quality Metrics (All Datasets)
- **Silhouette Coefficient** [60]: Cluster cohesion and separation
  - Range: [-1, 1], Higher is better
  - Measures how well samples fit their assigned cluster

- **Calinski-Harabasz Index** [61]: Ratio of between-cluster to within-cluster dispersion
  - Range: [0, ∞), Higher is better
  - Also known as Variance Ratio Criterion

- **Davies-Bouldin Index** [62]: Average similarity between clusters
  - Range: [0, ∞), Lower is better
  - Measures average similarity ratio of each cluster with its most similar cluster

#### Additional BioC Metrics (With Ground Truth)
- **ASW (cell type)**: Average silhouette width for cell types
  - Range: [-1, 1], Higher is better
  - Silhouette analysis using cell type labels

- **Graph cLISI** [63]: Local inverse Simpson's index for clustering
  - Range: [1, k], Higher is better (k = number of cell types)
  - Measures local clustering diversity

### 3. Batch Effect Removal (BER)
Evaluates how well methods remove technical batch effects while preserving biological signal.

**Note**: BER metrics are **NOT** used for vertical integration evaluation as vertical integration focuses on cross-modality integration within the same batch.

#### BER Metrics (For Horizontal and Mosaic Tasks)
- **ASW (batch)**: Average silhouette width for batches
  - Range: [-1, 1], Lower is better (negative indicates good batch mixing)

- **Graph iLISI**: Local inverse Simpson's index for batch mixing
  - Range: [1, b], Higher is better (b = number of batches)
  - Measures local batch diversity

- **kBET** [64]: k-nearest neighbor batch effect test
  - Range: [0, 1], Higher is better
  - Tests for batch effect in local neighborhoods

- **Graph Connectivity**: Preservation of local neighborhood structure
  - Range: [0, 1], Higher is better
  - Measures preservation of original neighborhood connections

## Task-Specific Evaluation

### Vertical Integration
**Definition**: Cross-modality, same batch integration (e.g., RNA + ADT from same sample)

**Evaluation Categories**: **BioC + SC** (BER not applicable)

#### With Ground Truth Datasets
**Full BioC + SC Metrics**:
- All clustering accuracy metrics (ARI, NMI, AMI, FMI, Purity, etc.)
- All quality metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin)
- Additional BioC metrics (ASW cell type, Graph cLISI)
- Spatial coherence metrics (Moran's I, Geary's C)

**Example Output** (19 metrics):
```
Metric                    SpatialGlue_HT_S1
asw_celltype             0.511
graph_clisi              0.886
ARI                      0.104
NMI                      0.274
FMI                      0.313
Silhouette Coefficient   0.147
Calinski-Harabasz Index  893.968
Davies-Bouldin Index     1.673
Purity                   0.750
AMI                      0.272
Homogeneity              0.450
Completeness             0.197
V-measure                0.274
F-measure                0.241
Jaccard Index            0.137
Dice Index               0.241
Moran Index              0.448
Geary C                  0.554
```

#### Without Ground Truth Datasets
**Limited BioC + SC Metrics**:
- Quality metrics only (no clustering accuracy metrics)
- Spatial coherence metrics

**Example Output** (5 metrics):
```
Metric                    SpatialGlue_HT_S1
Silhouette Coefficient   0.147
Calinski-Harabasz Index  893.968
Davies-Bouldin Index     1.673
Moran Index              0.448
Geary C                  0.554
```

### Horizontal Integration
**Definition**: Same modality, cross-batch integration (e.g., RNA from multiple batches)

**Evaluation Categories**: **BioC + SC + BER**

### Mosaic Integration  
**Definition**: Mixed modality and batch integration

**Evaluation Categories**: **BioC + SC + BER**

## Evaluation Pipeline

### 1. Method Execution and Clustering
```python
# Run integration method
adata = run_integration_method(rna_data, secondary_data)

# Apply multiple clustering methods
for clustering_method in ['leiden', 'louvain', 'kmeans', 'mclust']:
    adata = universal_clustering(adata, n_clusters, method=clustering_method)
```

### 2. Metric Calculation
```python
# Calculate appropriate metrics based on task and GT availability
if has_ground_truth:
    if task == 'vertical':
        metrics = calculate_bioc_sc_metrics(adata, gt_labels)  # 19 metrics
    else:  # horizontal or mosaic
        metrics = calculate_bioc_sc_ber_metrics(adata, gt_labels, batch_labels)
else:
    if task == 'vertical':
        metrics = calculate_limited_bioc_sc_metrics(adata)  # 5 metrics
    else:  # horizontal or mosaic
        metrics = calculate_limited_bioc_sc_ber_metrics(adata, batch_labels)
```

### 3. Result Storage
Individual results stored as:
`Results/evaluation/{task}/{gt_status}/method_results/{Method}_{Dataset}_{Clustering}_{GTStatus}_metrics.csv`

### 4. Aggregation and Normalization
- **Method-level aggregation**: Average across clustering methods for each method-dataset pair
- **Dataset normalization**: Min-max scaling within each dataset across methods
- **Cross-dataset averaging**: Mean performance across all datasets
- **Final ranking**: Comprehensive method ranking across all criteria

## Quality Control

### Metric Validation
- **Range checking**: Ensure all metrics fall within expected ranges
- **Sanity testing**: Verify metrics make biological sense
- **Reproducibility**: Fixed random seeds for all stochastic processes

### Statistical Analysis
- **Significance testing**: Paired comparisons between methods
- **Effect size**: Cohen's d for practical significance
- **Confidence intervals**: Bootstrap confidence intervals for metrics
- **Robustness**: Performance across different hyperparameter settings

## Implementation Notes

### Spatial Coordinates
- Moran's I and Geary's C require spatial coordinates
- Use `adata.obsm['spatial']` or `adata.obs[['x', 'y']]`
- Handle missing spatial information gracefully

### Graph Construction
- Graph-based metrics (cLISI, iLISI, Connectivity) require neighborhood graphs
- Use consistent k=50 neighbors for fair comparison
- Build graphs on integrated embeddings, not original data

### Batch Information
- BER metrics require batch labels in `adata.obs['batch']`
- Handle single-batch datasets (vertical integration) appropriately
- Validate batch label consistency across samples

## References

[49] Moran, P. A. (1950). Notes on continuous stochastic phenomena.
[50] Geary, R. C. (1954). The contiguity ratio and statistical mapping.
[51] Hubert, L. & Arabie, P. (1985). Comparing partitions.
[52] Strehl, A. & Ghosh, J. (2002). Cluster ensembles.
[53] Vinh, N. X., et al. (2010). Information theoretic measures.
[54] Fowlkes, E. B. & Mallows, C. L. (1983). A method for comparing two hierarchical clusterings.
[55] Manning, C. D., et al. (2008). Introduction to information retrieval.
[56] Rosenberg, A. & Hirschberg, J. (2007). V-measure.
[57] Jaccard, P. (1912). The distribution of the flora in the alpine zone.
[58] Dice, L. R. (1945). Measures of the amount of ecologic association.
[59] Van Rijsbergen, C. J. (1979). Information retrieval.
[60] Rousseeuw, P. J. (1987). Silhouettes.
[61] Calinski, T. & Harabasz, J. (1974). A dendrite method.
[62] Davies, D. L. & Bouldin, D. W. (1979). A cluster separation measure.
[63] Korsunsky, I., et al. (2019). Fast, sensitive and accurate integration.
[64] Büttner, M., et al. (2019). A test metric for assessing single-cell RNA-seq batch correction.