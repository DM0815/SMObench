# SMOBench Research Workflow

## Overview
This document outlines the systematic workflow for benchmarking spatial multi-omics integration methods in SMOBench. The workflow ensures reproducible, comprehensive evaluation across methods, datasets, and clustering approaches.

## Phase 1: Environment Setup and Data Preparation

### 1.1 Environment Initialization
```bash
# Activate conda environment
conda activate smobench

# Verify GPU availability
nvidia-smi

# Set CUDA device if needed
export CUDA_VISIBLE_DEVICES=0
```

### 1.2 Dataset Organization
The datasets are organized in the standardized structure:
```
Dataset/
├── withGT/          # Datasets with ground truth annotations
└── woGT/            # Datasets without ground truth
```

### 1.3 Data Verification
```python
# Verify data integrity and structure
conda run -n smobench python Dataset/data_info/generate_data_info.py
```

## Phase 2: Method Execution Pipeline

### 2.1 Individual Method Testing
For each integration method, follow this sequence:

1. **Parameter Configuration**: Set method-specific hyperparameters
2. **Data Loading**: Load RNA and secondary modality (ADT/ATAC)
3. **Preprocessing**: Apply var_names_make_unique() and method-specific preprocessing
4. **Integration**: Execute spatial multi-omics integration
5. **Result Storage**: Save integrated embeddings in AnnData format

### 2.2 Batch Execution Framework
Use the PBS template system for systematic evaluation:

```bash
# Generate PBS scripts from template
python Scripts/generate_pbs_scripts.py

# Submit batch jobs
for pbs in run_*.pbs; do qsub $pbs; done
```

### 2.3 Method-Specific Execution

#### SpatialGlue
```bash
conda run -n smobench python Scripts/integration/SpatialGlue/run_SpatialGlue.py \
  --data_type 10x \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Tonsils/S1/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Tonsils/S1/adata_ADT.h5ad \
  --save_path Results/adata/SpatialGlue/HT/S1/SpatialGlue_HT_S1.h5ad
```

#### SpaMosaic
```bash
conda run -n smobench python Scripts/integration/SpaMosaic/run_SpaMosaic.py \
  --dataset Human_Tonsils \
  --sample S1 \
  --output_dir Results/adata/SpaMosaic/
```

## Phase 3: Universal Clustering Application

### 3.1 Clustering Interface
Apply multiple clustering methods to each integration result:

```python
from Utils.SMOBench_clustering import universal_clustering

# Load integration result
adata = sc.read_h5ad('Results/adata/SpatialGlue/HT/S1/SpatialGlue_HT_S1.h5ad')

# Apply all clustering methods
for method in ['leiden', 'louvain', 'kmeans', 'mclust']:
    adata = universal_clustering(
        adata, 
        n_clusters=6, 
        used_obsm='spatial_emb',  # or method-specific embedding key
        method=method, 
        key=f'{method}_clusters'
    )

# Save updated result
adata.write('Results/adata/SpatialGlue/HT/S1/SpatialGlue_HT_S1_clustered.h5ad')
```

### 3.2 Parameter Optimization
For leiden/louvain methods:
- Automatic resolution search to match target cluster numbers
- Range: 0.1 to 3.0 with 0.01 increments
- Early stopping when target cluster count is achieved

## Phase 4: Comprehensive Evaluation Pipeline

### 4.1 Evaluation Framework Overview

SMOBench employs three categories of evaluation metrics:

1. **Spatial Coherence (SC)**: Moran's I, Geary's C  
2. **Biological Conservation (BioC)**: Clustering accuracy and quality metrics
3. **Batch Effect Removal (BER)**: Batch mixing metrics (horizontal/mosaic only)

**Task-Specific Evaluation**:
- **Vertical Integration**: BioC + SC (19 metrics with GT, 5 without GT)
- **Horizontal/Mosaic Integration**: BioC + SC + BER

### 4.2 Vertical Integration Evaluation

#### 4.2.1 With Ground Truth Datasets (Full BioC + SC)
```python
# Calculate comprehensive metrics (19 metrics)
conda run -n smobench python Scripts/evaluation/metrics_calculator.py \
  --adata Results/adata/SpatialGlue/HT/S1/SpatialGlue_HT_S1_clustered.h5ad \
  --cluster-keys leiden louvain kmeans mclust \
  --task vertical \
  --embedding-key spatial_emb \
  --gt-key final_annot \
  --output-dir Results/evaluation/
```

**Metrics Calculated** (19 total):
- **Clustering Accuracy**: ARI, NMI, AMI, FMI, Purity, Homogeneity, Completeness, V-measure, F-measure, Jaccard Index, Dice Index
- **Quality Metrics**: Silhouette Coefficient, Calinski-Harabasz Index, Davies-Bouldin Index
- **Additional BioC**: ASW (cell type), Graph cLISI  
- **Spatial Coherence**: Moran's I, Geary's C

#### 4.2.2 Without Ground Truth Datasets (Limited BioC + SC)
```python
# Calculate limited metrics (5 metrics)
conda run -n smobench python Scripts/evaluation/metrics_calculator.py \
  --adata Results/adata/SpatialGlue/MS/Spleen1/SpatialGlue_MS_Spleen1_clustered.h5ad \
  --cluster-keys leiden louvain kmeans mclust \
  --task vertical \
  --embedding-key spatial_emb \
  --output-dir Results/evaluation/
```

**Metrics Calculated** (5 total):
- **Quality Metrics**: Silhouette Coefficient, Calinski-Harabasz Index, Davies-Bouldin Index
- **Spatial Coherence**: Moran's I, Geary's C

### 4.3 Horizontal/Mosaic Integration Evaluation

```python
# Calculate full metrics including BER
conda run -n smobench python Scripts/evaluation/metrics_calculator.py \
  --adata Results/adata/Method/Dataset/clustered.h5ad \
  --cluster-keys leiden louvain kmeans mclust \
  --task horizontal \
  --embedding-key integrated_emb \
  --gt-key cell_type \
  --batch-key batch \
  --output-dir Results/evaluation/
```

**Additional BER Metrics**:
- ASW (batch), Graph iLISI, kBET, Graph Connectivity

### 4.4 Batch Evaluation Processing

#### 4.4.1 Systematic Dataset Processing
```bash
# Process all results by task type
for task in vertical horizontal mosaic; do
  for gt_status in withGT woGT; do
    conda run -n smobench python Scripts/evaluation/batch_evaluate.py \
      --task $task \
      --gt_status $gt_status \
      --results_dir Results/adata/ \
      --output_dir Results/evaluation/$task/$gt_status/
  done
done
```

#### 4.4.2 Individual Method Evaluation Script
```python
#!/usr/bin/env python3
import scanpy as sc
from Scripts.evaluation.metrics_calculator import evaluate_method_result

# Example: Evaluate SpatialGlue on Human Tonsils S1
results = evaluate_method_result(
    adata_path='Results/adata/SpatialGlue/HT/S1/SpatialGlue_HT_S1_clustered.h5ad',
    cluster_keys=['leiden', 'louvain', 'kmeans', 'mclust'],
    task='vertical',
    embedding_key='spatial_emb', 
    gt_key='final_annot',
    output_dir='Results/evaluation/'
)
```

### 4.5 Result File Structure

Evaluation results are saved with the following structure:
```
Results/evaluation/
├── vertical/
│   ├── withGT/
│   │   ├── method_results/
│   │   │   ├── SpatialGlue_HT_S1_leiden_metrics.csv
│   │   │   ├── SpatialGlue_HT_S1_louvain_metrics.csv
│   │   │   └── ... (all method-dataset-clustering combinations)
│   │   └── aggregated/
│   │       ├── vertical_withGT_summary.csv
│   │       └── vertical_withGT_rankings.csv
│   └── woGT/
│       ├── method_results/
│       └── aggregated/
├── horizontal/
└── mosaic/
```

#### Example Output Files

**With Ground Truth** (SpatialGlue_HT_S1_leiden_metrics.csv):
```csv
Metric,Value
asw_celltype,0.511
graph_clisi,0.886  
ARI,0.104
NMI,0.274
FMI,0.313
Silhouette Coefficient,0.147
Calinski-Harabasz Index,893.968
Davies-Bouldin Index,1.673
Purity,0.750
AMI,0.272
Homogeneity,0.450
Completeness,0.197
V-measure,0.274
F-measure,0.241
Jaccard Index,0.137
Dice Index,0.241
Moran Index,0.448
Geary C,0.554
```

**Without Ground Truth** (SpatialGlue_MS_Spleen1_leiden_metrics.csv):
```csv
Metric,Value
Silhouette Coefficient,0.147
Calinski-Harabasz Index,893.968
Davies-Bouldin Index,1.673
Moran Index,0.448
Geary C,0.554
```

## Phase 5: Result Aggregation and Analysis

### 5.1 Metric Standardization
Within each dataset:
1. Calculate min/max values for each metric across all methods
2. Apply min-max normalization: `(value - min) / (max - min)`
3. Handle directionality (higher-is-better vs lower-is-better metrics)

### 5.2 Cross-Dataset Aggregation
```python
conda run -n smobench python Scripts/evaluation/generate_final_results.py \
  --metrics_dir Results/metrics/ \
  --output_path Results/final_benchmark.csv
```

Aggregation steps:
1. **Method-Dataset Averaging**: Mean across clustering methods for each method-dataset pair
2. **Batch Averaging**: Mean across sample replicates (S1, S2) within tissue types  
3. **Task-Specific Averaging**: Separate averages for vertical, horizontal, mosaic tasks
4. **Overall Ranking**: Weighted average across all evaluation criteria

### 5.3 Statistical Analysis
- Significance testing between methods (paired t-test, Wilcoxon)
- Effect size calculation (Cohen's d)
- Confidence intervals for performance metrics
- Robustness analysis across hyperparameter variations

## Phase 6: Visualization and Reporting

### 6.1 Spatial Visualizations
```python
conda run -n smobench python Scripts/visualization/plot_spatial_results.py \
  --adata_path Results/adata/SpatialGlue/HT/S1/SpatialGlue_HT_S1_clustered.h5ad \
  --output_dir Results/plots/SpatialGlue/Human_Tonsils/S1/
```

Generated plots:
- Spatial scatter plots colored by cluster assignment
- UMAP embeddings of integrated representations
- Side-by-side comparison: predicted vs ground truth (if available)

### 6.2 Performance Heatmaps
```python
conda run -n smobench python Scripts/visualization/plot_benchmark_heatmap.py \
  --results_path Results/final_benchmark.csv \
  --output_path Results/plots/benchmark_heatmap.png
```

### 6.3 Method Comparison Plots
- Box plots of performance distributions
- Radar charts for multi-metric comparison
- Scatter plots of metric correlations
- Runtime and memory usage comparison

## Phase 7: Quality Control and Reproducibility

### 7.1 Result Validation
- Cross-check results with known method implementations
- Verify metric calculations with reference implementations
- Test on synthetic datasets with known ground truth

### 7.2 Documentation Updates
- Update status.md with current progress
- Document any parameter modifications or workflow changes
- Record computational resource usage and runtime statistics

### 7.3 Version Control
```bash
# Commit major workflow milestones
git add .
git commit -m "Complete benchmark evaluation for [method/dataset]"
git push origin main
```

## Phase 8: Advanced Analysis (Optional)

### 8.1 Method Failure Analysis
- Identify datasets where methods consistently underperform
- Analyze correlation between dataset characteristics and method performance
- Document method-specific limitations and optimal use cases

### 8.2 Ensemble Methods
- Combine predictions from multiple integration methods
- Evaluate ensemble performance vs individual methods
- Identify complementary method combinations

### 8.3 Hyperparameter Sensitivity
- Systematic variation of key hyperparameters
- Performance robustness analysis
- Optimal hyperparameter identification for each tissue type

## Error Handling and Troubleshooting

### Common Issues
1. **GPU Memory**: Use `nvidia-smi` to check availability, clear cache with `torch.cuda.empty_cache()`
2. **R Interface**: Ensure rpy2 and mclust are properly installed for mclust clustering
3. **Data Format**: Verify AnnData objects have required obsm keys for embeddings
4. **Missing Dependencies**: Use conda to install method-specific requirements

### Recovery Procedures
- Checkpoint intermediate results after each major processing step
- Implement automatic retry logic for transient failures
- Maintain backup copies of processed datasets and results