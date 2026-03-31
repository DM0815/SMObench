# SMOBench Summary Scatter Plots

This directory contains scripts to generate summary scatter plots for SMOBench vertical integration results, adapted from the scib-reproducibility visualization framework.

## Overview

The scatter plots visualize the trade-off between **Spatial Coherence** (SC) and **Biological Conservation** (BioC) for spatial multi-omics integration methods. This visualization is specifically designed for vertical integration tasks where spatial relationships are crucial.

## Files

- `plotSMOBenchScatter.R` - Main R script for generating scatter plots
- `run_scatter_plots.sh` - Shell script for easy execution
- `README_scatter_plots.md` - This documentation file

## Generated Visualizations

### 1. Summary Scatter Plots (Individual Clustering Methods)
- **Four separate plots** for each clustering method: leiden, louvain, kmeans, mclust
- **X-axis**: Spatial Coherence Score (0-1, higher is better)
- **Y-axis**: Biological Conservation Score (0-1, higher is better)
- **Point size**: Overall Score (Total_Score = 0.5*SC + 0.5*BioC)
- **Point color**: Integration method (8 methods)
- **Point shape**: Data type (RNA+ADT vs RNA+ATAC)
- **Facets**: Dataset (7 datasets) - each dataset in separate panel
- **Purpose**: Comprehensive view of method performance across datasets

### 2. Best Methods Scatter Plots (Aggregated Performance)
- **Four separate plots** for each clustering method: leiden, louvain, kmeans, mclust
- **X-axis**: Spatial Coherence Score (mean across datasets)
- **Y-axis**: Biological Conservation Score (mean across datasets)
- **Point size**: Overall Score (aggregated)
- **Point color**: Integration method (all 8 methods)
- **Point shape**: Data type (RNA+ADT vs RNA+ATAC)
- **Error bars**: Standard error across datasets
- **Method labels**: Text labels for easy identification
- **Purpose**: Clean overview of overall method performance with uncertainty

### 3. Clustering Comparison Plot
- **Single comprehensive plot** comparing all clustering methods
- **Grid layout**: Clustering methods (rows) × Datasets (columns)
- Shows how clustering choice affects method performance

## Usage

### Quick Start
```bash
# From SMOBench-CLEAN root directory
bash Draw/run_scatter_plots.sh
```

### Manual Execution
```bash
# From SMOBench-CLEAN root directory
Rscript Draw/plotSMOBenchScatter.R
```

### Custom Parameters
```r
# In R console
source("Draw/plotSMOBenchScatter.R")

# Generate plots with custom settings
makeSMOBenchScatter(
  results_dir = "Results/evaluation/vertical_integration/final_results/",
  out_dir = "Results/plots/",
  clustering_methods = c("leiden", "louvain", "kmeans", "mclust"),
  weight_sc = 0.5
)
```

## Input Data Requirements

The script expects the following files in `Results/evaluation/vertical_integration/final_results/`:
- `detailed_results_leiden.csv`
- `detailed_results_louvain.csv`
- `detailed_results_kmeans.csv`
- `detailed_results_mclust.csv`

### Required Columns
- `Method`: Integration method name
- `Dataset`: Dataset identifier
- `Dataset_Type`: RNA_ADT or RNA_ATAC
- `SC_Score`: Spatial Coherence score
- `BioC_Score`: Biological Conservation score  
- `Total_Score`: Overall performance score
- `GT_Available`: Ground truth availability flag

## Output

### File Naming Convention
- Summary Scatter plots: `SMOBench_scatter_{clustering_method}.{format}`
- Best Methods Scatter plots: `SMOBench_best_methods_{clustering_method}.{format}`
- Comparison plot: `SMOBench_clustering_comparison.pdf`

### Formats
- **PDF**: Vector format, best for publications
- **PNG**: High-resolution raster, good for presentations
- **TIFF**: High-quality raster, good for print

### Dimensions
- Individual plots: A4 landscape (297×210mm)
- Comparison plot: A3 landscape (420×297mm)

## Interpretation Guide

### Ideal Performance Regions
- **Top-right quadrant**: High SC + High BioC (best methods)
- **Top-left**: High BioC, Low SC (biology preserved, spatial lost)
- **Bottom-right**: High SC, Low BioC (spatial preserved, biology lost)
- **Bottom-left**: Low SC + Low BioC (poor performance)

### Reference Lines
- **Blue dashed lines**: Median performance per dataset
- Help identify which methods perform above/below average

### Method Performance Patterns
- **Point clustering**: Methods with similar performance profiles
- **Size variation**: Overall score differences within methods
- **Shape patterns**: Data type preferences (RNA+ADT vs RNA+ATAC)

## Customization

### Method Colors
Edit `getSMOBenchMethodsPal()` function to change method colors:
```r
methods_pal <- c(
  "CANDIES" = "#E31A1C",     # Red
  "SpatialGlue" = "#1F78B4", # Blue
  "SpaMosaic" = "#33A02C",   # Green
  # ... add more methods
)
```

### Dataset Labels
Edit `loadSMOBenchScores()` function to change dataset display names:
```r
Dataset_Clean = case_when(
  Dataset == "HLN" ~ "Human Lymph Nodes",
  Dataset == "HT" ~ "Human Tonsils",
  # ... add more mappings
)
```

## Dependencies

### R Packages
```r
install.packages(c(
  "ggplot2", "dplyr", "readr", "purrr", 
  "stringr", "fs", "glue", "magrittr", "lubridate"
))
```

### System Requirements
- R 4.0+ 
- Sufficient disk space for output files (~50MB for all plots)
- Memory: ~2GB for processing all clustering results

## Troubleshooting

### Common Issues

1. **"Results directory not found"**
   - Ensure you're running from SMOBench-CLEAN root directory
   - Check that results files exist in the expected location

2. **"No data found for clustering method"**
   - Verify detailed_results_*.csv files exist and contain data
   - Check for missing SC_Score, BioC_Score, or Total_Score columns

3. **"Package not found"**
   - Install missing R packages using `install.packages()`

4. **Empty plots**
   - Check for NA values in score columns
   - Verify method names match between files

### Debug Mode
Add debug output by setting:
```r
options(verbose = TRUE)
```

## Contact

For issues specific to the visualization script, check:
1. Input data format and completeness
2. R package versions and compatibility
3. File permissions for output directory