# Evaluation Framework

Comprehensive evaluation system for spatial multi-omics integration methods using standardized metrics.

## Evaluation Scripts

**Core Evaluation:**
- `eval_adata.py`: Vertical integration evaluation (SC + BioC)
- `eval_horizontal_integration.py`: Horizontal integration evaluation (SC + BioC + BER)
- `generate_final_results.py`: Aggregate vertical integration results
- `generate_horizontal_results.py`: Aggregate horizontal integration results

**Validation:**
- `simple_validation.py`: Quick result validation
- `validate_compatibility.py`: Method-dataset compatibility check

## Usage

### Evaluate Integration Results
```bash
# Vertical integration (single method)
python eval_adata.py --method SpatialGlue --dataset HLN

# Horizontal integration (all methods)
python eval_horizontal_integration.py

# Generate final summaries
python generate_final_results.py
python generate_horizontal_results.py
```

### Validation
```bash
# Validate specific results
python simple_validation.py

# Check method compatibility
python validate_compatibility.py
```

## Evaluation Metrics

**Spatial Coherence (SC):**
- Moran's I: Spatial autocorrelation

**Biological Conservation (BioC):**
- With GT: ARI, NMI, ASW (cell type), Graph cLISI
- Without GT: Silhouette Coefficient, Davies-Bouldin Index, Calinski-Harabasz Index

**Batch Effect Removal (BER):**
- kBET, KNN connectivity, bASW, iLISI, PCR (horizontal/mosaic only)

## Output Format

Results saved as CSV files with standardized metrics and summary tables for method comparison.