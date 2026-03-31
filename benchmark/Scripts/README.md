# Execution Scripts

Standardized workflow scripts for running spatial multi-omics integration methods and evaluation.

## Directory Structure

### vertical_integration/
Method-specific scripts for cross-modality integration within samples.

**Usage:**
```bash
bash Scripts/vertical_integration/SpatialGlue/run.sh
```

**Methods Available:**
- `CANDIES/`
- `COSMOS/`
- `PRAGA/`
- `PRESENT/`
- `SpaMV/`
- `SpaMosaic/`
- `SpatialGlue/`
- `SpaMultiVAE/`

### horizontal_integration/
Scripts for cross-sample integration with batch effect removal.

**Usage:**
```bash  
bash Scripts/horizontal_integration/SpatialGlue/run.sh
```

### data_preparation/
Data preprocessing and format conversion scripts.

### evaluation/
Evaluation workflow scripts for metrics calculation.

## Script Execution

**Important:** Always use `bash` instead of `sh` for script execution:
```bash
# Correct
bash Scripts/vertical_integration/SpatialGlue/run.sh

# Incorrect  
sh Scripts/vertical_integration/SpatialGlue/run.sh
```

## Workflow

1. **Data Preparation**: Preprocess datasets using `data_preparation/`
2. **Integration**: Run methods using `vertical_integration/` or `horizontal_integration/`
3. **Evaluation**: Calculate metrics using `evaluation/` scripts
4. **Results**: Access outputs in `Results/` directory