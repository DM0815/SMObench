#!/bin/bash

# COSMOS Horizontal Integration Script
# Processes fusion datasets for batch effect removal and horizontal integration
# Use relative paths for portability - script should be run from SMOBench root directory

echo "Starting COSMOS horizontal integration processing..."
echo "Start time: $(date)"

# Create base results directory for horizontal integration
mkdir -p Results/adata/horizontal_integration/COSMOS Results/plot/horizontal_integration/COSMOS

# === withGT RNA+ADT Fusion Datasets ===

echo "Processing RNA+ADT fusion datasets with ground truth..."

# Human_Lymph_Nodes fusion (combines A1 + D1)
echo "Processing Human_Lymph_Nodes fusion dataset..."
python Scripts/horizontal_integration/COSMOS/run_cosmos.py \
  --data_type fusion \
  --RNA_path Dataset/fusionWithGT/RNA_ADT/HLN_Fusion_RNA.h5ad \
  --ADT_path Dataset/fusionWithGT/RNA_ADT/HLN_Fusion_ADT.h5ad \
  --save_path Results/adata/horizontal_integration/COSMOS/HLN/fusion/COSMOS_HLN_fusion.h5ad \
  --method COSMOS \
  --dataset HLN \
  --cluster_nums 10

# Human_Tonsils fusion (combines S1 + S2 + S3)
echo "Processing Human_Tonsils fusion dataset..."
python Scripts/horizontal_integration/COSMOS/run_cosmos.py \
  --data_type fusion \
  --RNA_path Dataset/fusionWithGT/RNA_ADT/HT_Fusion_RNA.h5ad \
  --ADT_path Dataset/fusionWithGT/RNA_ADT/HT_Fusion_ADT.h5ad \
  --save_path Results/adata/horizontal_integration/COSMOS/HT/fusion/COSMOS_HT_fusion.h5ad \
  --method COSMOS \
  --dataset HT \
  --cluster_nums 5

# === withGT RNA+ATAC Fusion Datasets ===

echo "Processing RNA+ATAC fusion datasets with ground truth..."

# Mouse_Embryos_S1 fusion (combines E11 + E13 + E15 + E18)
echo "Processing Mouse_Embryos_S1 fusion dataset..."
python Scripts/horizontal_integration/COSMOS/run_cosmos.py \
  --data_type fusion \
  --RNA_path Dataset/fusionWithGT/RNA_ATAC/ME_S1_Fusion_RNA.h5ad \
  --ATAC_path Dataset/fusionWithGT/RNA_ATAC/ME_S1_Fusion_ATAC.h5ad \
  --save_path Results/adata/horizontal_integration/COSMOS/MISAR_S1/fusion/COSMOS_MISAR_S1_fusion.h5ad \
  --method COSMOS \
  --dataset MISAR_S1 \
  --cluster_nums 12

# Mouse_Embryos_S2 fusion (combines E11 + E13 + E15 + E18)
echo "Processing Mouse_Embryos_S2 fusion dataset..."
python Scripts/horizontal_integration/COSMOS/run_cosmos.py \
  --data_type fusion \
  --RNA_path Dataset/fusionWithGT/RNA_ATAC/ME_S2_Fusion_RNA.h5ad \
  --ATAC_path Dataset/fusionWithGT/RNA_ATAC/ME_S2_Fusion_ATAC.h5ad \
  --save_path Results/adata/horizontal_integration/COSMOS/MISAR_S2/fusion/COSMOS_MISAR_S2_fusion.h5ad \
  --method COSMOS \
  --dataset MISAR_S2 \
  --cluster_nums 14

# === woGT RNA+ADT Fusion Datasets ===

echo "Processing RNA+ADT fusion datasets without ground truth..."

# Mouse_Thymus fusion (combines multiple thymus samples)
echo "Processing Mouse_Thymus fusion dataset..."
python Scripts/horizontal_integration/COSMOS/run_cosmos.py \
  --data_type fusion \
  --RNA_path Dataset/fusionWoGT/RNA_ADT/Mouse_Thymus_Fusion_RNA.h5ad \
  --ADT_path Dataset/fusionWoGT/RNA_ADT/Mouse_Thymus_Fusion_ADT.h5ad \
  --save_path Results/adata/horizontal_integration/COSMOS/Mouse_Thymus/fusion/COSMOS_Mouse_Thymus_fusion.h5ad \
  --method COSMOS \
  --dataset Mouse_Thymus \
  --cluster_nums 8

# Mouse_Spleen fusion (combines multiple spleen samples)
echo "Processing Mouse_Spleen fusion dataset..."
python Scripts/horizontal_integration/COSMOS/run_cosmos.py \
  --data_type fusion \
  --RNA_path Dataset/fusionWoGT/RNA_ADT/Mouse_Spleen_Fusion_RNA.h5ad \
  --ADT_path Dataset/fusionWoGT/RNA_ADT/Mouse_Spleen_Fusion_ADT.h5ad \
  --save_path Results/adata/horizontal_integration/COSMOS/Mouse_Spleen/fusion/COSMOS_Mouse_Spleen_fusion.h5ad \
  --method COSMOS \
  --dataset Mouse_Spleen \
  --cluster_nums 5

# === woGT RNA+ATAC Fusion Datasets (if available) ===

# Check for Mouse_Brain fusion datasets
if [ -f "Dataset/fusionWoGT/RNA_ATAC/Mouse_Brain_Fusion_RNA.h5ad" ]; then
    echo "Processing Mouse_Brain fusion dataset..."
    python Scripts/horizontal_integration/COSMOS/run_cosmos.py \
      --data_type fusion \
      --RNA_path Dataset/fusionWoGT/RNA_ATAC/Mouse_Brain_Fusion_RNA.h5ad \
      --ATAC_path Dataset/fusionWoGT/RNA_ATAC/Mouse_Brain_Fusion_ATAC.h5ad \
      --save_path Results/adata/horizontal_integration/COSMOS/Mouse_Brain/fusion/COSMOS_Mouse_Brain_fusion.h5ad \
      --method COSMOS \
      --dataset Mouse_Brain \
      --cluster_nums 18
else
    echo "Mouse_Brain fusion dataset not found, skipping..."
fi

echo "COSMOS horizontal integration processing completed!"
echo "End time: $(date)"

# Generate summary report
echo "=== HORIZONTAL INTEGRATION PROCESSING SUMMARY ==="
echo "Results saved to Results/adata/horizontal_integration/COSMOS/"
echo "Plots saved to Results/plot/horizontal_integration/COSMOS/"
echo ""
echo "Processed fusion datasets:"
find Results/adata/horizontal_integration/COSMOS -name "*.h5ad" | sort | while read file; do
    echo "  - $file"
done

echo ""
echo "Total fusion datasets processed: $(find Results/adata/horizontal_integration/COSMOS -name "*.h5ad" 2>/dev/null | wc -l)"

echo ""
echo "Key differences from vertical integration:"
echo "  - Input: Fusion datasets containing multiple batches/samples"
echo "  - Goal: Remove batch effects while preserving biological signals"
echo "  - Training: Optimized hyperparameters for horizontal integration"
echo "  - Output: Integrated embeddings with reduced batch effects"
echo "  - Evaluation: Includes batch mixing metrics (BER) in addition to BioC and SC"