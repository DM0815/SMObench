#!/bin/bash

# SpaMultiVAE Horizontal Integration Script
# Processes fusion datasets for batch effect removal and horizontal integration
# Use relative paths for portability - script should be run from SMOBench root directory
# NOTE: SpaMultiVAE only supports RNA+ADT data, not RNA+ATAC

echo "Starting SpaMultiVAE horizontal integration processing..."
echo "Start time: $(date)"

# Create base results directory for horizontal integration
mkdir -p Results/adata/horizontal_integration/SpaMultiVAE Results/plot/horizontal_integration/SpaMultiVAE

# === withGT RNA+ADT Fusion Datasets ===

echo "Processing RNA+ADT fusion datasets with ground truth..."

# Human_Lymph_Nodes fusion (combines A1 + D1)
echo "Processing Human_Lymph_Nodes fusion dataset..."
python Scripts/horizontal_integration/SpaMultiVAE/run_spamultivae.py \
  --data_type fusion \
  --RNA_path Dataset/fusionWithGT/RNA_ADT/HLN_Fusion_RNA.h5ad \
  --ADT_path Dataset/fusionWithGT/RNA_ADT/HLN_Fusion_ADT.h5ad \
  --save_path Results/adata/horizontal_integration/SpaMultiVAE/HLN/fusion/SpaMultiVAE_HLN_fusion.h5ad \
  --method SpaMultiVAE \
  --dataset HLN \
  --cluster_nums 10

# Human_Tonsils fusion (combines S1 + S2 + S3)
echo "Processing Human_Tonsils fusion dataset..."
python Scripts/horizontal_integration/SpaMultiVAE/run_spamultivae.py \
  --data_type fusion \
  --RNA_path Dataset/fusionWithGT/RNA_ADT/HT_Fusion_RNA.h5ad \
  --ADT_path Dataset/fusionWithGT/RNA_ADT/HT_Fusion_ADT.h5ad \
  --save_path Results/adata/horizontal_integration/SpaMultiVAE/HT/fusion/SpaMultiVAE_HT_fusion.h5ad \
  --method SpaMultiVAE \
  --dataset HT \
  --cluster_nums 5

# === woGT RNA+ADT Fusion Datasets ===

echo "Processing RNA+ADT fusion datasets without ground truth..."

# Mouse_Thymus fusion (combines multiple thymus samples)
echo "Processing Mouse_Thymus fusion dataset..."
python Scripts/horizontal_integration/SpaMultiVAE/run_spamultivae.py \
  --data_type fusion \
  --RNA_path Dataset/fusionWoGT/RNA_ADT/Mouse_Thymus_Fusion_RNA.h5ad \
  --ADT_path Dataset/fusionWoGT/RNA_ADT/Mouse_Thymus_Fusion_ADT.h5ad \
  --save_path Results/adata/horizontal_integration/SpaMultiVAE/Mouse_Thymus/fusion/SpaMultiVAE_Mouse_Thymus_fusion.h5ad \
  --method SpaMultiVAE \
  --dataset Mouse_Thymus \
  --cluster_nums 8

# Mouse_Spleen fusion (combines multiple spleen samples)
echo "Processing Mouse_Spleen fusion dataset..."
python Scripts/horizontal_integration/SpaMultiVAE/run_spamultivae.py \
  --data_type fusion \
  --RNA_path Dataset/fusionWoGT/RNA_ADT/Mouse_Spleen_Fusion_RNA.h5ad \
  --ADT_path Dataset/fusionWoGT/RNA_ADT/Mouse_Spleen_Fusion_ADT.h5ad \
  --save_path Results/adata/horizontal_integration/SpaMultiVAE/Mouse_Spleen/fusion/SpaMultiVAE_Mouse_Spleen_fusion.h5ad \
  --method SpaMultiVAE \
  --dataset Mouse_Spleen \
  --cluster_nums 5

# === Skip RNA+ATAC Datasets ===

echo "Skipping RNA+ATAC fusion datasets..."
echo "Note: SpaMultiVAE only supports RNA+ADT integration, not RNA+ATAC"
echo "MISAR_S1, MISAR_S2, and Mouse_Brain datasets are not processed by SpaMultiVAE"

echo "SpaMultiVAE horizontal integration processing completed!"
echo "End time: $(date)"

# Generate summary report
echo "=== HORIZONTAL INTEGRATION PROCESSING SUMMARY ==="
echo "Results saved to Results/adata/horizontal_integration/SpaMultiVAE/"
echo "Plots saved to Results/plot/horizontal_integration/SpaMultiVAE/"
echo ""
echo "Processed fusion datasets (RNA+ADT only):"
find Results/adata/horizontal_integration/SpaMultiVAE -name "*.h5ad" | sort | while read file; do
    echo "  - $file"
done

echo ""
echo "Total fusion datasets processed: $(find Results/adata/horizontal_integration/SpaMultiVAE -name "*.h5ad" 2>/dev/null | wc -l)"

echo ""
echo "Key differences from vertical integration:"
echo "  - Input: RNA+ADT fusion datasets only (no ATAC support)"
echo "  - Goal: Remove batch effects while preserving biological signals"
echo "  - Training: Enhanced SpaMultiVAE parameters for horizontal integration"
echo "  - Output: Integrated embeddings with reduced batch effects"
echo "  - Evaluation: Includes batch mixing metrics (BER) in addition to BioC and SC"
echo ""
echo "Note: MISAR_S1, MISAR_S2, and Mouse_Brain datasets skipped (RNA+ATAC not supported)"