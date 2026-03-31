#!/bin/bash

# ==========================================================
# SMOPCA Horizontal Integration Script
# Performs horizontal integration (RNA + ADT / RNA + ATAC)
# Consistent with SpatialGlue pipeline for SMOBench framework
# ==========================================================

echo "Starting SMOPCA horizontal integration processing..."
echo "Start time: $(date)"

# Create base results directories
mkdir -p Results/adata/horizontal_integration/SMOPCA Results/plot/horizontal_integration/SMOPCA

# === withGT RNA+ADT Fusion Datasets ===
echo "Processing RNA+ADT fusion datasets with ground truth..."

# Human_Lymph_Nodes fusion (combines A1 + D1)
echo "Processing Human_Lymph_Nodes fusion dataset..."
python Scripts/horizontal_integration/SMOPCA/run_SMOPCA.py \
  --data_type fusion \
  --RNA_path Dataset/fusionWithGT/RNA_ADT/HLN_Fusion_RNA.h5ad \
  --ADT_path Dataset/fusionWithGT/RNA_ADT/HLN_Fusion_ADT.h5ad \
  --save_path Results/adata/horizontal_integration/SMOPCA/HLN/fusion/SMOPCA_HLN_fusion.h5ad \
  --method SMOPCA \
  --dataset HLN \
  --cluster_nums 10

# Human_Tonsils fusion (combines S1 + S2 + S3)
echo "Processing Human_Tonsils fusion dataset..."
python Scripts/horizontal_integration/SMOPCA/run_SMOPCA.py \
  --data_type fusion \
  --RNA_path Dataset/fusionWithGT/RNA_ADT/HT_Fusion_RNA.h5ad \
  --ADT_path Dataset/fusionWithGT/RNA_ADT/HT_Fusion_ADT.h5ad \
  --save_path Results/adata/horizontal_integration/SMOPCA/HT/fusion/SMOPCA_HT_fusion.h5ad \
  --method SMOPCA \
  --dataset HT \
  --cluster_nums 5

# === withGT RNA+ATAC Fusion Datasets ===
echo "Processing RNA+ATAC fusion datasets with ground truth..."

# Mouse_Embryos_S1 fusion
echo "Processing Mouse_Embryos_S1 fusion dataset..."
python Scripts/horizontal_integration/SMOPCA/run_SMOPCA.py \
  --data_type fusion \
  --RNA_path Dataset/fusionWithGT/RNA_ATAC/ME_S1_Fusion_RNA.h5ad \
  --ATAC_path Dataset/fusionWithGT/RNA_ATAC/ME_S1_Fusion_ATAC.h5ad \
  --save_path Results/adata/horizontal_integration/SMOPCA/MISAR_S1/fusion/SMOPCA_MISAR_S1_fusion.h5ad \
  --method SMOPCA \
  --dataset MISAR_S1 \
  --cluster_nums 12

# Mouse_Embryos_S2 fusion
echo "Processing Mouse_Embryos_S2 fusion dataset..."
python Scripts/horizontal_integration/SMOPCA/run_SMOPCA.py \
  --data_type fusion \
  --RNA_path Dataset/fusionWithGT/RNA_ATAC/ME_S2_Fusion_RNA.h5ad \
  --ATAC_path Dataset/fusionWithGT/RNA_ATAC/ME_S2_Fusion_ATAC.h5ad \
  --save_path Results/adata/horizontal_integration/SMOPCA/MISAR_S2/fusion/SMOPCA_MISAR_S2_fusion.h5ad \
  --method SMOPCA \
  --dataset MISAR_S2 \
  --cluster_nums 14

# === woGT RNA+ADT Fusion Datasets ===
echo "Processing RNA+ADT fusion datasets without ground truth..."

# Mouse_Thymus fusion
echo "Processing Mouse_Thymus fusion dataset..."
python Scripts/horizontal_integration/SMOPCA/run_SMOPCA.py \
  --data_type fusion \
  --RNA_path Dataset/fusionWoGT/RNA_ADT/Mouse_Thymus_Fusion_RNA.h5ad \
  --ADT_path Dataset/fusionWoGT/RNA_ADT/Mouse_Thymus_Fusion_ADT.h5ad \
  --save_path Results/adata/horizontal_integration/SMOPCA/Mouse_Thymus/fusion/SMOPCA_Mouse_Thymus_fusion.h5ad \
  --method SMOPCA \
  --dataset Mouse_Thymus \
  --cluster_nums 8

# Mouse_Spleen fusion
echo "Processing Mouse_Spleen fusion dataset..."
python Scripts/horizontal_integration/SMOPCA/run_SMOPCA.py \
  --data_type fusion \
  --RNA_path Dataset/fusionWoGT/RNA_ADT/Mouse_Spleen_Fusion_RNA.h5ad \
  --ADT_path Dataset/fusionWoGT/RNA_ADT/Mouse_Spleen_Fusion_ADT.h5ad \
  --save_path Results/adata/horizontal_integration/SMOPCA/Mouse_Spleen/fusion/SMOPCA_Mouse_Spleen_fusion.h5ad \
  --method SMOPCA \
  --dataset Mouse_Spleen \
  --cluster_nums 5

# === woGT RNA+ATAC Fusion Datasets ===
if [ -f "Dataset/fusionWoGT/RNA_ATAC/Mouse_Brain_Fusion_RNA.h5ad" ]; then
    echo "Processing Mouse_Brain fusion dataset..."
    python Scripts/horizontal_integration/SMOPCA/run_SMOPCA.py \
      --data_type fusion \
      --RNA_path Dataset/fusionWoGT/RNA_ATAC/Mouse_Brain_Fusion_RNA.h5ad \
      --ATAC_path Dataset/fusionWoGT/RNA_ATAC/Mouse_Brain_Fusion_ATAC.h5ad \
      --save_path Results/adata/horizontal_integration/SMOPCA/Mouse_Brain/fusion/SMOPCA_Mouse_Brain_fusion.h5ad \
      --method SMOPCA \
      --dataset Mouse_Brain \
      --cluster_nums 18
else
    echo "Mouse_Brain fusion dataset not found, skipping..."
fi

echo "SMOPCA horizontal integration processing completed!"
echo "End time: $(date)"

# ==========================================================
# Summary report
# ==========================================================
echo "=== HORIZONTAL INTEGRATION PROCESSING SUMMARY (SMOPCA) ==="
echo "Results saved to Results/adata/horizontal_integration/SMOPCA/"
echo "Plots saved to Results/plot/horizontal_integration/SMOPCA/"
echo ""
echo "Processed fusion datasets:"
find Results/adata/horizontal_integration/SMOPCA -name "*.h5ad" | sort | while read file; do
    echo "  - $file"
done

echo ""
echo "Total fusion datasets processed: $(find Results/adata/horizontal_integration/SMOPCA -name "*.h5ad" 2>/dev/null | wc -l)"

echo ""
echo "Key points:"
echo "  - Input: Fusion datasets containing multiple batches/samples"
echo "  - Goal: Remove batch effects while preserving biological structure"
echo "  - Method: SMOPCA (probabilistic canonical analysis for spatial omics)"
echo "  - Output: Integrated embeddings with batch-corrected latent representations"
echo "  - Evaluation: Uses same clustering & metrics as SpatialGlue benchmark"
