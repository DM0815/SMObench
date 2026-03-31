#!/bin/bash

# ==========================================================
# SpaMI Horizontal Integration Script
# Author: SMOBench Benchmark Framework
# Description:
#   Runs SpaMI horizontal integration across RNA+ADT and RNA+ATAC fusion datasets
#   Handles both withGT and woGT scenarios
#   Use relative paths for portability - run from SMOBench root directory
# ==========================================================

echo "Starting SpaMI horizontal integration processing..."
echo "Start time: $(date)"

# Create output directories
mkdir -p Results/adata/horizontal_integration/SpaMI Results/plot/horizontal_integration/SpaMI

# ==========================================================
# === withGT RNA+ADT Fusion Datasets ===
# ==========================================================
echo "Processing RNA+ADT fusion datasets with ground truth..."

# --- Human Lymph Nodes (HLN: A1 + D1) ---
echo "Processing Human_Lymph_Nodes fusion dataset..."
python Scripts/horizontal_integration/SpaMI/run_SpaMI.py \
  --RNA_path Dataset/fusionWithGT/RNA_ADT/HLN_Fusion_RNA.h5ad \
  --ADT_path Dataset/fusionWithGT/RNA_ADT/HLN_Fusion_ADT.h5ad \
  --save_path Results/adata/horizontal_integration/SpaMI/HLN/fusion/SpaMI_HLN_fusion.h5ad \
  --dataset HLN \
  --cluster_nums 10

# --- Human Tonsils (HT: S1 + S2 + S3) ---
echo "Processing Human_Tonsils fusion dataset..."
python Scripts/horizontal_integration/SpaMI/run_SpaMI.py \
  --RNA_path Dataset/fusionWithGT/RNA_ADT/HT_Fusion_RNA.h5ad \
  --ADT_path Dataset/fusionWithGT/RNA_ADT/HT_Fusion_ADT.h5ad \
  --save_path Results/adata/horizontal_integration/SpaMI/HT/fusion/SpaMI_HT_fusion.h5ad \
  --dataset HT \
  --cluster_nums 5


# ==========================================================
# === withGT RNA+ATAC Fusion Datasets ===
# ==========================================================
echo "Processing RNA+ATAC fusion datasets with ground truth..."

# --- Mouse Embryos S1 (MISAR_S1) ---
echo "Processing Mouse_Embryos_S1 fusion dataset..."
python Scripts/horizontal_integration/SpaMI/run_SpaMI.py \
  --RNA_path Dataset/fusionWithGT/RNA_ATAC/ME_S1_Fusion_RNA.h5ad \
  --ATAC_path Dataset/fusionWithGT/RNA_ATAC/ME_S1_Fusion_ATAC.h5ad \
  --save_path Results/adata/horizontal_integration/SpaMI/MISAR_S1/fusion/SpaMI_MISAR_S1_fusion.h5ad \
  --dataset MISAR_S1 \
  --cluster_nums 12

# --- Mouse Embryos S2 (MISAR_S2) ---
echo "Processing Mouse_Embryos_S2 fusion dataset..."
python Scripts/horizontal_integration/SpaMI/run_SpaMI.py \
  --RNA_path Dataset/fusionWithGT/RNA_ATAC/ME_S2_Fusion_RNA.h5ad \
  --ATAC_path Dataset/fusionWithGT/RNA_ATAC/ME_S2_Fusion_ATAC.h5ad \
  --save_path Results/adata/horizontal_integration/SpaMI/MISAR_S2/fusion/SpaMI_MISAR_S2_fusion.h5ad \
  --dataset MISAR_S2 \
  --cluster_nums 14


# ==========================================================
# === woGT RNA+ADT Fusion Datasets ===
# ==========================================================
echo "Processing RNA+ADT fusion datasets without ground truth..."

# --- Mouse Thymus ---
echo "Processing Mouse_Thymus fusion dataset..."
python Scripts/horizontal_integration/SpaMI/run_SpaMI.py \
  --RNA_path Dataset/fusionWoGT/RNA_ADT/Mouse_Thymus_Fusion_RNA.h5ad \
  --ADT_path Dataset/fusionWoGT/RNA_ADT/Mouse_Thymus_Fusion_ADT.h5ad \
  --save_path Results/adata/horizontal_integration/SpaMI/Mouse_Thymus/fusion/SpaMI_Mouse_Thymus_fusion.h5ad \
  --dataset Mouse_Thymus \
  --cluster_nums 8

# --- Mouse Spleen ---
echo "Processing Mouse_Spleen fusion dataset..."
python Scripts/horizontal_integration/SpaMI/run_SpaMI.py \
  --RNA_path Dataset/fusionWoGT/RNA_ADT/Mouse_Spleen_Fusion_RNA.h5ad \
  --ADT_path Dataset/fusionWoGT/RNA_ADT/Mouse_Spleen_Fusion_ADT.h5ad \
  --save_path Results/adata/horizontal_integration/SpaMI/Mouse_Spleen/fusion/SpaMI_Mouse_Spleen_fusion.h5ad \
  --dataset Mouse_Spleen \
  --cluster_nums 5


# ==========================================================
# === woGT RNA+ATAC Fusion Datasets (if available) ===
# ==========================================================
if [ -f "Dataset/fusionWoGT/RNA_ATAC/Mouse_Brain_Fusion_RNA.h5ad" ]; then
    echo "Processing Mouse_Brain fusion dataset..."
    python Scripts/horizontal_integration/SpaMI/run_SpaMI.py \
      --RNA_path Dataset/fusionWoGT/RNA_ATAC/Mouse_Brain_Fusion_RNA.h5ad \
      --ATAC_path Dataset/fusionWoGT/RNA_ATAC/Mouse_Brain_Fusion_ATAC.h5ad \
      --save_path Results/adata/horizontal_integration/SpaMI/Mouse_Brain/fusion/SpaMI_Mouse_Brain_fusion.h5ad \
      --dataset Mouse_Brain \
      --cluster_nums 18
else
    echo "Mouse_Brain fusion dataset not found, skipping..."
fi


# ==========================================================
# === Summary Report ===
# ==========================================================
echo "SpaMI horizontal integration processing completed!"
echo "End time: $(date)"
echo ""
echo "=== HORIZONTAL INTEGRATION SUMMARY ==="
echo "Results saved to Results/adata/horizontal_integration/SpaMI/"
echo ""
echo "Processed fusion datasets:"
find Results/adata/horizontal_integration/SpaMI -name "*.h5ad" | sort | while read file; do
    echo "  - $file"
done

echo ""
echo "Total fusion datasets processed: $(find Results/adata/horizontal_integration/SpaMI -name "*.h5ad" 2>/dev/null | wc -l)"
echo ""
echo "Key Notes:"
echo "  - Input: RNA + ADT / ATAC fusion datasets"
echo "  - Integration: Horizontal (batch effect removal across samples)"
echo "  - Method: SpaMI (contrastive multi-omics integration)"
echo "  - Output: Integrated embeddings + clustering results"
echo "  - Evaluation: Biological consistency + batch mixing metrics"
