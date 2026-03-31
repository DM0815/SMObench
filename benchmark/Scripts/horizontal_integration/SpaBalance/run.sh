#!/bin/bash

# SpaBalance Horizontal Integration Script
# Processes fusion datasets for batch effect removal and horizontal integration

echo "Starting SpaBalance horizontal integration processing..."
echo "Start time: $(date)"

mkdir -p Results/adata/horizontal_integration/SpaBalance Results/plot/horizontal_integration/SpaBalance

# === withGT RNA+ADT Fusion Datasets ===
echo "Processing RNA+ADT fusion datasets with ground truth..."

echo "Processing Human_Lymph_Nodes fusion dataset..."
python Scripts/horizontal_integration/SpaBalance/run_SpaBalance.py \
  --data_type fusion \
  --RNA_path Dataset/fusionWithGT/RNA_ADT/HLN_Fusion_RNA.h5ad \
  --ADT_path Dataset/fusionWithGT/RNA_ADT/HLN_Fusion_ADT.h5ad \
  --save_path Results/adata/horizontal_integration/SpaBalance/HLN/fusion/SpaBalance_HLN_fusion.h5ad \
  --dataset HLN \
  --cluster_nums 10

echo "Processing Human_Tonsils fusion dataset..."
python Scripts/horizontal_integration/SpaBalance/run_SpaBalance.py \
 --data_type fusion \
  --RNA_path Dataset/fusionWithGT/RNA_ADT/HT_Fusion_RNA.h5ad \
  --ADT_path Dataset/fusionWithGT/RNA_ADT/HT_Fusion_ADT.h5ad \
  --save_path Results/adata/horizontal_integration/SpaBalance/HT/fusion/SpaBalance_HT_fusion.h5ad \
  --dataset HT \
  --cluster_nums 5

# === withGT RNA+ATAC Fusion Datasets ===
echo "Processing RNA+ATAC fusion datasets with ground truth..."

python Scripts/horizontal_integration/SpaBalance/run_SpaBalance.py \
  --data_type fusion \
  --RNA_path Dataset/fusionWithGT/RNA_ATAC/ME_S1_Fusion_RNA.h5ad \
  --ATAC_path Dataset/fusionWithGT/RNA_ATAC/ME_S1_Fusion_ATAC.h5ad \
  --save_path Results/adata/horizontal_integration/SpaBalance/MISAR_S1/fusion/SpaBalance_MISAR_S1_fusion.h5ad \
  --dataset MISAR_S1 \
  --cluster_nums 12

python Scripts/horizontal_integration/SpaBalance/run_SpaBalance.py \
  --data_type fusion \
  --RNA_path Dataset/fusionWithGT/RNA_ATAC/ME_S2_Fusion_RNA.h5ad \
  --ATAC_path Dataset/fusionWithGT/RNA_ATAC/ME_S2_Fusion_ATAC.h5ad \
  --save_path Results/adata/horizontal_integration/SpaBalance/MISAR_S2/fusion/SpaBalance_MISAR_S2_fusion.h5ad \
  --dataset MISAR_S2 \
  --cluster_nums 14

# === woGT RNA+ADT Fusion Datasets ===
echo "Processing RNA+ADT fusion datasets without ground truth..."

python Scripts/horizontal_integration/SpaBalance/run_SpaBalance.py \
  --data_type fusion \ 
  --RNA_path Dataset/fusionWoGT/RNA_ADT/Mouse_Thymus_Fusion_RNA.h5ad \
  --ADT_path Dataset/fusionWoGT/RNA_ADT/Mouse_Thymus_Fusion_ADT.h5ad \
  --save_path Results/adata/horizontal_integration/SpaBalance/Mouse_Thymus/fusion/SpaBalance_Mouse_Thymus_fusion.h5ad \
  --dataset Mouse_Thymus \
  --cluster_nums 8

python Scripts/horizontal_integration/SpaBalance/run_SpaBalance.py \
  --data_type fusion \
  --RNA_path Dataset/fusionWoGT/RNA_ADT/Mouse_Spleen_Fusion_RNA.h5ad \
  --ADT_path Dataset/fusionWoGT/RNA_ADT/Mouse_Spleen_Fusion_ADT.h5ad \
  --save_path Results/adata/horizontal_integration/SpaBalance/Mouse_Spleen/fusion/SpaBalance_Mouse_Spleen_fusion.h5ad \
  --dataset Mouse_Spleen \
  --cluster_nums 5

# === woGT RNA+ATAC Fusion Datasets ===
if [ -f "Dataset/fusionWoGT/RNA_ATAC/Mouse_Brain_Fusion_RNA.h5ad" ]; then
    echo "Processing Mouse_Brain fusion dataset..."
    python Scripts/horizontal_integration/SpaBalance/run_SpaBalance.py \
  --data_type fusion \
      --RNA_path Dataset/fusionWoGT/RNA_ATAC/Mouse_Brain_Fusion_RNA.h5ad \
      --ATAC_path Dataset/fusionWoGT/RNA_ATAC/Mouse_Brain_Fusion_ATAC.h5ad \
      --save_path Results/adata/horizontal_integration/SpaBalance/Mouse_Brain/fusion/SpaBalance_Mouse_Brain_fusion.h5ad \
      --dataset Mouse_Brain \
      --cluster_nums 18
else
    echo "Mouse_Brain fusion dataset not found, skipping..."
fi

echo "SpaBalance horizontal integration processing completed!"
echo "End time: $(date)"

# Summary
echo ""
echo "=== HORIZONTAL INTEGRATION SUMMARY (SpaBalance) ==="
find Results/adata/horizontal_integration/SpaBalance -name "*.h5ad" | sort
echo ""
echo "Output directory: Results/adata/horizontal_integration/SpaBalance/"
