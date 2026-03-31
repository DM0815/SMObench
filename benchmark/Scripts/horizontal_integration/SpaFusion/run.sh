#!/bin/bash

# SpaFusion Horizontal Integration Script
# Processes fusion datasets for batch effect removal and horizontal integration
# Use relative paths for portability - script should be run from SMOBench root directory
# As SpaFusion cannot deal with ATAC data, RNA+ATAC datasets have been annotated.


echo "Starting SpaFusion horizontal integration processing..."
echo "Start time: $(date)"

# Create base results directory for horizontal integration
mkdir -p Results/adata/horizontal_integration/SpaFusion Results/plot/horizontal_integration/SpaFusion

# === withGT RNA+ADT Fusion Datasets ===

echo "Processing RNA+ADT fusion datasets with ground truth..."

# Human_Lymph_Nodes fusion (combines A1 + D1)
echo "Processing Human_Lymph_Nodes fusion dataset..."
python Scripts/horizontal_integration/SpaFusion/run_SpaFusion.py \
  --RNA_path Dataset/fusionWithGT/RNA_ADT/HLN_Fusion_RNA.h5ad \
  --ADT_path Dataset/fusionWithGT/RNA_ADT/HLN_Fusion_ADT.h5ad \
  --save_path Results/adata/horizontal_integration/SpaFusion/HLN/fusion/SpaFusion_HLN_fusion.h5ad \
  --dataset HLN \
  --cluster_nums 10 \
  --device cuda:0 \
  --pretrain_epoch 100 \
  --train_epoch 100

# Human_Tonsils fusion (combines S1 + S2 + S3)
echo "Processing Human_Tonsils fusion dataset..."
python Scripts/horizontal_integration/SpaFusion/run_SpaFusion.py \
  --RNA_path Dataset/fusionWithGT/RNA_ADT/HT_Fusion_RNA.h5ad \
  --ADT_path Dataset/fusionWithGT/RNA_ADT/HT_Fusion_ADT.h5ad \
  --save_path Results/adata/horizontal_integration/SpaFusion/HT/fusion/SpaFusion_HT_fusion.h5ad \
  --dataset HT \
  --cluster_nums 5 \
  --device cuda:0 \
  --pretrain_epoch 100 \
  --train_epoch 100


# === withGT RNA+ATAC Fusion Datasets ===

# echo "Processing RNA+ATAC fusion datasets with ground truth..."

# # Mouse_Embryos_S1 fusion (E11 + E13 + E15 + E18)
# echo "Processing Mouse_Embryos_S1 fusion dataset..."
# python Scripts/horizontal_integration/SpaFusion/run_SpaFusion.py \
#   --RNA_path Dataset/fusionWithGT/RNA_ATAC/ME_S1_Fusion_RNA.h5ad \
#   --ADT_path Dataset/fusionWithGT/RNA_ATAC/ME_S1_Fusion_ATAC.h5ad \
#   --save_path Results/adata/horizontal_integration/SpaFusion/MISAR_S1/fusion/SpaFusion_MISAR_S1_fusion.h5ad \
#   --dataset MISAR_S1 \
#   --cluster_nums 12 \
#   --device cuda:0 \
#   --pretrain_epoch 100 \
#   --train_epoch 100

# # Mouse_Embryos_S2 fusion (E11 + E13 + E15 + E18)
# echo "Processing Mouse_Embryos_S2 fusion dataset..."
# python Scripts/horizontal_integration/SpaFusion/run_SpaFusion.py \
#   --RNA_path Dataset/fusionWithGT/RNA_ATAC/ME_S2_Fusion_RNA.h5ad \
#   --ADT_path Dataset/fusionWithGT/RNA_ATAC/ME_S2_Fusion_ATAC.h5ad \
#   --save_path Results/adata/horizontal_integration/SpaFusion/MISAR_S2/fusion/SpaFusion_MISAR_S2_fusion.h5ad \
#   --dataset MISAR_S2 \
#   --cluster_nums 14 \
#   --device cuda:0 \
#   --pretrain_epoch 100 \
#   --train_epoch 100


# === woGT RNA+ADT Fusion Datasets ===

echo "Processing RNA+ADT fusion datasets without ground truth..."

# Mouse_Thymus fusion
echo "Processing Mouse_Thymus fusion dataset..."
python Scripts/horizontal_integration/SpaFusion/run_SpaFusion.py \
  --RNA_path Dataset/fusionWoGT/RNA_ADT/Mouse_Thymus_Fusion_RNA.h5ad \
  --ADT_path Dataset/fusionWoGT/RNA_ADT/Mouse_Thymus_Fusion_ADT.h5ad \
  --save_path Results/adata/horizontal_integration/SpaFusion/Mouse_Thymus/fusion/SpaFusion_Mouse_Thymus_fusion.h5ad \
  --dataset Mouse_Thymus \
  --cluster_nums 8 \
  --device cuda:0

# Mouse_Spleen fusion
echo "Processing Mouse_Spleen fusion dataset..."
python Scripts/horizontal_integration/SpaFusion/run_SpaFusion.py \
  --RNA_path Dataset/fusionWoGT/RNA_ADT/Mouse_Spleen_Fusion_RNA.h5ad \
  --ADT_path Dataset/fusionWoGT/RNA_ADT/Mouse_Spleen_Fusion_ADT.h5ad \
  --save_path Results/adata/horizontal_integration/SpaFusion/Mouse_Spleen/fusion/SpaFusion_Mouse_Spleen_fusion.h5ad \
  --dataset Mouse_Spleen \
  --cluster_nums 5 \
  --device cuda:0


# # === woGT RNA+ATAC Fusion Datasets (if available) ===

# if [ -f "Dataset/fusionWoGT/RNA_ATAC/Mouse_Brain_Fusion_RNA.h5ad" ]; then
#     echo "Processing Mouse_Brain fusion dataset..."
#     python Scripts/horizontal_integration/SpaFusion/run_SpaFusion.py \
#       --RNA_path Dataset/fusionWoGT/RNA_ATAC/Mouse_Brain_Fusion_RNA.h5ad \
#       --ADT_path Dataset/fusionWoGT/RNA_ATAC/Mouse_Brain_Fusion_ATAC.h5ad \
#       --save_path Results/adata/horizontal_integration/SpaFusion/Mouse_Brain/fusion/SpaFusion_Mouse_Brain_fusion.h5ad \
#       --dataset Mouse_Brain \
#       --cluster_nums 18 \
#       --device cuda:0
# else
#     echo "Mouse_Brain fusion dataset not found, skipping..."
# fi


echo "SpaFusion horizontal integration processing completed!"
echo "End time: $(date)"

# Generate summary report
echo "=== HORIZONTAL INTEGRATION PROCESSING SUMMARY ==="
echo "Results saved to Results/adata/horizontal_integration/SpaFusion/"
echo "Plots saved to Results/plot/horizontal_integration/SpaFusion/"
echo ""
echo "Processed fusion datasets:"
find Results/adata/horizontal_integration/SpaFusion -name "*.h5ad" | sort | while read file; do
    echo "  - $file"
done

echo ""
echo "Total fusion datasets processed: $(find Results/adata/horizontal_integration/SpaFusion -name "*.h5ad" 2>/dev/null | wc -l)"
