#!/bin/bash

# SpaMultiVAE run script for RNA+ADT datasets ONLY
# IMPORTANT: SpaMultiVAE only supports RNA+Proteome (ADT) integration
# RNA+ATAC datasets are commented out - use SpatialGlue for RNA+ATAC
# Use relative paths for portability - script should be run from SMOBench root directory
# Cluster numbers: HLN_A1:10 HLN_D1:11 HT_S1:4 HT_S2:5 HT_S3:5 mouse_thymus:8 mouse_spleen:5 mouse_brain:18 3M:5 MISAR_S1:8,12,12,14 MISAR_S2:13,14,15,16

echo "Starting SpaMultiVAE processing (RNA+ADT datasets only)..."
echo "Start time: $(date)"

# Create base results directory
mkdir -p Results/adata/SpaMultiVAE Results/plot/SpaMultiVAE

# === withGT RNA+ADT Datasets ===

echo "Processing Human_Lymph_Nodes datasets..."

# Human_Lymph_Nodes A1
echo "Processing Human_Lymph_Nodes A1..."
python Scripts/integration/SpaMultiVAE/run_spamultivae.py \
  --data_type 10x \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/A1/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/A1/adata_ADT.h5ad \
  --save_path Results/adata/SpaMultiVAE/HLN/A1/SpaMultiVAE_HLN_A1.h5ad \
  --method SpaMultiVAE \
  --dataset Human_Lymph_Nodes/A1 \
  --cluster_nums 10

# Human_Lymph_Nodes D1
echo "Processing Human_Lymph_Nodes D1..."
python Scripts/integration/SpaMultiVAE/run_spamultivae.py \
  --data_type 10x \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/D1/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/D1/adata_ADT.h5ad \
  --save_path Results/adata/SpaMultiVAE/HLN/D1/SpaMultiVAE_HLN_D1.h5ad \
  --method SpaMultiVAE \
  --dataset Human_Lymph_Nodes/D1 \
  --cluster_nums 11

echo "Processing Human_Tonsils datasets..."

# Human_Tonsils S1
echo "Processing Human_Tonsils S1..."
python Scripts/integration/SpaMultiVAE/run_spamultivae.py \
  --data_type 10x \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Tonsils/S1/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Tonsils/S1/adata_ADT.h5ad \
  --save_path Results/adata/SpaMultiVAE/HT/S1/SpaMultiVAE_HT_S1.h5ad \
  --method SpaMultiVAE \
  --dataset Human_Tonsils/S1 \
  --cluster_nums 4

# Human_Tonsils S2
echo "Processing Human_Tonsils S2..."
python Scripts/integration/SpaMultiVAE/run_spamultivae.py \
  --data_type 10x \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Tonsils/S2/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Tonsils/S2/adata_ADT.h5ad \
  --save_path Results/adata/SpaMultiVAE/HT/S2/SpaMultiVAE_HT_S2.h5ad \
  --method SpaMultiVAE \
  --dataset Human_Tonsils/S2 \
  --cluster_nums 5

# Human_Tonsils S3
echo "Processing Human_Tonsils S3..."
python Scripts/integration/SpaMultiVAE/run_spamultivae.py \
  --data_type 10x \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Tonsils/S3/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Tonsils/S3/adata_ADT.h5ad \
  --save_path Results/adata/SpaMultiVAE/HT/S3/SpaMultiVAE_HT_S3.h5ad \
  --method SpaMultiVAE \
  --dataset Human_Tonsils/S3 \
  --cluster_nums 5

# === woGT RNA+ADT Datasets ===

echo "Processing Mouse_Thymus datasets..."

# Mouse_Thymus datasets
for thymus_id in 1 2 3 4; do
    echo "Processing Mouse_Thymus${thymus_id}..."
    python Scripts/integration/SpaMultiVAE/run_spamultivae.py \
      --data_type Stereo-CITE-seq \
      --RNA_path Dataset/woGT/RNA_ADT/Mouse_Thymus/Mouse_Thymus${thymus_id}/adata_RNA.h5ad \
      --ADT_path Dataset/woGT/RNA_ADT/Mouse_Thymus/Mouse_Thymus${thymus_id}/adata_ADT.h5ad \
      --save_path Results/adata/SpaMultiVAE/Mouse_Thymus/Thymus${thymus_id}/SpaMultiVAE_MT_Thymus${thymus_id}.h5ad \
      --method SpaMultiVAE \
      --dataset Mouse_Thymus/Mouse_Thymus${thymus_id} \
      --cluster_nums 8
done

echo "Processing Mouse_Spleen datasets..."

# Mouse_Spleen datasets
for spleen_id in 1 2; do
    echo "Processing Mouse_Spleen${spleen_id}..."
    python Scripts/integration/SpaMultiVAE/run_spamultivae.py \
      --data_type SPOTS \
      --RNA_path Dataset/woGT/RNA_ADT/Mouse_Spleen/Mouse_Spleen${spleen_id}/adata_RNA.h5ad \
      --ADT_path Dataset/woGT/RNA_ADT/Mouse_Spleen/Mouse_Spleen${spleen_id}/adata_ADT.h5ad \
      --save_path Results/adata/SpaMultiVAE/Mouse_Spleen/Spleen${spleen_id}/SpaMultiVAE_MS_Spleen${spleen_id}.h5ad \
      --method SpaMultiVAE \
      --dataset Mouse_Spleen/Mouse_Spleen${spleen_id} \
      --cluster_nums 5
done

echo "SpaMultiVAE processing completed!"
echo "End time: $(date)"

# Generate summary report
echo "=== PROCESSING SUMMARY ==="
echo "Results saved to Results/adata/SpaMultiVAE/"
echo "Plots saved to Results/plot/SpaMultiVAE/"
echo ""
echo "Processed datasets:"
find Results/adata/SpaMultiVAE -name "*.h5ad" | sort | while read file; do
    echo "  - $file"
done

echo ""
echo "Total results: $(find Results/adata/SpaMultiVAE -name "*.h5ad" | wc -l) datasets processed"