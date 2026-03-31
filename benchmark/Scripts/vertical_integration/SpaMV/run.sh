#!/bin/bash

# Comprehensive SpaMV run script for all datasets
# Use relative paths for portability - script should be run from SMOBench root directory
# Cluster numbers: HLN_A1:10 HLN_D1:11 HT_S1:4 HT_S2:5 HT_S3:5 mouse_thymus:8 mouse_spleen:5 mouse_brain:18 3M:5 MISAR_S1:8,12,12,14 MISAR_S2:13,14,15,16

echo "Starting comprehensive SpaMV processing..."
echo "Start time: $(date)"

# Create base results directory
mkdir -p Results/adata/SpaMV Results/plot/SpaMV

# === withGT RNA+ADT Datasets ===

echo "Processing Human_Lymph_Nodes datasets..."

# Human_Lymph_Nodes A1
echo "Processing Human_Lymph_Nodes A1..."
python Scripts/integration/SpaMV/run_spamv.py \
  --data_type 10x \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/A1/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/A1/adata_ADT.h5ad \
  --save_path Results/adata/SpaMV/HLN/A1/SpaMV_HLN_A1.h5ad \
  --method SpaMV \
  --dataset Human_Lymph_Nodes/A1 \
  --cluster_nums 10

# Human_Lymph_Nodes D1
echo "Processing Human_Lymph_Nodes D1..."
python Scripts/integration/SpaMV/run_spamv.py \
  --data_type 10x \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/D1/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/D1/adata_ADT.h5ad \
  --save_path Results/adata/SpaMV/HLN/D1/SpaMV_HLN_D1.h5ad \
  --method SpaMV \
  --dataset Human_Lymph_Nodes/D1 \
  --cluster_nums 11

echo "Processing Human_Tonsils datasets..."

# Human_Tonsils S1
echo "Processing Human_Tonsils S1..."
python Scripts/integration/SpaMV/run_spamv.py \
  --data_type 10x \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Tonsils/S1/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Tonsils/S1/adata_ADT.h5ad \
  --save_path Results/adata/SpaMV/HT/S1/SpaMV_HT_S1.h5ad \
  --method SpaMV \
  --dataset Human_Tonsils/S1 \
  --cluster_nums 4

# Human_Tonsils S2
echo "Processing Human_Tonsils S2..."
python Scripts/integration/SpaMV/run_spamv.py \
  --data_type 10x \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Tonsils/S2/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Tonsils/S2/adata_ADT.h5ad \
  --save_path Results/adata/SpaMV/HT/S2/SpaMV_HT_S2.h5ad \
  --method SpaMV \
  --dataset Human_Tonsils/S2 \
  --cluster_nums 5

# Human_Tonsils S3
echo "Processing Human_Tonsils S3..."
python Scripts/integration/SpaMV/run_spamv.py \
  --data_type 10x \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Tonsils/S3/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Tonsils/S3/adata_ADT.h5ad \
  --save_path Results/adata/SpaMV/HT/S3/SpaMV_HT_S3.h5ad \
  --method SpaMV \
  --dataset Human_Tonsils/S3 \
  --cluster_nums 5

# === woGT RNA+ADT Datasets ===

echo "Processing Mouse_Thymus datasets..."

# Mouse_Thymus datasets
for thymus_id in 1 2 3 4; do
    echo "Processing Mouse_Thymus${thymus_id}..."
    python Scripts/integration/SpaMV/run_spamv.py \
      --data_type Stereo-CITE-seq \
      --RNA_path Dataset/woGT/RNA_ADT/Mouse_Thymus/Mouse_Thymus${thymus_id}/adata_RNA.h5ad \
      --ADT_path Dataset/woGT/RNA_ADT/Mouse_Thymus/Mouse_Thymus${thymus_id}/adata_ADT.h5ad \
      --save_path Results/adata/SpaMV/Mouse_Thymus/Thymus${thymus_id}/SpaMV_MT_Thymus${thymus_id}.h5ad \
      --method SpaMV \
      --dataset Mouse_Thymus/Mouse_Thymus${thymus_id} \
      --cluster_nums 8
done

echo "Processing Mouse_Spleen datasets..."

# Mouse_Spleen datasets
for spleen_id in 1 2; do
    echo "Processing Mouse_Spleen${spleen_id}..."
    python Scripts/integration/SpaMV/run_spamv.py \
      --data_type SPOTS \
      --RNA_path Dataset/woGT/RNA_ADT/Mouse_Spleen/Mouse_Spleen${spleen_id}/adata_RNA.h5ad \
      --ADT_path Dataset/woGT/RNA_ADT/Mouse_Spleen/Mouse_Spleen${spleen_id}/adata_ADT.h5ad \
      --save_path Results/adata/SpaMV/Mouse_Spleen/Spleen${spleen_id}/SpaMV_MS_Spleen${spleen_id}.h5ad \
      --method SpaMV \
      --dataset Mouse_Spleen/Mouse_Spleen${spleen_id} \
      --cluster_nums 5
done

# === RNA+ATAC Datasets ===

echo "Processing Mouse_Embryos RNA+ATAC datasets..."

# Check if Mouse_Embryos datasets exist
if [ -d "Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1" ]; then
    # E11: 8 clusters
    if [ -f "Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1/E11/adata_RNA.h5ad" ]; then
        echo "Processing Mouse_Embryos_S1 E11..."
        python Scripts/integration/SpaMV/run_spamv.py \
          --data_type MISAR \
          --RNA_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1/E11/adata_RNA.h5ad \
          --ATAC_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1/E11/adata_ATAC.h5ad \
          --save_path Results/adata/SpaMV/MISAR_S1/E11/SpaMV_MISAR_S1_E11.h5ad \
          --method SpaMV \
          --dataset Mouse_Embryos_S1/E11 \
          --cluster_nums 8
    fi
    
    # E13: 12 clusters
    if [ -f "Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1/E13/adata_RNA.h5ad" ]; then
        echo "Processing Mouse_Embryos_S1 E13..."
        python Scripts/integration/SpaMV/run_spamv.py \
          --data_type MISAR \
          --RNA_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1/E13/adata_RNA.h5ad \
          --ATAC_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1/E13/adata_ATAC.h5ad \
          --save_path Results/adata/SpaMV/MISAR_S1/E13/SpaMV_MISAR_S1_E13.h5ad \
          --method SpaMV \
          --dataset Mouse_Embryos_S1/E13 \
          --cluster_nums 12
    fi
    
    # E15: 12 clusters
    if [ -f "Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1/E15/adata_RNA.h5ad" ]; then
        echo "Processing Mouse_Embryos_S1 E15..."
        python Scripts/integration/SpaMV/run_spamv.py \
          --data_type MISAR \
          --RNA_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1/E15/adata_RNA.h5ad \
          --ATAC_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1/E15/adata_ATAC.h5ad \
          --save_path Results/adata/SpaMV/MISAR_S1/E15/SpaMV_MISAR_S1_E15.h5ad \
          --method SpaMV \
          --dataset Mouse_Embryos_S1/E15 \
          --cluster_nums 12
    fi
    
    # E18: 14 clusters
    if [ -f "Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1/E18/adata_RNA.h5ad" ]; then
        echo "Processing Mouse_Embryos_S1 E18..."
        python Scripts/integration/SpaMV/run_spamv.py \
          --data_type MISAR \
          --RNA_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1/E18/adata_RNA.h5ad \
          --ATAC_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1/E18/adata_ATAC.h5ad \
          --save_path Results/adata/SpaMV/MISAR_S1/E18/SpaMV_MISAR_S1_E18.h5ad \
          --method SpaMV \
          --dataset Mouse_Embryos_S1/E18 \
          --cluster_nums 14
    fi
fi

if [ -d "Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2" ]; then
    # E11: 13 clusters
    if [ -f "Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2/E11/adata_RNA.h5ad" ]; then
        echo "Processing Mouse_Embryos_S2 E11..."
        python Scripts/integration/SpaMV/run_spamv.py \
          --data_type MISAR \
          --RNA_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2/E11/adata_RNA.h5ad \
          --ATAC_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2/E11/adata_ATAC.h5ad \
          --save_path Results/adata/SpaMV/MISAR_S2/E11/SpaMV_MISAR_S2_E11.h5ad \
          --method SpaMV \
          --dataset Mouse_Embryos_S2/E11 \
          --cluster_nums 13
    fi
    
    # E13: 14 clusters
    if [ -f "Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2/E13/adata_RNA.h5ad" ]; then
        echo "Processing Mouse_Embryos_S2 E13..."
        python Scripts/integration/SpaMV/run_spamv.py \
          --data_type MISAR \
          --RNA_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2/E13/adata_RNA.h5ad \
          --ATAC_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2/E13/adata_ATAC.h5ad \
          --save_path Results/adata/SpaMV/MISAR_S2/E13/SpaMV_MISAR_S2_E13.h5ad \
          --method SpaMV \
          --dataset Mouse_Embryos_S2/E13 \
          --cluster_nums 14
    fi
    
    # E15: 15 clusters
    if [ -f "Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2/E15/adata_RNA.h5ad" ]; then
        echo "Processing Mouse_Embryos_S2 E15..."
        python Scripts/integration/SpaMV/run_spamv.py \
          --data_type MISAR \
          --RNA_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2/E15/adata_RNA.h5ad \
          --ATAC_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2/E15/adata_ATAC.h5ad \
          --save_path Results/adata/SpaMV/MISAR_S2/E15/SpaMV_MISAR_S2_E15.h5ad \
          --method SpaMV \
          --dataset Mouse_Embryos_S2/E15 \
          --cluster_nums 15
    fi
    
    # E18: 16 clusters
    if [ -f "Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2/E18/adata_RNA.h5ad" ]; then
        echo "Processing Mouse_Embryos_S2 E18..."
        python Scripts/integration/SpaMV/run_spamv.py \
          --data_type MISAR \
          --RNA_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2/E18/adata_RNA.h5ad \
          --ATAC_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2/E18/adata_ATAC.h5ad \
          --save_path Results/adata/SpaMV/MISAR_S2/E18/SpaMV_MISAR_S2_E18.h5ad \
          --method SpaMV \
          --dataset Mouse_Embryos_S2/E18 \
          --cluster_nums 16
    fi
fi

echo "Processing Mouse_Brain RNA+ATAC datasets..."

# Mouse_Brain datasets (woGT RNA+ATAC)
brain_types=("ATAC" "H3K4me3" "H3K27ac" "H3K27me3")
for brain_type in "${brain_types[@]}"; do
    if [ -f "Dataset/woGT/RNA_ATAC/Mouse_Brain/Mouse_Brain_${brain_type}/adata_RNA.h5ad" ]; then
        echo "Processing Mouse_Brain ${brain_type}..."
        # Check if the ATAC file exists with different naming
        atac_file="Dataset/woGT/RNA_ATAC/Mouse_Brain/Mouse_Brain_${brain_type}/adata_ATAC.h5ad"
        if [ ! -f "$atac_file" ]; then
            atac_file="Dataset/woGT/RNA_ATAC/Mouse_Brain/Mouse_Brain_${brain_type}/adata_peaks_normalized.h5ad"
        fi
        
        if [ -f "$atac_file" ]; then
            python Scripts/integration/SpaMV/run_spamv.py \
              --data_type Spatial-epigenome-transcriptome \
              --RNA_path Dataset/woGT/RNA_ATAC/Mouse_Brain/Mouse_Brain_${brain_type}/adata_RNA.h5ad \
              --ATAC_path "$atac_file" \
              --save_path Results/adata/SpaMV/Mouse_Brain/${brain_type}/SpaMV_MB_${brain_type}.h5ad \
              --method SpaMV \
              --dataset Mouse_Brain/Mouse_Brain_${brain_type} \
              --cluster_nums 18
        fi
    fi
done

echo "SpaMV processing completed!"
echo "End time: $(date)"

# Generate summary report
echo "=== PROCESSING SUMMARY ==="
echo "Results saved to Results/adata/SpaMV/"
echo "Plots saved to Results/plot/SpaMV/"
echo ""
echo "Processed datasets:"
find Results/adata/SpaMV -name "*.h5ad" | sort | while read file; do
    echo "  - $file"
done

echo ""
echo "Total results: $(find Results/adata/SpaMV -name "*.h5ad" | wc -l) datasets processed"