#!/bin/bash

# Comprehensive SpaMI vertical integration benchmark script
# Run from SMOBench root directory:
#   bash Scripts/vertical_integration/SpaMI/run.sh

echo "=== Starting SpaMI vertical integration benchmark ==="
echo "Start time: $(date)"

# Create base results directories (consistent with project structure)
mkdir -p Results/adata/vertical_integration/SpaMI
mkdir -p Results/plot/vertical_integration/SpaMI

# =========================================================
# === withGT RNA + ADT datasets ===========================
# =========================================================

echo "Processing Human_Lymph_Nodes datasets..."

echo "Processing Human_Lymph_Nodes A1..."
python Scripts/vertical_integration/SpaMI/run_SpaMI.py \
  --data_type 10x \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/A1/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/A1/adata_ADT.h5ad \
  --save_path Results/adata/vertical_integration/SpaMI/HLN/A1/SpaMI_HLN_A1.h5ad \
  --method SpaMI \
  --dataset Human_Lymph_Nodes/A1 \
  --cluster_nums 10

echo "Processing Human_Lymph_Nodes D1..."
python Scripts/vertical_integration/SpaMI/run_SpaMI.py \
  --data_type 10x \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/D1/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/D1/adata_ADT.h5ad \
  --save_path Results/adata/vertical_integration/SpaMI/HLN/D1/SpaMI_HLN_D1.h5ad \
  --method SpaMI \
  --dataset Human_Lymph_Nodes/D1 \
  --cluster_nums 11

echo "Processing Human_Tonsils datasets..."

echo "Processing Human_Tonsils S1..."
python Scripts/vertical_integration/SpaMI/run_SpaMI.py \
  --data_type 10x \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Tonsils/S1/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Tonsils/S1/adata_ADT.h5ad \
  --save_path Results/adata/vertical_integration/SpaMI/HT/S1/SpaMI_HT_S1.h5ad \
  --method SpaMI \
  --dataset Human_Tonsils/S1 \
  --cluster_nums 4

echo "Processing Human_Tonsils S2..."
python Scripts/vertical_integration/SpaMI/run_SpaMI.py \
  --data_type 10x \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Tonsils/S2/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Tonsils/S2/adata_ADT.h5ad \
  --save_path Results/adata/vertical_integration/SpaMI/HT/S2/SpaMI_HT_S2.h5ad \
  --method SpaMI \
  --dataset Human_Tonsils/S2 \
  --cluster_nums 5

echo "Processing Human_Tonsils S3..."
python Scripts/vertical_integration/SpaMI/run_SpaMI.py \
  --data_type 10x \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Tonsils/S3/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Tonsils/S3/adata_ADT.h5ad \
  --save_path Results/adata/vertical_integration/SpaMI/HT/S3/SpaMI_HT_S3.h5ad \
  --method SpaMI \
  --dataset Human_Tonsils/S3 \
  --cluster_nums 5

# =========================================================
# === woGT RNA + ADT datasets =============================
# =========================================================

echo "Processing Mouse_Thymus datasets..."
for thymus_id in 1 2 3 4; do
  echo "Processing Mouse_Thymus${thymus_id}..."
  python Scripts/vertical_integration/SpaMI/run_SpaMI.py \
    --data_type Stereo-CITE-seq \
    --RNA_path Dataset/woGT/RNA_ADT/Mouse_Thymus/Mouse_Thymus${thymus_id}/adata_RNA.h5ad \
    --ADT_path Dataset/woGT/RNA_ADT/Mouse_Thymus/Mouse_Thymus${thymus_id}/adata_ADT.h5ad \
    --save_path Results/adata/vertical_integration/SpaMI/Mouse_Thymus/Thymus${thymus_id}/SpaMI_MT_Thymus${thymus_id}.h5ad \
    --method SpaMI \
    --dataset Mouse_Thymus/Mouse_Thymus${thymus_id} \
    --cluster_nums 8
done

echo "Processing Mouse_Spleen datasets..."
for spleen_id in 1 2; do
  echo "Processing Mouse_Spleen${spleen_id}..."
  python Scripts/vertical_integration/SpaMI/run_SpaMI.py \
    --data_type SPOTS \
    --RNA_path Dataset/woGT/RNA_ADT/Mouse_Spleen/Mouse_Spleen${spleen_id}/adata_RNA.h5ad \
    --ADT_path Dataset/woGT/RNA_ADT/Mouse_Spleen/Mouse_Spleen${spleen_id}/adata_ADT.h5ad \
    --save_path Results/adata/vertical_integration/SpaMI/Mouse_Spleen/Spleen${spleen_id}/SpaMI_MS_Spleen${spleen_id}.h5ad \
    --method SpaMI \
    --dataset Mouse_Spleen/Mouse_Spleen${spleen_id} \
    --cluster_nums 5
done

# =========================================================
# === RNA + ATAC datasets =================================
# =========================================================

echo "Processing Mouse_Embryos RNA+ATAC datasets..."

if [ -d "Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1" ]; then
  for stage in E11 E13 E15 E18; do
    if [ -f "Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1/${stage}/adata_RNA.h5ad" ]; then
      echo "Processing Mouse_Embryos_S1 ${stage}..."
      case "$stage" in
        E11) cluster_num=8 ;;
        E13) cluster_num=12 ;;
        E15) cluster_num=12 ;;
        E18) cluster_num=14 ;;
        *) cluster_num=12 ;;
      esac
      python Scripts/vertical_integration/SpaMI/run_SpaMI.py \
        --data_type MISAR \
        --RNA_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1/${stage}/adata_RNA.h5ad \
        --ATAC_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1/${stage}/adata_ATAC.h5ad \
        --save_path Results/adata/vertical_integration/SpaMI/MISAR_S1/${stage}/SpaMI_MISAR_S1_${stage}.h5ad \
        --method SpaMI \
        --dataset Mouse_Embryos_S1/${stage} \
        --cluster_nums ${cluster_num}
    fi
  done
fi

if [ -d "Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2" ]; then
  for stage in E11 E13 E15 E18; do
    if [ -f "Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2/${stage}/adata_RNA.h5ad" ]; then
      echo "Processing Mouse_Embryos_S2 ${stage}..."
      case "$stage" in
        E11) cluster_num=13 ;;
        E13) cluster_num=14 ;;
        E15) cluster_num=15 ;;
        E18) cluster_num=16 ;;
        *) cluster_num=14 ;;
      esac
      python Scripts/vertical_integration/SpaMI/run_SpaMI.py \
        --data_type MISAR \
        --RNA_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2/${stage}/adata_RNA.h5ad \
        --ATAC_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2/${stage}/adata_ATAC.h5ad \
        --save_path Results/adata/vertical_integration/SpaMI/MISAR_S2/${stage}/SpaMI_MISAR_S2_${stage}.h5ad \
        --method SpaMI \
        --dataset Mouse_Embryos_S2/${stage} \
        --cluster_nums ${cluster_num}
    fi
  done
fi

echo "Processing Mouse_Brain RNA+ATAC datasets..."

brain_types=("ATAC" "H3K4me3" "H3K27ac" "H3K27me3")
for brain_type in "${brain_types[@]}"; do
  if [ -f "Dataset/woGT/RNA_ATAC/Mouse_Brain/Mouse_Brain_${brain_type}/adata_RNA.h5ad" ]; then
    echo "Processing Mouse_Brain ${brain_type}..."
    atac_file="Dataset/woGT/RNA_ATAC/Mouse_Brain/Mouse_Brain_${brain_type}/adata_ATAC.h5ad"
    if [ ! -f "$atac_file" ]; then
      atac_file="Dataset/woGT/RNA_ATAC/Mouse_Brain/Mouse_Brain_${brain_type}/adata_peaks_normalized.h5ad"
    fi
    if [ -f "$atac_file" ]; then
      python Scripts/vertical_integration/SpaMI/run_SpaMI.py \
        --data_type Spatial-epigenome-transcriptome \
        --RNA_path Dataset/woGT/RNA_ATAC/Mouse_Brain/Mouse_Brain_${brain_type}/adata_RNA.h5ad \
        --ATAC_path "$atac_file" \
        --save_path Results/adata/vertical_integration/SpaMI/Mouse_Brain/${brain_type}/SpaMI_MB_${brain_type}.h5ad \
        --method SpaMI \
        --dataset Mouse_Brain/Mouse_Brain_${brain_type} \
        --cluster_nums 18
    fi
  fi
done

# =========================================================
# === Summary =============================================
# =========================================================

echo "SpaMI processing completed!"
echo "End time: $(date)"
echo ""
echo "=== PROCESSING SUMMARY ==="
echo "Results saved to Results/adata/vertical_integration/SpaMI/"
echo "Plots saved to Results/plot/vertical_integration/SpaMI/"
echo ""
echo "Processed datasets:"
find Results/adata/vertical_integration/SpaMI -name "*.h5ad" 2>/dev/null | sort | while read -r file; do
  echo "  - $file"
done
echo ""
echo "Total results: $(find Results/adata/vertical_integration/SpaMI -name \"*.h5ad\" 2>/dev/null | wc -l) datasets processed"
