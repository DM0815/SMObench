#!/bin/bash

# MISO vertical integration pipeline across all SMOBench datasets
# Cluster numbers: HLN_A1:10 HLN_D1:11 HT_S1:4 HT_S2:5 HT_S3:5
# Mouse_Thymus:8 Mouse_Spleen:5 Mouse_Brain:18
# MISAR_S1: E11=8, E13=12, E15=12, E18=14
# MISAR_S2: E11=13, E13=14, E15=15, E18=16

echo "Starting MISO vertical integration pipeline..."
echo "Start time: $(date)"

BASE_RESULTS="Results/adata/vertical_integration/MISO"
mkdir -p "${BASE_RESULTS}"

run_miso() {
  local data_type=$1
  local rna_path=$2
  local second_arg_flag=$3
  local second_path=$4
  local save_path=$5
  local cluster_num=$6

  echo "Running MISO for ${save_path} ..."
  python Scripts/vertical_integration/MISO/run_MISO.py \
    --data_type "${data_type}" \
    --RNA_path "${rna_path}" \
    ${second_arg_flag} "${second_path}" \
    --save_path "${save_path}" \
    --cluster_nums "${cluster_num}"
}

# # === withGT RNA + ADT ===
# echo "Processing withGT RNA+ADT datasets..."

# run_miso 10x \
#   Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/A1/adata_RNA.h5ad \
#   --ADT_path Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/A1/adata_ADT.h5ad \
#   "${BASE_RESULTS}/HLN/A1/MISO_HLN_A1.h5ad" \
#   10

# run_miso 10x \
#   Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/D1/adata_RNA.h5ad \
#   --ADT_path Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/D1/adata_ADT.h5ad \
#   "${BASE_RESULTS}/HLN/D1/MISO_HLN_D1.h5ad" \
#   11

# run_miso 10x \
#   Dataset/withGT/RNA_ADT/Human_Tonsils/S1/adata_RNA.h5ad \
#   --ADT_path Dataset/withGT/RNA_ADT/Human_Tonsils/S1/adata_ADT.h5ad \
#   "${BASE_RESULTS}/HT/S1/MISO_HT_S1.h5ad" \
#   4

# run_miso 10x \
#   Dataset/withGT/RNA_ADT/Human_Tonsils/S2/adata_RNA.h5ad \
#   --ADT_path Dataset/withGT/RNA_ADT/Human_Tonsils/S2/adata_ADT.h5ad \
#   "${BASE_RESULTS}/HT/S2/MISO_HT_S2.h5ad" \
#   5

# run_miso 10x \
#   Dataset/withGT/RNA_ADT/Human_Tonsils/S3/adata_RNA.h5ad \
#   --ADT_path Dataset/withGT/RNA_ADT/Human_Tonsils/S3/adata_ADT.h5ad \
#   "${BASE_RESULTS}/HT/S3/MISO_HT_S3.h5ad" \
#   5

# # === woGT RNA + ADT ===
# echo "Processing woGT RNA+ADT datasets..."

# for thymus_id in 1 2 3 4; do
#   run_miso Stereo-CITE-seq \
#     "Dataset/woGT/RNA_ADT/Mouse_Thymus/Mouse_Thymus${thymus_id}/adata_RNA.h5ad" \
#     --ADT_path "Dataset/woGT/RNA_ADT/Mouse_Thymus/Mouse_Thymus${thymus_id}/adata_ADT.h5ad" \
#     "${BASE_RESULTS}/Mouse_Thymus/Thymus${thymus_id}/MISO_Mouse_Thymus${thymus_id}.h5ad" \
#     8
# done

# for spleen_id in 1 2; do
#   run_miso SPOTS \
#     "Dataset/woGT/RNA_ADT/Mouse_Spleen/Mouse_Spleen${spleen_id}/adata_RNA.h5ad" \
#     --ADT_path "Dataset/woGT/RNA_ADT/Mouse_Spleen/Mouse_Spleen${spleen_id}/adata_ADT.h5ad" \
#     "${BASE_RESULTS}/Mouse_Spleen/Spleen${spleen_id}/MISO_Mouse_Spleen${spleen_id}.h5ad" \
#     5
# done

# # === withGT RNA + ATAC ===
# echo "Processing withGT RNA+ATAC (MISAR) datasets..."

# MISAR_S1_DIR="Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1"
# if [ -d "$MISAR_S1_DIR" ]; then
#   for slice in E11 E13 E15 E18; do
#     cluster=8
#     if [ "$slice" = "E13" ] || [ "$slice" = "E15" ]; then cluster=12; fi
#     if [ "$slice" = "E18" ]; then cluster=14; fi
#     run_miso MISAR \
#       "$MISAR_S1_DIR/${slice}/adata_RNA.h5ad" \
#       --ATAC_path "$MISAR_S1_DIR/${slice}/adata_ATAC.h5ad" \
#       "${BASE_RESULTS}/MISAR_S1/${slice}/MISO_MISAR_S1_${slice}.h5ad" \
#       $cluster
#   done
# fi

# MISAR_S2_DIR="Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2"
# if [ -d "$MISAR_S2_DIR" ]; then
#   for slice in E11 E13 E15 E18; do
#     cluster=13
#     if [ "$slice" = "E13" ]; then cluster=14; fi
#     if [ "$slice" = "E15" ]; then cluster=15; fi
#     if [ "$slice" = "E18" ]; then cluster=16; fi
#     run_miso MISAR \
#       "$MISAR_S2_DIR/${slice}/adata_RNA.h5ad" \
#       --ATAC_path "$MISAR_S2_DIR/${slice}/adata_ATAC.h5ad" \
#       "${BASE_RESULTS}/MISAR_S2/${slice}/MISO_MISAR_S2_${slice}.h5ad" \
#       $cluster
#   done
# fi

# === woGT RNA + ATAC ===
echo "Processing woGT RNA+ATAC Mouse Brain datasets..."

BRAIN_DIR="Dataset/woGT/RNA_ATAC/Mouse_Brain"
if [ -d "$BRAIN_DIR" ]; then
  for region in ATAC H3K27ac H3K27me3 H3K4me3; do
    RUN_DIR="${BRAIN_DIR}/Mouse_Brain_${region}"
    if [ -f "${RUN_DIR}/adata_RNA.h5ad" ] && [ -f "${RUN_DIR}/adata_peaks_normalized.h5ad" ]; then
      run_miso "RNA_ATAC" \
        "${RUN_DIR}/adata_RNA.h5ad" \
        --ATAC_path "${RUN_DIR}/adata_peaks_normalized.h5ad" \
        "${BASE_RESULTS}/Mouse_Brain/${region}/MISO_Mouse_Brain_${region}.h5ad" \
        18
    else
      echo "Skipping Mouse_Brain_${region}: files not found."
    fi
  done
else
  echo "Mouse_Brain directory not found, skipping woGT RNA+ATAC."
fi

echo "MISO vertical integration pipeline completed."
echo "End time: $(date)"
