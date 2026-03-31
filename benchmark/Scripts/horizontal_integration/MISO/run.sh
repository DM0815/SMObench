#!/bin/bash

# MISO horizontal integration script
# Processes fusion datasets (RNA + ADT / ATAC) for batch effect removal

echo "Starting MISO horizontal integration..."
echo "Start time: $(date)"

BASE_DIR="Results/adata/horizontal_integration/MISO"
mkdir -p "${BASE_DIR}"
export NUMBA_DISABLE_JIT=1
PYTHON_CMD="conda run -n smobench python"

run_miso_h() {
  local rna_path=$1
  local second_flag=$2
  local second_path=$3
  local save_path=$4
  local clusters=$5

  echo "Processing ${save_path} ..."
  ${PYTHON_CMD} Scripts/horizontal_integration/MISO/run_MISO_h.py \
    --RNA_path "${rna_path}" \
    ${second_flag} "${second_path}" \
    --save_path "${save_path}" \
    --cluster_nums "${clusters}"
}

# echo "Processing MISAR_S1 fusion dataset..."
# run_miso_h \
#   Dataset/fusionWithGT/RNA_ATAC/ME_S1_Fusion_RNA.h5ad \
#   --ATAC_path Dataset/fusionWithGT/RNA_ATAC/ME_S1_Fusion_ATAC.h5ad \
#   "${BASE_DIR}/MISAR_S1/MISO_MISAR_S1_fusion.h5ad" \
#   12

# echo "Processing MISAR_S2 fusion dataset..."
# run_miso_h \
#   Dataset/fusionWithGT/RNA_ATAC/ME_S2_Fusion_RNA.h5ad \
#   --ATAC_path Dataset/fusionWithGT/RNA_ATAC/ME_S2_Fusion_ATAC.h5ad \
#   "${BASE_DIR}/MISAR_S2/MISO_MISAR_S2_fusion.h5ad" \
#   14

# echo "Processing Mouse_Brain fusion dataset..."
# run_miso_h \
#   Dataset/fusionWoGT/RNA_ATAC/Mouse_Brain_Fusion_RNA.h5ad \
#   --ATAC_path Dataset/fusionWoGT/RNA_ATAC/Mouse_Brain_Fusion_ATAC.h5ad \
#   "${BASE_DIR}/Mouse_Brain/MISO_Mouse_Brain_fusion.h5ad" \
#   18

# echo "Processing HLN fusion dataset..."
# run_miso_h \
#   Dataset/fusionWithGT/RNA_ADT/HLN_Fusion_RNA.h5ad \
#   --ADT_path Dataset/fusionWithGT/RNA_ADT/HLN_Fusion_ADT.h5ad \
#   "${BASE_DIR}/HLN/MISO_HLN_fusion.h5ad" \
#   10

# echo "Processing HT fusion dataset..."
# run_miso_h \
#   Dataset/fusionWithGT/RNA_ADT/HT_Fusion_RNA.h5ad \
#   --ADT_path Dataset/fusionWithGT/RNA_ADT/HT_Fusion_ADT.h5ad \
#   "${BASE_DIR}/HT/MISO_HT_fusion.h5ad" \
#   5

# echo "Processing Mouse Thymus fusion dataset..."
# run_miso_h \
#   Dataset/fusionWoGT/RNA_ADT/Mouse_Thymus_Fusion_RNA.h5ad \
#   --ADT_path Dataset/fusionWoGT/RNA_ADT/Mouse_Thymus_Fusion_ADT.h5ad \
#   "${BASE_DIR}/Mouse_Thymus/MISO_Mouse_Thymus_fusion.h5ad" \
#   8

echo "Processing Mouse Spleen fusion dataset..."
run_miso_h \
  Dataset/fusionWoGT/RNA_ADT/Mouse_Spleen_Fusion_RNA.h5ad \
  --ADT_path Dataset/fusionWoGT/RNA_ADT/Mouse_Spleen_Fusion_ADT.h5ad \
  "${BASE_DIR}/Mouse_Spleen/MISO_Mouse_Spleen_fusion.h5ad" \
  5

echo "MISO horizontal integration completed."
echo "End time: $(date)"
