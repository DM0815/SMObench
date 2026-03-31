#!/bin/bash

# Comprehensive MultiGATE run script for all datasets
# Use relative paths for portability - script should be run from SMOBench root directory
# Cluster numbers: HLN_A1:10 HLN_D1:11 HT_S1:4 HT_S2:5 HT_S3:5 mouse_thymus:8 mouse_spleen:5 mouse_brain:18 MISAR_S1:8,12,12,14 MISAR_S2:13,14,15,16

if [ -n "$CONDA_PREFIX" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
fi

echo "Starting comprehensive MultiGATE processing..."
echo "Start time: $(date)"

# Create base results directory
mkdir -p Results/adata/MultiGATE Results/plot/MultiGATE Logs/MultiGATE

MULTIGATE_SCRIPT="Scripts/vertical_integration/MultiGATE/run_MultiGATE.py"
GENE_PROTEIN_CSV="Methods/MultiGATE/protein_gene_graph.csv"
PYTHON_CMD="env LD_LIBRARY_PATH=${LD_LIBRARY_PATH} python"

# === withGT RNA+ADT Datasets ===

echo "Processing Human_Lymph_Nodes datasets..."

# Human_Lymph_Nodes A1
echo "Processing Human_Lymph_Nodes A1..."
${PYTHON_CMD} ${MULTIGATE_SCRIPT} \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/A1/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/A1/adata_ADT.h5ad \
  --save_path Results/adata/MultiGATE/HLN/A1/MultiGATE_HLN_A1.h5ad \
  --dataset Human_Lymph_Nodes/A1 \
  --cluster_nums 10

# Human_Lymph_Nodes D1
echo "Processing Human_Lymph_Nodes D1..."
${PYTHON_CMD} ${MULTIGATE_SCRIPT} \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/D1/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/D1/adata_ADT.h5ad \
  --save_path Results/adata/MultiGATE/HLN/D1/MultiGATE_HLN_D1.h5ad \
  --dataset Human_Lymph_Nodes/D1 \
  --cluster_nums 11

echo "Processing Human_Tonsils datasets..."

# Human_Tonsils S1
echo "Processing Human_Tonsils S1..."
${PYTHON_CMD} ${MULTIGATE_SCRIPT} \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Tonsils/S1/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Tonsils/S1/adata_ADT.h5ad \
  --save_path Results/adata/MultiGATE/HT/S1/MultiGATE_HT_S1.h5ad \
  --dataset Human_Tonsils/S1 \
  --cluster_nums 4

# Human_Tonsils S2
echo "Processing Human_Tonsils S2..."
${PYTHON_CMD} ${MULTIGATE_SCRIPT} \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Tonsils/S2/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Tonsils/S2/adata_ADT.h5ad \
  --save_path Results/adata/MultiGATE/HT/S2/MultiGATE_HT_S2.h5ad \
  --dataset Human_Tonsils/S2 \
  --cluster_nums 5

# Human_Tonsils S3
echo "Processing Human_Tonsils S3..."
${PYTHON_CMD} ${MULTIGATE_SCRIPT} \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Tonsils/S3/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Tonsils/S3/adata_ADT.h5ad \
  --save_path Results/adata/MultiGATE/HT/S3/MultiGATE_HT_S3.h5ad \
  --dataset Human_Tonsils/S3 \
  --cluster_nums 5

# === woGT RNA+ADT Datasets ===

echo "Processing Mouse_Thymus datasets..."

for thymus_id in 1 2 3 4; do
  echo "Processing Mouse_Thymus${thymus_id}..."
  ${PYTHON_CMD} ${MULTIGATE_SCRIPT} \
    --RNA_path Dataset/woGT/RNA_ADT/Mouse_Thymus/Mouse_Thymus${thymus_id}/adata_RNA.h5ad \
    --ADT_path Dataset/woGT/RNA_ADT/Mouse_Thymus/Mouse_Thymus${thymus_id}/adata_ADT.h5ad \
    --save_path Results/adata/MultiGATE/Mouse_Thymus/Thymus${thymus_id}/MultiGATE_MT_Thymus${thymus_id}.h5ad \
    --dataset Mouse_Thymus/Mouse_Thymus${thymus_id} \
    --cluster_nums 8
done

echo "Processing Mouse_Spleen datasets..."

for spleen_id in 1 2; do
  echo "Processing Mouse_Spleen${spleen_id}..."
  ${PYTHON_CMD} ${MULTIGATE_SCRIPT} \
    --RNA_path Dataset/woGT/RNA_ADT/Mouse_Spleen/Mouse_Spleen${spleen_id}/adata_RNA.h5ad \
    --ADT_path Dataset/woGT/RNA_ADT/Mouse_Spleen/Mouse_Spleen${spleen_id}/adata_ADT.h5ad \
    --save_path Results/adata/MultiGATE/Mouse_Spleen/Spleen${spleen_id}/MultiGATE_MS_Spleen${spleen_id}.h5ad \
    --dataset Mouse_Spleen/Mouse_Spleen${spleen_id} \
    --cluster_nums 5 \
    --gene_protein_csv ${GENE_PROTEIN_CSV}
done

# === withGT RNA+ATAC Datasets ===

echo "Processing Mouse_Embryos RNA+ATAC datasets..."

if [ -d "Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1" ]; then
  declare -A S1_clusters=( [E11]=8 [E13]=12 [E15]=12 [E18]=14 )
  for stage in E11 E13 E15 E18; do
    if [ -f "Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1/${stage}/adata_RNA.h5ad" ]; then
      echo "Processing Mouse_Embryos_S1 ${stage}..."
      ${PYTHON_CMD} ${MULTIGATE_SCRIPT} \
        --RNA_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1/${stage}/adata_RNA.h5ad \
        --ATAC_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1/${stage}/adata_ATAC.h5ad \
        --save_path Results/adata/MultiGATE/MISAR_S1/${stage}/MultiGATE_MISAR_S1_${stage}.h5ad \
        --dataset Mouse_Embryos_S1/${stage} \
        --cluster_nums ${S1_clusters[$stage]} \
        --gene_peak_max_distance 200000
    fi
  done
fi

if [ -d "Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2" ]; then
  declare -A S2_clusters=( [E11]=13 [E13]=14 [E15]=15 [E18]=16 )
  for stage in E11 E13 E15 E18; do
    if [ -f "Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2/${stage}/adata_RNA.h5ad" ]; then
      echo "Processing Mouse_Embryos_S2 ${stage}..."
      ${PYTHON_CMD} ${MULTIGATE_SCRIPT} \
        --RNA_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2/${stage}/adata_RNA.h5ad \
        --ATAC_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2/${stage}/adata_ATAC.h5ad \
        --save_path Results/adata/MultiGATE/MISAR_S2/${stage}/MultiGATE_MISAR_S2_${stage}.h5ad \
        --dataset Mouse_Embryos_S2/${stage} \
        --cluster_nums ${S2_clusters[$stage]} \
        --gene_peak_max_distance 200000
    fi
  done
fi

# === woGT RNA+ATAC Datasets ===

echo "Processing Mouse_Brain RNA+ATAC datasets..."

brain_types=(ATAC H3K4me3 H3K27ac H3K27me3)
for brain_type in "${brain_types[@]}"; do
  rna_path="Dataset/woGT/RNA_ATAC/Mouse_Brain/Mouse_Brain_${brain_type}/adata_RNA.h5ad"
  atac_path="Dataset/woGT/RNA_ATAC/Mouse_Brain/Mouse_Brain_${brain_type}/adata_peaks_normalized.h5ad"
  if [ -f "${rna_path}" ] && [ -f "${atac_path}" ]; then
    echo "Processing Mouse_Brain ${brain_type}..."
    ${PYTHON_CMD} ${MULTIGATE_SCRIPT} \
      --RNA_path ${rna_path} \
      --ATAC_path ${atac_path} \
      --save_path Results/adata/MultiGATE/Mouse_Brain/${brain_type}/MultiGATE_MB_${brain_type}.h5ad \
      --dataset Mouse_Brain/Mouse_Brain_${brain_type} \
      --cluster_nums 18 \
      --gene_peak_max_distance 200000
  fi
done

echo "MultiGATE processing completed!"
echo "End time: $(date)"

# Summary
echo "=== PROCESSING SUMMARY ==="
echo "Results saved to Results/adata/MultiGATE/"
echo "Plots saved to Results/plot/MultiGATE/"
echo ""
echo "Processed datasets:"
find Results/adata/MultiGATE -name "*.h5ad" | sort | while read file; do
  echo "  - $file"
done

echo ""
echo "Total results: $(find Results/adata/MultiGATE -name "*.h5ad" | wc -l) datasets processed"
