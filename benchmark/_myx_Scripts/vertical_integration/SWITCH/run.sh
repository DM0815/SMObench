#!/bin/bash

# SWITCH vertical integration benchmark script
# Run from the SMOBench root directory:
#   bash Scripts/vertical_integration/SWITCH/run.sh

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  if [[ ":${LD_LIBRARY_PATH:-}:" != *":$CONDA_PREFIX/lib:"* ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp}"

GTF_PATH="${SWITCH_GTF_PATH:-}"
GTF_BY="${SWITCH_GTF_BY:-gene_name}"

echo "=== Starting SWITCH vertical integration benchmark ==="
echo "Start time: $(date)"

mkdir -p Results/adata/vertical_integration/SWITCH
mkdir -p Results/plot/vertical_integration/SWITCH

clean_switch_cache() {
  rm -rf pre_adj/* 2>/dev/null || true
}

run_switch_dataset() {
  clean_switch_cache
  python Scripts/vertical_integration/SWITCH/run_SWITCH.py "$@"
}

# === withGT RNA + ADT datasets ===
echo "Processing Human_Lymph_Nodes datasets..."
run_switch_dataset \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/A1/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/A1/adata_ADT.h5ad \
  --save_path Results/adata/vertical_integration/SWITCH/HLN/A1/SWITCH_HLN_A1.h5ad \
  --dataset Human_Lymph_Nodes/A1 \
  --cluster_nums 10 \
  --method SWITCH \
  --gtf_path "$GTF_PATH" \
  --gtf_by "$GTF_BY" \
  --data_type RNA_ADT

run_switch_dataset \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/D1/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/D1/adata_ADT.h5ad \
  --save_path Results/adata/vertical_integration/SWITCH/HLN/D1/SWITCH_HLN_D1.h5ad \
  --dataset Human_Lymph_Nodes/D1 \
  --cluster_nums 11 \
  --method SWITCH \
  --gtf_path "$GTF_PATH" \
  --gtf_by "$GTF_BY" \
  --data_type RNA_ADT

echo "Processing Human_Tonsils datasets..."
run_switch_dataset \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Tonsils/S1/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Tonsils/S1/adata_ADT.h5ad \
  --save_path Results/adata/vertical_integration/SWITCH/HT/S1/SWITCH_HT_S1.h5ad \
  --dataset Human_Tonsils/S1 \
  --cluster_nums 4 \
  --method SWITCH \
  --gtf_path "$GTF_PATH" \
  --gtf_by "$GTF_BY" \
  --data_type RNA_ADT

run_switch_dataset \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Tonsils/S2/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Tonsils/S2/adata_ADT.h5ad \
  --save_path Results/adata/vertical_integration/SWITCH/HT/S2/SWITCH_HT_S2.h5ad \
  --dataset Human_Tonsils/S2 \
  --cluster_nums 5 \
  --method SWITCH \
  --gtf_path "$GTF_PATH" \
  --gtf_by "$GTF_BY" \
  --data_type RNA_ADT

run_switch_dataset \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Tonsils/S3/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Tonsils/S3/adata_ADT.h5ad \
  --save_path Results/adata/vertical_integration/SWITCH/HT/S3/SWITCH_HT_S3.h5ad \
  --dataset Human_Tonsils/S3 \
  --cluster_nums 5 \
  --method SWITCH \
  --gtf_path "$GTF_PATH" \
  --gtf_by "$GTF_BY" \
  --data_type RNA_ADT

# === withGT RNA + ATAC datasets ===
if [[ -n "$GTF_PATH" && -f "$GTF_PATH" ]]; then
  echo "Processing Mouse_Embryos_S1 RNA+ATAC datasets..."
  for stage in E11 E13 E15 E18; do
    run_switch_dataset \
      --RNA_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1/${stage}/adata_RNA.h5ad \
      --ATAC_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1/${stage}/adata_ATAC.h5ad \
      --save_path Results/adata/vertical_integration/SWITCH/MISAR_S1/${stage}/SWITCH_MISAR_S1_${stage}.h5ad \
      --dataset Mouse_Embryos_S1/${stage} \
      --cluster_nums $(case $stage in E11) echo 8;; E13) echo 12;; E15) echo 12;; E18) echo 14;; esac) \
      --method SWITCH \
      --gtf_path "$GTF_PATH" \
      --gtf_by "$GTF_BY" \
      --data_type RNA_ATAC
  done

  echo "Processing Mouse_Embryos_S2 RNA+ATAC datasets..."
  for stage in E11 E13 E15 E18; do
    run_switch_dataset \
      --RNA_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2/${stage}/adata_RNA.h5ad \
      --ATAC_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2/${stage}/adata_ATAC.h5ad \
      --save_path Results/adata/vertical_integration/SWITCH/MISAR_S2/${stage}/SWITCH_MISAR_S2_${stage}.h5ad \
      --dataset Mouse_Embryos_S2/${stage} \
      --cluster_nums $(case $stage in E11) echo 13;; E13) echo 14;; E15) echo 15;; E18) echo 16;; esac) \
      --method SWITCH \
      --gtf_path "$GTF_PATH" \
      --gtf_by "$GTF_BY" \
      --data_type RNA_ATAC
  done
else
  echo "Skipping Mouse_Embryos RNA+ATAC datasets (set SWITCH_GTF_PATH to enable)."
fi

# === woGT RNA + ADT datasets ===
echo "Processing Mouse_Thymus datasets..."
for thymus_id in 1 2 3 4; do
  run_switch_dataset \
    --RNA_path Dataset/woGT/RNA_ADT/Mouse_Thymus/Mouse_Thymus${thymus_id}/adata_RNA.h5ad \
    --ADT_path Dataset/woGT/RNA_ADT/Mouse_Thymus/Mouse_Thymus${thymus_id}/adata_ADT.h5ad \
    --save_path Results/adata/vertical_integration/SWITCH/Mouse_Thymus/Thymus${thymus_id}/SWITCH_MT_Thymus${thymus_id}.h5ad \
    --dataset Mouse_Thymus/Mouse_Thymus${thymus_id} \
    --cluster_nums 8 \
    --method SWITCH \
    --gtf_path "$GTF_PATH" \
    --gtf_by "$GTF_BY" \
    --data_type RNA_ADT
done

echo "Processing Mouse_Spleen datasets..."
for spleen_id in 1 2; do
  run_switch_dataset \
    --RNA_path Dataset/woGT/RNA_ADT/Mouse_Spleen/Mouse_Spleen${spleen_id}/adata_RNA.h5ad \
    --ADT_path Dataset/woGT/RNA_ADT/Mouse_Spleen/Mouse_Spleen${spleen_id}/adata_ADT.h5ad \
    --save_path Results/adata/vertical_integration/SWITCH/Mouse_Spleen/Spleen${spleen_id}/SWITCH_MS_Spleen${spleen_id}.h5ad \
    --dataset Mouse_Spleen/Mouse_Spleen${spleen_id} \
    --cluster_nums 5 \
    --method SWITCH \
    --gtf_path "$GTF_PATH" \
    --gtf_by "$GTF_BY" \
    --data_type RNA_ADT
done

# === woGT RNA + ATAC datasets ===
if [[ -n "$GTF_PATH" && -f "$GTF_PATH" ]]; then
  echo "Processing Mouse_Brain RNA+ATAC datasets..."
  brain_types=("ATAC" "H3K4me3" "H3K27ac" "H3K27me3")
  for brain_type in "${brain_types[@]}"; do
    run_switch_dataset \
      --RNA_path Dataset/woGT/RNA_ATAC/Mouse_Brain/Mouse_Brain_${brain_type}/adata_RNA.h5ad \
      --ATAC_path Dataset/woGT/RNA_ATAC/Mouse_Brain/Mouse_Brain_${brain_type}/adata_ATAC.h5ad \
      --save_path Results/adata/vertical_integration/SWITCH/Mouse_Brain/${brain_type}/SWITCH_MB_${brain_type}.h5ad \
      --dataset Mouse_Brain/Mouse_Brain_${brain_type} \
      --cluster_nums 18 \
      --method SWITCH \
      --gtf_path "$GTF_PATH" \
      --gtf_by "$GTF_BY" \
      --data_type RNA_ATAC
  done
else
  echo "Skipping Mouse_Brain RNA+ATAC datasets (set SWITCH_GTF_PATH to enable)."
fi

# === Summary ===
echo "SWITCH vertical integration completed!"
echo "End time: $(date)"
echo ""
echo "=== RESULTS SUMMARY ==="
echo "Results saved to Results/adata/vertical_integration/SWITCH/"
echo "Plots saved to Results/plot/vertical_integration/SWITCH/"
echo ""
find Results/adata/vertical_integration/SWITCH -name "*.h5ad" 2>/dev/null | sort | while read -r file; do
  echo "  - $file"
done
echo ""
echo "Total results: $(find Results/adata/vertical_integration/SWITCH -name \"*.h5ad\" 2>/dev/null | wc -l) datasets processed"
