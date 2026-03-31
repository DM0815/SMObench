#!/bin/bash

# Comprehensive SpaFusion vertical integration benchmark script
# Run from SMOBench root directory:
#   bash Scripts/vertical_integration/SpaFusion/run.sh
# Cluster numbers:
#   HLN_A1:10 HLN_D1:11 HT_S1:4 HT_S2:5 HT_S3:5
#   Mouse_Thymus:8 Mouse_Spleen:5
#   MISAR_S1:8/12/12/14 MISAR_S2:13/14/15/16 Mouse_Brain:18

echo "=== Starting SpaFusion vertical integration benchmark ==="
echo "Start time: $(date)"

mkdir -p Results/adata/vertical_integration/SpaFusion
mkdir -p Results/plot/vertical_integration/SpaFusion

clean_spafusion_temp() {
  for dir in results pretrain pre_adj; do
    if [ -d "$dir" ]; then
      rm -rf "$dir"
    fi
  done
}

run_spafusion_dataset() {
  clean_spafusion_temp
  python Scripts/vertical_integration/SpaFusion/run_SpaFusion.py "$@"
}

# =========================================================
# === withGT RNA + ADT datasets ===========================
# =========================================================

echo "Processing Human_Lymph_Nodes datasets..."

echo "Processing Human_Lymph_Nodes A1..."
run_spafusion_dataset \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/A1/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/A1/adata_ADT.h5ad \
  --save_path Results/adata/vertical_integration/SpaFusion/HLN/A1/SpaFusion_HLN_A1.h5ad \
  --dataset Human_Lymph_Nodes/A1 \
  --cluster_nums 10

echo "Processing Human_Lymph_Nodes D1..."
run_spafusion_dataset \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/D1/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/D1/adata_ADT.h5ad \
  --save_path Results/adata/vertical_integration/SpaFusion/HLN/D1/SpaFusion_HLN_D1.h5ad \
  --dataset Human_Lymph_Nodes/D1 \
  --cluster_nums 11

echo "Processing Human_Tonsils datasets..."

echo "Processing Human_Tonsils S1..."
run_spafusion_dataset \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Tonsils/S1/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Tonsils/S1/adata_ADT.h5ad \
  --save_path Results/adata/vertical_integration/SpaFusion/HT/S1/SpaFusion_HT_S1.h5ad \
  --dataset Human_Tonsils/S1 \
  --cluster_nums 4

echo "Processing Human_Tonsils S2..."
run_spafusion_dataset \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Tonsils/S2/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Tonsils/S2/adata_ADT.h5ad \
  --save_path Results/adata/vertical_integration/SpaFusion/HT/S2/SpaFusion_HT_S2.h5ad \
  --dataset Human_Tonsils/S2 \
  --cluster_nums 5

echo "Processing Human_Tonsils S3..."
run_spafusion_dataset \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Tonsils/S3/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Tonsils/S3/adata_ADT.h5ad \
  --save_path Results/adata/vertical_integration/SpaFusion/HT/S3/SpaFusion_HT_S3.h5ad \
  --dataset Human_Tonsils/S3 \
  --cluster_nums 5

# =========================================================
# === woGT RNA + ADT datasets =============================
# =========================================================

echo "Processing Mouse_Thymus datasets..."
for thymus_id in 1 2 3 4; do
  echo "Processing Mouse_Thymus${thymus_id}..."
  run_spafusion_dataset \
    --RNA_path Dataset/woGT/RNA_ADT/Mouse_Thymus/Mouse_Thymus${thymus_id}/adata_RNA.h5ad \
    --ADT_path Dataset/woGT/RNA_ADT/Mouse_Thymus/Mouse_Thymus${thymus_id}/adata_ADT.h5ad \
    --save_path Results/adata/vertical_integration/SpaFusion/Mouse_Thymus/Thymus${thymus_id}/SpaFusion_MT_Thymus${thymus_id}.h5ad \
    --dataset Mouse_Thymus/Mouse_Thymus${thymus_id} \
    --cluster_nums 8
done

echo "Processing Mouse_Spleen datasets..."
for spleen_id in 1 2; do
  echo "Processing Mouse_Spleen${spleen_id}..."
  run_spafusion_dataset \
    --RNA_path Dataset/woGT/RNA_ADT/Mouse_Spleen/Mouse_Spleen${spleen_id}/adata_RNA.h5ad \
    --ADT_path Dataset/woGT/RNA_ADT/Mouse_Spleen/Mouse_Spleen${spleen_id}/adata_ADT.h5ad \
    --save_path Results/adata/vertical_integration/SpaFusion/Mouse_Spleen/Spleen${spleen_id}/SpaFusion_MS_Spleen${spleen_id}.h5ad \
    --dataset Mouse_Spleen/Mouse_Spleen${spleen_id} \
    --cluster_nums 5
done

# =========================================================
# === RNA + ATAC datasets (not supported) =================
# =========================================================

echo "SpaFusion does not support RNA+ATAC datasets; skipping Mouse_Embryos and Mouse_Brain tasks."

# =========================================================
# === Summary =============================================
# =========================================================

echo "SpaFusion processing completed!"
echo "End time: $(date)"
echo ""
echo "=== PROCESSING SUMMARY ==="
echo "Results saved to Results/adata/vertical_integration/SpaFusion/"
echo "Plots saved to Results/plot/vertical_integration/SpaFusion/"
echo ""
echo "Processed datasets:"
find Results/adata/vertical_integration/SpaFusion -name "*.h5ad" 2>/dev/null | sort | while read -r file; do
  echo "  - $file"
done
echo ""
echo "Total results: $(find Results/adata/vertical_integration/SpaFusion -name \"*.h5ad\" 2>/dev/null | wc -l) datasets processed"

clean_spafusion_temp
