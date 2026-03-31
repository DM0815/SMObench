#!/bin/bash

# Comprehensive SMOPCA run script for all datasets

echo "Starting comprehensive SpaBalance_3M processing..."
echo "Start time: $(date)"

# Create base results directory
mkdir -p Results/adata/vertical_integration/SpaBalance_3M Results/plot/vertical_integration/SpaBalance_3M

# === 3M Simulation Dataset ===
echo "Processing 3M_Simulation datasets..."

python Scripts/vertical_integration/SpaBalance/run_SpaBalance_3M.py \
  --data_type Triplet \
  --RNA_path Dataset/withGT/3M_Simulation/adata_RNA.h5ad \
  --ADT_path Dataset/withGT/3M_Simulation/adata_ADT.h5ad \
  --ATAC_path Dataset/withGT/3M_Simulation/adata_ATAC.h5ad \
  --save_path Results/adata/vertical_integration/SpaBalance_3M/SpaBalance_3M_adata_3M.h5ad \
  --cluster_nums 5 \

# # === 3M Simulation Dataset ===

# # echo "Processing 3M Simulation dataset..."

# if [ -f "Dataset/withGT/3M_Simulation/adata_RNA.h5ad" ] && [ -f "Dataset/withGT/3M_Simulation/adata_ADT.h5ad" ]; then
#     echo "Processing 3M Simulation (RNA+ADT)..."
#     python Scripts/vertical_integration/SpaBalance/run_SpaBalance_3M.py \
#       --data_type simulation \
#       --RNA_path Dataset/withGT/3M_Simulation/adata_RNA.h5ad \
#       --ADT_path Dataset/withGT/3M_Simulation/adata_ADT.h5ad \
#       --save_path Results/adata/vertical_integration/3M_Simulation/adata_3M.h5ad \
#       --method Simulation \
#       --dataset 3M_Simulation/Simulation \
#       --cluster_nums 5
# fi
