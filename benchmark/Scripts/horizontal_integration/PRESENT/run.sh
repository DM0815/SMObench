#!/bin/bash

# PRESENT Horizontal Integration Script
# Uses PRESENT's native horizontal integration support with individual datasets
# Use relative paths for portability - script should be run from SMOBench root directory

echo "Starting PRESENT horizontal integration processing..."
echo "Start time: $(date)"

# Create base results directory for horizontal integration
mkdir -p Results/adata/horizontal_integration/PRESENT Results/plot/horizontal_integration/PRESENT

# === withGT RNA+ADT Datasets ===

echo "Processing RNA+ADT datasets with ground truth..."

# Human_Lymph_Nodes (combines A1 + D1)
echo "Processing Human_Lymph_Nodes horizontal integration..."
python Scripts/horizontal_integration/PRESENT/run_present.py \
  --data_type horizontal \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Lymph_Nodes \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Lymph_Nodes \
  --save_path Results/adata/horizontal_integration/PRESENT/HLN/fusion/PRESENT_HLN_fusion.h5ad \
  --method PRESENT \
  --dataset HLN \
  --cluster_nums 10 \
  --seed 2024 \
  --device cuda:0

# Human_Tonsils (combines S1 + S2 + S3)
echo "Processing Human_Tonsils horizontal integration..."
python Scripts/horizontal_integration/PRESENT/run_present.py \
  --data_type horizontal \
  --RNA_path Dataset/withGT/RNA_ADT/Human_Tonsils \
  --ADT_path Dataset/withGT/RNA_ADT/Human_Tonsils \
  --save_path Results/adata/horizontal_integration/PRESENT/HT/fusion/PRESENT_HT_fusion.h5ad \
  --method PRESENT \
  --dataset HT \
  --cluster_nums 5 \
  --seed 2024 \
  --device cuda:0

# === withGT RNA+ATAC Datasets ===

echo "Processing RNA+ATAC datasets with ground truth..."

# Mouse_Embryos_S1 (combines E11 + E13 + E15 + E18)
echo "Processing Mouse_Embryos_S1 horizontal integration..."
python Scripts/horizontal_integration/PRESENT/run_present.py \
  --data_type horizontal \
  --RNA_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1 \
  --ATAC_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1 \
  --save_path Results/adata/horizontal_integration/PRESENT/MISAR_S1/fusion/PRESENT_MISAR_S1_fusion.h5ad \
  --method PRESENT \
  --dataset MISAR_S1 \
  --cluster_nums 12 \
  --seed 2024 \
  --device cuda:0

# Mouse_Embryos_S2 (combines E11 + E13 + E15 + E18)
echo "Processing Mouse_Embryos_S2 horizontal integration..."
python Scripts/horizontal_integration/PRESENT/run_present.py \
  --data_type horizontal \
  --RNA_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2 \
  --ATAC_path Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2 \
  --save_path Results/adata/horizontal_integration/PRESENT/MISAR_S2/fusion/PRESENT_MISAR_S2_fusion.h5ad \
  --method PRESENT \
  --dataset MISAR_S2 \
  --cluster_nums 14 \
  --seed 2024 \
  --device cuda:0

# === woGT RNA+ADT Datasets ===

echo "Processing RNA+ADT datasets without ground truth..."

# Mouse_Thymus (combines multiple thymus samples)
echo "Processing Mouse_Thymus horizontal integration..."
python Scripts/horizontal_integration/PRESENT/run_present.py \
  --data_type horizontal \
  --RNA_path Dataset/woGT/RNA_ADT/Mouse_Thymus \
  --ADT_path Dataset/woGT/RNA_ADT/Mouse_Thymus \
  --save_path Results/adata/horizontal_integration/PRESENT/Mouse_Thymus/fusion/PRESENT_Mouse_Thymus_fusion.h5ad \
  --method PRESENT \
  --dataset Mouse_Thymus \
  --cluster_nums 8 \
  --seed 2024 \
  --device cuda:0

# Mouse_Spleen (combines multiple spleen samples)
echo "Processing Mouse_Spleen horizontal integration..."
python Scripts/horizontal_integration/PRESENT/run_present.py \
  --data_type horizontal \
  --RNA_path Dataset/woGT/RNA_ADT/Mouse_Spleen \
  --ADT_path Dataset/woGT/RNA_ADT/Mouse_Spleen \
  --save_path Results/adata/horizontal_integration/PRESENT/Mouse_Spleen/fusion/PRESENT_Mouse_Spleen_fusion.h5ad \
  --method PRESENT \
  --dataset Mouse_Spleen \
  --cluster_nums 5 \
  --seed 2024 \
  --device cuda:0

# === woGT RNA+ATAC Datasets ===

echo "Processing RNA+ATAC datasets without ground truth..."

# Mouse_Brain (combines multiple brain modalities/regions)
echo "Processing Mouse_Brain horizontal integration..."
python Scripts/horizontal_integration/PRESENT/run_present.py \
  --data_type horizontal \
  --RNA_path Dataset/woGT/RNA_ATAC/Mouse_Brain \
  --ATAC_path Dataset/woGT/RNA_ATAC/Mouse_Brain \
  --save_path Results/adata/horizontal_integration/PRESENT/Mouse_Brain/fusion/PRESENT_Mouse_Brain_fusion.h5ad \
  --method PRESENT \
  --dataset Mouse_Brain \
  --cluster_nums 18 \
  --seed 2024 \
  --device cuda:0

echo "PRESENT horizontal integration processing completed!"
echo "End time: $(date)"

# Generate summary report
echo "=== HORIZONTAL INTEGRATION PROCESSING SUMMARY ==="
echo "Results saved to Results/adata/horizontal_integration/PRESENT/"
echo "Plots saved to Results/plot/horizontal_integration/PRESENT/"
echo ""
echo "Processed datasets:"
find Results/adata/horizontal_integration/PRESENT -name "*.h5ad" | sort | while read file; do
    echo "  - $file"
done

echo ""
echo "Total datasets processed: $(find Results/adata/horizontal_integration/PRESENT -name "*.h5ad" 2>/dev/null | wc -l)"

echo ""
echo "Key features of PRESENT horizontal integration:"
echo "  - Native horizontal integration support (no fusion data needed)"
echo "  - Input: Individual batch datasets loaded separately"
echo "  - Goal: Remove batch effects while preserving biological signals"
echo "  - Training: PRESENT_BC_function with batch correction capabilities"
echo "  - Output: Integrated embeddings with reduced batch effects"
echo "  - Evaluation: Includes batch mixing metrics (BER) in addition to BioC and SC"
echo ""
echo "PRESENT successfully processes both RNA+ADT and RNA+ATAC data types"