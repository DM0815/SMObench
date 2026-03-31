#!/bin/bash
# Run all remaining mosaic integration (HLN already done in debug)
# Execute on interactive GPU node

set -e

cd /home/users/nus/e1724738/_main/_private/NUS/_Proj1/SMOBench-CLEAN
source /home/users/nus/e1724738/miniconda3/etc/profile.d/conda.sh
conda activate _Proj1_1_SpaMosaic

SCRIPT=_myx_Scripts/mosaic_integration/SpaMosaic/run_spamosaic_mosaic.py
BASE=_myx_Results/adata/mosaic_integration/SpaMosaic

echo "Start: $(date)"
echo "Node: $(hostname)"

# Dataset list (HLN already done)
declare -A DATASETS
DATASETS[HT]=5
DATASETS[Mouse_Spleen]=5
DATASETS[Mouse_Thymus]=8
DATASETS[MISAR_S1]=12
DATASETS[MISAR_S2]=14
DATASETS[Mouse_Brain]=18

for DS in HT Mouse_Spleen Mouse_Thymus MISAR_S1 MISAR_S2 Mouse_Brain; do
    K=${DATASETS[$DS]}
    echo ""
    echo "====== $DS (cluster_nums=$K) ======"

    echo "--- without_rna ---"
    python $SCRIPT --dataset $DS --scenario without_rna \
        --save_path ${BASE}/${DS}/without_rna/SpaMosaic_${DS}_without_rna.h5ad \
        --cluster_nums $K --device cuda:0

    echo "--- without_second ---"
    python $SCRIPT --dataset $DS --scenario without_second \
        --save_path ${BASE}/${DS}/without_second/SpaMosaic_${DS}_without_second.h5ad \
        --cluster_nums $K --device cuda:0

    echo "[$DS] DONE"
done

echo ""
echo "=============================="
echo "All mosaic integration complete!"
echo "End: $(date)"
echo "Output files:"
find $BASE -name "*.h5ad" | sort
echo "=============================="
