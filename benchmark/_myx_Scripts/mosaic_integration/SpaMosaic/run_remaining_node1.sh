#!/bin/bash
# Node 1: Mouse_Thymus + MISAR_S1
cd /home/users/nus/e1724738/_main/_private/NUS/_Proj1/SMOBench-CLEAN
source /home/users/nus/e1724738/miniconda3/etc/profile.d/conda.sh
conda activate _Proj1_1_SpaMosaic

SCRIPT=_myx_Scripts/mosaic_integration/SpaMosaic/run_spamosaic_mosaic.py
BASE=_myx_Results/adata/mosaic_integration/SpaMosaic

echo "Node1 Start: $(date)"

for DS_K in "Mouse_Thymus 8" "MISAR_S1 12"; do
    DS=$(echo $DS_K | cut -d' ' -f1)
    K=$(echo $DS_K | cut -d' ' -f2)
    echo "====== $DS (k=$K) ======"
    for SC in without_rna without_second; do
        echo "--- $SC ---"
        python $SCRIPT --dataset $DS --scenario $SC \
            --save_path ${BASE}/${DS}/${SC}/SpaMosaic_${DS}_${SC}.h5ad \
            --cluster_nums $K --device cuda:0 || echo "[WARN] $DS $SC failed"
    done
    echo "[$DS] DONE"
done

echo "Node1 End: $(date)"
