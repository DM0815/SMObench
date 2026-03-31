#!/bin/bash
# Debug mosaic integration with smallest dataset (HLN, 2 batches)
# Run on interactive GPU node

set -e

cd /home/users/nus/e1724738/_main/_private/NUS/_Proj1/SMOBench-CLEAN
source /home/users/nus/e1724738/miniconda3/etc/profile.d/conda.sh
conda activate _Proj1_1_SpaMosaic

SCRIPT=_myx_Scripts/mosaic_integration/SpaMosaic/run_spamosaic_mosaic.py
OUTDIR=_myx_Results/adata/mosaic_integration/SpaMosaic/HLN

echo "============================================"
echo "Debug: SpaMosaic Mosaic — HLN (2 batches)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo "============================================"

echo ""
echo "=== Test 1/2: Without RNA ==="
python $SCRIPT --dataset HLN --scenario without_rna \
    --save_path ${OUTDIR}/without_rna/SpaMosaic_HLN_without_rna.h5ad \
    --cluster_nums 10 --device cuda:0

echo ""
echo "=== Test 2/2: Without Second (ADT) ==="
python $SCRIPT --dataset HLN --scenario without_second \
    --save_path ${OUTDIR}/without_second/SpaMosaic_HLN_without_second.h5ad \
    --cluster_nums 10 --device cuda:0

echo ""
echo "============================================"
echo "Debug PASSED! Output files:"
ls -lh ${OUTDIR}/without_rna/*.h5ad ${OUTDIR}/without_second/*.h5ad 2>/dev/null
echo "End: $(date)"
echo "============================================"
