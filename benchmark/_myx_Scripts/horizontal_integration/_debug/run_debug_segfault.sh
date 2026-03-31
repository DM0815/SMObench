#!/bin/bash
# Quick debug: test ONLY CANDIES mclust segfault with added debug prints
set -o pipefail

ROOT="/home/users/nus/e1724738/_main/_private/NUS/_Proj1/SMOBench-CLEAN"
cd "$ROOT"

DEBUG_DATA="Dataset/_myx_fusion_debug"
DEBUG_OUT="_myx_Results/_debug"
SCRIPTS="_myx_Scripts/horizontal_integration"

THY_RNA="$DEBUG_DATA/RNA_ADT/Mouse_Thymus_Fusion_RNA.h5ad"
THY_ADT="$DEBUG_DATA/RNA_ADT/Mouse_Thymus_Fusion_ADT.h5ad"

mkdir -p "$DEBUG_OUT/CANDIES"

echo "=== Segfault Debug Test ==="
echo "Start: $(date)"
echo ""

echo ">>> CANDIES x Thymus (ADT) — mclust segfault test"
python $SCRIPTS/CANDIES/run_candies.py \
    --data_type fusion \
    --RNA_path "$THY_RNA" \
    --ADT_path "$THY_ADT" \
    --save_path "$DEBUG_OUT/CANDIES/CANDIES_Thymus_debug.h5ad" \
    --method CANDIES \
    --dataset Mouse_Thymus \
    --cluster_nums 8 2>&1

rc=$?
echo ""
echo "Exit code: $rc"
if [ $rc -eq 139 ]; then
    echo ">>> CONFIRMED SEGFAULT (SIGSEGV)"
elif [ $rc -eq 134 ]; then
    echo ">>> ABORT (SIGABRT)"
elif [ $rc -eq 0 ]; then
    echo ">>> PASS"
else
    echo ">>> FAIL (rc=$rc)"
fi
echo "End: $(date)"
