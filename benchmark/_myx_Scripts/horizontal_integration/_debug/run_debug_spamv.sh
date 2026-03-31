#!/bin/bash
# =============================================================================
# Debug script for SpaMV only (uses separate conda env: _Proj1_1_SpaMV)
# Tests SpaMV on Mouse_Brain (ATAC, woGT, ~300 cells)
#
# Usage: bash run_debug_spamv.sh   (from any directory)
#    or: qsub pbs_debug_spamv.pbs  (submit as PBS job)
# =============================================================================

ROOT="/home/users/nus/e1724738/_main/_private/NUS/_Proj1/SMOBench-CLEAN"
cd "$ROOT"

DEBUG_DATA="Dataset/_myx_fusion_debug"
DEBUG_OUT="_myx_Results/_debug"
SCRIPTS="_myx_Scripts/horizontal_integration"

BRAIN_RNA="$DEBUG_DATA/RNA_ATAC/Mouse_Brain_Fusion_RNA.h5ad"
BRAIN_ATAC="$DEBUG_DATA/RNA_ATAC/Mouse_Brain_Fusion_ATAC.h5ad"

mkdir -p "$DEBUG_OUT/SpaMV"

# Create debug data if not exists
if [ ! -f "$ROOT/$BRAIN_RNA" ]; then
    echo "Creating debug data..."
    python "$SCRIPTS/_debug/create_debug_data.py"
fi

echo "=== SpaMV Debug Test ==="
echo "Start: $(date)"

python $SCRIPTS/SpaMV/run_spamv.py \
    --data_type fusion \
    --RNA_path "$BRAIN_RNA" \
    --ATAC_path "$BRAIN_ATAC" \
    --save_path "$DEBUG_OUT/SpaMV/SpaMV_Brain_debug.h5ad" \
    --method SpaMV \
    --dataset Mouse_Brain \
    --cluster_nums 18

rc=$?
echo ""
if [ $rc -eq 0 ] && [ -f "$DEBUG_OUT/SpaMV/SpaMV_Brain_debug.h5ad" ]; then
    echo "<<< PASS: SpaMV x Brain (ATAC)"
else
    echo "<<< FAIL: SpaMV x Brain (ATAC) rc=$rc"
fi
echo "End: $(date)"
