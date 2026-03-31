#!/bin/bash
# =============================================================================
# Debug script: run all 7 methods on small (~300 cells) subsampled data.
# Tests one representative tissue per data type to catch code-level bugs fast.
#
# Methods tested:
#   ADT (Mouse_Thymus):  SMOPCA, CANDIES, PRAGA, SpaBalance, SpaFusion
#   ATAC (Mouse_Embryos_S1): SMOPCA, SpaMI
#   ATAC (Mouse_Brain):  SpaMV (separate env, see run_debug_spamv.sh)
#
# Usage: bash run_debug.sh        (from any directory)
#    or: qsub pbs_debug.pbs       (submit as PBS job)
# =============================================================================
set -o pipefail

ROOT="/home/users/nus/e1724738/_main/_private/NUS/_Proj1/SMOBench-CLEAN"
cd "$ROOT"

DEBUG_DATA="Dataset/_myx_fusion_debug"
DEBUG_OUT="_myx_Results/_debug"
SCRIPTS="_myx_Scripts/horizontal_integration"

# Tissue paths
THY_RNA="$DEBUG_DATA/RNA_ADT/Mouse_Thymus_Fusion_RNA.h5ad"
THY_ADT="$DEBUG_DATA/RNA_ADT/Mouse_Thymus_Fusion_ADT.h5ad"
S1_RNA="$DEBUG_DATA/RNA_ATAC/Mouse_Embryos_S1_Fusion_RNA.h5ad"
S1_ATAC="$DEBUG_DATA/RNA_ATAC/Mouse_Embryos_S1_Fusion_ATAC.h5ad"
BRAIN_RNA="$DEBUG_DATA/RNA_ATAC/Mouse_Brain_Fusion_RNA.h5ad"
BRAIN_ATAC="$DEBUG_DATA/RNA_ATAC/Mouse_Brain_Fusion_ATAC.h5ad"

PASS=0
FAIL=0
FAIL_LIST=""

run_test() {
    local name="$1"
    shift
    echo ""
    echo "========================================"
    echo ">>> TEST: $name"
    echo "========================================"
    local start=$(date +%s)

    "$@" 2>&1
    local rc=$?

    local end=$(date +%s)
    local elapsed=$((end - start))

    if [ $rc -eq 0 ]; then
        echo "<<< PASS: $name (${elapsed}s)"
        PASS=$((PASS + 1))
    else
        echo "<<< FAIL: $name (rc=$rc, ${elapsed}s)"
        FAIL=$((FAIL + 1))
        FAIL_LIST="$FAIL_LIST  - $name\n"
    fi
}

echo "=== Debug Test Suite ==="
echo "Start: $(date)"
echo ""

# --- Step 0: Create debug data if not exists ---
if [ ! -f "$ROOT/$THY_RNA" ]; then
    echo "Creating debug data (subsampling to 300 cells)..."
    python "$SCRIPTS/_debug/create_debug_data.py"
    echo ""
fi

# Create output dirs
mkdir -p "$DEBUG_OUT"/{SMOPCA,CANDIES,PRAGA,SpaBalance,SpaFusion,SpaMI,SpaMV}

# =============================================================================
# ADT methods — Mouse_Thymus (cluster_nums=8)
# =============================================================================

run_test "SMOPCA x Thymus (ADT)" \
    python $SCRIPTS/SMOPCA/run_SMOPCA.py \
        --data_type fusion \
        --RNA_path "$THY_RNA" \
        --ADT_path "$THY_ADT" \
        --save_path "$DEBUG_OUT/SMOPCA/SMOPCA_Thymus_debug.h5ad" \
        --method SMOPCA \
        --dataset Mouse_Thymus \
        --cluster_nums 8

run_test "CANDIES x Thymus (ADT)" \
    python $SCRIPTS/CANDIES/run_candies.py \
        --data_type fusion \
        --RNA_path "$THY_RNA" \
        --ADT_path "$THY_ADT" \
        --save_path "$DEBUG_OUT/CANDIES/CANDIES_Thymus_debug.h5ad" \
        --method CANDIES \
        --dataset Mouse_Thymus \
        --cluster_nums 8

run_test "PRAGA x Thymus (ADT)" \
    python $SCRIPTS/PRAGA/run_praga.py \
        --data_type fusion \
        --RNA_path "$THY_RNA" \
        --ADT_path "$THY_ADT" \
        --save_path "$DEBUG_OUT/PRAGA/PRAGA_Thymus_debug.h5ad" \
        --method PRAGA \
        --dataset Mouse_Thymus \
        --cluster_nums 8

run_test "SpaBalance x Thymus (ADT)" \
    python $SCRIPTS/SpaBalance/run_SpaBalance.py \
        --data_type fusion \
        --RNA_path "$THY_RNA" \
        --ADT_path "$THY_ADT" \
        --save_path "$DEBUG_OUT/SpaBalance/SpaBalance_Thymus_debug.h5ad" \
        --dataset Mouse_Thymus \
        --cluster_nums 8

run_test "SpaFusion x Thymus (ADT)" \
    python $SCRIPTS/SpaFusion/run_SpaFusion.py \
        --RNA_path "$THY_RNA" \
        --ADT_path "$THY_ADT" \
        --save_path "$DEBUG_OUT/SpaFusion/SpaFusion_Thymus_debug.h5ad" \
        --dataset Mouse_Thymus \
        --cluster_nums 8 \
        --device cuda:0

# =============================================================================
# ATAC methods — Mouse_Embryos_S1 (cluster_nums=12)
# =============================================================================

run_test "SMOPCA x Embryos_S1 (ATAC)" \
    python $SCRIPTS/SMOPCA/run_SMOPCA.py \
        --data_type fusion \
        --RNA_path "$S1_RNA" \
        --ATAC_path "$S1_ATAC" \
        --save_path "$DEBUG_OUT/SMOPCA/SMOPCA_S1_debug.h5ad" \
        --method SMOPCA \
        --dataset Mouse_Embryos_S1 \
        --cluster_nums 12

run_test "SpaMI x Embryos_S1 (ATAC)" \
    python $SCRIPTS/SpaMI/run_SpaMI.py \
        --RNA_path "$S1_RNA" \
        --ATAC_path "$S1_ATAC" \
        --save_path "$DEBUG_OUT/SpaMI/SpaMI_S1_debug.h5ad" \
        --dataset Mouse_Embryos_S1 \
        --cluster_nums 12

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "========================================"
echo "=== DEBUG SUMMARY ==="
echo "========================================"
echo "Passed: $PASS"
echo "Failed: $FAIL"
if [ $FAIL -gt 0 ]; then
    echo "Failed tests:"
    echo -e "$FAIL_LIST"
fi
echo "End: $(date)"

# Verify output files
echo ""
echo "Output files:"
find "$DEBUG_OUT" -name "*.h5ad" -ls 2>/dev/null || echo "  (none)"
