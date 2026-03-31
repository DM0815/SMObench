#!/bin/bash
# =============================================================================
# Debug: Horizontal integration — 7 methods on fusion data (~500 cells)
#
# ADT: Mouse_Thymus (cluster_nums=8)
# ATAC: Mouse_Embryos_S1 (cluster_nums=12), Mouse_Brain (cluster_nums=18)
#
# Methods:
#   SMOPCA: ADT + ATAC (CPU method)
#   CANDIES, PRAGA, SpaBalance, SpaFusion: ADT only (Mouse_Thymus)
#   SpaMI: ATAC only (Mouse_Embryos_S1)
#   SpaMV: ATAC only (Mouse_Brain) — separate script/env
# =============================================================================
set -o pipefail

ROOT="/home/users/nus/e1724738/_main/_private/NUS/_Proj1/SMOBench-CLEAN"
cd "$ROOT"

DATA="Dataset/_myx_debug_all/horizontal"
OUT="_myx_Results/_debug_all/horizontal"
HS="_myx_Scripts/horizontal_integration"

# --- Data paths ---
THY_RNA="$DATA/RNA_ADT/Mouse_Thymus_Fusion_RNA.h5ad"
THY_ADT="$DATA/RNA_ADT/Mouse_Thymus_Fusion_ADT.h5ad"
S1_RNA="$DATA/RNA_ATAC/Mouse_Embryos_S1_Fusion_RNA.h5ad"
S1_ATAC="$DATA/RNA_ATAC/Mouse_Embryos_S1_Fusion_ATAC.h5ad"
BRAIN_RNA="$DATA/RNA_ATAC/Mouse_Brain_Fusion_RNA.h5ad"
BRAIN_ATAC="$DATA/RNA_ATAC/Mouse_Brain_Fusion_ATAC.h5ad"

PASS=0; FAIL=0; SKIP=0; FAIL_LIST=""

run_test() {
    local name="$1"; shift
    echo ""
    echo "========================================"
    echo ">>> TEST: $name"
    echo "========================================"
    local start=$(date +%s)
    "$@" 2>&1
    local rc=$?
    local elapsed=$(( $(date +%s) - start ))
    if [ $rc -eq 0 ]; then
        echo "<<< PASS: $name (${elapsed}s)"
        PASS=$((PASS + 1))
    else
        echo "<<< FAIL: $name (rc=$rc, ${elapsed}s)"
        FAIL=$((FAIL + 1))
        FAIL_LIST="$FAIL_LIST  - $name (rc=$rc)\n"
    fi
}

echo "=== Horizontal Debug Test Suite ==="
echo "Start: $(date)"

# Create debug data if needed
if [ ! -f "$ROOT/$THY_RNA" ]; then
    echo "Creating debug data..."
    python "$ROOT/_myx_Scripts/horizontal_integration/_debug_all/create_debug_data.py"
fi

mkdir -p "$OUT"/{SMOPCA,CANDIES,PRAGA,SpaBalance,SpaFusion,SpaMI,SpaMV}

# =============================================================================
# SMOPCA — CPU method, tests both ADT and ATAC
# =============================================================================

run_test "SMOPCA x Thymus (ADT)" \
    python $HS/SMOPCA/run_SMOPCA.py \
        --data_type fusion \
        --RNA_path "$THY_RNA" --ADT_path "$THY_ADT" \
        --save_path "$OUT/SMOPCA/SMOPCA_Thymus_ADT.h5ad" \
        --method SMOPCA --dataset debug_Mouse_Thymus --cluster_nums 8

run_test "SMOPCA x Embryos_S1 (ATAC)" \
    python $HS/SMOPCA/run_SMOPCA.py \
        --data_type fusion \
        --RNA_path "$S1_RNA" --ATAC_path "$S1_ATAC" \
        --save_path "$OUT/SMOPCA/SMOPCA_S1_ATAC.h5ad" \
        --method SMOPCA --dataset debug_Mouse_Embryos_S1 --cluster_nums 12

# =============================================================================
# ADT methods — Mouse_Thymus (cluster_nums=8)
# =============================================================================

run_test "CANDIES x Thymus (ADT)" \
    python $HS/CANDIES/run_candies.py \
        --data_type fusion \
        --RNA_path "$THY_RNA" --ADT_path "$THY_ADT" \
        --save_path "$OUT/CANDIES/CANDIES_Thymus_ADT.h5ad" \
        --method CANDIES --dataset debug_Mouse_Thymus --cluster_nums 8

run_test "PRAGA x Thymus (ADT)" \
    python $HS/PRAGA/run_praga.py \
        --data_type fusion \
        --RNA_path "$THY_RNA" --ADT_path "$THY_ADT" \
        --save_path "$OUT/PRAGA/PRAGA_Thymus_ADT.h5ad" \
        --method PRAGA --dataset debug_Mouse_Thymus --cluster_nums 8

run_test "SpaBalance x Thymus (ADT)" \
    python $HS/SpaBalance/run_SpaBalance.py \
        --data_type fusion \
        --RNA_path "$THY_RNA" --ADT_path "$THY_ADT" \
        --save_path "$OUT/SpaBalance/SpaBalance_Thymus_ADT.h5ad" \
        --dataset debug_Mouse_Thymus --cluster_nums 8

run_test "SpaFusion x Thymus (ADT)" \
    python $HS/SpaFusion/run_SpaFusion.py \
        --RNA_path "$THY_RNA" --ADT_path "$THY_ADT" \
        --save_path "$OUT/SpaFusion/SpaFusion_Thymus_ADT.h5ad" \
        --dataset debug_Mouse_Thymus --cluster_nums 8 --device cuda:0

# =============================================================================
# ATAC methods — Mouse_Embryos_S1 (cluster_nums=12)
# =============================================================================

run_test "SpaMI x Embryos_S1 (ATAC)" \
    python $HS/SpaMI/run_SpaMI.py \
        --RNA_path "$S1_RNA" --ATAC_path "$S1_ATAC" \
        --save_path "$OUT/SpaMI/SpaMI_S1_ATAC.h5ad" \
        --dataset debug_Mouse_Embryos_S1 --cluster_nums 12

# SpaMV — separate env, see run_debug_spamv.sh
echo ""
echo ">>> SKIP: SpaMV x Brain (ATAC) — see run_debug_spamv.sh"
SKIP=$((SKIP + 1))

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "=== HORIZONTAL DEBUG SUMMARY ==="
echo "========================================"
echo "Passed: $PASS"
echo "Failed: $FAIL"
echo "Skipped: $SKIP"
if [ $FAIL -gt 0 ]; then
    echo "Failed tests:"
    echo -e "$FAIL_LIST"
fi
echo "End: $(date)"

echo ""
echo "Output files:"
find "$OUT" -name "*.h5ad" -ls 2>/dev/null || echo "  (none)"
