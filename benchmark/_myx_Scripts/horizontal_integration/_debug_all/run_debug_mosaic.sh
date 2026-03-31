#!/bin/bash
# =============================================================================
# Debug: Mosaic (3M) integration — 3 methods on 3M_Simulation (1296 cells)
#
# Data: 3M_Simulation (RNA + ADT + ATAC, cluster_nums=5)
# Methods: SpaBalance_3M, PRAGA_3M, SpatialGlue_3M
# =============================================================================
set -o pipefail

ROOT="/home/users/nus/e1724738/_main/_private/NUS/_Proj1/SMOBench-CLEAN"
cd "$ROOT"

DATA="Dataset/_myx_debug_all/mosaic/3M_Simulation"
OUT="_myx_Results/_debug_all/mosaic"
VS="_myx_Scripts/vertical_integration"

RNA="$DATA/adata_RNA.h5ad"
ADT="$DATA/adata_ADT.h5ad"
ATAC="$DATA/adata_ATAC.h5ad"

PASS=0; FAIL=0; FAIL_LIST=""

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

echo "=== Mosaic (3M) Debug Test Suite ==="
echo "Start: $(date)"

# Create debug data if needed
if [ ! -f "$ROOT/$RNA" ]; then
    echo "Creating debug data..."
    python "$ROOT/_myx_Scripts/horizontal_integration/_debug_all/create_debug_data.py"
fi

mkdir -p "$OUT"/{SpaBalance_3M,PRAGA_3M,SpatialGlue_3M}

# =============================================================================
# SpaBalance_3M (uses dedicated run_SpaBalance_3M.py)
# =============================================================================

run_test "SpaBalance_3M x 3M_Simulation" \
    python $VS/SpaBalance/run_SpaBalance_3M.py \
        --data_type simulation \
        --RNA_path "$RNA" --ADT_path "$ADT" --ATAC_path "$ATAC" \
        --save_path "$OUT/SpaBalance_3M/SpaBalance_3M_sim.h5ad" \
        --cluster_nums 5 --device cuda:0

# =============================================================================
# PRAGA_3M (run_praga.py only supports 2 modalities; test RNA+ADT with simulation)
# =============================================================================

run_test "PRAGA_3M x 3M_Simulation (RNA+ADT)" \
    python $VS/PRAGA/run_praga.py \
        --data_type simulation \
        --RNA_path "$RNA" --ADT_path "$ADT" \
        --save_path "$OUT/PRAGA_3M/PRAGA_3M_sim_ADT.h5ad" \
        --method PRAGA --dataset 3M_Simulation --cluster_nums 5 --device cuda:0

run_test "PRAGA_3M x 3M_Simulation (RNA+ATAC)" \
    python $VS/PRAGA/run_praga.py \
        --data_type simulation \
        --RNA_path "$RNA" --ATAC_path "$ATAC" \
        --save_path "$OUT/PRAGA_3M/PRAGA_3M_sim_ATAC.h5ad" \
        --method PRAGA --dataset 3M_Simulation --cluster_nums 5 --device cuda:0

# =============================================================================
# SpatialGlue_3M (run_SpatialGlue.py only supports 2 modalities; test RNA+ADT with simulation)
# =============================================================================

run_test "SpatialGlue_3M x 3M_Simulation (RNA+ADT)" \
    python $VS/SpatialGlue/run_SpatialGlue.py \
        --data_type simulation \
        --RNA_path "$RNA" --ADT_path "$ADT" \
        --save_path "$OUT/SpatialGlue_3M/SpatialGlue_3M_sim_ADT.h5ad" \
        --method SpatialGlue --dataset 3M_Simulation --cluster_nums 5 --device cuda:0

run_test "SpatialGlue_3M x 3M_Simulation (RNA+ATAC)" \
    python $VS/SpatialGlue/run_SpatialGlue.py \
        --data_type simulation \
        --RNA_path "$RNA" --ATAC_path "$ATAC" \
        --save_path "$OUT/SpatialGlue_3M/SpatialGlue_3M_sim_ATAC.h5ad" \
        --method SpatialGlue --dataset 3M_Simulation --cluster_nums 5 --device cuda:0

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "=== MOSAIC DEBUG SUMMARY ==="
echo "========================================"
echo "Passed: $PASS"
echo "Failed: $FAIL"
if [ $FAIL -gt 0 ]; then
    echo "Failed tests:"
    echo -e "$FAIL_LIST"
fi
echo "End: $(date)"

echo ""
echo "Output files:"
find "$OUT" -name "*.h5ad" -ls 2>/dev/null || echo "  (none)"
