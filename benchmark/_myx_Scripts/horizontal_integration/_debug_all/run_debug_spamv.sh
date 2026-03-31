#!/bin/bash
# =============================================================================
# Debug: SpaMV — separate conda env (_Proj1_1_SpaMV)
#
# Tests:
#   1. Vertical: HLN/A1 (ADT, cluster_nums=10)
#   2. Horizontal: Mouse_Brain (ATAC, cluster_nums=18)
# =============================================================================
set -o pipefail

ROOT="/home/users/nus/e1724738/_main/_private/NUS/_Proj1/SMOBench-CLEAN"
cd "$ROOT"

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

echo "=== SpaMV Debug Test Suite ==="
echo "Start: $(date)"

# Create debug data if needed
if [ ! -f "$ROOT/Dataset/_myx_debug_all/vertical/withGT/RNA_ADT/Human_Lymph_Nodes/A1/adata_RNA.h5ad" ]; then
    echo "Creating debug data..."
    python "$ROOT/_myx_Scripts/horizontal_integration/_debug_all/create_debug_data.py"
fi

# --- Vertical: HLN/A1 ADT ---
V_DATA="Dataset/_myx_debug_all/vertical"
V_OUT="_myx_Results/_debug_all/vertical/SpaMV"
mkdir -p "$V_OUT"

run_test "SpaMV x HLN/A1 (vertical, ADT)" \
    python _myx_Scripts/vertical_integration/SpaMV/run_spamv.py \
        --data_type 10x \
        --RNA_path "$V_DATA/withGT/RNA_ADT/Human_Lymph_Nodes/A1/adata_RNA.h5ad" \
        --ADT_path "$V_DATA/withGT/RNA_ADT/Human_Lymph_Nodes/A1/adata_ADT.h5ad" \
        --save_path "$V_OUT/SpaMV_HLN_A1_ADT.h5ad" \
        --method SpaMV --dataset HLN_A1 --cluster_nums 10 --device cuda:0

# --- Vertical: E11 ATAC ---
run_test "SpaMV x E11 (vertical, ATAC)" \
    python _myx_Scripts/vertical_integration/SpaMV/run_spamv.py \
        --data_type MISAR \
        --RNA_path "$V_DATA/withGT/RNA_ATAC/Mouse_Embryos_S1/E11/adata_RNA.h5ad" \
        --ATAC_path "$V_DATA/withGT/RNA_ATAC/Mouse_Embryos_S1/E11/adata_ATAC.h5ad" \
        --save_path "$V_OUT/SpaMV_E11_ATAC.h5ad" \
        --method SpaMV --dataset MISAR_E11 --cluster_nums 8 --device cuda:0

# --- Horizontal: Mouse_Brain ATAC ---
H_DATA="Dataset/_myx_debug_all/horizontal"
H_OUT="_myx_Results/_debug_all/horizontal/SpaMV"
mkdir -p "$H_OUT"

run_test "SpaMV x Brain (horizontal, ATAC)" \
    python _myx_Scripts/horizontal_integration/SpaMV/run_spamv.py \
        --data_type fusion \
        --RNA_path "$H_DATA/RNA_ATAC/Mouse_Brain_Fusion_RNA.h5ad" \
        --ATAC_path "$H_DATA/RNA_ATAC/Mouse_Brain_Fusion_ATAC.h5ad" \
        --save_path "$H_OUT/SpaMV_Brain_ATAC.h5ad" \
        --method SpaMV --dataset debug_Mouse_Brain --cluster_nums 18 --device cuda:0

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "=== SpaMV DEBUG SUMMARY ==="
echo "========================================"
echo "Passed: $PASS"
echo "Failed: $FAIL"
if [ $FAIL -gt 0 ]; then
    echo "Failed tests:"
    echo -e "$FAIL_LIST"
fi
echo "End: $(date)"
