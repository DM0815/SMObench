#!/bin/bash
# =============================================================================
# Debug: Vertical integration — 15 methods × representative slices
#
# ADT (withGT): HLN/A1 (cluster_nums=10)
# ATAC (withGT): Mouse_Embryos_S1/E11 (cluster_nums=8)
#
# All methods except SpaMV use env _Proj1_1.
# SpaMV tested separately in run_debug_spamv.sh.
# =============================================================================
set -o pipefail

ROOT="/home/users/nus/e1724738/_main/_private/NUS/_Proj1/SMOBench-CLEAN"
cd "$ROOT"

DATA="Dataset/_myx_debug_all/vertical"
OUT="_myx_Results/_debug_all/vertical"
VS="_myx_Scripts/vertical_integration"

# --- Data paths ---
HLN_RNA="$DATA/withGT/RNA_ADT/Human_Lymph_Nodes/A1/adata_RNA.h5ad"
HLN_ADT="$DATA/withGT/RNA_ADT/Human_Lymph_Nodes/A1/adata_ADT.h5ad"
E11_RNA="$DATA/withGT/RNA_ATAC/Mouse_Embryos_S1/E11/adata_RNA.h5ad"
E11_ATAC="$DATA/withGT/RNA_ATAC/Mouse_Embryos_S1/E11/adata_ATAC.h5ad"

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

echo "=== Vertical Debug Test Suite ==="
echo "Start: $(date)"

# Create debug data if needed
if [ ! -f "$ROOT/$HLN_RNA" ]; then
    echo "Creating debug data..."
    python "$ROOT/_myx_Scripts/horizontal_integration/_debug_all/create_debug_data.py"
fi

# Create output dirs
mkdir -p "$OUT"/{CANDIES,COSMOS,MISO,MultiGATE,PRAGA,PRESENT,SMOPCA,SpaBalance,SpaFusion,SpaMI,SpaMosaic,SpaMultiVAE,SpatialGlue,SWITCH}

# =============================================================================
# ADT methods — Human_Lymph_Nodes / A1 (cluster_nums=10, data_type=10x)
# =============================================================================

run_test "CANDIES x HLN/A1 (ADT)" \
    python $VS/CANDIES/run_candies.py \
        --data_type 10x \
        --RNA_path "$HLN_RNA" --ADT_path "$HLN_ADT" \
        --save_path "$OUT/CANDIES/CANDIES_HLN_A1_ADT.h5ad" \
        --method CANDIES --dataset HLN_A1 --cluster_nums 10 --device cuda:0

run_test "COSMOS x HLN/A1 (ADT)" \
    python $VS/COSMOS/run_cosmos.py \
        --data_type 10x \
        --RNA_path "$HLN_RNA" --ADT_path "$HLN_ADT" \
        --save_path "$OUT/COSMOS/COSMOS_HLN_A1_ADT.h5ad" \
        --method COSMOS --dataset HLN_A1 --cluster_nums 10 --device cuda:0

run_test "MISO x HLN/A1 (ADT)" \
    python $VS/MISO/run_MISO.py \
        --data_type 10x \
        --RNA_path "$HLN_RNA" --ADT_path "$HLN_ADT" \
        --save_path "$OUT/MISO/MISO_HLN_A1_ADT.h5ad" \
        --cluster_nums 10 --device cuda:0

run_test "MultiGATE x HLN/A1 (ADT)" \
    python $VS/MultiGATE/run_MultiGATE.py \
        --RNA_path "$HLN_RNA" --ADT_path "$HLN_ADT" \
        --save_path "$OUT/MultiGATE/MultiGATE_HLN_A1_ADT.h5ad" \
        --dataset HLN_A1 --cluster_nums 10

run_test "PRAGA x HLN/A1 (ADT)" \
    python $VS/PRAGA/run_praga.py \
        --data_type 10x \
        --RNA_path "$HLN_RNA" --ADT_path "$HLN_ADT" \
        --save_path "$OUT/PRAGA/PRAGA_HLN_A1_ADT.h5ad" \
        --method PRAGA --dataset HLN_A1 --cluster_nums 10 --device cuda:0

run_test "PRESENT x HLN/A1 (ADT)" \
    python $VS/PRESENT/run_present.py \
        --data_type 10x \
        --RNA_path "$HLN_RNA" --ADT_path "$HLN_ADT" \
        --save_path "$OUT/PRESENT/PRESENT_HLN_A1_ADT.h5ad" \
        --method PRESENT --dataset HLN_A1 --cluster_nums 10 --device cuda:0

run_test "SMOPCA x HLN/A1 (ADT)" \
    python $VS/SMOPCA/run_SMOPCA.py \
        --data_type 10x \
        --RNA_path "$HLN_RNA" --ADT_path "$HLN_ADT" \
        --save_path "$OUT/SMOPCA/SMOPCA_HLN_A1_ADT.h5ad" \
        --dataset HLN_A1 --cluster_nums 10 --device cpu

run_test "SpaBalance x HLN/A1 (ADT)" \
    python $VS/SpaBalance/run_SpaBalance.py \
        --data_type 10x \
        --RNA_path "$HLN_RNA" --ADT_path "$HLN_ADT" \
        --save_path "$OUT/SpaBalance/SpaBalance_HLN_A1_ADT.h5ad" \
        --method SpaBalance --dataset HLN_A1 --cluster_nums 10 --device cuda:0

run_test "SpaFusion x HLN/A1 (ADT)" \
    python $VS/SpaFusion/run_SpaFusion.py \
        --RNA_path "$HLN_RNA" --ADT_path "$HLN_ADT" \
        --save_path "$OUT/SpaFusion/SpaFusion_HLN_A1_ADT.h5ad" \
        --dataset debug_HLN_A1 --cluster_nums 10 --device cuda:0

run_test "SpaMI x HLN/A1 (ADT)" \
    python $VS/SpaMI/run_SpaMI.py \
        --data_type 10x \
        --RNA_path "$HLN_RNA" --ADT_path "$HLN_ADT" \
        --save_path "$OUT/SpaMI/SpaMI_HLN_A1_ADT.h5ad" \
        --dataset HLN_A1 --cluster_nums 10 --device cuda:0

run_test "SpaMosaic x HLN/A1 (ADT)" \
    python $VS/SpaMosaic/run_spamosaic.py \
        --data_type 10x \
        --RNA_path "$HLN_RNA" --ADT_path "$HLN_ADT" \
        --save_path "$OUT/SpaMosaic/SpaMosaic_HLN_A1_ADT.h5ad" \
        --method SpaMosaic --dataset HLN_A1 --cluster_nums 10 --device cuda:0

run_test "SpaMultiVAE x HLN/A1 (ADT)" \
    python $VS/SpaMultiVAE/run_spamultivae.py \
        --data_type 10x \
        --RNA_path "$HLN_RNA" --ADT_path "$HLN_ADT" \
        --save_path "$OUT/SpaMultiVAE/SpaMultiVAE_HLN_A1_ADT.h5ad" \
        --method SpaMultiVAE --dataset HLN_A1 --cluster_nums 10

run_test "SpatialGlue x HLN/A1 (ADT)" \
    python $VS/SpatialGlue/run_SpatialGlue.py \
        --data_type 10x \
        --RNA_path "$HLN_RNA" --ADT_path "$HLN_ADT" \
        --save_path "$OUT/SpatialGlue/SpatialGlue_HLN_A1_ADT.h5ad" \
        --method SpatialGlue --dataset HLN_A1 --cluster_nums 10 --device cuda:0

run_test "SWITCH x HLN/A1 (ADT)" \
    python $VS/SWITCH/run_SWITCH.py \
        --data_type 10x \
        --RNA_path "$HLN_RNA" --ADT_path "$HLN_ADT" \
        --save_path "$OUT/SWITCH/SWITCH_HLN_A1_ADT.h5ad" \
        --dataset HLN_A1 --cluster_nums 10 --device cuda:0

# NOTE: SpaMV tested in separate script (different conda env)
echo ""
echo ">>> SKIP: SpaMV x HLN/A1 (ADT) — see run_debug_spamv.sh"
SKIP=$((SKIP + 1))

# =============================================================================
# ATAC methods — Mouse_Embryos_S1 / E11 (cluster_nums=8, data_type=MISAR)
# Only methods that support ATAC
# =============================================================================

run_test "CANDIES x E11 (ATAC)" \
    python $VS/CANDIES/run_candies.py \
        --data_type MISAR \
        --RNA_path "$E11_RNA" --ATAC_path "$E11_ATAC" \
        --save_path "$OUT/CANDIES/CANDIES_E11_ATAC.h5ad" \
        --method CANDIES --dataset MISAR_E11 --cluster_nums 8 --device cuda:0

run_test "COSMOS x E11 (ATAC)" \
    python $VS/COSMOS/run_cosmos.py \
        --data_type MISAR \
        --RNA_path "$E11_RNA" --ATAC_path "$E11_ATAC" \
        --save_path "$OUT/COSMOS/COSMOS_E11_ATAC.h5ad" \
        --method COSMOS --dataset MISAR_E11 --cluster_nums 8 --device cuda:0

run_test "MISO x E11 (ATAC)" \
    python $VS/MISO/run_MISO.py \
        --data_type MISAR \
        --RNA_path "$E11_RNA" --ATAC_path "$E11_ATAC" \
        --save_path "$OUT/MISO/MISO_E11_ATAC.h5ad" \
        --cluster_nums 8 --device cuda:0

run_test "MultiGATE x E11 (ATAC)" \
    python $VS/MultiGATE/run_MultiGATE.py \
        --RNA_path "$E11_RNA" --ATAC_path "$E11_ATAC" \
        --save_path "$OUT/MultiGATE/MultiGATE_E11_ATAC.h5ad" \
        --dataset MISAR_E11 --cluster_nums 8

run_test "PRAGA x E11 (ATAC)" \
    python $VS/PRAGA/run_praga.py \
        --data_type MISAR \
        --RNA_path "$E11_RNA" --ATAC_path "$E11_ATAC" \
        --save_path "$OUT/PRAGA/PRAGA_E11_ATAC.h5ad" \
        --method PRAGA --dataset MISAR_E11 --cluster_nums 8 --device cuda:0

run_test "PRESENT x E11 (ATAC)" \
    python $VS/PRESENT/run_present.py \
        --data_type MISAR \
        --RNA_path "$E11_RNA" --ATAC_path "$E11_ATAC" \
        --save_path "$OUT/PRESENT/PRESENT_E11_ATAC.h5ad" \
        --method PRESENT --dataset MISAR_E11 --cluster_nums 8 --device cuda:0

run_test "SMOPCA x E11 (ATAC)" \
    python $VS/SMOPCA/run_SMOPCA.py \
        --data_type MISAR \
        --RNA_path "$E11_RNA" --ATAC_path "$E11_ATAC" \
        --save_path "$OUT/SMOPCA/SMOPCA_E11_ATAC.h5ad" \
        --dataset MISAR_E11 --cluster_nums 8 --device cpu

run_test "SpaBalance x E11 (ATAC)" \
    python $VS/SpaBalance/run_SpaBalance.py \
        --data_type MISAR \
        --RNA_path "$E11_RNA" --ATAC_path "$E11_ATAC" \
        --save_path "$OUT/SpaBalance/SpaBalance_E11_ATAC.h5ad" \
        --method SpaBalance --dataset MISAR_E11 --cluster_nums 8 --device cuda:0

run_test "SpaMI x E11 (ATAC)" \
    python $VS/SpaMI/run_SpaMI.py \
        --data_type MISAR \
        --RNA_path "$E11_RNA" --ATAC_path "$E11_ATAC" \
        --save_path "$OUT/SpaMI/SpaMI_E11_ATAC.h5ad" \
        --dataset MISAR_E11 --cluster_nums 8 --device cuda:0

run_test "SpaMosaic x E11 (ATAC)" \
    python $VS/SpaMosaic/run_spamosaic.py \
        --data_type MISAR \
        --RNA_path "$E11_RNA" --ATAC_path "$E11_ATAC" \
        --save_path "$OUT/SpaMosaic/SpaMosaic_E11_ATAC.h5ad" \
        --method SpaMosaic --dataset MISAR_E11 --cluster_nums 8 --device cuda:0

run_test "SpatialGlue x E11 (ATAC)" \
    python $VS/SpatialGlue/run_SpatialGlue.py \
        --data_type MISAR \
        --RNA_path "$E11_RNA" --ATAC_path "$E11_ATAC" \
        --save_path "$OUT/SpatialGlue/SpatialGlue_E11_ATAC.h5ad" \
        --method SpatialGlue --dataset MISAR_E11 --cluster_nums 8 --device cuda:0

run_test "SWITCH x E11 (ATAC)" \
    python $VS/SWITCH/run_SWITCH.py \
        --data_type MISAR \
        --RNA_path "$E11_RNA" --ATAC_path "$E11_ATAC" \
        --save_path "$OUT/SWITCH/SWITCH_E11_ATAC.h5ad" \
        --dataset MISAR_E11 --cluster_nums 8 --device cuda:0

run_test "SpaFusion x E11 (ATAC)" \
    python $VS/SpaFusion/run_SpaFusion.py \
        --RNA_path "$E11_RNA" --ATAC_path "$E11_ATAC" \
        --save_path "$OUT/SpaFusion/SpaFusion_E11_ATAC.h5ad" \
        --dataset debug_MISAR_E11 --cluster_nums 8 --device cuda:0

# NOTE: SpaMultiVAE does NOT support ATAC — skip
echo ""
echo ">>> SKIP: SpaMultiVAE x E11 (ATAC) — SpaMultiVAE only supports ADT"
SKIP=$((SKIP + 1))

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "=== VERTICAL DEBUG SUMMARY ==="
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
