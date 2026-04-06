#!/bin/bash
# ============================================================================
# run_all_datasets.sh - Run ALL methods on ALL datasets (full benchmark)
#
# Usage:
#   bash run_all_datasets.sh [max_parallel] [gpu_id]
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAX_PARALLEL="${1:-4}"
GPU_ID="${2:-0}"

echo "============================================================"
echo "SMObench Full Benchmark Run"
echo "Max parallel: ${MAX_PARALLEL}  |  GPU: ${GPU_ID}"
echo "Start time: $(date)"
echo "============================================================"

run() {
    bash "${SCRIPT_DIR}/run_all.sh" "$@" --max_parallel "$MAX_PARALLEL" --gpu "$GPU_ID"
}

# ---- Vertical Integration: 10x (RNA+ADT) ----
echo ""
echo ">>> Vertical: 10x RNA+ADT <<<"

for slice in A1 D1; do
    echo "--- Human_Lymph_Nodes / ${slice} ---"
    run --task vertical --data_type 10x --dataset Human_Lymph_Nodes --slice "$slice"
done

for slice in S1 S2 S3; do
    echo "--- Human_Tonsils / ${slice} ---"
    run --task vertical --data_type 10x --dataset Human_Tonsils --slice "$slice"
done

# ---- Vertical Integration: MISAR (RNA+ATAC) ----
echo ""
echo ">>> Vertical: MISAR RNA+ATAC <<<"

for slice in MISAR_S1 MISAR_S2; do
    echo "--- Mouse_Embryos / ${slice} ---"
    run --task vertical --data_type MISAR --dataset Mouse_Embryos --slice "$slice"
done

# ---- Mosaic Integration: 3M Simulation ----
echo ""
echo ">>> Mosaic: 3M Simulation <<<"
run --task mosaic --data_type simulation --dataset 3M_Simulation

# ---- Horizontal Integration ----
echo ""
echo ">>> Horizontal Integration <<<"

for dataset in Human_Tonsils Mouse_Spleen Mouse_Thymus; do
    echo "--- Horizontal: ${dataset} ---"
    run --task horizontal --data_type 10x --dataset "$dataset"
done

echo ""
echo "============================================================"
echo "Full benchmark complete at $(date)"
echo "============================================================"
