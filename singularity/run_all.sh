#!/bin/bash
# ============================================================================
# run_all.sh - Run all/selected SMObench methods in parallel via Apptainer
#
# Architecture:
#   1 shared SIF (PyTorch+CUDA) + per-method pip prefix (PYTHONPATH isolation)
#
# Usage:
#   bash run_all.sh --task vertical --data_type RNA_ADT --dataset Human_Tonsils --slice S1
#   bash run_all.sh --task vertical --data_type RNA_ADT --dataset Human_Tonsils --slice S1 --methods "SpatialGlue,PRAGA"
#   bash run_all.sh --dry_run --task vertical --data_type RNA_ADT --dataset Human_Tonsils --slice S1
# ============================================================================

set -euo pipefail

source /etc/profile.d/modules.sh 2>/dev/null || true
module load singularity/4.3.1 2>/dev/null

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(dirname "$SCRIPT_DIR")/benchmark"
SIF="${SCRIPT_DIR}/images/pytorch_2.1.2-cuda11.8-cudnn8-runtime.sif"
ENV_DIR="${SCRIPT_DIR}/envs"
LOG_DIR="${SCRIPT_DIR}/logs/run"
RESULT_DIR="${BENCH_DIR}/Results/adata"

export SINGULARITY_CACHEDIR="${SCRIPT_DIR}/.cache"
export SINGULARITY_TMPDIR="${SCRIPT_DIR}/.tmp"

# Defaults
TASK=""
DATA_TYPE=""
DATASET=""
SLICE=""
MAX_PARALLEL=4
GPU_ID=0
DRY_RUN=false
CLUSTER_NUMS=""

ALL_METHODS=(
    SpatialGlue SpaMosaic PRAGA COSMOS PRESENT CANDIES
    MISO SMOPCA SpaBalance SpaFusion SpaMI spaMultiVAE
    SpaMV SWITCH MultiGATE
)
SELECTED_METHODS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --task)          TASK="$2"; shift 2 ;;
        --data_type)     DATA_TYPE="$2"; shift 2 ;;
        --dataset)       DATASET="$2"; shift 2 ;;
        --slice)         SLICE="$2"; shift 2 ;;
        --methods)       IFS=',' read -ra SELECTED_METHODS <<< "$2"; shift 2 ;;
        --max_parallel)  MAX_PARALLEL="$2"; shift 2 ;;
        --gpu)           GPU_ID="$2"; shift 2 ;;
        --cluster_nums)  CLUSTER_NUMS="$2"; shift 2 ;;
        --dry_run)       DRY_RUN=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

PLATFORM_TYPE=""

if [[ -z "$TASK" || -z "$DATA_TYPE" || -z "$DATASET" ]]; then
    echo "Usage: bash run_all.sh --task vertical --data_type 10x --dataset Human_Tonsils --slice S1"
    echo "  --data_type: 10x, SPOTS, Stereo-CITE-seq, MISAR, simulation"
    exit 1
fi

# DATA_TYPE is now the platform type passed directly to method scripts
PLATFORM_TYPE="$DATA_TYPE"

# Map platform type to directory structure for data paths
case "$PLATFORM_TYPE" in
    10x|SPOTS|Stereo-CITE-seq) DATA_SUBDIR="RNA_ADT" ;;
    MISAR)                      DATA_SUBDIR="RNA_ATAC" ;;
    simulation)                 DATA_SUBDIR="3M" ;;
    *)                          DATA_SUBDIR="$PLATFORM_TYPE" ;;
esac
[[ ${#SELECTED_METHODS[@]} -eq 0 ]] && SELECTED_METHODS=("${ALL_METHODS[@]}")

mkdir -p "$LOG_DIR"

echo "============================================"
echo " SMObench Parallel Runner (Apptainer)"
echo "============================================"
echo " Task:      ${TASK}"
echo " Data:      ${DATA_TYPE} / ${DATASET} / ${SLICE:-all}"
echo " Methods:   ${SELECTED_METHODS[*]}"
echo " Parallel:  ${MAX_PARALLEL}"
echo " GPU:       ${GPU_ID}"
echo "============================================"

# ---- Run one method ----
run_method() {
    local method="$1"
    local log="${LOG_DIR}/${method}_${DATASET}_${SLICE:-all}_$(date +%Y%m%d_%H%M%S).log"

    # Find run script (handle case-insensitive naming)
    local script_dir="${BENCH_DIR}/Scripts/${TASK}_integration/${method}"
    # Also try capitalized dir name for spaMultiVAE -> SpaMultiVAE
    if [[ ! -d "$script_dir" ]]; then
        # Try common alternatives
        for alt in "$(echo "${method}" | sed 's/^spa/Spa/')" "$(echo "${method}" | tr '[:lower:]' '[:upper:]')"; do
            [[ -d "${BENCH_DIR}/Scripts/${TASK}_integration/${alt}" ]] && script_dir="${BENCH_DIR}/Scripts/${TASK}_integration/${alt}" && break
        done
    fi
    # Find the run script: try run_{method}.py, run_{lowercase}.py
    local run_script=""
    local method_lower="$(echo "$method" | tr '[:upper:]' '[:lower:]')"
    for candidate in \
        "${script_dir}/run_${method}.py" \
        "${script_dir}/run_${method_lower}.py"; do
        if [[ -f "$candidate" ]]; then
            run_script="$candidate"
            break
        fi
    done
    if [[ -z "$run_script" ]]; then
        echo "[SKIP] ${method}: script not found in ${script_dir}/"
        return 1
    fi
    local run_script_rel="${run_script#${BENCH_DIR}/}"

    # Build PYTHONPATH: method-specific env + base env
    local pypath="${ENV_DIR}/${method}:${ENV_DIR}/base"

    # Build save path
    local save_dir="${RESULT_DIR}/${method}/${DATASET}/${SLICE:-}"
    mkdir -p "$save_dir"
    local save_name="${method}_${DATASET}_${SLICE:-all}.h5ad"

    # Build singularity command
    local cmd="CUDA_VISIBLE_DEVICES=${GPU_ID} singularity exec --nv"
    cmd+=" --bind ${BENCH_DIR}:/bench"
    cmd+=" --bind ${ENV_DIR}:/envs"
    cmd+=" ${SIF}"
    cmd+=" bash -c '"
    cmd+="export PYTHONPATH=/envs/${method}:/envs/base:\${PYTHONPATH:-}; "
    cmd+="cd /bench; "
    cmd+="python ${run_script_rel}"

    # Only pass --data_type if the script accepts it (check argparse)
    local NO_DATATYPE_METHODS="SpaFusion"
    if [[ ! " $NO_DATATYPE_METHODS " =~ " $method " ]]; then
        cmd+=" --data_type ${PLATFORM_TYPE}"
    fi

    # Data paths based on directory structure
    if [[ "$DATA_SUBDIR" == "RNA_ADT" ]]; then
        cmd+=" --RNA_path /bench/Dataset/withGT/${DATA_SUBDIR}/${DATASET}/${SLICE}/adata_RNA.h5ad"
        cmd+=" --ADT_path /bench/Dataset/withGT/${DATA_SUBDIR}/${DATASET}/${SLICE}/adata_ADT.h5ad"
    elif [[ "$DATA_SUBDIR" == "RNA_ATAC" ]]; then
        cmd+=" --RNA_path /bench/Dataset/withGT/${DATA_SUBDIR}/${DATASET}/${SLICE}/adata_RNA.h5ad"
        cmd+=" --ATAC_path /bench/Dataset/withGT/${DATA_SUBDIR}/${DATASET}/${SLICE}/adata_ATAC.h5ad"
    elif [[ "$DATA_SUBDIR" == "3M" ]]; then
        cmd+=" --RNA_path /bench/Dataset/withGT/${DATA_SUBDIR}/${DATASET}/adata_RNA.h5ad"
        cmd+=" --ADT_path /bench/Dataset/withGT/${DATA_SUBDIR}/${DATASET}/adata_ADT.h5ad"
        cmd+=" --ATAC_path /bench/Dataset/withGT/${DATA_SUBDIR}/${DATASET}/adata_ATAC.h5ad"
    fi

    cmd+=" --save_path /bench/Results/adata/${method}/${DATASET}/${SLICE:-}/${save_name}"
    [[ -n "$CLUSTER_NUMS" ]] && cmd+=" --cluster_nums ${CLUSTER_NUMS}"
    cmd+="'"

    if $DRY_RUN; then
        echo "[DRY] $cmd"
        return 0
    fi

    echo "[RUN] ${method} @ $(date '+%H:%M:%S')"
    eval "$cmd" > "$log" 2>&1
    local rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "[OK]  ${method} @ $(date '+%H:%M:%S')"
    else
        echo "[FAIL] ${method} (rc=$rc) -> $log"
    fi
    return $rc
}

# ---- Parallel job control ----
PIDS=()
MNAMES=()
SUCCESS=()
FAILED=()

reap_finished() {
    local new_pids=() new_names=()
    for i in "${!PIDS[@]}"; do
        if kill -0 "${PIDS[$i]}" 2>/dev/null; then
            new_pids+=("${PIDS[$i]}")
            new_names+=("${MNAMES[$i]}")
        else
            wait "${PIDS[$i]}" 2>/dev/null && SUCCESS+=("${MNAMES[$i]}") || FAILED+=("${MNAMES[$i]}")
        fi
    done
    PIDS=("${new_pids[@]+"${new_pids[@]}"}")
    MNAMES=("${new_names[@]+"${new_names[@]}"}")
}

for method in "${SELECTED_METHODS[@]}"; do
    # Wait for a slot
    while [[ ${#PIDS[@]} -ge $MAX_PARALLEL ]]; do
        reap_finished
        [[ ${#PIDS[@]} -ge $MAX_PARALLEL ]] && sleep 3
    done
    run_method "$method" &
    PIDS+=($!)
    MNAMES+=("$method")
done

# Wait for all remaining
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" 2>/dev/null && SUCCESS+=("${MNAMES[$i]}") || FAILED+=("${MNAMES[$i]}")
done

echo ""
echo "============================================"
echo " Done: ${#SUCCESS[@]} ok, ${#FAILED[@]} failed"
echo " OK:   ${SUCCESS[*]:-none}"
echo " FAIL: ${FAILED[*]:-none}"
echo " Logs: ${LOG_DIR}/"
echo "============================================"
