#!/bin/bash
# Run a single SMObench method inside the Singularity/Apptainer container
#
# Usage:
#   bash run_method.sh --method SpatialGlue \
#     --rna_path /path/to/RNA.h5ad --adt_path /path/to/ADT.h5ad \
#     --save_path /path/to/output.h5ad --cluster_nums 4 \
#     [--integrated_path /path/to/adata_integrated.h5ad] \
#     [--data_type 10x] [--gpu 0] [--task vertical]
#
# Requires: module load apptainer/1.2.2 (or singularity/4.3.1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(dirname "${SCRIPT_DIR}")/benchmark"
SIF="${SCRIPT_DIR}/images/smobench_full.sif"

# Auto-detect container runtime
if command -v apptainer &>/dev/null; then
    RUNTIME=apptainer
elif command -v singularity &>/dev/null; then
    RUNTIME=singularity
else
    echo "ERROR: neither apptainer nor singularity found. Run: module load apptainer/1.2.2"
    exit 1
fi

# ---- Parse arguments ----
METHOD="" RNA_PATH="" ADT_PATH="" ATAC_PATH="" SAVE_PATH=""
INTEGRATED_PATH="" DATA_TYPE="10x" CLUSTER_NUMS=4 GPU_ID="0" TASK="vertical"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)          METHOD="$2"; shift 2 ;;
        --rna_path)        RNA_PATH="$2"; shift 2 ;;
        --adt_path)        ADT_PATH="$2"; shift 2 ;;
        --atac_path)       ATAC_PATH="$2"; shift 2 ;;
        --save_path)       SAVE_PATH="$2"; shift 2 ;;
        --integrated_path) INTEGRATED_PATH="$2"; shift 2 ;;
        --data_type)       DATA_TYPE="$2"; shift 2 ;;
        --cluster_nums)    CLUSTER_NUMS="$2"; shift 2 ;;
        --gpu)             GPU_ID="$2"; shift 2 ;;
        --task)            TASK="$2"; shift 2 ;;
        *)                 EXTRA_ARGS="${EXTRA_ARGS} $1"; shift ;;
    esac
done

[[ -n "$METHOD" && -n "$RNA_PATH" && -n "$SAVE_PATH" ]] || {
    echo "ERROR: --method, --rna_path, and --save_path are required"; exit 1;
}
[[ -f "$SIF" ]] || { echo "ERROR: Image not found: $SIF"; exit 1; }

# ---- Map method to script ----
declare -A SCRIPT_MAP=(
    [SpatialGlue]="SpatialGlue/run_SpatialGlue.py"
    [SpaMosaic]="SpaMosaic/run_spamosaic.py"
    [PRAGA]="PRAGA/run_praga.py"
    [COSMOS]="COSMOS/run_cosmos.py"
    [PRESENT]="PRESENT/run_present.py"
    [CANDIES]="CANDIES/run_candies.py"
    [MISO]="MISO/run_MISO.py"
    [MultiGATE]="MultiGATE/run_MultiGATE.py"
    [SMOPCA]="SMOPCA/run_SMOPCA.py"
    [SpaBalance]="SpaBalance/run_SpaBalance.py"
    [SpaFusion]="SpaFusion/run_SpaFusion.py"
    [SpaMI]="SpaMI/run_SpaMI.py"
    [spaMultiVAE]="SpaMultiVAE/run_spamultivae.py"
    [SpaMV]="SpaMV/run_spamv.py"
    [SWITCH]="SWITCH/run_SWITCH.py"
)

RUN_SCRIPT="${SCRIPT_MAP[$METHOD]:-}"
[[ -n "$RUN_SCRIPT" ]] || { echo "ERROR: Unknown method '$METHOD'"; exit 1; }

FULL_SCRIPT="/benchmark/Scripts/${TASK}_integration/${RUN_SCRIPT}"

# ---- Select python binary based on method ----
declare -A PYTHON_MAP=(
    [MultiGATE]="/opt/env_tf/bin/python"
    [PRESENT]="/opt/env_bio/bin/python"
    [SpaMV]="/opt/env_bio/bin/python"
)
PYTHON_BIN="${PYTHON_MAP[$METHOD]:-python}"

# ---- Build python command ----
PY_CMD="${PYTHON_BIN} ${FULL_SCRIPT} --data_type ${DATA_TYPE} --RNA_path ${RNA_PATH} --save_path ${SAVE_PATH} --cluster_nums ${CLUSTER_NUMS}"
[[ -n "$ADT_PATH" ]]        && PY_CMD="${PY_CMD} --ADT_path ${ADT_PATH}"
[[ -n "$ATAC_PATH" ]]       && PY_CMD="${PY_CMD} --ATAC_path ${ATAC_PATH}"
[[ -n "$INTEGRATED_PATH" ]] && PY_CMD="${PY_CMD} --integrated_path ${INTEGRATED_PATH}"
[[ -n "$EXTRA_ARGS" ]]      && PY_CMD="${PY_CMD} ${EXTRA_ARGS}"

# ---- Build bind mounts ----
BINDS="${BENCH_DIR}:/benchmark"

# Bind data directories (read-only)
for p in "$RNA_PATH" "$ADT_PATH" "$ATAC_PATH"; do
    [[ -n "$p" && -f "$p" ]] && BINDS="${BINDS},$(dirname $(readlink -f $p))"
done

# Bind save directory (read-write)
mkdir -p "$(dirname ${SAVE_PATH})"
BINDS="${BINDS},$(dirname $(readlink -f ${SAVE_PATH}))"

# Bind integrated path directory
if [[ -n "$INTEGRATED_PATH" ]]; then
    mkdir -p "$(dirname ${INTEGRATED_PATH})"
    BINDS="${BINDS},$(dirname $(readlink -f ${INTEGRATED_PATH}))"
fi

# ---- Run ----
echo "=========================================="
echo "SMObench: ${METHOD} (${TASK})"
echo "=========================================="
echo "Runtime: ${RUNTIME}"
echo "GPU:     ${GPU_ID}"
echo "RNA:     ${RNA_PATH}"
echo "Save:    ${SAVE_PATH}"
[[ -n "$INTEGRATED_PATH" ]] && echo "Integrated: ${INTEGRATED_PATH}"
echo "=========================================="

# ---- Load host R for mclust (if available) ----
R_BIND=""
R_ENV=""
if module load cray-R/4.1.2.0 2>/dev/null; then
    R_BIN=$(dirname $(which Rscript 2>/dev/null))
    R_LIB=$(Rscript -e 'cat(.libPaths()[1])' 2>/dev/null)
    if [[ -n "$R_BIN" ]]; then
        R_BIND=",${R_BIN}"
        [[ -d "${R_LIB}" ]] && R_BIND="${R_BIND},${R_LIB}"
        [[ -d "${HOME}/R/libs" ]] && R_BIND="${R_BIND},${HOME}/R/libs"
        R_ENV="--env R_LIBS_USER=${HOME}/R/libs --env PATH=${R_BIN}:\$PATH"
        echo "R:       ${R_BIN} (host module)"
    fi
fi

CUDA_VISIBLE_DEVICES=${GPU_ID} ${RUNTIME} exec --nv \
    --bind "${BINDS}${R_BIND}" \
    --env NUMBA_CACHE_DIR=/tmp/numba_cache \
    --env MPLCONFIGDIR=/tmp/mpl_cache \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env R_LIBS_USER=${HOME}/R/libs \
    --env "PATH=${R_BIN:-/usr/bin}:/opt/conda/bin:/usr/local/bin:/usr/bin:/bin" \
    "${SIF}" \
    ${PY_CMD}
