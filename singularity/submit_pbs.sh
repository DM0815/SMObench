#!/bin/bash
# Submit SMObench benchmark as a PBS job
#
# Usage:
#   qsub submit_pbs.sh
#   qsub -v METHOD=SpatialGlue submit_pbs.sh     # single method
#   qsub -v TASK=horizontal submit_pbs.sh          # horizontal integration

#PBS -N smobench
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1
#PBS -l walltime=12:00:00
#PBS -j oe

module load apptainer/1.2.2

cd "${PBS_O_WORKDIR:-$(dirname "${BASH_SOURCE[0]}")}"
SCRIPT_DIR="$(pwd)"
BENCH="$(dirname ${SCRIPT_DIR})/benchmark"
SIF="${SCRIPT_DIR}/images/smobench_full.sif"

# Defaults (override with qsub -v)
METHOD="${METHOD:-all}"
TASK="${TASK:-vertical}"
DATASET="${DATASET:-all}"
CLUSTERING="${CLUSTERING:-leiden,kmeans}"
GPU_ID="0"

echo "=========================================="
echo "SMObench PBS Job"
echo "Task:     ${TASK}"
echo "Method:   ${METHOD}"
echo "Dataset:  ${DATASET}"
echo "=========================================="

if [[ "$METHOD" == "all" ]]; then
    # Run via smobench CLI inside container
    CUDA_VISIBLE_DEVICES=${GPU_ID} apptainer exec --nv \
        --bind ${BENCH}:/benchmark \
        --env NUMBA_CACHE_DIR=/tmp/numba_cache \
        --env MPLCONFIGDIR=/tmp/mpl_cache \
        --env SMOBENCH_DATA_ROOT=/benchmark/Dataset \
        --env PYTHONPATH=/benchmark/src \
        ${SIF} python -m smobench.cli run \
        --task ${TASK} \
        --dataset ${DATASET} \
        --methods all \
        --clustering ${CLUSTERING} \
        --output /benchmark/Results/smobench_${TASK}_results.csv
else
    # Run single method
    bash ${SCRIPT_DIR}/run_method.sh \
        --method ${METHOD} \
        --task ${TASK} \
        --rna_path ${BENCH}/Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/A1/adata_RNA.h5ad \
        --adt_path ${BENCH}/Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/A1/adata_ADT.h5ad \
        --save_path ${BENCH}/Results/adata/${TASK}_integration/${METHOD}/test.h5ad \
        --cluster_nums 10 \
        --gpu ${GPU_ID}
fi
