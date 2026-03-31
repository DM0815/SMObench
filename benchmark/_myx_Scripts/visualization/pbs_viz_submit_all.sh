#!/bin/bash
# Submit 7 PBS jobs (one per dataset) for visualization generation
# Each job generates horizontal + vertical PDFs for that dataset

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

declare -A SLICES=(
    [Human_Lymph_Nodes]="A1 D1"
    [Human_Tonsils]="S1 S2 S3"
    [Mouse_Embryos_S1]="E11 E13 E15 E18"
    [Mouse_Embryos_S2]="E11 E13 E15 E18"
    [Mouse_Spleen]="Spleen1 Spleen2"
    [Mouse_Thymus]="Thymus1 Thymus2 Thymus3 Thymus4"
    [Mouse_Brain]="ATAC H3K27ac H3K27me3"
)

# HLN horizontal already done, skip it
declare -A SKIP_HORIZONTAL=(
    [Human_Lymph_Nodes]=1
)

for DS in Human_Lymph_Nodes Human_Tonsils Mouse_Embryos_S1 Mouse_Embryos_S2 Mouse_Spleen Mouse_Thymus Mouse_Brain; do
    SL_LIST="${SLICES[$DS]}"
    SKIP_H="${SKIP_HORIZONTAL[$DS]:-0}"

    PBS_FILE="$SCRIPT_DIR/pbs_viz_${DS}.pbs"
    LOG="/scratch/users/nus/e1724738/SMOBench_manuscript/viz_${DS}.log"

    cat > "$PBS_FILE" << PBSEOF
#!/bin/bash
#PBS -N viz_${DS}
#PBS -l select=1:ncpus=4:mem=64gb
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -o /dev/null
#PBS -P 11003054
#PBS -q normal

cd /data/projects/11003054/e1724738/_private/NUS/_Proj1/SMOBench-CLEAN
mkdir -p "$(dirname "$LOG")"
exec > "$LOG" 2>&1
export PYTHONUNBUFFERED=1
source /home/users/nus/e1724738/miniconda3/etc/profile.d/conda.sh
conda activate _Proj1_1

ROOT=/data/projects/11003054/e1724738/_private/NUS/_Proj1/SMOBench-CLEAN
OUTDIR=/home/users/nus/e1724738/_main/_private/NUS/_Proj1/writing/picture/5_visualization
SCRIPTS=\$ROOT/_myx_Scripts/visualization
PLOTS=\$ROOT/_myx_Results/plots
mkdir -p "\$OUTDIR"

DS="${DS}"
echo "=========================================="
echo "Dataset: \$DS"
echo "Start: \$(date)"
echo "=========================================="

# --- Horizontal ---
PBSEOF

    if [ "$SKIP_H" = "1" ]; then
        cat >> "$PBS_FILE" << 'PBSEOF'
echo "Horizontal: skipped (already exists)"
PBSEOF
    else
        cat >> "$PBS_FILE" << PBSEOF
echo "--- Horizontal ---"
python -u \$SCRIPTS/plot_fig3de_batch_umap.py --root \$ROOT --dataset \$DS
cp "\$PLOTS/fig3ef_batch_umap_\${DS}_leiden.pdf" "\$OUTDIR/horizontal_batch_\${DS}.pdf"
echo "Copied -> horizontal_batch_\${DS}.pdf"
PBSEOF
    fi

    cat >> "$PBS_FILE" << PBSEOF

# --- Vertical (per-slice) ---
for SL in ${SL_LIST}; do
    echo ""
    echo "--- Vertical: \$DS / \$SL ---"
    echo "\$(date)"
    python -u \$SCRIPTS/plot_fig2de_fig3de_umap_grid_v2.py --root \$ROOT --task vertical --dataset \$DS --slice \$SL
    cp "\$PLOTS/fig2de_umap_spatial_\${DS}_leiden.pdf" "\$OUTDIR/vertical_\${DS}_\${SL}.pdf"
    echo "Copied -> vertical_\${DS}_\${SL}.pdf"
done

echo ""
echo "=========================================="
echo "DONE \$DS: \$(date)"
echo "=========================================="
PBSEOF

    echo "Submitting: $DS"
    qsub "$PBS_FILE"
done
