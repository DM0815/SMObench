#!/bin/bash
# ============================================================================
# setup_envs.sh - Install per-method Python dependencies into isolated prefixes
#
# Strategy: One shared SIF image (PyTorch+CUDA) + per-method pip prefix dirs
# No root/fakeroot needed - all installed to project disk
#
# Usage:
#   bash setup_envs.sh                    # install all methods
#   bash setup_envs.sh SpatialGlue        # install one method
#   bash setup_envs.sh base               # install shared deps only
# ============================================================================

set -euo pipefail

source /etc/profile.d/modules.sh 2>/dev/null || true
module load singularity/4.3.1 2>/dev/null

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIF="${SCRIPT_DIR}/images/pytorch_2.1.2-cuda11.8-cudnn8-runtime.sif"
ENV_DIR="${SCRIPT_DIR}/envs"
LOG_DIR="${SCRIPT_DIR}/logs/setup"

export SINGULARITY_CACHEDIR="${SCRIPT_DIR}/.cache"
export SINGULARITY_TMPDIR="${SCRIPT_DIR}/.tmp"
mkdir -p "$ENV_DIR" "$LOG_DIR" "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

if [[ ! -f "$SIF" ]]; then
    echo "Error: Base image not found: $SIF"
    echo "Run: apptainer pull --dir ${SCRIPT_DIR}/images docker://pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime"
    exit 1
fi

# Helper: install packages into a prefix dir using the SIF container
pip_install() {
    local prefix="$1"
    shift
    mkdir -p "$prefix"
    singularity exec \
        --bind "${ENV_DIR}:/envs" \
        --bind "${SCRIPT_DIR}:/sing" \
        "$SIF" \
        pip install --target="$prefix" --no-warn-script-location "$@"
}

# ---- Base/shared dependencies ----
install_base() {
    echo "=== Installing base (shared) dependencies ==="
    local prefix="${ENV_DIR}/base"
    pip_install "$prefix" \
        numpy==1.24.4 \
        pandas==2.0.3 \
        scipy==1.10.1 \
        scikit-learn==1.3.2 \
        matplotlib==3.7.5 \
        seaborn==0.12.2 \
        h5py==3.10.0 \
        scanpy==1.9.8 \
        anndata==0.9.2 \
        episcanpy==0.4.0 \
        squidpy==1.6.2 \
        tqdm==4.66.2 \
        networkx==3.1 \
        umap-learn \
        leidenalg==0.9.1 \
        louvain==0.8.0 \
        plotly \
        PyYAML \
        scikit-misc==0.2.0 \
        2>&1 | tee "${LOG_DIR}/base.log" | tail -3
    echo "[OK] base"
}

# ---- Per-method dependencies ----
# Each method gets: base + its own extras

install_SpatialGlue() {
    echo "=== Installing SpatialGlue ==="
    local prefix="${ENV_DIR}/SpatialGlue"
    pip_install "$prefix" \
        torch-geometric==2.3.1 \
        torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
        torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
        2>&1 | tee "${LOG_DIR}/SpatialGlue.log" | tail -3
    echo "[OK] SpatialGlue"
}

install_SpaMosaic() {
    echo "=== Installing SpaMosaic ==="
    local prefix="${ENV_DIR}/SpaMosaic"
    pip_install "$prefix" \
        torch-geometric==2.3.1 \
        torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
        torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
        dgl -f https://data.dgl.ai/wheels/cu118/repo.html \
        hnswlib \
        annoy \
        2>&1 | tee "${LOG_DIR}/SpaMosaic.log" | tail -3
    echo "[OK] SpaMosaic"
}

install_PRAGA() {
    echo "=== Installing PRAGA ==="
    local prefix="${ENV_DIR}/PRAGA"
    pip_install "$prefix" \
        torch-geometric==2.3.1 \
        torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
        torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
        scvi-tools \
        2>&1 | tee "${LOG_DIR}/PRAGA.log" | tail -3
    echo "[OK] PRAGA"
}

install_COSMOS() {
    echo "=== Installing COSMOS ==="
    local prefix="${ENV_DIR}/COSMOS"
    pip_install "$prefix" \
        torch-geometric==2.3.1 \
        torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
        torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
        gudhi \
        cmcrameri \
        2>&1 | tee "${LOG_DIR}/COSMOS.log" | tail -3
    echo "[OK] COSMOS"
}

install_PRESENT() {
    echo "=== Installing PRESENT ==="
    local prefix="${ENV_DIR}/PRESENT"
    pip_install "$prefix" \
        torch-geometric==2.3.1 \
        torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
        torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
        genomicranges==0.4.2 \
        iranges==0.2.1 \
        biocutils==0.1.3 \
        2>&1 | tee "${LOG_DIR}/PRESENT.log" | tail -3
    echo "[OK] PRESENT"
}

install_CANDIES() {
    echo "=== Installing CANDIES ==="
    local prefix="${ENV_DIR}/CANDIES"
    pip_install "$prefix" \
        torch-geometric==2.6.1 \
        torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
        torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
        einops==0.8.0 \
        gudhi \
        scikit-image \
        gseapy==1.1.6 \
        esda \
        2>&1 | tee "${LOG_DIR}/CANDIES.log" | tail -3
    echo "[OK] CANDIES"
}

install_MISO() {
    echo "=== Installing MISO ==="
    local prefix="${ENV_DIR}/MISO"
    pip_install "$prefix" \
        einops==0.6.0 \
        opencv-python-headless \
        scikit-image \
        Pillow \
        2>&1 | tee "${LOG_DIR}/MISO.log" | tail -3
    echo "[OK] MISO"
}

install_SMOPCA() {
    echo "=== Installing SMOPCA ==="
    # Only needs base deps
    mkdir -p "${ENV_DIR}/SMOPCA"
    echo "[OK] SMOPCA (base deps only)"
}

install_SpaBalance() {
    echo "=== Installing SpaBalance ==="
    local prefix="${ENV_DIR}/SpaBalance"
    pip_install "$prefix" \
        torch-geometric==2.6.1 \
        torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
        torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
        2>&1 | tee "${LOG_DIR}/SpaBalance.log" | tail -3
    echo "[OK] SpaBalance"
}

install_SpaFusion() {
    echo "=== Installing SpaFusion ==="
    local prefix="${ENV_DIR}/SpaFusion"
    pip_install "$prefix" \
        torch-geometric==2.3.1 \
        torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
        torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
        2>&1 | tee "${LOG_DIR}/SpaFusion.log" | tail -3
    echo "[OK] SpaFusion"
}

install_SpaMI() {
    echo "=== Installing SpaMI ==="
    local prefix="${ENV_DIR}/SpaMI"
    pip_install "$prefix" \
        torch-geometric==2.5.1 \
        torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
        torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
        2>&1 | tee "${LOG_DIR}/SpaMI.log" | tail -3
    echo "[OK] SpaMI"
}

install_spaMultiVAE() {
    echo "=== Installing spaMultiVAE ==="
    # Only needs base deps
    mkdir -p "${ENV_DIR}/spaMultiVAE"
    echo "[OK] spaMultiVAE (base deps only)"
}

install_SpaMV() {
    echo "=== Installing SpaMV ==="
    local prefix="${ENV_DIR}/SpaMV"
    pip_install "$prefix" \
        torch-geometric==2.3.1 \
        torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
        torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
        pyro-ppl \
        2>&1 | tee "${LOG_DIR}/SpaMV.log" | tail -3
    echo "[OK] SpaMV"
}

install_SWITCH() {
    echo "=== Installing SWITCH ==="
    local prefix="${ENV_DIR}/SWITCH"
    pip_install "$prefix" \
        torch-geometric==2.6.1 \
        torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
        torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
        pybedtools==0.9.1 \
        pynvml \
        2>&1 | tee "${LOG_DIR}/SWITCH.log" | tail -3
    echo "[OK] SWITCH"
}

install_MultiGATE() {
    echo "=== Installing MultiGATE ==="
    local prefix="${ENV_DIR}/MultiGATE"
    # MultiGATE needs TensorFlow - install alongside PyTorch
    pip_install "$prefix" \
        tensorflow==2.12.0 \
        pybedtools==0.9.1 \
        pynvml \
        gseapy \
        2>&1 | tee "${LOG_DIR}/MultiGATE.log" | tail -3
    echo "[OK] MultiGATE"
}

# ---- Main ----
ALL_METHODS=(
    SpatialGlue SpaMosaic PRAGA COSMOS PRESENT CANDIES
    MISO SMOPCA SpaBalance SpaFusion SpaMI spaMultiVAE
    SpaMV SWITCH MultiGATE
)

TARGET="${1:-all}"

if [[ "$TARGET" == "all" ]]; then
    install_base
    echo ""
    for m in "${ALL_METHODS[@]}"; do
        install_"$m" || echo "[FAIL] $m"
        echo ""
    done
elif [[ "$TARGET" == "base" ]]; then
    install_base
else
    install_"$TARGET"
fi

echo "============================================"
echo "Setup complete. Env dirs:"
du -sh "${ENV_DIR}"/*/ 2>/dev/null
echo "============================================"
