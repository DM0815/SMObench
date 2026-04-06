#!/bin/bash
# ============================================================================
# build_all.sh - Build all Singularity images for SMObench
# Usage: bash build_all.sh [--method METHOD_NAME]
# Must run on a node with sudo/fakeroot privileges (or use --remote)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEF_DIR="${SCRIPT_DIR}/defs"
IMG_DIR="${SCRIPT_DIR}/images"
LOG_DIR="${SCRIPT_DIR}/logs/build"

mkdir -p "${IMG_DIR}" "${LOG_DIR}"

# All methods (order: base first, then alphabetical)
METHODS=(
    SpatialGlue SpaMosaic PRAGA COSMOS PRESENT CANDIES
    MISO SMOPCA SpaBalance SpaFusion SpaMI spaMultiVAE
    SpaMV SWITCH MultiGATE
)

# Parse args
TARGET_METHOD=""
if [[ "${1:-}" == "--method" ]] && [[ -n "${2:-}" ]]; then
    TARGET_METHOD="$2"
fi

build_image() {
    local name="$1"
    local def="${DEF_DIR}/${name}.def"
    local sif="${IMG_DIR}/${name}.sif"
    local log="${LOG_DIR}/${name}.log"

    if [[ ! -f "$def" ]]; then
        echo "[ERROR] Definition file not found: $def"
        return 1
    fi

    if [[ -f "$sif" ]]; then
        echo "[SKIP] ${name}.sif already exists. Delete it to rebuild."
        return 0
    fi

    echo "[BUILD] ${name} ..."
    singularity build --fakeroot "$sif" "$def" > "$log" 2>&1
    if [[ $? -eq 0 ]]; then
        echo "[OK] ${name}.sif built successfully ($(du -h "$sif" | cut -f1))"
    else
        echo "[FAIL] ${name} - check ${log}"
        return 1
    fi
}

# Step 1: Build base image
echo "============================================"
echo "Step 1: Building base image"
echo "============================================"
if [[ -z "$TARGET_METHOD" ]] || [[ "$TARGET_METHOD" == "base" ]]; then
    build_image "base" || { echo "Base image failed. Cannot continue."; exit 1; }
fi

# Step 2: Build method images
echo ""
echo "============================================"
echo "Step 2: Building method images"
echo "============================================"

if [[ -n "$TARGET_METHOD" ]] && [[ "$TARGET_METHOD" != "base" ]]; then
    build_image "$TARGET_METHOD"
else
    for method in "${METHODS[@]}"; do
        build_image "$method" || echo "[WARN] ${method} failed, continuing..."
    done
fi

echo ""
echo "============================================"
echo "Build complete. Images in: ${IMG_DIR}/"
echo "============================================"
ls -lh "${IMG_DIR}"/*.sif 2>/dev/null || echo "No images found."
