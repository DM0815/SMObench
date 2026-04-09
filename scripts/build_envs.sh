#!/usr/bin/env bash
#
# Build per-method pip prefix directories for Singularity-based execution.
#
# These prefix dirs contain ONLY the extra packages that each method needs
# beyond what the base container (smobench_full.sif) provides.
#
# Usage:
#   bash scripts/build_envs.sh              # Build all
#   bash scripts/build_envs.sh SpatialGlue  # Build one method
#   bash scripts/build_envs.sh --clean       # Remove all and rebuild
#
# The resulting directories are placed under singularity/envs/<MethodName>/
# and are automatically picked up by SMObench's subprocess isolation layer.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVS_DIR="${PROJECT_ROOT}/singularity/envs"

# Packages already in the base container (smobench_full.sif) — do NOT install these
# into per-method prefixes, as they would conflict with container versions.
CONTAINER_PKGS=(
    torch torchvision torchaudio numpy scipy pandas scikit-learn anndata scanpy
    matplotlib h5py setuptools nvidia cuda pip wheel
    torch-geometric torch_geometric torch-scatter torch_scatter
    torch-sparse torch_sparse
)

install_prefix() {
    local method="$1"
    shift
    local packages=("$@")

    local prefix_dir="${ENVS_DIR}/${method}"
    echo "=== Building prefix for ${method} ==="
    mkdir -p "${prefix_dir}"

    pip install --no-cache-dir --target="${prefix_dir}" "${packages[@]}" 2>&1 | \
        grep -v "already satisfied" || true

    # Remove packages that the container already provides
    for pkg in "${CONTAINER_PKGS[@]}"; do
        rm -rf "${prefix_dir}/${pkg}" \
               "${prefix_dir}/${pkg//-/_}" \
               "${prefix_dir}/${pkg//-/_}"*.dist-info \
               "${prefix_dir}/${pkg}"*.dist-info \
               2>/dev/null || true
    done

    echo "  -> ${prefix_dir} ($(du -sh "${prefix_dir}" 2>/dev/null | cut -f1))"
    echo ""
}

build_method() {
    local method="$1"
    case "${method}" in
        SpatialGlue)
            # Uses torch-geometric (in container), just needs scikit-learn update sometimes
            install_prefix SpatialGlue
            ;;
        SpaMosaic)
            install_prefix SpaMosaic dgl hnswlib annoy
            ;;
        PRAGA)
            install_prefix PRAGA scvi-tools
            ;;
        COSMOS)
            install_prefix COSMOS gudhi cmcrameri
            ;;
        PRESENT)
            install_prefix PRESENT genomicranges iranges biocutils
            ;;
        CANDIES)
            install_prefix CANDIES einops timm esda gseapy
            ;;
        MISO)
            install_prefix MISO einops opencv-python-headless scikit-image
            ;;
        MultiGATE)
            install_prefix MultiGATE "tensorflow>=2.10,<2.16" gseapy
            ;;
        SMOPCA)
            # Pure numpy/scanpy — no extra deps needed
            mkdir -p "${ENVS_DIR}/SMOPCA"
            echo "  SMOPCA: no extra deps needed"
            ;;
        SpaBalance)
            install_prefix SpaBalance
            ;;
        SpaFusion)
            install_prefix SpaFusion munkres
            ;;
        SpaMI)
            install_prefix SpaMI POT
            ;;
        spaMultiVAE)
            # Uses torch (in container) — no extra deps
            mkdir -p "${ENVS_DIR}/spaMultiVAE"
            echo "  spaMultiVAE: no extra deps needed"
            ;;
        SpaMV)
            install_prefix SpaMV pyro-ppl
            ;;
        SWITCH)
            install_prefix SWITCH pynvml
            ;;
        SMART)
            install_prefix SMART muon harmony-pytorch scikit-misc
            ;;
        GROVER)
            # All deps (torch, torch-geometric) are in container.
            # zeta.nn.FeedForward is vendored — no zetascale needed.
            mkdir -p "${ENVS_DIR}/GROVER"
            echo "  GROVER: no extra deps needed (FeedForward vendored)"
            ;;
        base)
            # Shared base prefix with common extras
            install_prefix base \
                leidenalg umap-learn seaborn tqdm networkx \
                plotly PyYAML squidpy episcanpy harmonypy
            ;;
        *)
            echo "ERROR: Unknown method '${method}'"
            echo "Available: SpatialGlue SpaMosaic PRAGA COSMOS PRESENT CANDIES"
            echo "           MISO MultiGATE SMOPCA SpaBalance SpaFusion SpaMI"
            echo "           spaMultiVAE SpaMV SWITCH SMART GROVER base"
            return 1
            ;;
    esac
}

ALL_METHODS=(
    base
    SpatialGlue SpaMosaic PRAGA COSMOS PRESENT CANDIES
    MISO MultiGATE SMOPCA SpaBalance SpaFusion SpaMI
    spaMultiVAE SpaMV SWITCH SMART GROVER
)

# --- Main ---
if [[ "${1:-}" == "--clean" ]]; then
    echo "Cleaning all prefix dirs..."
    rm -rf "${ENVS_DIR}"
    shift
fi

mkdir -p "${ENVS_DIR}"

if [[ $# -gt 0 ]]; then
    # Build specific methods
    for method in "$@"; do
        build_method "${method}"
    done
else
    # Build all
    for method in "${ALL_METHODS[@]}"; do
        build_method "${method}"
    done
fi

echo "=== Done ==="
echo "Prefix dirs are in: ${ENVS_DIR}/"
ls -d "${ENVS_DIR}"/*/ 2>/dev/null | while read d; do
    echo "  $(basename "$d"): $(du -sh "$d" 2>/dev/null | cut -f1)"
done
