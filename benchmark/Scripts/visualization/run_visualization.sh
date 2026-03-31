#!/bin/bash

# SMOBench Visualization Script
# Generate UMAP and spatial plots from integration results
#
# Usage:
#   bash run_visualization.sh [options]
#
# Options:
#   -i, --input       Input AnnData file path (required)
#   -m, --method      Integration method name (required)
#   -e, --embedding   Embedding key in adata.obsm (required)
#   -d, --dataset     Dataset name (optional, auto-extracted if not provided)
#   -s, --subset      Subset/sample name (optional, auto-extracted if not provided)
#   -c, --clustering  Comma-separated clustering methods (default: mclust,leiden,louvain,kmeans)
#   -o, --output      Output plot directory (default: Results/plot)
#   --point-size      Point size for plots (default: 20)
#   --n-neighbors     Number of neighbors for UMAP (default: 30)
#   --no-spatial-flip Don't flip Y coordinates for spatial plots
#   --force-recompute Force recomputation of UMAP coordinates
#   --compare-emb     Comma-separated embedding keys to compare
#   -h, --help        Show this help message

# Set default values
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/plot_umap_spatial.py"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default parameters
CLUSTERING_METHODS="mclust,leiden,louvain,kmeans"
OUTPUT_DIR="Results/plot"
POINT_SIZE=20
N_NEIGHBORS=30
SPATIAL_FLIP=true
FORCE_RECOMPUTE=false
COMPARE_EMBEDDINGS=""

# Function to show usage
show_usage() {
    echo "Usage: bash run_visualization.sh -i <adata_path> -m <method> -e <embedding_key> [options]"
    echo ""
    echo "Required arguments:"
    echo "  -i, --input       Input AnnData file path"
    echo "  -m, --method      Integration method name (e.g., SpatialGlue, PRAGA)"
    echo "  -e, --embedding   Embedding key in adata.obsm"
    echo ""
    echo "Optional arguments:"
    echo "  -d, --dataset     Dataset name (auto-extracted if not provided)"
    echo "  -s, --subset      Subset/sample name (auto-extracted if not provided)"
    echo "  -c, --clustering  Comma-separated clustering methods (default: $CLUSTERING_METHODS)"
    echo "  -o, --output      Output plot directory (default: $OUTPUT_DIR)"
    echo "  --point-size      Point size for plots (default: $POINT_SIZE)"
    echo "  --n-neighbors     Number of neighbors for UMAP (default: $N_NEIGHBORS)"
    echo "  --no-spatial-flip Don't flip Y coordinates for spatial plots"
    echo "  --force-recompute Force recomputation of UMAP coordinates"
    echo "  --compare-emb     Comma-separated embedding keys to compare"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Basic usage"
    echo "  bash run_visualization.sh -i Results/adata/SpatialGlue/Human_Lymph_Nodes/A1/adata_integrated.h5ad -m SpatialGlue -e SpatialGlue"
    echo ""
    echo "  # With custom clustering methods"
    echo "  bash run_visualization.sh -i Results/adata/PRAGA/MISAR/sample1/adata_integrated.h5ad -m PRAGA -e PRAGA -c mclust,leiden"
    echo ""
    echo "  # Compare multiple embeddings"
    echo "  bash run_visualization.sh -i Results/adata/SpaMosaic/simulation/data1/adata_integrated.h5ad -m SpaMosaic -e merged_emb --compare-emb spatial_emb,merged_emb"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_PATH="$2"
            shift 2
            ;;
        -m|--method)
            METHOD="$2"
            shift 2
            ;;
        -e|--embedding)
            EMBEDDING_KEY="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -s|--subset)
            SUBSET="$2"
            shift 2
            ;;
        -c|--clustering)
            CLUSTERING_METHODS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --point-size)
            POINT_SIZE="$2"
            shift 2
            ;;
        --n-neighbors)
            N_NEIGHBORS="$2"
            shift 2
            ;;
        --no-spatial-flip)
            SPATIAL_FLIP=false
            shift
            ;;
        --force-recompute)
            FORCE_RECOMPUTE=true
            shift
            ;;
        --compare-emb)
            COMPARE_EMBEDDINGS="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check required arguments
if [[ -z "$INPUT_PATH" ]]; then
    echo "Error: Input AnnData path is required (-i/--input)"
    show_usage
    exit 1
fi

if [[ -z "$METHOD" ]]; then
    echo "Error: Method name is required (-m/--method)"
    show_usage
    exit 1
fi

if [[ -z "$EMBEDDING_KEY" ]]; then
    echo "Error: Embedding key is required (-e/--embedding)"
    show_usage
    exit 1
fi

# Check if input file exists
if [[ ! -f "$INPUT_PATH" ]]; then
    echo "Error: Input file does not exist: $INPUT_PATH"
    exit 1
fi

# Check if Python script exists
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Error: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Build command
CMD="conda run -n smobench python \"$PYTHON_SCRIPT\""
CMD="$CMD --adata_path \"$INPUT_PATH\""
CMD="$CMD --method \"$METHOD\""
CMD="$CMD --embedding_key \"$EMBEDDING_KEY\""
CMD="$CMD --clustering_methods \"$CLUSTERING_METHODS\""
CMD="$CMD --plot_dir \"$OUTPUT_DIR\""
CMD="$CMD --point_size $POINT_SIZE"
CMD="$CMD --n_neighbors $N_NEIGHBORS"

# Add optional arguments
if [[ -n "$DATASET" ]]; then
    CMD="$CMD --dataset \"$DATASET\""
fi

if [[ -n "$SUBSET" ]]; then
    CMD="$CMD --subset \"$SUBSET\""
fi

if [[ "$SPATIAL_FLIP" == "false" ]]; then
    CMD="$CMD --no-spatial-flip"
fi

if [[ "$FORCE_RECOMPUTE" == "true" ]]; then
    CMD="$CMD --force_recompute"
fi

if [[ -n "$COMPARE_EMBEDDINGS" ]]; then
    CMD="$CMD --compare_embeddings \"$COMPARE_EMBEDDINGS\""
fi

# Change to project root directory
cd "$PROJECT_ROOT" || {
    echo "Error: Cannot change to project root directory: $PROJECT_ROOT"
    exit 1
}

# Print command for transparency
echo "==================== SMOBench Visualization ===================="
echo "Input file: $INPUT_PATH"
echo "Method: $METHOD"
echo "Embedding key: $EMBEDDING_KEY"
echo "Clustering methods: $CLUSTERING_METHODS"
echo "Output directory: $OUTPUT_DIR"
echo "Working directory: $(pwd)"
echo "=============================================================="
echo ""
echo "Executing command:"
echo "$CMD"
echo ""

# Execute the command
eval "$CMD"

# Check execution status
if [[ $? -eq 0 ]]; then
    echo ""
    echo "=============================================================="
    echo "Visualization completed successfully!"
    echo "=============================================================="
else
    echo ""
    echo "=============================================================="
    echo "Visualization failed with exit code: $?"
    echo "=============================================================="
    exit 1
fi