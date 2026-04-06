#!/bin/bash
# Build the SMObench Singularity/Apptainer image
#
# Usage: bash build.sh
# Requires: module load apptainer/1.2.2
# Output: images/smobench_full.sif (~9G)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF="${SCRIPT_DIR}/images/smobench_full.sif"
DEF="${SCRIPT_DIR}/defs/smobench_full.def"

export APPTAINER_CACHEDIR=/data/projects/51001003/dmeng/.apptainer_cache
export APPTAINER_TMPDIR=/tmp
mkdir -p "$(dirname $SIF)" "$APPTAINER_CACHEDIR"

if [[ -f "$SIF" ]]; then
    echo "Image already exists: $SIF ($(du -h $SIF | cut -f1))"
    echo "Delete it first to rebuild."
    exit 0
fi

echo "Building SMObench image..."
apptainer build "$SIF" "$DEF"
echo "Done: $SIF ($(du -h $SIF | cut -f1))"
