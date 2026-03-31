#!/bin/bash
# Submit all SWITCH scalability jobs
# Run this when queue has space (after MultiGATE jobs finish)
PBS_DIR="/data/projects/11003054/e1724738/_private/NUS/_Proj1/SMOBench-CLEAN/_myx_Scripts/scalability/pbs_scripts"
cd "$PBS_DIR"

submitted=0
failed=0
for f in f5_SW_A[0-9]*_r*.pbs f5_SW_AT*_r*.pbs; do
  result=$(qsub "$f" 2>&1)
  if echo "$result" | grep -q "Maximum"; then
    echo "Queue full, stopping. Submitted $submitted jobs so far."
    echo "Re-run this script later to submit remaining."
    exit 1
  fi
  echo "$result  $f"
  submitted=$((submitted+1))
done
echo "All $submitted SWITCH jobs submitted!"
