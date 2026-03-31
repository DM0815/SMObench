#!/bin/bash
echo "=== Rerun Job Status ($(date)) ==="
qstat -u e1724738 2>/dev/null | grep -E "rr_CANDIE|rr_SpaFus" || echo "No rerun jobs in queue"

echo ""
echo "=== Completed Logs ==="
LOG_DIR="/home/users/nus/e1724738/_main/_private/NUS/_Proj1/SMOBench-CLEAN/_myx_Scripts/scalability/pbs/rerun_nopatch"
for f in "$LOG_DIR"/rr_*.log; do
  [ -f "$f" ] || continue
  name=$(basename "$f" .log)
  success=$(grep -c '"success": true' "$f" 2>/dev/null)
  fail=$(grep -c '"success": false' "$f" 2>/dev/null)
  oom=$(grep -ci 'OutOfMemory\|CUDA out of memory\|Killed' "$f" 2>/dev/null)
  err=$(grep -ci 'Error\|Exception' "$f" 2>/dev/null)
  echo "  $name: success=$success fail=$fail oom=$oom errors=$err"
done
