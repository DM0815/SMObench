#!/usr/bin/env python3
"""Generate and submit PBS jobs for missing horizontal evaluations."""
import os, subprocess

ROOT = '/data/projects/11003054/e1724738/_private/NUS/_Proj1/SMOBench-CLEAN'
PBS_DIR = os.path.join(ROOT, '_myx_Scripts', 'evaluation', 'pbs')
LOG_DIR = '/scratch/users/nus/e1724738/SMOBench_manuscript'

# method -> missing datasets
MISSING = {
    'CANDIES':    ['Mouse_Brain', 'Mouse_Thymus'],
    'PRAGA':      ['Mouse_Brain', 'Mouse_Embryos_S1', 'Mouse_Embryos_S2', 'Mouse_Thymus'],
    'SpaBalance': ['Mouse_Brain', 'Mouse_Thymus'],
    'SpaMI':      ['Mouse_Brain', 'Mouse_Embryos_S1', 'Mouse_Embryos_S2'],
    'SpaMV':      ['Mouse_Brain'],
}

TEMPLATE = """#!/bin/bash
#PBS -N eval_h_{short}
#PBS -l select=1:ncpus=8:mem=128gb
#PBS -l walltime=8:00:00
#PBS -j oe
#PBS -o /dev/null
#PBS -P 11003054
#PBS -q normal

cd {root}
LOG="{log_dir}/eval_h_{method}.log"
exec > "$LOG" 2>&1
export PYTHONUNBUFFERED=1
source /home/users/nus/e1724738/miniconda3/etc/profile.d/conda.sh
conda activate _Proj1_1

echo "=== Horizontal eval: {method} ==="
echo "Datasets: {datasets}"
echo "Start: $(date)"
python -u _myx_Scripts/evaluation/eval_horizontal.py \\
    --root {root} \\
    --methods {method} \\
    --datasets {datasets}
echo "=== DONE: $(date) ==="
"""

for method, ds_list in MISSING.items():
    short = method[:8].lower()
    ds_str = ' '.join(ds_list)
    content = TEMPLATE.format(
        root=ROOT, log_dir=LOG_DIR, method=method,
        short=short, datasets=ds_str
    )
    pbs_path = os.path.join(PBS_DIR, f'eval_h_{method}.pbs')
    with open(pbs_path, 'w') as f:
        f.write(content)
    print(f"Created: {pbs_path}")

# Submit all
for method in MISSING:
    pbs_path = os.path.join(PBS_DIR, f'eval_h_{method}.pbs')
    result = subprocess.run(['qsub', pbs_path], capture_output=True, text=True)
    print(f"  {method}: {result.stdout.strip()}")
