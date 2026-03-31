#!/usr/bin/env python3
"""Generate and submit PBS scripts for CANDIES + SpaFusion rerun without patches."""
import os, subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
SCALABILITY_SCRIPT = os.path.join(SCRIPT_DIR, "..", "run_scalability.py")
PBS_OUT = os.path.join(SCRIPT_DIR, "rerun_nopatch")
os.makedirs(PBS_OUT, exist_ok=True)

# CANDIES: only sizes where patch was active (N > 10000)
# RNA_ADT: 17824; RNA_ATAC: 20000, 37885
CANDIES_RUNS = [
    ("CANDIES", "RNA_ADT", 17824),
    ("CANDIES", "RNA_ATAC", 20000),
    ("CANDIES", "RNA_ATAC", 37885),
]

# SpaFusion: ALL sizes (patch applied to all runs), RNA_ADT only
SPAFUSION_RUNS = [
    ("SpaFusion", "RNA_ADT", n) for n in [1000, 2500, 5000, 10000, 17824]
]

ALL_RUNS = CANDIES_RUNS + SPAFUSION_RUNS

ENV_MAP = {
    "CANDIES": "_Proj1_1",
    "SpaFusion": "_Proj1_1",
}

PBS_TEMPLATE = """#!/bin/bash
#PBS -N rr_{method}_{mod_short}_{n}
#PBS -l select=1:ncpus=4:mem=200gb:ngpus=1
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o {log_file}
#PBS -P 11003054
#PBS -q normal

cd $PBS_O_WORKDIR

source /home/users/nus/e1724738/miniconda3/etc/profile.d/conda.sh
conda activate {env}

echo "=== Rerun {method} {modality} n={n} (no patch) ==="
for rep in 1 2 3 4 5; do
    echo "--- Repeat $rep ---"
    python {script} --method {method} --modality {modality} --n_cells {n} --repeat $rep
done
echo "=== Done ==="
"""

submitted = []
for method, modality, n in ALL_RUNS:
    mod_short = "ADT" if "ADT" in modality else "ATAC"
    env = ENV_MAP[method]
    pbs_name = f"rr_{method}_{mod_short}_{n}.pbs"
    log_file = os.path.join(PBS_OUT, f"rr_{method}_{mod_short}_{n}.log")
    pbs_path = os.path.join(PBS_OUT, pbs_name)

    content = PBS_TEMPLATE.format(
        method=method, modality=modality, n=n,
        mod_short=mod_short, env=env,
        script=SCALABILITY_SCRIPT, log_file=log_file,
    )
    with open(pbs_path, 'w') as f:
        f.write(content)

    # Submit
    result = subprocess.run(["qsub", pbs_path], capture_output=True, text=True)
    job_id = result.stdout.strip()
    submitted.append((method, modality, n, job_id))
    print(f"Submitted: {method} {modality} n={n} -> {job_id}")

print(f"\nTotal: {len(submitted)} jobs submitted")
