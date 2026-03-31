#!/usr/bin/env python3
"""
Generate and optionally submit PBS scripts for scalability analysis.
Following [3] Hu et al., Nature Methods 2024:
  - RNA_ADT:  Mouse_Thymus (17,824 cells), sample [1000, 2500, 5000, 10000, 17824]
  - RNA_ATAC: Mouse_Brain  (37,885 cells), sample [1000, 2500, 5000, 10000, 20000, 37885]
  - 5 repeats per cell count
  - 13 methods

Usage:
    python submit_scalability.py --dry_run          # Generate PBS scripts only
    python submit_scalability.py --submit           # Generate and submit
    python submit_scalability.py --method COSMOS     # Single method only
    python submit_scalability.py --modality RNA_ADT  # Single modality only
"""

import os
import argparse
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PBS_DIR = os.path.join(SCRIPT_DIR, "pbs_scripts")

# Methods and their conda environments
METHOD_ENVS = {
    "CANDIES":     "_Proj1_1",
    "COSMOS":      "_Proj1_1",
    "MISO":        "_Proj1_1",
    "PRAGA":       "_Proj1_1",
    "SpaMosaic":   "_Proj1_1_SpaMosaic",
    "SpatialGlue": "_Proj1_1",
    "SpaMultiVAE": "_Proj1_1",
    "PRESENT":     "_Proj1_1_PRESENT",
    "SMOPCA":      "_Proj1_1",
    "SpaBalance":  "_Proj1_1",
    "SpaMI":       "_Proj1_1_PRESENT",
    "SpaMV":       "_Proj1_1_SpaMV",
    "SpaFusion":   "_Proj1_1",
}

CELL_COUNTS = {
    "RNA_ADT":  [1000, 2500, 5000, 10000, 17824],
    "RNA_ATAC": [1000, 2500, 5000, 10000, 20000, 37885],
}

N_REPEATS = 5

# Walltime estimates - generous to avoid re-runs (each PBS runs all cell_counts × 5 repeats)
WALLTIME = {
    "CANDIES": "48:00:00",
    "COSMOS": "48:00:00",
    "MISO": "48:00:00",
    "PRAGA": "48:00:00",
    "SpaMosaic": "48:00:00",
    "SpatialGlue": "48:00:00",
    "SpaMultiVAE": "48:00:00",
    "PRESENT": "48:00:00",
    "SMOPCA": "48:00:00",
    "SpaBalance": "48:00:00",
    "SpaMI": "48:00:00",
    "SpaMV": "48:00:00",
    "SpaFusion": "48:00:00",
}

PBS_TEMPLATE = """#!/bin/bash
#PBS -N scal_{method}_{modality_short}
#PBS -l select=1:ncpus=4:mem=400gb:ngpus=1
#PBS -l walltime={walltime}
#PBS -j oe
#PBS -o {log_dir}/{method}_{modality_short}.log
#PBS -P 11003054
#PBS -q normal

cd $PBS_O_WORKDIR

# Load conda
source /home/users/nus/e1724738/miniconda3/etc/profile.d/conda.sh
conda activate {conda_env}

echo "=============================="
echo "Scalability: {method} x {modality}"
echo "Start: $(date)"
echo "=============================="

SCRIPT="{script_dir}/run_scalability.py"

{run_commands}

echo "=============================="
echo "End: $(date)"
echo "=============================="
"""


def generate_pbs(method, modality, dry_run=True):
    """Generate PBS script for one method x one modality."""
    os.makedirs(PBS_DIR, exist_ok=True)
    log_dir = os.path.join(PROJECT_ROOT, "_myx_Results/scalability/logs")
    os.makedirs(log_dir, exist_ok=True)

    modality_short = "ADT" if modality == "RNA_ADT" else "ATAC"

    # Build run commands for all cell_counts x repeats
    commands = []
    for n_cells in CELL_COUNTS[modality]:
        for rep in range(1, N_REPEATS + 1):
            cmd = (
                f'echo "--- {method} {modality} {n_cells}cells rep{rep} ---"\n'
                f'python "$SCRIPT" --method {method} --modality {modality} '
                f'--n_cells {n_cells} --repeat {rep} --cluster_nums 7\n'
            )
            commands.append(cmd)

    pbs_content = PBS_TEMPLATE.format(
        method=method,
        modality=modality,
        modality_short=modality_short,
        walltime=WALLTIME.get(method, "04:00:00"),
        log_dir=log_dir,
        conda_env=METHOD_ENVS[method],
        script_dir=SCRIPT_DIR,
        run_commands="\n".join(commands),
    )

    pbs_file = os.path.join(PBS_DIR, f"scal_{method}_{modality_short}.pbs")
    with open(pbs_file, "w") as f:
        f.write(pbs_content)

    n_runs = len(CELL_COUNTS[modality]) * N_REPEATS
    print(f"  Generated: {pbs_file} ({n_runs} runs)")
    return pbs_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default=None, help="Single method (default: all)")
    parser.add_argument("--modality", type=str, default=None, choices=["RNA_ADT", "RNA_ATAC"])
    parser.add_argument("--submit", action="store_true", help="Submit PBS scripts")
    parser.add_argument("--dry_run", action="store_true", help="Generate scripts only (default)")
    args = parser.parse_args()

    methods = [args.method] if args.method else list(METHOD_ENVS.keys())
    modalities = [args.modality] if args.modality else ["RNA_ADT", "RNA_ATAC"]

    print(f"Generating scalability PBS scripts...")
    print(f"  Methods: {len(methods)}")
    print(f"  Modalities: {modalities}")
    print(f"  Cell counts: {CELL_COUNTS}")
    print(f"  Repeats: {N_REPEATS}")

    total_runs = 0
    pbs_files = []

    for method in methods:
        for modality in modalities:
            # Skip methods that don't support RNA_ATAC
            if modality == "RNA_ATAC" and method in ("SpaFusion", "SpaMultiVAE"):
                print(f"  Skipping: {method} x {modality} (not supported)")
                continue

            pbs_file = generate_pbs(method, modality)
            pbs_files.append(pbs_file)
            total_runs += len(CELL_COUNTS[modality]) * N_REPEATS

    print(f"\nTotal: {len(pbs_files)} PBS scripts, {total_runs} individual runs")

    if args.submit:
        print("\nSubmitting PBS scripts...")
        for pbs_file in pbs_files:
            result = subprocess.run(["qsub", pbs_file], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  Submitted: {os.path.basename(pbs_file)} -> {result.stdout.strip()}")
            else:
                print(f"  FAILED: {os.path.basename(pbs_file)} -> {result.stderr.strip()}")
    else:
        print("\nDry run complete. Use --submit to submit PBS scripts.")
        print(f"PBS scripts saved to: {PBS_DIR}")


if __name__ == "__main__":
    main()
