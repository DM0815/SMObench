#!/usr/bin/env python3
"""
Run scalability experiment for a single (method, modality, cell_count, repeat).
Calls the existing horizontal integration run scripts with subsampled data.

Usage:
    python run_scalability.py --method COSMOS --modality RNA_ADT --n_cells 5000 --repeat 1

This script:
1. Locates the pre-generated subsampled h5ad files
2. Calls the method's run script via subprocess
3. Collects the timing JSON output
"""

import os
import sys
import json
import time
import argparse
import subprocess
import glob
import traceback
import resource
import threading

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Method -> (script_path, conda_env, extra_args)
METHOD_CONFIG = {
    "CANDIES":     ("horizontal_integration/CANDIES/run_candies.py",         "_Proj1_1", {}),
    "COSMOS":      ("horizontal_integration/COSMOS/run_cosmos.py",           "_Proj1_1", {}),
    "MISO":        ("horizontal_integration/MISO/run_MISO_h.py",            "_Proj1_1", {}),
    "PRAGA":       ("horizontal_integration/PRAGA/run_praga.py",            "_Proj1_1", {}),
    "SpaMosaic":   ("horizontal_integration/SpaMosaic/run_spamosaic.py",    "_Proj1_1_SpaMosaic", {}),
    "SpatialGlue": ("horizontal_integration/SpatialGlue/run_SpatialGlue.py","_Proj1_1", {}),
    "SpaMultiVAE": ("horizontal_integration/SpaMultiVAE/run_spamultivae.py","_Proj1_1", {}),
    "PRESENT":     ("horizontal_integration/PRESENT/run_present.py",        "_Proj1_1_PRESENT", {}),
    "SMOPCA":      ("horizontal_integration/SMOPCA/run_SMOPCA.py",          "_Proj1_1", {}),
    "SpaBalance":  ("horizontal_integration/SpaBalance/run_SpaBalance.py",  "_Proj1_1", {}),
    "SpaMI":       ("horizontal_integration/SpaMI/run_SpaMI.py",            "_Proj1_1_PRESENT", {}),
    "SpaMV":       ("horizontal_integration/SpaMV/run_spamv.py",            "_Proj1_1_SpaMV", {}),
    "SpaFusion":   ("horizontal_integration/SpaFusion/run_SpaFusion.py",    "_Proj1_1", {}),
    "MultiGATE":   ("horizontal_integration/MultiGATE/run_MultiGATE.py",  "_Proj1_1_MultiGATE", {}),
    "SWITCH":      ("horizontal_integration/SWITCH/run_SWITCH.py",        "_Proj1_1", {}),
}

DATASET_INFO = {
    "RNA_ADT": {"name": "Mouse_Thymus", "other_key": "ADT"},
    "RNA_ATAC": {"name": "Mouse_Brain", "other_key": "ATAC"},
}


def find_subsampled_files(data_dir, modality, n_cells, repeat):
    """Find the pre-generated subsampled h5ad files."""
    info = DATASET_INFO[modality]
    dataset_name = info["name"]
    other_key = info["other_key"]
    seed = repeat * 1000 + n_cells

    mod_dir = os.path.join(data_dir, modality)
    rna_pattern = f"{dataset_name}_RNA_{n_cells}cells_rep{seed}.h5ad"
    other_pattern = f"{dataset_name}_{other_key}_{n_cells}cells_rep{seed}.h5ad"

    rna_path = os.path.join(mod_dir, rna_pattern)
    other_path = os.path.join(mod_dir, other_pattern)

    if not os.path.exists(rna_path):
        raise FileNotFoundError(f"RNA file not found: {rna_path}")
    if not os.path.exists(other_path):
        raise FileNotFoundError(f"Other file not found: {other_path}")

    return rna_path, other_path


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB via nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0:
            # Sum across all GPUs (usually 1)
            return sum(int(x.strip()) for x in out.stdout.strip().split("\n") if x.strip())
    except Exception:
        pass
    return 0


def get_proc_peak_rss_mb(pid):
    """Get VmPeak (peak virtual memory) and VmHWM (peak RSS) from /proc/{pid}/status."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024  # KB -> MB
    except Exception:
        pass
    return 0


def get_children_pids(pid):
    """Get all descendant PIDs of a process."""
    try:
        out = subprocess.run(
            ["pgrep", "-P", str(pid)], capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0:
            children = [int(p) for p in out.stdout.strip().split("\n") if p.strip()]
            all_pids = list(children)
            for cpid in children:
                all_pids.extend(get_children_pids(cpid))
            return all_pids
    except Exception:
        pass
    return []


class MemoryMonitor:
    """Background thread that polls GPU and system memory during execution."""

    def __init__(self, pid, interval=2.0):
        self.pid = pid
        self.interval = interval
        self.peak_gpu_mb = 0
        self.peak_rss_mb = 0
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _poll(self):
        while not self._stop.is_set():
            # GPU memory
            gpu_mb = get_gpu_memory_mb()
            if gpu_mb > self.peak_gpu_mb:
                self.peak_gpu_mb = gpu_mb

            # System RSS: sum over process tree
            all_pids = [self.pid] + get_children_pids(self.pid)
            total_rss = 0
            for p in all_pids:
                try:
                    with open(f"/proc/{p}/status") as f:
                        for line in f:
                            if line.startswith("VmRSS:"):
                                total_rss += int(line.split()[1]) / 1024  # KB -> MB
                                break
                except Exception:
                    pass
            if total_rss > self.peak_rss_mb:
                self.peak_rss_mb = total_rss

            self._stop.wait(self.interval)


def run_method(method, rna_path, other_path, modality, save_path, cluster_nums):
    """Run a method's script and return timing info + memory usage."""
    script_rel, conda_env, extra = METHOD_CONFIG[method]
    script_path = os.path.join(PROJECT_ROOT, "_myx_Scripts", script_rel)

    info = DATASET_INFO[modality]

    # Build command
    cmd_args = [
        sys.executable, script_path,
        "--RNA_path", rna_path,
        "--save_path", save_path,
        "--dataset", f"scalability_{info['name']}",
        "--cluster_nums", str(cluster_nums),
    ]

    # Add modality-specific path
    if modality == "RNA_ADT":
        cmd_args.extend(["--ADT_path", other_path])
    else:
        cmd_args.extend(["--ATAC_path", other_path])

    # Add extra args from METHOD_CONFIG (e.g. GTF path for ATAC)
    for k, v in extra.items():
        cmd_args.extend([k, str(v)])

    # Auto-add GTF path for methods that need it for ATAC
    if modality == "RNA_ATAC" and method in ("MultiGATE", "SWITCH"):
        gtf_mouse = os.path.join(PROJECT_ROOT, "Dataset/data_info/gencode.vM25.annotation.gtf.gz")
        if os.path.exists(gtf_mouse) and "--gtf_path" not in cmd_args:
            cmd_args.extend(["--gtf_path", gtf_mouse])

    print(f"Running: {method} on {modality}, cmd:")
    print(f"  {' '.join(cmd_args)}")

    # Record baseline GPU memory before run
    baseline_gpu_mb = get_gpu_memory_mb()

    start = time.time()
    proc = subprocess.Popen(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Start memory monitoring
    monitor = MemoryMonitor(proc.pid, interval=2.0)
    monitor.start()

    try:
        stdout, stderr = proc.communicate(timeout=172800)  # 48h per run (no practical limit)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()

    wall_time = time.time() - start
    monitor.stop()

    # Also get peak RSS via resource (cumulative for children)
    child_rusage = resource.getrusage(resource.RUSAGE_CHILDREN)
    rusage_peak_rss_mb = child_rusage.ru_maxrss / 1024  # KB -> MB on Linux

    memory_info = {
        "peak_gpu_memory_mb": monitor.peak_gpu_mb,
        "gpu_memory_delta_mb": max(0, monitor.peak_gpu_mb - baseline_gpu_mb),
        "peak_system_rss_mb": round(monitor.peak_rss_mb, 1),
        "rusage_maxrss_mb": round(rusage_peak_rss_mb, 1),
    }

    if proc.returncode != 0:
        print(f"FAILED (returncode={proc.returncode})")
        print(f"STDERR: {stderr[-2000:]}")
        return None, wall_time, memory_info

    # Create a result-like object
    class Result:
        pass
    result = Result()
    result.returncode = proc.returncode
    result.stdout = stdout
    result.stderr = stderr

    return result, wall_time, memory_info


def main():
    parser = argparse.ArgumentParser(description="Run scalability experiment")
    parser.add_argument("--method", type=str, required=True, choices=list(METHOD_CONFIG.keys()))
    parser.add_argument("--modality", type=str, required=True, choices=["RNA_ADT", "RNA_ATAC"])
    parser.add_argument("--n_cells", type=int, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "_myx_Results/scalability/subsampled_data"))
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "_myx_Results/scalability/results"))
    parser.add_argument("--cluster_nums", type=int, default=7)
    args = parser.parse_args()

    # Find subsampled data
    rna_path, other_path = find_subsampled_files(
        args.data_dir, args.modality, args.n_cells, args.repeat
    )

    # Setup output path
    out_subdir = os.path.join(
        args.out_dir, args.method, args.modality,
        f"{args.n_cells}cells", f"rep{args.repeat}"
    )
    os.makedirs(out_subdir, exist_ok=True)
    save_path = os.path.join(out_subdir, f"{args.method}.h5ad")

    # Run
    print(f"\n{'='*60}")
    print(f"Scalability: {args.method} | {args.modality} | {args.n_cells} cells | rep {args.repeat}")
    print(f"{'='*60}")

    result, wall_time, memory_info = run_method(
        args.method, rna_path, other_path, args.modality,
        save_path, args.cluster_nums
    )

    # Save scalability result
    scalability_result = {
        "method": args.method,
        "modality": args.modality,
        "n_cells": args.n_cells,
        "repeat": args.repeat,
        "wall_time_s": round(wall_time, 2),
        "success": result is not None and result.returncode == 0 if result else False,
        "peak_gpu_memory_mb": memory_info["peak_gpu_memory_mb"],
        "gpu_memory_delta_mb": memory_info["gpu_memory_delta_mb"],
        "peak_system_rss_mb": memory_info["peak_system_rss_mb"],
    }

    # Try to read the timing JSON generated by the method script
    timing_json_path = save_path.replace(".h5ad", "_timing_info.json")
    if os.path.exists(timing_json_path):
        with open(timing_json_path) as f:
            method_timing = json.load(f)
        scalability_result["training_time_s"] = method_timing.get("training_time_s")
        scalability_result["total_time_s"] = method_timing.get("total_time_s")

    # Save
    out_json = os.path.join(out_subdir, f"{args.method}_scalability.json")
    with open(out_json, "w") as f:
        json.dump(scalability_result, f, indent=2)
    print(f"\nScalability result saved to: {out_json}")
    print(json.dumps(scalability_result, indent=2))


if __name__ == "__main__":
    main()
