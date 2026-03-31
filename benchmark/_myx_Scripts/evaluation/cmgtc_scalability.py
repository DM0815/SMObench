#!/usr/bin/env python3
"""
CM-GTC Scalability Benchmark
Measures CM-GTC computation time across different sample sizes.
Uses synthetic Gaussian mixture data (2 modalities, 5 clusters).

Usage:
    python cmgtc_scalability.py --out_dir /path/to/output
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add CM-GTC path
sys.path.insert(0, '/data/projects/11003054/e1724738/_private/NUS/_Proj1/storage/_2_metric')
from cm_gtc import CMGTC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'visualization'))
from style_config import apply_style
apply_style()


SAMPLE_SIZES = [500, 1000, 2500, 5000, 10000]
N_REPEATS = 3
N_CLUSTERS = 5
N_FEATURES_RNA = 50
N_FEATURES_ADT = 20
JOINT_DIM = 30


def generate_synthetic(n_samples, n_clusters=5, seed=42):
    """Generate synthetic multi-modal data with known cluster structure."""
    rng = np.random.RandomState(seed)
    labels = rng.randint(0, n_clusters, n_samples)

    # RNA modality
    centers_rna = rng.randn(n_clusters, N_FEATURES_RNA) * 3
    rna = centers_rna[labels] + rng.randn(n_samples, N_FEATURES_RNA) * 0.5

    # ADT modality
    centers_adt = rng.randn(n_clusters, N_FEATURES_ADT) * 3
    adt = centers_adt[labels] + rng.randn(n_samples, N_FEATURES_ADT) * 0.5

    # Joint embedding (simulated integration output)
    centers_joint = rng.randn(n_clusters, JOINT_DIM) * 2
    joint = centers_joint[labels] + rng.randn(n_samples, JOINT_DIM) * 0.3

    return joint, {'RNA': rna, 'ADT': adt}


def benchmark_cmgtc(n_samples, seed=42):
    """Run CM-GTC and return wall-clock time in seconds."""
    joint, modalities = generate_synthetic(n_samples, seed=seed)

    evaluator = CMGTC()

    start = time.perf_counter()
    score, _ = evaluator.compute_cm_gtc(joint, modalities)
    elapsed = time.perf_counter() - start

    return elapsed, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='.')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for n in SAMPLE_SIZES:
        times = []
        for rep in range(N_REPEATS):
            print(f"  N={n:>6d}, rep={rep+1}/{N_REPEATS} ...", end=' ', flush=True)
            try:
                elapsed, score = benchmark_cmgtc(n, seed=42 + rep)
                times.append(elapsed)
                print(f"{elapsed:.2f}s (CM-GTC={score:.4f})")
            except Exception as e:
                print(f"FAILED: {e}")

        if times:
            results.append({
                'N': n,
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
            })

    df = pd.DataFrame(results)
    csv_path = os.path.join(args.out_dir, 'cmgtc_scalability.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.errorbar(df['N'], df['mean_time'], yerr=df['std_time'],
                fmt='o-', color='#E24A33', linewidth=2, markersize=7,
                capsize=4, label='CM-GTC')

    ax.set_xlabel('Number of Spots', fontsize=12)
    ax.set_ylabel('Runtime (seconds)', fontsize=12)
    ax.set_title('CM-GTC Computational Scalability', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(0.5, 1000)

    # Reference line
    ax.axhline(y=60, color='#AAAAAA', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.text(df['N'].max() * 0.95, 60, '1 min', fontsize=8, ha='right', va='bottom',
            color='#999999')

    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, 'cmgtc_scalability.pdf')
    plt.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Plot saved to {out_path}")


if __name__ == '__main__':
    main()
