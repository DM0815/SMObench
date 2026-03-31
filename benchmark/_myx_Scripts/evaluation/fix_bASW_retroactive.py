#!/usr/bin/env python3
"""
Retroactive fix for bASW values in horizontal evaluation CSVs.

Problem: bASW was computed as max(0, 1 - raw_silhouette) where raw ∈ [-1, 1],
         producing values up to ~2.0 instead of [0, 1].
Fix:     bASW_new = old_bASW / 2.0  (equivalent to (1 - raw) / 2)
         Then recompute BER_Score and Final_Score.

Usage:
    python fix_bASW_retroactive.py --root /path/to/SMOBench-CLEAN
"""

import os
import argparse
import glob
import pandas as pd
import numpy as np


BER_METRICS = ['kBET', 'KNN_connectivity', 'bASW', 'iLISI', 'PCR']


def fix_csv(csv_path, dry_run=False):
    """Fix bASW in one horizontal CSV (Metric/Value row format)."""
    df = pd.read_csv(csv_path)
    if 'Metric' not in df.columns or 'Value' not in df.columns:
        return False

    metrics = dict(zip(df['Metric'], df['Value'].astype(float)))

    if 'bASW' not in metrics:
        return False

    old_bASW = metrics['bASW']
    new_bASW = old_bASW / 2.0
    metrics['bASW'] = new_bASW

    # Recompute BER_Score
    ber_vals = [metrics[m] for m in BER_METRICS if m in metrics and pd.notna(metrics[m])]
    if ber_vals:
        metrics['BER_Score'] = np.mean(ber_vals)

    # Recompute Final_Score
    score_keys = ['SC_Score', 'BVC_Score', 'BER_Score']
    score_vals = [metrics[k] for k in score_keys if k in metrics and pd.notna(metrics[k])]
    if score_vals:
        metrics['Final_Score'] = np.mean(score_vals)

    if dry_run:
        changed = old_bASW != new_bASW
        if old_bASW > 1.0:
            print(f"  {os.path.basename(csv_path)}: bASW {old_bASW:.4f} -> {new_bASW:.4f}")
        return changed

    # Write back
    df_new = pd.DataFrame({'Metric': list(metrics.keys()), 'Value': list(metrics.values())})
    df_new.to_csv(csv_path, index=False)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    horiz_dir = os.path.join(args.root, '_myx_Results', 'evaluation', 'horizontal')
    csvs = glob.glob(os.path.join(horiz_dir, '**', '*.csv'), recursive=True)

    print(f"Found {len(csvs)} horizontal CSVs")
    fixed = 0
    gt1_count = 0

    for csv_path in sorted(csvs):
        df = pd.read_csv(csv_path)
        if 'Metric' in df.columns:
            metrics = dict(zip(df['Metric'], df['Value'].astype(float)))
            if 'bASW' in metrics and metrics['bASW'] > 1.0:
                gt1_count += 1
        if fix_csv(csv_path, dry_run=args.dry_run):
            fixed += 1

    print(f"\nbASW > 1.0 count: {gt1_count}")
    print(f"Fixed: {fixed} CSVs")
    if args.dry_run:
        print("(dry run — no files modified)")


if __name__ == '__main__':
    main()
