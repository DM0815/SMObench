#!/usr/bin/env python3
"""
Fig 5a: Clustering Robustness — Bump Chart (rank stability across 4 clustering algorithms).
Each line = one method. X = clustering algorithm. Y = rank. Stable methods = flat lines.

Usage:
    python plot_fig5a_radar.py --root /path/to/SMOBench-CLEAN
"""
import os, sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from style_config import apply_style, PAL13
apply_style()

CLUSTERING_METHODS = ['leiden', 'louvain', 'kmeans', 'mclust']
CLUST_LABELS = ['Leiden', 'Louvain', 'K-means', 'Mclust']

# 2×2 dataset groups: modality × GT
DATASET_GROUPS = [
    ('RNA+ADT',  'withGT', ['Human_Lymph_Nodes', 'Human_Tonsils']),
    ('RNA+ADT',  'woGT',   ['Mouse_Thymus', 'Mouse_Spleen']),
    ('RNA+ATAC', 'withGT', ['Mouse_Embryos_S1', 'Mouse_Embryos_S2']),
    ('RNA+ATAC', 'woGT',   ['Mouse_Brain']),
]

palette = PAL13 + ['#333333', '#777777']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--task', choices=['vertical', 'horizontal', 'both'], default='both')
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--dpi', type=int, default=300)
    return parser.parse_args()


def load_rankings(root, task, datasets=None):
    """Load overall method rankings for each clustering algorithm."""
    summary = os.path.join(root, '_myx_Results', 'evaluation', 'summary')
    score_col = 'SMOBench_V' if task == 'vertical' else 'SMOBench_H'

    rankings = {}
    for clust in CLUSTERING_METHODS:
        for prefix in [f'{task}_final', f'{task}_detailed']:
            path = os.path.join(summary, f'{prefix}_{clust}.csv')
            if os.path.isfile(path):
                df = pd.read_csv(path)
                df.columns = [c.replace(' ', '_') for c in df.columns]
                if datasets is not None:
                    df = df[df['Dataset'].isin(datasets)]
                if score_col in df.columns and not df.empty:
                    avg = df.groupby('Method')[score_col].mean().sort_values(ascending=False)
                    rankings[clust] = {m: rank + 1 for rank, (m, _) in enumerate(avg.items())}
                break
    return rankings


def plot_bump_chart(rankings, task, out_dir, dpi=300):
    """Draw a bump chart: x = clustering method, y = rank, one line per method."""
    if not rankings:
        print(f"  No rankings for {task}")
        return

    # Collect all methods
    all_methods = set()
    for r in rankings.values():
        all_methods.update(r.keys())
    all_methods = sorted(all_methods)

    # Build rank matrix
    clust_list = [c for c in CLUSTERING_METHODS if c in rankings]
    n_clust = len(clust_list)
    n_methods = len(all_methods)

    # Sort methods by average rank
    avg_ranks = {}
    for m in all_methods:
        ranks = [rankings[c].get(m, n_methods) for c in clust_list]
        avg_ranks[m] = np.mean(ranks)
    all_methods = sorted(all_methods, key=lambda m: avg_ranks[m])

    method_colors = {m: palette[i % len(palette)] for i, m in enumerate(all_methods)}

    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.arange(n_clust)

    for m in all_methods:
        ranks = [rankings[c].get(m, np.nan) for c in clust_list]
        avg_r = avg_ranks[m]

        lw = 2.5 if avg_r <= 3 else 1.8 if avg_r <= 7 else 1.2
        alpha = 1.0 if avg_r <= 5 else 0.6

        ax.plot(x, ranks, 'o-', color=method_colors[m], linewidth=lw,
                markersize=7 if avg_r <= 3 else 5, alpha=alpha, zorder=10 - avg_r)

        last_rank = ranks[-1] if not np.isnan(ranks[-1]) else ranks[-2] if len(ranks) > 1 else 0
        ax.text(n_clust - 1 + 0.15, last_rank, m, fontsize=8, va='center',
                ha='left', color=method_colors[m], fontweight='bold' if avg_r <= 3 else 'normal',
                clip_on=False)

    ax.set_xticks(x)
    ax.set_xticklabels([CLUST_LABELS[CLUSTERING_METHODS.index(c)] for c in clust_list],
                       fontsize=11)
    ax.set_ylabel('Rank', fontsize=12)
    ax.set_ylim(n_methods + 0.5, 0.5)
    ax.set_yticks(range(1, n_methods + 1))
    ax.set_xlim(-0.2, n_clust - 1 + 0.3)

    # Light horizontal grid at each rank
    for r in range(1, n_methods + 1):
        ax.axhline(r, color='#EEEEEE', lw=0.4, zorder=0)

    task_label = 'Vertical' if task == 'vertical' else 'Horizontal'
    ax.set_title(f'{task_label} — Clustering Robustness',
                 fontsize=13, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    out = os.path.join(out_dir, f'fig5a_radar_grid_{task}')
    fig.savefig(out + '.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(out + '.png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out}.pdf")


def plot_bump_chart_on_ax(ax, rankings, task, all_methods, method_colors):
    """Draw bump chart on a given axes."""
    clust_list = [c for c in CLUSTERING_METHODS if c in rankings]
    n_clust = len(clust_list)
    n_methods = len(all_methods)

    avg_ranks = {}
    for m in all_methods:
        ranks = [rankings[c].get(m, n_methods) for c in clust_list]
        avg_ranks[m] = np.mean(ranks)

    x = np.arange(n_clust)

    for m in all_methods:
        ranks = [rankings[c].get(m, np.nan) for c in clust_list]
        avg_r = avg_ranks[m]
        lw = 2.5 if avg_r <= 3 else 1.8 if avg_r <= 7 else 1.2
        alpha = 1.0 if avg_r <= 5 else 0.6

        ax.plot(x, ranks, 'o-', color=method_colors[m], linewidth=lw,
                markersize=7 if avg_r <= 3 else 5, alpha=alpha, zorder=10 - avg_r)

        last_rank = ranks[-1] if not np.isnan(ranks[-1]) else ranks[-2] if len(ranks) > 1 else 0
        ax.text(n_clust - 1 + 0.15, last_rank, m, fontsize=7, va='center',
                ha='left', color=method_colors[m], fontweight='bold' if avg_r <= 3 else 'normal',
                clip_on=False)

    ax.set_xticks(x)
    ax.set_xticklabels([CLUST_LABELS[CLUSTERING_METHODS.index(c)] for c in clust_list],
                       fontsize=10)
    ax.set_ylabel('Rank', fontsize=11)
    ax.set_ylim(n_methods + 0.5, 0.5)
    ax.set_yticks(range(1, n_methods + 1))
    ax.set_xlim(-0.2, n_clust - 1 + 0.3)

    for r in range(1, n_methods + 1):
        ax.axhline(r, color='#EEEEEE', lw=0.4, zorder=0)

    task_label = 'Vertical' if task == 'vertical' else 'Horizontal'
    ax.set_title(f'{task_label} — Clustering Robustness',
                 fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    out_dir = os.path.join(root, '_myx_Results', 'plots')
    os.makedirs(out_dir, exist_ok=True)

    for mod_label, gt_type, ds_list in DATASET_GROUPS:
        if len(ds_list) < 1:
            continue

        mod_safe = mod_label.replace('+', '_')
        prefix = 'fig5a' if gt_type == 'withGT' else 'supp'
        suffix = f'_{mod_safe}_{gt_type}'

        rankings_v = load_rankings(root, 'vertical', ds_list)
        rankings_h = load_rankings(root, 'horizontal', ds_list)

        # Unified method order
        all_methods_set = set()
        for r in list(rankings_v.values()) + list(rankings_h.values()):
            all_methods_set.update(r.keys())
        all_methods = sorted(all_methods_set)
        n_methods = len(all_methods)

        clust_list_v = [c for c in CLUSTERING_METHODS if c in rankings_v]
        avg_ranks_v = {}
        for m in all_methods:
            ranks = [rankings_v[c].get(m, n_methods) for c in clust_list_v]
            avg_ranks_v[m] = np.mean(ranks)
        all_methods = sorted(all_methods, key=lambda m: avg_ranks_v[m])
        method_colors = {m: palette[i % len(palette)] for i, m in enumerate(all_methods)}

        # Individual bump charts
        for task, rankings in [('vertical', rankings_v), ('horizontal', rankings_h)]:
            if rankings:
                print(f'\n=== {task.title()} {mod_label} {gt_type} ===')
                out_path = os.path.join(out_dir, f'{prefix}_radar_grid_{task}{suffix}')
                # Use plot_bump_chart but with custom output path
                plot_bump_chart(rankings, task, out_dir, args.dpi)
                # Rename to include suffix
                default_out = os.path.join(out_dir, f'fig5a_radar_grid_{task}')
                for ext in ['.pdf', '.png']:
                    src = default_out + ext
                    dst = out_path + ext
                    if os.path.isfile(src):
                        os.rename(src, dst)
                        print(f"  -> {dst}")


if __name__ == '__main__':
    main()
