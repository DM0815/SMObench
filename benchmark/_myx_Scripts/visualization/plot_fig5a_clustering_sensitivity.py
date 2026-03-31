#!/usr/bin/env python3
"""
Supplementary Fig S3: Clustering Sensitivity — Kendall's tau heatmap
Computes pairwise Kendall's tau between method rankings from
different clustering algorithms (Leiden, Louvain, K-means, Mclust).

Usage:
    python plot_fig5a_clustering_sensitivity.py --root /path/to/SMOBench-CLEAN
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import sys as _sys
_sys.path.insert(0, __import__('os').path.dirname(__file__))
from style_config import apply_style, PAL13
apply_style()



CLUSTERING_METHODS = ['leiden', 'louvain', 'kmeans', 'mclust']
CLUSTERING_LABELS = ['Leiden', 'Louvain', 'K-means', 'Mclust']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--dpi', type=int, default=300)
    return parser.parse_args()


# 2x2 dataset groups: modality × GT
DATASET_GROUPS = [
    ('RNA+ADT',  'withGT', ['Human_Lymph_Nodes', 'Human_Tonsils']),
    ('RNA+ADT',  'woGT',   ['Mouse_Thymus', 'Mouse_Spleen']),
    ('RNA+ATAC', 'withGT', ['Mouse_Embryos_S1', 'Mouse_Embryos_S2']),
    ('RNA+ATAC', 'woGT',   ['Mouse_Brain']),
]


def load_rankings(root, task, gt_datasets=None):
    """Load method rankings for all clustering methods, filtered by gt_datasets."""
    if gt_datasets is None:
        gt_datasets = ['Human_Lymph_Nodes', 'Human_Tonsils', 'Mouse_Embryos_S1', 'Mouse_Embryos_S2']
    summary_dir = os.path.join(root, '_myx_Results', 'evaluation', 'summary')
    rankings = {}

    for clust in CLUSTERING_METHODS:
        for prefix in ['final', 'detailed']:
            path = os.path.join(summary_dir, f'{task}_{prefix}_{clust}.csv')
            if os.path.isfile(path):
                df = pd.read_csv(path)
                df = df[df['Dataset'].isin(gt_datasets)].copy()
                # Compute overall score per method
                method_scores = {}
                for method, grp in df.groupby('Method'):
                    if task == 'vertical':
                        sc = grp.get('SC_Score', pd.Series(0)).mean()
                        bioc = grp.get('BioC_Score', pd.Series(0)).mean()
                        cmgtc = grp.get('CM_GTC', pd.Series(0)).mean()
                        method_scores[method] = 0.2*sc + 0.4*bioc + 0.4*cmgtc
                    else:
                        sc = grp.get('SC_Score', pd.Series(0)).mean()
                        bioc = grp.get('BioC_Score', pd.Series(0)).mean()
                        ber = grp.get('BER_Score', pd.Series(0)).mean()
                        cmgtc = grp.get('CM_GTC', pd.Series(0)).mean()
                        method_scores[method] = 0.15*sc + 0.3*bioc + 0.3*ber + 0.25*cmgtc
                rankings[clust] = pd.Series(method_scores).rank(ascending=False)
                break

    return rankings


def plot_tau_heatmap(rankings, task, out_dir, dpi=300,
                     out_prefix='suppfig3_clustering_tau', out_suffix=''):
    """Plot pairwise Kendall's tau heatmap."""
    n = len(CLUSTERING_METHODS)
    tau_matrix = np.ones((n, n))

    for i in range(n):
        for j in range(i+1, n):
            ci, cj = CLUSTERING_METHODS[i], CLUSTERING_METHODS[j]
            if ci in rankings and cj in rankings:
                common = rankings[ci].index.intersection(rankings[cj].index)
                if len(common) >= 3:
                    tau, _ = kendalltau(rankings[ci][common], rankings[cj][common])
                    tau_matrix[i, j] = tau
                    tau_matrix[j, i] = tau

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(tau_matrix, cmap='RdYlGn', vmin=0.0, vmax=1.0, aspect='equal')

    for i in range(n):
        for j in range(n):
            text = f'{tau_matrix[i,j]:.2f}'
            ax.text(j, i, text, ha='center', va='center', fontsize=12,
                    color='white' if tau_matrix[i,j] < 0.7 else 'black')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(CLUSTERING_LABELS, fontsize=11)
    ax.set_yticklabels(CLUSTERING_LABELS, fontsize=11)
    ax.set_title(f"{task.title()} — Pairwise Kendall's τ", fontsize=13)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Kendall's τ", fontsize=11)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'{out_prefix}_{task}{out_suffix}.png')
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    out_dir = os.path.join(root, '_myx_Results', 'plots')
    os.makedirs(out_dir, exist_ok=True)

    for task in ['vertical', 'horizontal']:
        print(f"\n--- {task.title()} ---")

        for mod_label, gt_type, ds_list in DATASET_GROUPS:
            if len(ds_list) < 1:
                print(f"  {mod_label} {gt_type}: only {len(ds_list)} dataset, skipping")
                continue

            rankings = load_rankings(root, task, gt_datasets=ds_list)
            if len(rankings) < 2:
                print(f"  {mod_label} {gt_type}: need ≥2 clustering methods, found {len(rankings)}")
                continue

            prefix = 'suppfig3_clustering_tau' if gt_type == 'withGT' else 'supp_clustering_tau'
            mod_safe = mod_label.replace('+', '_')
            suffix = f'_{mod_safe}_{gt_type}'

            for clust, ranks in rankings.items():
                top3 = ranks.sort_values().head(3)
                print(f"  [{mod_label} {gt_type}] {clust}: "
                      f"{', '.join(f'{m}({int(r)})' for m, r in top3.items())}")
            plot_tau_heatmap(rankings, task, out_dir, args.dpi,
                             out_prefix=prefix, out_suffix=suffix)

    print("\nDone.")


if __name__ == '__main__':
    main()
