#!/usr/bin/env python3
"""
SMOBench Final Score Aggregation
Merges traditional metrics (SC, BioC, BER) with CM-GTC into weighted final scores.

Scoring formulas (from experiment design):
  Vertical:   SMOBench_V = 0.2×SC + 0.4×BioC + 0.4×CM-GTC
  Horizontal: SMOBench_H = 0.15×SC + 0.3×BioC + 0.3×BER + 0.25×CM-GTC

Usage:
    python aggregate_scores.py --root /path/to/SMOBench-CLEAN
    python aggregate_scores.py --root /path/to/SMOBench-CLEAN --clustering leiden
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
VERTICAL_WEIGHTS = {'SC': 0.2, 'BioC': 0.4, 'CM_GTC': 0.4}
HORIZONTAL_WEIGHTS = {'SC': 0.15, 'BioC': 0.3, 'BER': 0.3, 'CM_GTC': 0.25}

CLUSTERING_METHODS = ['leiden', 'louvain', 'kmeans', 'mclust']

DATASET_TYPES = {
    'Human_Lymph_Nodes': 'RNA_ADT', 'Human_Tonsils': 'RNA_ADT',
    'Mouse_Embryos_S1': 'RNA_ATAC', 'Mouse_Embryos_S2': 'RNA_ATAC',
    'Mouse_Thymus': 'RNA_ADT', 'Mouse_Spleen': 'RNA_ADT',
    'Mouse_Brain': 'RNA_ATAC',
}

WITHGT_DATASETS = ['Human_Lymph_Nodes', 'Human_Tonsils', 'Mouse_Embryos_S1', 'Mouse_Embryos_S2']
WOGT_DATASETS = ['Mouse_Thymus', 'Mouse_Spleen', 'Mouse_Brain']


def parse_args():
    parser = argparse.ArgumentParser(description='SMOBench Score Aggregation')
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--clustering', nargs='+', default=None,
                        help='Clustering methods to aggregate (default: all 4)')
    return parser.parse_args()


def load_vertical_metrics(summary_dir, clustering):
    """Load vertical detailed results CSV."""
    path = os.path.join(summary_dir, f'vertical_detailed_{clustering}.csv')
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def load_horizontal_metrics(summary_dir, clustering):
    """Load horizontal detailed results CSV."""
    path = os.path.join(summary_dir, f'horizontal_detailed_{clustering}.csv')
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def load_cmgtc(cmgtc_dir, task):
    """Load CM-GTC results CSV."""
    path = os.path.join(cmgtc_dir, f'cmgtc_{task}.csv')
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def aggregate_cmgtc_by_dataset(cmgtc_df, task):
    """Aggregate slice-level CM-GTC to dataset level (mean across slices)."""
    if task == 'vertical':
        grouped = cmgtc_df.groupby(['Method', 'Dataset'])['CM_GTC'].mean().reset_index()
    else:
        # Horizontal: one score per method-dataset
        grouped = cmgtc_df.groupby(['Method', 'Dataset'])['CM_GTC'].mean().reset_index()
    return grouped


def compute_vertical_final(metrics_df, cmgtc_df):
    """Compute weighted vertical scores: 0.2×SC + 0.4×BioC + 0.4×CM-GTC."""
    if cmgtc_df is None or cmgtc_df.empty:
        print("  WARNING: No CM-GTC data, using SC+BioC only")
        metrics_df = metrics_df.copy()
        metrics_df['SMOBench_V'] = (
            0.2 * metrics_df.get('SC_Score', 0) +
            0.8 * metrics_df.get('BioC_Score', 0)
        )
        return metrics_df

    # Aggregate CM-GTC to dataset level
    cmgtc_agg = aggregate_cmgtc_by_dataset(cmgtc_df, 'vertical')

    # Merge
    merged = metrics_df.merge(
        cmgtc_agg[['Method', 'Dataset', 'CM_GTC']],
        on=['Method', 'Dataset'],
        how='left'
    )

    # Fill missing CM-GTC with 0
    merged['CM_GTC'] = merged['CM_GTC'].fillna(0.0)

    # Weighted score
    w = VERTICAL_WEIGHTS
    merged['SMOBench_V'] = (
        w['SC'] * merged['SC_Score'].fillna(0) +
        w['BioC'] * merged['BioC_Score'].fillna(0) +
        w['CM_GTC'] * merged['CM_GTC']
    )

    return merged


def compute_horizontal_final(metrics_df, cmgtc_df):
    """Compute weighted horizontal scores: 0.15×SC + 0.3×BioC + 0.3×BER + 0.25×CM-GTC."""
    if cmgtc_df is None or cmgtc_df.empty:
        print("  WARNING: No CM-GTC data, using SC+BioC+BER only")
        metrics_df = metrics_df.copy()
        metrics_df['SMOBench_H'] = (
            0.2 * metrics_df.get('SC_Score', 0) +
            0.4 * metrics_df.get('BVC_Score', 0) +
            0.4 * metrics_df.get('BER_Score', 0)
        )
        return metrics_df

    cmgtc_agg = aggregate_cmgtc_by_dataset(cmgtc_df, 'horizontal')

    merged = metrics_df.merge(
        cmgtc_agg[['Method', 'Dataset', 'CM_GTC']],
        on=['Method', 'Dataset'],
        how='left'
    )

    merged['CM_GTC'] = merged['CM_GTC'].fillna(0.0)

    w = HORIZONTAL_WEIGHTS
    merged['SMOBench_H'] = (
        w['SC'] * merged['SC_Score'].fillna(0) +
        w['BioC'] * merged.get('BVC_Score', merged.get('BioC_Score', pd.Series(0))).fillna(0) +
        w['BER'] * merged['BER_Score'].fillna(0) +
        w['CM_GTC'] * merged['CM_GTC']
    )

    return merged


def create_ranking_table(df, score_col, task, datasets=None):
    """Create method ranking table (pivot: Method × Dataset).

    Parameters
    ----------
    datasets : list[str] or None
        If given, restrict ranking to these datasets only.
    """
    if df.empty:
        return pd.DataFrame()

    df_use = df[df['Dataset'].isin(datasets)].copy() if datasets else df.copy()
    if df_use.empty:
        return pd.DataFrame()

    pivot = df_use.pivot_table(
        index='Method',
        columns='Dataset',
        values=score_col,
        aggfunc='mean'
    ).round(4)

    pivot['Average'] = pivot.mean(axis=1)
    pivot['Rank'] = pivot['Average'].rank(ascending=False).astype(int)
    pivot = pivot.sort_values('Rank')
    return pivot


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    eval_dir = os.path.join(root, '_myx_Results', 'evaluation')
    summary_dir = os.path.join(eval_dir, 'summary')
    cmgtc_dir = os.path.join(eval_dir, 'cmgtc')
    os.makedirs(summary_dir, exist_ok=True)

    clustering_list = args.clustering or CLUSTERING_METHODS

    # Load CM-GTC
    cmgtc_v = load_cmgtc(cmgtc_dir, 'vertical')
    cmgtc_h = load_cmgtc(cmgtc_dir, 'horizontal')

    if cmgtc_v is not None:
        print(f"Loaded vertical CM-GTC: {len(cmgtc_v)} entries")
    if cmgtc_h is not None:
        print(f"Loaded horizontal CM-GTC: {len(cmgtc_h)} entries")

    for clust in clustering_list:
        print(f"\n{'='*60}")
        print(f"Clustering: {clust}")
        print(f"{'='*60}")

        # --- Vertical ---
        v_metrics = load_vertical_metrics(summary_dir, clust)
        if v_metrics is not None and not v_metrics.empty:
            v_final = compute_vertical_final(v_metrics, cmgtc_v)
            v_final.to_csv(os.path.join(summary_dir, f'vertical_final_{clust}.csv'), index=False)

            # Main ranking: withGT only
            ranking = create_ranking_table(v_final, 'SMOBench_V', 'vertical',
                                           datasets=WITHGT_DATASETS)
            if not ranking.empty:
                ranking.to_csv(os.path.join(summary_dir, f'vertical_ranking_{clust}.csv'))
                print(f"\nVertical Ranking — withGT ({clust}):")
                print(ranking[['Average', 'Rank']].to_string())

            # Supplementary ranking: woGT only
            ranking_wogt = create_ranking_table(v_final, 'SMOBench_V', 'vertical',
                                                datasets=WOGT_DATASETS)
            if not ranking_wogt.empty:
                ranking_wogt.to_csv(os.path.join(summary_dir, f'vertical_ranking_wogt_{clust}.csv'))
                print(f"\nVertical Ranking — woGT ({clust}):")
                print(ranking_wogt[['Average', 'Rank']].to_string())
        else:
            print(f"  No vertical metrics for {clust}")

        # --- Horizontal ---
        h_metrics = load_horizontal_metrics(summary_dir, clust)
        if h_metrics is not None and not h_metrics.empty:
            h_final = compute_horizontal_final(h_metrics, cmgtc_h)
            h_final.to_csv(os.path.join(summary_dir, f'horizontal_final_{clust}.csv'), index=False)

            # Main ranking: withGT only
            ranking = create_ranking_table(h_final, 'SMOBench_H', 'horizontal',
                                           datasets=WITHGT_DATASETS)
            if not ranking.empty:
                ranking.to_csv(os.path.join(summary_dir, f'horizontal_ranking_{clust}.csv'))
                print(f"\nHorizontal Ranking — withGT ({clust}):")
                print(ranking[['Average', 'Rank']].to_string())

            # Supplementary ranking: woGT only
            ranking_wogt = create_ranking_table(h_final, 'SMOBench_H', 'horizontal',
                                                datasets=WOGT_DATASETS)
            if not ranking_wogt.empty:
                ranking_wogt.to_csv(os.path.join(summary_dir, f'horizontal_ranking_wogt_{clust}.csv'))
                print(f"\nHorizontal Ranking — woGT ({clust}):")
                print(ranking_wogt[['Average', 'Rank']].to_string())
        else:
            print(f"  No horizontal metrics for {clust}")

    # --- Overall Score (average of V and H) ---
    print(f"\n{'='*60}")
    print("Overall Score = mean(SMOBench_V, SMOBench_H)")
    print(f"{'='*60}")

    for clust in clustering_list:
        # Main overall: withGT only
        v_path = os.path.join(summary_dir, f'vertical_ranking_{clust}.csv')
        h_path = os.path.join(summary_dir, f'horizontal_ranking_{clust}.csv')
        if os.path.isfile(v_path) and os.path.isfile(h_path):
            v_rank = pd.read_csv(v_path, index_col=0)
            h_rank = pd.read_csv(h_path, index_col=0)
            overall = pd.DataFrame({
                'V_Score': v_rank['Average'],
                'H_Score': h_rank['Average'],
            })
            overall['Overall'] = overall.mean(axis=1)
            overall['Rank'] = overall['Overall'].rank(ascending=False).astype(int)
            overall = overall.sort_values('Rank')
            overall.to_csv(os.path.join(summary_dir, f'overall_ranking_{clust}.csv'))
            print(f"\nOverall Ranking — withGT ({clust}):")
            print(overall.to_string())

        # Supplementary overall: woGT only
        v_wogt_path = os.path.join(summary_dir, f'vertical_ranking_wogt_{clust}.csv')
        h_wogt_path = os.path.join(summary_dir, f'horizontal_ranking_wogt_{clust}.csv')
        if os.path.isfile(v_wogt_path) and os.path.isfile(h_wogt_path):
            v_rank_wogt = pd.read_csv(v_wogt_path, index_col=0)
            h_rank_wogt = pd.read_csv(h_wogt_path, index_col=0)
            overall_wogt = pd.DataFrame({
                'V_Score': v_rank_wogt['Average'],
                'H_Score': h_rank_wogt['Average'],
            })
            overall_wogt['Overall'] = overall_wogt.mean(axis=1)
            overall_wogt['Rank'] = overall_wogt['Overall'].rank(ascending=False).astype(int)
            overall_wogt = overall_wogt.sort_values('Rank')
            overall_wogt.to_csv(os.path.join(summary_dir, f'overall_ranking_wogt_{clust}.csv'))
            print(f"\nOverall Ranking — woGT ({clust}):")
            print(overall_wogt.to_string())

    print(f"\nAll results saved to: {summary_dir}")


if __name__ == '__main__':
    main()
