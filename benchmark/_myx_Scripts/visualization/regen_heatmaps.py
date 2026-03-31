#!/usr/bin/env python3
"""
Reaggregate evaluation summaries (after graph_clisi patch),
then regenerate Fig 2a and Fig 3a heatmaps.

Also fixes RNA_ADT -> RNA+ADT title format.

Usage:
    python regen_heatmaps.py --root /path/to/SMOBench-CLEAN
"""

import os
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--dpi', type=int, default=300)
    return parser.parse_args()


def reaggregate(root):
    """Re-run aggregation to regenerate summary CSVs."""
    eval_scripts = os.path.join(root, '_myx_Scripts', 'evaluation')
    sys.path.insert(0, eval_scripts)
    sys.path.insert(0, os.path.join(root, 'Eval'))

    print("=" * 60)
    print("Re-aggregating VERTICAL results...")
    print("=" * 60)
    from eval_vertical import aggregate_results as agg_vertical
    vertical_output = os.path.join(root, '_myx_Results', 'evaluation', 'vertical')
    agg_vertical(vertical_output, root)

    print()
    print("=" * 60)
    print("Re-aggregating HORIZONTAL results...")
    print("=" * 60)
    from eval_horizontal import aggregate_results as agg_horizontal
    horizontal_output = os.path.join(root, '_myx_Results', 'evaluation', 'horizontal')
    agg_horizontal(horizontal_output, root)


def merge_cmgtc_and_build_final(root):
    """Merge CM-GTC into summary and build final scores."""
    sys.path.insert(0, os.path.join(root, '_myx_Scripts', 'evaluation'))
    try:
        from aggregate_scores import main as build_scores
        print("\n" + "=" * 60)
        print("Building final scores with CM-GTC...")
        print("=" * 60)
        # Simulate args
        sys.argv = ['aggregate_scores.py', '--root', root]
        build_scores()
    except Exception as e:
        print(f"  [WARN] aggregate_scores failed: {e}")
        print("  Continuing with existing summary CSVs...")


def regen_heatmaps(root, dpi):
    """Regenerate heatmaps for all 4 clusterings."""
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_dir = os.path.join(root, '_myx_Results', 'plots')
    os.makedirs(out_dir, exist_ok=True)
    summary_dir = os.path.join(root, '_myx_Results', 'evaluation', 'summary')

    METRIC_COLS_WITHGT = ['Moran_Index', 'ARI', 'NMI', 'asw_celltype', 'graph_clisi']
    METRIC_COLS_WOGT = [
        'Moran_Index', 'Davies-Bouldin_Index_normalized',
        'Silhouette_Coefficient', 'Calinski-Harabaz_Index_normalized',
    ]
    SCORE_COLS_V = ['SC_Score', 'BioC_Score', 'CM_GTC', 'SMOBench_V']
    SCORE_COLS_H = ['SC_Score', 'BioC_Score', 'BER_Score', 'CM_GTC', 'SMOBench_H']
    METRIC_COLS_H_WITHGT = [
        'Moran_Index', 'ARI', 'NMI', 'asw_celltype', 'graph_clisi',
        'kBET', 'KNN_connectivity', 'bASW', 'iLISI', 'PCR',
    ]
    METRIC_COLS_H_WOGT = [
        'Moran_Index', 'Davies-Bouldin_Index_normalized',
        'Silhouette_Coefficient', 'Calinski-Harabaz_Index_normalized',
        'kBET', 'KNN_connectivity', 'bASW', 'iLISI', 'PCR',
    ]

    METHOD_ORDER_V = [
        'CANDIES', 'COSMOS', 'MISO', 'MultiGATE', 'PRAGA', 'PRESENT',
        'SMOPCA', 'SpaBalance', 'SpaFusion', 'SpaMI', 'SpaMosaic',
        'SpaMultiVAE', 'SpaMV', 'SpatialGlue', 'SWITCH',
    ]
    METHOD_ORDER_H = [
        'CANDIES', 'COSMOS', 'MISO', 'PRAGA', 'PRESENT',
        'SMOPCA', 'SpaBalance', 'SpaFusion', 'SpaMI', 'SpaMosaic',
        'SpaMultiVAE', 'SpaMV', 'SpatialGlue',
    ]

    # withGT datasets (main figure)
    RNA_ADT_WITHGT = ['Human_Lymph_Nodes', 'Human_Tonsils']
    RNA_ATAC_WITHGT = ['Mouse_Embryos_S1', 'Mouse_Embryos_S2']
    # woGT datasets (supplementary)
    RNA_ADT_WOGT = ['Mouse_Thymus', 'Mouse_Spleen']
    RNA_ATAC_WOGT = ['Mouse_Brain']

    # Display labels for metrics (NM-quality formatting)
    LABEL_MAP = {
        'Moran_Index': "Moran's I",
        'ARI': 'ARI', 'NMI': 'NMI',
        'asw_celltype': 'ASW',
        'graph_clisi': 'cLISI',
        'Davies-Bouldin_Index_normalized': 'DBI',
        'Silhouette_Coefficient': 'Sil.',
        'Calinski-Harabaz_Index_normalized': 'CHI',
        'kBET': 'kBET',
        'KNN_connectivity': 'KNN\nconn.',
        'bASW': 'bASW', 'iLISI': 'iLISI', 'PCR': 'PCR',
        'SC_Score': 'SC', 'BioC_Score': 'BioC',
        'BER_Score': 'BER', 'CM_GTC': 'CM-GTC',
        'SMOBench_V': 'Total', 'SMOBench_H': 'Total',
    }

    def _plot_heatmap(method_avg, available_cols, score_cols, title, out_path):
        # Fix -0.0 display
        annot_data = method_avg.round(3).copy()
        for col in annot_data.columns:
            annot_data[col] = annot_data[col].apply(lambda x: 0.0 if x == 0 else x)

        norm_data = method_avg.copy()
        for col in norm_data.columns:
            cmin, cmax = norm_data[col].min(), norm_data[col].max()
            if cmax > cmin:
                norm_data[col] = (norm_data[col] - cmin) / (cmax - cmin)
            else:
                norm_data[col] = 0.5

        n_cols = len(available_cols)
        fig_w = max(8, n_cols * 0.9 + 2)
        fig_h = max(4, len(method_avg) * 0.45 + 1.8)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        cmap = sns.color_palette("YlOrRd", as_cmap=True)
        xlabels = [LABEL_MAP.get(c, c) for c in available_cols]
        sns.heatmap(
            norm_data, annot=annot_data, fmt='g',
            cmap=cmap, linewidths=0.5, linecolor='white', ax=ax,
            cbar_kws={'label': 'Normalized Score', 'shrink': 0.6},
            xticklabels=xlabels,
            annot_kws={'fontsize': 9},
        )

        n_metrics = len([c for c in available_cols if c not in score_cols])
        if n_metrics < n_cols:
            ax.axvline(x=n_metrics, color='black', linewidth=2)

        ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
        ax.set_ylabel('Method', fontsize=12)
        ax.set_xlabel('')
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")

    def _load_csv(pattern_list):
        for fname in pattern_list:
            fpath = os.path.join(summary_dir, fname)
            if os.path.isfile(fpath):
                df = pd.read_csv(fpath)
                df.columns = [c.replace(' ', '_') for c in df.columns]
                # Unify BVC_Score → BioC_Score
                if 'BVC_Score' in df.columns and 'BioC_Score' not in df.columns:
                    df.rename(columns={'BVC_Score': 'BioC_Score'}, inplace=True)
                return df
        return None

    def _make_heatmap(df, ds_list, metric_cols, score_cols, method_order,
                      title, out_path):
        df_sub = df[df['Dataset'].isin(ds_list)]
        if df_sub.empty:
            return
        available = [c for c in metric_cols + score_cols if c in df_sub.columns]
        if not available:
            return
        method_avg = df_sub.groupby('Method')[available].mean()
        methods_present = [m for m in method_order if m in method_avg.index]
        method_avg = method_avg.loc[methods_present]
        sort_col = score_cols[-1] if score_cols[-1] in method_avg.columns else available[-1]
        method_avg = method_avg.sort_values(sort_col, ascending=False)
        _plot_heatmap(method_avg, available, score_cols, title, out_path)

    for clustering in ['leiden', 'louvain', 'kmeans', 'mclust']:
        # --- Fig 2a: Vertical ---
        df_v = _load_csv([f'vertical_final_{clustering}.csv',
                          f'vertical_detailed_{clustering}.csv'])
        if df_v is None:
            print(f"  [SKIP] No vertical summary for {clustering}")
            continue

        # Main figure: withGT
        for mod_label, ds_list in [('RNA+ADT', RNA_ADT_WITHGT),
                                   ('RNA+ATAC', RNA_ATAC_WITHGT)]:
            title = f'Vertical Integration \u2014 {mod_label} ({clustering})'
            out_path = os.path.join(out_dir,
                f'fig2a_vertical_heatmap_{mod_label}_withGT_{clustering}.pdf')
            _make_heatmap(df_v, ds_list, METRIC_COLS_WITHGT, SCORE_COLS_V,
                          METHOD_ORDER_V, title, out_path)

        # Supplementary: woGT
        for mod_label, ds_list in [('RNA+ADT', RNA_ADT_WOGT),
                                   ('RNA+ATAC', RNA_ATAC_WOGT)]:
            title = f'Vertical Integration \u2014 {mod_label} (woGT, {clustering})'
            out_path = os.path.join(out_dir,
                f'supp_vertical_heatmap_{mod_label}_woGT_{clustering}.pdf')
            _make_heatmap(df_v, ds_list, METRIC_COLS_WOGT, SCORE_COLS_V,
                          METHOD_ORDER_V, title, out_path)

        # --- Fig 3a: Horizontal ---
        df_h = _load_csv([f'horizontal_final_{clustering}.csv',
                          f'horizontal_detailed_{clustering}.csv'])
        if df_h is None:
            print(f"  [SKIP] No horizontal summary for {clustering}")
            continue

        # Main figure: withGT
        for mod_label, ds_list in [('RNA+ADT', RNA_ADT_WITHGT),
                                   ('RNA+ATAC', RNA_ATAC_WITHGT)]:
            title = f'Horizontal Integration \u2014 {mod_label} ({clustering})'
            out_path = os.path.join(out_dir,
                f'fig3a_horizontal_heatmap_{mod_label}_withGT_{clustering}.pdf')
            _make_heatmap(df_h, ds_list, METRIC_COLS_H_WITHGT, SCORE_COLS_H,
                          METHOD_ORDER_H, title, out_path)

        # Supplementary: woGT
        for mod_label, ds_list in [('RNA+ADT', RNA_ADT_WOGT),
                                   ('RNA+ATAC', RNA_ATAC_WOGT)]:
            title = f'Horizontal Integration \u2014 {mod_label} (woGT, {clustering})'
            out_path = os.path.join(out_dir,
                f'supp_horizontal_heatmap_{mod_label}_woGT_{clustering}.pdf')
            _make_heatmap(df_h, ds_list, METRIC_COLS_H_WOGT, SCORE_COLS_H,
                          METHOD_ORDER_H, title, out_path)

    print("\nAll heatmaps regenerated.")


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    reaggregate(root)
    merge_cmgtc_and_build_final(root)
    regen_heatmaps(root, args.dpi)


if __name__ == '__main__':
    main()
