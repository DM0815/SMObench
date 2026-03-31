#!/usr/bin/env python3
"""
Mosaic CM-GTC Transfer Analysis — Paired Bar Chart

For each dataset × scenario, compare per-modality CM-GTC between:
  - Bridge batches (had all modalities during integration)
  - Query batch (one modality was HIDDEN from SpaMosaic)

The hidden modality's CM-GTC on the query batch measures whether
SpaMosaic successfully transferred topological information from bridge batches.

Layout:
  Top row:    "Without RNA" scenario    (7 datasets)
  Bottom row: "Without ADT/ATAC" scenario (7 datasets)
  Each panel: grouped bars (RNA vs ADT/ATAC), bridge vs query side by side
  Hidden modality bar gets a ★ marker and dashed edge

Usage:
    python plot_mosaic_cmgtc_transfer.py --root /path/to/SMOBench-CLEAN
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))
from style_config import apply_style, METRIC_COLORS
apply_style()

DATASETS_ORDER = [
    ('HLN',          'Human\nLymph'),
    ('HT',           'Human\nTonsils'),
    ('Mouse_Spleen', 'Mouse\nSpleen'),
    ('MISAR_S1',     'Mouse\nEmbryos(S1)'),
    ('MISAR_S2',     'Mouse\nEmbryos(S2)'),
    ('Mouse_Thymus', 'Mouse\nThymus'),
    ('Mouse_Brain',  'Mouse\nBrain'),
]

SCENARIOS = ['without_rna', 'without_second']
SCENARIO_LABELS = ['Without RNA', 'Without ADT/ATAC']

# Colors
CLR_RNA = '#e75b58'       # coral red (SC color family)
CLR_SECOND = '#52a65e'    # green (BioC color family)
CLR_BRIDGE = 0.9          # high alpha for bridge
CLR_QUERY = 0.65          # lower alpha for query


def load_per_slice(root):
    csv_path = os.path.join(root, '_myx_Results', 'evaluation', 'mosaic',
                             'mosaic_cmgtc_per_slice.csv')
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Not found: {csv_path}")
    return pd.read_csv(csv_path)


def plot_transfer(df, out_path, dpi=300):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    n_datasets = len(DATASETS_ORDER)
    fig, axes = plt.subplots(2, 1, figsize=(14, 6.5), sharex=False)

    for row_idx, (scenario, scenario_label) in enumerate(zip(SCENARIOS, SCENARIO_LABELS)):
        ax = axes[row_idx]
        df_sc = df[df['Scenario'] == scenario]

        # Determine modality columns — unify ADT/ATAC into a single "second" column
        rna_col = 'CM_GTC_rna'
        # Merge CM_GTC_adt and CM_GTC_atac into one "second modality" column
        if 'CM_GTC_adt' in df_sc.columns and 'CM_GTC_atac' in df_sc.columns:
            df_sc = df_sc.copy()
            df_sc['CM_GTC_second'] = df_sc['CM_GTC_adt'].fillna(df_sc['CM_GTC_atac'])
        elif 'CM_GTC_adt' in df_sc.columns:
            df_sc = df_sc.copy()
            df_sc['CM_GTC_second'] = df_sc['CM_GTC_adt']
        elif 'CM_GTC_atac' in df_sc.columns:
            df_sc = df_sc.copy()
            df_sc['CM_GTC_second'] = df_sc['CM_GTC_atac']
        else:
            df_sc = df_sc.copy()
            df_sc['CM_GTC_second'] = np.nan
        sec_col = 'CM_GTC_second'

        x_positions = np.arange(n_datasets)
        bar_width = 0.18
        offsets = [-1.5, -0.5, 0.5, 1.5]  # RNA_bridge, RNA_query, SEC_bridge, SEC_query

        for di, (ds_key, ds_label) in enumerate(DATASETS_ORDER):
            df_ds = df_sc[df_sc['Dataset'] == ds_key]
            if df_ds.empty:
                continue

            bridge = df_ds[df_ds['Role'] == 'bridge']
            query = df_ds[df_ds['Role'] == 'query']

            # RNA bars
            rna_bridge = bridge[rna_col].mean() if not bridge.empty and rna_col in bridge else np.nan
            rna_query = query[rna_col].mean() if not query.empty and rna_col in query else np.nan

            # Second modality bars
            if sec_col and sec_col in df_sc.columns:
                sec_bridge = bridge[sec_col].mean() if not bridge.empty and sec_col in bridge.columns else np.nan
                sec_query = query[sec_col].mean() if not query.empty and sec_col in query.columns else np.nan
            else:
                sec_bridge = sec_query = np.nan

            vals = [rna_bridge, rna_query, sec_bridge, sec_query]
            colors = [CLR_RNA, CLR_RNA, CLR_SECOND, CLR_SECOND]
            alphas = [CLR_BRIDGE, CLR_QUERY, CLR_BRIDGE, CLR_QUERY]
            labels_short = ['B', 'Q', 'B', 'Q']

            for bi, (val, color, alpha, lbl) in enumerate(zip(vals, colors, alphas, labels_short)):
                if pd.isna(val):
                    continue
                x = x_positions[di] + offsets[bi] * bar_width

                # Determine if this is the hidden modality on query batch
                is_hidden = False
                if bi == 1 and scenario == 'without_rna':  # RNA query in without_rna
                    is_hidden = True
                elif bi == 3 and scenario == 'without_second':  # SEC query in without_second
                    is_hidden = True

                edgecolor = '#FFD700' if is_hidden else 'white'
                linestyle = '--' if is_hidden else '-'
                linewidth = 2.0 if is_hidden else 0.5
                hatch = '///' if is_hidden else None

                bar = ax.bar(x, val, bar_width * 0.95,
                             color=color, alpha=alpha,
                             edgecolor=edgecolor, linewidth=linewidth,
                             linestyle=linestyle, hatch=hatch, zorder=3)

                # Value text on top (always show)
                ax.text(x, val + 0.01, f'{val:.2f}',
                        ha='center', va='bottom', fontsize=5.5,
                        color='#333333', fontweight='bold' if is_hidden else 'normal')

                # Star marker for hidden modality
                if is_hidden:
                    ax.text(x, val + 0.045, '★', ha='center', va='bottom',
                            fontsize=8, color='#FFD700', fontweight='bold', zorder=5)

        # Axis formatting
        ax.set_xticks(x_positions)
        ax.set_xticklabels([dl for _, dl in DATASETS_ORDER], fontsize=9)
        ax.set_ylabel('CM-GTC', fontsize=11)
        ax.set_ylim(0, min(ax.get_ylim()[1] * 1.15, 0.7))
        ax.set_title(scenario_label, fontsize=13, fontweight='bold', pad=8)
        ax.axhline(y=0, color='black', linewidth=0.5)

        # Add sub-labels for RNA / Second modality groups
        for di in range(n_datasets):
            x_rna = x_positions[di] + (-1) * bar_width
            x_sec = x_positions[di] + (1) * bar_width
            y_label = -0.04 * ax.get_ylim()[1]

        # Grid
        ax.yaxis.grid(True, alpha=0.3, linewidth=0.3)
        ax.set_axisbelow(True)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=CLR_RNA, alpha=CLR_BRIDGE, label='RNA (bridge)'),
        mpatches.Patch(facecolor=CLR_RNA, alpha=CLR_QUERY, label='RNA (query)'),
        mpatches.Patch(facecolor=CLR_SECOND, alpha=CLR_BRIDGE, label='ADT/ATAC (bridge)'),
        mpatches.Patch(facecolor=CLR_SECOND, alpha=CLR_QUERY, label='ADT/ATAC (query)'),
        mpatches.Patch(facecolor='#EEEEEE', edgecolor='#FFD700', linewidth=2,
                       linestyle='--', hatch='///', label='★ Hidden modality\n(never seen by SpaMosaic)'),
    ]
    fig.legend(handles=legend_elements, loc='upper center',
               ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, 1.02),
               frameon=False, handlelength=1.5)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.savefig(out_path.replace('.pdf', '.png'), dpi=dpi, bbox_inches='tight',
                facecolor='white', pad_inches=0.1)
    plt.close()
    print(f'Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    df = load_per_slice(root)
    print(f"Loaded {len(df)} rows")

    # Print summary
    for scenario in SCENARIOS:
        df_sc = df[df['Scenario'] == scenario]
        print(f"\n--- {scenario} ---")
        for ds_key, _ in DATASETS_ORDER:
            df_ds = df_sc[df_sc['Dataset'] == ds_key]
            if df_ds.empty:
                continue
            bridge = df_ds[df_ds['Role'] == 'bridge']
            query = df_ds[df_ds['Role'] == 'query']
            mod_cols = [c for c in df.columns if c.startswith('CM_GTC_') and c != 'CM_GTC_global']
            parts = [f"  {ds_key}:"]
            for mc in mod_cols:
                b = bridge[mc].mean() if mc in bridge and not bridge.empty else float('nan')
                q = query[mc].mean() if mc in query and not query.empty else float('nan')
                mod_label = mc.replace('CM_GTC_', '')
                parts.append(f"    {mod_label}: B={b:.3f} Q={q:.3f}")
            print('\n'.join(parts))

    if args.output:
        out = args.output
    else:
        out = os.path.join(root, '_myx_Results', 'plots', 'mosaic_cmgtc_transfer.pdf')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plot_transfer(df, out)


if __name__ == '__main__':
    main()
