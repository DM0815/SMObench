"""
Generate Fig 3b,c: BER vs BVC scatter plots for horizontal integration.
Uses global style_config palette + CM-GTC color encoding.
"""
import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from adjustText import adjust_text

sys.path.insert(0, os.path.dirname(__file__))
from style_config import apply_style, PAL13
apply_style()

SUMMARY_DIR = "/data/projects/11003054/e1724738/_private/NUS/_Proj1/SMOBench-CLEAN/_myx_Results/evaluation/summary"
OUT_DIR = "/data/projects/11003054/e1724738/_private/NUS/_Proj1/SMOBench-CLEAN/_myx_Results/plots"

WITHGT_DATASETS = ['Human_Lymph_Nodes', 'Human_Tonsils', 'Mouse_Embryos_S1', 'Mouse_Embryos_S2']
WOGT_DATASETS = ['Mouse_Thymus', 'Mouse_Spleen', 'Mouse_Brain']

df = pd.read_csv(f"{SUMMARY_DIR}/horizontal_final_leiden.csv")

# CM-GTC colormap (same as fig2bc)
cmap = mcolors.LinearSegmentedColormap.from_list(
    'cmgtc', [PAL13[12], PAL13[11], PAL13[8], PAL13[0]], N=256)

def make_summary(df_sub):
    records = []
    for m in df_sub['Method'].unique():
        sub = df_sub[df_sub['Method'] == m]
        records.append({
            'Method': m,
            'BVC_mean': np.nanmean(sub['BVC_Score']),
            'BVC_std': np.nanstd(sub['BVC_Score']),
            'BER_mean': np.nanmean(sub['BER_Score']),
            'BER_std': np.nanstd(sub['BER_Score']),
            'CM_GTC_mean': np.nanmean(sub['CM_GTC']),
        })
    return pd.DataFrame(records)

def plot_ber_bvc(ax, data, title):
    """Text labels with adjustText + leader lines for crowded areas."""
    cmgtc = data['CM_GTC_mean'].values
    cmin, cmax = cmgtc.min(), cmgtc.max()
    crange = cmax - cmin if cmax > cmin else 1.0

    ax.axhline(data['BER_mean'].median(), color='#CCCCCC', ls='--', lw=0.5, zorder=1)
    ax.axvline(data['BVC_mean'].median(), color='#CCCCCC', ls='--', lw=0.5, zorder=1)

    # Manual offsets for crowded methods (applied to ALL panels)
    # Positive dx = right, positive dy = up (in offset points)
    MANUAL_OFFSETS = {
        'CANDIES':     (-60, -25),
        'COSMOS':      (10, 12),
        'SpaBalance':  (-65, 18),
        'SpatialGlue': (-70, 5),
        'SpaMI':       (10, 10),
        'SMOPCA':      (-55, 10),
        'SpaMV':       (10, -12),
        'SpaMultiVAE': (10, -10),
        'MISO':        (10, -8),
        'SpaFusion':   (10, 8),
        'PRAGA':       (10, -10),
        'PRESENT':     (-55, -8),
        'SpaMosaic':   (10, 8),
    }

    for _, row in data.iterrows():
        norm = (row['CM_GTC_mean'] - cmin) / crange
        color = cmap(norm)
        ax.errorbar(row['BVC_mean'], row['BER_mean'],
                    xerr=row['BVC_std'] * 0.5, yerr=row['BER_std'] * 0.5,
                    fmt='none', ecolor=color, alpha=0.5, capsize=2, linewidth=0.8, zorder=2)
        ax.scatter(row['BVC_mean'], row['BER_mean'], s=100, c=[color],
                   alpha=0.85, edgecolors='white', linewidth=0.4, zorder=5)

        method = row['Method']
        offset = MANUAL_OFFSETS.get(method, (8, 6))
        needs_arrow = method in MANUAL_OFFSETS and (abs(offset[0]) > 20 or abs(offset[1]) > 20)

        ax.annotate(method, (row['BVC_mean'], row['BER_mean']),
                    xytext=offset, textcoords='offset points',
                    fontsize=8, color='#000000',
                    arrowprops=dict(arrowstyle='-', color='#999999', lw=0.5)
                    if needs_arrow else None)

    ax.set_xlabel('BVC Score (Biological Conservation)', fontsize=12)
    ax.set_ylabel('BER Score (Batch Effect Removal)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold', color='#000000')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=cmin, vmax=cmax))
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, shrink=0.7, pad=0.02, aspect=25)
    cbar.set_label('CM-GTC', fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(0.4)

# ── withGT (main figure) ──────────────────────────────────────────────────
df_withgt = df[df['Dataset'].isin(WITHGT_DATASETS)]
df_adt = df_withgt[df_withgt['Dataset_Type'] == 'RNA_ADT']
df_atac = df_withgt[df_withgt['Dataset_Type'] == 'RNA_ATAC']
sum_adt = make_summary(df_adt)
sum_atac = make_summary(df_atac)

# Fig 3b: RNA+ADT (withGT)
fig, ax = plt.subplots(figsize=(7, 5.5))
plot_ber_bvc(ax, sum_adt, 'RNA+ADT')
fig.savefig(f"{OUT_DIR}/fig3b_ber_bvc_RNA_ADT_leiden.pdf", bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig3b (pdf)")

# Fig 3c: RNA+ATAC (withGT)
fig, ax = plt.subplots(figsize=(7, 5.5))
plot_ber_bvc(ax, sum_atac, 'RNA+ATAC')
fig.savefig(f"{OUT_DIR}/fig3c_ber_bvc_RNA_ATAC_leiden.pdf", bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig3c (pdf)")

# Combined (withGT)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
plot_ber_bvc(ax1, sum_adt, 'RNA+ADT')
plot_ber_bvc(ax2, sum_atac, 'RNA+ATAC')
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig3bc_ber_bvc_combined_leiden.pdf", bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig3bc combined (pdf)")

# ── woGT (supplementary) ─────────────────────────────────────────────────
df_wogt = df[df['Dataset'].isin(WOGT_DATASETS)]
df_adt_wo = df_wogt[df_wogt['Dataset_Type'] == 'RNA_ADT']
df_atac_wo = df_wogt[df_wogt['Dataset_Type'] == 'RNA_ATAC']
sum_adt_wo = make_summary(df_adt_wo)
sum_atac_wo = make_summary(df_atac_wo)

# Supp fig3b: RNA+ADT (woGT)
if not sum_adt_wo.empty:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    plot_ber_bvc(ax, sum_adt_wo, 'RNA+ADT (woGT)')
    fig.savefig(f"{OUT_DIR}/supp_fig3b_ber_bvc_RNA_ADT_woGT_leiden.pdf", bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved supp_fig3b woGT (pdf)")

# Supp fig3c: RNA+ATAC (woGT)
if not sum_atac_wo.empty:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    plot_ber_bvc(ax, sum_atac_wo, 'RNA+ATAC (woGT)')
    fig.savefig(f"{OUT_DIR}/supp_fig3c_ber_bvc_RNA_ATAC_woGT_leiden.pdf", bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved supp_fig3c woGT (pdf)")

# Combined (woGT)
wo_panels = [(sum_adt_wo, 'RNA+ADT (woGT)'), (sum_atac_wo, 'RNA+ATAC (woGT)')]
wo_panels = [(s, t) for s, t in wo_panels if not s.empty]
if wo_panels:
    fig, axes = plt.subplots(1, len(wo_panels), figsize=(7.5*len(wo_panels), 5.5))
    if len(wo_panels) == 1:
        axes = [axes]
    for ax, (s, t) in zip(axes, wo_panels):
        plot_ber_bvc(ax, s, t)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/supp_fig3bc_ber_bvc_combined_woGT_leiden.pdf", bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved supp_fig3bc combined woGT (pdf)")
