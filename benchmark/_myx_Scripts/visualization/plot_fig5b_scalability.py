"""
Fig 5b: Scalability — runtime & GPU memory vs dataset size.
2×2 layout: Row 1 = RNA+ADT (Runtime | GPU Memory), Row 2 = RNA+ATAC (Runtime | GPU Memory).
Legend on top, wide aspect ratio.
CPU-only methods (SMOPCA, PRESENT) excluded from GPU Memory panels.
"""
import json, os, glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(__file__))
from style_config import apply_style, PAL13
apply_style()

BASE = "/data/projects/11003054/e1724738/_private/NUS/_Proj1/SMOBench-CLEAN/_myx_Results/scalability/results"
OUT_DIR = "/data/projects/11003054/e1724738/_private/NUS/_Proj1/SMOBench-CLEAN/_myx_Results/plots"

CPU_ONLY_METHODS = {'SMOPCA', 'PRESENT'}  # no meaningful GPU computation (peak ≤ 5 MB)
MIN_POINTS = 2  # minimum data points to draw a line

# Collect
records = []
for method_dir in sorted(os.listdir(BASE)):
    method_path = os.path.join(BASE, method_dir)
    if not os.path.isdir(method_path):
        continue
    for jf in glob.glob(f"{method_path}/**/*_scalability.json", recursive=True):
        try:
            with open(jf) as f:
                d = json.load(f)
            records.append({
                'Method': d.get('method', method_dir),
                'modality': d.get('modality', ''),
                'n_cells': d.get('n_cells', 0),
                'wall_time_s': d.get('wall_time_s', d.get('total_time_s', np.nan)),
                'peak_gpu_mb': d.get('peak_gpu_memory_mb', np.nan),
                'success': d.get('success', True),
            })
        except:
            pass

df = pd.DataFrame(records)
print(f"Collected {len(df)} records from {df['Method'].nunique()} methods")
df = df[df['success'] == True]

agg = df.groupby(['Method', 'modality', 'n_cells']).agg({
    'wall_time_s': 'median',
    'peak_gpu_mb': 'median',
}).reset_index()

# Stable color/marker assignment across all methods
all_methods = sorted(agg['Method'].unique())
palette = PAL13 + ['#333333', '#777777']
method_colors = {m: palette[i % len(palette)] for i, m in enumerate(all_methods)}
markers_list = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', '<', '>', '8', 'P', 'H', 'X', 'd']
method_markers = {m: markers_list[i % len(markers_list)] for i, m in enumerate(all_methods)}

# --- 2×2 figure: rows=modality, cols=metric ---
fig, axes = plt.subplots(2, 2, figsize=(16, 8))
ax_rt_adt, ax_gm_adt = axes[0]   # Row 1: ADT
ax_rt_atac, ax_gm_atac = axes[1]  # Row 2: ATAC

modalities = [
    ('RNA_ADT', 'RNA+ADT (Mouse Thymus)', ax_rt_adt, ax_gm_adt),
    ('RNA_ATAC', 'RNA+ATAC (Mouse Brain)', ax_rt_atac, ax_gm_atac),
]

all_handles, all_labels = [], []

for mod_key, mod_title, ax_rt, ax_gm in modalities:
    sub_mod = agg[agg['modality'] == mod_key]
    mod_methods = sorted(sub_mod['Method'].unique())

    # --- Runtime ---
    for m in mod_methods:
        sub = sub_mod[sub_mod['Method'] == m].sort_values('n_cells')
        if len(sub) < MIN_POINTS:
            continue
        line, = ax_rt.plot(sub['n_cells'], sub['wall_time_s'],
                           marker=method_markers[m], color=method_colors[m],
                           label=m, markersize=5, linewidth=1.8, alpha=0.9)
        if m not in all_labels:
            all_handles.append(line)
            all_labels.append(m)

    ax_rt.set_xlabel('Number of Spots')
    ax_rt.set_ylabel('Runtime (seconds)')
    ax_rt.set_title(f'Runtime — {mod_title}')
    ax_rt.set_yscale('log')
    max_n = sub_mod['n_cells'].max() if len(sub_mod) > 0 else 40000
    ax_rt.axhline(3600, color='#AAAAAA', ls=':', lw=1.0)
    ax_rt.text(max_n * 0.85, 4500, '1 h', fontsize=8, color='#888888', ha='right')

    # --- GPU Memory (exclude CPU-only) ---
    sub_gpu = sub_mod[~sub_mod['Method'].isin(CPU_ONLY_METHODS)]
    gpu_methods = sorted(sub_gpu['Method'].unique())

    for m in gpu_methods:
        sub = sub_gpu[sub_gpu['Method'] == m].sort_values('n_cells')
        if len(sub) < MIN_POINTS or sub['peak_gpu_mb'].max() <= 5:
            continue
        ax_gm.plot(sub['n_cells'], sub['peak_gpu_mb'],
                   marker=method_markers[m], color=method_colors[m],
                   label=m, markersize=5, linewidth=1.8, alpha=0.9)

    ax_gm.set_xlabel('Number of Spots')
    ax_gm.set_ylabel('Peak GPU Memory (MB)')
    ax_gm.set_title(f'GPU Memory — {mod_title}')
    ax_gm.set_yscale('log')
    ax_gm.axhline(40000, color='#AAAAAA', ls=':', lw=1.0)
    ax_gm.text(max_n * 0.85, 47000, '40 GB', fontsize=8, color='#888888', ha='right')

# --- Legend on top ---
plt.tight_layout()

# Legend: 2 rows, 1.3x larger, width matches full figure (ylabel to right edge)
n_labels = len(all_labels)
ncol = (n_labels + 1) // 2  # 2 rows

# Get figure-coordinate x positions from axes to align legend width
left_edge = axes[0, 0].get_position().x0   # left ylabel edge
right_edge = axes[0, 1].get_position().x1  # right plot edge
center_x = (left_edge + right_edge) / 2

leg = fig.legend(all_handles, all_labels, loc='upper center',
                 bbox_to_anchor=(center_x, 1.06),
                 fontsize=12.5, frameon=False, ncol=ncol,
                 handlelength=2.5, columnspacing=1.2,
                 handletextpad=0.5, borderaxespad=0, markerscale=1.5)

out = f"{OUT_DIR}/fig5b_scalability_time_memory"
fig.savefig(out + '.pdf', bbox_inches='tight', facecolor='white')
fig.savefig(out + '.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out}.pdf")
