#!/usr/bin/env python3
"""
Plot 3M (triple-modality) vertical integration results as a grouped bar chart.
Output: figure/3M_vertical.pdf for the manuscript.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Data (Leiden clustering, 3M_Simulation_v2) ──
methods = ['SpatialGlue', 'SpaBalance', 'SMOPCA', 'MISO', 'PRESENT', 'SpaMV', 'PRAGA']

data = {
    'ARI':      [0.488, 0.453, 0.386, 0.376, 0.001, 0.214, 0.079],
    'NMI':      [0.507, 0.499, 0.395, 0.410, 0.005, 0.306, 0.106],
    "Moran's I":[0.767, 0.697, 0.623, 0.276, 0.446, 0.062, 0.148],
    'Vertical': [63.52, 60.21, 53.57, 36.05, 34.84, 22.22, 22.22],
}

# ── Figure: 2-panel layout ──
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={'width_ratios': [3, 1.2]})

# ── Left panel: grouped bar chart for ARI, NMI, Moran's I ──
ax = axes[0]
metrics = ['ARI', 'NMI', "Moran's I"]
n_methods = len(methods)
n_metrics = len(metrics)
x = np.arange(n_methods)
bar_width = 0.22
colors = ['#4C72B0', '#55A868', '#C44E52']

for i, metric in enumerate(metrics):
    vals = data[metric]
    bars = ax.bar(x + i * bar_width, vals, bar_width, label=metric, color=colors[i], edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, vals):
        if v > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=7, rotation=0)

ax.set_xticks(x + bar_width)
ax.set_xticklabels(methods, rotation=25, ha='right', fontsize=9)
ax.set_ylabel('Score', fontsize=11)
ax.set_ylim(0, 1.0)
ax.legend(fontsize=9, loc='upper right')
ax.set_title('3M Vertical Integration: Per-metric Comparison', fontsize=11, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── Right panel: horizontal bar chart for Vertical score ──
ax2 = axes[1]
sorted_idx = np.argsort(data['Vertical'])
sorted_methods = [methods[i] for i in sorted_idx]
sorted_scores = [data['Vertical'][i] for i in sorted_idx]
bar_colors = ['#4C72B0' if s >= 50 else '#8DA0CB' if s >= 30 else '#C6DBEF' for s in sorted_scores]

bars = ax2.barh(range(n_methods), sorted_scores, color=bar_colors, edgecolor='white', linewidth=0.5)
ax2.set_yticks(range(n_methods))
ax2.set_yticklabels(sorted_methods, fontsize=9)
ax2.set_xlabel('Vertical Score', fontsize=10)
ax2.set_title('Overall Ranking', fontsize=11, fontweight='bold')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

for bar, v in zip(bars, sorted_scores):
    ax2.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2,
             f'{v:.1f}', ha='left', va='center', fontsize=8)

ax2.set_xlim(0, max(sorted_scores) * 1.2)

plt.tight_layout()

# ── Save ──
out_dir = '/home/users/nus/e1724738/_main/_private/NUS/_Proj1/writing/NeurIPS_2025/figure'
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, '3M_vertical.pdf')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f'Saved: {out_path}')

# Also save PNG for preview
plt.savefig(out_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
print(f'Saved: {out_path.replace(".pdf", ".png")}')
