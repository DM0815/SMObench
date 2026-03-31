from pathlib import Path
from itertools import cycle

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / 'Results' / 'adata' / 'vertical_integration' / 'train_time_summary.csv'
OUTPUT_DIR = BASE_DIR / 'Results' / 'train_time'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HIGH_METHODS = [
    'CANDIES',
    'SpaMultiVAE',
    'PRAGA',
]

LOW_METHODS = [
    'SpatialGlue',
    'SpaFusion',
    'SpaMosaic',
    'SpaMI',
    'SpaBalance',
    'SpaBalance_3M',
    'COSMOS',
    'PRESENT',
    'SpaMV',
    'SMOPCA',
]

df = pd.read_csv(CSV_PATH)
train_df = df.dropna(subset=['train_time']).copy()

ALL_METHODS = HIGH_METHODS + [m for m in LOW_METHODS if m not in HIGH_METHODS]
remaining_methods = [
    m for m in train_df['method'].unique() if m not in ALL_METHODS
]
ALL_METHODS.extend(sorted(remaining_methods))

base_method = 'CANDIES'
base_order = train_df[train_df['method'] == base_method].copy()
if base_order.empty:
    raise ValueError(f"Base method '{base_method}' not found in train_time_summary.csv")

base_order = base_order.sort_values('n_spots').reset_index(drop=True)
position_map = {
    (row['dataset'], row['subset']): idx for idx, row in base_order.iterrows()
}
tick_labels = [f"{int(row['n_spots'])}" for _, row in base_order.iterrows()]

palette = plt.get_cmap('tab20', max(len(ALL_METHODS), 1))
color_map = {
    method: palette(idx)
    for idx, method in enumerate(ALL_METHODS)
}


def plot_methods(methods, title, filename, ylabel, color_map=None):
    fig, ax = plt.subplots(figsize=(11, 6))

    color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    for method in methods:
        method_df = train_df[train_df['method'] == method].copy()
        if method_df.empty:
            continue
        method_df['pos'] = method_df.apply(
            lambda row: position_map.get((row['dataset'], row['subset'])), axis=1
        )
        method_df = method_df.dropna(subset=['pos'])
        if method_df.empty:
            continue
        method_df = method_df.sort_values('pos')
        color = None
        if color_map:
            color = color_map.get(method)
        if color is None:
            color = next(color_cycle)
        ax.plot(
            method_df['pos'],
            method_df['train_time'],
            marker='o',
            markersize=4,
            linewidth=1.2,
            label=method,
            color=color
        )

    ax.set_xlabel('Number of Spots', fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(title)

    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlim(-0.5, len(base_order) - 0.5)
    ax.set_xticks(range(len(base_order)))
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=12)
    ax.tick_params(axis='y', labelsize=12)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='best', ncol=1)
    plt.tight_layout()

    out_path = OUTPUT_DIR / filename
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


plot_methods(
    HIGH_METHODS,
    'Computational Scalability (High Training Time Methods)',
    'train_time_high.png',
    'Average Training Time (s)',
    color_map=color_map,
)

plot_methods(
    LOW_METHODS,
    'Computational Scalability (Low Training Time Methods)',
    'train_time_low.png',
    'Average Training Time (s)',
    color_map=color_map,
)

plot_methods(
    ALL_METHODS,
    'Computational Scalability (All Methods)',
    'train_time_all.png',
    'Average Training Time (s)',
    color_map=color_map,
)
