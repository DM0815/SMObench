"""
SMOBench global figure style — aligned with TEAM paper (Nature Medicine).

Import this at the top of any plotting script:
    from style_config import apply_style, COLORS

Then call apply_style() before creating figures.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl


# ── 13-color palette (from user reference) ──────────────────────────────────
PAL13 = [
    '#e75b58',  #  0: coral red
    '#e49ac3',  #  1: pink
    '#ab3181',  #  2: dark magenta
    '#20452e',  #  3: dark green
    '#bb9369',  #  4: tan/brown
    '#8c529a',  #  5: purple
    '#e4d1dc',  #  6: light mauve
    '#52a65e',  #  7: green
    '#f0b971',  #  8: golden orange
    '#f2b09e',  #  9: peach/salmon
    '#d4e6a1',  # 10: light lime
    '#55c0f2',  # 11: sky blue
    '#496d87',  # 12: dark teal
]

# Convenience alias
PAL = PAL13

COLORS = {
    'black':       '#000000',
    'gray_dark':   '#444444',
    'gray':        '#888888',
    'gray_light':  '#CCCCCC',
    'white':       '#FFFFFF',
}

# Metric-group colors for dot-matrix heatmap (3 groups clearly distinct)
METRIC_COLORS = {
    'SC':    '#e75b58',    # coral red — Spatial Coherence
    'BioC':  '#52a65e',    # green — Biological Conservation
    'BER':   '#8c529a',    # purple — Batch Effect Removal
    'CMGTC': '#bb9369',    # tan — CM-GTC
    'agg':   '#496d87',    # dark teal — Aggregate scores
}


def apply_style():
    """Apply global matplotlib style for all SMOBench figures."""
    plt.rcParams.update({
        # Font — larger for publication
        'font.family':         'sans-serif',
        'font.sans-serif':     ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size':           11,

        # Axes — all black
        'axes.linewidth':      0.8,
        'axes.edgecolor':      '#000000',
        'axes.labelcolor':     '#000000',
        'axes.labelsize':      12,
        'axes.titlesize':      13,
        'axes.titleweight':    'bold',
        'axes.spines.top':     False,
        'axes.spines.right':   False,

        # Ticks
        'xtick.major.width':   0.4,
        'ytick.major.width':   0.4,
        'xtick.major.size':    4,
        'ytick.major.size':    4,
        'xtick.labelsize':     10,
        'ytick.labelsize':     10,
        'xtick.color':         '#000000',
        'ytick.color':         '#000000',

        # Grid
        'axes.grid':           False,
        'grid.linewidth':      0.3,
        'grid.color':          COLORS['gray_light'],
        'grid.alpha':          0.5,

        # Lines
        'lines.linewidth':     1.0,
        'lines.markersize':    5,

        # Legend
        'legend.fontsize':     10,
        'legend.frameon':      False,
        'legend.borderpad':    0.3,

        # Figure
        'figure.facecolor':    'white',
        'figure.dpi':          300,
        'savefig.dpi':         300,
        'savefig.bbox':        'tight',
        'savefig.pad_inches':  0.05,
        'savefig.facecolor':   'white',

        # Scatter
        'scatter.edgecolors':  'none',
    })
