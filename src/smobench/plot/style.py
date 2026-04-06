"""Publication-quality style configuration for SMObench figures."""

import matplotlib.pyplot as plt

# 15-method color palette
METHOD_COLORS = {
    'SpatialGlue':  '#e75b58',
    'SpaMosaic':    '#e49ac3',
    'PRAGA':        '#ab3181',
    'COSMOS':       '#20452e',
    'PRESENT':      '#bb9369',
    'CANDIES':      '#8c529a',
    'MISO':         '#e4d1dc',
    'MultiGATE':    '#52a65e',
    'SMOPCA':       '#f0b971',
    'SpaBalance':   '#f2b09e',
    'SpaFusion':    '#d4e6a1',
    'SpaMI':        '#55c0f2',
    'spaMultiVAE':  '#496d87',
    'SpaMV':        '#e75b58',
    'SWITCH':       '#8c529a',
}

PAL15 = list(METHOD_COLORS.values())

# Extended palette for >15 methods (auto-generated from matplotlib tab20 + Set2)
_EXTRA_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
]


def get_method_color(method_name: str, index: int = 0) -> str:
    """Get color for a method. Known methods get fixed colors, new ones get auto-assigned."""
    if method_name in METHOD_COLORS:
        return METHOD_COLORS[method_name]
    # Auto-assign from extended palette
    all_colors = PAL15 + _EXTRA_COLORS
    return all_colors[index % len(all_colors)]


# Metric group colors
METRIC_COLORS = {
    'SC':    '#e75b58',
    'BioC':  '#52a65e',
    'BVC':   '#52a65e',
    'BER':   '#8c529a',
    'CMGTC': '#bb9369',
    'agg':   '#496d87',
}

COLORS = {
    'black':       '#000000',
    'gray_dark':   '#444444',
    'gray':        '#888888',
    'gray_light':  '#CCCCCC',
    'white':       '#FFFFFF',
}


def apply_style():
    """Apply publication-quality matplotlib style."""
    plt.rcParams.update({
        'font.family':         'sans-serif',
        'font.sans-serif':     ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size':           11,
        'axes.linewidth':      0.8,
        'axes.edgecolor':      '#000000',
        'axes.labelsize':      12,
        'axes.titlesize':      13,
        'axes.titleweight':    'bold',
        'axes.spines.top':     False,
        'axes.spines.right':   False,
        'axes.grid':           False,
        'xtick.major.width':   0.4,
        'ytick.major.width':   0.4,
        'xtick.labelsize':     10,
        'ytick.labelsize':     10,
        'lines.linewidth':     1.0,
        'legend.fontsize':     10,
        'legend.frameon':      False,
        'figure.facecolor':    'white',
        'figure.dpi':          300,
        'savefig.dpi':         300,
        'savefig.bbox':        'tight',
        'savefig.pad_inches':  0.05,
        'scatter.edgecolors':  'none',
    })
