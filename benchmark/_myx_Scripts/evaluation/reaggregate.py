#!/usr/bin/env python3
"""
Re-run aggregation only (no evaluation).
Reads existing per-file CSVs and regenerates summary tables.
"""

import os
import sys

ROOT = '/data/projects/11003054/e1724738/_private/NUS/_Proj1/SMOBench-CLEAN'

# Add paths
sys.path.insert(0, os.path.join(ROOT, '_myx_Scripts', 'evaluation'))
sys.path.insert(0, os.path.join(ROOT, 'Eval'))

# --- Vertical ---
print("=" * 60)
print("Re-aggregating VERTICAL results...")
print("=" * 60)
from eval_vertical import aggregate_results as agg_vertical
vertical_output = os.path.join(ROOT, '_myx_Results', 'evaluation', 'vertical')
agg_vertical(vertical_output, ROOT)

# --- Horizontal ---
print()
print("=" * 60)
print("Re-aggregating HORIZONTAL results...")
print("=" * 60)
from eval_horizontal import aggregate_results as agg_horizontal
horizontal_output = os.path.join(ROOT, '_myx_Results', 'evaluation', 'horizontal')
agg_horizontal(horizontal_output, ROOT)

print()
print("Reaggregation complete.")
