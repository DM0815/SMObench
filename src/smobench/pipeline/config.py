"""YAML configuration for benchmark runs.

Example config.yaml:
    task: vertical
    datasets:
      - Human_Lymph_Nodes
      - Human_Tonsils
    methods:
      - SpatialGlue
      - SpaMosaic
      - PRAGA
    clustering:
      - leiden
      - kmeans
    device: cuda:0
    seed: 42
    n_jobs: 4
    output: results/vertical_results.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml


def load_config(path: str | Path) -> dict:
    """Load benchmark configuration from YAML."""
    with open(path) as f:
        config = yaml.safe_load(f)

    # Set defaults
    config.setdefault("task", "vertical")
    config.setdefault("datasets", "all")
    config.setdefault("methods", "all")
    config.setdefault("clustering", ["leiden", "kmeans"])
    config.setdefault("device", "cuda:0")
    config.setdefault("seed", 42)
    config.setdefault("n_jobs", 1)
    config.setdefault("output", "smobench_results.csv")
    config.setdefault("data_root", None)
    config.setdefault("save_dir", None)

    return config


def save_config(config: dict, path: str | Path):
    """Save benchmark configuration to YAML."""
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def generate_default_config(path: str | Path = "smobench_config.yaml"):
    """Generate a default config file."""
    config = {
        "task": "vertical",
        "datasets": "all",
        "methods": "all",
        "clustering": ["leiden", "kmeans"],
        "device": "cuda:0",
        "seed": 42,
        "n_jobs": 1,
        "output": "smobench_results.csv",
        "data_root": None,
        "save_dir": None,
    }
    save_config(config, path)
    print(f"Config saved to {path}")
    return config
