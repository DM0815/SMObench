"""Method hyperparameter configuration.

Loads per-method parameters from a YAML config file, with support for
dataset-specific overrides.

Usage::

    from smobench.config import load_config, get_method_params

    cfg = load_config("configs/method_params.yaml")
    params = get_method_params(cfg, "GROVER", dataset="Human_Tonsil")
    # {'epochs': 300, 'learning_rate': 0.0001, ...}
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def load_config(path: str | Path | None = None) -> dict:
    """Load method params from YAML file.

    Parameters
    ----------
    path : str or Path, optional
        Path to YAML config file. If None, looks for:
        1. ``SMOBENCH_CONFIG`` env var
        2. ``configs/method_params.yaml`` relative to project root
        3. Returns empty dict if nothing found
    """
    import yaml

    if path is None:
        path = os.environ.get("SMOBENCH_CONFIG")

    if path is None:
        # Try default location relative to package
        candidates = [
            Path(__file__).parent.parent.parent / "configs" / "method_params.yaml",
            Path.cwd() / "configs" / "method_params.yaml",
        ]
        for c in candidates:
            if c.is_file():
                path = c
                break

    if path is None:
        return {}

    path = Path(path)
    if not path.is_file():
        return {}

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    return cfg


def get_method_params(
    config: dict,
    method_name: str,
    dataset: str | None = None,
) -> dict[str, Any]:
    """Get resolved params for a method, with optional dataset override.

    Priority: dataset-specific > method default > empty dict

    Parameters
    ----------
    config : dict
        Config loaded by :func:`load_config`.
    method_name : str
        Method name (e.g. "GROVER", "SpatialGlue").
    dataset : str, optional
        Dataset name for dataset-specific overrides.

    Returns
    -------
    dict
        Merged parameter dict.
    """
    method_cfg = config.get(method_name, {})
    if not method_cfg:
        return {}

    # Start with defaults
    params = dict(method_cfg.get("default", {}) or {})

    # Apply dataset-specific overrides
    if dataset:
        ds_overrides = (method_cfg.get("datasets") or {}).get(dataset, {})
        if ds_overrides:
            params.update(ds_overrides)

    return params
