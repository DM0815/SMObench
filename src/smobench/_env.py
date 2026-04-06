"""Environment resolution for subprocess-based method isolation.

Architecture:
    - One shared SIF image (PyTorch+CUDA base)
    - Per-method pip prefix directories under singularity/envs/{MethodName}/
    - Base shared deps in singularity/envs/base/
    - PYTHONPATH = base_prefix:method_prefix to resolve deps at runtime
    - For users without singularity: current env or conda fallback
"""

from __future__ import annotations

import os
import shutil
from enum import Enum
from pathlib import Path


class EnvKind(Enum):
    CURRENT = "current"         # deps available in current Python
    CONDA = "conda"             # use `conda run -n <env>`
    SINGULARITY = "singularity" # use singularity exec with per-method prefix


# Method name -> environment group (for conda fallback grouping)
ENV_GROUPS: dict[str, str] = {
    "SpatialGlue": "torch-pyg",
    "PRAGA": "torch-pyg",
    "COSMOS": "torch-pyg",
    "PRESENT": "torch-pyg",
    "CANDIES": "torch-pyg",
    "MISO": "torch-pyg",
    "SpaBalance": "torch-pyg",
    "SpaFusion": "torch-pyg",
    "SpaMI": "torch-pyg",
    "spaMultiVAE": "base",
    "SpaMV": "torch-pyg",
    "SWITCH": "torch-pyg",
    "SpaMosaic": "spamosaic",
    "MultiGATE": "multigate",
    "SMOPCA": "base",
    "SMART": "torch-pyg",
}

# Group -> conda env name convention (fallback)
GROUP_CONDA_NAMES: dict[str, str] = {
    "torch-pyg": "smobench_torch",
    "spamosaic": "smobench_spamosaic",
    "multigate": "smobench_multigate",
    "base": "smobench_base",
}


def _get_singularity_root() -> Path:
    """Get the singularity directory path."""
    # Relative to package: src/smobench/_env.py -> ../../singularity/
    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir.parent.parent / "singularity",  # dev layout
        Path(os.environ.get("SMOBENCH_SINGULARITY_DIR", "")),
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return pkg_dir.parent.parent / "singularity"


def _find_sif() -> str | None:
    """Find the shared SIF image."""
    root = _get_singularity_root()
    # Prefer smobench_full.sif, then pytorch base
    for name in [
        "smobench_full.sif",
        "pytorch_2.1.2-cuda11.8-cudnn8-runtime.sif",
        "cuda_11.8.0-cudnn8-runtime-ubuntu22.04.sif",
    ]:
        path = root / "images" / name
        if path.is_file():
            return str(path)
    # Also check env var
    sif_dir = os.environ.get("SMOBENCH_SIF_DIR", "")
    if sif_dir:
        for name in ["smobench_full.sif", "pytorch_2.1.2-cuda11.8-cudnn8-runtime.sif"]:
            path = Path(sif_dir) / name
            if path.is_file():
                return str(path)
    return None


def _find_method_prefix(method_name: str) -> str | None:
    """Find per-method pip prefix directory."""
    root = _get_singularity_root()
    prefix = root / "envs" / method_name
    if prefix.is_dir() and any(prefix.iterdir()):
        return str(prefix)
    return None


def _find_base_prefix() -> str | None:
    """Find shared base prefix directory."""
    root = _get_singularity_root()
    prefix = root / "envs" / "base"
    if prefix.is_dir() and any(prefix.iterdir()):
        return str(prefix)
    return None


def _conda_env_exists(env_name: str) -> bool:
    """Check if a conda environment exists."""
    import subprocess
    try:
        result = subprocess.run(
            ["conda", "env", "list", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return False
        import json
        envs = json.loads(result.stdout).get("envs", [])
        return any(env_name == os.path.basename(p) for p in envs)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_current_env(method_name: str) -> bool:
    """Check if method deps are available in current environment."""
    try:
        from smobench.methods.registry import MethodRegistry
        method = MethodRegistry.get(method_name)
        return method.check_deps()
    except Exception:
        return False


def resolve_env(method_name: str) -> tuple[EnvKind, dict]:
    """Resolve the best available environment for a method.

    Returns (EnvKind, info) where info contains:
    - CURRENT: {}
    - CONDA: {"conda_env": str}
    - SINGULARITY: {"sif": str, "base_prefix": str|None, "method_prefix": str|None}

    Raises RuntimeError if no environment is available.
    """
    # 1. Current environment
    if check_current_env(method_name):
        return EnvKind.CURRENT, {}

    # 2. Singularity (preferred for HPC reproducibility)
    sif = _find_sif()
    if sif:
        base_prefix = _find_base_prefix()
        method_prefix = _find_method_prefix(method_name)
        return EnvKind.SINGULARITY, {
            "sif": sif,
            "base_prefix": base_prefix,
            "method_prefix": method_prefix,
        }

    # 3. Conda fallback
    group = ENV_GROUPS.get(method_name, "torch-pyg")
    conda_name = GROUP_CONDA_NAMES.get(group, f"smobench_{group}")
    if shutil.which("conda") and _conda_env_exists(conda_name):
        return EnvKind.CONDA, {"conda_env": conda_name}

    # No environment found
    group = ENV_GROUPS.get(method_name, "torch-pyg")
    raise RuntimeError(
        f"No environment found for method '{method_name}' (group: {group}).\n"
        f"Options:\n"
        f"  1. Install deps in current env: pip install smobench[{group}]\n"
        f"  2. Set up singularity: smobench setup --backend singularity\n"
        f"  3. Create conda env: smobench setup --group {group} --backend conda\n"
    )
