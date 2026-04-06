"""Subprocess-based method isolation for conflicting dependencies.

Supports three execution modes:
1. CURRENT: run in-process (zero overhead, deps available locally)
2. SINGULARITY: one shared SIF + per-method pip prefix via PYTHONPATH
3. CONDA: conda run -n <env> python -m smobench._runner
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from anndata import AnnData

# Common HPC paths where singularity/apptainer may be installed
_CONTAINER_SEARCH_PATHS = [
    "/app/apps/singularity/3.10.0/bin",
    "/app/apps/singularity/4.0.0/bin",
    "/app/apps/singularity/4.2.2/bin",
    "/app/apps/singularity/4.3.1/bin",
    "/usr/local/bin",
    "/opt/singularity/bin",
]


def _find_container_runtime() -> str | None:
    """Find singularity/apptainer, checking PATH and common HPC install locations."""
    import shutil

    # Check env override first
    env_path = os.environ.get("SMOBENCH_SINGULARITY_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    # Check PATH
    for name in ("apptainer", "singularity"):
        found = shutil.which(name)
        if found:
            return found

    # Fallback: scan common HPC paths
    for search_dir in _CONTAINER_SEARCH_PATHS:
        for name in ("apptainer", "singularity"):
            candidate = os.path.join(search_dir, name)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate

    return None


def subprocess_integrate(
    method_name: str,
    adata_rna: AnnData,
    adata_mod2: AnnData,
    device: str = "cuda:0",
    seed: int = 42,
    cache_dir: str | None = None,
    **kwargs,
) -> np.ndarray:
    """Run method.integrate() with automatic environment isolation.

    If deps are available in the current env, runs in-process (no overhead).
    Otherwise, spawns a subprocess in the appropriate singularity/conda env.

    Parameters
    ----------
    cache_dir : str, optional
        If set, cache embeddings under this directory.  Subsequent calls
        with the same method / data shape / seed will load from cache.
    """
    import hashlib

    from smobench._env import resolve_env, EnvKind

    env_kind, env_info = resolve_env(method_name)

    # Merge device/seed into kwargs for subprocess
    kwargs["device"] = device
    kwargs["seed"] = seed

    # Check cache
    if cache_dir is not None:
        cache_key = _cache_key(method_name, adata_rna, adata_mod2, seed, kwargs)
        cache_path = os.path.join(cache_dir, f"{method_name}_{cache_key}.npy")
        if os.path.isfile(cache_path):
            print(f"  [{method_name}] Loading cached embedding from {cache_path}")
            emb = np.load(cache_path)
            kept_path = cache_path.replace(".npy", "_kept.npy")
            kept = np.load(kept_path) if os.path.isfile(kept_path) else None
            return emb, kept

    if env_kind == EnvKind.CURRENT:
        emb, kept = _run_inprocess(method_name, adata_rna, adata_mod2, **kwargs)
    else:
        emb, kept = _run_subprocess(method_name, adata_rna, adata_mod2, env_kind, env_info, **kwargs)

    # Save to cache
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_path, emb)
        if kept is not None:
            np.save(cache_path.replace(".npy", "_kept.npy"), kept)
        print(f"  [{method_name}] Cached embedding to {cache_path}")

    return emb, kept


def _cache_key(method_name, adata_rna, adata_mod2, seed, kwargs):
    """Generate a short hash key for caching based on method + data identity."""
    import hashlib
    parts = [
        method_name,
        str(adata_rna.shape),
        str(adata_mod2.shape),
        str(seed),
        str(sorted((k, v) for k, v in kwargs.items() if k not in ("device", "verbose"))),
    ]
    # Include obs_names fingerprint for data identity
    if len(adata_rna.obs_names) > 0:
        parts.append(str(adata_rna.obs_names[0]) + str(adata_rna.obs_names[-1]))
    h = hashlib.md5("|".join(parts).encode()).hexdigest()[:12]
    return h


def _run_inprocess(
    method_name: str,
    adata_rna: AnnData,
    adata_mod2: AnnData,
    **kwargs,
) -> np.ndarray:
    """Run method directly in current process."""
    from smobench.methods.registry import MethodRegistry

    method = MethodRegistry.get(method_name)
    result = method.integrate(adata_rna, adata_mod2, **kwargs)
    kept = None
    if isinstance(result, tuple):
        embedding, kept = result
    else:
        embedding = result
    if not isinstance(embedding, np.ndarray):
        embedding = np.asarray(embedding)
    if kept is not None:
        kept = np.asarray(kept)
    return embedding, kept


def _run_subprocess(
    method_name: str,
    adata_rna: AnnData,
    adata_mod2: AnnData,
    env_kind,
    env_info: dict,
    **kwargs,
) -> np.ndarray:
    """Run method in a subprocess with the appropriate environment."""
    from smobench._env import EnvKind

    tmpdir = tempfile.mkdtemp(prefix=f"smobench_{method_name}_")
    rna_path = os.path.join(tmpdir, "rna.h5ad")
    mod2_path = os.path.join(tmpdir, "mod2.h5ad")
    out_path = os.path.join(tmpdir, "embedding.npy")

    try:
        # Write data to temp files
        adata_rna.write_h5ad(rna_path)
        adata_mod2.write_h5ad(mod2_path)

        # Build command
        runner_args = [
            "-m", "smobench._runner",
            "--method", method_name,
            "--rna", rna_path,
            "--mod2", mod2_path,
            "--out", out_path,
            "--kwargs", json.dumps(kwargs),
        ]

        # Common env vars for reproducibility
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["NUMBA_CACHE_DIR"] = os.path.join(tmpdir, "numba_cache")
        env["MPLCONFIGDIR"] = os.path.join(tmpdir, "mpl_cache")
        env["PYTHONNOUSERSITE"] = "1"

        if env_kind == EnvKind.SINGULARITY:
            cmd = _build_singularity_cmd(env_info, runner_args, tmpdir, env)
        elif env_kind == EnvKind.CONDA:
            cmd = ["conda", "run", "-n", env_info["conda_env"],
                   "--no-capture-output", "python"] + runner_args
        else:
            cmd = [sys.executable] + runner_args

        verbose = kwargs.pop("verbose", True)
        print(f"  [{method_name}] Running in {env_kind.value} env")

        if verbose:
            # Stream output in real time so user sees training progress
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                bufsize=1,
            )
            output_lines = []
            try:
                for line in proc.stdout:
                    output_lines.append(line)
                    sys.stdout.write(f"  [{method_name}] {line}")
                    sys.stdout.flush()
                proc.wait(timeout=7200)
            except subprocess.TimeoutExpired:
                proc.kill()
                raise
            returncode = proc.returncode
            full_output = "".join(output_lines)
        else:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,
                env=env,
            )
            returncode = result.returncode
            full_output = result.stdout + "\n" + result.stderr

        if returncode != 0:
            raise RuntimeError(
                f"Method '{method_name}' failed (exit code {returncode}).\n"
                f"Output:\n{full_output[-5000:]}"
            )

        # Read embedding
        if not os.path.isfile(out_path):
            raise RuntimeError(
                f"Method '{method_name}' did not produce output.\n"
                f"Output:\n{full_output[-5000:]}"
            )

        embedding = np.load(out_path)
        kept_path = out_path.replace(".npy", "_kept.npy")
        kept = np.load(kept_path) if os.path.isfile(kept_path) else None
        return embedding, kept

    finally:
        # Cleanup
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def _build_singularity_cmd(
    env_info: dict,
    runner_args: list[str],
    tmpdir: str,
    env: dict,
) -> list[str]:
    """Build singularity exec command with per-method PYTHONPATH."""
    import shutil

    sif = env_info["sif"]
    base_prefix = env_info.get("base_prefix")
    method_prefix = env_info.get("method_prefix")

    # Determine singularity/apptainer with fallback to common HPC paths
    runtime = _find_container_runtime()
    if runtime is None:
        raise FileNotFoundError(
            "Neither singularity nor apptainer found.\n"
            "Try: module load singularity\n"
            "Or set SMOBENCH_SINGULARITY_PATH=/path/to/singularity"
        )

    # Build PYTHONPATH for per-method prefix isolation
    pythonpath_parts = []
    if method_prefix:
        pythonpath_parts.append(method_prefix)
    if base_prefix:
        pythonpath_parts.append(base_prefix)
    if pythonpath_parts:
        env["PYTHONPATH"] = ":".join(pythonpath_parts)

    # Build bind mounts
    binds = [f"{tmpdir}:{tmpdir}"]

    # Bind the singularity envs dir so PYTHONPATH works
    sing_root = os.path.dirname(os.path.dirname(sif))  # singularity/
    envs_dir = os.path.join(sing_root, "envs")
    if os.path.isdir(envs_dir):
        binds.append(f"{envs_dir}:{envs_dir}")

    # Bind smobench package source so `python -m smobench._runner` works
    # Use realpath to resolve symlinks (singularity doesn't follow them)
    smobench_pkg = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    smobench_src = os.path.dirname(smobench_pkg)
    binds.append(f"{smobench_src}:{smobench_src}")

    # Also resolve envs_dir with realpath
    if os.path.isdir(envs_dir):
        real_envs = os.path.realpath(envs_dir)
        if real_envs != envs_dir:
            binds[-2] = f"{real_envs}:{real_envs}"
            # Update PYTHONPATH entries to real paths
            if method_prefix:
                method_prefix = os.path.realpath(method_prefix)
            if base_prefix:
                base_prefix = os.path.realpath(base_prefix)
            pythonpath_parts = []
            if method_prefix:
                pythonpath_parts.append(method_prefix)
            if base_prefix:
                pythonpath_parts.append(base_prefix)
            if pythonpath_parts:
                env["PYTHONPATH"] = ":".join(pythonpath_parts)

    # Add smobench src to PYTHONPATH so the runner can import smobench
    env["PYTHONPATH"] = f"{smobench_src}:" + env.get("PYTHONPATH", "")

    # Bind data root if set
    data_root = os.environ.get("SMOBENCH_DATA_ROOT", "")
    if data_root and os.path.isdir(data_root):
        binds.append(f"{data_root}:{data_root}")

    cmd = [
        runtime, "exec",
        "--nv",  # GPU passthrough
    ]
    for bind in binds:
        cmd.extend(["--bind", bind])

    # Pass env vars
    for key in ["PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMBA_CACHE_DIR", "MPLCONFIGDIR", "PYTHONNOUSERSITE",
                 "CUBLAS_WORKSPACE_CONFIG"]:
        if key in env:
            cmd.extend(["--env", f"{key}={env[key]}"])

    cmd.extend(["--env", "TF_CPP_MIN_LOG_LEVEL=2"])

    cmd.append(sif)

    # Use /usr/bin/env bash -c to resolve python from container PATH
    # (singularity exec resolves executables on host, not in container)
    inner_cmd = "python " + " ".join(f"'{a}'" for a in runner_args)
    cmd.extend(["/usr/bin/env", "bash", "-c", inner_cmd])

    return cmd
