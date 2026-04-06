"""Main benchmark orchestration with parallel execution support."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from anndata import AnnData


@dataclass
class BenchmarkResult:
    """Container for benchmark results with built-in plotting."""

    records: list[dict] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def summary(self, group_by: str = "Method") -> pd.DataFrame:
        df = self.to_dataframe()
        if df.empty:
            return df
        score_cols = [c for c in df.columns if c.endswith("_Score") or c in
                      ("ARI", "NMI", "Moran_I", "Silhouette")]
        return df.groupby(group_by)[score_cols].mean().round(4)

    def ranking(self, score_col: str = "BioC_Score") -> pd.DataFrame:
        """Rank methods by a score column."""
        summary = self.summary()
        if score_col in summary.columns:
            return summary.sort_values(score_col, ascending=False)
        return summary

    def save(self, path: str):
        self.to_dataframe().to_csv(path, index=False)
        print(f"Results saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BenchmarkResult":
        df = pd.read_csv(path)
        return cls(records=df.to_dict("records"))

    @property
    def plot(self):
        from smobench.plot import ResultPlotter
        return ResultPlotter(self)

    def __repr__(self):
        df = self.to_dataframe()
        n_methods = df["Method"].nunique() if "Method" in df.columns else 0
        n_datasets = df["Dataset"].nunique() if "Dataset" in df.columns else 0
        return f"BenchmarkResult({n_methods} methods × {n_datasets} datasets, {len(df)} records)"

    def __add__(self, other: "BenchmarkResult") -> "BenchmarkResult":
        return BenchmarkResult(records=self.records + other.records)


def benchmark(
    dataset: str | list[str] = "all",
    methods: str | list[str] = "all",
    task: str = "vertical",
    clustering: list[str] | None = None,
    metrics: str = "standard",
    device: str = "cuda:0",
    seed: int = 42,
    n_jobs: int = 1,
    data_root: str | None = None,
    save_dir: str | None = None,
) -> BenchmarkResult:
    """Run benchmark: integrate → cluster → evaluate.

    Parameters
    ----------
    dataset : str or list[str]
        Dataset name(s). "all" for all datasets.
    methods : str or list[str]
        Method name(s). "all" for all methods supporting the task.
    task : str
        "vertical", "horizontal", "mosaic", or "all".
    clustering : list[str]
        Clustering methods. Default: ["leiden", "kmeans"].
    metrics : str
        "fast", "standard", "all".
    device : str
        GPU device.
    seed : int
        Random seed.
    n_jobs : int
        Number of parallel jobs. 1 = sequential.
    data_root : str
        Dataset root directory.
    save_dir : str
        Directory to save integrated adata files.

    Returns
    -------
    BenchmarkResult
    """
    from smobench.data import DATASET_REGISTRY
    from smobench.methods import list_methods

    if clustering is None:
        clustering = ["leiden", "kmeans"]

    # Resolve tasks
    tasks = ["vertical", "horizontal", "mosaic"] if task == "all" else [task]

    # Resolve datasets
    if dataset == "all":
        datasets_to_run = list(DATASET_REGISTRY.keys())
    elif isinstance(dataset, str):
        datasets_to_run = [dataset]
    else:
        datasets_to_run = list(dataset)

    # Resolve methods
    if methods == "all":
        method_df = list_methods()
        method_names = method_df["Method"].tolist()
    elif isinstance(methods, str):
        method_names = [methods]
    else:
        method_names = list(methods)

    result = BenchmarkResult()

    for cur_task in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {cur_task} integration")
        print(f"Methods: {method_names}")
        print(f"Datasets: {datasets_to_run}")
        print(f"{'='*60}")

        # Filter methods that support this task
        method_df = list_methods()
        task_methods = []
        for m in method_names:
            row = method_df[method_df["Method"] == m]
            if not row.empty and cur_task in row.iloc[0]["Tasks"]:
                task_methods.append(m)
            else:
                print(f"  SKIP {m}: does not support {cur_task}")

        # Build job list
        jobs = _build_jobs(cur_task, datasets_to_run, task_methods, clustering,
                          device, seed, data_root, save_dir)

        if n_jobs > 1 and len(jobs) > 1:
            records = _run_parallel(jobs, n_jobs)
        else:
            records = _run_sequential(jobs)

        result.records.extend(records)

    print(f"\n{'='*60}")
    print(f"Benchmark complete: {result}")
    print(f"{'='*60}")

    # Print runtime summary
    df = result.to_dataframe()
    if "Runtime" in df.columns and "Method" in df.columns and len(df) > 0:
        runtime_summary = (
            df.drop_duplicates(subset=["Method", "Dataset", "Slice"] if "Slice" in df.columns else ["Method", "Dataset"])
            .groupby("Method")["Runtime"]
            .agg(["mean", "min", "max"])
            .rename(columns={"mean": "Avg(s)", "min": "Min(s)", "max": "Max(s)"})
            .sort_values("Avg(s)")
        )
        print(f"\nRuntime per method:")
        print(runtime_summary.round(1).to_string())

    return result


def _build_jobs(task, datasets, methods, clustering, device, seed, data_root, save_dir):
    """Build list of (task, dataset, slice, method, kwargs) tuples."""
    from smobench.data import DATASET_REGISTRY

    jobs = []
    for ds_name in datasets:
        if ds_name not in DATASET_REGISTRY:
            continue
        ds_info = DATASET_REGISTRY[ds_name]

        for method_name in methods:
            if task == "vertical":
                for slice_name in ds_info["slices"]:
                    integrated_path = None
                    if save_dir:
                        import os
                        integrated_path = os.path.join(
                            save_dir, task, ds_name, slice_name, "adata_integrated.h5ad"
                        )

                    jobs.append({
                        "task": task,
                        "dataset": ds_name,
                        "slice_name": slice_name,
                        "method_name": method_name,
                        "clustering": clustering,
                        "device": device,
                        "seed": seed,
                        "data_root": data_root,
                        "save_integrated": integrated_path,
                    })
            elif task == "horizontal":
                jobs.append({
                    "task": task,
                    "dataset": ds_name,
                    "method_name": method_name,
                    "clustering": clustering,
                    "device": device,
                    "seed": seed,
                    "data_root": data_root,
                })
            elif task == "mosaic":
                jobs.append({
                    "task": task,
                    "dataset": ds_name,
                    "method_name": method_name,
                    "clustering": clustering,
                    "device": device,
                    "seed": seed,
                    "data_root": data_root,
                })

    return jobs


def _run_job(job: dict) -> list[dict]:
    """Execute a single benchmark job."""
    task = job.pop("task")
    ds = job["dataset"]
    method = job["method_name"]

    print(f"\n  [{task}] {method} × {ds}" + (f"/{job.get('slice_name', '')}" if "slice_name" in job else ""))

    try:
        if task == "vertical":
            from smobench.pipeline.vertical import run_vertical
            return run_vertical(**job)
        elif task == "horizontal":
            from smobench.pipeline.horizontal import run_horizontal
            return run_horizontal(**job)
        elif task == "mosaic":
            from smobench.pipeline.mosaic import run_mosaic
            return run_mosaic(**job)
    except Exception as e:
        print(f"    FAILED: {e}")
        return []


def _run_sequential(jobs: list[dict]) -> list[dict]:
    """Run jobs sequentially."""
    all_records = []
    for job in jobs:
        records = _run_job(job)
        all_records.extend(records)
    return all_records


def _run_parallel(jobs: list[dict], n_jobs: int) -> list[dict]:
    """Run jobs in parallel using ProcessPoolExecutor."""
    all_records = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(_run_job, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                records = future.result()
                all_records.extend(records)
            except Exception as e:
                job = futures[future]
                print(f"  Job failed: {job.get('method_name')} × {job.get('dataset')}: {e}")
    return all_records
