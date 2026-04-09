"""Run all (or selected) methods on all datasets, save results to h5ad.

Five task types:
    1. vertical    — cross-modality integration on individual slices (all methods)
    2. horizontal  — batch correction across fused slices (14 methods)
    3. mosaic      — mixed modality + batch integration (SpaMosaic only)
    4. image       — triplet-modality with image embeddings (MISO, GROVER)
    5. 3m          — three-modality simulated data (8 methods)

Usage:
    python run_all_methods.py                          # vertical task (default)
    python run_all_methods.py --task horizontal         # horizontal (batch correction)
    python run_all_methods.py --task 3m                 # three-modality
    python run_all_methods.py --task all                # all 5 tasks
    python run_all_methods.py --methods SpatialGlue SMOPCA  # specific methods
    python run_all_methods.py --parallel --n-gpus 4    # 4 methods in parallel on 4 GPUs
    python run_all_methods.py --parallel --gpus 0,2,5  # parallel on specific GPU IDs
"""

import os, sys, time, argparse, traceback, warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Variable names are not unique")

# ── Config ──────────────────────────────────────────────────────
DATA_ROOT = "/home/project/11003054/e1724738/_public/SMOBench/_github/benchmark/Dataset"
SAVE_DIR = os.path.join(os.path.dirname(__file__), "results")
DEVICE = "cuda:0"
SEED = 2026
CLUSTERING = ["leiden", "louvain", "kmeans", "mclust"]
GTF_PATH = os.path.join(DATA_ROOT, "data_info", "gencode.vM25.annotation.gtf.gz")

ALL_METHODS = [
    "SpatialGlue", "CANDIES", "COSMOS", "MISO", "PRAGA", "PRESENT",
    "SpaBalance", "SpaFusion", "SpaMI", "SpaMosaic", "SpaMV",
    "spaMultiVAE", "SWITCH", "MultiGATE", "SMOPCA", "SMART",
    "GROVER",
]

ALL_TASKS = ["vertical", "horizontal", "mosaic", "image", "3m"]


# ── Shared helpers ──────────────────────────────────────────────

def _check_task_support(method_name, task):
    """Check if method supports a task. Returns (method_obj, supported)."""
    from smobench.methods.registry import MethodRegistry
    method_obj = MethodRegistry.get(method_name)
    return method_obj, task in method_obj.tasks


def _integrate_and_evaluate(method_name, adata_rna, adata_mod2, device, seed,
                            extra_kwargs, label_key, n_clusters, has_gt,
                            batch_key=None):
    """Common integration → clustering → evaluation logic.

    Returns (embedding, adata_rna, metrics_dict, runtime).
    adata_rna may be subsetted if method filters cells.
    """
    from smobench.pipeline._isolation import subprocess_integrate
    from smobench.clustering import cluster
    from smobench.metrics.evaluate import evaluate

    t0 = time.time()
    embedding, kept_indices = subprocess_integrate(
        method_name, adata_rna, adata_mod2,
        device=device, seed=seed, **extra_kwargs,
    )
    runtime = time.time() - t0
    print(f"  Integration done: {runtime:.1f}s, embedding {embedding.shape}")

    # Handle cell filtering
    if embedding.shape[0] != adata_rna.n_obs:
        if kept_indices is not None:
            print(f"  Method kept {len(kept_indices)}/{adata_rna.n_obs} cells.")
            adata_rna = adata_rna[kept_indices].copy()
        else:
            adata_rna = adata_rna[:embedding.shape[0]].copy()

    adata_rna.obsm[method_name] = embedding

    # Compute CMGTC (modality-agnostic, before clustering)
    from smobench.metrics.cmgtc import cmgtc
    import scipy.sparse as sp
    try:
        rna_X = adata_rna.X.toarray() if sp.issparse(adata_rna.X) else np.array(adata_rna.X)
        mod2_X = adata_mod2.X.toarray() if sp.issparse(adata_mod2.X) else np.array(adata_mod2.X)
        # Align mod2 if cell filtering happened
        if mod2_X.shape[0] != rna_X.shape[0] and kept_indices is not None:
            mod2_X = mod2_X[kept_indices]
        cmgtc_score = cmgtc(adata_rna, method_name, rna_X, mod2_X)
        print(f"    CMGTC = {cmgtc_score:.3f}")
    except Exception as e:
        print(f"    CMGTC failed: {e}")
        cmgtc_score = None

    # Cluster and evaluate
    metrics_dict = {}
    for clust in CLUSTERING:
        clust_key = f"{method_name}_{clust}"
        try:
            cluster(adata_rna, method=clust, n_clusters=n_clusters,
                    embedding_key=method_name, key_added=clust_key)
            scores = evaluate(
                adata_rna, embedding_key=method_name, cluster_key=clust_key,
                label_key=label_key, batch_key=batch_key, has_gt=has_gt,
            )
            if cmgtc_score is not None:
                scores["CMGTC"] = cmgtc_score
                # Recompute Total with CMGTC included
                total_keys = ["SC_Score", "BioC_Score", "BVC_Score", "BER_Score", "CMGTC"]
                total_vals = [scores[k] for k in total_keys if k in scores]
                scores["Total"] = float(np.mean(total_vals)) if total_vals else 0.0
            metrics_dict[clust] = scores
            if has_gt:
                print(f"    {clust:10s} -> ARI={scores.get('ARI','?'):.3f} "
                      f"NMI={scores.get('NMI','?'):.3f} "
                      f"Moran_I={scores.get('Moran_I','?'):.3f} "
                      f"Total={scores.get('Total','?'):.3f}")
            else:
                print(f"    {clust:10s} -> Silhouette={scores.get('Silhouette','?'):.3f} "
                      f"DBI={scores.get('DBI','?'):.3f} "
                      f"Moran_I={scores.get('Moran_I','?'):.3f} "
                      f"Total={scores.get('Total','?'):.3f}")
        except Exception as e:
            print(f"    {clust:10s} -> FAILED: {e}")

    return embedding, adata_rna, metrics_dict, runtime


def _save_results(adata_rna, method_name, embedding, metrics_dict, runtime,
                  task, ds_name, slice_name=None):
    """Save embedding + metrics to h5ad."""
    from smobench.io import save_embedding

    if metrics_dict:
        adata_rna.uns[f"{method_name}_metrics"] = metrics_dict
    adata_rna.uns[f"{method_name}_train_time"] = runtime

    if slice_name:
        out_dir = os.path.join(SAVE_DIR, task, ds_name, slice_name)
    else:
        out_dir = os.path.join(SAVE_DIR, task, ds_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "adata_integrated.h5ad")
    save_embedding(adata_rna, method_name, embedding, out_path, train_time=runtime)


# ── Task 1: Vertical ───────────────────────────────────────────

def run_one_method_vertical(method_name, device, seed, method_config=None):
    """Run a single method on all standard datasets (vertical integration)."""
    os.environ["SMOBENCH_DATA_ROOT"] = DATA_ROOT
    os.makedirs(SAVE_DIR, exist_ok=True)

    from smobench._constants import DATASETS, get_n_clusters, METHOD_DATASET_SKIP, IMAGE_DATASETS, THREE_M_DATASETS
    from smobench.data import load_dataset
    from smobench.config import get_method_params

    if method_config is None:
        method_config = {}

    skip_datasets = METHOD_DATASET_SKIP.get(method_name, set())
    summary = []

    for ds_name, ds_info in DATASETS.items():
        # Skip image-only and 3M datasets (they have their own tasks)
        if ds_name in IMAGE_DATASETS or ds_name in THREE_M_DATASETS:
            continue
        if ds_name in skip_datasets:
            print(f"\n  [SKIP] {method_name} does not support {ds_name} (upstream limitation)")
            continue

        for slice_name in ds_info["slices"]:
            print(f"\n{'='*70}")
            print(f"  [vertical] {method_name} | {ds_name} / {slice_name}")
            print(f"  modality={ds_info['modality']} gt={ds_info['gt']} "
                  f"n_clusters={get_n_clusters(ds_name, slice_name)}")
            print(f"{'='*70}")

            modality = "ADT" if "ADT" in ds_info["modality"] else "ATAC"
            label_key = "Spatial_Label" if ds_info["gt"] else None
            n_clusters = get_n_clusters(ds_name, slice_name)

            try:
                adata_rna, adata_mod2 = load_dataset(ds_name, slice_name, DATA_ROOT)
                print(f"  Data loaded: RNA {adata_rna.shape}, mod2 {adata_mod2.shape}")

                extra_kwargs = get_method_params(method_config, method_name, dataset=ds_name)
                extra_kwargs["modality"] = modality
                if "data_type" not in extra_kwargs and "data_type" in ds_info:
                    extra_kwargs["data_type"] = ds_info["data_type"]
                # SWITCH needs GTF for ATAC guidance graph
                if method_name == "SWITCH" and modality == "ATAC" and os.path.isfile(GTF_PATH):
                    extra_kwargs["gtf_path"] = GTF_PATH
                shown = {k: v for k, v in extra_kwargs.items() if k not in ("modality", "gtf_path")}
                if shown:
                    print(f"  Config params: {shown}")

                embedding, adata_rna, metrics_dict, runtime = _integrate_and_evaluate(
                    method_name, adata_rna, adata_mod2, device, seed,
                    extra_kwargs, label_key, n_clusters, ds_info["gt"],
                )
                _save_results(adata_rna, method_name, embedding, metrics_dict,
                              runtime, "vertical", ds_name, slice_name)
                summary.append(f"  OK  {ds_name}/{slice_name} ({runtime:.1f}s)")

            except Exception as e:
                print(f"  INTEGRATION FAILED: {e}")
                traceback.print_exc()
                summary.append(f"  FAIL {ds_name}/{slice_name}: {e}")

    print(f"\n--- {method_name} vertical summary ---")
    for line in summary:
        print(line)
    return summary


# ── Task 2: Horizontal ─────────────────────────────────────────

def run_one_method_horizontal(method_name, device, seed, method_config=None):
    """Run a single method on all datasets (horizontal / batch correction)."""
    _, supported = _check_task_support(method_name, "horizontal")
    if not supported:
        print(f"\n  [SKIP] {method_name} does not support horizontal integration")
        return []

    os.environ["SMOBENCH_DATA_ROOT"] = DATA_ROOT
    os.makedirs(SAVE_DIR, exist_ok=True)

    from smobench._constants import DATASETS, get_n_clusters, METHOD_DATASET_SKIP, IMAGE_DATASETS, THREE_M_DATASETS
    from smobench.config import get_method_params

    import scanpy as sc
    from pathlib import Path

    if method_config is None:
        method_config = {}

    skip_datasets = METHOD_DATASET_SKIP.get(method_name, set())
    summary = []

    for ds_name, ds_info in DATASETS.items():
        # Skip image-only and 3M datasets
        if ds_name in IMAGE_DATASETS or ds_name in THREE_M_DATASETS:
            continue
        if ds_name in skip_datasets:
            print(f"\n  [SKIP] {method_name} does not support {ds_name}")
            continue
        # Skip single-slice datasets (no batch to correct)
        if len(ds_info["slices"]) < 2:
            continue

        print(f"\n{'='*70}")
        print(f"  [horizontal] {method_name} | {ds_name} (fusion)")
        print(f"  modality={ds_info['modality']} gt={ds_info['gt']}")
        print(f"{'='*70}")

        modality = ds_info["modality"]
        mod2_name = "ADT" if "ADT" in modality else "ATAC"
        label_key = "Spatial_Label" if ds_info["gt"] else None
        n_clusters = get_n_clusters(ds_name, ds_info["slices"][0])

        # Find fusion data
        root = Path(DATA_ROOT)
        fusion_dir = "fusionWithGT" if ds_info["gt"] else "fusionWoGT"
        fusion_base = root / fusion_dir / modality
        if not fusion_base.is_dir():
            fusion_base = root / f"_myx_{fusion_dir}" / modality
        if not fusion_base.is_dir():
            print(f"  No fusion data found for {ds_name}, skipping")
            summary.append(f"  SKIP {ds_name}: no fusion data")
            continue

        rna_path = fusion_base / f"{ds_name}_Fusion_RNA.h5ad"
        mod2_path = fusion_base / f"{ds_name}_Fusion_{mod2_name}.h5ad"

        if not rna_path.exists() or not mod2_path.exists():
            print(f"  Fusion files not found: {rna_path.name} / {mod2_path.name}")
            summary.append(f"  SKIP {ds_name}: fusion files missing")
            continue

        try:
            adata_rna = sc.read_h5ad(str(rna_path))
            adata_mod2 = sc.read_h5ad(str(mod2_path))
            print(f"  Fusion data loaded: RNA {adata_rna.shape}, mod2 {adata_mod2.shape}")

            if "batch" not in adata_rna.obs.columns:
                print(f"  WARNING: No 'batch' column in fusion data, skipping")
                summary.append(f"  SKIP {ds_name}: no batch column")
                continue

            batches = adata_rna.obs["batch"].unique()
            print(f"  Batches: {list(batches)}")

            extra_kwargs = get_method_params(method_config, method_name, dataset=ds_name)
            extra_kwargs["modality"] = mod2_name
            if "data_type" not in extra_kwargs and "data_type" in ds_info:
                extra_kwargs["data_type"] = ds_info["data_type"]

            embedding, adata_rna, metrics_dict, runtime = _integrate_and_evaluate(
                method_name, adata_rna, adata_mod2, device, seed,
                extra_kwargs, label_key, n_clusters, ds_info["gt"],
                batch_key="batch",
            )
            _save_results(adata_rna, method_name, embedding, metrics_dict,
                          runtime, "horizontal", ds_name)
            summary.append(f"  OK  {ds_name} ({runtime:.1f}s)")

        except Exception as e:
            print(f"  INTEGRATION FAILED: {e}")
            traceback.print_exc()
            summary.append(f"  FAIL {ds_name}: {e}")

    print(f"\n--- {method_name} horizontal summary ---")
    for line in summary:
        print(line)
    return summary


# ── Task 3: Mosaic ──────────────────────────────────────────────

def run_one_method_mosaic(method_name, device, seed, method_config=None):
    """Run mosaic integration (mixed modality + batch).

    Currently only SpaMosaic supports this natively.
    Mosaic data: different slices have different modality combinations.
    """
    _, supported = _check_task_support(method_name, "mosaic")
    if not supported:
        print(f"\n  [SKIP] {method_name} does not support mosaic integration")
        return []

    # TODO: Implement mosaic data loading and evaluation.
    # Mosaic integration requires special data preparation where
    # different slices have different modality combinations.
    # SpaMosaic handles this natively via its mosaic data loader.
    print(f"\n  [TODO] Mosaic integration for {method_name}: pipeline not yet implemented")
    print(f"         SpaMosaic mosaic mode requires custom data preparation.")
    return []


# ── Task 4: Image (triplet-modality) ───────────────────────────

def run_one_method_image(method_name, device, seed, method_config=None):
    """Run triplet-modality integration on datasets with image embeddings.

    Currently supported by MISO and GROVER on 4 image datasets.
    """
    _, supported = _check_task_support(method_name, "image")
    if not supported:
        print(f"\n  [SKIP] {method_name} does not support image integration")
        return []

    os.environ["SMOBENCH_DATA_ROOT"] = DATA_ROOT
    os.makedirs(SAVE_DIR, exist_ok=True)

    from smobench._constants import DATASETS, IMAGE_DATASETS
    from smobench.config import get_method_params

    import scanpy as sc
    from pathlib import Path

    if method_config is None:
        method_config = {}

    summary = []

    for ds_name in sorted(IMAGE_DATASETS):
        ds_info = DATASETS[ds_name]
        slice_name = ds_info["slices"][0]

        print(f"\n{'='*70}")
        print(f"  [image] {method_name} | {ds_name}")
        print(f"  modality={ds_info['modality']} n_clusters={ds_info['n_clusters']}")
        print(f"{'='*70}")

        root = Path(DATA_ROOT)
        ds_path = root / ds_info["path"]

        # Load RNA + protein + image embeddings
        rna_path = ds_path / "rna.h5ad"
        protein_path = ds_path / "protein.h5ad"
        img_emb_path = ds_path / "image_embeddings.npy"

        if not rna_path.exists():
            print(f"  Image dataset not found: {rna_path}")
            summary.append(f"  SKIP {ds_name}: data not found")
            continue

        try:
            adata_rna = sc.read_h5ad(str(rna_path))
            adata_mod2 = sc.read_h5ad(str(protein_path))

            if img_emb_path.exists():
                img_emb = np.load(str(img_emb_path))
                print(f"  Data loaded: RNA {adata_rna.shape}, protein {adata_mod2.shape}, "
                      f"image_emb {img_emb.shape}")
            else:
                print(f"  WARNING: image_embeddings.npy not found at {img_emb_path}")
                img_emb = None

            n_clusters = ds_info["n_clusters"]
            label_key = "Spatial_Label" if ds_info["gt"] else None

            extra_kwargs = get_method_params(method_config, method_name, dataset=ds_name)
            extra_kwargs["modality"] = "ADT"
            if img_emb is not None:
                extra_kwargs["image_embeddings"] = img_emb
            if "data_type" not in extra_kwargs and "data_type" in ds_info:
                extra_kwargs["data_type"] = ds_info["data_type"]

            embedding, adata_rna, metrics_dict, runtime = _integrate_and_evaluate(
                method_name, adata_rna, adata_mod2, device, seed,
                extra_kwargs, label_key, n_clusters, ds_info["gt"],
            )
            _save_results(adata_rna, method_name, embedding, metrics_dict,
                          runtime, "image", ds_name, slice_name)
            summary.append(f"  OK  {ds_name} ({runtime:.1f}s)")

        except Exception as e:
            print(f"  INTEGRATION FAILED: {e}")
            traceback.print_exc()
            summary.append(f"  FAIL {ds_name}: {e}")

    print(f"\n--- {method_name} image summary ---")
    for line in summary:
        print(line)
    return summary


# ── Task 5: 3M (three-modality) ────────────────────────────────

def run_one_method_3m(method_name, device, seed, method_config=None):
    """Run three-modality integration on simulated 3M datasets (RNA + ADT + ATAC).

    Supported by 8 methods: SpatialGlue, SpaMosaic, PRAGA, SpaBalance (native),
    MISO, PRESENT, SMOPCA, SpaMV (adapted).
    """
    _, supported = _check_task_support(method_name, "3m")
    if not supported:
        print(f"\n  [SKIP] {method_name} does not support 3M integration")
        return []

    os.environ["SMOBENCH_DATA_ROOT"] = DATA_ROOT
    os.makedirs(SAVE_DIR, exist_ok=True)

    from smobench._constants import DATASETS, THREE_M_DATASETS
    from smobench.config import get_method_params
    from smobench.pipeline._isolation import subprocess_integrate
    from smobench.clustering import cluster
    from smobench.metrics.evaluate import evaluate

    import scanpy as sc
    from pathlib import Path

    if method_config is None:
        method_config = {}

    summary = []

    for ds_name in sorted(THREE_M_DATASETS):
        ds_info = DATASETS[ds_name]
        slice_name = ds_info["slices"][0]

        print(f"\n{'='*70}")
        print(f"  [3m] {method_name} | {ds_name}")
        print(f"  modality={ds_info['modality']} gt={ds_info['gt']}")
        print(f"{'='*70}")

        root = Path(DATA_ROOT)
        ds_path = root / ds_info["path"]

        # 3M datasets have 3 files: RNA, ADT, ATAC
        rna_path = ds_path / "adata_RNA.h5ad"
        adt_path = ds_path / "adata_ADT.h5ad"
        atac_path = ds_path / "adata_ATAC.h5ad"

        if not rna_path.exists():
            print(f"  3M dataset not found: {rna_path}")
            summary.append(f"  SKIP {ds_name}: data not found")
            continue

        try:
            adata_rna = sc.read_h5ad(str(rna_path))
            adata_adt = sc.read_h5ad(str(adt_path))
            adata_atac = sc.read_h5ad(str(atac_path))
            print(f"  Data loaded: RNA {adata_rna.shape}, ADT {adata_adt.shape}, "
                  f"ATAC {adata_atac.shape}")

            n_clusters = ds_info["n_clusters"]
            label_key = "Spatial_Label" if ds_info["gt"] else None

            extra_kwargs = get_method_params(method_config, method_name, dataset=ds_name)
            extra_kwargs["modality"] = "ADT_ATAC"  # Signal 3M mode
            extra_kwargs["adata_atac"] = adata_atac
            if "data_type" not in extra_kwargs and "data_type" in ds_info:
                extra_kwargs["data_type"] = ds_info["data_type"]

            # 3M integration: pass RNA + ADT as primary pair, ATAC via extra_kwargs
            t0 = time.time()
            embedding, kept_indices = subprocess_integrate(
                method_name, adata_rna, adata_adt,
                device=device, seed=seed, **extra_kwargs,
            )
            runtime = time.time() - t0
            print(f"  Integration done: {runtime:.1f}s, embedding {embedding.shape}")

            # Handle cell filtering
            if embedding.shape[0] != adata_rna.n_obs:
                if kept_indices is not None:
                    print(f"  Method kept {len(kept_indices)}/{adata_rna.n_obs} cells.")
                    adata_rna = adata_rna[kept_indices].copy()
                else:
                    adata_rna = adata_rna[:embedding.shape[0]].copy()

            adata_rna.obsm[method_name] = embedding

            # CMGTC for 3M: average over all 3 modality pairs
            from smobench.metrics.cmgtc import cmgtc as _cmgtc
            import scipy.sparse as sp
            try:
                rna_X = adata_rna.X.toarray() if sp.issparse(adata_rna.X) else np.array(adata_rna.X)
                adt_X = adata_adt.X.toarray() if sp.issparse(adata_adt.X) else np.array(adata_adt.X)
                atac_X = adata_atac.X.toarray() if sp.issparse(adata_atac.X) else np.array(adata_atac.X)
                if kept_indices is not None:
                    adt_X = adt_X[kept_indices]
                    atac_X = atac_X[kept_indices]
                # CM-GTC with all 3 modalities: use mean of RNA+ADT pair
                cmgtc_score = _cmgtc(adata_rna, method_name, rna_X,
                                     np.hstack([adt_X, atac_X]))
                print(f"    CMGTC = {cmgtc_score:.3f}")
            except Exception as e:
                print(f"    CMGTC failed: {e}")
                cmgtc_score = None

            # Cluster and evaluate
            metrics_dict = {}
            for clust in CLUSTERING:
                clust_key = f"{method_name}_{clust}"
                try:
                    cluster(adata_rna, method=clust, n_clusters=n_clusters,
                            embedding_key=method_name, key_added=clust_key)
                    scores = evaluate(
                        adata_rna, embedding_key=method_name, cluster_key=clust_key,
                        label_key=label_key, has_gt=ds_info["gt"],
                    )
                    if cmgtc_score is not None:
                        scores["CMGTC"] = cmgtc_score
                        total_keys = ["SC_Score", "BioC_Score", "BVC_Score", "BER_Score", "CMGTC"]
                        total_vals = [scores[k] for k in total_keys if k in scores]
                        scores["Total"] = float(np.mean(total_vals)) if total_vals else 0.0
                    metrics_dict[clust] = scores
                    if ds_info["gt"]:
                        print(f"    {clust:10s} -> ARI={scores.get('ARI','?'):.3f} "
                              f"NMI={scores.get('NMI','?'):.3f} "
                              f"Moran_I={scores.get('Moran_I','?'):.3f} "
                              f"Total={scores.get('Total','?'):.3f}")
                    else:
                        print(f"    {clust:10s} -> Silhouette={scores.get('Silhouette','?'):.3f} "
                              f"DBI={scores.get('DBI','?'):.3f} "
                              f"Total={scores.get('Total','?'):.3f}")
                except Exception as e:
                    print(f"    {clust:10s} -> FAILED: {e}")

            _save_results(adata_rna, method_name, embedding, metrics_dict,
                          runtime, "3m", ds_name, slice_name)
            summary.append(f"  OK  {ds_name} ({runtime:.1f}s)")

        except Exception as e:
            print(f"  INTEGRATION FAILED: {e}")
            traceback.print_exc()
            summary.append(f"  FAIL {ds_name}: {e}")

    print(f"\n--- {method_name} 3m summary ---")
    for line in summary:
        print(line)
    return summary


# ── Parallel & orchestration ────────────────────────────────────

TASK_FUNCTIONS = {
    "vertical": run_one_method_vertical,
    "horizontal": run_one_method_horizontal,
    "mosaic": run_one_method_mosaic,
    "image": run_one_method_image,
    "3m": run_one_method_3m,
}


def _worker(args_tuple):
    """Worker function for parallel execution."""
    method_name, device, seed, method_config, task = args_tuple
    print(f"\n[PARALLEL] Starting {method_name} ({task}) on {device} (pid={os.getpid()})")
    try:
        run_fn = TASK_FUNCTIONS[task]
        summary = run_fn(method_name, device, seed, method_config)
        return method_name, summary, None
    except Exception as e:
        traceback.print_exc()
        return method_name, [], str(e)


def run_parallel(methods, gpus, seed, method_config, task):
    """Run methods in parallel, round-robin assigned to available GPUs."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    tasks = []
    for i, method in enumerate(methods):
        gpu_id = gpus[i % len(gpus)]
        device = f"cuda:{gpu_id}"
        tasks.append((method, device, seed, method_config, task))

    print(f"\nParallel execution ({task}): {len(methods)} methods on {len(gpus)} GPUs {gpus}")
    for method, device, _, _, _ in tasks:
        print(f"  {method:20s} -> {device}")

    with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {executor.submit(_worker, t): t[0] for t in tasks}
        for future in as_completed(futures):
            method_name = futures[future]
            try:
                name, summary, error = future.result()
                if error:
                    print(f"\n[PARALLEL] {name} FAILED: {error}")
                else:
                    print(f"\n[PARALLEL] {name} completed ({len(summary)} datasets)")
            except Exception as e:
                print(f"\n[PARALLEL] {method_name} exception: {e}")


def run_task(task, methods, device, seed, method_config, gpus=None):
    """Run a single task for all methods."""
    print(f"\n\n{'#'*70}")
    print(f"# Task: {task}")
    print(f"{'#'*70}")

    run_fn = TASK_FUNCTIONS[task]

    if gpus:
        run_parallel(methods, gpus, seed, method_config, task)
    else:
        for method in methods:
            print(f"\n\n{'#'*70}")
            print(f"# Running: {method} ({task})")
            print(f"{'#'*70}")
            run_fn(method, device, seed, method_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Methods to run (default: all)")
    parser.add_argument("--task", default="vertical",
                        choices=ALL_TASKS + ["all"],
                        help="Integration task (default: vertical)")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--config", default=None,
                        help="YAML config file for method hyperparameters")
    parser.add_argument("--parallel", action="store_true",
                        help="Run methods in parallel across GPUs")
    parser.add_argument("--n-gpus", type=int, default=None,
                        help="Number of GPUs to use (0..N-1)")
    parser.add_argument("--gpus", default=None,
                        help="Comma-separated GPU IDs, e.g. '0,2,5'")
    args = parser.parse_args()

    DATA_ROOT = args.data_root
    methods = args.methods or ALL_METHODS

    from smobench.config import load_config
    method_config = load_config(args.config)
    if method_config:
        print(f"Loaded config with params for: {', '.join(method_config.keys())}")

    gpus = None
    if args.parallel:
        if args.gpus:
            gpus = [int(g) for g in args.gpus.split(",")]
        elif args.n_gpus:
            gpus = list(range(args.n_gpus))
        else:
            try:
                import torch
                n = torch.cuda.device_count()
                gpus = list(range(n)) if n > 0 else [0]
                print(f"Auto-detected {n} GPUs")
            except ImportError:
                gpus = [0]

    tasks = ALL_TASKS if args.task == "all" else [args.task]
    for task in tasks:
        run_task(task, methods, args.device, args.seed, method_config, gpus)

    print(f"\n{'='*70}")
    print(f"All results saved to h5ad files in: {SAVE_DIR}/")
    for task in tasks:
        print(f"  {task}: {SAVE_DIR}/{task}/")
    print(f"Load with: from smobench.io import load_results")
    print(f"{'='*70}")
