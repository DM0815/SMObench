"""SMObench command-line interface.

Usage:
    smobench run --dataset Human_Lymph_Nodes --methods SpatialGlue --task vertical
    smobench eval --input results.h5ad --metrics all
    smobench list methods
    smobench list datasets
"""

from __future__ import annotations

import sys


def app():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        _print_help()
        return

    cmd = sys.argv[1]

    if cmd == "run":
        _cmd_run(sys.argv[2:])
    elif cmd == "eval":
        _cmd_eval(sys.argv[2:])
    elif cmd == "list":
        _cmd_list(sys.argv[2:])
    elif cmd == "plot":
        _cmd_plot(sys.argv[2:])
    elif cmd == "setup":
        _cmd_setup(sys.argv[2:])
    elif cmd == "init":
        _cmd_init(sys.argv[2:])
    elif cmd in ("-h", "--help", "help"):
        _print_help()
    elif cmd == "--version":
        from smobench import __version__
        print(f"smobench {__version__}")
    else:
        print(f"Unknown command: {cmd}")
        _print_help()
        sys.exit(1)


def _print_help():
    print("""SMObench: Spatial Multi-Omics Integration Benchmark

Usage:
    smobench run       Run benchmark (integrate + cluster + evaluate)
    smobench eval      Evaluate existing integration results
    smobench list      List available methods or datasets
    smobench plot      Generate plots from results
    smobench setup     Set up environments for methods
    smobench init      Generate default config file
    smobench --version Show version

Run options:
    --dataset       Dataset name(s), comma-separated or "all"
    --methods       Method name(s), comma-separated or "all"
    --task          vertical, horizontal, mosaic, or all
    --clustering    Clustering methods, comma-separated (default: leiden,kmeans)
    --device        GPU device (default: cuda:0)
    --n_jobs        Parallel jobs (default: 1)
    --config        YAML config file (overrides other args)
    --output        Output CSV path
    --data_root     Dataset root directory
    --save_dir      Save integrated adata files

Examples:
    smobench run --dataset Human_Lymph_Nodes --methods SpatialGlue,SpaMosaic --task vertical
    smobench run --config smobench_config.yaml
    smobench list methods
    smobench list datasets
    smobench plot --input results.csv --type heatmap
""")


def _parse_args(argv, valid_args):
    """Simple arg parser: --key value pairs."""
    args = {}
    i = 0
    while i < len(argv):
        if argv[i].startswith("--"):
            key = argv[i][2:]
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                args[key] = argv[i + 1]
                i += 2
            else:
                args[key] = True
                i += 1
        else:
            i += 1
    return args


def _cmd_run(argv):
    """Run benchmark."""
    args = _parse_args(argv, [])

    # Support --config for YAML config file
    if "config" in args:
        from smobench.pipeline.config import load_config
        config = load_config(args["config"])
        # CLI args override config
        for k, v in args.items():
            if k != "config":
                config[k] = v
    else:
        config = {}

    dataset = args.get("dataset", config.get("datasets", "all"))
    methods = args.get("methods", config.get("methods", "all"))
    task = args.get("task", config.get("task", "vertical"))
    output = args.get("output", config.get("output", "smobench_results.csv"))
    device = args.get("device", config.get("device", "cuda:0"))
    clustering = args.get("clustering", config.get("clustering", "leiden,kmeans"))
    n_jobs = int(args.get("n_jobs", config.get("n_jobs", 1)))
    seed = int(args.get("seed", config.get("seed", 42)))
    data_root = args.get("data_root", config.get("data_root"))
    save_dir = args.get("save_dir", config.get("save_dir"))

    if isinstance(methods, str) and methods != "all":
        methods = [m.strip() for m in methods.split(",")]
    if isinstance(dataset, str) and dataset != "all":
        dataset = [d.strip() for d in dataset.split(",")]

    if isinstance(clustering, str):
        clustering = [c.strip() for c in clustering.split(",")]

    from smobench.pipeline import benchmark
    results = benchmark(
        dataset=dataset,
        methods=methods,
        task=task,
        clustering=clustering,
        device=device,
        seed=seed,
        n_jobs=n_jobs,
        data_root=data_root,
        save_dir=save_dir,
    )

    results.save(output)
    print(f"\nResults saved to: {output}")


def _cmd_eval(argv):
    """Evaluate existing results."""
    args = _parse_args(argv, [])
    input_path = args.get("input")
    if not input_path:
        print("ERROR: --input required")
        sys.exit(1)

    from smobench.metrics.evaluate import evaluate
    from smobench.io import load_integrated

    adata = load_integrated(input_path)
    embedding_keys = [k for k in adata.obsm.keys() if k not in ("spatial", "X_umap", "X_pca")]

    print(f"Found embeddings: {embedding_keys}")
    for key in embedding_keys:
        print(f"\n--- {key} ---")
        scores = evaluate(adata, embedding_key=key, cluster_key=f"{key}_leiden")
        for metric, value in sorted(scores.items()):
            print(f"  {metric}: {value:.4f}")


def _cmd_list(argv):
    """List methods or datasets."""
    if not argv:
        print("Usage: smobench list [methods|datasets]")
        return

    what = argv[0]
    if what == "methods":
        from smobench.methods import list_methods
        df = list_methods()
        # Add env status column
        from smobench._env import ENV_GROUPS, check_current_env
        df["EnvGroup"] = df["Method"].map(lambda m: ENV_GROUPS.get(m, "?"))
        df["Available"] = df["Method"].map(
            lambda m: "yes" if check_current_env(m) else "no"
        )
        print(df.to_string(index=False))
    elif what == "datasets":
        from smobench.data import list_datasets
        df = list_datasets()
        print(df.to_string(index=False))
    else:
        print(f"Unknown: {what}. Use 'methods' or 'datasets'.")


def _cmd_plot(argv):
    """Generate plots."""
    args = _parse_args(argv, [])
    input_path = args.get("input")
    plot_type = args.get("type", "heatmap")
    output = args.get("output")

    if not input_path:
        print("ERROR: --input required")
        sys.exit(1)

    import pandas as pd
    from smobench.pipeline.benchmark import BenchmarkResult

    df = pd.read_csv(input_path)
    result = BenchmarkResult(records=df.to_dict("records"))

    if plot_type == "heatmap":
        result.plot.heatmap(save=output)
    elif plot_type == "scatter":
        result.plot.scatter(save=output)
    elif plot_type == "radar":
        result.plot.radar(save=output)
    else:
        print(f"Unknown plot type: {plot_type}")


def _cmd_setup(argv):
    """Set up environments for method groups."""
    args = _parse_args(argv, [])
    backend = args.get("backend", "conda")
    groups = args.get("group", args.get("groups", "all"))

    from smobench._env import ENV_GROUPS, GROUP_CONDA_NAMES

    # Determine which groups to set up
    all_groups = sorted(set(ENV_GROUPS.values()))
    if groups == "all" or groups is True:
        target_groups = all_groups
    else:
        target_groups = [g.strip() for g in str(groups).split(",")]

    # Also support --methods to resolve groups from method names
    methods_arg = args.get("methods")
    if methods_arg:
        method_names = [m.strip() for m in str(methods_arg).split(",")]
        target_groups = sorted(set(
            ENV_GROUPS.get(m, "torch-pyg") for m in method_names
        ))

    print(f"Setting up environments for groups: {', '.join(target_groups)}")
    print(f"Backend: {backend}\n")

    if backend == "conda":
        _setup_conda(target_groups)
    elif backend == "singularity":
        _setup_singularity(target_groups)
    else:
        print(f"Unknown backend: {backend}. Use 'conda' or 'singularity'.")
        sys.exit(1)


def _setup_conda(groups):
    """Create conda environments for method groups."""
    import subprocess
    from smobench._env import GROUP_CONDA_NAMES

    # Base deps shared by all envs
    base_deps = "python=3.10 scanpy anndata numpy scipy"

    group_deps = {
        "base": base_deps,
        "torch-pyg": f"{base_deps} pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia && pip install torch-geometric pyg-lib torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu118.html",
        "spamosaic": f"{base_deps} pytorch torchvision pytorch-cuda=11.8 dgl -c pytorch -c nvidia -c dglteam/label/th21_cu118 && pip install torch-geometric",
        "multigate": f"{base_deps} tensorflow==1.15 keras==2.3.1",
    }

    for group in groups:
        env_name = GROUP_CONDA_NAMES.get(group, f"smobench_{group}")
        deps = group_deps.get(group, base_deps)
        print(f"--- Creating conda env: {env_name} ---")
        print(f"  conda create -n {env_name} {deps}")
        print(f"  conda run -n {env_name} pip install smobench")
        print()

    print("Run the commands above to create the environments.")
    print("Or use --auto to create them automatically (experimental).")

    # TODO: --auto flag to actually run conda create


def _setup_singularity(groups):
    """Show instructions for singularity setup."""
    from smobench._env import GROUP_SIF_NAMES

    for group in groups:
        sif = GROUP_SIF_NAMES.get(group, f"smobench_{group}.sif")
        print(f"--- Group: {group} ---")
        print(f"  Image: {sif}")
        print(f"  Build: singularity build {sif} docker://smobench/{group}:latest")
        print()


def _cmd_init(argv):
    """Generate default config file."""
    args = _parse_args(argv, [])
    path = args.get("output", "smobench_config.yaml")
    from smobench.pipeline.config import generate_default_config
    generate_default_config(path)


if __name__ == "__main__":
    app()
