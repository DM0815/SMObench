"""Run all (or selected) methods on all datasets, save results.

Usage:
    python run_all_methods.py                          # all 15 methods
    python run_all_methods.py --methods SpatialGlue SMOPCA  # specific methods
    python run_all_methods.py --methods SMOPCA --device cpu  # CPU only
"""

import os, sys, time, argparse, traceback, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Variable names are not unique")

# ── Config ──────────────────────────────────────────────────────
DATA_ROOT = "/home/project/11003054/e1724738/_public/SMOBench/_github/benchmark/Dataset"
SAVE_DIR = os.path.join(os.path.dirname(__file__), "results")
DEVICE = "cuda:0"
SEED = 2026
CLUSTERING = ["leiden", "louvain", "kmeans", "mclust"]

ALL_METHODS = [
    "SpatialGlue", "CANDIES", "COSMOS", "MISO", "PRAGA", "PRESENT",
    "SpaBalance", "SpaFusion", "SpaMI", "SpaMosaic", "SpaMV",
    "spaMultiVAE", "SWITCH", "MultiGATE", "SMOPCA", "SMART",
]


def run_one_method(method_name, device, seed):
    """Run a single method on all datasets, return list of records."""
    os.environ["SMOBENCH_DATA_ROOT"] = DATA_ROOT
    os.makedirs(SAVE_DIR, exist_ok=True)

    from smobench._constants import DATASETS, get_n_clusters, METHOD_DATASET_SKIP
    from smobench.data import load_dataset
    from smobench.pipeline._isolation import subprocess_integrate
    from smobench.clustering import cluster
    from smobench.metrics.evaluate import evaluate
    from smobench.io import save_embedding

    skip_datasets = METHOD_DATASET_SKIP.get(method_name, set())
    records = []

    for ds_name, ds_info in DATASETS.items():
        if ds_name in skip_datasets:
            print(f"\n  [SKIP] {method_name} does not support {ds_name} (upstream limitation)")
            continue

        for slice_name in ds_info["slices"]:
            print(f"\n{'='*70}")
            print(f"  {method_name} | {ds_name} / {slice_name}")
            print(f"  modality={ds_info['modality']} gt={ds_info['gt']} "
                  f"n_clusters={get_n_clusters(ds_name, slice_name)}")
            print(f"{'='*70}")

            modality = "ADT" if "ADT" in ds_info["modality"] else "ATAC"
            label_key = "Spatial_Label" if ds_info["gt"] else None
            n_clusters = get_n_clusters(ds_name, slice_name)

            try:
                # Load
                adata_rna, adata_mod2 = load_dataset(ds_name, slice_name, DATA_ROOT)
                print(f"  Data loaded: RNA {adata_rna.shape}, mod2 {adata_mod2.shape}")

                # Integrate
                t0 = time.time()
                embedding, kept_indices = subprocess_integrate(
                    method_name, adata_rna, adata_mod2,
                    device=device, seed=seed, modality=modality,
                )
                runtime = time.time() - t0
                print(f"  Integration done: {runtime:.1f}s, embedding {embedding.shape}")

                # If method filtered cells, subset adata
                if embedding.shape[0] != adata_rna.n_obs:
                    if kept_indices is not None:
                        print(f"  Method kept {len(kept_indices)}/{adata_rna.n_obs} cells. "
                              f"Subsetting adata for metrics.")
                        adata_rna = adata_rna[kept_indices].copy()
                    else:
                        print(f"  WARNING: embedding {embedding.shape[0]} != adata {adata_rna.n_obs}, "
                              f"no kept_indices. Using first {embedding.shape[0]} cells.")
                        adata_rna = adata_rna[:embedding.shape[0]].copy()

                adata_rna.obsm[method_name] = embedding

                # Save to h5ad
                out_dir = os.path.join(SAVE_DIR, "vertical", ds_name, slice_name)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{method_name}_integrated.h5ad")
                save_embedding(adata_rna, method_name, embedding, out_path, train_time=runtime)

                # Cluster + evaluate
                for clust in CLUSTERING:
                    clust_key = f"{method_name}_{clust}"
                    try:
                        cluster(adata_rna, method=clust, n_clusters=n_clusters,
                                embedding_key=method_name, key_added=clust_key)
                        scores = evaluate(
                            adata_rna, embedding_key=method_name, cluster_key=clust_key,
                            label_key=label_key, has_gt=ds_info["gt"],
                        )
                        record = {
                            "Method": method_name,
                            "Dataset": ds_name, "Slice": slice_name,
                            "Modality": ds_info["modality"], "GT": ds_info["gt"],
                            "Clustering": clust, "Runtime": round(runtime, 1),
                            "Status": "OK", **scores,
                        }
                        records.append(record)
                        if ds_info["gt"]:
                            print(f"    {clust:10s} -> ARI={scores.get('ARI','?'):.3f} "
                                  f"NMI={scores.get('NMI','?'):.3f} "
                                  f"Moran_I={scores.get('Moran_I','?'):.3f}")
                        else:
                            print(f"    {clust:10s} -> Silhouette={scores.get('Silhouette','?'):.3f} "
                                  f"DBI={scores.get('DBI','?'):.3f} "
                                  f"Moran_I={scores.get('Moran_I','?'):.3f}")
                    except Exception as e:
                        print(f"    {clust:10s} -> FAILED: {e}")
                        records.append({
                            "Method": method_name,
                            "Dataset": ds_name, "Slice": slice_name,
                            "Modality": ds_info["modality"], "GT": ds_info["gt"],
                            "Clustering": clust, "Runtime": round(runtime, 1),
                            "Status": "CLUSTER_FAIL", "Error": str(e)[:200],
                        })

            except Exception as e:
                print(f"  INTEGRATION FAILED: {e}")
                traceback.print_exc()
                for clust in CLUSTERING:
                    records.append({
                        "Method": method_name,
                        "Dataset": ds_name, "Slice": slice_name,
                        "Modality": ds_info["modality"], "GT": ds_info["gt"],
                        "Clustering": clust, "Runtime": np.nan,
                        "Status": "INTEGRATE_FAIL", "Error": str(e)[:300],
                    })

            # Save progress after each slice
            df_tmp = pd.DataFrame(records)
            csv_tmp = os.path.join(SAVE_DIR, f"{method_name.lower()}_all_results.csv")
            df_tmp.to_csv(csv_tmp, index=False)

    return records


def print_summary(df):
    """Print summary of results."""
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total records: {len(df)}")
    print(f"Status: {df['Status'].value_counts().to_dict()}")

    ok = df[df["Status"] == "OK"]
    if len(ok) == 0:
        return

    for method in ok["Method"].unique():
        m_df = ok[ok["Method"] == method]
        rt = m_df.drop_duplicates(subset=["Dataset", "Slice"])["Runtime"]
        print(f"\n  {method}: {len(m_df)} records, "
              f"runtime mean={rt.mean():.1f}s")

        gt_df = m_df[m_df["GT"] == True]
        nogt_df = m_df[m_df["GT"] == False]
        if len(gt_df) > 0:
            for col in ["ARI", "NMI", "cASW", "Moran_I"]:
                if col in gt_df.columns and gt_df[col].notna().any():
                    print(f"    GT   {col:10s}: {gt_df[col].mean():.3f}")
        if len(nogt_df) > 0:
            for col in ["Silhouette", "DBI", "CHI", "Moran_I"]:
                if col in nogt_df.columns and nogt_df[col].notna().any():
                    print(f"    noGT {col:10s}: {nogt_df[col].mean():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Methods to run (default: all 15)")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--data-root", default=DATA_ROOT)
    args = parser.parse_args()

    DATA_ROOT = args.data_root
    methods = args.methods or ALL_METHODS

    all_records = []
    for method in methods:
        print(f"\n\n{'#'*70}")
        print(f"# Running: {method}")
        print(f"{'#'*70}")
        records = run_one_method(method, args.device, args.seed)
        all_records.extend(records)

    # Save combined results
    df = pd.DataFrame(all_records)
    combined_csv = os.path.join(SAVE_DIR, "all_methods_results.csv")
    df.to_csv(combined_csv, index=False)
    print(f"\nCombined results saved to {combined_csv}")

    print_summary(df)
