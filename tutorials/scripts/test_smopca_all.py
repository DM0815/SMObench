"""Test SMOPCA on all datasets with all clustering methods."""

import os, sys, time, traceback
import numpy as np
import pandas as pd

# Config
DATA_ROOT = "/home/project/11003054/e1724738/_public/SMOBench/_github/benchmark/Dataset"
SAVE_DIR = os.path.join(os.path.dirname(__file__), "results")
DEVICE = "cuda:0"
SEED = 2026
METHOD = "SMOPCA"
CLUSTERING = ["leiden", "louvain", "kmeans", "mclust"]

os.environ["SMOBENCH_DATA_ROOT"] = DATA_ROOT
os.makedirs(SAVE_DIR, exist_ok=True)

from smobench._constants import DATASETS, get_n_clusters
from smobench.data import load_dataset
from smobench.pipeline._isolation import subprocess_integrate
from smobench.clustering import cluster
from smobench.metrics.evaluate import evaluate
from smobench.io import save_embedding, load_integrated

all_records = []

for ds_name, ds_info in DATASETS.items():
    for slice_name in ds_info["slices"]:
        print(f"\n{'='*70}")
        print(f"  {METHOD} | {ds_name} / {slice_name}")
        print(f"  modality={ds_info['modality']} gt={ds_info['gt']} n_clusters={get_n_clusters(ds_name, slice_name)}")
        print(f"{'='*70}")

        modality = "ADT" if "ADT" in ds_info["modality"] else "ATAC"
        label_key = "Spatial_Label" if ds_info["gt"] else None

        try:
            # Load
            adata_rna, adata_mod2 = load_dataset(ds_name, slice_name, DATA_ROOT)
            print(f"  Data loaded: RNA {adata_rna.shape}, mod2 {adata_mod2.shape}")

            # Integrate
            t0 = time.time()
            embedding, kept_indices = subprocess_integrate(
                METHOD, adata_rna, adata_mod2,
                device=DEVICE, seed=SEED, modality=modality,
            )
            runtime = time.time() - t0
            print(f"  Integration done: {runtime:.1f}s, embedding {embedding.shape}")

            # If method filtered cells, subset adata
            if embedding.shape[0] != adata_rna.n_obs:
                if kept_indices is not None:
                    print(f"  Method kept {len(kept_indices)}/{adata_rna.n_obs} cells.")
                    adata_rna = adata_rna[kept_indices].copy()
                else:
                    adata_rna = adata_rna[:embedding.shape[0]].copy()

            adata_rna.obsm[METHOD] = embedding

            # Save to h5ad
            out_dir = os.path.join(SAVE_DIR, "vertical", ds_name, slice_name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{METHOD}_integrated.h5ad")
            save_embedding(adata_rna, METHOD, embedding, out_path, train_time=runtime)

            # Cluster + evaluate
            for clust in CLUSTERING:
                clust_key = f"{METHOD}_{clust}"
                try:
                    cluster(adata_rna, method=clust, n_clusters=get_n_clusters(ds_name, slice_name),
                            embedding_key=METHOD, key_added=clust_key)
                    scores = evaluate(
                        adata_rna, embedding_key=METHOD, cluster_key=clust_key,
                        label_key=label_key, has_gt=ds_info["gt"],
                    )
                    record = {
                        "Dataset": ds_name, "Slice": slice_name,
                        "Modality": ds_info["modality"], "GT": ds_info["gt"],
                        "Clustering": clust, "Runtime": round(runtime, 1),
                        "Status": "OK", **scores,
                    }
                    all_records.append(record)
                    # Print key metrics
                    if ds_info["gt"]:
                        print(f"    {clust:10s} -> ARI={scores.get('ARI','?'):.3f} NMI={scores.get('NMI','?'):.3f} Moran_I={scores.get('Moran_I','?'):.3f}")
                    else:
                        print(f"    {clust:10s} -> Silhouette={scores.get('Silhouette','?'):.3f} DBI={scores.get('DBI','?'):.3f} Moran_I={scores.get('Moran_I','?'):.3f}")
                except Exception as e:
                    print(f"    {clust:10s} -> FAILED: {e}")
                    all_records.append({
                        "Dataset": ds_name, "Slice": slice_name,
                        "Modality": ds_info["modality"], "GT": ds_info["gt"],
                        "Clustering": clust, "Runtime": round(runtime, 1),
                        "Status": "CLUSTER_FAIL", "Error": str(e)[:200],
                    })

        except Exception as e:
            print(f"  INTEGRATION FAILED: {e}")
            traceback.print_exc()
            for clust in CLUSTERING:
                all_records.append({
                    "Dataset": ds_name, "Slice": slice_name,
                    "Modality": ds_info["modality"], "GT": ds_info["gt"],
                    "Clustering": clust, "Runtime": np.nan,
                    "Status": "INTEGRATE_FAIL", "Error": str(e)[:300],
                })

# Save progress
df = pd.DataFrame(all_records)
csv_path = os.path.join(SAVE_DIR, "smopca_all_results.csv")
df.to_csv(csv_path, index=False)

# Summary
print(f"\n\n{'='*70}")
print(f"SUMMARY: {METHOD} on all datasets")
print(f"{'='*70}")
print(f"Total records: {len(df)}")
print(f"Status: {df['Status'].value_counts().to_dict()}")

if "Runtime" in df.columns:
    ok = df[df["Status"] == "OK"]
    if len(ok) > 0:
        rt = ok.drop_duplicates(subset=["Dataset", "Slice"])["Runtime"]
        print(f"\nRuntime: mean={rt.mean():.1f}s, min={rt.min():.1f}s, max={rt.max():.1f}s")

        print(f"\nPer-dataset runtime:")
        for ds in ok["Dataset"].unique():
            ds_rt = ok[ok["Dataset"] == ds].drop_duplicates(subset=["Slice"])["Runtime"].mean()
            print(f"  {ds:25s}: {ds_rt:.1f}s")

        # Metrics by clustering
        print(f"\nMetrics by clustering method (mean across all slices):")
        for clust in CLUSTERING:
            c_df = ok[ok["Clustering"] == clust]
            if len(c_df) == 0:
                continue
            gt_df = c_df[c_df["GT"] == True]
            nogt_df = c_df[c_df["GT"] == False]
            print(f"\n  {clust}:")
            if len(gt_df) > 0:
                for col in ["ARI", "NMI", "cASW", "Moran_I"]:
                    if col in gt_df.columns:
                        print(f"    GT   {col:10s}: {gt_df[col].mean():.3f}")
            if len(nogt_df) > 0:
                for col in ["Silhouette", "DBI", "CHI", "Moran_I"]:
                    if col in nogt_df.columns:
                        print(f"    noGT {col:10s}: {nogt_df[col].mean():.3f}")

print(f"\nResults saved to {csv_path}")
