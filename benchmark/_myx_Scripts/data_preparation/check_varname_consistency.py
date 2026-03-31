#!/usr/bin/env python3
"""Check var_names consistency across batches for all datasets."""
import anndata as ad
from pathlib import Path

base = Path("/home/users/nus/e1724738/_main/_private/NUS/_Proj1/SMOBench-CLEAN/Dataset")

datasets = {
    "withGT/RNA_ADT/Human_Lymph_Nodes": {"batches": ["A1", "D1"], "modalities": ["ADT", "RNA"]},
    "withGT/RNA_ADT/Human_Tonsils": {"batches": ["S1", "S2", "S3"], "modalities": ["ADT", "RNA"]},
    "withGT/RNA_ATAC/Mouse_Embryos_S1": {"batches": ["E11", "E13", "E15", "E18"], "modalities": ["ATAC", "RNA"]},
    "withGT/RNA_ATAC/Mouse_Embryos_S2": {"batches": ["E11", "E13", "E15", "E18"], "modalities": ["ATAC", "RNA"]},
    "woGT/RNA_ADT/Mouse_Spleen": {"batches": ["Mouse_Spleen1", "Mouse_Spleen2"], "modalities": ["ADT", "RNA"]},
    "woGT/RNA_ADT/Mouse_Thymus": {"batches": ["Mouse_Thymus1", "Mouse_Thymus2", "Mouse_Thymus3", "Mouse_Thymus4"], "modalities": ["ADT", "RNA"]},
}

for ds_path, info in datasets.items():
    for mod in info["modalities"]:
        var_counts = []
        var_sets = []
        for batch in info["batches"]:
            f = base / ds_path / batch / f"adata_{mod}.h5ad"
            if f.exists():
                a = ad.read_h5ad(f, backed='r')
                vn = set(a.var_names)
                var_counts.append((batch, a.n_vars, a.n_obs))
                var_sets.append(vn)
                a.file.close()

        if len(var_sets) < 2:
            continue

        intersection = var_sets[0]
        union = var_sets[0]
        for s in var_sets[1:]:
            intersection = intersection & s
            union = union | s

        counts_str = ", ".join(f"{b}={n}" for b, n, _ in var_counts)
        status = "OK" if len(intersection) == len(union) else "MISMATCH"

        if status == "MISMATCH":
            print(f"[{status}] {ds_path} {mod}: intersection={len(intersection)}, union={len(union)} ({counts_str})")
            # Show first few differences
            for i, (batch, nv, _) in enumerate(var_counts):
                only_here = var_sets[i] - intersection
                if only_here:
                    examples = sorted(only_here)[:5]
                    print(f"  {batch} unique ({len(only_here)}): {examples}...")
        else:
            print(f"[{status}]      {ds_path} {mod}: {len(intersection)} features ({counts_str})")
