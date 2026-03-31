#!/usr/bin/env python3
"""
Regenerate Mouse_Thymus ADT fusion data with improved protein name normalization.

Problem: Original fusion only got 10 proteins because Thymus1 uses different
naming convention (Mouse-CD8a) from Thymus2/3/4 (mouse_CD8a), plus chained
species prefixes (mouse_rat_CD29, mouse_human_CD44) weren't handled.

Fix: Improved normalize_protein_name in generate_fusion_data.py now handles
chained prefixes and alternate suffixes.
"""
import os
import sys

# Setup paths
project_root = "/home/users/nus/e1724738/_main/_private/NUS/_Proj1/SMOBench-CLEAN"
sys.path.insert(0, os.path.join(project_root, "_myx_Scripts/data_preparation"))
os.chdir(project_root)

import scanpy as sc
import numpy as np
from pathlib import Path
from generate_fusion_data import fuse_adt_data, fuse_rna_data, _sanity_check

base_path = Path(project_root) / "Dataset"

# Also check Mouse_Spleen for similar issues
print("=" * 60)
print("Step 0: Check Mouse_Spleen for naming inconsistencies")
print("=" * 60)
spleen_dir = base_path / "woGT" / "RNA_ADT" / "Mouse_Spleen"
for s in ["Mouse_Spleen1", "Mouse_Spleen2"]:
    a = sc.read_h5ad(spleen_dir / s / "adata_ADT.h5ad")
    print(f"  {s}: {a.n_vars} proteins: {list(a.var_names)}")

# Regenerate Mouse_Thymus
print("\n" + "=" * 60)
print("Step 1: Load Mouse_Thymus ADT data from all 4 sections")
print("=" * 60)

thymus_dir = base_path / "woGT" / "RNA_ADT" / "Mouse_Thymus"
sections = ["Mouse_Thymus1", "Mouse_Thymus2", "Mouse_Thymus3", "Mouse_Thymus4"]
dataset_names = [f"Mouse_Thymus_{s.replace('Mouse_Thymus', 'Thymus')}" for s in sections]

adatas = []
for i, section in enumerate(sections):
    f = thymus_dir / section / "adata_ADT.h5ad"
    a = sc.read_h5ad(f)
    a.obs['batch'] = dataset_names[i]
    a.var_names_make_unique()
    adatas.append(a)
    print(f"  {section}: {a.n_obs} cells, {a.n_vars} proteins")
    print(f"    var_names: {list(a.var_names)}")

print(f"\n" + "=" * 60)
print("Step 2: Fuse ADT data with improved normalization")
print("=" * 60)

merged_adt = fuse_adt_data(adatas, dataset_names)

if merged_adt is not None:
    print(f"\n  Result: {merged_adt.n_obs} cells, {merged_adt.n_vars} proteins")
    print(f"  Protein names: {list(merged_adt.var_names)}")

    # Save
    output_dir = base_path / "_myx_fusionWoGT" / "RNA_ADT"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "Mouse_Thymus_Fusion_ADT.h5ad"

    # Backup old file
    if output_file.exists():
        backup = output_file.with_suffix('.h5ad.bak_10proteins')
        print(f"\n  Backing up old file ({10} proteins) to: {backup.name}")
        output_file.rename(backup)

    _sanity_check(merged_adt, name="Mouse_Thymus_ADT")
    merged_adt.write(output_file)
    print(f"  Saved new fusion: {output_file}")
    print(f"  Old: 10 proteins -> New: {merged_adt.n_vars} proteins")
else:
    print("  ERROR: ADT fusion failed!")
    sys.exit(1)

# Also regenerate RNA to keep cell counts consistent
print(f"\n" + "=" * 60)
print("Step 3: Regenerate Mouse_Thymus RNA fusion (for consistency)")
print("=" * 60)

rna_adatas = []
for i, section in enumerate(sections):
    f = thymus_dir / section / "adata_RNA.h5ad"
    a = sc.read_h5ad(f)
    a.obs['batch'] = dataset_names[i]
    a.var_names_make_unique()
    rna_adatas.append(a)
    print(f"  {section}: {a.n_obs} cells, {a.n_vars} genes")

merged_rna = fuse_rna_data(rna_adatas, dataset_names)
if merged_rna is not None:
    rna_output = output_dir / "Mouse_Thymus_Fusion_RNA.h5ad"
    _sanity_check(merged_rna, name="Mouse_Thymus_RNA")
    merged_rna.write(rna_output)
    print(f"  Saved: {rna_output}")
    print(f"  {merged_rna.n_obs} cells, {merged_rna.n_vars} genes")

print(f"\n" + "=" * 60)
print("DONE")
print("=" * 60)
