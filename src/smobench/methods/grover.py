"""GROVER method wrapper.

GROVER (Graph-guided Representation of Omics and Vision with Expert Regulation)
is a triplet-modality method that integrates RNA, protein/ADT, and image embeddings
using graph neural networks with a Mixture-of-Experts attention mechanism.

Requires: image_embeddings (pre-computed from a pathology foundation model).
Only runs on datasets with image data.
"""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from smobench.methods.base import BaseMethod
from smobench.methods.registry import MethodRegistry


class GROVERMethod(BaseMethod):
    name = "GROVER"
    tasks = ["vertical", "image"]
    modalities = ["RNA+ADT"]
    requires_gpu = True
    url = "https://github.com/Xubin-s-Lab/GROVER"

    def check_deps(self):
        try:
            import torch
            return True
        except ImportError:
            return False

    def integrate(
        self,
        adata_rna: AnnData,
        adata_mod2: AnnData,
        data_type: str = "10x",
        device: str = "cuda:0",
        seed: int = 2025,
        image_embeddings: np.ndarray | None = None,
        image_embeddings_path: str | None = None,
        locs_path: str | None = None,
        n_neighbors: int = 3,
        epochs: int = 300,
        learning_rate: float = 0.0001,
        dim_output: int = 64,
        weight_factors: list | None = None,
        **kwargs,
    ) -> np.ndarray:
        import os
        import sys
        import torch
        import scanpy as sc
        import anndata
        import pandas as pd

        _vendor_dir = os.path.join(os.path.dirname(__file__), "_vendor", "grover")
        if os.path.isdir(_vendor_dir) and _vendor_dir not in sys.path:
            sys.path.insert(0, _vendor_dir)

        from smobench.methods._vendor.grover.preprocess import (
            clr_normalize_each_cell, pca, construct_neighbor_graph, fix_seed,
        )
        from smobench.methods._vendor.grover.GROVER import Train_GROVER

        fix_seed(seed)
        device_obj = self.resolve_device(device)

        # --- Copy and align ---
        adata_omics1 = adata_rna.copy()
        adata_omics2 = adata_mod2.copy()
        adata_omics1.var_names_make_unique()
        adata_omics2.var_names_make_unique()

        common = adata_omics1.obs_names.intersection(adata_omics2.obs_names)
        if len(common) == 0:
            raise ValueError("No shared cells between RNA and secondary modality.")
        adata_omics1 = adata_omics1[common].copy()
        adata_omics2 = adata_omics2[common].copy()

        # --- Spatial coordinates ---
        if locs_path and os.path.isfile(locs_path):
            locs = pd.read_csv(locs_path, index_col=0)
            common_locs = adata_omics1.obs_names.intersection(locs.index)
            adata_omics1 = adata_omics1[common_locs].copy()
            adata_omics2 = adata_omics2[common_locs].copy()
            # Column names may be '2','3' or 'x','y' etc — use last two numeric cols
            num_cols = [c for c in locs.columns if locs[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]
            if len(num_cols) >= 2:
                spatial_coords = locs.loc[common_locs, num_cols[-2:]].values.astype(np.float64)
            else:
                spatial_coords = locs.loc[common_locs].iloc[:, -2:].values.astype(np.float64)
            adata_omics1.obsm['spatial'] = spatial_coords
            adata_omics2.obsm['spatial'] = spatial_coords
        elif 'spatial' not in adata_omics1.obsm:
            raise ValueError("No spatial coordinates found. Provide locs_path or ensure adata has .obsm['spatial'].")
        else:
            # Ensure both have spatial
            if 'spatial' not in adata_omics2.obsm:
                adata_omics2.obsm['spatial'] = adata_omics1.obsm['spatial'].copy()

        # --- Image embeddings ---
        if image_embeddings is None and image_embeddings_path:
            image_embeddings = np.load(image_embeddings_path)

        if image_embeddings is None:
            raise ValueError(
                "GROVER requires image embeddings. Provide image_embeddings "
                "or image_embeddings_path pointing to a .npy file."
            )

        n_cells = adata_omics1.n_obs
        if image_embeddings.shape[0] != n_cells:
            raise ValueError(
                f"Image embeddings shape {image_embeddings.shape[0]} != "
                f"number of cells {n_cells}. Ensure embeddings match cell count."
            )

        # Build image AnnData (stub, obsm only)
        adata_omics3 = anndata.AnnData(
            obs=adata_omics1.obs.copy(),
        )
        adata_omics3.obsm['feat'] = image_embeddings.astype(np.float32).copy()
        adata_omics3.obsm['spatial'] = adata_omics1.obsm['spatial'].copy()

        # --- Preprocess RNA ---
        n_protein = adata_omics2.n_vars
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=n_protein)

        # --- Preprocess Protein ---
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=n_protein)

        # --- Build neighbor graphs ---
        data = construct_neighbor_graph(adata_omics1, adata_omics2, adata_omics3,
                                        n_neighbors=n_neighbors)

        # --- Train ---
        if weight_factors is None:
            weight_factors = [1, 1, 1, 2, 2, 2]

        model = Train_GROVER(
            data,
            datatype='Triplet',
            device=device_obj,
            random_seed=seed,
            learning_rate=learning_rate,
            weight_decay=0.0,
            epochs=epochs,
            dim_input=3000,
            dim_output=dim_output,
            weight_factors=weight_factors,
        )
        output = model.train()

        embedding = output['GROVER'].astype(np.float32)
        return embedding


# Auto-register on import
MethodRegistry.register("GROVER", method=GROVERMethod())
