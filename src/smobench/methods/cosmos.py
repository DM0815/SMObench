"""COSMOS method wrapper."""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from smobench.methods.base import BaseMethod
from smobench.methods.registry import MethodRegistry


class COSMOSMethod(BaseMethod):
    name = "COSMOS"
    tasks = ["vertical", "horizontal"]
    modalities = ["RNA+ADT", "RNA+ATAC"]
    requires_gpu = True

    def check_deps(self):
        try:
            import torch
            import torch_geometric
            import gudhi
            return True
        except ImportError:
            return False

    def integrate(
        self,
        adata_rna: AnnData,
        adata_mod2: AnnData,
        data_type: str = "10x",
        device: str = "cuda:0",
        seed: int = 2026,
        **kwargs,
    ) -> np.ndarray:
        import torch

        self.resolve_device(device)  # fail fast if CUDA unavailable

        from smobench.methods._vendor.cosmos import cosmos

        adata_omics1 = adata_rna.copy()
        adata_omics2 = adata_mod2.copy()
        adata_omics1.var_names_make_unique()
        adata_omics2.var_names_make_unique()

        # Align cells
        common_obs = adata_omics1.obs_names.intersection(adata_omics2.obs_names)
        adata_omics1 = adata_omics1[common_obs].copy()
        adata_omics2 = adata_omics2[common_obs].copy()

        # Initialize COSMOS model
        cosmos_model = cosmos.Cosmos(adata1=adata_omics1, adata2=adata_omics2)

        # Preprocessing (uses cell_ranger HVG flavor, matching upstream).
        # COSMOS was developed with pandas 1.5.2 where pd.cut() silently
        # dropped duplicate bin edges. Newer pandas raises ValueError.
        # Patch pd.cut to restore original behavior for sparse data.
        import pandas as pd
        _orig_cut = pd.cut

        def _pd_cut_compat(*args, **kwargs):
            kwargs.setdefault("duplicates", "drop")
            return _orig_cut(*args, **kwargs)

        pd.cut = _pd_cut_compat
        try:
            cosmos_model.preprocessing_data(
                do_norm=True,
                do_log=True,
                n_top_genes=3000,
                do_pca=False,
                n_neighbors=10,
            )
        finally:
            pd.cut = _orig_cut

        # Extract GPU ID
        gpu_id = 0
        if "cuda:" in device:
            gpu_id = int(device.split(":")[1])

        # Train
        embedding = cosmos_model.train(
            spatial_regularization_strength=0.01,
            z_dim=50,
            lr=1e-3,
            wnn_epoch=500,
            total_epoch=1000,
            max_patience_bef=10,
            max_patience_aft=30,
            min_stop=200,
            random_seed=seed,
            gpu=gpu_id,
            regularization_acceleration=True,
            edge_subset_sz=1000000,
        )

        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        # Clean NaN/Inf
        if np.any(~np.isfinite(embedding)):
            embedding[np.isinf(embedding)] = (
                np.sign(embedding[np.isinf(embedding)]) * 1e10
            )
            embedding[np.isnan(embedding)] = 0

        return embedding


# Auto-register on import
MethodRegistry.register("COSMOS", method=COSMOSMethod())
