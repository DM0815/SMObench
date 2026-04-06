"""SpaMV method wrapper."""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from smobench.methods.base import BaseMethod
from smobench.methods.registry import MethodRegistry


class SpaMVMethod(BaseMethod):
    name = "SpaMV"
    tasks = ["vertical", "horizontal"]
    modalities = ["RNA+ADT", "RNA+ATAC"]
    requires_gpu = True

    def check_deps(self):
        try:
            import torch
            import pyro
            import torch_geometric
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

        from smobench.methods._vendor.spamv.spamv import SpaMV
        from smobench.methods._vendor.spamv.utils import preprocess_dc

        device_obj = self.resolve_device(device)

        n_cells_original = adata_rna.n_obs
        _original_obs = adata_rna.obs_names.copy()

        adata_omics1 = adata_rna.copy()
        adata_omics2 = adata_mod2.copy()
        adata_omics1.var_names_make_unique()
        adata_omics2.var_names_make_unique()

        modality = kwargs.get("modality", "ADT")

        if modality == "ADT":
            datasets = [adata_omics1, adata_omics2]
            omics_names = ["Transcriptome", "Proteome"]
            scale = True
        else:
            datasets = [adata_omics1, adata_omics2]
            omics_names = ["Transcriptome", "Epigenome"]
            scale = False

        # Preprocess using SpaMV built-in
        datasets = preprocess_dc(
            datasets,
            omics_names,
            prune=True,
            min_cells=10,
            min_genes=200,
            hvg=True,
            n_top_genes=3000,
            normalization=True,
            target_sum=1e4,
            log1p=True,
            scale=scale,
        )

        # Train
        model = SpaMV(
            adatas=datasets,
            interpretable=False,
            device=device_obj,
            random_seed=seed,
            max_epochs_stage1=400,
            max_epochs_stage2=400,
            early_stopping=True,
            patience=200,
        )

        model.train()
        embedding = model.get_embedding()

        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        # Clean NaN/Inf
        if np.any(~np.isfinite(embedding)):
            embedding[np.isinf(embedding)] = (
                np.sign(embedding[np.isinf(embedding)]) * 1e10
            )
            embedding[np.isnan(embedding)] = 0

        # If cells were filtered, return kept indices
        if embedding.shape[0] < n_cells_original:
            kept_obs = datasets[0].obs_names
            kept_indices = np.where(np.isin(_original_obs, kept_obs))[0]
            return embedding, kept_indices

        return embedding


# Auto-register on import
MethodRegistry.register("SpaMV", method=SpaMVMethod())
