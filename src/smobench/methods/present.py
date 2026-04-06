"""PRESENT method wrapper."""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from smobench.methods.base import BaseMethod
from smobench.methods.registry import MethodRegistry


class PRESENTMethod(BaseMethod):
    name = "PRESENT"
    tasks = ["vertical", "horizontal"]
    modalities = ["RNA+ADT", "RNA+ATAC"]
    requires_gpu = True

    def check_deps(self):
        try:
            import torch
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

        self.resolve_device(device)  # fail fast if CUDA unavailable
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        from smobench.methods._vendor.present import PRESENT_function

        n_cells_original = adata_rna.n_obs
        _original_obs = adata_rna.obs_names.copy()

        adata_omics1 = adata_rna.copy()
        adata_omics2 = adata_mod2.copy()
        adata_omics1.var_names_make_unique()
        adata_omics2.var_names_make_unique()

        # Align cells
        common_obs = adata_omics1.obs_names.intersection(adata_omics2.obs_names)
        adata_omics1 = adata_omics1[common_obs].copy()
        adata_omics2 = adata_omics2[common_obs].copy()

        modality = kwargs.get("modality", "ADT")
        n_clusters = kwargs.get("n_clusters", 10)

        # Extract GPU device ID
        device_id = 0
        if "cuda:" in device:
            device_id = int(device.split(":")[1])

        adata_adt = adata_omics2 if modality == "ADT" else None
        adata_atac = adata_omics2 if modality != "ADT" else None

        adata_integrated = PRESENT_function(
            spatial_key="spatial",
            batch_key=None,
            adata_rna=adata_omics1,
            adata_atac=adata_atac,
            adata_adt=adata_adt,
            rdata_rna=None,
            rdata_rna_anno=None,
            rdata_atac=None,
            rdata_atac_anno=None,
            rdata_adt=None,
            rdata_adt_anno=None,
            gene_min_cells=1,
            peak_min_cells_fraction=0.03,
            protein_min_cells=1,
            num_hvg=3000,
            nclusters=n_clusters,
            d_lat=50,
            k_neighbors=6,
            intra_neighbors=6,
            inter_neighbors=6,
            epochs=100,
            lr=1e-3,
            batch_size=320,
            device="cuda",
            device_id=device_id,
        )

        embedding = adata_integrated.obsm["embeddings"].copy()

        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        # Clean NaN/Inf
        if np.any(~np.isfinite(embedding)):
            embedding[np.isinf(embedding)] = (
                np.sign(embedding[np.isinf(embedding)]) * 1e10
            )
            embedding[np.isnan(embedding)] = 0

        # PRESENT filters cells internally; return kept_indices if size differs
        if embedding.shape[0] < n_cells_original:
            kept_obs = adata_integrated.obs_names
            kept_indices = np.where(np.isin(_original_obs, kept_obs))[0]
            return embedding, kept_indices

        return embedding


# Auto-register on import
MethodRegistry.register("PRESENT", method=PRESENTMethod())
