"""SpaMosaic method wrapper."""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from smobench.methods.base import BaseMethod
from smobench.methods.registry import MethodRegistry


class SpaMosaicMethod(BaseMethod):
    name = "SpaMosaic"
    tasks = ["vertical", "horizontal", "mosaic"]
    modalities = ["RNA+ADT", "RNA+ATAC"]
    requires_gpu = True
    env_group = "spamosaic"

    def check_deps(self):
        try:
            import torch
            import dgl
            import torch_geometric
            import hnswlib
            import annoy
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
        import os
        import sys
        import torch

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # Inject vendor directory so 'import spamosaic' resolves to _vendor copy
        _vendor_dir = os.path.join(os.path.dirname(__file__), "_vendor")
        _vendor_spamosaic = os.path.join(_vendor_dir, "spamosaic")
        if os.path.isdir(_vendor_spamosaic) and _vendor_dir not in sys.path:
            sys.path.insert(0, _vendor_dir)

        try:
            from spamosaic.framework import SpaMosaic
            from spamosaic.preprocessing import (
                RNA_preprocess,
                ADT_preprocess,
                Epigenome_preprocess,
            )
        except ImportError as e:
            raise ImportError(
                f"SpaMosaic dependencies not installed: {e}\n"
                f"Install with: pip install smobench[spamosaic]"
            ) from e

        device_obj = self.resolve_device(device)

        adata_omics1 = adata_rna.copy()
        adata_omics2 = adata_mod2.copy()
        adata_omics1.var_names_make_unique()
        adata_omics2.var_names_make_unique()

        modality = kwargs.get("modality", "ADT")
        if modality == "ADT":
            input_dict = {"rna": [adata_omics1], "adt": [adata_omics2]}
        else:
            input_dict = {"rna": [adata_omics1], "atac": [adata_omics2]}

        # Add batch key for SpaMosaic
        for key in input_dict:
            for adata in input_dict[key]:
                if adata is not None and "src" not in adata.obs.columns:
                    adata.obs["src"] = "batch0"

        input_key = "dimred_bc"

        RNA_preprocess(
            input_dict["rna"],
            batch_corr=True,
            favor="scanpy",
            n_hvg=5000,
            batch_key="src",
            key=input_key,
        )

        if modality == "ADT":
            ADT_preprocess(
                input_dict["adt"], batch_corr=True, batch_key="src", key=input_key
            )
        else:
            Epigenome_preprocess(
                input_dict["atac"], batch_corr=True, batch_key="src", key=input_key
            )

        model = SpaMosaic(
            modBatch_dict=input_dict,
            input_key=input_key,
            batch_key="src",
            intra_knns=10,
            seed=seed,
            device=device,
        )

        model.train(net="wlgcn", lr=0.01, T=0.01, n_epochs=100)

        ad_embs = model.infer_emb(input_dict, emb_key="emb", final_latent_key="merged_emb")
        embedding = ad_embs[0].obsm["merged_emb"].copy()

        # Clean NaN/Inf
        if np.any(~np.isfinite(embedding)):
            embedding[np.isinf(embedding)] = (
                np.sign(embedding[np.isinf(embedding)]) * 1e10
            )
            embedding[np.isnan(embedding)] = 0

        return embedding


# Auto-register on import
MethodRegistry.register("SpaMosaic", method=SpaMosaicMethod())
