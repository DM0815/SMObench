"""MISO method wrapper."""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from smobench.methods.base import BaseMethod
from smobench.methods.registry import MethodRegistry


class MISOMethod(BaseMethod):
    name = "MISO"
    tasks = ["vertical", "horizontal", "3m", "image"]
    modalities = ["RNA+ADT", "RNA+ATAC"]
    requires_gpu = True

    def check_deps(self):
        try:
            import torch
            import einops
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
        import scanpy as sc

        _vendor_dir = os.path.join(os.path.dirname(__file__), "_vendor", "miso")
        if os.path.isdir(_vendor_dir) and _vendor_dir not in sys.path:
            sys.path.insert(0, _vendor_dir)

        from miso import Miso
        from miso.utils import preprocess, set_random_seed

        set_random_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_obj = self.resolve_device(device)

        adata_omics1 = adata_rna.copy()
        adata_omics2 = adata_mod2.copy()
        adata_omics1.var_names_make_unique()
        adata_omics2.var_names_make_unique()

        # Align cells
        common = adata_omics1.obs_names.intersection(adata_omics2.obs_names)
        if len(common) == 0:
            raise ValueError("No shared cells between RNA and secondary modality.")
        adata_omics1 = adata_omics1[common].copy()
        adata_omics2 = adata_omics2[common].copy()

        # Preprocess via MISO utilities
        modality = kwargs.get("modality", "ADT")
        rna_feat = preprocess(adata_omics1, modality="rna")
        if modality == "ADT":
            other_feat = preprocess(adata_omics2, modality="protein")
        else:
            other_feat = preprocess(adata_omics2, modality="atac")

        features = [rna_feat.astype(np.float32), other_feat.astype(np.float32)]

        model = Miso(features, ind_views="all", combs="all", sparse=False, device=device_obj)
        model.train()
        embedding = model.emb.astype(np.float32)

        return embedding


# Auto-register on import
MethodRegistry.register("MISO", method=MISOMethod())
