"""SpatialGlue method wrapper."""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from smobench.methods.base import BaseMethod
from smobench.methods.registry import MethodRegistry


class SpatialGlueMethod(BaseMethod):
    name = "SpatialGlue"
    tasks = ["vertical", "horizontal", "3m"]
    modalities = ["RNA+ADT", "RNA+ATAC"]
    requires_gpu = True
    extras = ["torch", "pyg"]
    paper = "Long et al., Nature Methods, 2024"
    url = "https://github.com/JinmiaoChenLab/SpatialGlue"

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
        import os
        import sys

        # Inject vendor directory so internal relative imports work
        _vendor_sg = os.path.join(os.path.dirname(__file__), "_vendor", "spatialglue")
        if os.path.isdir(_vendor_sg) and _vendor_sg not in sys.path:
            sys.path.insert(0, _vendor_sg)

        try:
            import torch
            from smobench.methods._vendor.spatialglue.preprocess import (
                fix_seed, clr_normalize_each_cell, pca, lsi,
                construct_neighbor_graph,
            )
            from smobench.methods._vendor.spatialglue.SpatialGlue_pyG import Train_SpatialGlue
        except ImportError as e:
            raise ImportError(
                f"SpatialGlue dependencies not installed: {e}\n"
                f"Install with: pip install smobench[pyg]"
            ) from e

        import scanpy as sc

        fix_seed(seed)
        device = self.resolve_device(device)

        adata_omics1 = adata_rna.copy()
        adata_omics2 = adata_mod2.copy()

        # Preprocess RNA (matching original GitHub tutorials)
        adata_omics1.var_names_make_unique()
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)

        # Preprocess secondary modality
        adata_omics2.var_names_make_unique()
        modality = kwargs.get("modality", "ADT")
        if modality == "ADT":
            clr_normalize_each_cell(adata_omics2)
            sc.pp.scale(adata_omics2)
            n_comps_pca = min(adata_omics2.n_vars - 1, adata_omics1.n_obs - 1)
            adata_omics2.obsm["feat"] = pca(adata_omics2, n_comps=n_comps_pca)
            adata_omics1_high = adata_omics1[:, adata_omics1.var["highly_variable"]]
            adata_omics1.obsm["feat"] = pca(adata_omics1_high, n_comps=n_comps_pca)
        else:  # ATAC
            sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
            lsi(adata_omics2, use_highly_variable=False, n_components=51)
            adata_omics2.obsm["feat"] = adata_omics2.obsm["X_lsi"].copy()
            adata_omics1_high = adata_omics1[:, adata_omics1.var["highly_variable"]]
            adata_omics1.obsm["feat"] = pca(adata_omics1_high, n_comps=adata_omics2.obsm["X_lsi"].shape[1])

        # Build neighbor graphs
        construct_neighbor_graph(adata_omics1, adata_omics2, datatype=data_type)

        # Train
        data = {"adata_omics1": adata_omics1, "adata_omics2": adata_omics2}
        model = Train_SpatialGlue(data, datatype=data_type, device=device)
        output = model.train()
        emb = output["SpatialGlue"]
        embedding = emb.detach().cpu().numpy() if hasattr(emb, 'detach') else np.asarray(emb)

        return embedding


# Auto-register on import
MethodRegistry.register("SpatialGlue", method=SpatialGlueMethod())
