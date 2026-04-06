"""SpaBalance method wrapper."""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from smobench.methods.base import BaseMethod
from smobench.methods.registry import MethodRegistry


class SpaBalanceMethod(BaseMethod):
    name = "SpaBalance"
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
        import os
        import sys
        import torch
        import scanpy as sc

        _vendor_dir = os.path.join(os.path.dirname(__file__), "_vendor", "spabalance")
        if os.path.isdir(_vendor_dir) and _vendor_dir not in sys.path:
            sys.path.insert(0, _vendor_dir)

        from smobench.methods._vendor.spabalance.preprocess import fix_seed, clr_normalize_each_cell, pca, lsi, construct_neighbor_graph
        from smobench.methods._vendor.spabalance.Train_model import Train_SpaBalance

        fix_seed(seed)
        device_obj = self.resolve_device(device)

        adata_omics1 = adata_rna.copy()
        adata_omics2 = adata_mod2.copy()
        adata_omics1.var_names_make_unique()
        adata_omics2.var_names_make_unique()

        modality = kwargs.get("modality", "ADT")

        # RNA preprocessing
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)

        # Second modality preprocessing
        if modality == "ADT":
            adata_omics2 = clr_normalize_each_cell(adata_omics2)
            sc.pp.scale(adata_omics2)
            n_comps = min(adata_omics2.n_vars - 1, adata_omics1.n_obs - 1)
            adata_omics2.obsm["feat"] = pca(adata_omics2, n_comps=n_comps)
        else:  # ATAC
            if "X_lsi" not in adata_omics2.obsm:
                sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
                lsi(adata_omics2, use_highly_variable=False, n_components=51)
            adata_omics2.obsm["feat"] = adata_omics2.obsm["X_lsi"].copy()
            n_comps = adata_omics2.obsm["feat"].shape[1]

        adata_omics1_high = adata_omics1[:, adata_omics1.var["highly_variable"]]
        adata_omics1.obsm["feat"] = pca(adata_omics1_high, n_comps=n_comps)

        # Construct graph and train
        data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=data_type)
        model = Train_SpaBalance(data, datatype=data_type, device=device_obj)
        output = model.train()

        embedding = output["SpaBalance"].copy()

        # Clean NaN/Inf
        if np.any(~np.isfinite(embedding)):
            embedding[np.isnan(embedding)] = 0
            embedding[np.isinf(embedding)] = np.sign(embedding[np.isinf(embedding)]) * 1e10

        return embedding


# Auto-register on import
MethodRegistry.register("SpaBalance", method=SpaBalanceMethod())
