"""SMART method wrapper."""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from smobench.methods.base import BaseMethod
from smobench.methods.registry import MethodRegistry


class SMARTMethod(BaseMethod):
    name = "SMART"
    tasks = ["vertical", "horizontal"]
    modalities = ["RNA+ADT", "RNA+ATAC"]
    requires_gpu = True
    extras = ["torch", "pyg"]
    paper = "Huang et al., 2025"
    url = "https://github.com/Xubin-s-Lab/SMART-main"

    def check_deps(self):
        try:
            import torch
            import torch_geometric
            import numba
            import muon
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

        device_obj = self.resolve_device(device)

        # Inject vendor directory so `smart` package resolves
        _vendor_smart = os.path.join(os.path.dirname(__file__), "_vendor", "smart")
        if os.path.isdir(_vendor_smart) and _vendor_smart not in sys.path:
            sys.path.insert(0, _vendor_smart)

        import torch
        import scanpy as sc
        from muon import prot as pt
        from muon import atac as ac
        from smart.train import train_SMART
        from smart.utils import set_seed, pca
        from smart.build_graph import Cal_Spatial_Net
        from smart.MNN import Mutual_Nearest_Neighbors

        set_seed(seed)

        adata_omics1 = adata_rna.copy()
        adata_omics2 = adata_mod2.copy()
        adata_omics1.var_names_make_unique()
        adata_omics2.var_names_make_unique()

        # Align cells
        common_obs = adata_omics1.obs_names.intersection(adata_omics2.obs_names)
        adata_omics1 = adata_omics1[common_obs].copy()
        adata_omics2 = adata_omics2[common_obs].copy()

        modality = kwargs.get("modality", "ADT")

        # ── RNA preprocessing (same for both modalities) ──
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)

        adata_omics1_high = adata_omics1[:, adata_omics1.var["highly_variable"]]

        if modality == "ADT":
            # Protein preprocessing (matching Tutorial 2)
            adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
            pt.pp.clr(adata_omics2)
            sc.pp.scale(adata_omics2)
            adata_omics1.obsm["feat"] = pca(adata_omics1_high, n_comps=30)
            adata_omics2.obsm["feat"] = pca(adata_omics2, n_comps=30)
            n_neighbors = 6
            lr = 5e-3
            weight_decay = 1e-6
        else:
            # ATAC preprocessing (matching Tutorial 3)
            adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
            ac.pp.tfidf(adata_omics2, scale_factor=1e4)
            sc.pp.normalize_per_cell(adata_omics2, counts_per_cell_after=1e4)
            sc.pp.log1p(adata_omics2)
            adata_omics1.obsm["feat"] = pca(adata_omics1_high, n_comps=30)
            adata_omics2.obsm["feat"] = pca(adata_omics2, n_comps=60)
            n_neighbors = 4
            lr = 1e-3
            weight_decay = 1e-5

        # ── Spatial neighbor graph ──
        Cal_Spatial_Net(adata_omics1, model="KNN", n_neighbors=n_neighbors)
        Cal_Spatial_Net(adata_omics2, model="KNN", n_neighbors=n_neighbors)

        # ── MNN triplet samples ──
        adata_list = [adata_omics1, adata_omics2]
        x = [torch.FloatTensor(adata.obsm["feat"]).to(device_obj) for adata in adata_list]
        edges = [torch.LongTensor(adata.uns["edgeList"]).to(device_obj) for adata in adata_list]
        triplet_samples_list = [
            Mutual_Nearest_Neighbors(adata, key="feat", n_nearest_neighbors=3, farthest_ratio=0.6)
            for adata in adata_list
        ]

        # ── Train ──
        model = train_SMART(
            features=x,
            edges=edges,
            triplet_samples_list=triplet_samples_list,
            weights=[1, 1, 1, 1],
            emb_dim=64,
            n_epochs=300,
            lr=lr,
            weight_decay=weight_decay,
            device=device_obj,
            window_size=10,
            slope=1e-4,
        )

        embedding = model(x, edges)[0].cpu().detach().numpy()

        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        return embedding


# Auto-register on import
MethodRegistry.register("SMART", method=SMARTMethod())
