"""SMOPCA method wrapper."""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from smobench.methods.base import BaseMethod
from smobench.methods.registry import MethodRegistry


class SMOPCAMethod(BaseMethod):
    name = "SMOPCA"
    tasks = ["vertical", "horizontal"]
    modalities = ["RNA+ADT", "RNA+ATAC"]
    requires_gpu = False
    env_group = "base"

    def check_deps(self):
        try:
            import scanpy
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
        import scanpy as sc

        from smobench.methods._vendor.spatialglue.preprocess import fix_seed
        fix_seed(seed)

        from smobench.methods._vendor.smopca import model as smopca_model_module

        adata_omics1 = adata_rna.copy()
        adata_omics2 = adata_mod2.copy()
        adata_omics1.var_names_make_unique()
        adata_omics2.var_names_make_unique()

        # Align cells
        common = adata_omics1.obs_names.intersection(adata_omics2.obs_names)
        adata_omics1 = adata_omics1[common].copy()
        adata_omics2 = adata_omics2[common].copy()

        modality = kwargs.get("modality", "ADT")

        # RNA preprocessing (HVG on raw counts with seurat_v3, then normalize)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=1000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        adata_omics1 = adata_omics1[:, adata_omics1.var["highly_variable"]].copy()
        sc.pp.scale(adata_omics1)

        # Second modality preprocessing
        if modality == "ADT":
            # CLR normalization for protein data
            from smobench.methods._vendor.spatialglue.preprocess import clr_normalize_each_cell
            clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)

        X1 = adata_omics1.X.A if hasattr(adata_omics1.X, "A") else adata_omics1.X
        X2 = adata_omics2.X.A if hasattr(adata_omics2.X, "A") else adata_omics2.X

        # Spatial coordinates
        if "spatial" not in adata_omics1.obsm:
            sc.pp.neighbors(adata_omics1)
            sc.tl.umap(adata_omics1)
            adata_omics1.obsm["spatial"] = adata_omics1.obsm["X_umap"]

        pos = adata_omics1.obsm["spatial"]

        # Train SMOPCA
        z_dim = kwargs.get("z_dim", 20)
        max_latent_dim = min(z_dim, X1.shape[1], X2.shape[1])

        smopca = smopca_model_module.SMOPCA(
            Y_list=[X1.T, X2.T],
            Z_dim=max_latent_dim,
            pos=pos,
            intercept=False,
            omics_weight=False,
        )

        smopca.estimateParams(
            sigma_init_list=(1, 1),
            tol_sigma=2e-5,
            sigma_xtol_list=(1e-6, 1e-6),
            gamma_init=1,
            estimate_gamma=True,
        )
        z = smopca.calculatePosterior()
        embedding = np.asarray(z)

        return embedding


# Auto-register on import
MethodRegistry.register("SMOPCA", method=SMOPCAMethod())
