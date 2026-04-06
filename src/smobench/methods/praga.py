"""PRAGA method wrapper."""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from smobench.methods.base import BaseMethod
from smobench.methods.registry import MethodRegistry


class PRAGAMethod(BaseMethod):
    name = "PRAGA"
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
        import scanpy as sc

        from smobench.methods._vendor.praga.Train_model import Train
        from smobench.methods._vendor.praga.preprocess import (
            construct_neighbor_graph,
            pca,
            clr_normalize_each_cell,
            lsi,
            fix_seed,
        )

        fix_seed(seed)
        device_obj = self.resolve_device(device)

        # Build PRAGA args object
        class PRAGAArgs:
            pass

        args_praga = PRAGAArgs()
        args_praga.device = device
        args_praga.seed = seed
        args_praga.feat_n_comps = 50
        args_praga.n_neighbors = 3
        args_praga.KNN_k = 20

        # Data-type specific parameters
        # (RNA_weight, ADT_weight, cl_weight, alpha, tau, init_k) per data_type
        param_map = {
            "10x": (5, 5, 1, 0.9, 2, 10),
            "SPOTS": (1, 3, 1, 0.9, 2, 10),
            "Stereo-CITE-seq": (1, 3, 1, 0.9, 2, 10),
            "Spatial-epigenome-transcriptome": (1, 10, 2, 0.9, 2, 14),
            "MISAR": (1, 1, 5, 0.9, 2, 16),
        }
        rw, aw, cw, alpha, tau, init_k = param_map.get(data_type, (1, 3, 1, 0.9, 1, 6))
        args_praga.RNA_weight = rw
        args_praga.ADT_weight = aw
        args_praga.cl_weight = cw
        args_praga.alpha = alpha
        args_praga.tau = tau
        args_praga.init_k = init_k

        n_cells_original = adata_rna.n_obs

        adata_omics1 = adata_rna.copy()
        adata_omics2 = adata_mod2.copy()
        adata_omics1.var_names_make_unique()
        adata_omics2.var_names_make_unique()

        # Align cells
        common_obs = adata_omics1.obs_names.intersection(adata_omics2.obs_names)
        adata_omics1 = adata_omics1[common_obs].copy()
        adata_omics2 = adata_omics2[common_obs].copy()

        # RNA preprocessing (matching original GitHub)
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)
        adata_omics1.raw = adata_omics1
        adata_omics1 = adata_omics1[:, adata_omics1.var.highly_variable]

        # Second modality preprocessing (branch on data_type, matching original)
        if data_type in ("10x", "SPOTS", "Stereo-CITE-seq"):
            # Protein (ADT) data
            sc.pp.filter_genes(adata_omics2, min_cells=10)
            clr_normalize_each_cell(adata_omics2)
        else:
            # ATAC / chromatin data (LSI on raw counts, matching original)
            sc.pp.filter_genes(adata_omics2, min_cells=10)
            if "X_lsi" not in adata_omics2.obsm:
                sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
                lsi(adata_omics2, use_highly_variable=False, n_components=51)
            adata_omics2.obsm["feat"] = adata_omics2.obsm["X_lsi"].copy()

        # Post-HVG cell filter: only for Spatial-epigenome-transcriptome (matching original)
        _cell_mask = None
        if data_type == "Spatial-epigenome-transcriptome":
            import scipy.sparse
            X = adata_omics1.X
            if scipy.sparse.issparse(X):
                gene_counts = (X > 0).sum(axis=1).A1
            else:
                gene_counts = (X > 0).sum(axis=1)
            _cell_mask = np.asarray(gene_counts >= 200).ravel()
            adata_omics1 = adata_omics1[_cell_mask].copy()
            adata_omics2 = adata_omics2[_cell_mask].copy()

        # PCA n_comps: use adata_omics2.n_vars-1 for ADT types, feat_n_comps for ATAC
        if data_type in ("10x", "SPOTS", "Stereo-CITE-seq"):
            n_comps_target = min(adata_omics2.n_vars - 1, adata_omics1.n_obs - 1)
        else:
            n_comps_target = min(args_praga.feat_n_comps, adata_omics1.n_obs - 1, adata_omics1.n_vars - 1)

        # PCA for RNA
        adata_omics1.obsm["feat"] = pca(adata_omics1, n_comps=n_comps_target)

        # PCA for second modality if needed
        if "feat" not in adata_omics2.obsm:
            max_comps = min(adata_omics2.n_obs, adata_omics2.n_vars) - 1
            n_comps = min(n_comps_target, max_comps)
            adata_omics2.obsm["feat"] = pca(adata_omics2, n_comps=n_comps)

        # Construct neighbor graphs
        data = construct_neighbor_graph(
            adata_omics1,
            adata_omics2,
            datatype=data_type,
            n_neighbors=args_praga.n_neighbors,
            Arg=args_praga,
        )

        # Train
        model = Train(data, data_type, device_obj, seed, Arg=args_praga)
        output = model.train()
        embedding = output["PRAGA"]

        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        # Clean NaN/Inf
        if np.any(~np.isfinite(embedding)):
            embedding[np.isinf(embedding)] = (
                np.sign(embedding[np.isinf(embedding)]) * 1e10
            )
            embedding[np.isnan(embedding)] = 0

        # If cells were filtered, return kept indices so pipeline can subset
        if _cell_mask is not None and embedding.shape[0] < n_cells_original:
            kept_indices = np.where(_cell_mask)[0]
            return embedding, kept_indices

        return embedding


# Auto-register on import
MethodRegistry.register("PRAGA", method=PRAGAMethod())
