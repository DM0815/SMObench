"""SpaMI method wrapper."""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from smobench.methods.base import BaseMethod
from smobench.methods.registry import MethodRegistry


class SpaMIMethod(BaseMethod):
    name = "SpaMI"
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
        import scipy
        import scanpy as sc

        self.resolve_device(device)  # fail fast if CUDA unavailable
        import sklearn.decomposition
        import sklearn.preprocessing
        import sklearn.utils.extmath

        _vendor_dir = os.path.join(os.path.dirname(__file__), "_vendor", "spami")
        if os.path.isdir(_vendor_dir) and _vendor_dir not in sys.path:
            sys.path.insert(0, _vendor_dir)

        from smobench.methods._vendor.spami.preprocess import tfidf, construct_adj, add_contrastive_label
        from smobench.methods._vendor.spami.utils import fix_seed
        from smobench.methods._vendor.spami.main import train

        fix_seed(seed)

        adata_omics1 = adata_rna.copy()
        adata_omics2 = adata_mod2.copy()
        adata_omics1.var_names_make_unique()
        adata_omics2.var_names_make_unique()

        modality = kwargs.get("modality", "ADT")

        # RNA preprocessing
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.highly_variable_genes(adata_omics1, n_top_genes=3000)
        adata_omics1 = adata_omics1[:, adata_omics1.var["highly_variable"]]
        X = adata_omics1.X.toarray() if scipy.sparse.issparse(adata_omics1.X) else adata_omics1.X
        feat_omics1 = sklearn.decomposition.PCA(n_components=50).fit_transform(X)

        # Second modality preprocessing
        if modality == "ATAC":
            try:
                import episcanpy
                episcanpy.pp.binarize(adata_omics2)
                episcanpy.pp.filter_features(
                    adata_omics2, min_cells=np.ceil(0.005 * adata_omics2.shape[0])
                )
            except ImportError:
                sc.pp.filter_genes(adata_omics2, min_cells=int(np.ceil(0.005 * adata_omics2.shape[0])))
            count_mat = adata_omics2.X.copy()
            X2 = tfidf(count_mat)
            X2_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X2)
            X2_norm = np.log1p(X2_norm * 1e4)
            U, _, _ = sklearn.utils.extmath.randomized_svd(X2_norm, n_components=51)
            X_lsi = (U - U.mean(1, keepdims=True)) / U.std(1, ddof=1, keepdims=True)
            feat_omics2 = X_lsi[:, 1:]
        else:  # ADT
            sc.pp.filter_genes(adata_omics2, min_cells=50)

            def seurat_clr(x):
                s = np.sum(np.log1p(x[x > 0]))
                exp = np.exp(s / len(x)) if len(x) > 0 else 1
                return np.log1p(x / exp)

            X2 = adata_omics2.X.toarray() if scipy.sparse.issparse(adata_omics2.X) else np.array(adata_omics2.X)
            adata_omics2.X = np.apply_along_axis(seurat_clr, 1, X2)
            feat_omics2 = adata_omics2.X

        # Construct graphs and contrastive labels
        adj_omics1, graph_neigh_omics1 = construct_adj(adata_omics1, n_neighbors=3)
        adj_omics2, graph_neigh_omics2 = construct_adj(adata_omics2, n_neighbors=3)
        label_CSL_omics1 = add_contrastive_label(adata_omics1)
        label_CSL_omics2 = add_contrastive_label(adata_omics2)

        omics1_data = {
            "feat": feat_omics1,
            "adj": adj_omics1,
            "graph_neigh": graph_neigh_omics1,
            "label_CSL": label_CSL_omics1,
        }
        omics2_data = {
            "feat": feat_omics2,
            "adj": adj_omics2,
            "graph_neigh": graph_neigh_omics2,
            "label_CSL": label_CSL_omics2,
        }

        out_dim = 64 if modality == "ADT" else 20
        embedding = train(omics1_data, omics2_data, data_type, out_dim=out_dim)

        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        return embedding


# Auto-register on import
MethodRegistry.register("SpaMI", method=SpaMIMethod())
