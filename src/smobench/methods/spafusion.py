"""SpaFusion method wrapper."""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from smobench.methods.base import BaseMethod
from smobench.methods.registry import MethodRegistry


class SpaFusionMethod(BaseMethod):
    name = "SpaFusion"
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
        import tempfile
        from pathlib import Path

        _vendor_dir = os.path.join(os.path.dirname(__file__), "_vendor", "spafusion")
        if os.path.isdir(_vendor_dir) and _vendor_dir not in sys.path:
            sys.path.insert(0, _vendor_dir)

        from smobench.methods._vendor.spafusion.main import pre_train, train, setup_seed, load_data, norm_adj, adjacent_matrix_preprocessing
        from smobench.methods._vendor.spafusion.high_order_matrix import process_adjacency_matrix

        setup_seed(seed)
        device_obj = self.resolve_device(device)

        adata_omics1 = adata_rna.copy()
        adata_omics2 = adata_mod2.copy()
        adata_omics1.var_names_make_unique()
        adata_omics2.var_names_make_unique()

        modality = kwargs.get("modality", "ADT")
        n_clusters = kwargs.get("n_clusters", 10)
        spatial_k = kwargs.get("spatial_k", 9)
        adj_k = kwargs.get("adj_k", 20)
        dataset_name = kwargs.get("dataset_name", "default")

        view2 = "Protein" if modality == "ADT" else "Chromatin"

        adata_omics1, adata_omics2 = load_data(
            adata_omics1=adata_omics1,
            view1="RNA",
            adata_omics2=adata_omics2,
            view2=view2,
            n_neighbors=spatial_k,
            k=adj_k,
        )

        data1 = adata_omics1.obsm["feat"]
        data2 = adata_omics2.obsm["feat"]

        pre_adj_dir = Path(kwargs.get("pre_adj_dir", tempfile.mkdtemp(prefix="spafusion_")))
        pre_adj_dir.mkdir(parents=True, exist_ok=True)

        adj = adjacent_matrix_preprocessing(adata_omics1, adata_omics2, pre_adj_dir)

        feature_adj1 = adj["adj_feature_omics1"]
        feature_adj2 = adj["adj_feature_omics2"]
        spatial_adj1 = adj["adj_spatial_omics1"]
        spatial_adj2 = adj["adj_spatial_omics2"]

        Mt1_path = pre_adj_dir / f"Mt1_{dataset_name}.npy"
        Mt2_path = pre_adj_dir / f"Mt2_{dataset_name}.npy"
        Mt1 = process_adjacency_matrix(feature_adj1, Mt1_path)
        Mt2 = process_adjacency_matrix(feature_adj2, Mt2_path)

        # Normalize all matrices first (matching upstream main.py lines 205-210)
        feature_adj1 = norm_adj(feature_adj1)
        feature_adj2 = norm_adj(feature_adj2)
        spatial_adj1 = norm_adj(spatial_adj1)
        spatial_adj2 = norm_adj(spatial_adj2)
        Mt1 = norm_adj(Mt1)
        Mt2 = norm_adj(Mt2)

        # Ablation: spatial * feature on normalized matrices (upstream lines 220-222)
        spatial_adj1 = spatial_adj1 * feature_adj1
        spatial_adj2 = spatial_adj2 * feature_adj2

        to_tensor = lambda x: torch.tensor(x, dtype=torch.float32).to(device_obj)
        data1, data2 = to_tensor(data1), to_tensor(data2)
        feature_adj1, feature_adj2 = to_tensor(feature_adj1), to_tensor(feature_adj2)
        spatial_adj1, spatial_adj2 = to_tensor(spatial_adj1), to_tensor(spatial_adj2)
        Mt1, Mt2 = to_tensor(Mt1), to_tensor(Mt2)

        weight_list = kwargs.get("weight_list", [1, 1, 1, 1, 1, 1])
        lr = kwargs.get("lr", 1e-3)

        # Pretrain
        emb_comb, emb1, emb2 = pre_train(
            x1=data1, x2=data2,
            spatial_adj1=spatial_adj1, feature_adj1=feature_adj1,
            spatial_adj2=spatial_adj2, feature_adj2=feature_adj2,
            Mt1=Mt1, Mt2=Mt2,
            y=None, n_clusters=n_clusters,
            num_epoch=5000, device=device_obj, lr=lr,
            dataset_name=dataset_name, weight_list=weight_list,
        )

        # Train
        emb_comb, emb1, emb2 = train(
            x1=data1, x2=data2,
            spatial_adj1=spatial_adj1, feature_adj1=feature_adj1,
            spatial_adj2=spatial_adj2, feature_adj2=feature_adj2,
            y=None, n_clusters=n_clusters,
            Mt1=Mt1, Mt2=Mt2,
            num_epoch=2500,
            lambda1=1.0, lambda2=0.1,
            device=device_obj, seed=seed, lr=lr, num=0,
            dataset_name=dataset_name, weight_list=weight_list,
            spatial_K=spatial_k, adj_K=adj_k,
        )

        embedding = emb_comb.detach().cpu().numpy()

        return embedding


# Auto-register on import
MethodRegistry.register("SpaFusion", method=SpaFusionMethod())
