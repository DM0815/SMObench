"""CANDIES method wrapper."""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from smobench.methods.base import BaseMethod
from smobench.methods.registry import MethodRegistry


class CANDIESMethod(BaseMethod):
    name = "CANDIES"
    tasks = ["vertical", "horizontal"]
    modalities = ["RNA+ADT", "RNA+ATAC"]
    requires_gpu = True

    def check_deps(self):
        try:
            import torch
            import torch_geometric
            import einops
            import gudhi
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
        import pandas as pd
        import scanpy as sc

        _vendor_dir = os.path.join(os.path.dirname(__file__), "_vendor", "candies", "codes")
        if os.path.isdir(_vendor_dir) and _vendor_dir not in sys.path:
            sys.path.insert(0, _vendor_dir)

        from smobench.methods._vendor.candies.codes.preprocess1 import clr_normalize_each_cell, pca, lsi
        from smobench.methods._vendor.candies.codes.get_graph import construct_neighbor_graph, adjacent_matrix_preprocessing
        from smobench.methods._vendor.candies.codes.ZINB_encoder import encoder_ZINB
        from smobench.methods._vendor.candies.codes.AutoEncoder import train_model, train_atac
        from smobench.methods._vendor.candies.codes.integration import train_and_infer
        from smobench.methods._vendor.candies.codes.train_diff import run_diff

        def _seed_everything(s):
            torch.manual_seed(s)
            torch.cuda.manual_seed_all(s)
            np.random.seed(s)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        _seed_everything(seed)
        device_obj = self.resolve_device(device)

        n_cells_original = adata_rna.n_obs
        _original_obs = adata_rna.obs_names.copy()

        adata_omics1 = adata_rna.copy()
        adata_omics2 = adata_mod2.copy()
        adata_omics1.var_names_make_unique()
        adata_omics2.var_names_make_unique()

        # Align cells — track which original cells are kept
        common_obs = adata_omics1.obs_names.intersection(adata_omics2.obs_names)
        _common_mask = np.isin(_original_obs, common_obs)
        _kept_indices = np.where(_common_mask)[0]
        adata_omics1 = adata_omics1[common_obs].copy()
        adata_omics2 = adata_omics2[common_obs].copy()

        modality = kwargs.get("modality", "ADT")
        n_clusters = kwargs.get("n_clusters", 10)

        # RNA preprocessing
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)

        if modality == "ADT":
            adata_omics2 = clr_normalize_each_cell(adata_omics2)
            sc.pp.scale(adata_omics2)
            adata_omics2.obsm["feat"] = pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)
            adata_omics1_high = adata_omics1[:, adata_omics1.var["highly_variable"]]
            adata_omics1.obsm["feat"] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars - 1)
        else:
            if "X_lsi" not in adata_omics2.obsm:
                sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
                lsi(adata_omics2, use_highly_variable=False, n_components=51)
            adata_omics2.obsm["feat"] = adata_omics2.obsm["X_lsi"].copy()
            adata_omics1_high = adata_omics1[:, adata_omics1.var["highly_variable"]]
            adata_omics1.obsm["feat"] = pca(
                adata_omics1_high, n_comps=adata_omics2.obsm["X_lsi"].shape[1]
            )

        # Construct neighbor graphs
        adata_omics1, adata_omics2 = construct_neighbor_graph(adata_omics1, adata_omics2)
        adj = adjacent_matrix_preprocessing(adata_omics1, adata_omics2)
        adj_spatial_omics1 = adj["adj_spatial_omics1"].to(device_obj)
        adj_spatial_omics2 = adj["adj_spatial_omics2"].to(device_obj)

        # ZINB encoding for RNA
        ae_model = encoder_ZINB(
            adata=adata_omics1,
            device=device_obj,
            epochs=300,
            dim_output=64 if modality == "ADT" else 128,
        )
        adata_omics1.obsm["emb_ZINB"], _ = ae_model.train()

        # Second modality encoding
        _seed_everything(seed)
        if modality == "ADT":
            train_model(adata_omics1, adata_omics2, adj_spatial_omics1, adj_spatial_omics2, epochs=400)
        else:
            train_atac(adata_omics2, adj_spatial_omics2, epochs=600)

        # Pre-integration clustering (matching original)
        def _run_leiden_candies(adata1, n_cluster, use_rep="embeddings", key_added="Nleiden",
                                range_min=0, range_max=3, max_steps=30, tolerance=0):
            adata = adata1.copy()
            sc.pp.neighbors(adata, use_rep=use_rep)
            this_min = float(range_min)
            this_max = float(range_max)
            for _ in range(max_steps):
                this_resolution = this_min + ((this_max - this_min) / 2)
                sc.tl.leiden(adata, resolution=this_resolution)
                this_clusters = adata.obs["leiden"].nunique()
                if this_clusters > n_cluster + tolerance:
                    this_max = this_resolution
                elif this_clusters < n_cluster - tolerance:
                    this_min = this_resolution
                else:
                    adata1.obs[key_added] = adata.obs["leiden"]
                    return adata1
            adata1.obs[key_added] = adata.obs["leiden"]
            return adata1

        if modality == "ADT":
            cluster_num = min(10, n_clusters)
            adata_omics1 = _run_leiden_candies(adata_omics1, n_cluster=cluster_num, use_rep="emb_ZINB", key_added="AE")
            adata_omics2 = _run_leiden_candies(adata_omics2, n_cluster=cluster_num, use_rep="emb_latent_omics2", key_added="AE")
        else:
            cluster_num = min(14, n_clusters)
            adata_omics1 = _run_leiden_candies(adata_omics1, n_cluster=cluster_num, use_rep="emb_ZINB", key_added="AE")
            # ATAC uses mclust for omics2 (matching original mouse_embryo.ipynb)
            from smobench.clustering import cluster as _cluster
            _tmp = adata_omics2.copy()
            _tmp.obsm["X_emb"] = adata_omics2.obsm["emb_latent_omics2"]
            _tmp = _cluster(_tmp, method="mclust", n_clusters=cluster_num, embedding_key="X_emb", key_added="AE")
            adata_omics2.obs["AE"] = _tmp.obs["AE"]

        # Diffusion denoising
        slices_spatial = adata_omics1.obsm["spatial"]
        emb_omics1 = adata_omics1.obsm["emb_ZINB"]

        # For ADT: use 32-dim label embeddings from clusters (matching original)
        if modality == "ADT":
            ad2_ae = adata_omics2.obs["AE"]
            labels = torch.tensor(ad2_ae.values.astype(int), dtype=torch.long)
            num_classes = labels.max().item() + 1
            embedding_layer = torch.nn.Embedding(num_classes, 32)
            label_embeddings = embedding_layer(labels).detach().numpy()
            adata_omics2.obsm["label_embeddings_AE"] = label_embeddings
            emb_omics2 = adata_omics2.obsm["label_embeddings_AE"]
        else:
            emb_omics2 = adata_omics2.obsm["emb_latent_omics2"]

        df1 = pd.DataFrame(emb_omics1, index=[tuple(c) for c in slices_spatial])
        df2 = pd.DataFrame(emb_omics2, index=[tuple(c) for c in adata_omics2.obsm["spatial"]])
        df2_aligned = df2.reindex(df1.index)
        aligned_omics1 = df1.to_numpy()
        aligned_omics2 = df2_aligned.to_numpy()

        class ConditionalDiffusionDataset:
            def __init__(self, data1, data2):
                self.adata_omics1 = data1
                self.adata_omics2 = data2
                self.st_sample = torch.tensor(data1, dtype=torch.float32)
                self.con_sample = torch.tensor(data2, dtype=torch.float32)
                self.con_data = torch.tensor(data2, dtype=torch.float32)

            def __len__(self):
                return len(self.st_sample)

            def __getitem__(self, idx):
                return self.st_sample[idx], self.con_sample[idx], self.con_data

        # ADT: denoise RNA (omics1); ATAC: denoise ATAC (omics2)
        if modality == "ADT":
            dataset = ConditionalDiffusionDataset(aligned_omics1, aligned_omics2)
            denoise_target = "omics1"
        else:
            dataset = ConditionalDiffusionDataset(aligned_omics2, aligned_omics1)
            denoise_target = "omics2"

        _seed_everything(seed)
        com_mtx = run_diff(
            dataset,
            k=3,
            batch_size=min(512, adata_omics1.n_obs // 4),
            hidden_size=256,
            learning_rate=1e-3,
            num_epoch=1000,
            diffusion_step=800,
            depth=6,
            head=16,
            pca_dim=50,
            device=device_obj.type,
            classes=6,
            patience=40,
            bias=0.5 if modality == "ADT" else 1,
        )

        if denoise_target == "omics1":
            adata_omics1.obsm["denoise_emb"] = com_mtx
        else:
            adata_omics2.obsm["denoise_emb"] = com_mtx

        # Final integration
        adata1 = adata_omics1.copy()
        adata2 = adata_omics2.copy()

        # Spatial coordinate merge (matching original)
        spatial1 = pd.DataFrame(adata1.obsm["spatial"], columns=["x", "y"])
        spatial2 = pd.DataFrame(adata2.obsm["spatial"], columns=["x", "y"])
        spatial1["index1"] = spatial1.index
        spatial2["index2"] = spatial2.index
        merged = pd.merge(spatial1, spatial2, on=["x", "y"], how="inner")
        if len(merged) > 0:
            sorted_index1 = merged["index1"].values
            sorted_index2 = merged["index2"].values
            # Update kept_indices: further subset
            _kept_indices = _kept_indices[sorted_index1]
            adata1 = adata1[sorted_index1]
            adata2 = adata2[sorted_index2]

        if denoise_target == "omics1":
            adata1.obsm["feat"] = adata1.obsm["denoise_emb"]
            adata2.obsm["feat"] = adata2.obsm["emb_latent_omics2"]
        else:
            adata1.obsm["feat"] = adata1.obsm["emb_ZINB"]
            adata2.obsm["feat"] = adata2.obsm["denoise_emb"]

        adata1, adata2 = construct_neighbor_graph(adata1, adata2)
        adj = adjacent_matrix_preprocessing(adata1, adata2)

        features_omics1 = torch.FloatTensor(adata1.obsm["feat"].copy()).to(device_obj)
        features_omics2 = torch.FloatTensor(adata2.obsm["feat"].copy()).to(device_obj)

        _seed_everything(seed)
        result = train_and_infer(
            features_omics1=features_omics1,
            features_omics2=features_omics2,
            adj_spatial_omics1=adj["adj_spatial_omics1"].to(device_obj),
            adj_feature_omics1=adj["adj_feature_omics1"].to(device_obj),
            adj_spatial_omics2=adj["adj_spatial_omics2"].to(device_obj),
            adj_feature_omics2=adj["adj_feature_omics2"].to(device_obj),
            device=device_obj,
            epochs=200,
        )

        embedding = result["emb_latent_combined"].detach().cpu().numpy()

        # Clean NaN/Inf
        if np.any(~np.isfinite(embedding)):
            embedding[np.isinf(embedding)] = (
                np.sign(embedding[np.isinf(embedding)]) * 1e10
            )
            embedding[np.isnan(embedding)] = 0

        # Return kept_indices if cells were filtered
        if len(_kept_indices) < n_cells_original:
            return embedding, _kept_indices

        return embedding


# Auto-register on import
MethodRegistry.register("CANDIES", method=CANDIESMethod())
