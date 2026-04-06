"""spaMultiVAE method wrapper."""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from smobench.methods.base import BaseMethod
from smobench.methods.registry import MethodRegistry


class SpaMultiVAEMethod(BaseMethod):
    name = "spaMultiVAE"
    tasks = ["vertical", "horizontal"]
    modalities = ["RNA+ADT", "RNA+ATAC"]
    requires_gpu = True

    def check_deps(self):
        try:
            import torch
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
        import tempfile
        import torch
        import scanpy as sc
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.mixture import GaussianMixture

        from smobench.methods._vendor.spamultivae.spaMultiVAE import SPAMULTIVAE
        from smobench.methods._vendor.spamultivae.preprocess import normalize, geneSelection

        self.resolve_device(device)  # fail fast if CUDA unavailable
        torch.manual_seed(seed)
        np.random.seed(seed)

        n_cells_original = adata_rna.n_obs

        adata_omics1 = adata_rna.copy()
        adata_omics2 = adata_mod2.copy()
        adata_omics1.var_names_make_unique()
        adata_omics2.var_names_make_unique()

        # Align cells
        common = adata_omics1.obs_names.intersection(adata_omics2.obs_names)
        adata_omics1 = adata_omics1[common].copy()
        adata_omics2 = adata_omics2[common].copy()

        # Get expression matrices
        x1 = adata_omics1.X.toarray() if hasattr(adata_omics1.X, "toarray") else adata_omics1.X
        x2 = adata_omics2.X.toarray() if hasattr(adata_omics2.X, "toarray") else adata_omics2.X
        x1 = x1.astype("float64")
        x2 = x2.astype("float64")

        # Spatial coordinates
        if "spatial" in adata_omics1.obsm:
            loc = adata_omics1.obsm["spatial"].astype("float64")
        else:
            loc = np.random.rand(adata_omics1.n_obs, 2).astype("float64") * 100

        # Parameters
        GP_dim = kwargs.get("GP_dim", 2)
        Normal_dim = kwargs.get("Normal_dim", 18)
        encoder_layers = kwargs.get("encoder_layers", [128, 64])
        gene_decoder_layers = kwargs.get("gene_decoder_layers", [128])
        protein_decoder_layers = kwargs.get("protein_decoder_layers", [128])
        lr = kwargs.get("lr", 5e-3)
        maxiter = kwargs.get("maxiter", 5000)
        patience = kwargs.get("patience", 200)
        batch_size = kwargs.get("batch_size", "auto")
        loc_range = kwargs.get("loc_range", 20.0)
        kernel_scale = kwargs.get("kernel_scale", 20.0)
        inducing_point_steps = kwargs.get("inducing_point_steps", 19)

        if batch_size == "auto":
            if x1.shape[0] <= 1024:
                batch_size = 128
            elif x1.shape[0] <= 2048:
                batch_size = 256
            else:
                batch_size = 512

        # Gene selection
        select_genes = kwargs.get("select_genes", 0)
        if select_genes > 0:
            important_genes = geneSelection(x1, n=select_genes, plot=False)
            x1 = x1[:, important_genes]

        # Scale spatial coordinates
        scaler = MinMaxScaler()
        loc = scaler.fit_transform(loc) * loc_range

        # Grid inducing points
        eps = 1e-5
        initial_inducing_points = (
            np.mgrid[
                0 : (1 + eps) : (1.0 / inducing_point_steps),
                0 : (1 + eps) : (1.0 / inducing_point_steps),
            ]
            .reshape(2, -1)
            .T
            * loc_range
        )

        # Preprocess gene data
        adata1 = sc.AnnData(x1, dtype="float64")
        adata1 = normalize(adata1, size_factors=True, normalize_input=True, logtrans_input=True)

        # Preprocess protein data
        adata2 = sc.AnnData(x2, dtype="float64")
        adata2 = normalize(adata2, size_factors=False, normalize_input=True, logtrans_input=True)

        adata2_no_scale = sc.AnnData(x2, dtype="float64")
        adata2_no_scale = normalize(adata2_no_scale, size_factors=False, normalize_input=False, logtrans_input=True)

        # Align after normalization (normalize may filter cells)
        n_min = min(adata1.n_obs, adata2.n_obs, adata2_no_scale.n_obs, loc.shape[0])
        if adata1.n_obs != n_min or adata2.n_obs != n_min:
            adata1 = adata1[:n_min]
            adata2 = adata2[:n_min]
            adata2_no_scale = adata2_no_scale[:n_min]
            loc = loc[:n_min]

        # GMM for protein background
        gm = GaussianMixture(n_components=2, covariance_type="diag", n_init=20).fit(
            adata2_no_scale.X
        )
        back_idx = np.argmin(gm.means_, axis=0)
        protein_log_back_mean = np.log(
            np.expm1(gm.means_[back_idx, np.arange(adata2_no_scale.n_vars)])
        )
        protein_log_back_scale = np.sqrt(
            gm.covariances_[back_idx, np.arange(adata2_no_scale.n_vars)]
        )

        # Initialize model
        model = SPAMULTIVAE(
            gene_dim=adata1.n_vars,
            protein_dim=adata2.n_vars,
            GP_dim=GP_dim,
            Normal_dim=Normal_dim,
            encoder_layers=encoder_layers,
            gene_decoder_layers=gene_decoder_layers,
            protein_decoder_layers=protein_decoder_layers,
            gene_noise=0,
            protein_noise=0,
            encoder_dropout=0,
            decoder_dropout=0,
            fixed_inducing_points=True,
            initial_inducing_points=initial_inducing_points,
            fixed_gp_params=False,
            kernel_scale=kernel_scale,
            N_train=adata1.n_obs,
            KL_loss=0.025,
            dynamicVAE=True,
            init_beta=10,
            min_beta=4,
            max_beta=25,
            protein_back_mean=protein_log_back_mean,
            protein_back_scale=protein_log_back_scale,
            dtype=torch.float64,
            device=device,
        )

        # Train
        model.train_model(
            pos=loc,
            gene_ncounts=adata1.X,
            gene_raw_counts=adata1.raw.X,
            gene_size_factors=adata1.obs.size_factors,
            protein_ncounts=adata2.X,
            protein_raw_counts=adata2.raw.X,
            lr=lr,
            weight_decay=1e-6,
            batch_size=batch_size,
            num_samples=1,
            train_size=0.95,
            maxiter=maxiter,
            patience=patience,
            save_model=False,
            model_weights=os.path.join(tempfile.mkdtemp(prefix="spamultivae_"), "model.pt"),
        )

        # Extract embeddings
        embedding = model.batching_latent_samples(
            X=loc, gene_Y=adata1.X, protein_Y=adata2.X, batch_size=batch_size
        )

        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        # If cells were filtered (obs intersection or normalize), return kept indices
        if embedding.shape[0] < n_cells_original:
            # Best effort: return first N indices (normalize trims from end)
            kept_indices = np.arange(embedding.shape[0])
            return embedding, kept_indices

        return embedding


# Auto-register on import
MethodRegistry.register("spaMultiVAE", method=SpaMultiVAEMethod())
