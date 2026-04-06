"""MultiGATE method wrapper."""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from smobench.methods.base import BaseMethod
from smobench.methods.registry import MethodRegistry


class MultiGATEMethod(BaseMethod):
    name = "MultiGATE"
    tasks = ["vertical"]
    modalities = ["RNA+ADT", "RNA+ATAC"]
    requires_gpu = True
    env_group = "multigate"

    def check_deps(self):
        try:
            import tensorflow
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
        import warnings
        import pandas as pd
        import scanpy as sc

        import sys

        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

        # Inject vendor parent so 'import MultiGATE' resolves to _vendor/multigate
        _vendor_parent = os.path.join(os.path.dirname(__file__), "_vendor")
        if os.path.isdir(_vendor_parent) and _vendor_parent not in sys.path:
            sys.path.insert(0, _vendor_parent)

        import tensorflow.compat.v1 as _tf1
        import smobench.methods._vendor.multigate as MultiGATE
        from smobench.methods._vendor.multigate.MultiGATE import MultiGATE as _MultiGATEClass

        # Monkey-patch build_session for allow_soft_placement
        _orig_build_session = _MultiGATEClass.build_session

        def _patched_build_session(self, gpu=True):
            if self.config is not None:
                config = self.config
            else:
                config = _tf1.ConfigProto()
                config.gpu_options.allow_growth = True
                config.gpu_options.per_process_gpu_memory_fraction = 0.9
            config.allow_soft_placement = True
            if not gpu:
                config.intra_op_parallelism_threads = 0
                config.inter_op_parallelism_threads = 0
            self.session = _tf1.Session(config=config)
            self.session.run(
                [_tf1.global_variables_initializer(), _tf1.local_variables_initializer()]
            )

        _MultiGATEClass.build_session = _patched_build_session

        # Monkey-patch train_MultiGATE for gene_peak_Net propagation
        _orig_train = MultiGATE.train_MultiGATE

        def _patched_train(adata1, adata2, *args, **kw):
            if "gene_peak_Net" not in adata1.uns:
                if "gene_peak_Net" in adata2.uns:
                    adata1.uns["gene_peak_Net"] = adata2.uns["gene_peak_Net"]
            return _orig_train(adata1, adata2, *args, **kw)

        MultiGATE.train_MultiGATE = _patched_train

        warnings.filterwarnings("ignore")
        np.random.seed(seed)
        import tensorflow as _tf
        _tf.random.set_seed(seed)

        adata_omics1 = adata_rna.copy()
        adata_omics2 = adata_mod2.copy()
        adata_omics1.var_names_make_unique()
        adata_omics2.var_names_make_unique()

        # Align cells
        common = adata_omics1.obs_names.intersection(adata_omics2.obs_names)
        adata_omics1 = adata_omics1[common].copy()
        adata_omics2 = adata_omics2[common].copy()

        modality = kwargs.get("modality", "ADT")

        # Ensure spatial coordinates
        for ad, label in [(adata_omics1, "RNA"), (adata_omics2, modality)]:
            if "spatial" not in ad.obsm:
                sc.pp.neighbors(ad)
                sc.tl.umap(ad)
                ad.obsm["spatial"] = ad.obsm["X_umap"].copy()
        adata_omics1.obsm["spatial"][:, 1] *= -1
        adata_omics2.obsm["spatial"][:, 1] *= -1

        # RNA preprocessing
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        n_top = min(3000, adata_omics1.n_vars)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=n_top)
        adata_omics1 = adata_omics1[:, adata_omics1.var["highly_variable"]].copy()
        sc.pp.scale(adata_omics1, max_value=10)

        if modality == "ADT":
            mg_type = "protein"
            # CLR normalize
            data = adata_omics2.X.toarray() if not isinstance(adata_omics2.X, np.ndarray) else adata_omics2.X
            data = np.asarray(data, dtype=np.float64)
            clr = np.log1p(data)
            clr -= clr.mean(axis=1, keepdims=True)
            adata_omics2.X = clr
            sc.pp.scale(adata_omics2, max_value=10)
        else:
            mg_type = "ATAC_RNA"
            sc.pp.normalize_total(adata_omics2, target_sum=1e4)
            sc.pp.log1p(adata_omics2)
            n_top_feat = min(30000, adata_omics2.n_vars)
            sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=n_top_feat)
            adata_omics2 = adata_omics2[:, adata_omics2.var["highly_variable"]].copy()
            sc.pp.scale(adata_omics2, max_value=10)

        # Spatial graphs
        rad = kwargs.get("radius", 40 if modality == "ADT" else 100)
        MultiGATE.Cal_Spatial_Net(adata_omics1, rad_cutoff=rad)
        stats1 = MultiGATE.Stats_Spatial_Net(adata_omics1)
        # Warn if spatial graph is nearly fully connected
        spatial_net = adata_omics1.uns.get("Spatial_Net")
        if spatial_net is not None:
            avg_neigh = len(spatial_net) / max(adata_omics1.n_obs, 1)
            if avg_neigh > 100:
                sp = adata_omics1.obsm["spatial"]
                print(
                    f"  [MultiGATE WARNING] avg {avg_neigh:.0f} neighbors/cell (rad_cutoff={rad}). "
                    f"Spatial coords range [{sp.min():.0f}, {sp.max():.0f}]. "
                    f"Consider passing radius=<smaller value> if coords are in grid space."
                )
        MultiGATE.Cal_Spatial_Net(adata_omics2, rad_cutoff=rad)
        MultiGATE.Stats_Spatial_Net(adata_omics2)

        # Gene-feature relationships
        if modality == "ADT":
            MultiGATE.Cal_gene_protein_Net(adata_omics1, adata_omics2, verbose=True)
        else:
            gtf_path = kwargs.get("gtf_path", "")
            if not gtf_path:
                # Try to find GTF in data_info directory
                from smobench.data.registry import _find_default_root
                _data_root = _find_default_root()
                _candidate = os.path.join(_data_root, "data_info", "gencode.vM25.annotation.gtf.gz")
                if os.path.exists(_candidate):
                    gtf_path = _candidate
                else:
                    raise FileNotFoundError(
                        "GTF file not found. Please provide gtf_path via kwargs, e.g. "
                        "method.integrate(..., gtf_path='/path/to/gencode.gtf.gz')"
                    )
            max_distance = kwargs.get("gene_peak_max_distance", 200000)
            MultiGATE.Cal_gene_peak_Net_new(
                adata_omics1, adata_omics2, range=max_distance, file=gtf_path, verbose=True
            )
            # Bidirectional gene_peak_Net sync (matching original)
            in_rna = "gene_peak_Net" in adata_omics1.uns
            in_atac = "gene_peak_Net" in adata_omics2.uns
            if in_atac and not in_rna:
                adata_omics1.uns["gene_peak_Net"] = adata_omics2.uns["gene_peak_Net"]
            elif in_rna and not in_atac:
                adata_omics2.uns["gene_peak_Net"] = adata_omics1.uns["gene_peak_Net"]
            elif not in_rna and not in_atac:
                raise RuntimeError("gene_peak_Net not found in either adata after Cal_gene_peak_Net_new!")

        # Train
        n_epochs = kwargs.get("n_epochs", 1000)
        temperature = kwargs.get("temperature", 1.0)
        adata_omics1, adata_omics2 = MultiGATE.train_MultiGATE(
            adata_omics1,
            adata_omics2,
            n_epochs=n_epochs,
            temp=temperature,
            type=mg_type,
            save_attention=False,
            protein_value=0.001,
        )

        embedding = adata_omics1.obsm["MultiGATE_clip_all"]

        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        return embedding


# Auto-register on import
MethodRegistry.register("MultiGATE", method=MultiGATEMethod())
