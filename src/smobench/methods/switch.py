"""SWITCH method wrapper."""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from smobench.methods.base import BaseMethod
from smobench.methods.registry import MethodRegistry


class SWITCHMethod(BaseMethod):
    name = "SWITCH"
    tasks = ["vertical"]
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
        import networkx as nx
        from itertools import chain
        from scipy import sparse

        # Inject vendor directory so 'import switch' resolves to _vendor copy
        _vendor_dir = os.path.join(os.path.dirname(__file__), "_vendor")
        _vendor_switch = os.path.join(_vendor_dir, "switch")
        if os.path.isdir(_vendor_switch) and _vendor_dir not in sys.path:
            sys.path.insert(0, _vendor_dir)

        import switch as sw

        device_obj = self.resolve_device(device)

        rna = adata_rna.copy()
        other = adata_mod2.copy()
        rna.var_names_make_unique()
        other.var_names_make_unique()

        # Sanitize var columns
        for ad in [rna, other]:
            for col in ["gene_ids", "feature_types", "genome"]:
                if col in ad.var.columns:
                    ad.var[col] = ad.var[col].astype(str)

        # Align cells
        common = rna.obs_names.intersection(other.obs_names)
        if common.empty:
            raise ValueError("No overlapping cells between RNA and secondary modality.")
        rna = rna[common].copy()
        other = other[common].copy()

        modality = kwargs.get("modality", "ADT")

        # Ensure spatial coordinates
        for ad in [rna, other]:
            if "spatial" not in ad.obsm:
                sc.pp.neighbors(ad)
                sc.tl.umap(ad)
                ad.obsm["spatial"] = ad.obsm["X_umap"].copy()

        # Gene annotation for ATAC
        gtf_path = kwargs.get("gtf_path", "")
        if modality == "ATAC" and gtf_path:
            required_cols = {"chrom", "chromStart", "chromEnd"}
            if not required_cols.issubset(rna.var.columns):
                sw.pp.get_gene_annotation(
                    rna, gtf=gtf_path, gtf_by=kwargs.get("gtf_by", "gene_name"), drop_na=True
                )

        # RNA preprocessing (matching original GitHub: HVG on raw counts, no normalize/log1p/scale)
        rna_hv = kwargs.get("rna_hv_genes", 2000)
        rna.layers["counts"] = rna.X.copy()
        sc.pp.highly_variable_genes(rna, n_top_genes=min(rna_hv, rna.n_vars), flavor="seurat_v3")

        # Second modality preprocessing
        if modality == "ATAC":
            # Parse peak coordinates
            split = other.var_names.str.split(r"[:-]")
            other.var["chrom"] = split.str[0]
            other.var["chromStart"] = split.str[1].astype(int)
            other.var["chromEnd"] = split.str[2].astype(int)
            sc.pp.filter_genes(other, min_cells=20)

            X_raw = other.X.toarray() if sparse.issparse(other.X) else np.asarray(other.X)
            X_raw = np.nan_to_num(X_raw, nan=0.0).astype(np.float32)
            X_raw = np.clip(X_raw, 0, None)

            is_integer = np.allclose(X_raw, np.round(X_raw))
            if not is_integer:
                _X = other.X.toarray() if sparse.issparse(other.X) else np.asarray(other.X, dtype=np.float64)
                _X = np.clip(_X, 0, None)
                _row_sums = _X.sum(axis=1, keepdims=True)
                _row_sums[_row_sums == 0] = 1.0
                _X = np.round(_X / _row_sums * 1e4).astype(np.float32)
                _X = np.clip(_X, 0, None)
                other.layers["counts"] = _X
            else:
                other.layers["counts"] = X_raw

            sc.pp.log1p(other)
            other_hv = kwargs.get("other_hv_features", 2000)
            sc.pp.highly_variable_genes(other, n_top_genes=min(other_hv, other.n_vars), flavor="seurat_v3")
            sc.pp.scale(other)
        else:  # ADT
            X = other.X.toarray() if sparse.issparse(other.X) else np.asarray(other.X, dtype=np.float32)
            X = np.nan_to_num(X, nan=0.0)
            X = np.clip(X, 0, None) + 1
            other.layers["counts"] = X.copy()
            other.X = X
            other.var["highly_variable"] = True

        # Build guidance graph
        if modality == "ATAC":
            guidance = sw.pp.rna_anchored_guidance_graph(
                rna, other,
                promoter_len=kwargs.get("promoter_len", 2000),
                extend_range=kwargs.get("extend_range", 0),
            )
        else:
            # Build protein guidance graph
            graph = nx.MultiDiGraph()
            rna_names = {name.upper(): name for name in rna.var_names}
            for item in set(rna.var_names).union(other.var_names):
                graph.add_edge(item, item, weight=1.0, sign=1, type="loop")
            import re as _re
            for prot in other.var_names:
                gene = rna_names.get(prot.upper())
                if gene is None:
                    stripped = _re.sub(r"^(?:(?:mouse|human|rat|ms|hu)[_-])+", "", prot, flags=_re.IGNORECASE)
                    gene = rna_names.get(stripped.upper())
                if gene is None:
                    continue
                graph.add_edge(gene, prot, weight=1.0, sign=1, type="fwd")
                graph.add_edge(prot, gene, weight=1.0, sign=1, type="rev")
            guidance = graph

        # Filter guidance to HV features
        guidance_hvf = guidance.subgraph(
            chain(
                rna.var.query("highly_variable").index,
                other.var.query("highly_variable").index,
            )
        ).copy()

        # Add missing features as self-loops
        all_hvf = set(rna.var.query("highly_variable").index).union(
            other.var.query("highly_variable").index
        )
        missing = all_hvf - set(guidance_hvf.nodes)
        for feat in missing:
            guidance_hvf.add_edge(feat, feat, weight=1.0, sign=1, type="loop")

        # Setup data
        sw.pp.setup_data(rna, prob_model="NB", use_highly_variable=True, use_layer="counts")
        sw.pp.setup_data(other, prob_model="NB", use_highly_variable=True, use_layer="counts")

        # Spatial graph with auto radius
        from sklearn.neighbors import NearestNeighbors

        coords = rna.obsm["spatial"]
        nn = NearestNeighbors(n_neighbors=11).fit(coords)
        dists, _ = nn.kneighbors(coords)
        auto_radius = float(np.median(dists[:, -1]) * 1.1)
        sw.pp.cal_spatial_net(rna, cutoff=auto_radius, model="Radius")
        sw.pp.cal_spatial_net(other, cutoff=auto_radius, model="Radius")

        # Monkey-patch GATEncoder.normalize for safety
        from switch.model import GATEncoder as _GATEncoder

        _orig_normalize = _GATEncoder.normalize

        def _safe_normalize(self, x, l):
            l = l.clamp(min=1.0)
            return (x * (self.TOTAL_COUNT / l)).log1p()

        _GATEncoder.normalize = _safe_normalize

        # Model
        key_other = modality.lower()
        adata_dict = {"rna": rna, key_other: other}

        latent_dim = kwargs.get("latent_dim", 50)
        hidden_dim = kwargs.get("hidden_dim", 256)

        model = sw.SWITCH(
            adatas=adata_dict,
            vertices=sorted(guidance_hvf.nodes),
            latent_dim=latent_dim,
            h_dim=hidden_dim,
            h_depth_enc=1,
            dropout=0.25,
            conv_layer="GAT",
            seed=seed,
            device=device_obj,
        )

        model.compile(
            lam_graph=0.35,
            lam_align=0.3,
            lam_cycle=1.0,
            lam_adv=0.02,
            lam_kl=1.0,
            vae_lr=2e-4,
            dsc_lr=5e-5,
        )

        pretrain_epochs = kwargs.get("pretrain_epochs", 3000)
        train_epochs = kwargs.get("train_epochs", 800)

        model.pretrain(adatas=adata_dict, graph=guidance_hvf, max_epochs=pretrain_epochs, dsc_k=4, warmup=True)
        model.train(adatas=adata_dict, graph=guidance_hvf, max_epochs=train_epochs, dsc_k=12)

        # Get RNA embedding
        embedding = np.asarray(model.encode_data("rna", rna))

        return embedding


# Auto-register on import
MethodRegistry.register("SWITCH", method=SWITCHMethod())
