"""Base class for integration methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from anndata import AnnData


class BaseMethod(ABC):
    """Abstract base class for spatial multi-omics integration methods.

    Subclass this to add a new method to SMObench:

        class MyMethod(BaseMethod):
            name = "MyMethod"
            tasks = ["vertical"]
            modalities = ["RNA+ADT", "RNA+ATAC"]

            def integrate(self, adata_rna, adata_mod2, **kwargs):
                # Your integration logic
                return embedding  # np.ndarray of shape (n_cells, n_dims)
    """

    name: str = ""
    tasks: list[str] = []          # "vertical", "horizontal", "mosaic"
    modalities: list[str] = []     # "RNA+ADT", "RNA+ATAC"
    requires_gpu: bool = True
    env_group: str = "torch-pyg"   # environment group for isolation
    extras: list[str] = []         # pip extras needed, e.g. ["pyg"]
    paper: str = ""                # Citation
    url: str = ""                  # GitHub URL

    @abstractmethod
    def integrate(
        self,
        adata_rna: AnnData,
        adata_mod2: AnnData,
        device: str = "cuda:0",
        seed: int = 2026,
        **kwargs,
    ) -> np.ndarray:
        """Run integration and return joint embedding.

        Parameters
        ----------
        adata_rna : AnnData
            RNA modality data.
        adata_mod2 : AnnData
            Secondary modality (ADT or ATAC).
        device : str
            Compute device.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Joint embedding of shape (n_cells, n_dims).
        """
        ...

    def check_deps(self) -> bool:
        """Check if method dependencies are installed. Override if needed."""
        return True

    def install_hint(self) -> str:
        """Return pip install command for missing dependencies."""
        if self.extras:
            extras = ",".join(self.extras)
            return f"pip install smobench[{extras}]"
        return "pip install smobench"

    @staticmethod
    def resolve_device(device: str) -> "torch.device":
        """Resolve device string to torch.device with strict GPU validation.

        If the caller requests a CUDA device but CUDA is not available,
        raises RuntimeError instead of silently falling back to CPU.
        """
        import torch

        if "cuda" in device:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"Device '{device}' requested but CUDA is not available. "
                    f"Please run on a GPU node (e.g. srun --gres=gpu:1) "
                    f"or pass device='cpu' explicitly."
                )
        return torch.device(device)

    def __repr__(self):
        return (
            f"{self.name}(tasks={self.tasks}, "
            f"modalities={self.modalities}, gpu={self.requires_gpu})"
        )
