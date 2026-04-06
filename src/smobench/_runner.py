"""Subprocess entry point for running a method in an isolated environment.

Usage:
    python -m smobench._runner \
        --method SpatialGlue \
        --rna /tmp/rna.h5ad \
        --mod2 /tmp/mod2.h5ad \
        --out /tmp/embedding.npy \
        --kwargs '{"device": "cuda:0", "seed": 42, "modality": "ADT"}'
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
import warnings

# Suppress noisy deprecation warnings from vendored/container packages
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="SMObench method runner (subprocess)")
    parser.add_argument("--method", required=True, help="Method name")
    parser.add_argument("--rna", required=True, help="Path to RNA h5ad")
    parser.add_argument("--mod2", required=True, help="Path to second modality h5ad")
    parser.add_argument("--out", required=True, help="Output path for embedding (.npy)")
    parser.add_argument("--kwargs", default="{}", help="JSON kwargs for integrate()")
    args = parser.parse_args()

    try:
        import scanpy as sc

        # Load data
        adata_rna = sc.read_h5ad(args.rna)
        adata_mod2 = sc.read_h5ad(args.mod2)

        # Get method
        from smobench.methods.registry import MethodRegistry
        method = MethodRegistry.get(args.method)

        # Parse kwargs
        kwargs = json.loads(args.kwargs)

        # Run integration
        result = method.integrate(adata_rna, adata_mod2, **kwargs)

        # Support (embedding, kept_indices) tuple for methods that filter cells
        kept_indices = None
        if isinstance(result, tuple):
            embedding, kept_indices = result
        else:
            embedding = result

        if not isinstance(embedding, np.ndarray):
            embedding = np.asarray(embedding)

        # Save result
        np.save(args.out, embedding)
        if kept_indices is not None:
            kept_path = args.out.replace(".npy", "_kept.npy")
            np.save(kept_path, np.asarray(kept_indices))

    except ImportError as e:
        msg = str(e)
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"IMPORT ERROR in {args.method}: {msg}", file=sys.stderr)
        if "torch_geometric" in msg or "torch_sparse" in msg:
            print("  → Install PyG: pip install torch-geometric", file=sys.stderr)
        elif "tensorflow" in msg:
            print("  → Install TF: pip install tensorflow", file=sys.stderr)
        elif "dgl" in msg:
            print("  → Install DGL: pip install dgl", file=sys.stderr)
        elif ".A" in msg or "toarray" in msg:
            print("  → scipy version conflict: use .toarray() instead of .A", file=sys.stderr)
        else:
            print(f"  → Missing dependency: {msg}", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
