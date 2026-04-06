"""
Method registry for spatial multi-omics integration methods.

Usage:
    # List available methods
    smobench.list_methods()

    # Register a custom method
    @smobench.register_method("MyMethod", tasks=["vertical"])
    def my_method(adata_rna, adata_mod2, **kwargs):
        return embedding  # np.ndarray

    # Or use class-based registration
    class MyMethod(smobench.methods.BaseMethod):
        name = "MyMethod"
        tasks = ["vertical", "horizontal"]

        def integrate(self, adata_rna, adata_mod2, **kwargs):
            return embedding
"""

from smobench.methods.base import BaseMethod
from smobench.methods.registry import (
    MethodRegistry,
    register_method,
    get_method,
    list_methods,
)

__all__ = [
    "BaseMethod",
    "MethodRegistry",
    "register_method",
    "get_method",
    "list_methods",
]
