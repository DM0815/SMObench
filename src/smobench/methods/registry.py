"""Method registry with decorator and entry_points support."""

from __future__ import annotations

import functools
from typing import Callable, Optional

import numpy as np
import pandas as pd

from smobench.methods.base import BaseMethod


class _FunctionMethod(BaseMethod):
    """Wraps a plain function as a BaseMethod."""

    def __init__(self, func, name, tasks, modalities, requires_gpu):
        self.name = name
        self.tasks = tasks
        self.modalities = modalities
        self.requires_gpu = requires_gpu
        self._func = func

    def integrate(self, adata_rna, adata_mod2, **kwargs):
        return self._func(adata_rna, adata_mod2, **kwargs)


class MethodRegistry:
    """Global registry of integration methods."""

    _methods: dict[str, BaseMethod] = {}

    @classmethod
    def register(
        cls,
        name: str,
        method: BaseMethod | None = None,
        func: Callable | None = None,
        tasks: list[str] | None = None,
        modalities: list[str] | None = None,
        requires_gpu: bool = True,
    ):
        if method is not None:
            cls._methods[name] = method
        elif func is not None:
            cls._methods[name] = _FunctionMethod(
                func=func,
                name=name,
                tasks=tasks or ["vertical"],
                modalities=modalities or ["RNA+ADT", "RNA+ATAC"],
                requires_gpu=requires_gpu,
            )
        else:
            raise ValueError("Provide either method or func")

    @classmethod
    def get(cls, name: str) -> BaseMethod:
        if name not in cls._methods:
            cls._load_builtins()
            cls._load_entry_points()
        if name not in cls._methods:
            available = ", ".join(sorted(cls._methods.keys()))
            raise KeyError(
                f"Method '{name}' not found. Available: {available}"
            )
        return cls._methods[name]

    @classmethod
    def list_all(cls) -> pd.DataFrame:
        cls._load_entry_points()
        cls._load_builtins()
        rows = []
        for name, m in sorted(cls._methods.items()):
            rows.append({
                "Method": name,
                "Tasks": ", ".join(m.tasks),
                "Modalities": ", ".join(m.modalities),
                "GPU": m.requires_gpu,
            })
        return pd.DataFrame(rows)

    @classmethod
    def _load_entry_points(cls):
        """Load methods registered via entry_points in other packages."""
        try:
            from importlib.metadata import entry_points
            eps = entry_points()
            if hasattr(eps, "select"):
                method_eps = eps.select(group="smobench.methods")
            else:
                method_eps = eps.get("smobench.methods", [])
            for ep in method_eps:
                if ep.name not in cls._methods:
                    try:
                        method_cls = ep.load()
                        cls._methods[ep.name] = method_cls()
                    except Exception:
                        pass
        except Exception:
            pass

    @classmethod
    def _load_builtins(cls):
        """Lazy-load built-in method wrappers."""
        from smobench._constants import BUILTIN_METHODS
        builtins = BUILTIN_METHODS
        for name in builtins:
            # Module-level registration happens on import
            try:
                __import__(f"smobench.methods.{name}")
            except ImportError:
                pass


def register_method(
    name: str,
    tasks: list[str] | None = None,
    modalities: list[str] | None = None,
    requires_gpu: bool = True,
):
    """Decorator to register a function as an integration method.

    Usage:
        @register_method("MyMethod", tasks=["vertical", "horizontal"])
        def my_method(adata_rna, adata_mod2, **kwargs):
            embedding = ...
            return embedding  # np.ndarray
    """
    def decorator(func):
        MethodRegistry.register(
            name=name,
            func=func,
            tasks=tasks,
            modalities=modalities,
            requires_gpu=requires_gpu,
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def get_method(name: str) -> BaseMethod:
    return MethodRegistry.get(name)


def list_methods() -> pd.DataFrame:
    return MethodRegistry.list_all()
