"""
predictors/__init__.py

Predictor factory and protocol.

All predictors share one interface:

    class MyPredictor:
        adapter_type: str        # 'tabular' or 'sequence'
        task_type: str           # 'regression' or 'classification'

        def fit(self, X, y, meta) -> 'MyPredictor': ...
        def predict(self, X) -> np.ndarray: ...

Construction goes through `make_predictor(name, task_type, **kwargs)` so
external code (configs, CLI) refers to predictors by string name.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Predictor(Protocol):
    """Common interface for all predictors."""

    adapter_type: str   # 'tabular' or 'sequence'
    task_type: str      # 'regression' or 'classification'

    def fit(self, X, y, meta) -> "Predictor": ...
    def predict(self, X) -> np.ndarray: ...


# Lazy imports so optional deps (torch, himalaya) don't break the package
# if a user only wants the simple benchmarks.

def make_predictor(name: str, task_type: str = "regression", **kwargs) -> Predictor:
    """
    Construct a predictor by name.

    Parameters
    ----------
    name : str
        One of: 'persistence', 'moving_average', 'ar', 'banded_ridge', 'tft'
    task_type : str
        'regression' or 'classification'. Not all predictors support
        classification (Persistence and MA do; AR, BandedRidge do regression;
        TFT supports both).
    **kwargs
        Forwarded to the predictor constructor.
    """
    name = name.lower()

    if name == "persistence":
        from .persistence import PersistencePredictor
        return PersistencePredictor(task_type=task_type, **kwargs)

    if name in {"moving_average", "ma"}:
        from .moving_average import MovingAveragePredictor
        return MovingAveragePredictor(task_type=task_type, **kwargs)

    if name == "ar":
        from .ar import ARPredictor
        if task_type != "regression":
            raise ValueError("AR predictor supports regression only.")
        return ARPredictor(**kwargs)

    if name in {"banded_ridge", "ridge"}:
        from .banded_ridge import BandedRidgePredictor
        if task_type != "regression":
            raise ValueError("Banded ridge supports regression only.")
        return BandedRidgePredictor(**kwargs)

    if name == "tft":
        from .tft import TFTPredictor
        return TFTPredictor(task_type=task_type, **kwargs)

    raise ValueError(f"Unknown predictor: {name}")


__all__ = ["Predictor", "make_predictor"]
