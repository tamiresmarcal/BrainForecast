"""
predictors/persistence.py

Persistence baseline: ŷ(t+H) = y(t).

The simplest possible benchmark. Any real model must beat this to claim it
has learned anything beyond "tomorrow looks like today."

Implementation detail: at inference time we don't actually have y(t) as a
feature column — that would be circular. Instead, we use the most recent
target value from the `target_history` columns produced by the tabular
adapter (i.e. the `__hist1` columns). This means Persistence depends on
the TabularAdapter being configured with 'target_history' in its ops.

For classification, the prediction is the integer state at time t. For
regression, it is the scalar/vector value at time t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PersistencePredictor:
    adapter_type: str = "tabular"
    task_type: str = "regression"
    target_history_suffix: str = "__hist1"

    def fit(self, X, y, meta) -> "PersistencePredictor":
        # Remember which columns to read at predict time. We pick the
        # __hist1 columns; one per target if multi-target.
        self._hist_cols_ = [c for c in X.columns if c.endswith(self.target_history_suffix)]
        if not self._hist_cols_:
            raise ValueError(
                "PersistencePredictor needs target_history columns. "
                "Add 'target_history' to TabularAdapter ops."
            )
        return self

    def predict(self, X) -> np.ndarray:
        out = X[self._hist_cols_].to_numpy()
        if self.task_type == "classification":
            # State ids are integers; cast back
            return out.astype(int).ravel() if out.shape[1] == 1 else out.astype(int)
        return out.ravel() if out.shape[1] == 1 else out
