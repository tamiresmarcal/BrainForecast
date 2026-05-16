"""
predictors/moving_average.py

Moving-average baseline: ŷ(t+H) = mean(y[t-k+1 .. t]).

Slightly stronger than persistence — averages over a recent window so it
absorbs short-term noise. For classification, the prediction is the
*mode* of the recent state sequence rather than the mean.

Like Persistence, this reads from `target_history` columns produced by
the TabularAdapter. It uses the first `k` history lags (hist1..histk).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MovingAveragePredictor:
    k: int = 5
    adapter_type: str = "tabular"
    task_type: str = "regression"
    target_history_prefix: str = "__hist"

    def fit(self, X, y, meta) -> "MovingAveragePredictor":
        # Discover the first k history columns per target.
        # Column naming from TabularAdapter is "<target>__hist<k>".
        hist_cols = [c for c in X.columns if self.target_history_prefix in c]
        if not hist_cols:
            raise ValueError(
                "MovingAveragePredictor needs target_history columns. "
                "Add 'target_history' to TabularAdapter ops."
            )

        # Group by target stem
        by_target: dict[str, list[str]] = {}
        for c in hist_cols:
            stem, _ = c.split(self.target_history_prefix)
            by_target.setdefault(stem, []).append(c)
        # Sort each target's columns by lag number, take first k
        for stem in by_target:
            by_target[stem] = sorted(
                by_target[stem], key=lambda c: int(c.rsplit(self.target_history_prefix, 1)[-1])
            )[: self.k]
        self._cols_by_target_ = by_target
        return self

    def predict(self, X) -> np.ndarray:
        targets = list(self._cols_by_target_.keys())
        n = len(X)
        out = np.empty((n, len(targets)), dtype=float)

        if self.task_type == "classification":
            for j, stem in enumerate(targets):
                arr = X[self._cols_by_target_[stem]].to_numpy()
                # Mode across the k columns per row (majority vote in recent window)
                out[:, j] = pd.DataFrame(arr).mode(axis=1)[0].to_numpy()
            return out.astype(int).ravel() if out.shape[1] == 1 else out.astype(int)

        # Regression: mean across the k columns per row
        for j, stem in enumerate(targets):
            out[:, j] = X[self._cols_by_target_[stem]].mean(axis=1).to_numpy()
        return out.ravel() if out.shape[1] == 1 else out
