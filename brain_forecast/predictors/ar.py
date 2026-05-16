"""
predictors/ar.py

Autoregressive baseline: ŷ(t+H) = β₀ + Σ βᵢ y(t-i).

Linear regression where the only features are past values of the target
itself. No stimulus, no other brain features. This isolates "how much of
future brain activity is predictable from its own immediate past."

For multi-target regression, one AR is fit per target column. For
classification, AR is not applicable (use Persistence or MA instead).

Like Persistence and MA, this consumes `__hist` columns from the
TabularAdapter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


@dataclass
class ARPredictor:
    p: int = 5  # AR order: how many past lags to use as predictors
    adapter_type: str = "tabular"
    task_type: str = "regression"
    target_history_prefix: str = "__hist"
    ridge_alpha: float = 1.0  # mild regularisation for stability

    def fit(self, X, y, meta) -> "ARPredictor":
        # Discover history columns and group by target
        hist_cols = [c for c in X.columns if self.target_history_prefix in c]
        if not hist_cols:
            raise ValueError(
                "ARPredictor needs target_history columns. Add 'target_history' "
                "to TabularAdapter ops, with target_history_lags >= p."
            )

        by_target: dict[str, list[str]] = {}
        for c in hist_cols:
            stem, _ = c.split(self.target_history_prefix)
            by_target.setdefault(stem, []).append(c)
        for stem in by_target:
            by_target[stem] = sorted(
                by_target[stem], key=lambda c: int(c.rsplit(self.target_history_prefix, 1)[-1])
            )[: self.p]
        self._cols_by_target_ = by_target

        # Fit one ridge regression per target on its own history
        y_arr = y.to_numpy() if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y)
        if y_arr.ndim == 1:
            y_arr = y_arr[:, None]

        targets = list(by_target.keys())
        self._models_ = {}
        for j, stem in enumerate(targets):
            cols = by_target[stem]
            Xt = X[cols].to_numpy()
            model = Ridge(alpha=self.ridge_alpha)
            model.fit(Xt, y_arr[:, j])
            self._models_[stem] = model
        return self

    def predict(self, X) -> np.ndarray:
        targets = list(self._cols_by_target_.keys())
        out = np.empty((len(X), len(targets)))
        for j, stem in enumerate(targets):
            cols = self._cols_by_target_[stem]
            out[:, j] = self._models_[stem].predict(X[cols].to_numpy())
        return out.ravel() if out.shape[1] == 1 else out
