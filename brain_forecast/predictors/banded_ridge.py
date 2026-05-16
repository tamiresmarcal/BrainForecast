"""
predictors/banded_ridge.py

Banded ridge regression (Dupré la Tour et al., NeuroImage 2022).

Linear encoding model with a separate regularisation hyperparameter per
feature "band" (group). Standard tool for naturalistic-fMRI encoding —
solves the spurious-correlation problem when feature spaces are
correlated (e.g. semantic features bleeding into low-level visual ROIs).

Feature bands here are simple by default:
  - 'movie'  : columns starting with the movie prefix
  - 'brain'  : everything else from the original feature_cols
  - 'history': any column containing '__hist' (target autoregressive lags)

You can pass a custom band assignment via `bands` if your naming differs.

Multi-output regression is handled natively by himalaya — one solve, all
targets at once, with a separate alpha per band shared across targets.

The HRF convolution should already be applied via TabularAdapter(ops=['hrf', ...]).
Banded ridge itself does not modify features.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _origin_column(derived: str) -> str:
    """
    Trace a TabularAdapter-derived column back to its origin feature name.

    TabularAdapter appends suffixes:
      <col>__lag{k}, <col>__rmean{w}, <col>__rstd{w}, <col>__hrf,
      <tgt>__hist{k}
    The origin is everything before the first '__'.
    """
    return derived.split("__", 1)[0]


def _role_band_assigner(col: str, role_of: dict[str, str],
                        static_cols: Optional[frozenset] = None) -> str:
    """
    Assign a band from the feature's typed role.

    Bands:
      'history' — autoregressive target-history columns (<tgt>__hist{k})
      'static'  — time-invariant covariates (incl. one-hot encoded, e.g. sex_F)
      'stimulus'— known_dynamic (the movie)
      'brain'   — observed_dynamic (brain history) / fallback
    """
    if "__hist" in col:
        return "history"
    origin = _origin_column(col)
    # Direct role hit on the origin column.
    role = role_of.get(origin)
    if role == "static":
        return "static"
    if role == "known_dynamic":
        return "stimulus"
    if role == "observed_dynamic":
        return "brain"
    # One-hot-encoded static columns look like '<staticcol>_<level>' and have
    # no '__' suffix, so the origin == col. Match by static-name prefix.
    if static_cols:
        for s in static_cols:
            if col == s or col.startswith(s + "_"):
                return "static"
    return "brain"  # fallback for anything untyped


@dataclass
class BandedRidgePredictor:
    """
    Banded ridge wrapper around himalaya.

    Parameters
    ----------
    n_iter : int
        Number of hyperparameter random search iterations for himalaya.
    n_targets_batch : int
        Number of targets fit in parallel (memory-vs-speed tradeoff).
    band_fn : callable str -> str, optional
        Override for column → band assignment. By default bands are derived
        from the typed feature roles passed via the fit context:
        'static', 'stimulus' (known_dynamic), 'brain' (observed_dynamic),
        and 'history' (autoregressive target lags). This is exactly the
        feature-family separation banded ridge is designed for and matches
        the feature-exclusion benchmark families.
    backend : str
        himalaya backend: 'numpy', 'torch' (CPU), or 'torch_cuda' (GPU).
    """

    n_iter: int = 20
    n_targets_batch: int = 200
    band_fn: Optional[Callable[[str], str]] = None
    backend: str = "torch_cuda"

    # Required protocol attributes
    adapter_type: str = "tabular"
    task_type: str = "regression"

    def fit(self, X, y, meta) -> "BandedRidgePredictor":
        from himalaya.backend import set_backend
        from himalaya.ridge import GroupRidgeCV
        from himalaya.kernel_ridge import ColumnKernelizer, Kernelizer

        # Try requested backend, fall back if unavailable
        try:
            set_backend(self.backend)
        except (ValueError, ImportError) as e:
            log.warning(f"himalaya backend '{self.backend}' unavailable ({e}); falling back to numpy")
            set_backend("numpy")

        # Build origin-column -> role map from the fit context (passed by the
        # evaluation harness as `meta = {'roles': FeatureRoles}`). This lets us
        # band by the TFT-style typed roles: static / stimulus / brain / history.
        role_of: dict[str, str] = {}
        static_names: set[str] = set()
        if isinstance(meta, dict) and "roles" in meta and meta["roles"] is not None:
            roles = meta["roles"]
            for c in roles.static:
                role_of[c] = "static"
                static_names.add(c)
            for c in roles.known_dynamic:
                role_of[c] = "known_dynamic"
            for c in roles.observed_dynamic:
                role_of[c] = "observed_dynamic"

        static_fs = frozenset(static_names)
        assigner = self.band_fn or (
            lambda c: _role_band_assigner(c, role_of, static_fs)
        )
        col_to_band = {c: assigner(c) for c in X.columns}
        bands = sorted(set(col_to_band.values()))
        self._bands_ = bands

        # Build column slices per band (himalaya expects integer indices)
        band_slices = []
        for b in bands:
            idx = [i for i, c in enumerate(X.columns) if col_to_band[c] == b]
            band_slices.append((b, Kernelizer(), idx))
        log.info(
            f"BandedRidge bands: " + ", ".join(f"{b}={len([1 for c in col_to_band.values() if c==b])}" for b in bands)
        )

        # Standardise inputs (per band conceptually; here we standardise everything)
        X_arr = X.to_numpy().astype(np.float32)
        self._X_mean_ = X_arr.mean(axis=0)
        self._X_std_ = X_arr.std(axis=0) + 1e-6
        X_arr = (X_arr - self._X_mean_) / self._X_std_

        y_arr = y.to_numpy().astype(np.float32) if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y, dtype=np.float32)
        if y_arr.ndim == 1:
            y_arr = y_arr[:, None]
        self._y_mean_ = y_arr.mean(axis=0)
        y_arr = y_arr - self._y_mean_

        # himalaya kernel ridge with column kernelizer = banded ridge
        kernelizer = ColumnKernelizer(band_slices)
        self._model_ = GroupRidgeCV(
            groups="input",
            random_state=0,
            cv=5,
            solver_params={"n_iter": self.n_iter, "n_targets_batch": self.n_targets_batch},
        )
        # Pipeline: column kernelizer then group ridge
        from sklearn.pipeline import make_pipeline
        self._pipeline_ = make_pipeline(kernelizer, self._model_)
        self._pipeline_.fit(X_arr, y_arr)

        self._n_targets_ = y_arr.shape[1]
        return self

    def predict(self, X) -> np.ndarray:
        X_arr = X.to_numpy().astype(np.float32)
        X_arr = (X_arr - self._X_mean_) / self._X_std_
        y_pred = np.asarray(self._pipeline_.predict(X_arr)) + self._y_mean_
        if self._n_targets_ == 1:
            return y_pred.ravel()
        return y_pred
