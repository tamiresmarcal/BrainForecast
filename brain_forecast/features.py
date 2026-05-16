"""
features.py

Adapters that turn a FeatureBundle into model-ready inputs.

Two adapter families exist, matching the two predictor families:

  TabularAdapter
      Materialises lag / rolling / HRF features as extra columns. Output:
      a flat (N_samples, N_features) matrix. Used by Persistence, Moving
      Average, AR, and Banded Ridge.

  SequenceAdapter
      Stacks a window of past frames into a 3D tensor (N_samples, window,
      N_features). Used by the Temporal Fusion Transformer.

Both adapters produce a `meta` dataframe alongside X, y so the CV layer
knows which sample belongs to which (subject, cohort, time). This is
essential for valid subject-out splits.

Temporal alignment policy:
  - Sequence models (TFT): NO HRF, NO explicit lag stack. The model
    learns alignment implicitly from the window of past frames.
  - Tabular models: HRF convolution applied to MOVIE features only for
    Banded Ridge by default. Brain features are not convolved (they are
    already BOLD). Simpler benchmarks (Persistence, MA, AR) operate on
    target history only and ignore movie features entirely.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import gamma

from .data import META_COHORT, META_SUB, META_TIME, FeatureBundle

log = logging.getLogger(__name__)


# ─── HRF utilities (Glover / SPM canonical double-gamma) ─────────────────

def canonical_hrf(tr: float, duration: float = 32.0) -> np.ndarray:
    """
    Glover (1999) / SPM canonical double-gamma HRF.

    Peak ~5–6 s after stimulus, undershoot ~12–16 s. Standard fMRI choice
    for convolving stimulus regressors before linear modelling.
    """
    t = np.arange(0, duration, tr)
    hrf = (
        t ** 5 * np.exp(-t) / gamma(6)
        - (1.0 / 6.0) * t ** 15 * np.exp(-t) / gamma(16)
    )
    hrf = hrf / hrf.sum()
    return hrf


def hrf_convolve_series(x: np.ndarray, tr: float) -> np.ndarray:
    """Convolve a 1D timeseries with the canonical HRF and trim to original length."""
    hrf = canonical_hrf(tr=tr)
    return np.convolve(x, hrf, mode="full")[: len(x)]


# ─── Tabular adapter ─────────────────────────────────────────────────────

@dataclass
class TabularAdapter:
    """
    Materialise temporal features as columns. Output is a flat design matrix.

    Operations (applied per subject so they never leak across people):
      - lag:        shift(1), shift(2), ..., shift(k_lag)        on chosen cols
      - rolling:    rolling(window).{mean, std}                   on chosen cols
      - hrf:        convolve with canonical HRF                   on movie cols only
      - target_history:  past target values as features (for AR-style models)

    Parameters
    ----------
    ops : list[str]
        Which operations to apply. Subset of:
        {'lag', 'rolling', 'hrf', 'target_history'}.
    k_lag : int
        Number of lag shifts to create when 'lag' is in ops.
    rolling_window : int
        Window size in samples for the rolling mean/std.
    hrf_targets : str
        Which feature group to convolve with HRF. Default 'movie' means
        only columns starting with the movie prefix.
    movie_prefix : str
        Prefix that identifies movie feature columns. Used by 'hrf' to
        decide what to convolve. If your columns don't follow a prefix
        convention, pass an explicit list via `movie_cols`.
    movie_cols : list[str], optional
        Explicit movie column list. Overrides movie_prefix.
    """

    ops: list[str] = field(default_factory=lambda: ["lag", "rolling"])
    k_lag: int = 3
    rolling_window: int = 10
    movie_prefix: str = "mov_"
    movie_cols: Optional[list[str]] = None
    target_history_lags: int = 5  # only used if 'target_history' in ops

    def transform(
        self, bundle: FeatureBundle, horizon_min: float
    ) -> tuple[pd.DataFrame, pd.Series | pd.DataFrame, pd.DataFrame]:
        """
        Returns
        -------
        X    : (N, F) DataFrame of features
        y    : Series (single target) or DataFrame (multi-target) of y(t+H)
        meta : DataFrame with columns sub, cohort, start — for CV splitting
        """
        df = bundle.df.copy()
        roles = bundle.roles
        future_cols = bundle.future_target_cols(horizon_min)

        # Dynamic features get temporal ops; static features do not (they are
        # constant within a subject, so lag/rolling/HRF are meaningless and
        # would also break on categorical strings). Static columns are kept
        # as plain features (categoricals one-hot encoded).
        static_cols = list(roles.static)
        dynamic_cols = list(roles.known_dynamic) + list(roles.observed_dynamic)
        if not roles.all_features():
            # Fully untyped fallback (legacy flat usage): treat all as dynamic
            dynamic_cols = list(bundle.feature_cols)
            static_cols = []

        # Stimulus columns to HRF-convolve (typed role preferred).
        if self.movie_cols is not None:
            movie_set = set(self.movie_cols)
        elif roles.known_dynamic:
            movie_set = set(roles.known_dynamic)
        else:
            movie_set = {c for c in dynamic_cols if c.startswith(self.movie_prefix)}

        # One-hot encode categorical static columns; pass numeric statics through.
        static_feature_cols: list[str] = []
        for c in static_cols:
            if c in set(roles.static_categorical):
                dummies = pd.get_dummies(df[c], prefix=c, dtype=float)
                df = pd.concat([df, dummies], axis=1)
                static_feature_cols.extend(dummies.columns.tolist())
            else:
                static_feature_cols.append(c)

        new_cols: list[str] = []

        # 1) HRF convolution on stimulus cols (per subject, using their TR)
        if "hrf" in self.ops and movie_set:
            for sub_id, sub_df in df.groupby(META_SUB):
                tr = float(sub_df[META_TIME].diff().median())
                if not np.isfinite(tr) or tr <= 0:
                    continue
                for col in movie_set:
                    new_col = f"{col}__hrf"
                    df.loc[sub_df.index, new_col] = hrf_convolve_series(
                        sub_df[col].to_numpy(dtype=float), tr=tr
                    )
                    if new_col not in new_cols:
                        new_cols.append(new_col)

        # 2) Lag features (dynamic cols only, per subject)
        if "lag" in self.ops and dynamic_cols:
            for k in range(1, self.k_lag + 1):
                lagged = (
                    df.groupby(META_SUB, group_keys=False)[dynamic_cols].shift(k)
                )
                lagged.columns = [f"{c}__lag{k}" for c in dynamic_cols]
                df = pd.concat([df, lagged], axis=1)
                new_cols.extend(lagged.columns)

        # 3) Rolling stats (dynamic cols only, per subject)
        if "rolling" in self.ops and dynamic_cols:
            roll = df.groupby(META_SUB, group_keys=False)[dynamic_cols].rolling(
                window=self.rolling_window, min_periods=self.rolling_window
            )
            roll_mean = roll.mean().reset_index(level=0, drop=True)
            roll_std = roll.std().reset_index(level=0, drop=True)
            roll_mean.columns = [f"{c}__rmean{self.rolling_window}" for c in dynamic_cols]
            roll_std.columns = [f"{c}__rstd{self.rolling_window}" for c in dynamic_cols]
            df = pd.concat([df, roll_mean, roll_std], axis=1)
            new_cols.extend(roll_mean.columns)
            new_cols.extend(roll_std.columns)

        # 4) Target history (used by AR-style benchmarks). One lag per target.
        if "target_history" in self.ops:
            for tgt in bundle.target_cols:
                for k in range(1, self.target_history_lags + 1):
                    new_col = f"{tgt}__hist{k}"
                    df[new_col] = df.groupby(META_SUB, group_keys=False)[tgt].shift(k)
                    new_cols.append(new_col)

        # Assemble outputs. Feature matrix = static (encoded) + raw dynamic
        # + all derived (hrf/lag/rolling) + target-history columns.
        X_cols = static_feature_cols + dynamic_cols + new_cols
        # De-duplicate while preserving order (a column could be both raw
        # dynamic and, in legacy mode, otherwise listed).
        seen = set()
        X_cols = [c for c in X_cols if not (c in seen or seen.add(c))]
        keep_cols = X_cols + future_cols + [META_SUB, META_COHORT, META_TIME]
        out = df[keep_cols].dropna()

        X = out[X_cols].reset_index(drop=True)
        if len(future_cols) == 1:
            y = out[future_cols[0]].reset_index(drop=True)
        else:
            y = out[future_cols].reset_index(drop=True)
        meta = out[[META_SUB, META_COHORT, META_TIME]].reset_index(drop=True)
        return X, y, meta


# ─── Sequence adapter ────────────────────────────────────────────────────

@dataclass
class SequenceAdapter:
    """
    Pack a per-subject timeline for a sequence model, preserving feature roles.

    Unlike the tabular adapter, no lag/rolling columns are materialised — the
    Temporal Fusion Transformer learns temporal structure itself. The job here
    is to (a) build the integer time index pytorch_forecasting needs, (b) keep
    the static / known_dynamic / observed_dynamic typing intact so the TFT can
    wire each into the correct slot, and (c) trim each subject's tail so that
    BOTH the future target y(t+H) AND the future stimulus x up to t+H exist.

    Why the extra trim for the stimulus: the stimulus is a *known* input, so
    the TFT is allowed to look at it at and beyond the forecast time. That is
    only valid where those future stimulus frames actually exist in the data.
    Near the end of a film the future stimulus runs out, so those sample
    origins must be dropped (the same reasoning that drops rows where the
    future target is missing).

    Output dict consumed by TFTPredictor:
        {
            'data':                full per-subject timeline DataFrame,
            'time_idx_col':        'time_idx',
            'group_ids':           ['sub'],
            'static_categoricals': [...],   # s (categorical)
            'static_reals':        [...],   # s (continuous)
            'known_reals':         [...],   # x  (stimulus, past+future)
            'observed_reals':      [...],   # z  (brain history, past only)
            'target_cols':         [...],   # y(t+H), one per target
            'window_samples':      int,
            'horizon_samples':     int,
            'horizon_min':         float,
            'task_type':           str,
        }

    Parameters
    ----------
    window_min : float
        Past context window in minutes. Converted to samples from the
        median TR.
    """

    window_min: float = 5.0

    def transform(self, bundle: FeatureBundle, horizon_min: float) -> dict:
        df = bundle.df.copy()
        roles = bundle.roles

        # Per-subject integer time index (required by pytorch_forecasting)
        df["time_idx"] = df.groupby(META_SUB).cumcount().astype(int)

        future_cols = bundle.future_target_cols(horizon_min)

        # Window / horizon length in samples from the median TR
        tr = float(df.groupby(META_SUB)[META_TIME].diff().median())
        window_samples = max(1, int(round(self.window_min * 60 / tr)))
        horizon_samples = max(0, int(round(horizon_min * 60 / tr)))

        # 1) Drop rows where the future target is missing (end of each run)
        df = df.dropna(subset=future_cols)

        # 2) Drop rows where the future STIMULUS window (t .. t+H) would run
        #    past the end of the subject's recording. The stimulus is a known
        #    input, so the TFT may attend to x at and beyond t; those frames
        #    must actually exist. We enforce this by requiring at least
        #    horizon_samples rows to remain after each kept origin within the
        #    same subject.
        if roles.known_dynamic and horizon_samples > 0:
            def _trim_known_tail(g):
                # rows with enough future room inside this subject
                return g.iloc[: max(0, len(g) - horizon_samples)]
            df = df.groupby(META_SUB, group_keys=False).apply(_trim_known_tail)

        df = df.reset_index(drop=True)

        return {
            "data": df,
            "time_idx_col": "time_idx",
            "group_ids": [META_SUB],
            "static_categoricals": list(roles.static_categorical),
            "static_reals": list(roles.static_real()),
            "known_reals": list(roles.known_dynamic),
            "observed_reals": list(roles.observed_dynamic),
            "target_cols": future_cols,
            "window_samples": window_samples,
            "horizon_samples": horizon_samples,
            "horizon_min": horizon_min,
            "task_type": bundle.task_type,
        }
