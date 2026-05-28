"""
features.py

Adapters that turn a FeatureBundle into model-ready inputs.

Two adapter families exist, matching the two predictor families:

  TabularAdapter
      Materialises lag / rolling / HRF features as extra columns. Output:
      a flat (N_samples, N_features) matrix. Used by Persistence, Moving
      Average, AR, and Banded Ridge.

  SequenceAdapter
      Stacks a per-subject timeline for a sequence model. Used by the
      Temporal Fusion Transformer.

Both adapters produce a `meta` dataframe alongside X, y so the CV layer
knows which sample belongs to which (subject, cohort, time).
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
    """Glover (1999) / SPM canonical double-gamma HRF."""
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

    See module docstring; this is unchanged from the prior version.
    """

    ops: list[str] = field(default_factory=lambda: ["lag", "rolling"])
    k_lag: int = 3
    rolling_window: int = 10
    movie_prefix: str = "mov_"
    movie_cols: Optional[list[str]] = None
    target_history_lags: int = 5

    def transform(
        self, bundle: FeatureBundle, horizon_min: float
    ) -> tuple[pd.DataFrame, pd.Series | pd.DataFrame, pd.DataFrame]:
        df = bundle.df.copy()
        roles = bundle.roles
        future_cols = bundle.future_target_cols(horizon_min)

        static_cols = list(roles.static)
        dynamic_cols = list(roles.known_dynamic) + list(roles.observed_dynamic)
        if not roles.all_features():
            dynamic_cols = list(bundle.feature_cols)
            static_cols = []

        if self.movie_cols is not None:
            movie_set = set(self.movie_cols)
        elif roles.known_dynamic:
            movie_set = set(roles.known_dynamic)
        else:
            movie_set = {c for c in dynamic_cols if c.startswith(self.movie_prefix)}

        static_feature_cols: list[str] = []
        for c in static_cols:
            if c in set(roles.static_categorical):
                dummies = pd.get_dummies(df[c], prefix=c, dtype=float)
                df = pd.concat([df, dummies], axis=1)
                static_feature_cols.extend(dummies.columns.tolist())
            else:
                static_feature_cols.append(c)

        new_cols: list[str] = []

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

        if "lag" in self.ops and dynamic_cols:
            for k in range(1, self.k_lag + 1):
                lagged = (
                    df.groupby(META_SUB, group_keys=False)[dynamic_cols].shift(k)
                )
                lagged.columns = [f"{c}__lag{k}" for c in dynamic_cols]
                df = pd.concat([df, lagged], axis=1)
                new_cols.extend(lagged.columns)

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

        if "target_history" in self.ops:
            for tgt in bundle.target_cols:
                for k in range(1, self.target_history_lags + 1):
                    new_col = f"{tgt}__hist{k}"
                    df[new_col] = df.groupby(META_SUB, group_keys=False)[tgt].shift(k)
                    new_cols.append(new_col)

        X_cols = static_feature_cols + dynamic_cols + new_cols
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

    No lag/rolling columns are materialised — the TFT learns temporal
    structure itself. The job here is to (a) build the integer time index
    pytorch_forecasting needs, (b) preserve the static / known_dynamic /
    observed_dynamic typing, and (c) trim each subject's tail so future
    target *and* future stimulus exist for every kept origin.

    Parameters
    ----------
    window_min : float
        Past context window in minutes. Converted to samples from the
        median TR (after any stride is applied).
    stride : int
        Temporal subsampling factor for the TFT input. ``stride=1`` (default)
        keeps every row. ``stride=3`` keeps every 3rd row per subject (drops
        ~67% of data) and re-indexes ``time_idx`` so the strided sequence
        is contiguous 0,1,2,... The TFT therefore sees a uniform time grid
        whose step is ``stride × original_TR`` of real time.

        Important consequence: ``window_min`` and ``horizon_min`` keep their
        wall-clock meaning. ``window_min=5.0`` still means 5 real minutes of
        past context — it's just represented in fewer samples after stride.
        Same for the horizon. This is what you want for science consistency
        between strided dev runs and full-data production runs.

        Use this as a speed knob during development:
            stride=1   full data    (publish on this)
            stride=2   ~50% data    (dev iteration)
            stride=3   ~67% drop    (fast smoke tests; 70% drop target)
    """

    window_min: float = 5.0
    stride: int = 1

    def transform(self, bundle: FeatureBundle, horizon_min: float) -> dict:
        df = bundle.df.copy()
        roles = bundle.roles

        # Step 1: original TR (before any stride). Used to convert
        # window_min / horizon_min into sample counts in the strided frame.
        original_tr = float(df.groupby(META_SUB)[META_TIME].diff().median())
        if not np.isfinite(original_tr) or original_tr <= 0:
            raise ValueError(
                f"Could not infer a valid TR from the bundle (got {original_tr})."
            )

        # Step 2: apply stride per subject. Keep every Nth row and re-index
        # time_idx so the strided sequence is contiguous (0,1,2,...). The
        # TFT then sees a uniform grid; the only difference vs full data is
        # that each step represents stride × original_TR of real time.
        if self.stride > 1:
            n_before = len(df)
            df = (
                df.groupby(META_SUB, group_keys=False)
                .apply(lambda g: g.iloc[:: self.stride])
                .reset_index(drop=True)
            )
            log.info(
                f"SequenceAdapter stride={self.stride}: "
                f"{n_before:,} → {len(df):,} rows ({100*len(df)/n_before:.1f}% kept)"
            )

        # Step 3: per-subject integer time index in the (possibly strided) frame.
        df["time_idx"] = df.groupby(META_SUB).cumcount().astype(int)

        # Step 4: convert window / horizon to samples using the EFFECTIVE TR
        # (post-stride). This preserves the wall-clock meaning of the args:
        # "5 minutes of context" is still 5 real minutes, just in fewer steps.
        effective_tr = original_tr * self.stride
        window_samples = max(1, int(round(self.window_min * 60 / effective_tr)))
        horizon_samples = max(0, int(round(horizon_min * 60 / effective_tr)))
        log.info(
            f"SequenceAdapter: original_TR={original_tr:.2f}s "
            f"effective_TR={effective_tr:.2f}s | "
            f"window={self.window_min}min={window_samples} samples | "
            f"horizon={horizon_min}min={horizon_samples} samples"
        )

        future_cols = bundle.future_target_cols(horizon_min)

        # Step 5: drop rows where the future target column is missing (the
        # natural tail of each subject's recording). NOTE: future targets
        # were built by bundle.make_targets() BEFORE stride, using the
        # original TR. After stride, only rows whose target is non-NaN
        # survive — which automatically respects the horizon at the
        # original temporal resolution. Sanity check it.
        df = df.dropna(subset=future_cols)

        # Step 6: tail-trim by horizon_samples in the strided frame so the
        # known stimulus has enough future room. Necessary even though
        # step 5 already filtered by target, because the stimulus needs to
        # extend further than the target.
        if roles.known_dynamic and horizon_samples > 0:
            def _trim_known_tail(g):
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
            "stride": self.stride,
            "effective_tr_sec": effective_tr,
        }
