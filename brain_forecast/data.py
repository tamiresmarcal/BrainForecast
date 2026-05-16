"""
data.py

Load tabular data and expose typed feature columns + targets.

The TFT paper (Lim et al. 2021, Eq. 1) defines three input categories that
must be wired into different parts of the architecture:

    static            (s)  — time-invariant per subject: age, sex, ...
    known_dynamic     (x)  — known across past AND future: the stimulus
                              (the whole movie is available in advance)
    observed_dynamic  (z)  — known in the past ONLY: brain history
                              (we cannot know future brain — that is the target)

This module carries that typing as a `FeatureRoles` mapping so each model
can route columns to the correct slot. The simpler benchmarks (Persistence,
MA, AR) ignore the typing entirely — they only use the target's own past —
so a flat `feature_cols` view is still provided for them.

Convention:
  - 'start' is the time column, in seconds
  - 'sub' is the subject ID column
  - 'cohort' identifies the movie / stimulus context
These three are reserved metadata columns and must not appear in any
feature-role list or target list.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Reserved metadata column names
META_SUB = "sub"
META_TIME = "start"          # seconds
META_COHORT = "cohort"       # movie identifier
META_COLS = {META_SUB, META_TIME, META_COHORT}


@dataclass
class FeatureRoles:
    """
    Typed feature schema mapping each column to its TFT input category.

    Attributes
    ----------
    static : list[str]
        Time-invariant covariates (one value per subject). Demographics,
        subject id if used as a covariate, etc. Maps to TFT static slots.
    known_dynamic : list[str]
        Time-varying inputs known across the whole timeline including the
        future — i.e. the stimulus. Maps to TFT time_varying_known_*.
    observed_dynamic : list[str]
        Time-varying inputs known only in the past — i.e. brain history.
        Maps to TFT time_varying_unknown_*.
    static_categorical : list[str]
        Subset of `static` that is categorical (e.g. sex). The remainder
        of `static` is treated as continuous (real).
    """

    static: list[str] = field(default_factory=list)
    known_dynamic: list[str] = field(default_factory=list)
    observed_dynamic: list[str] = field(default_factory=list)
    static_categorical: list[str] = field(default_factory=list)

    def all_features(self) -> list[str]:
        """Flat list of every feature column. Order: static, known, observed."""
        return list(self.static) + list(self.known_dynamic) + list(self.observed_dynamic)

    def static_real(self) -> list[str]:
        cat = set(self.static_categorical)
        return [c for c in self.static if c not in cat]

    def validate(self) -> None:
        cat_not_in_static = set(self.static_categorical) - set(self.static)
        if cat_not_in_static:
            raise ValueError(
                f"static_categorical columns must also be listed in static: "
                f"{sorted(cat_not_in_static)}"
            )
        seen: dict[str, str] = {}
        for role in ("static", "known_dynamic", "observed_dynamic"):
            for c in getattr(self, role):
                if c in seen:
                    raise ValueError(
                        f"Column '{c}' appears in both '{seen[c]}' and '{role}'."
                    )
                seen[c] = role


@dataclass
class FeatureBundle:
    """
    Standardised container for tabular brain + movie data with typed roles.

    Attributes
    ----------
    df : pd.DataFrame
        Per-(subject, time) rows. Must contain META_COLS plus feature/target cols.
    roles : FeatureRoles
        Typed feature schema (static / known_dynamic / observed_dynamic).
    target_cols : list[str]
        Output columns. One for classification (brain state); many for
        multi-output regression (DFC connections, ROIs).
    task_type : str
        'regression' or 'classification'.
    horizons_sec : list[int]
        Forecast horizons in seconds. Populated by `make_targets`.
    """

    df: pd.DataFrame
    roles: FeatureRoles
    target_cols: list[str]
    task_type: str = "regression"
    horizons_sec: list[int] = field(default_factory=list)

    # ── Construction ─────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        missing = META_COLS - set(self.df.columns)
        if missing:
            raise ValueError(
                f"FeatureBundle requires metadata columns {META_COLS}; missing: {missing}"
            )
        self.roles.validate()
        overlap = (set(self.roles.all_features()) | set(self.target_cols)) & META_COLS
        if overlap:
            raise ValueError(
                f"Feature/target columns must not overlap with metadata: {overlap}"
            )
        if self.task_type not in {"regression", "classification"}:
            raise ValueError(
                f"task_type must be 'regression' or 'classification', got {self.task_type}"
            )
        # Stable ordering by (sub, time) — required for valid lag/horizon operations
        self.df = self.df.sort_values([META_SUB, META_TIME]).reset_index(drop=True)

    # ── Accessors ────────────────────────────────────────────────────────

    @property
    def meta_cols(self) -> set[str]:
        return META_COLS

    @property
    def feature_cols(self) -> list[str]:
        """Flat view of all features. Used by the simple (role-agnostic) benchmarks."""
        return self.roles.all_features()

    def subjects(self) -> np.ndarray:
        return self.df[META_SUB].unique()

    def cohorts(self) -> np.ndarray:
        return self.df[META_COHORT].unique()

    def n_subjects(self) -> int:
        return int(self.df[META_SUB].nunique())

    def iter_subjects(self) -> Iterator[tuple[str, pd.DataFrame]]:
        for sub_id, sub_df in self.df.groupby(META_SUB, sort=False):
            yield sub_id, sub_df

    def filter_subjects(self, subs: Iterable) -> "FeatureBundle":
        """Return a new bundle containing only the given subjects."""
        subs = set(subs)
        sub_df = self.df[self.df[META_SUB].isin(subs)].reset_index(drop=True)
        return FeatureBundle(
            df=sub_df,
            roles=FeatureRoles(
                static=list(self.roles.static),
                known_dynamic=list(self.roles.known_dynamic),
                observed_dynamic=list(self.roles.observed_dynamic),
                static_categorical=list(self.roles.static_categorical),
            ),
            target_cols=list(self.target_cols),
            task_type=self.task_type,
            horizons_sec=list(self.horizons_sec),
        )

    # ── Target construction ──────────────────────────────────────────────

    def make_targets(self, horizons_min: list[float]) -> "FeatureBundle":
        """
        Add y(t+H) columns for each requested horizon, per subject (no leakage).

        For each target column, creates `<tgt>__future_<H_sec>s`. The shift is
        computed per subject from that subject's median TR (the `start` spacing),
        so heterogeneous sampling rates and per-subject recording boundaries are
        respected — a future target never comes from a different subject's rows.
        """
        tr_per_sub = (
            self.df.groupby(META_SUB)[META_TIME]
            .diff()
            .groupby(self.df[META_SUB])
            .median()
        )
        global_tr = float(tr_per_sub.median())
        if tr_per_sub.std() > 0.1 * global_tr:
            log.warning(
                f"Heterogeneous TR across subjects (median±std = {global_tr:.2f}±"
                f"{tr_per_sub.std():.2f}s). Using per-subject shift counts."
            )

        for H_min in horizons_min:
            H_sec = int(round(H_min * 60))
            for tgt in self.target_cols:
                new_col = self._future_col_name(tgt, H_min)

                def _shift_group(g):
                    tr = g[META_TIME].diff().median()
                    if pd.isna(tr) or tr <= 0:
                        return pd.Series(np.nan, index=g.index)
                    n = int(round(H_sec / tr))
                    return g[tgt].shift(-n)

                self.df[new_col] = (
                    self.df.groupby(META_SUB, group_keys=False).apply(_shift_group)
                )

        self.horizons_sec = [int(round(H * 60)) for H in horizons_min]
        return self

    def future_target_cols(self, horizon_min: float) -> list[str]:
        return [self._future_col_name(t, horizon_min) for t in self.target_cols]

    @staticmethod
    def _future_col_name(target: str, horizon_min: float) -> str:
        # Seconds (integer) to avoid collisions when fractional minutes round equal.
        H_sec = int(round(horizon_min * 60))
        return f"{target}__future_{H_sec}s"


# ─── Loaders ─────────────────────────────────────────────────────────────

def load_bundle(
    path: str | Path,
    target_cols: list[str],
    task_type: str = "regression",
    *,
    static: Optional[list[str]] = None,
    known_dynamic: Optional[list[str]] = None,
    observed_dynamic: Optional[list[str]] = None,
    static_categorical: Optional[list[str]] = None,
    feature_cols: Optional[list[str]] = None,
) -> FeatureBundle:
    """
    Read a parquet or CSV file and return a typed FeatureBundle.

    Provide the typed lists (recommended):
      - static            : time-invariant covariates (demographics)
      - known_dynamic     : the stimulus (known past + future)
      - observed_dynamic  : brain history (past only)
      - static_categorical: subset of `static` that is categorical

    Backward-compatible fallback:
      - feature_cols : if given (and the typed lists are not), every column is
        treated as observed_dynamic. This reproduces the old flat behaviour and
        is only appropriate for the simple benchmarks; a real TFT run should
        use the typed lists so the stimulus is correctly marked as known.

    target_cols : columns to predict. One for classification; many for
                  multi-output regression.
    """
    path = Path(path)
    if path.suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    if feature_cols is not None and not any([static, known_dynamic, observed_dynamic]):
        log.warning(
            "load_bundle called with flat feature_cols and no typed roles. "
            "All features will be treated as observed_dynamic (past-only). "
            "For a correct TFT run, pass static / known_dynamic / observed_dynamic."
        )
        roles = FeatureRoles(observed_dynamic=list(feature_cols))
    else:
        roles = FeatureRoles(
            static=list(static or []),
            known_dynamic=list(known_dynamic or []),
            observed_dynamic=list(observed_dynamic or []),
            static_categorical=list(static_categorical or []),
        )

    needed = set(roles.all_features()) | set(target_cols) | META_COLS
    missing = needed - set(df.columns)
    if missing:
        shown = sorted(missing)[:10]
        raise ValueError(
            f"Columns missing from {path}: {shown}{'...' if len(missing) > 10 else ''}"
        )

    df = df[sorted(needed)].copy()

    log.info(
        f"Loaded {len(df):,} rows from {path.name}: "
        f"{df[META_SUB].nunique()} subjects, {df[META_COHORT].nunique()} cohorts | "
        f"roles: static={len(roles.static)}, "
        f"known={len(roles.known_dynamic)}, observed={len(roles.observed_dynamic)} | "
        f"{len(target_cols)} targets"
    )
    return FeatureBundle(df=df, roles=roles, target_cols=target_cols, task_type=task_type)
