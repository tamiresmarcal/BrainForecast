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

Identity model (see AGREEMENTS.md):
  - `sub` is only unique WITHIN a cohort (each cohort numbers participants
    1..N independently). The true subject identity is the pair
    (cohort, sub). Internally we materialise a composite UID column
    `__uid__ = "<cohort>/<sub>"` and key CV / target-shifting / filtering
    on it. The original sub/cohort/movie columns are untouched.
  - `movie` identifies the stimulus/film and is the stimulus join key.
    One cohort may contain multiple movies; a subject may watch several.
  - `start` is seconds and RESTARTS at 0 for every movie. Future-target
    shifting is therefore done per (cohort, sub, movie) block.

Reserved metadata columns: sub, start, cohort, movie. They must not appear
in any feature-role list or target list.

Two loaders are provided:

  load_bundle(path, ...)
      Single pre-joined file. Original entry point. Unchanged behaviour,
      but now also builds the UID and accepts a `movie` column if present
      (falls back to cohort-as-movie if absent, for legacy single-file
      tables).

  load_bundle_multi(brain_path, stimulus_path=None, static_path=None, ...)
      Three-channel input following AGREEMENTS.md. Brain is the spine;
      stimulus joins on (movie, start); static joins on (cohort, sub).
      All joins are key-based and validated.
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
META_TIME = "time_s"          # seconds, restarts per movie
META_COHORT = "cohort"       # grouping; may contain multiple movies
META_MOVIE = "movie"         # stimulus/film identifier; stimulus join key
META_UID = "__uid__"         # internal composite subject identity
META_COLS = {META_SUB, META_TIME, META_COHORT, META_MOVIE}


def _make_uid(df: pd.DataFrame) -> pd.Series:
    """Composite subject identity: '<cohort>/<sub>' (sub is cohort-local)."""
    return df[META_COHORT].astype(str) + "/" + df[META_SUB].astype(str)


@dataclass
class FeatureRoles:
    """
    Typed feature schema mapping each column to its TFT input category.

    Attributes
    ----------
    static : list[str]
        Time-invariant covariates (one value per subject). Maps to TFT
        static slots.
    known_dynamic : list[str]
        Stimulus; known across the whole timeline including the future.
        Maps to TFT time_varying_known_*.
    observed_dynamic : list[str]
        Brain history; known only in the past. Maps to
        TFT time_varying_unknown_*.
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
        Per-(cohort, sub, movie, start) rows. Contains META_COLS, the
        internal UID, plus feature/target cols.
    roles : FeatureRoles
        Typed feature schema (static / known_dynamic / observed_dynamic).
    target_cols : list[str]
        Output columns. One for classification; many for multi-output
        regression.
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
                f"FeatureBundle requires metadata columns {sorted(META_COLS)}; "
                f"missing: {sorted(missing)}"
            )
        self.roles.validate()
        overlap = (set(self.roles.all_features()) | set(self.target_cols)) & META_COLS
        if overlap:
            raise ValueError(
                f"Feature/target columns must not overlap with metadata: {overlap}"
            )
        if self.task_type not in {"regression", "classification"}:
            raise ValueError(
                f"task_type must be 'regression' or 'classification', "
                f"got {self.task_type}"
            )
        # Ensure the composite UID exists.
        if META_UID not in self.df.columns:
            self.df[META_UID] = _make_uid(self.df)
        # Stable ordering: subject identity, then movie, then time.
        # Required for valid lag/horizon ops because `start` restarts per
        # movie.
        self.df = self.df.sort_values(
            [META_UID, META_MOVIE, META_TIME]
        ).reset_index(drop=True)

    # ── Accessors ────────────────────────────────────────────────────────

    @property
    def meta_cols(self) -> set[str]:
        return META_COLS

    @property
    def feature_cols(self) -> list[str]:
        """Flat view of all features. Used by the simple benchmarks."""
        return self.roles.all_features()

    def subjects(self) -> np.ndarray:
        """Unique subject identities (composite UID '<cohort>/<sub>')."""
        return self.df[META_UID].unique()

    def cohorts(self) -> np.ndarray:
        return self.df[META_COHORT].unique()

    def movies(self) -> np.ndarray:
        return self.df[META_MOVIE].unique()

    def n_subjects(self) -> int:
        """Number of unique (cohort, sub) subjects."""
        return int(self.df[META_UID].nunique())

    def iter_subjects(self) -> Iterator[tuple[str, pd.DataFrame]]:
        for uid, sub_df in self.df.groupby(META_UID, sort=False):
            yield uid, sub_df

    def filter_subjects(self, uids: Iterable) -> "FeatureBundle":
        """
        Return a new bundle containing only the given subjects.

        `uids` are composite identities ('<cohort>/<sub>'), as produced by
        `subjects()` and consumed by the CV splitter.
        """
        uids = set(uids)
        sub_df = self.df[self.df[META_UID].isin(uids)].reset_index(drop=True)
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
        Add y(t+H) columns for each horizon, per (cohort, sub, movie) block.

        For each target column, creates `<tgt>__future_<H_sec>s`. The shift
        is computed per (subject, movie) block from that block's median TR
        (the `start` spacing). Because `start` restarts at 0 for every
        movie and a subject can watch multiple movies, shifting must NOT
        cross a movie boundary or a subject boundary — otherwise the
        "future" of the last frame of one movie would be taken from the
        first frame of the next. Grouping by (UID, movie) prevents both
        leaks; the future of a block's tail is correctly NaN.
        """
        block_keys = [META_UID, META_MOVIE]

        tr_per_block = (
            self.df.groupby(block_keys)[META_TIME]
            .diff()
            .groupby([self.df[META_UID], self.df[META_MOVIE]])
            .median()
        )
        global_tr = float(tr_per_block.median())
        if tr_per_block.std() > 0.1 * global_tr:
            log.warning(
                f"Heterogeneous TR across (subject, movie) blocks "
                f"(median±std = {global_tr:.2f}±{tr_per_block.std():.2f}s). "
                f"Using per-block shift counts."
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
                    self.df.groupby(block_keys, group_keys=False)
                    .apply(_shift_group)
                )

        self.horizons_sec = [int(round(H * 60)) for H in horizons_min]
        return self

    def future_target_cols(self, horizon_min: float) -> list[str]:
        return [self._future_col_name(t, horizon_min) for t in self.target_cols]

    @staticmethod
    def _future_col_name(target: str, horizon_min: float) -> str:
        H_sec = int(round(horizon_min * 60))
        return f"{target}__future_{H_sec}s"


# ─── Shared helpers ──────────────────────────────────────────────────────

def _read_table(path: str | Path) -> pd.DataFrame:
    """Read one parquet or CSV file into a DataFrame."""
    path = Path(path)
    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix} ({path})")


def _build_roles(
    *,
    static: Optional[list[str]],
    known_dynamic: Optional[list[str]],
    observed_dynamic: Optional[list[str]],
    static_categorical: Optional[list[str]],
    feature_cols: Optional[list[str]],
) -> FeatureRoles:
    """Construct FeatureRoles, honouring the legacy flat-feature_cols fallback."""
    if feature_cols is not None and not any([static, known_dynamic, observed_dynamic]):
        log.warning(
            "Flat feature_cols and no typed roles given. "
            "All features will be treated as observed_dynamic (past-only). "
            "For a correct TFT run, pass static / known_dynamic / observed_dynamic."
        )
        return FeatureRoles(observed_dynamic=list(feature_cols))
    return FeatureRoles(
        static=list(static or []),
        known_dynamic=list(known_dynamic or []),
        observed_dynamic=list(observed_dynamic or []),
        static_categorical=list(static_categorical or []),
    )


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
    Read a single pre-joined parquet or CSV file and return a FeatureBundle.

    Original entry point. Use it when you already have one flat table with
    brain + stimulus + static columns. For separate tables, use
    `load_bundle_multi`.

    Metadata: the table must contain `sub`, `start`, `cohort`. If it also
    has a `movie` column it is used as-is; if not (legacy single-file
    tables), `movie` is set equal to `cohort` so the rest of the pipeline
    still has a movie axis to group on.
    """
    path = Path(path)
    df = _read_table(path)

    roles = _build_roles(
        static=static,
        known_dynamic=known_dynamic,
        observed_dynamic=observed_dynamic,
        static_categorical=static_categorical,
        feature_cols=feature_cols,
    )

    base_meta = {META_SUB, META_TIME, META_COHORT}
    missing_base = base_meta - set(df.columns)
    if missing_base:
        raise ValueError(
            f"Columns missing from {path}: {sorted(missing_base)}"
        )
    if META_MOVIE not in df.columns:
        log.warning(
            f"No '{META_MOVIE}' column in {path.name}; treating each cohort "
            f"as a single movie (legacy single-file mode)."
        )
        df[META_MOVIE] = df[META_COHORT]

    needed = set(roles.all_features()) | set(target_cols) | META_COLS
    missing = needed - set(df.columns)
    if missing:
        shown = sorted(missing)[:10]
        raise ValueError(
            f"Columns missing from {path}: {shown}"
            f"{'...' if len(missing) > 10 else ''}"
        )

    df = df[sorted(needed)].copy()
    df[META_UID] = _make_uid(df)

    log.info(
        f"Loaded {len(df):,} rows from {path.name}: "
        f"{df[META_UID].nunique()} subjects, {df[META_COHORT].nunique()} "
        f"cohorts, {df[META_MOVIE].nunique()} movies | roles: "
        f"static={len(roles.static)}, known={len(roles.known_dynamic)}, "
        f"observed={len(roles.observed_dynamic)} | {len(target_cols)} targets"
    )
    return FeatureBundle(
        df=df, roles=roles, target_cols=target_cols, task_type=task_type
    )


def load_bundle_multi(
    brain_path: str | Path,
    stimulus_path: Optional[str | Path] = None,
    static_path: Optional[str | Path] = None,
    *,
    target_cols: list[str],
    task_type: str = "regression",
    static: Optional[list[str]] = None,
    known_dynamic: Optional[list[str]] = None,
    observed_dynamic: Optional[list[str]] = None,
    static_categorical: Optional[list[str]] = None,
    stimulus_join: str = "exact",
    asof_tolerance_sec: Optional[float] = None,
    on_missing_subjects: str = "error",
) -> FeatureBundle:
    """
    Read up to three channel tables and return a single FeatureBundle.

    See AGREEMENTS.md for the full contract. Summary:

      brain    — SPINE, required. Keys (cohort, sub, movie, start).
                 Carries observed_dynamic + targets.
      stimulus — optional. Keys (movie, start). Carries known_dynamic.
                 Same for every subject who watched that movie.
      static   — optional. Keys (cohort, sub). Carries static. Exactly
                 one row per (cohort, sub).

    `sub` is cohort-local; the true identity is (cohort, sub), materialised
    as the internal UID. Stimulus joins on (movie, start); static joins on
    (cohort, sub). Both joins are key-based and validated — a sort or
    dropped row upstream raises instead of silently misaligning.

    Within-channel concatenation (10 movie files, several static files) is
    NOT done here in v0; assemble one table per channel yourself.

    Parameters
    ----------
    brain_path : path (required)
    stimulus_path : path, optional
    static_path : path, optional
    target_cols : list[str]  (must live in the brain table)
    task_type : 'regression' | 'classification'
    static / known_dynamic / observed_dynamic / static_categorical : roles
    stimulus_join : {'exact', 'asof'}
        'exact' (default): brain.(movie,start) must match stimulus exactly.
        'asof': nearest start per movie within `asof_tolerance_sec`.
    asof_tolerance_sec : float, optional (only for stimulus_join='asof')
    on_missing_subjects : {'error', 'drop', 'warn'}
        Behaviour when a brain (cohort, sub) has no static row.

    Returns
    -------
    FeatureBundle  (identical object to load_bundle's output)
    """
    if stimulus_join not in {"exact", "asof"}:
        raise ValueError(
            f"stimulus_join must be 'exact' or 'asof', got {stimulus_join!r}"
        )
    if on_missing_subjects not in {"error", "drop", "warn"}:
        raise ValueError(
            f"on_missing_subjects must be 'error', 'drop' or 'warn', "
            f"got {on_missing_subjects!r}"
        )

    roles = _build_roles(
        static=static,
        known_dynamic=known_dynamic,
        observed_dynamic=observed_dynamic,
        static_categorical=static_categorical,
        feature_cols=None,
    )

    # ── 1. Brain table (the spine) ──────────────────────────────────────
    brain = _read_table(brain_path)
    bname = Path(brain_path).name

    brain_missing = META_COLS - set(brain.columns)
    if brain_missing:
        raise ValueError(
            f"Brain table {bname} must contain metadata columns "
            f"{sorted(META_COLS)}; missing: {sorted(brain_missing)}"
        )
    tgt_missing = set(target_cols) - set(brain.columns)
    if tgt_missing:
        raise ValueError(
            f"Target columns missing from brain table {bname}: "
            f"{sorted(tgt_missing)}"
        )
    obs_missing = set(roles.observed_dynamic) - set(brain.columns)
    if obs_missing:
        raise ValueError(
            f"observed_dynamic columns missing from brain table {bname}: "
            f"{sorted(obs_missing)[:10]}"
            f"{'...' if len(obs_missing) > 10 else ''}"
        )

    df = brain.copy()
    df[META_UID] = _make_uid(df)
    log.info(
        f"Brain spine: {len(df):,} rows, {df[META_UID].nunique()} subjects, "
        f"{df[META_COHORT].nunique()} cohorts, {df[META_MOVIE].nunique()} movies"
    )

    # ── 2. Stimulus join on (movie, start) ──────────────────────────────
    if stimulus_path is not None:
        stim = _read_table(stimulus_path)
        sname = Path(stimulus_path).name
        for req in (META_MOVIE, META_TIME):
            if req not in stim.columns:
                raise ValueError(
                    f"Stimulus table {sname} must contain '{req}'; "
                    f"columns include {list(stim.columns)[:8]}..."
                )
        kd_missing = set(roles.known_dynamic) - set(stim.columns)
        if kd_missing:
            raise ValueError(
                f"known_dynamic columns missing from stimulus table "
                f"{sname}: {sorted(kd_missing)[:10]}"
                f"{'...' if len(kd_missing) > 10 else ''}"
            )

        # Every movie in brain must exist in the stimulus table.
        brain_movies = set(df[META_MOVIE].unique())
        stim_movies = set(stim[META_MOVIE].unique())
        movies_absent = sorted(brain_movies - stim_movies)
        if movies_absent:
            raise ValueError(
                f"{len(movies_absent)} movie(s) present in the brain table "
                f"have no rows in the stimulus table {sname}: "
                f"{movies_absent[:10]}"
                f"{'...' if len(movies_absent) > 10 else ''}"
            )

        stim_keep = [META_MOVIE, META_TIME] + list(roles.known_dynamic)
        stim = stim[stim_keep].copy()

        n_before = len(df)
        if stimulus_join == "exact":
            df = df.merge(stim, on=[META_MOVIE, META_TIME], how="left")
            if len(df) != n_before:
                raise ValueError(
                    f"Stimulus exact-merge changed row count "
                    f"({n_before:,} → {len(df):,}). Duplicate (movie, start) "
                    f"keys in the stimulus table {sname}. Deduplicate it "
                    f"before loading."
                )
            if roles.known_dynamic:
                null_mask = df[roles.known_dynamic].isna().any(axis=1)
                if null_mask.any():
                    bad = (
                        df.loc[null_mask, [META_MOVIE, META_TIME]]
                        .drop_duplicates()
                        .head(10)
                    )
                    raise ValueError(
                        f"{int(null_mask.sum()):,} brain rows had no exact "
                        f"stimulus match on (movie, start). The brain and "
                        f"stimulus time grids differ for some movie(s) — use "
                        f"stimulus_join='asof' (with asof_tolerance_sec) if "
                        f"they are sampled differently. First unmatched "
                        f"(movie, start):\n{bad}"
                    )
        else:  # asof — nearest start within each movie
            # merge_asof requires BOTH frames globally sorted by the 'on'
            # key (`start`). `start` resets per movie so it is not globally
            # monotonic; the `by=movie` argument scopes matching per movie,
            # the frames just need to be sorted by `start`. We restore the
            # original row order afterwards.
            df = df.reset_index(drop=True)
            df["__orig_order__"] = np.arange(len(df))
            df = df.sort_values(META_TIME, kind="mergesort").reset_index(drop=True)
            stim = stim.sort_values(META_TIME, kind="mergesort").reset_index(drop=True)
            df = pd.merge_asof(
                df,
                stim,
                on=META_TIME,
                by=META_MOVIE,
                direction="nearest",
                tolerance=asof_tolerance_sec,
            )
            df = (
                df.sort_values("__orig_order__", kind="mergesort")
                .drop(columns="__orig_order__")
                .reset_index(drop=True)
            )
            if roles.known_dynamic:
                null_mask = df[roles.known_dynamic].isna().any(axis=1)
                if null_mask.any():
                    log.warning(
                        f"asof stimulus merge: {int(null_mask.sum()):,} brain "
                        f"rows had no stimulus within tolerance "
                        f"({asof_tolerance_sec}s) and carry NaN stimulus "
                        f"columns. Consider widening asof_tolerance_sec."
                    )
        log.info(
            f"Stimulus joined ({stimulus_join}) on (movie, start): "
            f"+{len(roles.known_dynamic)} known_dynamic columns"
        )
    elif roles.known_dynamic:
        raise ValueError(
            f"known_dynamic columns declared "
            f"({sorted(roles.known_dynamic)[:5]}...) but no stimulus_path "
            f"given. Pass the stimulus table or drop known_dynamic for a "
            f"brain-only run."
        )

    # ── 3. Static join on (cohort, sub) ─────────────────────────────────
    if static_path is not None:
        st = _read_table(static_path)
        stname = Path(static_path).name
        for req in (META_COHORT, META_SUB):
            if req not in st.columns:
                raise ValueError(
                    f"Static table {stname} must contain '{req}'; columns "
                    f"include {list(st.columns)[:8]}..."
                )
        s_missing = set(roles.static) - set(st.columns)
        if s_missing:
            raise ValueError(
                f"static columns missing from static table {stname}: "
                f"{sorted(s_missing)}"
            )

        st_keep = [META_COHORT, META_SUB] + list(roles.static)
        st = st[st_keep].copy()
        # Exactly one row per (cohort, sub).
        dup = st.duplicated(subset=[META_COHORT, META_SUB])
        if dup.any():
            ex = (
                st.loc[dup, [META_COHORT, META_SUB]]
                .drop_duplicates()
                .head(10)
            )
            raise ValueError(
                f"Static table {stname} is not one row per (cohort, sub); "
                f"conflicting duplicates exist. Examples:\n{ex}"
            )
        st[META_UID] = _make_uid(st)
        st = st.drop(columns=[META_COHORT, META_SUB])

        brain_uids = set(df[META_UID].unique())
        st_uids = set(st[META_UID].unique())
        unmatched = sorted(brain_uids - st_uids)
        if unmatched:
            msg = (
                f"{len(unmatched)} brain subjects (cohort/sub) have no static "
                f"row (e.g. {unmatched[:5]})"
            )
            if on_missing_subjects == "error":
                raise ValueError(
                    msg + ". Set on_missing_subjects='drop' or 'warn' to "
                    "proceed, or fix the static table."
                )
            elif on_missing_subjects == "drop":
                log.warning(msg + " → dropping them from the study.")
                df = df[df[META_UID].isin(st_uids)].reset_index(drop=True)
            else:  # warn
                log.warning(msg + " → keeping them with NaN static columns.")

        n_before = len(df)
        df = df.merge(st, on=META_UID, how="left")
        if len(df) != n_before:
            raise ValueError(
                f"Static merge changed row count "
                f"({n_before:,} → {len(df):,}); the static table is not one "
                f"row per (cohort, sub) after key selection."
            )
        log.info(
            f"Static joined on (cohort, sub): +{len(roles.static)} static "
            f"columns ({df[META_UID].nunique()} subjects)"
        )
    elif roles.static:
        raise ValueError(
            f"static columns declared ({sorted(roles.static)}) but no "
            f"static_path given. Pass the static table or drop static for a "
            f"brain-only run."
        )

    # ── 4. Degenerate-config warning ────────────────────────────────────
    if not roles.known_dynamic:
        log.warning(
            "No known_dynamic (stimulus) columns. The simple benchmarks "
            "(persistence / MA / AR) are valid, but TFT becomes pure "
            "autoregression with no known-future input — fine for a smoke "
            "test, not a real result."
        )

    # ── 5. Trim to needed columns + final validation ────────────────────
    needed = set(roles.all_features()) | set(target_cols) | META_COLS
    needed.add(META_UID)
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(
            f"After all joins, still missing columns: {sorted(missing)[:10]}"
            f"{'...' if len(missing) > 10 else ''}"
        )
    df = df[sorted(needed)].copy()

    log.info(
        f"Assembled bundle: {len(df):,} rows, {df[META_UID].nunique()} "
        f"subjects, {df[META_COHORT].nunique()} cohorts, "
        f"{df[META_MOVIE].nunique()} movies | roles: "
        f"static={len(roles.static)}, known={len(roles.known_dynamic)}, "
        f"observed={len(roles.observed_dynamic)} | {len(target_cols)} targets"
    )
    return FeatureBundle(
        df=df, roles=roles, target_cols=target_cols, task_type=task_type
    )
