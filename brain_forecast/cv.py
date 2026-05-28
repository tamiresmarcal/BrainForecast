"""
cv.py

Cross-validation for subject-out experiments, stratified by movie.

Core rule: no subject appears in both train and test of any fold, where a
"subject" is the composite identity (cohort, sub) — because `sub` is only
unique within a cohort. The bundle exposes this as `__uid__`.

A subject can watch multiple movies, so one subject (UID) contributes rows
under several movies. Folds are built on **unique UIDs**, not on rows, so a
subject can never leak into both train and test via one of their movies.
Stratification balances movie representation across folds: each fold's test
set draws subjects from every movie, proportional to that movie's subject
count.

Adapts to per-movie subject counts:
  - Movies with n_subjects >= loso_threshold use k_default folds
  - Movies with n_subjects < loso_threshold use leave-one-subject-out
    (one fold per subject in that movie)

Because a subject may appear under several movies, a UID is assigned to a
fold by the FIRST movie (alphabetical) in which it appears, so each UID is
placed exactly once. Stratification counts are therefore by movie-of-record;
the subject-out guarantee is global regardless.

Folds are aligned globally by taking the max fold count across movies.
Movies with fewer folds wrap around (modulo).

Note: the total fold count is NOT necessarily k_default. If any movie has
fewer than loso_threshold subjects it triggers LOSO and the global fold
count becomes the max across movies. To force exactly k_default folds, set
loso_threshold at or below the smallest movie's subject count.

Typical use:
    cv = StratifiedSubjectOutCV(k_default=5, loso_threshold=10)
    for fold_idx, (train_uids, test_uids) in enumerate(cv.split(bundle)):
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from .data import META_MOVIE, META_UID, FeatureBundle

log = logging.getLogger(__name__)


@dataclass
class StratifiedSubjectOutCV:
    """
    Subject-level k-fold stratified by movie.

    The held-out unit is the composite subject identity (cohort, sub),
    carried by the bundle as `__uid__`. Stratification is by `movie`.

    Parameters
    ----------
    k_default : int
        Number of folds for movies large enough to support it.
    loso_threshold : int
        Movies with fewer than this many subjects switch to LOSO.
    random_state : int
        Seed for the per-movie subject shuffle.
    """

    k_default: int = 5
    loso_threshold: int = 10
    random_state: int = 0

    def _uid_movie_of_record(self, bundle: FeatureBundle) -> pd.DataFrame:
        """
        One row per UID: the movie it is assigned to for stratification.

        A subject may appear under several movies; we assign each UID to the
        alphabetically-first movie it appears in, so every UID is placed in
        exactly one stratum and cannot be split across folds.
        """
        pairs = (
            bundle.df[[META_UID, META_MOVIE]]
            .drop_duplicates()
            .sort_values([META_UID, META_MOVIE])
        )
        first = pairs.groupby(META_UID, sort=False)[META_MOVIE].first()
        return first.reset_index().rename(columns={META_MOVIE: "_movie_rec"})

    def split(
        self, bundle: FeatureBundle
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Yields (train_uids, test_uids) per fold.

        Each is a numpy array of composite subject identities ('<cohort>/
        <sub>'), consumed by FeatureBundle.filter_subjects.
        """
        rng = np.random.default_rng(self.random_state)

        uid_rec = self._uid_movie_of_record(bundle)

        # Build per-movie UID lists (movie-of-record), shuffled.
        movie_uids: dict = {}
        for movie, grp in uid_rec.groupby("_movie_rec"):
            uids = grp[META_UID].to_numpy()
            uids = rng.permutation(uids)
            movie_uids[movie] = uids

        # Decide fold count per movie.
        per_movie_k = {
            m: (self.k_default if len(u) >= self.loso_threshold else len(u))
            for m, u in movie_uids.items()
        }
        per_movie_k = {m: max(1, k) for m, k in per_movie_k.items()}
        n_folds = max(per_movie_k.values())
        log.info(
            f"Subject-out CV (held-out unit = cohort/sub): {n_folds} folds. "
            "Per-movie scheme: "
            + ", ".join(
                f"{m}(n={len(u)},k={per_movie_k[m]})"
                for m, u in movie_uids.items()
            )
        )

        # Precompute fold assignments for each movie.
        movie_folds: dict = {}
        for m, u in movie_uids.items():
            k = per_movie_k[m]
            movie_folds[m] = [np.asarray(g) for g in np.array_split(u, k)]

        all_uids = np.concatenate(list(movie_uids.values())) if movie_uids \
            else np.array([])

        for fold in range(n_folds):
            test_pieces = []
            for m, folds in movie_folds.items():
                k = per_movie_k[m]
                idx = fold % k  # wrap for movies with fewer folds
                test_pieces.append(folds[idx])
            test_uids = (
                np.concatenate(test_pieces) if test_pieces else np.array([])
            )
            # A UID is in exactly one stratum, so test_uids is already unique.
            train_uids = np.setdiff1d(all_uids, test_uids, assume_unique=True)
            yield train_uids, test_uids

    def n_folds(self, bundle: FeatureBundle) -> int:
        """Compute the number of folds without running the iterator."""
        uid_rec = self._uid_movie_of_record(bundle)
        per_movie_k = []
        for _, grp in uid_rec.groupby("_movie_rec"):
            n = grp[META_UID].nunique()
            per_movie_k.append(
                self.k_default if n >= self.loso_threshold else max(1, n)
            )
        return max(per_movie_k) if per_movie_k else 0
