"""
cv.py

Cross-validation for subject-out experiments, stratified by cohort (movie).

Core rule: no subject appears in both train and test of any fold.
Stratification: each fold's test set has subjects from every cohort,
proportional to that cohort's subject count.

Adapts to per-cohort subject counts:
  - Cohorts with n_subjects >= loso_threshold use k_default folds
  - Cohorts with n_subjects < loso_threshold use leave-one-subject-out
    (one fold per subject in that cohort)

Folds are aligned globally by taking the max fold count across cohorts.
Cohorts with fewer folds wrap around (modulo).

Typical use:
    cv = StratifiedSubjectOutCV(k_default=5, loso_threshold=10)
    for fold_idx, (train_subs, test_subs) in enumerate(cv.split(bundle)):
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from .data import META_COHORT, META_SUB, FeatureBundle

log = logging.getLogger(__name__)


@dataclass
class StratifiedSubjectOutCV:
    """
    Subject-level k-fold stratified by cohort.

    Parameters
    ----------
    k_default : int
        Number of folds for cohorts large enough to support it.
    loso_threshold : int
        Cohorts with fewer than this many subjects switch to LOSO.
    random_state : int
        Seed for the per-cohort subject shuffle.
    """

    k_default: int = 5
    loso_threshold: int = 10
    random_state: int = 0

    def split(self, bundle: FeatureBundle) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Yields (train_subs, test_subs) per fold.

        Each is a numpy array of subject IDs.
        """
        rng = np.random.default_rng(self.random_state)

        # Build per-cohort subject lists, shuffled
        cohort_subs: dict = {}
        for cohort, sub_df in bundle.df.groupby(META_COHORT):
            subs = sub_df[META_SUB].unique()
            subs = rng.permutation(subs)
            cohort_subs[cohort] = subs

        # Decide fold count per cohort
        per_cohort_k = {
            c: (self.k_default if len(s) >= self.loso_threshold else len(s))
            for c, s in cohort_subs.items()
        }
        n_folds = max(per_cohort_k.values())
        log.info(
            f"Subject-out CV: {n_folds} folds. Per-cohort scheme: "
            + ", ".join(f"{c}(n={len(s)},k={per_cohort_k[c]})" for c, s in cohort_subs.items())
        )

        # Precompute fold assignments for each cohort
        cohort_folds: dict = {}
        for c, s in cohort_subs.items():
            k = per_cohort_k[c]
            cohort_folds[c] = [np.asarray(g) for g in np.array_split(s, k)]

        # Compose folds
        all_subs = np.concatenate(list(cohort_subs.values()))
        for fold in range(n_folds):
            test_pieces = []
            for c, folds in cohort_folds.items():
                k = per_cohort_k[c]
                idx = fold % k  # wrap for cohorts with fewer folds
                test_pieces.append(folds[idx])
            test_subs = np.concatenate(test_pieces) if test_pieces else np.array([])
            train_subs = np.setdiff1d(all_subs, test_subs, assume_unique=True)
            yield train_subs, test_subs

    def n_folds(self, bundle: FeatureBundle) -> int:
        """Compute the number of folds without running the iterator."""
        per_cohort_k = []
        for _, sub_df in bundle.df.groupby(META_COHORT):
            n = sub_df[META_SUB].nunique()
            per_cohort_k.append(self.k_default if n >= self.loso_threshold else n)
        return max(per_cohort_k) if per_cohort_k else 0
