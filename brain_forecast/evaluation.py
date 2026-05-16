"""
evaluation.py

Main evaluation loop. Runs subject-out CV across horizons and predictors,
saving per-(predictor, fold, cohort, horizon) scores.

Workflow per fold:
    1. Split bundle into train_bundle and test_bundle by subject.
    2. For each predictor:
        a. Build adapter inputs (tabular or sequence) using train_bundle.
        b. Fit the predictor.
        c. Build adapter inputs from test_bundle.
        d. Predict and score per (cohort, horizon).
    3. Append scores to the results table.

Scoring:
    - regression: Pearson r and R² per target, averaged across targets
      within a (cohort, horizon, fold).
    - classification: macro F1 and accuracy per (cohort, horizon, fold).

For multi-target regression we report the **mean across targets** as the
fold score. Per-target detail is saved separately for downstream analysis.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, r2_score
from scipy.stats import pearsonr

from .cv import StratifiedSubjectOutCV
from .data import META_COHORT, META_SUB, META_TIME, FeatureBundle
from .features import SequenceAdapter, TabularAdapter
from .predictors import Predictor, make_predictor

log = logging.getLogger(__name__)


# ─── Scoring helpers ─────────────────────────────────────────────────────

def _safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson r with sanity guards. Returns NaN if undefined."""
    if y_true.std() == 0 or y_pred.std() == 0:
        return float("nan")
    r, _ = pearsonr(y_true, y_pred)
    return float(r)


def score_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute R² and Pearson r averaged across target columns."""
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_pred = y_pred.reshape(-1, 1)

    per_target_r = [_safe_pearson(y_true[:, j], y_pred[:, j]) for j in range(y_true.shape[1])]
    per_target_r2 = [r2_score(y_true[:, j], y_pred[:, j]) for j in range(y_true.shape[1])]

    return {
        "r_mean": float(np.nanmean(per_target_r)),
        "r2_mean": float(np.nanmean(per_target_r2)),
        "n_targets": y_true.shape[1],
        "per_target_r": per_target_r,
        "per_target_r2": per_target_r2,
    }


def score_classification(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n_classes": int(len(np.unique(y_true))),
    }


# ─── Main loop ───────────────────────────────────────────────────────────

def run_experiment(
    bundle: FeatureBundle,
    predictor_specs: list[dict],
    horizons_min: list[float],
    cv: Optional[StratifiedSubjectOutCV] = None,
    tabular_adapter: Optional[TabularAdapter] = None,
    sequence_adapter: Optional[SequenceAdapter] = None,
    output_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Run the full subject-out CV across predictors and horizons.

    Parameters
    ----------
    bundle : FeatureBundle
        Loaded data with feature_cols and target_cols set. `make_targets`
        will be called inside.
    predictor_specs : list of dicts
        Each dict like {'name': 'tft', 'kwargs': {...}}.
    horizons_min : list of floats
        Forecast horizons in minutes.
    cv : StratifiedSubjectOutCV, optional
        Defaults to k_default=5, loso_threshold=10.
    tabular_adapter : TabularAdapter, optional
        Used by Persistence, MA, AR, BandedRidge. Default has lag+rolling+target_history.
    sequence_adapter : SequenceAdapter, optional
        Used by TFT. Default window=5 min.
    output_dir : path-like, optional
        If given, scores are saved as CSV at output_dir / 'scores.csv'.

    Returns
    -------
    pd.DataFrame
        One row per (predictor, fold, cohort, horizon) with score columns.
    """
    cv = cv or StratifiedSubjectOutCV()
    tabular_adapter = tabular_adapter or TabularAdapter(
        ops=["lag", "rolling", "target_history"], k_lag=3, rolling_window=10
    )
    sequence_adapter = sequence_adapter or SequenceAdapter(window_min=5.0)

    # Add future-target columns once, up front
    bundle = bundle.make_targets(horizons_min=horizons_min)

    task = bundle.task_type
    scorer = score_classification if task == "classification" else score_regression

    rows: list[dict] = []

    for fold_idx, (train_subs, test_subs) in enumerate(cv.split(bundle)):
        log.info(
            f"Fold {fold_idx + 1}: {len(train_subs)} train / {len(test_subs)} test subjects"
        )
        train_bundle = bundle.filter_subjects(train_subs)
        test_bundle = bundle.filter_subjects(test_subs)

        for spec in predictor_specs:
            name = spec["name"]
            kwargs = dict(spec.get("kwargs", {}))
            # For TFT, inject horizons so it knows what to predict
            if name == "tft":
                kwargs.setdefault("horizons_min", horizons_min)

            predictor: Predictor = make_predictor(name, task_type=task, **kwargs)
            t0 = time.time()

            if predictor.adapter_type == "sequence":
                # TFT path: one sequence adapter that knows all horizon target columns
                # but the adapter only takes one horizon at a time. We loop horizons
                # inside the fit/predict (TFT trains one model per horizon internally).
                # For simplicity, the sequence adapter outputs include only horizon_min's
                # future target. So we call it once per horizon — TFT's wrapper handles
                # the model dict internally.
                #
                # To keep things efficient: pass all horizons as columns by reusing
                # bundle.make_targets (already done). The sequence adapter exposes
                # `target_cols` listing future cols for one chosen horizon at a time.
                # Here we configure the adapter to expose ALL future cols at once by
                # temporarily monkey-patching its transform output.
                X_train = _sequence_pack_all_horizons(sequence_adapter, train_bundle, horizons_min)
                predictor.fit(X_train, None, None)
                X_test = _sequence_pack_all_horizons(sequence_adapter, test_bundle, horizons_min)
                y_pred_all = predictor.predict(X_test)  # shape (N, n_targets*n_horizons)

                # Score per horizon
                test_data = X_test["data"]
                meta_test = test_data[[META_SUB, META_COHORT, META_TIME]].reset_index(drop=True)
                target_cols = X_test["target_cols"]  # ordered: tgt for each horizon

                # Reshape predictions by horizon
                n_targets = len(bundle.target_cols)
                n_h = len(horizons_min)
                # target_cols is [tgt0_h0, tgt1_h0, ..., tgt0_h1, ...] if we made them
                # in horizon-major order. We did target-major in make_targets per horizon,
                # so the layout per horizon is contiguous: [tgt0_h, tgt1_h, ...].
                # Build a per-horizon slice.
                for h_idx, H in enumerate(horizons_min):
                    h_target_cols = bundle.future_target_cols(H)
                    h_slice_start = h_idx * n_targets
                    h_slice_end = h_slice_start + n_targets
                    y_true = test_data[h_target_cols].to_numpy()
                    y_pred = y_pred_all[:, h_slice_start:h_slice_end] if y_pred_all.ndim == 2 else y_pred_all[:, None]
                    _accumulate_scores(
                        rows, name, fold_idx, H, y_true, y_pred, meta_test, scorer
                    )

            else:
                # Tabular path: one adapter call per horizon, one fit per horizon
                # (because target columns differ per horizon).
                fit_ctx = {"roles": train_bundle.roles}
                for H in horizons_min:
                    Xtr, ytr, _ = tabular_adapter.transform(train_bundle, horizon_min=H)
                    if Xtr.empty:
                        log.warning(f"Empty train at H={H} for {name}; skipping")
                        continue
                    p = make_predictor(name, task_type=task, **spec.get("kwargs", {}))
                    p.fit(Xtr, ytr, fit_ctx)
                    Xte, yte, meta_te = tabular_adapter.transform(test_bundle, horizon_min=H)
                    if Xte.empty:
                        continue
                    y_pred = p.predict(Xte)
                    y_true = yte.to_numpy() if hasattr(yte, "to_numpy") else np.asarray(yte)
                    _accumulate_scores(
                        rows, name, fold_idx, H, y_true, np.asarray(y_pred), meta_te, scorer
                    )

            log.info(f"  {name} fold {fold_idx + 1} done in {time.time() - t0:.1f}s")

    results = pd.DataFrame(rows)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "scores.csv"
        results.to_csv(out_path, index=False)
        log.info(f"Wrote scores to {out_path}")

    return results


# ─── Helpers ─────────────────────────────────────────────────────────────

def _accumulate_scores(
    rows: list[dict],
    predictor_name: str,
    fold_idx: int,
    horizon_min: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta: pd.DataFrame,
    scorer,
) -> None:
    """Compute per-cohort scores and append rows to the running results."""
    # Per-cohort breakdown so we can plot per-movie curves
    for cohort, idx in meta.groupby(META_COHORT).indices.items():
        idx = np.asarray(idx)
        yt = y_true[idx]
        yp = y_pred[idx] if y_pred.ndim > 1 else y_pred[idx]
        s = scorer(yt, yp)
        rows.append({
            "predictor": predictor_name,
            "fold": fold_idx,
            "horizon_min": horizon_min,
            "cohort": cohort,
            "n_samples": len(idx),
            **{k: v for k, v in s.items() if not isinstance(v, list)},
        })


def _sequence_pack_all_horizons(
    adapter: SequenceAdapter, bundle: FeatureBundle, horizons_min: list[float]
) -> dict:
    """
    Pack all-horizon target columns into the sequence-adapter output.

    Standard SequenceAdapter.transform handles one horizon. We extend its
    output here to include the future columns for every requested horizon
    in `target_cols`, ordered as [h0_tgt0, h0_tgt1, ..., h1_tgt0, ...].
    """
    # Use the smallest horizon to build the base (largest valid N)
    base = adapter.transform(bundle, horizon_min=min(horizons_min))
    all_target_cols = []
    for H in horizons_min:
        all_target_cols.extend(bundle.future_target_cols(H))
    base["target_cols"] = all_target_cols
    return base
