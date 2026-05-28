"""
reporting.py

Aggregation tables and the horizon-curve plot.

Three main entry points:

  load_fold_scores(output_dir)
      Read per-fold scores.csv files written by a SLURM-array fold-parallel
      run and return them concatenated as one DataFrame, the same shape
      run_experiment() would return for a single-process N-fold run.

  aggregate_scores(results_df, by=('predictor', 'horizon_min'))
      Returns mean ± std (over folds, optionally over cohorts) of the
      primary metric.

  plot_horizon_curves(results_df, metric='r2_mean', ax=None, ...)
      Reproduces the style of the user's existing benchmark plot:
      one line per predictor (or per ablation), x = horizon in minutes,
      y = metric. Shaded band = std across folds.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

log = logging.getLogger(__name__)


# ─── Per-fold loading (for SLURM-array fold-parallel runs) ───────────────

def load_fold_scores(
    output_dir: str | Path,
    pattern: str = "fold_*",
    scores_filename: str = "scores.csv",
) -> pd.DataFrame:
    """
    Concatenate per-fold scores.csv files into one DataFrame.

    Each fold-parallel SLURM array task writes its own
    ``{output_dir}/fold_<idx>/scores.csv``. This helper reads them all back
    and returns a single DataFrame that looks just like the in-memory
    ``results`` a sequential ``run_experiment`` call would have produced.

    Skipped folds (missing scores.csv) are logged as warnings — partial
    array failures are visible but don't crash aggregation.

    Parameters
    ----------
    output_dir : path-like
        Parent dir containing ``fold_0/``, ``fold_1/``, ...
    pattern : str
        Glob pattern for per-fold subdirs. Default ``"fold_*"``.
    scores_filename : str
        Name of the per-fold scores file. Default ``"scores.csv"``.

    Returns
    -------
    pd.DataFrame
        Concatenated rows from every fold dir that had a scores file.
        Includes an extra column ``_source_fold_dir`` for traceability.

    Raises
    ------
    FileNotFoundError
        If ``output_dir`` does not exist, no fold dirs match ``pattern``,
        or no fold contained ``scores_filename``.
    """
    out = Path(output_dir)
    if not out.exists():
        raise FileNotFoundError(f"Output dir does not exist: {out}")

    fold_dirs = sorted(out.glob(pattern))
    if not fold_dirs:
        raise FileNotFoundError(f"No fold subdirs matched {out}/{pattern}")

    dfs: list[pd.DataFrame] = []
    skipped: list[str] = []
    for d in fold_dirs:
        scores = d / scores_filename
        if not scores.exists():
            skipped.append(d.name)
            continue
        df = pd.read_csv(scores)
        df["_source_fold_dir"] = d.name
        dfs.append(df)

    if skipped:
        log.warning(
            f"Skipped {len(skipped)} fold dir(s) with no {scores_filename}: {skipped}"
        )
    if not dfs:
        raise FileNotFoundError(
            f"No {scores_filename} files found under {out}/{pattern}"
        )

    combined = pd.concat(dfs, ignore_index=True)
    log.info(f"Loaded {len(combined)} rows from {len(dfs)} fold(s) in {out}")
    return combined


# ─── Aggregation ─────────────────────────────────────────────────────────

def aggregate_scores(
    results: pd.DataFrame,
    metric: str,
    by: Iterable[str] = ("predictor", "horizon_min"),
) -> pd.DataFrame:
    """
    Aggregate per-fold per-cohort scores into mean ± std.

    Parameters
    ----------
    results : pd.DataFrame
        From run_experiment() or load_fold_scores().
    metric : str
        Column name to aggregate (e.g. 'r2_mean', 'r_mean', 'f1_macro').
    by : iterable
        Grouping columns. Default = ('predictor', 'horizon_min') gives one
        row per (predictor, horizon) averaged across folds and cohorts.
        Add 'cohort' to get per-movie aggregates.
    """
    by = list(by)
    grouped = results.groupby(by)[metric].agg(["mean", "std", "count"]).reset_index()
    grouped.rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std", "count": "n"}, inplace=True)
    return grouped


# ─── Plotting ────────────────────────────────────────────────────────────

def plot_horizon_curves(
    results: pd.DataFrame,
    metric: str = "r2_mean",
    ax: Optional[plt.Axes] = None,
    per_cohort: bool = False,
    title: Optional[str] = None,
    output_path: Optional[str | Path] = None,
) -> plt.Axes:
    """
    R² (or any metric) vs horizon, one curve per predictor.

    Parameters
    ----------
    results : pd.DataFrame
        From run_experiment() or load_fold_scores().
    metric : str
        Column to plot on Y axis.
    ax : matplotlib axis, optional
        Plot into existing axis. If None, creates a new figure.
    per_cohort : bool
        If True, draws one subplot per cohort. Otherwise pools across cohorts.
    title : str, optional
    output_path : path-like, optional
        If given, saves the figure.

    Returns
    -------
    matplotlib.axes.Axes (single) or list (per_cohort).
    """
    sns.set_theme(style="whitegrid")

    if per_cohort:
        cohorts = sorted(results["cohort"].unique())
        n = len(cohorts)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=True, squeeze=False)
        for ax_i, cohort in zip(axes.ravel(), cohorts):
            sub = results[results["cohort"] == cohort]
            _draw_curves(sub, metric=metric, ax=ax_i)
            ax_i.set_title(cohort, fontsize=11)
        # Hide unused axes
        for ax_i in axes.ravel()[len(cohorts):]:
            ax_i.set_visible(False)
        if title:
            fig.suptitle(title, fontsize=13)
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=120, bbox_inches="tight")
            log.info(f"Saved figure to {output_path}")
        return axes

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    _draw_curves(results, metric=metric, ax=ax)
    if title:
        ax.set_title(title, fontsize=13)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        log.info(f"Saved figure to {output_path}")
    return ax


def _draw_curves(df: pd.DataFrame, metric: str, ax: plt.Axes) -> None:
    """Draw one line per predictor with std band."""
    agg = (
        df.groupby(["predictor", "horizon_min"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )
    predictors = sorted(agg["predictor"].unique())

    palette = sns.color_palette("tab10", n_colors=len(predictors))
    for color, pred in zip(palette, predictors):
        sub = agg[agg["predictor"] == pred].sort_values("horizon_min")
        ax.plot(sub["horizon_min"], sub["mean"], marker="o", label=pred, color=color)
        std = sub["std"].fillna(0)
        ax.fill_between(
            sub["horizon_min"], sub["mean"] - std, sub["mean"] + std, alpha=0.15, color=color
        )

    ax.set_xlabel("Forecast horizon (minutes)")
    ax.set_ylabel(metric)
    ax.legend(loc="best", frameon=True, fontsize=9)


# ─── Per-target heatmap (optional supplementary view) ────────────────────

def plot_per_movie_heatmap(
    results: pd.DataFrame,
    metric: str = "r2_mean",
    predictor: Optional[str] = None,
    output_path: Optional[str | Path] = None,
) -> plt.Axes:
    """
    Heatmap of metric by (cohort × horizon), for a single predictor.

    Parameters
    ----------
    predictor : str, optional
        Restrict to one predictor. If None and multiple are present, takes
        the one with the highest mean metric across all rows.
    """
    if predictor is None:
        means = results.groupby("predictor")[metric].mean()
        predictor = means.idxmax()
        log.info(f"plot_per_movie_heatmap: defaulting to best predictor '{predictor}'")

    sub = results[results["predictor"] == predictor]
    pivot = (
        sub.groupby(["cohort", "horizon_min"])[metric]
        .mean()
        .reset_index()
        .pivot(index="cohort", columns="horizon_min", values=metric)
    )
    fig, ax = plt.subplots(figsize=(1.0 + 0.6 * pivot.shape[1], 0.5 + 0.4 * pivot.shape[0]))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax, cbar_kws={"label": metric})
    ax.set_title(f"{predictor}: {metric} per cohort × horizon")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
    return ax
