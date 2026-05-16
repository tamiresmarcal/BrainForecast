"""
cli.py

Single-command entry point. Reads a YAML config and runs an experiment.

    python -m brain_forecast run --config configs/example.yaml

The config declares everything: paths, feature/target columns, predictors,
horizons, CV scheme, output directory.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from .cv import StratifiedSubjectOutCV
from .data import load_bundle
from .evaluation import run_experiment
from .features import SequenceAdapter, TabularAdapter
from .reporting import plot_horizon_curves, plot_per_movie_heatmap


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_from_config(config_path: str | Path) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())

    _setup_logging(cfg.get("log_level", "INFO"))
    log = logging.getLogger("brain_forecast.cli")
    log.info(f"Loading config from {config_path}")

    # ── Data ────────────────────────────────────────────────────────────
    data_cfg = cfg["data"]
    feat_cfg = data_cfg.get("features", {}) or {}
    bundle = load_bundle(
        path=data_cfg["path"],
        target_cols=data_cfg["target_cols"],
        task_type=data_cfg.get("task_type", "regression"),
        static=feat_cfg.get("static"),
        known_dynamic=feat_cfg.get("known_dynamic"),
        observed_dynamic=feat_cfg.get("observed_dynamic"),
        static_categorical=feat_cfg.get("static_categorical"),
        # Backward-compat: a flat list still works (treated as observed-only)
        feature_cols=data_cfg.get("feature_cols"),
    )

    # ── Adapters ────────────────────────────────────────────────────────
    tab_cfg = cfg.get("tabular_adapter", {}) or {}
    tabular_adapter = TabularAdapter(
        ops=tab_cfg.get("ops", ["lag", "rolling", "target_history"]),
        k_lag=tab_cfg.get("k_lag", 3),
        rolling_window=tab_cfg.get("rolling_window", 10),
        movie_cols=tab_cfg.get("movie_cols"),
        target_history_lags=tab_cfg.get("target_history_lags", 5),
    )
    seq_cfg = cfg.get("sequence_adapter", {}) or {}
    sequence_adapter = SequenceAdapter(window_min=seq_cfg.get("window_min", 5.0))

    # ── CV ──────────────────────────────────────────────────────────────
    cv_cfg = cfg.get("cv", {}) or {}
    cv = StratifiedSubjectOutCV(
        k_default=cv_cfg.get("k_default", 5),
        loso_threshold=cv_cfg.get("loso_threshold", 10),
        random_state=cv_cfg.get("random_state", 0),
    )

    # ── Predictors ──────────────────────────────────────────────────────
    predictor_specs = cfg["predictors"]  # list of dicts {name, kwargs?}

    # ── Run ─────────────────────────────────────────────────────────────
    output_dir = cfg.get("output_dir", "runs/latest")
    results = run_experiment(
        bundle=bundle,
        predictor_specs=predictor_specs,
        horizons_min=cfg["horizons_min"],
        cv=cv,
        tabular_adapter=tabular_adapter,
        sequence_adapter=sequence_adapter,
        output_dir=output_dir,
    )

    # ── Plots ───────────────────────────────────────────────────────────
    plot_cfg = cfg.get("plots", {}) or {}
    metric = plot_cfg.get("metric", "r2_mean" if bundle.task_type == "regression" else "f1_macro")
    out = Path(output_dir)
    plot_horizon_curves(
        results,
        metric=metric,
        output_path=out / "horizon_curves.png",
        title=f"{metric} vs forecast horizon",
    )
    if plot_cfg.get("per_cohort", True):
        plot_horizon_curves(
            results,
            metric=metric,
            per_cohort=True,
            output_path=out / "horizon_curves_per_cohort.png",
            title=f"{metric} per cohort",
        )
    if plot_cfg.get("heatmap", True):
        plot_per_movie_heatmap(
            results, metric=metric, output_path=out / "heatmap.png"
        )

    log.info("Done.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="brain-forecast")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="Run an experiment from a YAML config")
    run_p.add_argument("--config", required=True, help="Path to YAML config")

    args = parser.parse_args(argv)

    if args.cmd == "run":
        run_from_config(args.config)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
