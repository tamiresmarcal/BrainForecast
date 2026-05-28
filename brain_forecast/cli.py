"""
cli.py

Single entry point for brain_forecast. Two subcommands:

    python -m brain_forecast run --predictors tft --brain ... [flags]
    python -m brain_forecast aggregate --output-dir ...

`run` executes one cluster experiment (formerly runs/run_experiment.py).
Supports fold parallelism via --fold-idx for SLURM arrays. Predictor set
and hyperparameters come from CLI flags. Outputs:

  - basic (default):  scores.csv, resource_profile.csv
  - --full-outputs:   also horizon_curves.png, per_cohort.png, heatmap.png

`aggregate` concatenates per-fold scores.csv files from a fold-parallel
SLURM array run and prints the aggregate metric table.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import resource
import sys
import time
from pathlib import Path

# ── GPU lock: before any TF-bearing import (Nibi dev-guide pattern) ──────
os.environ.pop("SSL_CERT_FILE", None)
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402

from .cv import StratifiedSubjectOutCV
from .data import load_bundle_multi
from .evaluation import run_experiment
from .features import SequenceAdapter, TabularAdapter
from .reporting import (
    aggregate_scores,
    load_fold_scores,
    plot_horizon_curves,
    plot_per_movie_heatmap,
)


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


log = logging.getLogger("bf.run")


# ── Predictor bundles (convenience aliases for --predictors) ─────────────
# These expand to the comma-separated list a user could also write directly.
PREDICTOR_BUNDLES = {
    "bench": ["persistence", "moving_average", "ar"],
    "tft":   ["tft"],
    "all":   ["persistence", "moving_average", "ar", "banded_ridge", "tft"],
}
VALID_PREDICTORS = {"persistence", "moving_average", "ar", "banded_ridge", "tft"}
GPU_PREDICTORS = {"tft", "banded_ridge"}


def _parse_predictors(spec: str) -> list[str]:
    """Resolve a comma-separated list (with optional bundle names) into
    a canonical predictor name list, deduped, in user-given order."""
    raw = [s.strip() for s in spec.split(",") if s.strip()]
    out: list[str] = []
    seen: set[str] = set()
    for token in raw:
        names = PREDICTOR_BUNDLES.get(token, [token])
        for name in names:
            if name not in VALID_PREDICTORS:
                raise ValueError(
                    f"Unknown predictor '{name}'. Valid: "
                    f"{sorted(VALID_PREDICTORS)} or bundle "
                    f"{sorted(PREDICTOR_BUNDLES)}."
                )
            if name not in seen:
                out.append(name)
                seen.add(name)
    if not out:
        raise ValueError("--predictors resolved to an empty list.")
    return out


def _build_predictor_specs(names: list[str], args) -> list[dict]:
    """Turn predictor names + parsed args into the dict list run_experiment
    expects. Each predictor reads its own hyperparameter flags."""
    specs: list[dict] = []
    for name in names:
        if name == "persistence":
            specs.append({"name": "persistence"})
        elif name == "moving_average":
            specs.append({"name": "moving_average", "kwargs": {"k": args.ma_k}})
        elif name == "ar":
            specs.append({"name": "ar", "kwargs": {"p": args.ar_p}})
        elif name == "banded_ridge":
            specs.append({
                "name": "banded_ridge",
                "kwargs": {
                    "n_iter": args.banded_ridge_n_iter,
                    "backend": args.banded_ridge_backend,
                },
            })
        elif name == "tft":
            specs.append({
                "name": "tft",
                "kwargs": {
                    "max_epochs": args.max_epochs,
                    "hidden_size": args.hidden_size,
                    "attention_head_size": args.attention_head_size,
                    "dropout": args.dropout,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "num_workers": args.num_workers,
                    "device": "cuda",
                },
            })
        else:
            raise ValueError(f"Unhandled predictor: {name}")
    return specs


# ── resource probe (was in run_experiment.py) ───────────────────────────
_marks: list[tuple] = []


def _snapshot(label: str) -> None:
    gc.collect()
    ru = resource.getrusage(resource.RUSAGE_SELF)
    ram_gb = ru.ru_maxrss / (1024 ** 2)
    vram_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    _marks.append((label, time.time(), ram_gb, vram_gb))
    log.info(f"[{label:22s}] peakRAM={ram_gb:6.2f}GB  peakVRAM={vram_gb:6.2f}GB")


def _write_profile(output_dir: Path) -> None:
    rows = []
    for i, (label, t, ram, vram) in enumerate(_marks):
        dt = 0.0 if i == 0 else t - _marks[i - 1][1]
        rows.append({
            "stage": label,
            "dt_sec": round(dt, 1),
            "peakRAM_GB": round(ram, 2),
            "peakVRAM_GB": round(vram, 2),
        })
    prof = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    prof.to_csv(output_dir / "resource_profile.csv", index=False)
    log.info("Resource profile:\n" + prof.to_string(index=False))
    peak_ram = prof["peakRAM_GB"].max()
    peak_vram = prof["peakVRAM_GB"].max()
    total_sec = sum(r["dt_sec"] for r in rows)
    log.info(f"TOTAL wall (probed stages): {total_sec/60:.1f} min")
    log.info(f"PEAK RAM : {peak_ram:.2f} GB -> next --mem ~ {int(max(8, peak_ram*1.5))}G")
    log.info(f"PEAK VRAM: {peak_vram:.2f} GB (H100 has 80 GB)")
    log.info(
        "SIZING_SUMMARY "
        f"peakRAM_GB={peak_ram:.2f} peakVRAM_GB={peak_vram:.2f} "
        f"probed_wall_min={total_sec/60:.1f} "
        f"suggest_mem_G={int(max(8, peak_ram*1.5))}"
    )


def _log_hardware() -> None:
    parts: list[str] = []
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        parts.append(f"gpu={props.name.replace(' ', '_')}")
        parts.append(f"vram_total_GB={props.total_memory/1e9:.1f}")
        parts.append(f"cuda_cap={props.major}.{props.minor}")
    else:
        parts.append("gpu=NONE")
    parts.append(f"cpu_count={os.cpu_count()}")
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    parts.append(f"host_ram_GB={kb/1024/1024:.1f}")
                    break
    except OSError:
        pass
    for env_key, label in [
        ("SLURM_CPUS_PER_TASK", "slurm_cpus"),
        ("SLURM_MEM_PER_NODE", "slurm_mem_MB"),
        ("SLURMD_NODENAME", "node"),
        ("SLURM_JOB_ID", "jobid"),
        ("SLURM_ARRAY_TASK_ID", "array_task"),
    ]:
        v = os.environ.get(env_key)
        if v:
            parts.append(f"{label}={v}")
    log.info("HARDWARE " + " ".join(parts))


def _build_roles(brain_path, stimulus_path, static_path, known_dynamic_cap):
    brain_cols = pq.ParquetFile(brain_path).schema.names
    stim_cols = pq.ParquetFile(stimulus_path).schema.names if stimulus_path else []
    stat_cols = (
        pd.read_csv(static_path, nrows=0).columns.tolist() if static_path else []
    )
    static = [c for c in stat_cols if c in ("age", "sex")]
    static_categorical = [c for c in static if c == "sex"]
    known_dynamic_all = [c for c in stim_cols if c.startswith("mov_")]
    if known_dynamic_cap > 0 and len(known_dynamic_all) > known_dynamic_cap:
        known_dynamic = known_dynamic_all[:known_dynamic_cap]
        log.warning(
            f"known_dynamic capped at {known_dynamic_cap} of "
            f"{len(known_dynamic_all)} mov_* columns. "
            "Pass --known-dynamic-cap 0 to disable the cap."
        )
    else:
        known_dynamic = known_dynamic_all
    observed_dynamic: list[str] = []
    target_cols = [c for c in brain_cols if c.startswith("b_")]
    log.info(
        f"schemas: brain={len(brain_cols)} stim={len(stim_cols)} stat={len(stat_cols)} "
        f"| static={len(static)} known={len(known_dynamic)}/{len(known_dynamic_all)} "
        f"observed={len(observed_dynamic)} targets(all b_*)={len(target_cols)}"
    )
    return static, static_categorical, known_dynamic, observed_dynamic, target_cols


class _SingleFoldCV:
    """Wraps a StratifiedSubjectOutCV so it yields only fold `fold_idx`.

    The underlying CV is deterministic (fixed random_state), so fold i in
    a fold-parallel SLURM array is identical to fold i in a single-process
    run. That's what makes the array a valid parallelization.
    """

    def __init__(self, inner: StratifiedSubjectOutCV, fold_idx: int, n_folds: int):
        self.inner = inner
        self.fold_idx = fold_idx
        self.n_folds = n_folds

    def split(self, bundle):
        all_folds = list(self.inner.split(bundle))
        n_actual = len(all_folds)
        if n_actual != self.n_folds:
            log.warning(
                f"--n-folds={self.n_folds} but the CV produced {n_actual} folds "
                f"(per-cohort LOSO can override k_default). Using the actual count."
            )
        if not (0 <= self.fold_idx < n_actual):
            raise ValueError(
                f"--fold-idx={self.fold_idx} is out of range; "
                f"CV produced folds 0..{n_actual - 1}."
            )
        for i, (train, test) in enumerate(all_folds):
            if i == self.fold_idx:
                yield train, test
                return


# ── `run` subcommand ─────────────────────────────────────────────────────

def run_from_args(args) -> int:
    """Execute one experiment from parsed CLI args."""
    # Validate.
    if args.fold_idx is not None and args.n_folds is None:
        log.error("--fold-idx requires --n-folds.")
        return 2
    if args.stride < 1:
        log.error("--stride must be >= 1.")
        return 2
    if args.window_min <= 0:
        log.error("--window-min must be > 0.")
        return 2

    # Resolve predictor list and decide GPU need.
    predictor_names = _parse_predictors(args.predictors)
    needs_gpu = any(p in GPU_PREDICTORS for p in predictor_names)
    out = Path(args.output_dir)

    # Compute the minimal TabularAdapter ops set for this predictor list.
    # Persistence, MA, and AR only read `target_history` columns — the
    # 600 derived stimulus columns from lag/rolling/HRF (at cap=100) are
    # dead weight for them, and the materialization is what was OOM-ing
    # bench runs at --mem=15G. Banded ridge is the only predictor that
    # reads those features, so we add them back when it's in the list.
    adapter_ops = ["target_history"]
    if "banded_ridge" in predictor_names:
        adapter_ops += ["lag", "rolling", "hrf"]

    # ── EXPERIMENT_CONFIG: grep-friendly, all knobs visible ──────────────
    import brain_forecast
    log.info(
        "EXPERIMENT_CONFIG "
        f"predictors={','.join(predictor_names)} "
        f"window_min={args.window_min} "
        f"horizon_min={args.horizon_min} "
        f"stride={args.stride} "
        f"n_targets={args.n_targets} "
        f"known_dynamic_cap={args.known_dynamic_cap} "
        f"k_default={args.k_default} "
        f"loso_threshold={args.loso_threshold} "
        f"fold_idx={args.fold_idx} "
        f"n_folds={args.n_folds} "
        f"full_outputs={args.full_outputs} "
        # TFT hyperparams (logged regardless; ignored if tft not in predictors)
        f"tft_max_epochs={args.max_epochs} "
        f"tft_hidden_size={args.hidden_size} "
        f"tft_attention_head_size={args.attention_head_size} "
        f"tft_dropout={args.dropout} "
        f"tft_batch_size={args.batch_size} "
        f"tft_learning_rate={args.learning_rate} "
        f"tft_num_workers={args.num_workers} "
        # Other predictor kwargs
        f"ar_p={args.ar_p} "
        f"ma_k={args.ma_k} "
        f"banded_ridge_n_iter={args.banded_ridge_n_iter} "
        f"banded_ridge_backend={args.banded_ridge_backend} "
        f"tabular_adapter_ops={'+'.join(adapter_ops)} "
        f"output_dir={out} "
        f"bf_version={getattr(brain_forecast, '__version__', '?')}"
    )

    if needs_gpu:
        if not torch.cuda.is_available():
            log.error("Predictor list needs a GPU but torch.cuda.is_available() is False.")
            return 1
        torch.zeros(1).cuda()
        torch.cuda.reset_peak_memory_stats()
        torch.set_float32_matmul_precision("high")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        log.info("CPU-only run (no GPU predictor in list).")

    _log_hardware()
    _snapshot("start")

    static, static_categorical, known_dynamic, observed_dynamic, targets_all = (
        _build_roles(args.brain, args.stimulus, args.static, args.known_dynamic_cap)
    )
    target_subset = targets_all[: args.n_targets]
    log.info(f"Predicting {len(target_subset)} target(s): {target_subset[:10]}"
             + (f" ... ({len(target_subset)-10} more)" if len(target_subset) > 10 else ""))

    bundle = load_bundle_multi(
        brain_path=args.brain,
        stimulus_path=args.stimulus,
        static_path=args.static,
        target_cols=target_subset,
        task_type="regression",
        static=static,
        static_categorical=static_categorical,
        known_dynamic=known_dynamic,
        observed_dynamic=observed_dynamic,
        stimulus_join="exact",
        asof_tolerance_sec=None,
        on_missing_subjects=args.on_missing_subjects,
    )
    _snapshot("bundle loaded")
    log.info(f"subjects={bundle.n_subjects()} cohorts={list(bundle.cohorts())}")

    if needs_gpu and not known_dynamic:
        log.warning(
            "No known_dynamic (stimulus) columns: at horizon>0 the TFT has no "
            "future-side signal and degenerates to an autoregressor."
        )

    base_cv = StratifiedSubjectOutCV(
        k_default=args.k_default,
        loso_threshold=args.loso_threshold,
        random_state=0,
    )
    if args.fold_idx is not None:
        log.info(
            f"FOLD-PARALLEL: running fold {args.fold_idx} of {args.n_folds} "
            "(this process trains one fold only)."
        )
        cv = _SingleFoldCV(base_cv, fold_idx=args.fold_idx, n_folds=args.n_folds)
    else:
        cv = base_cv

    if args.stride > 1:
        log.info(
            f"DEV-MODE stride={args.stride}: TFT will see ~{100/args.stride:.0f}%% "
            f"of timesteps. window_min and horizon_min retain wall-clock meaning. "
            "Set --stride 1 for publishable runs."
        )

    predictor_specs = _build_predictor_specs(predictor_names, args)

    results = run_experiment(
        bundle=bundle,
        predictor_specs=predictor_specs,
        horizons_min=[args.horizon_min],
        cv=cv,
        tabular_adapter=TabularAdapter(
            ops=adapter_ops,
            k_lag=3,
            rolling_window=10,
            target_history_lags=5,
        ),
        sequence_adapter=SequenceAdapter(
            window_min=args.window_min,
            stride=args.stride,
        ),
        output_dir=str(out),
    )
    if args.fold_idx is not None and "fold" in results.columns:
        results["fold"] = args.fold_idx
        results.to_csv(out / "scores.csv", index=False)
    _snapshot("run done")

    try:
        agg = aggregate_scores(
            results, metric="r2_mean", by=["predictor", "horizon_min"]
        )
        log.info("Aggregate r2_mean (this fold only):\n" + agg.to_string(index=False))
    except Exception as e:
        log.warning(f"aggregate_scores skipped: {e}")

    _write_profile(out)

    # ── full outputs: plots, only when explicitly requested ──────────────
    if args.full_outputs:
        metric = "r2_mean"
        try:
            plot_horizon_curves(
                results, metric=metric,
                output_path=out / "horizon_curves.png",
                title=f"{metric} vs forecast horizon",
            )
            plot_horizon_curves(
                results, metric=metric, per_cohort=True,
                output_path=out / "horizon_curves_per_cohort.png",
                title=f"{metric} per cohort",
            )
            plot_per_movie_heatmap(
                results, metric=metric, output_path=out / "heatmap.png"
            )
            log.info(f"Wrote plots to {out} (--full-outputs)")
        except Exception as e:
            log.warning(f"Plot generation failed: {e}")
    else:
        log.info("Basic outputs only. Pass --full-outputs for plots.")

    log.info(f"Wrote scores + profile to {out}")
    return 0


# ── `aggregate` subcommand ───────────────────────────────────────────────

def aggregate_from_dir(
    output_dir: str | Path,
    metric: str = "r2_mean",
    pattern: str = "fold_*",
) -> None:
    combined = load_fold_scores(output_dir, pattern=pattern)
    out = Path(output_dir)
    out_path = out / "scores_all_folds.csv"
    combined.to_csv(out_path, index=False)
    log.info(f"Wrote combined scores to {out_path}")

    if metric not in combined.columns:
        log.warning(
            f"Metric '{metric}' not in scores; available columns: "
            f"{sorted(c for c in combined.columns if c != '_source_fold_dir')}"
        )
        return
    group_cols = [c for c in ("predictor", "horizon_min") if c in combined.columns]
    if not group_cols:
        log.warning("No 'predictor' or 'horizon_min' column; skipping aggregation.")
        return
    agg = aggregate_scores(combined, metric=metric, by=group_cols)
    log.info(f"Aggregated {metric} across folds:\n" + agg.to_string(index=False))


# ── argparse plumbing ────────────────────────────────────────────────────

def _add_run_args(p: argparse.ArgumentParser) -> None:
    # Required
    p.add_argument("--brain", required=True, help="BRAIN parquet (required)")
    p.add_argument("--stimulus", default=None, help="STIMULUS parquet (optional)")
    p.add_argument("--static", default=None, help="STATIC csv (optional)")
    p.add_argument("--output-dir", required=True)

    # Predictor selection
    p.add_argument(
        "--predictors", required=True,
        help="Comma-separated predictor list. Names: persistence, moving_average, "
        "ar, banded_ridge, tft. Bundle aliases also accepted: bench, tft, all. "
        "Example: --predictors persistence,ar,tft or --predictors bench",
    )

    # Experiment knobs
    p.add_argument(
        "--window-min", type=float, default=1.0,
        help="Past context length in minutes (SequenceAdapter / TFT). Default 1.0.",
    )
    p.add_argument(
        "--horizon-min", type=float, default=1.0,
        help="Forecast horizon in minutes. Default 1.0.",
    )
    p.add_argument(
        "--stride", type=int, default=1,
        help="Temporal stride for TFT (SequenceAdapter). 1=full data (default), "
        "2=~50%% data, 3=drops ~67%%. window/horizon keep wall-clock meaning.",
    )
    p.add_argument("--n-targets", type=int, default=1)
    p.add_argument(
        "--known-dynamic-cap", type=int, default=100,
        help="Limit on number of mov_* columns (0 = no cap). Default 100.",
    )

    # CV
    p.add_argument("--k-default", type=int, default=5)
    p.add_argument("--loso-threshold", type=int, default=10)
    p.add_argument(
        "--on-missing-subjects", default="error",
        choices=["error", "drop", "warn"],
    )
    p.add_argument(
        "--fold-idx", type=int, default=None,
        help="Run only this fold (0-indexed). Required for SLURM array fold-parallel.",
    )
    p.add_argument(
        "--n-folds", type=int, default=None,
        help="Total folds this run is part of. Required with --fold-idx.",
    )

    # TFT hyperparameters
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--hidden-size", type=int, default=16)
    p.add_argument("--attention-head-size", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--learning-rate", type=float, default=5e-3)
    p.add_argument("--num-workers", type=int, default=8,
                   help="DataLoader worker processes for TFT. Default 8.")

    # Other predictor kwargs (used only if that predictor is in the list)
    p.add_argument("--ar-p", type=int, default=5, help="AR order p")
    p.add_argument("--ma-k", type=int, default=5, help="Moving average window k")
    p.add_argument("--banded-ridge-n-iter", type=int, default=20)
    p.add_argument("--banded-ridge-backend", default="torch_cuda",
                   choices=["numpy", "torch", "torch_cuda"])

    # Output verbosity
    p.add_argument(
        "--full-outputs", action="store_true",
        help="Generate plots (horizon_curves.png, per_cohort.png, heatmap.png) "
        "in addition to scores.csv + resource_profile.csv. Default: basic only.",
    )


def main(argv: list[str] | None = None) -> int:
    _setup_logging("INFO")

    parser = argparse.ArgumentParser(prog="brain-forecast")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run one experiment")
    _add_run_args(run_p)

    agg_p = sub.add_parser(
        "aggregate",
        help="Concatenate per-fold scores from a fold-parallel SLURM array",
    )
    agg_p.add_argument("--output-dir", required=True)
    agg_p.add_argument("--metric", default="r2_mean")
    agg_p.add_argument("--pattern", default="fold_*")

    args = parser.parse_args(argv)

    if args.cmd == "run":
        return run_from_args(args)
    if args.cmd == "aggregate":
        aggregate_from_dir(args.output_dir, metric=args.metric, pattern=args.pattern)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
