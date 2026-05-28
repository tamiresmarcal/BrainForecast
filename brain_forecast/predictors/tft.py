"""
predictors/tft.py

Temporal Fusion Transformer (Lim et al., Int. J. Forecasting 2021).

MULTI-TARGET DESIGN (Design A)
------------------------------
ONE TFT predicts ALL targets simultaneously via a multi-output head. Shared
encoder, only the output projection grows with n_targets — so 10 targets
costs ~1.2× single-target, not 10×.

  * TimeSeriesDataSet(target=target_cols)  -- list, not str (when n_targets>1)
  * loss = MultiLoss([QuantileLoss() for _ in target_cols])  when n_targets>1
  * time_varying_unknown_reals includes ALL target columns
  * predict() returns (N, n_targets) for multi-target, (N,) for single-target

IMPORTANT: pytorch-forecasting treats single-target and multi-target as two
different code paths. For n_targets == 1, the target must be a string and the
loss must be a bare QuantileLoss — wrapping a single loss in MultiLoss causes
a shape mismatch (the model produces an unwrapped output tensor while the
loss expects a list). For n_targets > 1, the target must be a list and the
loss must be MultiLoss. This file handles both cases via a branch on n_targets.

PERFORMANCE NOTES
-----------------
  * num_workers=8, persistent_workers=True, pin_memory=True, prefetch_factor=4
  * precision="bf16-mixed" on H100
  * gradient_clip_val=1.0
  * EarlyStopping removed (no val split)

DIAGNOSTIC LOGGING
------------------
EpochLogger writes per-epoch wall time + train loss to Python logging
(survives SLURM log redirection). `predict start` / `predict done` bracket
inference.

SUBJECT-OUT CV
--------------
NaNLabelEncoder(add_nan=True) for group_ids and static_categoricals so
unseen test subjects don't crash the encoder. Correct behavior for
subject-out CV.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
train_log = logging.getLogger("bf.train")


# ─── Epoch-level logging callback ────────────────────────────────────────

def _make_epoch_logger(max_epochs: int):
    import lightning.pytorch as pl

    class EpochLogger(pl.Callback):
        def __init__(self, total_epochs: int):
            super().__init__()
            self.total_epochs = total_epochs
            self._epoch_start: float | None = None
            self._fit_start: float | None = None

        def on_fit_start(self, trainer, pl_module):
            self._fit_start = time.time()
            train_log.info(
                f"fit start | max_epochs={self.total_epochs} "
                f"device={trainer.accelerator.__class__.__name__}"
            )

        def on_train_epoch_start(self, trainer, pl_module):
            self._epoch_start = time.time()
            train_log.info(f"epoch {trainer.current_epoch:2d}/{self.total_epochs} start")

        def on_train_epoch_end(self, trainer, pl_module):
            now = time.time()
            dt = now - (self._epoch_start or now)
            elapsed = now - (self._fit_start or now)
            metrics = trainer.callback_metrics or {}
            loss_val = metrics.get("train_loss") or metrics.get("train_loss_epoch")
            loss_str = ""
            if loss_val is not None:
                try:
                    loss_str = f" | train_loss={float(loss_val):.4f}"
                except (TypeError, ValueError):
                    pass
            epochs_done = trainer.current_epoch + 1
            avg_per_epoch = elapsed / max(1, epochs_done)
            remaining = avg_per_epoch * max(0, self.total_epochs - epochs_done)
            train_log.info(
                f"epoch {trainer.current_epoch:2d}/{self.total_epochs} done"
                f"{loss_str} | dt={dt/60:.1f}min | elapsed={elapsed/60:.1f}min"
                f" | eta={remaining/3600:.2f}h"
            )

        def on_fit_end(self, trainer, pl_module):
            if self._fit_start is None:
                return
            total = time.time() - self._fit_start
            train_log.info(f"fit end | total={total/60:.1f}min ({total/3600:.2f}h)")

    return EpochLogger(total_epochs=max_epochs)


# ─── Predictor ───────────────────────────────────────────────────────────

@dataclass
class TFTPredictor:
    horizons_min: list[float] = field(default_factory=lambda: [0, 5, 10, 15, 30, 45, 60])
    max_epochs: int = 30
    learning_rate: float = 1e-3
    hidden_size: int = 64
    attention_head_size: int = 4
    dropout: float = 0.1
    batch_size: int = 128
    device: Optional[str] = None
    output_dir: Optional[str] = None
    num_workers: int = 8

    adapter_type: str = "sequence"
    task_type: str = "regression"

    # ── lifecycle ────────────────────────────────────────────────────────

    def fit(self, X, y, meta) -> "TFTPredictor":
        import torch
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data import NaNLabelEncoder
        from pytorch_forecasting.metrics import (
            CrossEntropy, MultiLoss, QuantileLoss,
        )
        import lightning.pytorch as pl

        data = X["data"]
        time_col = X["time_idx_col"]
        group_ids = X["group_ids"]
        static_categoricals = list(X.get("static_categoricals", []))
        static_reals = list(X.get("static_reals", []))
        known_reals = list(X.get("known_reals", []))
        observed_reals = list(X.get("observed_reals", []))
        target_cols = list(X["target_cols"])
        window = X["window_samples"]
        task = X["task_type"]
        n_targets = len(target_cols)

        if n_targets == 0:
            raise ValueError("TFTPredictor requires at least one target column.")

        # Single-target vs multi-target paths differ in pytorch-forecasting:
        #   n_targets == 1: target is a str, loss is a bare metric
        #   n_targets >  1: target is a list, loss is MultiLoss([...])
        # Wrapping a single loss in MultiLoss causes a shape mismatch in the
        # quantile loss's gradient computation. Branch explicitly here.
        single_target = n_targets == 1
        target_arg = target_cols[0] if single_target else target_cols

        # Pre-build NaN-tolerant encoders. Subject-out CV: every test fold has
        # unseen group_ids by design; tiny test folds could miss a static
        # category. Bucketing unseen levels as NaN is what we want.
        categorical_encoders: dict = {}
        for col in group_ids:
            categorical_encoders[col] = NaNLabelEncoder(add_nan=True)
        for col in static_categoricals:
            categorical_encoders[col] = NaNLabelEncoder(add_nan=True)

        self._target_cols_ = target_cols
        self._n_targets_ = n_targets
        self._single_target_ = single_target
        self._window_ = window
        self._task_ = task
        self._group_ids_ = group_ids
        self._time_col_ = time_col

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        accelerator = "gpu" if device == "cuda" else "cpu"
        precision = "bf16-mixed" if accelerator == "gpu" else 32

        out_dir = self.output_dir or tempfile.mkdtemp(prefix="tft_")
        os.makedirs(out_dir, exist_ok=True)

        loader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=(accelerator == "gpu"),
        )
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4

        mode_str = "single-target" if single_target else f"MULTI-TARGET ({n_targets} targets)"
        log.info(
            f"Fitting {mode_str} TFT | "
            f"static(cat={len(static_categoricals)},real={len(static_reals)}) "
            f"known={len(known_reals)} observed={len(observed_reals)} "
            f"bs={self.batch_size} workers={self.num_workers} precision={precision}"
        )
        if n_targets <= 5:
            log.info(f"Targets: {target_cols}")
        else:
            log.info(
                f"Targets (first 3 of {n_targets}): {target_cols[:3]} ... "
                f"last: {target_cols[-1]}"
            )
        log.info(
            f"NaN-tolerant encoders enabled for: "
            f"group_ids={group_ids} static_categoricals={static_categoricals}"
        )

        # time_varying_unknown_reals must include every target column —
        # each one's past is "observed only" (we don't know future brain).
        ts_kwargs = dict(
            time_idx=time_col,
            target=target_arg,
            group_ids=group_ids,
            min_encoder_length=window // 2,
            max_encoder_length=window,
            min_prediction_length=1,
            max_prediction_length=1,
            static_categoricals=static_categoricals,
            static_reals=static_reals,
            time_varying_known_reals=[time_col] + known_reals,
            time_varying_unknown_reals=observed_reals + target_cols,
            target_normalizer="auto" if task != "classification" else None,
            categorical_encoders=categorical_encoders,
            allow_missing_timesteps=True,
            add_relative_time_idx=True,
            add_target_scales=False,
        )
        try:
            training = TimeSeriesDataSet(data, **ts_kwargs)
        except Exception as e:
            log.warning(
                f"TimeSeriesDataSet with target_normalizer='auto' failed: {e}. "
                "Retrying with target_normalizer=None."
            )
            ts_kwargs["target_normalizer"] = None
            training = TimeSeriesDataSet(data, **ts_kwargs)

        train_loader = training.to_dataloader(train=True, **loader_kwargs)

        # Loss: bare metric for single-target, MultiLoss for multi-target.
        if task == "classification":
            single_loss_cls = CrossEntropy
        else:
            single_loss_cls = QuantileLoss
        if single_target:
            loss = single_loss_cls()
        else:
            loss = MultiLoss([single_loss_cls() for _ in range(n_targets)])

        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=min(self.hidden_size, 32),
            loss=loss,
            log_interval=0,
        )
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=accelerator,
            devices=1,
            precision=precision,
            gradient_clip_val=1.0,
            callbacks=[_make_epoch_logger(self.max_epochs)],
            default_root_dir=out_dir,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        trainer.fit(tft, train_dataloaders=train_loader)

        self._model_ = tft
        self._dataset_ = training
        return self

    def predict(self, X) -> np.ndarray:
        """
        Returns:
          (N,)           when n_targets == 1
          (N, n_targets) when n_targets >  1
        """
        from pytorch_forecasting import TimeSeriesDataSet

        data = X["data"]
        target_cols = list(X["target_cols"])
        n_targets = len(target_cols)

        loader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=(self.device or "cuda") == "cuda" and self.num_workers >= 0,
        )
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4

        ds = TimeSeriesDataSet.from_dataset(
            self._dataset_, data, predict=False, stop_randomization=True
        )
        loader = ds.to_dataloader(train=False, **loader_kwargs)

        train_log.info(f"predict start | n_targets={n_targets}")
        t0 = time.time()
        yhat = self._model_.predict(loader)
        train_log.info(f"predict done  | n_targets={n_targets} | dt={(time.time()-t0)/60:.1f}min")

        # Single-target: yhat is a Tensor of shape (N, prediction_length).
        # Multi-target:  yhat is a list of Tensors, len == n_targets.
        if isinstance(yhat, (list, tuple)):
            cols = []
            for t in yhat:
                arr = t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)
                cols.append(arr.reshape(-1))
            out = np.column_stack(cols)
        else:
            arr = yhat.cpu().numpy() if hasattr(yhat, "cpu") else np.asarray(yhat)
            out = arr.reshape(-1, max(1, n_targets))

        return out.ravel() if out.shape[1] == 1 else out

    # ── interpretability hooks ───────────────────────────────────────────

    def get_attention_weights(self, X):
        from pytorch_forecasting import TimeSeriesDataSet
        ds = TimeSeriesDataSet.from_dataset(
            self._dataset_, X["data"], predict=False, stop_randomization=True
        )
        loader = ds.to_dataloader(train=False, batch_size=self.batch_size, num_workers=0)
        raw_preds = self._model_.predict(loader, mode="raw", return_x=True)
        return self._model_.interpret_output(raw_preds.output, reduction="mean")

    def get_variable_importance(self, X):
        return self.get_attention_weights(X)
