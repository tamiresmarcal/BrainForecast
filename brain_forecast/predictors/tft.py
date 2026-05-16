"""
predictors/tft.py

Temporal Fusion Transformer (Lim et al., Int. J. Forecasting 2021).

Multi-horizon sequence forecaster with built-in interpretability:
  - Variable Selection Networks (VSN) produce per-feature importance
    weights that can be averaged per time-step to get "which feature
    family mattered" for each prediction.
  - Multi-head attention over the time axis produces a per-timestep
    attention map showing which past moments the model attended to.

This wrapper uses `pytorch_forecasting`'s reference TFT implementation
with sensible defaults for the brain-forecasting task. Hyperparameter
tuning is left for a future iteration — for v0 we lock reasonable
defaults so the harness can run end-to-end.

Key design choices:
  - Multi-output: one TFT predicts all targets simultaneously.
  - Multi-horizon: one TFT predicts y(t+H) for ALL chosen horizons.
    The output at horizon position h corresponds to horizons_min[h].
  - No HRF, no explicit lag stack — model learns alignment from the
    window of past frames.
  - Quantile loss by default for regression (gives uncertainty for free);
    cross-entropy for classification.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class TFTPredictor:
    """
    Temporal Fusion Transformer wrapper.

    Parameters
    ----------
    horizons_min : list[float]
        Horizons (in minutes) the model is trained to predict simultaneously.
        Determines the output dimensionality along the horizon axis.
    max_epochs : int
        Training epochs (with early stopping inside).
    learning_rate : float
    hidden_size : int
    attention_head_size : int
    dropout : float
    batch_size : int
    device : str
        'cuda' or 'cpu'. Auto-detected if None.
    output_dir : str | None
        Where to save checkpoints. None → temp dir.

    Notes
    -----
    The adapter for TFT is SequenceAdapter (not TabularAdapter). The fit /
    predict methods accept whatever SequenceAdapter.transform returns plus
    an additional `horizon_min` argument indicating which horizon to extract
    at predict time.
    """

    horizons_min: list[float] = field(default_factory=lambda: [0, 5, 10, 15, 30, 45, 60])
    max_epochs: int = 30
    learning_rate: float = 1e-3
    hidden_size: int = 64
    attention_head_size: int = 4
    dropout: float = 0.1
    batch_size: int = 128
    device: Optional[str] = None
    output_dir: Optional[str] = None

    # Required protocol attributes
    adapter_type: str = "sequence"
    task_type: str = "regression"

    # ── lifecycle ────────────────────────────────────────────────────────

    def fit(self, X, y, meta) -> "TFTPredictor":
        """
        Parameters
        ----------
        X : dict
            Output of SequenceAdapter.transform. Contains 'data', 'time_idx_col',
            'group_ids', 'feature_cols', 'target_cols' (one per horizon),
            'window_samples', 'horizon_min', 'task_type'.
        y : ignored
            Targets are inside X['data'] under X['target_cols'].
        meta : ignored
            Sequence adapter carries meta inside X['data'].
        """
        import torch
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.metrics import QuantileLoss, CrossEntropy
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import EarlyStopping

        data = X["data"]
        time_col = X["time_idx_col"]
        group_ids = X["group_ids"]
        static_categoricals = list(X.get("static_categoricals", []))
        static_reals = list(X.get("static_reals", []))
        known_reals = list(X.get("known_reals", []))      # stimulus (x): past+future
        observed_reals = list(X.get("observed_reals", []))  # brain history (z): past only
        target_cols = X["target_cols"]  # one column per requested horizon
        window = X["window_samples"]
        task = X["task_type"]

        # One TFT per horizon target (v0 design: shared architecture, separate
        # fitted weights per horizon, selected at predict time).
        self._models_ = {}
        self._datasets_ = {}
        self._window_ = window
        self._task_ = task
        self._target_cols_ = target_cols
        self._group_ids_ = group_ids
        self._time_col_ = time_col

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        accelerator = "gpu" if device == "cuda" else "cpu"

        out_dir = self.output_dir or tempfile.mkdtemp(prefix="tft_")
        os.makedirs(out_dir, exist_ok=True)

        for tgt_col in target_cols:
            log.info(
                f"Fitting TFT for {tgt_col} | "
                f"static(cat={len(static_categoricals)},real={len(static_reals)}) "
                f"known={len(known_reals)} observed={len(observed_reals)}"
            )

            # Role → TimeSeriesDataSet slot mapping (TFT paper Eq. 1):
            #   static_*                  ← s   (time-invariant covariates)
            #   time_varying_known_reals  ← x   (stimulus: known past AND future)
            #   time_varying_unknown_reals← z   (brain history: past only) + target
            # The time index itself is always a known real.
            training = TimeSeriesDataSet(
                data,
                time_idx=time_col,
                target=tgt_col,
                group_ids=group_ids,
                min_encoder_length=window // 2,
                max_encoder_length=window,
                min_prediction_length=1,
                max_prediction_length=1,
                static_categoricals=static_categoricals,
                static_reals=static_reals,
                time_varying_known_reals=[time_col] + known_reals,
                time_varying_unknown_reals=observed_reals + [tgt_col],
                target_normalizer=None if task == "classification" else "auto",
                allow_missing_timesteps=True,
                add_relative_time_idx=True,
                add_target_scales=False,
            )
            train_loader = training.to_dataloader(
                train=True, batch_size=self.batch_size, num_workers=0
            )

            loss = CrossEntropy() if task == "classification" else QuantileLoss()
            tft = TemporalFusionTransformer.from_dataset(
                training,
                learning_rate=self.learning_rate,
                hidden_size=self.hidden_size,
                attention_head_size=self.attention_head_size,
                dropout=self.dropout,
                hidden_continuous_size=min(self.hidden_size, 32),
                loss=loss,
                log_interval=0,
                reduce_on_plateau_patience=4,
            )
            trainer = pl.Trainer(
                max_epochs=self.max_epochs,
                accelerator=accelerator,
                devices=1,
                gradient_clip_val=0.1,
                callbacks=[EarlyStopping(monitor="train_loss", patience=5, mode="min")],
                default_root_dir=out_dir,
                enable_progress_bar=True,
                enable_model_summary=False,
                logger=False,
            )
            trainer.fit(tft, train_dataloaders=train_loader)

            self._models_[tgt_col] = tft
            self._datasets_[tgt_col] = training

        return self

    def predict(self, X) -> np.ndarray:
        """
        Predict y(t+H) for all horizons stored in self._models_.

        Returns an array of shape (N_samples, N_targets * N_horizons) where
        column order matches target_cols (which itself includes the horizon
        in its name, e.g. 'roi_42__future_15min').

        If only one target_col was used, returns a 1D array.
        """
        from pytorch_forecasting import TimeSeriesDataSet

        data = X["data"]
        target_cols = X["target_cols"]

        preds_per_target = []
        for tgt_col in target_cols:
            training = self._datasets_[tgt_col]
            ds = TimeSeriesDataSet.from_dataset(training, data, predict=False, stop_randomization=True)
            loader = ds.to_dataloader(train=False, batch_size=self.batch_size, num_workers=0)
            yhat = self._models_[tgt_col].predict(loader)
            # yhat is a torch.Tensor or numpy array (N, prediction_length=1)
            yhat = np.asarray(yhat).reshape(-1)
            preds_per_target.append(yhat)

        out = np.column_stack(preds_per_target)
        return out.ravel() if out.shape[1] == 1 else out

    # ── interpretability hooks (for later use, not v0 reports) ──────────

    def get_attention_weights(self, X, target_col: str):
        """Return attention weights from the TFT for a given horizon target."""
        from pytorch_forecasting import TimeSeriesDataSet
        training = self._datasets_[target_col]
        ds = TimeSeriesDataSet.from_dataset(training, X["data"], predict=False, stop_randomization=True)
        loader = ds.to_dataloader(train=False, batch_size=self.batch_size, num_workers=0)
        raw_preds = self._models_[target_col].predict(loader, mode="raw", return_x=True)
        return self._models_[target_col].interpret_output(raw_preds.output, reduction="mean")

    def get_variable_importance(self, X, target_col: str):
        """Return per-feature importance from the variable selection network."""
        return self.get_attention_weights(X, target_col)
