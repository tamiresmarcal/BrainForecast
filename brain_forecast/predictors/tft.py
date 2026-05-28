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

CHANNEL PROJECTION (Algonauts-style)
------------------------------------
When `channel_proj=True`, the 4900 known_dynamic columns are projected
per channel BEFORE the Variable Selection Network, jointly trained with
the rest of the TFT.

Why: the VSN builds one Gated Residual Network per input feature.
4900 GRNs is genuine compute, AND the softmax over 4900 logits flattens,
making per-feature attention weights statistically noisy under
subject-out CV. TRIBE (1st), VIBE (2nd), MedARC (4th) at Algonauts 2025
all project per modality/channel to a small latent (D=192-1024) before
any attention or VSN — see BrainForecast_Learnings notes.

Mechanism (the surgery on pytorch-forecasting):
  1. TimeSeriesDataSet is built with the RAW known_dynamic columns, so
     the dataloader correctly carries x_cont[:, :, raw_indices] every
     batch.
  2. TFT is built via from_dataset(), so all its internal modules
     (embeddings, VSN, LSTM, attention, output) are wired for the raw
     known_dynamic names.
  3. We then SURGICALLY REBUILD the encoder/decoder VSNs so they take
     the *projected* slot names (proj_<channel>_<i>) as inputs instead
     of the raw column names, AND update the matching name lists in
     hparams (time_varying_reals_encoder / time_varying_reals_decoder)
     so the parent's forward() consumes input_vectors using projected
     names rather than raw ones.
  4. forward() is overridden to: build input_vectors as normal, but for
     the raw stim columns route them through the projector first and
     replace those entries with the projected ones before VSN sees them.

Static reals, observed reals, targets, and categoricals are untouched.

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
inference. CHANNEL_PROJ log line records the resolved channel groups
once per fit for grep-friendly forensics.

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

from .channel_projection import (
    DEFAULT_CHANNEL_PATTERN,
    ChannelGroups,
    ChannelProjector,
    parse_channels,
)

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


# ─── Channel-projection patching of a vanilla TFT ────────────────────────

def _attach_channel_projector(
    tft_module,                  # TemporalFusionTransformer instance
    channel_groups: ChannelGroups,
    raw_known_reals: list[str],
    proj_dim: int,
):
    """
    Surgically rewire `tft_module` so its encoder/decoder VSNs accept
    projected stim slots instead of raw 4900 columns.

    Strategy:
      1. Build a ChannelProjector and attach it as an nn.Module.
      2. Build fresh VariableSelectionNetwork (VSN) instances per
         encoder/decoder that use the projected variable names.
      3. Update hparams.time_varying_reals_encoder / _decoder so the
         parent's forward() iterates over the new names (this is the
         half I forgot the first time — without this update, the parent
         tries to look up "mov_L1_0000" in an input_vectors dict that
         only has "proj_L1_0...63" keys, and crashes with KeyError).
      4. Wrap forward() so x_cont's raw stim slice is routed through the
         projector before becoming input_vectors entries.

    Modifies tft_module in place. Returns nothing.
    """
    import types
    import torch
    from torch import nn
    from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import (
        VariableSelectionNetwork,
    )

    # ── 1. Build and attach the projector ───────────────────────────────
    projector = ChannelProjector(
        groups=channel_groups.groups,
        n_scalars=len(channel_groups.scalars),
        proj_dim=proj_dim,
    )
    tft_module.channel_projector = projector  # nn.Module registered
    tft_module._cp_raw_known_reals = list(raw_known_reals)
    tft_module._cp_ordered_raw = list(channel_groups.ordered_raw)
    tft_module._cp_ordered_proj = list(channel_groups.ordered_proj)

    # Index of each raw known-real in the parent's `x_reals` tensor order.
    # x_reals is set up by from_dataset() and is the canonical column
    # order TFT uses to index into x_cont.
    x_reals = list(tft_module.hparams.x_reals)
    try:
        raw_indices = [x_reals.index(c) for c in channel_groups.ordered_raw]
    except ValueError as e:
        missing = [c for c in channel_groups.ordered_raw if c not in x_reals]
        raise RuntimeError(
            f"Channel-projection setup error: {len(missing)} raw known-real "
            f"columns are not in TFT's x_reals. First few: {missing[:5]}. "
            "This usually means the bundle's known_dynamic list does not "
            "match the columns the dataset was built with."
        ) from e
    tft_module._cp_raw_indices_tensor = torch.tensor(
        raw_indices, dtype=torch.long
    )

    # ── 2. Rebuild encoder/decoder VSNs around projected slots ──────────
    hidden_cont = int(tft_module.hparams.hidden_continuous_size)
    hidden_cont_sizes_user = dict(tft_module.hparams.hidden_continuous_sizes or {})

    def _size_for(name: str) -> int:
        return int(hidden_cont_sizes_user.get(name, hidden_cont))

    raw_known_set = set(raw_known_reals)

    def _rewrite_var_list(names: list[str]) -> list[str]:
        """Replace any raw stim col with the projected slot names, in order.
        Other names (time_idx, observed_reals, etc.) pass through unchanged."""
        out: list[str] = []
        seen_stim = False
        for n in names:
            if n in raw_known_set:
                if not seen_stim:
                    out.extend(channel_groups.ordered_proj)
                    seen_stim = True
                # raw stim cols are dropped from the VSN input list
            else:
                out.append(n)
        return out

    encoder_inputs_old = list(tft_module.encoder_variable_selection.input_sizes.keys())
    decoder_inputs_old = list(tft_module.decoder_variable_selection.input_sizes.keys())
    encoder_inputs_new = _rewrite_var_list(encoder_inputs_old)
    decoder_inputs_new = _rewrite_var_list(decoder_inputs_old)

    def _new_vsn(input_names: list[str], context_size: int) -> VariableSelectionNetwork:
        input_sizes = {n: _size_for(n) for n in input_names}
        prescalers = nn.ModuleDict(
            {n: nn.Linear(1, _size_for(n)) for n in input_names}
        )
        return VariableSelectionNetwork(
            input_sizes=input_sizes,
            hidden_size=int(tft_module.hparams.hidden_size),
            input_embedding_flags={},  # all reals here, no embeddings
            dropout=float(tft_module.hparams.dropout),
            context_size=context_size,
            prescalers=prescalers,
        )

    ctx = int(tft_module.hparams.hidden_size)
    tft_module.encoder_variable_selection = _new_vsn(encoder_inputs_new, ctx)
    tft_module.decoder_variable_selection = _new_vsn(decoder_inputs_new, ctx)

    if getattr(tft_module.hparams, "share_single_variable_networks", False):
        log.warning(
            "share_single_variable_networks=True is incompatible with "
            "channel projection in this implementation; the decoder VSN "
            "will not share networks with the encoder VSN."
        )

    # ── 3. Update hparams name lists so the parent's forward() iterates
    #      over projected names (NOT raw names) when consuming input_vectors.
    #
    # In pytorch-forecasting's TFT, the parent's forward() does roughly:
    #   for name in self.encoder_variables:
    #       enc_emb[name] = input_vectors[name][...]
    # `self.encoder_variables` is a property derived from
    # `hparams.time_varying_categoricals_encoder` +
    # `hparams.time_varying_reals_encoder`. Likewise for decoder.
    # The categoricals are untouched; we only rewrite the reals lists.
    #
    # We rewrite by replacing all raw stim names with the projected
    # slot names, preserving everything else (time_idx, observed reals,
    # target columns) in place.
    def _rewrite_reals_list_keep_others(names_attr: str) -> None:
        old = list(getattr(tft_module.hparams, names_attr, []) or [])
        new = _rewrite_var_list(old)
        setattr(tft_module.hparams, names_attr, new)

    _rewrite_reals_list_keep_others("time_varying_reals_encoder")
    _rewrite_reals_list_keep_others("time_varying_reals_decoder")

    # ── 4. Wrap forward() to route raw stim through the projector ──────
    # Layout of x_cont last dim:
    #   - `x_reals` is the canonical column order pytorch-forecasting uses.
    #   - raw_indices = positions of the raw stim cols within x_reals.
    # We project those columns and synthesize a new x_cont where:
    #   - non-stim cols keep their position and name (preserves the
    #     existing input_vectors construction in the parent for time_idx,
    #     observed reals, target cols, etc.)
    #   - the raw stim block at the end is REPLACED by the projected slots,
    #     and we extend x_reals with the projected names in that same order.

    original_forward = tft_module.forward

    # Pre-compute on CPU (once) the non-stim column names in their original
    # x_reals order, and the new x_reals list. The new layout is:
    #   [non_stim_cols (in original order)] + [projected_slot_names]
    n_x_reals = len(tft_module.hparams.x_reals)
    non_stim_mask_np = np.ones(n_x_reals, dtype=bool)
    non_stim_mask_np[np.array(raw_indices, dtype=np.int64)] = False
    non_stim_names = [
        n for i, n in enumerate(tft_module.hparams.x_reals) if non_stim_mask_np[i]
    ]
    tft_module._cp_non_stim_names = non_stim_names
    tft_module._cp_new_x_reals = non_stim_names + tft_module._cp_ordered_proj
    tft_module._cp_orig_x_reals = list(tft_module.hparams.x_reals)
    tft_module._cp_non_stim_mask = torch.tensor(
        non_stim_mask_np, dtype=torch.bool
    )

    def patched_forward(x):
        import torch as _torch

        encoder_cont = x["encoder_cont"]
        decoder_cont = x["decoder_cont"]
        x_cont = _torch.cat([encoder_cont, decoder_cont], dim=1)

        # Project the raw stim slice
        raw_idx = tft_module._cp_raw_indices_tensor.to(x_cont.device)
        raw_slice = x_cont.index_select(dim=-1, index=raw_idx)
        projected = tft_module.channel_projector(raw_slice)  # (..., n_proj)

        # Build new x_cont: non-stim columns (in original order) then
        # projected slots.
        non_stim_mask = tft_module._cp_non_stim_mask.to(x_cont.device)
        non_stim_slice = x_cont[..., non_stim_mask]
        new_cont = _torch.cat([non_stim_slice, projected], dim=-1)

        # Split back into encoder/decoder along time
        enc_len = encoder_cont.shape[1]
        new_encoder_cont = new_cont[:, :enc_len, :]
        new_decoder_cont = new_cont[:, enc_len:, :]

        # Temporarily swap hparams.x_reals so the parent's name-indexed
        # loop builds input_vectors with the projected names. This must
        # match the column order in new_cont exactly.
        saved_x_reals = tft_module.hparams.x_reals
        tft_module.hparams.x_reals = tft_module._cp_new_x_reals

        x_patched = dict(x)
        x_patched["encoder_cont"] = new_encoder_cont
        x_patched["decoder_cont"] = new_decoder_cont

        try:
            out = original_forward(x_patched)
        finally:
            tft_module.hparams.x_reals = saved_x_reals
        return out

    tft_module.forward = types.MethodType(
        lambda self, x: patched_forward(x), tft_module
    )

    n_proj_params = sum(p.numel() for p in projector.parameters() if p.requires_grad)
    log.info(
        f"CHANNEL_PROJ attached | channels={len(channel_groups.groups)} "
        f"scalars={len(channel_groups.scalars)} "
        f"raw_known={len(raw_known_reals)} -> projected_known={projector.n_projected_cols} "
        f"proj_dim_target={proj_dim} proj_params={n_proj_params:,}"
    )
    log.info(
        f"CHANNEL_PROJ hparams updated | "
        f"time_varying_reals_encoder: {len(encoder_inputs_old)} -> "
        f"{len(tft_module.hparams.time_varying_reals_encoder)} names | "
        f"time_varying_reals_decoder: {len(decoder_inputs_old)} -> "
        f"{len(tft_module.hparams.time_varying_reals_decoder)} names"
    )


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

    # ── channel projection (Algonauts-style) ─────────────────────────────
    channel_proj: bool = False
    channel_proj_dim: int = 64
    channel_pattern: str = DEFAULT_CHANNEL_PATTERN

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

        # Plan channel projection if enabled. We do this BEFORE building
        # TimeSeriesDataSet because the dataset is unchanged — it still
        # carries the raw 4900 known_dynamic columns. The projection
        # lives inside the model.
        self._channel_groups_ = None
        if self.channel_proj and known_reals:
            groups = parse_channels(
                columns=known_reals,
                pattern=self.channel_pattern,
                proj_dim=self.channel_proj_dim,
            )
            self._channel_groups_ = groups
            log.info(
                f"channel projection enabled: "
                f"{len(known_reals)} raw known_reals -> "
                f"{len(groups.ordered_proj)} projected slots"
            )
        elif self.channel_proj and not known_reals:
            log.warning(
                "channel_proj=True but known_reals is empty — "
                "projection has nothing to do; running plain TFT."
            )

        single_target = n_targets == 1
        target_arg = target_cols[0] if single_target else target_cols

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
        proj_str = (
            f" channel_proj(dim<={self.channel_proj_dim})"
            if self._channel_groups_ is not None else ""
        )
        log.info(
            f"Fitting {mode_str} TFT{proj_str} | "
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

        # Channel projection: rewire VSNs and patch forward().
        if self._channel_groups_ is not None:
            _attach_channel_projector(
                tft_module=tft,
                channel_groups=self._channel_groups_,
                raw_known_reals=known_reals,  # NOT including time_col
                proj_dim=self.channel_proj_dim,
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

    def get_channel_groups(self) -> Optional[ChannelGroups]:
        """Diagnostic accessor: the channel grouping used during fit."""
        return self._channel_groups_
