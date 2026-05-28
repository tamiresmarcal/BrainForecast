"""
predictors/channel_projection.py

Per-channel learned projection for high-dimensional stimulus features.

Motivation: the TFT's Variable Selection Network (VSN) builds one Gated
Residual Network (GRN) per input feature. With 4900 stim columns this is
both compute-heavy (linear in #features per epoch) and statistically
unstable (softmax over 4900 logits flattens; per-feature attention
weights become noisy under subject-out CV). Algonauts 2025 winners
(TRIBE, VIBE, MedARC) all sidestep this by projecting each modality /
channel into a small latent (D=192-1024) before any attention or VSN.

This module implements the same idea per channel, with two pieces:

  parse_channels(columns, pattern)
      Group a flat list of column names into per-channel buckets by
      regex on the column name. Default pattern recognises
      ``mov_<channel>_<integer>`` as a projectable channel-dim slot;
      anything else (e.g. ``mov_onset``, ``mov_entropy_L1``,
      ``mov_divergence_spec``) is kept as a scalar pass-through. The
      function returns:
        - groups:   dict[channel_name -> list[column_name]]   (>=2 dims)
        - scalars:  list[column_name]                          (1-d kept as-is)

  ChannelProjector
      A torch ``nn.Module`` containing one ``nn.Linear`` per group.
      ``raw_dim -> min(proj_dim, raw_dim)`` per channel; if the raw_dim
      is already <= proj_dim the projection is the identity (no params).
      Forward takes a tensor of shape (..., n_raw_cols) where columns are
      ordered as ``[group1_cols..., group2_cols..., ..., scalar_cols...]``
      and returns a tensor of shape (..., n_proj_cols) in the matching
      order. Used both by the TFT subclass (joint training) and by
      offline diagnostics.

Naming convention for projected slots: ``proj_<channel>_<i>`` for i in
[0, projected_dim). Scalars keep their original name. This keeps slot
ordering deterministic and easy to grep in EXPERIMENT_CONFIG logs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

log = logging.getLogger(__name__)


# Default regex: matches "mov_<channel>_<integer>" where <integer> is one or
# more digits at the very end of the name. The channel name may itself
# contain underscores (e.g. mov_audio_mel_0 -> channel=audio_mel, dim=0).
#
# Columns that do NOT match this pattern are kept as scalars. That covers:
#   - mov_onset            (no trailing integer)
#   - mov_entropy_L1       (trailing token is L1, not digits)
#   - mov_divergence_spec  (trailing token is non-numeric)
#   - any non-mov_ column
DEFAULT_CHANNEL_PATTERN = r"^mov_(?P<channel>.+)_(?P<dim>\d+)$"


@dataclass
class ChannelGroups:
    """
    Result of grouping a column list by channel.

    Attributes
    ----------
    groups : dict[str, list[str]]
        channel name -> ordered list of raw column names (>= 2 dims). The
        order in the list is the column order in the source dataframe,
        which is what determines the index order fed to the per-channel
        Linear. Channels with only 1 matching column are demoted to
        scalars (no point projecting a 1-d input).
    scalars : list[str]
        Columns kept as-is (1-d channels, surprise/uncertainty cols,
        anything not matching the channel pattern). These pass through
        the projector untouched.
    ordered_raw : list[str]
        Concatenation [group1_cols, group2_cols, ..., scalars]. This is
        the canonical column order the ChannelProjector expects on input.
    ordered_proj : list[str]
        Names of the projected output slots, in matching order.
        Multi-dim channels become proj_<ch>_<i>; scalars keep their name.
    """

    groups: dict[str, list[str]]
    scalars: list[str]
    ordered_raw: list[str]
    ordered_proj: list[str]


def parse_channels(
    columns: list[str],
    pattern: str = DEFAULT_CHANNEL_PATTERN,
    proj_dim: int = 64,
) -> ChannelGroups:
    """
    Group columns by channel using a regex pattern.

    Parameters
    ----------
    columns : list[str]
        The full ordered list of column names to group (typically the
        ``known_dynamic`` slice of bundle.feature_cols).
    pattern : str
        Regex with at least one named group ``channel``. Default matches
        ``mov_<channel>_<integer>``. Columns not matching the pattern OR
        whose channel only has one matching column become scalars.
    proj_dim : int
        Per-channel target dim. Used only here to compute the projected
        slot names; the actual projection size is
        ``min(proj_dim, raw_dim)``. Pass the same value to
        ChannelProjector.

    Returns
    -------
    ChannelGroups
    """
    rx = re.compile(pattern)
    raw_groups: dict[str, list[str]] = {}
    scalars: list[str] = []

    for col in columns:
        m = rx.match(col)
        if m is None:
            scalars.append(col)
            continue
        try:
            ch = m.group("channel")
        except IndexError:
            scalars.append(col)
            continue
        raw_groups.setdefault(ch, []).append(col)

    # Demote single-column channels to scalars: projecting a 1-d input
    # is a waste of a parameter matrix.
    groups: dict[str, list[str]] = {}
    for ch, cols in raw_groups.items():
        if len(cols) >= 2:
            groups[ch] = cols
        else:
            scalars.extend(cols)

    # Canonical ordering: groups in insertion order, scalars at the end.
    ordered_raw: list[str] = []
    ordered_proj: list[str] = []
    for ch, cols in groups.items():
        ordered_raw.extend(cols)
        proj_d = min(proj_dim, len(cols))
        for i in range(proj_d):
            ordered_proj.append(f"proj_{ch}_{i}")
    ordered_raw.extend(scalars)
    ordered_proj.extend(scalars)  # scalars pass through with their original names

    log.info(
        f"parse_channels: {len(columns)} raw cols -> "
        f"{len(groups)} multi-dim channels + {len(scalars)} scalars "
        f"-> {len(ordered_proj)} projected slots "
        f"(target proj_dim={proj_dim})"
    )
    if groups:
        details = ", ".join(
            f"{ch}={len(cols)}->({min(proj_dim, len(cols))})"
            for ch, cols in groups.items()
        )
        log.info(f"  channels: {details}")

    return ChannelGroups(
        groups=groups,
        scalars=scalars,
        ordered_raw=ordered_raw,
        ordered_proj=ordered_proj,
    )


class ChannelProjector(nn.Module):
    """
    Per-channel learned projection.

    Holds one ``nn.Linear(raw_dim, proj_dim)`` per channel where
    ``proj_dim = min(target_proj_dim, raw_dim)``. Channels whose raw_dim
    is already <= target_proj_dim use an identity (no parameters added).
    Scalar pass-through columns also use identity.

    Forward
    -------
    Input  : Tensor of shape (..., n_raw_cols) where the last-dim
             ordering matches ``ChannelGroups.ordered_raw``.
    Output : Tensor of shape (..., n_proj_cols), last-dim ordering
             matches ``ChannelGroups.ordered_proj``.

    Notes
    -----
    A LayerNorm is applied per channel BEFORE projection. This matches
    TRIBE/VIBE's pattern of normalizing each modality before fusion and
    is critical for stability when channels have very different scales
    (CNN activations vs mel spectrogram vs categorical one-hots).
    """

    def __init__(
        self,
        groups: dict[str, list[str]],
        n_scalars: int,
        proj_dim: int = 64,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self._channel_order = list(groups.keys())  # deterministic order
        self._channel_raw_sizes = {ch: len(cols) for ch, cols in groups.items()}
        self._channel_proj_sizes = {
            ch: min(proj_dim, n_raw) for ch, n_raw in self._channel_raw_sizes.items()
        }
        self.n_scalars = n_scalars

        self.norms = nn.ModuleDict()
        self.projs = nn.ModuleDict()
        for ch in self._channel_order:
            raw = self._channel_raw_sizes[ch]
            proj = self._channel_proj_sizes[ch]
            self.norms[ch] = nn.LayerNorm(raw)
            if proj < raw:
                self.projs[ch] = nn.Linear(raw, proj, bias=True)
            else:
                # raw <= target_proj_dim: identity, no params
                self.projs[ch] = nn.Identity()

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(
            f"ChannelProjector built: {len(self._channel_order)} channels, "
            f"{n_scalars} scalars passing through, "
            f"{n_params:,} trainable params"
        )

    @property
    def channel_proj_sizes(self) -> dict[str, int]:
        return dict(self._channel_proj_sizes)

    @property
    def channel_order(self) -> list[str]:
        return list(self._channel_order)

    @property
    def n_projected_cols(self) -> int:
        return sum(self._channel_proj_sizes.values()) + self.n_scalars

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project the channel slice of x and concatenate the scalar slice.

        x : (..., n_raw_cols)  ordering = ordered_raw
        out: (..., n_proj_cols) ordering = ordered_proj
        """
        out_pieces: list[torch.Tensor] = []
        cursor = 0
        for ch in self._channel_order:
            n_raw = self._channel_raw_sizes[ch]
            slc = x[..., cursor : cursor + n_raw]
            slc = self.norms[ch](slc)
            slc = self.projs[ch](slc)
            out_pieces.append(slc)
            cursor += n_raw
        if self.n_scalars > 0:
            scalars = x[..., cursor : cursor + self.n_scalars]
            out_pieces.append(scalars)
            cursor += self.n_scalars
        return torch.cat(out_pieces, dim=-1)
