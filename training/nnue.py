"""Sparse king-conditioned student network and quantized export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from distill.features import FEATURE_DIM, PADDING_INDEX

FORMAT_VERSION = 1


class SparseValueNetwork(nn.Module):
    """A compact HalfKP-style scalar evaluator."""

    def __init__(self, accumulator: int = 128, hidden: int = 64, bottleneck: int = 32) -> None:
        super().__init__()
        self.accumulator = accumulator
        self.hidden = hidden
        self.bottleneck = bottleneck
        self.feature = nn.Embedding(
            FEATURE_DIM + 1, accumulator, padding_idx=PADDING_INDEX
        )
        self.feature_bias = nn.Parameter(torch.zeros(accumulator))
        self.hidden_one = nn.Linear(2 * accumulator, hidden)
        self.hidden_two = nn.Linear(hidden, bottleneck)
        self.output = nn.Linear(bottleneck, 1)
        nn.init.normal_(self.feature.weight, std=0.01)
        with torch.no_grad():
            self.feature.weight[PADDING_INDEX].zero_()

    def _head(self, mover: torch.Tensor, opponent: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.hidden_one(torch.cat((mover, opponent), dim=-1)))
        hidden = torch.relu(self.hidden_two(hidden))
        return torch.tanh(self.output(hidden)).squeeze(-1)

    def forward_with_flipped_turns(
        self, features: torch.Tensor, turns: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate actual and turn-flipped mover order with one sparse accumulation."""
        accumulator = self.feature(features).sum(dim=-2) + self.feature_bias
        white = accumulator[..., 0, :]
        black = accumulator[..., 1, :]
        white_to_move = self._head(white, black)
        black_to_move = self._head(black, white)
        turn = turns.bool()
        prediction = torch.where(turn, white_to_move, black_to_move)
        flipped = torch.where(turn, black_to_move, white_to_move)
        return prediction, flipped

    def forward(self, features: torch.Tensor, turns: torch.Tensor) -> torch.Tensor:
        """Evaluate ``(..., 2, 32)`` sparse indices from the side to move."""
        prediction, _ = self.forward_with_flipped_turns(features, turns)
        return prediction


def _quantize_int8(weights: np.ndarray) -> tuple[np.ndarray, np.float32]:
    maximum = float(np.max(np.abs(weights)))
    scale = np.float32(max(maximum / 127.0, 1e-8))
    quantized = np.rint(weights / scale).clip(-127, 127).astype(np.int8)
    return quantized, scale


def export_quantized(
    model: SparseValueNetwork,
    destination: Path,
    *,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Export int8 sparse weights plus small float32 dense layers."""
    model = model.to("cpu").eval()
    feature = model.feature.weight[:FEATURE_DIM].detach().numpy().astype(np.float32)
    feature_q, feature_scale = _quantize_int8(feature)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        format_version=np.array(FORMAT_VERSION, dtype=np.int32),
        feature_dim=np.array(FEATURE_DIM, dtype=np.int32),
        accumulator=np.array(model.accumulator, dtype=np.int32),
        feature_q=feature_q,
        feature_scale=np.array(feature_scale, dtype=np.float32),
        feature_bias=model.feature_bias.detach().numpy().astype(np.float32),
        hidden_one_weight=model.hidden_one.weight.detach().numpy().astype(np.float32),
        hidden_one_bias=model.hidden_one.bias.detach().numpy().astype(np.float32),
        hidden_two_weight=model.hidden_two.weight.detach().numpy().astype(np.float32),
        hidden_two_bias=model.hidden_two.bias.detach().numpy().astype(np.float32),
        output_weight=model.output.weight.detach().numpy().astype(np.float32),
        output_bias=model.output.bias.detach().numpy().astype(np.float32),
    )
    report: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "feature_dim": FEATURE_DIM,
        "accumulator": model.accumulator,
        "hidden": model.hidden,
        "bottleneck": model.bottleneck,
        "bytes": destination.stat().st_size,
        "feature_scale": float(feature_scale),
        **(metadata or {}),
    }
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report
