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


def initialize_from_quantized(
    model: SparseValueNetwork,
    source: Path,
    *,
    freeze_feature: bool = False,
) -> list[str]:
    """Initialize compatible layers from an exported quantized evaluator.

    A narrower student can reuse a teacher's sparse representation even when its
    dense head has different shapes. Dense layers are copied only when their
    shapes match exactly.
    """
    copied = ["feature", "feature_bias"]
    with np.load(source, allow_pickle=False) as archive:
        feature_dim = int(archive["feature_dim"])
        accumulator = int(archive["accumulator"])
        if feature_dim != FEATURE_DIM:
            raise ValueError(
                f"feature dimension mismatch: checkpoint={feature_dim}, model={FEATURE_DIM}"
            )
        if accumulator != model.accumulator:
            raise ValueError(
                f"accumulator mismatch: checkpoint={accumulator}, model={model.accumulator}"
            )

        feature = archive["feature_q"].astype(np.float32) * np.float32(
            archive["feature_scale"]
        )
        with torch.no_grad():
            model.feature.weight[:FEATURE_DIM].copy_(torch.from_numpy(feature))
            model.feature.weight[PADDING_INDEX].zero_()
            model.feature_bias.copy_(torch.from_numpy(archive["feature_bias"]))

            dense_layers = (
                ("hidden_one", model.hidden_one),
                ("hidden_two", model.hidden_two),
                ("output", model.output),
            )
            for name, layer in dense_layers:
                weight = archive[f"{name}_weight"]
                bias = archive[f"{name}_bias"]
                if tuple(weight.shape) != tuple(layer.weight.shape) or tuple(
                    bias.shape
                ) != tuple(layer.bias.shape):
                    continue
                layer.weight.copy_(torch.from_numpy(weight))
                layer.bias.copy_(torch.from_numpy(bias))
                copied.append(name)

    if freeze_feature:
        model.feature.weight.requires_grad_(False)
        model.feature_bias.requires_grad_(False)
    return copied


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
