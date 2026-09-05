"""Small legal-move policy head over a frozen sparse chess representation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from distill.features import FEATURE_DIM, METADATA_DIM, PADDING_INDEX


class SparsePolicyNetwork(nn.Module):
    """Score legal moves from one board context and factorized move embeddings."""

    def __init__(self, accumulator: int, policy_width: int = 32) -> None:
        super().__init__()
        self.accumulator = accumulator
        self.policy_width = policy_width
        self.feature = nn.Embedding(
            FEATURE_DIM + 1,
            accumulator,
            padding_idx=PADDING_INDEX,
        )
        self.feature_bias = nn.Parameter(torch.zeros(accumulator))
        self.context = nn.Linear(2 * accumulator + METADATA_DIM, policy_width)
        self.from_embedding = nn.Embedding(64, policy_width)
        self.to_embedding = nn.Embedding(64, policy_width)
        self.promotion_embedding = nn.Embedding(7, policy_width)
        self.move_bias = nn.Linear(policy_width, 1)

    def forward(
        self,
        features: torch.Tensor,
        turns: torch.Tensor,
        metadata: torch.Tensor,
        move_from: torch.Tensor,
        move_to: torch.Tensor,
        promotion: torch.Tensor,
    ) -> torch.Tensor:
        accumulator = self.feature(features).sum(dim=-2) + self.feature_bias
        white = accumulator[:, 0]
        black = accumulator[:, 1]
        mover = torch.where(turns[:, None], white, black)
        opponent = torch.where(turns[:, None], black, white)
        reversed_metadata = torch.cat((metadata[:, 2:4], metadata[:, :2], metadata[:, 4:]), dim=-1)
        oriented_metadata = torch.where(turns[:, None], metadata, reversed_metadata)
        context = torch.relu(self.context(torch.cat((mover, opponent, oriented_metadata), dim=-1)))
        moves = (
            self.from_embedding(move_from)
            + self.to_embedding(move_to)
            + self.promotion_embedding(promotion)
        )
        compatibility = torch.sum(moves * context[:, None, :], dim=-1) / math.sqrt(
            self.policy_width
        )
        return compatibility + self.move_bias(moves).squeeze(-1)


def initialize_sparse_policy(model: SparsePolicyNetwork, source: Path) -> str:
    """Load and freeze a quantized evaluator's learned sparse representation."""
    with np.load(source, allow_pickle=False) as archive:
        accumulator = int(archive["accumulator"])
        if accumulator != model.accumulator:
            raise ValueError(
                f"accumulator mismatch: checkpoint={accumulator}, policy={model.accumulator}"
            )
        feature = archive["feature_q"].astype(np.float32) * np.float32(archive["feature_scale"])
        with torch.no_grad():
            model.feature.weight[:FEATURE_DIM].copy_(torch.from_numpy(feature))
            model.feature.weight[PADDING_INDEX].zero_()
            model.feature_bias.copy_(torch.from_numpy(archive["feature_bias"]))
    model.feature.weight.requires_grad_(False)
    model.feature_bias.requires_grad_(False)
    import hashlib

    return hashlib.sha256(source.read_bytes()).hexdigest()


def export_policy(
    model: SparsePolicyNetwork,
    base: Path,
    destination: Path,
    metadata: dict[str, object],
) -> dict[str, object]:
    """Append the trained policy head to a submission-ready value artifact."""
    model = model.to("cpu").eval()
    with np.load(base, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays.update(
        {
            "policy_format_version": np.array(1, dtype=np.int32),
            "policy_width": np.array(model.policy_width, dtype=np.int32),
            "policy_context_weight": model.context.weight.detach().numpy().astype(np.float32),
            "policy_context_bias": model.context.bias.detach().numpy().astype(np.float32),
            "policy_from_embedding": model.from_embedding.weight.detach()
            .numpy()
            .astype(np.float32),
            "policy_to_embedding": model.to_embedding.weight.detach().numpy().astype(np.float32),
            "policy_promotion_embedding": model.promotion_embedding.weight.detach()
            .numpy()
            .astype(np.float32),
            "policy_move_bias_weight": model.move_bias.weight.detach().numpy().astype(np.float32),
            "policy_move_bias_bias": model.move_bias.bias.detach().numpy().astype(np.float32),
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    report = {
        "bytes": destination.stat().st_size,
        "accumulator": model.accumulator,
        "policy_width": model.policy_width,
        **metadata,
    }
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report
