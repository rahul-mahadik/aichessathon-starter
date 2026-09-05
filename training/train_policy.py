"""Train a selective legal-move policy head on MultiPV labels and negatives."""

from __future__ import annotations

import argparse
import json
import math
import random
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
import torch.nn.functional as functional

from training.policy import (
    SparsePolicyNetwork,
    export_policy,
    initialize_sparse_policy,
)
from training.train_distilled import resolve_shards


class PolicyBatch(TypedDict):
    features: torch.Tensor
    turns: torch.Tensor
    metadata: torch.Tensor
    move_from: torch.Tensor
    move_to: torch.Tensor
    promotion: torch.Tensor
    scores: torch.Tensor
    mask: torch.Tensor


def iterate_policy_batches(
    shards: list[Path],
    *,
    batch_size: int,
    device: str,
    seed: int,
    epoch: int,
    validation: bool,
) -> Iterator[PolicyBatch]:
    shard_order = list(shards)
    random.Random(seed + epoch).shuffle(shard_order)
    for shard in shard_order:
        with np.load(shard, allow_pickle=False) as archive:
            required = {
                "features",
                "turns",
                "metadata",
                "policy_from",
                "policy_to",
                "policy_promotion",
                "policy_scores",
                "policy_mask",
            }
            if not required.issubset(archive.files):
                raise ValueError(f"{shard} is not a policy/metadata dataset")
            arrays = {name: archive[name] for name in required}
        count = len(arrays["features"])
        identity = f"{shard.parent.name}/{shard.name}"
        shard_seed = seed + zlib.crc32(identity.encode())
        partition = np.random.default_rng(shard_seed).permutation(count)
        validation_size = max(1, count // 10)
        indices = partition[:validation_size] if validation else partition[validation_size:]
        if not validation:
            np.random.default_rng(shard_seed + epoch).shuffle(indices)
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            yield {
                "features": torch.as_tensor(arrays["features"][selected, 0], device=device).long(),
                "turns": torch.as_tensor(arrays["turns"][selected, 0], device=device).bool(),
                "metadata": torch.as_tensor(arrays["metadata"][selected, 0], device=device).float(),
                "move_from": torch.as_tensor(arrays["policy_from"][selected], device=device).long(),
                "move_to": torch.as_tensor(arrays["policy_to"][selected], device=device).long(),
                "promotion": torch.as_tensor(
                    arrays["policy_promotion"][selected], device=device
                ).long(),
                "scores": torch.as_tensor(arrays["policy_scores"][selected], device=device).float(),
                "mask": torch.as_tensor(arrays["policy_mask"][selected], device=device).bool(),
            }


def policy_loss(
    logits: torch.Tensor,
    batch: PolicyBatch,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["mask"]
    masked_logits = logits.masked_fill(~mask, -1e4)
    teacher_logits = (batch["scores"] / temperature).masked_fill(~mask, -1e4)
    targets = functional.softmax(teacher_logits, dim=1)
    loss = -torch.mean(torch.sum(targets * functional.log_softmax(masked_logits, dim=1), dim=1))
    accuracy = torch.mean((masked_logits.argmax(dim=1) == 0).float())
    top_three = torch.topk(masked_logits, k=3, dim=1).indices
    recall = torch.mean(torch.any(top_three == 0, dim=1).float())
    return loss, {
        "policy_loss": float(loss.detach()),
        "top_move_accuracy": float(accuracy.detach()),
        "top_three_recall": float(recall.detach()),
    }


def evaluate(
    model: SparsePolicyNetwork,
    batches: Iterator[PolicyBatch],
    temperature: float,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    count = 0
    model.eval()
    with torch.inference_mode():
        for batch in batches:
            _, metrics = policy_loss(
                model(
                    batch["features"],
                    batch["turns"],
                    batch["metadata"],
                    batch["move_from"],
                    batch["move_to"],
                    batch["promotion"],
                ),
                batch,
                temperature,
            )
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + value
            count += 1
    return {name: value / max(count, 1) for name, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--policy-width", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=23)
    arguments = parser.parse_args()
    if (
        min(
            arguments.epochs,
            arguments.batch_size,
            arguments.learning_rate,
            arguments.policy_width,
            arguments.temperature,
        )
        <= 0
    ):
        parser.error("training counts, rate, width, and temperature must be positive")
    device = (
        "cuda"
        if arguments.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if arguments.device == "auto"
        else arguments.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    shards = resolve_shards(arguments.data)
    with np.load(arguments.base, allow_pickle=False) as archive:
        accumulator = int(archive["accumulator"])
    model = SparsePolicyNetwork(accumulator, arguments.policy_width).to(device)
    base_sha256 = initialize_sparse_policy(model, arguments.base)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=arguments.learning_rate,
    )
    best_loss = math.inf
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(arguments.epochs):
        model.train()
        train_loss = 0.0
        batches = 0
        for batch in iterate_policy_batches(
            shards,
            batch_size=arguments.batch_size,
            device=device,
            seed=arguments.seed,
            epoch=epoch,
            validation=False,
        ):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = policy_loss(
                model(
                    batch["features"],
                    batch["turns"],
                    batch["metadata"],
                    batch["move_from"],
                    batch["move_to"],
                    batch["promotion"],
                ),
                batch,
                arguments.temperature,
            )
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach())
            batches += 1
        metrics = evaluate(
            model,
            iterate_policy_batches(
                shards,
                batch_size=arguments.batch_size,
                device=device,
                seed=arguments.seed,
                epoch=epoch,
                validation=True,
            ),
            arguments.temperature,
        )
        if metrics["policy_loss"] < best_loss:
            best_loss = metrics["policy_loss"]
            best_epoch = epoch + 1
            best_metrics = dict(metrics)
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss / max(batches, 1),
                    **metrics,
                }
            )
        )
    assert best_state is not None
    model.load_state_dict(best_state)
    report = export_policy(
        model,
        arguments.base,
        arguments.output,
        {
            "base": str(arguments.base),
            "base_sha256": base_sha256,
            "device": device,
            "epochs": arguments.epochs,
            "batch_size": arguments.batch_size,
            "learning_rate": arguments.learning_rate,
            "temperature": arguments.temperature,
            "seed": arguments.seed,
            "best_epoch": best_epoch,
            **best_metrics,
        },
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
