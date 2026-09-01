"""Train the sparse evaluator with root value and candidate-ranking supervision."""

from __future__ import annotations

import argparse
import json
import math
import random
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

import chess
import numpy as np
import torch
import torch.nn.functional as functional

from distill.features import (
    GROUP_SIZE,
    MAX_CANDIDATES,
    MAX_PIECES,
    PADDING_INDEX,
    encode_board,
)
from training.nnue import SparseValueNetwork, export_quantized


class Batch(TypedDict):
    features: torch.Tensor
    turns: torch.Tensor
    targets: torch.Tensor
    value_mask: torch.Tensor
    candidate_scores: torch.Tensor
    candidate_mask: torch.Tensor


def resolve_shards(paths: list[Path]) -> list[Path]:
    shards: list[Path] = []
    for path in paths:
        if path.is_dir():
            shards.extend(sorted(path.glob("part-*.npz")))
        else:
            shards.append(path)
    missing = [path for path in shards if not path.exists()]
    if missing or not shards:
        raise SystemExit(f"no training shards found: {missing or paths}")
    return shards


def _tensor_batch(arrays: dict[str, np.ndarray], indices: np.ndarray, device: str) -> Batch:
    return {
        "features": torch.as_tensor(arrays["features"][indices], device=device).long(),
        "turns": torch.as_tensor(arrays["turns"][indices], device=device).bool(),
        "targets": torch.as_tensor(arrays["targets"][indices], device=device).float(),
        "value_mask": torch.as_tensor(arrays["value_mask"][indices], device=device).bool(),
        "candidate_scores": torch.as_tensor(
            arrays["candidate_scores"][indices], device=device
        ).float(),
        "candidate_mask": torch.as_tensor(
            arrays["candidate_mask"][indices], device=device
        ).bool(),
    }


def iterate_batches(
    shards: list[Path],
    *,
    batch_size: int,
    device: str,
    seed: int,
    epoch: int,
    validation: bool,
) -> Iterator[Batch]:
    shard_order = list(shards)
    random.Random(seed + epoch).shuffle(shard_order)
    for shard in shard_order:
        with np.load(shard, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        count = len(arrays["features"])
        shard_identity = f"{shard.parent.name}/{shard.name}"
        shard_seed = seed + zlib.crc32(shard_identity.encode())
        partition = np.random.default_rng(shard_seed).permutation(count)
        validation_size = max(1, count // 10) if count > 1 else 1
        indices = partition[:validation_size] if validation else partition[validation_size:]
        if not validation:
            np.random.default_rng(shard_seed + epoch).shuffle(indices)
        for start in range(0, len(indices), batch_size):
            yield _tensor_batch(arrays, indices[start : start + batch_size], device)


def _material_value(board: chess.Board) -> float:
    values = {1: 100, 2: 320, 3: 330, 4: 500, 5: 900, 6: 0}
    white = sum(values[piece.piece_type] for piece in board.piece_map().values() if piece.color)
    black = sum(
        values[piece.piece_type] for piece in board.piece_map().values() if not piece.color
    )
    score = white - black
    if board.turn == chess.BLACK:
        score = -score
    return math.tanh(score / 600.0)


def synthetic_shard(path: Path, records: int, seed: int) -> Path:
    """Create a deterministic tiny dataset that exercises root and ranking losses."""
    rng = random.Random(seed)
    features = np.full(
        (records, GROUP_SIZE, 2, MAX_PIECES), PADDING_INDEX, dtype=np.int32
    )
    turns = np.zeros((records, GROUP_SIZE), dtype=np.bool_)
    targets = np.zeros((records, GROUP_SIZE), dtype=np.float32)
    value_mask = np.zeros((records, GROUP_SIZE), dtype=np.bool_)
    candidate_scores = np.zeros((records, MAX_CANDIDATES), dtype=np.float32)
    candidate_mask = np.zeros((records, MAX_CANDIDATES), dtype=np.bool_)
    board = chess.Board()
    for record_index in range(records):
        if board.is_game_over() or board.ply() > 80:
            board.reset()
        for _ in range(rng.randrange(1, 5)):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        features[record_index, 0] = encode_board(board)
        turns[record_index, 0] = board.turn
        targets[record_index, 0] = _material_value(board)
        value_mask[record_index, 0] = True
        for candidate_index, move in enumerate(list(board.legal_moves)[:MAX_CANDIDATES]):
            board.push(move)
            root_value = -_material_value(board)
            features[record_index, candidate_index + 1] = encode_board(board)
            turns[record_index, candidate_index + 1] = board.turn
            targets[record_index, candidate_index + 1] = -root_value
            value_mask[record_index, candidate_index + 1] = True
            candidate_scores[record_index, candidate_index] = root_value
            candidate_mask[record_index, candidate_index] = True
            board.pop()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=features,
        turns=turns,
        targets=targets,
        value_mask=value_mask,
        candidate_scores=candidate_scores,
        candidate_mask=candidate_mask,
    )
    return path


def losses(
    predictions: torch.Tensor,
    batch: Batch,
    ranking_weight: float,
    top_move_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    value_errors = (predictions - batch["targets"])[batch["value_mask"]]
    value_loss = torch.mean(value_errors.square())

    predicted_q = -predictions[:, 1:]
    teacher_q = batch["candidate_scores"]
    teacher_difference = teacher_q.unsqueeze(2) - teacher_q.unsqueeze(1)
    predicted_difference = predicted_q.unsqueeze(2) - predicted_q.unsqueeze(1)
    valid = batch["candidate_mask"]
    pair_mask = valid.unsqueeze(2) & valid.unsqueeze(1) & (teacher_difference > 0.02)
    if torch.any(pair_mask):
        ranking_loss = functional.softplus(-4.0 * predicted_difference[pair_mask]).mean()
        ranking_accuracy = torch.mean(
            (predicted_difference[pair_mask] > 0).float()
        ).item()
    else:
        ranking_loss = predictions.new_zeros(())
        ranking_accuracy = 0.0

    top_move_logits = 4.0 * predicted_q.masked_fill(~batch["candidate_mask"], -1e4)
    top_move_targets = torch.zeros(
        top_move_logits.shape[0], dtype=torch.long, device=top_move_logits.device
    )
    top_move_loss = functional.cross_entropy(top_move_logits, top_move_targets)
    top_move_accuracy = torch.mean((top_move_logits.argmax(dim=1) == 0).float()).item()
    total = value_loss + ranking_weight * ranking_loss + top_move_weight * top_move_loss
    return total, {
        "value_mse": float(value_loss.detach()),
        "ranking_loss": float(ranking_loss.detach()),
        "ranking_accuracy": ranking_accuracy,
        "top_move_loss": float(top_move_loss.detach()),
        "top_move_accuracy": top_move_accuracy,
        "value_mae": float(value_errors.abs().mean().detach()),
    }


def evaluate(
    model: SparseValueNetwork,
    batches: Iterator[Batch],
    ranking_weight: float,
    top_move_weight: float,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    count = 0
    model.eval()
    with torch.inference_mode():
        for batch in batches:
            predictions = model(batch["features"], batch["turns"])
            loss, metrics = losses(predictions, batch, ranking_weight, top_move_weight)
            metrics["total_loss"] = float(loss.detach())
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + value
            count += 1
    return {name: value / max(count, 1) for name, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("weights/nnue.npz"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--ranking-weight", type=float, default=0.25)
    parser.add_argument("--top-move-weight", type=float, default=0.25)
    parser.add_argument("--accumulator", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--bottleneck", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.epochs < 1 or arguments.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if arguments.ranking_weight < 0 or arguments.top_move_weight < 0:
        parser.error("loss weights must be non-negative")

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    if arguments.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = arguments.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    if arguments.smoke:
        shards = [synthetic_shard(Path("/tmp/aichessathon-distill-smoke.npz"), 64, arguments.seed)]
    elif arguments.data:
        shards = resolve_shards(arguments.data)
    else:
        parser.error("pass --data or --smoke")

    model = SparseValueNetwork(
        accumulator=arguments.accumulator,
        hidden=arguments.hidden,
        bottleneck=arguments.bottleneck,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate)
    last_validation: dict[str, float] = {}
    best_validation_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_validation: dict[str, float] = {}
    for epoch in range(arguments.epochs):
        model.train()
        batches = iterate_batches(
            shards,
            batch_size=arguments.batch_size,
            device=device,
            seed=arguments.seed,
            epoch=epoch,
            validation=False,
        )
        training_loss = 0.0
        batch_count = 0
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch["features"], batch["turns"])
            loss, _ = losses(
                predictions,
                batch,
                arguments.ranking_weight,
                arguments.top_move_weight,
            )
            loss.backward()
            optimizer.step()
            training_loss += float(loss.detach())
            batch_count += 1
        last_validation = evaluate(
            model,
            iterate_batches(
                shards,
                batch_size=arguments.batch_size,
                device=device,
                seed=arguments.seed,
                epoch=epoch,
                validation=True,
            ),
            arguments.ranking_weight,
            arguments.top_move_weight,
        )
        if last_validation["total_loss"] < best_validation_loss:
            best_validation_loss = last_validation["total_loss"]
            best_epoch = epoch + 1
            best_validation = dict(last_validation)
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "train_loss": training_loss / max(batch_count, 1),
                    **last_validation,
                }
            )
        )

    assert best_state is not None
    model.load_state_dict(best_state)

    report = export_quantized(
        model,
        arguments.output,
        metadata={
            "device": device,
            "epochs": arguments.epochs,
            "batch_size": arguments.batch_size,
            "learning_rate": arguments.learning_rate,
            "ranking_weight": arguments.ranking_weight,
            "top_move_weight": arguments.top_move_weight,
            "seed": arguments.seed,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            **best_validation,
        },
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
