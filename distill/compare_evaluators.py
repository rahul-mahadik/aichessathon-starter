"""Compare learned and handcrafted evaluators against retained teacher traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import chess

from distill.features import DEFAULT_CP_SCALE, score_to_value
from distill.schema import read_records
from nnue_runtime import QuantizedEvaluator
from search_engine import handcrafted_evaluation


class ErrorMetrics:
    def __init__(self) -> None:
        self.count = 0
        self.squared_error = 0.0
        self.absolute_error = 0.0

    def add(self, prediction: float, target: float) -> None:
        error = prediction - target
        self.count += 1
        self.squared_error += error * error
        self.absolute_error += abs(error)

    def report(self) -> dict[str, float | int]:
        divisor = max(self.count, 1)
        return {
            "count": self.count,
            "mse": self.squared_error / divisor,
            "mae": self.absolute_error / divisor,
        }


class RankingMetrics:
    def __init__(self) -> None:
        self.records = 0
        self.top_move_matches = 0
        self.pairs = 0
        self.correct_pairs = 0

    def add(self, teacher: list[float], predicted: list[float], margin: float) -> None:
        self.records += 1
        if max(range(len(predicted)), key=predicted.__getitem__) == 0:
            self.top_move_matches += 1
        for left in range(len(teacher)):
            for right in range(len(teacher)):
                if teacher[left] - teacher[right] > margin:
                    self.pairs += 1
                    self.correct_pairs += predicted[left] > predicted[right]

    def report(self) -> dict[str, float | int]:
        return {
            "records": self.records,
            "top_move_matches": self.top_move_matches,
            "top_move_accuracy": self.top_move_matches / max(self.records, 1),
            "ordered_pairs": self.pairs,
            "correct_pairs": self.correct_pairs,
            "pairwise_accuracy": self.correct_pairs / max(self.pairs, 1),
        }


def normalized_handcrafted(board: chess.Board, cp_scale: float) -> float:
    return math.tanh(handcrafted_evaluation(board) / cp_scale)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare(
    inputs: list[Path],
    model_path: Path,
    *,
    max_records: int | None,
    cp_scale: float,
    ranking_margin: float,
    antisymmetric: bool,
) -> dict[str, object]:
    learned = QuantizedEvaluator(model_path, antisymmetric=antisymmetric)
    learned.warmup()
    root_learned = ErrorMetrics()
    root_handcrafted = ErrorMetrics()
    candidate_learned = ErrorMetrics()
    candidate_handcrafted = ErrorMetrics()
    learned_rankings = RankingMetrics()
    handcrafted_rankings = RankingMetrics()
    records = 0
    candidates = 0

    for record in read_records(inputs):
        if max_records is not None and records >= max_records:
            break
        board = chess.Board(record.fen)
        root_target = score_to_value(record.root_score, cp_scale)
        root_learned.add(learned.raw_value(board), root_target)
        root_handcrafted.add(normalized_handcrafted(board, cp_scale), root_target)

        teacher_scores: list[float] = []
        learned_scores: list[float] = []
        handcrafted_scores: list[float] = []
        for candidate in record.candidates:
            teacher_value = score_to_value(candidate.score, cp_scale)
            move = chess.Move.from_uci(candidate.move)
            board.push(move)
            learned_value = -learned.raw_value(board)
            handcrafted_value = -normalized_handcrafted(board, cp_scale)
            board.pop()
            candidate_learned.add(learned_value, teacher_value)
            candidate_handcrafted.add(handcrafted_value, teacher_value)
            teacher_scores.append(teacher_value)
            learned_scores.append(learned_value)
            handcrafted_scores.append(handcrafted_value)
            candidates += 1

        learned_rankings.add(teacher_scores, learned_scores, ranking_margin)
        handcrafted_rankings.add(teacher_scores, handcrafted_scores, ranking_margin)
        records += 1

    return {
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "antisymmetric": antisymmetric,
        "inputs": [str(path) for path in inputs],
        "records": records,
        "candidates": candidates,
        "cp_scale": cp_scale,
        "ranking_margin": ranking_margin,
        "learned": {
            "root_value": root_learned.report(),
            "candidate_value": candidate_learned.report(),
            "ranking": learned_rankings.report(),
        },
        "handcrafted": {
            "root_value": root_handcrafted.report(),
            "candidate_value": candidate_handcrafted.report(),
            "ranking": handcrafted_rankings.report(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--cp-scale", type=float, default=DEFAULT_CP_SCALE)
    parser.add_argument("--ranking-margin", type=float, default=0.02)
    parser.add_argument("--antisymmetric", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.max_records is not None and arguments.max_records < 1:
        parser.error("--max-records must be positive")
    if arguments.cp_scale <= 0 or arguments.ranking_margin < 0:
        parser.error("--cp-scale must be positive and --ranking-margin non-negative")
    report = compare(
        arguments.inputs,
        arguments.model,
        max_records=arguments.max_records,
        cp_scale=arguments.cp_scale,
        ranking_margin=arguments.ranking_margin,
        antisymmetric=arguments.antisymmetric,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
