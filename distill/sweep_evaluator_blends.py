"""Sweep convex evaluator blends against unseen teacher-labelled positions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import chess
import numpy as np

from controls.classical.evaluation import evaluate as classical_evaluation
from distill.compare_evaluators import ErrorMetrics, RankingMetrics, sha256
from distill.features import DEFAULT_CP_SCALE, score_to_value
from distill.schema import read_records
from nnue_runtime import QuantizedEvaluator


def _compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(total,)]
    return [
        (head, *tail)
        for head in range(total + 1)
        for tail in _compositions(total - head, parts - 1)
    ]


def _metrics(
    root_targets: np.ndarray,
    candidate_targets: list[list[float]],
    root_prediction: np.ndarray,
    candidate_prediction: list[np.ndarray],
    ranking_margin: float,
) -> dict[str, object]:
    root_error = ErrorMetrics()
    candidate_error = ErrorMetrics()
    ranking = RankingMetrics()
    for prediction, target in zip(root_prediction, root_targets, strict=True):
        root_error.add(float(prediction), float(target))
    for predictions, targets in zip(candidate_prediction, candidate_targets, strict=True):
        for prediction, target in zip(predictions, targets, strict=True):
            candidate_error.add(float(prediction), target)
        ranking.add(targets, predictions.tolist(), ranking_margin)
    return {
        "root_value": root_error.report(),
        "candidate_value": candidate_error.report(),
        "ranking": ranking.report(),
    }


def sweep(
    inputs: list[Path],
    models: list[Path],
    *,
    include_classical: bool,
    step: float,
    cp_scale: float,
    ranking_margin: float,
    max_records: int | None,
) -> dict[str, object]:
    evaluators = [QuantizedEvaluator(path, antisymmetric=True) for path in models]
    for evaluator in evaluators:
        evaluator.warmup()
    names = [path.stem for path in models]
    if include_classical:
        names.append("classical")

    root_targets: list[float] = []
    root_sources: list[list[float]] = [[] for _ in names]
    candidate_targets: list[list[float]] = []
    candidate_sources: list[list[list[float]]] = [[] for _ in names]
    records = 0
    candidates = 0
    for record in read_records(inputs):
        if max_records is not None and records >= max_records:
            break
        board = chess.Board(record.fen)
        root_targets.append(score_to_value(record.root_score, cp_scale))
        for index, evaluator in enumerate(evaluators):
            root_sources[index].append(evaluator.raw_value(board))
        if include_classical:
            root_sources[-1].append(math.tanh(classical_evaluation(board) / cp_scale))

        record_targets: list[float] = []
        record_sources: list[list[float]] = [[] for _ in names]
        for candidate in record.candidates:
            target = score_to_value(candidate.score, cp_scale)
            board.push(chess.Move.from_uci(candidate.move))
            for index, evaluator in enumerate(evaluators):
                record_sources[index].append(-evaluator.raw_value(board))
            if include_classical:
                record_sources[-1].append(-math.tanh(classical_evaluation(board) / cp_scale))
            board.pop()
            record_targets.append(target)
            candidates += 1
        candidate_targets.append(record_targets)
        for index, predictions in enumerate(record_sources):
            candidate_sources[index].append(predictions)
        records += 1

    root_arrays = [np.asarray(values, dtype=np.float64) for values in root_sources]
    candidate_arrays = [
        [np.asarray(values, dtype=np.float64) for values in source] for source in candidate_sources
    ]
    units = round(1.0 / step)
    if not math.isclose(units * step, 1.0, abs_tol=1e-9):
        raise ValueError("step must divide 1.0 exactly")

    results: list[dict[str, object]] = []
    root_target_array = np.asarray(root_targets, dtype=np.float64)
    for composition in _compositions(units, len(names)):
        weights = np.asarray(composition, dtype=np.float64) / units
        root_prediction = sum(
            weight * values for weight, values in zip(weights, root_arrays, strict=True)
        )
        blended_candidates = [
            sum(
                weights[source_index] * candidate_arrays[source_index][record_index]
                for source_index in range(len(names))
            )
            for record_index in range(records)
        ]
        metrics = _metrics(
            root_target_array,
            candidate_targets,
            root_prediction,
            blended_candidates,
            ranking_margin,
        )
        results.append(
            {
                "weights": dict(zip(names, weights.tolist(), strict=True)),
                **metrics,
            }
        )

    def top(metric: str) -> list[dict[str, object]]:
        def value(item: dict[str, object]) -> float:
            section_name = "candidate_value" if metric == "mse" else "ranking"
            value_name = {
                "pairwise": "pairwise_accuracy",
                "top_move": "top_move_accuracy",
                "mse": "mse",
            }[metric]
            section = item[section_name]
            assert isinstance(section, dict)
            result = section[value_name]
            assert isinstance(result, int | float)
            return float(result)

        return sorted(results, key=value, reverse=metric != "mse")[:10]

    return {
        "inputs": [str(path) for path in inputs],
        "models": [
            {"name": path.stem, "path": str(path), "sha256": sha256(path)} for path in models
        ],
        "include_classical": include_classical,
        "records": records,
        "candidates": candidates,
        "step": step,
        "top_by_pairwise_accuracy": top("pairwise"),
        "top_by_top_move_accuracy": top("top_move"),
        "top_by_candidate_mse": top("mse"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--include-classical", action="store_true")
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--cp-scale", type=float, default=DEFAULT_CP_SCALE)
    parser.add_argument("--ranking-margin", type=float, default=0.02)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if not 0 < arguments.step <= 1 or arguments.cp_scale <= 0:
        parser.error("--step must be in (0, 1] and --cp-scale must be positive")
    report = sweep(
        arguments.inputs,
        arguments.model,
        include_classical=arguments.include_classical,
        step=arguments.step,
        cp_scale=arguments.cp_scale,
        ranking_margin=arguments.ranking_margin,
        max_records=arguments.max_records,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
