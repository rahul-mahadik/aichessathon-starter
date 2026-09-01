"""Validate raw teacher shards and report label-quality statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import chess

from distill.schema import read_records


def inspect(paths: list[Path]) -> dict[str, Any]:
    records = 0
    candidates = 0
    wdl_scores = 0
    mate_scores = 0
    cp_scores = 0
    duplicates = 0
    incomplete_multipv_records = 0
    requested_multipv: dict[str, int] = {}
    minimum_depth: int | None = None
    maximum_depth: int | None = None
    budgets: dict[str, int] = {}
    seen: set[str] = set()

    for record in read_records(paths):
        records += 1
        duplicates += int(record.fen in seen)
        seen.add(record.fen)
        budget = (
            f"nodes:{record.node_budget}"
            if record.node_budget is not None
            else f"depth:{record.depth_budget}"
        )
        budgets[budget] = budgets.get(budget, 0) + 1
        board = chess.Board(record.fen)
        legal_moves = set(board.legal_moves)
        requested_key = str(record.multipv)
        requested_multipv[requested_key] = requested_multipv.get(requested_key, 0) + 1
        expected_candidates = min(record.multipv, len(legal_moves))
        incomplete_multipv_records += int(len(record.candidates) != expected_candidates)
        for candidate in record.candidates:
            move = chess.Move.from_uci(candidate.move)
            if move not in legal_moves:
                raise ValueError(f"illegal candidate {candidate.move} in {record.fen}")
            candidates += 1
            score = candidate.score
            wdl_scores += int(score.wdl is not None)
            mate_scores += int(score.mate is not None)
            cp_scores += int(score.cp is not None)
            if candidate.depth is not None:
                minimum_depth = (
                    candidate.depth
                    if minimum_depth is None
                    else min(minimum_depth, candidate.depth)
                )
                maximum_depth = (
                    candidate.depth
                    if maximum_depth is None
                    else max(maximum_depth, candidate.depth)
                )

    if records == 0:
        raise ValueError("no teacher records found")
    return {
        "records": records,
        "unique_fens": len(seen),
        "duplicate_fens": duplicates,
        "incomplete_multipv_records": incomplete_multipv_records,
        "candidates": candidates,
        "candidates_per_record": candidates / records,
        "wdl_coverage": wdl_scores / candidates,
        "cp_coverage": cp_scores / candidates,
        "mate_scores": mate_scores,
        "minimum_depth": minimum_depth,
        "maximum_depth": maximum_depth,
        "budgets": budgets,
        "requested_multipv": requested_multipv,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--expected-records", type=int)
    parser.add_argument("--expected-candidates", type=int, default=8)
    arguments = parser.parse_args()
    summary = inspect(arguments.inputs)
    print(json.dumps(summary, indent=2))
    if arguments.expected_records is not None and summary["records"] != arguments.expected_records:
        raise SystemExit(
            f"expected {arguments.expected_records} records, found {summary['records']}"
        )
    expected_key = str(arguments.expected_candidates)
    expected_requests = summary["requested_multipv"].get(expected_key, 0)
    if expected_requests != summary["records"]:
        raise SystemExit(
            f"expected every record to request MultiPV {arguments.expected_candidates}, "
            f"found {expected_requests} of {summary['records']}"
        )
    if summary["incomplete_multipv_records"]:
        raise SystemExit(
            f"found {summary['incomplete_multipv_records']} incomplete MultiPV records"
        )
    if summary["duplicate_fens"]:
        raise SystemExit(f"found {summary['duplicate_fens']} duplicate FENs")


if __name__ == "__main__":
    main()
