"""Mine positions where deeper Stockfish search changes values or move ordering."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import chess

from distill.features import score_to_value
from distill.schema import TeacherRecord, read_records, write_records


@dataclass(frozen=True)
class DepthDistance:
    """A bounded, interpretable distance between two MultiPV teacher records."""

    top_move_changed: bool
    top_three_distance: float
    root_value_delta: float
    candidate_value_mae: float
    ranking_disagreement: float
    common_candidates: int
    distance: float


@dataclass(frozen=True)
class DepthRow:
    """One position's shallow-to-medium and medium-to-deep changes."""

    fen: str
    low_to_medium: DepthDistance
    medium_to_deep: DepthDistance
    quiet: bool
    mate: bool


def resolve_inputs(paths: list[Path]) -> list[Path]:
    inputs: list[Path] = []
    for path in paths:
        if path.is_dir():
            inputs.extend(sorted(path.rglob("*.jsonl")))
            inputs.extend(sorted(path.rglob("*.jsonl.gz")))
        else:
            inputs.append(path)
    missing = [path for path in inputs if not path.exists()]
    if missing or not inputs:
        raise ValueError(f"no teacher traces found: {missing or paths}")
    return inputs


def _records_by_fen(paths: list[Path]) -> dict[str, TeacherRecord]:
    records: dict[str, TeacherRecord] = {}
    for record in read_records(resolve_inputs(paths)):
        if record.fen in records:
            raise ValueError(f"duplicate teacher record for {record.fen}")
        records[record.fen] = record
    return records


def _candidate_values(record: TeacherRecord) -> dict[str, float]:
    return {candidate.move: score_to_value(candidate.score) for candidate in record.candidates}


def _top_k_distance(
    shallower: TeacherRecord, deeper: TeacherRecord, *, top_k: int = 3
) -> float:
    """Return a bounded rank/set distance between the teachers' top-k moves."""
    shallow_moves = [candidate.move for candidate in shallower.candidates[:top_k]]
    deep_moves = [candidate.move for candidate in deeper.candidates[:top_k]]
    moves = set(shallow_moves) | set(deep_moves)
    if not moves:
        return 0.0
    shallow_ranks = {move: rank for rank, move in enumerate(shallow_moves)}
    deep_ranks = {move: rank for rank, move in enumerate(deep_moves)}
    displacement = sum(
        abs(shallow_ranks.get(move, top_k) - deep_ranks.get(move, top_k))
        for move in moves
    )
    return displacement / (top_k * len(moves))


def _is_quiet(record: TeacherRecord) -> bool:
    board = chess.Board(record.fen)
    move = chess.Move.from_uci(record.best_move)
    return not (board.is_capture(move) or board.gives_check(move) or move.promotion)


def _contains_mate(record: TeacherRecord) -> bool:
    return record.root_score.mate is not None or any(
        candidate.score.mate is not None for candidate in record.candidates
    )


def depth_distance(
    shallower: TeacherRecord,
    deeper: TeacherRecord,
    *,
    ranking_margin: float = 0.02,
) -> DepthDistance:
    """Compare the same position at two teacher budgets."""
    if shallower.fen != deeper.fen:
        raise ValueError("depth comparison requires identical FENs")
    shallow_values = _candidate_values(shallower)
    deep_values = _candidate_values(deeper)
    common = sorted(set(shallow_values) & set(deep_values))
    if not common:
        raise ValueError(f"teacher traces share no candidates for {shallower.fen}")

    candidate_value_mae = mean(
        abs(shallow_values[move] - deep_values[move]) for move in common
    )
    discordant = 0
    comparable = 0
    for first_index, first in enumerate(common):
        for second in common[first_index + 1 :]:
            shallow_delta = shallow_values[first] - shallow_values[second]
            deep_delta = deep_values[first] - deep_values[second]
            if abs(shallow_delta) <= ranking_margin or abs(deep_delta) <= ranking_margin:
                continue
            comparable += 1
            discordant += int(shallow_delta * deep_delta < 0)
    ranking_disagreement = discordant / comparable if comparable else 0.0
    root_value_delta = abs(
        score_to_value(shallower.root_score) - score_to_value(deeper.root_score)
    )
    top_move_changed = shallower.best_move != deeper.best_move
    top_three_distance = _top_k_distance(shallower, deeper)
    distance = (
        1.5 * float(top_move_changed)
        + 0.75 * top_three_distance
        + root_value_delta
        + 0.5 * candidate_value_mae
        + ranking_disagreement
    )
    return DepthDistance(
        top_move_changed=top_move_changed,
        top_three_distance=top_three_distance,
        root_value_delta=root_value_delta,
        candidate_value_mae=candidate_value_mae,
        ranking_disagreement=ranking_disagreement,
        common_candidates=len(common),
        distance=distance,
    )


def _quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    quantiles = (("p00", 0.0), ("p50", 0.5), ("p90", 0.9), ("p99", 0.99), ("p100", 1.0))
    return {
        label: ordered[round((len(ordered) - 1) * quantile)]
        for label, quantile in quantiles
    }


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _stratified_selection(
    rows: list[DepthRow], selection_size: int, seed: int
) -> tuple[list[tuple[DepthRow, str]], dict[str, int]]:
    """Select a deterministic mix of decision changes, outcome shifts, and coverage."""
    quotas = {
        "top-move-flip": round(selection_size * 0.35),
        "top-three-reorder": round(selection_size * 0.20),
        "outcome-change": round(selection_size * 0.20),
        "quiet-disagreement": round(selection_size * 0.15),
    }
    quotas["random-coverage"] = selection_size - sum(quotas.values())
    selected: list[tuple[DepthRow, str]] = []
    used: set[str] = set()

    def take_ranked(name: str, candidates: list[DepthRow], count: int) -> None:
        available = [row for row in candidates if row.fen not in used]
        for row in available[:count]:
            selected.append((row, name))
            used.add(row.fen)

    descending = sorted(rows, key=lambda row: row.medium_to_deep.distance, reverse=True)
    take_ranked(
        "top-move-flip",
        [row for row in descending if row.medium_to_deep.top_move_changed],
        quotas["top-move-flip"],
    )
    take_ranked(
        "top-three-reorder",
        [
            row
            for row in descending
            if not row.medium_to_deep.top_move_changed
            and row.medium_to_deep.top_three_distance > 0
        ],
        quotas["top-three-reorder"],
    )
    outcome = sorted(
        rows,
        key=lambda row: (
            row.medium_to_deep.root_value_delta,
            row.medium_to_deep.distance,
        ),
        reverse=True,
    )
    take_ranked("outcome-change", outcome, quotas["outcome-change"])
    take_ranked(
        "quiet-disagreement",
        [row for row in descending if row.quiet and not row.mate],
        quotas["quiet-disagreement"],
    )
    coverage = [row for row in rows if row.fen not in used]
    random.Random(seed).shuffle(coverage)
    take_ranked("random-coverage", coverage, quotas["random-coverage"])
    take_ranked("score-backfill", descending, selection_size - len(selected))
    if len(selected) != selection_size:
        raise ValueError(
            f"could select only {len(selected)} of {selection_size} requested positions"
        )
    realized: dict[str, int] = {}
    for _, stratum in selected:
        realized[stratum] = realized.get(stratum, 0) + 1
    return selected, realized


def _difficulty_buckets(rows: list[DepthRow]) -> dict[str, list[DepthRow]]:
    """Bucket positions by when teacher conclusions change along the node ladder."""
    low_values = [row.low_to_medium.distance for row in rows]
    deep_values = [row.medium_to_deep.distance for row in rows]
    low_median = _quantile(low_values, 0.5)
    low_high = _quantile(low_values, 0.75)
    deep_median = _quantile(deep_values, 0.5)
    deep_high = _quantile(deep_values, 0.75)
    buckets: dict[str, list[DepthRow]] = {
        "easy": [],
        "medium": [],
        "deep": [],
        "unstable": [],
        "transitional": [],
    }
    for row in rows:
        low = row.low_to_medium.distance
        deep = row.medium_to_deep.distance
        if low <= low_median and deep <= deep_median:
            bucket = "easy"
        elif low >= low_high and deep <= deep_median:
            bucket = "medium"
        elif low <= low_median and deep >= deep_high:
            bucket = "deep"
        elif low >= low_high and deep >= deep_high:
            bucket = "unstable"
        else:
            bucket = "transitional"
        buckets[bucket].append(row)
    return buckets


def _fen_digest(rows: list[DepthRow]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item.fen):
        digest.update(row.fen.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def mine(
    low_paths: list[Path],
    medium_paths: list[Path],
    deep_paths: list[Path],
    *,
    output: Path,
    select: int,
    ranking_margin: float,
    seed: int,
) -> dict[str, object]:
    low = _records_by_fen(low_paths)
    medium = _records_by_fen(medium_paths)
    deep = _records_by_fen(deep_paths)
    common_fens = sorted(set(low) & set(medium) & set(deep))
    if not common_fens:
        raise ValueError("the 10k/100k/1M trace sets have no positions in common")
    same_positions = (
        len(common_fens) == len(low) == len(medium) == len(deep)
    )
    if not same_positions:
        raise ValueError(
            "paired-depth mining requires identical position sets: "
            f"low={len(low)}, medium={len(medium)}, deep={len(deep)}, common={len(common_fens)}"
        )

    rows: list[DepthRow] = []
    for fen in common_fens:
        low_to_medium = depth_distance(low[fen], medium[fen], ranking_margin=ranking_margin)
        medium_to_deep = depth_distance(
            medium[fen], deep[fen], ranking_margin=ranking_margin
        )
        rows.append(
            DepthRow(
                fen=fen,
                low_to_medium=low_to_medium,
                medium_to_deep=medium_to_deep,
                quiet=_is_quiet(deep[fen]),
                mate=_contains_mate(deep[fen]),
            )
        )
    selection_size = min(select, len(rows))
    selected, strata = _stratified_selection(rows, selection_size, seed)
    high_rows = [row for row, _ in selected]
    random_rows = random.Random(seed + 1).sample(rows, selection_size)
    selected_strata = {row.fen: stratum for row, stratum in selected}

    output.mkdir(parents=True, exist_ok=True)
    selections = {"high": high_rows, "random": random_rows}
    for name, selection in selections.items():
        for label, records in (("low", low), ("medium", medium), ("deep", deep)):
            write_records(
                output / f"{name}-{label}.jsonl.gz",
                [records[row.fen] for row in selection],
            )
        (output / f"{name}.epd").write_text("".join(f"{row.fen}\n" for row in selection))
        with (output / f"{name}-distances.jsonl").open("w") as destination:
            for row in selection:
                destination.write(
                    json.dumps(
                        {
                            "fen": row.fen,
                            "stratum": (
                                selected_strata[row.fen]
                                if name == "high"
                                else "random-control"
                            ),
                            "quiet": row.quiet,
                            "mate": row.mate,
                            "low_to_medium": asdict(row.low_to_medium),
                            "medium_to_deep": asdict(row.medium_to_deep),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )

    buckets = _difficulty_buckets(rows)
    bucket_directory = output / "buckets"
    bucket_directory.mkdir(parents=True, exist_ok=True)
    for name, bucket_rows in buckets.items():
        (bucket_directory / f"{name}.epd").write_text(
            "".join(f"{row.fen}\n" for row in bucket_rows)
        )
        write_records(
            bucket_directory / f"{name}-deep.jsonl.gz",
            [deep[row.fen] for row in bucket_rows],
        )

    def summary_for(index: int) -> dict[str, object]:
        distances = [
            row.low_to_medium if index == 1 else row.medium_to_deep for row in rows
        ]
        return {
            "top_move_change_rate": mean(float(item.top_move_changed) for item in distances),
            "mean_top_three_distance": mean(item.top_three_distance for item in distances),
            "mean_root_value_delta": mean(item.root_value_delta for item in distances),
            "mean_candidate_value_mae": mean(item.candidate_value_mae for item in distances),
            "mean_ranking_disagreement": mean(item.ranking_disagreement for item in distances),
            "distance_quantiles": _quantiles([item.distance for item in distances]),
        }

    report: dict[str, object] = {
        "positions": len(rows),
        "selected_high": selection_size,
        "selected_random": selection_size,
        "ranking_margin": ranking_margin,
        "seed": seed,
        "selection_strata": strata,
        "high_fen_sha256": _fen_digest(high_rows),
        "random_fen_sha256": _fen_digest(random_rows),
        "high_random_overlap": len(
            {row.fen for row in high_rows} & {row.fen for row in random_rows}
        ),
        "difficulty_buckets": {name: len(bucket_rows) for name, bucket_rows in buckets.items()},
        "low_to_medium": summary_for(1),
        "medium_to_deep": summary_for(2),
        "inputs": {
            "low": [str(path) for path in resolve_inputs(low_paths)],
            "medium": [str(path) for path in resolve_inputs(medium_paths)],
            "deep": [str(path) for path in resolve_inputs(deep_paths)],
        },
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low", type=Path, nargs="+", required=True, help="10k-node traces")
    parser.add_argument(
        "--medium", type=Path, nargs="+", required=True, help="100k-node traces"
    )
    parser.add_argument("--deep", type=Path, nargs="+", required=True, help="1M-node traces")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--select", type=int, required=True)
    parser.add_argument("--ranking-margin", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    arguments = parser.parse_args()
    if arguments.select < 1 or arguments.ranking_margin < 0:
        parser.error("--select must be positive and --ranking-margin non-negative")
    report = mine(
        arguments.low,
        arguments.medium,
        arguments.deep,
        output=arguments.output,
        select=arguments.select,
        ranking_margin=arguments.ranking_margin,
        seed=arguments.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
