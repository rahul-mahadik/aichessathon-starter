"""Mine positions where deeper Stockfish search changes values or move ordering."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from distill.features import score_to_value
from distill.schema import TeacherRecord, read_records, write_records


@dataclass(frozen=True)
class DepthDistance:
    """A bounded, interpretable distance between two MultiPV teacher records."""

    top_move_changed: bool
    root_value_delta: float
    candidate_value_mae: float
    ranking_disagreement: float
    common_candidates: int
    distance: float


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
    distance = (
        float(top_move_changed)
        + root_value_delta
        + candidate_value_mae
        + ranking_disagreement
    )
    return DepthDistance(
        top_move_changed=top_move_changed,
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


def mine(
    low_paths: list[Path],
    medium_paths: list[Path],
    deep_paths: list[Path],
    *,
    output: Path,
    select: int,
    ranking_margin: float,
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

    rows: list[tuple[str, DepthDistance, DepthDistance]] = []
    for fen in common_fens:
        low_to_medium = depth_distance(low[fen], medium[fen], ranking_margin=ranking_margin)
        medium_to_deep = depth_distance(
            medium[fen], deep[fen], ranking_margin=ranking_margin
        )
        rows.append((fen, low_to_medium, medium_to_deep))
    rows.sort(key=lambda row: row[2].distance, reverse=True)
    selection_size = min(select, len(rows))
    high_rows = rows[:selection_size]
    low_rows = list(reversed(rows[-selection_size:]))

    output.mkdir(parents=True, exist_ok=True)
    for name, selected in (("high", high_rows), ("low", low_rows)):
        selected_records = [deep[fen] for fen, _, _ in selected]
        write_records(output / f"{name}-deep.jsonl.gz", selected_records)
        (output / f"{name}.epd").write_text(
            "".join(f"{fen}\n" for fen, _, _ in selected)
        )
        with (output / f"{name}-distances.jsonl").open("w") as destination:
            for fen, low_to_medium, medium_to_deep in selected:
                destination.write(
                    json.dumps(
                        {
                            "fen": fen,
                            "low_to_medium": asdict(low_to_medium),
                            "medium_to_deep": asdict(medium_to_deep),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )

    def summary_for(index: int) -> dict[str, object]:
        distances = [row[index] for row in rows]
        return {
            "top_move_change_rate": mean(float(item.top_move_changed) for item in distances),
            "mean_root_value_delta": mean(item.root_value_delta for item in distances),
            "mean_candidate_value_mae": mean(item.candidate_value_mae for item in distances),
            "mean_ranking_disagreement": mean(item.ranking_disagreement for item in distances),
            "distance_quantiles": _quantiles([item.distance for item in distances]),
        }

    report: dict[str, object] = {
        "positions": len(rows),
        "selected_high": selection_size,
        "selected_low": selection_size,
        "ranking_margin": ranking_margin,
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
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
