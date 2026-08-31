"""Convert raw teacher JSONL shards into fixed-shape sparse training shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from distill.features import DEFAULT_CP_SCALE, TrainingGroup, record_to_group
from distill.schema import read_records


def write_shard(path: Path, groups: list[TrainingGroup]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=np.stack([group.features for group in groups]),
        turns=np.stack([group.turns for group in groups]),
        targets=np.stack([group.targets for group in groups]),
        value_mask=np.stack([group.value_mask for group in groups]),
        candidate_scores=np.stack([group.candidate_scores for group in groups]),
        candidate_mask=np.stack([group.candidate_mask for group in groups]),
    )


def build(
    inputs: list[Path], output: Path, records_per_shard: int, cp_scale: float
) -> dict[str, object]:
    buffer: list[TrainingGroup] = []
    shards: list[str] = []
    records = 0
    for record in read_records(inputs):
        buffer.append(record_to_group(record, cp_scale))
        records += 1
        if len(buffer) >= records_per_shard:
            shard = output / f"part-{len(shards):05d}.npz"
            write_shard(shard, buffer)
            shards.append(shard.name)
            buffer.clear()
    if buffer:
        shard = output / f"part-{len(shards):05d}.npz"
        write_shard(shard, buffer)
        shards.append(shard.name)
    if not shards:
        raise SystemExit("no teacher records were found")
    metadata: dict[str, object] = {
        "records": records,
        "shards": shards,
        "records_per_shard": records_per_shard,
        "cp_scale": cp_scale,
        "inputs": [str(path) for path in inputs],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records-per-shard", type=int, default=100_000)
    parser.add_argument("--cp-scale", type=float, default=DEFAULT_CP_SCALE)
    arguments = parser.parse_args()
    if arguments.records_per_shard < 1 or arguments.cp_scale <= 0:
        parser.error("--records-per-shard and --cp-scale must be positive")
    metadata = build(
        arguments.inputs, arguments.output, arguments.records_per_shard, arguments.cp_scale
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
