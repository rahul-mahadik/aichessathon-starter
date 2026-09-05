"""Convert raw teacher JSONL shards into fixed-shape sparse training shards."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import tempfile
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
        metadata=np.stack([group.metadata for group in groups]),
        policy_from=np.stack([group.policy_from for group in groups]),
        policy_to=np.stack([group.policy_to for group in groups]),
        policy_promotion=np.stack([group.policy_promotion for group in groups]),
        policy_scores=np.stack([group.policy_scores for group in groups]),
        policy_mask=np.stack([group.policy_mask for group in groups]),
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


def _build_partition(
    index: int,
    input_path: Path,
    temporary_root: Path,
    records_per_shard: int,
    cp_scale: float,
) -> tuple[int, dict[str, object]]:
    partition = temporary_root / f"partition-{index:05d}"
    return index, build([input_path], partition, records_per_shard, cp_scale)


def build_parallel(
    inputs: list[Path],
    output: Path,
    records_per_shard: int,
    cp_scale: float,
    workers: int,
) -> dict[str, object]:
    """Convert independent raw shards concurrently and merge deterministically."""
    if workers == 1 or len(inputs) == 1:
        return build(inputs, output, records_per_shard, cp_scale)
    output.mkdir(parents=True, exist_ok=True)
    results: list[tuple[int, dict[str, object]]] = []
    with tempfile.TemporaryDirectory(prefix=".parallel-", dir=output) as temporary:
        temporary_root = Path(temporary)
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _build_partition,
                    index,
                    input_path,
                    temporary_root,
                    records_per_shard,
                    cp_scale,
                )
                for index, input_path in enumerate(inputs)
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        records = 0
        shards: list[str] = []
        for index, partition_metadata in sorted(results):
            records += int(partition_metadata["records"])
            partition = temporary_root / f"partition-{index:05d}"
            for source_name in partition_metadata["shards"]:
                destination = output / f"part-{len(shards):05d}.npz"
                (partition / str(source_name)).replace(destination)
                shards.append(destination.name)

    metadata: dict[str, object] = {
        "records": records,
        "shards": shards,
        "records_per_shard": records_per_shard,
        "cp_scale": cp_scale,
        "inputs": [str(path) for path in inputs],
        "workers": workers,
    }
    (output / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records-per-shard", type=int, default=100_000)
    parser.add_argument("--cp-scale", type=float, default=DEFAULT_CP_SCALE)
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args()
    if arguments.records_per_shard < 1 or arguments.cp_scale <= 0 or arguments.workers < 1:
        parser.error("--records-per-shard, --cp-scale, and --workers must be positive")
    metadata = build_parallel(
        arguments.inputs,
        arguments.output,
        arguments.records_per_shard,
        arguments.cp_scale,
        arguments.workers,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
