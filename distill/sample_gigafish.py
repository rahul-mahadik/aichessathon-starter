"""Create deterministic, disjoint teacher corpora from pinned Gigafish shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import chess

DATASET_REPOSITORY = "lukesalamone/gigafish-3.8b-d10"
DATASET_REVISION = "47100399529ac17e9fdf2c8d0f49bfae89ae0c30"
DEFAULT_SOURCE_SHARDS = (0, 1000, 2000, 3000)
DOWNLOAD_ATTEMPTS = 10
DOWNLOAD_BACKOFF_CAP_S = 60.0


def reservoir_sample(rows: Iterable[str], sample_size: int, seed: int) -> list[str]:
    """Return a deterministic reservoir sample without materialising every source row."""
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    rng = random.Random(seed)
    reservoir: list[str] = []
    for index, row in enumerate(rows):
        if index < sample_size:
            reservoir.append(row)
            continue
        replacement = rng.randrange(index + 1)
        if replacement < sample_size:
            reservoir[replacement] = row
    return reservoir


def _download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "aichessathon-distill/1"})
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            with urllib.request.urlopen(request) as source, partial.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
            partial.replace(destination)
            return
        except urllib.error.HTTPError as error:
            if error.code != 429 and error.code < 500:
                raise
            retry_after = error.headers.get("Retry-After")
            if attempt + 1 == DOWNLOAD_ATTEMPTS:
                raise
        except urllib.error.URLError:
            retry_after = None
            if attempt + 1 == DOWNLOAD_ATTEMPTS:
                raise
        try:
            delay = float(retry_after) if retry_after is not None else 2.0**attempt
        except ValueError:
            delay = 2.0**attempt
        time.sleep(min(DOWNLOAD_BACKOFF_CAP_S, max(1.0, delay)))


def _parquet_rows(paths: list[Path]) -> Iterator[str]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise SystemExit("install the teacher dependency group: uv sync --group teacher") from error

    for path in paths:
        source = parquet.ParquetFile(path)
        if "fen" not in source.schema.names:
            raise ValueError(f"{path} does not contain a fen column")
        for batch in source.iter_batches(batch_size=65_536, columns=["fen"]):
            for fen in batch.column(0).to_pylist():
                if isinstance(fen, str):
                    yield fen


def _feature_position_key(board: chess.Board) -> str:
    """Identify exactly the placement and turn visible to the sparse evaluator."""
    return f"{board.board_fen()} {'w' if board.turn == chess.WHITE else 'b'}"


def _excluded_feature_positions(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            board = chess.Board(line.partition(";")[0].strip())
            excluded.add(_feature_position_key(board))
    return excluded


def _valid_unique_positions(
    candidates: Iterable[str], required: int, excluded: set[str] | None = None
) -> list[str]:
    positions: list[str] = []
    seen = set(excluded or ())
    for fen in candidates:
        try:
            board = chess.Board(fen)
        except ValueError:
            continue
        missing_king = board.king(chess.WHITE) is None or board.king(chess.BLACK) is None
        if board.is_game_over() or missing_king:
            continue
        key = _feature_position_key(board)
        if key in seen:
            continue
        seen.add(key)
        positions.append(board.fen())
        if len(positions) >= required:
            return positions
    raise ValueError(f"only found {len(positions)} valid unique positions; needed {required}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_tier(
    output: Path, tier: str, positions: list[str], shard_count: int
) -> list[dict[str, Any]]:
    if shard_count < 1 or shard_count > len(positions):
        raise ValueError("shard_count must be between one and the number of positions")
    tier_directory = output / tier
    tier_directory.mkdir(parents=True, exist_ok=True)
    shards = [[] for _ in range(shard_count)]
    for index, fen in enumerate(positions):
        shards[index % shard_count].append(fen)

    metadata: list[dict[str, Any]] = []
    for index, rows in enumerate(shards):
        path = tier_directory / f"part-{index:05d}.epd"
        path.write_text("\n".join(rows) + "\n")
        metadata.append(
            {
                "path": str(path.relative_to(output)),
                "positions": len(rows),
                "sha256": _sha256(path),
            }
        )
    return metadata


def sample_corpus(
    *,
    output: Path,
    cache: Path,
    medium_positions: int,
    deep_positions: int,
    shards_per_tier: int,
    source_shards: list[int],
    seed: int,
    exclude_paths: list[Path] | None = None,
) -> dict[str, Any]:
    required = medium_positions + deep_positions
    if required < 1 or medium_positions < 0 or deep_positions < 0:
        raise ValueError("medium_positions and deep_positions must be non-negative")
    if not source_shards:
        raise ValueError("at least one source shard is required")

    parquet_paths: list[Path] = []
    sources: list[dict[str, Any]] = []
    for shard in source_shards:
        filename = f"data-{shard:05d}.parquet"
        url = (
            f"https://huggingface.co/datasets/{DATASET_REPOSITORY}/resolve/"
            f"{DATASET_REVISION}/{filename}"
        )
        path = cache / filename
        _download(url, path)
        parquet_paths.append(path)
        sources.append({"file": filename, "url": url, "sha256": _sha256(path)})

    oversample = max(required + 1_000, int(required * 1.05))
    candidates = reservoir_sample(_parquet_rows(parquet_paths), oversample, seed)
    excluded = _excluded_feature_positions(exclude_paths or [])
    positions = _valid_unique_positions(candidates, required, excluded)
    random.Random(seed + 1).shuffle(positions)
    medium = positions[:medium_positions]
    deep = positions[medium_positions:]

    output.mkdir(parents=True, exist_ok=True)
    tiers: dict[str, dict[str, Any]] = {}
    for tier, tier_positions in (("medium", medium), ("deep", deep)):
        if tier_positions:
            tiers[tier] = {
                "positions": len(tier_positions),
                "shards": _write_tier(output, tier, tier_positions, shards_per_tier),
            }
    manifest: dict[str, Any] = {
        "format": "aichessathon-teacher-corpus-v1",
        "dataset": DATASET_REPOSITORY,
        "dataset_revision": DATASET_REVISION,
        "seed": seed,
        "excluded_feature_positions": len(excluded),
        "exclude_paths": [str(path) for path in (exclude_paths or [])],
        "sources": sources,
        "tiers": tiers,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/aichessathon-gigafish"))
    parser.add_argument("--medium-positions", type=int, default=100_000)
    parser.add_argument("--deep-positions", type=int, default=10_000)
    parser.add_argument("--shards-per-tier", type=int, default=8)
    parser.add_argument("--source-shards", type=int, nargs="+", default=DEFAULT_SOURCE_SHARDS)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--exclude",
        type=Path,
        nargs="*",
        default=[],
        help="EPD/FEN files whose evaluator-visible positions must not enter the sample",
    )
    arguments = parser.parse_args()
    manifest = sample_corpus(
        output=arguments.output,
        cache=arguments.cache,
        medium_positions=arguments.medium_positions,
        deep_positions=arguments.deep_positions,
        shards_per_tier=arguments.shards_per_tier,
        source_shards=arguments.source_shards,
        seed=arguments.seed,
        exclude_paths=arguments.exclude,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
