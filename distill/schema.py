"""Versioned JSONL schema for raw Stockfish teacher records."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TeacherScore:
    """A raw score from the root side-to-move's point of view."""

    cp: int | None
    mate: int | None
    wdl: tuple[int, int, int] | None

    def __post_init__(self) -> None:
        if self.cp is None and self.mate is None:
            raise ValueError("a teacher score needs cp or mate")
        if self.wdl is not None and (len(self.wdl) != 3 or min(self.wdl) < 0):
            raise ValueError("wdl must be non-negative (wins, draws, losses)")


@dataclass(frozen=True)
class Candidate:
    """One MultiPV candidate, scored from the root side-to-move's point of view."""

    move: str
    score: TeacherScore
    pv: tuple[str, ...]
    depth: int | None
    seldepth: int | None
    nodes: int | None


@dataclass(frozen=True)
class TeacherRecord:
    """One root position and its fixed-budget MultiPV analysis."""

    fen: str
    teacher: str
    node_budget: int | None
    depth_budget: int | None
    multipv: int
    root_score: TeacherScore
    candidates: tuple[Candidate, ...]
    best_move: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported teacher schema {self.schema_version}")
        if (self.node_budget is None) == (self.depth_budget is None):
            raise ValueError("exactly one of node_budget or depth_budget is required")
        if not self.candidates:
            raise ValueError("a teacher record needs at least one candidate")
        if self.multipv < len(self.candidates):
            raise ValueError("record contains more candidates than requested MultiPV")
        if self.best_move != self.candidates[0].move:
            raise ValueError("best_move must match the first MultiPV candidate")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TeacherRecord:
        def score(value: dict[str, Any]) -> TeacherScore:
            raw_wdl = value.get("wdl")
            wdl = tuple(int(item) for item in raw_wdl) if raw_wdl is not None else None
            return TeacherScore(cp=value.get("cp"), mate=value.get("mate"), wdl=wdl)

        candidates = tuple(
            Candidate(
                move=item["move"],
                score=score(item["score"]),
                pv=tuple(item.get("pv", ())),
                depth=item.get("depth"),
                seldepth=item.get("seldepth"),
                nodes=item.get("nodes"),
            )
            for item in payload["candidates"]
        )
        return cls(
            fen=payload["fen"],
            teacher=payload["teacher"],
            node_budget=payload.get("node_budget"),
            depth_budget=payload.get("depth_budget"),
            multipv=int(payload["multipv"]),
            root_score=score(payload["root_score"]),
            candidates=candidates,
            best_move=payload["best_move"],
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
        )


def _open_text(path: Path, mode: str) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def read_records(paths: Iterable[Path]) -> Iterator[TeacherRecord]:
    """Stream teacher records from plain or gzip-compressed JSONL shards."""
    for path in paths:
        with _open_text(path, "rt") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    yield TeacherRecord.from_dict(payload)
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    message = f"{path}:{line_number}: invalid teacher record: {error}"
                    raise ValueError(message) from error


def write_records(path: Path, records: Iterable[TeacherRecord]) -> int:
    """Write one JSON object per line and return the number of records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with _open_text(path, "wt") as destination:
        for record in records:
            destination.write(json.dumps(record.to_dict(), separators=(",", ":")) + "\n")
            count += 1
    return count
