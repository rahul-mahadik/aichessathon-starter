"""Build a deterministic near-equal opening suite from held-out teacher records."""

from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path
from typing import TextIO


def _open(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def select_positions(
    paths: list[Path],
    *,
    count: int,
    max_abs_cp: int,
    max_fullmove: int,
    seed: int,
) -> list[tuple[str, int]]:
    candidates: list[tuple[str, int]] = []
    seen: set[str] = set()
    for path in paths:
        with _open(path) as source:
            for line in source:
                payload = json.loads(line)
                fen = str(payload["fen"])
                cp = payload["root_score"].get("cp")
                if cp is None or abs(int(cp)) > max_abs_cp:
                    continue
                fields = fen.split()
                if len(fields) != 6 or int(fields[5]) > max_fullmove or fen in seen:
                    continue
                seen.add(fen)
                candidates.append((fen, int(cp)))
    if len(candidates) < count:
        raise SystemExit(f"only {len(candidates)} eligible positions for requested count {count}")
    random.Random(seed).shuffle(candidates)
    return candidates[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--max-abs-cp", type=int, default=50)
    parser.add_argument("--max-fullmove", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260904)
    arguments = parser.parse_args()
    if min(arguments.count, arguments.max_abs_cp, arguments.max_fullmove) < 1:
        parser.error("count and filters must be positive")

    selected = select_positions(
        arguments.inputs,
        count=arguments.count,
        max_abs_cp=arguments.max_abs_cp,
        max_fullmove=arguments.max_fullmove,
        seed=arguments.seed,
    )
    lines = [
        "# Deterministic near-equal calibration suite; held out from student training.",
        f"# count={arguments.count} max_abs_cp={arguments.max_abs_cp} "
        f"max_fullmove={arguments.max_fullmove} seed={arguments.seed}",
    ]
    lines.extend(
        f"{fen} ; id calibration_{index:03d}_cp_{cp:+d};"
        for index, (fen, cp) in enumerate(selected, 1)
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
