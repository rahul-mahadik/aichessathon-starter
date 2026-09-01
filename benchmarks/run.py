"""Run paired, multi-position matches and write machine-readable results."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from harness.referee import FAILED_TERMINATIONS, Outcome, play_match
from harness.rules import PLY_CAP
from harness.sandbox import local

DEFAULT_POSITIONS = Path(__file__).with_name("openings.epd")


@dataclass(frozen=True)
class Position:
    name: str
    fen: str


@dataclass(frozen=True)
class GameRecord:
    round: int
    position: str
    candidate_colour: str
    result: str
    candidate_score: float
    termination: str
    elapsed_seconds: float
    pgn: str


@dataclass(frozen=True)
class GameSpec:
    round: int
    position: Position
    candidate: Path
    opponent: Path
    candidate_is_white: bool
    base_ms: int
    increment_ms: int
    ply_cap: int


def load_positions(path: Path) -> list[Position]:
    """Load full FENs with optional ``; id name;`` metadata."""
    positions: list[Position] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fen, _, operations = line.partition(";")
        try:
            import chess

            chess.Board(fen.strip())
        except ValueError as error:
            raise SystemExit(f"{path}:{line_number}: invalid FEN: {error}") from error
        name = f"position_{len(positions) + 1:02d}"
        for operation in operations.split(";"):
            key, _, value = operation.strip().partition(" ")
            if key == "id" and value:
                name = value.strip()
        positions.append(Position(name=name, fen=fen.strip()))
    if not positions:
        raise SystemExit(f"{path} contains no positions")
    return positions


def candidate_score(outcome: Outcome, candidate_is_white: bool) -> float:
    if outcome.result in {"draw", "void"}:
        return 0.5
    candidate_won = (outcome.result == "white") == candidate_is_white
    return 1.0 if candidate_won else 0.0


def git_metadata() -> dict[str, Any]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={Path.cwd().resolve()}", *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(git("status", "--porcelain")),
    }


def elo_from_score(score: float) -> float | None:
    if not 0.0 < score < 1.0:
        return None
    return 400.0 * math.log10(score / (1.0 - score))


def score_interval(scores: list[float], z_score: float = 1.96) -> tuple[float, float]:
    """Return a simple normal 95% interval over per-game scores."""
    mean = sum(scores) / len(scores)
    if len(scores) < 2:
        return mean, mean
    variance = sum((score - mean) ** 2 for score in scores) / (len(scores) - 1)
    margin = z_score * math.sqrt(variance / len(scores))
    return max(0.0, mean - margin), min(1.0, mean + margin)


def play_game(spec: GameSpec) -> GameRecord:
    white = spec.candidate if spec.candidate_is_white else spec.opponent
    black = spec.opponent if spec.candidate_is_white else spec.candidate
    started = time.monotonic()
    outcome = play_match(
        local(white),
        local(black),
        spec.base_ms,
        spec.increment_ms,
        ply_cap=spec.ply_cap,
        start_fen=spec.position.fen,
    )
    return GameRecord(
        round=spec.round,
        position=spec.position.name,
        candidate_colour="white" if spec.candidate_is_white else "black",
        result=outcome.result,
        candidate_score=candidate_score(outcome, spec.candidate_is_white),
        termination=outcome.termination,
        elapsed_seconds=round(time.monotonic() - started, 3),
        pgn=outcome.pgn,
    )


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    candidate = arguments.agent.resolve()
    opponent = arguments.opponent.resolve()
    positions = load_positions(arguments.positions)
    specs = [
        GameSpec(
            round=round_number,
            position=position,
            candidate=candidate,
            opponent=opponent,
            candidate_is_white=candidate_is_white,
            base_ms=arguments.base_ms,
            increment_ms=arguments.increment_ms,
            ply_cap=arguments.ply_cap,
        )
        for round_number in range(1, arguments.rounds + 1)
        for position in positions
        for candidate_is_white in (True, False)
    ]
    if arguments.workers == 1:
        records = map(play_game, specs)
    else:
        # Each game does its work in two child processes, so threads only coordinate I/O and
        # avoid adding another process layer around the platform-like runner.
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=arguments.workers)
        records = executor.map(play_game, specs)
    games: list[GameRecord] = []
    try:
        for record in records:
            games.append(record)
            print(
                f"round {record.round} {record.position} {record.candidate_colour}: "
                f"{record.result} by {record.termination}"
            )
    finally:
        if arguments.workers != 1:
            executor.shutdown()

    game_scores = [game.candidate_score for game in games]
    score = sum(game_scores) / len(game_scores)
    score_low, score_high = score_interval(game_scores)
    wins = sum(game.candidate_score == 1.0 for game in games)
    draws = sum(game.candidate_score == 0.5 for game in games)
    losses = sum(game.candidate_score == 0.0 for game in games)
    failures = sum(game.termination in FAILED_TERMINATIONS for game in games)
    return {
        "metadata": {
            **git_metadata(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "aws_instance_id": os.environ.get("AWS_EC2_INSTANCE_ID"),
            "aws_instance_type": os.environ.get("AWS_EC2_INSTANCE_TYPE"),
            "aws_ami_id": os.environ.get("AWS_EC2_AMI_ID"),
            "candidate": str(candidate),
            "opponent": str(opponent),
            "candidate_model": os.environ.get("BENCH_CANDIDATE_MODEL"),
            "candidate_model_sha256": os.environ.get("BENCH_CANDIDATE_SHA256"),
            "opponent_model": os.environ.get("BENCH_OPPONENT_MODEL"),
            "opponent_model_sha256": os.environ.get("BENCH_OPPONENT_SHA256"),
            "positions": str(arguments.positions),
            "rounds": arguments.rounds,
            "base_ms": arguments.base_ms,
            "increment_ms": arguments.increment_ms,
            "ply_cap": arguments.ply_cap,
            "workers": arguments.workers,
        },
        "summary": {
            "games": len(games),
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "score": score,
            "score_95pct_interval": [score_low, score_high],
            "estimated_elo_difference": elo_from_score(score),
            "estimated_elo_95pct_interval": [
                elo_from_score(score_low),
                elo_from_score(score_high),
            ],
            "failed_terminations": failures,
        },
        "games": [asdict(game) for game in games],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--opponent", type=Path, default=Path("baselines/minimax"))
    parser.add_argument("--positions", type=Path, default=DEFAULT_POSITIONS)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--base-ms", type=int, default=10_000)
    parser.add_argument("--increment-ms", type=int, default=100)
    parser.add_argument("--ply-cap", type=int, default=PLY_CAP)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/latest.json"))
    arguments = parser.parse_args()
    if arguments.rounds < 1 or arguments.workers < 1:
        parser.error("--rounds and --workers must be positive")

    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(variable, "1")

    report = run(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    summary = report["summary"]
    print(
        f"\n+{summary['wins']} ={summary['draws']} -{summary['losses']}, "
        f"score {summary['score']:.1%}"
    )
    print(f"report: {arguments.output}")
    if summary["failed_terminations"]:
        raise SystemExit("benchmark contained a crash, illegal move, flag, or init failure")


if __name__ == "__main__":
    main()
