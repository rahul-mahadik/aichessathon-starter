"""Annotate FEN/EPD positions with fixed-budget Stockfish MultiPV analysis."""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import chess
import chess.engine

from distill.schema import Candidate, TeacherRecord, TeacherScore, write_records

MATE_SCORE_CP = 100_000


def load_fens(path: Path, limit: int | None = None) -> Iterator[str]:
    """Read FEN or EPD lines, ignoring comments and EPD operations."""
    emitted = 0
    for line_number, raw_line in enumerate(path.read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        position = line.partition(";")[0].strip()
        fields = position.split()
        if len(fields) == 4:
            position = f"{position} 0 1"
        try:
            board = chess.Board(position)
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: invalid FEN/EPD: {error}") from error
        if board.is_game_over():
            continue
        yield board.fen()
        emitted += 1
        if limit is not None and emitted >= limit:
            return


def _score(info: dict[str, Any], turn: chess.Color) -> TeacherScore:
    pov_score = info["score"].pov(turn)
    cp = pov_score.score()
    mate = pov_score.mate()
    raw_wdl = info.get("wdl")
    wdl: tuple[int, int, int] | None = None
    if raw_wdl is not None:
        relative_wdl = raw_wdl.pov(turn)
        wdl = (relative_wdl.wins, relative_wdl.draws, relative_wdl.losses)
    return TeacherScore(cp=cp, mate=mate, wdl=wdl)


def _candidate(info: dict[str, Any], board: chess.Board, keep_pv: bool) -> Candidate:
    pv_moves = tuple(move.uci() for move in info.get("pv", ()))
    if not pv_moves:
        raise ValueError("Stockfish returned a MultiPV line without a principal variation")
    return Candidate(
        move=pv_moves[0],
        score=_score(info, board.turn),
        pv=pv_moves if keep_pv else (),
        depth=info.get("depth"),
        seldepth=info.get("seldepth"),
        nodes=info.get("nodes"),
    )


def annotate(
    engine: chess.engine.SimpleEngine,
    fens: Iterator[str],
    *,
    teacher: str,
    nodes: int | None,
    depth: int | None,
    multipv: int,
    keep_pv: bool,
) -> Iterator[TeacherRecord]:
    limit = chess.engine.Limit(nodes=nodes, depth=depth)
    for index, fen in enumerate(fens, 1):
        board = chess.Board(fen)
        raw = engine.analyse(board, limit, multipv=multipv, info=chess.engine.INFO_ALL)
        lines = raw if isinstance(raw, list) else [raw]
        candidates = tuple(_candidate(info, board, keep_pv) for info in lines)
        yield TeacherRecord(
            fen=fen,
            teacher=teacher,
            node_budget=nodes,
            depth_budget=depth,
            multipv=multipv,
            root_score=candidates[0].score,
            candidates=candidates,
            best_move=candidates[0].move,
        )
        print(f"annotated {index}: {fen} -> {candidates[0].move}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="FEN/EPD input file")
    parser.add_argument("--output", type=Path, required=True, help=".jsonl or .jsonl.gz shard")
    parser.add_argument(
        "--stockfish", default=os.environ.get("STOCKFISH_PATH", "stockfish")
    )
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--nodes", type=int)
    budget.add_argument("--depth", type=int)
    parser.add_argument("--multipv", type=int, default=8)
    parser.add_argument("--hash-mb", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--keep-pv", action="store_true")
    arguments = parser.parse_args()
    if arguments.multipv < 1 or arguments.hash_mb < 1:
        parser.error("--multipv and --hash-mb must be positive")

    with chess.engine.SimpleEngine.popen_uci(arguments.stockfish) as engine:
        options: dict[str, int | bool] = {"Threads": 1, "Hash": arguments.hash_mb}
        if "UCI_ShowWDL" in engine.options:
            options["UCI_ShowWDL"] = True
        engine.configure(options)
        teacher = " ".join(
            value for value in (engine.id.get("name"), engine.id.get("author")) if value
        )
        records = annotate(
            engine,
            load_fens(arguments.input, arguments.limit),
            teacher=teacher,
            nodes=arguments.nodes,
            depth=arguments.depth,
            multipv=arguments.multipv,
            keep_pv=arguments.keep_pv,
        )
        count = write_records(arguments.output, records)
    print(f"wrote {count} records to {arguments.output}")


if __name__ == "__main__":
    main()

