"""Local-only Stockfish 18 control behind the Chessathon agent contract.

This file intentionally wraps a prohibited third-party engine. It exists only
for offline research and must never be packaged or uploaded as a submission.
"""

from __future__ import annotations

import atexit
import os
import shutil
from pathlib import Path

import chess
import chess.engine


def _stockfish_path() -> str:
    configured = os.environ.get("STOCKFISH_PATH")
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return str(path)
        raise RuntimeError(f"STOCKFISH_PATH does not exist: {path}")
    discovered = shutil.which("stockfish")
    if discovered:
        return discovered
    raise RuntimeError(
        "Stockfish is required for this local-only control; install it or set STOCKFISH_PATH"
    )


_engine = chess.engine.SimpleEngine.popen_uci(_stockfish_path())
_engine.configure(
    {
        "Threads": 1,
        "Hash": 128,
        "Skill Level": 20,
        "UCI_LimitStrength": False,
        "Move Overhead": 20,
    }
)
atexit.register(_engine.quit)

_fixed_nodes = int(os.environ.get("AICHESSATHON_FIXED_NODES", "0"))
if _fixed_nodes < 0:
    raise ValueError("AICHESSATHON_FIXED_NODES must be non-negative")

_game_board: chess.Board | None = None


def _synchronize(fen: str) -> chess.Board:
    global _game_board
    incoming = chess.Board(fen)
    if _game_board is None:
        _game_board = incoming
        return _game_board
    if _game_board.fen() == incoming.fen():
        return _game_board
    for move in list(_game_board.legal_moves):
        _game_board.push(move)
        if _game_board.fen() == incoming.fen():
            return _game_board
        _game_board.pop()
    _game_board = incoming
    return _game_board


def _budget_ms(time_left_ms: int) -> int:
    # Keep this identical to both team-written engines for equal-clock tests.
    if time_left_ms <= 1_000:
        return max(10, int(time_left_ms * 0.08))
    target = int(time_left_ms / 40 + 300)
    return max(60, min(5_000, int(time_left_ms * 0.15), target))


def get_move(fen: str, time_left_ms: int) -> str:
    board = _synchronize(fen)
    limit = (
        chess.engine.Limit(nodes=_fixed_nodes)
        if _fixed_nodes
        else chess.engine.Limit(time=max(0.001, (_budget_ms(time_left_ms) - 25) / 1_000))
    )
    result = _engine.play(board, limit, info=chess.engine.INFO_BASIC)
    if result.move is None:
        raise RuntimeError("Stockfish returned no move for a non-terminal position")
    board.push(result.move)
    print(
        "stockfish-local "
        f"depth={result.info.get('depth')} nodes={result.info.get('nodes')} "
        f"score={result.info.get('score')}"
    )
    return result.move.uci()
