"""Submission-shaped entrypoint for the non-distillation classical control."""

from __future__ import annotations

import os

import chess
from classical_engine import ClassicalEngine

_engine = ClassicalEngine()
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


def get_move(fen: str, time_left_ms: int) -> str:
    board = _synchronize(fen)
    result = (
        _engine.choose_fixed_nodes(board, _fixed_nodes)
        if _fixed_nodes
        else _engine.choose(board, time_left_ms)
    )
    board.push(result.move)
    print(
        f"classical depth={result.depth} score={result.score} "
        f"nodes={result.nodes} elapsed_ms={result.elapsed_ms}"
    )
    return result.move.uci()
