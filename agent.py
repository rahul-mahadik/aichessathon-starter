"""Search-distilled AI Chessathon submission entrypoint."""

import json
import os
from pathlib import Path

import chess

from nnue_runtime import QuantizedEvaluator
from search_engine import SearchEngine, handcrafted_evaluation

_WEIGHTS = Path(__file__).with_name("weights") / "nnue.npz"
_EVALUATOR_CONFIG = _WEIGHTS.with_name("evaluator.json")
if _WEIGHTS.exists():
    _antisymmetric = False
    if _EVALUATOR_CONFIG.exists():
        _config = json.loads(_EVALUATOR_CONFIG.read_text())
        _antisymmetric = bool(_config.get("antisymmetric", False))
    _learned = QuantizedEvaluator(_WEIGHTS, antisymmetric=_antisymmetric)
    _learned.warmup()
    _search = SearchEngine(_learned)
else:
    _search = SearchEngine(handcrafted_evaluation)

_FIXED_NODES = int(os.environ.get("AICHESSATHON_FIXED_NODES", "0"))
if _FIXED_NODES < 0:
    raise ValueError("AICHESSATHON_FIXED_NODES must be non-negative")

_game_board: chess.Board | None = None


def _synchronize(fen: str) -> chess.Board:
    """Preserve move-stack history when the next FEN follows the prior position."""
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
    """Return the last move from a completed iterative-deepening iteration."""
    board = _synchronize(fen)
    result = (
        _search.choose_fixed_nodes(board, _FIXED_NODES)
        if _FIXED_NODES
        else _search.choose(board, time_left_ms)
    )
    board.push(result.move)
    print(
        f"depth={result.depth} score={result.score:.0f} "
        f"nodes={result.nodes} elapsed_ms={result.elapsed_ms}"
    )
    return result.move.uci()
