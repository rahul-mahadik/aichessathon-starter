"""Search-distilled AI Chessathon submission entrypoint."""

import json
import os
from collections.abc import Callable
from pathlib import Path

import chess

from frontier_search_engine import FrontierSearchEngine
from nnue_runtime import (
    BufferedIncrementalQuantizedEvaluator,
    IncrementalQuantizedEvaluator,
    QuantizedEvaluator,
)
from search_engine import SearchEngine, handcrafted_evaluation
from strong_search_engine import StrongSearchEngine

_WEIGHTS = Path(__file__).with_name("weights") / "nnue.npz"
_EVALUATOR_CONFIG = _WEIGHTS.with_name("evaluator.json")
_config: dict[str, object] = {}
if _EVALUATOR_CONFIG.exists():
    _config = json.loads(_EVALUATOR_CONFIG.read_text())

_search_mode = str(_config.get("search", os.environ.get("AICHESSATHON_SEARCH", "baseline")))
_runtime_mode = str(
    _config.get("runtime", os.environ.get("AICHESSATHON_NNUE_RUNTIME", "reference"))
)
if _search_mode not in {"baseline", "strong", "frontier"}:
    raise ValueError(f"unsupported search mode: {_search_mode}")
if _runtime_mode not in {"reference", "incremental", "buffered"}:
    raise ValueError(f"unsupported NNUE runtime mode: {_runtime_mode}")

if _WEIGHTS.exists():
    _antisymmetric = bool(_config.get("antisymmetric", False))
    evaluator_class = {
        "reference": QuantizedEvaluator,
        "incremental": IncrementalQuantizedEvaluator,
        "buffered": BufferedIncrementalQuantizedEvaluator,
    }[_runtime_mode]
    _learned = evaluator_class(_WEIGHTS, antisymmetric=_antisymmetric)
    _learned.warmup()
    _evaluator: Callable[[chess.Board], float] = _learned
else:
    _evaluator = handcrafted_evaluation

if _search_mode == "frontier":
    _search: SearchEngine | StrongSearchEngine = FrontierSearchEngine(_evaluator)
elif _search_mode == "strong":
    _search = StrongSearchEngine(_evaluator)
else:
    _search = SearchEngine(_evaluator)

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
        f"nodes={result.nodes} evals={getattr(result, 'evaluations', 'na')} "
        f"elapsed_ms={result.elapsed_ms} search={_search_mode} runtime={_runtime_mode}"
    )
    return result.move.uci()
