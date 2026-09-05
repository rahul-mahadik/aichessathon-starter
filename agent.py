"""Search-distilled AI Chessathon submission entrypoint."""

import json
import os
import threading
from collections.abc import Callable
from pathlib import Path

import chess

from frontier_search_engine import FrontierSearchEngine
from nnue_runtime import (
    BufferedIncrementalQuantizedEvaluator,
    IncrementalQuantizedEvaluator,
    IntegerQuantizedEvaluator,
    PolicyQuantizedEvaluator,
    QuantizedEvaluator,
)
from ordered_search_engine import OrderedSearchEngine
from policy_search_engine import PolicySearchEngine
from search_engine import SearchEngine, handcrafted_evaluation
from see_search_engine import SeeSearchEngine
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
_ponder_enabled = bool(_config.get("ponder", False))
_ponder_max_ms = int(str(_config.get("ponder_max_ms", 60_000)))
_value_scale_cp = float(str(_config.get("value_scale_cp", 1_000.0)))
if not 1 <= _ponder_max_ms <= 120_000:
    raise ValueError("ponder_max_ms must be between 1 and 120000")
if _value_scale_cp <= 0:
    raise ValueError("value_scale_cp must be positive")
if _search_mode not in {"baseline", "strong", "ordered", "policy", "see", "frontier"}:
    raise ValueError(f"unsupported search mode: {_search_mode}")
if _runtime_mode not in {"reference", "integer", "incremental", "buffered"}:
    raise ValueError(f"unsupported NNUE runtime mode: {_runtime_mode}")


def _load_evaluator() -> Callable[[chess.Board], float]:
    if not _WEIGHTS.exists():
        return handcrafted_evaluation
    _antisymmetric = bool(_config.get("antisymmetric", False))
    evaluator_class: type[QuantizedEvaluator] = {
        "reference": QuantizedEvaluator,
        "integer": IntegerQuantizedEvaluator,
        "incremental": IncrementalQuantizedEvaluator,
        "buffered": BufferedIncrementalQuantizedEvaluator,
    }[_runtime_mode]
    if _search_mode == "policy":
        if _runtime_mode != "reference":
            raise ValueError("policy search currently requires reference runtime")
        evaluator_class = PolicyQuantizedEvaluator
    learned = evaluator_class(
        _WEIGHTS,
        antisymmetric=_antisymmetric,
        value_scale_cp=_value_scale_cp,
    )
    learned.warmup()
    return learned


def _make_search(evaluator: Callable[[chess.Board], float]) -> SearchEngine | StrongSearchEngine:
    if _search_mode == "policy":
        return PolicySearchEngine(evaluator)
    if _search_mode == "see":
        return SeeSearchEngine(evaluator)
    if _search_mode == "ordered":
        return OrderedSearchEngine(evaluator)
    if _search_mode == "frontier":
        return FrontierSearchEngine(evaluator)
    if _search_mode == "strong":
        return StrongSearchEngine(evaluator)
    return SearchEngine(evaluator)


_evaluator = _load_evaluator()
_search = _make_search(_evaluator)
_ponder_evaluator = _load_evaluator() if _ponder_enabled else None
_ponder_search: StrongSearchEngine | None = None
_ponder_thread: threading.Thread | None = None

_FIXED_NODES = int(os.environ.get("AICHESSATHON_FIXED_NODES", "0"))
if _FIXED_NODES < 0:
    raise ValueError("AICHESSATHON_FIXED_NODES must be non-negative")

_game_board: chess.Board | None = None


def _stop_ponder() -> None:
    """Stop the background engine and import only its completed search state."""
    global _ponder_search, _ponder_thread
    if _ponder_search is None or _ponder_thread is None:
        return
    _ponder_search.request_stop()
    _ponder_thread.join()
    if isinstance(_search, StrongSearchEngine):
        _search.absorb(_ponder_search)
    _ponder_search = None
    _ponder_thread = None


def _start_ponder(board: chess.Board) -> None:
    """Use the otherwise-idle process to search the opponent's whole position."""
    global _ponder_search, _ponder_thread
    if not _ponder_enabled or _ponder_evaluator is None:
        return
    candidate = _make_search(_ponder_evaluator)
    if not isinstance(candidate, StrongSearchEngine):
        return
    ponder_board = board.copy(stack=True)
    _ponder_search = candidate

    def run() -> None:
        try:
            candidate.choose_for_ms(ponder_board, _ponder_max_ms)
        except Exception as error:
            print(f"ponder stopped after error: {error}")

    _ponder_thread = threading.Thread(target=run, name="aichessathon-ponder", daemon=True)
    _ponder_thread.start()


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
    _stop_ponder()
    board = _synchronize(fen)
    result = (
        _search.choose_fixed_nodes(board, _FIXED_NODES)
        if _FIXED_NODES
        else _search.choose(board, time_left_ms)
    )
    board.push(result.move)
    _start_ponder(board)
    print(
        f"depth={result.depth} score={result.score:.0f} "
        f"nodes={result.nodes} evals={getattr(result, 'evaluations', 'na')} "
        f"elapsed_ms={result.elapsed_ms} search={_search_mode} runtime={_runtime_mode}"
    )
    return result.move.uci()
