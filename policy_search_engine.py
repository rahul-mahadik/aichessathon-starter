"""Root move ordering driven by a distilled Stockfish policy head."""

from __future__ import annotations

from collections.abc import Callable

import chess

from ordered_search_engine import OrderedSearchEngine


class PolicySearchEngine(OrderedSearchEngine):
    """Use policy priors before iterative deepening has search scores."""

    def __init__(self, evaluator: Callable[[chess.Board], float]) -> None:
        super().__init__(evaluator)
        scorer = getattr(evaluator, "policy_scores", None)
        if scorer is None:
            raise TypeError("policy search requires an evaluator with policy_scores")
        self._policy_scorer = scorer
        self._policy_key: int | None = None
        self._policy_scores: dict[chess.Move, float] = {}

    def _root_prior(self, board: chess.Board, moves: list[chess.Move]) -> dict[chess.Move, float]:
        key = self._key(board)
        if key != self._policy_key:
            self._policy_key = key
            self._policy_scores = self._policy_scorer(board, moves)
        return self._policy_scores
