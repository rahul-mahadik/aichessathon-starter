"""Combine distilled root policy ordering with SEE capture ordering."""

from __future__ import annotations

from collections.abc import Callable

import chess

from see_search_engine import SeeSearchEngine


class PolicySeeSearchEngine(SeeSearchEngine):
    """Use policy priors at the root and SEE throughout the move tree."""

    def __init__(self, evaluator: Callable[[chess.Board], float]) -> None:
        super().__init__(evaluator)
        scorer = getattr(evaluator, "policy_scores", None)
        if scorer is None:
            raise TypeError("policy-SEE search requires an evaluator with policy_scores")
        self._policy_scorer = scorer
        self._policy_key: int | None = None
        self._policy_scores: dict[chess.Move, float] = {}

    def _root_prior(self, board: chess.Board, moves: list[chess.Move]) -> dict[chess.Move, float]:
        key = self._key(board)
        if key != self._policy_key:
            self._policy_key = key
            self._policy_scores = self._policy_scorer(board, moves)
        return self._policy_scores
