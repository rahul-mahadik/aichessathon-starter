"""Strong search with full root-score ordering across iterative deepening.

The frozen strong engine remembers only the best move from the preceding
iteration. This isolated challenger retains the score of every searched root
move, so later iterations examine the entire prior root ranking first without
spending an extra evaluator call.
"""

from __future__ import annotations

import chess

from strong_search_engine import INFINITY, Evaluator, StrongSearchEngine


class OrderedSearchEngine(StrongSearchEngine):
    """Reuse completed root scores as the next iteration's move ordering."""

    def __init__(self, evaluator: Evaluator) -> None:
        super().__init__(evaluator)
        self._root_position_key: int | None = None
        self._root_scores: dict[chess.Move, int] = {}

    def _root_prior(self, board: chess.Board, moves: list[chess.Move]) -> dict[chess.Move, float]:
        return {}

    def _root(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        previous: chess.Move,
    ) -> tuple[int, chess.Move]:
        root_key = self._key(board)
        if root_key != self._root_position_key:
            self._root_position_key = root_key
            self._root_scores.clear()

        entry = self.table.get(root_key)
        tt_move = entry.move if entry is not None else previous
        moves = self._ordered_moves(board, list(board.legal_moves), tt_move, 0)
        priors = self._root_prior(board, moves)
        # Python's sort is stable, retaining the strong engine's tactical,
        # history, and TT ordering for moves without a prior root score.
        moves.sort(
            key=lambda move: (
                move == tt_move,
                self._root_scores.get(move, -INFINITY),
                priors.get(move, -float("inf")),
            ),
            reverse=True,
        )

        best_move = moves[0]
        best_score = -INFINITY
        iteration_scores: dict[chess.Move, int] = {}
        for move_index, move in enumerate(moves):
            self._push(board, move)
            try:
                if move_index == 0:
                    score = -self._negamax(
                        board,
                        depth - 1,
                        -beta,
                        -alpha,
                        1,
                        pv_node=True,
                        allow_null=True,
                    )
                else:
                    score = -self._negamax(
                        board,
                        depth - 1,
                        -alpha - 1,
                        -alpha,
                        1,
                        pv_node=False,
                        allow_null=True,
                    )
                    if alpha < score < beta:
                        score = -self._negamax(
                            board,
                            depth - 1,
                            -beta,
                            -alpha,
                            1,
                            pv_node=True,
                            allow_null=True,
                        )
            finally:
                self._pop(board)
            iteration_scores[move] = score
            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        self._root_scores.update(iteration_scores)
        return best_score, best_move
