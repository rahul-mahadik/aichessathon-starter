"""Experimental selective search layered on the known-good strong engine.

This candidate deliberately lives beside :mod:`strong_search_engine` so every
new pruning rule can be tested against the frozen implementation before it is
allowed into a submission build.
"""

from __future__ import annotations

import math

import chess

from strong_search_engine import (
    EXACT,
    INFINITY,
    LOWER,
    MATE,
    MAX_PLY,
    MAX_QPLY,
    PIECE_VALUES,
    UPPER,
    Evaluator,
    StrongSearchEngine,
)


def _build_lmr_table() -> tuple[tuple[int, ...], ...]:
    """Precompute conservative late-move reductions outside the hot loop."""
    rows: list[tuple[int, ...]] = []
    for depth in range(65):
        reductions: list[int] = []
        for move_number in range(65):
            if depth < 3 or move_number < 4:
                reductions.append(0)
            else:
                reductions.append(max(1, int(math.log(depth) * math.log(move_number) / 2.4)))
        rows.append(tuple(reductions))
    return tuple(rows)


LMR_TABLE = _build_lmr_table()
LMP_LIMITS = (0, 6, 10, 16)


class FrontierSearchEngine(StrongSearchEngine):
    """Ablation candidate with wider, still readable alpha-beta techniques.

    Additions over ``StrongSearchEngine`` are mate-distance pruning, razoring,
    shallow late-move pruning, adaptive LMR, capture-history ordering, limited
    ProbCut, and checks in the first quiescence ply.
    """

    def __init__(self, evaluator: Evaluator) -> None:
        super().__init__(evaluator)
        self.capture_history: dict[tuple[int, int, int], int] = {}

    @staticmethod
    def _capture_details(board: chess.Board, move: chess.Move) -> tuple[int, int]:
        victim = board.piece_at(move.to_square)
        victim_type = chess.PAWN if victim is None else victim.piece_type
        attacker = board.piece_at(move.from_square)
        attacker_type = chess.PAWN if attacker is None else attacker.piece_type
        promotion_gain = 0
        if move.promotion is not None:
            promotion_gain = PIECE_VALUES[move.promotion] - PIECE_VALUES[chess.PAWN]
        return PIECE_VALUES[victim_type] + promotion_gain, PIECE_VALUES[attacker_type]

    def _move_score(
        self,
        board: chess.Board,
        move: chess.Move,
        tt_move: chess.Move | None,
        ply: int,
    ) -> int:
        base = super()._move_score(board, move, tt_move, ply)
        if board.is_capture(move):
            attacker = board.piece_at(move.from_square)
            attacker_type = chess.PAWN if attacker is None else attacker.piece_type
            return base + self.capture_history.get(
                (attacker_type, move.from_square, move.to_square), 0
            )
        return base

    def _qsearch(self, board: chess.Board, alpha: int, beta: int, ply: int, qply: int) -> int:
        self._visit()
        if self._draw(board):
            return 0
        moves = list(board.legal_moves)
        if not moves:
            return self._terminal(board, ply)
        if qply >= MAX_QPLY or ply >= MAX_PLY - 1:
            return self._evaluate(board)

        in_check = board.is_check()
        stand_pat: int | None = None
        if in_check:
            candidates = moves
        else:
            stand_pat = self._evaluate(board)
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat
            candidates = [
                move
                for move in moves
                if board.is_capture(move)
                or move.promotion is not None
                or (qply == 0 and board.gives_check(move))
            ]

        for move in self._ordered_moves(board, candidates, None, ply):
            capture = board.is_capture(move)
            if not in_check and capture and move.promotion is None:
                gain, attacker = self._capture_details(board, move)
                if stand_pat is not None and stand_pat + gain + 150 < alpha:
                    continue
                # Only prune captures that are both materially implausible and
                # unable to recover alpha even with a generous tactical margin.
                if attacker > gain + 250 and stand_pat is not None and stand_pat + gain < alpha:
                    continue
            self._push(board, move)
            try:
                score = -self._qsearch(board, -beta, -alpha, ply + 1, qply + 1)
            finally:
                self._pop(board)
            if score >= beta:
                if capture:
                    attacker_piece = board.piece_at(move.from_square)
                    attacker_type = (
                        chess.PAWN if attacker_piece is None else attacker_piece.piece_type
                    )
                    key = (attacker_type, move.from_square, move.to_square)
                    self.capture_history[key] = min(
                        250_000, self.capture_history.get(key, 0) + 32
                    )
                return score
            if score > alpha:
                alpha = score
        return alpha

    def _negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        *,
        pv_node: bool,
        allow_null: bool,
    ) -> int:
        self._visit()
        if self._draw(board):
            return 0
        if ply >= MAX_PLY - 1:
            return self._evaluate(board)

        # Scores outside this interval cannot improve the distance to mate.
        alpha = max(alpha, -MATE + ply)
        beta = min(beta, MATE - ply - 1)
        if alpha >= beta:
            return alpha

        in_check = board.is_check()
        if in_check and depth > 0:
            depth += 1
        if depth <= 0:
            return self._qsearch(board, alpha, beta, ply, 0)

        key = self._key(board)
        entry = self.table.get(key)
        tt_move = entry.move if entry is not None else None
        original_alpha = alpha
        original_beta = beta
        if entry is not None and entry.depth >= depth:
            tt_score = self._from_tt(entry.score, ply)
            if entry.bound == EXACT:
                return tt_score
            if entry.bound == LOWER:
                alpha = max(alpha, tt_score)
            elif entry.bound == UPPER:
                beta = min(beta, tt_score)
            if alpha >= beta:
                return tt_score

        static_score = self._evaluate(board)
        narrow = beta - alpha <= 1
        if not pv_node and not in_check:
            # Razoring asks quiescence to verify that a shallow node is truly
            # too far below alpha before discarding its ordinary move tree.
            if depth <= 2 and static_score + 180 * depth < alpha:
                razor = self._qsearch(board, alpha - 1, alpha, ply, 0)
                if razor < alpha:
                    return razor
            if depth <= 3 and narrow and static_score - 105 * depth >= beta:
                return static_score
            if (
                allow_null
                and depth >= 3
                and narrow
                and static_score >= beta
                and self._has_non_pawn_material(board)
            ):
                reduction = 2 + depth // 4 + int(static_score >= beta + 180)
                self._push(board, chess.Move.null())
                try:
                    score = -self._negamax(
                        board,
                        max(0, depth - 1 - reduction),
                        -beta,
                        -beta + 1,
                        ply + 1,
                        pv_node=False,
                        allow_null=False,
                    )
                finally:
                    self._pop(board)
                if score >= beta:
                    return score

            # Limited ProbCut: only obviously non-losing captures are probes,
            # and only well away from mate scores.
            if depth >= 5 and narrow and beta < MATE - MAX_PLY:
                prob_beta = beta + 180
                captures = []
                for move in board.generate_legal_captures():
                    gain, attacker = self._capture_details(board, move)
                    if gain + 120 >= attacker:
                        captures.append(move)
                for move in self._ordered_moves(board, captures, tt_move, ply)[:4]:
                    self._push(board, move)
                    try:
                        score = -self._negamax(
                            board,
                            depth - 3,
                            -prob_beta,
                            -prob_beta + 1,
                            ply + 1,
                            pv_node=False,
                            allow_null=True,
                        )
                    finally:
                        self._pop(board)
                    if score >= prob_beta:
                        return score

        moves = list(board.legal_moves)
        if not moves:
            return self._terminal(board, ply)
        self._ordered_moves(board, moves, tt_move, ply)

        best_score = -INFINITY
        best_move: chess.Move | None = None
        searched = 0
        quiets_seen = 0
        for move_index, move in enumerate(moves):
            capture = board.is_capture(move)
            tactical = capture or move.promotion is not None
            gives_check = board.gives_check(move)
            quiet = not tactical and not gives_check
            if quiet:
                quiets_seen += 1

            if searched > 0 and not pv_node and not in_check and quiet:
                if depth <= 3 and quiets_seen > LMP_LIMITS[depth] and static_score <= alpha:
                    continue
                if depth == 1 and static_score + 125 <= alpha:
                    continue

            reduction = 0
            if depth >= 3 and move_index >= 4 and not pv_node and not in_check and quiet:
                reduction = LMR_TABLE[min(depth, 64)][min(move_index + 1, 64)]
                history_score = self.history.get(
                    (board.turn, move.from_square, move.to_square), 0
                )
                if history_score > 8_000:
                    reduction = max(0, reduction - 1)
                reduction = min(reduction, max(0, depth - 2))

            self._push(board, move)
            try:
                if searched == 0:
                    score = -self._negamax(
                        board,
                        depth - 1,
                        -beta,
                        -alpha,
                        ply + 1,
                        pv_node=pv_node,
                        allow_null=True,
                    )
                else:
                    score = -self._negamax(
                        board,
                        depth - 1 - reduction,
                        -alpha - 1,
                        -alpha,
                        ply + 1,
                        pv_node=False,
                        allow_null=True,
                    )
                    if reduction and score > alpha:
                        score = -self._negamax(
                            board,
                            depth - 1,
                            -alpha - 1,
                            -alpha,
                            ply + 1,
                            pv_node=False,
                            allow_null=True,
                        )
                    if alpha < score < beta:
                        score = -self._negamax(
                            board,
                            depth - 1,
                            -beta,
                            -alpha,
                            ply + 1,
                            pv_node=pv_node,
                            allow_null=True,
                        )
            finally:
                self._pop(board)

            searched += 1
            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                bonus = min(16_384, depth * depth * 16)
                if tactical:
                    attacker_piece = board.piece_at(move.from_square)
                    attacker_type = (
                        chess.PAWN if attacker_piece is None else attacker_piece.piece_type
                    )
                    history_key = (attacker_type, move.from_square, move.to_square)
                    self.capture_history[history_key] = min(
                        250_000, self.capture_history.get(history_key, 0) + bonus
                    )
                else:
                    killers = self.killers[ply]
                    if move not in killers:
                        killers.insert(0, move)
                        del killers[2:]
                    history_key = (board.turn, move.from_square, move.to_square)
                    self.history[history_key] = min(
                        2_000_000, self.history.get(history_key, 0) + bonus
                    )
                    if board.move_stack:
                        previous = board.peek()
                        self.countermoves[(previous.from_square, previous.to_square)] = move
                break

        if best_move is None:
            best_score = static_score
        bound = EXACT
        if best_score <= original_alpha:
            bound = UPPER
        elif best_score >= original_beta:
            bound = LOWER
        self._store(key, depth, best_score, bound, best_move, ply)
        return best_score

    def _trim_state(self) -> None:
        super()._trim_state()
        if len(self.capture_history) > 20_000:
            self.capture_history = {
                key: value // 2 for key, value in self.capture_history.items() if value > 64
            }
