"""Team-written strong PVS search with a pluggable position evaluator.

This is an isolated post-distillation experiment. The original ``search_engine``
remains frozen so Phase D evaluator comparisons keep their causal baseline.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import chess
import chess.polyglot

MATE = 30_000
INFINITY = 32_000
MAX_PLY = 96
MAX_QPLY = 10
MAX_TABLE_ENTRIES = 180_000

EXACT = 0
LOWER = 1
UPPER = 2

PIECE_VALUES = (0, 100, 320, 335, 500, 950, 20_000)
Evaluator = Callable[[chess.Board], float]


class SearchTimeout(Exception):
    """Stop the current iteration while preserving its caller's board."""


@dataclass(slots=True)
class TTEntry:
    depth: int
    score: int
    bound: int
    move: chess.Move | None
    generation: int


@dataclass(frozen=True, slots=True)
class SearchResult:
    move: chess.Move
    score: int
    depth: int
    nodes: int
    evaluations: int
    elapsed_ms: int


class StrongSearchEngine:
    """Iterative-deepening PVS whose leaf evaluator is supplied by the caller."""

    def __init__(self, evaluator: Evaluator) -> None:
        self.evaluator = evaluator
        self._begin_evaluation = getattr(evaluator, "begin", None)
        self._push_evaluation = getattr(evaluator, "push", None)
        self._pop_evaluation = getattr(evaluator, "pop", None)
        self.table: dict[int, TTEntry] = {}
        self.history: dict[tuple[bool, int, int], int] = {}
        self.killers: list[list[chess.Move]] = [[] for _ in range(MAX_PLY)]
        self.countermoves: dict[tuple[int, int], chess.Move] = {}
        self.deadline = float("inf")
        self.node_limit: int | None = None
        self.nodes = 0
        self.evaluations = 0
        self.generation = 0
        self._stop_requested = False

    def _evaluate(self, board: chess.Board) -> int:
        self.evaluations += 1
        return round(self.evaluator(board))

    def _push(self, board: chess.Board, move: chess.Move) -> None:
        if self._push_evaluation is None:
            board.push(move)
            return
        self._push_evaluation(board, move)
        try:
            board.push(move)
        except Exception:
            if self._pop_evaluation is not None:
                self._pop_evaluation()
            raise

    def _pop(self, board: chess.Board) -> None:
        if self._pop_evaluation is not None:
            self._pop_evaluation()
        board.pop()

    @staticmethod
    def _key(board: chess.Board) -> int:
        return chess.polyglot.zobrist_hash(board) ^ (min(board.halfmove_clock, 127) << 64)

    def _visit(self) -> None:
        self.nodes += 1
        if self._stop_requested:
            raise SearchTimeout
        if self.node_limit is not None:
            if self.nodes >= self.node_limit:
                raise SearchTimeout
        elif self.nodes & 127 == 0 and time.monotonic() >= self.deadline:
            raise SearchTimeout

    @staticmethod
    def _terminal(board: chess.Board, ply: int) -> int:
        return -MATE + ply if board.is_check() else 0

    @staticmethod
    def _draw(board: chess.Board) -> bool:
        if board.is_fifty_moves() or board.is_insufficient_material():
            return True
        return board.halfmove_clock >= 8 and board.is_repetition(3)

    @staticmethod
    def _to_tt(score: int, ply: int) -> int:
        if score >= MATE - MAX_PLY:
            return score + ply
        if score <= -MATE + MAX_PLY:
            return score - ply
        return score

    @staticmethod
    def _from_tt(score: int, ply: int) -> int:
        if score >= MATE - MAX_PLY:
            return score - ply
        if score <= -MATE + MAX_PLY:
            return score + ply
        return score

    @staticmethod
    def _has_non_pawn_material(board: chess.Board) -> bool:
        side = board.turn
        return bool(
            board.knights & board.occupied_co[side]
            or board.bishops & board.occupied_co[side]
            or board.rooks & board.occupied_co[side]
            or board.queens & board.occupied_co[side]
        )

    def _move_score(
        self,
        board: chess.Board,
        move: chess.Move,
        tt_move: chess.Move | None,
        ply: int,
    ) -> int:
        if move == tt_move:
            return 30_000_000
        if move.promotion is not None:
            return 20_000_000 + PIECE_VALUES[move.promotion]
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            victim_type = chess.PAWN if victim is None else victim.piece_type
            attacker = board.piece_at(move.from_square)
            attacker_type = chess.PAWN if attacker is None else attacker.piece_type
            return 10_000_000 + 16 * PIECE_VALUES[victim_type] - PIECE_VALUES[attacker_type]
        if ply < MAX_PLY and move in self.killers[ply]:
            return 8_000_000 - self.killers[ply].index(move)
        if board.move_stack:
            previous = board.peek()
            if self.countermoves.get((previous.from_square, previous.to_square)) == move:
                return 7_000_000
        return self.history.get((board.turn, move.from_square, move.to_square), 0)

    def _ordered_moves(
        self,
        board: chess.Board,
        moves: list[chess.Move],
        tt_move: chess.Move | None,
        ply: int,
    ) -> list[chess.Move]:
        moves.sort(key=lambda move: self._move_score(board, move, tt_move, ply), reverse=True)
        return moves

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
                move for move in moves if board.is_capture(move) or move.promotion is not None
            ]

        for move in self._ordered_moves(board, candidates, None, ply):
            if not in_check and move.promotion is None:
                victim = board.piece_at(move.to_square)
                gain = PIECE_VALUES[chess.PAWN if victim is None else victim.piece_type]
                if stand_pat is not None and stand_pat + gain + 180 < alpha:
                    continue
            self._push(board, move)
            try:
                score = -self._qsearch(board, -beta, -alpha, ply + 1, qply + 1)
            finally:
                self._pop(board)
            if score >= beta:
                return score
            if score > alpha:
                alpha = score
        return alpha

    def _store(
        self,
        key: int,
        depth: int,
        score: int,
        bound: int,
        move: chess.Move | None,
        ply: int,
    ) -> None:
        old = self.table.get(key)
        if old is None or depth >= old.depth or old.generation != self.generation:
            self.table[key] = TTEntry(
                depth=depth,
                score=self._to_tt(score, ply),
                bound=bound,
                move=move,
                generation=self.generation,
            )

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
            if depth <= 3 and narrow and static_score - 110 * depth >= beta:
                return static_score
            if (
                allow_null
                and depth >= 3
                and narrow
                and static_score >= beta
                and self._has_non_pawn_material(board)
            ):
                reduction = 2 + depth // 5
                self._push(board, chess.Move.null())
                try:
                    score = -self._negamax(
                        board,
                        depth - 1 - reduction,
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

        moves = list(board.legal_moves)
        if not moves:
            return self._terminal(board, ply)
        self._ordered_moves(board, moves, tt_move, ply)

        best_score = -INFINITY
        best_move: chess.Move | None = None
        searched = 0
        for move_index, move in enumerate(moves):
            capture = board.is_capture(move)
            tactical = capture or move.promotion is not None
            gives_check = board.gives_check(move)
            if (
                searched > 0
                and depth == 1
                and not pv_node
                and not in_check
                and not tactical
                and not gives_check
                and static_score + 130 <= alpha
            ):
                continue

            reduction = 0
            if (
                depth >= 3
                and move_index >= 4
                and not pv_node
                and not in_check
                and not tactical
                and not gives_check
            ):
                reduction = 1 + int(depth >= 6 and move_index >= 10)

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
                if not tactical:
                    killers = self.killers[ply]
                    if move not in killers:
                        killers.insert(0, move)
                        del killers[2:]
                    history_key = (board.turn, move.from_square, move.to_square)
                    self.history[history_key] = min(
                        2_000_000,
                        self.history.get(history_key, 0) + depth * depth * 16,
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

    def _root(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        previous: chess.Move,
    ) -> tuple[int, chess.Move]:
        entry = self.table.get(self._key(board))
        tt_move = entry.move if entry is not None else previous
        moves = self._ordered_moves(board, list(board.legal_moves), tt_move, 0)
        best_move = moves[0]
        best_score = -INFINITY
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
            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        return best_score, best_move

    @staticmethod
    def _budget_ms(time_left_ms: int) -> int:
        # Match the experimental engine exactly so wall-clock comparisons isolate
        # engine architecture instead of silently testing two time managers.
        if time_left_ms <= 1_000:
            return max(10, int(time_left_ms * 0.08))
        target = int(time_left_ms / 40 + 300)
        return max(60, min(5_000, int(time_left_ms * 0.15), target))

    def _trim_state(self) -> None:
        if len(self.table) > MAX_TABLE_ENTRIES:
            cutoff = self.generation - 2
            self.table = {
                key: entry for key, entry in self.table.items() if entry.generation >= cutoff
            }
            if len(self.table) > MAX_TABLE_ENTRIES:
                self.table.clear()
        if len(self.history) > 30_000:
            self.history = {key: value // 2 for key, value in self.history.items() if value > 64}

    def _choose(
        self,
        board: chess.Board,
        *,
        budget_ms: int | None,
        node_limit: int | None,
    ) -> SearchResult:
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("cannot choose a move in a terminal position")
        started = time.monotonic()
        self._stop_requested = False
        self.generation += 1
        self._trim_state()
        self.node_limit = node_limit
        self.deadline = (
            float("inf")
            if budget_ms is None
            else started + max(2, budget_ms - 12) / 1_000.0
        )
        self.nodes = 0
        self.evaluations = 0
        if self._begin_evaluation is not None:
            self._begin_evaluation(board)
        best_move = legal[0]
        best_score = -INFINITY
        completed_depth = 0

        for depth in range(1, 65):
            if completed_depth >= 2:
                window = 35
                alpha = max(-INFINITY, best_score - window)
                beta = min(INFINITY, best_score + window)
            else:
                alpha, beta = -INFINITY, INFINITY
            try:
                score, move = self._root(board, depth, alpha, beta, best_move)
                if score <= alpha or score >= beta:
                    score, move = self._root(board, depth, -INFINITY, INFINITY, best_move)
            except SearchTimeout:
                break
            best_score = score
            best_move = move
            completed_depth = depth
            if abs(score) >= MATE - MAX_PLY:
                break

        elapsed_ms = int((time.monotonic() - started) * 1_000)
        return SearchResult(
            best_move,
            best_score,
            completed_depth,
            self.nodes,
            self.evaluations,
            elapsed_ms,
        )

    def choose(self, board: chess.Board, time_left_ms: int) -> SearchResult:
        return self._choose(board, budget_ms=self._budget_ms(time_left_ms), node_limit=None)

    def choose_fixed_nodes(self, board: chess.Board, node_limit: int) -> SearchResult:
        if node_limit < 1:
            raise ValueError("node_limit must be positive")
        return self._choose(board, budget_ms=None, node_limit=node_limit)

    def choose_for_ms(self, board: chess.Board, budget_ms: int) -> SearchResult:
        """Search for an explicit wall-time budget, primarily for pondering."""
        if budget_ms < 1:
            raise ValueError("budget_ms must be positive")
        return self._choose(board, budget_ms=budget_ms, node_limit=None)

    def request_stop(self) -> None:
        """Ask an active search to stop at its next visited node."""
        self._stop_requested = True

    def absorb(self, other: StrongSearchEngine) -> None:
        """Import completed, position-keyed knowledge from an idle engine."""
        for key, entry in other.table.items():
            current = self.table.get(key)
            if current is None or entry.depth > current.depth:
                self.table[key] = TTEntry(
                    depth=entry.depth,
                    score=entry.score,
                    bound=entry.bound,
                    move=entry.move,
                    generation=self.generation,
                )
        for history_key, value in other.history.items():
            self.history[history_key] = max(self.history.get(history_key, 0), value)
        for counter_key, move in other.countermoves.items():
            self.countermoves.setdefault(counter_key, move)
        self._trim_state()
