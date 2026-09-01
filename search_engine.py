"""Team-written iterative-deepening alpha-beta chess search."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import chess
import chess.polyglot

MATE = 30_000.0
INFINITY = 32_000.0
MAX_QUIESCENCE_PLY = 8
MAX_TABLE_ENTRIES = 150_000
EXACT = 0
LOWER = 1
UPPER = 2

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

Evaluator = Callable[[chess.Board], float]


class SearchTimeout(Exception):
    """Internal control flow for returning the last completed iteration."""


@dataclass(frozen=True)
class TableEntry:
    depth: int
    score: float
    bound: int
    move: chess.Move | None


@dataclass(frozen=True)
class SearchResult:
    move: chess.Move
    score: float
    depth: int
    nodes: int
    elapsed_ms: int


def handcrafted_evaluation(board: chess.Board) -> float:
    """Material and cheap positional terms, returned for the side to move."""
    white_score = 0.0
    non_pawn_material = 0
    bishops = {chess.WHITE: 0, chess.BLACK: 0}
    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        relative_rank = rank if piece.color == chess.WHITE else 7 - rank
        centre = 7.0 - (abs(file - 3.5) + abs(rank - 3.5))
        bonus = 0.0
        if piece.piece_type == chess.PAWN:
            bonus = 7.0 * relative_rank + 2.0 * centre
        elif piece.piece_type == chess.KNIGHT:
            bonus = 9.0 * centre
        elif piece.piece_type == chess.BISHOP:
            bonus = 5.0 * centre
            bishops[piece.color] += 1
        elif piece.piece_type == chess.ROOK:
            bonus = 12.0 if relative_rank == 6 else 0.0
        elif piece.piece_type == chess.QUEEN:
            bonus = 1.5 * centre
        if piece.piece_type not in (chess.PAWN, chess.KING):
            non_pawn_material += PIECE_VALUES[piece.piece_type]
        value = PIECE_VALUES[piece.piece_type] + bonus
        white_score += value if piece.color == chess.WHITE else -value

    if bishops[chess.WHITE] >= 2:
        white_score += 30.0
    if bishops[chess.BLACK] >= 2:
        white_score -= 30.0

    for colour in (chess.WHITE, chess.BLACK):
        king = board.king(colour)
        if king is None:
            continue
        king_rank = chess.square_rank(king)
        king_file = chess.square_file(king)
        centre = 7.0 - (abs(king_file - 3.5) + abs(king_rank - 3.5))
        king_bonus = 6.0 * centre if non_pawn_material < 2_000 else -8.0 * centre
        white_score += king_bonus if colour == chess.WHITE else -king_bonus

    return white_score if board.turn == chess.WHITE else -white_score


class SearchEngine:
    def __init__(self, evaluator: Evaluator = handcrafted_evaluation) -> None:
        self.evaluator = evaluator
        self.table: dict[int, TableEntry] = {}
        self.history: dict[tuple[bool, int, int], int] = {}
        self.killers: list[list[chess.Move]] = [[] for _ in range(128)]
        self.deadline = 0.0
        self.node_limit: int | None = None
        self.nodes = 0

    @staticmethod
    def _key(board: chess.Board) -> int:
        return chess.polyglot.zobrist_hash(board) ^ (min(board.halfmove_clock, 127) << 64)

    def _check_time(self) -> None:
        if self.node_limit is not None:
            if self.nodes >= self.node_limit:
                raise SearchTimeout
            return
        if self.nodes & 127 == 0 and time.monotonic() >= self.deadline:
            raise SearchTimeout

    @staticmethod
    def _terminal_score(board: chess.Board, ply: int) -> float:
        if board.is_check():
            return -MATE + ply
        return 0.0

    @staticmethod
    def _is_draw(board: chess.Board) -> bool:
        return board.is_repetition(3) or board.is_fifty_moves() or board.is_insufficient_material()

    def _move_score(
        self, board: chess.Board, move: chess.Move, tt_move: chess.Move | None, ply: int
    ) -> int:
        if move == tt_move:
            return 20_000_000
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            victim_value = PIECE_VALUES[chess.PAWN if victim is None else victim.piece_type]
            attacker = board.piece_at(move.from_square)
            attacker_value = PIECE_VALUES[attacker.piece_type] if attacker else 0
            return 10_000_000 + 16 * victim_value - attacker_value
        if move.promotion:
            return 9_000_000 + PIECE_VALUES[move.promotion]
        if ply < len(self.killers) and move in self.killers[ply]:
            return 8_000_000 - self.killers[ply].index(move)
        return self.history.get((board.turn, move.from_square, move.to_square), 0)

    def _ordered_moves(
        self,
        board: chess.Board,
        moves: list[chess.Move],
        tt_move: chess.Move | None,
        ply: int,
    ) -> list[chess.Move]:
        return sorted(
            moves,
            key=lambda move: self._move_score(board, move, tt_move, ply),
            reverse=True,
        )

    def _quiescence(
        self, board: chess.Board, alpha: float, beta: float, ply: int, qply: int
    ) -> float:
        self.nodes += 1
        self._check_time()
        if self._is_draw(board):
            return 0.0
        in_check = board.is_check()
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return self._terminal_score(board, ply)
        if qply >= MAX_QUIESCENCE_PLY:
            return self.evaluator(board)

        if not in_check:
            stand_pat = self.evaluator(board)
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
            moves = [
                move for move in legal_moves if board.is_capture(move) or move.promotion is not None
            ]
        else:
            moves = legal_moves

        for move in self._ordered_moves(board, moves, None, ply):
            board.push(move)
            try:
                score = -self._quiescence(board, -beta, -alpha, ply + 1, qply + 1)
            finally:
                board.pop()
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha

    def _negamax(
        self, board: chess.Board, depth: int, alpha: float, beta: float, ply: int
    ) -> float:
        self.nodes += 1
        self._check_time()
        if self._is_draw(board):
            return 0.0
        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply, 0)

        key = self._key(board)
        entry = self.table.get(key)
        if entry is not None and entry.depth >= depth:
            if entry.bound == EXACT:
                return entry.score
            if entry.bound == LOWER:
                alpha = max(alpha, entry.score)
            elif entry.bound == UPPER:
                beta = min(beta, entry.score)
            if alpha >= beta:
                return entry.score

        moves = list(board.legal_moves)
        if not moves:
            return self._terminal_score(board, ply)
        original_alpha = alpha
        best_score = -INFINITY
        best_move: chess.Move | None = None
        tt_move = entry.move if entry else None
        for move_index, move in enumerate(self._ordered_moves(board, moves, tt_move, ply)):
            quiet = not board.is_capture(move) and move.promotion is None
            board.push(move)
            try:
                if move_index == 0:
                    score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
                else:
                    score = -self._negamax(board, depth - 1, -alpha - 1.0, -alpha, ply + 1)
                    if alpha < score < beta:
                        score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
            if alpha >= beta:
                if quiet:
                    if ply < len(self.killers):
                        killers = self.killers[ply]
                        if move not in killers:
                            killers.insert(0, move)
                            del killers[2:]
                    history_key = (board.turn, move.from_square, move.to_square)
                    self.history[history_key] = self.history.get(history_key, 0) + depth * depth
                break

        bound = EXACT
        if best_score <= original_alpha:
            bound = UPPER
        elif best_score >= beta:
            bound = LOWER
        self.table[key] = TableEntry(depth, best_score, bound, best_move)
        return best_score

    def _root(
        self, board: chess.Board, depth: int, previous: chess.Move
    ) -> tuple[float, chess.Move]:
        moves = self._ordered_moves(board, list(board.legal_moves), previous, 0)
        alpha = -INFINITY
        beta = INFINITY
        best_move = moves[0]
        best_score = -INFINITY
        for move_index, move in enumerate(moves):
            board.push(move)
            try:
                if move_index == 0:
                    score = -self._negamax(board, depth - 1, -beta, -alpha, 1)
                else:
                    score = -self._negamax(board, depth - 1, -alpha - 1.0, -alpha, 1)
                    if alpha < score < beta:
                        score = -self._negamax(board, depth - 1, -beta, -alpha, 1)
            finally:
                board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
        return best_score, best_move

    @staticmethod
    def _budget_ms(time_left_ms: int) -> int:
        if time_left_ms <= 1_000:
            return max(10, int(time_left_ms * 0.08))
        target = int(time_left_ms / 40 + 300)
        return max(60, min(5_000, int(time_left_ms * 0.15), target))

    def _choose(
        self, board: chess.Board, *, budget_ms: int | None, node_limit: int | None
    ) -> SearchResult:
        moves = list(board.legal_moves)
        if not moves:
            raise ValueError("cannot choose a move in a terminal position")
        if len(self.table) > MAX_TABLE_ENTRIES:
            self.table.clear()
        started = time.monotonic()
        self.node_limit = node_limit
        self.deadline = (
            float("inf")
            if budget_ms is None
            else started + max(budget_ms - 10, 5) / 1_000.0
        )
        self.nodes = 0
        best_move = moves[0]
        best_score = -INFINITY
        completed_depth = 0
        for depth in range(1, 65):
            try:
                score, move = self._root(board, depth, best_move)
            except SearchTimeout:
                break
            best_move = move
            best_score = score
            completed_depth = depth
            if abs(score) >= MATE - 128:
                break
        elapsed_ms = int((time.monotonic() - started) * 1_000)
        return SearchResult(best_move, best_score, completed_depth, self.nodes, elapsed_ms)

    def choose(self, board: chess.Board, time_left_ms: int) -> SearchResult:
        return self._choose(
            board,
            budget_ms=self._budget_ms(time_left_ms),
            node_limit=None,
        )

    def choose_fixed_nodes(self, board: chess.Board, node_limit: int) -> SearchResult:
        if node_limit < 1:
            raise ValueError("node_limit must be positive")
        return self._choose(board, budget_ms=None, node_limit=node_limit)
