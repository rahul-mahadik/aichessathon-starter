"""Root-ordered search with static-exchange capture ordering."""

from __future__ import annotations

import chess

from ordered_search_engine import OrderedSearchEngine
from strong_search_engine import PIECE_VALUES, Evaluator


class SeeSearchEngine(OrderedSearchEngine):
    """Order losing captures after quiet heuristics without pruning them."""

    def __init__(self, evaluator: Evaluator) -> None:
        super().__init__(evaluator)

    @staticmethod
    def static_exchange(board: chess.Board, move: chess.Move) -> int:
        if not board.is_capture(move) and move.promotion is None:
            return 0
        moving_piece = board.piece_at(move.from_square)
        if moving_piece is None:
            return 0
        captured_square = move.to_square
        if board.is_en_passant(move):
            captured_square += -8 if board.turn == chess.WHITE else 8
        captured = board.piece_at(captured_square)
        captured_value = 0 if captured is None else PIECE_VALUES[captured.piece_type]
        placed_type = move.promotion or moving_piece.piece_type
        promotion_gain = (
            PIECE_VALUES[placed_type] - PIECE_VALUES[chess.PAWN]
            if move.promotion is not None
            else 0
        )
        gains = [captured_value + promotion_gain]
        occupied = int(board.occupied)
        occupied &= ~int(chess.BB_SQUARES[move.from_square])
        occupied &= ~int(chess.BB_SQUARES[captured_square])
        occupied |= int(chess.BB_SQUARES[move.to_square])
        target_piece_value = PIECE_VALUES[placed_type]
        side = not board.turn

        while True:
            attackers = int(board.attackers_mask(side, move.to_square, occupied)) & occupied
            attacker_square: chess.Square | None = None
            attacker_type = 0
            for piece_type in range(chess.PAWN, chess.KING + 1):
                candidates = attackers & int(board.pieces_mask(piece_type, side))
                if candidates:
                    attacker_square = chess.lsb(candidates)
                    attacker_type = piece_type
                    break
            if attacker_square is None:
                break
            gains.append(target_piece_value - gains[-1])
            occupied &= ~int(chess.BB_SQUARES[attacker_square])
            target_piece_value = PIECE_VALUES[attacker_type]
            side = not side

        for index in range(len(gains) - 1, 0, -1):
            gains[index - 1] = -max(-gains[index - 1], gains[index])
        return gains[0]

    def _move_score(
        self,
        board: chess.Board,
        move: chess.Move,
        tt_move: chess.Move | None,
        ply: int,
    ) -> int:
        base = super()._move_score(board, move, tt_move, ply)
        if move == tt_move or move.promotion is not None or not board.is_capture(move):
            return base
        exchange = self.static_exchange(board, move)
        if exchange >= 0:
            return 10_000_000 + exchange * 1_024 + base % 1_000_000
        return 1_000_000 + exchange * 1_024
