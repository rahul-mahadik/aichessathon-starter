"""Numba inference for the team's quantized sparse evaluator."""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
from numba import njit

PIECE_CATEGORIES = 12
FEATURE_DIM = 64 * PIECE_CATEGORIES * 64
MAX_PIECES = 32
PADDING_INDEX = FEATURE_DIM
FORMAT_VERSION = 1
VALUE_SCALE_CP = 1_000.0


def _orient(square: chess.Square, perspective: chess.Color) -> int:
    return square if perspective == chess.WHITE else chess.square_mirror(square)


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a board exactly as the offline distillation pipeline does."""
    encoded = np.full((2, MAX_PIECES), PADDING_INDEX, dtype=np.int32)
    pieces = sorted(board.piece_map().items())
    for perspective_index, perspective in enumerate((chess.WHITE, chess.BLACK)):
        king_square = board.king(perspective)
        if king_square is None:
            raise ValueError("position must contain both kings")
        oriented_king = _orient(king_square, perspective)
        for offset, (piece_square, piece) in enumerate(pieces):
            relative_colour = 0 if piece.color == perspective else 1
            category = relative_colour * 6 + piece.piece_type - 1
            encoded[perspective_index, offset] = (
                (oriented_king * PIECE_CATEGORIES + category) * 64
                + _orient(piece_square, perspective)
            )
    return encoded


@njit(cache=False)
def _dense_value(
    accumulators: np.ndarray,
    mover: int,
    hidden_one_weight: np.ndarray,
    hidden_one_bias: np.ndarray,
    hidden_two_weight: np.ndarray,
    hidden_two_bias: np.ndarray,
    output_weight: np.ndarray,
    output_bias: np.ndarray,
) -> float:
    accumulator_size = accumulators.shape[1]
    opponent = 1 - mover
    hidden_one = np.empty(hidden_one_bias.shape[0], dtype=np.float32)
    for output in range(hidden_one.shape[0]):
        value = hidden_one_bias[output]
        for column in range(accumulator_size):
            value += hidden_one_weight[output, column] * accumulators[mover, column]
            value += (
                hidden_one_weight[output, accumulator_size + column]
                * accumulators[opponent, column]
            )
        hidden_one[output] = max(value, 0.0)

    hidden_two = np.empty(hidden_two_bias.shape[0], dtype=np.float32)
    for output in range(hidden_two.shape[0]):
        value = hidden_two_bias[output]
        for column in range(hidden_one.shape[0]):
            value += hidden_two_weight[output, column] * hidden_one[column]
        hidden_two[output] = max(value, 0.0)

    value = output_bias[0]
    for column in range(hidden_two.shape[0]):
        value += output_weight[0, column] * hidden_two[column]
    return float(np.tanh(value))


@njit(cache=False)
def _infer(
    indices: np.ndarray,
    white_to_move: bool,
    feature_q: np.ndarray,
    feature_scale: float,
    feature_bias: np.ndarray,
    hidden_one_weight: np.ndarray,
    hidden_one_bias: np.ndarray,
    hidden_two_weight: np.ndarray,
    hidden_two_bias: np.ndarray,
    output_weight: np.ndarray,
    output_bias: np.ndarray,
    antisymmetric: bool,
) -> float:
    accumulator_size = feature_bias.shape[0]
    accumulators = np.empty((2, accumulator_size), dtype=np.float32)
    for perspective in range(2):
        for column in range(accumulator_size):
            total = 0
            for slot in range(indices.shape[1]):
                index = indices[perspective, slot]
                if index < feature_q.shape[0]:
                    total += int(feature_q[index, column])
            accumulators[perspective, column] = total * feature_scale + feature_bias[column]

    mover = 0 if white_to_move else 1
    value = _dense_value(
        accumulators,
        mover,
        hidden_one_weight,
        hidden_one_bias,
        hidden_two_weight,
        hidden_two_bias,
        output_weight,
        output_bias,
    )
    if not antisymmetric:
        return value
    reverse = _dense_value(
        accumulators,
        1 - mover,
        hidden_one_weight,
        hidden_one_bias,
        hidden_two_weight,
        hidden_two_bias,
        output_weight,
        output_bias,
    )
    return 0.5 * (value - reverse)


@njit(cache=False)
def _quantized_accumulators(indices: np.ndarray, feature_q: np.ndarray) -> np.ndarray:
    accumulators = np.zeros((2, feature_q.shape[1]), dtype=np.int32)
    for perspective in range(2):
        for slot in range(indices.shape[1]):
            index = indices[perspective, slot]
            if index < feature_q.shape[0]:
                for column in range(feature_q.shape[1]):
                    accumulators[perspective, column] += int(feature_q[index, column])
    return accumulators


@njit(cache=False)
def _apply_quantized_delta(
    parent: np.ndarray,
    removed: np.ndarray,
    added: np.ndarray,
    rebuild: np.ndarray,
    feature_q: np.ndarray,
) -> np.ndarray:
    child = parent.copy()
    for perspective in range(2):
        if rebuild[perspective]:
            for column in range(child.shape[1]):
                child[perspective, column] = 0
        else:
            for slot in range(removed.shape[1]):
                index = removed[perspective, slot]
                if index < feature_q.shape[0]:
                    for column in range(child.shape[1]):
                        child[perspective, column] -= int(feature_q[index, column])
        for slot in range(added.shape[1]):
            index = added[perspective, slot]
            if index < feature_q.shape[0]:
                for column in range(child.shape[1]):
                    child[perspective, column] += int(feature_q[index, column])
    return child


@njit(cache=False)
def _infer_quantized_accumulators(
    accumulator_q: np.ndarray,
    white_to_move: bool,
    feature_scale: float,
    feature_bias: np.ndarray,
    hidden_one_weight: np.ndarray,
    hidden_one_bias: np.ndarray,
    hidden_two_weight: np.ndarray,
    hidden_two_bias: np.ndarray,
    output_weight: np.ndarray,
    output_bias: np.ndarray,
    antisymmetric: bool,
) -> float:
    accumulators = accumulator_q.astype(np.float32) * feature_scale + feature_bias
    mover = 0 if white_to_move else 1
    value = _dense_value(
        accumulators,
        mover,
        hidden_one_weight,
        hidden_one_bias,
        hidden_two_weight,
        hidden_two_bias,
        output_weight,
        output_bias,
    )
    if not antisymmetric:
        return value
    reverse = _dense_value(
        accumulators,
        1 - mover,
        hidden_one_weight,
        hidden_one_bias,
        hidden_two_weight,
        hidden_two_bias,
        output_weight,
        output_bias,
    )
    return 0.5 * (value - reverse)


class QuantizedEvaluator:
    """Load and evaluate the team's versioned ``weights/nnue.npz`` artifact."""

    def __init__(self, path: Path, *, antisymmetric: bool = False) -> None:
        with np.load(path, allow_pickle=False) as archive:
            version = int(archive["format_version"])
            feature_dim = int(archive["feature_dim"])
            if version != FORMAT_VERSION or feature_dim != FEATURE_DIM:
                raise ValueError(
                    f"unsupported NNUE artifact: format={version}, features={feature_dim}"
                )
            self.feature_q = archive["feature_q"].copy()
            self.feature_scale = float(archive["feature_scale"])
            self.feature_bias = archive["feature_bias"].copy()
            self.hidden_one_weight = archive["hidden_one_weight"].copy()
            self.hidden_one_bias = archive["hidden_one_bias"].copy()
            self.hidden_two_weight = archive["hidden_two_weight"].copy()
            self.hidden_two_bias = archive["hidden_two_bias"].copy()
            self.output_weight = archive["output_weight"].copy()
            self.output_bias = archive["output_bias"].copy()
        self.antisymmetric = antisymmetric

    def raw_value(self, board: chess.Board) -> float:
        return _infer(
            encode_board(board),
            board.turn == chess.WHITE,
            self.feature_q,
            self.feature_scale,
            self.feature_bias,
            self.hidden_one_weight,
            self.hidden_one_bias,
            self.hidden_two_weight,
            self.hidden_two_bias,
            self.output_weight,
            self.output_bias,
            self.antisymmetric,
        )

    def __call__(self, board: chess.Board) -> float:
        return VALUE_SCALE_CP * self.raw_value(board)

    def warmup(self) -> None:
        self.raw_value(chess.Board())


class IncrementalQuantizedEvaluator(QuantizedEvaluator):
    """Quantized evaluator with move-stack-aware NNUE accumulator updates."""

    def __init__(self, path: Path, *, antisymmetric: bool = False) -> None:
        super().__init__(path, antisymmetric=antisymmetric)
        self._accumulator_stack: list[np.ndarray] = []
        self._king_stack: list[tuple[chess.Square, chess.Square]] = []

    @staticmethod
    def _feature_index(
        king_square: chess.Square,
        perspective: chess.Color,
        piece_square: chess.Square,
        piece: chess.Piece,
    ) -> int:
        relative_colour = 0 if piece.color == perspective else 1
        category = relative_colour * 6 + piece.piece_type - 1
        return (
            (_orient(king_square, perspective) * PIECE_CATEGORIES + category) * 64
            + _orient(piece_square, perspective)
        )

    def begin(self, board: chess.Board) -> None:
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        if white_king is None or black_king is None:
            raise ValueError("position must contain both kings")
        self._king_stack = [(white_king, black_king)]
        self._accumulator_stack = [
            _quantized_accumulators(encode_board(board), self.feature_q)
        ]

    def push(self, board: chess.Board, move: chess.Move) -> None:
        """Update state for ``move`` while ``board`` is still the parent position."""
        if not self._accumulator_stack:
            raise RuntimeError("incremental evaluator was pushed before begin")
        if move == chess.Move.null():
            self._accumulator_stack.append(self._accumulator_stack[-1].copy())
            self._king_stack.append(self._king_stack[-1])
            return

        moving_piece = board.piece_at(move.from_square)
        if moving_piece is None:
            raise ValueError(f"no piece on incremental move origin: {move}")
        captured_square = move.to_square
        if board.is_en_passant(move):
            captured_square += -8 if board.turn == chess.WHITE else 8
        captured_piece = board.piece_at(captured_square)
        placed_piece = (
            chess.Piece(move.promotion, moving_piece.color)
            if move.promotion is not None
            else moving_piece
        )

        removed_pieces = [(move.from_square, moving_piece)]
        added_pieces = [(move.to_square, placed_piece)]
        if captured_piece is not None:
            removed_pieces.append((captured_square, captured_piece))
        if board.is_castling(move):
            rank = 0 if moving_piece.color == chess.WHITE else 7
            kingside = board.is_kingside_castling(move)
            rook_from = chess.square(7 if kingside else 0, rank)
            rook_to = chess.square(5 if kingside else 3, rank)
            rook = board.piece_at(rook_from)
            if rook is None:
                raise ValueError(f"castling move has no rook: {move}")
            removed_pieces.append((rook_from, rook))
            added_pieces.append((rook_to, rook))

        removed = np.full((2, MAX_PIECES), PADDING_INDEX, dtype=np.int32)
        added = np.full((2, MAX_PIECES), PADDING_INDEX, dtype=np.int32)
        rebuild = np.zeros(2, dtype=np.bool_)
        parent_kings = self._king_stack[-1]
        child_kings = list(parent_kings)
        if moving_piece.piece_type == chess.KING:
            child_kings[0 if moving_piece.color == chess.WHITE else 1] = move.to_square
            board.push(move)
            try:
                child_indices = encode_board(board)
            finally:
                board.pop()
        else:
            child_indices = None
        for perspective_index, perspective in enumerate((chess.WHITE, chess.BLACK)):
            parent_king = parent_kings[perspective_index]
            child_king = child_kings[perspective_index]
            if parent_king != child_king:
                rebuild[perspective_index] = True
                if child_indices is None:
                    raise RuntimeError("king move did not build child features")
                added[perspective_index] = child_indices[perspective_index]
                continue
            for slot, (square, piece) in enumerate(removed_pieces):
                removed[perspective_index, slot] = self._feature_index(
                    parent_king, perspective, square, piece
                )
            for slot, (square, piece) in enumerate(added_pieces):
                added[perspective_index, slot] = self._feature_index(
                    child_king, perspective, square, piece
                )

        child_accumulators = _apply_quantized_delta(
            self._accumulator_stack[-1], removed, added, rebuild, self.feature_q
        )
        self._accumulator_stack.append(child_accumulators)
        self._king_stack.append((child_kings[0], child_kings[1]))

    def pop(self) -> None:
        if len(self._accumulator_stack) <= 1:
            raise RuntimeError("incremental evaluator cannot pop its root")
        self._accumulator_stack.pop()
        self._king_stack.pop()

    def raw_value(self, board: chess.Board) -> float:
        if not self._accumulator_stack:
            return super().raw_value(board)
        return _infer_quantized_accumulators(
            self._accumulator_stack[-1],
            board.turn == chess.WHITE,
            self.feature_scale,
            self.feature_bias,
            self.hidden_one_weight,
            self.hidden_one_bias,
            self.hidden_two_weight,
            self.hidden_two_bias,
            self.output_weight,
            self.output_bias,
            self.antisymmetric,
        )

    def warmup(self) -> None:
        board = chess.Board()
        self.begin(board)
        self.raw_value(board)
        move = chess.Move.from_uci("e2e4")
        self.push(board, move)
        board.push(move)
        self.raw_value(board)
        self.pop()
        board.pop()
