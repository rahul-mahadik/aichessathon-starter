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
                oriented_king * PIECE_CATEGORIES + category
            ) * 64 + _orient(piece_square, perspective)
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
def _apply_quantized_delta_into(
    parent: np.ndarray,
    child: np.ndarray,
    removed: np.ndarray,
    added: np.ndarray,
    rebuild: np.ndarray,
    feature_q: np.ndarray,
) -> None:
    """Apply a move into preallocated storage instead of allocating per node."""
    for perspective in range(2):
        if rebuild[perspective]:
            for column in range(child.shape[1]):
                child[perspective, column] = 0
        else:
            for column in range(child.shape[1]):
                child[perspective, column] = parent[perspective, column]
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


@njit(cache=False)
def _dense_value_integer(
    accumulators: np.ndarray,
    mover: int,
    hidden_one_weight_q: np.ndarray,
    hidden_one_bias_q: np.ndarray,
    hidden_one_scale: float,
    hidden_two_weight_q: np.ndarray,
    hidden_two_bias_q: np.ndarray,
    hidden_two_scale: float,
    output_weight_q: np.ndarray,
    output_bias_q: np.ndarray,
    output_scale: float,
) -> float:
    accumulator_size = accumulators.shape[1]
    opponent = 1 - mover
    hidden_one = np.empty(hidden_one_bias_q.shape[0], dtype=np.int64)
    for output in range(hidden_one.shape[0]):
        value = np.int64(hidden_one_bias_q[output])
        for column in range(accumulator_size):
            value += np.int64(hidden_one_weight_q[output, column]) * np.int64(
                accumulators[mover, column]
            )
            value += np.int64(hidden_one_weight_q[output, accumulator_size + column]) * np.int64(
                accumulators[opponent, column]
            )
        hidden_one[output] = max(value, 0)

    hidden_two = np.empty(hidden_two_bias_q.shape[0], dtype=np.int64)
    for output in range(hidden_two.shape[0]):
        value = np.int64(hidden_two_bias_q[output])
        for column in range(hidden_one.shape[0]):
            value += np.int64(hidden_two_weight_q[output, column]) * hidden_one[column]
        hidden_two[output] = max(value, 0)

    value = np.int64(output_bias_q[0])
    for column in range(hidden_two.shape[0]):
        value += np.int64(output_weight_q[0, column]) * hidden_two[column]
    return float(np.tanh(float(value) * output_scale))


@njit(cache=False)
def _infer_integer_accumulators(
    accumulators: np.ndarray,
    white_to_move: bool,
    hidden_one_weight_q: np.ndarray,
    hidden_one_bias_q: np.ndarray,
    hidden_one_scale: float,
    hidden_two_weight_q: np.ndarray,
    hidden_two_bias_q: np.ndarray,
    hidden_two_scale: float,
    output_weight_q: np.ndarray,
    output_bias_q: np.ndarray,
    output_scale: float,
    antisymmetric: bool,
) -> float:
    mover = 0 if white_to_move else 1
    value = _dense_value_integer(
        accumulators,
        mover,
        hidden_one_weight_q,
        hidden_one_bias_q,
        hidden_one_scale,
        hidden_two_weight_q,
        hidden_two_bias_q,
        hidden_two_scale,
        output_weight_q,
        output_bias_q,
        output_scale,
    )
    if not antisymmetric:
        return value
    reverse = _dense_value_integer(
        accumulators,
        1 - mover,
        hidden_one_weight_q,
        hidden_one_bias_q,
        hidden_one_scale,
        hidden_two_weight_q,
        hidden_two_bias_q,
        hidden_two_scale,
        output_weight_q,
        output_bias_q,
        output_scale,
    )
    return 0.5 * (value - reverse)


def _quantize_effective_layer(
    weight: np.ndarray,
    bias: np.ndarray,
    input_scale: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    effective = weight.astype(np.float64) * input_scale
    maximum = float(np.max(np.abs(effective)))
    output_scale = max(maximum / 127.0, 1e-12)
    weight_q = np.rint(effective / output_scale).clip(-127, 127).astype(np.int8)
    bias_q = np.rint(bias.astype(np.float64) / output_scale).astype(np.int64)
    return weight_q, bias_q, output_scale


class QuantizedEvaluator:
    """Load and evaluate the team's versioned ``weights/nnue.npz`` artifact."""

    def __init__(
        self,
        path: Path,
        *,
        antisymmetric: bool = False,
        value_scale_cp: float = VALUE_SCALE_CP,
    ) -> None:
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
        if value_scale_cp <= 0:
            raise ValueError("value_scale_cp must be positive")
        self.value_scale_cp = value_scale_cp

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
        return self.value_scale_cp * self.raw_value(board)

    def warmup(self) -> None:
        self.raw_value(chess.Board())


class IntegerQuantizedEvaluator(QuantizedEvaluator):
    """Use an int8/int64 dense head derived from the exported float weights."""

    def __init__(
        self,
        path: Path,
        *,
        antisymmetric: bool = False,
        value_scale_cp: float = VALUE_SCALE_CP,
    ) -> None:
        super().__init__(
            path,
            antisymmetric=antisymmetric,
            value_scale_cp=value_scale_cp,
        )
        doubled_bias = np.concatenate((self.feature_bias, self.feature_bias))
        first_bias = self.hidden_one_bias + self.hidden_one_weight @ doubled_bias
        (
            self.hidden_one_weight_q,
            self.hidden_one_bias_q,
            self.hidden_one_scale,
        ) = _quantize_effective_layer(
            self.hidden_one_weight,
            first_bias,
            self.feature_scale,
        )
        (
            self.hidden_two_weight_q,
            self.hidden_two_bias_q,
            self.hidden_two_scale,
        ) = _quantize_effective_layer(
            self.hidden_two_weight,
            self.hidden_two_bias,
            self.hidden_one_scale,
        )
        (
            self.output_weight_q,
            self.output_bias_q,
            self.output_scale,
        ) = _quantize_effective_layer(
            self.output_weight,
            self.output_bias,
            self.hidden_two_scale,
        )

    def raw_value(self, board: chess.Board) -> float:
        accumulators = _quantized_accumulators(encode_board(board), self.feature_q)
        return _infer_integer_accumulators(
            accumulators,
            board.turn == chess.WHITE,
            self.hidden_one_weight_q,
            self.hidden_one_bias_q,
            self.hidden_one_scale,
            self.hidden_two_weight_q,
            self.hidden_two_bias_q,
            self.hidden_two_scale,
            self.output_weight_q,
            self.output_bias_q,
            self.output_scale,
            self.antisymmetric,
        )


class IncrementalQuantizedEvaluator(QuantizedEvaluator):
    """Quantized evaluator with move-stack-aware NNUE accumulator updates."""

    def __init__(
        self,
        path: Path,
        *,
        antisymmetric: bool = False,
        value_scale_cp: float = VALUE_SCALE_CP,
    ) -> None:
        super().__init__(
            path,
            antisymmetric=antisymmetric,
            value_scale_cp=value_scale_cp,
        )
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
        return (_orient(king_square, perspective) * PIECE_CATEGORIES + category) * 64 + _orient(
            piece_square, perspective
        )

    def begin(self, board: chess.Board) -> None:
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        if white_king is None or black_king is None:
            raise ValueError("position must contain both kings")
        self._king_stack = [(white_king, black_king)]
        self._accumulator_stack = [_quantized_accumulators(encode_board(board), self.feature_q)]

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


class BufferedIncrementalQuantizedEvaluator(QuantizedEvaluator):
    """Incremental evaluator with fixed move-stack and scratch buffers.

    The original incremental evaluator remains available as a frozen control.
    This version removes the per-push accumulator, index, and king-stack array
    allocations that become expensive in a one-core Python search.
    """

    MAX_STACK_PLY = 128

    def __init__(
        self,
        path: Path,
        *,
        antisymmetric: bool = False,
        value_scale_cp: float = VALUE_SCALE_CP,
    ) -> None:
        super().__init__(
            path,
            antisymmetric=antisymmetric,
            value_scale_cp=value_scale_cp,
        )
        width = self.feature_q.shape[1]
        self._accumulators = np.empty((self.MAX_STACK_PLY + 1, 2, width), dtype=np.int32)
        self._kings = np.empty((self.MAX_STACK_PLY + 1, 2), dtype=np.int16)
        self._removed = np.full((2, MAX_PIECES), PADDING_INDEX, dtype=np.int32)
        self._added = np.full((2, MAX_PIECES), PADDING_INDEX, dtype=np.int32)
        self._rebuild = np.zeros(2, dtype=np.bool_)
        self._depth = -1

    def begin(self, board: chess.Board) -> None:
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        if white_king is None or black_king is None:
            raise ValueError("position must contain both kings")
        self._depth = 0
        self._accumulators[0] = _quantized_accumulators(encode_board(board), self.feature_q)
        self._kings[0] = (white_king, black_king)

    def push(self, board: chess.Board, move: chess.Move) -> None:
        if self._depth < 0:
            raise RuntimeError("buffered evaluator was pushed before begin")
        child_depth = self._depth + 1
        if child_depth > self.MAX_STACK_PLY:
            raise RuntimeError("buffered evaluator exceeded its move-stack capacity")
        if move == chess.Move.null():
            np.copyto(self._accumulators[child_depth], self._accumulators[self._depth])
            self._kings[child_depth] = self._kings[self._depth]
            self._depth = child_depth
            return

        moving_piece = board.piece_at(move.from_square)
        if moving_piece is None:
            raise ValueError(f"no piece on buffered move origin: {move}")
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

        self._removed.fill(PADDING_INDEX)
        self._added.fill(PADDING_INDEX)
        self._rebuild.fill(False)
        parent_kings = self._kings[self._depth]
        child_kings = [int(parent_kings[0]), int(parent_kings[1])]
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
            parent_king = int(parent_kings[perspective_index])
            child_king = child_kings[perspective_index]
            if parent_king != child_king:
                self._rebuild[perspective_index] = True
                if child_indices is None:
                    raise RuntimeError("king move did not build buffered child features")
                self._added[perspective_index] = child_indices[perspective_index]
                continue
            for slot, (square, piece) in enumerate(removed_pieces):
                self._removed[perspective_index, slot] = (
                    IncrementalQuantizedEvaluator._feature_index(
                        parent_king, perspective, square, piece
                    )
                )
            for slot, (square, piece) in enumerate(added_pieces):
                self._added[perspective_index, slot] = IncrementalQuantizedEvaluator._feature_index(
                    child_king, perspective, square, piece
                )

        _apply_quantized_delta_into(
            self._accumulators[self._depth],
            self._accumulators[child_depth],
            self._removed,
            self._added,
            self._rebuild,
            self.feature_q,
        )
        self._kings[child_depth] = child_kings
        self._depth = child_depth

    def pop(self) -> None:
        if self._depth <= 0:
            raise RuntimeError("buffered evaluator cannot pop its root")
        self._depth -= 1

    def raw_value(self, board: chess.Board) -> float:
        if self._depth < 0:
            return super().raw_value(board)
        return _infer_quantized_accumulators(
            self._accumulators[self._depth],
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
