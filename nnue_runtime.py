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
