"""King-conditioned sparse features and teacher-target transforms."""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass

import chess
import numpy as np

from distill.schema import TeacherRecord, TeacherScore

PIECE_CATEGORIES = 12
FEATURE_DIM = 64 * PIECE_CATEGORIES * 64
MAX_PIECES = 32
MAX_CANDIDATES = 8
MAX_POLICY_MOVES = 16
GROUP_SIZE = 1 + MAX_CANDIDATES
PADDING_INDEX = FEATURE_DIM
DEFAULT_CP_SCALE = 600.0
METADATA_DIM = 21


@dataclass(frozen=True)
class TrainingGroup:
    """One root plus up to eight candidate child positions."""

    features: np.ndarray
    turns: np.ndarray
    targets: np.ndarray
    value_mask: np.ndarray
    candidate_scores: np.ndarray
    candidate_mask: np.ndarray
    metadata: np.ndarray
    policy_from: np.ndarray
    policy_to: np.ndarray
    policy_promotion: np.ndarray
    policy_scores: np.ndarray
    policy_mask: np.ndarray


def _orient(square: chess.Square, perspective: chess.Color) -> int:
    return square if perspective == chess.WHITE else chess.square_mirror(square)


def feature_index(
    king_square: chess.Square,
    piece: chess.Piece,
    piece_square: chess.Square,
    perspective: chess.Color,
) -> int:
    """Return the HalfKP-style feature index for one piece and perspective."""
    oriented_king = _orient(king_square, perspective)
    oriented_piece = _orient(piece_square, perspective)
    relative_colour = 0 if piece.color == perspective else 1
    category = relative_colour * 6 + piece.piece_type - 1
    return (oriented_king * PIECE_CATEGORIES + category) * 64 + oriented_piece


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a board as white- and black-perspective padded sparse indices."""
    encoded = np.full((2, MAX_PIECES), PADDING_INDEX, dtype=np.int32)
    pieces = sorted(board.piece_map().items())
    if len(pieces) > MAX_PIECES:
        raise ValueError(f"position has {len(pieces)} pieces, expected at most {MAX_PIECES}")
    for perspective_index, perspective in enumerate((chess.WHITE, chess.BLACK)):
        king_square = board.king(perspective)
        if king_square is None:
            raise ValueError("training positions must contain both kings")
        for offset, (piece_square, piece) in enumerate(pieces):
            encoded[perspective_index, offset] = feature_index(
                king_square, piece, piece_square, perspective
            )
    return encoded


def encode_metadata(board: chess.Board) -> np.ndarray:
    """Encode castling, en-passant, and rule-50 state omitted by HalfKP."""
    encoded = np.zeros(METADATA_DIM, dtype=np.float32)
    encoded[0] = board.has_kingside_castling_rights(chess.WHITE)
    encoded[1] = board.has_queenside_castling_rights(chess.WHITE)
    encoded[2] = board.has_kingside_castling_rights(chess.BLACK)
    encoded[3] = board.has_queenside_castling_rights(chess.BLACK)
    ep_index = 8 if board.ep_square is None else chess.square_file(board.ep_square)
    encoded[4 + ep_index] = 1.0
    rule50_bucket = min(board.halfmove_clock // 16, 7)
    encoded[13 + rule50_bucket] = 1.0
    return encoded


def score_to_value(score: TeacherScore, cp_scale: float = DEFAULT_CP_SCALE) -> float:
    """Convert raw teacher output to an expected score in [-1, 1]."""
    if score.wdl is not None:
        wins, draws, losses = score.wdl
        total = wins + draws + losses
        if total > 0:
            return float(wins - losses) / total
    if score.mate is not None:
        return 1.0 if score.mate > 0 else -1.0
    assert score.cp is not None
    return math.tanh(score.cp / cp_scale)


def record_to_group(record: TeacherRecord, cp_scale: float = DEFAULT_CP_SCALE) -> TrainingGroup:
    """Create root and sign-flipped child-state targets from a teacher record."""
    features = np.full((GROUP_SIZE, 2, MAX_PIECES), PADDING_INDEX, dtype=np.int32)
    turns = np.zeros(GROUP_SIZE, dtype=np.bool_)
    targets = np.zeros(GROUP_SIZE, dtype=np.float32)
    value_mask = np.zeros(GROUP_SIZE, dtype=np.bool_)
    candidate_scores = np.zeros(MAX_CANDIDATES, dtype=np.float32)
    candidate_mask = np.zeros(MAX_CANDIDATES, dtype=np.bool_)
    metadata = np.zeros((GROUP_SIZE, METADATA_DIM), dtype=np.float32)
    policy_from = np.zeros(MAX_POLICY_MOVES, dtype=np.uint8)
    policy_to = np.zeros(MAX_POLICY_MOVES, dtype=np.uint8)
    policy_promotion = np.zeros(MAX_POLICY_MOVES, dtype=np.uint8)
    policy_scores = np.zeros(MAX_POLICY_MOVES, dtype=np.float32)
    policy_mask = np.zeros(MAX_POLICY_MOVES, dtype=np.bool_)

    board = chess.Board(record.fen)
    features[0] = encode_board(board)
    metadata[0] = encode_metadata(board)
    turns[0] = board.turn
    targets[0] = score_to_value(record.root_score, cp_scale)
    value_mask[0] = True

    for index, candidate in enumerate(record.candidates[:MAX_CANDIDATES]):
        move = chess.Move.from_uci(candidate.move)
        if move not in board.legal_moves:
            raise ValueError(f"teacher move {candidate.move} is illegal in {record.fen}")
        board.push(move)
        root_value = score_to_value(candidate.score, cp_scale)
        features[index + 1] = encode_board(board)
        metadata[index + 1] = encode_metadata(board)
        turns[index + 1] = board.turn
        targets[index + 1] = -root_value
        value_mask[index + 1] = True
        candidate_scores[index] = root_value
        candidate_mask[index] = True
        board.pop()

        policy_from[index] = move.from_square
        policy_to[index] = move.to_square
        policy_promotion[index] = move.promotion or 0
        policy_scores[index] = root_value
        policy_mask[index] = True

    teacher_moves = {candidate.move for candidate in record.candidates[:MAX_CANDIDATES]}
    negatives = [move for move in board.legal_moves if move.uci() not in teacher_moves]
    rng = np.random.default_rng(zlib.crc32(record.fen.encode()))
    rng.shuffle(negatives)
    negative_score = (
        float(np.min(candidate_scores[candidate_mask])) - 0.25 if np.any(candidate_mask) else -1.0
    )
    for offset, move in enumerate(negatives[: MAX_POLICY_MOVES - MAX_CANDIDATES]):
        index = MAX_CANDIDATES + offset
        policy_from[index] = move.from_square
        policy_to[index] = move.to_square
        policy_promotion[index] = move.promotion or 0
        policy_scores[index] = negative_score
        policy_mask[index] = True

    return TrainingGroup(
        features=features,
        turns=turns,
        targets=targets,
        value_mask=value_mask,
        candidate_scores=candidate_scores,
        candidate_mask=candidate_mask,
        metadata=metadata,
        policy_from=policy_from,
        policy_to=policy_to,
        policy_promotion=policy_promotion,
        policy_scores=policy_scores,
        policy_mask=policy_mask,
    )
