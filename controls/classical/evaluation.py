"""Fast, handcrafted evaluation for the non-distillation control engine.

All placement values and masks are generated from board geometry at import. No
piece-square table is copied from another engine. Runtime evaluation is almost
entirely integer bitboard arithmetic, which matters at every quiet search leaf.
"""

from __future__ import annotations

import chess

MAX_PHASE = 24
MATERIAL_MG = (0, 100, 320, 332, 505, 940, 0)
MATERIAL_EG = (0, 118, 304, 326, 526, 936, 0)
PHASE_WEIGHT = (0, 0, 1, 1, 2, 4, 0)
MOBILITY_MG = (0, 0, 4, 5, 2, 1, 0)
MOBILITY_EG = (0, 0, 3, 5, 4, 2, 0)
PASSED_MG = (0, 0, 5, 12, 25, 48, 85, 0)
PASSED_EG = (0, 0, 10, 24, 48, 88, 145, 0)
FILE_MASKS = tuple(int(mask) for mask in chess.BB_FILES)


def _relative_rank(square: chess.Square, colour: chess.Color) -> int:
    rank = chess.square_rank(square)
    return rank if colour == chess.WHITE else 7 - rank


def _centre(square: chess.Square) -> int:
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    return 14 - abs(2 * file - 7) - abs(2 * rank - 7)


def _generated_piece_square(
    piece_type: chess.PieceType,
    square: chess.Square,
    colour: chess.Color,
) -> tuple[int, int]:
    relative_rank = _relative_rank(square, colour)
    centre = _centre(square)
    file = chess.square_file(square)
    file_centre = 7 - abs(2 * file - 7)
    if piece_type == chess.PAWN:
        return 4 * relative_rank + centre, 7 * relative_rank + centre
    if piece_type == chess.KNIGHT:
        return 5 * centre - 14, 4 * centre - 12
    if piece_type == chess.BISHOP:
        return 3 * centre + file_centre - 10, 3 * centre - 8
    if piece_type == chess.ROOK:
        seventh = 18 if relative_rank == 6 else 0
        return 2 * centre + seventh - 8, 2 * centre + seventh - 6
    if piece_type == chess.QUEEN:
        return centre - 2 * relative_rank - 5, 2 * centre - 8
    if piece_type == chess.KING:
        return -5 * centre - 7 * relative_rank + 42, 6 * centre - 24
    return 0, 0


def _make_psqt() -> tuple[
    tuple[tuple[tuple[int, ...], ...], ...],
    tuple[tuple[tuple[int, ...], ...], ...],
]:
    middlegame: list[tuple[tuple[int, ...], ...]] = []
    endgame: list[tuple[tuple[int, ...], ...]] = []
    for colour in (chess.BLACK, chess.WHITE):
        colour_mg = [tuple(0 for _ in chess.SQUARES)]
        colour_eg = [tuple(0 for _ in chess.SQUARES)]
        for piece_type in range(chess.PAWN, chess.KING + 1):
            values = tuple(
                _generated_piece_square(piece_type, square, colour)
                for square in chess.SQUARES
            )
            colour_mg.append(tuple(value[0] for value in values))
            colour_eg.append(tuple(value[1] for value in values))
        middlegame.append(tuple(colour_mg))
        endgame.append(tuple(colour_eg))
    return tuple(middlegame), tuple(endgame)


def _files_near(file: int) -> range:
    return range(max(0, file - 1), min(8, file + 2))


def _make_pawn_masks() -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
]:
    passed: list[tuple[int, ...]] = []
    for colour in (chess.BLACK, chess.WHITE):
        colour_masks: list[int] = []
        for square in chess.SQUARES:
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            mask = 0
            ranks = range(rank) if colour == chess.BLACK else range(rank + 1, 8)
            for forward_rank in ranks:
                for nearby_file in _files_near(file):
                    mask |= chess.BB_SQUARES[chess.square(nearby_file, forward_rank)]
            colour_masks.append(mask)
        passed.append(tuple(colour_masks))

    connected: list[int] = []
    for square in chess.SQUARES:
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        mask = 0
        for adjacent_file in (file - 1, file + 1):
            if 0 <= adjacent_file < 8:
                for nearby_rank in range(max(0, rank - 1), min(8, rank + 2)):
                    mask |= chess.BB_SQUARES[chess.square(adjacent_file, nearby_rank)]
        connected.append(mask)
    return (tuple(passed), tuple(connected))


def _make_king_shields() -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
]:
    close_masks: list[tuple[int, ...]] = []
    far_masks: list[tuple[int, ...]] = []
    widths = tuple(len(_files_near(chess.square_file(square))) for square in chess.SQUARES)
    for colour in (chess.BLACK, chess.WHITE):
        direction = 1 if colour == chess.WHITE else -1
        colour_close: list[int] = []
        colour_far: list[int] = []
        for square in chess.SQUARES:
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            close = 0
            far = 0
            for shield_file in _files_near(file):
                if 0 <= rank + direction < 8:
                    close |= chess.BB_SQUARES[chess.square(shield_file, rank + direction)]
                if 0 <= rank + 2 * direction < 8:
                    far |= chess.BB_SQUARES[chess.square(shield_file, rank + 2 * direction)]
            colour_close.append(close)
            colour_far.append(far)
        close_masks.append(tuple(colour_close))
        far_masks.append(tuple(colour_far))
    return tuple(close_masks), tuple(far_masks), widths


PSQT_MG, PSQT_EG = _make_psqt()
PASSED_MASKS, CONNECTED_MASKS = _make_pawn_masks()
KING_CLOSE_MASKS, KING_FAR_MASKS, KING_SHIELD_WIDTH = _make_king_shields()


def _pawn_terms(pawns: int, enemy_pawns: int, colour: chess.Color) -> tuple[int, int]:
    middlegame = 0
    endgame = 0
    occupied_files = 0
    for file, file_mask in enumerate(FILE_MASKS):
        count = (pawns & file_mask).bit_count()
        if count:
            occupied_files |= 1 << file
            if count > 1:
                middlegame -= 13 * (count - 1)
                endgame -= 19 * (count - 1)

    for square in chess.scan_reversed(pawns):
        file = chess.square_file(square)
        adjacent_files = ((1 << (file - 1)) if file else 0) | (
            (1 << (file + 1)) if file < 7 else 0
        )
        if not occupied_files & adjacent_files:
            middlegame -= 11
            endgame -= 15
        elif pawns & CONNECTED_MASKS[square]:
            middlegame += 4
            endgame += 6
        if not enemy_pawns & PASSED_MASKS[colour][square]:
            relative_rank = _relative_rank(square, colour)
            middlegame += PASSED_MG[relative_rank]
            endgame += PASSED_EG[relative_rank]
    return middlegame, endgame


def _mobility_terms(board: chess.Board, colour: chess.Color, ours: int) -> tuple[int, int]:
    middlegame = 0
    endgame = 0
    for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for square in chess.scan_reversed(board.pieces_mask(piece_type, colour)):
            count = (board.attacks_mask(square) & ~ours).bit_count()
            middlegame += MOBILITY_MG[piece_type] * count
            endgame += MOBILITY_EG[piece_type] * count
    return middlegame, endgame


def _rook_file_terms(rooks: int, friendly_pawns: int, enemy_pawns: int) -> tuple[int, int]:
    middlegame = 0
    endgame = 0
    for square in chess.scan_reversed(rooks):
        file_mask = FILE_MASKS[chess.square_file(square)]
        if friendly_pawns & file_mask:
            continue
        middlegame += 10
        endgame += 7
        if not enemy_pawns & file_mask:
            middlegame += 8
            endgame += 8
    return middlegame, endgame


def _king_shield(king: int, pawns: int, colour: chess.Color) -> int:
    if not king:
        return 0
    square = chess.msb(king)
    close = (pawns & KING_CLOSE_MASKS[colour][square]).bit_count()
    far = (pawns & KING_FAR_MASKS[colour][square]).bit_count()
    missing = KING_SHIELD_WIDTH[square] - close - far
    return 10 * close + 4 * far - 8 * max(0, missing)


def evaluate(board: chess.Board) -> int:
    """Evaluate ``board`` in centipawns from the side-to-move perspective."""
    middlegame = 0
    endgame = 0
    phase = 0

    for colour, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        friendly_pawns = board.pieces_mask(chess.PAWN, colour)
        enemy_pawns = board.pieces_mask(chess.PAWN, not colour)
        ours = board.occupied_co[colour]
        bishops = 0
        rooks = 0
        king = 0

        for piece_type in range(chess.PAWN, chess.KING + 1):
            pieces = board.pieces_mask(piece_type, colour)
            count = pieces.bit_count()
            middlegame += sign * MATERIAL_MG[piece_type] * count
            endgame += sign * MATERIAL_EG[piece_type] * count
            phase += PHASE_WEIGHT[piece_type] * count
            for square in chess.scan_reversed(pieces):
                middlegame += sign * PSQT_MG[colour][piece_type][square]
                endgame += sign * PSQT_EG[colour][piece_type][square]
            if piece_type == chess.BISHOP:
                bishops = count
            elif piece_type == chess.ROOK:
                rooks = pieces
            elif piece_type == chess.KING:
                king = pieces

        pawn_mg, pawn_eg = _pawn_terms(friendly_pawns, enemy_pawns, colour)
        mobility_mg, mobility_eg = _mobility_terms(board, colour, ours)
        rook_mg, rook_eg = _rook_file_terms(rooks, friendly_pawns, enemy_pawns)
        middlegame += sign * (
            pawn_mg + mobility_mg + rook_mg + _king_shield(king, friendly_pawns, colour)
        )
        endgame += sign * (pawn_eg + mobility_eg + rook_eg)
        if bishops >= 2:
            middlegame += sign * 28
            endgame += sign * 38

    phase = min(phase, MAX_PHASE)
    white_score = round((middlegame * phase + endgame * (MAX_PHASE - phase)) / MAX_PHASE)
    return white_score if board.turn == chess.WHITE else -white_score
