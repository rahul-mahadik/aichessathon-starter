"""Local-only adapter around a pinned copy of the GPLv3 Sunfish engine.

This is a copied-engine research control. It is intentionally excluded from
the submission and must not be uploaded to the Chessathon.
"""

from __future__ import annotations

import os
import re
import time

import chess

import sunfish_upstream as sunfish


_fixed_nodes = int(os.environ.get("AICHESSATHON_FIXED_NODES", "0"))
if _fixed_nodes < 0:
    raise ValueError("AICHESSATHON_FIXED_NODES must be non-negative")

_searcher = sunfish.Searcher()
_game_board: chess.Board | None = None
_history: list[sunfish.Position] = []


def _from_board(board: chess.Board) -> sunfish.Position:
    board_text = re.sub(r"\d", lambda match: "." * int(match.group()), board.board_fen())
    mailbox = list(21 * " " + "  ".join(board_text.split("/")) + 21 * " ")
    mailbox[9::10] = ["\n"] * 12
    board_120 = "".join(mailbox)
    castling = board.castling_xfen()
    wc = ("Q" in castling, "K" in castling)
    bc = ("k" in castling, "q" in castling)
    ep = sunfish.parse(chess.square_name(board.ep_square)) if board.ep_square is not None else 0
    score = sum(
        sunfish.pst[piece][index]
        for index, piece in enumerate(board_120)
        if piece.isupper()
    )
    score -= sum(
        sunfish.pst[piece.upper()][119 - index]
        for index, piece in enumerate(board_120)
        if piece.islower()
    )
    position = sunfish.Position(board_120, score, wc, bc, ep, 0)
    return position if board.turn == chess.WHITE else position.rotate()


def _to_uci(move: sunfish.Move, white_to_move: bool) -> str:
    origin, target = move.i, move.j
    if not white_to_move:
        origin, target = 119 - origin, 119 - target
    return sunfish.render(origin) + sunfish.render(target) + move.prom.lower()


def _synchronize(fen: str) -> chess.Board:
    global _game_board, _history
    incoming = chess.Board(fen)
    if _game_board is None:
        _game_board, _history = incoming, [_from_board(incoming)]
        return _game_board
    if _game_board.fen() == incoming.fen():
        return _game_board
    for move in list(_game_board.legal_moves):
        _game_board.push(move)
        if _game_board.fen() == incoming.fen():
            _history.append(_from_board(_game_board))
            return _game_board
        _game_board.pop()
    _game_board, _history = incoming, [_from_board(incoming)]
    return _game_board


def _budget_ms(time_left_ms: int) -> int:
    if time_left_ms <= 1_000:
        return max(10, int(time_left_ms * 0.08))
    target = int(time_left_ms / 40 + 300)
    return max(60, min(5_000, int(time_left_ms * 0.15), target))


def _choose(board: chess.Board, time_left_ms: int) -> tuple[sunfish.Move, int]:
    position = _history[-1]
    budget_seconds = _budget_ms(max(1, time_left_ms)) / 1_000
    _searcher.node_cap = _fixed_nodes
    _searcher.deadline = (1 << 63) if _fixed_nodes else time.time() + budget_seconds
    _searcher.soft = _searcher.deadline

    committed = candidate = None
    current_depth = 1
    try:
        for depth, gamma, score, move in _searcher.search(_history):
            if depth != current_depth:
                committed = candidate or committed
                candidate = None
                current_depth = depth
            if score >= gamma and move is not None:
                candidate = move
    except sunfish.Stop:
        pass

    ordered = [move for move in (committed, candidate) if move is not None]
    ordered.extend(position.gen_moves())
    for move in ordered:
        try:
            parsed = chess.Move.from_uci(_to_uci(move, board.turn == chess.WHITE))
        except ValueError:
            continue
        if parsed in board.legal_moves:
            return move, current_depth - (candidate is not None)
    raise RuntimeError("Sunfish produced no legal move for a non-terminal position")


def get_move(fen: str, time_left_ms: int) -> str:
    board = _synchronize(fen)
    if board.is_game_over():
        raise ValueError("cannot choose a move from a terminal position")
    started = time.monotonic()
    move, depth = _choose(board, time_left_ms)
    uci = _to_uci(move, board.turn == chess.WHITE)
    parsed = chess.Move.from_uci(uci)
    board.push(parsed)
    _history.append(_history[-1].move(move))
    print(
        f"sunfish-dupe depth={depth} nodes={_searcher.nodes} "
        f"elapsed_ms={round((time.monotonic() - started) * 1_000)}",
        flush=True,
    )
    return uci
