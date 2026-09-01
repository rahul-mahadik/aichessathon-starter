from __future__ import annotations

import unittest

import chess

from search_engine import SearchEngine


class SearchEngineTests(unittest.TestCase):
    def test_search_restores_board_after_timeout(self) -> None:
        board = chess.Board()
        original = board.fen()
        result = SearchEngine().choose(board, 10)
        self.assertIn(result.move, board.legal_moves)
        self.assertEqual(board.fen(), original)

    def test_finds_mate_in_one(self) -> None:
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
        result = SearchEngine().choose(board, 1_000)
        board.push(result.move)
        self.assertTrue(board.is_checkmate(), result.move.uci())

    def test_only_legal_move(self) -> None:
        board = chess.Board("3n4/4r1b1/r1P5/3NR2P/3p3P/P5pN/3q4/k2K4 w - - 2 52")
        legal = list(board.legal_moves)
        self.assertEqual(len(legal), 1)
        result = SearchEngine().choose(board, 100)
        self.assertEqual(result.move, legal[0])

    def test_fixed_node_search_is_bounded_and_restores_board(self) -> None:
        board = chess.Board()
        original = board.fen()
        result = SearchEngine().choose_fixed_nodes(board, 1_000)
        self.assertIn(result.move, board.legal_moves)
        self.assertEqual(result.nodes, 1_000)
        self.assertEqual(board.fen(), original)


if __name__ == "__main__":
    unittest.main()
