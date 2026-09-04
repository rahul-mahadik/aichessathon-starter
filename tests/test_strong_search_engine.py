from __future__ import annotations

import unittest

import chess

from search_engine import handcrafted_evaluation
from strong_search_engine import StrongSearchEngine


class StrongSearchEngineTests(unittest.TestCase):
    def test_fixed_node_search_is_bounded_and_restores_board(self) -> None:
        board = chess.Board()
        original = board.fen()
        result = StrongSearchEngine(handcrafted_evaluation).choose_fixed_nodes(board, 1_000)
        self.assertIn(result.move, board.legal_moves)
        self.assertEqual(result.nodes, 1_000)
        self.assertGreater(result.evaluations, 0)
        self.assertEqual(board.fen(), original)

    def test_finds_mate_in_one(self) -> None:
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
        result = StrongSearchEngine(handcrafted_evaluation).choose(board, 1_000)
        board.push(result.move)
        self.assertTrue(board.is_checkmate(), result.move.uci())


if __name__ == "__main__":
    unittest.main()
