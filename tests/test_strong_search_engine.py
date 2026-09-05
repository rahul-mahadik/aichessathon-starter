from __future__ import annotations

import threading
import time
import unittest

import chess

from frontier_search_engine import FrontierSearchEngine
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

    def test_frontier_fixed_node_search_is_bounded_and_restores_board(self) -> None:
        board = chess.Board()
        original = board.fen()
        result = FrontierSearchEngine(handcrafted_evaluation).choose_fixed_nodes(board, 1_000)
        self.assertIn(result.move, board.legal_moves)
        self.assertEqual(result.nodes, 1_000)
        self.assertGreater(result.evaluations, 0)
        self.assertEqual(board.fen(), original)

    def test_frontier_finds_mate_in_one(self) -> None:
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
        result = FrontierSearchEngine(handcrafted_evaluation).choose(board, 1_000)
        board.push(result.move)
        self.assertTrue(board.is_checkmate(), result.move.uci())

    def test_stopped_ponder_state_can_be_absorbed(self) -> None:
        board = chess.Board()
        pondering = StrongSearchEngine(handcrafted_evaluation)
        pondering.choose_fixed_nodes(board, 1_000)
        foreground = StrongSearchEngine(handcrafted_evaluation)
        foreground.absorb(pondering)
        self.assertTrue(foreground.table)
        for key, entry in pondering.table.items():
            if key in foreground.table:
                self.assertGreaterEqual(foreground.table[key].depth, entry.depth)

    def test_external_stop_ends_search_and_restores_board(self) -> None:
        board = chess.Board()
        original = board.fen()
        engine = StrongSearchEngine(handcrafted_evaluation)
        thread = threading.Thread(target=engine.choose_for_ms, args=(board, 5_000))
        thread.start()
        time.sleep(0.02)
        engine.request_stop()
        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(board.fen(), original)


if __name__ == "__main__":
    unittest.main()
