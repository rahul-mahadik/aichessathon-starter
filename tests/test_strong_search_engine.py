from __future__ import annotations

import threading
import time
import unittest

import chess

from frontier_search_engine import FrontierSearchEngine
from ordered_search_engine import OrderedSearchEngine
from policy_search_engine import PolicySearchEngine
from policy_see_search_engine import PolicySeeSearchEngine
from search_engine import handcrafted_evaluation
from see_search_engine import SeeSearchEngine
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

    def test_ordered_search_is_bounded_and_restores_board(self) -> None:
        board = chess.Board()
        original = board.fen()
        result = OrderedSearchEngine(handcrafted_evaluation).choose_fixed_nodes(board, 1_000)
        self.assertIn(result.move, board.legal_moves)
        self.assertEqual(result.nodes, 1_000)
        self.assertGreater(result.evaluations, 0)
        self.assertEqual(board.fen(), original)

    def test_ordered_search_finds_mate_in_one(self) -> None:
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
        result = OrderedSearchEngine(handcrafted_evaluation).choose(board, 1_000)
        board.push(result.move)
        self.assertTrue(board.is_checkmate(), result.move.uci())

    def test_policy_search_uses_legal_move_priors(self) -> None:
        class PolicyEvaluator:
            def __call__(self, board: chess.Board) -> float:
                return handcrafted_evaluation(board)

            def policy_scores(
                self, board: chess.Board, moves: list[chess.Move]
            ) -> dict[chess.Move, float]:
                return {move: float(move == chess.Move.from_uci("e2e4")) for move in moves}

        board = chess.Board()
        original = board.fen()
        result = PolicySearchEngine(PolicyEvaluator()).choose_fixed_nodes(board, 1_000)
        self.assertIn(result.move, board.legal_moves)
        self.assertEqual(result.nodes, 1_000)
        self.assertEqual(board.fen(), original)

        combined = PolicySeeSearchEngine(PolicyEvaluator()).choose_fixed_nodes(board, 1_000)
        self.assertIn(combined.move, board.legal_moves)
        self.assertEqual(combined.nodes, 1_000)
        self.assertEqual(board.fen(), original)

    def test_static_exchange_distinguishes_winning_and_losing_captures(self) -> None:
        winning = chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")
        losing = chess.Board("3rk3/8/8/3p4/4Q3/8/8/4K3 w - - 0 1")
        self.assertGreater(SeeSearchEngine.static_exchange(winning, chess.Move.from_uci("e4d5")), 0)
        self.assertLess(SeeSearchEngine.static_exchange(losing, chess.Move.from_uci("e4d5")), 0)

    def test_see_search_is_bounded_and_restores_board(self) -> None:
        board = chess.Board()
        original = board.fen()
        result = SeeSearchEngine(handcrafted_evaluation).choose_fixed_nodes(board, 1_000)
        self.assertIn(result.move, board.legal_moves)
        self.assertEqual(result.nodes, 1_000)
        self.assertEqual(board.fen(), original)

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
