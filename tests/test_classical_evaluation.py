from __future__ import annotations

import unittest

import chess

from controls.classical.evaluation import evaluate


class ClassicalEvaluationTests(unittest.TestCase):
    def test_starting_position_is_balanced(self) -> None:
        self.assertEqual(evaluate(chess.Board()), 0.0)

    def test_evaluation_is_symmetric_under_mirroring(self) -> None:
        positions = [
            chess.Board(),
            chess.Board("4k3/8/2p5/3P4/5N2/8/8/4K3 w - - 0 1"),
            chess.Board("2kr3r/ppp2ppp/2n1bn2/3qp3/8/2NP1N2/PPP2PPP/2KR3R w - - 0 1"),
        ]
        for board in positions:
            with self.subTest(fen=board.fen()):
                self.assertAlmostEqual(evaluate(board), evaluate(board.mirror()))

    def test_material_advantage_follows_side_to_move(self) -> None:
        board = chess.Board("4k3/8/8/8/8/8/4Q3/4K3 w - - 0 1")
        white_score = evaluate(board)
        board.turn = chess.BLACK
        self.assertGreater(white_score, 800.0)
        self.assertAlmostEqual(white_score, -evaluate(board))

    def test_central_knight_is_better_than_corner_knight(self) -> None:
        corner = chess.Board("4k3/8/8/8/8/8/8/N3K3 w - - 0 1")
        centre = chess.Board("4k3/8/8/3N4/8/8/8/4K3 w - - 0 1")
        self.assertGreater(evaluate(centre), evaluate(corner) + 35.0)

    def test_advanced_passed_pawn_is_rewarded(self) -> None:
        held_back = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
        advanced = chess.Board("4k3/4P3/8/8/8/8/8/4K3 w - - 0 1")
        self.assertGreater(evaluate(advanced), evaluate(held_back) + 80.0)

    def test_doubled_isolated_pawns_are_penalized(self) -> None:
        healthy = chess.Board("4k3/7p/8/8/8/8/2PP4/4K3 w - - 0 1")
        doubled = chess.Board("4k3/7p/8/8/8/2P5/2P5/4K3 w - - 0 1")
        self.assertGreater(evaluate(healthy), evaluate(doubled) + 10.0)

    def test_pawn_shield_improves_king_safety_with_heavy_material(self) -> None:
        shielded = chess.Board("3qk3/8/8/8/8/8/5PPP/3Q2K1 w - - 0 1")
        exposed = chess.Board("3qk3/8/8/8/8/8/PPP5/3Q2K1 w - - 0 1")
        self.assertGreater(evaluate(shielded), evaluate(exposed))

    def test_evaluation_does_not_mutate_board(self) -> None:
        board = chess.Board("r3k2r/ppp2ppp/2n1bn2/3qp3/8/2NP1N2/PPP2PPP/R2QK2R w KQkq - 4 10")
        fen = board.fen()
        stack = list(board.move_stack)
        first = evaluate(board)
        second = evaluate(board)
        self.assertEqual(first, second)
        self.assertEqual(board.fen(), fen)
        self.assertEqual(board.move_stack, stack)


if __name__ == "__main__":
    unittest.main()
