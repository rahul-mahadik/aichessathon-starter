from __future__ import annotations

import os
import unittest
from pathlib import Path

import chess

from harness.sandbox import local

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "controls" / "sunfish_dupe"


class SunfishDupeControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.old_fixed_nodes = os.environ.get("AICHESSATHON_FIXED_NODES")
        os.environ["AICHESSATHON_FIXED_NODES"] = "1000"

    def tearDown(self) -> None:
        if self.old_fixed_nodes is None:
            os.environ.pop("AICHESSATHON_FIXED_NODES", None)
        else:
            os.environ["AICHESSATHON_FIXED_NODES"] = self.old_fixed_nodes

    def test_contract_returns_legal_moves_for_both_colours(self) -> None:
        agent = local(CONTROL)
        try:
            agent.start(10)
            board = chess.Board()
            first = chess.Move.from_uci(agent.move(board.fen(), 10_000))
            self.assertIn(first, board.legal_moves)
            board.push(first)
            reply = chess.Move.from_uci(agent.move(board.fen(), 10_000))
            self.assertIn(reply, board.legal_moves)
        finally:
            agent.stop()

    def test_handles_arbitrary_fen_and_exact_node_cap(self) -> None:
        agent = local(CONTROL)
        try:
            agent.start(10)
            board = chess.Board(
                "r3k2r/ppp2ppp/2n1bn2/3qp3/8/2NPBN2/PPP2PPP/R2Q1RK1 b kq - 3 10"
            )
            move = chess.Move.from_uci(agent.move(board.fen(), 10_000))
            self.assertIn(move, board.legal_moves)
        finally:
            agent.stop()
        self.assertIn("nodes=1000", agent.stderr_tail)


if __name__ == "__main__":
    unittest.main()
