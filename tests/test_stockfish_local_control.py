from __future__ import annotations

import os
import shutil
import unittest
from pathlib import Path

import chess

from harness.sandbox import local

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "controls" / "stockfish_local"


@unittest.skipUnless(shutil.which("stockfish"), "Stockfish is not installed locally")
class StockfishLocalControlTests(unittest.TestCase):
    def test_contract_returns_a_legal_move_at_fixed_nodes(self) -> None:
        old = os.environ.get("AICHESSATHON_FIXED_NODES")
        os.environ["AICHESSATHON_FIXED_NODES"] = "1000"
        agent = local(CONTROL)
        try:
            agent.start(10)
            board = chess.Board()
            move = chess.Move.from_uci(agent.move(board.fen(), 10_000))
            self.assertIn(move, board.legal_moves)
        finally:
            agent.stop()
            if old is None:
                os.environ.pop("AICHESSATHON_FIXED_NODES", None)
            else:
                os.environ["AICHESSATHON_FIXED_NODES"] = old


if __name__ == "__main__":
    unittest.main()
