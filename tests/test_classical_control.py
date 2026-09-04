from __future__ import annotations

import importlib.util
import sys
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import chess

from harness.package import build

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "controls" / "classical"


def load_engine_module():  # type: ignore[no-untyped-def]
    sys.path.insert(0, str(CONTROL))
    try:
        spec = importlib.util.spec_from_file_location(
            "classical_control_engine", CONTROL / "classical_engine.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load classical engine")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(CONTROL))


ENGINE = load_engine_module()


class ClassicalControlTests(unittest.TestCase):
    def test_search_restores_board_at_node_limit(self) -> None:
        board = chess.Board()
        original = board.fen()
        result = ENGINE.ClassicalEngine().choose_fixed_nodes(board, 1_000)
        self.assertIn(result.move, board.legal_moves)
        self.assertEqual(result.nodes, 1_000)
        self.assertEqual(board.fen(), original)

    def test_finds_mate_in_one(self) -> None:
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
        result = ENGINE.ClassicalEngine().choose(board, 1_000)
        board.push(result.move)
        self.assertTrue(board.is_checkmate(), result.move.uci())

    def test_returns_only_legal_move(self) -> None:
        board = chess.Board("3n4/4r1b1/r1P5/3NR2P/3p3P/P5pN/3q4/k2K4 w - - 2 52")
        legal = list(board.legal_moves)
        self.assertEqual(len(legal), 1)
        result = ENGINE.ClassicalEngine().choose(board, 100)
        self.assertEqual(result.move, legal[0])

    def test_rejects_terminal_position(self) -> None:
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        with self.assertRaises(ValueError):
            ENGINE.ClassicalEngine().choose(board, 100)

    def test_control_builds_as_small_submission_shaped_zip(self) -> None:
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "classical-control.zip"
            written = build(CONTROL, destination, ())
            self.assertIn("agent.py", written)
            self.assertLess(sum((CONTROL / name).stat().st_size for name in written), 50 * 2**20)
            with zipfile.ZipFile(destination) as archive:
                self.assertEqual(set(archive.namelist()), set(written))
                self.assertFalse(any(name.endswith((".so", ".dylib", ".exe")) for name in written))


if __name__ == "__main__":
    unittest.main()
