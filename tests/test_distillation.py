from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np
import torch

from distill.features import FEATURE_DIM, PADDING_INDEX, encode_board, record_to_group
from distill.inspect_teacher import inspect
from distill.sample_gigafish import _valid_unique_positions, reservoir_sample
from distill.schema import Candidate, TeacherRecord, TeacherScore, read_records, write_records
from nnue_runtime import QuantizedEvaluator
from training.nnue import SparseValueNetwork, export_quantized


class DistillationTests(unittest.TestCase):
    def test_reservoir_sample_is_bounded_and_deterministic(self) -> None:
        rows = [str(index) for index in range(100)]
        first = reservoir_sample(rows, 10, seed=7)
        second = reservoir_sample(rows, 10, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertEqual(len(set(first)), 10)

    def test_corpus_exclusion_uses_evaluator_visible_position(self) -> None:
        starting = chess.Board()
        excluded = {f"{starting.board_fen()} w"}
        same_features = starting.copy()
        same_features.halfmove_clock = 12
        same_features.fullmove_number = 42
        alternative = starting.copy()
        alternative.push_uci("e2e4")
        positions = _valid_unique_positions(
            [same_features.fen(), alternative.fen()], 1, excluded
        )
        self.assertEqual(positions, [alternative.fen()])

    def record(self) -> TeacherRecord:
        score = TeacherScore(cp=42, mate=None, wdl=(320, 500, 180))
        candidate = Candidate(
            move="e2e4",
            score=score,
            pv=("e2e4", "e7e5"),
            depth=12,
            seldepth=18,
            nodes=10_000,
        )
        return TeacherRecord(
            fen=chess.STARTING_FEN,
            teacher="Stockfish 18",
            node_budget=10_000,
            depth_budget=None,
            multipv=1,
            root_score=score,
            candidates=(candidate,),
            best_move="e2e4",
        )

    def test_schema_gzip_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.jsonl.gz"
            self.assertEqual(write_records(path, [self.record()]), 1)
            self.assertEqual(list(read_records([path])), [self.record()])

    def test_teacher_inspection_validates_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.jsonl.gz"
            write_records(path, [self.record()])
            summary = inspect([path])
        self.assertEqual(summary["records"], 1)
        self.assertEqual(summary["candidates_per_record"], 1.0)
        self.assertEqual(summary["duplicate_fens"], 0)
        self.assertEqual(summary["incomplete_multipv_records"], 0)

    def test_child_target_flips_perspective(self) -> None:
        group = record_to_group(self.record())
        self.assertTrue(group.value_mask[0])
        self.assertTrue(group.value_mask[1])
        self.assertAlmostEqual(float(group.targets[0]), 0.14, places=5)
        self.assertAlmostEqual(float(group.targets[1]), -0.14, places=5)
        self.assertTrue(np.all(group.features[2:] == PADDING_INDEX))

    def test_sparse_indices_are_in_range(self) -> None:
        encoded = encode_board(chess.Board())
        active = encoded[encoded != PADDING_INDEX]
        self.assertEqual(active.size, 64)
        self.assertTrue(np.all(active >= 0))
        self.assertTrue(np.all(active < FEATURE_DIM))

    def test_quantized_runtime_tracks_pytorch(self) -> None:
        torch.manual_seed(3)
        model = SparseValueNetwork(accumulator=16, hidden=12, bottleneck=8).eval()
        board = chess.Board()
        features = torch.as_tensor(encode_board(board)).long().unsqueeze(0)
        turns = torch.tensor([True])
        expected = float(model(features, turns).item())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nnue.npz"
            export_quantized(model, path)
            actual = QuantizedEvaluator(path).raw_value(board)
        self.assertAlmostEqual(actual, expected, delta=0.02)

    def test_antisymmetric_runtime_flips_with_turn(self) -> None:
        torch.manual_seed(5)
        model = SparseValueNetwork(accumulator=16, hidden=12, bottleneck=8).eval()
        board = chess.Board()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nnue.npz"
            export_quantized(model, path)
            evaluator = QuantizedEvaluator(path, antisymmetric=True)
            first = evaluator.raw_value(board)
            board.turn = not board.turn
            second = evaluator.raw_value(board)
        self.assertAlmostEqual(first, -second, places=6)


if __name__ == "__main__":
    unittest.main()
