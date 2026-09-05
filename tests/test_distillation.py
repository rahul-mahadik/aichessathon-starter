from __future__ import annotations

import io
import tempfile
import unittest
import urllib.error
from pathlib import Path

import chess
import numpy as np
import torch

from distill.build_dataset import build_parallel
from distill.features import FEATURE_DIM, PADDING_INDEX, encode_board, record_to_group
from distill.inspect_teacher import inspect
from distill.mine_depth_disagreements import depth_distance, mine
from distill.sample_gigafish import (
    _download,
    _valid_unique_positions,
    reservoir_sample,
    sample_corpus,
)
from distill.schema import Candidate, TeacherRecord, TeacherScore, read_records, write_records
from nnue_runtime import IncrementalQuantizedEvaluator, QuantizedEvaluator
from training.nnue import SparseValueNetwork, export_quantized, initialize_from_quantized
from training.train_distilled import Batch, losses


class DistillationTests(unittest.TestCase):
    def test_download_retries_rate_limits(self) -> None:
        rate_limit = urllib.error.HTTPError(
            "https://example.test/data", 429, "rate limited", {"Retry-After": "1"}, None
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "data.parquet"
            with (
                unittest.mock.patch(
                    "distill.sample_gigafish.urllib.request.urlopen",
                    side_effect=[rate_limit, io.BytesIO(b"payload")],
                ),
                unittest.mock.patch("distill.sample_gigafish.time.sleep") as sleep,
            ):
                _download("https://example.test/data", destination)
            self.assertEqual(destination.read_bytes(), b"payload")
            sleep.assert_called_once_with(1.0)

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
        positions = _valid_unique_positions([same_features.fen(), alternative.fen()], 1, excluded)
        self.assertEqual(positions, [alternative.fen()])

    def test_corpus_can_create_a_single_nonempty_tier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            source = cache / "data-00001.parquet"
            source.touch()
            rows = [chess.STARTING_FEN]
            with (
                unittest.mock.patch("distill.sample_gigafish._download"),
                unittest.mock.patch("distill.sample_gigafish._sha256", return_value="test"),
                unittest.mock.patch(
                    "distill.sample_gigafish._parquet_rows", return_value=iter(rows)
                ),
            ):
                manifest = sample_corpus(
                    output=root / "corpus",
                    cache=cache,
                    medium_positions=1,
                    deep_positions=0,
                    shards_per_tier=1,
                    source_shards=[1],
                    seed=7,
                )

        self.assertEqual(set(manifest["tiers"]), {"medium"})
        self.assertEqual(manifest["tiers"]["medium"]["positions"], 1)

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

    def test_parallel_dataset_build_merges_partitions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = [root / "raw-0.jsonl.gz", root / "raw-1.jsonl.gz"]
            for path in inputs:
                write_records(path, [self.record()])
            metadata = build_parallel(
                inputs,
                root / "dataset",
                records_per_shard=10,
                cp_scale=600.0,
                workers=2,
            )
            shards = sorted((root / "dataset").glob("part-*.npz"))

        self.assertEqual(metadata["records"], 2)
        self.assertEqual(metadata["workers"], 2)
        self.assertEqual(len(shards), 2)

    def test_child_target_flips_perspective(self) -> None:
        group = record_to_group(self.record())
        self.assertTrue(group.value_mask[0])
        self.assertTrue(group.value_mask[1])
        self.assertAlmostEqual(float(group.targets[0]), 0.14, places=5)
        self.assertAlmostEqual(float(group.targets[1]), -0.14, places=5)
        self.assertTrue(np.all(group.features[2:] == PADDING_INDEX))

    def test_depth_distance_detects_best_move_and_ranking_change(self) -> None:
        shallow = TeacherRecord(
            fen=chess.STARTING_FEN,
            teacher="Stockfish 18",
            node_budget=100_000,
            depth_budget=None,
            multipv=2,
            root_score=TeacherScore(cp=20, mate=None, wdl=None),
            candidates=(
                Candidate(
                    move="e2e4",
                    score=TeacherScore(cp=30, mate=None, wdl=None),
                    pv=(),
                    depth=12,
                    seldepth=16,
                    nodes=100_000,
                ),
                Candidate(
                    move="d2d4",
                    score=TeacherScore(cp=10, mate=None, wdl=None),
                    pv=(),
                    depth=12,
                    seldepth=16,
                    nodes=100_000,
                ),
            ),
            best_move="e2e4",
        )
        deep = TeacherRecord(
            fen=chess.STARTING_FEN,
            teacher="Stockfish 18",
            node_budget=1_000_000,
            depth_budget=None,
            multipv=2,
            root_score=TeacherScore(cp=45, mate=None, wdl=None),
            candidates=(
                Candidate(
                    move="d2d4",
                    score=TeacherScore(cp=60, mate=None, wdl=None),
                    pv=(),
                    depth=18,
                    seldepth=24,
                    nodes=1_000_000,
                ),
                Candidate(
                    move="e2e4",
                    score=TeacherScore(cp=5, mate=None, wdl=None),
                    pv=(),
                    depth=18,
                    seldepth=24,
                    nodes=1_000_000,
                ),
            ),
            best_move="d2d4",
        )
        distance = depth_distance(shallow, deep, ranking_margin=0.0)
        self.assertTrue(distance.top_move_changed)
        self.assertGreater(distance.top_three_distance, 0.0)
        self.assertEqual(distance.common_candidates, 2)
        self.assertEqual(distance.ranking_disagreement, 1.0)
        self.assertGreater(distance.distance, 2.0)

    def test_depth_distance_treats_disjoint_candidates_as_disagreement(self) -> None:
        shallow = self.record()
        score = TeacherScore(cp=-35, mate=None, wdl=(200, 500, 300))
        deep = TeacherRecord(
            fen=shallow.fen,
            teacher=shallow.teacher,
            node_budget=1_000_000,
            depth_budget=None,
            multipv=1,
            root_score=score,
            candidates=(
                Candidate(
                    move="d2d4",
                    score=score,
                    pv=(),
                    depth=20,
                    seldepth=28,
                    nodes=1_000_000,
                ),
            ),
            best_move="d2d4",
        )
        distance = depth_distance(shallow, deep)
        self.assertEqual(distance.common_candidates, 0)
        self.assertEqual(distance.candidate_value_mae, 2.0)
        self.assertEqual(distance.ranking_disagreement, 1.0)
        self.assertTrue(distance.top_move_changed)

    def test_depth_miner_emits_matched_ablation_controls(self) -> None:
        boards: list[chess.Board] = []
        for move in (None, "e2e4", "d2d4", "c2c4", "g1f3", "b2b3"):
            board = chess.Board()
            if move is not None:
                board.push_uci(move)
            boards.append(board)

        def teacher_record(board: chess.Board, node_budget: int, reverse: bool) -> TeacherRecord:
            moves = list(board.legal_moves)[:3]
            if reverse:
                moves.reverse()
            candidates = tuple(
                Candidate(
                    move=move.uci(),
                    score=TeacherScore(cp=90 - 30 * index, mate=None, wdl=None),
                    pv=(),
                    depth=12,
                    seldepth=16,
                    nodes=node_budget,
                )
                for index, move in enumerate(moves)
            )
            return TeacherRecord(
                fen=board.fen(),
                teacher="Stockfish 18",
                node_budget=node_budget,
                depth_budget=None,
                multipv=3,
                root_score=candidates[0].score,
                candidates=candidates,
                best_move=candidates[0].move,
            )

        low = [teacher_record(board, 10_000, False) for board in boards]
        medium = [
            teacher_record(board, 100_000, index % 3 == 0) for index, board in enumerate(boards)
        ]
        deep = [
            teacher_record(board, 1_000_000, index % 2 == 0) for index, board in enumerate(boards)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            low_path = root / "low.jsonl.gz"
            medium_path = root / "medium.jsonl.gz"
            deep_path = root / "deep.jsonl.gz"
            write_records(low_path, low)
            write_records(medium_path, medium)
            write_records(deep_path, deep)
            report = mine(
                [low_path],
                [medium_path],
                [deep_path],
                output=root / "mined",
                select=3,
                ranking_margin=0.0,
                seed=11,
                holdout=1,
            )
            high_medium = list(read_records([root / "mined/high-medium.jsonl.gz"]))
            high_deep = list(read_records([root / "mined/high-deep.jsonl.gz"]))
            random_deep = list(read_records([root / "mined/random-deep.jsonl.gz"]))
            holdout_deep = list(read_records([root / "mined/holdout-deep.jsonl.gz"]))
            bucket_positions = sum(
                len(path.read_text().splitlines())
                for path in (root / "mined/buckets").glob("*.epd")
            )

        self.assertEqual(report["positions"], 6)
        self.assertEqual(report["training_pool"], 5)
        self.assertEqual(len(high_medium), 3)
        self.assertEqual(len(random_deep), 3)
        self.assertEqual(len(holdout_deep), 1)
        self.assertEqual(
            [record.fen for record in high_medium],
            [record.fen for record in high_deep],
        )
        self.assertTrue(all(record.node_budget == 100_000 for record in high_medium))
        self.assertTrue(all(record.node_budget == 1_000_000 for record in high_deep))
        self.assertFalse(
            {record.fen for record in holdout_deep}
            & {record.fen for record in high_deep + random_deep}
        )
        self.assertEqual(bucket_positions, 1)

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

    def test_training_reuses_accumulator_for_turn_flip(self) -> None:
        torch.manual_seed(4)
        model = SparseValueNetwork(accumulator=16, hidden=12, bottleneck=8).eval()
        features = torch.as_tensor(encode_board(chess.Board())).long().unsqueeze(0)
        turns = torch.tensor([True])
        direct, flipped = model.forward_with_flipped_turns(features, turns)
        self.assertTrue(torch.equal(direct, model(features, turns)))
        self.assertTrue(torch.equal(flipped, model(features, ~turns)))

    def test_narrow_student_can_reuse_and_freeze_sparse_teacher(self) -> None:
        torch.manual_seed(11)
        teacher = SparseValueNetwork(accumulator=16, hidden=12, bottleneck=8).eval()
        student = SparseValueNetwork(accumulator=16, hidden=6, bottleneck=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.npz"
            export_quantized(teacher, path)
            copied = initialize_from_quantized(student, path, freeze_feature=True)
            with np.load(path, allow_pickle=False) as archive:
                expected = archive["feature_q"].astype(np.float32) * np.float32(
                    archive["feature_scale"]
                )

        self.assertEqual(copied, ["feature", "feature_bias"])
        np.testing.assert_allclose(
            student.feature.weight[:FEATURE_DIM].detach().numpy(), expected
        )
        np.testing.assert_allclose(
            student.feature_bias.detach().numpy(), teacher.feature_bias.detach().numpy()
        )
        self.assertFalse(student.feature.weight.requires_grad)
        self.assertFalse(student.feature_bias.requires_grad)
        self.assertTrue(student.hidden_one.weight.requires_grad)

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

    def test_incremental_runtime_matches_reference_through_move_stacks(self) -> None:
        torch.manual_seed(6)
        model = SparseValueNetwork(accumulator=16, hidden=12, bottleneck=8).eval()
        cases = [
            (
                chess.Board(),
                ["e2e4", "c7c5", "g1f3", "d7d6", "f1b5", "b8c6"],
            ),
            (chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"), ["e1g1"]),
            (chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"), ["e5d6"]),
            (chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1"), ["a7a8q"]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nnue.npz"
            export_quantized(model, path)
            reference = QuantizedEvaluator(path, antisymmetric=True)
            incremental = IncrementalQuantizedEvaluator(path, antisymmetric=True)
            for board, moves in cases:
                incremental.begin(board)
                self.assertAlmostEqual(
                    incremental.raw_value(board), reference.raw_value(board), places=6
                )
                for move_text in moves:
                    move = chess.Move.from_uci(move_text)
                    self.assertIn(move, board.legal_moves)
                    incremental.push(board, move)
                    board.push(move)
                    self.assertAlmostEqual(
                        incremental.raw_value(board), reference.raw_value(board), places=6
                    )
                for _ in moves:
                    incremental.pop()
                    board.pop()
                    self.assertAlmostEqual(
                        incremental.raw_value(board), reference.raw_value(board), places=6
                    )

    def test_training_antisymmetry_loss_uses_turn_flipped_predictions(self) -> None:
        predictions = torch.tensor([[0.2, -0.8, -0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        batch: Batch = {
            "features": torch.zeros((1, 9, 2, 32), dtype=torch.long),
            "turns": torch.ones((1, 9), dtype=torch.bool),
            "targets": predictions.clone(),
            "value_mask": torch.tensor(
                [[True, True, True, False, False, False, False, False, False]]
            ),
            "candidate_scores": torch.tensor([[0.8, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            "candidate_mask": torch.tensor(
                [[True, True, False, False, False, False, False, False]]
            ),
        }
        _, symmetric_metrics = losses(
            predictions,
            -predictions,
            batch,
            ranking_weight=0.25,
            top_move_weight=0.25,
            antisymmetry_weight=1.0,
            top_k=2,
            top_k_ranking_boost=4.0,
        )
        _, violating_metrics = losses(
            predictions,
            predictions,
            batch,
            ranking_weight=0.25,
            top_move_weight=0.25,
            antisymmetry_weight=1.0,
            top_k=2,
            top_k_ranking_boost=4.0,
        )
        self.assertEqual(symmetric_metrics["antisymmetry_mse"], 0.0)
        self.assertGreater(violating_metrics["antisymmetry_mse"], 0.0)


if __name__ == "__main__":
    unittest.main()
