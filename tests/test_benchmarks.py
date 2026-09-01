from __future__ import annotations

import unittest

from benchmarks.run import elo_from_score, score_interval


class BenchmarkTests(unittest.TestCase):
    def test_score_interval_contains_sample_mean(self) -> None:
        scores = [1.0, 1.0, 0.5, 0.0]
        low, high = score_interval(scores)
        mean = sum(scores) / len(scores)
        self.assertLess(low, mean)
        self.assertGreater(high, mean)

    def test_elo_is_zero_at_even_score(self) -> None:
        self.assertEqual(elo_from_score(0.5), 0.0)


if __name__ == "__main__":
    unittest.main()
