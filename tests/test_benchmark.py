from __future__ import annotations

import unittest

from cudarepo.benchmark import _percentile, effective_bandwidth_gbps


class BenchmarkMathTests(unittest.TestCase):
    def test_percentile_interpolates(self) -> None:
        values = [4.0, 1.0, 3.0, 2.0]
        self.assertEqual(_percentile(values, 0), 1.0)
        self.assertEqual(_percentile(values, 50), 2.5)
        self.assertEqual(_percentile(values, 100), 4.0)

    def test_effective_bandwidth(self) -> None:
        # 1 GB moved in 100 ms is 10 GB/s.
        self.assertAlmostEqual(effective_bandwidth_gbps(1_000_000_000, 100.0), 10.0)

    def test_invalid_inputs_fail_fast(self) -> None:
        with self.assertRaises(ValueError):
            _percentile([], 50)
        with self.assertRaises(ValueError):
            effective_bandwidth_gbps(1, 0)


if __name__ == "__main__":
    unittest.main()

