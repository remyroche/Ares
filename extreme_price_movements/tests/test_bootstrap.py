import unittest
import numpy as np
import pandas as pd
from extreme_price_movements.sequential_bootstrap import (
    get_label_intervals,
    _seq_bootstrap_numba,
    get_sequential_bootstrap_samples,
    seq_bootstrap
)

class TestSequentialBootstrap(unittest.TestCase):
    def test_get_label_intervals(self):
        price_times = pd.to_datetime(["2021-01-01 10:00", "2021-01-01 11:00", "2021-01-01 12:00", "2021-01-01 13:00"])
        # Indices: 0, 1, 2, 3

        # L1: 10:00 to 11:00 -> [0, 1]
        # L2: 12:00 to 13:00 -> [2, 3]
        # L3: 11:00 to 12:00 -> [1, 2]

        label_times = pd.DataFrame({
            "t_start": pd.to_datetime(["2021-01-01 10:00", "2021-01-01 12:00", "2021-01-01 11:00"]),
            "t_end":   pd.to_datetime(["2021-01-01 11:00", "2021-01-01 13:00", "2021-01-01 12:00"])
        })

        starts, ends, valid = get_label_intervals(label_times, price_times)

        np.testing.assert_array_equal(starts, [0, 2, 1])
        np.testing.assert_array_equal(ends,   [1, 3, 2])
        self.assertTrue(np.all(valid))

    def test_seq_bootstrap_logic(self):
        # 5 bars
        # A: [0, 2] (3 bars)
        # B: [2, 4] (3 bars)
        # C: [0, 4] (5 bars)
        starts = np.array([0, 2, 0])
        ends = np.array([2, 4, 4])
        valid = np.array([True, True, True])
        num_bars = 5
        num_samples = 2
        random_state = 42

        selected = _seq_bootstrap_numba(starts, ends, valid, num_bars, num_samples, random_state)

        self.assertEqual(len(selected), 2)
        self.assertEqual(len(np.unique(selected)), 2)

    def test_full_pipeline(self):
        price_times = pd.date_range("2021-01-01", periods=100, freq="h")
        # Create 50 labels
        starts = price_times[0:50]
        ends = price_times[10:60]
        label_times = pd.DataFrame({"t_start": starts, "t_end": ends})

        samples = get_sequential_bootstrap_samples(label_times, price_times, n_samples=10, random_state=1)
        self.assertEqual(len(samples), 10)
        # Check uniqueness
        self.assertEqual(len(np.unique(samples)), 10)

    def test_empty_case(self):
        price_times = pd.date_range("2021-01-01", periods=10, freq="h")
        label_times = pd.DataFrame({"t_start": [], "t_end": []})

        # Should handle empty gracefully
        samples = get_sequential_bootstrap_samples(label_times, price_times, n_samples=0, random_state=1)
        self.assertEqual(len(samples), 0)

        # If n_samples > available (0), it returns 0 samples
        samples2 = get_sequential_bootstrap_samples(label_times, price_times, n_samples=5, random_state=1)
        self.assertEqual(len(samples2), 0)

if __name__ == '__main__':
    unittest.main()
