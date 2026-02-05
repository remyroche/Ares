import unittest
import numpy as np
import pandas as pd
from extreme_price_movements.labeling import _numba_triple_barrier, compute_triple_barrier_labels

class TestTripleBarrier(unittest.TestCase):
    def test_numba_triple_barrier_long(self):
        # Create a simple price path
        # 100, 102, 106 (TP), 100
        # TP=0.05 (105). SL=0.02 (98).

        closes = np.array([100, 102, 106, 100], dtype=np.float32)
        highs = np.array([100, 102, 106, 100], dtype=np.float32)
        lows = np.array([100, 102, 106, 100], dtype=np.float32)
        times = np.array([0, 3600*1e9, 2*3600*1e9, 3*3600*1e9], dtype=np.int64) # Hourly

        tp = 0.05
        sl = 0.02
        horizon = 5 # Sufficient
        side = 1 # Long

        lbs, rets, idxs = _numba_triple_barrier(times, highs, lows, closes, tp, sl, horizon, side)

        # Test bar 0
        self.assertEqual(lbs[0], 1) # TP
        self.assertAlmostEqual(rets[0], 0.05)
        self.assertEqual(idxs[0], 2)

        # Test SL
        # 100, 99, 97 (SL), 100
        # SL=0.02 (98). 97 <= 98. SL hit.
        closes = np.array([100, 99, 97, 100], dtype=np.float32)
        highs = closes.copy()
        lows = closes.copy()

        lbs, rets, idxs = _numba_triple_barrier(times, highs, lows, closes, tp, sl, horizon, side)

        self.assertEqual(lbs[0], -1) # SL
        self.assertAlmostEqual(rets[0], -0.02)
        self.assertEqual(idxs[0], 2)

        # Test Time Exit
        closes = np.array([100, 101, 102, 101], dtype=np.float32) # No TP (105)
        highs = closes.copy()
        lows = closes.copy()
        horizon = 2

        lbs, rets, idxs = _numba_triple_barrier(times, highs, lows, closes, tp, sl, horizon, side)

        self.assertEqual(lbs[0], 0) # Time
        self.assertAlmostEqual(rets[0], 0.02)
        self.assertEqual(idxs[0], 2)

    def test_numba_triple_barrier_short(self):
        # Short Side
        # TP=0.05. Entry=100. TP Price = 95. SL Price = 102.

        # Test TP
        # 100, 98, 94 (TP), 96
        closes = np.array([100, 98, 94, 96], dtype=np.float32)
        highs = closes.copy()
        lows = closes.copy()
        times = np.array([0, 3600*1e9, 2*3600*1e9, 3*3600*1e9], dtype=np.int64)

        tp = 0.05
        sl = 0.02
        horizon = 5
        side = -1 # Short

        lbs, rets, idxs = _numba_triple_barrier(times, highs, lows, closes, tp, sl, horizon, side)

        # At bar 2: Low is 94 <= 95. TP hit.
        self.assertEqual(lbs[0], 1) # TP (Profit)
        self.assertAlmostEqual(rets[0], 0.05)
        self.assertEqual(idxs[0], 2)

        # Test SL
        # 100, 101, 103 (SL), 100
        # SL Price = 102. High 103 >= 102. SL hit.
        closes = np.array([100, 101, 103, 100], dtype=np.float32)
        highs = closes.copy()
        lows = closes.copy()

        lbs, rets, idxs = _numba_triple_barrier(times, highs, lows, closes, tp, sl, horizon, side)

        self.assertEqual(lbs[0], -1) # SL (Loss)
        self.assertAlmostEqual(rets[0], -0.02)
        self.assertEqual(idxs[0], 2)

    def test_wrapper(self):
        dates = pd.date_range("2021-01-01", periods=10, freq="1h")
        df = pd.DataFrame({
            "close": np.linspace(100, 110, 10, dtype=np.float32),
            "open": np.linspace(100, 110, 10, dtype=np.float32),
            "high": np.linspace(100, 110, 10, dtype=np.float32),
            "low": np.linspace(100, 110, 10, dtype=np.float32),
        }, index=dates)
        panel = {
            "close": pd.DataFrame({"A": df["close"]}),
            "open": pd.DataFrame({"A": df["open"]}),
            "high": pd.DataFrame({"A": df["high"]}),
            "low": pd.DataFrame({"A": df["low"]}),
        }

        # Default Long
        labels, rets = compute_triple_barrier_labels(panel, 0.05, 0.05, 5)
        self.assertEqual(labels.iloc[0, 0], 1)
        self.assertAlmostEqual(rets.iloc[0, 0], 0.05)

        # Short
        # 100 -> 110. Short hits SL (Price increase).
        # SL = 0.05. Entry=100. SL Price = 105.
        # Hit at 105.55 (index 5).
        labels_s, rets_s = compute_triple_barrier_labels(panel, 0.05, 0.05, 5, side="short")
        self.assertEqual(labels_s.iloc[0, 0], -1)
        self.assertAlmostEqual(rets_s.iloc[0, 0], -0.05)

if __name__ == '__main__':
    unittest.main()
