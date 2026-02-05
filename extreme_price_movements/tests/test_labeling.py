import unittest
import numpy as np
import pandas as pd
from extreme_price_movements.labeling import _numba_triple_barrier, compute_triple_barrier_labels

class TestTripleBarrier(unittest.TestCase):
    def test_numba_triple_barrier(self):
        # Create a simple price path
        # 100, 102, 106 (TP), 100
        # TP=0.05 (105). SL=0.02 (98).
        # i=0 (100). Next bars: 102, 106. 106 >= 105. TP hit at idx 2.

        closes = np.array([100, 102, 106, 100], dtype=np.float32)
        opens = np.array([100, 102, 106, 100], dtype=np.float32)
        highs = np.array([100, 102, 106, 100], dtype=np.float32)
        lows = np.array([100, 102, 106, 100], dtype=np.float32)
        times = np.array([0, 3600*1e9, 2*3600*1e9, 3*3600*1e9], dtype=np.int64) # Hourly

        tp = 0.05
        sl = 0.02
        horizon = 5 # Sufficient

        lbs, rets, idxs = _numba_triple_barrier(times, highs, lows, closes, tp, sl, horizon)

        # Test bar 0
        self.assertEqual(lbs[0], 1) # TP
        self.assertAlmostEqual(rets[0], 0.05)
        self.assertEqual(idxs[0], 2)

        # Test SL
        # 100, 99, 97 (SL), 100
        # SL=0.02 (98). 97 <= 98. SL hit.
        closes = np.array([100, 99, 97, 100], dtype=np.float32)
        opens = closes.copy()
        highs = closes.copy()
        lows = closes.copy()

        lbs, rets, idxs = _numba_triple_barrier(times, highs, lows, closes, tp, sl, horizon)

        self.assertEqual(lbs[0], -1) # SL
        self.assertAlmostEqual(rets[0], -0.02)
        self.assertEqual(idxs[0], 2)

        # Test Time Exit
        # 100, 101, 101, 101
        # Horizon = 2 hours.
        # t=0. horizon = 2h. cutoff = t+2h.
        # Bar 0 (0h). Bar 1 (1h). Bar 2 (2h). Bar 3 (3h).
        # At bar 2 (2h), time >= cutoff. Exit.

        closes = np.array([100, 101, 102, 101], dtype=np.float32) # No TP (105)
        opens = closes.copy()
        highs = closes.copy()
        lows = closes.copy()
        horizon = 2

        lbs, rets, idxs = _numba_triple_barrier(times, highs, lows, closes, tp, sl, horizon)

        self.assertEqual(lbs[0], 0) # Time
        # Return at idx 2 (102) from 100 -> 0.02
        self.assertAlmostEqual(rets[0], 0.02)
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

        labels, rets = compute_triple_barrier_labels(panel, 0.05, 0.05, 5)

        # 100 -> 105 is +5%. Should hit TP around index 5.
        self.assertEqual(labels.iloc[0, 0], 1)
        self.assertAlmostEqual(rets.iloc[0, 0], 0.05)

if __name__ == '__main__':
    unittest.main()
