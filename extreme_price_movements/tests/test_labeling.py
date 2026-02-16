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
        opens = closes.copy()
        times = np.array([0, 3600*1e9, 2*3600*1e9, 3*3600*1e9], dtype=np.int64) # Hourly

        tp = np.full(4, 0.05, dtype=np.float32)
        sl = np.full(4, 0.02, dtype=np.float32)
        horizon = 5 # Sufficient
        side = 1 # Long

        lbs, rets, idxs = _numba_triple_barrier(times, opens, highs, lows, closes, tp, sl, horizon, side)

        # Test bar 0
        # Activated at 106 (>105).
        # Trail dev = 0.5 * 0.05 * 100 = 2.5.
        # Extreme = 106. SL = 103.5.
        # Bar 3 (100). Low 100 <= 103.5. SL Hit.
        # Return = 103.5/100 - 1 = 0.035.
        # Label = 1 (Trailing Active).
        self.assertEqual(lbs[0], 1)
        self.assertAlmostEqual(rets[0], 0.035)
        self.assertEqual(idxs[0], 3)

        # Test SL
        # 100, 99, 97 (SL), 100
        # SL=0.02 (98). 97 <= 98. SL hit.
        closes = np.array([100, 99, 97, 100], dtype=np.float32)
        highs = closes.copy()
        lows = closes.copy()
        opens = closes.copy()

        lbs, rets, idxs = _numba_triple_barrier(times, opens, highs, lows, closes, tp, sl, horizon, side)

        self.assertEqual(lbs[0], -1) # SL
        self.assertAlmostEqual(rets[0], -0.02)
        self.assertEqual(idxs[0], 2)

        # Test Time Exit (triggered by Stall Check at 1h)
        # Horizon 2. Stall 1h.
        # Bar 1 (1h): High 101. MFE 1%. Threshold 2.5% (0.5 * 5%).
        # 1% < 2.5%. Stall Exit.
        # Exit at Close[1] = 101. Return 0.01.
        closes = np.array([100, 101, 102, 101], dtype=np.float32) # No TP (105)
        highs = closes.copy()
        lows = closes.copy()
        opens = closes.copy()
        horizon = 2

        lbs, rets, idxs = _numba_triple_barrier(times, opens, highs, lows, closes, tp, sl, horizon, side)

        self.assertEqual(lbs[0], 0) # Time/Stall
        self.assertAlmostEqual(rets[0], 0.01)
        self.assertEqual(idxs[0], 1)

    def test_numba_triple_barrier_short(self):
        # Short Side
        # TP=0.05. Entry=100. TP Price = 95. SL Price = 102.

        # Test TP
        # 100, 98, 94 (TP), 96
        closes = np.array([100, 98, 94, 96], dtype=np.float32)
        highs = closes.copy()
        lows = closes.copy()
        opens = closes.copy()
        times = np.array([0, 3600*1e9, 2*3600*1e9, 3*3600*1e9], dtype=np.int64)

        tp = np.full(4, 0.05, dtype=np.float32)
        sl = np.full(4, 0.02, dtype=np.float32)
        horizon = 5
        side = -1 # Short

        lbs, rets, idxs = _numba_triple_barrier(times, opens, highs, lows, closes, tp, sl, horizon, side)

        # At bar 2: Low is 94 <= 95. Activated.
        # Trail dev = 2.5. Extreme = 94.
        # SL = 94 + 2.5 = 96.5.
        # Bar 3: High 96. 96 < 96.5. No Hit.
        # End of data -> Time Exit at Close[3] (96).
        # Return = 100/96 - 1 = 0.0416666.
        # Label = 0 (Time Exit overwrites Trailing Active? Yes per current logic).
        self.assertEqual(lbs[0], 0)
        self.assertAlmostEqual(rets[0], 100.0/96.0 - 1.0)
        self.assertEqual(idxs[0], 3)

        # Test SL
        # 100, 101, 103 (SL), 100
        # SL Price = 102. High 103 >= 102. SL hit.
        closes = np.array([100, 101, 103, 100], dtype=np.float32)
        highs = closes.copy()
        lows = closes.copy()
        opens = closes.copy()

        lbs, rets, idxs = _numba_triple_barrier(times, opens, highs, lows, closes, tp, sl, horizon, side)

        self.assertEqual(lbs[0], -1) # SL (Loss)
        self.assertAlmostEqual(rets[0], 100.0/102.0 - 1.0)
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
        # TP 0.05 (105). Hit at 105.55 (idx 5).
        # Trailing active. Linear up -> Time Exit at idx 5 (Horizon 5).
        # Return at 105.55: 105.55/100 - 1 = 5/90.
        # Label 0 (Time Exit).
        labels, rets = compute_triple_barrier_labels(panel, 0.05, 0.05, 5)
        self.assertEqual(labels.iloc[0, 0], 0)
        self.assertAlmostEqual(rets.iloc[0, 0], 5.0/90.0)

        # Short
        # 100 -> 110.
        # SL = 0.05. Entry=100. SL Price = 105.
        # Horizon 5. Stall 2.5h.
        # At index 3 (3h), Price 103.33. MFE=0 (Low never dropped below 100).
        # Stall triggers (0 < 2.5%). Exit at 103.33.
        # Return = 100/103.33 - 1 = -0.032258
        labels_s, rets_s = compute_triple_barrier_labels(panel, 0.05, 0.05, 5, side="short")
        self.assertEqual(labels_s.iloc[0, 0], 0) # Stall/Time Exit
        self.assertAlmostEqual(rets_s.iloc[0, 0], 100.0/(100.0 + 30.0/9.0) - 1.0)

    def test_dynamic_barrier(self):
        # Test array input to wrapper
        dates = pd.date_range("2021-01-01", periods=10, freq="1h")
        df = pd.DataFrame({
            "close": np.linspace(100, 110, 10, dtype=np.float32),
            "high": np.linspace(100, 110, 10, dtype=np.float32),
            "low": np.linspace(100, 110, 10, dtype=np.float32),
        }, index=dates)

        # Variable TP/SL
        # First 5: tp=0.01 (hit immediately)
        # Last 5: tp=0.20 (not hit)
        tp_arr = np.concatenate([np.full(5, 0.01), np.full(5, 0.20)])
        sl_arr = np.full(10, 0.05)

        panel = {
            "close": pd.DataFrame({"A": df["close"]}),
            "high": pd.DataFrame({"A": df["high"]}),
            "low": pd.DataFrame({"A": df["low"]}),
            "open": pd.DataFrame({"A": df["close"]}), # dummy
        }

        # We need to construct DataFrames for TP/SL
        tp_df = pd.DataFrame({"A": tp_arr}, index=dates)
        sl_df = pd.DataFrame({"A": sl_arr}, index=dates)

        labels, rets = compute_triple_barrier_labels(panel, tp_df, sl_df, 5)

        # First one should be Time Exit (because linear up, never hits trailing SL)
        # Entry 100. TP 101. Hit. Trailing.
        # Exit at Horizon 5 (105.55).
        # Return 0.0555. Label 0.
        self.assertEqual(labels.iloc[0, 0], 0)
        self.assertAlmostEqual(rets.iloc[0, 0], 5.0/90.0)

        # Check an index where TP is large
        # Index 6. Entry ~106. TP 20% -> 127. Close goes to 110.
        # Time exit.
        # But lookahead limit is 5.
        # 6+5=11 > 10. So it goes to end (110).
        self.assertEqual(labels.iloc[6, 0], 0)

if __name__ == '__main__':
    unittest.main()
