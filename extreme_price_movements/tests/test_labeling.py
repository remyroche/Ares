import unittest
import pandas as pd
import numpy as np
from extreme_price_movements.labeling import compute_triple_barrier_labels

class TestTripleBarrier(unittest.TestCase):
    def test_basic_long(self):
        # Create synthetic price data
        # 0: 100
        # 1: 102 (Hit +2% profit if pt=0.015)
        # 2: 90 (Hit stop if not already exited)

        dates = pd.date_range("2021-01-01", periods=10, freq="h")
        prices = pd.DataFrame({
            "close": [100, 102, 90, 100, 100, 100, 100, 100, 100, 100],
            "high":  [101, 103, 91, 101, 101, 101, 101, 101, 101, 101],
            "low":   [99,  101, 89, 99,  99,  99,  99,  99,  99,  99]
        }, index=dates)

        # Event at t=0
        events = pd.DataFrame(index=[dates[0]])
        events['pt'] = 0.015
        events['sl'] = 0.05
        events['horizon'] = 5

        # Expectation:
        # Entry at t=0 close=100.
        # t=1: High=103. 103 > 100*(1.015)=101.5. Profit hit.
        # Should exit at t=1 with label=1.

        res = compute_triple_barrier_labels(prices, events)

        self.assertEqual(res.iloc[0]['label'], 1)
        self.assertAlmostEqual(res.iloc[0]['ret'], 0.015) # Returns pt
        self.assertEqual(res.iloc[0]['exit_ts'], dates[1])

    def test_basic_short(self):
        dates = pd.date_range("2021-01-01", periods=10, freq="h")
        prices = pd.DataFrame({
            "close": [100, 98, 110, 100, 100, 100, 100, 100, 100, 100],
            "high":  [101, 99, 111, 101, 101, 101, 101, 101, 101, 101],
            "low":   [99,  97, 109, 99,  99,  99,  99,  99,  99,  99]
        }, index=dates)

        # Event at t=0, Side=-1 (Short)
        events = pd.DataFrame(index=[dates[0]])
        events['pt'] = 0.015
        events['sl'] = 0.05
        events['horizon'] = 5
        events['side'] = -1

        # Expectation:
        # Entry 100.
        # t=1: Low=97. 97 < 100*(1-0.015)=98.5. Profit hit.

        res = compute_triple_barrier_labels(prices, events, side_col='side')

        self.assertEqual(res.iloc[0]['label'], 1)
        self.assertAlmostEqual(res.iloc[0]['ret'], 0.015)
        self.assertEqual(res.iloc[0]['exit_ts'], dates[1])

if __name__ == '__main__':
    unittest.main()
