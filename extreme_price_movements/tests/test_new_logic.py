import unittest
import numpy as np
import pandas as pd
from extreme_price_movements.candidates import select_trade_candidates_vectorized
from extreme_price_movements.fast_funcs import simulate_trade_numba
from extreme_price_movements.meta_model import MetaModel

class TestNewLogic(unittest.TestCase):
    def test_select_trade_candidates_vectorized(self):
        # Setup data
        dates = pd.date_range("2021-01-01", periods=100, freq="1h")
        # Sym A: Top mover at t=50.
        # Sym B: Flat.
        metric = pd.DataFrame(index=dates, columns=["A", "B"], dtype=float)
        metric[:] = 0.0
        metric.loc[dates[50], "A"] = 1.0 # High value
        metric.loc[dates[50], "B"] = -1.0 # Low value

        feats = {"ret24h": metric}

        # Panel for Vol Filter
        # Make Vol High enough (>= 8%)
        # 12h range.
        c = pd.DataFrame(100.0, index=dates, columns=["A", "B"])
        h = c.copy()
        l = c.copy()

        # At t=50 and surrounding, make volatility high for A
        # Range = 10. Close = 100. Diff = 10/100 = 10% > 8%.
        h.loc[dates[40:60], "A"] = 110.0
        l.loc[dates[40:60], "A"] = 100.0
        c.loc[dates[40:60], "A"] = 100.0

        panel = {"close": c, "high": h, "low": l}

        # Test
        # pct=0.1. A is top (rank 1.0), B is bot (rank 0.0). Both candidates.
        mask = select_trade_candidates_vectorized(panel, feats, pct=0.1, metric="ret24h")

        # Check base candidate t=50
        # A should be True (Vol OK, Rank OK)
        self.assertTrue(mask.loc[dates[50], "A"])

        # Check shifted candidates
        # t-12 (38), t+4 (54)
        # Shift offsets: -12, -8, -4, 4, 8, 12, 16.
        # If t=50 is candidate, then t+4=54 should be True (Shift(4) logic) IF Vol OK.
        # Wait, Shift(4) shifts data at t to t+4.
        # So mask[54] comes from mask[50].
        # Vol check is applied AFTER shift?
        # My code:
        # expanded_mask = ...
        # vol_mask = ...
        # final = expanded & vol
        # So Vol must be valid AT t=54.

        # At t=54: Vol window [43, 54]. High=110, Low=100. Vol=10%. OK.
        self.assertTrue(mask.loc[dates[54], "A"])

        # t=38 (shift -12). From t=50.
        # Vol check at t=38. Window [27, 38]. Prices flat (100). Vol=0.
        # So should be False.
        self.assertFalse(mask.loc[dates[38], "A"])

        # B: Rank OK (Bot). But Vol is 0. So False.
        self.assertFalse(mask.loc[dates[50], "B"])

    def test_simulate_trade_numba(self):
        # 100 -> 102 -> 108 -> 90
        opens = np.array([100, 102, 108, 90], dtype=np.float32)
        highs = np.array([101, 103, 109, 92], dtype=np.float32)
        lows = np.array([99, 101, 107, 85], dtype=np.float32)
        closes = np.array([100, 102, 108, 90], dtype=np.float32)

        entry = 100.0
        side = 1 # Long

        # Params
        sl_dist = 5.0 # SL at 95
        act_dist = 6.0 # Activate at 106.
        tr_dist = 2.0 # Trail by 2.

        # Path:
        # 0: L=99 > 95. H=101 < 106. No Act.
        # 1: L=101. H=103 < 106. No Act.
        # 2: L=107. H=109 >= 106 (Act).
        #    Trail Active. MaxH=109. New SL = 109-2 = 107.
        #    L=107 <= 107. Hit SL (Profit).

        ret, idx, reason = simulate_trade_numba(opens, highs, lows, closes, entry, side, sl_dist, act_dist, tr_dist)

        # Exit at 107. Ret = 0.07.
        # Logic updates SL for *next* bar to be conservative (don't assume High before Low).
        # So exit happens at Bar 3 (Low=85 hits SL=107).
        self.assertAlmostEqual(ret, 0.07)
        self.assertEqual(idx, 3)

    def test_meta_model(self):
        # Simple data
        dates = pd.date_range("2021-01-01", periods=20, freq="1h")
        X_meta = pd.DataFrame(np.random.randn(20, 10), index=dates, columns=[
            "pred_tf_logit", "pred_mr_logit",
            "realized_vol", "vol_z", "log_volume",
            "norm_momentum", "dist_ma_z",
            "atr_slope", "dist_vwap_norm", "mom_accel"
        ])
        y = np.random.randn(20)

        m = MetaModel()
        m.fit(X_meta, y)
        preds = m.predict(X_meta)
        self.assertEqual(len(preds), 20)

if __name__ == '__main__':
    unittest.main()
