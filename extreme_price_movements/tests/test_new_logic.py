import unittest
import numpy as np
import pandas as pd
from extreme_price_movements.candidates import select_trade_candidates_vectorized
from extreme_price_movements.meta_model import MetaModel
from unittest.mock import patch

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
        # Make Vol High enough (>= 7%)
        # 12h range.
        c = pd.DataFrame(100.0, index=dates, columns=["A", "B"])
        h = c.copy()
        l = c.copy()

        # At t=50 and surrounding, make volatility high for A AND B
        # Range = 10. Close = 100. Diff = 10/100 = 10% > 7%.
        # B needs to pass volatility check too!
        h.loc[dates[40:60], "A"] = 110.0
        l.loc[dates[40:60], "A"] = 100.0
        c.loc[dates[40:60], "A"] = 100.0

        h.loc[dates[40:60], "B"] = 110.0
        l.loc[dates[40:60], "B"] = 100.0
        c.loc[dates[40:60], "B"] = 100.0

        panel = {"close": c, "high": h, "low": l}

        # Ensure Volatility Z-Score passes (> 1.6)
        feats["volatility_zscore"] = pd.DataFrame(2.0, index=dates, columns=["A", "B"])

        # Ensure new Exhaustion features pass for Short MR (Top Performer A)
        feats["wick_ratio_4h_max"] = pd.DataFrame(0.5, index=dates, columns=["A", "B"])
        feats["vol_price_div"] = pd.DataFrame(-0.5, index=dates, columns=["A", "B"])
        feats["rsi"] = pd.DataFrame(75.0, index=dates, columns=["A", "B"])
        feats["rsi_lag1"] = pd.DataFrame(80.0, index=dates, columns=["A", "B"])

        # Ensure new Falling Knife features pass for Long MR (Bottom Performer B)
        feats["vol_z_4h"] = pd.DataFrame(3.0, index=dates, columns=["A", "B"])
        feats["rsi"].loc[dates[50], "B"] = 25.0
        feats["rsi_1h_slope"] = pd.DataFrame(1.0, index=dates, columns=["A", "B"])
        feats["atr_pct_change"] = pd.DataFrame(-0.1, index=dates, columns=["A", "B"])

        # Mock sign consistency to always pass (returns 1.0)
        # We need to mock ff.numba_sign_consistency used inside candidates.py
        with patch('extreme_price_movements.fast_funcs.numba_sign_consistency') as mock_sc:
            # Mock return value as a DataFrame of 1.0s
            mock_sc.return_value = np.ones((100, 2), dtype=np.float32)

            mask = select_trade_candidates_vectorized(panel, feats, pct=0.1, metric="ret24h")

            # Check base candidate t=50
            # A should be True (Vol OK, Rank OK, ShortMR Filters OK)
            self.assertTrue(mask.loc[dates[50], "A"])

            # Check shifted candidates
            # At t=54: Vol window [43, 54]. High=110, Low=100. Vol=10%. OK.
            self.assertTrue(mask.loc[dates[54], "A"])

            # t=38 (shift -12). From t=50.
            # Vol check at t=38 is 0, BUT mask logic applies filters at Event Time (t=50)
            # and THEN expands. Since t=50 is a valid event, t=38 (offset -12) is a valid candidate.
            self.assertTrue(mask.loc[dates[38], "A"])

            # B: Rank OK (Bot).
            # Vol check at t=50: range 10% (from dates[40:60]). OK.
            self.assertTrue(mask.loc[dates[50], "B"])

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
