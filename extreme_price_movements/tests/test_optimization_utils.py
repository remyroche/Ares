import unittest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
from extreme_price_movements.optimization_utils import filter_low_variance_assets

class TestOptimizationUtils(unittest.TestCase):
    def test_filter_low_variance_assets(self):
        # Mock Store
        store = MagicMock()

        dates = pd.date_range("2023-01-01", "2023-02-01", freq="1h", tz="UTC")

        # A: Sine wave (High Var)
        price_a = 100 + 10 * np.sin(np.linspace(0, 100, len(dates)))
        # B: Small noise (Low Var)
        np.random.seed(42)
        price_b = 100 + 0.1 * np.random.randn(len(dates))
        # C: Constant (Zero Var)
        price_c = 100 * np.ones(len(dates))

        df_a = pd.DataFrame({"close": price_a}, index=dates)
        df_b = pd.DataFrame({"close": price_b}, index=dates)
        df_c = pd.DataFrame({"close": price_c}, index=dates)

        def mock_load(sym, columns=None, start_ts=None, end_ts=None):
            df_map = {"A": df_a, "B": df_b, "C": df_c}
            base_df = df_map.get(sym, pd.DataFrame())

            if base_df.empty: return base_df

            # Simulate slicing if start_ts/end_ts passed
            # This mimics PartitionedOHLCVStore behavior roughly
            sliced = base_df
            if start_ts:
                sliced = sliced[sliced.index >= start_ts]
            if end_ts:
                sliced = sliced[sliced.index <= end_ts]

            return sliced

        store.load.side_effect = mock_load

        syms = ["A", "B", "C"]
        ts_sig = pd.Timestamp("2023-01-31 12:00:00", tz="UTC")

        # Expect A to be top variance.
        # Threshold 40% of 3 is 1 (int(3*0.4) = 1).
        # With lookback 10 days

        # Note: current implementation does not accept ts_sig, so this will fail until Step 2.
        result = filter_low_variance_assets(store, syms, lookback_days=10, threshold_pct=0.40, ts_sig=ts_sig)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "A")

        # Verify optimization: store.load called with start_ts
        expected_start = ts_sig - pd.Timedelta(days=10)

        # Get calls for "A"
        # Since logic iterates syms, we find call for "A"
        # store.load(s, columns=["close"], start_ts=cutoff, end_ts=ts_sig)

        calls = store.load.mock_calls
        # filter calls for "A"
        call_a = [c for c in calls if c.args and c.args[0] == "A"]
        if not call_a:
             # Try check if args[0] was passed as keyword?
             # But implementation calls store.load(s, ...)
             pass

        self.assertTrue(len(call_a) > 0, "store.load should be called for A")

        # check kwargs or args
        # Signature: load(symbol, columns=..., start_ts=..., end_ts=...)
        # We expect start_ts and end_ts to be passed as kwargs or pos args depending on implementation.
        # My plan is to use kwargs for start_ts/end_ts.

        kwargs = call_a[0].kwargs
        self.assertIn("start_ts", kwargs)
        self.assertIn("end_ts", kwargs)
        self.assertEqual(kwargs["start_ts"], expected_start)
        self.assertEqual(kwargs["end_ts"], ts_sig)

if __name__ == '__main__':
    unittest.main()
