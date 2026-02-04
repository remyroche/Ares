import os
import shutil
import pandas as pd
import numpy as np
import unittest
from unittest.mock import patch, MagicMock
from extreme_price_movements.data_store import PartitionedOHLCVStore, FileLock

class TestDataStoreImprovements(unittest.TestCase):
    def setUp(self):
        self.test_dir = "temp_test_data"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        self.store = PartitionedOHLCVStore(root_dir=self.test_dir)
        self.symbol = "BTC/USDT"

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_load_filtering(self):
        # Create dummy data for 3 months
        dates = pd.date_range("2023-01-01", "2023-03-31", freq="1h", tz="UTC")
        df = pd.DataFrame({
            "open": np.random.randn(len(dates)),
            "high": np.random.randn(len(dates)),
            "low": np.random.randn(len(dates)),
            "close": np.random.randn(len(dates)),
            "volume": np.random.randn(len(dates))
        }, index=dates)

        # Save partitioned
        self.store.save_partitioned(self.symbol, df)

        # Test load with start_ts in Feb
        start_ts = pd.Timestamp("2023-02-15", tz="UTC")
        loaded = self.store.load(self.symbol, start_ts=start_ts)

        self.assertEqual(loaded.index.min(), start_ts)
        self.assertEqual(loaded.index.max(), df.index.max())

    def test_gap_filling(self):
        # Create data with a gap
        dates1 = pd.date_range("2023-01-01 00:00", "2023-01-01 05:00", freq="1h", tz="UTC")
        dates2 = pd.date_range("2023-01-01 10:00", "2023-01-01 12:00", freq="1h", tz="UTC")

        df1 = pd.DataFrame(np.ones((len(dates1), 5)), columns=["open","high","low","close","volume"], index=dates1)
        df2 = pd.DataFrame(np.ones((len(dates2), 5)), columns=["open","high","low","close","volume"], index=dates2)

        self.store.save_partitioned(self.symbol, df1)
        self.store.save_partitioned(self.symbol, df2)

        # Load without gap fill
        loaded = self.store.load(self.symbol)
        self.assertEqual(len(loaded), len(dates1) + len(dates2))

        # Load with gap fill
        loaded_filled = self.store.load(self.symbol, fill_gaps=True)
        # Expected rows: 00:00 to 12:00 inclusive = 13 rows
        self.assertEqual(len(loaded_filled), 13)

        # Check filled values at 06:00
        filled_row = loaded_filled.loc[pd.Timestamp("2023-01-01 06:00", tz="UTC")]
        # Prices should be ffilled from 05:00 (which is 1.0)
        self.assertEqual(filled_row["close"], 1.0)
        self.assertEqual(filled_row["open"], 1.0) # Should be filled with close
        self.assertEqual(filled_row["volume"], 0.0)

    def test_overlap_deduplication(self):
        # Create initial data
        dates = pd.date_range("2023-01-01 00:00", "2023-01-01 05:00", freq="1h", tz="UTC")
        df = pd.DataFrame(np.random.randn(len(dates), 5), columns=["open","high","low","close","volume"], index=dates)
        self.store.save_partitioned(self.symbol, df)

        # Simulate overlap: Save overlapping range 05:00 - 07:00
        dates_overlap = pd.date_range("2023-01-01 05:00", "2023-01-01 07:00", freq="1h", tz="UTC")
        df_overlap = pd.DataFrame(np.random.randn(len(dates_overlap), 5), columns=["open","high","low","close","volume"], index=dates_overlap)

        self.store.save_partitioned(self.symbol, df_overlap)

        loaded = self.store.load(self.symbol)
        # 00:00 to 07:00 = 8 rows
        self.assertEqual(len(loaded), 8)
        # Verify 05:00 is from the second write (kept 'last')
        # Actually save_partitioned creates separate files. load() dedups.
        # But wait, we modified save_partitioned to clean input, not clean global state (that happens at compaction or load).
        # And compaction merges.
        # So deduplication happens at READ time (load) or COMPACTION time.

        # Let's verify load() returns clean index
        self.assertTrue(loaded.index.is_unique)

    def test_update_symbol_quality_check(self):
        # We want to verify check_data_health is called and warns
        # We can't easily capture tprint output without mocking, but we can verify logic via update_symbol flow

        with patch("extreme_price_movements.data_store.fetch_ohlcv_all_7d_chunks") as mock_fetch:
            # Return data with gap
            dates_gap = pd.to_datetime(["2023-01-01 00:00", "2023-01-01 02:00"]).tz_localize("UTC")
            df_gap = pd.DataFrame(np.random.randn(2, 5), columns=["open","high","low","close","volume"], index=dates_gap)
            mock_fetch.return_value = df_gap

            with patch("extreme_price_movements.data_store.tprint") as mock_tprint:
                 self.store.update_symbol(MagicMock(), self.symbol, 0)

                 # Check if warning was printed
                 # We look for "WARNING: Data gaps detected"
                 found_warning = False
                 for call in mock_tprint.call_args_list:
                     if "WARNING: Data gaps detected" in str(call):
                         found_warning = True
                         break
                 self.assertTrue(found_warning)

if __name__ == "__main__":
    unittest.main()
