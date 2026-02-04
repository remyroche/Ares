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

        # Verify files exist
        sym_dir = self.store._get_symbol_dir(self.symbol)
        # Should have Jan, Feb, Mar folders
        self.assertTrue(os.path.exists(os.path.join(sym_dir, "year=2023", "month=01")))
        self.assertTrue(os.path.exists(os.path.join(sym_dir, "year=2023", "month=02")))
        self.assertTrue(os.path.exists(os.path.join(sym_dir, "year=2023", "month=03")))

        # Test load with start_ts in Feb
        start_ts = pd.Timestamp("2023-02-15", tz="UTC")
        loaded = self.store.load(self.symbol, start_ts=start_ts)

        # Should only contain data >= Feb 15
        self.assertEqual(loaded.index.min(), start_ts)
        self.assertEqual(loaded.index.max(), df.index.max())

        # Test load with end_ts in Feb
        end_ts = pd.Timestamp("2023-02-20", tz="UTC")
        loaded_subset = self.store.load(self.symbol, start_ts=start_ts, end_ts=end_ts)
        self.assertEqual(loaded_subset.index.max(), end_ts)

    def test_compaction_atomic(self):
        # Create small chunks for one month to trigger compaction (or call explicitly)
        # save_partitioned triggers compaction if > 10 files.
        # Let's force it manually.

        # Create 2 files in Jan 2023 manually
        dates1 = pd.date_range("2023-01-01", "2023-01-05", freq="1h", tz="UTC")
        df1 = pd.DataFrame(np.random.randn(len(dates1), 5), columns=["open","high","low","close","volume"], index=dates1)
        self.store.save_partitioned(self.symbol, df1)

        dates2 = pd.date_range("2023-01-10", "2023-01-15", freq="1h", tz="UTC")
        df2 = pd.DataFrame(np.random.randn(len(dates2), 5), columns=["open","high","low","close","volume"], index=dates2)
        self.store.save_partitioned(self.symbol, df2)

        # Check files count
        part_dir = os.path.join(self.store._get_symbol_dir(self.symbol), "year=2023", "month=01")
        files = [f for f in os.listdir(part_dir) if f.endswith(".parquet")]
        self.assertEqual(len(files), 2)

        # Run compaction
        self.store.compact_partition(self.symbol, 2023, 1)

        # Should be 1 file now: compact-...
        files_after = [f for f in os.listdir(part_dir) if f.endswith(".parquet")]
        self.assertEqual(len(files_after), 1)
        self.assertTrue(files_after[0].startswith("compact-"))

        # Verify data integrity
        loaded = self.store.load(self.symbol)
        self.assertEqual(len(loaded), len(df1) + len(df2))

    def test_locking(self):
        # Test FileLock
        lock_path = os.path.join(self.test_dir, "test.lock")

        # Acquire lock
        with FileLock(lock_path) as lock:
            self.assertTrue(lock.handle is not None)
            # Try to acquire again in non-blocking way?
            # Standard flock is advisory.
            # Just verifying it doesn't crash is basic sanity check.
            pass

        # Verify update_symbol uses lock
        # We can mock FileLock and see if it's called
        with patch("extreme_price_movements.data_store.FileLock") as MockLock:
             # Mock instance context manager
            instance = MockLock.return_value
            instance.__enter__.return_value = instance

            # Mock fetch to avoid network
            with patch("extreme_price_movements.data_store.fetch_ohlcv_all_7d_chunks") as mock_fetch:
                mock_fetch.return_value = pd.DataFrame()

                self.store.update_symbol(MagicMock(), self.symbol, 0)

                # Verify FileLock was initialized with correct path
                sym_dir = self.store._get_symbol_dir(self.symbol)
                expected_lock = os.path.join(sym_dir, ".lock")
                MockLock.assert_called_with(expected_lock)
                instance.__enter__.assert_called()
                instance.__exit__.assert_called()

if __name__ == "__main__":
    unittest.main()
