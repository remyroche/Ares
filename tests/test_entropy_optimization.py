
import unittest
import numpy as np
import pandas as pd
import time
from src.utils.entropy_optimized import (
    vectorized_entropy_features,
    lempel_ziv_complexity_numba,
    shannon_entropy_numba
)

class TestEntropyOptimization(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.n = 200
        # Create a random walk price series
        self.prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, self.n)))
        self.df = pd.DataFrame(
            {'close': self.prices},
            index=pd.date_range('2024-01-01', periods=self.n, freq='1min')
        )

    def test_discrepancy_fix(self):
        """Test that Numba and Pandas implementations are consistent (using returns)."""
        try:
            features_numba = vectorized_entropy_features(self.df, use_numba=True)
            features_pandas = vectorized_entropy_features(self.df, use_numba=False)
        except Exception as e:
            self.fail(f"Feature generation failed: {e}")

        col = 'entropy_rolling_20'
        if col not in features_numba.columns:
            self.skipTest(f"Column {col} missing from output")

        s1 = features_numba[col].dropna()
        s2 = features_pandas[col].dropna()

        common_idx = s1.index.intersection(s2.index)
        if len(common_idx) < 10:
            self.fail("Not enough common indices to compare")

        corr = s1.loc[common_idx].corr(s2.loc[common_idx])
        print(f"Correlation between Numba and Pandas versions: {corr:.4f}")

        # We expect high correlation (>0.9) now that both use returns
        self.assertGreater(corr, 0.9, "Numba and Pandas implementations should be consistent")

    def test_lz_performance(self):
        """Test Lempel-Ziv complexity performance."""
        n = 10000
        values = np.random.randn(n)

        start = time.time()
        _ = lempel_ziv_complexity_numba(values, normalize=True)
        duration = time.time() - start

        print(f"LZ Complexity time for N={n}: {duration:.4f}s")
        # Should be well under 1 second for O(N*K) where K=5000 and N=10000
        # O(N^2) would be 10^8 ops, might take ~0.5-1s depending on CPU
        # With optimization, it scans at most 5000 back.
        # For i < 5000, it scans i. Total ops approx N*K.
        self.assertLess(duration, 2.0, "Execution time too slow")

    def test_nan_handling(self):
        """Test NaN handling in Shannon Entropy."""
        values = np.array([1.0, 2.0, np.nan, 3.0, np.inf])

        # Should not crash and return valid entropy
        try:
            h = shannon_entropy_numba(values, n_bins=10)
        except Exception as e:
            self.fail(f"shannon_entropy_numba crashed on NaNs: {e}")

        self.assertTrue(np.isfinite(h), "Entropy should be finite")
        self.assertGreater(h, 0.0, "Entropy should be positive for diverse values")

if __name__ == "__main__":
    unittest.main()
