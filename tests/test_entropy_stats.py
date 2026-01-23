
import unittest
import numpy as np
import pandas as pd
from src.utils.entropy_optimized import calculate_entropy_statistics_numba, vectorized_entropy_features

class TestEntropyStats(unittest.TestCase):
    def test_calculate_entropy_statistics_correctness(self):
        np.random.seed(42)
        N = 1000
        window = 10
        values = np.random.random(N)

        # Original (naive) implementation logic for comparison
        expected_ma = np.full(N, np.nan)
        expected_std = np.full(N, np.nan)
        expected_zscore = np.full(N, np.nan)

        for i in range(window - 1, N):
            window_data = values[i - window + 1:i + 1]
            ma = np.mean(window_data)
            std = np.std(window_data)
            expected_ma[i] = ma
            expected_std[i] = std
            if std > 0:
                expected_zscore[i] = (values[i] - ma) / std
            else:
                expected_zscore[i] = 0.0

        # Run optimized function
        ma, std, zscore = calculate_entropy_statistics_numba(values, window)

        # Compare results
        # Note: Numba online algo vs numpy two-pass might have small floating point differences
        np.testing.assert_allclose(ma[window-1:], expected_ma[window-1:], rtol=1e-5, atol=1e-8, err_msg="Mean mismatch")
        np.testing.assert_allclose(std[window-1:], expected_std[window-1:], rtol=1e-5, atol=1e-8, err_msg="Std mismatch")

        # Zscore comparison
        # Handle cases where std is very small or nan
        valid_mask = np.isfinite(expected_zscore) & np.isfinite(zscore)
        np.testing.assert_allclose(zscore[valid_mask], expected_zscore[valid_mask], rtol=1e-5, atol=1e-8, err_msg="Zscore mismatch")

    def test_vectorized_entropy_features_volatility(self):
        # This tests the fix for the volatility loop and timestamp issue
        N = 200
        prices = np.random.random(N) + 100
        df = pd.DataFrame({'close': prices}, index=pd.date_range('2024-01-01', periods=N, freq='1s'))

        # We need entropy_contribution to trigger the other path if we want
        # But volatility is calculated regardless

        try:
            features = vectorized_entropy_features(df, use_numba=True)
        except Exception as e:
            self.fail(f"vectorized_entropy_features raised exception: {e}")

        self.assertIn('staleness_adjusted_drift', features.columns)
        self.assertFalse(features['staleness_adjusted_drift'].isnull().all())

if __name__ == '__main__':
    unittest.main()
