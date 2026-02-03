
import unittest
import numpy as np
import pandas as pd
from src.feature_generation.categories.entropy import ShannonEntropyGenerator
from src.utils.numba_funcs import _numba_rolling_shannon_entropy

class TestShannonEntropyOptimized(unittest.TestCase):
    def test_numba_func_matches_python_logic(self):
        # Generate random data
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.normal(0, 1, n))

        # Calculate returns as per Generator logic
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate([[np.nan], returns])

        window = 20
        q_bins = 5

        # Run Numba implementation
        returns_clean = np.nan_to_num(returns, nan=0.0)
        numba_result = _numba_rolling_shannon_entropy(returns_clean, window, q_bins)

        # Run Manual Python Logic (simplified from original generator)
        python_result = np.full(n, np.nan)

        for i in range(window, n):
            window_returns = returns_clean[i - window + 1 : i + 1]
            # Valid check not needed as we cleaned returns, but logic assumes finite
            c_min = np.min(window_returns)
            c_max = np.max(window_returns)

            if c_min == c_max:
                python_result[i] = 0.0
                continue

            bins = np.linspace(c_min, c_max, q_bins + 1)
            # digitize returns 1..bins. We want 0..bins-1
            digitized = np.digitize(window_returns, bins) - 1
            # clip to ensure range
            digitized = np.clip(digitized, 0, q_bins - 1)

            counts = np.bincount(digitized, minlength=q_bins)
            probabilities = counts / len(window_returns)

            entropy = 0
            for p in probabilities:
                if p > 0:
                    entropy -= p * np.log2(p)

            python_result[i] = entropy

        # Compare valid parts (window onwards)
        # Allow small float diff
        np.testing.assert_array_almost_equal(numba_result[window:], python_result[window:], decimal=5)

    def test_generator_integration(self):
        # Test that the generator class uses the optimized function and returns expected format
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.normal(0, 1, n))
        data = pd.DataFrame({'close': close})

        generator = ShannonEntropyGenerator(window=20, q_bins=10)
        result = generator._generate_feature(data)

        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), n)
        # First window-1 values should be NaN or 0?
        # Numba func returns NaN for first window-1?
        # Actually _numba_rolling_shannon_entropy logic:
        # loop range(window, n). so 0 to window-1 are initialized as NaN.
        # But wait, python range(window, n) starts at window.
        # So index window is the first calculated one.
        # Indices 0 to window-1 are untouched (so whatever initialization).
        # Initialization: output = np.full(n, np.nan, dtype=np.float32)
        # So first 'window' elements (indices 0..window-1) are NaN.

        self.assertTrue(np.isnan(result.iloc[0]))
        self.assertTrue(not np.isnan(result.iloc[25]))

if __name__ == '__main__':
    unittest.main()
