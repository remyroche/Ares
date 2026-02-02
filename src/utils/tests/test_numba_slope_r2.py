
import unittest
import numpy as np
from src.utils.numba_funcs import _numba_rolling_slope, _numba_rolling_rsquared
from scipy.stats import linregress

class TestNumbaSlopeR2(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_rolling_slope_simple(self):
        y = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        window = 2
        # W1: [10, 20] -> slope 10
        # W2: [20, 30] -> slope 10
        # W3: [30, 40] -> slope 10
        res = _numba_rolling_slope(y, window)
        expected = np.array([0, 10, 10, 10], dtype=np.float32)
        np.testing.assert_allclose(res, expected, atol=1e-6)

    def test_rolling_slope_random(self):
        n = 100
        window = 10
        y = np.random.randn(n).astype(np.float32)
        res = _numba_rolling_slope(y, window)

        # Verify against scipy.stats.linregress
        for i in range(window, n):
            chunk = y[i-window+1 : i+1]
            slope, _, _, _, _ = linregress(np.arange(window), chunk)
            self.assertAlmostEqual(res[i], slope, places=5)

    def test_rolling_rsquared_simple(self):
        # Perfect correlation
        y = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        window = 3
        # W1: [10, 20, 30]. R2 = 1.0
        # W2: [20, 30, 40]. R2 = 1.0
        res = _numba_rolling_rsquared(y, window)
        np.testing.assert_allclose(res[window-1:], 1.0, atol=1e-6)

        # No correlation (horizontal line) -> R2 should be 0 or handled gracefully
        y_flat = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32)
        res_flat = _numba_rolling_rsquared(y_flat, window)
        # Variance of y is 0. R2 is technically undefined 0/0.
        # Implementation returns 0.0.
        np.testing.assert_allclose(res_flat[window-1:], 0.0, atol=1e-6)

    def test_rolling_rsquared_random(self):
        n = 100
        window = 10
        y = np.random.randn(n).astype(np.float32)
        res = _numba_rolling_rsquared(y, window)

        for i in range(window, n):
            chunk = y[i-window+1 : i+1]
            slope, intercept, r_value, p_value, std_err = linregress(np.arange(window), chunk)
            expected_r2 = r_value ** 2
            self.assertAlmostEqual(res[i], expected_r2, places=5)

if __name__ == '__main__':
    unittest.main()
