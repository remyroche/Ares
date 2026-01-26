
import unittest
import numpy as np
from src.training.steps.labeling.feature_engineering_utils import _numba_efficiency_ratio

class TestEfficiencyRatio(unittest.TestCase):
    def test_basic_correctness(self):
        # Window size 3
        # log_returns = [1, -1, 1, 1, -2]
        #
        # i=2 (window 0,1,2): [1, -1, 1]
        #   net = 1 - 1 + 1 = 1
        #   vol = |1| + |-1| + |1| = 3
        #   er = 1/3
        #
        # i=3 (window 1,2,3): [-1, 1, 1]
        #   net = -1 + 1 + 1 = 1
        #   vol = |-1| + |1| + |1| = 3
        #   er = 1/3
        #
        # i=4 (window 2,3,4): [1, 1, -2]
        #   net = 1 + 1 - 2 = 0
        #   vol = |1| + |1| + |-2| = 4
        #   er = 0/4 = 0

        log_returns = np.array([1.0, -1.0, 1.0, 1.0, -2.0], dtype=np.float64)
        window = 3
        result = _numba_efficiency_ratio(log_returns, window)

        expected = np.array([np.nan, np.nan, 1.0/3.0, 1.0/3.0, 0.0])

        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_nans_are_preserved(self):
        # The optimized implementation works with primitives, assuming valid inputs or handling NaNs as numbers (which breaks logic usually)
        # But if inputs have NaNs, what happens?
        # Numba with float64 handles NaN arithmetic (NaN + 1 = NaN).
        # So sums will become NaN.
        # We verify behavior.
        log_returns = np.array([1.0, np.nan, 1.0, 1.0, 1.0], dtype=np.float64)
        window = 3
        result = _numba_efficiency_ratio(log_returns, window)

        # Window 0 (0,1,2) -> contains NaN -> net=NaN, vol=NaN -> ER=NaN
        # Window 1 (1,2,3) -> contains NaN -> ER=NaN
        # Window 2 (2,3,4) -> [1, 1, 1] -> net=3, vol=3 -> ER=1

        # Note: The optimized implementation tracks NaNs. If any NaN in window, output is 0.0 (preserving original behavior).
        # Window 0 (0,1,2) -> [1, NaN, 1] -> Has NaN -> 0.0
        # Window 1 (1,2,3) -> [NaN, 1, 1] -> Has NaN -> 0.0
        # Window 2 (2,3,4) -> [1, 1, 1] -> No NaN -> sum=3, vol=3 -> ER=1.0

        self.assertEqual(result[2], 0.0)
        self.assertEqual(result[3], 0.0)
        self.assertAlmostEqual(result[4], 1.0)

    def test_window_larger_than_data(self):
        log_returns = np.array([1.0, 2.0], dtype=np.float64)
        window = 5
        result = _numba_efficiency_ratio(log_returns, window)
        self.assertTrue(np.all(np.isnan(result)))
        self.assertEqual(len(result), 2)

    def test_zeros(self):
        log_returns = np.zeros(10, dtype=np.float64)
        window = 5
        result = _numba_efficiency_ratio(log_returns, window)

        # Volatility is 0. Check safe division logic (should be 0.0)
        # First window-1 are NaN
        self.assertTrue(np.all(np.isnan(result[:window-1])))
        self.assertTrue(np.all(result[window-1:] == 0.0))

    def test_numerical_stability(self):
        # Create a case where floating point errors might accumulate
        # Though hard to deterministic trigger, we ensure basic consistency
        np.random.seed(42)
        log_returns = np.random.randn(1000).astype(np.float64)
        window = 50
        result = _numba_efficiency_ratio(log_returns, window)

        # Check no Infs or negative values (ER is ratio of abs / sum(abs) so must be >= 0 and <= 1)
        # Since |sum(x)| <= sum(|x|), ER must be <= 1.

        valid_res = result[window-1:]
        self.assertTrue(np.all(valid_res >= 0.0))
        self.assertTrue(np.all(valid_res <= 1.0 + 1e-9)) # Allow tiny epsilon

if __name__ == '__main__':
    unittest.main()
