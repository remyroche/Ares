import unittest
import numpy as np
import pandas as pd
from extreme_price_movements import fast_funcs as ff

def reference_rolling_rank(arr, window, pct=True):
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(n):
        val_curr = arr[i]
        if np.isnan(val_curr): continue

        count_valid = 0
        count_less = 0
        count_equal = 0

        for k in range(max(0, i-window+1), i+1):
            v = arr[k]
            if not np.isnan(v):
                count_valid += 1
                if v < val_curr:
                    count_less += 1
                elif v == val_curr:
                    count_equal += 1

        if count_valid > 0:
            rank = count_less + (count_equal + 1.0) / 2.0
            if pct:
                out[i] = rank / count_valid
            else:
                out[i] = rank
    return out

class TestNumbaRank(unittest.TestCase):
    def test_rolling_rank_basic(self):
        arr = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
        window = 3
        # Reference doesn't support min_periods, assumes min_periods=1
        expected = reference_rolling_rank(arr, window, pct=True)

        df = pd.DataFrame({'a': arr})
        # Override default to match reference which is effectively min_periods=1
        result = ff.numba_rolling_rank(df, window, pct=True, min_periods=1)
        res_arr = result['a'].values

        np.testing.assert_allclose(res_arr, expected, rtol=1e-5)

    def test_rolling_rank_nans(self):
        arr = np.array([1.0, np.nan, 2.0, 1.0], dtype=np.float32)
        window = 3
        expected = reference_rolling_rank(arr, window, pct=True)

        df = pd.DataFrame({'a': arr})
        result = ff.numba_rolling_rank(df, window, pct=True, min_periods=1)
        res_arr = result['a'].values

        np.testing.assert_allclose(res_arr, expected, rtol=1e-5)

    def test_rolling_rank_ties(self):
        arr = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        window = 3
        expected = reference_rolling_rank(arr, window, pct=True)

        df = pd.DataFrame({'a': arr})
        result = ff.numba_rolling_rank(df, window, pct=True, min_periods=1)
        res_arr = result['a'].values

        np.testing.assert_allclose(res_arr, expected, rtol=1e-5)

    def test_rolling_rank_min_periods(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        window = 3
        df = pd.DataFrame({'a': arr})

        # Default behavior (min_periods=window=3)
        result = ff.numba_rolling_rank(df, window, pct=True)
        res_arr = result['a'].values
        expected = np.array([np.nan, np.nan, 1.0], dtype=np.float32)
        np.testing.assert_allclose(res_arr, expected, rtol=1e-5)

        # Explicit min_periods=1
        result_1 = ff.numba_rolling_rank(df, window, pct=True, min_periods=1)
        res_arr_1 = result_1['a'].values
        expected_1 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        np.testing.assert_allclose(res_arr_1, expected_1, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()
