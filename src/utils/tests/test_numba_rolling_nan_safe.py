import numpy as np
import pandas as pd
import pytest
from src.utils.numba_funcs import _numba_rolling_mean_nan_safe, _numba_rolling_std_nan_safe

def test_rolling_mean_nan_safe_basic():
    x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    res = _numba_rolling_mean_nan_safe(x, window=3)

    # Expected:
    # i=0: [1] -> 1
    # i=1: [1,2] -> 1.5
    # i=2: [1,2,3] -> 2
    # i=3: [2,3,4] -> 3
    # i=4: [3,4,5] -> 4

    expected = np.array([1, 1.5, 2, 3, 4], dtype=np.float32)
    np.testing.assert_allclose(res, expected, rtol=1e-5)

def test_rolling_mean_nan_safe_with_nans():
    x = np.array([1, np.nan, 3, 4, np.nan], dtype=np.float32)
    res = _numba_rolling_mean_nan_safe(x, window=3)

    # i=0: [1] -> 1
    # i=1: [1, nan] -> 1
    # i=2: [1, nan, 3] -> 2
    # i=3: [nan, 3, 4] -> 3.5
    # i=4: [3, 4, nan] -> 3.5

    expected = np.array([1, 1, 2, 3.5, 3.5], dtype=np.float32)
    np.testing.assert_allclose(res, expected, rtol=1e-5)

def test_rolling_std_nan_safe_basic():
    x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    res = _numba_rolling_std_nan_safe(x, window=3)

    # i=0: [1] -> NaN (count < 2)
    # i=1: [1,2] -> std([1,2]) = 0.707106
    # i=2: [1,2,3] -> std([1,2,3]) = 1.0
    # i=3: [2,3,4] -> 1.0
    # i=4: [3,4,5] -> 1.0

    # Pandas min_periods=1 means std is NaN for count=1

    assert np.isnan(res[0])
    assert np.isclose(res[1], 0.707106, atol=1e-5)
    assert np.isclose(res[2], 1.0, atol=1e-5)
    assert np.isclose(res[3], 1.0, atol=1e-5)
    assert np.isclose(res[4], 1.0, atol=1e-5)

def test_rolling_std_nan_safe_with_nans():
    x = np.array([1, np.nan, 3, 4, 10], dtype=np.float32)
    res = _numba_rolling_std_nan_safe(x, window=3)

    s = pd.Series(x)
    expected = s.rolling(3, min_periods=1).std().to_numpy()

    # Mask NaNs for comparison
    mask = ~np.isnan(expected)
    np.testing.assert_allclose(res[mask], expected[mask], rtol=1e-4)

    # Verify NaN positions match
    assert np.all(np.isnan(res) == np.isnan(expected))

def test_all_nans():
    x = np.full(10, np.nan, dtype=np.float32)
    res_mean = _numba_rolling_mean_nan_safe(x, 5)
    res_std = _numba_rolling_std_nan_safe(x, 5)

    assert np.all(np.isnan(res_mean))
    assert np.all(np.isnan(res_std))

def test_single_value_window():
    x = np.array([1, 2, 3], dtype=np.float32)
    res_mean = _numba_rolling_mean_nan_safe(x, 1)
    res_std = _numba_rolling_std_nan_safe(x, 1)

    np.testing.assert_allclose(res_mean, x)
    # Std for window 1 should be NaN because count is 1.
    assert np.all(np.isnan(res_std))

def test_window_larger_than_array():
    x = np.array([1, 2, 3], dtype=np.float32)
    res_mean = _numba_rolling_mean_nan_safe(x, 10)

    # Behaves like accumulating mean
    expected = np.array([1, 1.5, 2], dtype=np.float32)
    np.testing.assert_allclose(res_mean, expected)
