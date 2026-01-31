import numpy as np
import pytest
import pandas as pd
from src.utils.numba_funcs import _numba_rolling_mean_nan_safe, _numba_rolling_std_nan_safe

def test_rolling_mean_nan_safe_basic():
    """Test rolling mean with NaNs against Pandas."""
    data = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, np.nan, 8.0])
    window = 3

    # Expected behavior: min_periods=1 effectively (since it computes if count > 0)
    # Pandas default min_periods is equal to window size if not specified,
    # but the numba func seems to behave like min_periods=1 (expanding start).
    # Let's check the logic:
    # "if count > 0: output[i] = sum/count"
    # So yes, min_periods=1.

    # Pandas equivalent
    series = pd.Series(data)
    expected = series.rolling(window=window, min_periods=1).mean().values

    result = _numba_rolling_mean_nan_safe(data, window)

    np.testing.assert_allclose(result, expected, equal_nan=True)

def test_rolling_std_nan_safe_basic():
    """Test rolling std with NaNs against Pandas."""
    data = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 10.0, 2.0])
    window = 3

    # Pandas equivalent
    series = pd.Series(data)
    # ddof=1 is default for pandas std
    expected = series.rolling(window=window, min_periods=2).std().values

    # Numba implementation output check:
    # "if count > 1: ... else: 0.0 or NaN"
    # The existing implementation might output 0.0 or NaN for count <= 1.
    # My optimized one outputs 0.0 for count=1? No, 0.0.
    # Pandas outputs NaN for count=1.
    # Let's align my test expectation with what the function returns.
    # Current unoptimized implementation returns 0.0?
    # No, it initializes with NaN.
    # "if count <= 1: continue" -> so it stays NaN.
    # So it matches pandas min_periods=2.

    result = _numba_rolling_std_nan_safe(data, window)

    np.testing.assert_allclose(result, expected, equal_nan=True)

def test_rolling_mean_all_nans():
    data = np.full(10, np.nan)
    window = 3
    result = _numba_rolling_mean_nan_safe(data, window)
    assert np.all(np.isnan(result))

def test_rolling_std_all_nans():
    data = np.full(10, np.nan)
    window = 3
    result = _numba_rolling_std_nan_safe(data, window)
    assert np.all(np.isnan(result))

def test_rolling_window_larger_than_data():
    data = np.array([1.0, 2.0, 3.0])
    window = 5

    # Should work as expanding window
    series = pd.Series(data)
    expected_mean = series.rolling(window=window, min_periods=1).mean().values
    expected_std = series.rolling(window=window, min_periods=2).std().values

    res_mean = _numba_rolling_mean_nan_safe(data, window)
    res_std = _numba_rolling_std_nan_safe(data, window)

    np.testing.assert_allclose(res_mean, expected_mean, equal_nan=True)
    np.testing.assert_allclose(res_std, expected_std, equal_nan=True)

def test_rolling_window_1():
    data = np.array([1.0, 2.0, 3.0])
    window = 1

    res_mean = _numba_rolling_mean_nan_safe(data, window)
    res_std = _numba_rolling_std_nan_safe(data, window)

    np.testing.assert_allclose(res_mean, data, equal_nan=True)
    # Std of 1 element is NaN in pandas (ddof=1), or 0 if ddof=0.
    # Pandas defaults to NaN.
    # Our function: "if count <= 1: continue" -> stays NaN.
    assert np.all(np.isnan(res_std))
