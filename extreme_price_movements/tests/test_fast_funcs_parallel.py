import numpy as np
import pandas as pd
import pytest
import extreme_price_movements.fast_funcs as ff

def test_numba_rolling_mean_correctness():
    np.random.seed(42)
    data = np.random.randn(100, 5).astype(np.float32)
    df = pd.DataFrame(data)

    # Numba implementation behaves like min_periods=1
    expected = df.rolling(10, min_periods=1).mean().astype(np.float32)
    actual = ff.numba_rolling_mean(df, 10)

    pd.testing.assert_frame_equal(expected, actual, atol=1e-5)

def test_numba_rolling_std_correctness():
    np.random.seed(42)
    data = np.random.randn(100, 5).astype(np.float32)
    df = pd.DataFrame(data)

    # Numba implementation behaves like min_periods=2
    expected = df.rolling(10, min_periods=2).std().astype(np.float32)

    actual = ff.numba_rolling_std(df, 10)

    pd.testing.assert_frame_equal(expected, actual, atol=1e-5)

def test_numba_rolling_std_parallel_correctness():
    np.random.seed(42)
    data = np.random.randn(100, 5).astype(np.float32)
    df = pd.DataFrame(data)

    # Numba implementation behaves like min_periods=2
    expected = df.rolling(10, min_periods=2).std().astype(np.float32)

    actual = ff.numba_rolling_std_parallel(df, 10)

    pd.testing.assert_frame_equal(expected, actual, atol=1e-5)

def test_numba_ewma_correctness():
    np.random.seed(42)
    data = np.random.randn(100, 5).astype(np.float32)
    df = pd.DataFrame(data)
    alpha = 0.5

    # Expected: Pandas ewm
    expected = df.ewm(alpha=alpha, adjust=False).mean().astype(np.float32)
    actual = ff.numba_ewma(df, alpha, adjust=False)

    pd.testing.assert_frame_equal(expected, actual, atol=1e-5)

def test_numba_rolling_sum_correctness():
    np.random.seed(42)
    data = np.random.randn(100, 5).astype(np.float32)
    df = pd.DataFrame(data)

    # min_periods=1
    expected = df.rolling(10, min_periods=1).sum().astype(np.float32)
    actual = ff.numba_rolling_sum(df, 10)

    pd.testing.assert_frame_equal(expected, actual, atol=1e-4)

def test_numba_rolling_max_correctness():
    np.random.seed(42)
    data = np.random.randn(100, 5).astype(np.float32)
    df = pd.DataFrame(data)

    # min_periods=1
    expected = df.rolling(10, min_periods=1).max().astype(np.float32)
    actual = ff.numba_rolling_max(df, 10)

    pd.testing.assert_frame_equal(expected, actual, atol=1e-5)

def test_numba_rolling_min_correctness():
    np.random.seed(42)
    data = np.random.randn(100, 5).astype(np.float32)
    df = pd.DataFrame(data)

    # min_periods=1
    expected = df.rolling(10, min_periods=1).min().astype(np.float32)
    actual = ff.numba_rolling_min(df, 10)

    pd.testing.assert_frame_equal(expected, actual, atol=1e-5)

def test_numba_rolling_median_correctness():
    np.random.seed(42)
    data = np.random.randn(100, 5).astype(np.float32)
    df = pd.DataFrame(data)
    window = 10

    # _numba_rolling_median outputs 0.0 for the first window-1 elements
    # Pandas outputs NaN.
    # We'll replace Pandas NaNs with 0.0 to match our legacy implementation behavior
    expected = df.rolling(window).median().fillna(0.0).astype(np.float32)

    actual = ff.numba_rolling_median(df, window)

    pd.testing.assert_frame_equal(expected, actual, atol=1e-5)

def test_numba_pct_change_correctness():
    np.random.seed(42)
    data = np.random.randn(100, 5).astype(np.float32)
    df = pd.DataFrame(data)

    expected = df.pct_change(1).astype(np.float32)
    actual = ff.numba_pct_change(df, 1)

    # Pandas fills first row with NaN.
    # _numba_pct_change also fills with NaN.

    pd.testing.assert_frame_equal(expected, actual, atol=1e-5)
