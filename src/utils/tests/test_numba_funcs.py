import numpy as np
import pytest
from src.utils.numba_funcs import (
    _numba_streak_persistence,
    _numba_rolling_mean_nan_safe,
    _numba_rolling_std_nan_safe
)

def test_streak_persistence_basic():
    """Test basic streak persistence logic."""
    close = np.array([100, 101, 102, 101, 100, 99], dtype=float)
    # diffs: [nan, 1, 1, -1, -1, -1]
    # signs: [0, 1, 1, -1, -1, -1]

    # Window 3.
    # i=3 (indices 1,2,3 -> signs 1, 1, -1). Streak: 2 (+), 1 (-).
    #   mean=(2+1)/2=1.5. std=0.5. Z=1.5/0.5=3.0? No, mean/std of lengths.
    # i=4 (indices 2,3,4 -> signs 1, -1, -1). Streak: 1 (+), 2 (-).
    #   mean=1.5. std=0.5.
    # i=5 (indices 3,4,5 -> signs -1, -1, -1). Streak: 3 (-).
    #   count=1. No stats (need count >= 2). Output 0.

    res = _numba_streak_persistence(close, window=3)

    # Verify shape
    assert len(res) == len(close)

    # Verify first window (i=3)
    # Streaks: [2, 1] -> mean=1.5, var=0.25, std=0.5. res=3.0
    assert np.isclose(res[3], 3.0)

    # Verify second window (i=4)
    # Streaks: [1, 2] -> mean=1.5, var=0.25, std=0.5. res=3.0
    assert np.isclose(res[4], 3.0)

    # Verify third window (i=5)
    # Streaks: [3] -> count=1. res=0.0
    assert res[5] == 0.0

def test_streak_persistence_zeros():
    """Test streak persistence with zeros in diffs."""
    close = np.array([100, 101, 101, 102, 102, 103], dtype=float)
    # diffs: [nan, 1, 0, 1, 0, 1]
    # signs: [0, 1, 0, 1, 0, 1]

    # Window 5. Indices 1..5. Signs: 1, 0, 1, 0, 1.
    # Ignoring zeros: 1, 1, 1. Streak: 3 (+). count=1. Result 0.

    res = _numba_streak_persistence(close, window=5)
    assert res[5] == 0.0

def test_streak_persistence_long():
    """Test with a longer sequence and verify output consistency."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(100))

    # Using window=10
    res = _numba_streak_persistence(close, window=10)

    assert len(res) == 100
    assert np.all(res[:10] == 0) # First window-1 elements are 0, plus window element (index 9) computed.
    # Wait, code computes for i in range(window, n).
    # So index 10 is computed.
    # indices 0..9 are 0.
    assert np.all(res[:10] == 0.0)
    # Index 10 is the first computed value?
    # No, range(window, n) starts at window.
    # If window=10, starts at 10.
    # So 0..9 are 0.

    # If output[window] is set (in the initialization block), then index 10 is set?
    # Initialization loop fills state.
    # Then `if window < n: output[window] = ...`
    # So index `window` (10) is set.

    # Check values are finite
    assert np.all(np.isfinite(res))

def test_streak_persistence_alternating():
    """Test alternating signs which create many short streaks."""
    close = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    # diffs: [nan, 1, -1, 1, -1, 1, -1, 1]
    # signs: [0, 1, -1, 1, -1, 1, -1, 1]

    # Window 4.
    # i=4. Signs 1, -1, 1, -1. Streaks: 1, 1, 1, 1.
    # Mean 1. Var 0. Res 0.

    res = _numba_streak_persistence(close, window=4)
    assert res[4] == 0.0

def test_rolling_mean_nan_safe_basic():
    """Test rolling mean with NaNs and basic values."""
    data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    window = 3

    # Expected behavior:
    # i=0: [1] -> 1
    # i=1: [1, 2] -> 1.5
    # i=2: [1, 2, nan] -> 1.5
    # i=3: [2, nan, 4] -> 3
    # i=4: [nan, 4, 5] -> 4.5

    res = _numba_rolling_mean_nan_safe(data, window)

    expected = np.array([1.0, 1.5, 1.5, 3.0, 4.5])
    np.testing.assert_allclose(res, expected)

def test_rolling_mean_nan_safe_all_nans():
    """Test rolling mean with all NaNs."""
    data = np.full(10, np.nan)
    window = 3

    res = _numba_rolling_mean_nan_safe(data, window)

    assert np.all(np.isnan(res))

def test_rolling_std_nan_safe_basic():
    """Test rolling std with NaNs."""
    data = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 7.0])
    window = 3

    # i=0: [1] -> count=1 -> NaN (std requires count > 1)
    # i=1: [1, 2] -> std=0.7071
    # i=2: [1, 2, 3] -> std=1.0
    # i=3: [2, 3, nan] -> std=0.7071
    # i=4: [3, nan, 5] -> std=1.4142
    # i=5: [nan, 5, 7] -> std=1.4142

    res = _numba_rolling_std_nan_safe(data, window)

    assert np.isnan(res[0])
    assert np.isclose(res[1], np.std([1, 2], ddof=1))
    assert np.isclose(res[2], np.std([1, 2, 3], ddof=1))
    assert np.isclose(res[3], np.std([2, 3], ddof=1))
    assert np.isclose(res[4], np.std([3, 5], ddof=1))
    assert np.isclose(res[5], np.std([5, 7], ddof=1))

def test_rolling_std_nan_safe_insufficient_data():
    """Test rolling std returns NaN when insufficient valid data."""
    data = np.array([1.0, np.nan, np.nan, 4.0])
    window = 3

    # i=0: [1] -> count=1 -> NaN
    # i=1: [1, nan] -> count=1 -> NaN
    # i=2: [1, nan, nan] -> count=1 -> NaN
    # i=3: [nan, nan, 4] -> count=1 -> NaN

    res = _numba_rolling_std_nan_safe(data, window)

    assert np.all(np.isnan(res))

def test_rolling_funcs_grow_window():
    """Test correctness of growing window at the start."""
    data = np.ones(5)
    window = 5

    res_mean = _numba_rolling_mean_nan_safe(data, window)
    res_std = _numba_rolling_std_nan_safe(data, window)

    assert np.all(res_mean == 1.0)
    assert np.isnan(res_std[0])
    assert np.all(res_std[1:] == 0.0)

def test_rolling_funcs_window_larger_than_data():
    """Test window larger than data length."""
    data = np.array([1.0, 2.0, 3.0])
    window = 10

    res_mean = _numba_rolling_mean_nan_safe(data, window)
    # Should behave like expanding
    assert np.isclose(res_mean[0], 1.0)
    assert np.isclose(res_mean[1], 1.5)
    assert np.isclose(res_mean[2], 2.0)
