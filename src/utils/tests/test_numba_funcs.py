import numpy as np
import pytest
from src.utils.numba_funcs import _numba_streak_persistence, _numba_rolling_beta

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

def test_rolling_beta_basic():
    """Test basic rolling beta logic."""
    # Market: [1, 2, 3, 4, 5]
    # Asset: [2, 4, 6, 8, 10] -> Beta should be 2.0

    m = np.array([1, 2, 3, 4, 5], dtype=float)
    x = np.array([2, 4, 6, 8, 10], dtype=float)

    # Window 3.
    # i=2: [1,2,3] vs [2,4,6].
    # Mean M=2, X=4.
    # Cov(X,M) = E[XM] - E[X]E[M] = (2+8+18)/3 - 8 = 28/3 - 24/3 = 4/3.
    # Var(M) = E[M^2] - E[M]^2 = (1+4+9)/3 - 4 = 14/3 - 12/3 = 2/3.
    # Beta = (4/3) / (2/3) = 2.0.

    res = _numba_rolling_beta(x, m, window=3)

    assert len(res) == 5
    assert np.isclose(res[2], 2.0)
    assert np.isclose(res[3], 2.0)
    assert np.isclose(res[4], 2.0)

def test_rolling_beta_zero_variance():
    """Test rolling beta with zero market variance."""
    m = np.array([1, 1, 1, 1, 1], dtype=float)
    x = np.array([1, 2, 3, 4, 5], dtype=float)

    res = _numba_rolling_beta(x, m, window=3)

    # Variance of M is 0. Beta should be 0 (handled by check).
    assert np.all(res == 0.0)
