import numpy as np
import pytest
from src.utils.numba_funcs import (
    _numba_streak_persistence,
    _numba_generate_range_bars,
    _numba_ewma,
    _numba_price_jump_frequency,
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

def test_range_bars_float32():
    """Verify that _numba_generate_range_bars outputs float32 durations."""
    n = 100
    times = np.arange(n, dtype=np.int64) * 1_000_000_000  # Seconds as ns
    opens = np.random.randn(n).astype(np.float32)
    highs = opens + 1
    lows = opens - 1
    closes = opens + 0.1
    vols = np.ones(n, dtype=np.float32)
    thresholds = np.ones(n, dtype=np.float32) * 0.1

    result = _numba_generate_range_bars(times, opens, highs, lows, closes, vols, thresholds)

    # Unpack
    out_durations = result[6]

    assert out_durations.dtype == np.float32, f"Expected float32, got {out_durations.dtype}"

def test_ewma_float32():
    """Verify _numba_ewma execution with float32."""
    x = np.random.randn(100).astype(np.float32)
    alpha = 0.5 # Python float (double)

    # Run with adjust=True (uses sum_weights)
    out = _numba_ewma(x, alpha, adjust=True)

    assert out.dtype == np.float32, f"Expected float32 output, got {out.dtype}"
    assert np.all(np.isfinite(out) | np.isnan(out))


def test_price_jump_frequency():
    """Test price jump frequency calculation."""
    # Create a deterministic pattern
    # Window 5. Threshold 1.0.
    # [1, 2, 3, 4, 5] -> Mean 3, Std sqrt(2) ~ 1.414.
    # Z-scores:
    # 1: |1-3|/1.414 = 2/1.414 = 1.41 > 1.0 (Jump)
    # 2: |2-3|/1.414 = 1/1.414 = 0.707 < 1.0
    # 3: |3-3| = 0 < 1.0
    # 4: 0.707 < 1.0
    # 5: 1.41 > 1.0 (Jump)
    # Count = 2. Freq = 2/5 = 0.4.

    returns = np.array([1, 2, 3, 4, 5, 100], dtype=np.float32)
    # Window 5 at index 5 (0..5 exclusive? No, range(window, n) starts at index window (5).
    # Returns at index 5 uses returns[0:5].
    # So expected output at index 5 is 0.4.

    res = _numba_price_jump_frequency(returns, window=5, threshold=1.0)

    assert np.isclose(res[5], 0.4), f"Expected 0.4, got {res[5]}"

    # Check zero variance case
    # [1, 1, 1, 1, 1] -> Std 0 -> Output 0
    returns_const = np.ones(10, dtype=np.float32)
    res_const = _numba_price_jump_frequency(returns_const, window=5, threshold=1.0)
    assert np.all(res_const == 0.0)
