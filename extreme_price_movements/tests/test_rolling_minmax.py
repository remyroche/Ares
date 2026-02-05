import numpy as np
import pandas as pd
import pytest
import extreme_price_movements.fast_funcs as ff

def test_rolling_max_min_correctness():
    N = 1000
    W = 24
    np.random.seed(42)
    data = np.random.randn(N, 5).astype(np.float32)
    # Inject NaNs
    data[np.random.rand(N, 5) < 0.1] = np.nan

    df = pd.DataFrame(data, columns=[f"c_{i}" for i in range(5)])

    # Run optimized functions
    res_max = ff.numba_rolling_max(df, W)
    res_min = ff.numba_rolling_min(df, W)

    # Run Pandas reference
    # min_periods=1 because the Numba implementation (apply_to_frame logic)
    # and the rolling logic usually behave like min_periods=1 or similar?
    # Wait, let's check the implementation of _numba_rolling_max
    # lower_bound = i - window + 1. If i < window - 1, lower_bound <= 0.
    # So it uses available history. This is effectively min_periods=1.

    ref_max = df.rolling(W, min_periods=1).max().astype(np.float32)
    ref_min = df.rolling(W, min_periods=1).min().astype(np.float32)

    # Check max
    # We might have NaNs where input is all NaNs in window.
    # Pandas rolling max with min_periods=1 handles this.

    # There's a subtle difference:
    # If the window has valid values, they should match.
    # If the window is all NaNs, both should be NaN.

    pd.testing.assert_frame_equal(res_max, ref_max, atol=1e-5)
    pd.testing.assert_frame_equal(res_min, ref_min, atol=1e-5)

if __name__ == "__main__":
    test_rolling_max_min_correctness()
