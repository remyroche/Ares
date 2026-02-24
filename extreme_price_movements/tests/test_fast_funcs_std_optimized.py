import numpy as np
import pandas as pd
import pytest
from extreme_price_movements.fast_funcs import numba_rolling_std_parallel

def test_numba_rolling_std_parallel_correctness():
    np.random.seed(42)
    rows = 1000
    cols = 5
    data = np.random.randn(rows, cols).astype(np.float32)
    # Add some NaNs to test robustness
    data[10:20, 0] = np.nan
    data[50:55, 1] = np.nan

    df = pd.DataFrame(data, columns=[f"c_{i}" for i in range(cols)])
    window = 50

    # Numba implementation behaves like min_periods=2 (outputing values as soon as count > 1)

    expected = df.rolling(window, min_periods=2).std().astype(np.float32)

    # Numba implementation outputs
    actual = numba_rolling_std_parallel(df, window)

    # Check max difference
    # Note: float32 precision might lead to small diffs especially with different algorithms
    # (Welford vs whatever Pandas uses - likely two-pass or similar)

    pd.testing.assert_frame_equal(expected, actual, atol=1e-4)

if __name__ == "__main__":
    test_numba_rolling_std_parallel_correctness()
