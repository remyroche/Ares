
import numpy as np
import pandas as pd
import pytest
from extreme_price_movements import fast_funcs as ff

def test_numba_zscore_accuracy():
    np.random.seed(42)
    # Test with standard values
    rows = 1000
    cols = 5
    data = np.random.randn(rows, cols).astype(np.float32)
    df = pd.DataFrame(data, columns=[f"c_{i}" for i in range(cols)])

    window = 24

    # Numba implementation
    res_numba = ff.numba_zscore(df, window)

    # Pandas reference
    # Numba implementation requires count > 1 (min_periods=2)
    ref_mu = df.rolling(window, min_periods=2).mean()
    ref_std = df.rolling(window, min_periods=2).std(ddof=1)
    ref_z = (df - ref_mu) / (ref_std + 1e-12)

    # Fill NaNs for comparison
    res_numba = res_numba.fillna(0)
    ref_z = ref_z.fillna(0)

    pd.testing.assert_frame_equal(res_numba, ref_z.astype(np.float32), atol=1e-4)

def test_numba_zscore_stability():
    np.random.seed(42)
    # Test with large offsets to check numerical stability
    rows = 1000
    cols = 5
    # Use large values but variance is small (standard normal scaled)
    # Float32 has ~7 digits precision. 1,000,000 has 7 digits.
    # So we are at the edge. But variance calculation using float64 accumulators should handle it.
    data = (np.random.randn(rows, cols) * 10.0 + 1_000_000.0).astype(np.float32)
    df = pd.DataFrame(data, columns=[f"c_{i}" for i in range(cols)])

    window = 50

    res_numba = ff.numba_zscore(df, window)

    ref_mu = df.rolling(window, min_periods=2).mean()
    ref_std = df.rolling(window, min_periods=2).std(ddof=1)
    ref_z = (df - ref_mu) / (ref_std + 1e-12)

    res_numba = res_numba.fillna(0)
    ref_z = ref_z.fillna(0)

    pd.testing.assert_frame_equal(res_numba, ref_z.astype(np.float32), atol=1e-3)

def test_numba_zscore_nans():
    data = np.array([
        [1.0, 10.0],
        [np.nan, 12.0],
        [3.0, 14.0],
        [4.0, np.nan],
        [5.0, 16.0]
    ], dtype=np.float32)
    df = pd.DataFrame(data, columns=["A", "B"])

    window = 3

    res_numba = ff.numba_zscore(df, window)

    ref_mu = df.rolling(window, min_periods=2).mean()
    ref_std = df.rolling(window, min_periods=2).std(ddof=1)
    ref_z = (df - ref_mu) / (ref_std + 1e-12)

    pd.testing.assert_frame_equal(res_numba, ref_z.astype(np.float32), atol=1e-4)

if __name__ == "__main__":
    test_numba_zscore_accuracy()
    test_numba_zscore_stability()
    test_numba_zscore_nans()
