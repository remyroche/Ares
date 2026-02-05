
import numpy as np
import pandas as pd
import pytest
from extreme_price_movements.fast_funcs import numba_rolling_quantile_dual

def test_numba_rolling_quantile_dual_accuracy():
    np.random.seed(42)
    # 5 assets, 1000 rows
    data = np.random.randn(1000, 5).astype(np.float32)
    df = pd.DataFrame(data, columns=[f"c_{i}" for i in range(5)])

    window = 50
    q1 = 0.02
    q2 = 0.98

    # Run Numba dual
    res1, res2 = numba_rolling_quantile_dual(df, window, q1, q2)

    # Run Pandas reference with min_periods=1 to match Numba implementation behavior
    ref1 = df.rolling(window, min_periods=1).quantile(q1).astype(np.float32)
    ref2 = df.rolling(window, min_periods=1).quantile(q2).astype(np.float32)

    # Fill NaNs for comparison (should be none now if inputs are valid, but safe to keep)
    res1 = res1.fillna(0)
    res2 = res2.fillna(0)
    ref1 = ref1.fillna(0)
    ref2 = ref2.fillna(0)

    pd.testing.assert_frame_equal(res1, ref1, atol=1e-5)
    pd.testing.assert_frame_equal(res2, ref2, atol=1e-5)

if __name__ == "__main__":
    test_numba_rolling_quantile_dual_accuracy()
