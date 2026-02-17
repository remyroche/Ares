import numpy as np
import pandas as pd
import pytest
from extreme_price_movements import fast_funcs as ff

def rolling_mad_standard_pandas(df: pd.DataFrame, window: int):
    # Standard MAD: median(|x - median(x)|) over window
    # rolling().apply() in Pandas applies function to array
    return df.rolling(window).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)

def test_numba_rolling_mad_accuracy():
    np.random.seed(42)
    # 5 assets, 200 rows
    data = np.random.randn(200, 5).astype(np.float32)
    df = pd.DataFrame(data, columns=[f"c_{i}" for i in range(5)])

    window = 24

    # Run Numba
    res = ff.numba_rolling_mad(df, window)

    # Run Pandas reference
    # Note: Pandas output will have first window-1 as NaNs.
    # Numba output will have first window-1 as 0.0 (default init).
    ref = rolling_mad_standard_pandas(df, window)

    # Compare excluding warmup period
    res_valid = res.iloc[window:]
    ref_valid = ref.iloc[window:]

    # Use float32 precision
    pd.testing.assert_frame_equal(res_valid, ref_valid.astype(np.float32), atol=1e-5)

if __name__ == "__main__":
    test_numba_rolling_mad_accuracy()
