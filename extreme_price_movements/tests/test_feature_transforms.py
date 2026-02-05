import pytest
import numpy as np
import pandas as pd
from extreme_price_movements.feature_transforms import CausalFeatureTransformer

def reference_transform(df, window, winsor_qt):
    # Log transform
    out = np.arcsinh(df)

    # Rolling Quantiles
    lower = out.rolling(window=window, min_periods=1).quantile(winsor_qt)
    upper = out.rolling(window=window, min_periods=1).quantile(1 - winsor_qt)

    # Forward fill limits
    lower = lower.ffill()
    upper = upper.ffill()

    # Clip
    out = out.clip(lower=lower, upper=upper, axis=0)

    # Z-Score
    mu = out.rolling(window=window, min_periods=1).mean()
    sigma = out.rolling(window=window, min_periods=1).std(ddof=1)

    z = (out - mu) / (sigma + 1e-12)
    return z

def test_causal_feature_transformer_equivalence():
    np.random.seed(42)
    rows = 1000
    cols = 5
    data = np.random.randn(rows, cols)
    # Add some NaNs
    data[10:20, 0] = np.nan
    data[50:60, 2] = np.nan

    df = pd.DataFrame(data, columns=[f"col_{i}" for i in range(cols)])

    window = 50
    winsor_qt = 0.02

    transformer = CausalFeatureTransformer(winsor_qt=winsor_qt, roll_window=window)

    # Current implementation run
    res_opt = transformer.transform(df.copy(), name="test_df")

    # Reference run
    res_ref = reference_transform(df.copy(), window, winsor_qt)

    # Compare
    # Note: Rolling implementations might differ slightly (min_periods, interpolation)
    # The existing CausalFeatureTransformer uses numba rolling which might handle min_periods differently or use different quantile interpolation (linear).
    # Numba implementation usually does "linear" interpolation. Pandas default is "linear".
    # However, standard deviation ddof might differ. Pandas uses ddof=1 by default. _numba_rolling_std usually uses ddof=1.

    # We expect close match, but maybe not exact due to float32 vs float64.
    # The optimized one will be float32. Reference is float64.

    # Also, initial window NaNs might differ if min_periods is different.
    # Numba rolling usually requires full window or min_periods=window?
    # Let's check fast_funcs.py logic.
    # _numba_rolling_mean_nan_safe logic: if current_count > 0 -> output. So min_periods=1 effectively.
    # _numba_rolling_quantile: count valid in window. if count == 0 continue. So min_periods=1.

    pd.testing.assert_frame_equal(res_opt.astype(np.float64), res_ref, atol=1e-4, check_dtype=False)

if __name__ == "__main__":
    test_causal_feature_transformer_equivalence()
