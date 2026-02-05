import numpy as np
import pandas as pd
import pytest
import extreme_price_movements.fast_funcs as ff
from extreme_price_movements.feature_transforms import CausalFeatureTransformer, log_winsor_zscore_rolling

# Reference implementation mimicking the original class logic
def reference_transform(df, winsor_qt, roll_window):
    # 1. Log
    out = np.arcsinh(df)

    # 2. Quantile
    lower, upper = ff.numba_rolling_quantile_dual(out, roll_window, winsor_qt, 1 - winsor_qt)

    # Forward fill (original logic)
    lower = lower.ffill()
    upper = upper.ffill()

    # Clip
    out = out.clip(lower=lower, upper=upper, axis=0)

    # 3. Z-score
    mu = ff.numba_rolling_mean(out, roll_window)
    sigma = ff.numba_rolling_std(out, roll_window)

    z = (out - mu) / (sigma + 1e-12)
    return z.astype(np.float32)

def reference_series_transform(series, window, qt):
    x = np.arcsinh(series)
    x_df = x.to_frame()

    lo_df, hi_df = ff.numba_rolling_quantile_dual(x_df, window, qt, 1-qt)
    lo = lo_df[lo_df.columns[0]]
    hi = hi_df[hi_df.columns[0]]

    # Original had no ffill in the Series helper?
    # Let's check the code I read.
    # "lo = lo_df[lo_df.columns[0]]; hi = ...; x = x.clip(lower=lo, upper=hi)"
    # It seems the Series helper did NOT have ffill in the original code I read.
    # CausalFeatureTransformer.transform had ffill.

    x = x.clip(lower=lo, upper=hi)

    x_df = x.to_frame()
    mu_df = ff.numba_rolling_mean(x_df, window)
    sd_df = ff.numba_rolling_std(x_df, window)

    mu = mu_df[mu_df.columns[0]]
    sd = sd_df[sd_df.columns[0]]

    return (x - mu) / (sd + 1e-12)

def test_causal_feature_transformer_equivalence():
    np.random.seed(42)
    rows = 1000
    cols = 5
    data = np.random.randn(rows, cols).astype(np.float32)
    # Add some NaNs and extremes
    data[10:20, 0] = np.nan
    data[100, 1] = 1e6

    df = pd.DataFrame(data, columns=[f"c_{i}" for i in range(cols)])

    winsor_qt = 0.02
    roll_window = 50

    # Run Reference
    ref_res = reference_transform(df.copy(), winsor_qt, roll_window)

    # Run Class (which we will modify)
    transformer = CausalFeatureTransformer(winsor_qt=winsor_qt, roll_window=roll_window)
    # Note: we haven't added debug yet, assuming it works or will be added
    # For now, current class doesn't have debug.
    new_res = transformer.transform(df.copy())

    # Compare
    # We expect high similarity. If ffill makes a diff, it will be in specific spots.
    # But with min_periods=1 (default behavior of numba_rolling_quantile_dual?), ffill usually does nothing.
    pd.testing.assert_frame_equal(ref_res, new_res, atol=1e-5)

def test_series_helper_equivalence():
    np.random.seed(42)
    data = np.random.randn(1000).astype(np.float32)
    series = pd.Series(data, name="s1")

    window = 50
    qt = 0.02

    ref_res = reference_series_transform(series.copy(), window, qt)
    new_res = log_winsor_zscore_rolling(series.copy(), window, qt)

    pd.testing.assert_series_equal(ref_res, new_res, atol=1e-5)
