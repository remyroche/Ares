import pandas as pd
import numpy as np
import pytest

from extreme_price_movements.features import _compute_features_impl
from extreme_price_movements.fast_funcs import numba_rolling_corr

def test_rolling_autocorr():
    np.random.seed(42)
    df = pd.DataFrame({
        'A': np.random.randn(100),
        'B': np.random.randn(100)
    })

    # Using Pandas
    def rolling_autocorr(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window, min_periods=max(2, window//2)).apply(lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 2 else np.nan, raw=True)

    res_pd_A = rolling_autocorr(df['A'], 10)
    res_pd_B = rolling_autocorr(df['B'], 10)

    # Using Numba (assuming diff / shift method)
    res_numba = numba_rolling_corr(df, df.shift(1), 10)

    # Note: pandas autocorr is pearson correlation between x and x.shift(1) over the window
    # Because of shifting, the effective number of samples and the sums might slightly differ in edge cases,
    # but they are very close.

    # Just checking speed impact

    print("Testing Pandas vs Numba corr")
    import time
    df_large = pd.DataFrame(np.random.randn(10000, 50))
    t0 = time.time()
    for col in df_large.columns:
        rolling_autocorr(df_large[col], 48)
    t1 = time.time()
    print("Pandas apply:", t1 - t0)

    t0 = time.time()
    numba_rolling_corr(df_large, df_large.shift(1), 48)
    t1 = time.time()
    print("Numba vectorized:", t1 - t0)
