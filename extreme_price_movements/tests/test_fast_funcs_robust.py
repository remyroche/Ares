import numpy as np
import pandas as pd
import pytest
import time
from extreme_price_movements.fast_funcs import numba_rolling_robust_zscore

def test_robust_zscore_exact():
    np.random.seed(42)
    # Generate some data
    n = 1000
    data = np.random.randn(n, 2).astype(np.float32)
    df = pd.DataFrame(data, columns=["A", "B"])

    window = 50
    # Exact calculation (no max_samples or large max_samples)
    res_exact = numba_rolling_robust_zscore(df, window, quantile=0.5, max_samples=None)

    # Check shape
    assert res_exact.shape == df.shape

    # Check consistency (not NaN everywhere)
    assert not res_exact.iloc[window:].isna().all().all()

def test_robust_zscore_approx_accuracy():
    np.random.seed(42)
    # Generate longer data
    n = 5000
    data = np.random.randn(n, 1).astype(np.float32)
    # Add trend and noise
    data += np.linspace(0, 10, n).reshape(-1, 1)
    df = pd.DataFrame(data, columns=["A"])

    window = 500
    # Exact
    res_exact = numba_rolling_robust_zscore(df, window, max_samples=None)

    # Approx (step = 500 // 50 = 10)
    res_approx = numba_rolling_robust_zscore(df, window, max_samples=50)

    # Check correlation
    # Ignore startup period
    valid_exact = res_exact.iloc[window:].fillna(0)
    valid_approx = res_approx.iloc[window:].fillna(0)

    corr = valid_exact["A"].corr(valid_approx["A"])
    print(f"Correlation between Exact and Approx (samples=50): {corr:.4f}")

    # Expect high correlation
    assert corr > 0.90

    # Check MAE
    mae = (valid_exact - valid_approx).abs().mean().item()
    print(f"MAE between Exact and Approx: {mae:.4f}")

    # Z-scores are usually around -3 to 3. MAE should be small.
    # With 50 samples, estimation error is 1/sqrt(50) ~ 0.14 relative?
    assert mae < 0.5

def test_robust_zscore_performance():
    np.random.seed(42)
    n_rows = 50000
    n_cols = 10
    data = np.random.randn(n_rows, n_cols).astype(np.float32)
    df = pd.DataFrame(data)

    window = 2160 # 90 days

    # Measure Exact
    t0 = time.time()
    res_exact = numba_rolling_robust_zscore(df, window, max_samples=None)
    t1 = time.time()
    time_exact = t1 - t0

    # Measure Approx
    t0 = time.time()
    res_approx = numba_rolling_robust_zscore(df, window, max_samples=300)
    t1 = time.time()
    time_approx = t1 - t0

    print(f"Performance (50k rows, 10 cols, W=2160): Exact={time_exact:.4f}s, Approx={time_approx:.4f}s")
    print(f"Speedup: {time_exact / time_approx:.2f}x")

    # Expect significant speedup (step = 2160 // 300 = 7) -> ~7x speedup ideally (minus overhead)
    assert time_approx < time_exact
    assert (time_exact / time_approx) > 2.0

if __name__ == "__main__":
    test_robust_zscore_exact()
    test_robust_zscore_approx_accuracy()
    test_robust_zscore_performance()
