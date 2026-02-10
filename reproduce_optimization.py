import time
import numpy as np
import pandas as pd
from extreme_price_movements.fast_funcs import numba_rolling_robust_zscore

def benchmark_robust_zscore():
    print("Generating data...")
    np.random.seed(42)
    # 50 assets, 50k rows
    n_assets = 50
    n_rows = 50000
    data = np.random.randn(n_rows, n_assets).astype(np.float32)
    # Add some outliers
    data[::100] *= 10.0

    df = pd.DataFrame(data, columns=[f"asset_{i}" for i in range(n_assets)])

    window = 2160 # 90 days
    quantile = 0.45

    print(f"Benchmarking numba_rolling_robust_zscore on {n_rows}x{n_assets} with window={window}...")

    # 1. Exact
    print("Running Exact...")
    start_time = time.time()
    res_exact = numba_rolling_robust_zscore(df, window, quantile, max_samples=None)
    end_time = time.time()
    exact_dur = end_time - start_time
    print(f"Exact Time: {exact_dur:.4f} seconds")

    # 2. Approx
    print("Running Approx (max_samples=300)...")
    start_time = time.time()
    res_approx = numba_rolling_robust_zscore(df, window, quantile, max_samples=300)
    end_time = time.time()
    approx_dur = end_time - start_time
    print(f"Approx Time: {approx_dur:.4f} seconds")

    print(f"Speedup: {exact_dur / approx_dur:.2f}x")

    # Check errors
    # Ignore startup
    valid_exact = res_exact.iloc[window:].fillna(0).to_numpy()
    valid_approx = res_approx.iloc[window:].fillna(0).to_numpy()

    mae = np.mean(np.abs(valid_exact - valid_approx))
    print(f"MAE: {mae:.4f}")

if __name__ == "__main__":
    benchmark_robust_zscore()
