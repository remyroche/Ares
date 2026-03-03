
import time
import numpy as np
import pandas as pd
from extreme_price_movements.features import _kalman_local_level_df, _robust_obs_var_per_col
from extreme_price_movements.fast_funcs import numba_kalman_filter

# Dummy original implementation for benchmark comparison since we already replaced it in features.py
def _kalman_original_py(y_df, lambda_qr, r_base=None):
    y = y_df.to_numpy(dtype=np.float64)
    t_len, n_cols = y.shape
    r = _robust_obs_var_per_col(y_df) if r_base is None else np.asarray(r_base, dtype=np.float64)
    r = np.clip(r, 1e-8, None)
    q = np.clip(lambda_qr, 1e-8, None) * r

    x = np.full_like(y, np.nan, dtype=np.float64)
    innov_var = np.full_like(y, np.nan, dtype=np.float64)
    p_state = np.full_like(y, np.nan, dtype=np.float64)

    first_obs = np.where(np.isfinite(y[0]), y[0], 0.0)
    x_prev = first_obs.copy()
    p_prev = r.copy()

    for t in range(t_len):
        y_t = y[t]
        x_pred = x_prev
        p_pred = p_prev + q

        s_t = p_pred + r
        k_t = p_pred / np.clip(s_t, 1e-12, None)
        innov_t = y_t - x_pred

        valid = np.isfinite(y_t)
        x_new = np.where(valid, x_pred + k_t * innov_t, x_pred)
        p_new = np.where(valid, (1.0 - k_t) * p_pred, p_pred)

        x[t] = x_new
        innov_var[t] = s_t
        p_state[t] = p_new

        x_prev = x_new
        p_prev = p_new

    return x, innov_var, p_state

def benchmark_loop():
    np.random.seed(42)
    rows = 20000
    cols = 200
    print(f"Data shape: {rows}x{cols}")
    data = np.random.randn(rows, cols).cumsum(axis=0)
    # Add NaNs
    mask = np.random.rand(rows, cols) < 0.1
    data[mask] = np.nan
    data[0, 0:10] = np.nan # Test init

    df = pd.DataFrame(data, columns=[f"col_{i}" for i in range(cols)])
    lambda_qr = 0.5

    # Warmup Numba
    print("Warming up Numba...")
    _kalman_local_level_df(df.iloc[:100], lambda_qr)

    # Benchmark Original
    print("Benchmarking Original (3 runs)...")
    t_orig = []
    for _ in range(3):
        st = time.time()
        _kalman_original_py(df, lambda_qr)
        t_orig.append(time.time() - st)
    avg_orig = np.mean(t_orig)
    print(f"Avg Original: {avg_orig:.4f}s")

    # Benchmark New
    print("Benchmarking New Numba (3 runs)...")
    t_new = []
    for _ in range(3):
        st = time.time()
        # Call the actual feature function which uses numba internally now
        _kalman_local_level_df(df, lambda_qr)
        t_new.append(time.time() - st)
    avg_new = np.mean(t_new)
    print(f"Avg New:      {avg_new:.4f}s")

    print(f"Speedup:      {avg_orig/avg_new:.2f}x")

if __name__ == "__main__":
    benchmark_loop()
