import numpy as np
import pandas as pd
import extreme_price_movements.fast_funcs as ff

def test_zscore():
    # Make a mock 2D array
    np.random.seed(42)
    c_log = pd.DataFrame(np.random.randn(1000, 50).astype(np.float32))

    c_log_arr = c_log.to_numpy()

    # Check if we can use ff._numba_rolling_mean_parallel
    mean_100 = ff._numba_rolling_mean_parallel(c_log_arr, 100)
    std_100 = ff._numba_rolling_std_parallel(c_log_arr, 100)
    std_100 = np.maximum(std_100, 1e-12)

    z_100 = (c_log_arr - mean_100) / std_100

    z_100_df = pd.DataFrame(z_100, index=c_log.index, columns=c_log.columns).astype(np.float32)
    print("Shape:", z_100_df.shape)

test_zscore()
