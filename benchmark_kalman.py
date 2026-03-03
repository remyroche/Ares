
import time
import numpy as np
import pandas as pd
from extreme_price_movements.features import _kalman_local_level_df

def benchmark():
    # Create dummy data
    rows = 10000
    cols = 100
    data = np.random.randn(rows, cols).cumsum(axis=0)
    # Add some NaNs
    mask = np.random.rand(rows, cols) < 0.1
    data[mask] = np.nan

    df = pd.DataFrame(data, columns=[f"col_{i}" for i in range(cols)])

    lambda_qr = 0.5

    print(f"Benchmarking _kalman_local_level_df with shape {df.shape}...")

    start_time = time.time()
    _kalman_local_level_df(df, lambda_qr)
    end_time = time.time()

    print(f"Time taken: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    benchmark()
