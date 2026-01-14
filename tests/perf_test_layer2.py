
import pandas as pd
import numpy as np
import time
from src.training.steps.labeling.label_based_layer_2 import roll_entropy, get_serial_correlation
from src.utils.numba_funcs import _numba_rolling_entropy, _numba_return_autocorrelation

def test_performance():
    # Create synthetic data
    n_samples = 10000
    price = pd.Series(np.cumprod(1 + np.random.normal(0, 0.01, n_samples)), name='close')
    returns = price.pct_change().fillna(0)

    # Test roll_entropy
    print(f"Testing roll_entropy with {n_samples} samples...")
    start_time = time.time()
    res_orig = roll_entropy(returns, window=50, bins=10)
    end_time = time.time()
    orig_time = end_time - start_time
    print(f"Original roll_entropy: {orig_time:.4f}s")

    start_time = time.time()
    # Numba implementation
    # Note: _numba_rolling_entropy uses bins=5 by default in definition, need to pass bins=10
    # Also it returns numpy array
    res_numba_vals = _numba_rolling_entropy(returns.values, window=50, bins=10)
    res_numba = pd.Series(res_numba_vals, index=returns.index)
    # Adjust for log base difference if needed (orig uses log2, numba uses log10)
    # log2(x) = log10(x) / log10(2)
    res_numba_adjusted = res_numba / np.log10(2)
    end_time = time.time()
    numba_time = end_time - start_time
    print(f"Numba roll_entropy: {numba_time:.4f}s")
    print(f"Speedup: {orig_time / numba_time:.2f}x")

    # Check correlation
    valid_mask = ~np.isnan(res_orig) & ~np.isnan(res_numba_adjusted) & (res_orig != 0)
    if valid_mask.sum() > 0:
        corr = np.corrcoef(res_orig[valid_mask], res_numba_adjusted[valid_mask])[0, 1]
        print(f"Correlation: {corr:.4f}")
    else:
        print("No valid overlapping data for correlation")

    # Test get_serial_correlation
    print(f"\nTesting get_serial_correlation with {n_samples} samples...")
    start_time = time.time()
    res_orig = get_serial_correlation(returns, window=20)
    end_time = time.time()
    orig_time = end_time - start_time
    print(f"Original get_serial_correlation: {orig_time:.4f}s")

    start_time = time.time()
    res_numba_vals = _numba_return_autocorrelation(returns.values, window=20, lag=1)
    res_numba = pd.Series(res_numba_vals, index=returns.index)
    end_time = time.time()
    numba_time = end_time - start_time
    print(f"Numba get_serial_correlation: {numba_time:.4f}s")
    print(f"Speedup: {orig_time / numba_time:.2f}x")

    valid_mask = ~np.isnan(res_orig) & ~np.isnan(res_numba) & (res_orig != 0)
    if valid_mask.sum() > 0:
        corr = np.corrcoef(res_orig[valid_mask], res_numba[valid_mask])[0, 1]
        print(f"Correlation: {corr:.4f}")

if __name__ == "__main__":
    test_performance()
