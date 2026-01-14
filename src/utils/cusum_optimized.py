import numpy as np
from numba import njit, prange

@njit(parallel=False)
def generate_dual_cusum_numba(
    diff_arr: np.ndarray,
    h_arr: np.ndarray,
    er_arr: np.ndarray,
    er_min: float
) -> np.ndarray:
    """
    Generate dual CUSUM signals using Numba optimization.

    Args:
        diff_arr: Array of price differences (close.diff())
        h_arr: Array of dynamic thresholds (k * vol)
        er_arr: Array of efficiency ratios
        er_min: Minimum efficiency ratio for trend regime

    Returns:
        Array of shape (N, 2) where col 0 is trend_signal and col 1 is reversal_signal.
        Values are -1, 0, 1.
    """
    n = len(diff_arr)
    # Using int8 to save memory, values are -1, 0, 1
    # 0: trend, 1: reversal
    signals = np.zeros((n, 2), dtype=np.int8)

    s_pos = 0.0
    s_neg = 0.0

    # Start from 1 because diff[0] is typically NaN or 0 (diff requires previous)
    # The original loop started from 1.

    for i in range(1, n):
        # Skip if h is NaN (can happen at start due to rolling window)
        h_val = h_arr[i]
        if np.isnan(h_val):
            continue

        diff_val = diff_arr[i]
        # Handle NaN diff just in case
        if np.isnan(diff_val):
            continue

        s_pos = max(0.0, s_pos + diff_val)
        s_neg = min(0.0, s_neg + diff_val)

        er_val = er_arr[i]

        if s_pos > h_val:
            # Upside break
            if er_val > er_min:
                signals[i, 0] = 1 # Trend Long
            else:
                signals[i, 1] = -1 # Reversal Short
            s_pos = 0.0

        elif s_neg < -h_val:
            # Downside break
            if er_val > er_min:
                signals[i, 0] = -1 # Trend Short
            else:
                signals[i, 1] = 1 # Reversal Long
            s_neg = 0.0

    return signals
