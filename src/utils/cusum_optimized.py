import numpy as np
from numba import jit

@jit(nopython=True)
def generate_dual_cusum_numba(diff_arr, h_arr, er_arr, er_min):
    """
    Numba optimized Dual CUSUM signal generation.
    """
    n = len(diff_arr)
    trend_signal = np.zeros(n, dtype=np.int8)
    reversal_signal = np.zeros(n, dtype=np.int8)

    s_pos = 0.0
    s_neg = 0.0

    # Iterate starting from 1 since index 0 has no diff
    for i in range(1, n):
        h_val = h_arr[i]
        if np.isnan(h_val):
            continue

        diff = diff_arr[i]

        # S+
        s_pos = max(0.0, s_pos + diff)
        # S-
        s_neg = min(0.0, s_neg + diff)

        if s_pos > h_val:
            # Regime Detection
            if er_arr[i] > er_min:
                trend_signal[i] = 1
            else:
                reversal_signal[i] = -1
            s_pos = 0.0

        elif s_neg < -h_val:
            if er_arr[i] > er_min:
                trend_signal[i] = -1
            else:
                reversal_signal[i] = 1
            s_neg = 0.0

    return trend_signal, reversal_signal
