import numpy as np
from numba import njit, prange

@njit(parallel=False)
def compute_volume_spike_numba(
    volume: np.ndarray,
    window: int
) -> np.ndarray:
    n = len(volume)
    result = np.zeros(n)

    if n < window:
        return result

    for i in prange(window - 1, n):
        window_data = volume[i - window + 1 : i + 1]

        # Calculate mean
        mean_val = 0.0
        for val in window_data:
            mean_val += val
        mean_val /= window

        # Calculate std
        var_val = 0.0
        for val in window_data:
            diff = val - mean_val
            var_val += diff * diff
        var_val /= window
        std_val = np.sqrt(var_val)

        if std_val < 1e-9:
            std_val = 1e-9

        result[i] = (volume[i] - mean_val) / std_val

    return result

@njit(parallel=False)
def compute_return_shock_numba(
    returns: np.ndarray,
    window: int
) -> np.ndarray:
    n = len(returns)
    result = np.zeros(n)

    if n < window:
        return result

    for i in prange(window - 1, n):
        window_data = returns[i - window + 1 : i + 1]

        # Mean
        mean_val = 0.0
        for val in window_data:
            mean_val += val
        mean_val /= window

        # Std
        var_val = 0.0
        for val in window_data:
            diff = val - mean_val
            var_val += diff * diff
        var_val /= window
        std_val = np.sqrt(var_val)

        if std_val < 1e-9:
            std_val = 1e-9

        result[i] = np.abs(returns[i]) / std_val

    return result

@njit(parallel=False)
def compute_trade_intensity_numba(
    volume: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int
) -> np.ndarray:
    n = len(volume)
    result = np.zeros(n)
    tr = np.zeros(n)
    intensity = np.zeros(n)

    # Calculate True Range and Intensity first
    tr[0] = high[0] - low[0]
    if tr[0] < 1e-9: tr[0] = 1e-9
    intensity[0] = volume[0] / tr[0]

    for i in range(1, n):
        h_l = high[i] - low[i]
        h_pc = np.abs(high[i] - close[i-1])
        l_pc = np.abs(low[i] - close[i-1])

        current_tr = max(h_l, max(h_pc, l_pc))
        if current_tr < 1e-9:
            current_tr = 1e-9
        tr[i] = current_tr
        intensity[i] = volume[i] / current_tr

    if n < window:
        return result

    for i in prange(window - 1, n):
        window_data = intensity[i - window + 1 : i + 1]

        mean_val = 0.0
        for val in window_data:
            mean_val += val
        mean_val /= window

        var_val = 0.0
        for val in window_data:
            diff = val - mean_val
            var_val += diff * diff
        var_val /= window
        std_val = np.sqrt(var_val)

        if std_val < 1e-9:
            std_val = 1e-9

        result[i] = (intensity[i] - mean_val) / std_val

    return result

@njit(parallel=False)
def compute_order_flow_imbalance_numba(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    window: int
) -> np.ndarray:
    """
    Compute order flow imbalance signal using Numba.
    Matches Pandas logic: close_position * volume (not centered).
    """
    n = len(close)
    result = np.zeros(n)
    vw_pos = np.zeros(n) # Volume weighted position

    for i in range(n):
        bar_range = high[i] - low[i]
        if bar_range < 1e-9:
            pos = 0.5
        else:
            pos = (close[i] - low[i]) / bar_range

        # Fixed logic: use pos * volume, not (pos - 0.5) * volume
        vw_pos[i] = pos * volume[i]

    if n < window:
        return result

    for i in prange(window - 1, n):
        window_data = vw_pos[i - window + 1 : i + 1]

        mean_val = 0.0
        for val in window_data:
            mean_val += val
        mean_val /= window

        var_val = 0.0
        for val in window_data:
            diff = val - mean_val
            var_val += diff * diff
        var_val /= window
        std_val = np.sqrt(var_val)

        if std_val < 1e-9:
            std_val = 1e-9

        result[i] = (vw_pos[i] - mean_val) / std_val

    return result
