import numpy as np
from numba import njit, prange

@njit(parallel=False)
def find_order_blocks_numba(
    open_arr: np.ndarray,
    close_arr: np.ndarray,
    volume_arr: np.ndarray,
    volume_ma_arr: np.ndarray,
    lookback: int,
    min_move_pct: float,
    volume_threshold: float
) -> np.ndarray:
    """
    Find order blocks using Numba for performance.

    Args:
        open_arr: Open prices array
        close_arr: Close prices array
        volume_arr: Volume array
        volume_ma_arr: Rolling mean of volume array
        lookback: Lookahead window size
        min_move_pct: Minimum move percentage (as a float, e.g. 0.5 for 0.5%)
        volume_threshold: Multiplier for volume threshold

    Returns:
        Boolean array where True indicates an order block event at that index.
    """
    n = len(close_arr)
    events = np.zeros(n, dtype=np.bool_)

    # Pre-convert percentage to decimal ratio
    move_thresh_decimal = min_move_pct / 100.0

    # We iterate until n - lookback because we need to look ahead
    # Also start from 'lookback' to respect the original logic's range,
    # though the original code passed lookback to range() start.
    # Original: range(lookback, len(df) - lookback)

    for i in prange(lookback, n - lookback):
        # Current candle data
        c_close = close_arr[i]
        c_open = open_arr[i]
        c_vol = volume_arr[i]
        c_vol_ma = volume_ma_arr[i]

        # Determine if candle is candidate
        is_bullish_candidate = (c_close < c_open) and (c_vol > c_vol_ma * volume_threshold)
        is_bearish_candidate = (c_close > c_open) and (c_vol > c_vol_ma * volume_threshold)

        if not (is_bullish_candidate or is_bearish_candidate):
            continue

        future_move = True

        # Look ahead
        # Determine check range limit
        limit = min(i + lookback, n)

        if is_bullish_candidate:
            # Check for strong up move
            for j in range(i + 1, limit):
                move = (close_arr[j] - c_close) / c_close
                if move > move_thresh_decimal:
                    future_move = True
                    break
                elif move < -move_thresh_decimal:
                    future_move = False
                    break
            # Note: if loop finishes without break, future_move remains True (based on original logic)

        elif is_bearish_candidate:
            # Check for strong down move
            for j in range(i + 1, limit):
                move = (close_arr[j] - c_close) / c_close
                if move < -move_thresh_decimal:
                    future_move = True
                    break
                elif move > move_thresh_decimal:
                    future_move = False
                    break
            # Note: if loop finishes without break, future_move remains True (based on original logic)

        if future_move:
            events[i] = True

    return events
