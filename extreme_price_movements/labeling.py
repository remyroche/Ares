import numpy as np
import pandas as pd
from numba import jit

@jit(nopython=True, cache=True)
def _numba_triple_barrier_kernel(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    event_indices: np.ndarray,
    sides: np.ndarray,
    pt_arr: np.ndarray,
    sl_arr: np.ndarray,
    horizon_arr: np.ndarray
):
    """
    Numba kernel for Triple Barrier Method.

    Args:
        close: Close prices array
        high: High prices array
        low: Low prices array
        event_indices: Array of integer indices where events occur (entry bars)
        sides: Array of side integers (1 for Long, -1 for Short)
        pt_arr: Array of profit targets (as fraction, e.g. 0.01 for 1%)
        sl_arr: Array of stop losses (as positive fraction, e.g. 0.01 for 1%)
        horizon_arr: Array of horizons (int bars)

    Returns:
        out_labels: Array of labels (1=Profit, -1=Loss, 0=Timeout)
        out_rets: Array of realized returns
        out_idx: Array of exit indices
    """
    n_events = len(event_indices)
    n_bars = len(close)

    out_labels = np.zeros(n_events, dtype=np.int8)
    out_rets = np.zeros(n_events, dtype=np.float32)
    out_idx = np.zeros(n_events, dtype=np.int64)

    for i in range(n_events):
        idx = event_indices[i]

        # Valid entry check
        if idx >= n_bars - 1:
            out_labels[i] = 0
            out_rets[i] = 0.0
            out_idx[i] = idx
            continue

        entry_price = close[idx]
        side = sides[i]
        pt = pt_arr[i]
        sl = sl_arr[i]
        horizon = horizon_arr[i]

        limit_idx = min(idx + horizon, n_bars - 1)
        exit_idx = limit_idx
        exit_reason = 0 # 0: Timeout, 1: Profit, -1: Stop

        # Determine barrier levels
        if side == 1: # Long
            upper_barrier = entry_price * (1.0 + pt)
            lower_barrier = entry_price * (1.0 - sl)
        else: # Short
            upper_barrier = entry_price * (1.0 - pt) # Profit for short is lower price
            lower_barrier = entry_price * (1.0 + sl) # Stop for short is higher price

        # Scan path
        # Start from idx + 1
        for t in range(idx + 1, limit_idx + 1):
            h = high[t]
            l = low[t]

            hit_profit = False
            hit_stop = False

            if side == 1: # Long
                if h >= upper_barrier:
                    hit_profit = True
                if l <= lower_barrier:
                    hit_stop = True
            else: # Short
                if l <= upper_barrier: # Profit hit
                    hit_profit = True
                if h >= lower_barrier: # Stop hit
                    hit_stop = True

            # Check First Touch
            # If both hit in same bar, conservative assumption: Stop is hit.

            if hit_stop and hit_profit:
                exit_idx = t
                exit_reason = -1
                break
            elif hit_stop:
                exit_idx = t
                exit_reason = -1
                break
            elif hit_profit:
                exit_idx = t
                exit_reason = 1
                break

        # Calculate Return
        exit_price_val = close[exit_idx]

        # If barrier hit, use barrier price (assuming execution at barrier)
        if exit_reason == 1:
            exit_price_val = upper_barrier
        elif exit_reason == -1:
            exit_price_val = lower_barrier

        # For Timeout (exit_reason 0), use close at limit

        if side == 1:
            ret = (exit_price_val - entry_price) / entry_price
        else:
            ret = (entry_price - exit_price_val) / entry_price

        out_labels[i] = exit_reason
        out_rets[i] = ret
        out_idx[i] = exit_idx

    return out_labels, out_rets, out_idx

def compute_triple_barrier_labels(
    prices_df: pd.DataFrame,
    events_df: pd.DataFrame,
    pt: float = 0.01,
    sl: float = 0.01,
    horizon: int = 24,
    side_col: str = None
) -> pd.DataFrame:
    """
    Compute Triple Barrier Method labels.

    Args:
        prices_df: DataFrame with 'close', 'high', 'low'.
        events_df: DataFrame identifying events. Index must be subset of prices_df index.
                   Columns can override defaults: 'pt', 'sl', 'horizon', 'side'.
        pt: Default Profit target (fraction). Used if 'pt' not in events_df.
        sl: Default Stop loss (fraction). Used if 'sl' not in events_df.
        horizon: Default Time horizon. Used if 'horizon' not in events_df.
        side_col: Column name in events_df for side (1/-1). Defaults to 1 (Long) if None/missing.

    Returns:
        DataFrame with index from events_df and columns:
        ['ret', 'label', 'exit_idx', 'exit_ts']
    """
    # 1. Align Data
    # We need arrays for Numba.
    # Map events index to integer positions in prices_df.

    if 'close' not in prices_df or 'high' not in prices_df or 'low' not in prices_df:
        raise ValueError("prices_df must contain close, high, low")

    close_arr = prices_df['close'].to_numpy(dtype=np.float32)
    high_arr = prices_df['high'].to_numpy(dtype=np.float32)
    low_arr = prices_df['low'].to_numpy(dtype=np.float32)

    # Get integer locations of events
    # Optimally, prices_df index is unique.
    if not prices_df.index.is_unique:
        raise ValueError("prices_df index must be unique")

    # intersection
    valid_events = events_df.index.intersection(prices_df.index)
    if len(valid_events) < len(events_df):
        # Warn or filter? Let's filter
        pass

    events_subset = events_df.loc[valid_events]

    # Map index to integers
    # This can be slow for large index.
    # If prices_df is sorted, searchsorted is fast.
    if prices_df.index.is_monotonic_increasing:
        event_indices = prices_df.index.searchsorted(valid_events)
    else:
        # Fallback to get_indexer
        event_indices = prices_df.index.get_indexer(valid_events)

    # 2. Prepare Parameter Arrays
    n = len(events_subset)

    # Side
    if side_col and side_col in events_subset.columns:
        sides = events_subset[side_col].to_numpy(dtype=np.int8)
    else:
        sides = np.ones(n, dtype=np.int8)

    # PT
    if 'pt' in events_subset.columns:
        pt_arr = events_subset['pt'].to_numpy(dtype=np.float32)
    else:
        pt_arr = np.full(n, pt, dtype=np.float32)

    # SL
    if 'sl' in events_subset.columns:
        sl_arr = events_subset['sl'].to_numpy(dtype=np.float32)
    else:
        sl_arr = np.full(n, sl, dtype=np.float32)

    # Horizon
    if 'horizon' in events_subset.columns:
        h_arr = events_subset['horizon'].to_numpy(dtype=np.int32)
    else:
        h_arr = np.full(n, horizon, dtype=np.int32)

    # 3. Call Numba Kernel
    out_labels, out_rets, out_idx = _numba_triple_barrier_kernel(
        close_arr, high_arr, low_arr,
        event_indices.astype(np.int64),
        sides,
        pt_arr,
        sl_arr,
        h_arr
    )

    # 4. Format Output
    res = pd.DataFrame(index=valid_events)
    res['ret'] = out_rets
    res['label'] = out_labels
    res['exit_idx'] = out_idx

    # Map exit_idx back to timestamps
    res['exit_ts'] = prices_df.index[out_idx]

    return res
