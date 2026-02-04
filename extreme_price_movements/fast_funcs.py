import numpy as np
import pandas as pd
from numba import jit

# Import optimized kernels from the main codebase
from src.utils.numba_funcs import (
    _numba_rolling_mean_nan_safe,
    _numba_rolling_std_nan_safe,
    _numba_ewma,
    _numba_rolling_vwap,
    _numba_rolling_kurt,
    _numba_rolling_skew,
    _numba_rolling_slope,
    _numba_rolling_rsquared,
    _numba_rolling_median,
    _numba_rolling_sum,
    _numba_rolling_correlation
)
from .utils import tprint

def apply_to_frame(df: pd.DataFrame, func, *args) -> pd.DataFrame:
    """
    Applies a Numba 1D function to each column of a DataFrame.
    Returns a DataFrame with float32 dtype.
    Handles pd.Series by converting to DataFrame and returning Series.
    """
    tprint(f"Entering function: apply_to_frame in fast_funcs.py")
    is_series = isinstance(df, pd.Series)
    if is_series:
        df = df.to_frame()

    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=np.float32)
    # We iterate over columns.
    for col in df.columns:
        # Convert to numpy float32 array
        vals = df[col].to_numpy(dtype=np.float32)
        # Apply function
        res = func(vals, *args)
        out[col] = res

    if is_series:
        return out[out.columns[0]]

    return out

def apply_to_frame_binary(df1: pd.DataFrame, df2: pd.DataFrame, func, *args) -> pd.DataFrame:
    """
    Applies a function taking two arrays (col1, col2) -> out_array.
    Assumes df1 and df2 have same columns and index.
    Handles pd.Series inputs.
    """
    tprint(f"Entering function: apply_to_frame_binary in fast_funcs.py")
    is_series1 = isinstance(df1, pd.Series)
    is_series2 = isinstance(df2, pd.Series)

    if is_series1:
        df1 = df1.to_frame()
    if is_series2:
        df2 = df2.to_frame()

    out = pd.DataFrame(index=df1.index, columns=df1.columns, dtype=np.float32)
    for col in df1.columns:
        if col in df2.columns:
            v1 = df1[col].to_numpy(dtype=np.float32)
            v2 = df2[col].to_numpy(dtype=np.float32)
            out[col] = func(v1, v2, *args)

    if is_series1:
        return out[out.columns[0]]

    return out

@jit(nopython=True, cache=True)
def numba_rsi_kernel(close, n):
    """
    RSI = 100 - 100 / (1 + RS)
    RS = Average Gain / Average Loss
    """
    delta = np.empty_like(close)
    delta[0] = np.nan
    delta[1:] = close[1:] - close[:-1]

    up = np.maximum(delta, 0.0)
    dn = np.maximum(-delta, 0.0)

    # EWMA with alpha = 1/n, adjust=False
    alpha = 1.0 / n

    avg_up = _numba_ewma(up, alpha, adjust=False)
    avg_dn = _numba_ewma(dn, alpha, adjust=False)

    out = np.empty_like(close)
    for i in range(len(close)):
        if np.isnan(avg_dn[i]) or avg_dn[i] == 0:
            if np.isnan(avg_up[i]):
                out[i] = np.nan
            elif avg_up[i] == 0:
                 out[i] = 50.0 # No move
            else:
                 out[i] = 100.0 # Only up moves
        else:
            rs = avg_up[i] / avg_dn[i]
            out[i] = 100.0 - (100.0 / (1.0 + rs))

    return out

def numba_rsi(close_df, n):
    tprint(f"Entering function: numba_rsi in fast_funcs.py")
    return apply_to_frame(close_df, numba_rsi_kernel, n)

@jit(nopython=True, cache=True)
def numba_atr_kernel(high, low, close, n):
    """
    ATR using EWM smoothing.
    TR = max(h-l, abs(h-prev_c), abs(l-prev_c))
    """
    sz = len(close)
    tr = np.empty(sz, dtype=np.float32)
    tr[0] = high[0] - low[0] # First TR is High - Low

    for i in range(1, sz):
        h = high[i]; l = low[i]; pc = close[i-1]
        v1 = h - l
        v2 = abs(h - pc)
        v3 = abs(l - pc)
        tr[i] = max(v1, max(v2, v3))

    # ATR is EWM of TR
    atr = _numba_ewma(tr, 1.0/n, adjust=False)

    # Return ATR percent: ATR / Close
    out = np.empty(sz, dtype=np.float32)
    for i in range(sz):
        c = close[i]
        if c == 0 or np.isnan(c):
             out[i] = np.nan
        else:
             out[i] = atr[i] / c

    return out

@jit(nopython=True, cache=True)
def numba_atr_no_norm_kernel(high, low, close, n):
    """
    ATR using EWM smoothing, without normalization by close.
    """
    sz = len(close)
    tr = np.empty(sz, dtype=np.float32)
    tr[0] = high[0] - low[0]

    for i in range(1, sz):
        h = high[i]; l = low[i]; pc = close[i-1]
        v1 = h - l
        v2 = abs(h - pc)
        v3 = abs(l - pc)
        tr[i] = max(v1, max(v2, v3))

    atr = _numba_ewma(tr, 1.0/n, adjust=False)
    return atr

def numba_atr_no_norm(high_df, low_df, close_df, n):
    tprint(f"Entering function: numba_atr_no_norm in fast_funcs.py")
    out = pd.DataFrame(index=close_df.index, columns=close_df.columns, dtype=np.float32)
    cols = close_df.columns
    for c in cols:
        h = high_df[c].to_numpy(dtype=np.float32)
        l = low_df[c].to_numpy(dtype=np.float32)
        cl = close_df[c].to_numpy(dtype=np.float32)
        res = numba_atr_no_norm_kernel(h, l, cl, n)
        out[c] = res
    return out

def numba_atr(high_df, low_df, close_df, n):
    # This requires synchronized iteration over 3 dataframes.
    tprint(f"Entering function: numba_atr in fast_funcs.py")
    out = pd.DataFrame(index=close_df.index, columns=close_df.columns, dtype=np.float32)
    cols = close_df.columns
    for c in cols:
        h = high_df[c].to_numpy(dtype=np.float32)
        l = low_df[c].to_numpy(dtype=np.float32)
        cl = close_df[c].to_numpy(dtype=np.float32)
        res = numba_atr_kernel(h, l, cl, n)
        out[c] = res
    return out

def numba_zscore(df, n):
    # (x - mean) / std
    # Using nan_safe versions
    tprint(f"Entering function: numba_zscore in fast_funcs.py")
    mu = apply_to_frame(df, _numba_rolling_mean_nan_safe, n)
    sd = apply_to_frame(df, _numba_rolling_std_nan_safe, n)

    # Vectorized pandas operation for final step is fine/fast
    return (df - mu) / (sd + 1e-12)

@jit(nopython=True, cache=True)
def simulate_trade_numba(
    opens, highs, lows, closes,
    entry_px, side_int, atr,
    k_sl, k_ts, k_td
):
    """
    Simulates a trade path using Numba optimization.
    side_int: 1 for long, -1 for short.
    Returns: (return_pct, exit_idx_offset, reason_code)
    reason_code: 0=no_entry/error, 1=sl_hit, 2=ambiguous, 3=time_exit
    """
    # Logic from TrailingStop
    initial_sl_dist = k_sl * atr * entry_px

    sl_px = 0.0
    highest_high = entry_px
    lowest_low = entry_px
    trailing_active = False

    if side_int == 1: # Long
        sl_px = entry_px - initial_sl_dist
    else: # Short
        sl_px = entry_px + initial_sl_dist

    n = len(opens)
    for i in range(n):
        curr_h = highs[i]
        curr_l = lows[i]
        curr_c = closes[i]

        if np.isnan(curr_h) or np.isnan(curr_l):
            continue

        if side_int == 1: # Long
            # Check Stop
            stop_hit = curr_l <= sl_px
            trail_would_trigger = False

            # Check Trailing Update (High)
            if curr_h > highest_high:
                highest_high = curr_h

            # Update Trailing State
            profit_dist = highest_high - entry_px
            if profit_dist >= k_ts * atr * entry_px:
                trailing_active = True

            if trailing_active:
                trail_dist_px = k_td * atr * entry_px
                new_sl = curr_h - trail_dist_px
                if new_sl > sl_px:
                    trail_would_trigger = True

            if stop_hit and trail_would_trigger:
                return 0.0, i, 2 # Ambiguous

            if stop_hit:
                return (sl_px / entry_px) - 1.0, i, 1 # SL Hit

            # Update High
            if curr_h > highest_high:
                highest_high = curr_h

            # Update Trailing State
            profit_dist = highest_high - entry_px
            if profit_dist >= k_ts * atr * entry_px:
                trailing_active = True

            if trailing_active:
                trail_dist_px = k_td * atr * entry_px
                new_sl = highest_high - trail_dist_px
                if new_sl > sl_px:
                    sl_px = new_sl

        else: # Short
            # Check Stop
            stop_hit = curr_h >= sl_px
            trail_would_trigger = False

            # Check Trailing Update (Low)
            if curr_l < lowest_low:
                profit_dist = entry_px - curr_l
                req_start_dist = k_ts * atr * entry_px

                is_active = trailing_active or (profit_dist >= req_start_dist)

                if is_active:
                    trail_dist_px = k_td * atr * entry_px
                    new_sl = curr_l + trail_dist_px
                    if new_sl < sl_px:
                        trail_would_trigger = True

            if stop_hit and trail_would_trigger:
                return 0.0, i, 2 # Ambiguous

            if stop_hit:
                return (entry_px / sl_px) - 1.0, i, 1 # SL Hit

            if curr_l < lowest_low:
                lowest_low = curr_l

            profit_dist = entry_px - lowest_low
            if profit_dist >= k_ts * atr * entry_px:
                trailing_active = True

            if trailing_active:
                trail_dist_px = k_td * atr * entry_px
                new_sl = lowest_low + trail_dist_px
                if new_sl < sl_px:
                    sl_px = new_sl

    # Time Exit
    last_c = closes[n-1]
    if side_int == 1:
        ret = (last_c / entry_px) - 1.0
    else:
        ret = (entry_px / last_c) - 1.0

    return ret, n-1, 3 # Time Exit

# --- NEW KERNELS & WRAPPERS ---

@jit(nopython=True, cache=True)
def _numba_rolling_max(x, window):
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        start = max(0, i - window + 1)
        end = i + 1
        valid = False
        m = -np.inf
        for j in range(start, end):
            val = x[j]
            if not np.isnan(val):
                if val > m:
                    m = val
                valid = True
        if valid:
            out[i] = m
    return out

@jit(nopython=True, cache=True)
def _numba_rolling_min(x, window):
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        start = max(0, i - window + 1)
        end = i + 1
        valid = False
        m = np.inf
        for j in range(start, end):
            val = x[j]
            if not np.isnan(val):
                if val < m:
                    m = val
                valid = True
        if valid:
            out[i] = m
    return out

@jit(nopython=True, cache=True)
def _numba_rolling_quantile(x, window, q):
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float32)

    for i in range(n):
        start = max(0, i - window + 1)
        end = i + 1

        # Count valid
        count = 0
        for j in range(start, end):
            if not np.isnan(x[j]):
                count += 1

        if count == 0:
            continue

        # Collect
        buf = np.empty(count, dtype=np.float32)
        idx = 0
        for j in range(start, end):
            val = x[j]
            if not np.isnan(val):
                buf[idx] = val
                idx += 1

        # Sort
        buf.sort()

        # Linear Interpolation
        v_idx = q * (count - 1)
        i_lower = int(np.floor(v_idx))
        i_upper = int(np.ceil(v_idx))
        fraction = v_idx - i_lower

        if i_lower == i_upper:
            out[i] = buf[i_lower]
        else:
            out[i] = buf[i_lower] + (buf[i_upper] - buf[i_lower]) * fraction

    return out

@jit(nopython=True, cache=True)
def _numba_pct_change(x, n_shift):
    l = len(x)
    out = np.full(l, np.nan, dtype=np.float32)
    for i in range(n_shift, l):
        prev = x[i - n_shift]
        curr = x[i]
        if prev != 0 and not np.isnan(prev) and not np.isnan(curr):
            out[i] = (curr - prev) / prev
    return out

def numba_grouped_rolling_mean(df: pd.DataFrame, group_series: pd.Series, window: int) -> pd.DataFrame:
    """
    Vectorized grouped rolling mean.
    """
    tprint(f"Entering function: numba_grouped_rolling_mean in fast_funcs.py")
    is_series = isinstance(df, pd.Series)
    if is_series:
        df = df.to_frame()

    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=np.float32)
    out[:] = np.nan

    # Ensure group series is aligned and numpy-ready
    groups_arr = group_series.reindex(df.index).to_numpy()
    unique_groups = np.unique(groups_arr)

    for col in df.columns:
        vals = df[col].to_numpy(dtype=np.float32)
        res_col = np.full_like(vals, np.nan)

        for g in unique_groups:
            if np.isnan(g):
                continue
            mask = (groups_arr == g)
            subset = vals[mask]

            # Use rolling mean on the subset
            rolled = _numba_rolling_mean_nan_safe(subset, window)

            res_col[mask] = rolled

        out[col] = res_col

    if is_series:
        return out[out.columns[0]]

    return out

# Wrappers
def numba_rolling_max(df, n):
    tprint(f"Entering function: numba_rolling_max in fast_funcs.py")
    return apply_to_frame(df, _numba_rolling_max, n)

def numba_rolling_min(df, n):
    tprint(f"Entering function: numba_rolling_min in fast_funcs.py")
    return apply_to_frame(df, _numba_rolling_min, n)

def numba_rolling_sum(df, n):
    tprint(f"Entering function: numba_rolling_sum in fast_funcs.py")
    return apply_to_frame(df, _numba_rolling_sum, n)

def numba_rolling_median(df, n):
    tprint(f"Entering function: numba_rolling_median in fast_funcs.py")
    return apply_to_frame(df, _numba_rolling_median, n)

def numba_rolling_quantile(df, n, q):
    tprint(f"Entering function: numba_rolling_quantile in fast_funcs.py")
    return apply_to_frame(df, _numba_rolling_quantile, n, q)

def numba_pct_change(df, n):
    tprint(f"Entering function: numba_pct_change in fast_funcs.py")
    return apply_to_frame(df, _numba_pct_change, n)

def numba_rolling_corr(df1, df2, n):
    tprint(f"Entering function: numba_rolling_corr in fast_funcs.py")
    return apply_to_frame_binary(df1, df2, _numba_rolling_correlation, n)

def numba_rolling_mean(df, n):
    tprint(f"Entering function: numba_rolling_mean in fast_funcs.py")
    return apply_to_frame(df, _numba_rolling_mean_nan_safe, n)

def numba_rolling_std(df, n):
    tprint(f"Entering function: numba_rolling_std in fast_funcs.py")
    return apply_to_frame(df, _numba_rolling_std_nan_safe, n)

@jit(nopython=True, cache=True)
def _numba_frac_diff_kernel(x, d, window):
    # Fixed Width Window Frac Diff
    # w_k = -w_{k-1} * (d - k + 1) / k
    # w_0 = 1

    n = len(x)
    out = np.full(n, np.nan, dtype=np.float32)

    # Precompute weights
    weights = np.empty(window, dtype=np.float32)
    w = 1.0
    weights[0] = w
    for k in range(1, window):
        w = -w * (d - k + 1) / k
        weights[k] = w

    # Convolve
    # x_tilde_t = sum(w_k * x_{t-k})

    for i in range(window - 1, n):
        val = 0.0
        valid = True
        for k in range(window):
            if np.isnan(x[i-k]):
                valid = False
                break
            val += weights[k] * x[i-k]

        if valid:
            out[i] = val

    return out

def numba_frac_diff(df, d, window):
    tprint(f"Entering function: numba_frac_diff in fast_funcs.py")
    return apply_to_frame(df, _numba_frac_diff_kernel, d, window)
