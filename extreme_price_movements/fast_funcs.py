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

@jit(nopython=True, cache=True)
def _numba_ewma_nan_safe(x, alpha, adjust=False):
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float32)

    # Find first valid
    first_valid = -1
    for i in range(n):
        if not np.isnan(x[i]):
            first_valid = i
            break

    if first_valid == -1:
        return out

    # Initialize
    out[first_valid] = x[first_valid]

    for i in range(first_valid + 1, n):
        val = x[i]
        if np.isnan(val):
            out[i] = out[i-1]
        else:
            out[i] = (1.0 - alpha) * out[i-1] + alpha * val

    return out

@jit(nopython=True, cache=True)
def _numba_rolling_sum_nan_safe(x, window):
    n = len(x)
    output = np.full(n, np.nan, dtype=np.float32)

    if window <= 0: return output

    current_sum = 0.0
    current_count = 0

    for i in range(n):
        val_in = x[i]
        if not np.isnan(val_in):
            current_sum += val_in
            current_count += 1

        if i >= window:
            val_out = x[i - window]
            if not np.isnan(val_out):
                current_sum -= val_out
                current_count -= 1

        # Require at least 1 valid value to report sum
        if current_count > 0:
            output[i] = current_sum

    return output

@jit(nopython=True, cache=True)
def simulate_trade_numba(
    opens, highs, lows, closes,
    entry_px, side_int,
    sl_dist, activation_dist, trailing_dist
):
    """
    Simulates a trade path using Numba optimization.
    side_int: 1 for long, -1 for short.
    sl_dist, activation_dist, trailing_dist: Absolute price distances.
    Returns: (return_pct, exit_idx_offset, reason_code)
    reason_code: 0=no_entry/error, 1=sl_hit, 2=ambiguous, 3=time_exit
    """
    sl_px = 0.0
    highest_high = entry_px
    lowest_low = entry_px
    trailing_active = False

    if side_int == 1: # Long
        sl_px = entry_px - sl_dist
    else: # Short
        sl_px = entry_px + sl_dist

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
            if profit_dist >= activation_dist:
                trailing_active = True

            if trailing_active:
                trail_dist_px = trailing_dist
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
            if profit_dist >= activation_dist:
                trailing_active = True

            if trailing_active:
                trail_dist_px = trailing_dist
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

                is_active = trailing_active or (profit_dist >= activation_dist)

                if is_active:
                    trail_dist_px = trailing_dist
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
            if profit_dist >= activation_dist:
                trailing_active = True

            if trailing_active:
                trail_dist_px = trailing_dist
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

def apply_to_frame(df: pd.DataFrame, func, *args) -> pd.DataFrame:
    """
    Applies a Numba 1D function to each column of a DataFrame.
    Returns a DataFrame with float32 dtype.
    Handles pd.Series by converting to DataFrame and returning Series.
    """
    # tprint(f"Entering function: apply_to_frame in fast_funcs.py")
    is_series = isinstance(df, pd.Series)
    if is_series:
        df = df.to_frame()

    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=np.float32)
    # We iterate over columns.
    total_cols = len(df.columns)
    for i, col in enumerate(df.columns):
        if i % 100 == 0:
            tprint(f"apply_to_frame progress: {i}/{total_cols} columns processed")
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
    total_cols = len(df1.columns)
    for i, col in enumerate(df1.columns):
        if i % 100 == 0:
            tprint(f"apply_to_frame_binary progress: {i}/{total_cols} columns processed")
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

    avg_up = _numba_ewma_nan_safe(up, alpha, adjust=False)
    avg_dn = _numba_ewma_nan_safe(dn, alpha, adjust=False)

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
    atr = _numba_ewma_nan_safe(tr, 1.0/n, adjust=False)

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

    atr = _numba_ewma_nan_safe(tr, 1.0/n, adjust=False)
    return atr

def numba_atr_no_norm(high_df, low_df, close_df, n):
    # tprint(f"Entering function: numba_atr_no_norm in fast_funcs.py")
    out = pd.DataFrame(index=close_df.index, columns=close_df.columns, dtype=np.float32)
    cols = close_df.columns
    total_cols = len(cols)
    for i, c in enumerate(cols):
        if i % 100 == 0:
            tprint(f"numba_atr_no_norm progress: {i}/{total_cols} columns processed")
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
    total_cols = len(cols)
    for i, c in enumerate(cols):
        if i % 100 == 0:
            tprint(f"numba_atr progress: {i}/{total_cols} columns processed")
        h = high_df[c].to_numpy(dtype=np.float32)
        l = low_df[c].to_numpy(dtype=np.float32)
        cl = close_df[c].to_numpy(dtype=np.float32)
        res = numba_atr_kernel(h, l, cl, n)
        out[c] = res
    return out

def numba_zscore(df, n):
    # (x - mean) / std
    # Using nan_safe versions
    # tprint(f"Entering function: numba_zscore in fast_funcs.py")
    mu = apply_to_frame(df, _numba_rolling_mean_nan_safe, n)
    sd = apply_to_frame(df, _numba_rolling_std_nan_safe, n)

    # Vectorized pandas operation for final step is fine/fast
    return (df - mu) / (sd + 1e-12)

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

    total_cols = len(df.columns)
    for i, col in enumerate(df.columns):
        if i % 100 == 0:
            tprint(f"numba_grouped_rolling_mean progress: {i}/{total_cols} columns processed")
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

@jit(nopython=True, cache=True)
def _numba_peak_label_and_weight(close, atr, horizon, near_k, rev_k, is_uptrend, max_near_pct, min_rev_pct):
    """
    Computes "Peak Proximity" label and sample weights.
    Returns: (labels, weights)

    Weight = log(1 + Y / (X^2 * ATR))
    X: Proximity (abs diff price to peak)
    Y: Reversal size (abs diff peak to subsequent extremum)
    ATR: Rolling ATR at t

    If label=0, weight=1.0.
    """
    n = len(close)
    labels = np.zeros(n, dtype=np.float32)
    weights = np.ones(n, dtype=np.float32)

    # We iterate up to n - horizon to ensure we have full lookahead
    # Last H bars will be 0 (unlabeled)
    for i in range(n - horizon):
        curr_p = close[i]
        curr_atr = atr[i]

        # Valid ATR check
        if np.isnan(curr_atr) or curr_atr <= 0 or np.isnan(curr_p):
            continue

        # Calculate base distances
        raw_limit = rev_k * curr_atr
        raw_near = near_k * curr_atr

        # Apply Clipping
        # near_dist = min(ATR-based, Cap)
        near_dist = min(raw_near, max_near_pct * curr_p)

        # limit_dist = max(ATR-based, Floor)
        limit_dist = max(raw_limit, min_rev_pct * curr_p)

        # Define search window: (i+1, i+1+horizon)
        start_search = i + 1
        end_search = min(n, i + 1 + horizon)

        if start_search >= end_search:
            continue

        if is_uptrend: # Looking for Peak (Top)
            # 1. Find Forward Max (Peak)
            fwd_max = -np.inf
            idx_max = -1

            for j in range(start_search, end_search):
                val = close[j]
                if not np.isnan(val):
                    if val > fwd_max:
                        fwd_max = val
                        idx_max = j

            if idx_max == -1: continue

            # 2. Check Peak Proximity
            if curr_p < (fwd_max - near_dist):
                continue

            # 3. Check Reversal AFTER Peak
            # We need a drawdown from fwd_max of size limit_dist
            # The drawdown must happen in [idx_max + 1, end_search)
            # If peak is the last bar (idx_max == end_search-1), no reversal can be checked
            if idx_max >= end_search - 1:
                continue

            # Find Max Reversal (Min Price after Peak)
            fwd_min_after = np.inf

            for j in range(idx_max + 1, end_search):
                val = close[j]
                if not np.isnan(val):
                    if val < fwd_min_after:
                        fwd_min_after = val

            if fwd_min_after == np.inf: continue

            # Check if reversal is big enough
            reversal_size = fwd_max - fwd_min_after
            if reversal_size >= limit_dist:
                labels[i] = 1.0

                # Weight Calculation
                # X = distance to peak = fwd_max - curr_p
                # Y = reversal size = reversal_size
                # ATR = curr_atr
                # W = log(1 + Y / (X^2 * ATR)) -> No, user said Y/(X^2 * ATR)
                # Wait, units.
                # Y is price diff. X is price diff. ATR is price diff.
                # X^2 is price^2.
                # Y / (X^2 * ATR) -> price / (price^3) = 1/price^2.
                # This seems dimensionally weird if not normalized?
                # User: "Ensure Y is defined in price units OR drop ATR normalisation if Y is already in ATR units"
                # If X, Y, ATR are all price units:
                # X ~ 100. X^2 ~ 10000. ATR ~ 100.
                # Y ~ 200.
                # Y / (X^2 * ATR) ~ 200 / (10000 * 100) ~ small.
                # Maybe user meant X in ATR units?
                # "timing X² matters more"
                # If X is small (near 0), X^2 is very small. Denom small -> Weight huge.
                # If X is in price units (e.g. 1.0), X^2 = 1.0.
                # Let's assume user wants raw values but we should protect against div/0.

                X_val = fwd_max - curr_p
                Y_val = reversal_size

                # Protect X=0
                X_safe = max(X_val, 1e-4)

                term = Y_val / ((X_safe**2) * curr_atr)
                weights[i] = np.log1p(term)

        else: # Looking for Trough (Bottom)
            # 1. Find Forward Min (Trough)
            fwd_min = np.inf
            idx_min = -1

            for j in range(start_search, end_search):
                val = close[j]
                if not np.isnan(val):
                    if val < fwd_min:
                        fwd_min = val
                        idx_min = j

            if idx_min == -1: continue

            # 2. Check Trough Proximity
            if curr_p > (fwd_min + near_dist):
                continue

            # 3. Check Rally AFTER Trough
            if idx_min >= end_search - 1:
                continue

            # Find Max Reversal (Max Price after Trough)
            fwd_max_after = -np.inf
            for j in range(idx_min + 1, end_search):
                val = close[j]
                if not np.isnan(val):
                    if val > fwd_max_after:
                        fwd_max_after = val

            if fwd_max_after == -np.inf: continue

            reversal_size = fwd_max_after - fwd_min
            if reversal_size >= limit_dist:
                labels[i] = 1.0

                # Weight Calculation
                # X = distance to trough = curr_p - fwd_min
                X_val = curr_p - fwd_min
                Y_val = reversal_size

                X_safe = max(X_val, 1e-4)
                term = Y_val / ((X_safe**2) * curr_atr)
                weights[i] = np.log1p(term)

    return labels, weights

# Wrappers
def numba_rolling_max(df, n):
    tprint(f"Entering function: numba_rolling_max in fast_funcs.py")
    return apply_to_frame(df, _numba_rolling_max, n)

def numba_rolling_min(df, n):
    tprint(f"Entering function: numba_rolling_min in fast_funcs.py")
    return apply_to_frame(df, _numba_rolling_min, n)

def numba_rolling_sum(df, n):
    tprint(f"Entering function: numba_rolling_sum in fast_funcs.py")
    # CHANGED: Use NaN-safe version
    return apply_to_frame(df, _numba_rolling_sum_nan_safe, n)

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

def compute_peak_labels_and_weights(close_df, atr_df, horizon, near_k, rev_k, is_uptrend, max_near_pct=0.02, min_rev_pct=0.005):
    tprint(f"Entering function: compute_peak_labels_and_weights in fast_funcs.py")

    # We need a custom apply because we return TWO frames (labels, weights)
    # The existing apply_to_frame_binary returns one.
    # We'll implement a custom loop here since it's cleaner than modifying the generic helper.

    cols = close_df.columns
    idx = close_df.index

    l_out = pd.DataFrame(index=idx, columns=cols, dtype=np.float32)
    w_out = pd.DataFrame(index=idx, columns=cols, dtype=np.float32)

    total_cols = len(cols)
    for i, c in enumerate(cols):
        if i % 100 == 0:
            tprint(f"compute_peak_labels_and_weights progress: {i}/{total_cols} columns processed")
        if c not in atr_df.columns: continue

        c_arr = close_df[c].to_numpy(dtype=np.float32)
        a_arr = atr_df[c].to_numpy(dtype=np.float32)

        l_arr, w_arr = _numba_peak_label_and_weight(
            c_arr, a_arr,
            horizon, near_k, rev_k, is_uptrend,
            max_near_pct, min_rev_pct
        )

        l_out[c] = l_arr
        w_out[c] = w_arr

    return l_out, w_out
  
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
