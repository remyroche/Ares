import numpy as np
import pandas as pd
from numba import jit, prange

# TODO: DECOUPLE src dependencies for standalone module
# These functions should be implemented locally or vendored into this module
# Currently used: _numba_rolling_mean_nan_safe, _numba_rolling_std_nan_safe, 
#                 _numba_rolling_median, _numba_rolling_correlation
# Unused imports: _numba_ewma, _numba_rolling_vwap, _numba_rolling_kurt,
#                 _numba_rolling_skew, _numba_rolling_slope, _numba_rolling_rsquared,
#                 _numba_rolling_sum (have local _numba_rolling_sum_nan_safe)
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
    one = np.float32(1.0)

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
            out[i] = (one - alpha) * out[i-1] + alpha * val

    return out

@jit(nopython=True, cache=True)
def _numba_rolling_sum_nan_safe(x, window):
    n = len(x)
    output = np.full(n, np.nan, dtype=np.float32)

    if window <= 0: return output

    current_sum = np.float32(0.0)
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
    # Use float32 constants
    one = np.float32(1.0)
    zero = np.float32(0.0)

    sl_px = zero
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

        if np.isnan(curr_h) or np.isnan(curr_l):
            continue

        if side_int == 1: # Long
            # 1. Update Extrema
            if curr_h > highest_high:
                highest_high = curr_h

            # 2. Check Activation
            if not trailing_active:
                if (highest_high - entry_px) >= activation_dist:
                    trailing_active = True

            # 3. Check Hits
            stop_hit = curr_l <= sl_px
            trail_moved = False
            new_sl_val = sl_px

            if trailing_active:
                # Trailing stop based on Highest High
                potential_new_sl = highest_high - trailing_dist
                if potential_new_sl > sl_px:
                    trail_moved = True
                    new_sl_val = potential_new_sl

            # Ambiguity Check: Hit Fixed Stop AND Trailing Stop moved UP in same bar
            # (implying we might have hit fixed stop before trailing moved, or trailing moved then we hit it?)
            # Conservative: If ambiguous, return special code.
            if stop_hit and trail_moved:
                return zero, i, 2 # Ambiguous

            if stop_hit:
                return (sl_px / entry_px) - one, i, 1 # SL Hit (Fixed)

            # Update SL for next bar (or if we decide to exit intrabar on trailing?)
            # Original code exits on 'stop_hit' which is against 'sl_px' (Start of Bar).
            # It updates 'sl_px' at end.
            # So effectively, trailing stop applies from NEXT bar.
            if trail_moved:
                sl_px = new_sl_val

        else: # Short
            # 1. Update Extrema
            if curr_l < lowest_low:
                lowest_low = curr_l

            # 2. Check Activation
            if not trailing_active:
                if (entry_px - lowest_low) >= activation_dist:
                    trailing_active = True

            # 3. Check Hits
            stop_hit = curr_h >= sl_px
            trail_moved = False
            new_sl_val = sl_px

            if trailing_active:
                potential_new_sl = lowest_low + trailing_dist
                if potential_new_sl < sl_px:
                    trail_moved = True
                    new_sl_val = potential_new_sl

            if stop_hit and trail_moved:
                return zero, i, 2 # Ambiguous

            if stop_hit:
                return (entry_px / sl_px) - one, i, 1 # SL Hit

            if trail_moved:
                sl_px = new_sl_val

    # Time Exit
    last_c = closes[n-1]
    if side_int == 1:
        ret = (last_c / entry_px) - one
    else:
        ret = (entry_px / last_c) - one

    return ret, n-1, 3 # Time Exit

def apply_to_matrix(df: pd.DataFrame, func, *args) -> pd.DataFrame:
    """
    Applies a Numba 1D function to each column of a DataFrame.
    Returns a DataFrame with float32 dtype.
    Handles pd.Series by converting to DataFrame and returning Series.
    Optimized to use 2D numpy array iteration to avoid Pandas overhead per column.
    """
    is_series = isinstance(df, pd.Series)
    if is_series:
        df = df.to_frame()

    # Convert to numpy float32 array (copy=False if possible)
    mat = df.to_numpy(dtype=np.float32, copy=False)
    n_rows, n_cols = mat.shape

    # Allocate output array
    out = np.empty((n_rows, n_cols), dtype=np.float32)

    # Apply function to each column
    for j in range(n_cols):
        out[:, j] = func(mat[:, j], *args)

    res_df = pd.DataFrame(out, index=df.index, columns=df.columns)

    if is_series:
        return res_df[res_df.columns[0]]

    return res_df

@jit(nopython=True, cache=True)
def _numba_rolling_robust_zscore_1d(x, window, quantile, eps):
    n = len(x)
    output = np.full(n, np.nan, dtype=np.float32)
    buf = np.empty(window, dtype=np.float32)
    devs_buf = np.empty(window, dtype=np.float32)

    if window <= 0: return output

    for i in range(n):
        # Window bounds
        start = max(0, i - window + 1)
        end = i + 1
        count = 0

        # Populate buffer
        for j in range(start, end):
            v = x[j]
            if not np.isnan(v):
                buf[count] = v
                count += 1

        if count < 10:
            continue

        valid_buf = buf[:count]

        # 1. Base Anchor
        idx_q = int(round(quantile * (count - 1)))
        # partition reorders valid_buf in-place
        part = np.partition(valid_buf, idx_q)
        base_t = part[idx_q]

        # 2. Scale (MAD)
        # Median of x
        idx_med = int((count - 1) // 2)
        part_med = np.partition(valid_buf, idx_med)
        median_val = part_med[idx_med]

        # Deviations
        for k in range(count):
            devs_buf[k] = abs(valid_buf[k] - median_val)

        # Median Absolute Deviation
        part_dev = np.partition(devs_buf[:count], idx_med)
        mad = part_dev[idx_med]

        scale_t = np.float32(1.4826) * mad + eps

        # 3. Z-score
        val_in = x[i]
        if not np.isnan(val_in):
            output[i] = (val_in - base_t) / scale_t

    return output

@jit(nopython=True, parallel=True, cache=True)
def _numba_rolling_robust_zscore_parallel(mat, window, quantile, eps):
    n_rows, n_cols = mat.shape
    out = np.empty((n_rows, n_cols), dtype=np.float32)

    for j in prange(n_cols):
        out[:, j] = _numba_rolling_robust_zscore_1d(mat[:, j], window, quantile, eps)

    return out

def numba_rolling_robust_zscore(df, window, quantile=0.45, eps=1e-12):
    tprint(f"Entering function: numba_rolling_robust_zscore in fast_funcs.py")
    is_series = isinstance(df, pd.Series)
    if is_series:
        df = df.to_frame()

    mat = df.to_numpy(dtype=np.float32, copy=False)
    res = _numba_rolling_robust_zscore_parallel(mat, window, quantile, eps)

    res_df = pd.DataFrame(res, index=df.index, columns=df.columns)

    if is_series:
        return res_df[res_df.columns[0]]

    return res_df

def apply_to_frame(df: pd.DataFrame, func, *args) -> pd.DataFrame:
    """
    Applies a Numba 1D function to each column of a DataFrame.
    Returns a DataFrame with float32 dtype.
    Handles pd.Series by converting to DataFrame and returning Series.
    """
    # tprint(f"Entering function: apply_to_frame in fast_funcs.py")
    return apply_to_matrix(df, func, *args)

def apply_to_matrix_binary(df1: pd.DataFrame, df2: pd.DataFrame, func, *args) -> pd.DataFrame:
    """
    Applies a function taking two arrays (col1, col2) -> out_array.
    Assumes df1 and df2 have same columns and index.
    Handles pd.Series inputs.
    """
    is_series1 = isinstance(df1, pd.Series)
    is_series2 = isinstance(df2, pd.Series)

    if is_series1:
        df1 = df1.to_frame()
    if is_series2:
        df2 = df2.to_frame()

    # Check if columns match exactly for fast path
    if df1.columns.equals(df2.columns):
        m1 = df1.to_numpy(dtype=np.float32, copy=False)
        m2 = df2.to_numpy(dtype=np.float32, copy=False)
        n_rows, n_cols = m1.shape
        out = np.empty((n_rows, n_cols), dtype=np.float32)

        for j in range(n_cols):
            out[:, j] = func(m1[:, j], m2[:, j], *args)

        res_df = pd.DataFrame(out, index=df1.index, columns=df1.columns)
        if is_series1: return res_df[res_df.columns[0]]
        return res_df

    # Fallback / Intersection path
    common = df1.columns.intersection(df2.columns)
    if len(common) == 0:
        out_df = pd.DataFrame(index=df1.index, columns=df1.columns, dtype=np.float32)
        out_df[:] = np.nan
        if is_series1: return out_df[out_df.columns[0]]
        return out_df

    m1 = df1[common].to_numpy(dtype=np.float32, copy=False)
    m2 = df2[common].to_numpy(dtype=np.float32, copy=False)

    n_rows, n_cols = m1.shape
    out_mat = np.empty((n_rows, n_cols), dtype=np.float32)

    for j in range(n_cols):
        out_mat[:, j] = func(m1[:, j], m2[:, j], *args)

    # Construct result with full columns (NaNs where missing)
    res_common = pd.DataFrame(out_mat, index=df1.index, columns=common)
    out_df = res_common.reindex(columns=df1.columns) # NaNs for others

    if is_series1:
        return out_df[out_df.columns[0]]

    return out_df

def apply_to_frame_binary(df1: pd.DataFrame, df2: pd.DataFrame, func, *args) -> pd.DataFrame:
    """
    Applies a function taking two arrays (col1, col2) -> out_array.
    Assumes df1 and df2 have same columns and index.
    Handles pd.Series inputs.
    """
    tprint(f"Entering function: apply_to_frame_binary in fast_funcs.py")
    return apply_to_matrix_binary(df1, df2, func, *args)

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
    alpha = np.float32(1.0) / n

    avg_up = _numba_ewma_nan_safe(up, alpha, adjust=False)
    avg_dn = _numba_ewma_nan_safe(dn, alpha, adjust=False)

    out = np.empty_like(close)
    c100 = np.float32(100.0)
    c50 = np.float32(50.0)
    c1 = np.float32(1.0)

    for i in range(len(close)):
        if np.isnan(avg_dn[i]) or avg_dn[i] == 0:
            if np.isnan(avg_up[i]):
                out[i] = np.nan
            elif avg_up[i] == 0:
                 out[i] = c50 # No move
            else:
                 out[i] = c100 # Only up moves
        else:
            rs = avg_up[i] / avg_dn[i]
            out[i] = c100 - (c100 / (c1 + rs))

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
    atr = _numba_ewma_nan_safe(tr, np.float32(1.0)/n, adjust=False)

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

    atr = _numba_ewma_nan_safe(tr, np.float32(1.0)/n, adjust=False)
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
    return (df - mu) / (sd + np.float32(1e-12))

# --- NEW KERNELS & WRAPPERS ---

@jit(nopython=True, cache=True)
def _numba_rolling_max(x, window):
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float32)

    # Deque stores indices of potential max candidates
    # Invariant: elements in deque are in decreasing order of value at those indices
    # We maintain a deque of indices
    # OPTIMIZATION: Use circular buffer of size 'window' instead of 'n' to save memory
    deque_indices = np.empty(window, dtype=np.int32)
    front = 0
    back = -1

    for i in range(n):
        val = x[i]

        # 1. Clean deque from front: remove indices that are out of window
        lower_bound = i - window + 1
        while front <= back:
            idx = deque_indices[front % window]
            if idx < lower_bound:
                front += 1
            else:
                break

        # 2. Add current element if it is valid (not NaN)
        if not np.isnan(val):
            # Maintain decreasing property: remove elements from back that are smaller than val
            while front <= back:
                idx = deque_indices[back % window]
                # x[idx] is guaranteed to be valid because we only add valid indices
                if x[idx] <= val:
                    back -= 1
                else:
                    break

            # Push current index
            back += 1
            deque_indices[back % window] = i

        # 3. Report max
        if front <= back:
            out[i] = x[deque_indices[front % window]]

    return out

@jit(nopython=True, cache=True)
def _numba_rolling_min(x, window):
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float32)

    # Deque stores indices of potential min candidates
    # Invariant: elements in deque are in increasing order of value at those indices
    # OPTIMIZATION: Use circular buffer of size 'window' instead of 'n' to save memory
    deque_indices = np.empty(window, dtype=np.int32)
    front = 0
    back = -1

    for i in range(n):
        val = x[i]

        # 1. Clean deque from front: remove indices that are out of window
        lower_bound = i - window + 1
        while front <= back:
            idx = deque_indices[front % window]
            if idx < lower_bound:
                front += 1
            else:
                break

        # 2. Add current element if it is valid (not NaN)
        if not np.isnan(val):
            # Maintain increasing property: remove elements from back that are larger than val
            while front <= back:
                idx = deque_indices[back % window]
                if x[idx] >= val:
                    back -= 1
                else:
                    break

            # Push current index
            back += 1
            deque_indices[back % window] = i

        # 3. Report min
        if front <= back:
            out[i] = x[deque_indices[front % window]]

    return out

@jit(nopython=True, cache=True)
def _numba_rolling_quantile(x, window, q):
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float32)

    # Pre-allocate buffer to avoid N allocations
    buf = np.empty(window, dtype=np.float32)

    for i in range(n):
        start = max(0, i - window + 1)
        end = i + 1

        # Count valid
        count = 0
        for j in range(start, end):
            val = x[j]
            if not np.isnan(val):
                buf[count] = val
                count += 1

        if count == 0:
            continue

        # Use valid slice
        valid_buf = buf[:count]

        # Linear Interpolation
        v_idx = q * (count - 1)
        i_lower = int(np.floor(v_idx))
        i_upper = int(np.ceil(v_idx))
        fraction = v_idx - i_lower

        # Optimization: Use np.partition instead of sort
        # O(W) instead of O(W log W)

        # Partition at i_upper puts the correct element at i_upper
        # and all smaller/equal elements to the left.
        # This allows finding i_lower (which is <= i_upper) efficiently.
        part = np.partition(valid_buf, i_upper)
        val_upper = part[i_upper]

        if i_lower == i_upper:
            out[i] = val_upper
        else:
            # i_lower is in part[:i_upper]
            # Since partitioned, we just need the max of the left side
            val_lower = np.max(part[:i_upper])
            out[i] = val_lower + (val_upper - val_lower) * fraction

    return out

@jit(nopython=True, cache=True)
def _numba_rolling_quantile_dual_1d(x, window, q1, q2, out1, out2):
    n = len(x)
    # out1, out2 passed in to avoid allocation

    # Initialize with NaN
    out1[:] = np.nan
    out2[:] = np.nan

    # Pre-allocate buffer
    buf = np.empty(window, dtype=np.float32)

    for i in range(n):
        start = max(0, i - window + 1)
        end = i + 1

        # Count valid
        count = 0
        for j in range(start, end):
            val = x[j]
            if not np.isnan(val):
                buf[count] = val
                count += 1

        if count == 0:
            continue

        valid_buf = buf[:count]

        # Calculate Indices
        v_idx1 = q1 * (count - 1)
        i_lower1 = int(np.floor(v_idx1))
        i_upper1 = int(np.ceil(v_idx1))
        frac1 = v_idx1 - i_lower1

        v_idx2 = q2 * (count - 1)
        i_lower2 = int(np.floor(v_idx2))
        i_upper2 = int(np.ceil(v_idx2))
        frac2 = v_idx2 - i_lower2

        # Optimization: Sequential Partitioning
        # We assume q1 <= q2, so i_upper1 <= i_upper2.
        # Logic handles generic case.

        if i_upper2 >= i_upper1:
            # Partition at Upper 2
            part2 = np.partition(valid_buf, i_upper2)
            val_upper2 = part2[i_upper2]

            if i_lower2 == i_upper2:
                val_lower2 = val_upper2
            else:
                val_lower2 = np.max(part2[:i_upper2])

            out2[i] = val_lower2 + (val_upper2 - val_lower2) * frac2

            if i_upper1 < i_upper2:
                # Partition again for Q1.
                # Avoid slicing part2 which might incur copy.
                # Instead, partition part2 (which is valid_buf reordered).
                part1 = np.partition(part2, i_upper1)
                val_upper1 = part1[i_upper1]

                if i_lower1 == i_upper1:
                    val_lower1 = val_upper1
                else:
                    val_lower1 = np.max(part1[:i_upper1])

                out1[i] = val_lower1 + (val_upper1 - val_lower1) * frac1

            else:
                # i_upper1 == i_upper2 (Collision)
                # Recompute using values we already found, but use frac1
                val_upper1 = val_upper2
                val_lower1 = val_lower2
                out1[i] = val_lower1 + (val_upper1 - val_lower1) * frac1

        else:
            # q1 > q2 (Unlikely path)
            part1 = np.partition(valid_buf, i_upper1)
            val_upper1 = part1[i_upper1]

            if i_lower1 == i_upper1:
                val_lower1 = val_upper1
            else:
                val_lower1 = np.max(part1[:i_upper1])
            out1[i] = val_lower1 + (val_upper1 - val_lower1) * frac1

            if i_upper2 < i_upper1:
                part2 = np.partition(part1, i_upper2)
                val_upper2 = part2[i_upper2]
                if i_lower2 == i_upper2:
                    val_lower2 = val_upper2
                else:
                    val_lower2 = np.max(part2[:i_upper2])
                out2[i] = val_lower2 + (val_upper2 - val_lower2) * frac2
            else:
                val_upper2 = val_upper1
                val_lower2 = val_lower1
                out2[i] = val_lower2 + (val_upper2 - val_lower2) * frac2

    # Return not strictly needed as in-place, but good for consistency/checking
    return out1, out2

@jit(nopython=True, parallel=True, cache=True)
def _numba_rolling_quantile_dual_parallel(mat, window, q1, q2):
    n_rows, n_cols = mat.shape
    out1 = np.empty((n_rows, n_cols), dtype=np.float32)
    out2 = np.empty((n_rows, n_cols), dtype=np.float32)

    for j in prange(n_cols):
        # Pass slices to write in-place, avoiding internal allocation
        _numba_rolling_quantile_dual_1d(mat[:, j], window, q1, q2, out1[:, j], out2[:, j])

    return out1, out2

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

@jit(nopython=True, cache=True)
def _numba_grouped_rolling_mean_gather_kernel(col_vals, indices, window, out_vals):
    # Computes rolling mean of col_vals[indices] and stores in out_vals[indices]

    n = len(indices)
    # Use float64 for accumulation to match original implementation precision
    current_sum = 0.0
    current_count = 0

    if window <= 0:
        return

    for i in range(n):
        idx_curr = indices[i]
        val_in = col_vals[idx_curr]

        if not np.isnan(val_in):
            current_sum += val_in
            current_count += 1

        if i >= window:
            idx_out = indices[i - window]
            val_out = col_vals[idx_out]
            if not np.isnan(val_out):
                current_sum -= val_out
                current_count -= 1

        if current_count > 0:
            out_vals[idx_curr] = current_sum / current_count
        else:
            out_vals[idx_curr] = np.nan

def numba_grouped_rolling_mean(df: pd.DataFrame, group_series: pd.Series, window: int) -> pd.DataFrame:
    """
    Vectorized grouped rolling mean.
    Optimized to use pre-computed indices per group to avoid repeated masking.
    Uses gather kernel to avoid subset allocation.
    """
    tprint(f"Entering function: numba_grouped_rolling_mean in fast_funcs.py")
    is_series = isinstance(df, pd.Series)
    if is_series:
        df = df.to_frame()

    # Align groups
    groups_arr = group_series.reindex(df.index).to_numpy()

    # Pre-compute indices for each group to avoid repeated O(N) scanning
    unique_groups = np.unique(groups_arr)
    # Filter NaNs from keys
    unique_groups = unique_groups[~np.isnan(unique_groups)]

    group_indices = {}
    for g in unique_groups:
        group_indices[g] = np.where(groups_arr == g)[0]

    # Use Matrix approach
    mat = df.to_numpy(dtype=np.float32, copy=False)
    n_rows, n_cols = mat.shape
    out_mat = np.full((n_rows, n_cols), np.nan, dtype=np.float32)

    # Iterating columns
    for j in range(n_cols):
        col_vals = mat[:, j]
        for g in unique_groups:
            indices = group_indices[g]
            if len(indices) == 0: continue

            # Use optimized gather kernel (avoids subset copy + output allocation)
            _numba_grouped_rolling_mean_gather_kernel(col_vals, indices, window, out_mat[:, j])

    res_df = pd.DataFrame(out_mat, index=df.index, columns=df.columns)

    if is_series:
        return res_df[res_df.columns[0]]

    return res_df

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
    # tprint(f"Entering function: numba_rolling_quantile in fast_funcs.py")
    return apply_to_frame(df, _numba_rolling_quantile, n, q)

def numba_rolling_quantile_dual(df, n, q1, q2):
    # tprint(f"Entering function: numba_rolling_quantile_dual in fast_funcs.py")
    is_series = isinstance(df, pd.Series)
    if is_series:
        df = df.to_frame()

    # Convert to 2D numpy array (float32)
    # We assume uniform float32 data as per features.py
    mat = df.to_numpy(dtype=np.float32)

    res1, res2 = _numba_rolling_quantile_dual_parallel(mat, n, q1, q2)

    out1 = pd.DataFrame(res1, index=df.index, columns=df.columns)
    out2 = pd.DataFrame(res2, index=df.index, columns=df.columns)

    if is_series:
        return out1[out1.columns[0]], out2[out2.columns[0]]

    return out1, out2

def numba_pct_change(df, n):
    tprint(f"Entering function: numba_pct_change in fast_funcs.py")
    return apply_to_frame(df, _numba_pct_change, n)

def numba_rolling_corr(df1, df2, n):
    tprint(f"Entering function: numba_rolling_corr in fast_funcs.py")
    return apply_to_frame_binary(df1, df2, _numba_rolling_correlation, n)

def numba_rolling_mean(df, n):
    # tprint(f"Entering function: numba_rolling_mean in fast_funcs.py")
    return apply_to_frame(df, _numba_rolling_mean_nan_safe, n)

def numba_rolling_std(df, n):
    # tprint(f"Entering function: numba_rolling_std in fast_funcs.py")
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
def _numba_frac_diff_kernel(x, d, window, thres):
    # Fixed Width Window Frac Diff
    # w_k = -w_{k-1} * (d - k + 1) / k
    # w_0 = 1

    n = len(x)
    out = np.full(n, np.nan, dtype=np.float32)

    # Precompute weights
    weights = np.empty(window, dtype=np.float32)
    w = np.float32(1.0)
    weights[0] = w
    effective_k = 1

    for k in range(1, window):
        w = -w * (d - k + 1) / k
        weights[k] = w
        effective_k = k + 1
        if abs(w) < thres:
            break

    # Convolve
    # x_tilde_t = sum(w_k * x_{t-k})

    for i in range(effective_k - 1, n):
        val = np.float32(0.0)
        valid = True
        for k in range(effective_k):
            if np.isnan(x[i-k]):
                valid = False
                break
            val += weights[k] * x[i-k]

        if valid:
            out[i] = val

    return out

def numba_frac_diff(df, d, window, thres=1e-5):
    tprint(f"Entering function: numba_frac_diff in fast_funcs.py")
    return apply_to_matrix(df, _numba_frac_diff_kernel, d, window, thres)

@jit(nopython=True, cache=True)
def _numba_rolling_zscore_nan_safe_1d(x, window, eps=1e-12):
    n = len(x)
    output = np.full(n, np.nan, dtype=np.float32)

    if window <= 0:
        return output

    sum_val = 0.0
    sum_sq = 0.0
    count = 0

    for i in range(n):
        # Entering
        val_in = x[i]
        if not np.isnan(val_in):
            sum_val += val_in
            sum_sq += val_in * val_in
            count += 1

        # Leaving
        if i >= window:
            val_out = x[i - window]
            if not np.isnan(val_out):
                sum_val -= val_out
                sum_sq -= val_out * val_out
                count -= 1

        # Output logic
        if count > 1:
            mean = sum_val / count
            # Var = (SumSq - (Sum^2)/N) / (N-1)
            var_num = sum_sq - (sum_val * sum_val) / count

            if var_num < 0: var_num = 0.0

            std = np.sqrt(var_num / (count - 1))

            if not np.isnan(val_in):
                 output[i] = (val_in - mean) / (std + eps)
            else:
                 output[i] = np.nan

        elif count == 1:
             # Std is undefined (0 or NaN) for N=1 depending on definition.
             output[i] = np.nan
        else:
             output[i] = np.nan

    return output

@jit(nopython=True, cache=True)
def _numba_causal_clip_with_ffill_1d(x, lo, hi):
    n = len(x)
    out = np.empty(n, dtype=np.float32)

    last_lo = np.nan
    last_hi = np.nan

    for i in range(n):
        # Update limits if valid
        l = lo[i]
        h = hi[i]

        if not np.isnan(l):
            last_lo = l

        if not np.isnan(h):
            last_hi = h

        val = x[i]
        if np.isnan(val):
            out[i] = np.nan
            continue

        res = val

        # Apply limits if we have them (current or carried forward)
        if not np.isnan(last_lo):
            if res < last_lo: res = last_lo

        if not np.isnan(last_hi):
            if res > last_hi: res = last_hi

        out[i] = res

    return out

@jit(nopython=True, parallel=True, cache=True)
def _numba_rolling_zscore_parallel(mat, window, eps=1e-12):
    n_rows, n_cols = mat.shape
    out = np.empty((n_rows, n_cols), dtype=np.float32)

    for j in prange(n_cols):
        out[:, j] = _numba_rolling_zscore_nan_safe_1d(mat[:, j], window, eps)

    return out

@jit(nopython=True, parallel=True, cache=True)
def _numba_causal_clip_parallel(mat, lo_mat, hi_mat):
    n_rows, n_cols = mat.shape
    out = np.empty((n_rows, n_cols), dtype=np.float32)

    for j in prange(n_cols):
        out[:, j] = _numba_causal_clip_with_ffill_1d(mat[:, j], lo_mat[:, j], hi_mat[:, j])

    return out

def numba_rolling_zscore_fused(df, window, eps=1e-12):
    is_series = isinstance(df, pd.Series)
    if is_series:
        df = df.to_frame()

    mat = df.to_numpy(dtype=np.float32, copy=False)

    res = _numba_rolling_zscore_parallel(mat, window, eps)

    res_df = pd.DataFrame(res, index=df.index, columns=df.columns)

    if is_series:
        return res_df[res_df.columns[0]]

    return res_df

def numba_causal_clip(df, lo, hi):
    # lo, hi should be same shape as df
    is_series = isinstance(df, pd.Series)
    if is_series:
        df = df.to_frame()
        lo = lo.to_frame()
        hi = hi.to_frame()

    mat = df.to_numpy(dtype=np.float32, copy=False)
    lo_mat = lo.to_numpy(dtype=np.float32, copy=False)
    hi_mat = hi.to_numpy(dtype=np.float32, copy=False)

    res = _numba_causal_clip_parallel(mat, lo_mat, hi_mat)

    res_df = pd.DataFrame(res, index=df.index, columns=df.columns)

    if is_series:
        return res_df[res_df.columns[0]]

    return res_df
