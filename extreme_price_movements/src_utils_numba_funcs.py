"""
Duplicated numba functions from src/utils/numba_funcs.py
Required for extreme_price_movements/ to be self-contained.
"""

import numpy as np
import math

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    prange = range
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

_EPS = 1e-12

@jit(nopython=True, cache=True, fastmath=True)
def _numba_rolling_mean_nan_safe(x, window):
    """
    Rolling mean ignoring NaNs (O(N) online algorithm).
    """
    n = len(x)
    output = np.full(n, np.nan, dtype=np.float32)

    if window <= 0:
        return output

    current_sum = 0.0
    current_count = 0

    for i in range(n):
        # Entering element
        val_in = x[i]
        if not np.isnan(val_in):
            current_sum += val_in
            current_count += 1

        # Leaving element
        # Window size W means we look at [i-W+1, i].
        # So we drop element at i-W.
        if i >= window:
            val_out = x[i - window]
            if not np.isnan(val_out):
                current_sum -= val_out
                current_count -= 1

        # Output logic
        if current_count > 0:
            output[i] = current_sum / current_count

    return output


@jit(nopython=True, cache=True, fastmath=True)
def _numba_rolling_std_nan_safe(x, window):
    """
    Rolling std ignoring NaNs (O(N) online algorithm).
    Uses sample standard deviation (denom=count-1).
    """
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
            # Var = (SumSq - (Sum^2)/N) / (N-1)
            var_num = sum_sq - (sum_val * sum_val) / count

            # Floating point noise can make var_num slightly negative
            if var_num < 1e-12:
                output[i] = 0.0
            else:
                output[i] = np.sqrt(var_num / (count - 1))

    return output


@jit(nopython=True, cache=True, fastmath=True)
def _numba_rolling_median(x, window):
    """
    Calculate rolling Median using Numba.
    Results are aligned to the right edge of the window.
    First (window-1) elements will be 0.
    """
    n = len(x)
    output = np.zeros(n, dtype=np.float32)

    if n < window:
        return output

    # Fixed off-by-one: range should be (window-1, n) to output at [window-1:]
    for i in range(window - 1, n):
        # Slice: ending at i+1 (exclusive) -> window size
        chunk = x[i - window + 1 : i + 1]

        # Median
        med = np.median(chunk)

        output[i] = med

    return output


@jit(nopython=True, cache=True, fastmath=True)
def _numba_rolling_correlation(x, y, window):
    """
    Rolling correlation Corr(x,y) aligned to the right edge of the window.
    NaN-safe O(N) implementation.
    """
    n = len(x)
    out = np.zeros(n, dtype=np.float32)

    if window <= 0:
        return out
    if len(y) != n:
        return out
    if window == 1:
        return out

    sx = 0.0
    sy = 0.0
    sxx = 0.0
    syy = 0.0
    sxy = 0.0
    count = 0

    for i in range(n):
        vx = x[i]
        vy = y[i]

        # Entering
        if not (np.isnan(vx) or np.isnan(vy)):
            sx += vx
            sy += vy
            sxx += vx * vx
            syy += vy * vy
            sxy += vx * vy
            count += 1

        # Leaving
        if i >= window:
            rx = x[i - window]
            ry = y[i - window]
            if not (np.isnan(rx) or np.isnan(ry)):
                sx -= rx
                sy -= ry
                sxx -= rx * rx
                syy -= ry * ry
                sxy -= rx * ry
                count -= 1

        if i >= window - 1:
            if count < 2:
                out[i] = 0.0
            else:
                inv_c = 1.0 / count
                mx = sx * inv_c
                my = sy * inv_c

                varx = (sxx * inv_c) - (mx * mx)
                vary = (syy * inv_c) - (my * my)
                cov = (sxy * inv_c) - (mx * my)

                if varx <= _EPS or vary <= _EPS:
                    out[i] = 0.0
                else:
                    denom = np.sqrt(varx * vary)
                    if denom <= _EPS:
                        out[i] = 0.0
                    else:
                        out[i] = cov / denom

    return out


@jit(nopython=True, cache=True, fastmath=True)
def _numba_ewma(x, alpha, adjust=False):
    """
    Calculate Exponential Weighted Moving Average (EWMA) using Numba.
    Matches pandas ewm(alpha=alpha, adjust=adjust).mean().
    """
    n = len(x)
    out = np.empty(n, dtype=np.float32)

    if n == 0:
        return out

    # Handle first value (and NaNs at start)
    # Find first valid index
    first_valid_idx = -1
    for i in range(n):
        if not np.isnan(x[i]):
            first_valid_idx = i
            break

    if first_valid_idx == -1:
        # All NaNs
        out[:] = np.nan
        return out

    # Fill NaNs before first valid with NaN
    out[:first_valid_idx] = np.nan

    # Initialize
    if adjust:
        weighted_sum = x[first_valid_idx]
        sum_weights = np.float32(1.0)
        out[first_valid_idx] = weighted_sum / sum_weights

        for i in range(first_valid_idx + 1, n):
            val = x[i]
            if np.isnan(val):
                out[i] = np.nan
                weighted_sum = np.nan
            else:
                weighted_sum = val + (np.float32(1.0) - alpha) * weighted_sum
                sum_weights = np.float32(1.0) + (np.float32(1.0) - alpha) * sum_weights
                out[i] = weighted_sum / sum_weights
    else:
        last_val = x[first_valid_idx]
        out[first_valid_idx] = last_val

        for i in range(first_valid_idx + 1, n):
            val = x[i]
            if np.isnan(val):
                out[i] = np.nan
                last_val = np.nan
            else:
                if np.isnan(last_val):
                    last_val = val
                else:
                    last_val = (1.0 - alpha) * last_val + alpha * val
                out[i] = last_val

    return out


@jit(nopython=True, cache=True, fastmath=True)
def _numba_rolling_vwap(price, volume, window):
    """
    Calculate rolling VWAP using Numba with O(N) complexity.
    Supports min_periods=1 behavior (accumulating from start).
    Handles NaNs correctly (matching Pandas behavior: ignoring NaNs in sums).
    Returns NaN when volume sum is near zero.
    """
    n = len(price)
    out = np.empty(n, dtype=np.float32)
    out[:] = np.nan

    sum_pv = 0.0
    sum_v = 0.0

    for i in range(n):
        p = price[i]
        v = volume[i]

        # Accumulate Volume and PV (both must be valid)
        if not np.isnan(p) and not np.isnan(v):
            sum_v += v
            sum_pv += p * v

        # Remove leaving elements (both must have been valid)
        if i >= window:
            p_old = price[i - window]
            v_old = volume[i - window]

            if not np.isnan(p_old) and not np.isnan(v_old):
                sum_v -= v_old
                sum_pv -= p_old * v_old

        # Compute VWAP
        # Avoid division by zero
        if sum_v > 1e-9:
            out[i] = sum_pv / sum_v
        else:
            out[i] = np.nan

    return out


@jit(nopython=True, cache=True, fastmath=True)
def _numba_rolling_kurt(x, window):
    """
    Calculate rolling kurtosis using Numba with online update algorithm (Sum of Powers).
    Optimized for speed O(N) instead of O(N*W).
    Assumes clean input (no NaNs), or NaNs will propagate.
    """
    n = len(x)
    out = np.empty(n, dtype=np.float32)
    out[:] = np.nan

    if window < 4:
        return out

    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    s4 = 0.0

    for i in range(window):
        val = x[i]
        s1 += val
        s2 += val*val
        s3 += val*val*val
        s4 += val*val*val*val

    w = float(window)

    # Constants for bias correction
    term1 = (w * (w + 1.0)) / ((w - 1.0) * (w - 2.0) * (w - 3.0))
    term2 = (3.0 * (w - 1.0)**2) / ((w - 2.0) * (w - 3.0))

    # Helper logic inlined (Numba inlining is automatic usually)
    mean = s1 / w
    m2 = (s2 / w) - (mean * mean)

    if m2 > 1e-12:
        e4 = s4 / w
        e3 = s3 / w
        e2 = s2 / w
        e1 = mean
        m4 = e4 - 4.0*e1*e3 + 6.0*e1*e1*e2 - 3.0*e1*e1*e1*e1
        kurt_pop = m4 / (m2 * m2)
        val = ((w - 1.0)**2 / w) * kurt_pop
        out[window-1] = term1 * val - term2
    else:
        out[window-1] = 0.0

    for i in range(window, n):
        leaving = x[i-window]
        entering = x[i]

        s1 = s1 - leaving + entering
        s2 = s2 - leaving*leaving + entering*entering
        s3 = s3 - leaving*leaving*leaving + entering*entering*entering
        s4 = s4 - leaving*leaving*leaving*leaving + entering*entering*entering*entering

        mean = s1 / w
        m2 = (s2 / w) - (mean * mean)

        if m2 > 1e-12:
            e4 = s4 / w
            e3 = s3 / w
            e2 = s2 / w
            e1 = mean
            m4 = e4 - 4.0*e1*e3 + 6.0*e1*e1*e2 - 3.0*e1*e1*e1*e1
            kurt_pop = m4 / (m2 * m2)
            val = ((w - 1.0)**2 / w) * kurt_pop
            out[i] = term1 * val - term2
        else:
            out[i] = 0.0

    return out


@jit(nopython=True, cache=True, fastmath=True)
def _numba_rolling_skew(x, window):
    """
    Calculate rolling skewness using Numba with online update algorithm (Sum of Powers).
    Optimized for speed O(N) instead of O(N*W).
    Assumes clean input (no NaNs), or NaNs will propagate.
    """
    n = len(x)
    out = np.empty(n, dtype=np.float32)
    out[:] = np.nan

    if window < 3:
        return out

    s1 = 0.0
    s2 = 0.0
    s3 = 0.0

    # Initialize first window
    for i in range(window):
        val = x[i]
        s1 += val
        s2 += val*val
        s3 += val*val*val

    w = float(window)
    # Compute first output
    mean = s1 / w
    # Variance = E[x^2] - (E[x])^2
    var = (s2 / w) - (mean * mean)

    # m3 = E[x^3] - 3*E[x]*E[x^2] + 2*(E[x])^3
    m3 = (s3 / w) - 3.0 * mean * (s2 / w) + 2.0 * (mean * mean * mean)

    if var > 1e-12:
        stdev = np.sqrt(var)
        pop_skew = m3 / (stdev * stdev * stdev)
        # Bias correction
        adj = np.sqrt(w * (w - 1.0)) / (w - 2.0)
        out[window-1] = adj * pop_skew
    else:
        out[window-1] = 0.0

    # Rolling update
    for i in range(window, n):
        leaving = x[i-window]
        entering = x[i]

        s1 = s1 - leaving + entering
        s2 = s2 - leaving*leaving + entering*entering
        s3 = s3 - leaving*leaving*leaving + entering*entering*entering

        mean = s1 / w
        var = (s2 / w) - (mean * mean)

        if var > 1e-12:
            m3 = (s3 / w) - 3.0 * mean * (s2 / w) + 2.0 * (mean * mean * mean)
            stdev = np.sqrt(var)
            pop_skew = m3 / (stdev * stdev * stdev)

            adj = np.sqrt(w * (w - 1.0)) / (w - 2.0)
            out[i] = adj * pop_skew
        else:
            out[i] = 0.0

    return out


@jit(nopython=True, cache=True, fastmath=True)
def _numba_rolling_slope(y, window):
    """
    Calculate rolling linear regression slope using Numba.
    Slope = (N*sum(xy) - sum(x)*sum(y)) / (N*sum(x^2) - (sum(x))^2)
    where x = [0, 1, ..., window-1]

    Optimized to O(N) using incremental updates:
    S_xy_new = S_xy_old - S_y_old + y_leaving + (W-1)*y_entering
    S_y_new = S_y_old - y_leaving + y_entering
    """
    n = len(y)
    output = np.zeros(n, dtype=np.float32)

    if n < window:
        return output

    # Pre-calculate common terms for x = range(window)
    n_w = float(window)
    sum_x = (n_w * (n_w - 1.0)) / 2.0
    sum_x2 = (n_w * (n_w - 1.0) * (2.0 * n_w - 1.0)) / 6.0
    denom = n_w * sum_x2 - sum_x**2

    if denom == 0:
        return output

    # Initialize first window
    sum_y = 0.0
    sum_xy = 0.0
    for j in range(window):
        val = y[j]
        sum_y += val
        sum_xy += j * val

    output[window - 1] = (n_w * sum_xy - sum_x * sum_y) / denom

    # Loop for remaining windows
    for i in range(window, n):
        y_leaving = y[i - window]
        y_entering = y[i]

        # Update sums incrementally
        sum_xy = sum_xy - sum_y + y_leaving + (n_w - 1) * y_entering
        sum_y = sum_y - y_leaving + y_entering

        output[i] = (n_w * sum_xy - sum_x * sum_y) / denom

    return output


@jit(nopython=True, cache=True, fastmath=True)
def _numba_rolling_rsquared(y, window):
    """
    Calculate rolling R-squared of linear regression y on x (x=0..window-1).
    R^2 = (N*sum(xy) - sum(x)*sum(y))^2 / ((N*sum(x^2) - (sum(x))^2) * (N*sum(y^2) - (sum(y))^2))
    """
    n = len(y)
    output = np.zeros(n, dtype=np.float32)

    if n < window:
        return output

    # Pre-calculate common terms for x = range(window)
    n_w = float(window)
    sum_x = (n_w * (n_w - 1.0)) / 2.0
    sum_x2 = (n_w * (n_w - 1.0) * (2.0 * n_w - 1.0)) / 6.0
    denom_x = n_w * sum_x2 - sum_x**2

    if denom_x <= _EPS:
        return output

    # Initialize first window
    sum_y = 0.0
    sum_y2 = 0.0
    sum_xy = 0.0
    for j in range(window):
        val = y[j]
        sum_y += val
        sum_y2 += val * val
        sum_xy += j * val

    denom_y = n_w * sum_y2 - sum_y**2
    numerator = n_w * sum_xy - sum_x * sum_y

    if denom_y > _EPS:
        output[window - 1] = (numerator * numerator) / (denom_x * denom_y)
    else:
        output[window - 1] = 0.0

    # Loop for remaining windows
    for i in range(window, n):
        y_leaving = y[i - window]
        y_entering = y[i]

        # Update sums incrementally
        sum_xy = sum_xy - sum_y + y_leaving + (n_w - 1) * y_entering
        sum_y = sum_y - y_leaving + y_entering
        sum_y2 = sum_y2 - y_leaving*y_leaving + y_entering*y_entering

        denom_y = n_w * sum_y2 - sum_y**2
        numerator = n_w * sum_xy - sum_x * sum_y

        if denom_y > _EPS:
            output[i] = (numerator * numerator) / (denom_x * denom_y)
        else:
            output[i] = 0.0

    return output


@jit(nopython=True, cache=True, fastmath=True)
def _numba_rolling_sum(x, window):
    """
    Rolling sum aligned to the right edge of the window.
    First (window-1) elements are 0.0.
    """
    n = len(x)
    out = np.zeros(n, dtype=np.float32)

    if window <= 0:
        return out
    if window == 1:
        # Right-aligned sum with window=1 is the value itself
        for i in range(n):
            out[i] = x[i]
        return out

    s = 0.0
    for i in range(n):
        s += x[i]
        if i >= window:
            s -= x[i - window]
        if i >= window - 1:
            out[i] = s
    return out


@jit(nopython=True, cache=True, fastmath=True)
def _numba_rolling_mad(x, window):
    """
    Calculate rolling MAD (Median Absolute Deviation) using Numba.
    MAD = median(|x - median(x)|)
    Results are aligned to the right edge of the window.
    First (window-1) elements will be 0.
    """
    n = len(x)
    output = np.zeros(n, dtype=np.float32)

    if n < window:
        return output

    # Fixed off-by-one: range should be (window-1, n) to output at [window-1:]
    for i in range(window - 1, n):
        # Slice: ending at i+1 (exclusive) -> window size
        chunk = x[i - window + 1 : i + 1]

        # 1. Median
        med = np.median(chunk)

        # 2. Abs Deviations
        devs = np.abs(chunk - med)

        # 3. MAD = median of deviations
        mad = np.median(devs)

        output[i] = mad

    return output

@jit(nopython=True, parallel=True, cache=True)
def _numba_rolling_vwap_parallel(price_mat, volume_mat, window):
    n_rows, n_cols = price_mat.shape
    out = np.empty((n_rows, n_cols), dtype=np.float32)

    for j in prange(n_cols):
        sum_pv = 0.0
        sum_v = 0.0
        for i in range(n_rows):
            out[i, j] = np.nan
            p = price_mat[i, j]
            v = volume_mat[i, j]

            if not np.isnan(p) and not np.isnan(v):
                sum_v += v
                sum_pv += p * v

            if i >= window:
                p_old = price_mat[i - window, j]
                v_old = volume_mat[i - window, j]

                if not np.isnan(p_old) and not np.isnan(v_old):
                    sum_v -= v_old
                    sum_pv -= p_old * v_old

            if sum_v > 1e-9:
                out[i, j] = sum_pv / sum_v

    return out

@jit(nopython=True, parallel=True, cache=True)
def _numba_rolling_bars_since_extreme_parallel(mat, window, mode):
    n_rows, n_cols = mat.shape
    out = np.empty((n_rows, n_cols), dtype=np.float32)

    for j in prange(n_cols):
        for i in range(n_rows):
            start_idx = max(0, i - window + 1)
            length = i - start_idx + 1

            if length == 0:
                out[i, j] = np.nan
                continue

            # Need to find argmax or argmin
            best_idx = 0
            best_val = mat[start_idx, j]

            for k in range(1, length):
                val = mat[start_idx + k, j]
                if mode == 1: # max
                    if val > best_val or np.isnan(best_val):
                        best_val = val
                        best_idx = k
                else: # min
                    if val < best_val or np.isnan(best_val):
                        best_val = val
                        best_idx = k

            # Distance from end of window
            out[i, j] = float(length - 1 - best_idx)

    return out
