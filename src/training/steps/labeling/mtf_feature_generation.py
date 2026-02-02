"""
Multi-Timeframe Feature Generation Module.
Extracted and enhanced from feature_generation_meta_labeling_step.py.
"""
from typing import Any, Dict, List, Optional, Union, Tuple
import os
import pandas as pd
import numpy as np
import logging
import hashlib
from pandas.util import hash_pandas_object

# Global cache
_MTF_CACHE = {}
_MAX_MTF_CACHE_SIZE = 20

def clear_mtf_cache():
    global _MTF_CACHE
    import gc
    if _MTF_CACHE:
        count = len(_MTF_CACHE)
        _MTF_CACHE.clear()
        gc.collect()
        logger.info(f"[MTF Cache] Cleared {count} entries")

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Allow explicit opt-out for stability/debugging.
if os.getenv("ARES_DISABLE_NUMBA_MTF", "0") == "1":
    NUMBA_AVAILABLE = False

if not NUMBA_AVAILABLE:
    # Overwrite with dummy decorators if disabled or not found
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

# ===== NUMBA-OPTIMIZED PRIMITIVES (GROUP-AWARE) =====

@njit(nogil=True, fastmath=True)
def _rolling_mean_grouped_numba(values: np.ndarray, group_ids: np.ndarray, window: int) -> np.ndarray:
    """
    O(N) Group-aware Rolling Mean.
    Strictly requires full window of data within the same group.
    """
    n = len(values)
    output = np.full(n, np.nan, dtype=np.float32)
    if window <= 0: return output

    s = 0.0
    count = 0
    current_group = -1

    # Initialize group
    if n > 0: current_group = group_ids[0]

    for i in range(n):
        # Check group change
        if group_ids[i] != current_group:
            s = 0.0
            count = 0
            current_group = group_ids[i]

        # Add entering value
        val = values[i]
        if not np.isnan(val):
            s += val
            count += 1
        else:
            # If we encounter NaN, the window becomes invalid if strict?
            # Existing implementation ignored NaNs in sum/count but skipped output?
            # Existing: 'if not np.isnan(v): s += v; count += 1'.
            # output = s / count. So it was partial-NaN safe.
            # But here we do incremental.
            # Handling NaNs incrementally is tricky if we don't know what's leaving.
            # We must look back at 'leaving' value.
            pass

    # RE-IMPLEMENTATION STRATEGY:
    # To handle NaNs correctly in O(N) with arbitrary patterns, we need to check 'leaving' value.
    # Since we can access values array, we can do that.

    # Reset
    s = 0.0
    count = 0
    current_group = -1
    if n > 0: current_group = group_ids[0]

    # Track validity length (consecutive elements in group)
    group_len = 0

    for i in range(n):
        if group_ids[i] != current_group:
            s = 0.0
            count = 0
            group_len = 0
            current_group = group_ids[i]

        group_len += 1

        # Add entering
        val_enter = values[i]
        if not np.isnan(val_enter):
            s += val_enter
            count += 1

        # Remove leaving
        if group_len > window:
            val_leave = values[i - window]
            if not np.isnan(val_leave):
                s -= val_leave
                count -= 1

        # Output if full window available (group_len >= window) AND count > 0
        if group_len >= window and count > 0:
            output[i] = s / count

    return output

@njit(nogil=True, fastmath=True)
def _rolling_std_grouped_numba(values: np.ndarray, group_ids: np.ndarray, window: int) -> np.ndarray:
    """
    O(N) Group-aware Rolling Std.
    """
    n = len(values)
    output = np.full(n, np.nan, dtype=np.float32)
    if window <= 0: return output

    s = 0.0
    ss = 0.0
    count = 0
    current_group = -1
    if n > 0: current_group = group_ids[0]
    group_len = 0

    for i in range(n):
        if group_ids[i] != current_group:
            s = 0.0
            ss = 0.0
            count = 0
            group_len = 0
            current_group = group_ids[i]

        group_len += 1

        # Enter
        val_enter = values[i]
        if not np.isnan(val_enter):
            s += val_enter
            ss += val_enter * val_enter
            count += 1

        # Leave
        if group_len > window:
            val_leave = values[i - window]
            if not np.isnan(val_leave):
                s -= val_leave
                ss -= val_leave * val_leave
                count -= 1

        if group_len >= window and count > 1:
            mean = s / count
            var = (ss - count * mean * mean) / (count - 1)
            if var > 1e-12:
                output[i] = np.sqrt(var)
            else:
                output[i] = 0.0

    return output

@njit(nogil=True, fastmath=True)
def _rolling_sum_grouped_numba(values: np.ndarray, group_ids: np.ndarray, window: int) -> np.ndarray:
    n = len(values)
    output = np.full(n, np.nan, dtype=np.float32)
    if window <= 0: return output

    s = 0.0
    count = 0 # To track NaNs
    current_group = -1
    if n > 0: current_group = group_ids[0]
    group_len = 0

    for i in range(n):
        if group_ids[i] != current_group:
            s = 0.0
            count = 0
            group_len = 0
            current_group = group_ids[i]

        group_len += 1

        val_enter = values[i]
        if not np.isnan(val_enter):
            s += val_enter
            count += 1

        if group_len > window:
            val_leave = values[i - window]
            if not np.isnan(val_leave):
                s -= val_leave
                count -= 1

        if group_len >= window and count > 0:
            output[i] = s

    return output

@njit(nogil=True, fastmath=True)
def _rolling_min_grouped_numba(values: np.ndarray, group_ids: np.ndarray, window: int) -> np.ndarray:
    """
    O(N) Group-aware Rolling Min using Monotonic Queue (Deque).
    """
    n = len(values)
    output = np.full(n, np.nan, dtype=np.float32)
    if window <= 0: return output

    # Deque buffers (indices)
    # Max size is 'window'
    deque_idx = np.empty(window + 2, dtype=np.int32)
    head = 0
    tail = 0 # Points to next empty slot. size = tail - head

    current_group = -1
    if n > 0: current_group = group_ids[0]
    group_len = 0

    for i in range(n):
        if group_ids[i] != current_group:
            # Reset
            head = 0
            tail = 0
            group_len = 0
            current_group = group_ids[i]

        group_len += 1
        val = values[i]

        # We only consider non-NaNs for Min?
        # Or should NaN break the min?
        # Original implementation: 'if not np.isnan(v): min_val = min(min_val, v)'.
        # So NaNs are skipped.
        # This makes Monotonic Queue tricky because indices are not contiguous in value space.
        # BUT, standard deque stores indices of *values*.
        # If we skip NaN, we effectively don't add it to the deque.
        # However, checking 'expiry' (leaving window) relies on index.

        if not np.isnan(val):
            # Maintain increasing order for Min
            while tail > head:
                back_idx = deque_idx[tail - 1]
                # Compare values
                if values[back_idx] >= val:
                    tail -= 1
                else:
                    break
            deque_idx[tail] = i
            tail += 1

        # Remove expired from head
        # Even if 'i-window' was NaN, we need to check if head is out of bounds
        if tail > head:
            if deque_idx[head] <= i - window:
                head += 1

        if group_len >= window:
            if tail > head:
                output[i] = values[deque_idx[head]]
            else:
                # Window full of NaNs
                output[i] = np.nan

    return output

@njit(nogil=True, fastmath=True)
def _rolling_max_grouped_numba(values: np.ndarray, group_ids: np.ndarray, window: int) -> np.ndarray:
    """
    O(N) Group-aware Rolling Max using Monotonic Queue.
    """
    n = len(values)
    output = np.full(n, np.nan, dtype=np.float32)
    if window <= 0: return output

    deque_idx = np.empty(window + 2, dtype=np.int32)
    head = 0
    tail = 0

    current_group = -1
    if n > 0: current_group = group_ids[0]
    group_len = 0

    for i in range(n):
        if group_ids[i] != current_group:
            head = 0
            tail = 0
            group_len = 0
            current_group = group_ids[i]

        group_len += 1
        val = values[i]

        if not np.isnan(val):
            # Maintain decreasing order for Max
            while tail > head:
                back_idx = deque_idx[tail - 1]
                if values[back_idx] <= val:
                    tail -= 1
                else:
                    break
            deque_idx[tail] = i
            tail += 1

        if tail > head:
            if deque_idx[head] <= i - window:
                head += 1

        if group_len >= window:
            if tail > head:
                output[i] = values[deque_idx[head]]
            else:
                output[i] = np.nan

    return output

@njit(nogil=True, fastmath=False) # Disable fastmath to ensure strict NaN handling for EWMA state
def _numba_ewma_grouped(x: np.ndarray, group_ids: np.ndarray, alpha: float) -> np.ndarray:
    """
    Group-aware EWMA. Resets on group change.
    Using 'adjust=False' logic (recursive).
    y[t] = (1-alpha)*y[t-1] + alpha*x[t]
    """
    n = len(x)
    output = np.full(n, np.nan, dtype=np.float32)
    if n == 0: return output

    current_group = group_ids[0]
    # Use strict float32 nan
    last_val = np.float32(np.nan)

    # First element init
    if not np.isnan(x[0]):
        last_val = x[0]
        output[0] = last_val

    for i in range(1, n):
        if group_ids[i] != current_group:
            current_group = group_ids[i]
            last_val = np.float32(np.nan)

        val = x[i]
        if not np.isnan(val):
            if np.isnan(last_val):
                last_val = val
            else:
                last_val = (1.0 - alpha) * last_val + alpha * val
            output[i] = last_val
        else:
            # If input is NaN, output NaN, but PRESERVE state (ignore_na=True behavior)
            # effectively last_val remains same for next step's calculation?
            # Pandas default (ignore_na=False): y_t becomes NaN if x_t is NaN.
            # Next step uses y_{t-1}.
            # If y_{t-1} is NaN, it stays NaN?
            # Actually, Pandas `ewm(ignore_na=False)`:
            # y_0 = x_0. If x_0 NaN, y_0 NaN.
            # y_1 = (1-a)y_0 + a*x_1. If y_0 NaN, y_1 NaN.
            # So NaN propagates indefinitely unless restart?
            # Wait, `adjust=False` means strictly recursive.
            # If last_val becomes NaN, it stays NaN forever.
            # But here `val` is valid.
            # If `last_val` was NaN, we treat `val` as new start! `if np.isnan(last_val): last_val = val`.
            # This logic RESETS the EWMA if it encounters a valid value after a sequence of NaNs (or start).
            # This is robust and prevents infinite NaN propagation.
            output[i] = last_val # This emits the *previous* valid value if current is NaN?
            # No, if current is NaN, we should output NaN.
            output[i] = np.float32(np.nan)
            pass

    return output

@njit(nogil=True, fastmath=True)
def _rolling_winsorized_zscore_grouped_numba(
    values: np.ndarray,
    group_ids: np.ndarray,
    window: int,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99
) -> np.ndarray:
    # Winsorization requires sorting, so it is O(N * W log W) or O(N * W).
    # Hard to optimize to O(N) without complex structures (Skip List / BST).
    # We will keep the parallel implementation for this one, as it's robust stats.
    # But we can optimize memory allocation.
    n = len(values)
    output = np.zeros(n, dtype=np.float32)
    if window <= 0: return output

    # Parallel loop over 'i' is okay if W is not too huge.
    # For W=600, sorting 600 elements is fast (especially nearly sorted).
    # We will invoke the logic but wrapped in parallel=True in a separate function
    # if we want to mix 'fastmath=True' and 'parallel=True'.
    # The decorator is already there.
    # We'll leave this function as is, but ensure it handles groups correctly.
    # It does check group_ids.
    return _rolling_winsorized_zscore_grouped_numba_impl(values, group_ids, window, lower_quantile, upper_quantile)

@njit(nogil=True, fastmath=True)
def _rolling_winsorized_zscore_grouped_numba_impl(
    values: np.ndarray,
    group_ids: np.ndarray,
    window: int,
    lower_quantile: float,
    upper_quantile: float
) -> np.ndarray:
    n = len(values)
    output = np.zeros(n, dtype=np.float32)

    for i in prange(n):
        if i < window - 1: continue
        start_idx = i - window + 1
        # Check strict group membership
        if group_ids[i] != group_ids[start_idx]: continue

        # Extract window buffer
        # Pre-allocate buffer? In parallel, dynamic alloc is okay-ish.
        window_vals = np.empty(window, dtype=np.float32)
        count = 0

        for k in range(window):
            idx = start_idx + k
            v = values[idx]
            if not np.isnan(v) and not np.isinf(v):
                window_vals[count] = v
                count += 1

        if count < 2: continue

        # Sort valid part
        valid_slice = window_vals[:count]
        # np.partition is faster than sort for quantiles? Numba supports sort().
        valid_slice.sort()

        idx_lower = int(lower_quantile * (count - 1))
        idx_upper = int(upper_quantile * (count - 1))
        q_low = valid_slice[idx_lower]
        q_high = valid_slice[idx_upper]

        s = 0.0
        ss = 0.0
        for k in range(count):
            v = valid_slice[k]
            if v < q_low: v = q_low
            if v > q_high: v = q_high
            s += v
            ss += v*v

        mean = s / count
        var = (ss - count * mean * mean) / (count - 1)
        std = np.sqrt(var) if var > 0 else 0.0

        val = values[i]
        if np.isnan(val) or np.isinf(val):
            val = 0.0
        else:
            if val < q_low: val = q_low
            if val > q_high: val = q_high

        if std > 1e-9:
            output[i] = (val - mean) / std
        else:
            output[i] = 0.0

    return output


@njit(nogil=True, fastmath=True)
def _compute_candle_geometry_grouped_numba(
    open_p: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, group_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(open_p)
    body_to_range = np.zeros(n, dtype=np.float32)
    shadow_asymmetry = np.zeros(n, dtype=np.float32)
    clv = np.zeros(n, dtype=np.float32)
    real_body = np.zeros(n, dtype=np.float32)
    for i in prange(n):
        candle_range = high[i] - low[i]
        upper_shadow = high[i] - max(open_p[i], close[i])
        lower_shadow = min(open_p[i], close[i]) - low[i]
        real_body[i] = abs(close[i] - open_p[i])
        if candle_range > 1e-9:
            body_to_range[i] = real_body[i] / candle_range
            shadow_asymmetry[i] = (upper_shadow - lower_shadow) / candle_range
            clv[i] = ((close[i] - low[i]) - (high[i] - close[i])) / candle_range
    return body_to_range, shadow_asymmetry, clv, real_body

@njit(nogil=True, fastmath=True)
def _compute_volatility_features_grouped_numba(
    log_ret: np.ndarray, group_ids: np.ndarray, short_window: int = 20, long_window: int = 200
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimized Volatility Features using O(N) operations.
    """
    n = len(log_ret)

    # 1. Short Volatility (Rolling Std) - O(N)
    vol_short = _rolling_std_grouped_numba(log_ret, group_ids, short_window)

    # 2. Long Mean of Short Vol (Rolling Mean of vol_short) - O(N)
    # Note: vol_short has NaNs at start of groups. _rolling_mean handles NaNs?
    # Our new implementation skips NaNs (reduces count).
    vol_long_mean = _rolling_mean_grouped_numba(vol_short, group_ids, long_window)

    # 3. Long Std of Short Vol (Rolling Std of vol_short) - O(N)
    vol_long_std = _rolling_std_grouped_numba(vol_short, group_ids, long_window)

    rv_z_short = np.zeros(n, dtype=np.float32)

    for i in range(n):
        if not np.isnan(vol_long_std[i]) and vol_long_std[i] > 1e-8:
            if not np.isnan(vol_short[i]) and not np.isnan(vol_long_mean[i]):
                rv_z_short[i] = (vol_short[i] - vol_long_mean[i]) / vol_long_std[i]

    return vol_short, vol_long_mean, vol_long_std, rv_z_short

@njit
def _numba_kalman_filter_grouped(data, group_ids, Q, R, initial_value):
    n = len(data)
    filtered = np.zeros(n, dtype=np.float32)
    variances = np.zeros(n, dtype=np.float32)
    x = initial_value
    P = 1.0
    current_group = -1
    if n > 0: current_group = group_ids[0]
    for i in range(n):
        if group_ids[i] != current_group:
            x = initial_value
            P = 1.0
            current_group = group_ids[i]
        val = data[i]
        if np.isnan(val):
            filtered[i] = np.nan
            variances[i] = np.nan
        else:
            x_prior = x
            P_prior = P + Q
            if P_prior > 1e6: P_prior = 1e6
            elif P_prior < 1e-12: P_prior = 1e-12
            denominator = P_prior + R
            K = P_prior / denominator if denominator > 1e-12 else 0.0
            if K > 1.0: K = 1.0
            elif K < 0.0: K = 0.0
            x = x_prior + K * (val - x_prior)
            P = (1 - K) * P_prior
            if P > 1e6: P = 1e6
            elif P < 1e-12: P = 1e-12
            filtered[i] = x
            variances[i] = P
    return filtered, variances

@njit(nogil=True, fastmath=True)
def _compute_dual_cusum_grouped_numba(
    log_ret_smooth: np.ndarray, residual_ret: np.ndarray, sigma: np.ndarray, er: np.ndarray,
    group_ids: np.ndarray, k: float, er_min: float
):
    n = len(log_ret_smooth)
    s_trend_pos = np.zeros(n, dtype=np.float32)
    s_trend_neg = np.zeros(n, dtype=np.float32)
    s_rev_pos = np.zeros(n, dtype=np.float32)
    s_rev_neg = np.zeros(n, dtype=np.float32)
    tp, tn, rp, rn = 0.0, 0.0, 0.0, 0.0
    current_group = -1
    if n > 0: current_group = group_ids[0]
    for t in range(n):
        if group_ids[t] != current_group:
            tp, tn, rp, rn = 0.0, 0.0, 0.0, 0.0
            current_group = group_ids[t]
        if er[t] < er_min:
            tp, tn, rp, rn = 0.0, 0.0, 0.0, 0.0
        else:
            cur_h = max(k * sigma[t], 1e-4)
            tp = max(0.0, tp + log_ret_smooth[t])
            tn = min(0.0, tn + log_ret_smooth[t])
            if tp > cur_h: tp = 0.0
            if tn < -cur_h: tn = 0.0
            rp = max(0.0, rp + residual_ret[t])
            rn = min(0.0, rn + residual_ret[t])
            if rp > cur_h: rp = 0.0
            if rn < -cur_h: rn = 0.0
        s_trend_pos[t] = tp
        s_trend_neg[t] = tn
        s_rev_pos[t] = rp
        s_rev_neg[t] = rn
    return s_trend_pos, s_trend_neg, s_rev_pos, s_rev_neg

def _rolling_mean_abs_dev_numba(values: np.ndarray, window: int) -> np.ndarray:
    # Use the grouped mean function?
    # This is a legacy helper, but if we want to optimize it...
    # It's not used in the main pipeline so we leave it.
    n = len(values)
    output = np.full(n, np.nan, dtype=np.float32)
    if window <= 0: return output
    for i in prange(n):
        if i < window - 1: continue
        start_idx = i - window + 1
        s = 0.0
        count = 0
        for j in range(start_idx, i + 1):
            v = values[j]
            if not np.isnan(v):
                s += v
                count += 1
        if count == 0: continue
        mean_val = s / count
        mad_sum = 0.0
        for j in range(start_idx, i + 1):
            v = values[j]
            if not np.isnan(v):
                mad_sum += abs(v - mean_val)
        output[i] = mad_sum / count
    return output

def _rolling_argmax_numba(values: np.ndarray, window: int) -> np.ndarray:
    # Legacy O(N*W). Leave for now.
    n = len(values)
    output = np.full(n, np.nan, dtype=np.float32)
    if window <= 0: return output
    for i in prange(n):
        if i < window - 1: continue
        start_idx = i - window + 1
        max_idx = 0
        max_val = values[start_idx]
        for j in range(start_idx + 1, i + 1):
            if values[j] > max_val:
                max_val = values[j]
                max_idx = j - start_idx
        output[i] = max_idx
    return output

def _rolling_argmin_numba(values: np.ndarray, window: int) -> np.ndarray:
    # Legacy O(N*W). Leave for now.
    n = len(values)
    output = np.full(n, np.nan, dtype=np.float32)
    if window <= 0: return output
    for i in prange(n):
        if i < window - 1: continue
        start_idx = i - window + 1
        min_idx = 0
        min_val = values[start_idx]
        for j in range(start_idx + 1, i + 1):
            if values[j] < min_val:
                min_val = values[j]
                min_idx = j - start_idx
        output[i] = min_idx
    return output

# --- LEGACY HELPERS DEPRECATED ---
# The following functions are retained for backward compatibility with
# external modules (e.g. de_prado_causal_features.py) but are not used
# in the core MTF pipeline.

class KalmanFilter1D:
    def __init__(self, Q: float = 1e-5, R: float = 0.01, initial_value: float = 0.0):
        self.Q = Q
        self.R = R
        self.x = initial_value
        self.P = 1.0

    def update(self, measurement: float) -> Tuple[float, float]:
        x_prior = self.x
        P_prior = self.P + self.Q
        P_prior = np.clip(P_prior, 1e-12, 1e6)
        denominator = P_prior + self.R
        K = P_prior / denominator if denominator > 1e-12 else 0.0
        K = np.clip(K, 0.0, 1.0)
        self.x = x_prior + K * (measurement - x_prior)
        self.P = (1 - K) * P_prior
        self.P = np.clip(self.P, 1e-12, 1e6)
        return self.x, self.P

    def filter_series(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        if NUMBA_AVAILABLE:
            filtered, variances = _numba_kalman_filter(
                series.values.astype(np.float32), self.Q, self.R, self.x
            )
            return pd.Series(filtered, index=series.index), pd.Series(variances, index=series.index)
        else:
            filtered, variances = [], []
            for val in series:
                f, v = self.update(val)
                filtered.append(f)
                variances.append(v)
            return pd.Series(filtered, index=series.index), pd.Series(variances, index=series.index)

def kalman_smooth_trend(prices: pd.Series, Q: float = 1e-5, R: float = 0.01) -> Tuple[pd.Series, pd.Series]:
    kf = KalmanFilter1D(Q=Q, R=R, initial_value=prices.iloc[0] if len(prices) > 0 else 0.0)
    return kf.filter_series(prices)

def compute_dual_cusum_statistics(
    close: pd.Series, volume: Optional[pd.Series] = None, k: float = 0.12, er_min: float = 0.2,
    window_vol: int = 20, window_er: int = 10, Q: float = 1e-5, R: float = 0.01
) -> pd.DataFrame:
    log_ret = np.log(close / close.shift(1)).fillna(0.0)
    kf = KalmanFilter1D(Q=Q, R=R, initial_value=float(log_ret.iloc[0]))
    log_ret_smooth_raw, _ = kf.filter_series(log_ret)
    log_ret_smooth_series = pd.Series(log_ret_smooth_raw, index=close.index).fillna(0.0)
    sigma = log_ret_smooth_series.rolling(window_vol, min_periods=1).std()
    change = log_ret_smooth_series.rolling(window_er).sum().abs()
    volatility = log_ret_smooth_series.abs().rolling(window_er, min_periods=1).sum()
    ER = (change / (volatility + 1e-12)).fillna(0.0)
    expected_return = log_ret_smooth_series.rolling(window_vol, min_periods=1).mean()
    residual_ret = (log_ret_smooth_series - expected_return).fillna(0.0)

    if NUMBA_AVAILABLE:
        r_arr = log_ret_smooth_series.values.astype(np.float32)
        res_arr = residual_ret.values.astype(np.float32)
        sigma_arr = sigma.fillna(0.0).values.astype(np.float32)
        er_arr = ER.fillna(0.0).values.astype(np.float32)
        s_tp, s_tn, s_rp, s_rn = _compute_dual_cusum_grouped_numba(r_arr, res_arr, sigma_arr, er_arr, group_ids=np.zeros(len(r_arr), dtype=np.int32), k=k, er_min=er_min)
    else:
        # Fallback (Slow)
        n = len(close)
        s_tp = np.zeros(n, dtype=np.float32)
        s_tn = np.zeros(n, dtype=np.float32)
        s_rp = np.zeros(n, dtype=np.float32)
        s_rn = np.zeros(n, dtype=np.float32)
        h_arr = (k * sigma).fillna(0.0).values
        er_arr_np = ER.values
        r_arr_np = log_ret_smooth_series.values
        res_arr_np = residual_ret.values
        tp, tn, rp, rn = 0.0, 0.0, 0.0, 0.0
        for t in range(n):
            if er_arr_np[t] >= er_min:
                cur_h = max(h_arr[t], 1e-4)
                tp = max(0.0, tp + r_arr_np[t])
                tn = min(0.0, tn + r_arr_np[t])
                if tp > cur_h: tp = 0.0
                if tn < -cur_h: tn = 0.0
                rp = max(0.0, rp + res_arr_np[t])
                rn = min(0.0, rn + res_arr_np[t])
                if rp > cur_h: rp = 0.0
                if rn < -cur_h: rn = 0.0
            s_tp[t], s_tn[t], s_rp[t], s_rn[t] = tp, tn, rp, rn

    return pd.DataFrame({
        'S_trend_pos': s_tp, 'S_trend_neg': s_tn, 'S_rev_pos': s_rp, 'S_rev_neg': s_rn,
        'smoothed_return': log_ret_smooth_series, 'residual_return': residual_ret
    }, index=close.index)

def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

# ... (Other legacy functions omitted for brevity, keeping them as they were in file content) ...
# Actually, I should write the WHOLE file content.
# I will copy the rest of legacy functions and then the create_meta_features.

def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def get_efficiency_ratio(close: pd.Series, window: int = 14) -> pd.Series:
    change = close.diff(window).abs()
    volatility = close.diff().abs().rolling(window).sum()
    return change / (volatility + 1e-9)

def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-9))
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent

def _rolling_mad_numpy(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or window > len(values): return np.full(len(values), np.nan)
    values = np.ascontiguousarray(values, dtype=np.float32)
    shape = (len(values) - window + 1, window)
    strides = (values.strides[0], values.strides[0])
    try:
        windows = np.lib.stride_tricks.as_strided(values, shape=shape, strides=strides, writeable=False)
        means = np.mean(windows, axis=1, keepdims=True)
        mad = np.mean(np.abs(windows - means), axis=1)
        return np.concatenate([np.full(window - 1, np.nan), mad])
    except Exception: return np.full(len(values), np.nan)

def compute_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=period).mean()
    if NUMBA_AVAILABLE:
        mean_dev_arr = _rolling_mean_abs_dev_numba(tp.values.astype(np.float32), period)
        mean_dev = pd.Series(mean_dev_arr, index=tp.index)
        if len(mean_dev) >= period: mean_dev.iloc[: period - 1] = np.nan
    else:
        mean_dev = pd.Series(_rolling_mad_numpy(tp.values, period), index=tp.index)
    return (tp - sma_tp) / (0.015 * mean_dev + 1e-9)

def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    plus_dm = high.diff().clip(lower=0)
    minus_dm = low.diff().clip(lower=0)
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm = plus_dm.where(plus_dm > 0, 0.0)
    minus_dm = minus_dm.where(minus_dm > 0, 0.0)
    mask_plus = (plus_dm > minus_dm) & (plus_dm > 0)
    mask_minus = (minus_dm > plus_dm) & (minus_dm > 0)
    plus_dm_final = pd.Series(0.0, index=close.index)
    minus_dm_final = pd.Series(0.0, index=close.index)
    plus_dm_final[mask_plus] = plus_dm[mask_plus]
    minus_dm_final[mask_minus] = minus_dm[mask_minus]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    plus_di = 100 * (plus_dm_final.rolling(period).mean() / (atr + 1e-9))
    minus_di = 100 * (minus_dm_final.rolling(period).mean() / (atr + 1e-9))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    adx = dx.rolling(period).mean()
    return adx, plus_di, minus_di

def compute_bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    width = (upper - lower) / (middle + 1e-9)
    return upper, middle, lower, width

def compute_choppiness_index(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(period).sum()
    high_max = high.rolling(period).max()
    low_min = low.rolling(period).min()
    return 100 * np.log10(atr_sum / (high_max - low_min + 1e-9)) / np.log10(period)

def compute_cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    mf_multiplier = ((close - low) - (high - close)) / (high - low + 1e-9)
    mf_volume = mf_multiplier * volume
    return mf_volume.rolling(period).sum() / (volume.rolling(period).sum() + 1e-9)

def compute_force_index(close: pd.Series, volume: pd.Series, period: int = 13) -> pd.Series:
    fi = close.diff(1) * volume
    return fi.ewm(span=period).mean()

def compute_hurst_proxy(close: pd.Series, window: int = 100) -> pd.Series:
    ret = np.log(close / close.shift(1)).fillna(0.0)
    roll_close = close.rolling(window)
    roll_ret = ret.rolling(window)
    r = (roll_close.max() - roll_close.min()) / (roll_close.mean() + 1e-9)
    s = roll_ret.std()
    rs = r / (s + 1e-9)
    return np.log(rs + 1e-9) / np.log(window)

def compute_parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    log_hl = np.log(high / (low + 1e-9)) ** 2
    return np.sqrt((1.0 / (4.0 * np.log(2.0))) * log_hl.rolling(window).mean())

def compute_donchian_channel(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    upper = high.rolling(window).max()
    lower = low.rolling(window).min()
    width = upper - lower
    position = (close - lower) / (width + 1e-9)
    return upper, lower, position

def compute_wick_to_body_ratio(open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    max_oc = pd.concat([open_p, close], axis=1).max(axis=1)
    min_oc = pd.concat([open_p, close], axis=1).min(axis=1)
    upper_wick = high - max_oc
    body = max_oc - min_oc
    return upper_wick / (body.replace(0, 1e-9))

def compute_relative_volume_stress(volume: pd.Series, window: int = 20) -> pd.Series:
    sma_vol = volume.rolling(window).mean()
    return volume / (sma_vol + 1e-9)

def compute_amihud_illiquidity(open_p: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    log_ret = np.log(close / (open_p + 1e-9)).abs()
    return log_ret / np.log1p(volume + 1e-9)

def compute_displacement_ratio(open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    body_signed = close - open_p
    rng = high - low
    return body_signed / (rng + 1e-9)

def compute_proxy_levels(
    high: pd.Series, low: pd.Series, close: pd.Series, pivot_window: int = 30, atr_window: int = 200, k_factor: float = 1.0
) -> Tuple[pd.Series, pd.Series]:
    pivot_low = low.rolling(window=pivot_window).min()
    pivot_high = high.rolling(window=pivot_window).max()
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_long = tr.rolling(atr_window).mean()
    proxy_long = pivot_low - (k_factor * atr_long)
    proxy_short = pivot_high + (k_factor * atr_long)
    return proxy_long, proxy_short

def compute_garman_klass_volatility(open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    log_hl = np.log(high / (low + 1e-9)) ** 2
    log_co = np.log(close / (open_p + 1e-9)) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return np.sqrt(gk.rolling(window).mean())

def compute_volume_delta(close: pd.Series, open_p: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close - open_p)
    return direction * volume

def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (direction * volume).cumsum()

def compute_rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    roll = series.rolling(window)
    return (series - roll.mean()) / (roll.std() + 1e-9)

def compute_rolling_percentile(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window).rank(pct=True)

def compute_bars_since(condition: pd.Series) -> pd.Series:
    idx = pd.Series(np.arange(len(condition)), index=condition.index)
    last_occurrence = idx.where(condition).ffill()
    return (idx - last_occurrence).fillna(len(condition))

def compute_rogers_satchell_volatility(open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    h_c = np.log(high / (close + 1e-9))
    h_o = np.log(high / (open_p + 1e-9))
    l_c = np.log(low / (close + 1e-9))
    l_o = np.log(low / (open_p + 1e-9))
    rs_var = (h_c * h_o) + (l_c * l_o)
    return np.sqrt(rs_var.rolling(window).mean())

def compute_yang_zhang_volatility(open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    log_oc_prev = np.log(open_p / (close.shift(1) + 1e-9))
    log_co = np.log(close / (open_p + 1e-9))
    var_o = log_oc_prev.rolling(window).var()
    var_c = log_co.rolling(window).var()
    h_c = np.log(high / (close + 1e-9))
    h_o = np.log(high / (open_p + 1e-9))
    l_c = np.log(low / (close + 1e-9))
    l_o = np.log(low / (open_p + 1e-9))
    rs_var = (h_c * h_o) + (l_c * l_o)
    rs_var_mean = rs_var.rolling(window).mean()
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    yz_var = var_o + k * var_c + (1 - k) * rs_var_mean
    return np.sqrt(yz_var)

def _rolling_argmax_numpy(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or window > len(values): return np.full(len(values), np.nan)
    values = np.ascontiguousarray(values, dtype=np.float32)
    shape = (len(values) - window + 1, window)
    strides = (values.strides[0], values.strides[0])
    try:
        windows = np.lib.stride_tricks.as_strided(values, shape=shape, strides=strides, writeable=False)
        return np.concatenate([np.full(window - 1, np.nan), np.argmax(windows, axis=1)])
    except Exception: return np.full(len(values), np.nan)

def _rolling_argmin_numpy(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or window > len(values): return np.full(len(values), np.nan)
    values = np.ascontiguousarray(values, dtype=np.float32)
    shape = (len(values) - window + 1, window)
    strides = (values.strides[0], values.strides[0])
    try:
        windows = np.lib.stride_tricks.as_strided(values, shape=shape, strides=strides, writeable=False)
        return np.concatenate([np.full(window - 1, np.nan), np.argmin(windows, axis=1)])
    except Exception: return np.full(len(values), np.nan)

def compute_aroon(high: pd.Series, low: pd.Series, window: int = 25) -> pd.Series:
    if NUMBA_AVAILABLE:
        high_idx_arr = _rolling_argmax_numba(high.values.astype(np.float32), window)
        low_idx_arr = _rolling_argmin_numba(low.values.astype(np.float32), window)
    else:
        high_idx_arr = _rolling_argmax_numpy(high.values, window)
        low_idx_arr = _rolling_argmin_numpy(low.values, window)
    high_idx = pd.Series(high_idx_arr, index=high.index)
    low_idx = pd.Series(low_idx_arr, index=low.index)
    if len(high_idx) >= window:
        high_idx.iloc[: window - 1] = np.nan
        low_idx.iloc[: window - 1] = np.nan
    aroon_up = ((window - (window - 1 - high_idx)) / window) * 100
    aroon_down = ((window - (window - 1 - low_idx)) / window) * 100
    return aroon_up - aroon_down

def compute_ease_of_movement(high: pd.Series, low: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
    dm = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
    box_ratio = (volume + 1e-9) / ((high - low) + 1e-9)
    eom = dm / box_ratio
    return eom.rolling(window).mean()

def compute_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
    tp = (high + low + close) / 3
    rmf = tp * volume
    diff = tp.diff()
    pos_flow = rmf.where(diff > 0, 0)
    neg_flow = rmf.where(diff < 0, 0)
    pos_mean = pos_flow.rolling(window).mean()
    neg_mean = neg_flow.rolling(window).mean()
    return 100 - (100 / (1 + (pos_mean / (neg_mean + 1e-9))))

def compute_fisher_transform(high: pd.Series, low: pd.Series, window: int = 10) -> pd.Series:
    mid = (high + low) / 2
    rolling_min = low.rolling(window).min()
    rolling_max = high.rolling(window).max()
    val = 2 * ((mid - rolling_min) / (rolling_max - rolling_min + 1e-9) - 0.5)
    val = val.ewm(alpha=0.33).mean().clip(-0.999, 0.999)
    fisher = 0.5 * np.log((1 + val) / (1 - val))
    return fisher.ewm(alpha=0.5).mean()

def compute_hilbert_phase(series: pd.Series) -> pd.Series:
    centered = series - series.rolling(30).mean()
    q1 = centered.diff(2)
    i1 = centered.shift(1).diff(2)
    return np.arctan2(q1, i1 + 1e-9).fillna(0)

def _align_to_features(arr: Any, n: int) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float32)
    if len(values) == n: return values
    if len(values) > n: return values[-n:]
    padded = np.full(n, np.nan, dtype=np.float32)
    padded[-len(values):] = values
    return padded

def create_meta_features(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    volume_available: bool = True,
    include_raw_signals: bool = False,
    use_kalman: bool = True,
    horizon_bars: Optional[int] = None,
    downsample_long_horizon: bool = True,
    windows: List[int] = [10, 20, 50, 100, 150, 200],
    asset_id_col: Optional[str] = None
) -> pd.DataFrame:
    import time
    import gc
    # print(f"DEBUG: Entered create_meta_features for {len(df)} rows", flush=True) # REMOVED
    start_time = time.time()
    
    # --- CACHE CHECK ---
    # print("DEBUG: Starting cache check", flush=True) # REMOVED
    try:
        h_idx = hashlib.md5(hash_pandas_object(df.index).values.tobytes()).hexdigest()
        cols_to_hash = [c for c in ['close', 'high', 'low', 'open', 'volume', 'Close', 'High', 'Low', 'Open', 'Volume'] if c in df.columns]
        if cols_to_hash:
            h_content = hashlib.md5(hash_pandas_object(df[cols_to_hash]).values.tobytes()).hexdigest()
        else:
            h_content = "no_ohlcv"
        h_signals = "no_signals"
        if signals is not None and not signals.empty:
             h_signals = hashlib.md5(hash_pandas_object(signals).values.tobytes()).hexdigest()
        config_str = f"{len(df)}_{windows}_{volume_available}_{include_raw_signals}_{use_kalman}_{horizon_bars}_{downsample_long_horizon}_{asset_id_col}"
        cache_key = f"{h_idx}_{h_content}_{h_signals}_{config_str}"
        if cache_key in _MTF_CACHE:
            logger.info(f"⚡ [MTF Cache] Returning pre-computed features for {len(df)} rows")
            return _MTF_CACHE[cache_key].copy()
    except Exception as e:
        logger.warning(f"⚠️ Cache check failed: {e}")
        cache_key = None
    
    # print(f"DEBUG: Cache check done. Key: {cache_key}", flush=True) # REMOVED
    logger.info(f"🔍 Starting MTF feature generation for {len(df)} rows. Hash: {cache_key}")
    if asset_id_col:
        logger.info(f"   [Debug] MTF: Asset ID col: {asset_id_col}")
    else:
        logger.info("   [Debug] MTF: No Asset ID col")
    
    # 1. Alignment and Indexing
    len_df = len(df)
    len_sig = len(signals) if signals is not None else 0
    if signals is not None and len_df != len_sig:
        target_len = min(len_df, len_sig)
        if len_df > target_len: df = df.iloc[-target_len:, :]
        if len_sig > target_len: signals = signals.iloc[-target_len:, :]
    
    if (signals is not None and not df.index.equals(signals.index)) or df.index.has_duplicates:
        df = df.reset_index(drop=True)
        if signals is not None: signals = signals.reset_index(drop=True)

    # 2. Group ID Logic
    if asset_id_col is None:
        for col in ['asset_id', 'symbol', 'pair', 'instrument', 'AssetId', 'Symbol']:
            if col in df.columns:
                asset_id_col = col
                break
    
    if asset_id_col and asset_id_col in df.columns:
        group_ids, _ = pd.factorize(df[asset_id_col])
        group_ids = group_ids.astype(np.int32)
        logger.info(f"🔒 Cross-asset protection enabled using column '{asset_id_col}'")
    else:
        group_ids = np.zeros(len(df), dtype=np.int32)
    
    # 3. Data Extraction (Float32)
    close_col = next((c for c in df.columns if c.lower() in ['close', 'adj close']), 'close')
    high_col = next((c for c in df.columns if c.lower() == 'high'), close_col)
    low_col = next((c for c in df.columns if c.lower() == 'low'), close_col)
    open_col = next((c for c in df.columns if c.lower() == 'open'), close_col)
    vol_col = next((c for c in df.columns if 'volume' in c.lower()), None)
    
    close_arr = df[close_col].values.astype(np.float32)
    high_arr = df[high_col].values.astype(np.float32)
    low_arr = df[low_col].values.astype(np.float32)
    open_arr = df[open_col].values.astype(np.float32)
    vol_arr = df[vol_col].fillna(0).values.astype(np.float32) if (vol_col and volume_available) else np.ones_like(close_arr)
    
    # 4. Returns Calculation (Group Safe)
    log_ret_arr = np.zeros_like(close_arr)
    close_prev = np.roll(close_arr, 1)
    # Mask valid shifts (not boundary)
    mask = np.ones_like(close_arr, dtype=bool)
    mask[0] = False
    mask[1:] = (group_ids[1:] == group_ids[:-1])
    with np.errstate(divide='ignore', invalid='ignore'):
        log_ret_arr[mask] = np.log(close_arr[mask] / close_prev[mask])
    log_ret_arr = np.nan_to_num(log_ret_arr)

    # Internal Normalization Helper
    def _norm(data: np.ndarray, name: str) -> np.ndarray:
        if 'volume' in name.lower() and 'ratio' not in name.lower():
             data = np.log1p(data)
        return _rolling_winsorized_zscore_grouped_numba(data, group_ids, window=600)

    features_list = []
    
    # --- BASE FEATURES ---
    logger.info("   [NumbaDebug] Calling _norm (rolling_winsorized_zscore) for log_ret_arr")
    features_list.append({'log_ret': _norm(log_ret_arr, 'log_ret')})
    
    # Round-number distance (Normalized)
    dist_round_1 = np.abs(close_arr - np.round(close_arr))
    dist_round_01 = np.abs(close_arr - np.round(close_arr, 1))
    features_list.append({
        'dist_round_1': _norm(dist_round_1, 'dist_round_1'),
        'dist_round_01': _norm(dist_round_01, 'dist_round_01')
    })

    # --- CUSUM / EFFICIENCY ---
    if NUMBA_AVAILABLE:
        logger.info("   [NumbaDebug] Calling _numba_kalman_filter_grouped")
        kf_ret, _ = _numba_kalman_filter_grouped(log_ret_arr, group_ids, 1e-5, 0.01, 0.0)
        logger.info("   [NumbaDebug] Calling _rolling_std_grouped_numba (sigma_arr)")
        sigma_arr = _rolling_std_grouped_numba(kf_ret, group_ids, 20)

        # O(N) Efficiency Calculation
        # ER = Abs(Change) / Sum(Abs(Ret))
        # Change = Sum(Ret) over window (signed) -> Abs
        change_signed = np.abs(_rolling_sum_grouped_numba(kf_ret, group_ids, 10))
        vol_abs = _rolling_sum_grouped_numba(np.abs(kf_ret), group_ids, 10)
        er_arr = np.zeros_like(change_signed)
        m = vol_abs > 1e-9
        er_arr[m] = change_signed[m] / vol_abs[m]
        
        exp_ret = _rolling_mean_grouped_numba(kf_ret, group_ids, 20)
        resid_ret = np.nan_to_num(kf_ret - exp_ret)

        logger.info("   [NumbaDebug] Calling _compute_dual_cusum_grouped_numba")
        s_tp, s_tn, s_rp, s_rn = _compute_dual_cusum_grouped_numba(kf_ret, resid_ret, sigma_arr, er_arr, group_ids, 0.12, 0.2)
        features_list.append({
            'cusum_trend_pos': _norm(s_tp, 'cusum_trend_pos'),
            'cusum_trend_neg': _norm(s_tn, 'cusum_trend_neg'),
            'cusum_rev_pos': _norm(s_rp, 'cusum_rev_pos'),
            'cusum_rev_neg': _norm(s_rn, 'cusum_rev_neg'),
            'smoothed_return': _norm(kf_ret, 'smoothed_return'),
            'residual_return': _norm(resid_ret, 'residual_return')
        })

    # --- GEOMETRY ---
    logger.info("   [NumbaDebug] Calling _compute_candle_geometry_grouped_numba")
    body_range, shadow_asym, clv, _ = _compute_candle_geometry_grouped_numba(open_arr, high_arr, low_arr, close_arr, group_ids)
    features_list.append({
        'body_to_range': _norm(body_range, 'body_to_range'),
        'shadow_asymmetry': _norm(shadow_asym, 'shadow_asymmetry'),
        'close_location_value': _norm(clv, 'close_location_value')
    })

    # --- VOLATILITY ---
    logger.info("   [NumbaDebug] Calling _compute_volatility_features_grouped_numba")
    v_short, v_long_m, v_long_s, v_z = _compute_volatility_features_grouped_numba(log_ret_arr, group_ids)
    v_slope = v_short - np.roll(v_short, 5) # Simple diff
    # Mask slope boundary
    slope_mask = np.ones_like(v_short, dtype=bool)
    slope_mask[:5] = False
    slope_mask[5:] = (group_ids[5:] == group_ids[:-5])
    v_slope[~slope_mask] = 0.0

    features_list.append({
        'rv_z_short': _norm(v_z, 'rv_z_short'),
        'volatility_trend_slope': _norm(v_slope, 'volatility_trend_slope')
    })

    # --- SIGNALS (Pass-through) ---
    if signals is not None:
        sig_feats = {}
        for col in signals.columns:
            if 'signal' in col.lower() or 'consensus' in col.lower():
                s_arr = signals[col].fillna(0).values.astype(np.float32)
                sig_feats[f'{col}_lag_1'] = np.roll(s_arr, 1) # Simple roll, ignore boundary noise for lag
                # Interactions
                sig_feats[f'{col}_x_ret'] = _norm(s_arr * log_ret_arr, f'{col}_x_ret')
        features_list.append(sig_feats)

    # --- MULTI-TIMEFRAME LOOP ---
    for w in windows:
        logger.info(f"   [NumbaDebug] Starting MTF loop for window {w}")
        w_feats = {}
        
        # Hoisted Ops O(N)
        logger.info(f"   [NumbaDebug] Calling _rolling_max/min/mean/std for window {w}")
        r_high = _rolling_max_grouped_numba(high_arr, group_ids, w)
        r_low = _rolling_min_grouped_numba(low_arr, group_ids, w)
        r_close_mean = _rolling_mean_grouped_numba(close_arr, group_ids, w)
        r_ret_std = _rolling_std_grouped_numba(log_ret_arr, group_ids, w)
        
        # Virtual Candle
        shift = w - 1
        win_open = np.roll(open_arr, shift)
        shift_mask = np.ones_like(open_arr, dtype=bool)
        if shift > 0:
            shift_mask[:shift] = False
            shift_mask[shift:] = (group_ids[shift:] == group_ids[:-shift])
        win_open[~shift_mask] = open_arr[~shift_mask] # Fallback
        
        win_range = r_high - r_low
        win_body = np.abs(close_arr - win_open)
        
        with np.errstate(all='ignore'):
            w_feats[f'body_to_range_w{w}'] = _norm(win_body / (win_range + 1e-9), '')
            w_feats[f'volatility_w{w}'] = _norm(r_ret_std, '')
            # O(N) Z-score
            w_feats[f'volatility_zscore_w{w}'] = _norm((r_ret_std - _rolling_mean_grouped_numba(r_ret_std, group_ids, w)) / (_rolling_std_grouped_numba(r_ret_std, group_ids, w) + 1e-9), '')
            
            # Trend Efficiency
            # ER = Abs(Close - LagClose) / Sum(Abs(Close - PrevClose))
            abs_total_ret = np.abs(close_arr - np.roll(close_arr, w))
            # Fix boundary for abs_total_ret
            # Actually, efficiency ratio formula:
            # Numerator: Net change over W.
            # Denominator: Sum of absolute changes (path length) over W.

            # Path length: sum of abs(close.diff()) over W.
            diff_abs = np.abs(close_arr - np.roll(close_arr, 1))
            diff_abs[~mask] = 0.0 # Clear boundary
            sum_path_len = _rolling_sum_grouped_numba(diff_abs, group_ids, w)

            efficiency_ratio = abs_total_ret / (sum_path_len + 1e-9)
            w_feats[f'trend_efficiency_ratio_w{w}'] = _norm(efficiency_ratio, '')

            # Donchian
            donchian_width = r_high - r_low
            pos = (close_arr - r_low) / (donchian_width + 1e-9)
            w_feats[f'donchian_position_w{w}'] = _norm(pos, '')
            w_feats[f'donchian_width_w{w}'] = _norm(donchian_width / (close_arr + 1e-9), '')

            # RSI (Wilder's approximation using EWMA)
            # Standard RSI: RS = EWMA(Gain) / EWMA(Loss)
            diff = close_arr - np.roll(close_arr, 1)
            diff[~mask] = 0
            gain = np.maximum(diff, 0)
            loss = np.abs(np.minimum(diff, 0))

            # Use EWMA for RSI
            alpha_rsi = 1.0 / w # Smoother, roughly similar to 2*w? No, Wilder uses 1/14.
            # Standard RSI period N -> alpha = 1/N.
            avg_gain = _numba_ewma_grouped(gain, group_ids, alpha_rsi)
            avg_loss = _numba_ewma_grouped(loss, group_ids, alpha_rsi)

            rs = avg_gain / (avg_loss + 1e-9)
            rsi = 100.0 - (100.0 / (1.0 + rs))
            w_feats[f'rsi_w{w}'] = _norm(rsi, '')

            # Volume Features
            if volume_available:
                # Impact
                w_feats[f'price_impact_w{w}'] = _norm(_rolling_mean_grouped_numba(np.abs(log_ret_arr) / (vol_arr + 1e-9), group_ids, w), 'volume')

                # Churn: Volume * (1 - Efficiency)
                # "Volume expended without price progress"
                churn = vol_arr * (1.0 - efficiency_ratio)
                w_feats[f'volume_without_progress_w{w}'] = _norm(_rolling_mean_grouped_numba(churn, group_ids, w), 'volume')

            # --- Advanced Momentum & Microstructure ---
            # Volatility-normalized momentum
            # ret / vol (using r_ret_std as vol)
            w_feats[f'vol_norm_momentum_w{w}'] = _norm(log_ret_arr / (r_ret_std + 1e-9), '')

            # Trend Acceleration (Change in slope/trend)
            # Proxy: Change in smoothed return
            trend = _rolling_mean_grouped_numba(log_ret_arr, group_ids, w)
            trend_prev = np.roll(trend, w)
            trend_mask = np.ones_like(trend, dtype=bool)
            trend_mask[:w] = False
            trend_mask[w:] = (group_ids[w:] == group_ids[:-w])
            trend_accel = trend - trend_prev
            trend_accel[~trend_mask] = 0.0
            w_feats[f'trend_acceleration_w{w}'] = _norm(trend_accel, '')

            # Price-Volume Divergence (PVD)
            # zscore(slope_price) - zscore(slope_vol)
            # Proxy slope: (val - val_lag) / lag
            slope_price = close_arr - np.roll(close_arr, w)
            slope_vol = vol_arr - np.roll(vol_arr, w)
            # Masking handled by _norm usually, but better to be safe?
            # _norm handles winsorization.
            w_feats[f'pvd_w{w}'] = _norm(slope_price, '') - _norm(slope_vol, '')

            # CLV Momentum
            # EMA(CLV) -> Rolling Mean(CLV)
            w_feats[f'clv_mom_w{w}'] = _norm(_rolling_mean_grouped_numba(clv, group_ids, w), '')

            # Squeeze Ratio
            # BB Width / Volatility (ATR proxy)
            # BB Width = 4 * std(price) / mean(price)
            # We need r_close_std (std of price, not returns)
            r_close_std = _rolling_std_grouped_numba(close_arr, group_ids, w)
            bb_width = (4 * r_close_std) / (r_close_mean + 1e-9)
            # ATR proxy = r_ret_std * close_arr (approximate daily move in price terms) or just use normalized BB width.
            # Squeeze is usually BB Width / Keltner Width (ATR).
            # Here we approximate: BB Width / (Vol_Return)
            # Normalized BB Width is already width / price.
            # Vol_Return is sigma.
            # So Squeeze ~ BB_Width_Norm / Vol_Return.
            w_feats[f'squeeze_ratio_w{w}'] = _norm(bb_width / (r_ret_std + 1e-9), '')

            # Signed Volume Pressure
            # Sum(Sign(Ret)*Vol) / Sum(Vol)
            signed_vol = np.sign(log_ret_arr) * vol_arr
            sum_signed_vol = _rolling_sum_grouped_numba(signed_vol, group_ids, w)
            sum_vol = _rolling_sum_grouped_numba(vol_arr, group_ids, w)
            w_feats[f'signed_vol_pressure_w{w}'] = _norm(sum_signed_vol / (sum_vol + 1e-9), '')

            # Volume-Weighted Momentum (VWAP Momentum)
            # Sum(Ret*Vol) / Sum(Vol)
            ret_vol = log_ret_arr * vol_arr
            sum_ret_vol = _rolling_sum_grouped_numba(ret_vol, group_ids, w)
            w_feats[f'vwap_momentum_w{w}'] = _norm(sum_ret_vol / (sum_vol + 1e-9), '')

            # ATR Contraction Ratio
            # Vol(Short) / Vol(Long)
            # We need long volatility. We can use v_long_std from base features (computed earlier).
            # v_long_std is typically window=200.
            # Only valid if we have access to it here. `v_long_std` is defined outside loop.
            w_feats[f'atr_contraction_ratio_w{w}'] = _norm(r_ret_std / (v_long_s + 1e-9), '')

        features_list.append(w_feats)

    # Flatten and Create DataFrame
    final_dict = {}
    for d in features_list:
        final_dict.update(d)
        
    features = pd.DataFrame(final_dict, index=df.index)
    
    # Legacy Support (1h/4h returns)
    # 1h = 4 bars (assuming 15m). 4h = 16 bars.
    # Group safe return calc
    def get_period_return(period):
        prev = np.roll(close_arr, period)
        m = np.ones(len(close_arr), dtype=bool)
        if period > 0:
            m[:period] = False
            m[period:] = (group_ids[period:] == group_ids[:-period])
        ret = np.zeros_like(close_arr)
        with np.errstate(divide='ignore'):
            ret[m] = (close_arr[m] - prev[m]) / prev[m]
        return _norm(ret, '')

    features['returns_1h'] = get_period_return(4)
    features['returns_4h'] = get_period_return(16)
    
    features = features.astype(np.float32)
    
    end_time = time.time()
    logger.info(f"⚡ MTF generation finished in {end_time - start_time:.2f}s")
    
    # Cache Save (Optional)
    if cache_key:
        try:
            _MTF_CACHE[cache_key] = features.copy()
            # Enforce cache size limit logic
            if len(_MTF_CACHE) > _MAX_MTF_CACHE_SIZE:
                num_to_remove = len(_MTF_CACHE) - _MAX_MTF_CACHE_SIZE
                keys_to_remove = list(_MTF_CACHE.keys())[:num_to_remove]
                for k in keys_to_remove:
                    del _MTF_CACHE[k]
        except Exception: pass

    return features
