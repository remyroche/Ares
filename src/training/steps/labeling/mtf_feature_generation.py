"""
Multi-Timeframe Feature Generation Module.
Extracted and enhanced from feature_generation_meta_labeling_step.py.
"""
from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import logging
import hashlib
from pandas.util import hash_pandas_object

from src.features_common.transforms.scaling_normalization import winsorized_zscore_normalize
from src.utils.feature_common.volume_transforms import log1p_zscore_normalize
from src.utils.feature_common.atr_normalization import atr_normalize, should_use_atr_normalization, calculate_atr

# Global cache to prevent redundant MTF generation
_MTF_CACHE = {}
_MAX_MTF_CACHE_SIZE = 20  # Increased for multi-asset support

def clear_mtf_cache():
    """Explicitly clear the MTF feature cache to free memory."""
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
    # Minimal mock for njit if numba missing
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range
    NUMBA_AVAILABLE = False

logger = logging.getLogger(__name__)

# ===== NUMBA-OPTIMIZED FEATURE COMPUTATIONS =====

@njit(nogil=True, fastmath=True, parallel=True)
def _compute_candle_geometry_numba(
    open_p: np.ndarray, 
    high: np.ndarray, 
    low: np.ndarray, 
    close: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Numba-optimized candle geometry calculations."""
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
        
        eps = 1e-9
        if candle_range > eps:
            body_to_range[i] = real_body[i] / candle_range
            shadow_asymmetry[i] = (upper_shadow - lower_shadow) / candle_range
            clv[i] = ((close[i] - low[i]) - (high[i] - close[i])) / candle_range
        else:
            body_to_range[i] = 0.0
            shadow_asymmetry[i] = 0.0
            clv[i] = 0.0
    
    return body_to_range, shadow_asymmetry, clv, real_body

@njit(nogil=True, fastmath=True, parallel=True)
def _compute_volatility_features_numba(
    log_ret: np.ndarray,
    short_window: int = 20,
    long_window: int = 200
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Numba-optimized volatility regime calculations."""
    n = len(log_ret)
    vol_short = np.zeros(n, dtype=np.float32)
    vol_long_mean = np.zeros(n, dtype=np.float32)
    vol_long_std = np.zeros(n, dtype=np.float32)
    rv_z_short = np.zeros(n, dtype=np.float32)
    
    # Compute rolling volatility
    for i in range(short_window - 1, n):
        start_idx = i - short_window + 1
        vol_short[i] = np.std(log_ret[start_idx:i+1])
    
    # Compute long-term statistics
    min_periods = max(50, long_window // 4)
    
    for i in range(long_window - 1, n):
        start_idx = i - long_window + 1
        vol_slice = vol_short[start_idx:i+1]
        
        valid_count = np.sum(~np.isnan(vol_slice))
        if valid_count >= min_periods:
            vol_long_mean[i] = np.nanmean(vol_slice)
            vol_long_std[i] = np.nanstd(vol_slice)
    
    # Compute Z-scores
    for i in range(n):
        if vol_long_std[i] > 1e-8:
            rv_z_short[i] = (vol_short[i] - vol_long_mean[i]) / vol_long_std[i]
        else:
            rv_z_short[i] = 0.0
    
    return vol_short, vol_long_mean, vol_long_std, rv_z_short


@njit(nogil=True, fastmath=True, parallel=True)
def _rolling_mean_numba(values: np.ndarray, window: int) -> np.ndarray:
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
        if count > 0:
            output[i] = s / count
    return output

@njit(nogil=True, fastmath=True, parallel=True)
def _rolling_std_numba(values: np.ndarray, window: int) -> np.ndarray:
    n = len(values)
    output = np.full(n, np.nan, dtype=np.float32)
    if window <= 0: return output
    for i in prange(n):
        if i < window - 1: continue
        start_idx = i - window + 1
        s = 0.0
        ss = 0.0
        count = 0
        for j in range(start_idx, i + 1):
            v = values[j]
            if not np.isnan(v):
                s += v
                ss += v*v
                count += 1
        if count > 1:
            mean = s / count
            var = (ss - count * mean * mean) / (count - 1)
            if var > 0:
                output[i] = np.sqrt(var)
            else:
                output[i] = 0.0
    return output

@njit(nogil=True, fastmath=True, parallel=True)
def _rolling_min_numba(values: np.ndarray, window: int) -> np.ndarray:
    n = len(values)
    output = np.full(n, np.nan, dtype=np.float32)
    if window <= 0: return output
    for i in prange(n):
        if i < window - 1: continue
        start_idx = i - window + 1
        min_val = np.inf
        valid = False
        for j in range(start_idx, i + 1):
            v = values[j]
            if not np.isnan(v):
                if v < min_val:
                    min_val = v
                valid = True
        if valid:
            output[i] = min_val
    return output

@njit(nogil=True, fastmath=True, parallel=True)
def _rolling_max_numba(values: np.ndarray, window: int) -> np.ndarray:
    n = len(values)
    output = np.full(n, np.nan, dtype=np.float32)
    if window <= 0: return output
    for i in prange(n):
        if i < window - 1: continue
        start_idx = i - window + 1
        max_val = -np.inf
        valid = False
        for j in range(start_idx, i + 1):
            v = values[j]
            if not np.isnan(v):
                if v > max_val:
                    max_val = v
                valid = True
        if valid:
            output[i] = max_val
    return output

@njit(nogil=True, fastmath=True, parallel=True)
def _rolling_sum_numba(values: np.ndarray, window: int) -> np.ndarray:
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
        if count > 0:
            output[i] = s
    return output

@njit(nogil=True, fastmath=True, parallel=True)
def _rolling_mean_abs_dev_numba(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean absolute deviation from mean (Numba)."""
    n = len(values)
    output = np.full(n, np.nan, dtype=np.float32)
    if window <= 0:
        return output
    for i in prange(n):
        if i < window - 1:
            continue
        start_idx = i - window + 1
        s = 0.0
        count = 0
        for j in range(start_idx, i + 1):
            v = values[j]
            if not np.isnan(v):
                s += v
                count += 1
        if count == 0:
            continue
        mean_val = s / count
        mad_sum = 0.0
        for j in range(start_idx, i + 1):
            v = values[j]
            if not np.isnan(v):
                mad_sum += abs(v - mean_val)
        output[i] = mad_sum / count
    return output


def _rolling_weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Vectorized weighted rolling mean with NaN-safe weights."""
    values_arr = np.asarray(values, dtype=np.float32)
    weight_arr = np.asarray(weights, dtype=np.float32)
    weight_sum = weight_arr.sum()
    if weight_sum <= 0:
        return np.zeros_like(values_arr)
    weight_arr = weight_arr / weight_sum
    clean = np.nan_to_num(values_arr, nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.isfinite(values_arr).astype(float)
    # Use weights directly (no reversal) for causal convolution where weights[0] applies to values[t]
    conv = np.convolve(clean * mask, weight_arr, mode="full")[: len(values_arr)]
    denom = np.convolve(mask, weight_arr, mode="full")[: len(values_arr)]
    return np.divide(conv, denom, out=np.zeros_like(conv), where=denom > 0)


@njit(nogil=True, fastmath=True, parallel=True)
def _rolling_argmax_numba(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling argmax index within the window (Numba)."""
    n = len(values)
    output = np.full(n, np.nan, dtype=np.float32)
    if window <= 0:
        return output
    for i in prange(n):
        if i < window - 1:
            continue
        start_idx = i - window + 1
        max_idx = 0
        max_val = values[start_idx]
        for j in range(start_idx + 1, i + 1):
            if values[j] > max_val:
                max_val = values[j]
                max_idx = j - start_idx
        output[i] = max_idx
    return output


@njit(nogil=True, fastmath=True, parallel=True)
def _rolling_argmin_numba(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling argmin index within the window (Numba)."""
    n = len(values)
    output = np.full(n, np.nan, dtype=np.float32)
    if window <= 0:
        return output
    for i in prange(n):
        if i < window - 1:
            continue
        start_idx = i - window + 1
        min_idx = 0
        min_val = values[start_idx]
        for j in range(start_idx + 1, i + 1):
            if values[j] < min_val:
                min_val = values[j]
                min_idx = j - start_idx
        output[i] = min_idx
    return output

@njit(nogil=True, fastmath=True, parallel=True)
def _compute_wick_to_body_ratio_numba(
    open_p: np.ndarray,
    high: np.ndarray, 
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """Numba-optimized wick-to-body ratio calculation."""
    n = len(open_p)
    wb_ratio = np.zeros(n, dtype=np.float32)
    
    for i in prange(n):
        real_body = abs(close[i] - open_p[i])
        candle_range = high[i] - low[i]
        
        if candle_range > 1e-9:
            if real_body > 1e-9:
                total_wick = candle_range - real_body
                wb_ratio[i] = total_wick / real_body
            else:
                # Doji case: Infinite ratio, cap at high value or proportional to range
                # Using a proxy: range / epsilon
                wb_ratio[i] = candle_range * 10000.0 
        else:
            wb_ratio[i] = 0.0
    
    return wb_ratio

@njit(nogil=True, fastmath=True, parallel=True)
def _compute_displacement_ratio_numba(
    open_p: np.ndarray,
    high: np.ndarray,
    low: np.ndarray, 
    close: np.ndarray
) -> np.ndarray:
    """Numba-optimized displacement ratio calculation."""
    n = len(open_p)
    displacement = np.zeros(n, dtype=np.float32)
    
    for i in prange(n):
        candle_range = high[i] - low[i]
        if candle_range > 1e-9:
            displacement[i] = abs(close[i] - open_p[i]) / candle_range
        else:
            displacement[i] = 0.0
    
    return displacement

@njit(nogil=True, fastmath=True)
def _compute_dual_cusum_numba(
    log_ret_smooth: np.ndarray,
    residual_ret: np.ndarray,
    sigma: np.ndarray,
    er: np.ndarray,
    k: float,
    er_min: float
):
    n = len(log_ret_smooth)
    s_trend_pos = np.zeros(n, dtype=np.float32)
    s_trend_neg = np.zeros(n, dtype=np.float32)
    s_rev_pos = np.zeros(n, dtype=np.float32)
    s_rev_neg = np.zeros(n, dtype=np.float32)
    
    tp, tn = 0.0, 0.0
    rp, rn = 0.0, 0.0
    
    for t in range(n):
        if er[t] < er_min:
            tp, tn = 0.0, 0.0
            rp, rn = 0.0, 0.0
        else:
            cur_h = k * sigma[t]
            if cur_h <= 1e-9: cur_h = 1e-4
            
            # Trend
            tp = max(0.0, tp + log_ret_smooth[t])
            tn = min(0.0, tn + log_ret_smooth[t])
            
            if tp > cur_h: tp = 0.0
            if tn < -cur_h: tn = 0.0
            
            # Reversal
            rp = max(0.0, rp + residual_ret[t])
            rn = min(0.0, rn + residual_ret[t])
            
            if rp > cur_h: rp = 0.0
            if rn < -cur_h: rn = 0.0
            
        s_trend_pos[t] = tp
        s_trend_neg[t] = tn
        s_rev_pos[t] = rp
        s_rev_neg[t] = rn
        
    return s_trend_pos, s_trend_neg, s_rev_pos, s_rev_neg

class KalmanFilter1D:
    """
    Simple 1D Kalman filter for signal smoothing.

    State-space model:
        x_t = x_{t-1} + mu + w_t    (state evolution)
        y_t = x_t + v_t              (observation)

    Where w_t ~ N(0, Q) and v_t ~ N(0, R)
    """

    def __init__(self, Q: float = 1e-5, R: float = 0.01, initial_value: float = 0.0):
        """
        Args:
            Q: Process variance (smaller = smoother evolution)
            R: Observation variance (larger = more smoothing)
            initial_value: Initial state estimate
        """
        self.Q = Q  # Process variance
        self.R = R  # Observation variance
        self.x = initial_value  # State estimate
        self.P = 1.0  # State variance

    def update(self, measurement: float) -> Tuple[float, float]:
        """
        Update filter with new measurement.

        Returns:
            Tuple of (filtered_value, state_variance)
        """
        # Predict
        x_prior = self.x
        P_prior = self.P + self.Q

        # Numerical stability: bound P_prior to prevent infinity
        P_prior = np.clip(P_prior, 1e-12, 1e6)

        # Update
        # Prevent division by zero and ensure numerical stability
        denominator = P_prior + self.R
        if denominator <= 1e-12:
            K = 0.0  # No update if denominator is too small
        else:
            K = P_prior / denominator
            K = np.clip(K, 0.0, 1.0)  # Bound Kalman gain
        
        self.x = x_prior + K * (measurement - x_prior)
        self.P = (1 - K) * P_prior
        
        # Bound state variance to prevent infinity
        self.P = np.clip(self.P, 1e-12, 1e6)

        return self.x, self.P

    def filter_series(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Filter entire time series. (Numba Optimized)
        """
        if NUMBA_AVAILABLE:
            filtered, variances = _numba_kalman_filter(
                series.values.astype(np.float32),
                self.Q,
                self.R,
                self.x
            )
            return pd.Series(filtered, index=series.index), pd.Series(variances, index=series.index)
        else:
            filtered = []
            variances = []
            for val in series:
                f, v = self.update(val)
                filtered.append(f)
                variances.append(v)
            return pd.Series(filtered, index=series.index), pd.Series(variances, index=series.index)

@njit
def _numba_kalman_filter(data, Q, R, initial_value):
    n = len(data)
    filtered = np.zeros(n, dtype=np.float32)
    variances = np.zeros(n, dtype=np.float32)
    
    x = initial_value
    P = 1.0
    
    for i in range(n):
        val = data[i]
        if np.isnan(val):
            filtered[i] = np.nan
            variances[i] = np.nan
        else:
            # Predict
            x_prior = x
            P_prior = P + Q
            
            # Numerical stability: bound P_prior to prevent infinity
            if P_prior > 1e6:
                P_prior = 1e6
            elif P_prior < 1e-12:
                P_prior = 1e-12
            
            # Update
            # Prevent division by zero and ensure numerical stability
            denominator = P_prior + R
            if denominator <= 1e-12:
                K = 0.0  # No update if denominator is too small
            else:
                K = P_prior / denominator
                # Bound Kalman gain
                if K > 1.0:
                    K = 1.0
                elif K < 0.0:
                    K = 0.0
            
            x = x_prior + K * (val - x_prior)
            P = (1 - K) * P_prior
            
            # Bound state variance to prevent infinity
            if P > 1e6:
                P = 1e6
            elif P < 1e-12:
                P = 1e-12
            
            filtered[i] = x
            variances[i] = P
            
    return filtered, variances


def kalman_smooth_trend(prices: pd.Series, Q: float = 1e-5, R: float = 0.01) -> Tuple[pd.Series, pd.Series]:
    """
    Extract smoothed trend from prices using Kalman filter.

    Args:
        prices: Price series
        Q: Process variance (smaller = smoother)
        R: Observation variance (larger = more smoothing)

    Returns:
        Tuple of (trend, uncertainty)
    """
    kf = KalmanFilter1D(Q=Q, R=R, initial_value=prices.iloc[0] if len(prices) > 0 else 0.0)
    return kf.filter_series(prices)

def compute_dual_cusum_statistics(
    close: pd.Series,
    volume: Optional[pd.Series] = None,
    k: float = 0.12,
    er_min: float = 0.2,
    window_vol: int = 20,
    window_er: int = 10,
    Q: float = 1e-5,
    R: float = 0.01
) -> pd.DataFrame:
    """
    Compute continuous Dual CUSUM statistics.

    Generates:
    - S_trend_pos, S_trend_neg: Trend CUSUM accumulators
    - S_rev_pos, S_rev_neg: Reversal CUSUM accumulators
    - smoothed_return: Kalman smoothed returns
    - residual_return: Deviation from smoothed return

    Args:
        close: Close prices
        volume: Volume series (optional)
        k: Threshold sensitivity
        er_min: Minimum Efficiency Ratio to accumulate
        window_vol: Window for volatility
        window_er: Window for ER
        Q, R: Kalman parameters

    Returns:
        DataFrame with columns: ['S_trend_pos', 'S_trend_neg', 'S_rev_pos', 'S_rev_neg', 'smoothed_return', 'residual_return']
    """
    # 1. Compute log returns
    log_ret = np.log(close / close.shift(1)).fillna(0.0)

    # 2. Kalman Filter
    kf = KalmanFilter1D(Q=Q, R=R, initial_value=float(log_ret.iloc[0]))
    log_ret_smooth_raw, _ = kf.filter_series(log_ret)

    if not isinstance(log_ret_smooth_raw, pd.Series):
        log_ret_smooth_series = pd.Series(log_ret_smooth_raw, index=close.index).fillna(0.0)
    else:
        log_ret_smooth_series = log_ret_smooth_raw.fillna(0.0)

    # 3. ER and Volatility
    # Volatility on smoothed returns
    sigma = log_ret_smooth_series.rolling(window_vol, min_periods=1).std()

    # ER Calculation
    change = log_ret_smooth_series.rolling(window_er).sum().abs()
    volatility = log_ret_smooth_series.abs().rolling(window_er, min_periods=1).sum()
    ER = (change / (volatility + 1e-12)).fillna(0.0)

    # Threshold h_t
    # Simplified threshold for feature generation (ignoring complex regime modulation)
    h_t = (k * sigma).fillna(0.0)

    # 5. Residuals for Reversal
    expected_return = log_ret_smooth_series.rolling(window_vol, min_periods=1).mean()
    residual_ret = (log_ret_smooth_series - expected_return).fillna(0.0)

    # 6. CUSUM Loop
    if NUMBA_AVAILABLE:
        r_arr = log_ret_smooth_series.values.astype(np.float32)
        res_arr = residual_ret.values.astype(np.float32)
        sigma_arr = sigma.fillna(0.0).values.astype(np.float32)
        er_arr = ER.fillna(0.0).values.astype(np.float32)
        
        S_trend_pos_arr, S_trend_neg_arr, S_rev_pos_arr, S_rev_neg_arr = _compute_dual_cusum_numba(
            r_arr, res_arr, sigma_arr, er_arr, k, er_min
        )
    else:
        n = len(close)
        r_arr = log_ret_smooth_series.to_numpy()
        res_arr = residual_ret.to_numpy()
        h_arr = h_t.to_numpy()
        er_arr = ER.to_numpy()

        S_trend_pos_arr = np.zeros(n, dtype=np.float32)
        S_trend_neg_arr = np.zeros(n, dtype=np.float32)
        S_rev_pos_arr = np.zeros(n, dtype=np.float32)
        S_rev_neg_arr = np.zeros(n, dtype=np.float32)

        S_tp, S_tn = 0.0, 0.0
        S_rp, S_rn = 0.0, 0.0

        for t in range(n):
            if er_arr[t] < er_min:
                S_tp, S_tn = 0.0, 0.0
                S_rp, S_rn = 0.0, 0.0
            else:
                cur_h = h_arr[t]
                if np.isnan(cur_h) or cur_h <= 0:
                    cur_h = 1e-4

                # Trend CUSUM
                S_tp = max(0.0, S_tp + r_arr[t])
                S_tn = min(0.0, S_tn + r_arr[t])

                # Reset if threshold hit (simulating signal generation resets)
                if S_tp > cur_h: S_tp = 0.0
                if S_tn < -cur_h: S_tn = 0.0

                # Reversal CUSUM
                S_rp = max(0.0, S_rp + res_arr[t])
                S_rn = min(0.0, S_rn + res_arr[t])

                if S_rp > cur_h: S_rp = 0.0
                if S_rn < -cur_h: S_rn = 0.0

            S_trend_pos_arr[t] = S_tp
            S_trend_neg_arr[t] = S_tn
            S_rev_pos_arr[t] = S_rp
            S_rev_neg_arr[t] = S_rn

    return pd.DataFrame({
        'S_trend_pos': S_trend_pos_arr,
        'S_trend_neg': S_trend_neg_arr,
        'S_rev_pos': S_rev_pos_arr,
        'S_rev_neg': S_rev_neg_arr,
        'smoothed_return': log_ret_smooth_series,
        'residual_return': residual_ret
    }, index=close.index)


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI (Relative Strength Index)."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute MACD (Moving Average Convergence Divergence).

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def get_efficiency_ratio(close: pd.Series, window: int = 14) -> pd.Series:
    """Compute Kaufman Efficiency Ratio."""
    change = close.diff(window).abs()
    volatility = close.diff().abs().rolling(window).sum()
    return change / (volatility + 1e-9)

def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Compute Stochastic Oscillator (%K and %D)."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-9))
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent

def _rolling_mad_numpy(values: np.ndarray, window: int) -> np.ndarray:
    """Numpy strided implementation of Rolling MAD."""
    if window <= 1 or window > len(values):
        return np.full(len(values), np.nan)
        
    # Ensure float type and contiguous memory
    values = np.ascontiguousarray(values, dtype=np.float32)
    
    # Create strided view (N-w+1, w)
    shape = (len(values) - window + 1, window)
    strides = (values.strides[0], values.strides[0])
    
    try:
        windows = np.lib.stride_tricks.as_strided(
            values, shape=shape, strides=strides, writeable=False
        )
        
        # MAD calculation: mean(|x - mean(x)|) per window
        means = np.mean(windows, axis=1, keepdims=True)
        mad = np.mean(np.abs(windows - means), axis=1)
        
        # Pad beginning
        return np.concatenate([np.full(window - 1, np.nan), mad])
    except Exception:
        # Fallback if striding fails
        return np.full(len(values), np.nan)

def compute_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Compute Commodity Channel Index (CCI)."""
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=period).mean()
    
    # Mean deviation from the moving average
    if NUMBA_AVAILABLE:
        mean_dev_arr = _rolling_mean_abs_dev_numba(tp.values.astype(np.float32), period)
        mean_dev = pd.Series(mean_dev_arr, index=tp.index)
        if len(mean_dev) >= period:
            mean_dev.iloc[: period - 1] = np.nan
    else:
        # Vectorized Numpy Fallback
        mean_dev_arr = _rolling_mad_numpy(tp.values, period)
        mean_dev = pd.Series(mean_dev_arr, index=tp.index)
        
    cci = (tp - sma_tp) / (0.015 * mean_dev + 1e-9)
    return cci

def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute ADX, Plus DI, Minus DI."""
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
    """Compute Bollinger Bands (Upper, Middle, Lower, Width)."""
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    width = (upper - lower) / (middle + 1e-9)
    return upper, middle, lower, width

def compute_choppiness_index(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Choppiness Index."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_sum = tr.rolling(period).sum()
    high_max = high.rolling(period).max()
    low_min = low.rolling(period).min()

    ci = 100 * np.log10(atr_sum / (high_max - low_min + 1e-9)) / np.log10(period)
    return ci

def compute_cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """Compute Chaikin Money Flow."""
    mf_multiplier = ((close - low) - (high - close)) / (high - low + 1e-9)
    mf_volume = mf_multiplier * volume
    cmf = mf_volume.rolling(period).sum() / (volume.rolling(period).sum() + 1e-9)
    return cmf

def compute_force_index(close: pd.Series, volume: pd.Series, period: int = 13) -> pd.Series:
    """Compute Force Index."""
    fi = close.diff(1) * volume
    return fi.ewm(span=period).mean()

def compute_hurst_proxy(close: pd.Series, window: int = 100) -> pd.Series:
    """
    Vectorized Hurst Exponent proxy using Rolling R/S analysis on Returns.
    Approximation:
    R ~ Range of Price (Range of cumulative returns)
    S ~ Std Dev of Returns
    """
    # 1. Log returns
    ret = np.log(close / close.shift(1)).fillna(0.0)
    
    # 2. Rolling stats
    roll_close = close.rolling(window)
    roll_ret = ret.rolling(window)
    
    # Range of Price (proxy for Range of Cumulative Returns)
    # Normalized by current price to get percentage range
    # Or simply: Log Price Range
    # Let's use (Max - Min) / Mean as percentage range R
    r = (roll_close.max() - roll_close.min()) / (roll_close.mean() + 1e-9)
    
    # Standard Deviation of Returns S
    s = roll_ret.std()
    
    # RS Ratio
    # If volatility is 0, we have issues, but handled by 1e-9 in normalization usually.
    rs = r / (s + 1e-9)
    
    # Hurst = log(RS) / log(N)
    hurst = np.log(rs + 1e-9) / np.log(window)
    return hurst

def compute_parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """Compute Parkinson Volatility."""
    log_hl = np.log(high / (low + 1e-9)) ** 2
    return np.sqrt((1.0 / (4.0 * np.log(2.0))) * log_hl.rolling(window).mean())

def compute_donchian_channel(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Donchian Channel (Upper, Lower, Position)."""
    upper = high.rolling(window).max()
    lower = low.rolling(window).min()
    width = upper - lower
    # Position: 0 to 1
    position = (close - lower) / (width + 1e-9)
    return upper, lower, position


def compute_wick_to_body_ratio(open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Compute Wick-to-Body Ratio.
    (High - max(O,C)) / (max(O,C) - min(O,C))
    High ratio (Long upper wick) suggests Short Squeeze / Rejection.
    """
    max_oc = pd.concat([open_p, close], axis=1).max(axis=1)
    min_oc = pd.concat([open_p, close], axis=1).min(axis=1)
    
    upper_wick = high - max_oc
    body = max_oc - min_oc
    
    # Handle div/0 for doji candles by capping
    return upper_wick / (body.replace(0, 1e-9))

def compute_relative_volume_stress(volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Compute Relative Volume Stress (RVS).
    Volume / SMA(Volume, 20)
    """
    sma_vol = volume.rolling(window).mean()
    return volume / (sma_vol + 1e-9)

def compute_amihud_illiquidity(open_p: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Compute Amihud Illiquidity.
    abs(log(C/O)) / log1p(Volume)
    """
    log_ret = np.log(close / (open_p + 1e-9)).abs()
    return log_ret / np.log1p(volume + 1e-9)

def compute_displacement_ratio(open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Compute Displacement Ratio.
    (Close - Open) / (High - Low)
    Small body but huge range -> choppy/hunting stops.
    """
    body_signed = close - open_p
    rng = high - low
    return body_signed / (rng + 1e-9)

def compute_proxy_levels(
    high: pd.Series, 
    low: pd.Series, 
    close: pd.Series, 
    pivot_window: int = 30, 
    atr_window: int = 200, 
    k_factor: float = 1.0
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Liquidation Proxy Levels.
    Using 30-period pivots and 200-period ATR.
    
    Long Proxy (below price) = PivotLow - (k * ATR)
    Short Proxy (above price) = PivotHigh + (k * ATR)
    """
    # 1. Calculate Pivot Points (Rolling Min/Max) - representing swing points
    pivot_low = low.rolling(window=pivot_window).min()
    pivot_high = high.rolling(window=pivot_window).max()
    
    # 2. Calculate Long-Horizon ATR
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_long = tr.rolling(atr_window).mean()
    
    # 3. Calculate Proxies
    proxy_long = pivot_low - (k_factor * atr_long)
    proxy_short = pivot_high + (k_factor * atr_long)
    
    return proxy_long, proxy_short

def compute_garman_klass_volatility(open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """Compute Garman-Klass Volatility."""
    log_hl = np.log(high / (low + 1e-9)) ** 2
    log_co = np.log(close / (open_p + 1e-9)) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return np.sqrt(gk.rolling(window).mean())

def compute_volume_delta(close: pd.Series, open_p: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute Volume Delta Proxy."""
    # 1 if Close > Open (Buy), -1 if Close < Open (Sell)
    direction = np.sign(close - open_p)
    # If flat, assume 0 or maintain previous? 0 is safer for proxy.
    return direction * volume

def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute On-Balance Volume (OBV)."""
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (direction * volume).cumsum()

def compute_rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """Compute Rolling Z-Score."""
    roll = series.rolling(window)
    return (series - roll.mean()) / (roll.std() + 1e-9)

def compute_rolling_percentile(series: pd.Series, window: int = 20) -> pd.Series:
    """Compute Rolling Percentile (Rank)."""
    return series.rolling(window).rank(pct=True)

def compute_bars_since(condition: pd.Series) -> pd.Series:
    """Count bars since condition was True."""
    idx = pd.Series(np.arange(len(condition)), index=condition.index)
    last_occurrence = idx.where(condition).ffill()
    return (idx - last_occurrence).fillna(len(condition))

def compute_rogers_satchell_volatility(open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """Compute Rogers-Satchell Volatility."""
    h_c = np.log(high / (close + 1e-9))
    h_o = np.log(high / (open_p + 1e-9))
    l_c = np.log(low / (close + 1e-9))
    l_o = np.log(low / (open_p + 1e-9))
    rs_var = (h_c * h_o) + (l_c * l_o)
    return np.sqrt(rs_var.rolling(window).mean())

def compute_yang_zhang_volatility(open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """Compute Yang-Zhang Volatility."""
    # Log variances
    log_oc_prev = np.log(open_p / (close.shift(1) + 1e-9))
    log_co = np.log(close / (open_p + 1e-9))

    var_o = log_oc_prev.rolling(window).var()
    var_c = log_co.rolling(window).var()

    # Rogers-Satchell part
    # Re-using internal RS calc for efficiency or call function?
    # Calling function does rolling mean, we need that.
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
    """Numpy strided implementation of Rolling Argmax."""
    if window <= 1 or window > len(values):
        return np.full(len(values), np.nan)
    values = np.ascontiguousarray(values, dtype=np.float32)
    shape = (len(values) - window + 1, window)
    strides = (values.strides[0], values.strides[0])
    try:
        windows = np.lib.stride_tricks.as_strided(
            values, shape=shape, strides=strides, writeable=False
        )
        return np.concatenate([np.full(window - 1, np.nan), np.argmax(windows, axis=1)])
    except Exception:
        return np.full(len(values), np.nan)

def _rolling_argmin_numpy(values: np.ndarray, window: int) -> np.ndarray:
    """Numpy strided implementation of Rolling Argmin."""
    if window <= 1 or window > len(values):
        return np.full(len(values), np.nan)
    values = np.ascontiguousarray(values, dtype=np.float32)
    shape = (len(values) - window + 1, window)
    strides = (values.strides[0], values.strides[0])
    try:
        windows = np.lib.stride_tricks.as_strided(
            values, shape=shape, strides=strides, writeable=False
        )
        return np.concatenate([np.full(window - 1, np.nan), np.argmin(windows, axis=1)])
    except Exception:
        return np.full(len(values), np.nan)

def compute_aroon(high: pd.Series, low: pd.Series, window: int = 25) -> pd.Series:
    """Compute Aroon Oscillator."""
    # How many bars since high/low
    if NUMBA_AVAILABLE:
        high_idx_arr = _rolling_argmax_numba(high.values.astype(np.float32), window)
        low_idx_arr = _rolling_argmin_numba(low.values.astype(np.float32), window)
    else:
        # Optimized Numpy fallback
        high_idx_arr = _rolling_argmax_numpy(high.values, window)
        low_idx_arr = _rolling_argmin_numpy(low.values, window)

    high_idx = pd.Series(high_idx_arr, index=high.index)
    low_idx = pd.Series(low_idx_arr, index=low.index)
    
    # Pad NaN at start
    if len(high_idx) >= window:
        high_idx.iloc[: window - 1] = np.nan
        low_idx.iloc[: window - 1] = np.nan

    aroon_up = ((window - (window - 1 - high_idx)) / window) * 100
    aroon_down = ((window - (window - 1 - low_idx)) / window) * 100
    return aroon_up - aroon_down

def compute_ease_of_movement(high: pd.Series, low: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
    """Compute Ease of Movement (EOM)."""
    dm = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
    box_ratio = (volume + 1e-9) / ((high - low) + 1e-9)
    eom = dm / box_ratio
    return eom.rolling(window).mean()

def compute_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
    """Compute Money Flow Index (Volume-Weighted RSI)."""
    tp = (high + low + close) / 3
    rmf = tp * volume

    diff = tp.diff()
    pos_flow = rmf.where(diff > 0, 0)
    neg_flow = rmf.where(diff < 0, 0)

    pos_mean = pos_flow.rolling(window).mean()
    neg_mean = neg_flow.rolling(window).mean()

    mfi = 100 - (100 / (1 + (pos_mean / (neg_mean + 1e-9))))
    return mfi

def compute_fisher_transform(high: pd.Series, low: pd.Series, window: int = 10) -> pd.Series:
    """Compute Ehlers Fisher Transform."""
    mid = (high + low) / 2
    # Normalize to -1..1 over window
    rolling_min = low.rolling(window).min()
    rolling_max = high.rolling(window).max()

    val = 2 * ((mid - rolling_min) / (rolling_max - rolling_min + 1e-9) - 0.5)
    # Smooth with EMA
    val = val.ewm(alpha=0.33).mean()
    # Clamp
    val = val.clip(-0.999, 0.999)

    fisher = 0.5 * np.log((1 + val) / (1 - val))
    return fisher.ewm(alpha=0.5).mean()

def compute_hilbert_phase(series: pd.Series) -> pd.Series:
    """
    Compute Causal Hilbert Transform Phase Proxy.
    Uses Ehlers' Two-Bar Hilbert Transform logic to avoid future leakage.
    (Simple implementation: Quadrature = Price[t] - Price[t-2], InPhase = Price[t-1] - Price[t-3])
    """
    # Detrend
    centered = series - series.rolling(30).mean()

    # Ehlers Simple Hilbert Transform (Causal)
    # Q[t] = trend[t] - trend[t-2]?
    # Using simple difference proxy for quadrature
    # Q = I shifted by 90 deg.
    # Q[t] = 0.0962*P[t] + 0.5769*P[t-2] - 0.5769*P[t-4] - 0.0962*P[t-6] (Ehlers Coefficients)
    # But simple diff is often used as rough proxy.

    # Using Quadrature = series.diff(3) and InPhase = series.shift(1).diff(2)
    # This is a crude proxy.

    # Better: Use discrete Hilbert filter kernel (Type III or IV) but that's complex.
    # We will use the simplest "Ehlers Loop" inputs:
    # Q1 = (Price - Price[2])
    # I1 = (Price[1] - Price[3])
    # Phase = arctan(Q1 / I1)

    q1 = centered.diff(2)
    i1 = centered.shift(1).diff(2)

    phase = np.arctan2(q1, i1 + 1e-9)
    return phase.fillna(0)

def _align_to_features(arr: Any, n: int) -> np.ndarray:
    """Helper to align 1D array to feature index length. Pads at the BEGINNING (left)."""
    values = np.asarray(arr, dtype=np.float32)
    if len(values) == n:
        return values
    if len(values) > n:
        return values[-n:] # Take the last n elements
    padded = np.full(n, np.nan, dtype=np.float32)
    # Pad at the beginning (shift values to the right end)
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
) -> pd.DataFrame:
    """
    Create features for the meta-model with Multi-Timeframe support.
    Enhanced with Numba JIT optimizations for performance.
    """
    import time
    import gc
    start_time = time.time()
    
    # --- CACHE CHECK ---
    try:
        # Hash inputs (df index + close column specific values for speed + config)
        # Using hash_pandas_object on full DF might be slow but 20s vs 1s is worth it.
        # Include high/low/volume in hash to capture full OHLCV context
        h_idx = hashlib.md5(hash_pandas_object(df.index).values.tobytes()).hexdigest()
        
        # Hash key columns content
        cols_to_hash = []
        for c in ['close', 'high', 'low', 'open', 'volume', 'Close', 'High', 'Low', 'Open', 'Volume']:
            if c in df.columns:
                cols_to_hash.append(c)
        
        if cols_to_hash:
            h_content = hashlib.md5(hash_pandas_object(df[cols_to_hash]).values.tobytes()).hexdigest()
        else:
            h_content = "no_ohlcv"

        # Hash signals if present
        h_signals = "no_signals"
        if signals is not None and not signals.empty:
             h_signals = hashlib.md5(hash_pandas_object(signals).values.tobytes()).hexdigest()
        
        # Hash config
        config_str = f"{len(df)}_{windows}_{volume_available}_{include_raw_signals}_{use_kalman}_{horizon_bars}_{downsample_long_horizon}"
        cache_key = f"{h_idx}_{h_content}_{h_signals}_{config_str}"
        
        if cache_key in _MTF_CACHE:
            logger.info(f"⚡ [MTF Cache] Returning pre-computed features for {len(df)} rows")
            return _MTF_CACHE[cache_key].copy()
            
    except Exception as e:
        logger.warning(f"⚠️ Cache check failed: {e}")
        cache_key = None
    
    # Add progress tracking
    logger.info(f"🔍 Starting MTF feature generation for {len(df)} rows...")
    
    # Preserve original index for alignment if downsampling occurs
    input_index = df.index

    # Hard-align df and signals to a shared tail window
    len_df = len(df)
    len_sig = len(signals)

    if len_df != len_sig:
        target_len = min(len_df, len_sig)
        if len_df > target_len: df = df.iloc[-target_len:, :]
        if len_sig > target_len: signals = signals.iloc[-target_len:, :]

    # Reset index to avoid duplicate index issues
    index_mismatch = not df.index.equals(signals.index)
    df_duplicates = bool(df.index.has_duplicates)
    signals_duplicates = bool(signals.index.has_duplicates)
    
    if index_mismatch or df_duplicates or signals_duplicates:
        df = df.reset_index(drop=True)
        signals = signals.reset_index(drop=True)

    features = pd.DataFrame(index=df.index)
    n_features = len(features)

    original_index = df.index

    # Log optimization status
    if NUMBA_AVAILABLE:
        logger.info(f"🚀 Using Numba JIT optimizations for {len(df)} rows...")
    else:
        logger.warning(f"⚠️  Numba not available, using pandas fallback for {len(df)} rows...")

    # Downsample for long horizon to reduce rolling window cost
    is_downsampled = False
    if (
        downsample_long_horizon
        and isinstance(horizon_bars, (int, float))
        and horizon_bars >= 48
        and isinstance(df.index, pd.DatetimeIndex)
        and len(df) > 0
    ):
        try:
            median_delta = df.index.to_series().diff().median()
            if pd.notna(median_delta) and median_delta <= pd.Timedelta("20min"):
                print("🧭 Downsampling long-horizon features to 60m bars for efficiency")
                agg_map = {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
                # Handle case variants
                for col in df.columns:
                    lower_col = col.lower()
                    if lower_col in agg_map and col not in agg_map:
                        agg_map[col] = agg_map[lower_col]
                
                df = df.resample("60min").agg({col: agg_map.get(col, "last") for col in df.columns}).dropna()
                signals = signals.resample("60min").last().ffill().reindex(df.index)
                is_downsampled = True
        except Exception as exc:
            print(f"⚠️  Downsampling skipped due to error: {exc}")

    # Memory optimization: limit data size for feature generation
    MAX_FEATURE_ROWS = 1_000_000  # Increased to prevent index mismatches in chunked runs
    if len(df) > MAX_FEATURE_ROWS:
        print(f"🧠 Limiting feature generation from {len(df)} to {MAX_FEATURE_ROWS} rows for memory efficiency")
        df = df.tail(MAX_FEATURE_ROWS)
        signals = signals.tail(MAX_FEATURE_ROWS)
        features = pd.DataFrame(index=df.index)
        print(f"✅ Limited to {len(df)} rows for feature generation")
    
    # IMPORTANT: Update n_features after limiting
    n_features = len(features)
    
    long_horizon = bool(isinstance(horizon_bars, (int, float)) and horizon_bars >= 48)
    if long_horizon:
        print(f"🧭 Long horizon detected ({int(horizon_bars)} bars); disabling heavy feature families")

    # Additional memory optimization: reduce windows for large datasets ONLY if not explicitly provided or if default
    # If user provided specific windows (not the default list), respect them unless forced.
    default_windows = [10, 20, 50, 100, 150, 200]
    if windows == default_windows and len(df) > 10000:
        print(f"🔧 Reducing feature windows for large dataset ({len(df)} rows)")
        windows = [10, 20, 50]  # Reduced windows for memory efficiency
        print(f"📊 Using reduced windows: {windows}")
    elif windows != default_windows:
        print(f"📊 Using user-provided windows: {windows}")
    else:
        # Default case, small dataset
        pass

    if not NUMBA_AVAILABLE and len(df) > 10000:
        print("⚠️  Numba unavailable for long-horizon run; disabling heavy interactions and limiting windows")

    # Limit interactions/cross-timeframe ratios to lower-timeframe windows when data is large
    interaction_windows = set(windows)
    if long_horizon:
        interaction_windows = set()
        print("🧩 Disabling interaction/cross-timeframe features for long-horizon runs")
    elif len(df) > 10000:
        interaction_windows = {w for w in windows if w <= 50}
        if len(interaction_windows) < len(windows):
            print(f"🧩 Limiting interaction/cross-timeframe features to windows <= 50: {sorted(interaction_windows)}")

    enable_tail_risk = not long_horizon

    # Initialize accumulation lists for batch concatenation
    feature_chunks: List[pd.DataFrame] = []
    chunk_window_size = 2

    def _downcast_float32(frame: pd.DataFrame) -> pd.DataFrame:
        float_cols = frame.select_dtypes(include=['float64']).columns
        if len(float_cols) > 0:
            frame[float_cols] = frame[float_cols].astype(np.float32)
        return frame

    def _flush_family(window_feature_map: Dict[str, np.ndarray], start_idx: int) -> None:
        nonlocal features
        if len(window_feature_map) <= start_idx:
            return
        family_keys = list(window_feature_map.keys())[start_idx:]
        family_data = {key: window_feature_map.pop(key) for key in family_keys}
        family_df = pd.DataFrame(family_data, index=features.index)
        family_df = _downcast_float32(family_df)
        feature_chunks.append(family_df)
        if len(feature_chunks) >= chunk_window_size:
            features = pd.concat([features] + feature_chunks, axis=1)
            feature_chunks.clear()
            features = _downcast_float32(features)
    
    # NOW extract columns and compute returns on the LIMITED data
    close_col = None
    for col in ['close', 'Close', 'CLOSE']:
        if col in df.columns:
            close_col = col
            break
    
    if close_col is None:
        for col in df.columns:
            if col.endswith('_close') or col.endswith('_Close') or col.endswith('_CLOSE'):
                close_col = col
                break
    
    if close_col is None:
        raise KeyError(f"None of ['close', 'Close', 'CLOSE'] or prefixed variants found in columns: {list(df.columns)}")
    
    close = df[close_col]
    log_ret = close.pct_change()

    def _norm(data: Union[pd.Series, np.ndarray], name: str) -> np.ndarray:
        """
        Normalize feature data based on feature name and type.
        """
        if not isinstance(data, pd.Series):
             if len(data) == len(df):
                 series = pd.Series(data, index=df.index)
             else:
                 series = pd.Series(data)
        else:
            series = data

        name_lower = name.lower()
        exclude_keywords = [
            'ratio', 'trend', 'corr', 'rel', 'per', 'norm',
            'imbalance', 'signed', 'pressure', 'delta', 'cvd', 'spike',
            'asymmetry', 'location', 'oscillator', 'phase', 'flag', 'count'
        ]

        if 'volume' in name_lower and not any(x in name_lower for x in exclude_keywords):
             return log1p_zscore_normalize(series, window=600).fillna(0).to_numpy(dtype=np.float32)

        if should_use_atr_normalization(name):
             if any(x in name_lower for x in ['position', 'ratio', 'score', 'percent', 'pct', 'oscillator', 'index', 'flag', 'count']):
                 pass
             else:
                 # Use robust variables if possible, but _norm is defined before high/low/close extraction.
                 # So we need to access df robustly here too.
                 # However, high/low/close are available in outer scope if we move _norm definition after extraction.
                 # Or we access them robustly here.
                 # Find robust column names (handle multi-timeframe prefixes)
                 def _find_col(df, suffixes):
                     for suffix in suffixes:
                         for col in df.columns:
                             if col == suffix or col.endswith(f'_{suffix}'):
                                 return col
                     return None
                 
                 _high_col = _find_col(df, ['high', 'High', 'HIGH'])
                 _low_col = _find_col(df, ['low', 'Low', 'LOW'])
                 _close_col = _find_col(df, ['close', 'Close', 'CLOSE'])
                 
                 _high = df[_high_col] if _high_col else close
                 _low = df[_low_col] if _low_col else close
                 _close = df[_close_col] if _close_col else close
                 
                 return atr_normalize(series, _high, _low, _close, window=14).fillna(0).to_numpy(dtype=np.float32)

        return winsorized_zscore_normalize(series, window=600).fillna(0).to_numpy(dtype=np.float32)

    # Handle prefixed high/low columns
    high_col = None
    low_col = None
    for suffix in ['high', 'High', 'HIGH']:
        for col in df.columns:
            if col == suffix or col.endswith(f'_{suffix}'):
                high_col = col
                break
        if high_col:
            break
    
    for suffix in ['low', 'Low', 'LOW']:
        for col in df.columns:
            if col == suffix or col.endswith(f'_{suffix}'):
                low_col = col
                break
        if low_col:
            break
    
    high = df[high_col] if high_col else close
    low = df[low_col] if low_col else close

    # Add log_ret feature (already computed above)
    features['log_ret'] = _align_to_features(_norm(log_ret, 'log_ret'), n_features)

    if 'open' in df.columns:
        open_p = df['open']
    elif 'Open' in df.columns:
        open_p = df['Open']
    else:
        open_p = close

    # Check for volume column (case-insensitive)
    vol_col = None
    if 'volume' in df.columns:
        vol_col = 'volume'
    elif 'Volume' in df.columns:
        vol_col = 'Volume'

    # Robust check for volume_available to avoid ambiguity errors
    vol_avail_check = False
    try:
        if isinstance(volume_available, (bool, np.bool_)):
            vol_avail_check = bool(volume_available)
        else:
            vol_avail_check = True # Default if passed something else
    except Exception:
        vol_avail_check = True

    if vol_avail_check and vol_col is not None:
        volume = pd.to_numeric(df[vol_col], errors='coerce').ffill().bfill().fillna(0.0)
        volume_available = volume.notna().any()
    else:
        volume = pd.Series(1.0, index=df.index, dtype='float32')
        volume_available = False

    # Define numpy arrays for Numba usage once, to avoid scope issues
    if NUMBA_AVAILABLE:
        # Pre-allocate float32 arrays
        # Use extracted series directly
        open_arr = open_p.values.astype(np.float32)
        high_arr = high.values.astype(np.float32)
        low_arr = low.values.astype(np.float32)
        close_arr = close.values.astype(np.float32)
    else:
        open_arr = high_arr = low_arr = close_arr = None

    # ===== 0. ORTHOGONAL CUSUM & EFFICIENCY FEATURES (NEW) =====
    print("📊 Computing CUSUM & Efficiency features...")
    # Dual CUSUM Continuous Stats
    cusum_stats = compute_dual_cusum_statistics(
        close,
        volume if volume_available else None,
        k=0.12, window_vol=20, window_er=10, er_min=0.2
    )
    features['cusum_trend_pos'] = _align_to_features(_norm(cusum_stats['S_trend_pos'], 'cusum_trend_pos'), n_features)
    features['cusum_trend_neg'] = _align_to_features(_norm(cusum_stats['S_trend_neg'], 'cusum_trend_neg'), n_features)
    features['cusum_rev_pos'] = _align_to_features(_norm(cusum_stats['S_rev_pos'], 'cusum_rev_pos'), n_features)
    features['cusum_rev_neg'] = _align_to_features(_norm(cusum_stats['S_rev_neg'], 'cusum_rev_neg'), n_features)
    features['smoothed_return'] = _align_to_features(_norm(cusum_stats['smoothed_return'], 'smoothed_return'), n_features)
    features['residual_return'] = _align_to_features(_norm(cusum_stats['residual_return'], 'residual_return'), n_features)

    # Rolling Efficiency Ratios
    # Ensure log_ret and er_30 are available/calculated
    er_30 = get_efficiency_ratio(close, 30)
    features['rolling_efficiency_ratio'] = _align_to_features(_norm(er_30.rolling(30).mean(), 'rolling_efficiency_ratio'), n_features)
    features['efficiency_ratio_volatility'] = _align_to_features(_norm(er_30.rolling(30).std(), 'efficiency_ratio_volatility'), n_features)

    # Geometry-Specific Lag Features from Signals
    signal_cols = [c for c in signals.columns if 'signal' in c.lower() or 'consensus' in c.lower()]
    abs_signal_sum = pd.Series(0.0, index=df.index, dtype='float32')

    # Use the log_ret already computed above
    vol_short_local = log_ret.rolling(window=20).std()

    print(f"📈 Processing {len(signal_cols)} signal columns...")
    for i, col in enumerate(signal_cols):
        if i % 10 == 0:  # Progress update every 10 signals
            print(f"   Processing signal {i+1}/{len(signal_cols)}: {col}")
            
        # IMPORTANT: Use signals that have been limited to match df
        sig_series = signals[col].fillna(0)
        if len(sig_series) > len(df):
            sig_series = sig_series.tail(len(df))
        
        abs_signal_sum += sig_series.abs()

        # Lags
        for lag in [1, 2, 3]:
            features[f'{col}_lag_{lag}'] = _align_to_features(sig_series.shift(lag), n_features)

        # Cumulative/Rolling
        features[f'{col}_rolling_sum_3'] = _align_to_features(sig_series.rolling(3).sum(), n_features)
        features[f'{col}_rolling_std_5'] = _align_to_features(sig_series.rolling(5).std(), n_features)

        # Interactions
        features[f'{col}_x_ret'] = _align_to_features(_norm(sig_series * log_ret, f'{col}_x_ret'), n_features)
        features[f'{col}_x_vol'] = _align_to_features(_norm(sig_series * vol_short_local, f'{col}_x_vol'), n_features)

        # Recent Performance (Momentum of signals)
        perf = (sig_series.shift(1) * log_ret).rolling(10).mean()
        features[f'{col}_recent_performance'] = _align_to_features(_norm(perf, f'{col}_recent_performance'), n_features)

    # Cross-Geometry Features
    features['signal_cluster_count'] = _align_to_features(_norm(abs_signal_sum, 'signal_cluster_count'), n_features)

    # ===== 1. CANDLE GEOMETRY & MICRO-SENTIMENT (BASE) =====
    # Use Numba-optimized calculations if available
    if NUMBA_AVAILABLE:
        # Convert to numpy arrays for Numba processing
        open_arr = open_p.values.astype(np.float32)
        high_arr = high.values.astype(np.float32)
        low_arr = low.values.astype(np.float32)
        close_arr = close.values.astype(np.float32)
        
        # Optimized candle geometry calculations
        body_to_range_arr, shadow_asymmetry_arr, clv_arr, real_body_arr = _compute_candle_geometry_numba(
            open_arr, high_arr, low_arr, close_arr
        )
        
        features['body_to_range'] = _align_to_features(_norm(body_to_range_arr, 'body_to_range'), n_features)
        features['shadow_asymmetry'] = _align_to_features(_norm(shadow_asymmetry_arr, 'shadow_asymmetry'), n_features)
        features['close_location_value'] = _align_to_features(_norm(clv_arr, 'close_location_value'), n_features)
    else:
        # Fallback to original pandas calculations
        candle_range = high - low
        upper_shadow = high - pd.concat([open_p, close], axis=1).max(axis=1)
        lower_shadow = pd.concat([open_p, close], axis=1).min(axis=1) - low
        real_body = (close - open_p).abs()

        features['body_to_range'] = _align_to_features(_norm(real_body / (candle_range + 1e-9), 'body_to_range'), n_features)
        features['shadow_asymmetry'] = _align_to_features(_norm((upper_shadow - lower_shadow) / (candle_range + 1e-9), 'shadow_asymmetry'), n_features)

        clv = ((close - low) - (high - close)) / (candle_range + 1e-9)
        features['close_location_value'] = _align_to_features(_norm(clv, 'close_location_value'), n_features)

    # ===== 2. VOLATILITY REGIME (GLOBAL) =====
    # Use Numba-optimized volatility calculations if available
    if NUMBA_AVAILABLE:
        # Optimized volatility calculations
        vol_short_arr, vol_long_mean_arr, vol_long_std_arr, rv_z_short_arr = _compute_volatility_features_numba(
            log_ret.values.astype(np.float32)
        )
        
        features['rv_z_short'] = _norm(rv_z_short_arr, 'rv_z_short')
        
        # Fix: Compute slope with proper alignment (lag 5 difference)
        # np.diff is recursive difference, not lag. We need lag difference.
        vol_slope = np.full(len(vol_short_arr), np.nan, dtype=np.float32)
        if len(vol_short_arr) > 5:
            vol_slope[5:] = vol_short_arr[5:] - vol_short_arr[:-5]
            
        features['volatility_trend_slope'] = _align_to_features(_norm(vol_slope, 'volatility_trend_slope'), n_features)
    else:
        # Fallback to original pandas calculations
        vol_short_20 = log_ret.rolling(window=20).std()
        vol_long_mean = vol_short_20.rolling(window=200, min_periods=50).mean()
        vol_long_std = vol_short_20.rolling(window=200, min_periods=50).std()
        rv_z_short = (vol_short_20 - vol_long_mean) / (vol_long_std + 1e-8)
        features['rv_z_short'] = _norm(rv_z_short.fillna(0.0), 'rv_z_short')

        # Volatility Trend Slope
        features['volatility_trend_slope'] = _align_to_features(_norm(vol_short_20.diff(5), 'volatility_trend_slope'), n_features)

    # ===== 3. LIQUIDATION SPECIALIST FEATURES (NEW) =====
    # Wick-to-Body - Use Numba optimization if available
    if NUMBA_AVAILABLE:
        wb_ratio_arr = _compute_wick_to_body_ratio_numba(
            open_arr, high_arr, low_arr, close_arr
        )
        features['wick_to_body_ratio'] = _align_to_features(_norm(wb_ratio_arr, 'wick_to_body_ratio'), n_features)
        
        displacement_arr = _compute_displacement_ratio_numba(
            open_arr, high_arr, low_arr, close_arr
        )
        features['displacement_ratio'] = _align_to_features(_norm(displacement_arr, 'displacement_ratio'), n_features)
    else:
        # Fallback to original calculations
        wb_ratio = compute_wick_to_body_ratio(open_p, high, low, close)
        features['wick_to_body_ratio'] = _align_to_features(_norm(wb_ratio, 'wick_to_body_ratio'), n_features)
        
        displacement = compute_displacement_ratio(open_p, high, low, close)
        features['displacement_ratio'] = _align_to_features(_norm(displacement, 'displacement_ratio'), n_features)
    
    # RVS
    rvs = compute_relative_volume_stress(volume, window=20)
    features['relative_volume_stress'] = _align_to_features(_norm(rvs, 'relative_volume_stress'), n_features)
    
    # Amihud
    amihud = compute_amihud_illiquidity(open_p, close, volume)
    features['amihud_illiquidity'] = _align_to_features(_norm(amihud, 'amihud_illiquidity'), n_features)
    
    # Proxy Levels & Distance
    # User Spec: Pivot=30, ATR=200, k=1.0 (mid-range of 0.5-1.5)
    proxy_long, proxy_short = compute_proxy_levels(high, low, close, pivot_window=30, atr_window=200, k_factor=1.0)
    
    # Distance to Proxy (normalized by ATR(14))
    # dist_to_proxy = (Price - Proxy) / ATR(14)
    # We use Close for current price
    atr_14_local = atr_14 if 'atr_14' in locals() else (high - low).rolling(14).mean() # Fallback approximation if atr_14 not computed yet

    def _sigmoid(arr, scale=1.0, shift=0.0):
        z = np.clip((arr - shift) / (scale + 1e-9), -20, 20)
        return 1.0 / (1.0 + np.exp(-z))
    
    # For Long Squeeze (price drops to proxy): Distance is positive if Price > Proxy
    dist_to_long_proxy = (close - proxy_long) / (atr_14_local + 1e-9)
    features['dist_to_long_proxy'] = _align_to_features(_norm(dist_to_long_proxy, 'dist_to_long_proxy'), n_features)
    
    # For Short Squeeze (price rises to proxy): Distance is positive if Proxy > Price
    dist_to_short_proxy = (proxy_short - close) / (atr_14_local + 1e-9)
    features['dist_to_short_proxy'] = _align_to_features(_norm(dist_to_short_proxy, 'dist_to_short_proxy'), n_features)

    # Liquidation Trigger Signal (Logic Trace)
    # IF (Price < Long_Proxy_Level) AND (RVS > 2.0)
    # Note: "Causal_Specialist_Signal" dependency is external/downstream, so we calculate the 'Liquidation Condition' component here.
    # We capture the "Liquidation Zone" state.
    
    in_long_liq_zone = (close < proxy_long).astype(float)
    high_stress = (rvs > 2.0).astype(float)
    
    # Composite feature: Liquidation Risk (Long)
    # 1.0 = In Zone + High Stress
    liq_risk_long = in_long_liq_zone * high_stress
    features['liquidation_risk_long'] = _align_to_features(_norm(liq_risk_long, 'liquidation_risk_long'), n_features)

    # Soft-encoded liquidation risk (probabilistic zone + stress intensity)
    zone_distance = (proxy_long - close) / (atr_14_local + 1e-9)
    long_zone_soft = _sigmoid(zone_distance, scale=1.0)
    stress_soft = _sigmoid(rvs, scale=0.5, shift=2.0)
    liq_risk_long_soft = (long_zone_soft * stress_soft).fillna(0.0)
    features['liquidation_risk_long_soft'] = _align_to_features(
        _norm(liq_risk_long_soft, 'liquidation_risk_long_soft'), n_features
    )

    # ===== INFORMATION_SPECIALIST FEATURES (MTF) =====
    print("📊 Computing INFORMATION_SPECIALIST features across multiple timeframes...")
    
    # Add INFORMATION_SPECIALIST features for each window in the MTF framework
    for w in windows:
        # volume_burst: Volume burst detection (sudden information arrival) - MTF version
        if volume_available:
            volume_ma = volume.rolling(w).mean()
            volume_std = volume.rolling(w).std()
            volume_burst = (volume - volume_ma) / (volume_std + 1e-9)
            features[f'volume_burst_w{w}'] = _align_to_features(_norm(volume_burst, f'volume_burst_w{w}'), n_features)
        
        # surprise_magnitude: Deviation from expected returns - MTF version
        returns_mean = log_ret.rolling(w).mean()
        returns_std = log_ret.rolling(w).std()
        surprise_magnitude = abs(log_ret - returns_mean) / (returns_std + 1e-9)
        features[f'surprise_magnitude_w{w}'] = _align_to_features(_norm(surprise_magnitude, f'surprise_magnitude_w{w}'), n_features)
        
        # cumulative_info_impact: Cumulative information impact - MTF version
        cumulative_impact_window = max(3, w // 4)  # Scale impact window with timeframe
        cumulative_info_impact = surprise_magnitude.rolling(cumulative_impact_window).sum()
        features[f'cumulative_info_impact_w{w}'] = _align_to_features(_norm(cumulative_info_impact, f'cumulative_info_impact_w{w}'), n_features)
        
        # trend_persistence: Trend persistence (information momentum) - MTF version
        rolling_min = log_ret.rolling(w).min()
        rolling_max = log_ret.rolling(w).max()
        trend_persistence = ((rolling_min > 0) | (rolling_max < 0)).astype(float)
        features[f'trend_persistence_w{w}'] = _align_to_features(_norm(trend_persistence, f'trend_persistence_w{w}'), n_features)

        same_sign = (((rolling_min >= 0) & (rolling_max >= 0)) | ((rolling_min <= 0) & (rolling_max <= 0))).astype(float)
        magnitude_ratio = np.minimum(rolling_max.abs(), rolling_min.abs()) / (
            np.maximum(rolling_max.abs(), rolling_min.abs()) + 1e-9
        )
        trend_persistence_soft = (same_sign * magnitude_ratio).fillna(0.0)
        features[f'trend_persistence_soft_w{w}'] = _align_to_features(
            _norm(trend_persistence_soft, f'trend_persistence_soft_w{w}'), n_features
        )
        
        # info_acceleration: Rate of change of information flow - MTF version
        info_flow = abs(log_ret)
        ema_span = max(3, w // 8)  # Scale EMA span with timeframe
        info_acceleration = info_flow.ewm(span=ema_span).mean().diff()
        features[f'info_acceleration_w{w}'] = _align_to_features(_norm(info_acceleration, f'info_acceleration_w{w}'), n_features)
        
        # Additional MTF-specific INFORMATION_SPECIALIST features
        # Information intensity: Normalized surprise magnitude by timeframe
        info_intensity = surprise_magnitude * np.sqrt(w)  # Scale by sqrt of window
        features[f'info_intensity_w{w}'] = _align_to_features(_norm(info_intensity, f'info_intensity_w{w}'), n_features)
        
        # Information decay: How quickly information dissipates over timeframe
        decay_weights = np.exp(-np.arange(w) / max(w, 1))
        decay_weights = decay_weights / decay_weights.sum()

        info_decay_arr = _rolling_weighted_mean(
            surprise_magnitude.to_numpy(dtype=float), decay_weights
        )
        info_decay = pd.Series(info_decay_arr, index=surprise_magnitude.index)
        features[f'info_decay_w{w}'] = _align_to_features(_norm(info_decay, f'info_decay_w{w}'), n_features)
        
        # Cross-timeframe information ratio: Compare current to longer timeframe
        if w < max(windows):
            longer_w = max([win for win in windows if win > w])
            if longer_w in windows:
                # This would require access to longer timeframe features, simplified for now
                cross_tf_ratio = surprise_magnitude / (log_ret.rolling(longer_w).std() + 1e-9)
                features[f'cross_tf_info_ratio_w{w}'] = _align_to_features(_norm(cross_tf_ratio, f'cross_tf_info_ratio_w{w}'), n_features)
    
    print(f"✅ Generated INFORMATION_SPECIALIST features for {len(windows)} timeframes: {windows}")
    # ============================================

    # ===== SPECIALIST SCALAR FEATURES =====
    specialist_cols = [
        "risk_score", "path_risk_score", "macro_trend_score_continuous",
        "mr_probability_dense", "mr_probability", "mr_raw_score",
        "mr_trend_state", "mr_trend_is_mr", "sr_labeling_xgb_prob",
        "vol_force_scalar", "smc_predicted"
    ]
    for col in df.columns:
        if col in specialist_cols or col.startswith("mr_") or col.startswith("smc_"):
            if col not in features.columns:
                features[col] = _norm(df[col], col)

    # ===== KALMAN & BASE INDICATORS =====
    df_local = df.copy()
    df_local['rsi'] = compute_rsi(close, period=14)
    df_local['momentum_30'] = close.pct_change(30)
    features['momentum_30'] = _align_to_features(_norm(df_local['momentum_30'], 'momentum_30'), n_features)

    # Kalman
    if use_kalman:
        kalman_trend, kalman_uncertainty = kalman_smooth_trend(close, Q=1e-4, R=0.1)
        features['kalman_trend'] = _align_to_features(_norm(kalman_trend, 'kalman_trend'), n_features)
        features['kalman_uncertainty'] = _align_to_features(_norm(kalman_uncertainty, 'kalman_uncertainty'), n_features)

        kf_rsi = KalmanFilter1D(Q=1e-4, R=0.1, initial_value=50.0)
        kalman_rsi, _ = kf_rsi.filter_series(df_local['rsi'])
        features['rsi_kalman'] = _align_to_features(_norm(kalman_rsi, 'rsi_kalman'), n_features)
    else:
        features['rsi'] = _align_to_features(_norm(df_local['rsi'], 'rsi'), n_features)

    # Base ATR
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = true_range.rolling(14).mean()
    features['atr_14'] = _align_to_features(_norm(atr_14, 'atr_14'), n_features)

    # Base Sharpe
    roll_mean_50 = log_ret.rolling(50).mean()
    roll_std_50 = log_ret.rolling(50).std()
    features['rolling_sharpe'] = _align_to_features(_norm((roll_mean_50/(roll_std_50+1e-9)).fillna(0), 'rolling_sharpe'), n_features)

    # Kaufman ER (Base)
    features['kaufman_efficiency_ratio'] = _align_to_features(_norm(get_efficiency_ratio(close, 30), 'kaufman_efficiency_ratio'), n_features)
    features = _downcast_float32(features)

    # ===== 6. MULTI-TIMEFRAME FEATURES =====
    # User requested multi-timeframe features
    print(f" Computing multi-timeframe features for windows: {windows}")

    feature_chunks: List[pd.DataFrame] = []
    chunk_window_size = 2

    last_heartbeat = start_time
    for i, w in enumerate(windows):
        if i % 2 == 0:  # Progress update every 2 windows
            print(f"   Processing window {i+1}/{len(windows)}: w={w}")
        if time.time() - last_heartbeat > 30:
            print(f"   ⏱️ Heartbeat: processed {i+1}/{len(windows)} windows in {time.time() - start_time:.1f}s")
            last_heartbeat = time.time()

        window_features: Dict[str, np.ndarray] = {}
        family_start = len(window_features)
        
        # --- HOISTED ROLLING OBJECTS ---
        # Compute shared rolling stats once per window to avoid redundant calculations
        r_high = high.rolling(w)
        r_low = low.rolling(w)
        r_close = close.rolling(w)
        r_ret = log_ret.rolling(w)
        
        # Pre-compute common aggregates
        win_high = r_high.max()
        win_low = r_low.min()
        win_close_mean = r_close.mean()
        win_ret_std = r_ret.std()
        win_ret_mean = r_ret.mean()
        
        # Optional Volume aggregates
        if volume_available:
            r_vol = volume.rolling(w)
            win_vol_mean = r_vol.mean()
            win_vol_std = r_vol.std()
            win_vol_sum = r_vol.sum()

        # --- 1. PRICE MOMENTUM ---(MTF Virtual Candle) ---
        # Construct virtual candle for window w
        # High = Rolling Max, Low = Rolling Min, Close = Close, Open = Open shifted
        # Open of the virtual candle is the Open of the bar w-1 periods ago
        win_open = open_p.shift(w - 1)
        # win_close is just current close
        
        win_range = win_high - win_low
        win_body = (close - win_open).abs()

        # Shadows
        # Upper: High - Max(Open, Close)
        win_upper = win_high - pd.concat([win_open, close], axis=1).max(axis=1)
        # Lower: Min(Open, Close) - Low
        win_lower = pd.concat([win_open, close], axis=1).min(axis=1) - win_low

        window_features[f'body_to_range_w{w}'] = _align_to_features(_norm(win_body / (win_range + 1e-9), f'body_to_range_w{w}'), n_features)
        window_features[f'shadow_asymmetry_w{w}'] = _align_to_features(_norm((win_upper - win_lower) / (win_range + 1e-9), f'shadow_asymmetry_w{w}'), n_features)

        win_clv = ((close - win_low) - (win_high - close)) / (win_range + 1e-9)
        window_features[f'close_location_value_w{w}'] = _align_to_features(_norm(win_clv, f'close_location_value_w{w}'), n_features)
        _flush_family(window_features, family_start)

        # --- 2. ADVANCED VOLATILITY ---
        family_start = len(window_features)
        # Yang-Zhang
        yz_vol = compute_yang_zhang_volatility(open_p, high, low, close, window=w)
        window_features[f'yang_zhang_volatility_w{w}'] = _align_to_features(_norm(yz_vol, f'yang_zhang_volatility_w{w}'), n_features)

        # Rogers-Satchell
        rs_vol = compute_rogers_satchell_volatility(open_p, high, low, close, window=w)
        window_features[f'rogers_satchell_volatility_w{w}'] = _align_to_features(_norm(rs_vol, f'rogers_satchell_volatility_w{w}'), n_features)

        # Parkinson
        park_vol = compute_parkinson_volatility(high, low, window=w)
        window_features[f'parkinson_volatility_w{w}'] = _align_to_features(_norm(park_vol, f'parkinson_volatility_w{w}'), n_features)

        # Standard & Z-Score Vol
        # Use hoisted
        vol_w = win_ret_std
        window_features[f'volatility_w{w}'] = _align_to_features(_norm(vol_w, f'volatility_w{w}'), n_features)
        vol_z = compute_rolling_zscore(vol_w, window=w)
        window_features[f'volatility_zscore_w{w}'] = _align_to_features(_norm(vol_z, f'volatility_zscore_w{w}'), n_features)
        _flush_family(window_features, family_start)

        # --- 3. VOLATILITY REGIME DYNAMICS ---
        family_start = len(window_features)
        # Volatility Trend Slope
        vol_slope = vol_w.diff(max(1, w//4))
        window_features[f'volatility_trend_slope_w{w}'] = _align_to_features(_norm(vol_slope, f'volatility_trend_slope_w{w}'), n_features)

        # Vol Acceleration
        vol_accel = vol_slope.diff()
        window_features[f'vol_acceleration_w{w}'] = _align_to_features(_norm(vol_accel, f'vol_acceleration_w{w}'), n_features)

        # Compression Duration: Bars since BB Width expanded?
        # Or bars since vol > mean?
        # Let's use Bars Since Vol Spike (Z > 2)
        is_vol_spike = vol_z > 2.0
        window_features[f'bars_since_vol_spike_w{w}'] = _align_to_features(_norm(compute_bars_since(is_vol_spike), f'bars_since_vol_spike_w{w}'), n_features)

        # Compression: Low volatility state. Z < -1.0
        is_compression = vol_z < -1.0
        # Bars since compression ENDED = bars since NOT compression.
        # But commonly we want "Duration of current compression".
        # Which is `bars_since_not_compression`.
        vol_compression_duration = _align_to_features(
            _norm(compute_bars_since(~is_compression), f'vol_compression_duration_w{w}'),
            n_features
        )
        window_features[f'vol_compression_duration_w{w}'] = vol_compression_duration

        # Bars since compression ended
        window_features[f'bars_since_compression_ended_w{w}'] = _align_to_features(_norm(compute_bars_since(is_compression), f'bars_since_compression_ended_w{w}'), n_features)

        # Breakout after compression: Interaction
        # If we broke out (price change high) AND we were compressed recently
        price_breakout = compute_rolling_zscore(log_ret, w).abs() > 2.0
        was_compressed = vol_compression_duration > w
        window_features[f'breakout_after_compression_flag_w{w}'] = _align_to_features(
            _norm(price_breakout.astype(float) * was_compressed.astype(float), f'breakout_after_compression_flag_w{w}'),
            n_features
        )
        _flush_family(window_features, family_start)

        # --- 4. TREND QUALITY & EFFICIENCY ---
        family_start = len(window_features)
        # Efficiency Ratio (Kaufman)
        er_w = get_efficiency_ratio(close, window=w)
        window_features[f'trend_efficiency_ratio_w{w}'] = _align_to_features(_norm(er_w, f'trend_efficiency_ratio_w{w}'), n_features)

        # Directional Consistency: sum(sign(returns)) / w
        dir_consistency = np.sign(log_ret).rolling(w).sum() / w
        window_features[f'directional_consistency_w{w}'] = _align_to_features(_norm(dir_consistency.abs(), f'directional_consistency_w{w}'), n_features)

        # Trend Duration: Bars since MA slope flip
        # Use hoisted mean
        ma_w = win_close_mean
        ma_slope = ma_w.diff()
        slope_sign_change = np.sign(ma_slope) != np.sign(ma_slope.shift(1))
        window_features[f'trend_duration_w{w}'] = _align_to_features(_norm(compute_bars_since(slope_sign_change), f'trend_duration_w{w}'), n_features)

        # Momentum Decay
        roc = close.pct_change(w)
        max_roc = roc.rolling(w).max()
        window_features[f'momentum_decay_w{w}'] = _align_to_features(_norm(roc / (max_roc + 1e-9), f'momentum_decay_w{w}'), n_features)

        # Trend per Vol (Sharpe proxy)
        # Use hoisted mean/std
        sharpe_w = (win_ret_mean / (win_ret_std + 1e-9)).fillna(0)
        window_features[f'trend_per_vol_w{w}'] = _align_to_features(_norm(sharpe_w, f'trend_per_vol_w{w}'), n_features)

        # Trend Slope Stability: Mean(Slope) / Std(Slope)
        slope = ma_w.diff()
        slope_stab = slope.rolling(w).mean() / (slope.rolling(w).std() + 1e-9)
        window_features[f'trend_slope_stability_w{w}'] = _align_to_features(_norm(slope_stab, f'trend_slope_stability_w{w}'), n_features)

        # ADX
        adx, pdi, mdi = compute_adx(high, low, close, period=w)
        window_features[f'adx_w{w}'] = _align_to_features(_norm(adx, f'adx_w{w}'), n_features)
        window_features[f'adx_trend_strength_w{w}'] = _align_to_features(_norm((pdi - mdi) / (adx + 1e-9), f'adx_trend_strength_w{w}'), n_features)

        # MACD (Scaled to window)
        # Fast=w/2, Slow=w, Signal=w/3
        macd, macd_sig, macd_hist = compute_macd(close, fast=max(2, w//2), slow=w, signal=max(2, w//3))
        window_features[f'macd_hist_w{w}'] = _align_to_features(_norm(macd_hist, f'macd_hist_w{w}'), n_features)

        # Hurst
        window_features[f'hurst_proxy_w{w}'] = _align_to_features(_norm(compute_hurst_proxy(close, window=w), f'hurst_proxy_w{w}'), n_features)

        # Donchian Channel
        d_upper, d_lower, d_pos = compute_donchian_channel(high, low, close, window=w)
        window_features[f'donchian_position_w{w}'] = _align_to_features(_norm(d_pos, f'donchian_position_w{w}'), n_features)
        window_features[f'donchian_width_w{w}'] = _align_to_features(_norm((d_upper - d_lower) / (close + 1e-9), f'donchian_width_w{w}'), n_features)
        _flush_family(window_features, family_start)

        # --- 5. CYCLE & REGIME ---
        family_start = len(window_features)
        # Fisher Transform
        fisher = compute_fisher_transform(high, low, window=w)
        window_features[f'ehlers_fisher_transform_w{w}'] = _align_to_features(
            _norm(fisher, f'ehlers_fisher_transform_w{w}'),
            n_features,
        )

        # Hilbert Phase
        # Only compute for one main window to save compute, or cheap version
        if w == 20:
            window_features[f'hilbert_transform_phase'] = _align_to_features(_norm(compute_hilbert_phase(close), f'hilbert_transform_phase'), n_features)

        # Aroon
        window_features[f'aroon_oscillator_w{w}'] = _align_to_features(_norm(compute_aroon(high, low, w), f'aroon_oscillator_w{w}'), n_features)

        # Stochastic
        stoch_k, stoch_d = compute_stochastic(high, low, close, k_period=w, d_period=max(3, w//5))
        window_features[f'stochastic_k_w{w}'] = _align_to_features(_norm(stoch_k, f'stochastic_k_w{w}'), n_features)
        
        # CCI
        window_features[f'cci_w{w}'] = _align_to_features(_norm(compute_cci(high, low, close, period=w), f'cci_w{w}'), n_features)
        _flush_family(window_features, family_start)

        # --- 6. VOLUME-PRICE EFFICIENCY ---
        family_start = len(window_features)
        if volume_available:
            # Price Impact: Abs(Ret) / Vol
            # Normalize volume first or ratio?
            # Log Return / Log Volume is better, or Amihud illiquidity
            impact = log_ret.abs() / (volume + 1e-9)
            window_features[f'price_impact_w{w}'] = _align_to_features(_norm(impact.rolling(w).mean(), f'price_impact_w{w}'), n_features)

            # Signed Price Impact
            signed_impact = log_ret / (volume + 1e-9)
            window_features[f'signed_price_impact_w{w}'] = _align_to_features(_norm(signed_impact.rolling(w).mean(), f'signed_price_impact_w{w}'), n_features)

            # Range per Volume (Kyber's liquidity metric)
            rpv = (high - low) / (volume + 1e-9)
            window_features[f'range_per_volume_w{w}'] = _align_to_features(_norm(rpv.rolling(w).mean(), f'range_per_volume_w{w}'), n_features)

            # Volume without progress (Churn): Vol * (1 - Efficiency)
            churn = volume * (1 - er_w)
            window_features[f'volume_without_progress_w{w}'] = _align_to_features(_norm(churn.rolling(w).mean(), f'volume_without_progress_w{w}'), n_features)

            # Delta-Volume Divergence: Corr(PriceChange, VolDelta)
            # Or Divergence between Price Trend and Vol Trend
            vol_trend = volume.rolling(w).mean().diff()
            price_trend = close.rolling(w).mean().diff()
            # Simple interaction
            window_features[f'delta_volume_divergence_w{w}'] = _align_to_features(_norm(vol_trend * price_trend, f'delta_volume_divergence_w{w}'), n_features)

            # Climax Volume: Vol > 3 * mean
            vol_mean = volume.rolling(w).mean()
            is_climax = volume > 3 * vol_mean
            window_features[f'climax_volume_flag_w{w}'] = _align_to_features(_norm(is_climax.astype(float), f'climax_volume_flag_w{w}'), n_features)

            # CMF
            window_features[f'cmf_w{w}'] = _align_to_features(_norm(compute_cmf(high, low, close, volume, period=w), f'cmf_w{w}'), n_features)

            # Force Index
            window_features[f'force_index_w{w}'] = _align_to_features(_norm(compute_force_index(close, volume, period=w), f'force_index_w{w}'), n_features)

            # Volume Delta (Rolling Sum)
            vol_delta = compute_volume_delta(close, open_p, volume)
            window_features[f'volume_delta_w{w}'] = _align_to_features(_norm(vol_delta.rolling(w).sum(), f'volume_delta_w{w}'), n_features)

            # OBV (Slope)
            obv = compute_obv(close, volume)
            window_features[f'obv_slope_w{w}'] = _align_to_features(_norm(obv.diff(w), f'obv_slope_w{w}'), n_features)
        _flush_family(window_features, family_start)

        # --- 7. VWAP & CONTEXT ---
        family_start = len(window_features)
        if volume_available:
            # VWAP Z-Score
            tp = (high + low + close) / 3
            # rolling vwap
            pv = tp * volume
            vwap = pv.rolling(w).sum() / (volume.rolling(w).sum() + 1e-9)
            vwap_std = tp.rolling(w).std() # Approx std dev of price
            window_features[f'vwap_zscore_w{w}'] = _align_to_features(_norm((close - vwap)/(vwap_std+1e-9), f'vwap_zscore_w{w}'), n_features)

            # EOM
            window_features[f'ease_of_movement_w{w}'] = _align_to_features(_norm(compute_ease_of_movement(high, low, volume, w), f'ease_of_movement_w{w}'), n_features)

            # MFI
            window_features[f'volume_weighted_rsi_w{w}'] = _align_to_features(_norm(compute_mfi(high, low, close, volume, w), f'volume_weighted_rsi_w{w}'), n_features)
        _flush_family(window_features, family_start)

        # --- 8. STRUCTURAL SR ---
        family_start = len(window_features)
        # Touch count: Count close near High/Low of window
        # Fix: Use PREVIOUS window peak to avoid look-ahead bias/autocorrelation
        # Previously used win_high (current window max), which always includes current high.
        win_high_prev = high.shift(1).rolling(w).max()
        win_low_prev = low.shift(1).rolling(w).min()
        
        # Near = within 0.5% range?
        rng = win_high - win_low # Use current range for scale context
        thresh = rng * 0.05
        
        # Current bar touch of PREVIOUS structure?
        touch_high = (high > (win_high_prev - thresh)).astype(float)
        touch_low = (low < (win_low_prev + thresh)).astype(float)
        
        # Sum touches in window
        window_features[f'touch_count_near_price_w{w}'] = _align_to_features(_norm((touch_high + touch_low).rolling(w).sum(), f'touch_count_near_price_w{w}'), n_features)

        # Time-weighted SR strength
        # Weighted sum of touches (decay over time)
        # Touch event series
        touch_series = (touch_high + touch_low)
        # Exponential moving sum
        window_features[f'time_weighted_sr_strength_w{w}'] = _align_to_features(_norm(touch_series.ewm(span=w).sum(), f'time_weighted_sr_strength_w{w}'), n_features)

        # Range Expansion Ratio: Range / Avg Range
        avg_range = (high - low).rolling(w).mean()
        window_features[f'range_expansion_ratio_w{w}'] = _align_to_features(_norm((high - low) / (avg_range + 1e-9), f'range_expansion_ratio_w{w}'), n_features)

        # Breakout Follow Through: (Close - BreakLevel) / BreakCandleSize
        # Defined as momentum continuity
        # Simplified: Return(t) * Return(t-1) > 0 ?
        window_features[f'breakout_follow_through_w{w}'] = _align_to_features(_norm((log_ret * log_ret.shift(1)), f'breakout_follow_through_w{w}'), n_features)

        # False Break Rate (Proxy)
        # Breakout: High > Rolling Max High (w)
        # False: Breakout AND Close < Rolling Max High (w)
        # Use previous high for breakout level
        roll_high_prev = win_high_prev # Re-use hoisted/computed prev high
        breakout = high > roll_high_prev
        failed_break = breakout & (close < roll_high_prev)
        # Rate in window
        window_features[f'false_break_rate_w{w}'] = _align_to_features(_norm(failed_break.rolling(w).sum() / (breakout.rolling(w).sum() + 1e-9), f'false_break_rate_w{w}'), n_features)
        _flush_family(window_features, family_start)

        if enable_tail_risk:
            # --- 9. TAIL RISK ---
            family_start = len(window_features)
            # Rolling Skew/Kurt
            window_features[f'rolling_skewness_w{w}'] = _align_to_features(_norm(r_ret.skew(), f'rolling_skewness_w{w}'), n_features)
            window_features[f'rolling_kurtosis_w{w}'] = _align_to_features(_norm(r_ret.kurt(), f'rolling_kurtosis_w{w}'), n_features)

            # Downside Semivariance
            neg_ret = log_ret.where(log_ret < 0, 0)
            downside_var = neg_ret.rolling(w).var()
            window_features[f'downside_semivariance_w{w}'] = _align_to_features(_norm(downside_var, f'downside_semivariance_w{w}'), n_features)

            # Left Tail Var Ratio
            # Use hoisted
            total_var = r_ret.var()
            window_features[f'left_tail_var_ratio_w{w}'] = _align_to_features(_norm(downside_var / (total_var + 1e-9), f'left_tail_var_ratio_w{w}'), n_features)

            # Max Runup (MFE Proxy)
            # Max high relative to close
            # "How high did it go in last w bars?" relative to min in window?
            # Rolling Max - Rolling Min / Rolling Min
            # Use hoisted
            window_features[f'max_runup_w{w}'] = _align_to_features(_norm((win_high - win_low) / (win_low + 1e-9), f'max_runup_w{w}'), n_features)
            # This is effectively max amplitude

            # Drawdown Depth (Current price vs Rolling Max)
            dd_w = (close / win_high) - 1.0
            window_features[f'drawdown_w{w}'] = _align_to_features(_norm(dd_w, f'drawdown_w{w}'), n_features)

            # Max Adverse Excursion (MAE) Proxy -> Alias to Drawdown
            window_features[f'max_adverse_excursion_w{w}'] = window_features[f'drawdown_w{w}']

            # Tail Event Flag
            is_tail = log_ret.abs() > (win_ret_std * 3)
            window_features[f'tail_event_flag_w{w}'] = _align_to_features(_norm(is_tail.astype(float), f'tail_event_flag_w{w}'), n_features)
            _flush_family(window_features, family_start)

        # --- 11. EVENT BASED TIME IN STATE ---
        family_start = len(window_features)
        # Bars since breakout
        # Breakout = Close > Rolling High (prev)
        # Use computed previous high
        win_high_prev = high.shift(1).rolling(w).max()
        is_breakout = close > win_high_prev
        window_features[f'bars_since_breakout_attempt_w{w}'] = _align_to_features(_norm(compute_bars_since(is_breakout), f'bars_since_breakout_attempt_w{w}'), n_features)

        # Bars since trend exhaustion (e.g. RSI > 80 or < 20)
        # Use RSI of window w
        rsi_w = compute_rsi(close, period=w)
        is_exhaustion = (rsi_w > 80) | (rsi_w < 20)
        window_features[f'bars_since_trend_exhaustion_signal_w{w}'] = _align_to_features(_norm(compute_bars_since(is_exhaustion), f'bars_since_trend_exhaustion_signal_w{w}'), n_features)
        _flush_family(window_features, family_start)

        if w in interaction_windows:
            family_start = len(window_features)
            # --- 12. INTERACTIONS ---
            # Trend Alignment: RSI direction == Slope direction
            rsi_slope = rsi_w.diff(max(1, w//4)) # Use window-aligned slope for RSI
            aligned = np.sign(rsi_slope) == np.sign(ma_slope)
            window_features[f'trend_alignment_score_w{w}'] = _align_to_features(_norm(aligned.astype(float), f'trend_alignment_score_w{w}'), n_features)

            # Price-Volume Correlation
            if volume_available:
                # Correlation between Returns and Volume Change
                vol_chg = volume.pct_change()
                pv_corr = log_ret.rolling(w).corr(vol_chg).fillna(0)
                window_features[f'price_vol_correlation_w{w}'] = _align_to_features(_norm(pv_corr, f'price_vol_correlation_w{w}'), n_features)

            # --- 13. CROSS-TIMEFRAME RATIOS ---
            # Compare w vs w_long (e.g. w*4)
            w_long = w * 4
            # SMA Ratio (Trend Extension)
            sma_w = ma_w
            sma_long = close.rolling(w_long).mean()
            window_features[f'sma_ratio_w{w}_vs_w{w_long}'] = _align_to_features(_norm(sma_w / (sma_long + 1e-9), f'sma_ratio_w{w}_vs_w{w_long}'), n_features)

            # RSI Ratio (Momentum Divergence)
            rsi_long = compute_rsi(close, period=w_long)
            window_features[f'rsi_ratio_w{w}_vs_w{w_long}'] = _align_to_features(_norm(rsi_w / (rsi_long + 1e-9), f'rsi_ratio_w{w}_vs_w{w_long}'), n_features)

            # Volatility Ratio (Vol Compression/Expansion)
            vol_long = log_ret.rolling(w_long).std()
            window_features[f'vol_ratio_w{w}_vs_w{w_long}'] = _align_to_features(_norm(vol_w / (vol_long + 1e-9), f'vol_ratio_w{w}_vs_w{w_long}'), n_features)

            # Vol Trend Conflict: Vol rising, Price falling (Panic?) or Vol falling, Price rising (Drift)
            # Vol Slope * Price Slope
            window_features[f'vol_trend_conflict_w{w}'] = _align_to_features(_norm(vol_slope * ma_slope, f'vol_trend_conflict_w{w}'), n_features)

            # Compression x Momentum (Already partially done)
            bb_width = compute_bollinger_bands(close, w)[3] # raw width
            mom_abs = roc.abs()
            window_features[f'compression_x_momentum_w{w}'] = _align_to_features(_norm(mom_abs / (bb_width + 1e-9), f'compression_x_momentum_w{w}'), n_features)

            # Absorption x Vol Spike
            if volume_available:
                window_features[f'absorption_x_vol_spike_w{w}'] = _align_to_features(_norm(rpv * is_vol_spike.astype(float), f'absorption_x_vol_spike_w{w}'), n_features)
            _flush_family(window_features, family_start)

        if window_features:
            window_df = pd.DataFrame(window_features, index=features.index)
            window_df = _downcast_float32(window_df)
            feature_chunks.append(window_df)
            window_features.clear()

        if len(feature_chunks) >= chunk_window_size:
            features = pd.concat([features] + feature_chunks, axis=1)
            feature_chunks.clear()
            features = _downcast_float32(features)

    if feature_chunks:
        features = pd.concat([features] + feature_chunks, axis=1)
        features = _downcast_float32(features)

    if isinstance(original_index, pd.DatetimeIndex) and not features.index.equals(original_index):
        features = features.reindex(original_index, method="ffill").fillna(0.0)
        # Update cached length so subsequent alignment helpers use the expanded index
        n_features = len(features)

    # ===== LEGACY SUPPORT / CROSS-TIMEFRAME SPECIFIC =====
    # Add back some key legacy features if not covered
    # Use robust 'close' extracted earlier
    close_1h = close.rolling(4).mean()
    features['returns_1h'] = _align_to_features(_norm(close_1h.pct_change(), 'returns_1h'), n_features)
    close_4h = close.rolling(16).mean()
    features['returns_4h'] = _align_to_features(_norm(close_4h.pct_change(), 'returns_4h'), n_features)

    # Final enforcement of float32
    features = _downcast_float32(features)

    try:
        logger.info(f"[MTF] create_meta_features produced {int(len(features.columns))} columns prior to Layer2 filtering.")
    except Exception:
        pass

    # Performance timing
    end_time = time.time()
    elapsed = end_time - start_time
    optimization_status = "Numba JIT" if NUMBA_AVAILABLE else "Pandas fallback"
    print(f"⚡ MTF feature generation completed in {elapsed:.2f}s using {optimization_status} ({len(features)} features)")
    
    # Memory cleanup
    gc.collect()
    print(f"🧹 Memory cleanup completed")

    # --- SAVE TO CACHE ---
    if cache_key:
        try:
            _MTF_CACHE[cache_key] = features.copy()
            print(f"💾 [MTF Cache] Saved {len(features.columns)} features to cache")
        except Exception:
            pass

    # --- CACHE MANAGEMENT ---
    # Enforce size limit
    if len(_MTF_CACHE) > _MAX_MTF_CACHE_SIZE:
        # Simple FIFO/Arbitrary pruning (dicts are insertion ordered in modern Python)
        # Remove oldest entries until we are under the limit
        num_to_remove = len(_MTF_CACHE) - _MAX_MTF_CACHE_SIZE
        try:
            # Create a list of keys to remove to avoid runtime error during iteration
            # We want to keep the NEWEST ones (last inserted). 
            # So remove the first 'num_to_remove' keys.
            keys_to_remove = list(_MTF_CACHE.keys())[:num_to_remove]
            for k in keys_to_remove:
                del _MTF_CACHE[k]
            
            # Force collection after pruning large objects
            gc.collect()
            print(f"🧹 [MTF Cache] Pruned {len(keys_to_remove)} old entries to enforce limit of {_MAX_MTF_CACHE_SIZE}")
        except Exception as e:
            print(f"⚠️ [MTF Cache] Pruning failed: {e}")

    return features
