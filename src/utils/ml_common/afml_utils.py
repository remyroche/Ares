import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union

def get_daily_vol(close: pd.Series, span: int = 100, use_volume_time: bool = False) -> pd.Series:
    """
    Calculate daily volatility (standard deviation of daily returns) as a threshold for barriers.
    
    Args:
        close: Close price series.
        span: EWM span.
        use_volume_time: If True, calculates volatility based on bar count (assuming volume bars)
                         rather than time-based '1 day' lookback.
    """
    if use_volume_time:
        returns = close.pct_change()
        vol = returns.ewm(span=span).std()
        return vol
    else:
        if not isinstance(close.index, pd.DatetimeIndex):
             returns = close.pct_change()
             return returns.ewm(span=span).std()

        try:
            prev_day_idx = close.index.searchsorted(close.index - pd.Timedelta(days=1))
            valid_mask = prev_day_idx > 0

            returns = pd.Series(index=close.index, dtype=float)
            idx_now = close.index[valid_mask]
            idx_prev = close.index[prev_day_idx[valid_mask] - 1]

            returns.loc[idx_now] = close.loc[idx_now].values / close.loc[idx_prev].values - 1
            vol = returns.ewm(span=span).std()
            return vol
        except Exception:
             returns = close.pct_change()
             return returns.ewm(span=span).std()

def get_t_events(close: pd.Series, threshold: Union[float, pd.Series]) -> pd.DatetimeIndex:
    """
    CUSUM Filter: Returns a DatetimeIndex of events where cumulative returns exceed a threshold.
    """
    t_events, s_pos, s_neg = [], 0, 0
    diff = close.pct_change().fillna(0)
    
    if isinstance(threshold, pd.Series):
        threshold_series = threshold.reindex(diff.index, method='ffill').fillna(method='bfill')
    else:
        threshold_series = pd.Series(threshold, index=diff.index)
    
    for i in diff.index[1:]:
        s_pos = max(0, s_pos + diff.loc[i])
        s_neg = min(0, s_neg + diff.loc[i])
        thresh = abs(threshold_series.loc[i])
        if np.isnan(thresh) or thresh == 0: continue
            
        if s_neg < -thresh:
            s_neg = 0
            t_events.append(i)
        elif s_pos > thresh:
            s_pos = 0
            t_events.append(i)
            
    return pd.DatetimeIndex(t_events)

def get_vertical_barrier(close: pd.Series, t_events: pd.DatetimeIndex, num_bars: int) -> pd.Series:
    """Returns a Series of timestamps representing the expiration (horizontal barrier)."""
    t1 = pd.Series(index=t_events, dtype='datetime64[ns]')
    
    try:
        if not close.index.is_unique:
             for evt in t_events:
                try:
                    loc = close.index.get_loc(evt)
                    if isinstance(loc, (slice, np.ndarray)):
                        loc = loc.start if isinstance(loc, slice) else loc[0]
                    target = loc + num_bars
                    if target < len(close):
                        t1.loc[evt] = close.index[target]
                except KeyError:
                    pass
        else:
            idx_map = pd.Series(np.arange(len(close)), index=close.index)
            event_ilocs = idx_map.loc[t_events].values
            target_ilocs = event_ilocs + num_bars
            valid = target_ilocs < len(close)

            if valid.any():
                valid_ilocs = target_ilocs[valid]
                valid_events = t_events[valid]
                t1.loc[valid_events] = close.index[valid_ilocs]

    except Exception:
        for evt in t_events:
            try:
                loc_idx = close.index.get_loc(evt)
                if isinstance(loc_idx, (slice, np.ndarray)):
                     loc_idx = loc_idx.start if isinstance(loc_idx, slice) else loc_idx[0]
                target_idx = loc_idx + num_bars
                if target_idx < len(close):
                    t1.loc[evt] = close.index[target_idx]
            except KeyError:
                continue

    return t1

def apply_triple_barrier(close: pd.Series, t_events: pd.DatetimeIndex, pt_sl: List[float], 
                         target: pd.Series, min_ret: float, vertical_barrier: Optional[pd.Series] = None) -> pd.DataFrame:
    """Triple Barrier Method."""
    if vertical_barrier is None:
        vertical_barrier = pd.Series(index=t_events, data=pd.NaT)
    
    out = pd.DataFrame(index=t_events, columns=['t1', 'trgt', 'type', 'ret', 'mfe', 'mae'])
    aligned_target = target.reindex(t_events, method='ffill')
    out['trgt'] = aligned_target
    
    for loc, trgt in out['trgt'].items():
        if np.isnan(trgt): continue
        
        pt = pt_sl[0] * trgt if pt_sl[0] > 0 else np.inf
        sl = -pt_sl[1] * trgt if pt_sl[1] > 0 else -np.inf
            
        t1 = vertical_barrier.loc[loc]
        if pd.isna(t1):
            window = close.iloc[close.index.get_loc(loc) + 1:]
        else:
            try:
                window = close.iloc[close.index.get_loc(loc) + 1 : close.index.get_loc(t1) + 1]
            except KeyError:
                window = close.iloc[close.index.get_loc(loc) + 1:]
            
        if window.empty:
            out.loc[loc, ['t1', 'type', 'ret', 'mfe', 'mae']] = [pd.NaT, 'none', 0.0, 0.0, 0.0]
            continue

        rets = (window / close.loc[loc] - 1)
        mfe = rets.max()
        mae = rets.min()

        hi_idx = rets[rets > pt].index
        lo_idx = rets[rets < sl].index
        
        if len(hi_idx) > 0 or len(lo_idx) > 0:
            if len(hi_idx) == 0:
                out.loc[loc, 't1'] = lo_idx[0]
                out.loc[loc, 'type'] = 'sl'
                out.loc[loc, 'ret'] = rets.loc[lo_idx[0]]
                touch_time = lo_idx[0]
                path_to_touch = rets.loc[:touch_time]
                out.loc[loc, 'mfe'] = path_to_touch.max()
                out.loc[loc, 'mae'] = path_to_touch.min()

            elif len(lo_idx) == 0:
                out.loc[loc, 't1'] = hi_idx[0]
                out.loc[loc, 'type'] = 'pt'
                out.loc[loc, 'ret'] = rets.loc[hi_idx[0]]
                touch_time = hi_idx[0]
                path_to_touch = rets.loc[:touch_time]
                out.loc[loc, 'mfe'] = path_to_touch.max()
                out.loc[loc, 'mae'] = path_to_touch.min()

            else:
                if hi_idx[0] < lo_idx[0]:
                    out.loc[loc, 't1'] = hi_idx[0]
                    out.loc[loc, 'type'] = 'pt'
                    out.loc[loc, 'ret'] = rets.loc[hi_idx[0]]
                    touch_time = hi_idx[0]
                    path_to_touch = rets.loc[:touch_time]
                    out.loc[loc, 'mfe'] = path_to_touch.max()
                    out.loc[loc, 'mae'] = path_to_touch.min()
                else:
                    out.loc[loc, 't1'] = lo_idx[0]
                    out.loc[loc, 'type'] = 'sl'
                    out.loc[loc, 'ret'] = rets.loc[lo_idx[0]]
                    touch_time = lo_idx[0]
                    path_to_touch = rets.loc[:touch_time]
                    out.loc[loc, 'mfe'] = path_to_touch.max()
                    out.loc[loc, 'mae'] = path_to_touch.min()
        else:
            if not pd.isna(t1):
                out.loc[loc, 't1'] = t1
                out.loc[loc, 'type'] = 'expiration'
                out.loc[loc, 'ret'] = rets.iloc[-1]
                out.loc[loc, 'mfe'] = mfe
                out.loc[loc, 'mae'] = mae
            else:
                out.loc[loc, ['t1', 'type', 'ret', 'mfe', 'mae']] = [pd.NaT, 'none', 0.0, 0.0, 0.0]
                
    return out

def get_bins(triple_barrier_events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    out = triple_barrier_events.copy()
    out['bin'] = (out['type'] == 'pt').astype(int)
    return out

def get_num_concurrent_events(close_index: pd.Index, t1: pd.Series) -> pd.Series:
    t1 = t1.fillna(close_index[-1])
    count = pd.Series(0, index=close_index)
    for i, j in t1.items():
        count.loc[i:j] += 1
    return count

def get_sample_uniqueness(t1: pd.Series, num_concurrent: pd.Series) -> pd.Series:
    unq = pd.Series(index=t1.index, dtype=float)
    for i, j in t1.items():
        if pd.isna(j): 
            j = num_concurrent.index[-1]
        unq.loc[i] = (1.0 / num_concurrent.loc[i:j]).mean()
    return unq

def compute_structural_inertia(series: pd.Series, window: int = 20) -> pd.Series:
    def get_slope_stat(s):
        if len(s) < window: return 0.0
        x = np.arange(len(s))
        y = s.values
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        residuals = y - y_pred
        sse = np.sum(residuals**2)
        if len(s) <= 2: return slope
        se_slope = np.sqrt(sse / (len(s) - 2)) / np.sqrt(np.sum((x - np.mean(x))**2) + 1e-8)
        return slope / (se_slope + 1e-8)
    return series.rolling(window=window).apply(get_slope_stat)

def compute_efficiency_ratio(series: pd.Series, window: int = 20) -> pd.Series:
    net_change = (series - series.shift(window)).abs()
    total_travel = series.diff().abs().rolling(window=window).sum()
    return net_change / (total_travel + 1e-8)

def compute_hunter_weight(uniqueness: np.ndarray, mfe: np.ndarray, mae: np.ndarray,
                          barrier: np.ndarray, wavelet_noise: np.ndarray) -> np.ndarray:
    """
    New Hunter System Weighting:
    wi = clip(u^0.3 * sigma(cleanliness) * sigma(1 - wavelet_noise), 0.2, 1.0)
    """
    def sigmoid(x, beta=10.0, offset=0.5):
        return 1.0 / (1.0 + np.exp(-beta * (x - offset)))

    # 1. Uniqueness component
    u_comp = np.power(np.clip(uniqueness, 0.0, 1.0), 0.3)
    
    # 2. Cleanliness component
    # C = (MFE - MAE) / barrier
    barrier_safe = np.where(barrier <= 0, 1e-8, barrier)
    cleanliness = np.clip((mfe - mae) / barrier_safe, 0.0, 1.0)
    c_comp = sigmoid(cleanliness, beta=10.0, offset=0.5)

    # 3. Wavelet Noise component
    # N = 1.0 - wavelet_noise_ratio
    # wavelet_noise is typically HF/Total Energy
    n_val = 1.0 - np.clip(wavelet_noise, 0.0, 1.0)
    w_comp = sigmoid(n_val, beta=10.0, offset=0.5)

    weights = u_comp * c_comp * w_comp

    return np.clip(weights, 0.2, 1.0)

def compute_master_weight(uniqueness: np.ndarray, mfe: np.ndarray, mae: np.ndarray,
                          barrier: np.ndarray, hf_lf_ratio: np.ndarray,
                          volatility: np.ndarray, raw_return: np.ndarray,
                          timestamp_index: pd.Index) -> np.ndarray:
    """Legacy master weight wrapper - redirecting to hunter weight for consistency if desired."""
    # For now, we keep original logic or can switch to hunter weight.
    # The prompt explicitly asked to implement the formula:
    # wi = clip(u^0.3 * sigma(cleanliness) * sigma(1-wavelet), 0.2, 1.0)
    
    # Wavelet noise ratio ~ hf_lf_ratio / (1+hf_lf_ratio) ?
    # Or just use hf_lf_ratio as proxy.
    # The input hf_lf_ratio is HighFreqEnergy / LowFreqEnergy.
    # Noise ratio = HF / (HF + LF) = Ratio / (1 + Ratio)
    noise_ratio = hf_lf_ratio / (1.0 + hf_lf_ratio + 1e-9)
    
    return compute_hunter_weight(uniqueness, mfe, mae, barrier, noise_ratio)

def seq_bootstrap(t1: pd.Series, close_index: pd.Index, num_samples: Optional[int] = None) -> List[pd.Timestamp]:
    if num_samples is None:
        num_samples = len(t1)
    phi = []
    while len(phi) < num_samples:
        avg_unq = pd.Series(0.0, index=t1.index)
        if len(phi) > 0:
            t1_phi = t1.loc[phi]
            count = pd.Series(0, index=close_index)
            for i, j in t1_phi.items():
                if pd.isna(j): j = close_index[-1]
                count.loc[i:j] += 1
            for k in t1.index:
                if k in phi: continue
                k_start = k
                k_end = t1.loc[k]
                if pd.isna(k_end): k_end = close_index[-1]
                avg_unq.loc[k] = (1.0 / (count.loc[k_start:k_end] + 1)).mean()
        else:
            avg_unq[:] = 1.0
        prob = avg_unq / avg_unq.sum()
        next_sample = np.random.choice(t1.index, p=prob.values)
        phi.append(next_sample)
    return phi

def get_sample_weights(t1: pd.Series, num_concurrent: pd.Series, returns: pd.Series) -> pd.Series:
    unq = get_sample_uniqueness(t1, num_concurrent)
    abs_ret = returns.abs().reindex(unq.index).fillna(0)
    weights = unq * abs_ret
    if weights.sum() > 0:
        weights = weights / weights.sum() * len(weights)
    return weights

def get_weights_by_uniqueness(t1: pd.Series, close_index: pd.Index) -> pd.Series:
    """Simplified sample weighting based only on uniqueness."""
    num_concurrent = get_num_concurrent_events(close_index, t1)
    unq = get_sample_uniqueness(t1, num_concurrent)
    return unq

def get_frac_diff_weights(d: float, size: int) -> np.ndarray:
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] / k * (d - k + 1)
        w.append(w_k)
    return np.array(w[::-1]).reshape(-1, 1)

def frac_diff_fixed(series: pd.Series, d: float, threshold: float = 1e-5) -> pd.Series:
    w = get_frac_diff_weights(d, 100)
    w_sum = np.cumsum(abs(w))
    w_sum /= w_sum[-1]
    skip = (w_sum > threshold).argmax()
    w = w[skip:]
    res = {}
    for i in range(len(w), series.shape[0]):
        res[series.index[i]] = np.dot(w.T, series.iloc[i-len(w)+1:i+1])[0]
    return pd.Series(res)

def calculate_rolling_volatility(prices_15min: pd.Series, window_days: int = 7) -> pd.Series:
    log_rets = np.log(prices_15min / prices_15min.shift(1))
    window_size = 96 * window_days
    vol = log_rets.rolling(window=window_size, min_periods=96).std()
    return vol

def calculate_dynamic_range_threshold(volatility: pd.Series, current_price: pd.Series, k: float = None) -> pd.Series:
    if k is None:
        k = np.sqrt(2 / np.pi)
    return k * volatility * current_price

try:
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    adfuller = None

def get_weights_ffd(d: float, thres: float, lim: Optional[int] = None) -> np.ndarray:
    """
    Get weights for fractional differentiation using Fixed Width Window (FFD) method.
    The window size is determined dynamically where weights drop below threshold.
    """
    w, k = [1.], 1
    while True:
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
        if lim and k >= lim:
            break
    w = np.array(w[::-1]).reshape(-1, 1)
    return w

def frac_diff_ffd(series: pd.Series, d: float, thres: float = 1e-4) -> pd.Series:
    """
    Apply fractional differentiation using FFD method to preserve memory.
    """
    # 1) Compute weights for the longest series
    w = get_weights_ffd(d, thres)
    width = len(w) - 1

    # 2) Apply weights to values
    # Loop implementation is robust for Series
    df = series.to_frame()
    output = {}

    series_f = df.iloc[:, 0].fillna(method='ffill')

    # Pre-compute valid indices to loop over
    # We need a window of size `len(w)`
    if len(series_f) < width + 1:
        return pd.Series(dtype=float)

    values = series_f.values
    indices = series_f.index
    w_flat = w.flatten()

    # Vectorized stride trick or simple loop.
    # For simplicity and correctness with DatetimeIndex alignment:
    for i in range(width, len(values)):
        # Window of prices
        window = values[i-width:i+1]
        # Skip if any NaN in window (though ffill helps)
        if np.isnan(window).any():
            continue

        # Dot product: sum(w * window)
        # w is reversed in get_weights_ffd so it aligns with [t-width ... t]
        # Actually standard implementation expects w to be applied such that:
        # y_t = w_0*x_t + w_1*x_{t-1} ...
        # My get_weights_ffd returns w[::-1], so w[0] is w_k (smallest), w[-1] is w_0 (1.0).
        # So if window is [x_{t-k} ... x_t], then dot product works directly.
        val = np.dot(window, w_flat)
        output[indices[i]] = val

    return pd.Series(output).sort_index()

def optimize_fractional_differentiation(series: pd.Series, d_min: float = 0.0, d_max: float = 1.0,
                                      step: float = 0.1, threshold: float = 1e-4,
                                      p_val_thresh: float = 0.05) -> Tuple[float, pd.Series]:
    """
    Find the minimum differentiation order 'd' that makes the series stationary (ADF test).
    Returns (best_d, best_series).
    """
    if not STATSMODELS_AVAILABLE:
        # Fallback to standard 1st diff if statsmodels missing
        return 1.0, series.diff().dropna()

    best_d = 1.0
    best_series = series.diff().dropna()

    # Generate range of d values
    ds = np.arange(d_min, d_max + step/2, step)

    for d in ds:
        # Skip d ~ 0 as it's likely non-stationary prices
        if d < 1e-3: continue

        try:
            diff_series = frac_diff_ffd(series, d, thres=threshold).dropna()

            # ADF requires some length
            if len(diff_series) < 20:
                continue

            # Run Augmented Dickey-Fuller test
            # Returns: adf_stat, pvalue, usedlag, nobs, critical_values, icbest
            res = adfuller(diff_series.values, maxlag=1, regression='c', autolag=None)
            p_val = res[1]

            # If p-value is low enough, reject null hypothesis (Unit Root) -> Stationary
            if p_val < p_val_thresh:
                best_d = d
                best_series = diff_series
                return float(best_d), best_series

        except Exception:
            continue

    return float(best_d), best_series
