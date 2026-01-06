import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union

def get_daily_vol(close: pd.Series, span: int = 100) -> pd.Series:
    """Calculate daily volatility (standard deviation of daily returns) as a threshold for barriers."""
    # Find timestamps exactly 1 day ago
    prev_day_idx = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    # Filter out out-of-bounds indices
    valid_mask = prev_day_idx > 0
    
    # Calculate returns relative to price ~1 day ago
    returns = pd.Series(index=close.index, dtype=float)
    idx_now = close.index[valid_mask]
    idx_prev = close.index[prev_day_idx[valid_mask] - 1]
    
    returns.loc[idx_now] = close.loc[idx_now].values / close.loc[idx_prev].values - 1
    
    # Exponential moving average of standard deviation
    vol = returns.ewm(span=span).std()
    return vol

def get_t_events(close: pd.Series, threshold: Union[float, pd.Series]) -> pd.DatetimeIndex:
    """
    CUSUM Filter: Returns a DatetimeIndex of events where cumulative returns exceed a threshold.
    threshold can be a constant or a series (adaptive).
    """
    t_events, s_pos, s_neg = [], 0, 0
    diff = close.pct_change().fillna(0)
    
    # Ensure threshold is aligned with diff index
    if isinstance(threshold, pd.Series):
        threshold_series = threshold.reindex(diff.index, method='ffill').fillna(method='bfill')
    else:
        threshold_series = pd.Series(threshold, index=diff.index)
    
    for i in diff.index[1:]:
        s_pos = max(0, s_pos + diff.loc[i])
        s_neg = min(0, s_neg + diff.loc[i])
        
        # Use absolute threshold value
        thresh = abs(threshold_series.loc[i])
        if np.isnan(thresh) or thresh == 0:
            continue
            
        if s_neg < -thresh:
            s_neg = 0
            t_events.append(i)
        elif s_pos > thresh:
            s_pos = 0
            t_events.append(i)
            
    return pd.DatetimeIndex(t_events)

def get_vertical_barrier(close: pd.Series, t_events: pd.DatetimeIndex, num_bars: int) -> pd.Series:
    """Returns a Series of timestamps representing the expiration (horizontal barrier)."""
    # Assuming 15m timeframe if not specified, or use the actual timeframe from index frequency
    delta = pd.Timedelta(minutes=num_bars * 15)
    t1 = close.index.searchsorted(t_events + delta)
    
    # Create output series indexed by full t_events
    out = pd.Series(index=t_events, dtype='datetime64[ns]')
    
    # Fill values where indexer is within bounds
    valid_mask = t1 < close.shape[0]
    out.iloc[valid_mask] = close.index[t1[valid_mask]]
    
    return out

def apply_triple_barrier(close: pd.Series, t_events: pd.DatetimeIndex, pt_sl: List[float], 
                         target: pd.Series, min_ret: float, vertical_barrier: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Triple Barrier Method: Returns a DataFrame with the time and label of each event.
    pt_sl: [profit_take_factor, stop_loss_factor]
    target: The volatility/target return to scale the barriers.
    """
    # 1. Get vertical barrier if not provided
    if vertical_barrier is None:
        vertical_barrier = pd.Series(index=t_events, data=pd.NaT)
    
    # 2. Get barriers
    out = pd.DataFrame(index=t_events, columns=['t1', 'trgt', 'type', 'ret'])
    
    # Align target to t_events safely
    aligned_target = target.reindex(t_events, method='ffill')
    out['trgt'] = aligned_target
    
    for loc, trgt in out['trgt'].items():
        if np.isnan(trgt): continue
        
        # Horizontal barriers
        if pt_sl[0] > 0:
            pt = pt_sl[0] * trgt
        else:
            pt = np.inf
            
        if pt_sl[1] > 0:
            sl = -pt_sl[1] * trgt
        else:
            sl = -np.inf
            
        # Vertical barrier expiration
        t1 = vertical_barrier.loc[loc]
        if pd.isna(t1):
            # Window starts from loc + 1 to avoid overlap with features at loc
            window = close.iloc[close.index.get_loc(loc) + 1:]
        else:
            # Window starts from loc + 1 to avoid overlap with features at loc
            window = close.iloc[close.index.get_loc(loc) + 1 : close.index.get_loc(t1) + 1]
            
        if window.empty:
            out.loc[loc, 't1'] = pd.NaT
            out.loc[loc, 'type'] = 'none'
            out.loc[loc, 'ret'] = 0
            continue

        # Check which barrier is hit first
        rets = (window / close.loc[loc] - 1)
        
        # Profit take
        hi_idx = rets[rets > pt].index
        # Stop loss
        lo_idx = rets[rets < sl].index
        
        if len(hi_idx) > 0 or len(lo_idx) > 0:
            # Hit a price barrier
            if len(hi_idx) == 0:
                out.loc[loc, 't1'] = lo_idx[0]
                out.loc[loc, 'type'] = 'sl'
                out.loc[loc, 'ret'] = rets.loc[lo_idx[0]]
            elif len(lo_idx) == 0:
                out.loc[loc, 't1'] = hi_idx[0]
                out.loc[loc, 'type'] = 'pt'
                out.loc[loc, 'ret'] = rets.loc[hi_idx[0]]
            else:
                if hi_idx[0] < lo_idx[0]:
                    out.loc[loc, 't1'] = hi_idx[0]
                    out.loc[loc, 'type'] = 'pt'
                    out.loc[loc, 'ret'] = rets.loc[hi_idx[0]]
                else:
                    out.loc[loc, 't1'] = lo_idx[0]
                    out.loc[loc, 'type'] = 'sl'
                    out.loc[loc, 'ret'] = rets.loc[lo_idx[0]]
        else:
            # Hit vertical barrier (expiration)
            if not pd.isna(t1):
                out.loc[loc, 't1'] = t1
                out.loc[loc, 'type'] = 'expiration'
                out.loc[loc, 'ret'] = rets.iloc[-1]
            else:
                out.loc[loc, 't1'] = pd.NaT
                out.loc[loc, 'type'] = 'none'
                out.loc[loc, 'ret'] = 0
                
    return out

def get_bins(triple_barrier_events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """Generate labels based on triple barrier events."""
    out = triple_barrier_events.copy()
    # 0 if hit stop loss or expiration with negative return, 1 if hit profit take
    out['bin'] = (out['type'] == 'pt').astype(int)
    return out

def get_num_concurrent_events(close_index: pd.Index, t1: pd.Series) -> pd.Series:
    """Calculate the number of concurrent events at each point in time."""
    t1 = t1.fillna(close_index[-1])
    count = pd.Series(0, index=close_index)
    for i, j in t1.items():
        count.loc[i:j] += 1
    return count

def get_sample_uniqueness(t1: pd.Series, num_concurrent: pd.Series) -> pd.Series:
    """Calculate average uniqueness of each sample over its lifespan."""
    unq = pd.Series(index=t1.index, dtype=float)
    for i, j in t1.items():
        if pd.isna(j): 
            j = num_concurrent.index[-1]
        # Average uniqueness (1/c_t) over the lifespan of the label
        unq.loc[i] = (1.0 / num_concurrent.loc[i:j]).mean()
    return unq

def compute_structural_inertia(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate Structural Inertia: Normalized Regression Slope by its Standard Error.
    Measures 'cleanliness' of a trend.
    """
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
    """
    Calculate Efficiency Ratio (Kaufman): Net Change / Total Absolute Travel.
    Fractal Dimension proxy.
    """
    net_change = (series - series.shift(window)).abs()
    total_travel = series.diff().abs().rolling(window=window).sum()
    return net_change / (total_travel + 1e-8)

def compute_spectral_energy(series: pd.Series, window: int = 100) -> Dict[str, pd.Series]:
    """
    Extract Frequency Domain features using Hilbert Transform and PSD proxy.
    """
    from scipy.signal import hilbert, periodogram
    
    def get_dominant_freq(s):
        if len(s) < 10: return 0.0
        f, pxx = periodogram(s - np.mean(s))
        return f[np.argmax(pxx)]

    def get_phase(s):
        if len(s) < 10: return 0.0
        analytic_signal = hilbert(s - np.mean(s))
        return np.angle(analytic_signal)[-1]
    
    dom_freq = series.rolling(window=window).apply(get_dominant_freq)
    phase = series.rolling(window=window).apply(get_phase)
    
    return {
        'dominant_freq': dom_freq,
        'phase': phase
    }

def seq_bootstrap(t1: pd.Series, close_index: pd.Index, num_samples: Optional[int] = None) -> List[pd.Timestamp]:
    """
    Sequential Bootstrapping AFML Solution:
    Step 1: Pick first sample randomly.
    Step 2: Calculate uniqueness of remaining samples relative to already picked.
    Step 3: Prob of picking next is proportional to uniqueness.
    """
    if num_samples is None:
        num_samples = len(t1)
        
    phi = []
    while len(phi) < num_samples:
        avg_unq = pd.Series(0.0, index=t1.index)
        if len(phi) > 0:
            # Calculate concurrent count only for already picked samples
            t1_phi = t1.loc[phi]
            # count[t] = number of active labels at time t
            count = pd.Series(0, index=close_index)
            for i, j in t1_phi.items():
                if pd.isna(j): j = close_index[-1]
                count.loc[i:j] += 1
            
            # Calculate potential uniqueness if we pick sample k
            for k in t1.index:
                if k in phi: continue
                # new_count[t] = count[t] + 1 if sample k is active at time t
                k_start = k
                k_end = t1.loc[k]
                if pd.isna(k_end): k_end = close_index[-1]
                
                # unq = 1 / (count + 1) for periods where k is active
                # unq = 1 / count for periods where k is NOT active (but count > 0)
                # But we only care about unq of sample k itself
                avg_unq.loc[k] = (1.0 / (count.loc[k_start:k_end] + 1)).mean()
        else:
            # First sample is picked with equal probability
            avg_unq[:] = 1.0
            
        # Normalize to get probabilities
        prob = avg_unq / avg_unq.sum()
        # Pick next sample
        next_sample = np.random.choice(t1.index, p=prob.values)
        phi.append(next_sample)
        
    return phi

def get_sample_weights(t1: pd.Series, num_concurrent: pd.Series, returns: pd.Series) -> pd.Series:
    """
    Calculate sample weights based on average uniqueness and absolute returns (AFML Hardening).
    Formula: w_i = u_bar_i * |ret_i|
    """
    unq = get_sample_uniqueness(t1, num_concurrent)
    abs_ret = returns.abs().reindex(unq.index).fillna(0)
    weights = unq * abs_ret
    
    # Normalize to mean=1
    if weights.sum() > 0:
        weights = weights / weights.sum() * len(weights)
    return weights

def get_weights_by_uniqueness(t1: pd.Series, close_index: pd.Index) -> pd.Series:
    """Simplified sample weighting based only on uniqueness."""
    num_concurrent = get_num_concurrent_events(close_index, t1)
    unq = get_sample_uniqueness(t1, num_concurrent)
    return unq

def get_frac_diff_weights(d: float, size: int) -> np.ndarray:
    """Calculate weights for fractional differentiation."""
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] / k * (d - k + 1)
        w.append(w_k)
    return np.array(w[::-1]).reshape(-1, 1)

def frac_diff_fixed(series: pd.Series, d: float, threshold: float = 1e-5) -> pd.Series:
    """Fixed-window fractional differentiation to ensure stationarity while preserving memory."""
    w = get_frac_diff_weights(d, 100) # Use a window of 100
    w_sum = np.cumsum(abs(w))
    w_sum /= w_sum[-1]
    skip = (w_sum > threshold).argmax()
    w = w[skip:]
    
    res = {}
    for i in range(len(w), series.shape[0]):
        res[series.index[i]] = np.dot(w.T, series.iloc[i-len(w)+1:i+1])[0]
    return pd.Series(res)
