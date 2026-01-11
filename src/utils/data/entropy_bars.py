
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List

def entropy_bars_abs_return(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Constructs entropy bars based on absolute return accumulation.
    
    Args:
        df: DataFrame with columns ['open','high','low','close','volume']
            indexed at 1-min frequency (or any fixed-time frequency)
        threshold: entropy accumulation threshold
        
    Returns:
        DataFrame of OHLCV bars indexed by end_time, with 'start_idx' and 'end_idx' columns.
    """
    # Ensure required columns exist
    req_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(c in df.columns for c in req_cols):
        raise ValueError(f"Input DataFrame must contain columns: {req_cols}")

    close = df['close'].values
    # Calculate returns (simple returns for speed/proxy)
    # ret[t] = (P[t] - P[t-1]) / P[t-1]
    ret = np.diff(close) / close[:-1]
    ret = np.insert(ret, 0, 0.0)

    # Entropy proxy (causal, cheap): Absolute Return
    entropy = np.abs(ret)

    # Accumulate
    bars = []
    acc = 0.0
    start = 0
    
    # Iterate through ticks/bars
    for t, e in enumerate(entropy):
        acc += e
        if acc >= threshold:
            bars.append((start, t))
            start = t + 1
            acc = 0.0

    # If the last bar is incomplete, we usually drop it or encompass it
    # Here strictly following logic: only yielding completed bars?
    # User's snippet implies strict threshold crossing.
    
    if not bars:
        return pd.DataFrame()

    # Construct OHLCV bars
    out = []
    
    # Pre-fetch columns for speed
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    volumes = df['volume'].values
    indices = df.index
    
    for s, e in bars:
        # Vectorized slice aggregation is faster, but loop is explicit
        o = opens[s]
        h = np.max(highs[s:e+1])
        l = np.min(lows[s:e+1])
        c = closes[e]
        v = np.sum(volumes[s:e+1])
        end_time = indices[e]
        
        out.append((end_time, o, h, l, c, v, s, e))

    out_df = pd.DataFrame(out, columns=[
        'end_time','open','high','low','close','volume','start_idx','end_idx'
    ]).set_index('end_time')

    return out_df

def calibrate_threshold_abs(df: pd.DataFrame, target_bars: int, q_low: float = 0.001, q_high: float = 0.999) -> float:
    """
    Calibrates the entropy threshold to achieve a target number of bars.
    Binary search on quantiles of the entropy distribution.
    """
    close = df['close'].values
    ret = np.diff(close) / close[:-1]
    ret = np.insert(ret, 0, 0.0)
    entropy = np.abs(ret)

    # Determine search range from distribution of individual tick entropies
    # The threshold will be significantly larger than single-tick entropy
    # Actually, threshold ~ (Total Entropy / Target Bars)
    
    # Heuristic: Calculate total entropy and divide by target bars for initial estimate
    total_entropy = np.sum(entropy)
    estimated_threshold = total_entropy / target_bars if target_bars > 0 else np.max(entropy)
    
    # User's procedure uses binary search on quantiles of single-tick entropy? 
    # That seems like it searches for "average bar size" via "tick entropy quantiles" which might be mis-scaled.
    # But let's follow the user's logic structure but fix the range if needed.
    # Wait, the user provided code:
    # low = quantile(entropy, q_low) -> this is per-tick entropy (~0)
    # high = quantile(entropy, q_high) -> this is per-tick max entropy (~0.05)
    # mid = (low+high)/2
    # est = int(entropy.sum() / mid)
    # 
    # If mid is small (per-tick), est (Total / small) will be huge.
    # If we want target_bars (e.g. 1000) from 1M ticks, each bar needs ~1000 ticks worth of entropy.
    # So the threshold should be ~1000 * avg_tick_entropy.
    # The user's provided code seems to search in range of [tick_min, tick_max], which would produce bars of size 1 tick?
    # This looks like a mistake in the user's snippet logic for the SEARCH RANGE.
    # "low = quantile(entropy, q_low)" is tiny.
    # "est = int(entropy.sum() / mid)" -> if mid is tiny, est is huge.
    # 
    # Let's corrected it to search in a reasonable range for *bar* size, not *tick* size.
    # However, strictly following user instructions often means strictly following the code provided if it works?
    # No, "Calibration Procedure... Target: 1 bar / 15 min". 
    # If I use the user's code literal, low/high are single-tick entropies.
    # Threshold will be < max_tick_return.
    # So bars will trigger every time a large candle appears, or every 2-3 small candles.
    # If 15-min bars are expected from 1-min data, we expect ~15 ticks per bar.
    # So threshold should be ~15 * mean(abs(ret)).
    # 
    # I will stick to the INTENT (1 bar / 15 min) and calculate threshold derived from data directly:
    # Threshold = Total_Entropy / Target_Bars.
    # This is analytically the exact threshold that yields *average* bar count = Target,
    # because Sum(Threshold * N_bars) ~ Sum(Entropy).
    #
    # But wait, user provided code:
    # "def calibrate_threshold_abs... for _ in range(30): mid = (low+high)/2 ... return (low+high)/2"
    # I will implement a robust calibration that *works*.
    # I'll use the analytic solution as the center of search if I use search, or just use the analytic solution.
    # Actually, simply: threshold = sum(entropy) / target_bars is the best estimator for *average* frequency.
    #
    # However, to respect the user's "binary search" preference, I'll adapt the search range to be meaningful.
    # Range: [Total/Target * 0.1, Total/Target * 10]
    
    total_entropy = np.sum(entropy)
    if target_bars <= 0: return np.max(entropy) * 10
    
    center_est = total_entropy / target_bars
    low = center_est * 0.1
    high = center_est * 10.0
    
    for _ in range(30):
        mid = (low + high) / 2.0
        # How many bars would this threshold yield?
        # We can approximate count = Total / mid
        # Or run the actual simulation? Simulation is safer but slower.
        # Given "core implementation" is fast (numpy), running sim is okay?
        # User code: est = int(entropy.sum() / mid) -> This assumes linearity.
        
        est = int(entropy.sum() / mid)
        
        if est > target_bars:
            # We have too many bars -> Threshold `mid` is too low.
            low = mid
        else:
            # We have too few bars -> Threshold `mid` is too high.
            high = mid
            
    return (low + high) / 2.0

def build_entropy_bars_15min(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Builds entropy bars calibrated to approximate 15-min frequency.
    """
    # Estimate total duration in minutes if index is datetime
    if isinstance(df.index, pd.DatetimeIndex):
        duration = (df.index[-1] - df.index[0]).total_seconds() / 60.0
        # Target: 1 bar per 15 mins
        target_bars = int(duration / 15.0)
    else:
        # Fallback: assume input is 1-min bars? User said "1-min frequency"
        M = len(df)
        target_bars = int(M / 15)

    if target_bars < 1:
        target_bars = 1

    threshold = calibrate_threshold_abs(df, target_bars)
    bars = entropy_bars_abs_return(df, threshold)
    
    return bars, threshold
