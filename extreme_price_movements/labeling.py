import numpy as np
import pandas as pd
import platform
from numba import jit, prange
from joblib import Parallel, delayed, cpu_count
from .fast_funcs import simulate_trade_numba
from .utils import tprint

OUT_SL = np.int8(0)
OUT_TO = np.int8(1)
OUT_TP = np.int8(2)
_QUALITY_EPS = 1e-9

@jit(nopython=True, nogil=True, cache=True)
def _clip_scalar(val, a_min, a_max):
    if val < a_min: return a_min
    if val > a_max: return a_max
    return val

@jit(nopython=True, nogil=True, cache=True)
def _soft_squash_pos(val):
    # Monotone compression to avoid hard saturation when raw ratios spike.
    # For val>=0: maps to [0,1).
    if val <= 0.0:
        return 0.0
    return val / (1.0 + val)

# Fast version with serial loop and binary search
@jit(nopython=True, nogil=True, cache=True, parallel=False)
def _numba_triple_barrier_outcomes_fast(
    times, opens, highs, lows, closes, tp_arr, sl_arr, horizon, side, horizons_arr=None
):
    """
    Fast vectorized triple barrier outcomes using binary search for time windows.
    Returns: outcomes, quality, returns, exit_idxs, conflict_j
    """
    # Remove nested import to fix Numba IMPORT_NAME error
    
    n = len(closes)
    outcomes = np.zeros(n, dtype=np.int8)
    quality = np.zeros(n, dtype=np.float32)
    returns = np.zeros(n, dtype=np.float32)
    exit_idxs = np.zeros(n, dtype=np.int64)
    conflict_j = np.full(n, -1, dtype=np.int64)
    mfe_arr = np.zeros(n, dtype=np.float32)
    mae_arr = np.zeros(n, dtype=np.float32)
    time_to_mfe = np.zeros(n, dtype=np.float32)
    time_to_mae = np.zeros(n, dtype=np.float32)
    
    limit_ns_base = int(horizon * 3600 * 1_000_000_000)
    
    if horizons_arr is not None:
        horizons_ns = (horizons_arr * 3600 * 1_000_000_000).astype(np.int64)
    else:
        horizons_ns = np.full(n, limit_ns_base, dtype=np.int64)
    
    cutoff_times = times + horizons_ns
    
    for i in prange(n - 1):
        entry_p = closes[i]
        entry_t = times[i]
        
        activation = tp_arr[i]
        sl = sl_arr[i]
        
        if np.isnan(activation) or np.isnan(sl) or np.isnan(entry_p) or entry_p <= 0:
            continue
        
        limit_ns = horizons_ns[i]
        cutoff_t = cutoff_times[i]
        stall_ns = limit_ns // 2
        stall_t = entry_t + stall_ns
        
        den_tp = max(abs(entry_p * activation), _QUALITY_EPS)
        
        if side == 1:
            sl_price = entry_p * (1.0 - sl)
            tp_price = entry_p * (1.0 + activation)
        else:
            sl_price = entry_p * (1.0 + sl)
            tp_price = entry_p * (1.0 - activation)
        
        j_start = i + 1
        j_end = np.searchsorted(times, cutoff_t, side="right")
        
        if j_end <= j_start:
            outcomes[i] = OUT_TO
            exit_idxs[i] = j_start if j_start < n else n - 1
            continue
        
        mfe_val = 0.0
        mae_val = 0.0
        t_mfe = 0.0
        t_mae = 0.0
        
        for j in range(j_start, min(j_end, n)):
            tt = times[j]
            hh = highs[j]
            ll = lows[j]
            cc = closes[j]
            
            # Update MFE/MAE
            if side == 1:
                cur_mfe = max(0.0, hh - entry_p)
                cur_mae = max(0.0, entry_p - ll)
            else:
                cur_mfe = max(0.0, entry_p - ll)
                cur_mae = max(0.0, hh - entry_p)
            
            if cur_mfe > mfe_val:
                mfe_val = cur_mfe
                t_mfe = (tt - entry_t) / 1e9 / 3600.0
            if cur_mae > mae_val:
                mae_val = cur_mae
                t_mae = (tt - entry_t) / 1e9 / 3600.0
            
            if np.isnan(hh) or np.isnan(ll):
                if tt >= cutoff_t:
                    outcomes[i] = OUT_TO
                    returns[i] = (cc / entry_p - 1.0) if side == 1 else (entry_p / cc - 1.0)
                    exit_idxs[i] = j
                    rel_prog = returns[i] / den_tp
                    quality[i] = 0.5 + _clip_scalar(rel_prog * 0.4, -0.4, 0.4)
                    break
                continue
            
            hit_tp = (hh >= tp_price) if side == 1 else (ll <= tp_price)
            hit_sl = (ll <= sl_price) if side == 1 else (hh >= sl_price)
            
            if hit_tp and hit_sl:
                conflict_j[i] = j
                outcomes[i] = OUT_SL
                returns[i] = -sl
                exit_idxs[i] = j
                qual_raw = (mfe_val / den_tp) * 0.5
                quality[i] = _clip_scalar(_soft_squash_pos(qual_raw), 0.0, 0.49)
                break
            
            if hit_sl:
                outcomes[i] = OUT_SL
                returns[i] = -sl
                exit_idxs[i] = j
                qual_raw = (mfe_val / den_tp) * 0.5
                quality[i] = _clip_scalar(_soft_squash_pos(qual_raw), 0.0, 0.49)
                break
            
            if hit_tp:
                outcomes[i] = OUT_TP
                returns[i] = activation
                exit_idxs[i] = j
                qual_raw = 1.0 - (mae_val / den_tp) * 0.5
                quality[i] = _clip_scalar(_soft_squash_pos(qual_raw), 0.51, 1.0)
                break
            
            if tt >= cutoff_t:
                outcomes[i] = OUT_TO
                returns[i] = (cc / entry_p - 1.0) if side == 1 else (entry_p / cc - 1.0)
                exit_idxs[i] = j
                rel_prog = returns[i] / den_tp
                quality[i] = 0.5 + _clip_scalar(rel_prog * 0.4, -0.4, 0.4)
                break

        mfe_arr[i] = mfe_val / entry_p
        mae_arr[i] = mae_val / entry_p
        time_to_mfe[i] = t_mfe
        time_to_mae[i] = t_mae
    
    return outcomes, quality, returns, exit_idxs, conflict_j, mfe_arr, mae_arr, time_to_mfe, time_to_mae


@jit(nopython=True, nogil=True, cache=True)
def _numba_triple_barrier_outcomes(times, opens, highs, lows, closes, tp_arr, sl_arr, horizon, side, horizons_arr=None):
    """
    Triple barrier labeling returning 3-way outcome and quality scores.

    Outcomes:
        2: TP_FIRST (Profit)
        1: TIMEOUT (Time expiry)
        0: SL_FIRST (Loss)

    Quality Scores (0.0 to 1.0):
        TP: f(MFE, MAE) = 1.0 - (MAE / SL_dist) * 0.5  (Clean win = 1.0, messy win < 1.0)
        SL: f(MAE, MFE) = 0.0 + (MFE / TP_dist) * 0.5  (Bad loss = 0.0, close call loss > 0.0)
        TIMEOUT: 0.5 + (Return / TP_dist) * 0.4        (Centered at 0.5)

    times: int64 (nanoseconds)
    side: 1 for Long, -1 for Short
    horizons_arr: optional float array of horizon hours per row. If None, uses scalar `horizon`.
    """
    n = len(closes)
    outcomes = np.zeros(n, dtype=np.int8)
    quality = np.zeros(n, dtype=np.float32)
    returns = np.zeros(n, dtype=np.float32)
    exit_idxs = np.zeros(n, dtype=np.int64)
    conflict_j = np.full(n, -1, dtype=np.int64)
    mfe_arr = np.zeros(n, dtype=np.float32)
    mae_arr = np.zeros(n, dtype=np.float32)
    time_to_mfe = np.zeros(n, dtype=np.float32)
    time_to_mae = np.zeros(n, dtype=np.float32)
    conflict_j = np.full(n, -1, dtype=np.int64)

    limit_ns_base = int(horizon * 3600 * 1_000_000_000)

    for i in range(n - 1):
        entry_p = closes[i]
        entry_t = times[i]

        activation = tp_arr[i] # TP distance (pct)
        sl = sl_arr[i]         # SL distance (pct)

        if np.isnan(activation) or np.isnan(sl) or np.isnan(entry_p) or entry_p <= 0:
            continue

        if horizons_arr is not None:
             # Per-trade dynamic horizon
             limit_ns = int(horizons_arr[i] * 3600 * 1_000_000_000)
        else:
             limit_ns = limit_ns_base

        cutoff_t = entry_t + limit_ns
        stall_ns = limit_ns // 2
        stall_t = entry_t + stall_ns

        trail_dev = 0.5 * activation
        stall_threshold = 0.5 * activation

        if side == 1:  # Long
            sl_price = entry_p * (1.0 - sl)
            tp_price = entry_p * (1.0 + activation)
        else:  # Short
            sl_price = entry_p * (1.0 + sl)
            tp_price = entry_p * (1.0 - activation)

        exit_found = False

        # Track MFE/MAE for quality calculation
        mfe_val = 0.0 # Max Favorable Excursion (price diff)
        mae_val = 0.0 # Max Adverse Excursion (price diff)

        for j in range(i + 1, n):
            tt = times[j]
            hh = highs[j]
            ll = lows[j]
            cc = closes[j]

            # Update MFE/MAE
            if side == 1:
                cur_mfe = max(0.0, hh - entry_p)
                cur_mae = max(0.0, entry_p - ll)
            else:
                cur_mfe = max(0.0, entry_p - ll)
                cur_mae = max(0.0, hh - entry_p)

            if cur_mfe > mfe_val: mfe_val = cur_mfe
            if cur_mae > mae_val: mae_val = cur_mae

            if tt > cutoff_t:
                # TIMEOUT
                outcomes[i] = OUT_TO
                if side == 1:
                    ret = (opens[j] / entry_p) - 1.0
                else:
                    ret = (entry_p / opens[j]) - 1.0
                returns[i] = ret
                exit_idxs[i] = j

                # Timeout Quality: 0.5 centered, +/- based on progress to TP
                # Cap at 0.1 to 0.9
                den_tp = max(abs(activation), _QUALITY_EPS)
                rel_prog = ret / den_tp
                quality[i] = 0.5 + _clip_scalar(rel_prog * 0.4, -0.4, 0.4)

                exit_found = True
                break

            if np.isnan(hh) or np.isnan(ll):
                if tt == cutoff_t:
                    # Timeout at close
                    outcomes[i] = OUT_TO
                    if side == 1: ret = (cc / entry_p) - 1.0
                    else: ret = (entry_p / cc) - 1.0
                    returns[i] = ret
                    exit_idxs[i] = j

                    den_tp = max(abs(activation), _QUALITY_EPS)
                    rel_prog = ret / den_tp
                    quality[i] = 0.5 + _clip_scalar(rel_prog * 0.4, -0.4, 0.4)

                    exit_found = True
                    break
                continue

            # Check Stops
            hit_tp = False
            hit_sl = False

            if side == 1:
                if ll <= sl_price: hit_sl = True
                if hh >= tp_price: hit_tp = True
            else:
                if hh >= sl_price: hit_sl = True
                if ll <= tp_price: hit_tp = True

            if hit_sl and hit_tp:
                # Record conflict for post-processing
                conflict_j[i] = j

                # Default assume SL if not resolved later
                outcomes[i] = OUT_SL
                returns[i] = -sl
                exit_idxs[i] = j
                den_tp = max(entry_p * abs(activation), _QUALITY_EPS)
                qual_raw = (mfe_val / den_tp) * 0.5
                qual = _soft_squash_pos(qual_raw)
                quality[i] = _clip_scalar(qual, 0.0, 0.49)
                exit_found = True
                break

            if hit_sl:
                outcomes[i] = OUT_SL
                returns[i] = -sl
                exit_idxs[i] = j
                den_tp = max(entry_p * abs(activation), _QUALITY_EPS)
                qual_raw = (mfe_val / den_tp) * 0.5
                qual = _soft_squash_pos(qual_raw)
                quality[i] = _clip_scalar(qual, 0.0, 0.49)
                exit_found = True
                break

            if hit_tp:
                outcomes[i] = OUT_TP
                returns[i] = activation
                exit_idxs[i] = j
                
                # Time Penalty: explicit penalization of duration exposure
                time_elapsed = max(0, tt - entry_t)
                time_penalty = min(0.15, 0.15 * (time_elapsed / max(limit_ns, 1)))
                
                # Win Quality: how much heat did we take?
                # 1.0 - (MAE / SL_dist) * 0.5 - time_penalty
                den_sl = max(entry_p * abs(sl), _QUALITY_EPS)
                mae_ratio = mae_val / den_sl
                qual = 1.0 - (mae_ratio * 0.5) - time_penalty
                quality[i] = _clip_scalar(qual, 0.51, 1.0)
                exit_found = True
                break

            # Exact cutoff at close
            if tt == cutoff_t:
                outcomes[i] = OUT_TO
                if side == 1: ret = (cc / entry_p) - 1.0
                else: ret = (entry_p / cc) - 1.0
                returns[i] = ret
                exit_idxs[i] = j
                den_tp = max(abs(activation), _QUALITY_EPS)
                rel_prog = ret / den_tp
                quality[i] = 0.5 + _clip_scalar(rel_prog * 0.4, -0.4, 0.4)
                exit_found = True
                break

        if not exit_found:
            outcomes[i] = OUT_TO
            if side == 1: returns[i] = (closes[n-1] / entry_p) - 1.0
            else: returns[i] = (entry_p / closes[n-1]) - 1.0
            exit_idxs[i] = n - 1
            den_tp = max(abs(activation), _QUALITY_EPS)
            rel_prog = returns[i] / den_tp
            quality[i] = 0.5 + _clip_scalar(rel_prog * 0.4, -0.4, 0.4)

        mfe_arr[i] = mfe_val / entry_p
        mae_arr[i] = mae_val / entry_p
        time_to_mfe[i] = t_mfe
        time_to_mae[i] = t_mae

    # Numba compatibility: avoid nan_to_num keyword args unsupported in some versions.
    quality = np.nan_to_num(quality)
    quality = np.clip(quality, 0.0, 1.0).astype(np.float32)
    return outcomes, returns, quality, exit_idxs, conflict_j, mfe_arr, mae_arr, time_to_mfe, time_to_mae


@jit(nopython=True, nogil=True, cache=True)
def _resolve_conflicts_with_15m_numba(
    ambiguous_indices,
    conflict_j,
    c_times_ns,
    c_arr,
    tp_arr,
    sl_arr,
    hf_times_ns,
    hf_high,
    hf_low,
    hf_close,
    horizon_ns,
    side_int,
):
    """Resolve ambiguous (TP+SL-in-same-1h-bar) events directly on 15m arrays."""
    n_amb = len(ambiguous_indices)
    resolved = np.zeros(n_amb, dtype=np.int8)  # 0=unresolved, 1=TP, 2=SL
    if n_amb == 0 or len(hf_times_ns) == 0:
        return resolved

    last_c_time = c_times_ns[len(c_times_ns) - 1]
    end_minus_1ms = 1_000_000  # ns

    for a in range(n_amb):
        i = int(ambiguous_indices[a])
        j = int(conflict_j[i])
        if j < 0:
            continue

        entry_p = c_arr[i]
        activation = tp_arr[i]
        sl_pct = sl_arr[i]
        if not np.isfinite(entry_p) or entry_p <= 0.0:
            continue
        if not np.isfinite(activation) or not np.isfinite(sl_pct):
            continue

        tp_price = entry_p * (1.0 + activation) if side_int == 1 else entry_p * (1.0 - activation)
        sl_price = entry_p * (1.0 - sl_pct) if side_int == 1 else entry_p * (1.0 + sl_pct)

        start_t = c_times_ns[j]
        cutoff_t = c_times_ns[i] + horizon_ns
        if cutoff_t > last_c_time:
            cutoff_t = last_c_time
        end_t = cutoff_t - end_minus_1ms
        if end_t < start_t:
            continue

        s = np.searchsorted(hf_times_ns, start_t, side="left")
        e = np.searchsorted(hf_times_ns, end_t, side="right")
        if e <= s:
            continue

        h_min = hf_low[s]
        h_max = hf_high[s]
        for k in range(s + 1, e):
            lv = hf_low[k]
            hv = hf_high[k]
            if lv < h_min:
                h_min = lv
            if hv > h_max:
                h_max = hv
        window_range = h_max - h_min
        range_threshold = entry_p * min(abs(activation), abs(sl_pct))
        if (not np.isfinite(window_range)) or (window_range <= max(range_threshold, 0.0)):
            continue

        for k in range(s, e):
            h15 = hf_high[k]
            l15 = hf_low[k]
            c15 = hf_close[k]
            if not (np.isfinite(h15) and np.isfinite(l15)):
                continue
            hit_tp = (h15 >= tp_price) if side_int == 1 else (l15 <= tp_price)
            hit_sl = (l15 <= sl_price) if side_int == 1 else (h15 >= sl_price)

            if hit_tp and hit_sl:
                d_tp = abs(c15 - h15) if side_int == 1 else abs(c15 - l15)
                d_sl = abs(c15 - l15) if side_int == 1 else abs(c15 - h15)
                resolved[a] = 1 if d_tp < d_sl else 2
                break
            if hit_tp:
                resolved[a] = 1
                break
            if hit_sl:
                resolved[a] = 2
                break

    return resolved

@jit(nopython=True, nogil=True, cache=True, parallel=False)
def _numba_trailing_atr_labeling_fast(
    times, opens, highs, lows, closes, atr_pct,
    k_sl, k_pt, k_tp, horizon_hours
):
    """
    Fast vectorized trailing ATR labeling using binary search for time windows.
    Uses parallel loop for independent entries.
    """
    # Remove nested import to fix Numba IMPORT_NAME error
    
    n = len(closes)
    labels = np.zeros(n, dtype=np.int8)
    returns = np.zeros(n, dtype=np.float32)
    
    limit_ns = horizon_hours * 3600 * 1_000_000_000
    cutoff_times = times + limit_ns
    
    for i in prange(n - 1):
        entry_p = closes[i]
        atr = atr_pct[i]
        
        if np.isnan(entry_p) or entry_p <= 0 or np.isnan(atr) or atr <= 0:
            continue
        
        # Calculate distances
        raw_sl = k_sl * atr
        sl_pct = min(max(raw_sl, 0.02), 0.05)
        raw_pt = k_pt * atr
        act_pct = min(max(raw_pt, 0.05), 0.10)
        raw_tp = k_tp * atr
        trail_pct = min(max(raw_tp, 0.02), 0.04)
        
        sl_dist = sl_pct * entry_p
        act_dist = act_pct * entry_p
        trail_dist = trail_pct * entry_p
        
        # Binary search for end index (O(log n) instead of O(horizon))
        end_idx = np.searchsorted(times, cutoff_times[i], side="right")
        
        if end_idx <= i + 1:
            continue
        
        o_slice = opens[i+1:end_idx]
        h_slice = highs[i+1:end_idx]
        l_slice = lows[i+1:end_idx]
        c_slice = closes[i+1:end_idx]
        
        ret, idx_off, reason = simulate_trade_numba(
            o_slice, h_slice, l_slice, c_slice,
            entry_p, 1,  # Long
            sl_dist, act_dist, trail_dist
        )
        
        returns[i] = ret
        if ret > 0:
            labels[i] = 1
        elif ret < 0:
            labels[i] = -1
        else:
            labels[i] = 0
    
    return labels, returns


@jit(nopython=True, nogil=True, cache=True)
def _numba_trailing_atr_labeling(
    times, opens, highs, lows, closes, atr_pct,
    k_sl, k_pt, k_tp, horizon_hours
):
    """
    Vectorized Labeling using Trailing ATR Strategy.
    Simulates the strategy for every potential entry point.
    """
    n = len(closes)
    labels = np.zeros(n, dtype=np.int8) # 1 if ret > 0, 0 otherwise? Or strict +1/-1?
    # Let's align with triple barrier: 1 (profit), -1 (loss), 0 (time exit or neutral).
    # simulate_trade_numba returns realized return.

    returns = np.zeros(n, dtype=np.float32)

    # We iterate until n-1
    for i in range(n - 1):
        # Entry assumed at Close[i] (signal time)
        # Simulation runs on data from i+1 onwards
        entry_p = closes[i]
        atr = atr_pct[i]

        if np.isnan(entry_p) or entry_p <= 0 or np.isnan(atr) or atr <= 0:
            continue

        # Calculate Distances
        # sl_pct = clamp(k_sl * ATR%, 2%, 5%)
        raw_sl = k_sl * atr
        sl_pct = min(max(raw_sl, 0.02), 0.05)

        # pt_pct (activation) = clamp(k_pt * ATR%, 5%, 10%)
        raw_pt = k_pt * atr
        act_pct = min(max(raw_pt, 0.05), 0.10)

        # tp_pct (trailing dist) = clamp(k_tp * ATR%, 2%, 4%)
        raw_tp = k_tp * atr
        trail_pct = min(max(raw_tp, 0.02), 0.04)

        sl_dist = sl_pct * entry_p
        act_dist = act_pct * entry_p
        trail_dist = trail_pct * entry_p

        # Determine Trade Direction
        # Usually we label for the BEST direction or a specific one?
        # Triple Barrier usually labels "If we go Long, what happens?"
        # TF/MR models will predict direction.
        # But for TRAINING data, we need the "Truth".
        # If we label Long Outcome:
        # If Long makes money -> 1. If Long loses -> -1.
        # If we want to train directional models, we usually train on (Long Result).
        # TF Long Model: y=1 if Long Profit, y=0 if Long Loss.
        # TF Short Model: y=1 if Short Profit, y=0 if Short Loss.

        # Here we simulate LONG. (Assuming symmetric or we call twice?)
        # Let's simulate LONG.
        # Note: simulate_trade_numba takes side_int (1=Long).

        # Slice arrays from i+1
        # Slicing creates views in Numba

        # We also need a horizon limit for the simulation?
        # simulate_trade_numba goes to end of array.
        # We should limit it to horizon_hours?
        # simulate_trade_numba returns "Time Exit" if it hits end.
        # So we should pass a slice of length roughly horizon.
        # Or calculate end index.

        # horizon in indices? We have timestamps.
        # Let's find index where time > t[i] + horizon
        # This search is O(Horizon).
        # Doing it for every i is O(N*Horizon). Acceptable.

        t_entry = times[i]
        limit_ns = horizon_hours * 3600 * 1_000_000_000
        cutoff = t_entry + limit_ns

        # Find end index
        end_idx = n
        for k in range(i + 1, n):
            if times[k] > cutoff:
                end_idx = k
                break

        if end_idx <= i + 1:
            continue

        o_slice = opens[i+1:end_idx]
        h_slice = highs[i+1:end_idx]
        l_slice = lows[i+1:end_idx]
        c_slice = closes[i+1:end_idx]

        ret, idx_off, reason = simulate_trade_numba(
            o_slice, h_slice, l_slice, c_slice,
            entry_p, 1, # Long
            sl_dist, act_dist, trail_dist
        )

        returns[i] = ret
        if ret > 0:
            labels[i] = 1
        elif ret < 0:
            labels[i] = -1
        else:
            labels[i] = 0

    return labels, returns


def compute_trailing_atr_labels(
    panel: pd.DataFrame,
    atr_df: pd.DataFrame,
    k_sl: float, k_pt: float, k_tp: float,
    horizon_hours: int
):
    """
    Computes labels using Trailing ATR Strategy.
    Simulates LONG outcome. For Short, we can flip sign of returns?
    Wait, Short Profit = Price Decrease.
    If we want to train directional models, we usually target "Profitability of Direction".
    For 'build_hourly_training_set_and_weights', it calculates 'y_bin' and 'y_ret'.
    If 'y_ret' is Long Return, then:
       If Trend=Up (Long): PnL = y_ret.
       If Trend=Down (Short): PnL = -y_ret (approx).
    So simulating LONG is sufficient if we assume symmetry or close enough.
    Or we can modify this to return Long Returns, and the caller flips for Short.
    """
    c = panel["close"]
    h = panel["high"]
    l = panel["low"]
    o = panel["open"]

    assets = c.columns
    # Intersection of assets
    valid_assets = [a for a in assets if a in atr_df.columns]

    times = c.index.to_numpy(dtype="datetime64[ns]").view(np.int64)

    out_labels = pd.DataFrame(0, index=c.index, columns=valid_assets, dtype=np.int8)
    out_returns = pd.DataFrame(0.0, index=c.index, columns=valid_assets, dtype=np.float32)

    for asset in valid_assets:
        c_arr = c[asset].to_numpy(dtype=np.float32)
        o_arr = o[asset].to_numpy(dtype=np.float32)
        h_arr = h[asset].to_numpy(dtype=np.float32)
        l_arr = l[asset].to_numpy(dtype=np.float32)
        atr_arr = atr_df[asset].to_numpy(dtype=np.float32)

        # Use fast vectorized version with parallel processing and binary search
        lbs, rets = _numba_trailing_atr_labeling_fast(
            times, o_arr, h_arr, l_arr, c_arr, atr_arr,
            k_sl, k_pt, k_tp, horizon_hours
        )

        out_labels[asset] = lbs
        out_returns[asset] = rets

    return out_labels, out_returns

@jit(nopython=True, nogil=True, cache=True, parallel=False)
def _numba_triple_barrier_fast(
    times, opens, highs, lows, closes, tp_arr, sl_arr, horizon, side, horizons_arr=None
):
    """
    Fast vectorized triple barrier labeling using binary search for time windows.
    Uses parallel loops for independent entries.
    
    Returns: labels, returns, exit_idxs, conflict_j
    """
    # Remove nested import to fix Numba IMPORT_NAME error
    
    n = len(closes)
    labels = np.zeros(n, dtype=np.int8)
    returns = np.zeros(n, dtype=np.float32)
    exit_idxs = np.zeros(n, dtype=np.int64)
    conflict_j = np.full(n, -1, dtype=np.int64)
    mfe_arr = np.zeros(n, dtype=np.float32)
    mae_arr = np.zeros(n, dtype=np.float32)
    time_to_mfe = np.zeros(n, dtype=np.float32)
    time_to_mae = np.zeros(n, dtype=np.float32)
    
    limit_ns_base = int(horizon * 3600 * 1_000_000_000)
    
    # Pre-compute cutoff times for all entries (vectorized)
    if horizons_arr is not None:
        horizons_ns = (horizons_arr * 3600 * 1_000_000_000).astype(np.int64)
    else:
        horizons_ns = np.full(n, limit_ns_base, dtype=np.int64)
    
    cutoff_times = times + horizons_ns
    
    for i in prange(n - 1):
        entry_p = closes[i]
        entry_t = times[i]
        
        activation = tp_arr[i]
        sl = sl_arr[i]
        
        # Early NaN exit
        if np.isnan(activation) or np.isnan(sl) or np.isnan(entry_p) or entry_p <= 0:
            continue
        
        limit_ns = horizons_ns[i]
        cutoff_t = cutoff_times[i]
        stall_ns = limit_ns // 2
        stall_t = entry_t + stall_ns
        
        trail_dev = 0.5 * activation
        stall_threshold = 0.5 * activation
        
        if side == 1:  # Long
            sl_price = entry_p * (1.0 - sl)
            activation_price = entry_p * (1.0 + activation)
        else:  # Short
            sl_price = entry_p * (1.0 + sl)
            activation_price = entry_p * (1.0 - activation)
        
        # Binary search to find time window (O(log n) instead of O(n))
        j_start = i + 1
        j_end = np.searchsorted(times, cutoff_t, side="right")
        
        if j_end <= j_start:
            # No bars within horizon
            labels[i] = OUT_TO
            returns[i] = 0.0
            exit_idxs[i] = j_start if j_start < n else n - 1
            continue
        
        trailing_active = False
        exit_found = False
        stall_checked = False
        extreme = entry_p
        
        mfe_val = 0.0
        mae_val = 0.0
        t_mfe = 0.0
        t_mae = 0.0

        # Scan only within the time window
        for j in range(j_start, min(j_end, n)):
            tt = times[j]
            hh = highs[j]
            ll = lows[j]
            cc = closes[j]
            
            # Handle NaN high/low
            if np.isnan(hh) or np.isnan(ll):
                if tt >= cutoff_t:
                    labels[i] = OUT_TO
                    returns[i] = (cc / entry_p - 1.0) if side == 1 else (entry_p / cc - 1.0)
                    exit_idxs[i] = j
                    exit_found = True
                    break
                continue
            
            # Check barriers
            if side == 1:
                hit_sl = ll <= sl_price
                hit_tp = hh >= activation_price
                if trailing_active:
                    # Update trailing stop
                    if hh > extreme:
                        extreme = hh
                    new_sl = extreme - (trail_dev * entry_p)
                    if new_sl > sl_price:
                        sl_price = new_sl
                        hit_sl = ll <= sl_price  # Re-check with new stop
                else:
                    if hh > extreme:
                        extreme = hh
                    if extreme >= activation_price:
                        trailing_active = True
            else:
                hit_sl = hh >= sl_price
                hit_tp = ll <= activation_price
                if trailing_active:
                    if ll < extreme:
                        extreme = ll
                    new_sl = extreme + (trail_dev * entry_p)
                    if new_sl < sl_price:
                        sl_price = new_sl
                        hit_sl = hh >= sl_price
                else:
                    if ll < extreme:
                        extreme = ll
                    if extreme <= activation_price:
                        trailing_active = True
            
            # Conflict detection
            if hit_sl and hit_tp:
                conflict_j[i] = j
            
            # Exit on SL/TP
            if hit_sl:
                labels[i] = OUT_TP if trailing_active else OUT_SL
                returns[i] = (sl_price / entry_p - 1.0) if side == 1 else (entry_p / sl_price - 1.0)
                exit_idxs[i] = j
                exit_found = True
                break
            
            # Stall check at 50% horizon
            if not stall_checked and not trailing_active and tt >= stall_t:
                stall_checked = True
                if side == 1:
                    mfe = (extreme / entry_p) - 1.0
                else:
                    mfe = (entry_p / extreme) - 1.0
                if mfe < stall_threshold:
                    labels[i] = OUT_TO
                    returns[i] = (cc / entry_p - 1.0) if side == 1 else (entry_p / cc - 1.0)
                    exit_idxs[i] = j
                    exit_found = True
                    break
            
            # Exact cutoff - exit at close
            if tt >= cutoff_t:
                labels[i] = OUT_TO
                returns[i] = (cc / entry_p - 1.0) if side == 1 else (entry_p / cc - 1.0)
                exit_idxs[i] = j
                exit_found = True
                break
        
        if not exit_found:
            # Timeout at end of window or data
            final_idx = min(j_end, n - 1)
            labels[i] = OUT_TO
            returns[i] = (closes[final_idx] / entry_p - 1.0) if side == 1 else (entry_p / closes[final_idx] - 1.0)
            exit_idxs[i] = final_idx

        mfe_arr[i] = mfe_val / entry_p
        mae_arr[i] = mae_val / entry_p
        time_to_mfe[i] = t_mfe
        time_to_mae[i] = t_mae
    
    return labels, returns, exit_idxs, conflict_j, mfe_arr, mae_arr, time_to_mfe, time_to_mae


@jit(nopython=True, nogil=True, cache=True)
def _numba_triple_barrier(times, opens, highs, lows, closes, tp_arr, sl_arr, horizon, side, horizons_arr=None):
    """
    Trailing-profit barrier labeling with early stall exit.
    tp_arr: Activation threshold (relative, >0) — once MFE reaches this, trailing stop activates.
    sl_arr: Stop-loss distance (relative, >0) — fixed from entry.
    Trail deviation = 0.5 * tp_arr (half the activation threshold).
    Stall exit: if after 50% of horizon, MFE < 50% of activation threshold, exit at close.
    times: int64 (nanoseconds)
    opens: used for overshoot time exits (first bar after horizon)
    side: 1 for Long, -1 for Short
    horizons_arr: optional float array of horizon hours per row. If None, uses scalar `horizon`.
    """
    n = len(closes)
    labels = np.zeros(n, dtype=np.int8)
    returns = np.zeros(n, dtype=np.float32)
    exit_idxs = np.zeros(n, dtype=np.int64)
    conflict_j = np.full(n, -1, dtype=np.int64)
    mfe_arr = np.zeros(n, dtype=np.float32)
    mae_arr = np.zeros(n, dtype=np.float32)
    time_to_mfe = np.zeros(n, dtype=np.float32)
    time_to_mae = np.zeros(n, dtype=np.float32)

    limit_ns_base = int(horizon * 3600 * 1_000_000_000)

    for i in range(n - 1):
        entry_p = closes[i]
        entry_t = times[i]

        activation = tp_arr[i]
        sl = sl_arr[i]

        # ── CRITICAL OPTIMIZATION: Early NaN Exit ──
        # Never perform int() casts or horizon arithmetic on NaN values.
        # Numba/LLVM handling of float.nan -> int triggers slow floating-point exception traps.
        # Since 15m structural arrays intentionally contain 75% NaNs, this must be skipped first.
        if np.isnan(activation) or np.isnan(sl) or np.isnan(entry_p) or entry_p <= 0:
            continue

        if horizons_arr is not None:
             limit_ns = int(horizons_arr[i] * 3600 * 1_000_000_000)
        else:
             limit_ns = limit_ns_base

        cutoff_t = entry_t + limit_ns
        stall_ns = limit_ns // 2
        stall_t = entry_t + stall_ns

        trail_dev = 0.5 * activation  # trailing deviation = half activation
        stall_threshold = 0.5 * activation  # MFE must exceed 50% of activation by half-horizon

        if side == 1:  # Long
            sl_price = entry_p * (1.0 - sl)
            activation_price = entry_p * (1.0 + activation)
            extreme = entry_p
        else:  # Short
            sl_price = entry_p * (1.0 + sl)
            activation_price = entry_p * (1.0 - activation)
            extreme = entry_p

        trailing_active = False
        exit_found = False
        stall_checked = False

        for j in range(i + 1, n):
            tt = times[j]
            hh = highs[j]
            ll = lows[j]
            cc = closes[j]

            if tt > cutoff_t:
                # Overshoot: first bar strictly after horizon -> time exit at open.
                labels[i] = OUT_TO
                if side == 1:
                    returns[i] = (opens[j] / entry_p) - 1.0
                else:
                    returns[i] = (entry_p / opens[j]) - 1.0
                exit_idxs[i] = j
                exit_found = True
                break

            if np.isnan(hh) or np.isnan(ll):
                # Even if high/low are missing, still honor exact-cutoff timeout at close.
                if tt == cutoff_t:
                    labels[i] = OUT_TO
                    if side == 1:
                        returns[i] = (cc / entry_p) - 1.0
                    else:
                        returns[i] = (entry_p / cc) - 1.0
                    exit_idxs[i] = j
                    exit_found = True
                    break
                continue

            # Check Stop-Loss and Activation
            hit_sl_this_bar = False
            hit_act_this_bar = False

            if side == 1:
                if ll <= sl_price: hit_sl_this_bar = True
                if hh >= activation_price: hit_act_this_bar = True
            else:
                if hh >= sl_price: hit_sl_this_bar = True
                if ll <= activation_price: hit_act_this_bar = True

            if hit_sl_this_bar and hit_act_this_bar:
                conflict_j[i] = j

            if side == 1:
                if hit_sl_this_bar:
                    ret = (sl_price / entry_p) - 1.0
                    returns[i] = ret
                    labels[i] = OUT_TP if trailing_active else OUT_SL
                    exit_idxs[i] = j
                    exit_found = True
                    break
            else:
                if hit_sl_this_bar:
                    ret = (entry_p / sl_price) - 1.0
                    returns[i] = ret
                    labels[i] = OUT_TP if trailing_active else OUT_SL
                    exit_idxs[i] = j
                    exit_found = True
                    break

            # Update extreme and check activation
            if side == 1:
                if hh > extreme:
                    extreme = hh
                if extreme >= activation_price:
                    trailing_active = True
            else:
                if ll < extreme:
                    extreme = ll
                if extreme <= activation_price:
                    trailing_active = True

            # Early stall exit: at 50% of horizon, check if MFE < 50% of activation
            if not stall_checked and not trailing_active and times[j] >= stall_t:
                stall_checked = True
                if side == 1:
                    mfe_so_far = (extreme / entry_p) - 1.0
                else:
                    mfe_so_far = (entry_p / extreme) - 1.0
                if mfe_so_far < stall_threshold:
                    # Stall exit at close price
                    labels[i] = OUT_TO
                    if side == 1:
                        returns[i] = (cc / entry_p) - 1.0
                    else:
                        returns[i] = (entry_p / cc) - 1.0
                    exit_idxs[i] = j
                    exit_found = True
                    break

            # Ratchet trailing stop
            if trailing_active:
                trail_dist = trail_dev * entry_p
                if side == 1:
                    new_sl = extreme - trail_dist
                    if new_sl > sl_price:
                        sl_price = new_sl
                else:
                    new_sl = extreme + trail_dist
                    if new_sl < sl_price:
                        sl_price = new_sl

            # Exact-cutoff bar: after processing hits on this bar, exit at close if still open.
            if tt == cutoff_t:
                labels[i] = OUT_TO
                if side == 1:
                    returns[i] = (cc / entry_p) - 1.0
                else:
                    returns[i] = (entry_p / cc) - 1.0
                exit_idxs[i] = j
                exit_found = True
                break

        if not exit_found:
            labels[i] = OUT_TO
            if side == 1:
                returns[i] = (closes[n-1] / entry_p) - 1.0
            else:
                returns[i] = (entry_p / closes[n-1]) - 1.0
            exit_idxs[i] = n - 1

    return labels, returns, exit_idxs, conflict_j

def compute_triple_barrier_labels(panel, tp, sl, horizon, side="long", return_outcomes=False, horizons_frame=None, resolve_conflicts=True):
    """
    Computes triple barrier labels for a panel.
    tp: Scalar float OR DataFrame/Series matching panel dimensions.
    sl: Scalar float OR DataFrame/Series matching panel dimensions.
    side: "long" or "short"
    return_outcomes: If True, returns (outcomes, quality, returns).
                     Outcomes: 2=TP, 1=TIMEOUT, 0=SL
    horizons_frame: Optional DataFrame matching panel dimensions with horizon hours (float).
                    Used for adaptive horizon scaling.
    resolve_conflicts: If True, attempts to resolve ambiguous bars by reading higher-fidelity 15m parquets.
    """
    required = {"close", "high", "low"}
    missing = required.difference(panel.keys())
    if missing:
        raise KeyError(f"compute_triple_barrier_labels requires panel keys {sorted(required)}; missing={sorted(missing)}")

    c = panel["close"]
    h = panel["high"]
    l = panel["low"]
    o = panel["open"] if "open" in panel else c

    # OPTIMIZATION: Pre-align all panel DataFrames once before the asset loop
    # This avoids per-asset reindexing overhead
    if not h.index.equals(c.index) or not h.columns.equals(c.columns):
        h = h.reindex(index=c.index, columns=c.columns)
    if not l.index.equals(c.index) or not l.columns.equals(c.columns):
        l = l.reindex(index=c.index, columns=c.columns)
    if not o.index.equals(c.index) or not o.columns.equals(c.columns):
        o = o.reindex(index=c.index, columns=c.columns)

    if horizons_frame is not None:
        if not horizons_frame.index.equals(c.index) or not horizons_frame.columns.equals(c.columns):
            horizons_frame = horizons_frame.reindex(index=c.index, columns=c.columns)

    assets = c.columns
    # Force ns resolution so Numba horizon arithmetic (in ns) is consistent across pandas versions.
    times = c.index.to_numpy(dtype="datetime64[ns]").view(np.int64)

    out_labels = pd.DataFrame(0, index=c.index, columns=assets, dtype=np.int8)
    out_returns = pd.DataFrame(0.0, index=c.index, columns=assets, dtype=np.float32)
    out_quality = None

    if return_outcomes:
        out_quality = pd.DataFrame(0.0, index=c.index, columns=assets, dtype=np.float32)

    side_int = 1 if side == "long" else -1

    # OPTIMIZATION: Fast path for scalar TP/SL - avoid DataFrame creation overhead
    tp_is_scalar = np.isscalar(tp)
    sl_is_scalar = np.isscalar(sl)
    
    if tp_is_scalar:
        tp_scalar_val = float(tp)
    else:
        tp_df = tp
    
    if sl_is_scalar:
        sl_scalar_val = float(sl)
    else:
        sl_df = sl

    def _process_asset(asset):
        c_arr = c[asset].to_numpy(dtype=np.float32)
        o_arr = o[asset].to_numpy(dtype=np.float32)
        h_arr = h[asset].to_numpy(dtype=np.float32)
        l_arr = l[asset].to_numpy(dtype=np.float32)
        
        # OPTIMIZATION: Use scalar directly when TP/SL are uniform
        if tp_is_scalar:
            tp_arr = np.full(len(c_arr), tp_scalar_val, dtype=np.float32)
        else:
            tp_arr = tp_df[asset].to_numpy(dtype=np.float32) if asset in tp_df.columns else np.full(len(c_arr), np.nan, dtype=np.float32)
        
        if sl_is_scalar:
            sl_arr = np.full(len(c_arr), sl_scalar_val, dtype=np.float32)
        else:
            sl_arr = sl_df[asset].to_numpy(dtype=np.float32) if asset in sl_df.columns else np.full(len(c_arr), np.nan, dtype=np.float32)
        
        h_arr_custom = horizons_frame[asset].to_numpy(dtype=np.float32) if (horizons_frame is not None and asset in horizons_frame.columns) else None

        if return_outcomes:
            out, rets, qual, _, conflict_j, mfe_arr, mae_arr, t_mfe, t_mae = _numba_triple_barrier_outcomes_fast(
                times, o_arr, h_arr, l_arr, c_arr, tp_arr, sl_arr, horizon, side_int, horizons_arr=h_arr_custom
            )
            return asset, out, rets, qual, conflict_j, tp_arr, sl_arr, mfe_arr, mae_arr, t_mfe, t_mae
        else:
            lbs, rets, _, conflict_j, mfe_arr, mae_arr, t_mfe, t_mae = _numba_triple_barrier_fast(times, o_arr, h_arr, l_arr, c_arr, tp_arr, sl_arr, horizon, side_int, horizons_arr=h_arr_custom)
            return asset, lbs, rets, None, conflict_j, tp_arr, sl_arr, mfe_arr, mae_arr, t_mfe, t_mae

    # OPTIMIZATION: Use all available cores for parallel processing
    n_jobs_cap = 8
    if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        # Keep memory pressure bounded on Apple Silicon unified memory.
        n_jobs_cap = 4
    n_jobs = min(cpu_count(), len(assets), n_jobs_cap)
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_process_asset)(asset) for asset in assets
    )

    horizon_ns = int(float(horizon) * 3600 * 1_000_000_000)

    # FIX #4: Pre-load 15m HF data once per asset (batch I/O) before iterating results.
    # Avoids one disk read per conflict; loads lazily only for assets that actually have conflicts.
    _hf_cache: dict = {}

    def _get_hf_data(asset: str):
        """Return cached 15m DataFrame for asset; loads from disk on first call."""
        if not resolve_conflicts:
            return pd.DataFrame()
        if asset not in _hf_cache:
            try:
                from extreme_price_movements.hf_data_loader import _load_existing_data
                _hf_cache[asset] = _load_existing_data(asset)
            except Exception as _e:
                tprint(f"labeling: WARNING could not load 15m HF data for {asset}: {_e}")
                _hf_cache[asset] = pd.DataFrame()
        return _hf_cache[asset]

    out_mfe = pd.DataFrame(0.0, index=c.index, columns=assets, dtype=np.float32)
    out_mae = pd.DataFrame(0.0, index=c.index, columns=assets, dtype=np.float32)
    out_t_mfe = pd.DataFrame(0.0, index=c.index, columns=assets, dtype=np.float32)
    out_t_mae = pd.DataFrame(0.0, index=c.index, columns=assets, dtype=np.float32)

    for asset, lbs_or_out, rets, qual, conflict_j, tp_arr, sl_arr, mfe_arr, mae_arr, t_mfe, t_mae in results:
        # Check for conflicts
        ambiguous_indices = np.where(conflict_j != -1)[0]
        if len(ambiguous_indices) > 0 and resolve_conflicts:
            # FIX #4: use cached per-asset 15m data (loaded once above, not per conflict).
            df_15m = _get_hf_data(asset)
            if not df_15m.empty:
                hf_index_ns = (
                    pd.to_datetime(df_15m.index, utc=True)
                    .tz_localize(None)
                    .to_numpy(dtype="datetime64[ns]")
                    .view(np.int64)
                )
                hf_h = df_15m["high"].to_numpy(dtype=np.float32, copy=False)
                hf_l = df_15m["low"].to_numpy(dtype=np.float32, copy=False)
                hf_c = df_15m["close"].to_numpy(dtype=np.float32, copy=False)
                c_arr = c[asset].to_numpy(dtype=np.float32, copy=False)

                resolved_codes = _resolve_conflicts_with_15m_numba(
                    ambiguous_indices.astype(np.int64, copy=False),
                    conflict_j.astype(np.int64, copy=False),
                    times,
                    c_arr,
                    tp_arr,
                    sl_arr,
                    hf_index_ns,
                    hf_h,
                    hf_l,
                    hf_c,
                    horizon_ns,
                    side_int,
                )
                tp_idx = ambiguous_indices[resolved_codes == 1]
                sl_idx = ambiguous_indices[resolved_codes == 2]

                if tp_idx.size > 0:
                    lbs_or_out[tp_idx] = OUT_TP
                    rets[tp_idx] = tp_arr[tp_idx]
                    if return_outcomes and qual is not None:
                        qual[tp_idx] = np.maximum(qual[tp_idx], 0.51)
                if sl_idx.size > 0:
                    lbs_or_out[sl_idx] = OUT_SL
                    if return_outcomes:
                        rets[sl_idx] = -sl_arr[sl_idx]
                        if qual is not None:
                            qual[sl_idx] = np.minimum(qual[sl_idx], 0.49)
                    else:
                        if side_int == 1:
                            rets[sl_idx] = -sl_arr[sl_idx]
                        else:
                            # Keep explicit short-side return formula for consistency.
                            rets[sl_idx] = (1.0 / (1.0 + sl_arr[sl_idx])) - 1.0
            else:
                # FIX #1: log that 15m data is unavailable; label remains the numba default (SL).
                tprint(f"labeling: WARNING no 15m HF data for {asset}; {len(ambiguous_indices)} ambiguous bars left as SL (numba default)")

        out_labels[asset] = lbs_or_out
        out_returns[asset] = rets
        out_mfe[asset] = mfe_arr
        out_mae[asset] = mae_arr
        out_t_mfe[asset] = t_mfe
        out_t_mae[asset] = t_mae
        if return_outcomes and qual is not None:
            out_quality[asset] = qual

    if return_outcomes:
        return out_labels, out_returns, out_quality, out_mfe, out_mae, out_t_mfe, out_t_mae
    return out_labels, out_returns, out_mfe, out_mae, out_t_mfe, out_t_mae
