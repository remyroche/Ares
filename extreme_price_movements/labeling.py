import numpy as np
import pandas as pd
from numba import jit
from .fast_funcs import simulate_trade_numba
from .utils import tprint

@jit(nopython=True, cache=True)
def _numba_triple_barrier(
    times, opens, highs, lows, closes,
    tp, sl, horizon_hours
):
    """
    Vectorized Triple Barrier Method (Fixed Barriers).
    Kept for legacy/compatibility.
    """
    n = len(closes)
    labels = np.zeros(n, dtype=np.int8)
    returns = np.zeros(n, dtype=np.float32)
    exit_idxs = np.full(n, -1, dtype=np.int64)
    horizon_ns = horizon_hours * 3600 * 1_000_000_000

    for i in range(n):
        entry_p = closes[i]
        if entry_p <= 0: continue
        entry_ts = times[i]
        cutoff_ts = entry_ts + horizon_ns
        barrier_up = entry_p * (1.0 + tp)
        barrier_dn = entry_p * (1.0 - sl)
        found = False

        for j in range(i + 1, n):
            ts = times[j]
            if ts > cutoff_ts:
                exit_idxs[i] = j - 1
                pass
            h = highs[j]
            l = lows[j]
            hit_sl = l <= barrier_dn
            hit_tp = h >= barrier_up

            if hit_sl and hit_tp:
                labels[i] = -1; returns[i] = -sl; exit_idxs[i] = j; found = True; break
            elif hit_sl:
                labels[i] = -1; returns[i] = -sl; exit_idxs[i] = j; found = True; break
            elif hit_tp:
                labels[i] = 1; returns[i] = tp; exit_idxs[i] = j; found = True; break

            if ts >= cutoff_ts:
                labels[i] = 0; returns[i] = (closes[j] / entry_p) - 1.0; exit_idxs[i] = j; found = True; break

        if not found:
            labels[i] = 0; returns[i] = (closes[n-1] / entry_p) - 1.0; exit_idxs[i] = n - 1

    return labels, returns, exit_idxs

@jit(nopython=True, cache=True)
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
    tprint(f"Starting compute_triple_barrier_labels with tp={tp}, sl={sl}, horizon={horizon_hours}")
    # Extract arrays
    # Ensure sorted by time
    # panel is usually dict of DataFrames or a Panel-like object (MultiIndex DF?)
    # In training.py: panel["close"], etc. So panel is a dict of DFs.

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

def compute_triple_barrier_labels(panel: pd.DataFrame, tp: float, sl: float, horizon_hours: int):
    # Legacy wrapper
    c = panel["close"]
    h = panel["high"]
    l = panel["low"]
    o = panel["open"]
    assets = c.columns
    times = c.index.view(np.int64)
    out_labels = pd.DataFrame(0, index=c.index, columns=assets, dtype=np.int8)
    out_returns = pd.DataFrame(0.0, index=c.index, columns=assets, dtype=np.float32)
    for asset in assets:
    tprint(f"Processing {len(assets)} assets.")
    times = c.index.view(np.int64) # Nanoseconds

    out_labels = pd.DataFrame(0, index=c.index, columns=assets, dtype=np.int8)
    out_returns = pd.DataFrame(0.0, index=c.index, columns=assets, dtype=np.float32)

    for i, asset in enumerate(assets):
        tprint(f"[{i+1}/{len(assets)}] Processing asset: {asset}")
        c_arr = c[asset].to_numpy(dtype=np.float32)
        h_arr = h[asset].to_numpy(dtype=np.float32)
        l_arr = l[asset].to_numpy(dtype=np.float32)
        o_arr = o[asset].to_numpy(dtype=np.float32)
        lbs, rets, _ = _numba_triple_barrier(times, o_arr, h_arr, l_arr, c_arr, tp, sl, horizon_hours)
        out_labels[asset] = lbs
        out_returns[asset] = rets
    return out_labels, out_returns

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

    times = c.index.view(np.int64)

    out_labels = pd.DataFrame(0, index=c.index, columns=valid_assets, dtype=np.int8)
    out_returns = pd.DataFrame(0.0, index=c.index, columns=valid_assets, dtype=np.float32)

    for asset in valid_assets:
        c_arr = c[asset].to_numpy(dtype=np.float32)
        h_arr = h[asset].to_numpy(dtype=np.float32)
        l_arr = l[asset].to_numpy(dtype=np.float32)
        o_arr = o[asset].to_numpy(dtype=np.float32)
        atr_arr = atr_df[asset].to_numpy(dtype=np.float32)

        lbs, rets = _numba_trailing_atr_labeling(
            times, o_arr, h_arr, l_arr, c_arr, atr_arr,
            k_sl, k_pt, k_tp, horizon_hours
        )

        out_labels[asset] = lbs
        out_returns[asset] = rets

    tprint("Finished compute_triple_barrier_labels.")
    return out_labels, out_returns
