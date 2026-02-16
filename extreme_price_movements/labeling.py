import numpy as np
import pandas as pd
from numba import jit
from .fast_funcs import simulate_trade_numba


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

    return out_labels, out_returns

@jit(nopython=True, cache=True)
def _numba_triple_barrier(times, highs, lows, closes, tp_arr, sl_arr, horizon, side):
    """
    Trailing-profit barrier labeling with early stall exit.
    tp_arr: Activation threshold (relative, >0) — once MFE reaches this, trailing stop activates.
    sl_arr: Stop-loss distance (relative, >0) — fixed from entry.
    Trail deviation = 0.5 * tp_arr (half the activation threshold).
    Stall exit: if after 50% of horizon, MFE < 50% of activation threshold, exit at close.
    times: int64 (nanoseconds)
    side: 1 for Long, -1 for Short
    """
    n = len(closes)
    labels = np.zeros(n, dtype=np.int8)
    returns = np.zeros(n, dtype=np.float32)
    exit_idxs = np.zeros(n, dtype=np.int64)

    limit_ns = horizon * 3600 * 1_000_000_000
    stall_ns = limit_ns // 2  # 50% of horizon

    for i in range(n - 1):
        entry_p = closes[i]
        entry_t = times[i]
        cutoff_t = entry_t + limit_ns
        stall_t = entry_t + stall_ns

        if np.isnan(entry_p) or entry_p <= 0:
            continue

        activation = tp_arr[i]
        sl = sl_arr[i]

        if np.isnan(activation) or np.isnan(sl):
            continue

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
            if times[j] >= cutoff_t:
                # Time exit
                labels[i] = 0
                if side == 1:
                    returns[i] = (closes[j] / entry_p) - 1.0
                else:
                    returns[i] = (entry_p / closes[j]) - 1.0
                exit_idxs[i] = j
                exit_found = True
                break

            hh = highs[j]
            ll = lows[j]
            cc = closes[j]
            if np.isnan(hh) or np.isnan(ll):
                continue

            # Check stop-loss
            if side == 1:
                if ll <= sl_price:
                    ret = (sl_price / entry_p) - 1.0
                    returns[i] = ret
                    labels[i] = 1 if trailing_active else -1
                    exit_idxs[i] = j
                    exit_found = True
                    break
            else:
                if hh >= sl_price:
                    ret = (entry_p / sl_price) - 1.0
                    returns[i] = ret
                    labels[i] = 1 if trailing_active else -1
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
                    labels[i] = 0
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

        if not exit_found:
            labels[i] = 0
            if side == 1:
                returns[i] = (closes[n-1] / entry_p) - 1.0
            else:
                returns[i] = (entry_p / closes[n-1]) - 1.0
            exit_idxs[i] = n - 1

    return labels, returns, exit_idxs

def compute_triple_barrier_labels(panel, tp, sl, horizon, side="long"):
    """
    Computes triple barrier labels for a panel.
    tp: Scalar float OR DataFrame/Series matching panel dimensions.
    sl: Scalar float OR DataFrame/Series matching panel dimensions.
    side: "long" or "short"

    Hit logic:
    - TP/SL hit detection is based on intrabar extremes (`high`/`low`), not `close`.
    - `close` is used only for non-hit exits (timeout/stall return calculation).
    """
    required = {"close", "high", "low"}
    missing = required.difference(panel.keys())
    if missing:
        raise KeyError(f"compute_triple_barrier_labels requires panel keys {sorted(required)}; missing={sorted(missing)}")

    c = panel["close"]
    h = panel["high"]
    l = panel["low"]

    # Ensure all price matrices are aligned so hit checks always use matching high/low bars.
    if not h.index.equals(c.index) or not h.columns.equals(c.columns):
        h = h.reindex(index=c.index, columns=c.columns)
    if not l.index.equals(c.index) or not l.columns.equals(c.columns):
        l = l.reindex(index=c.index, columns=c.columns)

    assets = c.columns
    times = c.index.view(np.int64)

    out_labels = pd.DataFrame(0, index=c.index, columns=assets, dtype=np.int8)
    out_returns = pd.DataFrame(0.0, index=c.index, columns=assets, dtype=np.float32)

    side_int = 1 if side == "long" else -1

    # Prepare TP/SL as dataframes if they are scalars
    if np.isscalar(tp):
        tp_df = pd.DataFrame(tp, index=c.index, columns=assets)
    else:
        tp_df = tp

    if np.isscalar(sl):
        sl_df = pd.DataFrame(sl, index=c.index, columns=assets)
    else:
        sl_df = sl

    for asset in assets:
        c_arr = c[asset].to_numpy(dtype=np.float32)
        h_arr = h[asset].to_numpy(dtype=np.float32)
        l_arr = l[asset].to_numpy(dtype=np.float32)

        # Extract TP/SL arrays for this asset
        # Handle case where tp_df might not have the column (if passed as partial df)
        # Assuming alignment for now
        if asset in tp_df.columns:
            tp_arr = tp_df[asset].to_numpy(dtype=np.float32)
        else:
             # Fallback or error? defaulting to NaN effectively skips
             tp_arr = np.full(len(c_arr), np.nan, dtype=np.float32)

        if asset in sl_df.columns:
            sl_arr = sl_df[asset].to_numpy(dtype=np.float32)
        else:
            sl_arr = np.full(len(c_arr), np.nan, dtype=np.float32)

        lbs, rets, _ = _numba_triple_barrier(times, h_arr, l_arr, c_arr, tp_arr, sl_arr, horizon, side_int)

        out_labels[asset] = lbs
        out_returns[asset] = rets

    return out_labels, out_returns
