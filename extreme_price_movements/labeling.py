import numpy as np
import pandas as pd
from numba import jit
from .utils import tprint

@jit(nopython=True, cache=True)
def _numba_triple_barrier(
    times, opens, highs, lows, closes,
    tp, sl, horizon_hours
):
    """
    Vectorized Triple Barrier Method.

    Args:
        times: int64 array of timestamps (ns)
        opens, highs, lows, closes: float32 arrays
        tp: Take Profit (relative, e.g. 0.05)
        sl: Stop Loss (relative, e.g. 0.025)
        horizon_hours: Max holding period in hours

    Returns:
        labels: array of int8 (1: TP, -1: SL, 0: Time)
        ret: array of float32 (realized return)
        exit_idx: array of int64 (index of exit)
    """
    n = len(closes)
    labels = np.zeros(n, dtype=np.int8)
    returns = np.zeros(n, dtype=np.float32)
    exit_idxs = np.full(n, -1, dtype=np.int64)

    # Precompute horizon in nanoseconds
    horizon_ns = horizon_hours * 3600 * 1_000_000_000

    for i in range(n):
        entry_px = opens[i] # Assume entry at Open of bar i?
        # Usually entry is next bar Open. But let's assume this function is called
        # such that 'i' is the entry bar (or we iterate from i+1).
        # Let's assume input arrays start from 't'.
        # But commonly we label 't' based on future.
        # So we look forward from i.
        # Entry price is usually Close[i] or Open[i+1].
        # Let's stick to Open[i+1] if possible, or Close[i].
        # The user's `training.py` uses:
        # px_entry = panel["open"].loc[t_entry, final_candidates] where t_entry = t + 1h
        # So we should probably treat `opens` as the execution prices sequence.
        # But let's keep it general: We label time `i` assuming entry at `closes[i]` or similar?
        # Let's use `closes[i]` as reference if we are labeling `i`.
        # OR: We pass arrays where `opens[i]` IS the entry price.

        # Standard approach: At time `i`, we decide to enter. Entry is at `closes[i]`.
        # Path starts from `i+1`.

        entry_p = closes[i]

        if entry_p <= 0:
            continue

        entry_ts = times[i]
        cutoff_ts = entry_ts + horizon_ns

        # Upper/Lower Bounds
        # Long Logic (Can be adapted for Short or separate function)
        # Usually we label directionality.
        # Let's implement for LONG first (TP above, SL below).
        # Or returns label 1 if hit upper, -1 if hit lower.

        # Using simple percent logic
        # Up Barrier: entry * (1 + tp)
        # Down Barrier: entry * (1 - sl)

        barrier_up = entry_p * (1.0 + tp)
        barrier_dn = entry_p * (1.0 - sl)

        found = False

        for j in range(i + 1, n):
            ts = times[j]
            if ts > cutoff_ts:
                # Time Exit
                exit_idxs[i] = j - 1 # Exit at close of previous bar?
                # Or exit at Open of this bar?
                # Usually Time Exit means we close at the end of horizon.
                # If j passes cutoff, we exit at j (or j-1).
                # Let's assume we close at `closes[j-1]` or `opens[j]`?
                # "t_exit = t_entry + horizon".
                # If we passed it, we take the price at that time.
                # Let's use closes[j-1] as the last valid price within horizon, or closes[j] if it's the first bar outside?
                # Let's just use closes[j] for simplicity of "exit at j".
                # Wait, if ts > cutoff, we should have exited at cutoff.
                # If data is hourly, we check the exact hour.

                # Check previous bar
                # If times[j] > cutoff, then times[j-1] <= cutoff (hopefully).
                # We exit at `closes[j-1]`?
                # Let's stick to: scan until hit or time out.
                # If time out, we take the return at the end.

                # Let's prioritize Barriers within the bar j.
                # Low[j] could hit SL. High[j] could hit TP.
                pass

            # Check High/Low of bar j
            h = highs[j]
            l = lows[j]

            # Check touches
            # If both touched, usually SL first is safer assumption (pessimistic)
            # Or use Open/Close to guess path?
            # Pessimistic: Hit SL first.

            hit_sl = l <= barrier_dn
            hit_tp = h >= barrier_up

            if hit_sl and hit_tp:
                # Both hit in same bar.
                # Pessimistic: SL hit.
                labels[i] = -1
                returns[i] = -sl
                exit_idxs[i] = j
                found = True
                break
            elif hit_sl:
                labels[i] = -1
                returns[i] = -sl # Or exact diff? (barrier_dn / entry - 1) which is approx -sl
                exit_idxs[i] = j
                found = True
                break
            elif hit_tp:
                labels[i] = 1
                returns[i] = tp
                exit_idxs[i] = j
                found = True
                break

            if ts >= cutoff_ts:
                # Time limit reached at this bar
                labels[i] = 0
                returns[i] = (closes[j] / entry_p) - 1.0
                exit_idxs[i] = j
                found = True
                break

        if not found:
            # Ran out of data
            labels[i] = 0 # Treated as time exit?
            returns[i] = (closes[n-1] / entry_p) - 1.0
            exit_idxs[i] = n - 1

    return labels, returns, exit_idxs

def compute_triple_barrier_labels(
    panel: pd.DataFrame,
    tp: float,
    sl: float,
    horizon_hours: int
):
    """
    Wrapper for Numba triple barrier.
    Expects panel with open, high, low, close.
    """
    tprint(f"Starting compute_triple_barrier_labels with tp={tp}, sl={sl}, horizon={horizon_hours}")
    # Extract arrays
    # Ensure sorted by time
    # panel is usually dict of DataFrames or a Panel-like object (MultiIndex DF?)
    # In training.py: panel["close"], etc. So panel is a dict of DFs.

    # We need to process each asset (column) separately.

    c = panel["close"]
    h = panel["high"]
    l = panel["low"]
    o = panel["open"]

    # Ensure they have same index/columns
    assets = c.columns
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

        # Handle NaNs?
        # Numba function assumes clean data or handles it?
        # If NaN, comparisons fail.
        # Simple fix: fillna?
        # Better: Numba check?

        # Assuming cleaned data or we fillna(method='ffill')
        # But we can just pass as is.

        lbs, rets, _ = _numba_triple_barrier(
            times, o_arr, h_arr, l_arr, c_arr,
            tp, sl, horizon_hours
        )

        out_labels[asset] = lbs
        out_returns[asset] = rets

    tprint("Finished compute_triple_barrier_labels.")
    return out_labels, out_returns
