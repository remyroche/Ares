import pandas as pd
import numpy as np
from .utils import tprint
import extreme_price_movements.fast_funcs as ff

def select_trade_candidates_hourly(feats, ts, syms, pct=0.05, min_n=10, max_n=60, metric="dist_ema_fast"):
    if ts not in feats[metric].index:
        return [], []
    s = feats[metric].loc[ts, syms].dropna()
    if s.empty:
        return [], []
    n = len(s)
    k = max(min_n, int(n * pct))
    k = min(k, max_n)
    top = s.sort_values(ascending=False).head(k).index.tolist()
    bot = s.sort_values(ascending=True).head(k).index.tolist()

    # Apply Filters (Abs Range 12h > 7% & Vol Z > 1.6 & Sign Consistency > 80%)
    # We calculate sign consistency on the fly for candidates to avoid massive precomputation if not needed
    def get_sign_consistency(sym):
        try:
            # We need the close price series history ending at ts
            # Assuming feats has 'ret1h' or we need panel?
            # feats usually contains derived features.
            # We need raw close or we can approximate with ret1h cumsum?
            # Ideally we have panel. But this function signature only has feats.
            # However, `feats` in live engine context is a dictionary of DataFrames.
            # And often contains "close" if not explicitly removed.
            # But let's check what's available.
            # If "close" is not in feats, we might struggle.
            # But `select_trade_candidates_vectorized` takes panel.
            # `select_trade_candidates_hourly` is used in `engine.py`.
            # In `engine.py`, `simulate_trade_hourly` has `o_s, h_s...`
            # But `select_trade_candidates_hourly` is called in `engine.py` with `feats`.
            # `feats` usually has `ret1h`.
            # Let's assume we can't easily get close history here without panel.
            # BUT: We can use `detect_extreme_movement_candidates` logic if we had panel.
            # Since we don't have panel here, we might need to skip this check or assume it's done elsewhere?
            # Or rely on `select_trade_candidates_vectorized` for training.
            # For live/sim, we might need to pass panel?
            # Let's skip expensive check here if data missing, or use a proxy if available.
            pass
        except:
            pass
        return 1.0 # Default pass if we can't check

    # Wait, the user wants "additional criteria: 80% ...".
    # If I can't implement it here, I should change the signature or use vectorized.
    # The `select_trade_candidates_vectorized` below has `panel` access.
    # `select_trade_candidates_hourly` is less critical for training but used in inference/sim.
    # In `engine.py`, `generate_hourly_signals` calls `select_trade_candidates_hourly`.
    # `generate_hourly_signals` does NOT pass panel.
    # This is a problem for live inference if I enforce this rule.
    # I should probably update `engine.py` to pass panel or calculate consistency beforehand.
    # BUT: `feats` likely has `close` or `ret1h`.
    # Constructing price from returns:
    # prices = (1 + feats["ret1h"][sym]).cumprod()
    # This is close enough for sign consistency check (monotony).

    def apply_filters(candidates):
        filtered = []
        for sym in candidates:
            try:
                r12 = feats["range_12h_pct"].loc[ts, sym]
                vz = feats["volatility_zscore"].loc[ts, sym]

                if r12 > 0.07 and vz > 1.6:
                    # Sign Consistency Check (12h)
                    # Use ret1h to reconstruct path
                    # Look back 12 hours
                    end_loc = feats["ret1h"].index.get_loc(ts)
                    start_loc = max(0, end_loc - 12 + 1)

                    # Need returns segment
                    rets_seg = feats["ret1h"][sym].iloc[start_loc : end_loc+1].values
                    # Reconstruct approx price path (start=1.0)
                    px = np.concatenate([[1.0], np.cumprod(1.0 + rets_seg)])

                    # Use numba function on this small array?
                    # `_numba_sign_consistency_1d` expects an array and window.
                    # Here we passed a window of data. Window size = len(px).
                    # But the function scans.
                    # We can just run the logic manually for single window.

                    # Logic:
                    # 1. Find Min/Max in window
                    idx_min = np.argmin(px)
                    idx_max = np.argmax(px)

                    curr_val = px[-1]
                    local_min = px[idx_min]
                    local_max = px[idx_max]

                    up_move = curr_val - local_min
                    dn_move = local_max - curr_val

                    target_sign = 0
                    anchor_idx = 0

                    if up_move > dn_move:
                        target_sign = 1
                        anchor_idx = idx_min
                    else:
                        target_sign = -1
                        anchor_idx = idx_max

                    # Slice from anchor to end
                    # px has length N+1. rets_seg has length N.
                    # px[0] corresponds to T-12 close (base).
                    # px[1] is T-11 close. rets_seg[0] is T-11 return.
                    # Anchor index in px.
                    # If anchor is at 0 (start), we check all returns.

                    # We need returns FROM anchor+1 to END.
                    # rets indices: 0..N-1.
                    # px indices: 0..N.
                    # If anchor at px[k], next return is rets_seg[k].

                    check_rets = rets_seg[anchor_idx:]
                    if len(check_rets) > 0:
                        signs = np.sign(check_rets)
                        # Count matches (ignore zeros)
                        valid = signs != 0
                        if valid.sum() > 0:
                            matches = (signs[valid] == target_sign).sum()
                            consistency = matches / valid.sum()
                        else:
                            consistency = 0.0
                    else:
                        consistency = 0.0 # Current is extremum?

                    if consistency >= 0.80:
                        filtered.append(sym)

            except KeyError:
                continue
        return filtered

    return apply_filters(top), apply_filters(bot)

def entry_price_next_hour_open(panel_open, ts_entry, symbol):
    tprint(f"Entering function: entry_price_next_hour_open in candidates.py")
    try:
        px = panel_open.loc[ts_entry, symbol]
        return float(px) if pd.notna(px) and px > 0 else np.nan
    except Exception:
        return np.nan

def select_trade_candidates_vectorized(panel, feats, pct=0.05, metric="ret24h"):
    """
    Vectorized candidate selection with time expansion and volatility filtering.
    Optimized using argpartition for faster top-K selection.

    1. Identify Top/Worst pct% performers based on 'metric' (e.g., ret24h).
    2. Expand candidates to t-12, t-8, t-4, t+4, t+8, t+12, t+16.
    3. Filter: Keep only if last 12h High/Low diff >= 8%.

    Returns:
        mask (pd.DataFrame): Boolean mask of valid candidates.
    """
    tprint(f"Entering function: select_trade_candidates_vectorized in candidates.py")

    # 1. Base Selection — fully vectorized rank-based top/bottom K
    if metric not in feats:
        tprint(f"Warning: Metric {metric} not found in feats.")
        return None

    df_metric = feats[metric]
    n_cols = df_metric.shape[1]
    k = max(1, int(n_cols * pct))

    # Rank each row (ascending): rank 1 = smallest, rank n = largest
    # method='first' avoids ties; pct=False gives integer ranks
    ranks = df_metric.rank(axis=1, method='first', na_option='keep')
    valid_counts = df_metric.notna().sum(axis=1)

    # Top K: rank > (valid_count - k);  Bottom K: rank <= k
    # Broadcast valid_counts as a column vector
    vc = valid_counts.values[:, np.newaxis]
    r = ranks.values
    base_mask_arr = (r > (vc - k)) | (r <= k)
    # Mask out NaN positions
    base_mask_arr = base_mask_arr & df_metric.notna().values
    # Mask out rows with too few valid values
    base_mask_arr[valid_counts.values < k, :] = False

    base_mask = pd.DataFrame(base_mask_arr, index=df_metric.index, columns=df_metric.columns)

    # 2. Volatility & Event Filters (Apply BEFORE Expansion)
    # This ensures we select events where conditions were met AT THE TIME of the event.

    # Filter 2: 12h High/Low range > 7%
    if "range_12h_pct" in feats:
        vol_metric = feats["range_12h_pct"]
        vol_mask = vol_metric > 0.07
    else:
        # Fallback if feature missing (legacy)
        c = panel["close"]
        h = panel["high"]
        l = panel["low"]
        roll_h = h.rolling(12).max()
        roll_l = l.rolling(12).min()
        vol_metric = (roll_h - roll_l) / (c + 1e-12)
        vol_mask = vol_metric > 0.07

    # Filter 3: Volatility Z-score > 1.6
    if "volatility_zscore" in feats:
        event_mask = feats["volatility_zscore"] > 1.6
    else:
        event_mask = pd.DataFrame(True, index=vol_mask.index, columns=vol_mask.columns)

    # Filter 4: Sign Consistency > 80%
    # We use Numba optimized function on close prices
    if "close" in panel:
        sc = ff.numba_sign_consistency(panel["close"], 12)
        sc_df = pd.DataFrame(sc, index=panel["close"].index, columns=panel["close"].columns)
        sc_mask = sc_df >= 0.80
    else:
        # Fallback if close not in panel (unlikely)
        sc_mask = pd.DataFrame(True, index=vol_mask.index, columns=vol_mask.columns)

    # Combine Filters into Base Mask
    base_mask = base_mask & vol_mask & event_mask & sc_mask

    # 3. Time Expansion
    # Offsets: t-12, t-8, t-4, t+4, t+8, t+12, t+16
    offsets = [-12, -8, -4, 4, 8, 12, 16]

    expanded_mask = base_mask.copy()
    for off in offsets:
        # Shift mask
        shifted = base_mask.shift(off)
        expanded_mask = expanded_mask | shifted.fillna(False)

    return expanded_mask

def detect_extreme_movement_candidates(
    panel,
    feats,
    ts,
    event_window_hours=12,
    move_threshold=0.07,
    perf_pct=0.10,
    draw_window_hours=8,
    sign_consistency_min=0.80,
):
    """Select top/worst performers gated by signed draw-extreme magnitude and sign consistency."""
    c = panel["close"]
    h = panel["high"]
    l = panel["low"]
    if ts not in c.index:
        return []

    ts_loc = c.index.get_loc(ts)
    start_draw = max(0, ts_loc - int(draw_window_hours) + 1)
    hs = h.iloc[start_draw:ts_loc + 1]
    ls = l.iloc[start_draw:ts_loc + 1]
    cs = c.iloc[start_draw:ts_loc + 1]
    if hs.empty or ls.empty or cs.empty:
        return []

    close_t = c.loc[ts]
    local_low = ls.min(axis=0)
    local_high = hs.max(axis=0)

    if ts_loc < int(draw_window_hours):
        ret_w = close_t / (c.iloc[0] + 1e-12) - 1.0
    else:
        ret_w = close_t / (c.iloc[ts_loc - int(draw_window_hours)] + 1e-12) - 1.0

    up_draw = (close_t - local_low) / (close_t.abs() + 1e-12)
    dn_draw = (local_high - close_t) / (close_t.abs() + 1e-12)
    draw_extreme = pd.Series(np.where(ret_w >= 0, up_draw, dn_draw), index=close_t.index, dtype=np.float32)
    signed_draw_extreme = draw_extreme * np.sign(ret_w).replace(0, np.nan)

    # Agreement is measured only on bars between current bar and the local extremum:
    # - up move: local low -> current
    # - down move: local high -> current
    sign_consistency = pd.Series(index=close_t.index, dtype=np.float32)
    dir_sign = np.sign(ret_w)

    for sym in close_t.index:
        d = dir_sign.get(sym, 0.0)
        if not np.isfinite(d) or d == 0:
            sign_consistency.loc[sym] = 0.0
            continue

        window_close = cs[sym]
        if d > 0:
            ext_ts = ls[sym].idxmin()
        else:
            ext_ts = hs[sym].idxmax()

        try:
            start_pos = window_close.index.get_loc(ext_ts)
        except Exception:
            sign_consistency.loc[sym] = 0.0
            continue

        segment = window_close.iloc[start_pos:]
        if len(segment) < 2:
            sign_consistency.loc[sym] = 0.0
            continue

        seg_rets = segment.pct_change().dropna()
        if seg_rets.empty:
            sign_consistency.loc[sym] = 0.0
            continue

        seg_sign = np.sign(seg_rets)
        valid = seg_sign != 0
        if valid.sum() == 0:
            sign_consistency.loc[sym] = 0.0
            continue

        agree_rate = (seg_sign[valid] == d).mean()
        sign_consistency.loc[sym] = float(agree_rate)

    event_mask = (signed_draw_extreme.abs() >= move_threshold) & (sign_consistency >= sign_consistency_min)
    event_syms = set(event_mask[event_mask].index.tolist())
    if not event_syms:
        return []

    perf_h = max(1, min(int(event_window_hours), len(feats.get("ret1h", c).index)))
    perf_key = f"ret{perf_h}h"
    perf = feats[perf_key].loc[ts].dropna() if perf_key in feats else feats.get("ret1h", c).loc[ts].dropna()
    if perf.empty:
        return []

    k = max(1, int(np.ceil(len(perf) * perf_pct)))
    top = set(perf.nlargest(k).index.tolist())
    bot = set(perf.nsmallest(k).index.tolist())
    return sorted((top | bot) & event_syms)
