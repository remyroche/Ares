import pandas as pd
import numpy as np
from .utils import tprint

def select_trade_candidates_hourly(feats, ts, syms, pct=0.05, min_n=10, max_n=60, metric="dist_ema_fast"):
    tprint(f"Entering function: select_trade_candidates_hourly in candidates.py")
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

    # Apply Filters (Abs Range > 7% & Vol Z > 1.6)
    def apply_filters(candidates):
        filtered = []
        for sym in candidates:
            try:
                r24 = feats["range_24h_pct"].loc[ts, sym]
                vz = feats["volatility_zscore"].loc[ts, sym]
                if r24 > 0.07 and vz > 1.6:
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

    # Filter 2: 24h High/Low range > 7%
    if "range_24h_pct" in feats:
        vol_metric = feats["range_24h_pct"]
        vol_mask = vol_metric > 0.07
    else:
        # Fallback if feature missing (legacy)
        c = panel["close"]
        h = panel["high"]
        l = panel["low"]
        roll_h = h.rolling(24).max()
        roll_l = l.rolling(24).min()
        vol_metric = (roll_h - roll_l) / (c + 1e-12)
        vol_mask = vol_metric > 0.07

    # Filter 3: Volatility Z-score > 1.6
    if "volatility_zscore" in feats:
        event_mask = feats["volatility_zscore"] > 1.6
    else:
        event_mask = pd.DataFrame(True, index=vol_mask.index, columns=vol_mask.columns)

    # Combine Filters into Base Mask
    base_mask = base_mask & vol_mask & event_mask

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
    draw_extreme = pd.Series(np.where(ret_w >= 0, up_draw, dn_draw), index=close_t.index, dtype=np.float64)
    signed_draw_extreme = draw_extreme * np.sign(ret_w).replace(0, np.nan)

    # Agreement is measured only on bars between current bar and the local extremum:
    # - up move: local low -> current
    # - down move: local high -> current
    sign_consistency = pd.Series(index=close_t.index, dtype=np.float64)
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
