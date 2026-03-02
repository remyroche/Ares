import pandas as pd
import numpy as np
from .utils import tprint
from scipy.ndimage import binary_dilation

_RANK_CACHE = {}

def select_trade_candidates_hourly(
    feats,
    ts,
    syms,
    pct=0.05,
    min_n=10,
    max_n=60,
    metric="dist_ema_fast",
    min_range_pct=0.07,
    min_vol_zscore=1.6,
):
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

    def apply_filters(candidates):
        filtered = []
        for sym in candidates:
            try:
                r12 = feats["range_12h_pct"].loc[ts, sym]
                vz = feats["volatility_zscore"].loc[ts, sym]

                if r12 > min_range_pct and vz > min_vol_zscore:
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

def select_trade_candidates_vectorized(
    panel,
    feats,
    pct=0.05,
    metric="ret24h",
    min_range_pct=0.07,
    min_vol_zscore=1.5,
    chop_thr=0.5,
    sign_consistency_min=None,
):
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

    # O(N) partial selection via argpartition (no full per-row ranking/sort).
    _rk = (id(df_metric), float(pct), "argpartition_topbot")
    cached = _RANK_CACHE.get(_rk)
    if cached is None:
        arr = df_metric.to_numpy(dtype=np.float32, copy=False)
        valid = np.isfinite(arr)
        valid_counts = valid.sum(axis=1)

        arr_top = np.where(valid, arr, -np.inf)
        arr_bot = np.where(valid, arr, np.inf)
        top_idx = np.argpartition(arr_top, kth=max(n_cols - k, 0), axis=1)[:, -k:]
        bot_idx = np.argpartition(arr_bot, kth=max(k - 1, 0), axis=1)[:, :k]

        rows = np.repeat(np.arange(arr.shape[0], dtype=np.int32), k)
        top_flat = top_idx.reshape(-1)
        bot_flat = bot_idx.reshape(-1)
        top_valid = valid[rows, top_flat]
        bot_valid = valid[rows, bot_flat]

        base_mask_arr = np.zeros_like(valid, dtype=bool)
        base_mask_arr[rows[top_valid], top_flat[top_valid]] = True
        base_mask_arr[rows[bot_valid], bot_flat[bot_valid]] = True
        base_mask_arr[valid_counts < k, :] = False

        _RANK_CACHE[_rk] = (base_mask_arr, valid_counts)
    else:
        base_mask_arr, valid_counts = cached

    base_mask = pd.DataFrame(base_mask_arr, index=df_metric.index, columns=df_metric.columns)

    # 2. Volatility & Event Filters (Apply BEFORE Expansion)
    # This ensures we select events where conditions were met AT THE TIME of the event.

    # Filter 2: 12h High/Low range exceeds configured pct
    if "range_12h_pct" in feats:
        vol_metric = feats["range_12h_pct"]
        vol_mask = vol_metric > min_range_pct
    else:
        # Fallback if feature missing (legacy)
        c = panel["close"]
        h = panel["high"]
        l = panel["low"]
        roll_h = h.rolling(12).max()
        roll_l = l.rolling(12).min()
        vol_metric = (roll_h - roll_l) / (c + 1e-12)
        vol_mask = vol_metric > min_range_pct

    # Filter 3: Volatility Z-score > dynamic threshold
    # Dynamic Z-scaling: scale min_vol_zscore based on global market volatility (48h)
    # We use 'mkt_rv_24h' as a proxy if 'mkt_rv_48h' is not explicitly in feats
    mkt_vol_key = "mkt_rv_48h" if "mkt_rv_48h" in feats else "mkt_rv_24h"
    if mkt_vol_key in feats and "volatility_zscore" in feats:
        mkt_vol = feats[mkt_vol_key]
        mkt_vol_mean = mkt_vol.rolling(24*30, min_periods=100).mean().ffill().bfill()
        # Scale threshold: when market is more volatile, we want higher conviction (higher z-score)
        dynamic_thr = min_vol_zscore * (mkt_vol / (mkt_vol_mean + 1e-12)).clip(0.5, 2.0)
        event_mask = feats["volatility_zscore"] > dynamic_thr
    elif "volatility_zscore" in feats:
        event_mask = feats["volatility_zscore"] > min_vol_zscore
    else:
        event_mask = pd.DataFrame(True, index=vol_mask.index, columns=vol_mask.columns)

    # Filter 5: Chop Filter (discard candidates with high chop_score)
    if "chop_score" in feats:
        chop_mask = feats["chop_score"] < chop_thr
    else:
        chop_mask = pd.DataFrame(True, index=vol_mask.index, columns=vol_mask.columns)

    # Deprecated: sign-consistency gating is intentionally disabled.
    _ = sign_consistency_min

    # Combine filters into base mask
    base_mask = base_mask & vol_mask & event_mask & chop_mask

    # 3. Time Expansion
    # Two-stage expansion (entry-delay windows only, not TBM horizon cap):
    # stage-1 late entries: +2,+4,+6,+8 ; stage-2 early entries: -2,-4
    offsets = [2, 4, 6, 8, -2, -4]

    mask_arr = base_mask.to_numpy(dtype=bool, copy=False)
    if offsets:
        max_lag = int(max(abs(int(o)) for o in offsets))
        struct = np.zeros((2 * max_lag + 1, 1), dtype=bool)
        center = max_lag
        struct[center, 0] = True
        for o in offsets:
            struct[center + int(o), 0] = True
        mask_arr = binary_dilation(mask_arr, structure=struct)

    expanded_mask = pd.DataFrame(mask_arr, index=base_mask.index, columns=base_mask.columns, dtype=bool)
    return expanded_mask

def detect_extreme_movement_candidates(
    panel,
    feats,
    ts,
    event_window_hours=12,
    move_threshold=0.07,
    perf_pct=0.10,
    draw_window_hours=8,
    sign_consistency_min=None,
):
    """Select top/worst performers gated by signed draw-extreme magnitude."""
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

    _ = sign_consistency_min
    event_mask = signed_draw_extreme.abs() >= move_threshold
    event_syms = set(event_mask[event_mask].index.tolist())
    if not event_syms:
        return []

    # Support both DataFrame and numpy array feats
    _fallback = c
    _ret1h = feats.get("ret1h", _fallback)
    if isinstance(_ret1h, pd.DataFrame):
        perf_h = max(1, min(int(event_window_hours), len(_ret1h.index)))
        perf_key = f"ret{perf_h}h"
        perf = feats[perf_key].loc[ts].dropna() if perf_key in feats else _ret1h.loc[ts].dropna()
    else:
        # numpy array mode — use panel close for perf ranking
        perf = (c.loc[ts] / (c.shift(int(event_window_hours)).loc[ts] + 1e-12) - 1.0).dropna()
    if perf.empty:
        return []

    k = max(1, int(np.ceil(len(perf) * perf_pct)))
    top = set(perf.nlargest(k).index.tolist())
    bot = set(perf.nsmallest(k).index.tolist())
    return sorted((top | bot) & event_syms)
