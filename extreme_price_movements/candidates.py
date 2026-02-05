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

    # Apply Filters (Abs Range > 10% & Vol Z > 1.6)
    def apply_filters(candidates):
        filtered = []
        for sym in candidates:
            try:
                r24 = feats["range_24h_pct"].loc[ts, sym]
                vz = feats["volatility_zscore"].loc[ts, sym]
                if r24 > 0.10 and vz > 1.6:
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

    # 1. Base Selection using argpartition (faster than full rank)
    if metric not in feats:
        tprint(f"Warning: Metric {metric} not found in feats.")
        return None

    df_metric = feats[metric]
    n_cols = df_metric.shape[1]
    k = max(1, int(n_cols * pct))
    
    # Use argpartition for top-K and bottom-K (O(n) vs O(n log n) for sorting)
    base_mask = pd.DataFrame(False, index=df_metric.index, columns=df_metric.columns)
    
    for idx in df_metric.index:
        row = df_metric.loc[idx].values
        valid_mask = ~np.isnan(row)
        
        if valid_mask.sum() < k:
            continue
        
        valid_vals = row[valid_mask]
        valid_cols = df_metric.columns[valid_mask]
        
        # Top K indices
        if len(valid_vals) > k:
            top_k_idx = np.argpartition(valid_vals, -k)[-k:]
            bot_k_idx = np.argpartition(valid_vals, k)[:k]
            
            top_cols = valid_cols[top_k_idx]
            bot_cols = valid_cols[bot_k_idx]
            
            base_mask.loc[idx, top_cols] = True
            base_mask.loc[idx, bot_cols] = True

    # 2. Volatility & Event Filters (Apply BEFORE Expansion)
    # This ensures we select events where conditions were met AT THE TIME of the event.

    # Filter 2: 24h High/Low range > 10%
    if "range_24h_pct" in feats:
        vol_metric = feats["range_24h_pct"]
        vol_mask = vol_metric > 0.10
    else:
        # Fallback if feature missing (legacy)
        c = panel["close"]
        h = panel["high"]
        l = panel["low"]
        roll_h = h.rolling(24).max()
        roll_l = l.rolling(24).min()
        vol_metric = (roll_h - roll_l) / (c + 1e-12)
        vol_mask = vol_metric > 0.10

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
