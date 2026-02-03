import numpy as np
import pandas as pd
from extreme_price_movements.risk import TrailingStop

def entry_price_next_hour_open(panel_open, ts_entry, symbol):
    try:
        px = panel_open.loc[ts_entry, symbol]
        return float(px) if pd.notna(px) and px > 0 else np.nan
    except Exception:
        return np.nan

def simulate_trade_hourly(o_s, h_s, l_s, c_s, feats_s, ts_entry, entry_px, side, cfg, max_hold_hours):
    if np.isnan(entry_px) or entry_px <= 0:
        return 0.0, ts_entry, "no_entry"

    ts_sig = ts_entry - pd.Timedelta(hours=1)
    if ts_sig not in feats_s.index:
        atr = 0.02
    else:
        atr = float(feats_s.loc[ts_sig])

    ts_manager = TrailingStop(
        entry_px=entry_px,
        side=side,
        atr_val=atr,
        k_sl=cfg["risk_k_sl"],
        k_trail_start=cfg["risk_k_trail_start"],
        k_trail_dist=cfg["risk_k_trail_dist"]
    )

    end_ts = ts_entry + pd.Timedelta(hours=max_hold_hours)
    path = o_s.loc[ts_entry:end_ts].index
    if len(path) == 0:
        return 0.0, ts_entry, "no_path"

    for ts in path:
        hh = h_s.loc[ts]; ll = l_s.loc[ts]; cc = c_s.loc[ts]
        if np.isnan(hh) or np.isnan(ll) or np.isnan(cc):
            continue

        stopped, exit_px, reason = ts_manager.update(hh, ll, cc)
        if stopped:
            if reason == "ambiguous_neutral":
                return 0.0, ts, reason
            if side == "long":
                return (exit_px / entry_px) - 1.0, ts, reason
            else:
                return (entry_px / exit_px) - 1.0, ts, reason

    last_ts = path[-1]
    last_close = c_s.loc[last_ts]
    if side == "long":
        return (last_close / entry_px) - 1.0, last_ts, "time_exit"
    else:
        return (entry_px / last_close) - 1.0, last_ts, "time_exit"
