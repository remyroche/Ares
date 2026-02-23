import numpy as np
import pandas as pd

from extreme_price_movements.pnl import CostModel
from extreme_price_movements.pnl_asserts import assert_units
from extreme_price_movements.telemetry.tprint_hooks import emit_bucket_summary
from extreme_price_movements.utils import tprint

from extreme_price_movements.risk import TrailingStop
from extreme_price_movements.training import (
    compute_p_exhaustion_at_t,
    select_best_horizon,
    apply_interaction_toggles,
    scaled_atr_pct
)
from extreme_price_movements.candidates import select_trade_candidates_hourly, entry_price_next_hour_open

def _estimate_global_percentile(score, q50, q90, q95=None, q98=None):
    """Linearly interpolate global percentile for a score magnitude."""
    z = abs(float(score))
    # Fallbacks for missing/None quantiles
    if q50 is None or q90 is None:
        return 0.5  # No calibration data yet — neutral rank
    if q50 <= 0: return 0.5 # protection
    if q90 <= q50: q90 = q50 * 1.5
    if q95 is None: q95 = q90 + 0.5 * (q90 - q50)
    if q98 is None: q98 = q95 + 0.5 * (q95 - q90)
    
    if z <= q50:
        return 0.5 * (z / q50)
    if z <= q90:
        return 0.5 + 0.4 * (z - q50) / (q90 - q50 + 1e-12)
    if z <= q95:
        return 0.9 + 0.05 * (z - q90) / (q95 - q90 + 1e-12)
    if z <= q98:
        return 0.95 + 0.03 * (z - q95) / (q98 - q95 + 1e-12)
    
    # Beyond P98, we linear extrapolate slightly but cap at 0.999
    extrap = 0.98 + 0.01 * (z - q98) / (q98 - q95 + 1e-12)
    return min(extrap, 0.999)


def _compute_barrier_pct(feats_s, ts_sig, atr, cfg):
    """Compute vol-scaled barrier percentage from ATR history.
    
    IMPORTANT: feats_s contains raw price-space ATR (not a fraction).
    scaled_atr_pct clips to [vol_lo, vol_hi] which are fractions (0.03-0.06).
    Fallback must also return a fraction, never raw price-space ATR.
    """
    vol_lo = float(cfg.get("vol_lo", 0.03))
    vol_hi = float(cfg.get("vol_hi", 0.06))
    vol_z_max = float(cfg.get("vol_z_max", 3.0))
    window_base = 24 * 30
    # Safe default: mid-range fraction (NOT raw price-space ATR)
    default_barrier = 0.5 * (vol_lo + vol_hi)

    if ts_sig not in feats_s.index:
        return default_barrier

    end_loc = feats_s.index.get_loc(ts_sig)
    start_loc = max(0, end_loc - window_base * 2)

    if end_loc - start_loc < window_base:
        return default_barrier

    subset = feats_s.iloc[start_loc : end_loc + 1]
    if len(subset) >= window_base:
        win = subset.iloc[-window_base:]
        base = win.median()
        std = win.std()
        z = (atr - base) / (std + 1e-12)
        return scaled_atr_pct(atr, z, base, z_max=vol_z_max, lo=vol_lo, hi=vol_hi)
    return default_barrier


def simulate_trade_hourly(o_s, h_s, l_s, c_s, feats_s, ts_entry, entry_px, side, cfg, max_hold_hours, exchange=None, symbol=None, cost: CostModel | None = None):
    """Simulate a trade using trailing-profit policy.

    Exit logic (all distances are vol-scaled via barrier_pct):
      - Stop-loss: fixed at sl_mult * barrier_pct below entry.
      - Trailing profit: once price reaches activation_mult * barrier_pct in
        profit, a trailing stop activates at trail_mult * barrier_pct from the
        running extreme. The trailing stop only ratchets in the profitable
        direction.
      - Time exit: close at last bar if neither SL nor trailing stop triggered.
    
    If exchange (CCXT) and symbol are provided and use_15m_precision is enabled,
    downloads 15m data for more accurate trailing stop simulation.

    Returns:
        (ret, exit_ts, reason, extras) where extras is a dict with:
          mae_pct: max adverse excursion as % of entry price
          mfe_pct: max favorable excursion as % of entry price
          bars_to_mfe: number of bars until MFE was reached
          sl_pct: stop-loss distance as % of entry price
          tp_pct: activation distance as % of entry price
    """
    _empty_extras = {"mae_pct": 0.0, "mfe_pct": 0.0, "bars_to_mfe": 0, "sl_pct": 0.0, "tp_pct": 0.0}
    if cost is not None:
        assert_units(cost)
    if np.isnan(entry_px) or entry_px <= 0:
        return 0.0, ts_entry, "no_entry", _empty_extras

    ts_sig = ts_entry - pd.Timedelta(hours=1)
    atr = float(feats_s.loc[ts_sig]) if ts_sig in feats_s.index else 0.02
    barrier_pct = _compute_barrier_pct(feats_s, ts_sig, atr, cfg)

    # Risk parameters (vol-scaled) — per-bucket TP/SL comes via cfg overrides
    sl_mult = float(cfg.get("sl_mult", 0.5))
    activation_mult = float(cfg.get("tp_mult", 1.0))      # full trail activation
    trail_mult = float(cfg.get("trail_mult", 0.25))        # trail = 25% of barrier — wide enough to survive bar noise

    # Early invalidation parameters (kill_score = mae_frac + a*bars - b*mfe_frac - c)
    kill_a = float(cfg.get("kill_a", 0.002))    # time penalty per bar
    kill_b = float(cfg.get("kill_b", 1.5))       # MFE credit multiplier
    kill_c = float(cfg.get("kill_c", 0.005))     # baseline tolerance
    kill_min_bars = int(cfg.get("kill_min_bars", 2))  # minimum bars before kill check

    # Profit-protection parameters (absolute % of price, NOT barrier-proportional)
    fee_rate = float(cfg.get("fee_bps", 25.0)) / 10000.0
    be_threshold_pct = float(cfg.get("be_threshold_pct", 0.005))    # Stage 1: BE after MFE >= 0.5%
    be_buffer_pct = float(cfg.get("be_buffer_pct", 2.0 * fee_rate)) # buffer above entry for BE stop
    profit_lock_pct = float(cfg.get("profit_lock_pct", 0.015))      # Stage 2: lock profit after MFE >= 1.5%
    profit_lock_amount = float(cfg.get("profit_lock_amount", 0.003)) # lock 0.3% real profit at stage 2
    giveback_pct = float(cfg.get("giveback_pct", 0.005))            # Max giveback: exit if ret drops this much from peak MFE
    max_loss_pct = float(cfg.get("max_loss_pct", 0.03))             # hard cap: max 3% loss in ret space

    # Sanity: barrier_pct must be a fraction (0.01–0.30), not price-space ATR
    if barrier_pct > 0.30:
        tprint(f"  WARNING: barrier_pct={barrier_pct:.4f} > 30% — likely price-space ATR, clamping to 6%")
        barrier_pct = 0.06
    
    sl_dist = sl_mult * barrier_pct * entry_px
    activation_dist = activation_mult * barrier_pct * entry_px
    trail_dist = trail_mult * barrier_pct * entry_px
    be_threshold_dist = be_threshold_pct * entry_px
    be_buffer_dist = be_buffer_pct * entry_px
    profit_lock_dist = profit_lock_pct * entry_px
    profit_lock_stop_dist = profit_lock_amount * entry_px
    giveback_dist = giveback_pct * entry_px
    max_loss_dist = max_loss_pct * entry_px

    # Clamp SL to max_loss_pct (hard ceiling)
    if sl_dist > max_loss_dist:
        sl_dist = max_loss_dist
    
    # Post-computation sanity: actual distances as fraction of entry must be sane
    sl_frac = sl_dist / entry_px
    tp_frac = activation_dist / entry_px
    if sl_frac > 0.15 or tp_frac > 0.30:
        tprint(f"  WARNING: Insane distances SL={sl_frac:.2%} TP={tp_frac:.2%} "
               f"(barrier={barrier_pct:.4f}, sl_mult={sl_mult}, tp_mult={activation_mult}). "
               f"Clamping barrier to 6%.")
        barrier_pct = 0.06
        sl_dist = sl_mult * barrier_pct * entry_px
        activation_dist = activation_mult * barrier_pct * entry_px
        trail_dist = trail_mult * barrier_pct * entry_px
        if sl_dist > max_loss_dist:
            sl_dist = max_loss_dist

    # Log TP/SL parameters for visibility
    if cfg.get("verbose_risk_logging", False):
        sl_pct_log = (sl_dist / entry_px) * 100
        tp_pct_log = (activation_dist / entry_px) * 100
        trail_pct_log = (trail_dist / entry_px) * 100
        tprint(f"  Risk params: SL={sl_pct_log:.2f}%, TP={tp_pct_log:.2f}%, Trail={trail_pct_log:.2f}% | "
               f"Entry=${entry_px:.2f}, Barrier={barrier_pct*100:.2f}% | "
               f"BE@{be_threshold_pct*100:.1f}%, Lock@{profit_lock_pct*100:.1f}%, MaxLoss={max_loss_pct*100:.1f}%")

    end_ts = ts_entry + pd.Timedelta(hours=max_hold_hours)
    
    # Check if 15m precision is enabled
    use_15m = cfg.get("use_15m_precision", False) and exchange is not None and symbol is not None
    
    if use_15m:
        # Download 15m data for precise simulation
        try:
            from extreme_price_movements.hf_data_loader import get_15m_ohlcv
            df_15m = get_15m_ohlcv(exchange, symbol, ts_entry, max_hold_hours)
            
            if not df_15m.empty:
                # Use 15m bars
                path = df_15m.index
                h_data = df_15m['high']
                l_data = df_15m['low']
                c_data = df_15m['close']
            else:
                # Fallback to 1h if download failed
                path = o_s.loc[ts_entry:end_ts].index
                h_data = h_s
                l_data = l_s
                c_data = c_s
        except Exception as e:
            # Fallback to 1h on error
            tprint(f"WARNING: 15m download failed for {symbol}: {e}, using 1h data")
            path = o_s.loc[ts_entry:end_ts].index
            h_data = h_s
            l_data = l_s
            c_data = c_s
    else:
        # Use 1h bars (original behavior)
        path = o_s.loc[ts_entry:end_ts].index
        h_data = h_s
        l_data = l_s
        c_data = c_s

    # Limit Order Fill Logic (Post-prediction offset)
    use_limit = cfg.get("use_limit_orders", False)
    limit_offset_bps = float(cfg.get("limit_offset_bps", 0.0))
    limit_offset_pct = limit_offset_bps / 10000.0
    
    if use_limit and limit_offset_pct > 0:
        if side == "long":
            limit_px = entry_px * (1.0 - limit_offset_pct)
        else:
            limit_px = entry_px * (1.0 + limit_offset_pct)
            
        # Check first 4 bars for fill (1h or 15m depending on precision)
        fill_window = path[:4]
        filled = False
        fill_ts = ts_entry
        
        for ts in fill_window:
            if ts not in h_data.index or ts not in l_data.index:
                continue
            
            bar_l = l_data.loc[ts]
            bar_h = h_data.loc[ts]
            
            if side == "long" and bar_l <= limit_px:
                filled = True
                fill_ts = ts
                break
            elif side == "short" and bar_h >= limit_px:
                filled = True
                fill_ts = ts
                break
        
        if not filled:
            return 0.0, ts_entry, "limit_not_filled", _empty_extras
        
        # Update entry price and start time to fill event
        entry_px = limit_px
        ts_entry = fill_ts
        # Resume simulation from fill_ts
        path = path[path.get_loc(fill_ts):]
        
        # For this simulation, we'll use 0.20% if filled via limit
        cfg["fee_bps"] = 20.0
        filled_via_limit = True
    else:
        filled_via_limit = False
    
    if len(path) == 0:
        return 0.0, ts_entry, "no_path", _empty_extras

    # Exit limit offset logic
    exit_limit_offset_bps = float(cfg.get("exit_limit_offset_bps", 0.0))
    exit_limit_offset_pct = exit_limit_offset_bps / 10000.0

    # MAE/MFE tracking
    mae_px = 0.0   # max adverse excursion (worst unrealised loss, as positive distance)
    mfe_px = 0.0   # max favorable excursion (best unrealised profit, as positive distance)
    mae_px_4bar = 0.0  # max adverse excursion in first 4 bars
    mfe_px_4bar = 0.0  # max favorable excursion in first 4 bars
    bars_to_mfe = 0
    bar_count = 0

    # Initial stop-loss price
    if side == "long":
        sl_price = entry_px - sl_dist
        activation_price = entry_px + activation_dist
        extreme = entry_px  # highest high seen
    else:
        sl_price = entry_px + sl_dist
        activation_price = entry_px - activation_dist
        extreme = entry_px  # lowest low seen

    # Exit state machine: 0=initial, 1=break-even, 2=tight-trail, 3=full-trail
    exit_stage = 0
    trailing_active = False

    for ts in path:
        if ts not in h_data.index or ts not in l_data.index or ts not in c_data.index:
            continue
        
        hh = h_data.loc[ts]
        ll = l_data.loc[ts]
        cc = c_data.loc[ts]
        
        if np.isnan(hh) or np.isnan(ll) or np.isnan(cc):
            continue

        bar_count += 1

        # Track MAE/MFE before exit checks
        if side == "long":
            adverse = entry_px - ll
            favorable = hh - entry_px
        else:
            adverse = hh - entry_px
            favorable = entry_px - ll

        if adverse > mae_px:
            mae_px = adverse
        if favorable > mfe_px:
            mfe_px = favorable
            bars_to_mfe = bar_count
            
        if bar_count <= 4:
            if adverse > mae_px_4bar:
                mae_px_4bar = adverse
            if favorable > mfe_px_4bar:
                mfe_px_4bar = favorable

        # --- Stage transitions (ratchet up only) ---
        if exit_stage == 0 and be_threshold_dist > 0 and favorable >= be_threshold_dist:
            # Stage 1: Break-even — move stop to entry ± buffer
            if side == "long":
                new_sl = entry_px + be_buffer_dist
                if new_sl > sl_price:
                    sl_price = new_sl
            else:
                new_sl = entry_px - be_buffer_dist
                if new_sl < sl_price:
                    sl_price = new_sl
            exit_stage = 1

        if exit_stage <= 1 and profit_lock_dist > 0 and favorable >= profit_lock_dist:
            # Stage 2: Profit lock — lock real profit
            if side == "long":
                new_sl = entry_px + profit_lock_stop_dist
                if new_sl > sl_price:
                    sl_price = new_sl
            else:
                new_sl = entry_px - profit_lock_stop_dist
                if new_sl < sl_price:
                    sl_price = new_sl
            exit_stage = 2
            trailing_active = True

        if exit_stage == 2 and favorable >= activation_dist:
            # Stage 3: Full trailing stop (same trail_dist, just a label upgrade)
            exit_stage = 3

        # --- Update extreme BEFORE stop check (Bug #1 fix: stale-level prevention) ---
        if side == "long":
            if hh > extreme:
                extreme = hh
        else:
            if ll < extreme:
                extreme = ll

        # --- Ratchet trailing stop based on current stage (before stop check) ---
        if trailing_active and exit_stage >= 2:
            # Full trail distance at all active stages (Bug #3 fix: no half-trail)
            if side == "long":
                new_sl = extreme - trail_dist
                if new_sl > sl_price:
                    sl_price = new_sl
            else:
                new_sl = extreme + trail_dist
                if new_sl < sl_price:
                    sl_price = new_sl

        # --- Max giveback exit: if we had meaningful profit and gave too much back ---
        if giveback_dist > 0 and mfe_px >= profit_lock_dist and mfe_px > 0:
            if side == "long":
                current_favorable = cc - entry_px
            else:
                current_favorable = entry_px - cc
            giveback = mfe_px - max(0.0, current_favorable)
            if giveback >= giveback_dist:
                ret = (cc / entry_px) - 1.0 if side == "long" else (entry_px / cc) - 1.0
                extras = {
                    "mae_pct": mae_px / entry_px,
                    "mfe_pct": mfe_px / entry_px,
                    "bars_to_mfe": bars_to_mfe,
                    "sl_pct": sl_dist / entry_px,
                    "tp_pct": activation_dist / entry_px,
                    "exit_stage": exit_stage,
                    "filled_via_limit": filled_via_limit,
                }
                return ret, ts, "giveback_exit", extras

        # --- Early invalidation: kill trades showing adverse drift without MFE ---
        if exit_stage == 0 and bar_count >= kill_min_bars:
            mae_frac = mae_px / entry_px
            mfe_frac = mfe_px / entry_px
            kill_score = mae_frac + kill_a * bar_count - kill_b * mfe_frac - kill_c
            if kill_score > 0:
                # Apply exit limit padding
                exit_price = cc
                if exit_limit_offset_pct > 0:
                    if side == "long":
                        exit_price = max(cc, cc * (1.0 + exit_limit_offset_pct))
                    else:
                        exit_price = min(cc, cc * (1.0 - exit_limit_offset_pct))

                ret = (exit_price / entry_px) - 1.0 if side == "long" else (entry_px / exit_price) - 1.0
                extras = {
                    "mae_pct": mae_frac,
                    "mfe_pct": mfe_frac,
                    "bars_to_mfe": bars_to_mfe,
                    "sl_pct": sl_dist / entry_px,
                    "tp_pct": activation_dist / entry_px,
                    "exit_stage": exit_stage,
                    "filled_via_limit": filled_via_limit,
                    "exit_limit_bonus": abs(exit_price - cc) / entry_px,
                }
                return ret, ts, "early_invalidation", extras

        # Check stop-loss / trailing-stop hit
        if side == "long":
            hit_sl = ll <= sl_price
        else:
            hit_sl = hh >= sl_price

        if hit_sl:
            # Apply exit limit padding
            exit_price = sl_price
            if exit_limit_offset_pct > 0:
                if side == "long":
                    exit_price = max(sl_price, sl_price * (1.0 + exit_limit_offset_pct))
                else:
                    exit_price = min(sl_price, sl_price * (1.0 - exit_limit_offset_pct))
            
            ret = (exit_price / entry_px) - 1.0 if side == "long" else (entry_px / exit_price) - 1.0
            if exit_stage >= 1:
                reason = "trailing_stop"
            else:
                reason = "stop_loss"
            extras = {
                "mae_pct": mae_px / entry_px,
                "mfe_pct": mfe_px / entry_px,
                "mae_pct_4bar": mae_px_4bar / entry_px,
                "mfe_pct_4bar": mfe_px_4bar / entry_px,
                "bars_to_mfe": bars_to_mfe,
                "sl_pct": sl_dist / entry_px,
                "tp_pct": activation_dist / entry_px,
                "exit_stage": exit_stage,
                "filled_via_limit": filled_via_limit,
                "exit_limit_bonus": abs(exit_price - sl_price) / entry_px,
            }
            return ret, ts, reason, extras

    # Time exit
    last_ts = path[-1]
    last_close = c_data.loc[last_ts]
    
    # Apply exit limit padding
    exit_price = last_close
    if exit_limit_offset_pct > 0:
        if side == "long":
            exit_price = max(last_close, last_close * (1.0 + exit_limit_offset_pct))
        else:
            exit_price = min(last_close, last_close * (1.0 - exit_limit_offset_pct))
            
    extras = {
        "mae_pct": mae_px / entry_px,
        "mfe_pct": mfe_px / entry_px,
        "mae_pct_4bar": mae_px_4bar / entry_px,
        "mfe_pct_4bar": mfe_px_4bar / entry_px,
        "bars_to_mfe": bars_to_mfe,
        "sl_pct": sl_dist / entry_px,
        "tp_pct": activation_dist / entry_px,
        "exit_stage": exit_stage,
        "filled_via_limit": filled_via_limit,
        "exit_limit_bonus": abs(exit_price - last_close) / entry_px,
    }
    if side == "long":
        return (exit_price / entry_px) - 1.0, last_ts, "time_exit", extras
    else:
        return (entry_px / exit_price) - 1.0, last_ts, "time_exit", extras



def _robust_norm(val, center, scale, eps=1e-12):
    return (float(val) - float(center)) / (float(scale) + eps)


def _bucket_mode_from_side_dom(side, dom):
    if dom == "tf":
        return "best" if side == "long" else "worst"
    return "worst" if side == "long" else "best"


def _calculate_disagreement_features(meta_data, h_preds, kind_name):
    """Replicate training.py disagreement logic."""
    p2 = h_preds.get(f"pred_{kind_name}_H2")
    p4 = h_preds.get(f"pred_{kind_name}_H4")
    p8 = h_preds.get(f"pred_{kind_name}_H8")
    if p2 is not None and p4 is not None and p8 is not None:
        stack = np.vstack([p2, p4, p8]).T.astype(np.float32)
        pair_abs = (np.abs(p2 - p4) + np.abs(p2 - p8) + np.abs(p4 - p8)) / 3.0
        vote_p = (stack > 0.5).mean(axis=1).astype(np.float32)
        meta_data[f"disagree_{kind_name}_std"] = np.std(stack, axis=1, dtype=np.float32)
        meta_data[f"disagree_{kind_name}_range"] = np.max(stack, axis=1) - np.min(stack, axis=1)
        meta_data[f"disagree_{kind_name}_pair_abs"] = pair_abs
        meta_data[f"disagree_{kind_name}_vote_mix"] = 4.0 * vote_p * (1.0 - vote_p)
        return pair_abs
    return None

def _meta_predict_or_fallback(meta_model, p_alpha, grp_df, label, side_key, mr_h_preds, tf_h_preds, cfg=None):
    """Predict with meta model; fall back to raw alpha if meta output is degenerate."""
    if meta_model is None:
        return (p_alpha - 0.5) * 0.1, np.ones(len(p_alpha), dtype=bool)

    num = grp_df.select_dtypes(include=[np.number]).copy()
    X_meta = meta_model.prepare_meta_features(p_alpha, num, pred_col_name="pred_logit")

    # REPLICATE training.py derived feature logic
    # 1. Per-horizon logit features
    from scipy.special import logit as _logit_fn
    _logit_parts = []
    for h in [2, 4, 8]:
        # Collect from both h_preds sets
        ph = mr_h_preds.get(f"pred_H{h}")
        if ph is None:
            ph = tf_h_preds.get(f"pred_H{h}")
        
        if ph is not None:
            X_meta[f"pred_H{h}"] = ph.astype(np.float32)
            _p_clip = np.clip(ph.astype(float), 1e-4, 1 - 1e-4)
            _lg_h = np.clip(_logit_fn(_p_clip), -4.0, 4.0)
            X_meta[f"pred_logit_H{h}"] = _lg_h.astype(np.float32)
            _logit_parts.append(_lg_h)
    
    # Store all individual scale predictions (meta model often selects them)
    for k, v in mr_h_preds.items():
        X_meta[k] = v.astype(np.float32)
    for k, v in tf_h_preds.items():
        X_meta[k] = v.astype(np.float32)

    # 2. Disagreement features
    pair_abs_mr = _calculate_disagreement_features(X_meta, mr_h_preds, "mr")
    pair_abs_tf = _calculate_disagreement_features(X_meta, tf_h_preds, "tf")

    # 3. Cross-kind agreement
    if pair_abs_mr is not None and pair_abs_tf is not None:
        agree_mr_avg = (1.0 - np.clip(pair_abs_mr, 0.0, 1.0)).astype(np.float32)
        agree_tf_avg = (1.0 - np.clip(pair_abs_tf, 0.0, 1.0)).astype(np.float32)
        X_meta["agree_tf_minus_mr_avg"] = agree_tf_avg - agree_mr_avg

    # 4. Cross-kind per-horizon diff
    for h in [2, 4, 8]:
        pmr = mr_h_preds.get(f"pred_mr_H{h}")
        ptf = tf_h_preds.get(f"pred_tf_H{h}")
        if pmr is not None and ptf is not None:
            X_meta[f"tf_minus_mr_H{h}"] = (ptf - pmr).astype(np.float32)

    # 5. Core Interaction features (must match training.py:4246)
    if "pred_logit" in X_meta.columns:
        pl = X_meta["pred_logit"].values
        for interact_feat in ["vol_z", "mkt_rv_ratio", "ambig", "exh_qual", "trend_pct",
                              "trend_t", "trend_z_t", "spike_score", "grind_score", "chop_score"]:
            if interact_feat in X_meta.columns:
                X_meta[f"pred_x_{interact_feat}"] = pl * X_meta[interact_feat].values
        
        # Regime bucket interactions (G_VOL, G_TREND)
        for rcol in ["G_VOL", "G_TREND"]:
            if rcol in grp_df.columns:
                rv = grp_df[rcol].values
                for bkt in [0, 1, 2]:
                    X_meta[f"pred_x_{rcol}_{bkt}"] = pl * (rv == bkt).astype(float)

        # 6. Granular regime interactions (vol, volume, trend at 12h/48h)
        # Replicating training.py:2667 map
        _regime_map = {
            "vol_12h": "rv_12h",
            "vol_48h": "rv_24h",
            "volume_12h": "vol_z_base",
            "volume_48h": "vol_z24_base",
            "trend_12h": "ret6h",
            "trend_48h": "trend_pct_base",
        }
        
        boundaries = cfg.get("granular_regime_boundaries", {}) if cfg else {}
        
        for rname, src_col in _regime_map.items():
            if src_col in grp_df.columns:
                # Source data exists: ensure derived columns are created to satisfy coverage check
                # Even if we don't have enough valid symbols, we must have the columns.
                for bkt in [0, 1, 2]:
                    X_meta[f"pred_x_{rname}_{bkt}"] = 0.0
                
                vals = grp_df[src_col].values.astype(float)
                valid_mask = np.isfinite(vals)
                
                # Check for stable pre-calculated boundaries first
                terciles = boundaries.get(rname)
                
                # If no stable boundaries, fallback to dynamic cross-sectional terciles
                if terciles is None and valid_mask.sum() > 5:
                    try:
                        terciles = np.nanpercentile(vals[valid_mask], [33.3, 66.7]).tolist()
                    except Exception:
                        terciles = None
                
                if terciles:
                    # Apply thresholds
                    mask0 = (vals <= terciles[0])
                    mask1 = (vals > terciles[0]) & (vals < terciles[1])
                    mask2 = (vals >= terciles[1])
                    
                    X_meta[f"pred_x_{rname}_0"] = pl * mask0.astype(float)
                    X_meta[f"pred_x_{rname}_1"] = pl * mask1.astype(float)
                    X_meta[f"pred_x_{rname}_2"] = pl * mask2.astype(float)

                    # Also store raw regime buckets (0, 1, 2) for agreement features
                    # mapping: Low=0, Mid=1, High=2
                    grp_df[f"__regime_{rname}__"] = (mask1.astype(int) + 2 * mask2.astype(int)).astype(float)
                elif valid_mask.sum() > 0:
                    # Sparse data and no boundaries: fallback to mid-bucket (1)
                    X_meta[f"pred_x_{rname}_1"] = pl * valid_mask.astype(float)
                    grp_df[f"__regime_{rname}__"] = 1.0

        # 7. Cross-temporal / Regime agreement features (Replicating training.py:4270-4284)
        if "trend_slope_48h" in grp_df.columns and "trend_slope_120h" in grp_df.columns:
            ts48 = grp_df["trend_slope_48h"].values
            ts120 = grp_df["trend_slope_120h"].values
            X_meta["trend_slope_ratio_48_120"] = np.where(
                np.abs(ts120) > 1e-9, ts48 / np.clip(np.abs(ts120), 1e-9, None), 0.0).astype(np.float32)

        if "__regime_vol_12h__" in grp_df.columns and "__regime_vol_48h__" in grp_df.columns:
            v12 = grp_df["__regime_vol_12h__"].values
            v48 = grp_df["__regime_vol_48h__"].values
            X_meta["vol_regime_agree"] = (v12 == v48).astype(np.float32)
            X_meta["vol_regime_diff"] = (v12 - v48).astype(np.float32)

        if "__regime_trend_12h__" in grp_df.columns and "__regime_trend_48h__" in grp_df.columns:
            t12 = grp_df["__regime_trend_12h__"].values
            t48 = grp_df["__regime_trend_48h__"].values
            X_meta["trend_regime_agree"] = (t12 == t48).astype(np.float32)
            X_meta["trend_regime_diff"] = (t12 - t48).astype(np.float32)

    if meta_model.selected_features:
        available = set(X_meta.columns)
        selected = set(meta_model.selected_features)
        present = selected & available
        missing = selected - available
        coverage = len(present) / max(len(selected), 1)
        if coverage < 1.0:
            tprint(f"  Meta {side_key}_{label}: DISABLED — feature coverage {coverage:.0%} "
                   f"({len(missing)} missing of {len(selected)})")
            if missing:
                missing_list = sorted(list(missing))
                tprint(f"    Missing keys: {missing_list[:10]} {'...' if len(missing_list) > 10 else ''}")
            # Strict: No signal if coverage is incomplete
            return np.zeros(len(p_alpha), dtype=np.float64), np.ones(len(p_alpha), dtype=bool)
        if missing:
            tprint(f"  Meta {side_key}_{label}: {len(missing)} features missing "
                   f"(coverage {coverage:.0%}), filling with 0")
        X_meta = X_meta.reindex(columns=meta_model.selected_features, fill_value=0.0)
    
    if hasattr(meta_model, "predict_proba"):
        probs = meta_model.predict_proba(X_meta)
        # Class 2 is TP, Class 0 is SL. Score by EV proxy.
        s = probs[:, 2] * 2.0 - probs[:, 0] * 1.0
    else:
        s = meta_model.predict(X_meta)
        
    # Variance gate: only check on large batches (>=10 symbols).
    # On small batches, predictions can legitimately be similar.
    # The _center_scale degenerate guard in pipeline_steps.py
    # already protects against systematic degeneracy.
    if len(s) >= 10 and np.std(s) < 1e-6:
        tprint(f"  Meta {side_key}_{label}: DISABLED — prediction std={np.std(s):.2e} (degenerate, n={len(s)})")
        return (p_alpha - 0.5) * 0.1, np.ones(len(p_alpha), dtype=bool)
    return s, np.zeros(len(p_alpha), dtype=bool)


def _build_side_score_df(ts_sig, feats, mkt_gates, model_bundle, cfg, p_exh_cand, current_positions_syms, tradeable_candidates=None):
    if ts_sig not in mkt_gates.index:
        return pd.DataFrame()

    candidates = set(tradeable_candidates or [])
    if not candidates:
        lookback_offsets = [0, 4, 8, 12, 16]
        for offset in lookback_offsets:
            t_check = ts_sig - pd.Timedelta(hours=offset)
            if t_check in feats["ret24h"].index:
                top, bot = select_trade_candidates_hourly(
                    feats,
                    t_check,
                    list(feats["ret24h"].columns),
                    cfg["trade_extreme_pct"],
                    cfg["trade_extreme_min"],
                    cfg["trade_extreme_max"],
                    cfg["trade_deviation_metric"],
                    cfg.get("train_min_range_pct", 0.07),
                    cfg.get("train_min_vol_zscore", 1.6),
                )
                candidates.update(top)
                candidates.update(bot)

    candidates = [s for s in candidates if s not in current_positions_syms]
    if not candidates:
        return pd.DataFrame()

    mrk = mkt_gates.loc[ts_sig]
    ts_lag = ts_sig - pd.Timedelta(hours=1)

    alpha_models = model_bundle["alpha_models"]
    meta_models = model_bundle["meta_models"]
    spike_model = model_bundle.get("spike_model")

    # Determine trend direction (Best vs Worst) for each candidate
    trend_map = {}
    metric_name = cfg.get("trade_deviation_metric", "dist_ema_fast")
    if metric_name in feats:
        try:
            m_vals = feats[metric_name].loc[ts_sig, list(candidates)]
            for s, v in m_vals.items():
                if np.isfinite(v):
                    trend_map[s] = 1 if v > 0 else -1
        except KeyError:
            pass

    rows = []
    for sym in candidates:
        try:
            t_dir = trend_map.get(sym, 0)
            if t_dir == 0: continue # Skip if no trend info

            p_lag = 0.5
            if ts_lag in p_exh_cand.index and sym in p_exh_cand.columns:
                p_lag = float(p_exh_cand.loc[ts_lag, sym])
            for side_key in ["long", "short"]:
                m_bundle = alpha_models.get(side_key)
                if not m_bundle or not m_bundle.get("mr") or not m_bundle.get("tf"):
                    continue
                model_mr = m_bundle["mr"]["model"]
                model_tf = m_bundle["tf"]["model"]
                fcols_mr = m_bundle["mr"]["feat_cols"]
                fcols_tf = m_bundle["tf"]["feat_cols"]
                # Multi-horizon models for averaging
                mr_by_h = m_bundle["mr"].get("models_by_h", {})
                tf_by_h = m_bundle["tf"].get("models_by_h", {})
                rec = {
                    "symbol": sym, "side_key": side_key, "model_mr": model_mr, "model_tf": model_tf,
                    "feat_cols_mr": fcols_mr, "feat_cols_tf": fcols_tf,
                    "mr_models_by_h": mr_by_h, "tf_models_by_h": tf_by_h,
                    "mkt_ret24h": float(mrk["mkt_ret24h"]), "mkt_ret6h": float(mrk["mkt_ret6h"]),
                    "mkt_trend": float(mrk["mkt_trend"]), "mkt_rv": float(mrk["mkt_rv"]),
                    "G_VOL": int(mrk["G_VOL"]), "G_TREND": int(mrk["G_TREND"]), "p_exh_lag1": p_lag,
                    "trend_dir": t_dir
                }
                # Collect feature columns from all horizons
                all_fcols_mr = set(fcols_mr)
                all_fcols_tf = set(fcols_tf)
                for _h_info in mr_by_h.values():
                    all_fcols_mr |= set(_h_info.get("feat_cols", []))
                for _h_info in tf_by_h.values():
                    all_fcols_tf |= set(_h_info.get("feat_cols", []))
                
                # IMPORTANT: Include source features for meta-interaction terms
                source_regime_features = {"rv_12h", "rv_24h", "vol_z_base", "vol_z24_base", "ret6h", "trend_pct_base"}
                interaction_base_features = {"vol_z", "mkt_rv_ratio", "ambig", "exh_qual", "trend_pct", "trend_t", "trend_z_t", "spike_score", "grind_score", "chop_score"}

                all_keys = all_fcols_mr | all_fcols_tf | set(cfg.get("spike_feature_keys", [])) | set(cfg.get("meta_feature_keys", [])) | source_regime_features | interaction_base_features
                
                for k in all_keys:
                    if k in feats and sym in feats[k].columns:
                        rec[k] = float(feats[k].loc[ts_sig, sym])
                rows.append(rec)
        except Exception:
            continue

    df_all = pd.DataFrame(rows)
    if df_all.empty:
        return pd.DataFrame()

    spike_keys = cfg.get("spike_feature_keys", [])
    if spike_model:
        if isinstance(spike_model, dict):
            gmm = spike_model["gmm"]
            scaler = spike_model.get("scaler")
            spike_cols = spike_model.get("columns", spike_keys)
            available_cols = [c for c in spike_cols if c in df_all.columns]
            X_spike = df_all[available_cols].fillna(0.0).values
            if scaler is not None:
                X_spike = scaler.transform(X_spike)
            probs = gmm.predict_proba(X_spike)
        else:
            X_spike = df_all[spike_keys].fillna(0.0)
            probs = spike_model.predict_proba(X_spike)
        for i in range(probs.shape[1]):
            df_all[f"spike_prob_{i}"] = probs[:, i]
    else:
        for i in range(4):
            df_all[f"spike_prob_{i}"] = 0.0
    
    # Specialist Models: Trap (quality filter) and Gamma (volatility prediction)
    specialist_models = model_bundle.get("specialist_models", {})
    
    # Trap Specialist: Quality Score
    trap_model = specialist_models.get("trap_model") if specialist_models else None
    if trap_model:
        trap_cols = trap_model["columns"]
        available_trap_cols = [c for c in trap_cols if c in df_all.columns]
        if len(available_trap_cols) == len(trap_cols):
            X_trap = df_all[trap_cols].fillna(0.0).values
            X_trap_scaled = trap_model["scaler"].transform(X_trap)
            trap_probs = trap_model["gmm"].predict_proba(X_trap_scaled)
            cluster_order = trap_model["cluster_order"]
            
            # Quality Score = Weighted sum (0=Trap, 3=Premium)
            quality_weights = np.array([0.0, 0.33, 0.67, 1.0])
            quality_score = trap_probs @ quality_weights[cluster_order]
            df_all["trap_quality"] = quality_score
        else:
            df_all["trap_quality"] = 1.0  # Default to accepting all signals
    else:
        df_all["trap_quality"] = 1.0
    
    # Gamma Specialist: Predicted Volatility
    gamma_model = specialist_models.get("gamma_model") if specialist_models else None
    if gamma_model and gamma_model.selected_features_:
        gamma_cols = gamma_model.selected_features_
        available_gamma_cols = [c for c in gamma_cols if c in df_all.columns]
        if len(available_gamma_cols) == len(gamma_cols):
            X_gamma = df_all[gamma_cols].fillna(0.0)
            predicted_vol = gamma_model.predict(X_gamma)
            df_all["predicted_vol_6h"] = predicted_vol
        else:
            df_all["predicted_vol_6h"] = 1.0  # Default to normal volatility
    else:
        df_all["predicted_vol_6h"] = 1.0

    score_rows = []
    for side_key, grp in df_all.groupby("side_key"):
        first = grp.iloc[0]
        model_mr = first["model_mr"]; model_tf = first["model_tf"]
        fcols_mr = first["feat_cols_mr"]; fcols_tf = first["feat_cols_tf"]
        mr_by_h = first.get("mr_models_by_h", {})
        tf_by_h = first.get("tf_models_by_h", {})

        keys_mr = cfg.get("mr_feature_keys", cfg["causal_cols"])
        keys_tf = cfg.get("tf_feature_keys", cfg["causal_cols"])

        # Store individual horizon predictions for meta-features
        mr_h_preds = {}
        tf_h_preds = {}

        # Multi-horizon MR prediction
        if mr_by_h:
            mr_preds_list = []
            for _h, _hi in mr_by_h.items():
                _m = _hi["model"]
                _fc = _hi.get("feat_cols", fcols_mr)
                _grp = apply_interaction_toggles(grp.copy(), keys_mr, ["G_VOL","G_TREND"], drop_raw=cfg["drop_raw_causal"])
                _X = _grp.reindex(columns=_fc, fill_value=0.0).fillna(0.0).astype(np.float32)
                _p = _m.predict(_X)
                mr_preds_list.append(_p)
                mr_h_preds[f"pred_mr_H{_h}"] = _p
                mr_h_preds[f"pred_H{_h}"] = _p # Also generic name for meta-compat
            p_mr = np.mean(mr_preds_list, axis=0) if mr_preds_list else np.zeros(len(grp))
        else:
            grp_mr = apply_interaction_toggles(grp.copy(), keys_mr, ["G_VOL","G_TREND"], drop_raw=cfg["drop_raw_causal"])
            X_mr_pred = grp_mr.reindex(columns=fcols_mr, fill_value=0.0).fillna(0.0).astype(np.float32)
            p_mr = model_mr.predict(X_mr_pred)
            mr_h_preds["pred_mr_H2"] = p_mr # Minimal fallback

        # Multi-horizon TF prediction
        if tf_by_h:
            tf_preds_list = []
            for _h, _hi in tf_by_h.items():
                _m = _hi["model"]
                _fc = _hi.get("feat_cols", fcols_tf)
                _grp = apply_interaction_toggles(grp.copy(), keys_tf, ["G_VOL","G_TREND"], drop_raw=cfg["drop_raw_causal"])
                _X = _grp.reindex(columns=_fc, fill_value=0.0).fillna(0.0).astype(np.float32)
                _p = _m.predict(_X)
                tf_preds_list.append(_p)
                tf_h_preds[f"pred_tf_H{_h}"] = _p
                if f"pred_H{_h}" not in mr_h_preds: # avoid collision
                     tf_h_preds[f"pred_H{_h}"] = _p
            p_tf = np.mean(tf_preds_list, axis=0) if tf_preds_list else np.zeros(len(grp))
        else:
            grp_tf = apply_interaction_toggles(grp.copy(), keys_tf, ["G_VOL","G_TREND"], drop_raw=cfg["drop_raw_causal"])
            X_tf_pred = grp_tf.reindex(columns=fcols_tf, fill_value=0.0).fillna(0.0).astype(np.float32)
            p_tf = model_tf.predict(X_tf_pred)
            tf_h_preds["pred_tf_H2"] = p_tf # Minimal fallback

        meta_mr = meta_models.get(f"{side_key}_mr") or meta_models.get(f"{side_key}_mr_clf")
        meta_tf = meta_models.get(f"{side_key}_tf") or meta_models.get(f"{side_key}_tf_clf")


        s_mr, fb_mr = _meta_predict_or_fallback(meta_mr, p_mr, grp, "mr", side_key, mr_h_preds, tf_h_preds, cfg=cfg)
        s_tf, fb_tf = _meta_predict_or_fallback(meta_tf, p_tf, grp, "tf", side_key, mr_h_preds, tf_h_preds, cfg=cfg)

        for i, idx in enumerate(grp.index):
            score_rows.append({
                "symbol": grp.loc[idx, "symbol"],
                "side_key": side_key,
                "score_mr": float(s_mr[i]),
                "score_tf": float(s_tf[i]),
                "trend_dir": int(grp.loc[idx, "trend_dir"]),
                "trap_quality": float(grp.loc[idx, "trap_quality"]),
                "predicted_vol_6h": float(grp.loc[idx, "predicted_vol_6h"]),
                "used_fallback_mr": bool(fb_mr[i]),
                "used_fallback_tf": bool(fb_tf[i]),
            })

    return pd.DataFrame(score_rows)

def generate_hourly_signals(ts_sig, feats, mkt_gates, model_bundle, risk_config, cfg, p_exh_cand, current_positions_syms, tradeable_candidates=None):
    if ts_sig not in mkt_gates.index:
        return []

    signal_params = (risk_config or {}).get("signal_params", {}) if isinstance(risk_config, dict) else {}
    thr_long = float(signal_params.get("thr_long", cfg.get("thr_long", 0.01)))
    # Unified score convention: higher score is always better (long and short).
    # For short buckets we keep side information separate from score orientation.
    thr_short = float(signal_params.get("thr_short", cfg.get("thr_short", -0.01)))
    k_long = int(signal_params.get("k_long", cfg.get("k_long", 10)))
    k_short = int(signal_params.get("k_short", cfg.get("k_short", 10)))

    size_min = float(signal_params.get("size_min", 0.03))
    size_max = float(signal_params.get("size_max", 0.15))
    size_k = float(signal_params.get("size_k", 2.0))
    size_x0 = float(signal_params.get("size_x0", 0.5))
    size_zcap = float(signal_params.get("size_zcap", 4.0))
    size_q50 = signal_params.get("size_q50")
    size_q90 = signal_params.get("size_q90")

    sc_df = _build_side_score_df(ts_sig, feats, mkt_gates, model_bundle, cfg, p_exh_cand, current_positions_syms, tradeable_candidates=tradeable_candidates)
    if sc_df.empty:
        return []

    if "used_fallback_mr" in sc_df.columns and "used_fallback_tf" in sc_df.columns:
        fb_any = (sc_df["used_fallback_mr"] | sc_df["used_fallback_tf"]).astype(float)
        emit_bucket_summary(
            tprint=tprint,
            run_id=str(ts_sig),
            bucket_id="META_ALL",
            kind="meta_fallback",
            stats={
                "pct_fallback_used": float(fb_any.mean()),
                "pct_meta_used": float(1.0 - fb_any.mean()),
                "n_rows": int(len(sc_df)),
            },
        )

    score_scale = signal_params.get("score_scale_params", {}) if isinstance(signal_params, dict) else {}

    final_orders = []
    for row in sc_df.to_dict('records'):
        sym = row["symbol"]
        side_key = row["side_key"]
        t_dir = int(row["trend_dir"])
        s_mr = float(row["score_mr"])
        s_tf = float(row["score_tf"])

        if score_scale:
            if side_key == "long":
                s_mr = _robust_norm(s_mr, score_scale.get("long_mr_center", 0.0), score_scale.get("long_mr_scale", 1.0))
                s_tf = _robust_norm(s_tf, score_scale.get("long_tf_center", 0.0), score_scale.get("long_tf_scale", 1.0))
            else:
                s_mr = _robust_norm(s_mr, score_scale.get("short_mr_center", 0.0), score_scale.get("short_mr_scale", 1.0))
                s_tf = _robust_norm(s_tf, score_scale.get("short_tf_center", 0.0), score_scale.get("short_tf_scale", 1.0))

        # Decoupled Logic (Report 2026-02-10)
        # Long candidates (from 'top'): check TF for momentum (Up Trend) or MR for recovery (Down Trend)
        # Short candidates (from 'bot'): check MR for reversal (Up Trend) or TF for momentum (Down Trend)
        potential_signal = None
        if side_key == "long":
            if t_dir > 0:
                mode = "best"
                thr = float(signal_params.get(f"thr_tf_{mode}", thr_long))
                if s_tf > thr:
                    potential_signal = {"symbol": sym, "side": "long", "score": s_tf, "dom": "tf", "mode": mode}
            else:
                mode = "worst"
                thr = float(signal_params.get(f"thr_mr_{mode}", thr_long))
                if s_mr > thr:
                    potential_signal = {"symbol": sym, "side": "long", "score": s_mr, "dom": "mr", "mode": mode}
        else: # side_key == "short"
            if t_dir > 0:
                mode = "best"
                thr = float(signal_params.get(f"thr_mr_{mode}", thr_short))
                if s_mr > thr:
                    potential_signal = {"symbol": sym, "side": "short", "score": s_mr, "dom": "mr", "mode": mode}
            else:
                mode = "worst"
                thr = float(signal_params.get(f"thr_tf_{mode}", thr_short))
                if s_tf > thr:
                    potential_signal = {"symbol": sym, "side": "short", "score": s_tf, "dom": "tf", "mode": mode}

        if potential_signal:
            final_orders.append(potential_signal)

    if not final_orders:
        return []

    longs = [o for o in final_orders if o["side"] == "long"]
    shorts = [o for o in final_orders if o["side"] == "short"]
    longs.sort(key=lambda x: x["score"], reverse=True)
    shorts.sort(key=lambda x: x["score"], reverse=True)
    final_orders = longs[:k_long] + shorts[:k_short]
    
    # --- Global Percentile Gating ---
    # Gating based on global distribution of scores (across all assets and time)
    # This addresses the user query: "if it's in the top x%, then act on it"
    q95 = signal_params.get("size_q95")
    q98 = signal_params.get("size_q98")
    for o in final_orders:
        o["global_rank"] = _estimate_global_percentile(o["score"], size_q50, size_q90, q95, q98)
    
    score_gate_q = float(signal_params.get("score_gate_q", 0.0))
    if score_gate_q > 0:
        filtered = [o for o in final_orders if o["global_rank"] >= score_gate_q]
        if len(filtered) < len(final_orders):
            # tprint(f"  Global gate (>{score_gate_q:.1%}) filtered {len(final_orders) - len(filtered)} signals")
            final_orders = filtered

    # Symbol mutual exclusivity: if same symbol has both long and short signals,
    # keep only the stronger one (higher |score|) to prevent ping-pong losses
    sym_best = {}
    for o in final_orders:
        s = o["symbol"]
        if s not in sym_best or abs(o["score"]) > abs(sym_best[s]["score"]):
            sym_best[s] = o
    final_orders = list(sym_best.values())

    if size_q50 is None or size_q90 is None:
        arr = np.array([abs(o["score"]) for o in final_orders], dtype=np.float64)
        size_q50 = float(np.quantile(arr, 0.5)) if arr.size else 0.0
        size_q90 = float(np.quantile(arr, 0.9)) if arr.size else 1.0

    orders_out = []
    gross_cap = float(cfg.get("wallet_gross_cap", 1.0))
    sizing_mode = signal_params.get("sizing_mode", cfg.get("sizing_mode", "rank"))
    raw_w = []
    # Mode "rank" now uses GLOBAL rank (percentile) rather than hourly local rank
    # mapping is: global_rank (interpolated) -> sigmoid weight
    
    for i, o in enumerate(final_orders):
        if sizing_mode == "score":
            z = abs(float(o["score"]))
            z_tilde = np.clip((z - size_q50) / (size_q90 - size_q50 + 1e-12), 0.0, size_zcap)
            fz = 1.0 / (1.0 + np.exp(-size_k * (z_tilde - size_x0)))
            w_alloc = size_min + (size_max - size_min) * fz
        elif sizing_mode == "rank":
            # Global Rank-based Sigmoid: rank within long-term distribution
            # Immune to local conviction paradox and magnitude drift
            ranks_i = o.get("global_rank", 0.5)
            fz = 1.0 / (1.0 + np.exp(-size_k * (ranks_i - size_x0)))
            w_alloc = size_min + (size_max - size_min) * fz
        else:
            w_alloc = (size_min + size_max) / 2.0
        raw_w.append(w_alloc)

    total_w = float(np.sum(raw_w))
    scale = min(1.0, gross_cap / max(total_w, 1e-12))

    for ord, w_alloc in zip(final_orders, raw_w):
        ord["weight"] = float(w_alloc * scale)
        mode = ord.get("mode") or _bucket_mode_from_side_dom(ord.get("side"), ord.get("dom"))
        r_keys = [
            f"risk_{ord['dom']}_{mode}",
            f"risk_{ord['side']}_{ord['dom']}",
        ]
        if risk_config and "granular_risk" in risk_config:
            for r_key in r_keys:
                g_risk = risk_config["granular_risk"].get(r_key)
                if g_risk:
                    ord["risk_params"] = g_risk
                    break
        orders_out.append(ord)

    return orders_out
