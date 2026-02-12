import numpy as np
import pandas as pd

from extreme_price_movements.utils import tprint

from extreme_price_movements.risk import TrailingStop
from extreme_price_movements.training import (
    compute_p_exhaustion_at_t,
    select_best_horizon,
    apply_interaction_toggles,
    scaled_atr_pct
)
from extreme_price_movements.candidates import select_trade_candidates_hourly, entry_price_next_hour_open

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


def simulate_trade_hourly(o_s, h_s, l_s, c_s, feats_s, ts_entry, entry_px, side, cfg, max_hold_hours, exchange=None, symbol=None):
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
    
    if len(path) == 0:
        return 0.0, ts_entry, "no_path", _empty_extras

    # MAE/MFE tracking
    mae_px = 0.0   # max adverse excursion (worst unrealised loss, as positive distance)
    mfe_px = 0.0   # max favorable excursion (best unrealised profit, as positive distance)
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
                }
                return ret, ts, "giveback_exit", extras

        # --- Early invalidation: kill trades showing adverse drift without MFE ---
        if exit_stage == 0 and bar_count >= kill_min_bars:
            mae_frac = mae_px / entry_px
            mfe_frac = mfe_px / entry_px
            kill_score = mae_frac + kill_a * bar_count - kill_b * mfe_frac - kill_c
            if kill_score > 0:
                ret = (cc / entry_px) - 1.0 if side == "long" else (entry_px / cc) - 1.0
                extras = {
                    "mae_pct": mae_frac,
                    "mfe_pct": mfe_frac,
                    "bars_to_mfe": bars_to_mfe,
                    "sl_pct": sl_dist / entry_px,
                    "tp_pct": activation_dist / entry_px,
                    "exit_stage": exit_stage,
                }
                return ret, ts, "early_invalidation", extras

        # Check stop-loss / trailing-stop hit
        if side == "long":
            hit_sl = ll <= sl_price
        else:
            hit_sl = hh >= sl_price

        if hit_sl:
            ret = (sl_price / entry_px) - 1.0 if side == "long" else (entry_px / sl_price) - 1.0
            if exit_stage >= 1:
                reason = "trailing_stop"
            else:
                reason = "stop_loss"
            extras = {
                "mae_pct": mae_px / entry_px,
                "mfe_pct": mfe_px / entry_px,
                "bars_to_mfe": bars_to_mfe,
                "sl_pct": sl_dist / entry_px,
                "tp_pct": activation_dist / entry_px,
                "exit_stage": exit_stage,
            }
            return ret, ts, reason, extras

    # Time exit
    last_ts = path[-1]
    last_close = c_data.loc[last_ts]
    extras = {
        "mae_pct": mae_px / entry_px,
        "mfe_pct": mfe_px / entry_px,
        "bars_to_mfe": bars_to_mfe,
        "sl_pct": sl_dist / entry_px,
        "tp_pct": activation_dist / entry_px,
        "exit_stage": exit_stage,
    }
    if side == "long":
        return (last_close / entry_px) - 1.0, last_ts, "time_exit", extras
    else:
        return (entry_px / last_close) - 1.0, last_ts, "time_exit", extras



def _robust_norm(val, center, scale, eps=1e-12):
    return (float(val) - float(center)) / (float(scale) + eps)


def _bucket_mode_from_side_dom(side, dom):
    if dom == "tf":
        return "best" if side == "long" else "worst"
    return "worst" if side == "long" else "best"


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
                all_keys = all_fcols_mr | all_fcols_tf | set(cfg.get("spike_feature_keys", [])) | set(cfg.get("meta_feature_keys", []))
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

        # Multi-horizon MR prediction (average across all horizons)
        if mr_by_h:
            mr_preds = []
            for _h, _hi in mr_by_h.items():
                _m = _hi["model"]
                _fc = _hi.get("feat_cols", fcols_mr)
                _grp = apply_interaction_toggles(grp.copy(), keys_mr, ["G_VOL","G_TREND"], drop_raw=cfg["drop_raw_causal"])
                _X = _grp.reindex(columns=_fc, fill_value=0.0).fillna(0.0).astype(np.float32)
                mr_preds.append(_m.predict(_X))
            p_mr = np.mean(mr_preds, axis=0)
        else:
            grp_mr = apply_interaction_toggles(grp.copy(), keys_mr, ["G_VOL","G_TREND"], drop_raw=cfg["drop_raw_causal"])
            X_mr_pred = grp_mr.reindex(columns=fcols_mr, fill_value=0.0).fillna(0.0).astype(np.float32)
            p_mr = model_mr.predict(X_mr_pred)

        # Multi-horizon TF prediction (average across all horizons)
        if tf_by_h:
            tf_preds = []
            for _h, _hi in tf_by_h.items():
                _m = _hi["model"]
                _fc = _hi.get("feat_cols", fcols_tf)
                _grp = apply_interaction_toggles(grp.copy(), keys_tf, ["G_VOL","G_TREND"], drop_raw=cfg["drop_raw_causal"])
                _X = _grp.reindex(columns=_fc, fill_value=0.0).fillna(0.0).astype(np.float32)
                tf_preds.append(_m.predict(_X))
            p_tf = np.mean(tf_preds, axis=0)
        else:
            grp_tf = apply_interaction_toggles(grp.copy(), keys_tf, ["G_VOL","G_TREND"], drop_raw=cfg["drop_raw_causal"])
            X_tf_pred = grp_tf.reindex(columns=fcols_tf, fill_value=0.0).fillna(0.0).astype(np.float32)
            p_tf = model_tf.predict(X_tf_pred)

        meta_mr = meta_models.get(f"{side_key}_mr")
        meta_tf = meta_models.get(f"{side_key}_tf")

        def _meta_predict_or_fallback(meta_model, p_alpha, grp_df, label):
            """Predict with meta model; fall back to raw alpha if meta output is degenerate."""
            if meta_model is None:
                return (p_alpha - 0.5) * 0.1
            num = grp_df.select_dtypes(include=[np.number]).copy()
            X_meta = meta_model.prepare_meta_features(p_alpha, num, pred_col_name="pred_logit")
            # Add feature interactions (must match training)
            if "pred_logit" in X_meta.columns:
                pl = X_meta["pred_logit"].values
                for ifeat in ["vol_z", "mkt_rv_ratio", "ambig", "exh_qual", "trend_pct"]:
                    if ifeat in X_meta.columns:
                        X_meta[f"pred_x_{ifeat}"] = pl * X_meta[ifeat].values
                # Regime bucket interactions (must match training)
                for rcol in ["G_VOL", "G_TREND"]:
                    if rcol in grp_df.columns:
                        rv = grp_df[rcol].values
                        for bkt in [0, 1, 2]:
                            X_meta[f"pred_x_{rcol}_{bkt}"] = pl * (rv == bkt).astype(float)
            if meta_model.selected_features:
                available = set(X_meta.columns)
                selected = set(meta_model.selected_features)
                present = selected & available
                missing = selected - available
                coverage = len(present) / max(len(selected), 1)
                if coverage < 0.8:
                    tprint(f"  Meta {side_key}_{label}: DISABLED — feature coverage {coverage:.0%} "
                           f"({len(missing)} missing of {len(selected)})")
                    return (p_alpha - 0.5) * 0.1
                if missing:
                    tprint(f"  Meta {side_key}_{label}: {len(missing)} features missing "
                           f"(coverage {coverage:.0%}), filling with 0")
                X_meta = X_meta.reindex(columns=meta_model.selected_features, fill_value=0.0)
            s = meta_model.predict(X_meta)
            # Variance gate: only check on large batches (>=10 symbols).
            # On small batches, predictions can legitimately be similar.
            # The _center_scale degenerate guard in pipeline_steps.py
            # already protects against systematic degeneracy.
            if len(s) >= 10 and np.std(s) < 1e-6:
                tprint(f"  Meta {side_key}_{label}: DISABLED — prediction std={np.std(s):.2e} (degenerate, n={len(s)})")
                return (p_alpha - 0.5) * 0.1
            return s

        s_mr = _meta_predict_or_fallback(meta_mr, p_mr, grp, "mr")
        s_tf = _meta_predict_or_fallback(meta_tf, p_tf, grp, "tf")

        for i, idx in enumerate(grp.index):
            score_rows.append({
                "symbol": grp.loc[idx, "symbol"],
                "side_key": side_key,
                "score_mr": float(s_mr[i]),
                "score_tf": float(s_tf[i]),
                "trend_dir": int(grp.loc[idx, "trend_dir"]),
                "trap_quality": float(grp.loc[idx, "trap_quality"]),
                "predicted_vol_6h": float(grp.loc[idx, "predicted_vol_6h"])
            })

    return pd.DataFrame(score_rows)

def generate_hourly_signals(ts_sig, feats, mkt_gates, model_bundle, risk_config, cfg, p_exh_cand, current_positions_syms, tradeable_candidates=None):
    if ts_sig not in mkt_gates.index:
        return []

    signal_params = (risk_config or {}).get("signal_params", {}) if isinstance(risk_config, dict) else {}
    thr_long = float(signal_params.get("thr_long", cfg.get("thr_long", 0.01)))
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

    long_df = sc_df[sc_df["side_key"] == "long"].set_index("symbol")
    short_df = sc_df[sc_df["side_key"] == "short"].set_index("symbol")
    syms = sorted(set(long_df.index).intersection(set(short_df.index)))
    if not syms:
        return []

    score_scale = signal_params.get("score_scale_params", {}) if isinstance(signal_params, dict) else {}

    final_orders = []
    abs_scores = []
    for sym in syms:
        l_mr = float(long_df.loc[sym, "score_mr"])
        s_mr = float(short_df.loc[sym, "score_mr"])
        l_tf = float(long_df.loc[sym, "score_tf"])
        s_tf = float(short_df.loc[sym, "score_tf"])

        # Determine trend direction from the dataframe
        # (It should be same for long_df and short_df for same symbol)
        t_dir = int(long_df.loc[sym, "trend_dir"])

        if score_scale:
            l_mr = _robust_norm(l_mr, score_scale.get("long_mr_center", 0.0), score_scale.get("long_mr_scale", 1.0))
            s_mr = _robust_norm(s_mr, score_scale.get("short_mr_center", 0.0), score_scale.get("short_mr_scale", 1.0))
            l_tf = _robust_norm(l_tf, score_scale.get("long_tf_center", 0.0), score_scale.get("long_tf_scale", 1.0))
            s_tf = _robust_norm(s_tf, score_scale.get("short_tf_center", 0.0), score_scale.get("short_tf_scale", 1.0))

        # Decoupled Logic (Report 2026-02-10)
        # Treat each model output as an independent signal.
        # Best (Up Trend, t_dir > 0): Long_TF (buy) OR Short_MR (sell)
        # Worst (Down Trend, t_dir < 0): Long_MR (buy) OR Short_TF (sell)

        potential_signals = []

        if t_dir > 0:
            # Best Performer Pipeline
            # Check Long_TF
            mode_l = "best" # long_tf is 'best' bucket
            thr_l = float(signal_params.get(f"thr_tf_{mode_l}", thr_long))
            if l_tf > thr_l:
                potential_signals.append({
                    "symbol": sym, "side": "long", "score": l_tf, "dom": "tf", "mode": mode_l
                })

            # Check Short_MR
            mode_s = "best" # short_mr is 'best' bucket (reversal of best)
            thr_s = float(signal_params.get(f"thr_mr_{mode_s}", thr_short))
            # Short thresholds are typically negative (e.g. -0.01).
            # But here meta models predict rank percentile centered?
            # If meta predicts > 0, it means "good trade".
            # Wait, `short_mr` meta predicts rank relative to Short PnL.
            # So a high POSITIVE score means "Good Short".
            # Thresholds in config might be positive for meta scores if they represent quality.
            # However, `thr_short` defaults to -0.01.
            # If the system expects signed scores (long > 0, short < 0), we need to negate s_mr.
            # Previous logic: `net_score = l_tf - s_mr`.
            # If s_mr was large (0.8), net_score becomes negative (-0.8), triggering short.
            # So s_mr is a POSITIVE quality score for a SHORT trade.
            # To create a negative signal score: signal = -s_mr.

            # If we use `thr_short` (e.g. -0.01), we check if -s_mr < thr_short => s_mr > -thr_short.
            # Example: thr_short = -0.01. We need -s_mr < -0.01 => s_mr > 0.01.

            neg_s_mr = -s_mr
            if neg_s_mr < thr_s:
                 potential_signals.append({
                    "symbol": sym, "side": "short", "score": neg_s_mr, "dom": "mr", "mode": mode_s
                })

        else:
            # Worst Performer Pipeline
            # Check Long_MR
            mode_l = "worst" # long_mr is 'worst' bucket
            thr_l = float(signal_params.get(f"thr_mr_{mode_l}", thr_long))
            if l_mr > thr_l:
                potential_signals.append({
                    "symbol": sym, "side": "long", "score": l_mr, "dom": "mr", "mode": mode_l
                })

            # Check Short_TF
            mode_s = "worst" # short_tf is 'worst' bucket
            thr_s = float(signal_params.get(f"thr_tf_{mode_s}", thr_short))

            neg_s_tf = -s_tf
            if neg_s_tf < thr_s:
                 potential_signals.append({
                    "symbol": sym, "side": "short", "score": neg_s_tf, "dom": "tf", "mode": mode_s
                })

        # Conflict Resolution (Mutual Exclusivity)
        # If we have both Long and Short for the same symbol (rare but possible with decoupled logic),
        # we treat it as High Uncertainty and take NO position.
        # User Instruction: "if there is a conflict, don't open any position"
        if len(potential_signals) > 1:
            # Conflict detected -> Ignore all signals for this symbol
            pass
        elif len(potential_signals) == 1:
            final_orders.append(potential_signals[0])
            abs_scores.append(abs(potential_signals[0]["score"]))

    if not final_orders:
        return []

    longs = [o for o in final_orders if o["side"] == "long"]
    shorts = [o for o in final_orders if o["side"] == "short"]
    longs.sort(key=lambda x: x["score"], reverse=True)
    shorts.sort(key=lambda x: x["score"])  # more negative first
    final_orders = longs[:k_long] + shorts[:k_short]

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

    # Pre-compute percentile ranks for rank-based sizing
    if sizing_mode == "rank" and len(final_orders) >= 2:
        abs_scores = np.array([abs(o["score"]) for o in final_orders], dtype=np.float64)
        order = np.argsort(abs_scores)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.linspace(0.0, 1.0, len(abs_scores))
    else:
        ranks = None

    for i, o in enumerate(final_orders):
        if sizing_mode == "score":
            # Legacy: raw |score| sigmoid (may have broken monotonicity)
            z = abs(float(o["score"]))
            z_tilde = np.clip((z - size_q50) / (size_q90 - size_q50 + 1e-12), 0.0, size_zcap)
            fz = 1.0 / (1.0 + np.exp(-size_k * (z_tilde - size_x0)))
            w_alloc = size_min + (size_max - size_min) * fz
        elif sizing_mode == "rank" and ranks is not None:
            # Rank-based: percentile rank within batch → sigmoid sizing
            # Immune to conviction paradox (uses ordinal rank, not magnitude)
            fz = 1.0 / (1.0 + np.exp(-size_k * (ranks[i] - size_x0)))
            w_alloc = size_min + (size_max - size_min) * fz
        else:
            # Equal: flat sizing
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