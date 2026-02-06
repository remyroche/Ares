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

def simulate_trade_hourly(o_s, h_s, l_s, c_s, feats_s, ts_entry, entry_px, side, cfg, max_hold_hours):
    tprint(f"Entering function: simulate_trade_hourly in engine.py")
    if np.isnan(entry_px) or entry_px <= 0:
        return 0.0, ts_entry, "no_entry"

    ts_sig = ts_entry - pd.Timedelta(hours=1)
    if ts_sig not in feats_s.index:
        atr = 0.02
        # No atr base/std available if using just scalar lookups
        # Fallback to fixed
        use_dynamic_barrier = False
    else:
        atr = float(feats_s.loc[ts_sig])
        use_dynamic_barrier = True

    # Check for optimized TP/SL mults
    tp_mult = cfg.get("tp_mult")
    sl_mult = cfg.get("sl_mult")

    if tp_mult is not None and sl_mult is not None:
        # TRIPLE BARRIER LOGIC
        # Compute dynamic barrier level if possible
        if use_dynamic_barrier:
            # We need atr_pct history to compute Z.
            # feats_s is just a Series (usually 'atr_pct' column).
            # But we need history for rolling metrics.
            # In simulate_trade_hourly, `feats_s` is passed as a Series: `atr_s[sym]`.
            # This contains the full history of ATR for the symbol.
            # So we can compute context.

            # Context window
            window_base = 24 * 30
            # Slicing history ending at ts_sig
            # This might be slow if done every time.
            # But optimize_risk_params uses fast vectorized approach.
            # simulate_trade_hourly is used in backtest loop (slow anyway).

            # slice context
            # We need Z and Base.
            # If feats_s is indeed the full series, we can look back.
            end_loc = feats_s.index.get_loc(ts_sig)
            start_loc = max(0, end_loc - window_base * 2)

            # Check sufficiency
            if end_loc - start_loc < window_base:
                # Not enough history
                barrier_pct = atr
            else:
                # Compute rolling stats on the fly? Or assume pre-computed?
                # Pre-computing in backtest loop would be better.
                # But here we are inside the function.
                # Let's do a quick calculation on the window.
                # slice
                subset = feats_s.iloc[start_loc : end_loc+1]

                # We need base and std at the END.
                # base = median of last window_base
                # std = std of last window_base
                if len(subset) >= window_base:
                    win = subset.iloc[-window_base:]
                    base = win.median()
                    std = win.std()
                    z = (atr - base) / (std + 1e-12)

                    # scaled_atr_pct
                    barrier_pct = scaled_atr_pct(
                        atr, z, base, z_max=3.0, lo=0.03, hi=0.06
                    )
                else:
                    barrier_pct = atr
        else:
            barrier_pct = atr # Fallback

        # Apply multipliers
        tp_dist = tp_mult * barrier_pct * entry_px
        sl_dist = sl_mult * barrier_pct * entry_px

        # Simulate Fixed Exit
        # We can reuse TrailingStop with activation=TP and trail=0 (tight stop)
        # Or simple loop

        end_ts = ts_entry + pd.Timedelta(hours=max_hold_hours)
        path = o_s.loc[ts_entry:end_ts].index
        if len(path) == 0:
            return 0.0, ts_entry, "no_path"

        tp_price = entry_px + tp_dist if side == "long" else entry_px - tp_dist
        sl_price = entry_px - sl_dist if side == "long" else entry_px + sl_dist

        for ts in path:
            hh = h_s.loc[ts]; ll = l_s.loc[ts]; cc = c_s.loc[ts]
            if np.isnan(hh): continue

            if side == "long":
                # Check SL first? Or TP?
                # Optimistic: TP first.
                if hh >= tp_price:
                    return (tp_price / entry_px) - 1.0, ts, "take_profit"
                if ll <= sl_price:
                    return (sl_price / entry_px) - 1.0, ts, "stop_loss"
            else:
                if ll <= tp_price:
                    return (entry_px / tp_price) - 1.0, ts, "take_profit"
                if hh >= sl_price:
                    return (entry_px / sl_price) - 1.0, ts, "stop_loss"

        # Time exit
        last_ts = path[-1]
        last_close = c_s.loc[last_ts]
        if side == "long":
            return (last_close / entry_px) - 1.0, last_ts, "time_exit"
        else:
            return (entry_px / last_close) - 1.0, last_ts, "time_exit"

    else:
        # TRAILING STOP LOGIC (Legacy)
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

def generate_hourly_signals(ts_sig, feats, mkt_gates, model_bundle, risk_config, cfg, p_exh_cand, current_positions_syms):
    tprint(f"Entering function: generate_hourly_signals in engine.py")
    if ts_sig not in mkt_gates.index:
        return []

    # 1. Live Candidate Selection
    candidates = set()
    lookback_offsets = [0, 4, 8, 12, 16]

    for offset in lookback_offsets:
        t_check = ts_sig - pd.Timedelta(hours=offset)
        if t_check in feats["ret24h"].index:
            top, bot = select_trade_candidates_hourly(
                feats, t_check, list(feats["ret24h"].columns),
                cfg["trade_extreme_pct"], cfg["trade_extreme_min"], cfg["trade_extreme_max"],
                cfg["trade_deviation_metric"]
            )
            candidates.update(top)
            candidates.update(bot)

    candidates = list(candidates)
    candidates = [s for s in candidates if s not in current_positions_syms]

    if not candidates:
        return []

    mrk = mkt_gates.loc[ts_sig]
    ts_lag = ts_sig - pd.Timedelta(hours=1)

    alpha_models = model_bundle["alpha_models"]
    meta_models = model_bundle["meta_models"]
    spike_model = model_bundle.get("spike_model")

    # Group candidates into "Best" (Trend > 0) and "Worst" (Trend <= 0)
    # Using 'trend_pct' for determination
    trend_vals = feats["trend_pct"].loc[ts_sig, candidates]
    best_performers = trend_vals[trend_vals > 0].index.tolist()
    worst_performers = trend_vals[trend_vals <= 0].index.tolist()

    rows = []

    # We iterate candidates and attach models for BOTH Long and Short possibilities
    # But based on Best/Worst classification, the strategy mapping is fixed:
    # Best: Long=TF, Short=MR
    # Worst: Long=MR, Short=TF

    def _prepare_rec(sym, kind):
        p_lag = 0.5
        if ts_lag in p_exh_cand.index and sym in p_exh_cand.columns:
            p_lag = float(p_exh_cand.loc[ts_lag, sym])

        rec = {
            "symbol": sym,
            "kind": kind, # 'best' or 'worst'
            "mkt_ret24h": float(mrk["mkt_ret24h"]),
            "mkt_ret6h": float(mrk["mkt_ret6h"]),
            "mkt_trend": float(mrk["mkt_trend"]),
            "mkt_rv": float(mrk["mkt_rv"]),
            "G_VOL": int(mrk["G_VOL"]),
            "G_TREND": int(mrk["G_TREND"]),
            "p_exh_lag1": p_lag
        }

        # We need all features potentially used by any of the 4 models
        # But we can optimize later. For now, fetch all known keys.
        # This includes alpha features, meta features, spike features.

        # Get required feature keys from config
        all_keys = set()
        for k_list in ["mr_feature_keys", "tf_feature_keys", "spike_feature_keys", "meta_feature_keys", "causal_cols"]:
            if k_list in cfg:
                all_keys.update(cfg[k_list])

        for k in all_keys:
            if k in feats and sym in feats[k].columns:
                rec[k] = float(feats[k].loc[ts_sig, sym])

        return rec

    for sym in best_performers:
        rows.append(_prepare_rec(sym, "best"))
    for sym in worst_performers:
        rows.append(_prepare_rec(sym, "worst"))

    df_all = pd.DataFrame(rows)
    orders_candidates = []

    if not df_all.empty:
        # Spike Inference
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

            if probs is not None:
                for i in range(probs.shape[1]):
                    df_all[f"spike_prob_{i}"] = probs[:, i]
            else:
                 for i in range(4): df_all[f"spike_prob_{i}"] = 0.0
        else:
             for i in range(4): df_all[f"spike_prob_{i}"] = 0.0

        # Process by Group (Best/Worst) to batch predict
        for kind, grp in df_all.groupby("kind"):
            # Setup Models based on Kind
            if kind == "best":
                # Long: TF, Short: MR
                m_long = alpha_models.get("long", {}).get("tf", {})
                m_short = alpha_models.get("short", {}).get("mr", {})

                meta_long = meta_models.get("long_tf")
                meta_short = meta_models.get("short_mr")

                feat_keys_long = cfg.get("tf_feature_keys", cfg["causal_cols"])
                feat_keys_short = cfg.get("mr_feature_keys", cfg["causal_cols"])

                dom_long = "tf"
                dom_short = "mr"

            else: # worst
                # Long: MR, Short: TF
                m_long = alpha_models.get("long", {}).get("mr", {})
                m_short = alpha_models.get("short", {}).get("tf", {})

                meta_long = meta_models.get("long_mr")
                meta_short = meta_models.get("short_tf")

                feat_keys_long = cfg.get("mr_feature_keys", cfg["causal_cols"])
                feat_keys_short = cfg.get("tf_feature_keys", cfg["causal_cols"])

                dom_long = "mr"
                dom_short = "tf"

            if not m_long or not m_short:
                continue

            # --- Predict Long ---
            # 1. Alpha
            grp_long = apply_interaction_toggles(grp.copy(), feat_keys_long, ["G_VOL","G_TREND"], drop_raw=cfg["drop_raw_causal"])

            cols_l = m_long["feat_cols"]
            X_long = pd.DataFrame(0.0, index=grp.index, columns=cols_l)
            avail_l = [c for c in cols_l if c in grp_long.columns]
            X_long[avail_l] = grp_long[avail_l].fillna(0.0)
            p_long = m_long["model"].predict(X_long)

            # 2. Meta Long
            if meta_long:
                numeric_cols = grp.select_dtypes(include=[np.number]).columns.tolist()
                grp_numeric = grp[numeric_cols].copy()
                grp_toggled = apply_interaction_toggles(grp_numeric, feat_keys_long, ["G_VOL", "G_TREND"], drop_raw=False)
                X_meta_l = meta_long.prepare_meta_features(p_long, grp_toggled, pred_col_name="pred_logit")
                # Ensure columns
                if meta_long.selected_features:
                    for c in meta_long.selected_features:
                         if c not in X_meta_l.columns: X_meta_l[c] = 0.0
                s_long = meta_long.predict(X_meta_l)
            else:
                s_long = (p_long - 0.5) * 0.1

            # --- Predict Short ---
            # 1. Alpha
            grp_short = apply_interaction_toggles(grp.copy(), feat_keys_short, ["G_VOL","G_TREND"], drop_raw=cfg["drop_raw_causal"])

            cols_s = m_short["feat_cols"]
            X_short = pd.DataFrame(0.0, index=grp.index, columns=cols_s)
            avail_s = [c for c in cols_s if c in grp_short.columns]
            X_short[avail_s] = grp_short[avail_s].fillna(0.0)
            p_short = m_short["model"].predict(X_short)

            # 2. Meta Short
            if meta_short:
                numeric_cols = grp.select_dtypes(include=[np.number]).columns.tolist()
                grp_numeric = grp[numeric_cols].copy()
                grp_toggled = apply_interaction_toggles(grp_numeric, feat_keys_short, ["G_VOL", "G_TREND"], drop_raw=False)
                X_meta_s = meta_short.prepare_meta_features(p_short, grp_toggled, pred_col_name="pred_logit")
                # Ensure columns
                if meta_short.selected_features:
                    for c in meta_short.selected_features:
                         if c not in X_meta_s.columns: X_meta_s[c] = 0.0
                s_short = meta_short.predict(X_meta_s)
            else:
                s_short = (p_short - 0.5) * 0.1

            # --- Net Score ---
            net_score = s_long - s_short

            for i, idx in enumerate(grp.index):
                sym = grp.loc[idx, "symbol"]
                ns = float(net_score[i])

                # Decision Logic
                # If ns > 0 => Long (using dom_long strategy)
                # If ns < 0 => Short (using dom_short strategy)

                # Thresholds
                thr_long = cfg.get("thr_long", 0.0)
                thr_short = cfg.get("thr_short", 0.0)

                if ns > thr_long:
                    orders_candidates.append({
                        "symbol": sym, "side": "long", "score": ns, "dom": dom_long
                    })
                elif ns < -thr_short:
                    orders_candidates.append({
                        "symbol": sym, "side": "short", "score": abs(ns), "dom": dom_short
                    })

    # Sort by absolute score (conviction)
    orders_candidates.sort(key=lambda x: x["score"], reverse=True)

    # Apply Limits (Max Longs, Max Shorts)
    longs = [o for o in orders_candidates if o["side"] == "long"][:cfg["k_long"]]
    shorts = [o for o in orders_candidates if o["side"] == "short"][:cfg["k_short"]]

    final_orders = longs + shorts

    # Allocation (Absolute Scaling based on Score)
    orders_out = []
    gross_cap = float(cfg["wallet_gross_cap"])
    pos_scale = cfg.get("pos_size_scale", 1.0)

    for ord in final_orders:
        # User request: "optimise the rate at which position sizing... increases with Net Score absolute value"
        # Formula: size_pct = clip(scale * |score|, 3%, 15%)
        # If score < thr, we don't trade. If score > thr, size logic applies.

        raw_size = pos_scale * abs(ord["score"])
        size_pct = np.clip(raw_size, 0.03, 0.15)

        ord["weight"] = gross_cap * size_pct

        # Inject Risk Params
        r_key = f"risk_{ord['side']}_{ord['dom']}"
        if risk_config and "granular_risk" in risk_config:
            g_risk = risk_config["granular_risk"].get(r_key)
            if g_risk:
                ord["risk_params"] = g_risk

        orders_out.append(ord)

    return orders_out
