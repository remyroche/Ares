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

            hit_tp = False
            hit_sl = False

            if side == "long":
                if hh >= tp_price: hit_tp = True
                if ll <= sl_price: hit_sl = True
            else:
                if ll <= tp_price: hit_tp = True
                if hh >= sl_price: hit_sl = True

            if hit_tp and hit_sl:
                # Ambiguous: both hit in same bar
                # Use deterministic tie-break based on timestamp parity to avoid bias
                # (Random would be better but deterministic is preferred for reproduction)
                if int(ts.value) % 2 == 0:
                    return (tp_price / entry_px) - 1.0 if side == "long" else (entry_px / tp_price) - 1.0, ts, "take_profit"
                else:
                    return (sl_price / entry_px) - 1.0 if side == "long" else (entry_px / sl_price) - 1.0, ts, "stop_loss"
            elif hit_tp:
                return (tp_price / entry_px) - 1.0 if side == "long" else (entry_px / tp_price) - 1.0, ts, "take_profit"
            elif hit_sl:
                return (sl_price / entry_px) - 1.0 if side == "long" else (entry_px / sl_price) - 1.0, ts, "stop_loss"

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

    # 1. Live Candidate Selection (t-12 to t+16 logic adapted for live)
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
    trend_df = feats.get("trend_pct")

    alpha_models = model_bundle["alpha_models"]
    meta_models = model_bundle["meta_models"]
    spike_model = model_bundle.get("spike_model")

    rows = []
    for sym in candidates:
        try:
            p_lag = 0.5
            if ts_lag in p_exh_cand.index and sym in p_exh_cand.columns: p_lag = float(p_exh_cand.loc[ts_lag, sym])

            # Evaluate BOTH long and short models for every candidate
            for side_key in ["long", "short"]:
                m_bundle = alpha_models.get(side_key)
                if not m_bundle or not m_bundle.get("mr") or not m_bundle.get("tf"): continue

                model_mr = m_bundle["mr"]["model"]
                model_tf = m_bundle["tf"]["model"]
                fcols_mr = m_bundle["mr"]["feat_cols"]
                fcols_tf = m_bundle["tf"]["feat_cols"]

                # meta_model retrieval deferred to aggregation step
                rec = {
                    "symbol": sym, "side_key": side_key, "model_mr": model_mr, "model_tf": model_tf,
                    "feat_cols_mr": fcols_mr, "feat_cols_tf": fcols_tf,
                    "mkt_ret24h": float(mrk["mkt_ret24h"]), "mkt_ret6h": float(mrk["mkt_ret6h"]), "mkt_trend": float(mrk["mkt_trend"]), "mkt_rv": float(mrk["mkt_rv"]),
                    "G_VOL": int(mrk["G_VOL"]), "G_TREND": int(mrk["G_TREND"]), "p_exh_lag1": p_lag
                }

                all_keys = set(fcols_mr) | set(fcols_tf) | set(cfg.get("spike_feature_keys", [])) | set(cfg.get("meta_feature_keys", []))

                for k in all_keys:
                    if k in feats and sym in feats[k].columns: rec[k] = float(feats[k].loc[ts_sig, sym])

                rows.append(rec)
        except Exception: continue

    df_all = pd.DataFrame(rows)
    score_raw_list = []
    if not df_all.empty:
        # Spike Inference
        spike_keys = cfg.get("spike_feature_keys", [])
        if spike_model:
            # spike_model can be a dict {"gmm", "scaler", "columns"} or a raw GMM
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

        for side_key, grp in df_all.groupby("side_key"):
            first = grp.iloc[0]
            model_mr = first["model_mr"]; model_tf = first["model_tf"]
            fcols_mr = first["feat_cols_mr"]
            fcols_tf = first["feat_cols_tf"]

            # Apply interaction toggles (same as training), then select trained feature columns
            keys_mr = cfg.get("mr_feature_keys", cfg["causal_cols"])
            grp_mr = apply_interaction_toggles(grp.copy(), keys_mr, ["G_VOL","G_TREND"], drop_raw=cfg["drop_raw_causal"])
            mr_avail = [c for c in fcols_mr if c in grp_mr.columns]
            mr_missing = [c for c in fcols_mr if c not in grp_mr.columns]
            X_mr_pred = grp_mr[mr_avail].fillna(0.0).astype(np.float32)
            for c in mr_missing:
                X_mr_pred[c] = 0.0
            X_mr_pred = X_mr_pred[fcols_mr]
            p_mr = model_mr.predict(X_mr_pred)

            keys_tf = cfg.get("tf_feature_keys", cfg["causal_cols"])
            grp_tf = apply_interaction_toggles(grp.copy(), keys_tf, ["G_VOL","G_TREND"], drop_raw=cfg["drop_raw_causal"])
            tf_avail = [c for c in fcols_tf if c in grp_tf.columns]
            tf_missing = [c for c in fcols_tf if c not in grp_tf.columns]
            X_tf_pred = grp_tf[tf_avail].fillna(0.0).astype(np.float32)
            for c in tf_missing:
                X_tf_pred[c] = 0.0
            X_tf_pred = X_tf_pred[fcols_tf]
            p_tf = model_tf.predict(X_tf_pred)

            # Resolve Meta Models for this side (one for MR, one for TF)
            meta_mr = meta_models.get(f"{side_key}_mr")
            meta_tf = meta_models.get(f"{side_key}_tf")

            score_mr = None
            score_tf = None

            # 1. Score MR
            if meta_mr:
                numeric_cols = grp.select_dtypes(include=[np.number]).columns.tolist()
                grp_numeric = grp[numeric_cols].copy()
                grp_toggled = apply_interaction_toggles(grp_numeric, keys_mr, ["G_VOL", "G_TREND"], drop_raw=False)

                X_meta_mr = meta_mr.prepare_meta_features(p_mr, grp_toggled, pred_col_name="pred_logit")

                if meta_mr.selected_features:
                    for c in meta_mr.selected_features:
                        if c not in X_meta_mr.columns:
                            X_meta_mr[c] = 0.0
                score_mr = meta_mr.predict(X_meta_mr)
            else:
                score_mr = (p_mr - 0.5) * 0.1

            # 2. Score TF
            if meta_tf:
                numeric_cols = grp.select_dtypes(include=[np.number]).columns.tolist()
                grp_numeric = grp[numeric_cols].copy()
                grp_toggled = apply_interaction_toggles(grp_numeric, keys_tf, ["G_VOL", "G_TREND"], drop_raw=False)

                X_meta_tf = meta_tf.prepare_meta_features(p_tf, grp_toggled, pred_col_name="pred_logit")

                if meta_tf.selected_features:
                    for c in meta_tf.selected_features:
                        if c not in X_meta_tf.columns:
                            X_meta_tf[c] = 0.0
                score_tf = meta_tf.predict(X_meta_tf)
            else:
                score_tf = (p_tf - 0.5) * 0.1

            for i, idx in enumerate(grp.index):
                sym = grp.loc[idx, "symbol"]
                s_mr_val = float(score_mr[i])
                s_tf_val = float(score_tf[i])

                if s_mr_val > s_tf_val:
                    best_s = s_mr_val
                    dom = "mr"
                else:
                    best_s = s_tf_val
                    dom = "tf"

                score_raw_list.append((sym, side_key, best_s, dom))

    # Separate long and short signals
    long_signals = [(sym, sc, dom) for sym, sk, sc, dom in score_raw_list if sk == "long"]
    short_signals = [(sym, sc, dom) for sym, sk, sc, dom in score_raw_list if sk == "short"]

    # For longs: highest scores win; for shorts: highest scores win (model predicts short conviction)
    long_signals.sort(key=lambda x: x[1], reverse=True)
    short_signals.sort(key=lambda x: x[1], reverse=True)

    longs = [x for x in long_signals if x[1] > cfg["thr_long"]][:cfg["k_long"]]
    shorts = [x for x in short_signals if x[1] > cfg["thr_long"]][:cfg["k_short"]]  # Same threshold: positive = conviction

    final_orders = []
    for s, sc, dom in longs: final_orders.append({"symbol": s, "side": "long", "score": sc, "dom": dom})
    for s, sc, dom in shorts: final_orders.append({"symbol": s, "side": "short", "score": sc, "dom": dom})

    # Allocation
    total_wt = sum(abs(x["score"]) for x in final_orders)
    orders_out = []
    if total_wt > 0:
        gross_cap = float(cfg["wallet_gross_cap"])
        for ord in final_orders:
            w_alloc = gross_cap * (abs(ord["score"]) / total_wt)
            ord["weight"] = w_alloc
            # Inject Risk Params from Config if available
            # Map side/dom to key
            r_key = f"risk_{ord['side']}_{ord['dom']}"
            if risk_config and "granular_risk" in risk_config:
                g_risk = risk_config["granular_risk"].get(r_key)
                if g_risk:
                    # Pass these params in the order dict, so main/executor can use them
                    # Or modify them here? executor usually takes order dict.
                    ord["risk_params"] = g_risk

            orders_out.append(ord)

    return orders_out
