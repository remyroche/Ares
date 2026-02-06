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

    # Retrieve optimized barrier params or defaults
    vol_lo = float(cfg.get("vol_lo", 0.03))
    vol_hi = float(cfg.get("vol_hi", 0.06))
    vol_z_max = float(cfg.get("vol_z_max", 3.0))

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
                        atr, z, base, z_max=vol_z_max, lo=vol_lo, hi=vol_hi
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
                    feats, t_check, list(feats["ret24h"].columns),
                    cfg["trade_extreme_pct"], cfg["trade_extreme_min"], cfg["trade_extreme_max"],
                    cfg["trade_deviation_metric"]
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

    rows = []
    for sym in candidates:
        try:
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
                rec = {
                    "symbol": sym, "side_key": side_key, "model_mr": model_mr, "model_tf": model_tf,
                    "feat_cols_mr": fcols_mr, "feat_cols_tf": fcols_tf,
                    "mkt_ret24h": float(mrk["mkt_ret24h"]), "mkt_ret6h": float(mrk["mkt_ret6h"]),
                    "mkt_trend": float(mrk["mkt_trend"]), "mkt_rv": float(mrk["mkt_rv"]),
                    "G_VOL": int(mrk["G_VOL"]), "G_TREND": int(mrk["G_TREND"]), "p_exh_lag1": p_lag
                }
                all_keys = set(fcols_mr) | set(fcols_tf) | set(cfg.get("spike_feature_keys", [])) | set(cfg.get("meta_feature_keys", []))
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

    score_rows = []
    for side_key, grp in df_all.groupby("side_key"):
        first = grp.iloc[0]
        model_mr = first["model_mr"]; model_tf = first["model_tf"]
        fcols_mr = first["feat_cols_mr"]; fcols_tf = first["feat_cols_tf"]

        keys_mr = cfg.get("mr_feature_keys", cfg["causal_cols"])
        grp_mr = apply_interaction_toggles(grp.copy(), keys_mr, ["G_VOL","G_TREND"], drop_raw=cfg["drop_raw_causal"])
        X_mr_pred = grp_mr.reindex(columns=fcols_mr, fill_value=0.0).fillna(0.0).astype(np.float32)
        p_mr = model_mr.predict(X_mr_pred)

        keys_tf = cfg.get("tf_feature_keys", cfg["causal_cols"])
        grp_tf = apply_interaction_toggles(grp.copy(), keys_tf, ["G_VOL","G_TREND"], drop_raw=cfg["drop_raw_causal"])
        X_tf_pred = grp_tf.reindex(columns=fcols_tf, fill_value=0.0).fillna(0.0).astype(np.float32)
        p_tf = model_tf.predict(X_tf_pred)

        meta_mr = meta_models.get(f"{side_key}_mr")
        meta_tf = meta_models.get(f"{side_key}_tf")

        if meta_mr:
            num = grp.select_dtypes(include=[np.number]).copy()
            toggled = apply_interaction_toggles(num, keys_mr, ["G_VOL", "G_TREND"], drop_raw=False)
            X_meta = meta_mr.prepare_meta_features(p_mr, toggled, pred_col_name="pred_logit")
            if meta_mr.selected_features:
                X_meta = X_meta.reindex(columns=meta_mr.selected_features, fill_value=0.0)
            s_mr = meta_mr.predict(X_meta)
        else:
            s_mr = (p_mr - 0.5) * 0.1

        if meta_tf:
            num = grp.select_dtypes(include=[np.number]).copy()
            toggled = apply_interaction_toggles(num, keys_tf, ["G_VOL", "G_TREND"], drop_raw=False)
            X_meta = meta_tf.prepare_meta_features(p_tf, toggled, pred_col_name="pred_logit")
            if meta_tf.selected_features:
                X_meta = X_meta.reindex(columns=meta_tf.selected_features, fill_value=0.0)
            s_tf = meta_tf.predict(X_meta)
        else:
            s_tf = (p_tf - 0.5) * 0.1

        for i, idx in enumerate(grp.index):
            score_rows.append({
                "symbol": grp.loc[idx, "symbol"],
                "side_key": side_key,
                "score_mr": float(s_mr[i]),
                "score_tf": float(s_tf[i]),
            })

    return pd.DataFrame(score_rows)

def generate_hourly_signals(ts_sig, feats, mkt_gates, model_bundle, risk_config, cfg, p_exh_cand, current_positions_syms, tradeable_candidates=None):
    tprint(f"Entering function: generate_hourly_signals in engine.py")
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

        if score_scale:
            l_mr = _robust_norm(l_mr, score_scale.get("long_mr_center", 0.0), score_scale.get("long_mr_scale", 1.0))
            s_mr = _robust_norm(s_mr, score_scale.get("short_mr_center", 0.0), score_scale.get("short_mr_scale", 1.0))
            l_tf = _robust_norm(l_tf, score_scale.get("long_tf_center", 0.0), score_scale.get("long_tf_scale", 1.0))
            s_tf = _robust_norm(s_tf, score_scale.get("short_tf_center", 0.0), score_scale.get("short_tf_scale", 1.0))

        net_mr = float(l_mr - s_mr)
        net_tf = float(l_tf - s_tf)
        if abs(net_mr) >= abs(net_tf):
            net_score = net_mr
            dom = "mr"
        else:
            net_score = net_tf
            dom = "tf"

        long_mode = _bucket_mode_from_side_dom("long", dom)
        short_mode = _bucket_mode_from_side_dom("short", dom)
        dom_thr_long = float(signal_params.get(f"thr_{dom}_{long_mode}", thr_long))
        dom_thr_short = float(signal_params.get(f"thr_{dom}_{short_mode}", thr_short))

        if net_score >= dom_thr_long:
            final_orders.append({"symbol": sym, "side": "long", "score": net_score, "dom": dom, "mode": long_mode})
            abs_scores.append(abs(net_score))
        elif net_score <= dom_thr_short:
            final_orders.append({"symbol": sym, "side": "short", "score": net_score, "dom": dom, "mode": short_mode})
            abs_scores.append(abs(net_score))

    if not final_orders:
        return []

    longs = [o for o in final_orders if o["side"] == "long"]
    shorts = [o for o in final_orders if o["side"] == "short"]
    longs.sort(key=lambda x: x["score"], reverse=True)
    shorts.sort(key=lambda x: x["score"])  # more negative first
    final_orders = longs[:k_long] + shorts[:k_short]

    if size_q50 is None or size_q90 is None:
        arr = np.array([abs(o["score"]) for o in final_orders], dtype=np.float64)
        size_q50 = float(np.quantile(arr, 0.5)) if arr.size else 0.0
        size_q90 = float(np.quantile(arr, 0.9)) if arr.size else 1.0

    orders_out = []
    gross_cap = float(cfg.get("wallet_gross_cap", 1.0))
    raw_w = []
    for o in final_orders:
        z = abs(float(o["score"]))
        z_tilde = np.clip((z - size_q50) / (size_q90 - size_q50 + 1e-12), 0.0, size_zcap)
        fz = 1.0 / (1.0 + np.exp(-size_k * (z_tilde - size_x0)))
        w_alloc = size_min + (size_max - size_min) * fz
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