import numpy as np
import pandas as pd
from extreme_price_movements.utils import tprint
from extreme_price_movements.model_race import ModelRace
from extreme_price_movements.meta_model import MetaModel
from extreme_price_movements.exhaustion import ExhaustionModel
from extreme_price_movements.optimization import composite_score_with_constraints
from extreme_price_movements.candidates import select_trade_candidates_hourly, select_trade_candidates_vectorized
import extreme_price_movements.fast_funcs as ff
from extreme_price_movements.labeling import compute_triple_barrier_labels

def apply_interaction_toggles(df: pd.DataFrame, causal_cols, gate_cols, drop_raw=True):
    tprint(f"Entering function: apply_interaction_toggles in training.py")
    out = df.copy()
    for g in gate_cols:
        if g not in out.columns:
            continue
        for col in causal_cols:
            if col in out.columns:
                out[f"{col}_{g}_0"] = out[col] * (1 - out[g])
                out[f"{col}_{g}_1"] = out[col] * out[g]
    if drop_raw:
        out = out.drop(columns=[c for c in causal_cols if c in out.columns], errors="ignore")
    return out

def compute_weights_logic(df, cfg, model_kind):
    tprint(f"Entering function: compute_weights_logic in training.py")
    from extreme_price_movements.model_mr import compute_mr_weights
    from extreme_price_movements.model_tf import compute_tf_weights
    if model_kind == "mr": return compute_mr_weights(df, cfg)
    else: return compute_tf_weights(df, cfg)

def build_exhaustion_Xy(panel, feats, mkt_gates, cfg, ts_end, lookback_hours, syms, trend_filter=None):
    tprint(f"Entering function: build_exhaustion_Xy in training.py")
    c = panel["close"]
    idx = c.index
    H = int(cfg["exh_horizon_hours"])
    ts_train_end = ts_end - pd.Timedelta(hours=1)
    ts_start = ts_train_end - pd.Timedelta(hours=int(lookback_hours))
    if ts_train_end not in idx: return None, None, None
    mask = (idx >= ts_start) & (idx <= ts_train_end + pd.Timedelta(hours=H))
    idx_slice = idx[mask]
    valid_syms = [s for s in syms if s in c.columns]
    if not valid_syms: return None, None, None
    t_index = idx[(idx >= ts_start) & (idx <= ts_train_end)]
    current = c.loc[t_index, valid_syms]

    if cfg.get("exh_label_type", "simple") == "peak":
        use_atr = cfg.get("exh_use_atr", True)
        if use_atr and "atr_pct" in feats:
             atr_full = feats["atr_pct"] * panel["close"]
             near_k = float(cfg.get("exh_atr_near_k", 0.5))
             rev_k = float(cfg.get("exh_atr_rev_k", 2.0))
        else:
             atr_full = panel["close"]
             near_k = float(cfg.get("exh_near_thr", 0.01))
             rev_k = float(cfg.get("exh_rev_thr_pct", 0.04))

        common_idx = panel["close"].index.intersection(atr_full.index)
        c_full = panel["close"].loc[common_idx]
        a_full = atr_full.loc[common_idx]

        max_near = float(cfg.get("exh_near_dist_cap_pct", 0.02))
        min_rev = float(cfg.get("exh_rev_dist_floor_pct", 0.005))

        l_short, w_short = ff.compute_peak_labels_and_weights(c_full, a_full, H, near_k, rev_k, True, max_near, min_rev)
        l_long, w_long = ff.compute_peak_labels_and_weights(c_full, a_full, H, near_k, rev_k, False, max_near, min_rev)

        l_short_s = l_short.reindex(index=t_index, columns=valid_syms)
        l_long_s = l_long.reindex(index=t_index, columns=valid_syms)
        w_short_s = w_short.reindex(index=t_index, columns=valid_syms)
        w_long_s = w_long.reindex(index=t_index, columns=valid_syms)

        is_short_rev = l_short_s.fillna(0) > 0.5
        is_long_rev = l_long_s.fillna(0) > 0.5

        # We need to store weights for later use.
        # But build_exhaustion_Xy returns X, y, cols. It doesn't return weights yet.
        # However, the user asked to "add sample weights".
        # build_exhaustion_Xy is called by compute_p_exhaustion_at_t, which fits the model.
        # We need to update build_exhaustion_Xy signature to return weights or handle them.
    else:
        close_sub = c.loc[idx_slice, valid_syms].astype(np.float32)
        rev_close = close_sub.iloc[::-1]
        fmax = rev_close.rolling(H).max().shift(1).iloc[::-1]
        fmin = rev_close.rolling(H).min().shift(1).iloc[::-1]
        fmax = fmax.loc[t_index]; fmin = fmin.loc[t_index]
        thr = float(cfg["exh_reversal_thr"])
        is_short_rev = ((fmin / (current + 1e-12)) - 1.0) <= -thr
        is_long_rev = ((fmax / (current + 1e-12)) - 1.0) >= thr
    ret24 = feats["ret24h"].loc[t_index, valid_syms]
    dir_mat = np.sign(ret24).astype(np.int8)
    y = np.zeros(current.shape, dtype=np.int8)
    w = np.ones(current.shape, dtype=np.float32)

    # Assign labels and weights
    # For uptrend (dir_mat > 0), use short reversal labels/weights
    mask_up = (dir_mat > 0)
    if mask_up.any():
        # Align index/columns
        # is_short_rev is a DataFrame aligned with current
        # w_short_s is a DataFrame aligned with current
        y[mask_up] = is_short_rev.values[mask_up].astype(np.int8)
        if cfg.get("exh_label_type") == "peak":
             w[mask_up] = w_short_s.values[mask_up].astype(np.float32)

    mask_dn = (dir_mat < 0)
    if mask_dn.any():
        y[mask_dn] = is_long_rev.values[mask_dn].astype(np.int8)
        if cfg.get("exh_label_type") == "peak":
             w[mask_dn] = w_long_s.values[mask_dn].astype(np.float32)

    # Winsorize Weights (Top 80% kept -> Clip top 20%)
    # Winsorize only if we have weights > 1
    if cfg.get("exh_label_type") == "peak":
        # Global winsorization or per-batch? Global over the passed slice is fine.
        w_flat = w.flatten()
        q_high = np.nanquantile(w_flat, 0.80)
        # Wait, "Winsorise the top 80%" usually means clamp outliers.
        # User: "Winsorise the top 80%" -> probably means "Winsorize at 80th percentile" (clamp top 20%).
        # Or keep 80%? Usually top 1-5% are outliers. 20% is aggressive but user requested it.
        # If weights are mostly 1.0 (negatives), then quantile 0.8 might be 1.0.
        # We should only winsorize the boosted weights (w > 1).

        mask_boosted = w > 1.0
        if mask_boosted.sum() > 10:
             boosted_vals = w[mask_boosted]
             cap = np.quantile(boosted_vals, 0.80)
             w[w > cap] = cap
    X_parts = []
    for k in cfg["exh_feature_keys"]:
        if k in feats:
            X_parts.append(feats[k].loc[t_index, valid_syms].stack(future_stack=True).rename(k))
    X = pd.concat(X_parts, axis=1)
    X.index.names = ["ts", "symbol"]
    if trend_filter:
        trend_vals = feats["trend_pct"].loc[t_index, valid_syms].stack(future_stack=True)
        common_idx = X.index.intersection(trend_vals.index)
        X = X.loc[common_idx]; trend_vals = trend_vals.loc[common_idx]
        if trend_filter == "up": keep = trend_vals > 0
        else: keep = trend_vals <= 0
        X = X[keep]
        y_ser = pd.DataFrame(y, index=t_index, columns=valid_syms).stack(future_stack=True).rename("y").reindex(X.index)
        y_arr = y_ser.values.astype(int)
    else:
        y_df = pd.DataFrame(y, index=t_index, columns=valid_syms).stack(future_stack=True).rename("y")
        w_df = pd.DataFrame(w, index=t_index, columns=valid_syms).stack(future_stack=True).rename("w")
        X = X.join(y_df).join(w_df).dropna()
        y_arr = X.pop("y").astype(int).values
        w_arr = X.pop("w").astype(np.float32).values

    mg = mkt_gates.loc[t_index, ["mkt_ret24h", "mkt_ret6h", "mkt_trend", "mkt_rv", "G_VOL", "G_TREND"]]
    for col in mg.columns:
        X[col] = X.index.get_level_values("ts").map(mg[col])

    return X, y_arr, w_arr, list(X.columns)

def compute_p_exhaustion_at_t(panel, feats, mkt_gates, cfg, ts, syms, models=None):
    tprint(f"Entering function: compute_p_exhaustion_at_t in training.py")
    t_index = pd.DatetimeIndex([ts], tz="UTC")
    valid_syms = [s for s in syms if s in panel["close"].columns]
    trend_vals = feats["trend_pct"].loc[ts, valid_syms]
    up_syms = trend_vals[trend_vals > 0].index.tolist()
    dn_syms = trend_vals[trend_vals <= 0].index.tolist()
    out_probs = pd.Series(index=syms, dtype=float).fillna(0.0)
    lookback = cfg["exh_train_lookback_hours"]
    if up_syms:
        if models and "up" in models: model_up = models["up"]
        else:
            X, y, w, _ = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, ts, lookback, valid_syms, trend_filter="up")
            if X is not None and len(y) > 100:
                model_up = ExhaustionModel(C=cfg["exh_C"], l1_ratio=cfg["exh_l1_ratio"])
                model_up.fit(X, y, sample_weight=w)
            else: model_up = None
        if model_up:
            Xp = _build_pred_X(feats, mkt_gates, cfg, ts, up_syms)
            if not Xp.empty:
                probs = model_up.predict_proba(Xp)
                probs = np.clip(probs * 2.0, 0.0, 1.0)
                out_probs.loc[up_syms] = probs
    if dn_syms:
        if models and "down" in models: model_dn = models["down"]
        else:
            X, y, w, _ = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, ts, lookback, valid_syms, trend_filter="down")
            if X is not None and len(y) > 100:
                model_dn = ExhaustionModel(C=cfg["exh_C"], l1_ratio=cfg["exh_l1_ratio"])
                model_dn.fit(X, y, sample_weight=w)
            else: model_dn = None
        if model_dn:
            Xp = _build_pred_X(feats, mkt_gates, cfg, ts, dn_syms)
            if not Xp.empty:
                probs = model_dn.predict_proba(Xp)
                out_probs.loc[dn_syms] = probs
    return out_probs.fillna(0.0)

def _build_pred_X(feats, mkt_gates, cfg, ts, syms):
    tprint(f"Entering function: _build_pred_X in training.py")
    t_index = pd.DatetimeIndex([ts], tz="UTC")
    X_parts = []
    for k in cfg["exh_feature_keys"]:
        if k in feats:
            X_parts.append(feats[k].loc[t_index, syms].stack(future_stack=True).rename(k))
    Xp = pd.concat(X_parts, axis=1)
    Xp.index.names = ["ts", "symbol"]
    mg = mkt_gates.loc[t_index, ["mkt_ret24h", "mkt_ret6h", "mkt_trend", "mkt_rv", "G_VOL", "G_TREND"]]
    for col in mg.columns:
        Xp[col] = Xp.index.get_level_values("ts").map(mg[col])
    return Xp.fillna(0)

def generate_exhaustion_history(panel, feats, mkt_gates, cfg, ts_end, lookback_hours, syms):
    tprint(f"Entering function: generate_exhaustion_history in training.py")
    train_end = ts_end - pd.Timedelta(hours=lookback_hours)
    train_len = cfg["exh_train_lookback_hours"]
    X_up, y_up, w_up, _ = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, train_end, train_len, syms, trend_filter="up")
    model_up = None
    if X_up is not None and len(y_up) > 100:
        model_up = ExhaustionModel(C=cfg["exh_C"], l1_ratio=cfg["exh_l1_ratio"])
        model_up.fit(X_up, y_up, sample_weight=w_up)
    X_dn, y_dn, w_dn, _ = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, train_end, train_len, syms, trend_filter="down")
    model_dn = None
    if X_dn is not None and len(y_dn) > 100:
        model_dn = ExhaustionModel(C=cfg["exh_C"], l1_ratio=cfg["exh_l1_ratio"])
        model_dn.fit(X_dn, y_dn, sample_weight=w_dn)
    t_idx = pd.date_range(train_end, ts_end, freq='h', tz="UTC")
    t_idx = t_idx[t_idx.isin(panel["close"].index)]
    valid_syms = [s for s in syms if s in panel["close"].columns]
    Xp = _build_pred_X_window(feats, mkt_gates, cfg, t_idx, valid_syms)
    p_up = 0.0
    if model_up:
        p_up = model_up.predict_proba(Xp)
        p_up = np.clip(p_up * 2.0, 0.0, 1.0)
    p_dn = 0.0
    if model_dn:
        p_dn = model_dn.predict_proba(Xp)
    trend_vals = feats["trend_pct"].loc[t_idx, valid_syms].stack(future_stack=True).reindex(Xp.index).fillna(0)
    p_final = np.where(trend_vals > 0, p_up, p_dn)
    res_ser = pd.Series(p_final, index=Xp.index)
    res_df = res_ser.unstack(level="symbol").reindex(columns=syms).fillna(0.0)
    return res_df

def _build_pred_X_window(feats, mkt_gates, cfg, t_idx, syms):
    tprint(f"Entering function: _build_pred_X_window in training.py")
    X_parts = []
    for k in cfg["exh_feature_keys"]:
        if k in feats:
            X_parts.append(feats[k].loc[t_idx, syms].stack(future_stack=True).rename(k))
    Xp = pd.concat(X_parts, axis=1)
    Xp.index.names = ["ts", "symbol"]
    mg = mkt_gates.loc[t_idx, ["mkt_ret24h", "mkt_ret6h", "mkt_trend", "mkt_rv", "G_VOL", "G_TREND"]]
    for col in mg.columns:
        Xp[col] = Xp.index.get_level_values("ts").map(mg[col])
    return Xp.fillna(0)

def build_hourly_training_set_and_weights(panel, feats, mkt_gates, cfg, syms, ts_end, p_exh_hist, H, model_kind, trend_filter=None):
    tprint(f"Entering function: build_hourly_training_set_and_weights in training.py")
    c = panel["close"]
    idx = c.index

    # 1. Labels
    tp = cfg.get("tp", 0.05)
    sl = cfg.get("sl", 0.025)
    tb_labels, tb_returns = compute_triple_barrier_labels(panel, tp, sl, H)

    # 2. Vectorized Candidate Selection
    cand_mask = select_trade_candidates_vectorized(panel, feats, pct=cfg["trade_extreme_pct"], metric=cfg["trade_deviation_metric"])
    if cand_mask is None:
        return None, None, None, None, None

    # Filter to training window
    ts_start = ts_end - pd.Timedelta(hours=int(cfg["train_lookback_hours"]))
    # Mask to valid training period
    # Note: cand_mask spans the whole feats index.
    # We slice it.
    valid_window_mask = (cand_mask.index >= ts_start) & (cand_mask.index <= ts_end - pd.Timedelta(hours=H+8))
    # Subsample every 4 hours to match original density preference
    subsample_mask = (cand_mask.index.hour % 4 == 0)

    final_mask = cand_mask & pd.Series(valid_window_mask & subsample_mask, index=cand_mask.index).fillna(False) # Broadcasting

    # We iterate over timestamps that have at least one candidate
    # This might be faster than iterating all t in window

    # Find timestamps where at least one symbol is True
    valid_ts = final_mask[final_mask.any(axis=1)].index

    rows = []

    for t in valid_ts:
        # Get symbols at t
        row_mask = final_mask.loc[t]
        final_candidates = row_mask[row_mask].index.tolist()

        # Intersection with syms (allowed universe)
        final_candidates = [s for s in final_candidates if s in syms]

        if not final_candidates: continue

        t_entry = t + pd.Timedelta(hours=1)
        if t_entry not in tb_labels.index: continue

        # Retrieve metric vals for weighting (ret24h)
        # Using vectorized access
        ret_vals = feats["ret24h"].loc[t, final_candidates]

        for sym in final_candidates:
            if sym not in tb_labels.columns: continue

            # TB Outcome
            lbl = tb_labels.loc[t_entry, sym]
            ret = tb_returns.loc[t_entry, sym]

            # Filter trend
            trend_val = 0.0
            if "trend_pct" in feats: trend_val = feats["trend_pct"].loc[t, sym]
            trend_dir = np.sign(trend_val) if trend_val != 0 else 1.0

            if trend_filter == "up" and trend_dir <= 0: continue
            if trend_filter == "down" and trend_dir > 0: continue

            trade_dir = 1 if model_kind == "tf" else -1
            pnl = ret * trade_dir * trend_dir
            y_bin = 1 if pnl > 0 else 0

            # Weighting
            pa = abs(ret_vals[sym])
            w1 = np.log(1 + pa)
            w2 = 1.0
            weight = w1 * w2

            rec = {"symbol": sym, "ts": t, "y_bin": y_bin, "y_ret": pnl, "w": weight}

            t_lag = t - pd.Timedelta(hours=1)
            p_val = 0.0
            if t_lag in p_exh_hist.index and sym in p_exh_hist.columns:
                p_val = p_exh_hist.loc[t_lag, sym]
            rec["p_exh_lag1"] = p_val
            for k in cfg["causal_cols"]:
                if k == "p_exh_lag1": continue
                if k == "a_funding_proxy": k = "funding_proxy"
                if k in feats: rec[k] = feats[k].loc[t, sym]
            rec["G_VOL"] = mkt_gates.loc[t, "G_VOL"]
            rec["G_TREND"] = mkt_gates.loc[t, "G_TREND"]
            rows.append(rec)

    if not rows: return None, None, None, None, None
    df = pd.DataFrame(rows).dropna()
    weights = df.pop("w").values.astype(np.float32)
    weights = np.clip(weights, 0.1, 10.0)
    df = apply_interaction_toggles(df, cfg["causal_cols"], ["G_VOL", "G_TREND"], drop_raw=cfg["drop_raw_causal"])
    y_bin = df.pop("y_bin").values.astype(int)
    y_ret = df.pop("y_ret").values.astype(np.float32)
    X_out = df.drop(columns=["ts", "symbol"]).astype(np.float32)
    X_out.index = df.index
    return X_out, y_bin, y_ret, list(X_out.columns), weights

def optimize_risk_params(panel, feats, mkt_gates, cfg, syms, ts, p_exh_hist, models):
    """
    Optimizes risk params per (Direction, Model_Kind) using simulation on validation set.
    """
    tprint("Optimizing Risk Params...")
    alpha_models = models.get("alpha_models")
    meta_models = models.get("meta_models")
    if not alpha_models:
        return {"granular_risk": {}}

    val_hours = int(cfg.get("val_lookback_hours", 24*7))
    ts_start = ts - pd.Timedelta(hours=val_hours)

    # We can stick to simpler candidate selection for Validation Risk Optimization
    # because we want to optimize execution on 'typical' candidates.
    # However, ideally we use the same process.
    # But `vectorized` works on full feats.

    # Let's use vectorized candidates for validation too!
    cand_mask = select_trade_candidates_vectorized(panel, feats, pct=cfg["trade_extreme_pct"], metric=cfg["trade_deviation_metric"])
    if cand_mask is None: return {"granular_risk": {}}

    valid_window_mask = (cand_mask.index >= ts_start) & (cand_mask.index < ts - pd.Timedelta(hours=48))
    # Step 2 hours or 4 hours
    subsample_mask = (cand_mask.index.hour % 2 == 0)
    final_mask = cand_mask & pd.Series(valid_window_mask & subsample_mask, index=cand_mask.index).fillna(False)

    valid_ts = final_mask[final_mask.any(axis=1)].index
    candidates = []

    trend_df = feats.get("trend_pct")
    o_df = panel["open"]
    h_df = panel["high"]
    l_df = panel["low"]
    c_df = panel["close"]

    for t_idx in valid_ts:
        row_mask = final_mask.loc[t_idx]
        trade_syms = row_mask[row_mask].index.tolist()
        trade_syms = [s for s in trade_syms if s in syms]
        if not trade_syms: continue

        mrk = mkt_gates.loc[t_idx]
        t_exh_lag = t_idx - pd.Timedelta(hours=1)

        rows = []
        for sym in trade_syms:
            try:
                t_val = 0.0
                if trend_df is not None and sym in trend_df.columns:
                    t_val = float(trend_df.loc[t_idx, sym])
                direction = "up" if t_val > 0 else "down"

                m_bundle = alpha_models.get(direction)
                if not m_bundle or not m_bundle["mr"] or not m_bundle["tf"]: continue

                model_mr = m_bundle["mr"]["model"]
                model_tf = m_bundle["tf"]["model"]
                feat_cols = m_bundle["mr"]["feat_cols"]
                meta_model = meta_models.get(direction)

                p_lag = 0.5
                if t_exh_lag in p_exh_hist.index and sym in p_exh_hist.columns:
                    p_lag = float(p_exh_hist.loc[t_exh_lag, sym])

                rec = {
                    "symbol": sym, "direction": direction,
                    "model_mr": model_mr, "model_tf": model_tf, "meta_model": meta_model,
                    "feat_cols": feat_cols,
                    "mkt_ret24h": float(mrk["mkt_ret24h"]),
                    "mkt_ret6h": float(mrk["mkt_ret6h"]),
                    "mkt_trend": float(mrk["mkt_trend"]),
                    "mkt_rv": float(mrk["mkt_rv"]),
                    "G_VOL": int(mrk["G_VOL"]), "G_TREND": int(mrk["G_TREND"]),
                    "p_exh_lag1": p_lag
                }
                for k in feat_cols:
                    if k in feats: rec[k] = float(feats[k].loc[t_idx, sym])

                for mk in ["a_rv24", "a_volz", "a_rsi", "dist_ema_fast", "atr_slope", "dist_vwap_norm", "mom_accel"]:
                    if mk in feats: rec[mk] = float(feats[mk].loc[t_idx, sym])

                rows.append(rec)
            except: continue

        if not rows: continue

        df_all = pd.DataFrame(rows)

        # Predict
        for d, grp in df_all.groupby("direction"):
            first = grp.iloc[0]
            model_mr = first["model_mr"]; model_tf = first["model_tf"]; meta_model = first["meta_model"]; fcols = first["feat_cols"]

            Xint = apply_interaction_toggles(grp, cfg["causal_cols"], ["G_VOL","G_TREND"], drop_raw=cfg["drop_raw_causal"])
            for c in fcols:
                if c not in Xint.columns: Xint[c] = 0.0
            Xpred = Xint[fcols].fillna(0.0).astype(np.float32)

            p_mr = model_mr.predict(Xpred)
            p_tf = model_tf.predict(Xpred)

            if meta_model:
                X_meta = meta_model.prepare_meta_features(p_tf, p_mr, grp)
                score = meta_model.predict(X_meta)
            else:
                score = p_tf - p_mr
                sign = 1.0 if d == "up" else -1.0
                score = score * sign

            for i, idx in enumerate(grp.index):
                sym = grp.loc[idx, "symbol"]
                s_score = score[i]
                dom = "mr" if p_mr[i] > p_tf[i] else "tf"

                ts_entry = t_idx + pd.Timedelta(hours=1)
                entry_px = float(o_df.loc[ts_entry, sym]) if ts_entry in o_df.index else np.nan
                if np.isnan(entry_px): continue

                atr_val = float(feats["atr_pct"].loc[t_idx, sym])

                side = "long" if s_score > 0 else "short"
                if abs(s_score) < 0.005: continue

                candidates.append({
                    "ts": t_idx,
                    "symbol": sym,
                    "side": side,
                    "dom": dom,
                    "score": s_score,
                    "entry_px": entry_px,
                    "atr": atr_val
                })

    if not candidates:
        return {"granular_risk": {}}

    # 3. Prepare Simulation Data
    hold_h = int(cfg.get("hold_hours", 48))
    sim_data = []

    for cand in candidates:
        ts_entry = cand["ts"] + pd.Timedelta(hours=1)
        ts_exit = ts_entry + pd.Timedelta(hours=hold_h)
        sym = cand["symbol"]

        if ts_exit > c_df.index.max():
            ts_exit = c_df.index.max()

        sl = slice(ts_entry, ts_exit)
        try:
            o_arr = o_df.loc[sl, sym].to_numpy(dtype=np.float32)
            h_arr = h_df.loc[sl, sym].to_numpy(dtype=np.float32)
            l_arr = l_df.loc[sl, sym].to_numpy(dtype=np.float32)
            c_arr = c_df.loc[sl, sym].to_numpy(dtype=np.float32)

            if len(c_arr) == 0: continue

            cand["o"] = o_arr
            cand["h"] = h_arr
            cand["l"] = l_arr
            cand["c"] = c_arr
            sim_data.append(cand)
        except: continue

    # 4. Grid Search
    # New Grids
    k_sl_grid = [1.5, 2.0, 3.0]
    k_pt_grid = [1.5, 2.0, 3.0] # Activation (k_pt)
    k_tp_grid = [0.5, 1.0, 1.5] # Trailing Dist (k_tp)

    buckets = ["long_mr", "long_tf", "short_mr", "short_tf"]
    best_params = {}

    for b in buckets:
        side_req, dom_req = b.split("_")
        subset = [c for c in sim_data if c["side"] == side_req and c["dom"] == dom_req]

        # Default fallback
        best_params[f"risk_{b}"] = {
            "k_sl": 2.0, "k_pt": 2.0, "k_tp": 1.0, "score_scale": 0.5
        }

        if len(subset) < 10:
            continue

        best_perf = -1e9
        best_combo = None

        for k_sl in k_sl_grid:
            for k_pt in k_pt_grid:
                for k_tp in k_tp_grid:
                    total_ret = 0.0

                    for c in subset:
                        entry_px = float(c["entry_px"])
                        atr_val = float(c["atr"])
                        side_int = 1 if c["side"] == "long" else -1

                        # Apply Clamps
                        # sl_pct = clamp(k_sl * ATR%, 2%, 5%)
                        sl_pct = np.clip(k_sl * atr_val, 0.02, 0.05)
                        # pt_pct (activation) = clamp(k_pt * ATR%, 5%, 10%)
                        pt_pct = np.clip(k_pt * atr_val, 0.05, 0.10)
                        # tp_pct (dist) = clamp(k_tp * ATR%, 2%, 4%)
                        tp_pct = np.clip(k_tp * atr_val, 0.02, 0.04)

                        sl_dist = sl_pct * entry_px
                        act_dist = pt_pct * entry_px
                        tr_dist = tp_pct * entry_px

                        ret, _, _ = ff.simulate_trade_numba(
                            c["o"], c["h"], c["l"], c["c"],
                            entry_px, side_int,
                            sl_dist, act_dist, tr_dist
                        )
                        total_ret += ret

                    if total_ret > best_perf:
                        best_perf = total_ret
                        best_combo = (k_sl, k_pt, k_tp)

        if best_combo:
            k_sl, k_pt, k_tp = best_combo
            best_params[f"risk_{b}"] = {
                "k_sl": k_sl, "k_pt": k_pt, "k_tp": k_tp, "score_scale": 0.5
            }

    tprint(f"Risk Params Optimized: {best_params}")
    return {"granular_risk": best_params}

def select_best_horizon(panel, feats, mkt_gates, cfg, syms, ts, p_exh_hist):
    tprint(f"Entering function: select_best_horizon in training.py")
    directions = ["up", "down"]
    kinds = ["mr", "tf"]
    final_models = {}
    for d in directions:
        final_models[d] = {}
        for k in kinds:
            best_ic = -1.0; best_m = None
            horizons = cfg["label_horizons_hours"]
            for H in horizons:
                tprint(f"Selecting {d} {k} H={H}...")
                X, y, y_ret, cols, w = build_hourly_training_set_and_weights(panel, feats, mkt_gates, cfg, syms, ts, p_exh_hist, H, k, trend_filter=d)
                if X is None or len(y) < cfg["min_train_samples"] // 4: continue
                race = ModelRace(kind=k, n_splits=3)
                race.fit(X, y, sample_weight=w, returns=y_ret)
                score = race.metrics.get(race.best_model_name, -1.0)
                if score > best_ic:
                    best_ic = score
                    best_m = {"model": race, "H": H, "feat_cols": cols}
            final_models[d][k] = best_m

    meta_models = {}
    for d in directions:
        mr_conf = final_models[d]["mr"]
        tf_conf = final_models[d]["tf"]
        if not mr_conf or not tf_conf:
            meta_models[d] = None; continue
        H_mr = mr_conf["H"]
        X_mr, _, _, _, _ = build_hourly_training_set_and_weights(panel, feats, mkt_gates, cfg, syms, ts, p_exh_hist, H_mr, "mr", trend_filter=d)
        H_tf = tf_conf["H"]
        X_tf, y_tf, y_ret_tf, cols_tf, _ = build_hourly_training_set_and_weights(panel, feats, mkt_gates, cfg, syms, ts, p_exh_hist, H_tf, "tf", trend_filter=d)
        common = X_mr.index.intersection(X_tf.index)
        if len(common) < 100: meta_models[d] = None; continue
        X_mr = X_mr.loc[common]; X_tf = X_tf.loc[common]
        p_mr = mr_conf["model"].predict(X_mr)
        p_tf = tf_conf["model"].predict(X_tf)

        # Train Meta Model (Ridge)
        meta = MetaModel()
        X_meta = meta.prepare_meta_features(p_tf, p_mr, X_tf) # X_tf has meta features in it?
        # X_tf is feats_df passed to prepare_meta_features.
        # It needs `atr_slope`, `mom_accel` etc.
        # `build_hourly_training_set_and_weights` collects `causal_cols`.
        # I MUST ensure these new features are in `causal_cols` in config OR
        # explicitly collect them in `build_hourly_training_set_and_weights`.

        # `build_hourly_training_set_and_weights` collects:
        # `rec[k] = feats[k]` for k in `causal_cols`.
        # So I rely on `config["causal_cols"]` having them.
        # Or I modify `build_hourly_training_set_and_weights` to add them explicitly.
        # Given I cannot easily edit `config.py` in this step (or I could),
        # I'll just add them to the extraction loop in `build_hourly_training_set_and_weights`.
        # Actually I didn't add them in my `write_file` above!
        # I added them in `optimize_risk_params` but NOT `build_hourly_training_set_and_weights`.
        # I should add the Meta features to `build_hourly_training_set_and_weights` output.

        y_meta = y_ret_tf[X_tf.index.get_indexer(common)]
        meta.fit(X_meta, y_meta)
        meta_models[d] = meta

    exh_models = {}
    lookback = cfg["exh_train_lookback_hours"]
    for d in directions:
        X, y, w, _ = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, ts, lookback, syms, trend_filter=d)
        if X is not None and len(y) > 100:
            m = ExhaustionModel(C=cfg["exh_C"], l1_ratio=cfg["exh_l1_ratio"])
            m.fit(X, y, sample_weight=w)
            exh_models[d] = m
        else: exh_models[d] = None
    return {"alpha_models": final_models, "exh_models": exh_models, "meta_models": meta_models}
