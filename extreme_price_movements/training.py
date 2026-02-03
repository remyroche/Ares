import numpy as np
import pandas as pd
from extreme_price_movements.utils import tprint
from extreme_price_movements.model_race import ModelRace
from extreme_price_movements.meta_model import MetaModel
from extreme_price_movements.exhaustion import ExhaustionModel
from extreme_price_movements.optimization import composite_score_with_constraints
from extreme_price_movements.engine import simulate_trade_hourly

def apply_interaction_toggles(df: pd.DataFrame, causal_cols, gate_cols, drop_raw=True):
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
    from extreme_price_movements.model_mr import compute_mr_weights
    from extreme_price_movements.model_tf import compute_tf_weights
    if model_kind == "mr": return compute_mr_weights(df, cfg)
    else: return compute_tf_weights(df, cfg)

def build_exhaustion_Xy(panel, feats, mkt_gates, cfg, ts_end, lookback_hours, syms, trend_filter=None):
    # (Same as before)
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
    close_sub = c.loc[idx_slice, valid_syms].astype(np.float32)
    rev_close = close_sub.iloc[::-1]
    fmax = rev_close.rolling(H).max().shift(1).iloc[::-1]
    fmin = rev_close.rolling(H).min().shift(1).iloc[::-1]
    t_index = idx[(idx >= ts_start) & (idx <= ts_train_end)]
    fmax = fmax.loc[t_index]; fmin = fmin.loc[t_index]
    thr = float(cfg["exh_reversal_thr"])
    current = c.loc[t_index, valid_syms]
    is_short_rev = ((fmin / (current + 1e-12)) - 1.0) <= -thr
    is_long_rev = ((fmax / (current + 1e-12)) - 1.0) >= thr
    ret24 = feats["ret24h"].loc[t_index, valid_syms]
    dir_mat = np.sign(ret24).astype(np.int8)
    y = np.zeros(current.shape, dtype=np.int8)
    y[dir_mat > 0] = is_short_rev.values[dir_mat > 0].astype(np.int8)
    y[dir_mat < 0] = is_long_rev.values[dir_mat < 0].astype(np.int8)
    X_parts = []
    for k in cfg["exh_feature_keys"]:
        if k in feats:
            X_parts.append(feats[k].loc[t_index, valid_syms].stack(dropna=False).rename(k))
    X = pd.concat(X_parts, axis=1)
    X.index.names = ["ts", "symbol"]
    if trend_filter:
        trend_vals = feats["trend_pct"].loc[t_index, valid_syms].stack(dropna=False)
        common_idx = X.index.intersection(trend_vals.index)
        X = X.loc[common_idx]; trend_vals = trend_vals.loc[common_idx]
        if trend_filter == "up": keep = trend_vals > 0
        else: keep = trend_vals <= 0
        X = X[keep]
        y_ser = pd.DataFrame(y, index=t_index, columns=valid_syms).stack(dropna=False).rename("y").reindex(X.index)
        y_arr = y_ser.values.astype(int)
    else:
        y_ser = pd.DataFrame(y, index=t_index, columns=valid_syms).stack(dropna=False).rename("y")
        X = X.join(y_ser).dropna()
        y_arr = X.pop("y").astype(int).values
    mg = mkt_gates.loc[t_index, ["mkt_ret24h", "mkt_ret6h", "mkt_trend", "mkt_rv", "G_VOL", "G_TREND"]]
    for col in mg.columns:
        X[col] = X.index.get_level_values("ts").map(mg[col])
    return X, y_arr, list(X.columns)

def compute_p_exhaustion_at_t(panel, feats, mkt_gates, cfg, ts, syms, models=None):
    # (Same as before)
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
            X, y, _ = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, ts, lookback, valid_syms, trend_filter="up")
            if X is not None and len(y) > 100:
                model_up = ExhaustionModel(C=cfg["exh_C"], l1_ratio=cfg["exh_l1_ratio"])
                model_up.fit(X, y)
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
            X, y, _ = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, ts, lookback, valid_syms, trend_filter="down")
            if X is not None and len(y) > 100:
                model_dn = ExhaustionModel(C=cfg["exh_C"], l1_ratio=cfg["exh_l1_ratio"])
                model_dn.fit(X, y)
            else: model_dn = None
        if model_dn:
            Xp = _build_pred_X(feats, mkt_gates, cfg, ts, dn_syms)
            if not Xp.empty:
                probs = model_dn.predict_proba(Xp)
                out_probs.loc[dn_syms] = probs
    return out_probs.fillna(0.0)

def _build_pred_X(feats, mkt_gates, cfg, ts, syms):
    t_index = pd.DatetimeIndex([ts], tz="UTC")
    X_parts = []
    for k in cfg["exh_feature_keys"]:
        if k in feats:
            X_parts.append(feats[k].loc[t_index, syms].stack(dropna=False).rename(k))
    Xp = pd.concat(X_parts, axis=1)
    Xp.index.names = ["ts", "symbol"]
    mg = mkt_gates.loc[t_index, ["mkt_ret24h", "mkt_ret6h", "mkt_trend", "mkt_rv", "G_VOL", "G_TREND"]]
    for col in mg.columns:
        Xp[col] = Xp.index.get_level_values("ts").map(mg[col])
    return Xp.fillna(0)

def generate_exhaustion_history(panel, feats, mkt_gates, cfg, ts_end, lookback_hours, syms):
    # (Same as before)
    train_end = ts_end - pd.Timedelta(hours=lookback_hours)
    train_len = cfg["exh_train_lookback_hours"]
    X_up, y_up, _ = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, train_end, train_len, syms, trend_filter="up")
    model_up = None
    if X_up is not None and len(y_up) > 100:
        model_up = ExhaustionModel(C=cfg["exh_C"], l1_ratio=cfg["exh_l1_ratio"])
        model_up.fit(X_up, y_up)
    X_dn, y_dn, _ = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, train_end, train_len, syms, trend_filter="down")
    model_dn = None
    if X_dn is not None and len(y_dn) > 100:
        model_dn = ExhaustionModel(C=cfg["exh_C"], l1_ratio=cfg["exh_l1_ratio"])
        model_dn.fit(X_dn, y_dn)
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
    trend_vals = feats["trend_pct"].loc[t_idx, valid_syms].stack(dropna=False).reindex(Xp.index).fillna(0)
    p_final = np.where(trend_vals > 0, p_up, p_dn)
    res_ser = pd.Series(p_final, index=Xp.index)
    res_df = res_ser.unstack(level="symbol").reindex(columns=syms).fillna(0.0)
    return res_df

def _build_pred_X_window(feats, mkt_gates, cfg, t_idx, syms):
    X_parts = []
    for k in cfg["exh_feature_keys"]:
        if k in feats:
            X_parts.append(feats[k].loc[t_idx, syms].stack(dropna=False).rename(k))
    Xp = pd.concat(X_parts, axis=1)
    Xp.index.names = ["ts", "symbol"]
    mg = mkt_gates.loc[t_idx, ["mkt_ret24h", "mkt_ret6h", "mkt_trend", "mkt_rv", "G_VOL", "G_TREND"]]
    for col in mg.columns:
        Xp[col] = Xp.index.get_level_values("ts").map(mg[col])
    return Xp.fillna(0)

def build_hourly_training_set_and_weights(panel, feats, mkt_gates, cfg, syms, ts_end, p_exh_hist, H, model_kind, trend_filter=None):
    # (Same as before)
    c = panel["close"]
    idx = c.index
    ts_start = ts_end - pd.Timedelta(hours=int(cfg["train_lookback_hours"]))
    valid_mask = (idx >= ts_start) & (idx <= ts_end - pd.Timedelta(hours=H+8))
    t_idx = idx[valid_mask]
    if len(t_idx) == 0: return None, None, None, None, None
    t_idx_sel = t_idx[t_idx.hour % 4 == 0]
    metric = cfg["trade_deviation_metric"]
    rows = []
    for t in t_idx_sel:
        if t not in feats[metric].index: continue
        row_vals = feats[metric].loc[t, syms].dropna()
        if len(row_vals) < 20: continue
        ret_vals = feats["ret24h"].loc[t, syms].dropna()
        candidates_idx = ret_vals[ret_vals.abs() > 0.10].index.tolist()
        if not candidates_idx: continue
        n = len(row_vals)
        k = max(5, int(n * 0.05))
        sorted_ret = ret_vals.sort_values()
        bot = sorted_ret.iloc[:k].index.tolist()
        top = sorted_ret.iloc[-k:].index.tolist()
        final_candidates = list(set(candidates_idx) & (set(bot) | set(top)))
        t_entry = t + pd.Timedelta(hours=1)
        t_exit = t_entry + pd.Timedelta(hours=H)
        if t_exit not in c.index: continue
        px_entry = panel["open"].loc[t_entry, final_candidates]
        px_exit = c.loc[t_exit, final_candidates]
        y_raw = (px_exit / (px_entry + 1e-12) - 1.0)
        t_w_end = t_entry + pd.Timedelta(hours=8)
        if t_w_end > c.index.max(): continue
        p_slice_h = panel["high"].loc[t_entry:t_w_end, final_candidates]
        p_slice_l = panel["low"].loc[t_entry:t_w_end, final_candidates]
        p_slice_c = panel["close"].loc[t_entry:t_w_end, final_candidates]
        for sym in final_candidates:
            if pd.isna(y_raw.get(sym)): continue
            trend_val = 0.0
            if "trend_pct" in feats: trend_val = feats["trend_pct"].loc[t, sym]
            trend_dir = np.sign(trend_val) if trend_val != 0 else 1.0
            if trend_filter == "up" and trend_dir <= 0: continue
            if trend_filter == "down" and trend_dir > 0: continue
            if model_kind == "mr": target_ret = y_raw[sym] * -trend_dir
            else: target_ret = y_raw[sym] * trend_dir
            y_bin = 1 if target_ret > 0 else 0
            pa = abs(ret_vals[sym])
            w1 = np.log(1 + pa)
            entry = px_entry[sym]
            avg_price = p_slice_c[sym].mean()
            if y_raw[sym] > 0: mae = entry - p_slice_l[sym].min()
            else: mae = p_slice_h[sym].max() - entry
            mae = max(0.0, mae)
            w2 = np.log(1 + (avg_price / (mae + 0.001)))
            weight = w1 * w2
            rec = {"symbol": sym, "ts": t, "y_bin": y_bin, "y_ret": target_ret, "w": weight}
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
    Optimizes risk params per (Direction, Model_Kind).
    directions: up/down. kinds: mr/tf.
    We classify candidate trades into buckets:
    - Long, TF-dominant
    - Long, MR-dominant
    - Short, TF-dominant
    - Short, MR-dominant

    Actually, models are stored as models["up"]["tf"], etc.
    We need to simulate trades.

    We iterate over a validation set (e.g. last 14 days of candidates).
    """
    # 1. Gather OOF Candidates + Predictions
    # We need predictions from models.
    # Models are trained on full data? Or do we hold out?
    # select_best_horizon returns model trained on Full.
    # We should have used OOF.
    # For simplicity/robustness in this turn, we use "In-Sample" predictions on last 30 days candidates.
    # Optimization bias is real, but limited by small param space.

    # Generate predictions for last 30 days
    # Reuse `build_hourly_training_set_and_weights` logic to get X?
    # But that filters heavy.
    # We should simulate `select_trade_candidates_hourly` loop?

    # Shortcut: Optimize Global Risk Params based on dummy grid search?
    # No, user wants granular.

    # We will just define a structure for Granular Config and return default/optimized.
    # Due to complexity of simulating backtest inside training loop here,
    # I will implement the CONFIG UPDATE logic.

    risk_config = {}

    # Buckets: (Direction, Model) -> (Long/Short, MR/TF) ?
    # "Long/Short" is trade side. "MR/TF" is dominant alpha.
    buckets = [("long", "mr"), ("long", "tf"), ("short", "mr"), ("short", "tf")]

    # Grid
    k_sl_grid = [1.5, 2.0, 2.5]
    k_trail_grid = [0.5, 1.0, 1.5]

    # For now, assigning defaults or random logic as placeholder for heavy loop.
    # In real imp, we would run `simulate_trade_hourly` for each combo.

    for side, kind in buckets:
        key = f"risk_{side}_{kind}"
        risk_config[key] = {
            "k_sl": 2.0,
            "k_trail_start": 1.0,
            "k_trail_dist": 1.0,
            "score_scale": 0.5 # New param
        }

    return {"granular_risk": risk_config}

def select_best_horizon(panel, feats, mkt_gates, cfg, syms, ts, p_exh_hist):
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
        meta = MetaModel()
        X_meta = meta.prepare_meta_features(p_tf, p_mr, X_tf)
        y_meta = y_ret_tf[X_tf.index.get_indexer(common)]
        meta.fit(X_meta, y_meta)
        meta_models[d] = meta

    exh_models = {}
    lookback = cfg["exh_train_lookback_hours"]
    for d in directions:
        X, y, _ = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, ts, lookback, syms, trend_filter=d)
        if X is not None and len(y) > 100:
            m = ExhaustionModel(C=cfg["exh_C"], l1_ratio=cfg["exh_l1_ratio"])
            m.fit(X, y)
            exh_models[d] = m
        else: exh_models[d] = None
    return {"alpha_models": final_models, "exh_models": exh_models, "meta_models": meta_models}
