import numpy as np
import pandas as pd
from extreme_price_movements.utils import tprint
from extreme_price_movements.model_race import ModelRace
from extreme_price_movements.meta_model import MetaModel
from extreme_price_movements.exhaustion import ExhaustionModel
from extreme_price_movements.candidates import select_trade_candidates_hourly, select_trade_candidates_vectorized
import extreme_price_movements.fast_funcs as ff
from extreme_price_movements.labeling import compute_trailing_atr_labels
from extreme_price_movements.sample_weights import build_label_time_ranges, compute_sample_weights_with_uniqueness
from sklearn.mixture import GaussianMixture

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

    label_type = cfg.get("exh_label_type", "simple")
    tprint(f"Exhaustion Label Type: {label_type}")
    if label_type == "peak":
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
    dir_mat = np.sign(ret24).fillna(0).astype(np.int8)
    y = np.zeros(current.shape, dtype=np.int8)
    w = np.ones(current.shape, dtype=np.float32)

    mask_up = (dir_mat > 0)
    if mask_up.values.any():
        y[mask_up] = is_short_rev.values[mask_up].astype(np.int8)
        if cfg.get("exh_label_type") == "peak":
             w[mask_up] = w_short_s.values[mask_up].astype(np.float32)

    mask_dn = (dir_mat < 0)
    if mask_dn.values.any():
        y[mask_dn] = is_long_rev.values[mask_dn].astype(np.int8)
        if cfg.get("exh_label_type") == "peak":
             w[mask_dn] = w_long_s.values[mask_dn].astype(np.float32)

    if cfg.get("exh_label_type") == "peak":
        mask_boosted = w > 1.0
        if mask_boosted.sum() > 10:
             boosted_vals = w[mask_boosted]
             cap = np.quantile(boosted_vals, 0.80)
             w[w > cap] = cap
    X_parts = []
    # Exhaustion features are specific now
    for k in cfg.get("exh_feature_keys", cfg.get("exh_feature_keys_legacy", [])):
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
        
        w_ser = pd.DataFrame(w, index=t_index, columns=valid_syms).stack(future_stack=True).rename("w").reindex(X.index)
        w_arr = w_ser.values.astype(np.float32)
    else:
        y_df = pd.DataFrame(y, index=t_index, columns=valid_syms).stack(future_stack=True).rename("y")
        w_df = pd.DataFrame(w, index=t_index, columns=valid_syms).stack(future_stack=True).rename("w")
        X = X.join(y_df).join(w_df).dropna()
        y_arr = X.pop("y").astype(int).values
        w_arr = X.pop("w").astype(np.float32).values

    tprint(f"Exhaustion X shape: {X.shape}, y shape: {y_arr.shape}")
    tprint(f"Exhaustion class dist: {np.bincount(y_arr)}")

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
    tprint(f"compute_p_exhaustion_at_t: {len(up_syms)} up, {len(dn_syms)} down")

    out_probs = pd.Series(index=syms, dtype=float).fillna(0.0)
    lookback = cfg["exh_train_lookback_hours"]
    if up_syms:
        if models and "up" in models: model_up = models["up"]
        else:
            tprint("Training UP model...")
            X, y, w, _ = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, ts, lookback, valid_syms, trend_filter="up")
            if X is not None and len(y) > 100:
                model_up = ExhaustionModel(C=cfg["exh_C"], l1_ratio=cfg["exh_l1_ratio"])
                model_up.fit(X, y, sample_weight=w)
            else:
                tprint("Not enough data for UP model.")
                model_up = None
        if model_up:
            Xp = _build_pred_X(feats, mkt_gates, cfg, ts, up_syms, feature_key="exh_feature_keys")
            if not Xp.empty:
                probs = model_up.predict_proba(Xp)
                probs = np.clip(probs * 2.0, 0.0, 1.0)
                out_probs.loc[up_syms] = probs
    if dn_syms:
        if models and "down" in models: model_dn = models["down"]
        else:
            tprint("Training DOWN model...")
            X, y, w, _ = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, ts, lookback, valid_syms, trend_filter="down")
            if X is not None and len(y) > 100:
                model_dn = ExhaustionModel(C=cfg["exh_C"], l1_ratio=cfg["exh_l1_ratio"])
                model_dn.fit(X, y, sample_weight=w)
            else:
                tprint("Not enough data for DOWN model.")
                model_dn = None
        if model_dn:
            Xp = _build_pred_X(feats, mkt_gates, cfg, ts, dn_syms, feature_key="exh_feature_keys")
            if not Xp.empty:
                probs = model_dn.predict_proba(Xp)
                out_probs.loc[dn_syms] = probs
    return out_probs.fillna(0.0)

def _build_pred_X(feats, mkt_gates, cfg, ts, syms, feature_key="exh_feature_keys"):
    tprint(f"Entering function: _build_pred_X in training.py")
    t_index = pd.DatetimeIndex([ts], tz="UTC")
    X_parts = []

    keys = cfg.get(feature_key, [])

    for k in keys:
        if k in feats:
            X_parts.append(feats[k].loc[t_index, syms].stack(future_stack=True).rename(k))
    if not X_parts: return pd.DataFrame()

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
    tprint("Generating UP history...")
    X_up, y_up, w_up, _ = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, train_end, train_len, syms, trend_filter="up")
    model_up = None
    if X_up is not None and len(y_up) > 100:
        model_up = ExhaustionModel(C=cfg["exh_C"], l1_ratio=cfg["exh_l1_ratio"])
        model_up.fit(X_up, y_up, sample_weight=w_up)
    tprint("Generating DOWN history...")
    X_dn, y_dn, w_dn, _ = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, train_end, train_len, syms, trend_filter="down")
    model_dn = None
    if X_dn is not None and len(y_dn) > 100:
        model_dn = ExhaustionModel(C=cfg["exh_C"], l1_ratio=cfg["exh_l1_ratio"])
        model_dn.fit(X_dn, y_dn, sample_weight=w_dn)
    t_idx = pd.date_range(train_end, ts_end, freq='h', tz="UTC")
    t_idx = t_idx[t_idx.isin(panel["close"].index)]
    valid_syms = [s for s in syms if s in panel["close"].columns]
    Xp = _build_pred_X_window(feats, mkt_gates, cfg, t_idx, valid_syms, feature_key="exh_feature_keys")
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

def _build_pred_X_window(feats, mkt_gates, cfg, t_idx, syms, feature_key="exh_feature_keys"):
    tprint(f"Entering function: _build_pred_X_window in training.py")
    X_parts = []
    keys = cfg.get(feature_key, [])
    for k in keys:
        if k in feats:
            X_parts.append(feats[k].loc[t_idx, syms].stack(future_stack=True).rename(k))
    Xp = pd.concat(X_parts, axis=1)
    Xp.index.names = ["ts", "symbol"]
    mg = mkt_gates.loc[t_idx, ["mkt_ret24h", "mkt_ret6h", "mkt_trend", "mkt_rv", "G_VOL", "G_TREND"]]
    for col in mg.columns:
        Xp[col] = Xp.index.get_level_values("ts").map(mg[col])
    return Xp.fillna(0)

def build_hourly_training_set_and_weights(panel, feats, mkt_gates, cfg, syms, ts_end, p_exh_hist, H, model_kind, trend_filter=None, feature_key=None):
    tprint(f"Entering function: build_hourly_training_set_and_weights in training.py")
    c = panel["close"]
    idx = c.index

    k_sl = cfg.get("train_k_sl", 2.0)
    k_pt = cfg.get("train_k_pt", 2.0)
    k_tp = cfg.get("train_k_tp", 1.0)

    if "atr_pct" in feats:
        atr_df = feats["atr_pct"]
    else:
        tprint("Warning: atr_pct not found, using default 1% ATR for labeling")
        atr_df = pd.DataFrame(0.01, index=c.index, columns=c.columns)

    tb_labels, tb_returns = compute_trailing_atr_labels(
        panel, atr_df,
        k_sl=k_sl, k_pt=k_pt, k_tp=k_tp,
        horizon_hours=H
    )

    cand_mask = select_trade_candidates_vectorized(panel, feats, pct=cfg["trade_extreme_pct"], metric=cfg["trade_deviation_metric"])
    if cand_mask is None:
        tprint("No candidates mask returned.")
        return None, None, None, None, None
    tprint(f"Candidates found: {cand_mask.sum().sum()}")

    ts_start = ts_end - pd.Timedelta(hours=int(cfg["train_lookback_hours"]))
    valid_window_mask = (cand_mask.index >= ts_start) & (cand_mask.index <= ts_end - pd.Timedelta(hours=H+8))
    subsample_mask = (cand_mask.index.hour % 4 == 0)

    final_mask = cand_mask & pd.Series(valid_window_mask & subsample_mask, index=cand_mask.index).fillna(False)

    valid_ts = final_mask[final_mask.any(axis=1)].index
    tprint(f"Processing {len(valid_ts)} valid timestamps...")
    rows = []

    if feature_key:
        feat_keys = cfg.get(feature_key, [])
    else:
        feat_keys = cfg.get("causal_cols", [])

    for t in valid_ts:
        row_mask = final_mask.loc[t]
        final_candidates = row_mask[row_mask].index.tolist()
        final_candidates = [s for s in final_candidates if s in syms]

        if not final_candidates: continue

        t_entry = t + pd.Timedelta(hours=1)
        if t_entry not in tb_labels.index: continue

        ret_vals = feats["ret24h"].loc[t, final_candidates]

        for sym in final_candidates:
            if sym not in tb_labels.columns: continue

            # TB Outcome
            lbl = tb_labels.loc[t_entry, sym]
            ret = tb_returns.loc[t_entry, sym]

            trend_val = 0.0
            if "trend_pct" in feats: trend_val = feats["trend_pct"].loc[t, sym]
            trend_dir = np.sign(trend_val) if trend_val != 0 else 1.0

            if trend_filter == "up" and trend_dir <= 0: continue
            if trend_filter == "down" and trend_dir > 0: continue

            trade_dir = 1 # Default Long
            if model_kind == "tf":
                if trend_dir > 0: trade_dir = 1
                else: trade_dir = -1
            elif model_kind == "mr":
                if trend_dir > 0: trade_dir = -1
                else: trade_dir = 1

            pnl = ret * trade_dir
            y_bin = 1 if pnl > 0 else 0

            pa = abs(ret_vals[sym])
            w1 = np.log(1 + pa)
            w2 = 1.0
            weight = w1 * w2

            rec = {"symbol": sym, "ts": t, "y_bin": y_bin, "y_ret": pnl, "w": weight}

            # Collect Features
            t_lag = t - pd.Timedelta(hours=1)
            p_val = 0.0
            if t_lag in p_exh_hist.index and sym in p_exh_hist.columns:
                p_val = p_exh_hist.loc[t_lag, sym]
            rec["p_exh_lag1"] = p_val

            missing_features = []
            for k in feat_keys:
                if k == "p_exh_lag1": continue
                if k in feats:
                    rec[k] = feats[k].loc[t, sym]
                else:
                    missing_features.append(k)
            
            if missing_features and len(missing_features) < 5:  # Log if few features missing
                tprint(f"WARNING: Missing features for {sym} at {t}: {missing_features}")

            rec["G_VOL"] = mkt_gates.loc[t, "G_VOL"]
            rec["G_TREND"] = mkt_gates.loc[t, "G_TREND"]
            rows.append(rec)

    if not rows:
        tprint("No rows generated for training set.")
        return None, None, None, None, None, None
    tprint(f"Final training set size: {len(rows)}")
    df = pd.DataFrame(rows).dropna()

    # Build label time ranges for uniqueness weighting
    entry_times = df["ts"].values
    exit_times = entry_times + pd.Timedelta(hours=H)  # H is the horizon
    label_times = build_label_time_ranges(
        pd.DatetimeIndex(entry_times),
        pd.DatetimeIndex(exit_times)
    )
    
    # Compute sample weights with uniqueness (AFML Chapter 4)
    base_weights = df["w"].values
    returns = df["y_ret"].values
    weights = compute_sample_weights_with_uniqueness(
        label_times=label_times,
        returns=returns,
        base_weights=base_weights
    )
    
    tprint(f"Applied uniqueness weighting: mean={weights.mean():.3f}, std={weights.std():.3f}")
    df.drop(columns=["w"], inplace=True)

    df = apply_interaction_toggles(df, feat_keys, ["G_VOL", "G_TREND"], drop_raw=cfg["drop_raw_causal"])
    y_bin = df.pop("y_bin").values.astype(int)
    y_ret = df.pop("y_ret").values.astype(np.float32)

    # Save metadata for return if needed?
    # For now, just drop
    meta_cols = ["ts", "symbol"]
    # We will need these for Spike Model alignment later, so let's keep them in a separate DF if we want.
    # But function signature returns X_out, y...
    # I'll return `df` but with meta cols dropped.

    # IMPORTANT: We need to pass Spike Features to Meta Model.
    # If this function is called for TF/MR, it only gathers TF/MR features.
    # The Meta model needs to gather ITS OWN features for the SAME rows.
    # We should probably have a `build_meta_training_set` helper or modify this to return metadata.

    X_out = df.drop(columns=["ts", "symbol"], errors="ignore").astype(np.float32)
    X_out.index = df.index

    # Return df_meta (ts, symbol) as extra return?
    df_meta = df[meta_cols] if "ts" in df.columns else pd.DataFrame(index=df.index)

    return X_out, y_bin, y_ret, list(X_out.columns), weights, df_meta

def train_spike_anatomy_model(panel, feats, mkt_gates, cfg, syms, ts_end):
    tprint(f"Entering function: train_spike_anatomy_model in training.py")
    cand_mask = select_trade_candidates_vectorized(panel, feats, pct=cfg["trade_extreme_pct"], metric=cfg["trade_deviation_metric"])
    if cand_mask is None: return None

    ts_start = ts_end - pd.Timedelta(hours=int(cfg["train_lookback_hours"]))
    mask = (cand_mask.index >= ts_start) & (cand_mask.index <= ts_end)
    final_mask = cand_mask & pd.Series(mask, index=cand_mask.index).fillna(False)

    valid_ts = final_mask[final_mask.any(axis=1)].index
    tprint(f"Spike Anatomy valid timestamps: {len(valid_ts)}")

    rows = []
    keys = cfg.get("spike_feature_keys", [])

    for t in valid_ts:
        row_mask = final_mask.loc[t]
        cands = row_mask[row_mask].index.tolist()
        cands = [s for s in cands if s in syms]

        for sym in cands:
            rec = {}
            for k in keys:
                if k in feats:
                    rec[k] = feats[k].loc[t, sym]
            rows.append(rec)

    if not rows: return None

    df = pd.DataFrame(rows).dropna()
    tprint(f"Spike Anatomy dataset shape: {df.shape}")
    if df.empty: return None

    tprint("Fitting Spike Anatomy GMM...")
    gmm = GaussianMixture(n_components=4, random_state=42)
    gmm.fit(df)

    return gmm

def select_best_horizon(panel, feats, mkt_gates, cfg, syms, ts, p_exh_hist):
    tprint(f"Entering function: select_best_horizon in training.py")
    directions = ["up", "down"]
    kinds = ["mr", "tf"]
    final_models = {}

    spike_model = train_spike_anatomy_model(panel, feats, mkt_gates, cfg, syms, ts)

    for d in directions:
        final_models[d] = {}
        for k in kinds:
            best_ic = -1.0; best_m = None
            horizons = cfg["label_horizons_hours"]
            feat_key = "tf_feature_keys" if k == "tf" else "mr_feature_keys"

            for H in horizons:
                tprint(f"Selecting {d} {k} H={H}...")
                X, y, y_ret, cols, w, _ = build_hourly_training_set_and_weights(
                    panel, feats, mkt_gates, cfg, syms, ts, p_exh_hist, H, k,
                    trend_filter=d, feature_key=feat_key
                )

                if X is None or len(y) < cfg["min_train_samples"] // 4:
                    tprint(f"Insufficient data for {d} {k} H={H}")
                    continue
                race = ModelRace(kind=k, n_splits=3)
                race.fit(X, y, sample_weight=w, returns=y_ret)
                score = race.metrics.get(race.best_model_name, -1.0)
                tprint(f"Score for {d} {k} H={H}: {score:.4f}")
                if score > best_ic:
                    best_ic = score
                    best_m = {"model": race, "H": H, "feat_cols": cols}
            tprint(f"Best score for {d} {k}: {best_ic:.4f}")
            final_models[d][k] = best_m

    meta_models = {}
    for d in directions:
        mr_conf = final_models[d]["mr"]
        tf_conf = final_models[d]["tf"]
        if not mr_conf or not tf_conf:
            meta_models[d] = None; continue

        H_mr = mr_conf["H"]
        X_mr, _, _, _, _, meta_idx_mr = build_hourly_training_set_and_weights(
            panel, feats, mkt_gates, cfg, syms, ts, p_exh_hist, H_mr, "mr",
            trend_filter=d, feature_key="mr_feature_keys"
        )
        H_tf = tf_conf["H"]
        X_tf, y_tf, y_ret_tf, cols_tf, _, meta_idx_tf = build_hourly_training_set_and_weights(
            panel, feats, mkt_gates, cfg, syms, ts, p_exh_hist, H_tf, "tf",
            trend_filter=d, feature_key="tf_feature_keys"
        )

        # We need to align MR and TF predictions on the same events.
        # But H_mr and H_tf might differ, so "events" might imply different horizons?
        # But "ts" is the signal time. If signal times match, we can blend.
        # We align by (ts, symbol).

        # Create MultiIndex for alignment
        X_mr.index = pd.MultiIndex.from_frame(meta_idx_mr)
        X_tf.index = pd.MultiIndex.from_frame(meta_idx_tf)
        y_tf_indexed = pd.Series(y_ret_tf, index=X_tf.index) # Target for meta is TF return?

        common = X_mr.index.intersection(X_tf.index)
        tprint(f"Meta alignment common size for {d}: {len(common)}")
        if len(common) < 100:
             tprint(f"Insufficient common events for Meta {d}")
             meta_models[d] = None; continue

        X_mr = X_mr.loc[common]
        X_tf = X_tf.loc[common]
        y_meta = y_tf_indexed.loc[common].values

        p_mr = mr_conf["model"].predict(X_mr)
        p_tf = tf_conf["model"].predict(X_tf)

        # Build Meta Features
        # 1. Base Meta Features (from config)
        meta_feat_keys = cfg.get("meta_feature_keys", [])

        # We need to fetch these values for the `common` (ts, symbol) pairs.
        # Efficient way: `_build_pred_X` but for specific list of indices?
        # Or construct a dataframe from `feats` using loop.

        meta_rows = []
        spike_rows = []
        spike_keys = cfg.get("spike_feature_keys", [])

        # Iterate common index to fetch features
        # This is slow if loop. Vectorized fetch preferred.
        # feats[k] is (Time x Symbol).
        # We can stack feats[k] to get (Time, Symbol) -> Value.
        # Then reindex.

        # Prepare Stacked Feats for Meta Keys
        stacked_meta = {}
        for k in meta_feat_keys:
            if k in feats:
                stacked_meta[k] = feats[k].stack()

        df_meta_feats = pd.DataFrame(stacked_meta) # Index (Time, Symbol)
        # Reindex to common
        df_meta_feats = df_meta_feats.reindex(common).fillna(0.0)

        # 2. Spike Probabilities
        if spike_model:
            # We need inputs for spike model for these common events
            stacked_spike = {}
            for k in spike_keys:
                if k in feats:
                    stacked_spike[k] = feats[k].stack()
            df_spike_in = pd.DataFrame(stacked_spike).reindex(common).fillna(0.0)

            if not df_spike_in.empty:
                probs = spike_model.predict_proba(df_spike_in)
                # Add probs as features
                for i in range(probs.shape[1]):
                    df_meta_feats[f"spike_prob_{i}"] = probs[:, i]
            else:
                 # fill 0
                 for i in range(4): df_meta_feats[f"spike_prob_{i}"] = 0.0

        # Train Meta Model
        meta = MetaModel()
        X_meta_final = meta.prepare_meta_features(p_tf, p_mr, df_meta_feats)

        meta.fit(X_meta_final, y_meta)
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

    return {"alpha_models": final_models, "exh_models": exh_models, "meta_models": meta_models, "spike_model": spike_model}

def optimize_risk_params(panel, feats, mkt_gates, cfg, train_syms, ts, p_exh_hist, alpha_models):
    tprint("Entering function: optimize_risk_params in training.py")
    tprint("optimize_risk_params not implemented, returning default config risk params.")
    # Return a minimal risk dict based on config defaults
    return {
        "k_sl": cfg.get("risk_k_sl", 2.0),
        "k_trail_start": cfg.get("risk_k_trail_start", 1.0),
        "k_trail_dist": cfg.get("risk_k_trail_dist", 0.5),
        "granular_risk": {}
    }
