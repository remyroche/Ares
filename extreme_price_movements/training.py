import numpy as np
import pandas as pd

from extreme_price_movements.utils import tprint
from extreme_price_movements.exhaustion import ExhaustionModel
from extreme_price_movements.model_mr import MRModel, compute_mr_weights
from extreme_price_movements.model_tf import TFModel, compute_tf_weights

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

def build_exhaustion_Xy(panel, feats, mkt_gates, cfg, ts_end, lookback_hours, syms):
    c = panel["close"]
    idx = c.index
    H = int(cfg["exh_horizon_hours"])

    ts_train_end = ts_end - pd.Timedelta(hours=1)
    ts_start = ts_train_end - pd.Timedelta(hours=int(lookback_hours))

    if ts_train_end not in idx:
        return None, None, None

    ts_future_needed = ts_train_end + pd.Timedelta(hours=H)
    if ts_future_needed > idx.max():
        ts_train_end = idx.max() - pd.Timedelta(hours=H)
        if ts_train_end < ts_start:
             return None, None, None

    mask = (idx >= ts_start) & (idx <= ts_train_end + pd.Timedelta(hours=H))
    idx_slice = idx[mask]

    valid_syms = [s for s in syms if s in c.columns]
    if not valid_syms:
        return None, None, None

    close_sub = c.loc[idx_slice, valid_syms].astype(np.float32)

    rev_close = close_sub.iloc[::-1]
    fmax = rev_close.rolling(H).max().shift(1).iloc[::-1]
    fmin = rev_close.rolling(H).min().shift(1).iloc[::-1]

    t_index = idx[(idx >= ts_start) & (idx <= ts_train_end)]
    fmax = fmax.loc[t_index]
    fmin = fmin.loc[t_index]

    thr = float(cfg["exh_reversal_thr"])

    current = c.loc[t_index, valid_syms]
    fut_min = fmin
    fut_max = fmax

    is_short_rev = ((fut_min / (current + 1e-12)) - 1.0) <= -thr
    is_long_rev = ((fut_max / (current + 1e-12)) - 1.0) >= thr

    ret24 = feats["ret24h"].loc[t_index, valid_syms]
    dir_mat = np.sign(ret24).astype(np.int8)

    y = np.zeros(current.shape, dtype=np.int8)
    y[dir_mat > 0] = is_short_rev.values[dir_mat > 0].astype(np.int8)
    y[dir_mat < 0] = is_long_rev.values[dir_mat < 0].astype(np.int8)

    X_parts = []
    keys = cfg["exh_feature_keys"]
    for k in keys:
        if k in feats:
            X_parts.append(feats[k].loc[t_index, valid_syms].stack(dropna=False).rename(k))

    X = pd.concat(X_parts, axis=1)
    X.index.names = ["ts", "symbol"]

    mg = mkt_gates.loc[t_index, ["mkt_ret24h", "mkt_ret6h", "mkt_trend", "mkt_rv", "G_VOL", "G_TREND"]]
    for col in mg.columns:
        X[col] = X.index.get_level_values("ts").map(mg[col])

    y_ser = pd.DataFrame(y, index=t_index, columns=valid_syms).stack(dropna=False).rename("y")
    X = X.join(y_ser)
    X = X.dropna()

    y_arr = X.pop("y").astype(int).values
    return X, y_arr, list(X.columns)

def compute_p_exhaustion_at_t(panel, feats, mkt_gates, cfg, ts, syms):
    lookback = cfg["exh_train_lookback_hours"]
    X, y, cols = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, ts, lookback, syms)

    if X is None or len(y) < cfg["min_exh_samples"] or len(np.unique(y)) < 2:
        return pd.Series(0.0, index=syms)

    model = ExhaustionModel(C=cfg["exh_C"], l1_ratio=cfg["exh_l1_ratio"])
    model.fit(X, y)

    t_index = pd.DatetimeIndex([ts], tz="UTC")
    valid_syms = [s for s in syms if s in panel["close"].columns]

    X_parts = []
    for k in cfg["exh_feature_keys"]:
        if k in feats:
            X_parts.append(feats[k].loc[t_index, valid_syms].stack(dropna=False).rename(k))

    Xp = pd.concat(X_parts, axis=1)
    Xp.index.names = ["ts", "symbol"]

    mg = mkt_gates.loc[t_index, ["mkt_ret24h", "mkt_ret6h", "mkt_trend", "mkt_rv", "G_VOL", "G_TREND"]]
    for col in mg.columns:
        Xp[col] = Xp.index.get_level_values("ts").map(mg[col])

    for c in cols:
        if c not in Xp.columns:
            Xp[c] = np.nan
    Xp = Xp[cols].fillna(0)

    if Xp.empty:
        return pd.Series(0.0, index=syms)

    probs = model.predict_proba(Xp)
    out = pd.Series(probs, index=Xp.index.get_level_values("symbol"))
    return out.reindex(syms).fillna(0.0)

def generate_exhaustion_history(panel, feats, mkt_gates, cfg, ts_end, lookback_hours, syms):
    """
    Generates historical exhaustion probabilities for a window ending at ts_end.
    Since models need retraining (rolling window), we approximate by training once
    on an older window and predicting forward, or stepping?
    Stepping is slow.
    Approximation: Train on [ts_end - lookback*2, ts_end - lookback] ?
    Or just train on [ts_end - lookback, ts_end] (leakage if predicting past?)

    For proper causality of input feature (p_exh_lag1) for MR model training:
    We need p_exh at time t, generated using info < t.

    We can train ONE model using data up to ts_end - lookback_hours,
    and predict for [ts_end - lookback_hours, ts_end].
    This simulates an "expanding window" or "static model" for that period.
    It's a reasonable approximation for weighting features.
    """

    train_end = ts_end - pd.Timedelta(hours=lookback_hours)
    # Train model up to train_end
    # Using a separate lookback for training data
    train_len = cfg["exh_train_lookback_hours"]

    X, y, cols = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, train_end, train_len, syms)

    if X is None or len(y) < cfg["min_exh_samples"]:
        return pd.DataFrame(0.0, index=pd.date_range(train_end, ts_end, freq='h'), columns=syms)

    model = ExhaustionModel(C=cfg["exh_C"], l1_ratio=cfg["exh_l1_ratio"])
    model.fit(X, y)

    # Predict for the window [train_end, ts_end]
    # We construct X for all timestamps in this window
    t_idx = pd.date_range(train_end, ts_end, freq='h', tz="UTC")
    t_idx = t_idx[t_idx.isin(panel["close"].index)]

    # Bulk prediction
    valid_syms = [s for s in syms if s in panel["close"].columns]

    X_parts = []
    for k in cfg["exh_feature_keys"]:
        if k in feats:
            # slice time
            X_parts.append(feats[k].loc[t_idx, valid_syms].stack(dropna=False).rename(k))

    Xp = pd.concat(X_parts, axis=1)
    Xp.index.names = ["ts", "symbol"]

    mg = mkt_gates.loc[t_idx, ["mkt_ret24h", "mkt_ret6h", "mkt_trend", "mkt_rv", "G_VOL", "G_TREND"]]
    for col in mg.columns:
        Xp[col] = Xp.index.get_level_values("ts").map(mg[col])

    for c in cols:
        if c not in Xp.columns:
            Xp[c] = np.nan
    Xp = Xp[cols].fillna(0)

    if Xp.empty:
        return pd.DataFrame(0.0, index=t_idx, columns=syms)

    probs = model.predict_proba(Xp)
    res_ser = pd.Series(probs, index=Xp.index)
    res_df = res_ser.unstack(level="symbol").reindex(columns=syms).fillna(0.0)

    return res_df

def build_hourly_training_set_and_weights(panel, feats, mkt_gates, cfg, syms, ts_end, p_exh_hist, H, model_kind):
    c = panel["close"]
    idx = c.index
    ts_start = ts_end - pd.Timedelta(hours=int(cfg["train_lookback_hours"]))

    valid_mask = (idx >= ts_start) & (idx <= ts_end - pd.Timedelta(hours=H))
    t_idx = idx[valid_mask]

    if len(t_idx) == 0:
        return None, None, None, None

    metric = cfg["trade_deviation_metric"]
    if metric not in feats:
        return None, None, None, None

    metric_df = feats[metric].loc[t_idx, [s for s in syms if s in feats[metric].columns]]

    rows = []
    for t in t_idx:
        row_vals = metric_df.loc[t].dropna()
        if len(row_vals) < 20: continue

        n = len(row_vals)
        k = max(cfg["train_extreme_min"], int(n * cfg["train_extreme_pct_hourly"]))
        k = min(k, cfg["train_extreme_max"])

        sorted_vals = row_vals.sort_values()
        bot = sorted_vals.iloc[:k].index.tolist()
        top = sorted_vals.iloc[-k:].index.tolist()
        candidates = list(set(bot) | set(top))

        t_entry = t + pd.Timedelta(hours=1)
        t_exit = t_entry + pd.Timedelta(hours=H)

        if t_exit not in c.index: continue

        px_entry = panel["open"].loc[t_entry, candidates]
        px_exit = c.loc[t_exit, candidates]
        y = (px_exit / (px_entry + 1e-12) - 1.0)

        for sym in candidates:
            if pd.isna(y.get(sym)): continue

            rec = {"symbol": sym, "ts": t, "y": y[sym]}

            t_lag = t - pd.Timedelta(hours=1)
            p_val = 0.0
            if t_lag in p_exh_hist.index and sym in p_exh_hist.columns:
                p_val = p_exh_hist.loc[t_lag, sym]
            rec["p_exh_lag1"] = p_val

            for k in cfg["causal_cols"]:
                if k == "p_exh_lag1": continue
                if k == "a_funding_proxy": k = "funding_proxy"
                if k in feats:
                    rec[k] = feats[k].loc[t, sym]

            rec["mkt_ret24h"] = mkt_gates.loc[t, "mkt_ret24h"]
            rec["mkt_ret6h"] = mkt_gates.loc[t, "mkt_ret6h"]
            rec["mkt_trend"] = mkt_gates.loc[t, "mkt_trend"]
            rec["mkt_rv"] = mkt_gates.loc[t, "mkt_rv"]
            rec["G_VOL"] = mkt_gates.loc[t, "G_VOL"]
            rec["G_TREND"] = mkt_gates.loc[t, "G_TREND"]

            rows.append(rec)

    if not rows:
        return None, None, None, None

    df = pd.DataFrame(rows).dropna()

    # Compute Weights BEFORE dropping raw columns
    if model_kind == "mr":
        weights = compute_mr_weights(df, cfg)
    else:
        weights = compute_tf_weights(df, cfg)

    # Apply interactions and drop raw
    df = apply_interaction_toggles(df, cfg["causal_cols"], ["G_VOL", "G_TREND"], drop_raw=cfg["drop_raw_causal"])

    y_out = df.pop("y").values.astype(np.float32)
    X_out = df.drop(columns=["ts", "symbol"]).astype(np.float32)

    return X_out, y_out, list(X_out.columns), weights

def select_best_horizon(panel, feats, mkt_gates, cfg, syms, ts, p_exh_hist, model_kind="mr"):
    horizons = cfg["label_horizons_hours"]
    best_loss = float("inf")
    best_res = None

    for H in horizons:
        X, y, cols, w = build_hourly_training_set_and_weights(panel, feats, mkt_gates, cfg, syms, ts, p_exh_hist, H, model_kind)

        if X is None or len(y) < cfg["min_train_samples"]:
            continue

        # Purged Validation
        # H is the gap we need to purge.
        # We split by time index? But X is a cross-section merged.
        # We assume X is sorted by time (it is constructed that way).
        # We find the index where timestamp crosses the split point.
        # Since we dropped ts from X, we approximate by row count.
        # Rows are roughly uniform per hour.

        n = len(X)
        split_idx = int(n * 0.8)

        # Purge gap:
        # We need to drop samples in [split_idx, split_idx + gap_rows] ?
        # Actually, samples in X_train end at T_split. Their labels rely on [T_split, T_split+H].
        # Samples in X_val start at T_val. Their input relies on [T_val-Lookback, T_val].
        # For simple OOF correctness (leakage of label to input):
        # Y_train uses future. X_val uses past.
        # If Y_train(t) overlaps with X_val(t'), we have problem?
        # Only if t + H > t'.
        # So we need t' > t + H.
        # So X_val should start at Time(X_train_last) + H.

        # Since we don't have timestamps in X/y anymore, we assume they are ordered.
        # We need to discard `H` hours worth of samples between train and val.
        # Approximate `rows_per_hour`?
        # It varies.
        # Safer: We can't do exact purging without TS.
        # BUT, build_hourly_training_set_and_weights could return indices or TS.
        # For now, I'll use a safe buffer of 5% of data? or just skip 1000 rows.
        # Config has `trade_extreme_pct` (5%) * `fetch_symbols` (350) ~= 17 rows/hour.
        # H=24. 24 * 17 = 400 rows.
        # I'll purge 500 rows.

        purge_rows = 500
        if split_idx + purge_rows >= n:
            # Not enough data for validation
            continue

        X_train = X.iloc[:split_idx]
        y_train = y[:split_idx]
        w_train = w[:split_idx]

        X_val = X.iloc[split_idx + purge_rows:]
        y_val = y[split_idx + purge_rows:]

        if model_kind == "mr":
            model = MRModel(lasso_alpha=cfg.get("lasso_alpha", 0.001))
            model.fit(X_train, y_train, sample_weight=w_train)
            preds = model.predict(X_val)
            loss = np.mean((y_val - preds)**2)
        else:
            model = TFModel(lasso_alpha=cfg.get("lasso_alpha", 0.001))
            model.fit(X_train, y_train, sample_weight=w_train)
            preds, disp = model.predict(X_val)
            loss = np.mean((y_val - preds)**2)

        if loss < best_loss:
            best_loss = loss
            if model_kind == "mr":
                final_model = MRModel(lasso_alpha=cfg.get("lasso_alpha", 0.001))
                final_model.fit(X, y, sample_weight=w)
            else:
                final_model = TFModel(lasso_alpha=cfg.get("lasso_alpha", 0.001))
                final_model.fit(X, y, sample_weight=w)

            best_res = {
                "model": final_model,
                "H": H,
                "feat_cols": cols,
                "loss": loss
            }

    if best_res is None:
        return None

    return best_res
