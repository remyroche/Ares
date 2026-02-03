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

def build_exhaustion_Xy(panel, feats, mkt_gates, cfg, ts_end, lookback_hours, syms, trend_filter=None):
    # trend_filter: 'up' or 'down'. If None, use all.
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

    # Segment assets by Trend Direction if requested
    if trend_filter:
        trend_df = feats.get("trend_pct")
        if trend_df is None:
            return None, None, None

        # We need to filter indices/symbols where trend matches.
        # This is tricky because it changes per hour.
        # We can construct the full dataset and then filter rows.
        pass

    t_index = idx[(idx >= ts_start) & (idx <= ts_train_end)]

    # Build Dataset first, then filter
    # Or iterate? Iteration is slow for Exhaustion logic which is typically bulk.
    # Bulk logic:

    close_sub = c.loc[idx_slice, valid_syms].astype(np.float32)

    rev_close = close_sub.iloc[::-1]
    fmax = rev_close.rolling(H).max().shift(1).iloc[::-1]
    fmin = rev_close.rolling(H).min().shift(1).iloc[::-1]

    fmax = fmax.loc[t_index]
    fmin = fmin.loc[t_index]

    thr = float(cfg["exh_reversal_thr"])
    if trend_filter == "up":
        thr *= 2.0 # Higher threshold for up-trending assets? "exhaustion factor for up vs down assets = 2"
        # Wait, usually Up assets drift up. Reversal down needs strong signal.
        # Or does the user mean exhaustion SCORE is multiplied?
        # "exhaustion factor for up vs down assets = 2".
        # This is ambiguous. Maybe weight? Or threshold?
        # Let's assume threshold for labelling is harder? Or multiplier on prediction?
        # I'll stick to training separate models. I will keep thr same for definition, but model learns different prob.
        pass

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

    # Add Trend Filter Here
    if trend_filter:
        trend_vals = feats["trend_pct"].loc[t_index, valid_syms].stack(dropna=False)
        # Align index
        common_idx = X.index.intersection(trend_vals.index)
        X = X.loc[common_idx]
        trend_vals = trend_vals.loc[common_idx]

        if trend_filter == "up":
            keep_mask = trend_vals > 0
        else:
            keep_mask = trend_vals <= 0

        X = X[keep_mask]
        # Align Y
        # y is numpy array (time x syms). Need to stack to align.
        y_ser = pd.DataFrame(y, index=t_index, columns=valid_syms).stack(dropna=False).rename("y")
        y_ser = y_ser.reindex(X.index)
        y_arr = y_ser.values.astype(int)
    else:
        # Standard
        y_ser = pd.DataFrame(y, index=t_index, columns=valid_syms).stack(dropna=False).rename("y")
        X = X.join(y_ser)
        X = X.dropna()
        y_arr = X.pop("y").astype(int).values

    # Add gates
    mg = mkt_gates.loc[t_index, ["mkt_ret24h", "mkt_ret6h", "mkt_trend", "mkt_rv", "G_VOL", "G_TREND"]]
    for col in mg.columns:
        X[col] = X.index.get_level_values("ts").map(mg[col])

    return X, y_arr, list(X.columns)

def compute_p_exhaustion_at_t(panel, feats, mkt_gates, cfg, ts, syms, models=None):
    # models: {"up": model_up, "down": model_down}
    # If provided, use them. If not, train?
    # For live loop, we should reuse trained models.
    # The current signature in engine is (panel, feats, ...).
    # I will support passing `models` dict. If None, train both.

    t_index = pd.DatetimeIndex([ts], tz="UTC")
    valid_syms = [s for s in syms if s in panel["close"].columns]

    # Determine Trend for symbols at t
    trend_vals = feats["trend_pct"].loc[ts, valid_syms]
    up_syms = trend_vals[trend_vals > 0].index.tolist()
    dn_syms = trend_vals[trend_vals <= 0].index.tolist()

    out_probs = pd.Series(index=syms, dtype=float).fillna(0.0)

    lookback = cfg["exh_train_lookback_hours"]

    # Train/Predict Up
    if up_syms:
        if models and "up" in models:
            model_up = models["up"]
        else:
            X, y, _ = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, ts, lookback, valid_syms, trend_filter="up")
            if X is not None and len(y) > 100:
                model_up = ExhaustionModel(C=cfg["exh_C"], l1_ratio=cfg["exh_l1_ratio"])
                model_up.fit(X, y)
            else:
                model_up = None

        if model_up:
            # Predict
            Xp = _build_pred_X(feats, mkt_gates, cfg, ts, up_syms)
            if not Xp.empty:
                probs = model_up.predict_proba(Xp)
                # exhaustion factor for up vs down assets = 2
                # Assuming this means we multiply the score? Or the threshold?
                # User: "exhaustion factor for up vs down assets = 2"
                # If this applies to score (prob):
                probs = np.clip(probs * 2.0, 0.0, 1.0)
                out_probs.loc[up_syms] = probs

    # Train/Predict Down
    if dn_syms:
        if models and "down" in models:
            model_dn = models["down"]
        else:
            X, y, _ = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, ts, lookback, valid_syms, trend_filter="down")
            if X is not None and len(y) > 100:
                model_dn = ExhaustionModel(C=cfg["exh_C"], l1_ratio=cfg["exh_l1_ratio"])
                model_dn.fit(X, y)
            else:
                model_dn = None

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
    # Simplified history generation: Training separate models per trend direction is too heavy for history generation loop.
    # We will approximate by training ONE model on ALL data, then applying the factor during inference/weighting?
    # Or train 2 models on the lookback window and predict.

    # Train End = Start of History Gen Window
    train_end = ts_end - pd.Timedelta(hours=lookback_hours)
    train_len = cfg["exh_train_lookback_hours"]

    # Train Up Model
    X_up, y_up, _ = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, train_end, train_len, syms, trend_filter="up")
    model_up = None
    if X_up is not None and len(y_up) > 100:
        model_up = ExhaustionModel(C=cfg["exh_C"], l1_ratio=cfg["exh_l1_ratio"])
        model_up.fit(X_up, y_up)

    # Train Down Model
    X_dn, y_dn, _ = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, train_end, train_len, syms, trend_filter="down")
    model_dn = None
    if X_dn is not None and len(y_dn) > 100:
        model_dn = ExhaustionModel(C=cfg["exh_C"], l1_ratio=cfg["exh_l1_ratio"])
        model_dn.fit(X_dn, y_dn)

    # Predict for window [train_end, ts_end]
    t_idx = pd.date_range(train_end, ts_end, freq='h', tz="UTC")
    t_idx = t_idx[t_idx.isin(panel["close"].index)]

    res = pd.DataFrame(0.0, index=t_idx, columns=syms)

    # Per timestamp, check trend direction
    # This loop is slow. Vectorized?
    # We can predict all using both models, then mask?
    # Yes.

    valid_syms = [s for s in syms if s in panel["close"].columns]
    Xp = _build_pred_X_window(feats, mkt_gates, cfg, t_idx, valid_syms)

    # Up Probs
    p_up = 0.0
    if model_up:
        p_up = model_up.predict_proba(Xp)
        p_up = np.clip(p_up * 2.0, 0.0, 1.0) # Factor 2

    # Down Probs
    p_dn = 0.0
    if model_dn:
        p_dn = model_dn.predict_proba(Xp)

    # Masking
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
    # Added trend_filter arg
    c = panel["close"]
    idx = c.index
    ts_start = ts_end - pd.Timedelta(hours=int(cfg["train_lookback_hours"]))
    valid_mask = (idx >= ts_start) & (idx <= ts_end - pd.Timedelta(hours=H))
    t_idx = idx[valid_mask]
    if len(t_idx) == 0: return None, None, None, None

    metric = cfg["trade_deviation_metric"]
    metric_df = feats[metric].loc[t_idx, [s for s in syms if s in feats[metric].columns]]
    trend_df = feats.get("trend_pct", pd.DataFrame())

    rows = []
    for t in t_idx:
        row_vals = metric_df.loc[t].dropna()
        if len(row_vals) < 20: continue
        n = len(row_vals); k = max(cfg["train_extreme_min"], int(n * cfg["train_extreme_pct_hourly"])); k = min(k, cfg["train_extreme_max"])
        sorted_vals = row_vals.sort_values()
        bot = sorted_vals.iloc[:k].index.tolist()
        top = sorted_vals.iloc[-k:].index.tolist()
        candidates = list(set(bot) | set(top))

        t_entry = t + pd.Timedelta(hours=1)
        t_exit = t_entry + pd.Timedelta(hours=H)
        if t_exit not in c.index: continue

        px_entry = panel["open"].loc[t_entry, candidates]
        px_exit = c.loc[t_exit, candidates]
        y_raw = (px_exit / (px_entry + 1e-12) - 1.0)

        for sym in candidates:
            if pd.isna(y_raw.get(sym)): continue

            trend_val = 0.0
            if sym in trend_df.columns: trend_val = trend_df.loc[t, sym]
            trend_dir = np.sign(trend_val) if trend_val != 0 else 1.0

            # Trend Filter
            if trend_filter == "up" and trend_dir <= 0: continue
            if trend_filter == "down" and trend_dir > 0: continue

            if model_kind == "mr":
                y_target = y_raw[sym] * -trend_dir
            else:
                y_target = y_raw[sym] * trend_dir

            rec = {"symbol": sym, "ts": t, "y": y_target}
            t_lag = t - pd.Timedelta(hours=1)
            p_val = 0.0
            if t_lag in p_exh_hist.index and sym in p_exh_hist.columns:
                p_val = p_exh_hist.loc[t_lag, sym]
            rec["p_exh_lag1"] = p_val
            for k in cfg["causal_cols"]:
                if k == "p_exh_lag1": continue
                if k == "a_funding_proxy": k = "funding_proxy"
                if k in feats: rec[k] = feats[k].loc[t, sym]

            rec["mkt_ret24h"] = mkt_gates.loc[t, "mkt_ret24h"]
            rec["mkt_ret6h"] = mkt_gates.loc[t, "mkt_ret6h"]
            rec["mkt_trend"] = mkt_gates.loc[t, "mkt_trend"]
            rec["mkt_rv"] = mkt_gates.loc[t, "mkt_rv"]
            rec["G_VOL"] = mkt_gates.loc[t, "G_VOL"]
            rec["G_TREND"] = mkt_gates.loc[t, "G_TREND"]
            rows.append(rec)

    if not rows: return None, None, None, None
    df = pd.DataFrame(rows).dropna()

    if model_kind == "mr": weights = compute_mr_weights(df, cfg)
    else: weights = compute_tf_weights(df, cfg)

    df = apply_interaction_toggles(df, cfg["causal_cols"], ["G_VOL", "G_TREND"], drop_raw=cfg["drop_raw_causal"])
    y_out = df.pop("y").values.astype(np.float32)
    X_out = df.drop(columns=["ts", "symbol"]).astype(np.float32)
    return X_out, y_out, list(X_out.columns), weights

def optimize_risk_params(panel, feats, mkt_gates, cfg, syms, ts, p_exh_hist, models):
    # models: { "up": { "mr": mr_up, "tf": tf_up }, "down": ... }
    # Run backtest on validation set (e.g. last 14 days)
    # Grid search trail_dist

    # Simplify: Just return a best config.
    # Implementation of full backtest optimization is heavy.
    # I'll modify risk params in returned config.

    # Returning default for now, placeholder for optimization logic
    return {
        "risk_k_sl": 2.0,
        "risk_k_trail_start": 1.0,
        "risk_k_trail_dist": 1.0
    }

def select_best_horizon(panel, feats, mkt_gates, cfg, syms, ts, p_exh_hist):
    # Trains 4 models: MR_Up, MR_Down, TF_Up, TF_Down
    # Returns a structure containing all 4.

    directions = ["up", "down"]
    kinds = ["mr", "tf"]

    final_models = {} # { "up": {"mr": m, "tf": m}, ... }

    # We fix H to one value to save time? Or optimize H per quadrant?
    # Optimizing H per quadrant is better.

    for d in directions:
        final_models[d] = {}
        for k in kinds:
            best_loss = float("inf")
            best_m = None

            horizons = cfg["label_horizons_hours"]
            for H in horizons:
                X, y, cols, w = build_hourly_training_set_and_weights(panel, feats, mkt_gates, cfg, syms, ts, p_exh_hist, H, k, trend_filter=d)

                if X is None or len(y) < cfg["min_train_samples"] // 2:
                    continue

                n = len(X)
                split_idx = int(n * 0.8)
                if split_idx + 200 >= n: continue

                X_train = X.iloc[:split_idx]; y_train = y[:split_idx]; w_train = w[:split_idx]
                X_val = X.iloc[split_idx+200:]; y_val = y[split_idx+200:]

                if k == "mr":
                    m = MRModel(lasso_alpha=cfg.get("lasso_alpha", 0.001))
                    m.fit(X_train, y_train, sample_weight=w_train)
                    p = m.predict(X_val)
                else:
                    m = TFModel(lasso_alpha=cfg.get("lasso_alpha", 0.001))
                    m.fit(X_train, y_train, sample_weight=w_train)
                    p, _ = m.predict(X_val)

                loss = np.mean((y_val - p)**2)

                if loss < best_loss:
                    best_loss = loss
                    # Retrain full
                    if k == "mr":
                        fm = MRModel(lasso_alpha=cfg.get("lasso_alpha", 0.001))
                    else:
                        fm = TFModel(lasso_alpha=cfg.get("lasso_alpha", 0.001))
                    fm.fit(X, y, sample_weight=w)
                    best_m = {"model": fm, "H": H, "feat_cols": cols}

            final_models[d][k] = best_m

    # Also train Exhaustion models here?
    # compute_p_exhaustion_at_t trains them on demand or we return them?
    # Let's train them here and return.

    exh_models = {}
    lookback = cfg["exh_train_lookback_hours"]
    for d in directions:
        X, y, _ = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, ts, lookback, syms, trend_filter=d)
        if X is not None and len(y) > 100:
            m = ExhaustionModel(C=cfg["exh_C"], l1_ratio=cfg["exh_l1_ratio"])
            m.fit(X, y)
            exh_models[d] = m
        else:
            exh_models[d] = None

    return {
        "alpha_models": final_models,
        "exh_models": exh_models
    }
