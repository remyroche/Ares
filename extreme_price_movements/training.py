import numpy as np
import pandas as pd
from extreme_price_movements.utils import tprint
from extreme_price_movements.model_race import ModelRace
from extreme_price_movements.meta_model import MetaModel
from extreme_price_movements.exhaustion import ExhaustionModel
from extreme_price_movements.candidates import select_trade_candidates_hourly, select_trade_candidates_vectorized
import extreme_price_movements.fast_funcs as ff
from extreme_price_movements.labeling import compute_trailing_atr_labels, compute_triple_barrier_labels
from extreme_price_movements.sample_weights import build_label_time_ranges, compute_sample_weights_with_uniqueness
from sklearn.mixture import GaussianMixture
from extreme_price_movements.optimise_tpsl_ratio import (
    run_tp_sl_selection_fast,
    calibrate_atr_base_pct,
    compute_vol_z_log_mad,
    PurgedKFold,
)

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

def scaled_atr_pct(
    atr_pct: float,
    z: float,
    atr_base_pct: float,
    *,
    z_max: float = 3.0,
    lo: float = 0.03,
    hi: float = 0.06,
    eps: float = 1e-12,
):
    """
    ATR-informed, shock-scaled, bounded barrier percent.
    Vectorized using NumPy.
    """
    # 1) Shock control
    shock = np.clip(z, 0.0, z_max)

    # 2) Dynamic multiplier 'a' so that:
    #    atr_base_pct * (1 + a*z_max) ≈ hi
    a = (hi / np.maximum(atr_base_pct, eps) - 1.0) / z_max

    # 3) Multiplicative scaling
    raw = atr_pct * (1.0 + a * shock)

    # 4) Enforce cross-asset low/high targets
    return np.clip(raw, lo, hi)

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
        if model_up and model_up.model:
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
        if model_dn and model_dn.model:
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
    if model_up and model_up.model:
        p_up = model_up.predict_proba(Xp)
        p_up = np.clip(p_up * 2.0, 0.0, 1.0)
    p_dn = 0.0
    if model_dn and model_dn.model:
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

def build_hourly_training_set_and_weights(
    panel, feats, mkt_gates, cfg, syms, ts_end, p_exh_hist, H, model_kind,
    trend_filter=None, feature_key=None,
    label_method="atr", fixed_tp=0.05, fixed_sl=0.025, side="long"
):
    tprint(f"Entering function: build_hourly_training_set_and_weights in training.py")
    c = panel["close"]
    idx = c.index

    if label_method == "triple_barrier":
        # Dynamic Barrier Logic
        if "atr_pct" in feats:
            atr_pct = feats["atr_pct"]

            # Compute Rolling Baseline (e.g. 30 days)
            # We need to compute this on the fly if not available.
            # Using simple rolling median for robustness.
            # Assuming aligned index.
            window_size = 24 * 30

            # Since feats are dict of DataFrames, we can process atr_pct
            tprint("Computing dynamic barriers...")
            atr_base = atr_pct.rolling(window_size, min_periods=24).median()
            atr_std = atr_pct.rolling(window_size, min_periods=24).std()

            # Z-score: (atr_pct - base) / std
            # Avoid div/0
            z_score = (atr_pct - atr_base) / (atr_std + 1e-12)

            # Compute Barrier
            # Convert to numpy for vectorization
            b_pct_vals = scaled_atr_pct(
                atr_pct.values,
                z_score.values,
                atr_base.values,
                z_max=3.0,
                lo=0.03,
                hi=0.06
            )

            # Reconstruct DataFrame
            barrier_pct = pd.DataFrame(b_pct_vals, index=atr_pct.index, columns=atr_pct.columns)

            # TP = Barrier
            tp_df = barrier_pct
            # SL = 0.5 * Barrier
            sl_df = 0.5 * barrier_pct

            tprint(f"Labeling: Dynamic Triple Barrier (Mean TP={tp_df.mean().mean():.4f}, Side={side})")

            tb_labels, tb_returns = compute_triple_barrier_labels(
                panel, tp_df, sl_df, H, side=side
            )
        else:
            tprint("Warning: atr_pct not found for dynamic barriers. Falling back to fixed.")
            tprint(f"Labeling: Fixed Triple Barrier (TP={fixed_tp}, SL={fixed_sl}, Side={side})")
            tb_labels, tb_returns = compute_triple_barrier_labels(
                panel, fixed_tp, fixed_sl, H, side=side
            )

    else:
        # Default ATR logic
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

            # If label_method is triple_barrier, `ret` already reflects the PnL of the specific side.
            # If label_method is atr (legacy), `ret` is simulated Long return, so we flip for Short strategies.

            pnl = 0.0
            if label_method == "triple_barrier":
                pnl = ret
                # Note: `ret` from `compute_triple_barrier_labels` for Short is (Entry/Exit - 1).
                # So positive `ret` means profit.
            else:
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

    X_out = df.drop(columns=["ts", "symbol"], errors="ignore").astype(np.float32)
    X_out.index = df.index

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
    return df

def generate_label_datasets(panel, feats, mkt_gates, cfg, syms, ts, p_exh_hist):
    tprint(f"Entering function: generate_label_datasets in training.py")
    datasets = {}

    # 1. Spike Anatomy
    spike_df = train_spike_anatomy_model(panel, feats, mkt_gates, cfg, syms, ts)
    if spike_df is not None:
        datasets["spike_anatomy"] = spike_df

    # 2. Alpha Models (MR/TF)
    trade_sides = ["long", "short"]
    kinds = ["mr", "tf"]
    horizons = cfg["label_horizons_hours"]

    for side in trade_sides:
        for k in kinds:
            # Determine Candidate Filter based on Strategy
            cand_filter = "unknown"
            if side == "long":
                if k == "mr": cand_filter = "worst" # ret < 0
                else: cand_filter = "best"          # ret > 0
            else: # short
                if k == "mr": cand_filter = "best"  # ret > 0
                else: cand_filter = "worst"         # ret < 0

            trend_filter = "up" if cand_filter == "best" else "down"

            feat_key = "tf_feature_keys" if k == "tf" else "mr_feature_keys"

            # Dynamic Barriers will be computed inside build_hourly_training_set_and_weights
            # We don't need to pass fixed params if we use atr-based logic.
            # But we keep fixed defaults just in case.
            fixed_tp = 0.05
            fixed_sl = 0.025

            for H in horizons:
                tprint(f"Generating labels for {side} {k} ({cand_filter}) H={H}...")

                X, y, y_ret, cols, w, meta_idx = build_hourly_training_set_and_weights(
                    panel, feats, mkt_gates, cfg, syms, ts, p_exh_hist, H, k,
                    trend_filter=trend_filter, feature_key=feat_key,
                    label_method="triple_barrier",
                    fixed_tp=fixed_tp, fixed_sl=fixed_sl, side=side
                )

                if X is not None:
                    df_out = X.copy()
                    df_out["__y_bin__"] = y
                    df_out["__y_ret__"] = y_ret
                    df_out["__w__"] = w

                    if meta_idx is not None:
                        df_out["__ts__"] = meta_idx["ts"]
                        df_out["__symbol__"] = meta_idx["symbol"]

                    datasets[f"train_{side}_{k}_{H}"] = df_out

    # 3. Exhaustion Models
    lookback = cfg["exh_train_lookback_hours"]
    directions = ["up", "down"]
    for d in directions:
        tprint(f"Generating exhaustion training set for {d}...")
        X, y, w, cols = build_exhaustion_Xy(panel, feats, mkt_gates, cfg, ts, lookback, syms, trend_filter=d)
        if X is not None:
            df_out = X.copy()
            df_out["__y__"] = y
            df_out["__w__"] = w
            datasets[f"exh_{d}"] = df_out.reset_index()

    return datasets

def train_models_from_artifacts(datasets, cfg):
    tprint(f"Entering function: train_models_from_artifacts in training.py")
    directions = ["up", "down"]
    kinds = ["mr", "tf"]
    final_models = {}

    # 1. Train Spike Model
    spike_model = None
    if "spike_anatomy" in datasets:
        tprint("Training Spike Model...")
        df_spike = datasets["spike_anatomy"]
        gmm = GaussianMixture(n_components=4, covariance_type='diag', random_state=42)
        gmm.fit(df_spike)
        spike_model = gmm

    # 2. Train Alpha Models
    # directions (up/down) replaced by sides (long/short)
    trade_sides = ["long", "short"]
    kinds = ["mr", "tf"]
    final_models = {}

    for side in trade_sides:
        final_models[side] = {}
        for k in kinds:
            best_ic = -1.0; best_m = None
            horizons = cfg["label_horizons_hours"]

            for H in horizons:
                key = f"train_{side}_{k}_{H}"
                if key not in datasets: continue

                df = datasets[key]
                if df.empty or len(df) < cfg["min_train_samples"] // 4:
                    continue

                y = df["__y_bin__"].values.astype(int)
                y_ret = df["__y_ret__"].values.astype(np.float32)
                w = df["__w__"].values.astype(np.float32)

                drop_cols = ["__y_bin__", "__y_ret__", "__w__", "__ts__", "__symbol__"]
                X = df.drop(columns=[c for c in drop_cols if c in df.columns])
                cols = list(X.columns)

                tprint(f"Training {side} {k} H={H} (n={len(X)})...")
                race = ModelRace(kind=k, n_splits=3)
                race.fit(X, y, sample_weight=w, returns=y_ret)
                score = race.metrics.get(race.best_model_name, -1.0)

                if score > best_ic:
                    best_ic = score
                    best_m = {"model": race, "H": H, "feat_cols": cols}

            final_models[side][k] = best_m

    # 3. Train Meta Models
    meta_models = {}
    for side in trade_sides:
        mr_conf = final_models[side]["mr"]
        tf_conf = final_models[side]["tf"]
        if not mr_conf or not tf_conf:
            meta_models[side] = None; continue

        H_mr = mr_conf["H"]
        H_tf = tf_conf["H"]

        key_mr = f"train_{side}_mr_{H_mr}"
        key_tf = f"train_{side}_tf_{H_tf}"

        if key_mr not in datasets or key_tf not in datasets:
            meta_models[side] = None; continue

        df_mr = datasets[key_mr]
        df_tf = datasets[key_tf]

        # Re-index for alignment
        # We need (ts, symbol)
        if "__ts__" not in df_mr.columns or "__symbol__" not in df_mr.columns:
            tprint("Meta alignment failed: missing meta columns")
            meta_models[d] = None; continue

        idx_mr = pd.MultiIndex.from_frame(df_mr[["__ts__", "__symbol__"]])
        idx_tf = pd.MultiIndex.from_frame(df_tf[["__ts__", "__symbol__"]])

        df_mr.index = idx_mr
        df_tf.index = idx_tf

        common = idx_mr.intersection(idx_tf)
        if len(common) < 100:
             meta_models[d] = None; continue

        df_mr_c = df_mr.loc[common]
        df_tf_c = df_tf.loc[common]

        # Prepare X_mr, X_tf
        drop_cols = ["__y_bin__", "__y_ret__", "__w__", "__ts__", "__symbol__"]
        X_mr = df_mr_c.drop(columns=[c for c in drop_cols if c in df_mr_c.columns])
        X_tf = df_tf_c.drop(columns=[c for c in drop_cols if c in df_tf_c.columns])

        y_meta = df_tf_c["__y_ret__"].values # TF returns as target

        p_mr = mr_conf["model"].predict(X_mr)
        p_tf = tf_conf["model"].predict(X_tf)

        # We need Meta Features (that were in feats, but now we only have datasets)
        # The `datasets` only contain features selected for MR/TF.
        # We need a way to pass extra features for Meta.
        # Ideally, `generate_label_datasets` should have included meta features in the saved parquet?
        # OR we save a separate "meta_features" dataset?
        # Or we assume MR/TF features overlap enough? (Probably not enough).
        # FIX: We can assume for now we use what we have, OR we simply skip meta features if not available.
        # The user said "Reuse of the same labeled dataset".
        # Let's try to construct `df_meta_feats` from X_mr + X_tf if possible, or just skip extra meta features?
        # Ideally we should have saved them.
        # But wait, `build_hourly_training_set_and_weights` only saves `feat_keys`.
        # If we want Meta features, we need them in the parquet.
        # For this refactor, I will attempt to proceed without extra meta features lookup from `feats` (since we don't load feats in `train` mode).
        # I'll rely on the model predictions mostly.

        # Spike Probs
        df_meta_feats = pd.DataFrame(index=common)
        if spike_model:
            # We need spike features. Are they in X_mr/X_tf?
            # Likely not.
            # This is a limitation of the current split plan unless we save ALL candidate features.
            # I will skip spike probs injection here for simplicity or assume we can't do it without loading features.
            pass

        meta = MetaModel()
        # Pass empty df_meta_feats if we can't reconstruct it easily without huge file I/O
        X_meta_final = meta.prepare_meta_features(p_tf, p_mr, df_meta_feats)
        meta.fit(X_meta_final, y_meta)
        meta_models[d] = meta

    # 4. Train Exhaustion Models
    exh_models = {}
    for d in directions:
        key = f"exh_{d}"
        if key in datasets:
            df = datasets[key]
            if len(df) > 100:
                y = df["__y__"].values.astype(int)
                w = df["__w__"].values.astype(np.float32)
                # Drop meta/targets
                # In `build_exhaustion_Xy`, X columns are features + G_VOL/G_TREND
                # The reset_index added `ts` `symbol`.
                drop_cols = ["__y__", "__w__", "ts", "symbol"]
                X = df.drop(columns=[c for c in drop_cols if c in df.columns])

                m = ExhaustionModel(C=cfg["exh_C"], l1_ratio=cfg["exh_l1_ratio"])
                m.fit(X, y, sample_weight=w)
                exh_models[d] = m
            else:
                exh_models[d] = None
        else:
            exh_models[d] = None

    return {"alpha_models": final_models, "exh_models": exh_models, "meta_models": meta_models, "spike_model": spike_model}

def select_best_horizon(panel, feats, mkt_gates, cfg, syms, ts, p_exh_hist):
    # DEPRECATED / LEGACY WRAPPER
    # This function is kept for backward compatibility if needed,
    # but strictly we should use the new split.
    # We can implement it by calling the new functions in sequence.
    datasets = generate_label_datasets(panel, feats, mkt_gates, cfg, syms, ts, p_exh_hist)
    return train_models_from_artifacts(datasets, cfg)

def optimize_risk_params(panel, feats, mkt_gates, cfg, train_syms, ts, p_exh_hist, alpha_models):
    tprint("Entering function: optimize_risk_params in training.py (High Throughput Selection)")

    granular_risk = {}

    # 1. Prepare shared price data
    # We need to process all candidates from the training history.
    # We can reuse select_trade_candidates_vectorized logic but we want ALL potential signals.
    # Or just use the signals that were actually generated by the strategy logic?
    # The selection script expects X (features) and prices.

    # Extract panel data
    open_df = panel["open"]
    high_df = panel["high"]
    low_df = panel["low"]
    close_df = panel["close"]

    # ATR stats
    if "atr_pct" not in feats:
        tprint("ATR pct missing, skipping optimization")
        return cfg

    atr_pct_df = feats["atr_pct"]
    window_base = 24 * 30

    tprint("Computing ATR baselines for optimization...")
    atr_base_df = atr_pct_df.rolling(window_base, min_periods=24).median().fillna(method='bfill')
    # Using the fast numpy functions if possible, but pandas is easier for alignment here
    # For Z, we need a robust one.
    # Let's use the one from fast_funcs if available or re-implement simple robust Z.
    atr_std_df = atr_pct_df.rolling(window_base, min_periods=24).std().fillna(method='bfill')
    z_df = (atr_pct_df - atr_base_df) / (atr_std_df + 1e-12)

    # 2. Iterate over strategies (buckets)
    trade_sides = ["long", "short"]
    kinds = ["mr", "tf"]

    # We need to gather events for each bucket.
    # This is non-trivial because `optimize_risk_params` is usually called on a small simulation window.
    # But `run_tp_sl_selection_fast` is designed for training time selection on historical data.
    # Assuming `ts` is the end of training.

    # We will scan the last N days (e.g. 90 or 180) for candidates.
    lookback_days = 90
    ts_start = ts - pd.Timedelta(days=lookback_days)

    # Select candidates
    cand_mask = select_trade_candidates_vectorized(panel, feats, pct=cfg["trade_extreme_pct"], metric=cfg["trade_deviation_metric"])
    if cand_mask is None:
        tprint("No candidates found.")
        return cfg

    mask = (cand_mask.index >= ts_start) & (cand_mask.index <= ts)
    final_mask = cand_mask & pd.Series(mask, index=cand_mask.index).fillna(False)

    valid_ts = final_mask[final_mask.any(axis=1)].index

    # Pre-fetch numpy arrays for the full period (aligned)
    # We need a unified index and columns
    # Let's align everything to `close_df`

    # Flatten everything to 1D arrays of events?
    # run_tp_sl_selection_fast takes 1D arrays for open, high, etc?
    # No, it takes full arrays and event indices.
    # BUT, if we have multiple assets, we can concatenate them?
    # Or run per asset? No, pooled.
    # To pool, we can concatenate all asset time series end-to-end, and adjust event indices.
    # That's one way.
    # Or simply: run_tp_sl_selection_fast expects single instrument arrays.
    # "Assumptions: Single instrument arrays, time-aligned."
    # So we cannot pass the whole panel directly.
    # We must flatten the panel into a long format (Time, Symbol) -> single timeline?
    # No, time-alignment breaks if we concatenate.
    # Effectively, we treat the dataset as one long series of (t, asset) observations.
    # We can concatenate the columns of the panel into one giant 1D array.
    # And compute event indices relative to this giant array.
    # Yes, that works if we insert NaNs or gaps between assets to prevent window crossover.
    # Or just careful indexing.
    # Let's concatenate with a small buffer of NaNs between assets.

    tprint("Flattening panel data for pooled optimization...")

    assets = close_df.columns
    # Collect arrays
    big_open = []
    big_high = []
    big_low = []
    big_close = []
    big_atr = []
    big_z = []
    big_atr_base = []
    big_X = [] # Features

    asset_offsets = {}
    current_offset = 0
    buffer_size = 100 # larger than horizon

    # We need features too.
    # Let's pick a standard set of features for X
    feat_keys = cfg.get("causal_cols", [])
    if not feat_keys:
        # Fallback
        feat_keys = ["trend_pct", "vol_pct", "ret24h"]

    for sym in assets:
        if sym not in atr_pct_df.columns: continue

        # Get data chunks
        o = open_df[sym].values.astype(np.float32)
        h = high_df[sym].values.astype(np.float32)
        l = low_df[sym].values.astype(np.float32)
        c = close_df[sym].values.astype(np.float32)

        a = atr_pct_df[sym].values.astype(np.float32)
        b = atr_base_df[sym].values.astype(np.float32)
        z_v = z_df[sym].values.astype(np.float32)

        # Features
        # Gather into (T, F)
        x_list = []
        for k in feat_keys:
            if k in feats:
                x_list.append(feats[k][sym].values.astype(np.float32))
            else:
                x_list.append(np.zeros(len(c), dtype=np.float32))
        x_arr = np.stack(x_list, axis=1)

        # Append
        big_open.append(o)
        big_high.append(h)
        big_low.append(l)
        big_close.append(c)
        big_atr.append(a)
        big_atr_base.append(b)
        big_z.append(z_v)
        big_X.append(x_arr)

        asset_offsets[sym] = current_offset
        current_offset += len(c) + buffer_size

        # Add buffer
        nan_buf = np.full(buffer_size, np.nan, dtype=np.float32)
        nan_buf_x = np.full((buffer_size, len(feat_keys)), np.nan, dtype=np.float32)

        big_open.append(nan_buf)
        big_high.append(nan_buf)
        big_low.append(nan_buf)
        big_close.append(nan_buf)
        big_atr.append(nan_buf)
        big_atr_base.append(nan_buf)
        big_z.append(nan_buf)
        big_X.append(nan_buf_x)

    # Concatenate
    full_open = np.concatenate(big_open)
    full_high = np.concatenate(big_high)
    full_low = np.concatenate(big_low)
    full_close = np.concatenate(big_close)
    full_atr = np.concatenate(big_atr)
    full_atr_base = np.concatenate(big_atr_base)
    full_z = np.concatenate(big_z)
    full_X = np.concatenate(big_X, axis=0)

    # Now iterate strategies and collect event indices
    for side in trade_sides:
        for k in kinds:
            # Filter logic
            cand_filter = "unknown"
            if side == "long":
                if k == "mr": cand_filter = "worst"
                else: cand_filter = "best"
            else:
                if k == "mr": cand_filter = "best"
                else: cand_filter = "worst"

            trend_filter = "up" if cand_filter == "best" else "down"

            # Collect indices
            indices = []

            # Iterate valid timestamps
            for t in valid_ts:
                # Get candidates at t
                row = final_mask.loc[t]
                cands = row[row].index.intersection(assets)

                # Check trend
                trend_vals = feats["trend_pct"].loc[t, cands]

                for sym in cands:
                    tv = trend_vals[sym]
                    tdir = np.sign(tv) if tv != 0 else 1.0

                    if trend_filter == "up" and tdir <= 0: continue
                    if trend_filter == "down" and tdir > 0: continue

                    # Found a candidate
                    # Get index in full arrays
                    # t is timestamp. We need integer index in the asset array.
                    # Assuming all assets have same index 'idx' (from panel)
                    # We can map t to integer index in panel

                    # idx is sorted? Yes.
                    # Find integer location
                    try:
                        time_idx = idx.get_loc(t)
                    except KeyError:
                        continue

                    flat_idx = asset_offsets[sym] + time_idx
                    indices.append(flat_idx)

            indices = np.array(indices, dtype=np.int32)
            tprint(f"Bucket {side} {k} ({cand_filter}): {len(indices)} events")

            if len(indices) < 50:
                tprint("Not enough events, using defaults.")
                granular_risk[f"risk_{side}_{k}"] = {
                    "k_sl": cfg["risk_k_sl"],
                    "k_trail_start": cfg["risk_k_trail_start"],
                    "k_trail_dist": cfg["risk_k_trail_dist"],
                    # "tp_mult": ... we store optimized params
                }
                continue

            # Run optimization
            # Note: run_tp_sl_selection_fast selects tp_mult and sl_mult for TRIPLE BARRIER.
            # But the system might use Trailing ATR logic at execution time?
            # If we switch to Triple Barrier execution, we use these.
            # If we use Trailing ATR, we map them: k_sl = sl_mult (approx).
            # The prompt implies we want to find optimal "TP:SL ratio" and levels.

            summary = run_tp_sl_selection_fast(
                X=full_X,
                open_=full_open,
                high=full_high,
                low=full_low,
                close=full_close,
                atr_pct=full_atr,
                z=full_z,
                atr_base_pct=full_atr_base,
                event_idx=indices,
                horizon=24, # Fixed horizon for now
                max_events=2000, # Cap for speed
                tp_mult_grid=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0], # Wider grid
                sl_mult_grid=[0.5, 1.0, 1.5, 2.0, 2.5],
                entry_mode="next_open"
            )

            tprint(f"Optimized {side} {k}: TP_mult={summary.final_tp_mult:.2f}, SL_mult={summary.final_sl_mult:.2f}")

            # Store in granular risk
            # We map these to the config keys used by Triple Barrier execution
            granular_risk[f"risk_{side}_{k}"] = {
                "tp_mult": summary.final_tp_mult,
                "sl_mult": summary.final_sl_mult,
                # Fallback keys for legacy
                "k_sl": 2.0,
                "k_trail_start": 2.0,
                "k_trail_dist": 1.0
            }

    # Best params for default (not really used if granular is active)
    best_params = {
        "k_sl": cfg["risk_k_sl"],
        "k_trail_start": cfg["risk_k_trail_start"],
        "k_trail_dist": cfg["risk_k_trail_dist"],
        "granular_risk": granular_risk
    }

    return best_params
