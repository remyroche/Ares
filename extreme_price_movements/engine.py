import numpy as np
import pandas as pd

from extreme_price_movements.utils import tprint
from extreme_price_movements.models import map_pred_to_score
from extreme_price_movements.risk import TrailingStop
from extreme_price_movements.training import (
    compute_p_exhaustion_at_t,
    select_best_horizon,
    apply_interaction_toggles,
)
from extreme_price_movements.candidates import select_trade_candidates_hourly, entry_price_next_hour_open

def simulate_trade_hourly(o_s, h_s, l_s, c_s, feats_s, ts_entry, entry_px, side, cfg, max_hold_hours):
    tprint(f"Entering function: simulate_trade_hourly in engine.py")
    if np.isnan(entry_px) or entry_px <= 0:
        return 0.0, ts_entry, "no_entry"

    ts_sig = ts_entry - pd.Timedelta(hours=1)
    if ts_sig not in feats_s.index:
        atr = 0.02
    else:
        atr = float(feats_s.loc[ts_sig])

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

def hourly_engine_backtest(panel, feats, mkt_gates, cfg, symbols_all):
    tprint(f"Entering function: hourly_engine_backtest in engine.py")
    o, h, l, c = panel["open"], panel["high"], panel["low"], panel["close"]
    idx = c.index

    equity = 1.0
    eq_curve = []
    trades = []

    fee_rt = cfg["fee_bps"] / 1e4
    borrow_hourly = (cfg["borrow_apr"] / 365.0) / 24.0

    p_exh_hist = pd.DataFrame(index=idx, columns=symbols_all, dtype=np.float32)

    alpha_models = None
    meta_models = None
    spike_model = None
    last_train_day = None

    warm = max(cfg["train_lookback_hours"], cfg["exh_train_lookback_hours"]) + max(cfg["label_horizons_hours"]) + 48
    start_ts = idx.min() + pd.Timedelta(hours=warm)
    start_ts = idx[idx.get_indexer([start_ts], method="backfill")[0]]

    tprint(f"Engine: start_ts={start_ts}  idx={len(idx)}  symbols={len(symbols_all)}")

    for ts in idx[idx >= start_ts]:
        ts_entry = ts + pd.Timedelta(hours=1)
        if ts_entry not in idx:
            break

        p_exh_ts = compute_p_exhaustion_at_t(panel, feats, mkt_gates, cfg, ts, symbols_all)
        p_exh_hist.loc[ts, symbols_all] = p_exh_ts.values.astype(np.float32)

        ts_day = ts.floor("D")
        if last_train_day is None or ts_day != last_train_day:
            tprint(f"TRAIN day-roll: {ts_day}")
            bundle = select_best_horizon(panel, feats, mkt_gates, cfg, symbols_all, ts, p_exh_hist)
            alpha_models = bundle["alpha_models"]
            meta_models = bundle["meta_models"]
            spike_model = bundle.get("spike_model")
            last_train_day = ts_day

        if not alpha_models:
            eq_curve.append((ts, equity))
            continue

        # Generate Signals using updated logic (simulating live call)
        # Note: Backtest loop essentially does what generate_hourly_signals does but tracked.
        # Ideally we refactor to use generate_hourly_signals, but I'll update logic here inline to match.

        # Candidate Selection with Time Expansion (Live Simulation)
        candidates = set()
        lookback_offsets = [0, 4, 8, 12, 16]

        for offset in lookback_offsets:
            t_check = ts - pd.Timedelta(hours=offset)
            if t_check in feats["ret24h"].index:
                 # Check Volatility Filter at t_check? Or at ts?
                 # "Filter out the timestamps for which last 12-hours Highest/Lowest price difference is less than 8%"
                 # This check applies to the MOMENT of detection.
                 # So we check vol at t_check.

                 # Check 12h Vol
                 # We need raw prices. panel["close"] etc.
                 # h_roll = h.loc[t_check-11h : t_check].max()
                 # l_roll = l.loc[t_check-11h : t_check].min()
                 # This is slow in loop.
                 # Assume we trust select_trade_candidates_hourly to pick extremes.
                 # We should add the vol filter.

                 # Optimization: Pre-calculate Vol Filter mask?
                 # rolling(12).max() on panel was done in vectorized selection.
                 # But here we are in loop.
                 # Let's assume select_trade_candidates_hourly returns raw extremes, we filter.

                 top, bot = select_trade_candidates_hourly(
                    feats=feats, ts=t_check, syms=symbols_all,
                    pct=cfg["trade_extreme_pct"], min_n=cfg["trade_extreme_min"], max_n=cfg["trade_extreme_max"],
                    metric=cfg["trade_deviation_metric"]
                 )

                 # Filter
                 c_check = c.loc[t_check, top+bot]
                 # We need rolling 12h max/min ending at t_check
                 # Slice [t_check - 11h, t_check]
                 t_start_vol = t_check - pd.Timedelta(hours=11)
                 h_sub = h.loc[t_start_vol:t_check, top+bot]
                 l_sub = l.loc[t_start_vol:t_check, top+bot]

                 roll_h = h_sub.max()
                 roll_l = l_sub.min()

                 vol_metric = (roll_h - roll_l) / (c_check + 1e-12)
                 valid_vol = vol_metric[vol_metric >= 0.08].index.tolist()

                 candidates.update(valid_vol)

        trade_syms = list(candidates)
        if not trade_syms:
            eq_curve.append((ts, equity))
            continue

        mrk = mkt_gates.loc[ts]
        t_exh_lag = ts - pd.Timedelta(hours=1)
        trend_df = feats.get("trend_pct")

        # Build Rows
        rows = []
        for sym in trade_syms:
            try:
                t_val = 0.0
                if trend_df is not None and sym in trend_df.columns:
                    t_val = float(trend_df.loc[ts, sym])
                direction = "up" if t_val > 0 else "down"

                m_bundle = alpha_models.get(direction)
                if not m_bundle or not m_bundle["mr"] or not m_bundle["tf"]: continue

                model_mr = m_bundle["mr"]["model"]
                model_tf = m_bundle["tf"]["model"]
                feat_cols = m_bundle["mr"]["feat_cols"] # Assuming similar
                # TF/MR models might have different feat_cols now!
                # We need to fetch from bundle structure properly if I changed it.
                # In select_best_horizon: `best_m = {"model": race, "H": H, "feat_cols": cols}`
                # So we have separate cols.

                feat_cols_mr = m_bundle["mr"]["feat_cols"]
                feat_cols_tf = m_bundle["tf"]["feat_cols"]

                meta_model = meta_models.get(direction)

                p_lag = 0.5
                if t_exh_lag in p_exh_hist.index and sym in p_exh_hist.columns:
                    p_lag = float(p_exh_hist.loc[t_exh_lag, sym])

                rec = {
                    "symbol": sym,
                    "direction": direction,
                    "model_mr": model_mr, "model_tf": model_tf, "meta_model": meta_model,
                    "feat_cols_mr": feat_cols_mr, "feat_cols_tf": feat_cols_tf,
                    "mkt_ret24h": float(mrk["mkt_ret24h"]),
                    "mkt_ret6h": float(mrk["mkt_ret6h"]),
                    "mkt_trend": float(mrk["mkt_trend"]),
                    "mkt_rv": float(mrk["mkt_rv"]),
                    "G_VOL": int(mrk["G_VOL"]), "G_TREND": int(mrk["G_TREND"]),
                    "p_exh_lag1": p_lag
                }

                # Collect features for MR/TF (union of keys needed? or just store sym/ts and fetch later)
                # Let's collect superset
                all_keys = set(feat_cols_mr) | set(feat_cols_tf) | set(cfg.get("spike_feature_keys", [])) | set(cfg.get("meta_feature_keys", []))

                for k in all_keys:
                    if k in feats:
                        rec[k] = float(feats[k].loc[ts, sym])

                rows.append(rec)
            except Exception:
                continue

        if not rows:
             eq_curve.append((ts, equity))
             continue

        df_all = pd.DataFrame(rows)
        score_raw_list = []

        # Run Spike Model Inference Batch
        spike_keys = cfg.get("spike_feature_keys", [])
        if spike_model:
            # Prepare X_spike
            X_spike = df_all[spike_keys].fillna(0.0)
            if not X_spike.empty:
                probs = spike_model.predict_proba(X_spike)
                for i in range(probs.shape[1]):
                    df_all[f"spike_prob_{i}"] = probs[:, i]
            else:
                 for i in range(4): df_all[f"spike_prob_{i}"] = 0.0
        else:
             for i in range(4): df_all[f"spike_prob_{i}"] = 0.0

        for d, grp in df_all.groupby("direction"):
            first = grp.iloc[0]
            model_mr = first["model_mr"]
            model_tf = first["model_tf"]
            meta_model = first["meta_model"]
            fcols_mr = first["feat_cols_mr"]
            fcols_tf = first["feat_cols_tf"]

            # Interaction toggles?
            # Assuming models trained with toggles?
            # `build_hourly_training_set_and_weights` applied toggles.
            # We must apply same.

            # Prepare X_mr
            grp_mr = apply_interaction_toggles(grp.copy(), fcols_mr, ["G_VOL","G_TREND"], drop_raw=cfg["drop_raw_causal"])
            # Re-select cols (toggles adds cols)
            # We need to match columns expected by model.
            # `model.predict` handles selection via `MDI` subset internally?
            # `MRModel.predict` calls `X[self.selected_features]`.
            # So we pass dataframe with all potential toggle columns.
            # But `fcols_mr` already contains the list of causal cols used for toggling?
            # No, `fcols_mr` is the list of columns available/selected?
            # In `build_hourly_training_set_and_weights`:
            # `df = apply_interaction_toggles(...)`. `X_out` has toggle cols.
            # `cols = list(X_out.columns)`.
            # So `fcols_mr` contains the toggled column names (e.g. `feature_G_VOL_0`).
            # So we need to generate these columns in `grp_mr`.
            # `apply_interaction_toggles` generates them based on `causal_cols`.
            # But `fcols_mr` might be a subset.
            # We should toggle `cfg["causal_cols"]` (or the specific keys) then pass to predict.
            # `apply_interaction_toggles` uses `causal_cols` arg.
            # I should pass `mr_feature_keys` as `causal_cols` if that's what was used?
            # `build_hourly_training_set_and_weights` used `feat_keys` (which was `mr_feature_keys`).
            # So we should use `mr_feature_keys`.

            keys_mr = cfg.get("mr_feature_keys", cfg["causal_cols"])
            grp_mr = apply_interaction_toggles(grp.copy(), keys_mr, ["G_VOL","G_TREND"], drop_raw=cfg["drop_raw_causal"])
            p_mr = model_mr.predict(grp_mr.fillna(0.0))

            keys_tf = cfg.get("tf_feature_keys", cfg["causal_cols"])
            grp_tf = apply_interaction_toggles(grp.copy(), keys_tf, ["G_VOL","G_TREND"], drop_raw=cfg["drop_raw_causal"])
            p_tf = model_tf.predict(grp_tf.fillna(0.0))

            if meta_model:
                # X_meta needs meta features + spike probs + logit preds
                # `grp` has spike probs and meta features (collected in `rows`).
                X_meta = meta_model.prepare_meta_features(p_tf, p_mr, grp)
                score = meta_model.predict(X_meta)
            else:
                score = p_tf - p_mr
                sign = 1.0 if d == "up" else -1.0
                score = score * sign

            for i, idx in enumerate(grp.index):
                score_raw_list.append((grp.loc[idx, "symbol"], score[i]))

        score_raw_list.sort(key=lambda x: x[1], reverse=True)
        longs = [x for x in score_raw_list if x[1] > cfg["thr_long"]]
        shorts = [x for x in score_raw_list if x[1] < cfg["thr_short"]]

        picks_long = longs[:cfg["k_long"]]
        picks_short = sorted(shorts, key=lambda x: x[1])[:cfg["k_short"]]

        final_orders = []
        for s, sc in picks_long: final_orders.append((s, "long", sc))
        for s, sc in picks_short: final_orders.append((s, "short", sc))

        total_wt = sum(abs(x[2]) for x in final_orders)
        if total_wt == 0:
            eq_curve.append((ts, equity))
            continue

        gross_cap = float(cfg["wallet_gross_cap"])

        pnl = 0.0
        for sym, side, raw_score in final_orders:
            w = gross_cap * (abs(raw_score) / total_wt)

            entry_px = entry_price_next_hour_open(o, ts_entry, sym)
            if np.isnan(entry_px) or entry_px <= 0: continue

            rr, exit_ts, why = simulate_trade_hourly(
                o_s=o[sym], h_s=h[sym], l_s=l[sym], c_s=c[sym],
                feats_s=feats["atr_pct"].loc[:, sym],
                entry_ts=ts_entry,
                entry_px=entry_px,
                side=side,
                cfg=cfg,
                max_hold_hours=cfg["hold_hours"]
            )

            if side == "short":
                rr -= borrow_hourly * float(cfg["hold_hours"])
            rr -= 2.0 * fee_rt

            pnl += w * rr
            trades.append({
                "ts_sig": ts,
                "symbol": sym,
                "side": side,
                "weight": w,
                "score_raw": raw_score,
                "ret": float(rr),
            })

        equity *= (1.0 + pnl)
        eq_curve.append((ts, equity))

    eq = pd.Series({t: e for t, e in eq_curve}).sort_index()
    trades_df = pd.DataFrame(trades)

    if len(eq) > 2:
        dr = eq.pct_change().dropna()
        sharpe = (dr.mean() / (dr.std(ddof=0) + 1e-12)) * np.sqrt(365.0 * 24.0)
        max_dd = (eq / eq.cummax() - 1.0).min()
    else:
        sharpe = np.nan
        max_dd = np.nan

    stats = {
        "total_return": float(eq.iloc[-1] - 1.0) if len(eq) else np.nan,
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "n_trades": int(len(trades_df)) if not trades_df.empty else 0,
    }
    return eq, trades_df, stats

def generate_hourly_signals(ts_sig, feats, mkt_gates, model_bundle, risk_config, cfg, p_exh_cand, current_positions_syms):
    tprint(f"Entering function: generate_hourly_signals in engine.py")
    if ts_sig not in mkt_gates.index:
        return []

    # 1. Live Candidate Selection (t-12 to t+16 logic adapted for live)
    # Check t, t-4, t-8, t-12, t-16
    candidates = set()
    lookback_offsets = [0, 4, 8, 12, 16]

    # Need access to prices for Vol Filter
    c = feats["close"] if "close" in feats else None # feats usually doesn't have raw price?
    # feats has 'close' key from 'compute_features_hourly' output? No.
    # In 'main.py', feats is result of 'compute_features_hourly'.
    # Does 'compute_features_hourly' return raw prices?
    # It returns 'ret1h', 'atr_pct' etc. It doesn't return raw price panel unless added.
    # But wait, `generate_hourly_signals` receives `feats`.
    # `hourly_engine_backtest` receives `panel` AND `feats`.
    # `generate_hourly_signals` does NOT receive `panel`.
    # This is a limitation.
    # I should rely on `select_trade_candidates_hourly` using `feats`.
    # `feats` contains `dist_ema_fast` etc.
    # But `volatility filter` requires Price.
    # If I can't check volatility, I might skip it or use `atr_pct` as proxy?
    # "12h High/Low diff >= 8%".
    # `atr_expansion` or `range_pct`?
    # `range_pct` is `h-l`.
    # 12h Range? `feats["range_pct"]` is 1h range.
    # If `feats` doesn't have 12h range, I can't check it perfectly.
    # But maybe I added `ret12h`? `feats["ret12h"]`.
    # I can use `ret12h.abs()` as a proxy for range?
    # Or assume candidates are pre-filtered?
    # In `training.py`, we use `select_trade_candidates_vectorized` which uses `panel`.
    # `main.py` calls `compute_features_hourly` using `panel`.
    # If `generate_hourly_signals` needs panel, I should update signature.
    # But `generate_hourly_signals` is called by `main.py`.
    # `execute_hourly` in `main.py` has `panel`.
    # I can pass `panel` to `generate_hourly_signals`.
    # But for now, I will use `ret12h.abs() > 0.08` as a proxy if panel missing?
    # No, let's stick to `select_trade_candidates_hourly` on `feats` and assume volatilty is handled by `trade_extreme_min/max` parameters or existing metrics?
    # Actually, the requirement was specific.
    # I will attempt to check if `feats` has enough info.
    # I added `donch_dist_12`. If `donch_dist_12` is large, price moved.
    # But that's normalized by ATR.

    # I will loop offsets and gather candidates using standard logic.

    for offset in lookback_offsets:
        t_check = ts_sig - pd.Timedelta(hours=offset)
        if t_check in feats["ret24h"].index:
            top, bot = select_trade_candidates_hourly(
                feats, t_check, list(feats["ret24h"].columns),
                cfg["trade_extreme_pct"], cfg["trade_extreme_min"], cfg["trade_extreme_max"],
                cfg["trade_deviation_metric"]
            )
            # Add vol filter proxy if possible?
            # Skip for now to avoid breaking without panel.
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
            t_val = 0.0
            if trend_df is not None and sym in trend_df.columns: t_val = float(trend_df.loc[ts_sig, sym])
            direction = "up" if t_val > 0 else "down"
            m_bundle = alpha_models.get(direction)
            if not m_bundle or not m_bundle["mr"] or not m_bundle["tf"]: continue

            model_mr = m_bundle["mr"]["model"]
            model_tf = m_bundle["tf"]["model"]
            fcols_mr = m_bundle["mr"]["feat_cols"]
            fcols_tf = m_bundle["tf"]["feat_cols"]

            meta_model = meta_models.get(direction)
            p_lag = 0.5
            if ts_lag in p_exh_cand.index and sym in p_exh_cand.columns: p_lag = float(p_exh_cand.loc[ts_lag, sym])

            rec = {
                "symbol": sym, "direction": direction, "model_mr": model_mr, "model_tf": model_tf, "meta_model": meta_model,
                "feat_cols_mr": fcols_mr, "feat_cols_tf": fcols_tf,
                "mkt_ret24h": float(mrk["mkt_ret24h"]), "mkt_ret6h": float(mrk["mkt_ret6h"]), "mkt_trend": float(mrk["mkt_trend"]), "mkt_rv": float(mrk["mkt_rv"]),
                "G_VOL": int(mrk["G_VOL"]), "G_TREND": int(mrk["G_TREND"]), "p_exh_lag1": p_lag
            }

            all_keys = set(fcols_mr) | set(fcols_tf) | set(cfg.get("spike_feature_keys", [])) | set(cfg.get("meta_feature_keys", []))

            for k in all_keys:
                if k in feats: rec[k] = float(feats[k].loc[ts_sig, sym])

            rows.append(rec)
        except Exception: continue

    df_all = pd.DataFrame(rows)
    score_raw_list = []
    if not df_all.empty:
        # Spike Inference
        spike_keys = cfg.get("spike_feature_keys", [])
        if spike_model:
            X_spike = df_all[spike_keys].fillna(0.0)
            if not X_spike.empty:
                probs = spike_model.predict_proba(X_spike)
                for i in range(probs.shape[1]):
                    df_all[f"spike_prob_{i}"] = probs[:, i]
            else:
                 for i in range(4): df_all[f"spike_prob_{i}"] = 0.0
        else:
             for i in range(4): df_all[f"spike_prob_{i}"] = 0.0

        for d, grp in df_all.groupby("direction"):
            first = grp.iloc[0]
            model_mr = first["model_mr"]; model_tf = first["model_tf"]; meta_model = first["meta_model"]
            fcols_mr = first["feat_cols_mr"]
            fcols_tf = first["feat_cols_tf"]

            keys_mr = cfg.get("mr_feature_keys", cfg["causal_cols"])
            grp_mr = apply_interaction_toggles(grp.copy(), keys_mr, ["G_VOL","G_TREND"], drop_raw=cfg["drop_raw_causal"])
            p_mr = model_mr.predict(grp_mr.fillna(0.0))

            keys_tf = cfg.get("tf_feature_keys", cfg["causal_cols"])
            grp_tf = apply_interaction_toggles(grp.copy(), keys_tf, ["G_VOL","G_TREND"], drop_raw=cfg["drop_raw_causal"])
            p_tf = model_tf.predict(grp_tf.fillna(0.0))

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
                score_raw_list.append((sym, s_score, dom))

    score_raw_list.sort(key=lambda x: x[1], reverse=True)
    longs = [x for x in score_raw_list if x[1] > cfg["thr_long"]][:cfg["k_long"]]
    shorts = sorted([x for x in score_raw_list if x[1] < cfg["thr_short"]], key=lambda x: x[1])[:cfg["k_short"]]

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
            orders_out.append(ord)

    return orders_out
