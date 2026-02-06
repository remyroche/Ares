import os
import pickle
import pandas as pd
import numpy as np

from extreme_price_movements.utils import tprint, Timer
from extreme_price_movements.data_store import load_features, save_features, save_artifact_df, load_artifact_df, to_panel
from extreme_price_movements.training import generate_label_datasets, generate_exhaustion_history, optimize_risk_params, train_models_from_artifacts
from extreme_price_movements.features import compute_market_features, add_regime_gates, compute_features_hourly
from extreme_price_movements.universe import get_training_universe
from extreme_price_movements.engine import simulate_trade_hourly, generate_hourly_signals, _build_side_score_df
from extreme_price_movements.metrics import MetricsLogger
from extreme_price_movements.risk import TrailingStop

def run_label_generation_step_v2(ts_sig, margin_symbols, cfg, store, ex):
    tprint("STEP: LABEL GENERATION START")
    train_syms = get_training_universe(margin_symbols, cfg, store, ts_sig=ts_sig)
    tprint(f"Universe: {len(train_syms)} symbols")

    # Load Data & Features
    dfs = {}

    lookback_days = max(90, int(cfg["fetch_years"] * 365))

    with Timer("Data Load"):
        for s in train_syms:
            df = store.load(s)
            if not df.empty: dfs[s] = df[df.index <= ts_sig].tail(24*lookback_days)

    if not dfs:
        tprint("No data available.")
        return

    panel = to_panel(dfs)
    mkt_df = compute_market_features(panel, cfg["market_basket"])
    mkt_gates = add_regime_gates(mkt_df, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"])

    feats = load_features(ts_sig, cfg["data_root"])
    if feats is None:
        tprint("ERROR: Features not found. Run feature_generation first.")
        return

    # Restrict to symbols present in both panel and features
    sample_feat = next(iter(feats.values()))
    feat_syms = set(sample_feat.columns)
    panel_syms = set(panel["close"].columns)
    valid_syms = sorted(feat_syms & panel_syms & set(train_syms))
    tprint(f"Symbol intersection: {len(valid_syms)} (feats={len(feat_syms)}, panel={len(panel_syms)}, universe={len(train_syms)})")
    if not valid_syms:
        tprint("ERROR: No overlapping symbols between features and panel.")
        return
    train_syms = valid_syms

    # Restrict panel to valid symbols to keep downstream operations aligned
    panel = {k: v[valid_syms] for k, v in panel.items() if isinstance(v, pd.DataFrame)}

    # 1. Exhaustion History
    p_exh_hist = generate_exhaustion_history(panel, feats, mkt_gates, cfg, ts_sig, cfg["train_lookback_hours"], train_syms)

    # Save Exhaustion History
    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    save_artifact_df(p_exh_hist, cfg["data_root"], run_id, "labels", "exhaustion_history")

    # 2. Label Datasets
    datasets = generate_label_datasets(panel, feats, mkt_gates, cfg, train_syms, ts_sig, p_exh_hist)

    for name, df in datasets.items():
        save_artifact_df(df, cfg["data_root"], run_id, "labels", name)

    tprint("STEP: LABEL GENERATION COMPLETE")

def run_training_step(ts_sig, cfg, store=None, margin_symbols=None):
    """Train all models from label artifacts. Saves trained state to disk."""
    tprint("STEP: MODEL TRAINING START")

    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    datasets = {}

    # 1. Load label artifacts
    tprint("Loading label datasets from artifacts...")

    # Spike anatomy
    missing_spike = []
    for mode in ["best", "worst"]:
        name = f"spike_anatomy_{mode}"
        df_spike = load_artifact_df(cfg["data_root"], run_id, "labels", name)
        if df_spike is not None:
            datasets[name] = df_spike
        else:
            missing_spike.append(mode)

    if missing_spike:
        tprint(f"Adding Missing Spike artifacts: {missing_spike} (Generating in-memory...)")
        if store is None:
            tprint("ERROR: store is None, cannot generate missing spike artifacts.")
            # Critical failure if we can't generate
        else:
            # Need features and panel. Load them.
            # Mirror run_label_generation_step_v2 logic roughly but localized
            tprint("Loading features and panel for Spike Anatomy generation...")
            feats = load_features(ts_sig, cfg["data_root"])
            if feats is None:
                tprint("ERROR: Features not found.")
            else:
                train_syms = get_training_universe(margin_symbols, cfg, store, ts_sig=ts_sig)
                dfs = {}
                lookback_days = max(90, int(cfg["fetch_years"] * 365))
                for s in train_syms:
                    df = store.load(s)
                    if not df.empty: dfs[s] = df[df.index <= ts_sig].tail(24*lookback_days)
                
                if dfs:
                    panel = to_panel(dfs)
                    mkt_df = compute_market_features(panel, cfg["market_basket"])
                    mkt_gates = add_regime_gates(mkt_df, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"])
                    
                    # Intersect symbols
                    sample_feat = next(iter(feats.values()))
                    valid_syms = sorted(set(sample_feat.columns) & set(panel["close"].columns) & set(train_syms))
                    panel = {k: v[valid_syms] for k, v in panel.items() if isinstance(v, pd.DataFrame)}
                    
                    from extreme_price_movements.training import train_spike_anatomy_model
                    
                    for mode in missing_spike:
                        tprint(f"Generating Spike Anatomy ({mode})...")
                        df_spike = train_spike_anatomy_model(panel, feats, mkt_gates, cfg, valid_syms, ts_sig, mode=mode)
                        if df_spike is not None:
                            datasets[f"spike_anatomy_{mode}"] = df_spike
                            save_artifact_df(df_spike, cfg["data_root"], run_id, "labels", f"spike_anatomy_{mode}")
                            tprint(f"Saved generated spike artifact: spike_anatomy_{mode}")
    
    # Alpha models (long/short × mr/tf × horizons)
    trade_sides = ["long", "short"]
    kinds = ["mr", "tf"]
    horizons = cfg["label_horizons_hours"]

    found_count = 0
    for side in trade_sides:
        for k in kinds:
            for H in horizons:
                name = f"train_{side}_{k}_{H}"
                df = load_artifact_df(cfg["data_root"], run_id, "labels", name)
                if df is not None:
                    datasets[name] = df
                    found_count += 1
                    tprint(f"  Loaded {name}: {len(df)} rows")

    # Exhaustion models
    for d in ["up", "down"]:
        name = f"exh_{d}"
        df = load_artifact_df(cfg["data_root"], run_id, "labels", name)
        if df is not None:
            datasets[name] = df
            tprint(f"  Loaded {name}: {len(df)} rows")

    if not found_count:
        tprint("ERROR: No alpha label datasets found. Run 'labels' mode first.")
        return None

    tprint(f"Loaded {len(datasets)} datasets total.")

    # 2. Train models
    with Timer("Model Training"):
        trained_bundle = train_models_from_artifacts(datasets, cfg)

    # 3. Save trained state
    default_risk = {
        "k_sl": cfg.get("risk_k_sl", 2.0),
        "k_trail_start": cfg.get("risk_k_trail_start", 1.0),
        "k_trail_dist": cfg.get("risk_k_trail_dist", 0.5),
        "granular_risk": {}
    }

    state = {
        "ts_trained": ts_sig,
        "bundle": trained_bundle,
        "risk_params": default_risk
    }

    state_dir = os.path.join(cfg["data_root"], "artifacts", run_id, "models")
    os.makedirs(state_dir, exist_ok=True)
    state_path = os.path.join(state_dir, "trained_state.pkl")
    with open(state_path, "wb") as f:
        pickle.dump(state, f)
    tprint(f"Saved trained state to {state_path}")

    # Log summary
    bundle = trained_bundle
    if bundle:
        alpha = bundle.get("alpha_models", {})
        for side in trade_sides:
            for k in kinds:
                m = alpha.get(side, {}).get(k)
                if m:
                    tprint(f"  {side} {k}: H={m['H']}, features={len(m['feat_cols'])}")
                else:
                    tprint(f"  {side} {k}: NO MODEL")

        spike = bundle.get("spike_models", {})
        for mode in ["best", "worst"]:
            m = spike.get(mode)
            tprint(f"  spike_{mode}: {'fitted' if m else 'NO MODEL'}")
            if m and "oof_scores" in m:
                oof_df = m["oof_scores"]
                save_artifact_df(oof_df, cfg["data_root"], run_id, "labels", f"spike_oof_{mode}")
                tprint(f"  Saved OOF scores: spike_oof_{mode}")

        exh = bundle.get("exh_models", {})
        for d in ["up", "down"]:
            m = exh.get(d)
            tprint(f"  exh_{d}: {'fitted' if m and m.model else 'NO MODEL'}")

        meta = bundle.get("meta_models", {})
        for side in trade_sides:
            for k in kinds:
                key = f"{side}_{k}"
                m = meta.get(key)
                tprint(f"  meta_{key}: {'fitted' if m and m.model else 'NO MODEL'}")

    tprint("STEP: MODEL TRAINING COMPLETE")
    return state

def run_risk_optimization_step(ts_sig, margin_symbols, cfg, store, state_file):
    tprint("STEP: RISK OPTIMIZATION START")
    if not os.path.exists(state_file):
        tprint(f"State file {state_file} not found.")
        return

    # Prevent OOS leakage
    run_ts = ts_sig # Keep original for loading artifacts
    opt_ts = ts_sig # Used for data filtering
    
    oos_days = cfg.get("oos_holdout_days", 0)
    if oos_days > 0:
        opt_ts = ts_sig - pd.Timedelta(days=oos_days)
        tprint(f"Risk Optimization: Excluding last {oos_days} days (OOS). Training end: {opt_ts}")

    with open(state_file, "rb") as f:
        state = pickle.load(f)

    bundle = state.get("bundle")
    if not bundle:
        tprint("No model bundle in state.")
        return

    # Need Data for optimization (simulation)
    # Use opt_ts to filter training universe and data
    train_syms = get_training_universe(margin_symbols, cfg, store, ts_sig=opt_ts)
    dfs = {}

    lookback_days = max(90, int(cfg["fetch_years"] * 365))

    for s in train_syms:
        df = store.load(s)
        if not df.empty: dfs[s] = df[df.index <= opt_ts].tail(24*lookback_days)

    if not dfs: return

    panel = to_panel(dfs)
    mkt_df = compute_market_features(panel, cfg["market_basket"])
    mkt_gates = add_regime_gates(mkt_df, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"])
    
    # Load features using run_ts (actual storage location)
    feats = load_features(run_ts, cfg["data_root"])
    if feats is None:
        tprint("ERROR: Features not found for risk optimization.")
        return

    # We also need p_exh_hist. Load from artifacts (using run_ts).
    run_id = run_ts.strftime("%Y%m%d_%H%M%S")
    from extreme_price_movements.data_store import load_artifact_df
    p_exh_hist = load_artifact_df(cfg["data_root"], run_id, "labels", "exhaustion_history")

    if p_exh_hist is None:
        tprint("Exhaustion history artifact missing. Regenerating...")
        # Generate up to opt_ts? Or regenerate full and slice?
        # Typically we want history up to opt_ts for optimization.
        p_exh_hist = generate_exhaustion_history(panel, feats, mkt_gates, cfg, opt_ts, cfg["train_lookback_hours"], train_syms)

    alpha_models = bundle["alpha_models"]
    best_risk = optimize_risk_params(panel, feats, mkt_gates, cfg, train_syms, opt_ts, p_exh_hist, alpha_models)

    prev_risk = state.get("risk_params", {}) if isinstance(state.get("risk_params"), dict) else {}
    if isinstance(prev_risk, dict) and "signal_params" in prev_risk and isinstance(best_risk, dict):
        best_risk["signal_params"] = prev_risk["signal_params"]

    state["risk_params"] = best_risk
    with open(state_file, "wb") as f:
        pickle.dump(state, f)

    tprint("Risk params updated in state file.")
    tprint("STEP: RISK OPTIMIZATION COMPLETE")

def run_backtest_step(ts_sig, margin_symbols, cfg, store, state_file):
    tprint("STEP: BACKTEST START")
    if not os.path.exists(state_file):
        tprint("State file not found.")
        return

    with open(state_file, "rb") as f:
        model_state = pickle.load(f)

    train_syms = get_training_universe(margin_symbols, cfg, store, ts_sig=ts_sig)
    dfs = {}
    lookback_days = max(90, int(cfg["fetch_years"] * 365))

    with Timer("Backtest Data Load"):
        for s in train_syms:
            df = store.load(s)
            if not df.empty:
                dfs[s] = df[df.index <= ts_sig].tail(24 * lookback_days)

    if not dfs:
        return

    panel = to_panel(dfs)
    mkt_df = compute_market_features(panel, cfg["market_basket"])
    mkt_gates = add_regime_gates(mkt_df, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"])
    feats = load_features(ts_sig, cfg["data_root"])

    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    from extreme_price_movements.data_store import load_artifact_df
    p_exh_hist = load_artifact_df(cfg["data_root"], run_id, "labels", "exhaustion_history")
    if p_exh_hist is None:
        p_exh_hist = generate_exhaustion_history(panel, feats, mkt_gates, cfg, ts_sig, cfg["train_lookback_hours"], train_syms)

    test_days = cfg.get("oos_holdout_days", 14)
    start_ts = ts_sig - pd.Timedelta(days=test_days)
    end_ts = ts_sig - pd.Timedelta(hours=24)
    valid_ts = [t for t in feats["ret1h"].index if t >= start_ts and t <= end_ts]
    tprint(f"Running backtest over {len(valid_ts)} hours...")
    if len(valid_ts) < 48:
        tprint("Not enough timestamps for backtest optimization.")
        return

    o_s = panel["open"]; h_s = panel["high"]; l_s = panel["low"]; c_s = panel["close"]
    atr_s = feats["atr_pct"]
    risk_conf = model_state.get("risk_params", {}) or {}
    bundle = model_state.get("bundle")

    fee_bps = cfg.get("fee_bps", 25.0)
    fee_rate = fee_bps / 10000.0

    def rank01(x: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
        pct = (ranks - 1.0) / max(1.0, x.size - 1.0)
        return pct if higher_is_better else (1.0 - pct)

    def pnl_sortino_dd_utility(pnls, sortinos, max_dds, w_pnl=0.65, w_sortino=0.25, w_dd=0.10):
        return w_pnl * rank01(pnls, True) + w_sortino * rank01(sortinos, True) + w_dd * rank01(max_dds, True)

    def compute_metrics(trades):
        if not trades:
            return 0.0, 0.0, 0.0, 0.0, 0
        rets = np.array([x["pnl"] for x in trades], dtype=np.float64)
        pnl = float(np.sum(rets))
        neg = rets[rets < 0]
        sortino = float(np.mean(rets) / (np.std(neg) + 1e-12)) if neg.size > 0 else float(np.mean(rets) / 1e-12)
        eq = np.cumsum(rets)
        peak = np.maximum.accumulate(eq)
        dd = eq - peak
        max_dd = float(np.min(dd)) if dd.size else 0.0
        count = len(trades)
        win_rate = float(np.mean(rets > 0)) if count > 0 else 0.0
        return pnl, sortino, max_dd, win_rate, count

    def run_slice(ts_list, signal_params):
        trades = []
        local_risk = dict(risk_conf)
        local_risk["signal_params"] = signal_params

        for t in ts_list:
            orders = generate_hourly_signals(t, feats, mkt_gates, bundle, local_risk, cfg, p_exh_hist, [])
            for order in orders:
                sym = order["symbol"]
                side = order["side"]
                score = float(order["score"])
                dom = order["dom"]
                weight = float(order.get("weight", 0.0))

                entry_ts = t + pd.Timedelta(hours=1)
                if entry_ts not in o_s.index or sym not in o_s.columns:
                    continue
                entry_px = float(o_s.loc[entry_ts, sym])

                if dom == "tf":
                    mode = "best" if side == "long" else "worst"
                else:
                    mode = "worst" if side == "long" else "best"
                risk_keys = [f"risk_{dom}_{mode}", f"risk_{side}_{dom}"]
                granular = local_risk.get("granular_risk", {})
                rp = {}
                for risk_key in risk_keys:
                    if risk_key in granular:
                        rp = granular[risk_key]
                        break

                k_sl = rp.get("k_sl", cfg["risk_k_sl"])
                k_ts = rp.get("k_trail_start", cfg["risk_k_trail_start"])
                k_td = rp.get("k_trail_dist", cfg["risk_k_trail_dist"])

                # Extract optimized TP/SL multipliers if available
                tp_mult = rp.get("tp_mult", cfg.get("tp_mult"))
                sl_mult = rp.get("sl_mult", cfg.get("sl_mult"))

                sc_scale = rp.get("score_scale", 0.0)
                adj = (1.0 + sc_scale * abs(score))

                temp_cfg = cfg.copy()
                temp_cfg["risk_k_sl"] = k_sl * adj
                temp_cfg["risk_k_trail_start"] = k_ts
                temp_cfg["risk_k_trail_dist"] = k_td

                if tp_mult is not None: temp_cfg["tp_mult"] = tp_mult
                if sl_mult is not None: temp_cfg["sl_mult"] = sl_mult

                ret, exit_ts, reason = simulate_trade_hourly(
                    o_s[sym], h_s[sym], l_s[sym], c_s[sym], atr_s[sym],
                    entry_ts, entry_px, side, temp_cfg, max_hold_hours=24
                )
                net_ret = ret - (2.0 * fee_rate)
                pnl = net_ret * weight
                trades.append({
                    "entry_ts": entry_ts, "symbol": sym, "side": side, "dom": dom,
                    "score": score, "weight": weight, "ret": net_ret, "pnl": pnl,
                    "gross_ret": ret, "exit_ts": exit_ts, "reason": reason
                })
        return trades

    split = max(24, int(len(valid_ts) * 0.6))
    train_ts = valid_ts[:split]
    test_ts = valid_ts[split:]

    raw_train = run_slice(train_ts, {
        "thr_long": -1e9, "thr_short": 1e9,
        "k_long": cfg.get("k_long", 10), "k_short": cfg.get("k_short", 10),
        "size_min": 0.03, "size_max": 0.15, "size_k": 2.0, "size_x0": 0.5, "size_zcap": 4.0,
    })
    train_abs = np.array([abs(t["score"]) for t in raw_train], dtype=np.float64)
    q50 = float(np.quantile(train_abs, 0.5)) if train_abs.size else 0.0
    q90 = float(np.quantile(train_abs, 0.9)) if train_abs.size else max(q50 + 1e-6, 1e-3)

    # Calibrate long/short meta-score comparability on train only.
    side_frames = []
    for t in train_ts:
        s_df = _build_side_score_df(t, feats, mkt_gates, bundle, cfg, p_exh_hist, [])
        if not s_df.empty:
            side_frames.append(s_df)

    if side_frames:
        side_all = pd.concat(side_frames, ignore_index=True)

        def _center_scale(arr):
            v = np.asarray(arr, dtype=np.float64)
            if v.size == 0:
                return 0.0, 1.0
            c = float(np.quantile(v, 0.5))
            s = float(np.quantile(v, 0.9) - np.quantile(v, 0.5))
            return c, max(s, 1e-6)

        lmr = side_all[side_all["side_key"] == "long"]["score_mr"].values
        smr = side_all[side_all["side_key"] == "short"]["score_mr"].values
        ltf = side_all[side_all["side_key"] == "long"]["score_tf"].values
        stf = side_all[side_all["side_key"] == "short"]["score_tf"].values

        lmr_c, lmr_s = _center_scale(lmr)
        smr_c, smr_s = _center_scale(smr)
        ltf_c, ltf_s = _center_scale(ltf)
        stf_c, stf_s = _center_scale(stf)

        score_scale_params = {
            "long_mr_center": lmr_c, "long_mr_scale": lmr_s,
            "short_mr_center": smr_c, "short_mr_scale": smr_s,
            "long_tf_center": ltf_c, "long_tf_scale": ltf_s,
            "short_tf_center": stf_c, "short_tf_scale": stf_s,
        }
    else:
        score_scale_params = {}

    thr_long_grid = [0.00, 0.01, 0.02, 0.03]
    thr_short_grid = [0.00, 0.01, 0.02, 0.03]
    x0_grid = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    k_grid = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

    combos = []
    for tl in thr_long_grid:
        for ts_ in thr_short_grid:
            for x0 in x0_grid:
                for k in k_grid:
                    params = {
                        "thr_long": tl,
                        "thr_short": -ts_,
                        "k_long": cfg.get("k_long", 10),
                        "k_short": cfg.get("k_short", 10),
                        "size_min": 0.03,
                        "size_max": 0.15,
                        "size_k": k,
                        "size_x0": x0,
                        "size_zcap": 4.0,
                        "size_q50": q50,
                        "size_q90": q90,
                        "score_scale_params": score_scale_params,
                    }
                    tr = run_slice(train_ts, params)
                    pnl, sortino, max_dd, win_rate, count = compute_metrics(tr)
                    tprint(f"SignalOpt tl={tl:.3f} ts={-ts_:.3f} k={k:.2f} x0={x0:.2f} -> pnl={pnl:.6f} sortino={sortino:.6f} maxdd={max_dd:.6f} wr={win_rate:.2f} n={count}")
                    combos.append((params, pnl, sortino, max_dd))

    if combos:
        pnls = np.array([c[1] for c in combos], dtype=np.float64)
        sorts = np.array([c[2] for c in combos], dtype=np.float64)
        dds = np.array([c[3] for c in combos], dtype=np.float64)
        util = pnl_sortino_dd_utility(pnls, sorts, dds)
        best_i = int(np.argmax(util))
        best_signal_params = combos[best_i][0]
    else:
        best_signal_params = {
            "thr_long": cfg.get("thr_long", 0.01), "thr_short": cfg.get("thr_short", -0.01),
            "k_long": cfg.get("k_long", 10), "k_short": cfg.get("k_short", 10),
            "size_min": 0.03, "size_max": 0.15, "size_k": 2.0, "size_x0": 0.5,
            "size_zcap": 4.0, "size_q50": q50, "size_q90": q90,
            "score_scale_params": score_scale_params,
        }

    tprint(f"Selected signal params: {best_signal_params}")

    test_trades = run_slice(test_ts, best_signal_params)
    pnl, sortino, max_dd, win_rate, count = compute_metrics(test_trades)
    avg_pnl = pnl / count if count > 0 else 0.0
    tprint(f"Backtest OOS Result: Trades={count}, PnL={pnl:.6f}, AvgPnL={avg_pnl:.6f}, Sortino={sortino:.6f}, MaxDD={max_dd:.6f}, WinRate={win_rate:.2f}")

    # Breakdown
    if test_trades:
        df_t = pd.DataFrame(test_trades)
        tprint("--- OOS Breakdown ---")
        for side in ["long", "short"]:
            df_s = df_t[df_t["side"] == side]
            if not df_s.empty:
                s_pnl = df_s["pnl"].sum()
                s_wr = (df_s["pnl"] > 0).mean()
                tprint(f"  {side.upper()}: Trades={len(df_s)}, PnL={s_pnl:.4f}, WinRate={s_wr:.2f}")
        for dom in ["mr", "tf"]:
            df_d = df_t[df_t["dom"] == dom]
            if not df_d.empty:
                d_pnl = df_d["pnl"].sum()
                d_wr = (df_d["pnl"] > 0).mean()
                tprint(f"  {dom.upper()}: Trades={len(df_d)}, PnL={d_pnl:.4f}, WinRate={d_wr:.2f}")
        tprint("-----------------------")

    if test_trades:
        df_res = pd.DataFrame(test_trades)
        out_path = os.path.join(cfg["data_root"], "artifacts", run_id, "backtest_results.csv")
        df_res.to_csv(out_path, index=False)
        tprint(f"Detailed results saved to {out_path}")

    risk_conf["signal_params"] = best_signal_params
    model_state["risk_params"] = risk_conf
    with open(state_file, "wb") as f:
        pickle.dump(model_state, f)
    tprint("Saved optimized signal params to trained state for inference use.")
    tprint("STEP: BACKTEST COMPLETE")

def run_feature_generation_step(ts_sig, margin_symbols, cfg, store):
    tprint("STEP: FEATURE GENERATION START")
    tprint(f"Target Timestamp: {ts_sig}")

    # 1. Define Universe
    # We want "all assets in our universe".
    # This implies the margin universe (Top M).
    train_syms = get_training_universe(margin_symbols, cfg, store, ts_sig=None) 
    # disable ts_sig in universe selection to ensure we see everything available currently?
    # actually getting training universe usually filters by variance over 30d.
    # User said "Ensure we generate features for ALL assets... if not log why".
    # So we should probably NOT filter by variance yet? Or just log the drops?
    # get_training_universe DOES filter.
    # Let's verify what we have vs what we drop.
    
    tprint(f"Universe (Top {cfg['fetch_symbols_M']} Vol + Basket + VarianceFilter): {len(train_syms)} symbols")
    
    # 2. Load Data
    dfs = {}
    lookback_days = max(180, int(cfg["fetch_years"] * 365))
    
    # Load Market Basket First (Critical)
    for s in cfg["market_basket"]:
        if s not in train_syms:
            train_syms.append(s)
            
    loaded_syms = []
    skipped_log = []

    with Timer("Feature Gen Data Load"):
        for s in train_syms:
            df = store.load(s, end_ts=ts_sig) # Load up to ts_sig
            
            # Constraints Check
            if df.empty:
                skipped_log.append(f"{s}: Empty DataFrame")
                continue
            
            # Check length (at least 60 days for basic moving averages + volatility)
            min_rows = 24 * 60 
            if len(df) < min_rows:
                skipped_log.append(f"{s}: Insufficient data ({len(df)} rows < {min_rows})")
                continue
                
            # Check recent data freshness?
            last_ts = df.index[-1]
            if (ts_sig - last_ts).days > 7:
                 skipped_log.append(f"{s}: Stale data (Last: {last_ts}, Target: {ts_sig})")
                 continue

            dfs[s] = df.tail(24 * lookback_days)
            loaded_syms.append(s)

    tprint(f"Loaded {len(loaded_syms)} symbols. Skipped {len(skipped_log)}.")
    for msg in skipped_log:
        tprint(f"  [SKIP] {msg}")

    if not dfs:
        tprint("CRITICAL: No valid data found for feature generation.")
        return

    # 3. Compute Features (Panel)
    tprint("Constructing Panel...")
    panel = to_panel(dfs)
    
    tprint("Computing Market Features...")
    mkt_df = compute_market_features(panel, cfg["market_basket"])
    mkt_gates = add_regime_gates(mkt_df, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"])
    
    tprint("Computing Asset Features (Hourly)...")
    feats = compute_features_hourly(panel, mkt_gates, cfg)
    
    # 4. Save
    save_features(feats, ts_sig, cfg["data_root"])
    
    tprint(f"Generated features for {len(loaded_syms)} symbols.")
    tprint("STEP: FEATURE GENERATION COMPLETE")
