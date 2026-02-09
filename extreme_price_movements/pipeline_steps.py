import os
import pickle
import pandas as pd
import numpy as np

from extreme_price_movements.utils import tprint, Timer
from extreme_price_movements.data_store import load_features, save_features, save_artifact_df, load_artifact_df, to_panel
from extreme_price_movements.training import generate_label_datasets, generate_exhaustion_history, optimize_risk_params, train_models_from_artifacts
from extreme_price_movements.features import compute_market_features, add_regime_gates, compute_features_hourly
from extreme_price_movements.universe import get_training_universe, refresh_margin_universe_daily
from extreme_price_movements.engine import simulate_trade_hourly, generate_hourly_signals, _build_side_score_df
from extreme_price_movements.metrics import MetricsLogger
from extreme_price_movements.risk import TrailingStop
from extreme_price_movements.reports.report_generator import generate_training_report, generate_risk_report, generate_backtest_report

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

    # Data coverage diagnostics
    _close = panel["close"]
    _ts_min = _close.index.min()
    _ts_max = _close.index.max()
    _n_hours = len(_close)
    _n_days = (_ts_max - _ts_min).total_seconds() / 86400 if _ts_max > _ts_min else 0
    _n_syms = _close.shape[1]
    _non_nan_pct = float(_close.notna().sum().sum()) / max(_close.size, 1) * 100
    tprint(f"DATA COVERAGE: {_n_syms} symbols, {_n_hours} hourly bars, "
           f"{_n_days:.0f} days ({_ts_min.date()} to {_ts_max.date()}), "
           f"{_non_nan_pct:.1f}% non-NaN")
    if _n_days < 365:
        tprint(f"WARNING: Only {_n_days:.0f} days of data — recommend >= 365 days for robust training")

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

    # Ensure we always have a margin universe for downstream specialist training
    if margin_symbols is None:
        try:
            margin_cache = refresh_margin_universe_daily(None, quotes=cfg.get("margin_quotes", ("USDT", "USDC", "BUSD", "EUR")))
            margin_symbols = margin_cache.symbols if margin_cache else []
        except Exception as exc:
            tprint(f"WARNING: Failed to refresh margin universe ({exc}); proceeding without specialist training")
            margin_symbols = []

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
    
    # 2b. Train Specialist Models (requires panel and feats)
    # Load panel and feats if they were used for spike generation
    if store is not None and margin_symbols:
        try:
            from extreme_price_movements.training import train_specialist_models
            
            tprint("Loading panel and features for Specialist training...")
            train_syms = get_training_universe(margin_symbols, cfg, store, ts_sig=ts_sig)
            
            dfs = {}
            lookback_days = max(90, int(cfg["fetch_years"] * 365))
            for s in train_syms:
                df = store.load(s)
                if not df.empty:
                    dfs[s] = df[df.index <= ts_sig].tail(24*lookback_days)
            
            if dfs:
                panel = to_panel(dfs)
                mkt_df = compute_market_features(panel, cfg["market_basket"])
                mkt_gates = add_regime_gates(mkt_df, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"])
                
                feats = load_features(ts_sig, cfg["data_root"])
                if feats is not None:
                    # Align symbols
                    sample_feat = next(iter(feats.values()))
                    valid_syms = sorted(set(sample_feat.columns) & set(panel["close"].columns) & set(train_syms))
                    panel = {k: v[valid_syms] for k, v in panel.items() if isinstance(v, pd.DataFrame)}
                    
                    with Timer("Specialist Training"):
                        specialist_models = train_specialist_models(panel, feats, mkt_gates, cfg, valid_syms, ts_sig)
                    
                    # Merge into bundle
                    trained_bundle["specialist_models"] = specialist_models
                else:
                    tprint("WARNING: Features not found, skipping Specialist training")
            else:
                tprint("WARNING: No panel data, skipping Specialist training")
        except Exception as e:
            tprint(f"WARNING: Specialist training failed: {e}")
            import traceback
            traceback.print_exc()


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

    # Generate training report
    try:
        report_path = generate_training_report(
            run_id=run_id,
            cfg=cfg,
            bundle=bundle or {},
            datasets=datasets or {},
            specialist_models=bundle.get("specialist_models") if bundle else None,
        )
        tprint(f"Training report saved to {report_path}")
    except Exception as e:
        tprint(f"WARNING: Failed to generate training report: {e}")

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

    # Generate risk optimization report
    try:
        run_id = run_ts.strftime("%Y%m%d_%H%M%S")
        granular = best_risk.get("granular_risk", {}) if isinstance(best_risk, dict) else {}
        report_path = generate_risk_report(
            run_id=run_id,
            cfg=cfg,
            granular_risk=granular,
        )
        tprint(f"Risk optimization report saved to {report_path}")
    except Exception as e:
        tprint(f"WARNING: Failed to generate risk report: {e}")

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
        max_concurrent = int(cfg.get("max_concurrent_trades", 5))
        max_portfolio_weight = float(cfg.get("max_portfolio_weight", 0.25))

        # Daily risk budget: per-specialist and total daily caps
        max_daily_per_specialist = int(cfg.get("max_daily_per_specialist", 8))
        max_daily_total = int(cfg.get("max_daily_total", 25))

        # Drawdown-based regime throttle parameters
        throttle_lookback = int(cfg.get("throttle_lookback_trades", 20))
        throttle_dd_thr = float(cfg.get("throttle_dd_threshold", -0.02))  # cumPnL drawdown trigger
        throttle_factor = float(cfg.get("throttle_sizing_factor", 0.5))   # reduce sizing to 50%

        from collections import defaultdict
        daily_bucket_counts = defaultdict(lambda: defaultdict(int))  # date -> bucket -> count
        daily_total_counts = defaultdict(int)  # date -> total count

        for t in ts_list:
            # --- Regime throttle: check recent closed-trade drawdown ---
            size_mult = 1.0
            if len(trades) >= throttle_lookback:
                recent_pnls = np.array([tr["pnl"] for tr in trades[-throttle_lookback:]], dtype=np.float64)
                cum = np.cumsum(recent_pnls)
                peak = np.maximum.accumulate(cum)
                dd = cum - peak
                if dd[-1] < throttle_dd_thr:
                    size_mult = throttle_factor

            # Count currently open trades and their total weight at this timestamp
            open_trades = [tr for tr in trades if pd.Timestamp(tr["entry_ts"]) <= t < pd.Timestamp(tr["exit_ts"])]
            open_count = len(open_trades)
            open_weight = sum(abs(tr.get("weight", 0.0)) for tr in open_trades)
            remaining_slots = max(0, max_concurrent - open_count)
            remaining_weight = max(0.0, max_portfolio_weight - open_weight)
            orders = generate_hourly_signals(t, feats, mkt_gates, bundle, local_risk, cfg, p_exh_hist, [])
            orders = orders[:remaining_slots]  # cap to available slots
            # Apply regime throttle to order weights
            if size_mult < 1.0:
                for o in orders:
                    o["weight"] = o.get("weight", 0.0) * size_mult
            # Further cap by portfolio weight
            capped_orders = []
            for o in orders:
                w = abs(o.get("weight", 0.0))
                if w <= remaining_weight:
                    capped_orders.append(o)
                    remaining_weight -= w
            orders = capped_orders

            # --- Daily concentration controls ---
            trade_date = t.date()
            budget_filtered = []
            for o in orders:
                bucket = f"{o['side'].upper()}_{o['dom'].upper()}"
                if daily_total_counts[trade_date] >= max_daily_total:
                    break
                if daily_bucket_counts[trade_date][bucket] >= max_daily_per_specialist:
                    continue
                budget_filtered.append(o)
            orders = budget_filtered
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
                matched_key = None
                for risk_key in risk_keys:
                    if risk_key in granular:
                        rp = granular[risk_key]
                        matched_key = risk_key
                        break
                if matched_key is None:
                    tprint(f"  WARNING: No granular risk for keys {risk_keys}, falling back to global cfg")

                k_sl = rp.get("k_sl", cfg["risk_k_sl"])
                k_ts = rp.get("k_trail_start", cfg["risk_k_trail_start"])
                k_td = rp.get("k_trail_dist", cfg["risk_k_trail_dist"])

                # Extract optimized trailing-profit params
                tp_mult = rp.get("tp_mult", cfg.get("tp_mult"))
                sl_mult = rp.get("sl_mult", cfg.get("sl_mult"))
                trail_mult = rp.get("trail_mult", cfg.get("trail_mult", 0.25))

                sc_scale = rp.get("score_scale", 0.0)
                adj = (1.0 + sc_scale * abs(score))

                temp_cfg = cfg.copy()
                temp_cfg["risk_k_sl"] = k_sl * adj
                temp_cfg["risk_k_trail_start"] = k_ts
                temp_cfg["risk_k_trail_dist"] = k_td

                if tp_mult is not None: temp_cfg["tp_mult"] = tp_mult
                if sl_mult is not None: temp_cfg["sl_mult"] = sl_mult
                temp_cfg["trail_mult"] = trail_mult

                # Per-bucket profit-protection params (absolute % of price)
                for pp_key in ("be_threshold_pct", "profit_lock_pct", "profit_lock_amount", "giveback_pct", "max_loss_pct"):
                    if pp_key in rp:
                        temp_cfg[pp_key] = rp[pp_key]

                # Per-bucket vol scaling params (from risk optimization)
                if "vol_lo" in rp: temp_cfg["vol_lo"] = rp["vol_lo"]
                if "vol_hi" in rp: temp_cfg["vol_hi"] = rp["vol_hi"]
                if "vol_z_max" in rp: temp_cfg["vol_z_max"] = rp["vol_z_max"]

                # Per-bucket max hold hours (from risk optimization, default 24)
                hold_hours = int(rp.get("max_hold_hours", 24))

                # Initialize CCXT exchange for 15m precision if enabled
                exchange = None
                if cfg.get("use_15m_precision", False):
                    try:
                        import ccxt
                        exchange = ccxt.binance()
                    except Exception as e:
                        tprint(f"WARNING: Failed to initialize CCXT exchange: {e}")

                ret, exit_ts, reason, trade_extras = simulate_trade_hourly(
                    o_s[sym], h_s[sym], l_s[sym], c_s[sym], atr_s[sym],
                    entry_ts, entry_px, side, temp_cfg, max_hold_hours=hold_hours,
                    exchange=exchange, symbol=sym if "/" in sym else sym.replace("USDT", "/USDT")  # CCXT format
                )
                net_ret = ret - (2.0 * fee_rate)
                pnl = net_ret * weight
                
                # Store risk parameters + MAE/MFE for aggregate statistics
                bucket_label = f"{side.upper()}_{dom.upper()}"
                trade_record = {
                    "entry_ts": entry_ts, "symbol": sym, "side": side, "dom": dom,
                    "bucket": bucket_label,
                    "score": score, "weight": weight, "ret": net_ret, "pnl": pnl,
                    "gross_ret": ret, "exit_ts": exit_ts, "reason": reason,
                    "sl_mult": temp_cfg.get("sl_mult", 0.5),
                    "tp_mult": temp_cfg.get("tp_mult", 1.0),
                    "trail_mult": temp_cfg.get("trail_mult", 0.25),
                    "entry_px": entry_px,
                    "atr": float(atr_s[sym].loc[entry_ts]) if entry_ts in atr_s[sym].index else 0.02,
                    "sl_pct": trade_extras.get("sl_pct", 0.0),
                    "tp_pct": trade_extras.get("tp_pct", 0.0),
                    "mae_pct": trade_extras.get("mae_pct", 0.0),
                    "mfe_pct": trade_extras.get("mfe_pct", 0.0),
                    "bars_to_mfe": trade_extras.get("bars_to_mfe", 0),
                    "exit_stage": trade_extras.get("exit_stage", 0),
                }
                # Add regime context for diagnostic reporting
                if t in mkt_gates.index:
                    mrk = mkt_gates.loc[t]
                    trade_record["G_VOL"] = int(mrk.get("G_VOL", 0)) if "G_VOL" in mrk.index else 0
                    trade_record["G_TREND"] = int(mrk.get("G_TREND", 0)) if "G_TREND" in mrk.index else 0
                    trade_record["mkt_rv"] = float(mrk.get("mkt_rv", 0.0)) if "mkt_rv" in mrk.index else 0.0
                    trade_record["mkt_ret24h"] = float(mrk.get("mkt_ret24h", 0.0)) if "mkt_ret24h" in mrk.index else 0.0
                trades.append(trade_record)
                # Update daily concentration counters
                daily_bucket_counts[trade_date][bucket_label] += 1
                daily_total_counts[trade_date] += 1
        
        # Log aggregate TP/SL statistics (using actual barrier-scaled distances from engine)
        if trades:
            avg_sl_pct = np.mean([t["sl_pct"] * 100 for t in trades])
            avg_tp_pct = np.mean([t["tp_pct"] * 100 for t in trades])
            avg_trail_pct = np.mean([t["trail_mult"] * t["sl_pct"] * 100 for t in trades])  # trail ≈ trail_mult * barrier
            avg_mae = np.mean([t["mae_pct"] * 100 for t in trades])
            avg_mfe = np.mean([t["mfe_pct"] * 100 for t in trades])
            
            tprint(f"\n  TP/SL Statistics ({len(trades)} trades) [actual barrier-scaled]:")
            tprint(f"    Avg SL:    {avg_sl_pct:.2f}%")
            tprint(f"    Avg TP:    {avg_tp_pct:.2f}%")
            tprint(f"    Avg Trail: {avg_trail_pct:.2f}%")
            tprint(f"    Avg MAE:   {avg_mae:.2f}%")
            tprint(f"    Avg MFE:   {avg_mfe:.2f}%\n")
        
        return trades

    split = max(24, int(len(valid_ts) * 0.2))  # 20% for signal calibration, 80% for OOS test
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

        def _center_scale(arr, channel_name=""):
            """Robust scaling: median / IQR with winsorization to [q05, q95]."""
            v = np.asarray(arr, dtype=np.float64)
            if v.size == 0:
                return 0.0, 1.0
            # Winsorize to [q05, q95] to tame heavy tails (esp. LONG_MR)
            q05 = float(np.quantile(v, 0.05))
            q95 = float(np.quantile(v, 0.95))
            v_w = np.clip(v, q05, q95)
            c = float(np.median(v_w))
            # IQR-based scale (robust to outliers)
            q25 = float(np.quantile(v_w, 0.25))
            q75 = float(np.quantile(v_w, 0.75))
            s = q75 - q25  # IQR
            min_meaningful_scale = 0.001
            if s < min_meaningful_scale:
                tprint(f"  ScoreScale WARNING: {channel_name} has degenerate IQR "
                       f"({s:.2e}). Disabling normalization (center=0, scale=1).")
                return 0.0, 1.0
            n_clipped = int(np.sum(arr < q05) + np.sum(arr > q95))
            if n_clipped > 0:
                tprint(f"  ScoreScale: {channel_name} winsorized {n_clipped}/{len(arr)} outliers "
                       f"to [{q05:.4f}, {q95:.4f}]")
            return c, s

        lmr = side_all[side_all["side_key"] == "long"]["score_mr"].values
        smr = side_all[side_all["side_key"] == "short"]["score_mr"].values
        ltf = side_all[side_all["side_key"] == "long"]["score_tf"].values
        stf = side_all[side_all["side_key"] == "short"]["score_tf"].values

        lmr_c, lmr_s = _center_scale(lmr, "long_mr")
        smr_c, smr_s = _center_scale(smr, "short_mr")
        ltf_c, ltf_s = _center_scale(ltf, "long_tf")
        stf_c, stf_s = _center_scale(stf, "short_tf")

        score_scale_params = {
            "long_mr_center": lmr_c, "long_mr_scale": lmr_s,
            "short_mr_center": smr_c, "short_mr_scale": smr_s,
            "long_tf_center": ltf_c, "long_tf_scale": ltf_s,
            "short_tf_center": stf_c, "short_tf_scale": stf_s,
        }
        tprint(f"Score scale params: lmr=({lmr_c:.4f},{lmr_s:.4f}) smr=({smr_c:.4f},{smr_s:.4f}) "
               f"ltf=({ltf_c:.4f},{ltf_s:.4f}) stf=({stf_c:.4f},{stf_s:.4f})")
        # Log raw score distributions for diagnostics
        for name, arr in [("long_mr", lmr), ("short_mr", smr), ("long_tf", ltf), ("short_tf", stf)]:
            if len(arr) > 0:
                tprint(f"  {name} scores: n={len(arr)}, mean={np.mean(arr):.6f}, std={np.std(arr):.6f}, "
                       f"q10={np.quantile(arr,0.1):.6f}, q50={np.quantile(arr,0.5):.6f}, q90={np.quantile(arr,0.9):.6f}")
    else:
        score_scale_params = {}

    thr_long_grid = [0.00, 0.02]
    thr_short_grid = [0.00]
    x0_grid = [0.5, 0.7, 0.9]
    k_grid = [2.0, 4.0]

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

        # Duration & frequency diagnostics
        ts_min = pd.Timestamp(df_t["entry_ts"].min())
        ts_max = pd.Timestamp(df_t["entry_ts"].max())
        n_days = max(1, (ts_max - ts_min).total_seconds() / 86400)
        tprint(f"--- OOS Period: {ts_min.date()} to {ts_max.date()} ({n_days:.0f} days) ---")
        tprint(f"  Total trades: {len(df_t)}, Trades/day: {len(df_t)/n_days:.1f}")

        tprint("--- OOS Breakdown ---")
        for side in ["long", "short"]:
            df_s = df_t[df_t["side"] == side]
            if not df_s.empty:
                s_pnl = df_s["pnl"].sum()
                s_wr = (df_s["pnl"] > 0).mean()
                tprint(f"  {side.upper()}: Trades={len(df_s)} ({len(df_s)/n_days:.1f}/day), PnL={s_pnl:.4f}, WinRate={s_wr:.2f}")
        for dom in ["mr", "tf"]:
            df_d = df_t[df_t["dom"] == dom]
            if not df_d.empty:
                d_pnl = df_d["pnl"].sum()
                d_wr = (df_d["pnl"] > 0).mean()
                tprint(f"  {dom.upper()}: Trades={len(df_d)} ({len(df_d)/n_days:.1f}/day), PnL={d_pnl:.4f}, WinRate={d_wr:.2f}")

        # Compute hold duration for all trades
        df_t["_entry"] = pd.to_datetime(df_t["entry_ts"])
        df_t["_exit"] = pd.to_datetime(df_t["exit_ts"])
        df_t["_hold_h"] = (df_t["_exit"] - df_t["_entry"]).dt.total_seconds() / 3600.0

        # --- Per-bucket deep diagnostics ---
        tprint("=" * 70)
        tprint("PER-BUCKET DIAGNOSTICS")
        tprint("=" * 70)
        for side in ["long", "short"]:
            for dom in ["mr", "tf"]:
                df_sd = df_t[(df_t["side"] == side) & (df_t["dom"] == dom)]
                if df_sd.empty:
                    continue
                bucket = f"{side}_{dom}"
                n = len(df_sd)
                sd_pnl = df_sd["pnl"].sum()
                sd_wr = (df_sd["pnl"] > 0).mean()
                avg_pnl = sd_pnl / n

                # Sortino ratio for this bucket
                rets = df_sd["pnl"].values
                neg = rets[rets < 0]
                down_std = np.sqrt(np.mean(neg ** 2)) if len(neg) > 0 else 1e-9
                bucket_sortino = np.mean(rets) / down_std if down_std > 1e-9 else 0.0

                # Max drawdown for this bucket (sequential equity curve)
                eq = np.cumsum(rets)
                running_max = np.maximum.accumulate(eq)
                dd = eq - running_max
                bucket_mdd = float(dd.min()) if len(dd) > 0 else 0.0

                # Profit factor
                gross_win = float(rets[rets > 0].sum()) if (rets > 0).any() else 0.0
                gross_loss = float(abs(rets[rets < 0].sum())) if (rets < 0).any() else 1e-9
                pf = gross_win / gross_loss

                tprint(f"\n--- {bucket.upper()} (n={n}, {n/n_days:.1f}/day) ---")
                tprint(f"  PnL={sd_pnl:.4f}  AvgPnL={avg_pnl:.6f}  WR={sd_wr:.2f}  Sortino={bucket_sortino:.3f}  MaxDD={bucket_mdd:.4f}  PF={pf:.2f}")

                # Win/loss asymmetry
                wins = df_sd[df_sd["pnl"] > 0]
                losses = df_sd[df_sd["pnl"] <= 0]
                avg_win = float(wins["pnl"].mean()) if len(wins) > 0 else 0.0
                avg_loss = float(losses["pnl"].mean()) if len(losses) > 0 else 0.0
                payoff = abs(avg_win / avg_loss) if abs(avg_loss) > 1e-9 else 0.0
                avg_win_ret = float(wins["gross_ret"].mean()) if len(wins) > 0 else 0.0
                avg_loss_ret = float(losses["gross_ret"].mean()) if len(losses) > 0 else 0.0
                tprint(f"  Win/Loss: AvgWin={avg_win:.6f} ({avg_win_ret:.4f} ret)  AvgLoss={avg_loss:.6f} ({avg_loss_ret:.4f} ret)  Payoff={payoff:.2f}")

                # Exit reason breakdown per bucket
                if "reason" in df_sd.columns:
                    reasons = df_sd["reason"].value_counts()
                    parts = []
                    for r in sorted(reasons.index):
                        r_df = df_sd[df_sd["reason"] == r]
                        r_pnl = r_df["pnl"].sum()
                        r_wr = (r_df["pnl"] > 0).mean()
                        parts.append(f"{r}:{len(r_df)}({r_pnl:+.4f}, WR={r_wr:.2f})")
                    tprint(f"  Exits: {' | '.join(parts)}")

                # Hold duration stats
                hold = df_sd["_hold_h"]
                tprint(f"  Hold(h): mean={hold.mean():.1f}  med={hold.median():.1f}  min={hold.min():.0f}  max={hold.max():.0f}")
                # Hold duration for wins vs losses
                if len(wins) > 0 and len(losses) > 0:
                    tprint(f"  Hold wins={wins['_hold_h'].mean():.1f}h  Hold losses={losses['_hold_h'].mean():.1f}h")

                # Score distribution
                sc = df_sd["score"].abs()
                tprint(f"  |Score|: mean={sc.mean():.3f}  med={sc.median():.3f}  q10={sc.quantile(0.1):.3f}  q90={sc.quantile(0.9):.3f}")

                # Spearman(|score|, ret) — key monotonicity diagnostic
                from scipy.stats import spearmanr
                if len(sc) >= 5:
                    sp_corr, sp_pval = spearmanr(sc.values, df_sd["ret"].values)
                    tprint(f"  Spearman(|score|, ret): {sp_corr:+.3f} (p={sp_pval:.3f})"
                           f"{'  *** NEGATIVE = conviction paradox ***' if sp_corr < -0.05 else ''}")

                # Survival metric: % of trades that reach trailing stop activation
                if "reason" in df_sd.columns:
                    n_trail = (df_sd["reason"] == "trailing_stop").sum()
                    n_sl = (df_sd["reason"] == "stop_loss").sum()
                    survival_rate = n_trail / max(1, n_trail + n_sl)
                    tprint(f"  Survival-to-trail: {survival_rate:.1%} ({n_trail} trail / {n_sl} SL)")

                # Score vs outcome: high-conviction vs low-conviction
                sc_med = sc.median()
                hi_conv = df_sd[sc >= sc_med]
                lo_conv = df_sd[sc < sc_med]
                if len(hi_conv) > 0 and len(lo_conv) > 0:
                    tprint(f"  Hi-conv(|s|>={sc_med:.3f}): n={len(hi_conv)} PnL={hi_conv['pnl'].sum():.4f} WR={(hi_conv['pnl']>0).mean():.2f}")
                    tprint(f"  Lo-conv(|s|< {sc_med:.3f}): n={len(lo_conv)} PnL={lo_conv['pnl'].sum():.4f} WR={(lo_conv['pnl']>0).mean():.2f}")
                    # Survival comparison by conviction
                    if "reason" in df_sd.columns:
                        hi_surv = (hi_conv["reason"] == "trailing_stop").sum() / max(1, len(hi_conv))
                        lo_surv = (lo_conv["reason"] == "trailing_stop").sum() / max(1, len(lo_conv))
                        tprint(f"  Trail-survival: Hi-conv={hi_surv:.1%}  Lo-conv={lo_surv:.1%}")

                # ========================================================================
                # CONFIDENCE QUARTILE ANALYSIS (ENHANCED)
                # ========================================================================
                try:
                    sc_abs = sc.abs()
                    df_sd = df_sd.copy()
                    df_sd["confidence_bin"] = pd.qcut(sc_abs, q=4, labels=["Q1_Low", "Q2", "Q3", "Q4_High"], duplicates='drop')
                    
                    tprint("  Confidence Calibration:")
                    tprint(f"    {'Quartile':<10} {'N':>4} {'WR':>5} {'PnL':>8} {'AvgRet':>8} {'MFE%':>6} {'MAE%':>6} {'MFE/MAE':>7} {'Capture':>7} {'Trail%':>6}")
                    for bin_label in ["Q1_Low", "Q2", "Q3", "Q4_High"]:
                        bt = df_sd[df_sd["confidence_bin"] == bin_label]
                        if len(bt) == 0:
                            continue
                        b_wr = (bt["pnl"] > 0).mean()
                        b_pnl = bt["pnl"].sum()
                        b_ret = bt["ret"].mean()
                        b_mfe = bt["mfe_pct"].mean() * 100
                        b_mae = bt["mae_pct"].mean() * 100
                        b_ratio = b_mfe / max(b_mae, 0.01)
                        # Capture ratio: avg gross_ret / avg MFE for winners
                        bt_w = bt[bt["pnl"] > 0]
                        b_cap = (bt_w["gross_ret"].mean() / max(bt_w["mfe_pct"].mean(), 1e-9)) if len(bt_w) > 0 else 0.0
                        # Trail survival %
                        b_trail = (bt["reason"] == "trailing_stop").mean() * 100 if "reason" in bt.columns else 0.0
                        tprint(f"    {bin_label:<10} {len(bt):>4} {b_wr:>5.2f} {b_pnl:>+8.4f} {b_ret:>+8.4f} {b_mfe:>6.2f} {b_mae:>6.2f} {b_ratio:>7.2f} {b_cap:>7.2f} {b_trail:>5.1f}%")
                    # Exit reason distribution per quartile
                    if "reason" in df_sd.columns:
                        tprint("  Exit Reasons by Confidence:")
                        for bin_label in ["Q1_Low", "Q2", "Q3", "Q4_High"]:
                            bt = df_sd[df_sd["confidence_bin"] == bin_label]
                            if len(bt) == 0:
                                continue
                            rc = bt["reason"].value_counts()
                            parts = [f"{r}:{c}" for r, c in rc.items()]
                            tprint(f"    {bin_label}: {' '.join(parts)}")
                except Exception as e:
                    tprint(f"  Warning: Confidence calibration failed: {e}")

                # ========================================================================
                # REGIME ANALYSIS: G_VOL × G_TREND
                # ========================================================================
                if "G_VOL" in df_sd.columns and "G_TREND" in df_sd.columns:
                    try:
                        tprint("  Regime Analysis (G_VOL × G_TREND):")
                        tprint(f"    {'Regime':<20} {'N':>4} {'WR':>5} {'PnL':>8} {'MFE%':>6} {'MAE%':>6} {'MFE/MAE':>7} {'Capture':>7}")
                        for gv in [0, 1]:
                            for gt in [0, 1]:
                                regime = df_sd[(df_sd["G_VOL"] == gv) & (df_sd["G_TREND"] == gt)]
                                if len(regime) < 3:
                                    continue
                                label = f"VOL={'Hi' if gv else 'Lo'}_TREND={'Hi' if gt else 'Lo'}"
                                r_wr = (regime["pnl"] > 0).mean()
                                r_pnl = regime["pnl"].sum()
                                r_mfe = regime["mfe_pct"].mean() * 100
                                r_mae = regime["mae_pct"].mean() * 100
                                r_ratio = r_mfe / max(r_mae, 0.01)
                                r_w = regime[regime["pnl"] > 0]
                                r_cap = (r_w["gross_ret"].mean() / max(r_w["mfe_pct"].mean(), 1e-9)) if len(r_w) > 0 else 0.0
                                tprint(f"    {label:<20} {len(regime):>4} {r_wr:>5.2f} {r_pnl:>+8.4f} {r_mfe:>6.2f} {r_mae:>6.2f} {r_ratio:>7.2f} {r_cap:>7.2f}")
                    except Exception as e:
                        tprint(f"  Warning: Regime analysis failed: {e}")


                # Weight/sizing stats
                wt = df_sd["weight"]
                tprint(f"  Weight: mean={wt.mean():.4f}  med={wt.median():.4f}  min={wt.min():.4f}  max={wt.max():.4f}")

                # Temporal half-split: is performance degrading?
                mid = len(df_sd) // 2
                if mid > 5:
                    first_half = df_sd.iloc[:mid]
                    second_half = df_sd.iloc[mid:]
                    fh_pnl = first_half["pnl"].sum()
                    sh_pnl = second_half["pnl"].sum()
                    fh_wr = (first_half["pnl"] > 0).mean()
                    sh_wr = (second_half["pnl"] > 0).mean()
                    tprint(f"  1st half: n={len(first_half)} PnL={fh_pnl:.4f} WR={fh_wr:.2f}  |  2nd half: n={len(second_half)} PnL={sh_pnl:.4f} WR={sh_wr:.2f}")

                # Top losing symbols
                sym_pnl = df_sd.groupby("symbol")["pnl"].agg(["sum", "count"])
                sym_pnl = sym_pnl.sort_values("sum")
                worst_3 = sym_pnl.head(3)
                best_3 = sym_pnl.tail(3).iloc[::-1]
                w_parts = [f"{s}({row['sum']:+.4f}, n={int(row['count'])})" for s, row in worst_3.iterrows()]
                b_parts = [f"{s}({row['sum']:+.4f}, n={int(row['count'])})" for s, row in best_3.iterrows()]
                tprint(f"  Worst syms: {', '.join(w_parts)}")
                tprint(f"  Best syms:  {', '.join(b_parts)}")

        # --- MAE/MFE DIAGNOSTIC REPORT ---
        if "mae_pct" in df_t.columns and "mfe_pct" in df_t.columns:
            tprint("\n" + "=" * 70)
            tprint("MAE / MFE DIAGNOSTIC REPORT")
            tprint("=" * 70)

            # Global MAE/MFE
            tprint(f"\n--- Global MAE/MFE (n={len(df_t)}) ---")
            tprint(f"  MAE: mean={df_t['mae_pct'].mean()*100:.2f}%  med={df_t['mae_pct'].median()*100:.2f}%  q90={df_t['mae_pct'].quantile(0.9)*100:.2f}%")
            tprint(f"  MFE: mean={df_t['mfe_pct'].mean()*100:.2f}%  med={df_t['mfe_pct'].median()*100:.2f}%  q90={df_t['mfe_pct'].quantile(0.9)*100:.2f}%")
            tprint(f"  MFE/MAE ratio: {df_t['mfe_pct'].mean() / max(df_t['mae_pct'].mean(), 1e-9):.2f}")

            # Per-bucket MAE/MFE
            bucket_col = "bucket" if "bucket" in df_t.columns else None
            if bucket_col is None:
                df_t["bucket"] = df_t["side"].str.upper() + "_" + df_t["dom"].str.upper()
                bucket_col = "bucket"

            for bkt in sorted(df_t[bucket_col].unique()):
                df_b = df_t[df_t[bucket_col] == bkt]
                if len(df_b) < 3:
                    continue
                tprint(f"\n  --- {bkt} (n={len(df_b)}) ---")
                tprint(f"    MAE: mean={df_b['mae_pct'].mean()*100:.2f}%  med={df_b['mae_pct'].median()*100:.2f}%")
                tprint(f"    MFE: mean={df_b['mfe_pct'].mean()*100:.2f}%  med={df_b['mfe_pct'].median()*100:.2f}%")
                ratio = df_b['mfe_pct'].mean() / max(df_b['mae_pct'].mean(), 1e-9)
                tprint(f"    MFE/MAE ratio: {ratio:.2f}  {'GOOD (>1.5)' if ratio > 1.5 else 'WEAK (<1.5) — entries or stops need work'}")

                # MAE/MFE by exit reason
                for reason in sorted(df_b["reason"].dropna().unique()):
                    df_br = df_b[df_b["reason"] == reason]
                    if len(df_br) < 2:
                        continue
                    tprint(f"    {reason}: n={len(df_br)}  MAE={df_br['mae_pct'].mean()*100:.2f}%  MFE={df_br['mfe_pct'].mean()*100:.2f}%  bars_to_mfe={df_br['bars_to_mfe'].mean():.0f}")

                # Key diagnostic: losers that had meaningful MFE (exit/stop problem)
                losers = df_b[df_b["pnl"] <= 0]
                if len(losers) > 0:
                    losers_with_mfe = losers[losers["mfe_pct"] > 0.005]  # >0.5% MFE
                    pct_losers_had_mfe = len(losers_with_mfe) / len(losers)
                    tprint(f"    Losers with MFE>0.5%: {pct_losers_had_mfe:.0%} ({len(losers_with_mfe)}/{len(losers)})"
                           f"{'  *** EXIT PROBLEM: many losers saw profit first ***' if pct_losers_had_mfe > 0.4 else ''}")
                    if len(losers_with_mfe) > 0:
                        tprint(f"      Avg MFE of those losers: {losers_with_mfe['mfe_pct'].mean()*100:.2f}%")

                # Key diagnostic: winners — how much MFE was captured?
                winners = df_b[df_b["pnl"] > 0]
                if len(winners) > 0:
                    capture_ratio = winners["gross_ret"].mean() / max(winners["mfe_pct"].mean(), 1e-9)
                    tprint(f"    Winner capture ratio (ret/MFE): {capture_ratio:.2f}"
                           f"{'  *** LOW CAPTURE: trailing too loose ***' if capture_ratio < 0.3 else ''}")

        # --- PnL RECONCILIATION TABLE ---
        tprint("\n" + "=" * 70)
        tprint("PnL RECONCILIATION TABLE")
        tprint("=" * 70)

        # All units in portfolio-weighted PnL (pnl = net_ret * weight)
        total_gross_profit = float(df_t.loc[df_t["pnl"] > 0, "pnl"].sum())
        total_gross_loss = float(df_t.loc[df_t["pnl"] <= 0, "pnl"].sum())
        total_net_pnl = total_gross_profit + total_gross_loss
        recon_pf = total_gross_profit / abs(total_gross_loss) if abs(total_gross_loss) > 1e-9 else float("inf")

        tprint(f"\n  Total Gross Profit:  {total_gross_profit:+.6f}  (portfolio-weighted PnL)")
        tprint(f"  Total Gross Loss:   {total_gross_loss:+.6f}")
        tprint(f"  Net PnL:            {total_net_pnl:+.6f}")
        tprint(f"  Profit Factor:      {recon_pf:.3f}")

        # Fee impact
        total_fees = float((2.0 * fee_rate * df_t["weight"]).sum())
        gross_pnl_before_fees = float(df_t["gross_ret"].mul(df_t["weight"]).sum())
        tprint(f"\n  Gross PnL (pre-fee): {gross_pnl_before_fees:+.6f}")
        tprint(f"  Total Fees:          {total_fees:+.6f}")
        tprint(f"  Net PnL (post-fee):  {gross_pnl_before_fees - total_fees:+.6f}")

        # Per-bucket contribution (same units)
        tprint(f"\n  --- Per-Bucket Contribution (portfolio-weighted PnL) ---")
        tprint(f"  {'Bucket':<15} {'N':>5} {'GrossProfit':>12} {'GrossLoss':>12} {'NetPnL':>12} {'PF':>6} {'WR':>6} {'AvgWin':>10} {'AvgLoss':>10}")
        for bkt in sorted(df_t[bucket_col].unique()):
            df_b = df_t[df_t[bucket_col] == bkt]
            b_gp = float(df_b.loc[df_b["pnl"] > 0, "pnl"].sum())
            b_gl = float(df_b.loc[df_b["pnl"] <= 0, "pnl"].sum())
            b_net = b_gp + b_gl
            b_pf = b_gp / abs(b_gl) if abs(b_gl) > 1e-9 else float("inf")
            b_wr = (df_b["pnl"] > 0).mean()
            b_aw = float(df_b.loc[df_b["pnl"] > 0, "pnl"].mean()) if (df_b["pnl"] > 0).any() else 0.0
            b_al = float(df_b.loc[df_b["pnl"] <= 0, "pnl"].mean()) if (df_b["pnl"] <= 0).any() else 0.0
            tprint(f"  {bkt:<15} {len(df_b):>5} {b_gp:>+12.6f} {b_gl:>+12.6f} {b_net:>+12.6f} {b_pf:>6.2f} {b_wr:>6.2f} {b_aw:>+10.6f} {b_al:>+10.6f}")

        # Units sanity check
        avg_win_ret = float(df_t.loc[df_t["pnl"] > 0, "ret"].mean()) if (df_t["pnl"] > 0).any() else 0.0
        avg_loss_ret = float(df_t.loc[df_t["pnl"] <= 0, "ret"].mean()) if (df_t["pnl"] <= 0).any() else 0.0
        avg_weight = float(df_t["weight"].mean())
        tprint(f"\n  --- Units Check ---")
        tprint(f"  Avg Win (ret space):  {avg_win_ret:+.4f}")
        tprint(f"  Avg Loss (ret space): {avg_loss_ret:+.4f}")
        tprint(f"  Avg Weight:           {avg_weight:.4f}")
        tprint(f"  Implied AvgWin PnL:   {avg_win_ret * avg_weight:+.6f}  (should ~ match AvgWin above)")
        tprint(f"  Implied AvgLoss PnL:  {avg_loss_ret * avg_weight:+.6f}  (should ~ match AvgLoss above)")

        # --- Global exit reason breakdown ---
        if "reason" in df_t.columns:
            tprint("\n--- Exit Reasons (global) ---")
            for reason in sorted(df_t["reason"].dropna().unique()):
                df_r = df_t[df_t["reason"] == reason]
                r_wr = (df_r["pnl"] > 0).mean()
                r_hold = df_r["_hold_h"].mean()
                tprint(f"  {reason}: n={len(df_r)} ({len(df_r)/len(df_t)*100:.0f}%), PnL={df_r['pnl'].sum():.4f}, WR={r_wr:.2f}, AvgHold={r_hold:.1f}h")

        # Daily concentration check
        df_t["_date"] = df_t["_entry"].dt.date
        daily_counts = df_t.groupby("_date").size()
        tprint(f"\n--- Daily Concentration: max={daily_counts.max()}/day, mean={daily_counts.mean():.1f}/day ---")
        if daily_counts.max() > 20:
            worst_day = daily_counts.idxmax()
            df_wd = df_t[df_t["_date"] == worst_day]
            tprint(f"  Worst day {worst_day}: {len(df_wd)} trades, PnL={df_wd['pnl'].sum():.4f}")

        # Per-bucket daily concentration
        tprint("  Per-bucket daily max:")
        for bkt in sorted(df_t[bucket_col].unique()):
            df_b = df_t[df_t[bucket_col] == bkt]
            bkt_daily = df_b.groupby("_date").size()
            tprint(f"    {bkt}: max={bkt_daily.max()}/day, mean={bkt_daily.mean():.1f}/day")

        # Weekly PnL trend
        df_t["_week"] = df_t["_entry"].dt.isocalendar().week.astype(int)
        weekly = df_t.groupby("_week").agg(
            n=("pnl", "count"),
            pnl=("pnl", "sum"),
            wr=("pnl", lambda x: (x > 0).mean())
        )
        tprint("--- Weekly PnL ---")
        for wk, row in weekly.iterrows():
            bar = "+" * int(max(0, row["pnl"]) * 500) + "-" * int(max(0, -row["pnl"]) * 500)
            tprint(f"  W{wk:02d}: n={int(row['n']):3d}  PnL={row['pnl']:+.4f}  WR={row['wr']:.2f}  {bar}")

        df_t.drop(columns=["_date", "_entry", "_exit", "_hold_h", "_week", bucket_col], inplace=True, errors="ignore")
        tprint("-----------------------")

    if test_trades:
        df_res = pd.DataFrame(test_trades)
        out_path = os.path.join(cfg["data_root"], "artifacts", run_id, "backtest_results.csv")
        df_res.to_csv(out_path, index=False)
        tprint(f"Detailed results saved to {out_path}")

    # Generate backtest report
    if test_trades:
        try:
            report_path = generate_backtest_report(
                run_id=run_id,
                cfg=cfg,
                trades=test_trades,
                signal_params=best_signal_params,
                fee_rate=fee_rate,
            )
            tprint(f"Backtest report saved to {report_path}")
        except Exception as e:
            tprint(f"WARNING: Failed to generate backtest report: {e}")

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
