import os
import pickle
import pandas as pd
import numpy as np

from extreme_price_movements.utils import tprint, Timer
from extreme_price_movements.data_store import load_features, save_features, save_artifact_df, load_artifact_df, to_panel
from extreme_price_movements.training import generate_label_datasets, generate_exhaustion_history, optimize_risk_params, train_models_from_artifacts
from extreme_price_movements.features import compute_market_features, add_regime_gates, compute_features_hourly
from extreme_price_movements.universe import get_training_universe
from extreme_price_movements.engine import simulate_trade_hourly, generate_hourly_signals
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

def run_training_step(ts_sig, cfg):
    """Train all models from label artifacts. Saves trained state to disk."""
    tprint("STEP: MODEL TRAINING START")

    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    datasets = {}

    # 1. Load label artifacts
    tprint("Loading label datasets from artifacts...")

    # Spike anatomy
    df_spike = load_artifact_df(cfg["data_root"], run_id, "labels", "spike_anatomy")
    if df_spike is not None:
        datasets["spike_anatomy"] = df_spike

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

    with open(state_file, "rb") as f:
        state = pickle.load(f)

    bundle = state.get("bundle")
    if not bundle:
        tprint("No model bundle in state.")
        return

    # Need Data for optimization (simulation)
    train_syms = get_training_universe(margin_symbols, cfg, store, ts_sig=ts_sig)
    dfs = {}

    lookback_days = max(90, int(cfg["fetch_years"] * 365))

    for s in train_syms:
        df = store.load(s)
        if not df.empty: dfs[s] = df[df.index <= ts_sig].tail(24*lookback_days)

    if not dfs: return

    panel = to_panel(dfs)
    mkt_df = compute_market_features(panel, cfg["market_basket"])
    mkt_gates = add_regime_gates(mkt_df, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"])
    feats = load_features(ts_sig, cfg["data_root"])

    # We also need p_exh_hist. Load from artifacts?
    # Or re-generate? Re-generation is safer but slower.
    # Let's try loading.
    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    from extreme_price_movements.data_store import load_artifact_df
    p_exh_hist = load_artifact_df(cfg["data_root"], run_id, "labels", "exhaustion_history")

    if p_exh_hist is None:
        tprint("Exhaustion history artifact missing. Regenerating...")
        p_exh_hist = generate_exhaustion_history(panel, feats, mkt_gates, cfg, ts_sig, cfg["train_lookback_hours"], train_syms)

    alpha_models = bundle["alpha_models"]
    best_risk = optimize_risk_params(panel, feats, mkt_gates, cfg, train_syms, ts_sig, p_exh_hist, alpha_models)

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

    # Load Data
    train_syms = get_training_universe(margin_symbols, cfg, store, ts_sig=ts_sig)
    dfs = {}

    lookback_days = max(90, int(cfg["fetch_years"] * 365))

    with Timer("Backtest Data Load"):
        for s in train_syms:
            df = store.load(s)
            if not df.empty: dfs[s] = df[df.index <= ts_sig].tail(24*lookback_days) # enough for features

    if not dfs: return

    panel = to_panel(dfs)
    mkt_df = compute_market_features(panel, cfg["market_basket"])
    mkt_gates = add_regime_gates(mkt_df, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"])
    feats = load_features(ts_sig, cfg["data_root"])

    # Load Exhaustion
    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    from extreme_price_movements.data_store import load_artifact_df
    p_exh_hist = load_artifact_df(cfg["data_root"], run_id, "labels", "exhaustion_history")
    if p_exh_hist is None:
        p_exh_hist = generate_exhaustion_history(panel, feats, mkt_gates, cfg, ts_sig, cfg["train_lookback_hours"], train_syms)

    # Backtest Loop (use OOS holdout window)
    test_days = cfg.get("oos_holdout_days", 14)
    start_ts = ts_sig - pd.Timedelta(days=test_days)
    end_ts = ts_sig - pd.Timedelta(hours=24) # Leave holding room

    valid_ts = [t for t in feats["ret1h"].index if t >= start_ts and t <= end_ts]

    tprint(f"Running backtest over {len(valid_ts)} hours...")

    trades = []

    # Cache price data
    o_s = panel["open"]
    h_s = panel["high"]
    l_s = panel["low"]
    c_s = panel["close"]
    atr_s = feats["atr_pct"]

    risk_conf = model_state.get("risk_params", {})
    bundle = model_state.get("bundle")

    for t in valid_ts:
        orders = generate_hourly_signals(t, feats, mkt_gates, bundle, risk_conf, cfg, p_exh_hist, [])
        for order in orders:
            sym = order["symbol"]
            side = order["side"]
            score = order["score"]
            dom = order["dom"]

            entry_ts = t + pd.Timedelta(hours=1)
            if entry_ts not in o_s.index: continue

            entry_px = float(o_s.loc[entry_ts, sym])

            # Risk logic from execute_hourly
            risk_key = f"risk_{side}_{dom}"
            granular = risk_conf.get("granular_risk", {})
            rp = granular.get(risk_key, {})

            k_sl = rp.get("k_sl", cfg["risk_k_sl"])
            k_ts = rp.get("k_trail_start", cfg["risk_k_trail_start"])
            k_td = rp.get("k_trail_dist", cfg["risk_k_trail_dist"])
            sc_scale = rp.get("score_scale", 0.0)

            adj = (1.0 + sc_scale * abs(score))
            k_sl_adj = k_sl * adj

            temp_cfg = cfg.copy()
            temp_cfg["risk_k_sl"] = k_sl_adj
            temp_cfg["risk_k_trail_start"] = k_ts
            temp_cfg["risk_k_trail_dist"] = k_td

            ret, exit_ts, reason = simulate_trade_hourly(
                o_s[sym], h_s[sym], l_s[sym], c_s[sym], atr_s[sym],
                entry_ts, entry_px, side, temp_cfg, max_hold_hours=24
            )

            trades.append({
                "entry_ts": entry_ts, "symbol": sym, "side": side,
                "ret": ret, "exit_ts": exit_ts, "reason": reason
            })

    if trades:
        df_res = pd.DataFrame(trades)
        win_rate = (df_res["ret"] > 0).mean()
        avg_ret = df_res["ret"].mean()
        total_ret = df_res["ret"].sum()

        tprint(f"Backtest Result: Trades={len(df_res)}, WinRate={win_rate:.2%}, AvgRet={avg_ret:.4%}, Total={total_ret:.4f}")

        # Save results
        out_path = os.path.join(cfg["data_root"], "artifacts", run_id, "backtest_results.csv")
        df_res.to_csv(out_path, index=False)
        tprint(f"Detailed results saved to {out_path}")
    else:
        tprint("No trades generated in backtest.")

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
