import os
import pickle
import pandas as pd
import numpy as np

from extreme_price_movements.utils import tprint, Timer
from extreme_price_movements.data_store import load_features, save_artifact_df, to_panel
from extreme_price_movements.training import generate_label_datasets, generate_exhaustion_history, optimize_risk_params
from extreme_price_movements.features import compute_market_features, add_regime_gates
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

    # Backtest Loop (last 14 days)
    test_days = 14
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
