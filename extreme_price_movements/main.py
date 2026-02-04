import time
import sys
import pandas as pd
import numpy as np
import uuid

from extreme_price_movements.config import CFG
from extreme_price_movements.utils import tprint, Timer
from extreme_price_movements.universe import refresh_margin_universe_daily, build_fetch_universe, select_live_candidates
from extreme_price_movements.data_store import make_spot_exchange, PartitionedOHLCVStore, to_panel, check_data_health, save_features, load_features
from extreme_price_movements.features import compute_market_features, add_regime_gates, compute_features_hourly
from extreme_price_movements.engine import generate_hourly_signals
from extreme_price_movements.candidates import select_trade_candidates_hourly, entry_price_next_hour_open
from extreme_price_movements.time_utils import get_ts_sig, floor_to_hour, now_utc
from extreme_price_movements.state import StateManager
from extreme_price_movements.metrics import MetricsLogger
from extreme_price_movements.training import select_best_horizon, compute_p_exhaustion_at_t, apply_interaction_toggles, generate_exhaustion_history, optimize_risk_params
from extreme_price_movements.risk import TrailingStop
from extreme_price_movements.optimization_utils import filter_low_variance_assets

def reconcile_state(ex, state):
    tprint("Reconciling state...")
    return True

def train_daily(ts_sig, margin_symbols, cfg, store, ex):
    tprint("DAILY TRAINING START")
    syms_all = build_fetch_universe(margin_symbols, cfg["market_basket"], cfg["fetch_symbols_M"])
    tprint(f"Fetch universe size: {len(syms_all)}")
    # with Timer("Training Data Fetch"):
    #     since = (ts_sig - pd.Timedelta(days=365)).floor("D")
    #     since_ms = int(since.value // 10**6)
    #     count_upd = 0
    #     for s in syms_all:
    #         try:
    #             store.update_symbol(ex, s, since_ms)
    #             count_upd += 1
    #         except Exception: pass
    #     tprint(f"Updated {count_upd}/{len(syms_all)} symbols")

    train_syms = filter_low_variance_assets(store, syms_all, lookback_days=30, threshold_pct=0.40)
    tprint(f"Training universe size (after variance filter): {len(train_syms)}")
    train_syms = sorted(list(set(train_syms).union(set(cfg["market_basket"]))))
    dfs = {}
    for s in train_syms:
        df = store.load(s)
        if not df.empty: dfs[s] = df[df.index <= ts_sig].tail(24*90)

    tprint(f"Loaded data for {len(dfs)}/{len(train_syms)} symbols")
    if not dfs:
        tprint("Training failed: No data.")
        return None
    with Timer("Training Pipeline"):
        panel = to_panel(dfs)
        mkt_df = compute_market_features(panel, cfg["market_basket"])
        tprint("Market features computed")
        mkt_gates = add_regime_gates(mkt_df, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"])
        tprint("Regime gates added")
        tprint("Regime gates added")
        
        # Try load features
        feats = load_features(ts_sig, cfg["data_root"])
        if feats is None:
            feats = compute_features_hourly(panel, mkt_gates, cfg)
            tprint("Hourly features computed")
            save_features(feats, ts_sig, cfg["data_root"])
        else:
            tprint("Loaded features from disk.")
        p_exh_hist = generate_exhaustion_history(panel, feats, mkt_gates, cfg, ts_sig, cfg["train_lookback_hours"], train_syms)
        tprint("Exhaustion history generated")
        trained_bundle = select_best_horizon(panel, feats, mkt_gates, cfg, train_syms, ts_sig, p_exh_hist)
        tprint("Best horizon selected / Models trained")
        alpha_models = trained_bundle["alpha_models"]
        best_risk = optimize_risk_params(panel, feats, mkt_gates, cfg, train_syms, ts_sig, p_exh_hist, alpha_models)
        tprint("Risk params optimized")

        # Return state dict
        new_state = {
            "ts_trained": ts_sig,
            "bundle": trained_bundle,
            "risk_params": best_risk
        }
    tprint("DAILY TRAINING COMPLETE")
    return new_state

def execute_hourly(ts_sig, margin_symbols, cfg, store, ex, state, logger, model_state):
    tprint(f"Entering function: execute_hourly in main.py")
    run_id = str(uuid.uuid4())
    tprint(f"HOURLY EXEC Start: {ts_sig} RunID={run_id}")
    candidates_pool = select_live_candidates(margin_symbols, cfg["market_basket"], pct=0.05)
    tprint(f"Candidates selected: {len(candidates_pool)}")

    current_positions = state.get_positions()
    active_syms = list(current_positions.keys())
    tprint(f"Active positions: {len(active_syms)}")
    # Ensure active symbols fetched
    fetch_syms = sorted(list(set(candidates_pool + active_syms)))

    dfs = {}
    since = (ts_sig - pd.Timedelta(days=90)).floor("D")
    since_ms = int(since.value // 10**6)
    with Timer("Candidate Data Fetch"):
        count_fetch = 0
        for s in fetch_syms:
            try:
                df = store.update_symbol(ex, s, since_ms)
                if not df.empty and df.index.max() >= ts_sig:
                    dfs[s] = df[df.index <= ts_sig].tail(24*90)
                    count_fetch += 1
            except Exception: pass
        tprint(f"Fetched data for {count_fetch}/{len(fetch_syms)} symbols")
    if not dfs:
        tprint("No data available for execution. Exiting.")
        return

    with Timer("Feature Gen (Candidates)"):
        panel = to_panel(dfs)
        mkt_df = compute_market_features(panel, cfg["market_basket"])
        mkt_gates = add_regime_gates(mkt_df, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"])
        feats = compute_features_hourly(panel, mkt_gates, cfg)
        tprint("Features generated")

    if not model_state or not model_state.get("bundle"):
        tprint("No trained models available. Skipping execution.")
        return

    bundle = model_state["bundle"]
    alpha_models = bundle["alpha_models"]
    meta_models = bundle["meta_models"]

    risk_conf = model_state.get("risk_params")
    granular_risk = risk_conf.get("granular_risk", {}) if risk_conf else {}

    o = panel["open"]; h = panel["high"]; l = panel["low"]; c = panel["close"]
    exits_count = 0
    for sym in active_syms:
        if sym not in c.columns or ts_sig not in c.index:
            tprint(f"Warning: {sym} not in data/index for position update")
            continue
        pos = current_positions[sym]
        ts_risk = TrailingStop.from_dict(pos["risk_state"])
        curr_h = float(h.loc[ts_sig, sym]); curr_l = float(l.loc[ts_sig, sym]); curr_c = float(c.loc[ts_sig, sym])
        stopped, exit_px, reason = ts_risk.update(curr_h, curr_l, curr_c)
        if stopped:
            entry_px = pos["entry_px"]; side = pos["side"]
            if reason == "ambiguous_neutral": ret = 0.0
            else: ret = (exit_px / entry_px - 1.0) if side == "long" else (entry_px / exit_px - 1.0)
            logger.log(ts_sig, {"event": "exit", "symbol": sym, "return": ret, "reason": reason})
            state.clear_position(sym)
            tprint(f"EXIT {sym} ({reason}): ret={ret:.4%}")
            exits_count += 1
        else:
            pos["risk_state"] = ts_risk.to_dict()
            state.set_position(sym, pos)

    tprint(f"Position updates complete. Exits: {exits_count}")
    p_exh_cand = generate_exhaustion_history(panel, feats, mkt_gates, cfg, ts_sig, 24, list(dfs.keys()))
    tprint("Exhaustion history for candidates generated")

    target_orders = generate_hourly_signals(
        ts_sig, feats, mkt_gates, bundle, risk_conf, cfg, p_exh_cand, active_syms
    )
    tprint(f"Generated {len(target_orders) if target_orders else 0} signals")

    if target_orders:
        for order in target_orders:
            sym = order["symbol"]
            side = order["side"]
            score = order["score"]
            dom = order["dom"]
            w_alloc = order["weight"]

            if sym not in c.columns: continue
            atr = float(feats["atr_pct"].loc[ts_sig, sym])
            entry_px = float(c.loc[ts_sig, sym])

            # Risk Params Lookup
            # Key: risk_{side}_{dom}
            risk_key = f"risk_{side}_{dom}"
            rp = granular_risk.get(risk_key, {})

            # Apply Score Confidence scaling
            k_sl = rp.get("k_sl", cfg["risk_k_sl"])
            k_ts = rp.get("k_trail_start", cfg["risk_k_trail_start"])
            k_td = rp.get("k_trail_dist", cfg["risk_k_trail_dist"])
            sc_scale = rp.get("score_scale", 0.0)

            # Adjust
            adj = (1.0 + sc_scale * abs(score))
            k_sl_adj = k_sl * adj

            ts_risk = TrailingStop(
                entry_px=entry_px, side=side, atr_val=atr,
                k_sl=k_sl_adj, k_trail_start=k_ts, k_trail_dist=k_td
            )
            pos = {
                "symbol": sym, "side": side, "entry_px": entry_px, "entry_ts": ts_sig.isoformat(),
                "score": float(score), "weight": float(w_alloc), "risk_state": ts_risk.to_dict(), "run_id": run_id
            }
            state.set_position(sym, pos)
            tprint(f"ENTRY {side} {sym} @ {entry_px} (score={score:.4f}, w={w_alloc:.4f}, dom={dom})")
            logger.log(ts_sig, {"event": "entry", "symbol": sym, "side": side, "score": score, "weight": w_alloc, "dom": dom})

    state.set_last_ts_sig(ts_sig)
    n_orders = len(target_orders) if target_orders else 0
    logger.log(ts_sig, {"n_orders": n_orders, "run_id": run_id})
    tprint("HOURLY EXEC COMPLETE")

def run_live_cycle(initial_model_state=None):
    # Maintain state in function scope (for live loop)
    # But usually this script restarts?
    # For robust persistent state, we need to save/load from disk (pickle).
    # But for this refactor, we just keep it in memory for the process life.

    # Initialize state
    tprint(f"Entering function: run_live_cycle in main.py")
    if initial_model_state:
        model_state = initial_model_state
    else:
        model_state = {
            "ts_trained": None,
            "bundle": None,
            "risk_params": None
        }
        # Try load from default file
        import os
        import pickle
        if os.path.exists("model_state.pkl"):
            try:
                with open("model_state.pkl", "rb") as f:
                    model_state = pickle.load(f)
                tprint("Loaded model state from model_state.pkl")
            except Exception as e:
                tprint(f"Failed to load model state: {e}")

    cfg = CFG.copy()
    state = StateManager()
    logger = MetricsLogger()

    # Start loop
    while True:
        try:
            ts_sig = get_ts_sig()
            last_ts = state.get_last_ts_sig()
            tprint(f"Current ts_sig: {ts_sig}")

            if last_ts and ts_sig <= last_ts:
                tprint(f"Already processed {ts_sig}. Waiting...")
                time.sleep(60)
                continue

            ex = make_spot_exchange()
            reconcile_state(ex, state)
            with Timer("Margin universe refresh"): mu = refresh_margin_universe_daily(None, quote="USDT")
            store = PartitionedOHLCVStore(root_dir=cfg["data_root"], timeframe=cfg["timeframe"])

            last_train = model_state["ts_trained"]
            need_train = False
            if last_train is None: need_train = True
            else:
                if ts_sig.floor("D") > last_train.floor("D"): need_train = True

            if need_train:
                new_state = train_daily(ts_sig, mu.symbols, cfg, store, ex)
                if new_state:
                    model_state = new_state

            execute_hourly(ts_sig, mu.symbols, cfg, store, ex, state, logger, model_state)

        except Exception as e:
            tprint(f"CRITICAL ERROR: {e}")
            import traceback; traceback.print_exc()

        tprint("Sleeping 60s...")
        time.sleep(60)

if __name__ == "__main__":
    run_live_cycle()
