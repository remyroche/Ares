import time
import sys
import pandas as pd
import numpy as np
import uuid
import pickle
import os

from extreme_price_movements.config import CFG
from extreme_price_movements.utils import tprint, Timer
from extreme_price_movements.universe import refresh_margin_universe_daily, build_fetch_universe, select_live_candidates
from extreme_price_movements.data_store import make_spot_exchange, PartitionedOHLCVStore, to_panel, check_data_health
from extreme_price_movements.features import compute_market_features, add_regime_gates, compute_features_hourly
from extreme_price_movements.engine import select_trade_candidates_hourly, entry_price_next_hour_open, generate_signals, execute_signals
from extreme_price_movements.time_utils import get_ts_sig, floor_to_hour, now_utc
from extreme_price_movements.state import StateManager
from extreme_price_movements.metrics import MetricsLogger
from extreme_price_movements.training import select_best_horizon, compute_p_exhaustion_at_t, generate_exhaustion_history, optimize_risk_params
from extreme_price_movements.risk import TrailingStop
from extreme_price_movements.optimization_utils import filter_low_variance_assets

TRAINED_MODELS = {
    "ts_trained": None, "bundle": None, "risk_params": None
}

def reconcile_state(ex, state):
    state.reconcile(ex)
    return True

def train_daily(ts_sig, margin_symbols, cfg, store, ex):
    tprint("DAILY TRAINING START")
    syms_all = build_fetch_universe(margin_symbols, cfg["market_basket"], cfg["fetch_symbols_M"])
    with Timer("Training Data Fetch"):
        since = (ts_sig - pd.Timedelta(days=365)).floor("D")
        since_ms = int(since.value // 10**6)
        for s in syms_all:
            try: store.update_symbol(ex, s, since_ms)
            except Exception: pass
    train_syms = filter_low_variance_assets(store, syms_all, lookback_days=30, threshold_pct=0.40)
    train_syms = sorted(list(set(train_syms).union(set(cfg["market_basket"]))))
    dfs = {}
    for s in train_syms:
        df = store.load(s)
        if not df.empty: dfs[s] = df[df.index <= ts_sig].tail(24*90)
    if not dfs:
        tprint("Training failed: No data.")
        return
    with Timer("Training Pipeline"):
        panel = to_panel(dfs)
        mkt_df = compute_market_features(panel, cfg["market_basket"])
        mkt_gates = add_regime_gates(mkt_df, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"])
        feats = compute_features_hourly(panel, mkt_gates, cfg)
        p_exh_hist = generate_exhaustion_history(panel, feats, mkt_gates, cfg, ts_sig, cfg["train_lookback_hours"], train_syms)
        trained_bundle = select_best_horizon(panel, feats, mkt_gates, cfg, train_syms, ts_sig, p_exh_hist)
        alpha_models = trained_bundle["alpha_models"]
        best_risk = optimize_risk_params(panel, feats, mkt_gates, cfg, train_syms, ts_sig, p_exh_hist, alpha_models)
        TRAINED_MODELS["ts_trained"] = ts_sig
        TRAINED_MODELS["bundle"] = trained_bundle
        TRAINED_MODELS["risk_params"] = best_risk

        try:
             with open("models.pkl", "wb") as f:
                  pickle.dump(TRAINED_MODELS, f)
             tprint("Saved models.pkl")
        except Exception as e:
             tprint(f"Failed to save models: {e}")

    tprint("DAILY TRAINING COMPLETE")

def execute_hourly(ts_sig, margin_symbols, cfg, store, ex, state, logger):
    run_id = str(uuid.uuid4())
    state.set_run_id(run_id)
    tprint(f"HOURLY EXEC Start: {ts_sig} RunID={run_id}")
    candidates_pool = select_live_candidates(margin_symbols, cfg["market_basket"], pct=0.05)

    current_positions = state.get_positions()
    active_syms = list(current_positions.keys())
    # Ensure active symbols fetched
    fetch_syms = sorted(list(set(candidates_pool + active_syms)))

    dfs = {}
    since = (ts_sig - pd.Timedelta(days=90)).floor("D")
    since_ms = int(since.value // 10**6)
    with Timer("Candidate Data Fetch"):
        for s in fetch_syms:
            try:
                df = store.update_symbol(ex, s, since_ms)
                if not df.empty and df.index.max() >= ts_sig: dfs[s] = df[df.index <= ts_sig].tail(24*90)
            except Exception: pass
    if not dfs: return

    with Timer("Feature Gen (Candidates)"):
        panel = to_panel(dfs)
        mkt_df = compute_market_features(panel, cfg["market_basket"])
        mkt_gates = add_regime_gates(mkt_df, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"])
        feats = compute_features_hourly(panel, mkt_gates, cfg)

    # Load persistence if global empty
    if TRAINED_MODELS["bundle"] is None:
         if os.path.exists("models.pkl"):
              try:
                   with open("models.pkl", "rb") as f:
                        data = pickle.load(f)
                        TRAINED_MODELS.update(data)
                   tprint("Loaded models.pkl")
              except Exception as e:
                   tprint(f"Failed to load models: {e}")

    bundle = TRAINED_MODELS["bundle"]
    if not bundle: return
    alpha_models = bundle["alpha_models"]
    meta_models = bundle["meta_models"]

    risk_conf = TRAINED_MODELS["risk_params"] # granular_risk dict inside

    # Update Active Positions (Risk Management)
    o = panel["open"]; h = panel["high"]; l = panel["low"]; c = panel["close"]
    for sym in active_syms:
        if sym not in c.columns or ts_sig not in c.index: continue
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
        else:
            pos["risk_state"] = ts_risk.to_dict()
            state.set_position(sym, pos)

    p_exh_cand = generate_exhaustion_history(panel, feats, mkt_gates, cfg, ts_sig, 24, list(dfs.keys()))
    top, bot = select_trade_candidates_hourly(feats, ts_sig, list(dfs.keys()), cfg["trade_extreme_pct"], cfg["trade_extreme_min"], cfg["trade_extreme_max"], cfg["trade_deviation_metric"])
    candidates = list(set(top) | set(bot))
    # Filter already active
    candidates = [s for s in candidates if s not in state.get_positions()]

    # Generate Signals
    signals = generate_signals(ts_sig, candidates, panel, feats, mkt_gates, cfg, alpha_models, meta_models, p_exh_cand, p_exh_ts=None, active_positions=state.get_positions())

    # Execute Signals
    execute_signals(signals, state, ex, panel, feats, ts_sig, cfg, logger, risk_conf)

    state.set_last_ts_sig(ts_sig)
    logger.log(ts_sig, {"n_candidates": len(candidates), "run_id": run_id})
    tprint("HOURLY EXEC COMPLETE")

def run_live_cycle():
    cfg = CFG.copy()
    state = StateManager()
    logger = MetricsLogger()
    ts_sig = get_ts_sig()
    last_ts = state.get_last_ts_sig()
    tprint(f"Current ts_sig: {ts_sig}")
    if last_ts and ts_sig <= last_ts: tprint(f"Already processed {ts_sig}. Waiting..."); return
    ex = make_spot_exchange()
    reconcile_state(ex, state)
    with Timer("Margin universe refresh"): mu = refresh_margin_universe_daily(None, quote="USDT")
    store = PartitionedOHLCVStore(root_dir=cfg["data_root"], timeframe=cfg["timeframe"])
    last_train = TRAINED_MODELS["ts_trained"]

    # Check if loaded from pickle
    if last_train is None and os.path.exists("models.pkl"):
          try:
               with open("models.pkl", "rb") as f:
                    data = pickle.load(f)
                    TRAINED_MODELS.update(data)
               last_train = TRAINED_MODELS["ts_trained"]
          except: pass

    need_train = False
    if last_train is None: need_train = True
    else:
        if ts_sig.floor("D") > last_train.floor("D"): need_train = True

    if need_train: train_daily(ts_sig, mu.symbols, cfg, store, ex)
    execute_hourly(ts_sig, mu.symbols, cfg, store, ex, state, logger)

if __name__ == "__main__":
    while True:
        try: run_live_cycle()
        except Exception as e:
            tprint(f"CRITICAL ERROR: {e}")
            import traceback; traceback.print_exc()
        tprint("Sleeping 60s..."); time.sleep(60)
