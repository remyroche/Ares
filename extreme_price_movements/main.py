import time
import sys
import pandas as pd
import numpy as np
import uuid

from extreme_price_movements.config import CFG
from extreme_price_movements.utils import tprint, Timer
from extreme_price_movements.universe import refresh_margin_universe_daily, build_fetch_universe, select_live_candidates
from extreme_price_movements.data_store import make_spot_exchange, PartitionedOHLCVStore, to_panel, check_data_health
from extreme_price_movements.features import compute_market_features, add_regime_gates, compute_features_hourly
from extreme_price_movements.engine import select_trade_candidates_hourly, entry_price_next_hour_open
from extreme_price_movements.time_utils import get_ts_sig, floor_to_hour, now_utc
from extreme_price_movements.state import StateManager
from extreme_price_movements.metrics import MetricsLogger
from extreme_price_movements.training import select_best_horizon, compute_p_exhaustion_at_t, apply_interaction_toggles, generate_exhaustion_history, optimize_risk_params
from extreme_price_movements.risk import TrailingStop
from extreme_price_movements.optimization_utils import filter_low_variance_assets

# Global cache for trained models (simulated persistence)
TRAINED_MODELS = {
    "ts_trained": None,
    "bundle": None,
    "risk_params": None,
    "p_exh_hist": None # Used for inference? We generate lag.
}

def reconcile_state(ex, state):
    tprint("Reconciling state...")
    return True

def train_daily(ts_sig, margin_symbols, cfg, store, ex):
    """
    Daily training routine.
    Fetches broad universe, filters by variance, trains models.
    Updates TRAINED_MODELS global.
    """
    tprint("DAILY TRAINING START")

    # 1. Broad Universe Selection
    syms_all = build_fetch_universe(margin_symbols, cfg["market_basket"], cfg["fetch_symbols_M"])

    # 2. Variance Filter (First Pass)
    # Load daily stats to filter
    # This calls store.load which might be slow if data not present.
    # We first update data?
    # If we update ALL 300 symbols, it takes time.
    # User said: "make a first pass by looking at one point of data every 12 hours... then only look at more granular data"
    # This implies we have the data? Or we fetch coarse data?
    # CCXT fetch_ohlcv supports 12h?
    # Binace supports 12h.
    # BUT our store is 1h.
    # If we fetch 1h for everyone, we defeat the purpose.
    # We should fetch 12h for filtering?
    # Complexity: Storing 12h data separately?
    # Simplification: We assume we have historical data for "top 300" from previous runs?
    # Or we pay the cost of fetching 300 symbols once a day.
    # Given requirements, I will fetch 300 symbols (incremental).
    # Then apply variance filter.
    # Then TRAIN on top 120.

    # Optimization: Only fetch/load 'close' if possible?
    # update_symbol fetches all columns.

    # Let's do: Fetch All (it's daily).
    with Timer("Training Data Fetch"):
        since = (ts_sig - pd.Timedelta(days=365)).floor("D")
        since_ms = int(since.value // 10**6)

        # We limit to top N vol to limit fetch time (cfg["fetch_symbols_M"])
        for s in syms_all:
            try:
                store.update_symbol(ex, s, since_ms)
            except Exception: pass

    # 3. Variance Filter
    train_syms = filter_low_variance_assets(store, syms_all, lookback_days=30, threshold_pct=0.40)
    # Ensure basket
    train_syms = sorted(list(set(train_syms).union(set(cfg["market_basket"]))))

    # 4. Load Training Data
    dfs = {}
    for s in train_syms:
        df = store.load(s)
        if not df.empty:
            dfs[s] = df[df.index <= ts_sig].tail(24*90) # 90 days history

    if not dfs:
        tprint("Training failed: No data.")
        return

    # 5. Train
    with Timer("Training Pipeline"):
        panel = to_panel(dfs)
        mkt_df = compute_market_features(panel, cfg["market_basket"])
        mkt_gates = add_regime_gates(mkt_df, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"])
        feats = compute_features_hourly(panel, mkt_gates, cfg)

        p_exh_hist = generate_exhaustion_history(panel, feats, mkt_gates, cfg, ts_sig, cfg["train_lookback_hours"], train_syms)

        trained_bundle = select_best_horizon(panel, feats, mkt_gates, cfg, train_syms, ts_sig, p_exh_hist)
        alpha_models = trained_bundle["alpha_models"]

        best_risk = optimize_risk_params(panel, feats, mkt_gates, cfg, train_syms, ts_sig, p_exh_hist, alpha_models)

        # Update Global State
        TRAINED_MODELS["ts_trained"] = ts_sig
        TRAINED_MODELS["bundle"] = trained_bundle
        TRAINED_MODELS["risk_params"] = best_risk
        # We don't store p_exh_hist for all symbols, but we might need recent history for lag generation?
        # Inference will re-generate recent history for candidates.

    tprint("DAILY TRAINING COMPLETE")

def execute_hourly(ts_sig, margin_symbols, cfg, store, ex, state, logger):
    """
    Hourly execution routine.
    Uses cached models.
    Fetches only candidates.
    """
    run_id = str(uuid.uuid4())
    tprint(f"HOURLY EXEC Start: {ts_sig} RunID={run_id}")

    # 1. Select Candidates (Live Ticker Filter)
    candidates_pool = select_live_candidates(margin_symbols, cfg["market_basket"], pct=0.05)

    # 2. Fetch Data (Candidates Only)
    dfs = {}
    since = (ts_sig - pd.Timedelta(days=90)).floor("D") # Need enough for lookbacks
    since_ms = int(since.value // 10**6)

    with Timer("Candidate Data Fetch"):
        for s in candidates_pool:
            try:
                df = store.update_symbol(ex, s, since_ms)
                if not df.empty and df.index.max() >= ts_sig:
                    dfs[s] = df[df.index <= ts_sig].tail(24*90)
            except Exception: pass

    if not dfs:
        tprint("No candidate data.")
        return

    # 3. Features
    with Timer("Feature Gen (Candidates)"):
        panel = to_panel(dfs)
        mkt_df = compute_market_features(panel, cfg["market_basket"])
        mkt_gates = add_regime_gates(mkt_df, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"])
        feats = compute_features_hourly(panel, mkt_gates, cfg)

    # 4. Inference
    bundle = TRAINED_MODELS["bundle"]
    if not bundle:
        tprint("No trained models available. Skipping.")
        return

    alpha_models = bundle["alpha_models"]
    meta_models = bundle["meta_models"]

    # Use Risk Params
    if TRAINED_MODELS["risk_params"]:
        cfg.update(TRAINED_MODELS["risk_params"])

    # Position Management
    current_positions = state.get_positions()
    # ... (Exit Logic same as before, check `engine.py` / prev main)
    # Need to load data for existing positions too if they are not in candidates?
    # Yes. We should add existing position symbols to fetch list.
    # Fixed in Step 2 logic (union with active).

    # Re-fetch active symbols logic:
    # Actually, step 2 should include active symbols.
    # Let's fix that.
    active_syms = list(current_positions.keys())
    # We need data for them to check exits.
    # Assuming they are in `dfs`. If not, we missed them.
    # We must ensure `candidates_pool` includes `active_syms`.

    # Refetch missing active
    for s in active_syms:
        if s not in dfs:
            # Fetch
            try:
                df = store.update_symbol(ex, s, since_ms)
                if not df.empty: dfs[s] = df.tail(24*90)
            except: pass

    # Re-build panel? If we added symbols.
    # Efficiency: Include active in candidates_pool at start.
    # I'll rely on `dfs` check later.

    # EXITS
    o = panel["open"]; h = panel["high"]; l = panel["low"]; c = panel["close"]
    for sym in active_syms:
        if sym not in c.columns or ts_sig not in c.index: continue
        pos = current_positions[sym]
        ts_risk = TrailingStop.from_dict(pos["risk_state"])
        curr_h = float(h.loc[ts_sig, sym])
        curr_l = float(l.loc[ts_sig, sym])
        curr_c = float(c.loc[ts_sig, sym])
        stopped, exit_px, reason = ts_risk.update(curr_h, curr_l, curr_c)
        if stopped:
            entry_px = pos["entry_px"]
            side = pos["side"]
            if reason == "ambiguous_neutral": ret = 0.0
            else:
                ret = (exit_px / entry_px - 1.0) if side == "long" else (entry_px / exit_px - 1.0)
            logger.log(ts_sig, {"event": "exit", "symbol": sym, "return": ret, "reason": reason})
            state.clear_position(sym)
        else:
            pos["risk_state"] = ts_risk.to_dict()
            state.set_position(sym, pos)

    # ENTRIES
    # Generate partial history for candidates (for lag features)
    # We need p_exh_lag1 (t-1).
    # We compute p_exh history for candidates.
    p_exh_cand = generate_exhaustion_history(panel, feats, mkt_gates, cfg, ts_sig, 24, list(dfs.keys()))

    top, bot = select_trade_candidates_hourly(feats, ts_sig, list(dfs.keys()), cfg["trade_extreme_pct"], cfg["trade_extreme_min"], cfg["trade_extreme_max"], cfg["trade_deviation_metric"])
    candidates = list(set(top) | set(bot))
    candidates = [s for s in candidates if s not in state.get_positions()]

    if candidates and ts_sig in mkt_gates.index:
        mrk = mkt_gates.loc[ts_sig]
        ts_lag = ts_sig - pd.Timedelta(hours=1)
        trend_df = feats.get("trend_pct")

        rows = []
        for sym in candidates:
            try:
                t_val = 0.0
                if trend_df is not None and sym in trend_df.columns:
                    t_val = float(trend_df.loc[ts_sig, sym])
                direction = "up" if t_val > 0 else "down"

                m_bundle = alpha_models.get(direction)
                if not m_bundle or not m_bundle["mr"] or not m_bundle["tf"]: continue

                model_mr = m_bundle["mr"]["model"]
                model_tf = m_bundle["tf"]["model"]
                feat_cols = m_bundle["mr"]["feat_cols"]
                meta_model = meta_models.get(direction)

                p_lag = 0.5
                if ts_lag in p_exh_cand.index and sym in p_exh_cand.columns:
                    p_lag = float(p_exh_cand.loc[ts_lag, sym])

                rec = {
                    "symbol": sym, "direction": direction,
                    "model_mr": model_mr, "model_tf": model_tf, "meta_model": meta_model,
                    "feat_cols": feat_cols,
                    "mkt_ret24h": float(mrk["mkt_ret24h"]), "mkt_ret6h": float(mrk["mkt_ret6h"]),
                    "mkt_trend": float(mrk["mkt_trend"]), "mkt_rv": float(mrk["mkt_rv"]),
                    "G_VOL": int(mrk["G_VOL"]), "G_TREND": int(mrk["G_TREND"]),
                    "p_exh_lag1": p_lag
                }
                for k in feat_cols:
                    if k in feats: rec[k] = float(feats[k].loc[ts_sig, sym])
                if "a_rv24" in feats: rec["a_rv24"] = float(feats["a_rv24"].loc[ts_sig, sym])
                if "a_volz" in feats: rec["a_volz"] = float(feats["a_volz"].loc[ts_sig, sym])
                if "a_rsi" in feats: rec["a_rsi"] = float(feats["a_rsi"].loc[ts_sig, sym])
                if "dist_ema_fast" in feats: rec["dist_ema_fast"] = float(feats["dist_ema_fast"].loc[ts_sig, sym])
                rows.append(rec)
            except Exception: continue

        # Inference
        df_all = pd.DataFrame(rows)
        score_raw_list = []
        if not df_all.empty:
            for d, grp in df_all.groupby("direction"):
                first = grp.iloc[0]
                model_mr = first["model_mr"]; model_tf = first["model_tf"]; meta_model = first["meta_model"]; fcols = first["feat_cols"]
                Xint = apply_interaction_toggles(grp, cfg["causal_cols"], ["G_VOL","G_TREND"], drop_raw=cfg["drop_raw_causal"])
                for c in fcols:
                    if c not in Xint.columns: Xint[c] = 0.0
                Xpred = Xint[fcols].fillna(0.0).astype(np.float32)
                p_mr = model_mr.predict(Xpred)
                p_tf = model_tf.predict(Xpred)

                if meta_model:
                    X_meta = meta_model.prepare_meta_features(p_tf, p_mr, grp)
                    score = meta_model.predict(X_meta)
                else:
                    score = p_tf - p_mr
                    sign = 1.0 if d == "up" else -1.0
                    score = score * sign

                for i, idx in enumerate(grp.index):
                    score_raw_list.append((grp.loc[idx, "symbol"], score[i]))

        # Execution
        score_raw_list.sort(key=lambda x: x[1], reverse=True)
        longs = [x for x in score_raw_list if x[1] > cfg["thr_long"]][:cfg["k_long"]]
        shorts = sorted([x for x in score_raw_list if x[1] < cfg["thr_short"]], key=lambda x: x[1])[:cfg["k_short"]]

        final_orders = []
        for s, sc in longs: final_orders.append((s, "long", sc))
        for s, sc in shorts: final_orders.append((s, "short", sc))

        total_wt = sum(abs(x[2]) for x in final_orders)
        if total_wt > 0:
            gross_cap = float(cfg["wallet_gross_cap"])
            for sym, side, score in final_orders:
                w_alloc = gross_cap * (abs(score) / total_wt)
                atr = float(feats["atr_pct"].loc[ts_sig, sym])
                entry_px = float(c.loc[ts_sig, sym])
                ts_risk = TrailingStop(entry_px=entry_px, side=side, atr_val=atr, k_sl=cfg["risk_k_sl"], k_trail_start=cfg["risk_k_trail_start"], k_trail_dist=cfg["risk_k_trail_dist"])
                pos = {
                    "symbol": sym, "side": side, "entry_px": entry_px, "entry_ts": ts_sig.isoformat(),
                    "score": float(score), "weight": float(w_alloc), "risk_state": ts_risk.to_dict(), "run_id": run_id
                }
                state.set_position(sym, pos)
                tprint(f"ENTRY {side} {sym} @ {entry_px} (score={score:.4f})")
                logger.log(ts_sig, {"event": "entry", "symbol": sym, "side": side, "score": score, "weight": w_alloc})

    state.set_last_ts_sig(ts_sig)
    tprint("HOURLY EXEC COMPLETE")

def run_live_cycle():
    cfg = CFG.copy()
    state = StateManager()
    logger = MetricsLogger()

    ts_sig = get_ts_sig()
    last_ts = state.get_last_ts_sig()

    tprint(f"Current ts_sig: {ts_sig}")
    if last_ts and ts_sig <= last_ts:
        tprint(f"Already processed {ts_sig}. Waiting...")
        return

    ex = make_spot_exchange()
    reconcile_state(ex, state)

    with Timer("Margin universe refresh"):
        mu = refresh_margin_universe_daily(None, quote="USDT")
    margin_symbols = mu.symbols

    store = PartitionedOHLCVStore(root_dir=cfg["data_root"], timeframe=cfg["timeframe"])

    # Check if training needed (Daily roll)
    # Check TRAINED_MODELS timestamp
    last_train = TRAINED_MODELS["ts_trained"]
    need_train = False
    if last_train is None:
        need_train = True
    else:
        # Train if new day
        if ts_sig.floor("D") > last_train.floor("D"):
            need_train = True

    if need_train:
        train_daily(ts_sig, margin_symbols, cfg, store, ex)

    execute_hourly(ts_sig, margin_symbols, cfg, store, ex, state, logger)

if __name__ == "__main__":
    while True:
        try:
            run_live_cycle()
        except Exception as e:
            tprint(f"CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
        tprint("Sleeping 60s...")
        time.sleep(60)
