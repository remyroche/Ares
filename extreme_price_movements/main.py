import time
import sys
import os
import pandas as pd
import numpy as np
import uuid

from extreme_price_movements.config import CFG
from extreme_price_movements.utils import tprint, Timer
from extreme_price_movements.universe import refresh_margin_universe_daily, build_fetch_universe, select_live_candidates, get_training_universe
from extreme_price_movements.data_store import make_spot_exchange, PartitionedOHLCVStore, to_panel, check_data_health, save_features, load_features, get_feature_path, load_artifact_df
from extreme_price_movements.features import compute_market_features, add_regime_gates, compute_features_hourly
from extreme_price_movements.engine import generate_hourly_signals
from extreme_price_movements.candidates import select_trade_candidates_hourly, entry_price_next_hour_open
from extreme_price_movements.time_utils import get_ts_sig, floor_to_hour, now_utc
from extreme_price_movements.state import StateManager
from extreme_price_movements.metrics import MetricsLogger
from extreme_price_movements.training import select_best_horizon, compute_p_exhaustion_at_t, apply_interaction_toggles, generate_exhaustion_history, optimize_risk_params, train_models_from_artifacts
from extreme_price_movements.risk import TrailingStop
from extreme_price_movements.optimization_utils import filter_low_variance_assets
from extreme_price_movements.pipeline_steps import run_label_generation_step_v2, run_risk_optimization_step, run_backtest_step

def reconcile_state(ex, state):
    tprint("Reconciling state...")
    return True

def generate_features_daily(ts_sig, margin_symbols, cfg, store, ex):
    tprint("DAILY FEATURE GENERATION START")
    train_syms = get_training_universe(margin_symbols, cfg, store, ts_sig=ts_sig)
    tprint(f"Target universe size: {len(train_syms)}")

    missing_syms = []
    for s in train_syms:
        fpath = get_feature_path(cfg["data_root"], ts_sig, s)
        if not os.path.exists(fpath):
            missing_syms.append(s)

    if not missing_syms:
        tprint("All features already generated.")
        return

    tprint(f"Generating features for {len(missing_syms)} missing symbols...")

    # We must load market basket to compute market features
    load_syms = sorted(list(set(missing_syms).union(set(cfg["market_basket"]))))

    dfs = {}

    # Use fetch_years to determine loading window, but ensure at least 90 days for feature safety
    lookback_days = max(90, int(cfg["fetch_years"] * 365))

    with Timer("Feature Data Fetch"):
        for s in load_syms:
            df = store.load(s)
            if not df.empty:
                # Load history based on config
                dfs[s] = df[df.index <= ts_sig].tail(24*lookback_days)

    if not dfs:
        tprint("No data available for feature generation.")
        return

    with Timer("Feature Computation"):
        panel = to_panel(dfs)
        mkt_df = compute_market_features(panel, cfg["market_basket"])
        tprint("Market features computed (generation)")
        mkt_gates = add_regime_gates(mkt_df, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"])

        feats = compute_features_hourly(panel, mkt_gates, cfg)
        tprint("Hourly features computed (generation)")

        # Filter to save only missing symbols
        feats_to_save = {}

        # Check available columns in features
        # feats is Dict[FeatName -> DataFrame]

        # Just to be safe, check one known feature like 'ret1h'
        if "ret1h" in feats:
            available_cols = feats["ret1h"].columns
            valid_missing = [s for s in missing_syms if s in available_cols]
        else:
            # Fallback if structure is different
            valid_missing = missing_syms

        if not valid_missing:
            tprint("No valid missing symbols found in computed features.")
            return

        for k, v in feats.items():
            if isinstance(v, pd.DataFrame):
                # Select only the columns for missing symbols
                cols = [c for c in valid_missing if c in v.columns]
                if cols:
                    feats_to_save[k] = v[cols]
            else:
                pass

        if feats_to_save:
            save_features(feats_to_save, ts_sig, cfg["data_root"])
        else:
            tprint("No features to save.")

    tprint("DAILY FEATURE GENERATION COMPLETE")

def train_daily(ts_sig, margin_symbols, cfg, store, ex):
    tprint("DAILY TRAINING START")

    # In the new split, train_daily only trains models from artifacts.
    # It assumes labels are generated.

    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    datasets = {}

    # 1. Load Datasets from artifacts
    # We need to know which keys to look for.
    # We can glob or just try standard keys.
    tprint("Loading label datasets from artifacts...")

    # Spike
    df_spike = load_artifact_df(cfg["data_root"], run_id, "labels", "spike_anatomy")
    if df_spike is not None: datasets["spike_anatomy"] = df_spike

    # Alpha models
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

    # Exhaustion
    for d in ["up", "down"]:
        name = f"exh_{d}"
        df = load_artifact_df(cfg["data_root"], run_id, "labels", name)
        if df is not None: datasets[name] = df

    if not found_count:
        tprint("ERROR: No label datasets found. Run 'labels' mode first.")
        return None

    with Timer("Model Training"):
        trained_bundle = train_models_from_artifacts(datasets, cfg)
        tprint("Models trained.")

        # Note: Risk optimization is now separate step.
        # But for backward compat, we should return a state that has 'risk_params' if we don't run risk step?
        # If we return None for risk, execute_hourly might fail.
        # We will initialize with default risk params here, and 'risk' step will update them.
        default_risk = {
            "k_sl": cfg.get("risk_k_sl", 2.0),
            "k_trail_start": cfg.get("risk_k_trail_start", 1.0),
            "k_trail_dist": cfg.get("risk_k_trail_dist", 0.5),
            "granular_risk": {}
        }

        new_state = {
            "ts_trained": ts_sig,
            "bundle": trained_bundle,
            "risk_params": default_risk
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

    # Execution typically only needs enough for feature calculation (90d is safe)
    # But to respect "light" vs full consistency, we can use the same logic if feasible.
    # However, for live execution, fetching 3 years of data every hour is wasteful and slow.
    # We will stick to a safe 90 days for execution as it doesn't affect model training depth.
    # Wait, the user said "model training, etc". Execution is "running" the model.
    # Ideally execution state is minimal. Let's keep 90 days or `fetch_years` if smaller?
    # No, features might break if window is too small. 90 days is a safe lower bound.
    # Let's keep 90 days for execution speed, as it's not training.

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
            # Already injected by generate_hourly_signals into order['risk_params']
            # But let's double check or use defaults
            g_risk = order.get("risk_params", {})

            # Check if Triple Barrier Params are present
            tp_mult = g_risk.get("tp_mult")
            sl_mult = g_risk.get("sl_mult")

            # Apply Score Confidence scaling to SL?
            # Standard logic: k_sl adj = k_sl * (1 + score_scale * abs(score))
            # If using fixed TP/SL mults, do we scale them?
            # Probably scaled SL multiplier is good.

            k_sl = g_risk.get("k_sl", cfg["risk_k_sl"])
            sc_scale = g_risk.get("score_scale", 0.0) # usually 0 in defaults
            adj = (1.0 + sc_scale * abs(score))

            # Config for TrailingStop
            # If tp_mult is present, we use it for activation (approx) or specialized logic?
            # TrailingStop class currently handles k_sl, k_trail_start...
            # We need to enhance TrailingStop or use a different class if we want fixed barriers.
            # But TrailingStop is serialized.
            # Let's map TP/SL mult to TrailingStop params if possible.
            # TP -> Activation? If we want fixed exit at TP, we can set activation=TP and trail_dist=tiny.
            # Then once activated, stop jumps to Price - tiny ~ Price. Next tick exit.

            if tp_mult and sl_mult:
                # Use dynamic barrier logic
                # We need ATR stats history for dynamic scaling?
                # simulate_trade_hourly computes it.
                # Here we are in live execution.
                # We need to compute the barrier level NOW.

                # To compute dynamic barrier, we need rolling Z.
                # We have `feats`. `feats["atr_pct"]` is the series.
                # We can compute it here.
                from extreme_price_movements.training import scaled_atr_pct

                atr_series = feats["atr_pct"][sym]
                # Slice history
                if len(atr_series) > 30*24:
                    win = atr_series.iloc[-(30*24):]
                    base = win.median()
                    std = win.std()
                    z = (atr - base) / (std + 1e-12)
                    barrier_pct = scaled_atr_pct(atr, z, base, z_max=3.0, lo=0.03, hi=0.06)
                else:
                    barrier_pct = atr # Fallback

                # Convert to k factors relative to CURRENT ATR?
                # barrier_pct is absolute percent.
                # TrailingStop expects k factors relative to `atr_val` passed to it.
                # k_effective = barrier_pct / atr

                k_barrier = barrier_pct / (atr + 1e-12)

                # TP distance = tp_mult * barrier_pct
                # SL distance = sl_mult * barrier_pct

                # Map to TrailingStop:
                # k_sl = sl_mult * k_barrier
                # k_trail_start = tp_mult * k_barrier
                # k_trail_dist = 0.001 (tight)

                k_sl_adj = sl_mult * k_barrier * adj # scaling SL
                k_ts = tp_mult * k_barrier
                k_td = 0.001

            else:
                # Legacy Trailing Logic
                k_sl_adj = k_sl * adj
                k_ts = g_risk.get("k_trail_start", cfg["risk_k_trail_start"])
                k_td = g_risk.get("k_trail_dist", cfg["risk_k_trail_dist"])

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
