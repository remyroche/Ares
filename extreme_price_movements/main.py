import time
import sys
import pandas as pd
import numpy as np

from extreme_price_movements.config import CFG
from extreme_price_movements.utils import tprint, Timer
from extreme_price_movements.universe import refresh_margin_universe_daily, build_fetch_universe
from extreme_price_movements.data_store import make_spot_exchange, PartitionedOHLCVStore, to_panel, check_data_health
from extreme_price_movements.features import compute_market_features, add_regime_gates, compute_features_hourly
from extreme_price_movements.engine import select_trade_candidates_hourly, entry_price_next_hour_open
from extreme_price_movements.time_utils import get_ts_sig, floor_to_hour, now_utc
from extreme_price_movements.state import StateManager
from extreme_price_movements.metrics import MetricsLogger
from extreme_price_movements.training import select_best_horizon, compute_p_exhaustion_at_t, apply_interaction_toggles, generate_exhaustion_history
from extreme_price_movements.risk import TrailingStop
from extreme_price_movements.models import map_pred_to_score

def run_live_cycle():
    cfg = CFG
    state = StateManager()
    logger = MetricsLogger()

    tprint("BOOT: Extreme Price Movements (Live/Paper)")

    # 1. Time Check
    ts_sig = get_ts_sig()
    last_ts = state.get_last_ts_sig()

    tprint(f"Current ts_sig: {ts_sig}")
    if last_ts and ts_sig <= last_ts:
        tprint(f"Already processed {ts_sig}. Waiting...")
        return

    # 2. Data Fetch
    ex = make_spot_exchange()
    with Timer("Margin universe refresh"):
        mu = refresh_margin_universe_daily(None, quote="USDT")
    margin_symbols = mu.symbols

    syms = build_fetch_universe(margin_symbols, cfg["market_basket"], cfg["fetch_symbols_M"])
    store = PartitionedOHLCVStore(root_dir=cfg["data_root"], timeframe=cfg["timeframe"])

    dfs = {}
    data_health_issues = 0

    since = (ts_sig - pd.Timedelta(days=365)).floor("D")
    since_ms = int(since.value // 10**6)

    with Timer("Fetch/Update symbols"):
        for s in syms:
            try:
                df = store.update_symbol(ex, s, since_ms)
                if df.empty or df.index.max() < ts_sig:
                    tprint(f"WARN: {s} missing data at {ts_sig}")
                    data_health_issues += 1
                    continue

                health = check_data_health(df.loc[since:ts_sig])
                if health["status"] != "ok":
                    tprint(f"WARN: {s} data health issues: {health}")
                    data_health_issues += 1

                dfs[s] = df[df.index <= ts_sig].tail(24*90)
            except Exception as e:
                tprint(f"Error fetching {s}: {e}")

    if not dfs:
        tprint("No data available.")
        return

    # 3. Build Panel & Features
    with Timer("Feature Generation"):
        panel = to_panel(dfs)
        basket_ok = True
        for b in cfg["market_basket"]:
            if b not in panel["close"].columns:
                tprint(f"Market basket missing: {b}")
                basket_ok = False
        if not basket_ok:
            tprint("Skipping: Market basket incomplete.")
            return

        mkt_df = compute_market_features(panel, cfg["market_basket"])
        mkt_gates = add_regime_gates(mkt_df, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"])
        feats = compute_features_hourly(panel, mkt_gates, cfg)

    # 4. Model Training (Daily)
    tprint("Model Selection / Training...")

    # Generate Exhaustion History for training weighting
    # This is expensive but necessary for correct weighting
    p_exh_hist = generate_exhaustion_history(panel, feats, mkt_gates, cfg, ts_sig, cfg["train_lookback_hours"], syms)

    p_exh_now = compute_p_exhaustion_at_t(panel, feats, mkt_gates, cfg, ts_sig, syms)

    best_mr = select_best_horizon(panel, feats, mkt_gates, cfg, syms, ts_sig, p_exh_hist, model_kind="mr")
    best_tf = select_best_horizon(panel, feats, mkt_gates, cfg, syms, ts_sig, p_exh_hist, model_kind="tf")

    if best_mr is None or best_tf is None:
        tprint("Training failed (no valid models).")
        return

    model_mr = best_mr["model"]
    model_tf = best_tf["model"]
    feat_cols = best_mr["feat_cols"]

    # 5. Position Management (Exits)
    current_positions = state.get_positions()
    active_symbols = list(current_positions.keys())

    o = panel["open"]; h = panel["high"]; l = panel["low"]; c = panel["close"]

    tprint(f"Checking exits for {len(active_symbols)} positions...")
    for sym in active_symbols:
        if sym not in c.columns or ts_sig not in c.index:
            tprint(f"WARN: No data for {sym} at {ts_sig}")
            continue

        pos = current_positions[sym]
        ts_risk = TrailingStop.from_dict(pos["risk_state"])

        curr_h = float(h.loc[ts_sig, sym])
        curr_l = float(l.loc[ts_sig, sym])
        curr_c = float(c.loc[ts_sig, sym])

        stopped, exit_px, reason = ts_risk.update(curr_h, curr_l, curr_c)

        if stopped:
            tprint(f"EXIT {sym}: {reason} at {exit_px}")
            entry_px = pos["entry_px"]
            side = pos["side"]
            if reason == "ambiguous_neutral":
                ret = 0.0
            else:
                if side == "long":
                    ret = (exit_px / entry_px) - 1.0
                else:
                    ret = (entry_px / exit_px) - 1.0

            logger.log(ts_sig, {"event": "exit", "symbol": sym, "return": ret, "reason": reason})
            state.clear_position(sym)
        else:
            pos["risk_state"] = ts_risk.to_dict()
            state.set_position(sym, pos)

    # 6. Candidate Selection & Entry
    top, bot = select_trade_candidates_hourly(feats, ts_sig, syms, cfg["trade_extreme_pct"], cfg["trade_extreme_min"], cfg["trade_extreme_max"], cfg["trade_deviation_metric"])
    candidates = list(set(top) | set(bot))

    candidates = [s for s in candidates if s not in state.get_positions()]

    if candidates:
        if ts_sig not in mkt_gates.index:
            tprint("Missing gates")
            return
        mrk = mkt_gates.loc[ts_sig]

        # We need Lag 1 of p_exh.
        # p_exh_hist contains history up to ts_sig.
        # Lag 1 is at ts_sig - 1h.
        ts_lag = ts_sig - pd.Timedelta(hours=1)

        rows = []
        for sym in candidates:
            try:
                p_lag_val = 0.5
                if ts_lag in p_exh_hist.index and sym in p_exh_hist.columns:
                    p_lag_val = float(p_exh_hist.loc[ts_lag, sym])

                rows.append({
                    "symbol": sym,
                    **{k: float(feats[k].loc[ts_sig, sym]) for k in feat_cols if k in feats},
                    "mkt_ret24h": float(mrk["mkt_ret24h"]),
                    "mkt_ret6h": float(mrk["mkt_ret6h"]),
                    "mkt_trend": float(mrk["mkt_trend"]),
                    "mkt_rv": float(mrk["mkt_rv"]),
                    "G_VOL": int(mrk["G_VOL"]),
                    "G_TREND": int(mrk["G_TREND"]),
                    "p_exh_lag1": p_lag_val,
                })
            except Exception:
                continue

        if rows:
            Xdf = pd.DataFrame(rows)
            Xint = apply_interaction_toggles(Xdf, cfg["causal_cols"], ["G_VOL","G_TREND"], drop_raw=cfg["drop_raw_causal"])
            for col in feat_cols:
                if col not in Xint.columns: Xint[col] = 0.0

            Xpred = Xint[feat_cols].fillna(0.0).astype(np.float32)
            pred_mr = model_mr.predict(Xpred)
            pred_tf, disp_tf = model_tf.predict(Xpred)

            score_raw = pred_tf - pred_mr

            res = pd.DataFrame({"symbol": Xint["symbol"], "score": score_raw})

            longs = res[res["score"] > cfg["thr_long"]].sort_values("score", ascending=False).head(cfg["k_long"])
            shorts = res[res["score"] < cfg["thr_short"]].sort_values("score", ascending=True).head(cfg["k_short"])

            picks = []
            for _, row in longs.iterrows():
                picks.append((row["symbol"], "long", row["score"]))
            for _, row in shorts.iterrows():
                picks.append((row["symbol"], "short", row["score"]))

            for sym, side, score in picks:
                atr = float(feats["atr_pct"].loc[ts_sig, sym])
                entry_px = float(c.loc[ts_sig, sym])

                ts_risk = TrailingStop(
                    entry_px=entry_px, side=side, atr_val=atr,
                    k_sl=cfg["risk_k_sl"], k_trail_start=cfg["risk_k_trail_start"], k_trail_dist=cfg["risk_k_trail_dist"]
                )

                pos = {
                    "symbol": sym,
                    "side": side,
                    "entry_px": entry_px,
                    "entry_ts": ts_sig.isoformat(),
                    "score": float(score),
                    "risk_state": ts_risk.to_dict()
                }
                state.set_position(sym, pos)
                tprint(f"ENTRY {side} {sym} @ {entry_px} (score={score:.4f})")
                logger.log(ts_sig, {"event": "entry", "symbol": sym, "side": side, "score": score})

    state.set_last_ts_sig(ts_sig)

    metrics = {
        "n_candidates": len(candidates),
        "data_health_issues": data_health_issues,
        "n_positions": len(state.get_positions())
    }
    logger.log(ts_sig, metrics)
    tprint("Cycle Complete.")

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
