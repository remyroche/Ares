import time
import sys
import pandas as pd
import numpy as np
import uuid

from extreme_price_movements.config import CFG
from extreme_price_movements.utils import tprint, Timer
from extreme_price_movements.universe import refresh_margin_universe_daily, build_fetch_universe
from extreme_price_movements.data_store import make_spot_exchange, PartitionedOHLCVStore, to_panel, check_data_health
from extreme_price_movements.features import compute_market_features, add_regime_gates, compute_features_hourly
from extreme_price_movements.engine import select_trade_candidates_hourly, entry_price_next_hour_open
from extreme_price_movements.time_utils import get_ts_sig, floor_to_hour, now_utc
from extreme_price_movements.state import StateManager
from extreme_price_movements.metrics import MetricsLogger
from extreme_price_movements.training import select_best_horizon, compute_p_exhaustion_at_t, apply_interaction_toggles, generate_exhaustion_history, optimize_risk_params
from extreme_price_movements.risk import TrailingStop
from extreme_price_movements.models import map_pred_to_score

def reconcile_state(ex, state):
    tprint("Reconciling state...")
    return True

def run_live_cycle():
    run_id = str(uuid.uuid4())
    cfg = CFG.copy()
    state = StateManager()
    logger = MetricsLogger()

    tprint(f"BOOT: Extreme Price Movements (Live/Paper) RunID={run_id}")

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

    # Selection logic updated in universe.py
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
                    data_health_issues += 1
                    continue
                dfs[s] = df[df.index <= ts_sig].tail(24*90)
            except Exception:
                pass

    if not dfs:
        return

    with Timer("Feature Generation"):
        panel = to_panel(dfs)
        mkt_df = compute_market_features(panel, cfg["market_basket"])
        mkt_gates = add_regime_gates(mkt_df, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"])
        feats = compute_features_hourly(panel, mkt_gates, cfg)

    # 4. Model Training (Daily)
    tprint("Model Selection / Training...")

    # 6 models logic
    # p_exh_hist used for input feature
    p_exh_hist = generate_exhaustion_history(panel, feats, mkt_gates, cfg, ts_sig, cfg["train_lookback_hours"], syms)

    # Train separate models
    trained_bundle = select_best_horizon(panel, feats, mkt_gates, cfg, syms, ts_sig, p_exh_hist)
    alpha_models = trained_bundle["alpha_models"] # { "up": { "mr": ..., "tf": ... }, ... }
    exh_models = trained_bundle["exh_models"]     # { "up": ..., "down": ... }

    # Optimize Risk
    best_risk = optimize_risk_params(panel, feats, mkt_gates, cfg, syms, ts_sig, p_exh_hist, alpha_models)
    cfg.update(best_risk)
    tprint(f"Risk Params: {best_risk}")

    # 5. Position Management (Exits)
    current_positions = state.get_positions()
    active_symbols = list(current_positions.keys())

    o = panel["open"]; h = panel["high"]; l = panel["low"]; c = panel["close"]

    tprint(f"Checking exits for {len(active_symbols)} positions...")
    for sym in active_symbols:
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

    # 6. Candidate Selection & Entry
    top, bot = select_trade_candidates_hourly(feats, ts_sig, syms, cfg["trade_extreme_pct"], cfg["trade_extreme_min"], cfg["trade_extreme_max"], cfg["trade_deviation_metric"])
    candidates = list(set(top) | set(bot))
    candidates = [s for s in candidates if s not in state.get_positions()]

    if candidates and ts_sig in mkt_gates.index:
        mrk = mkt_gates.loc[ts_sig]
        ts_lag = ts_sig - pd.Timedelta(hours=1)

        # Determine Trend for switching models
        trend_df = feats.get("trend_pct")

        # We process candidates in two batches: UP trend and DOWN trend
        # For efficiency, we can loop candidates and pick model per candidate

        rows = []
        for sym in candidates:
            try:
                # Identify Trend Direction
                t_val = 0.0
                if trend_df is not None and sym in trend_df.columns:
                    t_val = float(trend_df.loc[ts_sig, sym])

                direction = "up" if t_val > 0 else "down"

                # Get relevant models
                m_bundle = alpha_models[direction]
                if not m_bundle["mr"] or not m_bundle["tf"]:
                    continue

                model_mr = m_bundle["mr"]["model"]
                model_tf = m_bundle["tf"]["model"]
                feat_cols = m_bundle["mr"]["feat_cols"]

                # Get Exh Lag
                p_lag = 0.5
                if ts_lag in p_exh_hist.index and sym in p_exh_hist.columns:
                    p_lag = float(p_exh_hist.loc[ts_lag, sym])

                # Build Row
                rec = {
                    "symbol": sym,
                    "direction": direction, # meta
                    "model_mr": model_mr,   # meta
                    "model_tf": model_tf,   # meta
                    "feat_cols": feat_cols,  # meta
                    "mkt_ret24h": float(mrk["mkt_ret24h"]),
                    "mkt_ret6h": float(mrk["mkt_ret6h"]),
                    "mkt_trend": float(mrk["mkt_trend"]),
                    "mkt_rv": float(mrk["mkt_rv"]),
                    "G_VOL": int(mrk["G_VOL"]),
                    "G_TREND": int(mrk["G_TREND"]),
                    "p_exh_lag1": p_lag,
                }
                # Add features
                for k in feat_cols:
                    if k in feats:
                        rec[k] = float(feats[k].loc[ts_sig, sym])

                rows.append(rec)
            except Exception:
                continue

        # Inference Loop
        picks = []
        for r in rows:
            # We must process row by row because models might differ (different feature sets?)
            # Or group by direction?
            # Grouping by direction is safer.
            pass

        # Group by direction
        df_all = pd.DataFrame(rows)
        if not df_all.empty:
            for d, grp in df_all.groupby("direction"):
                # All rows in grp share same models and cols (assuming training consistency)
                # But check first row
                first = grp.iloc[0]
                model_mr = first["model_mr"]
                model_tf = first["model_tf"]
                fcols = first["feat_cols"]

                # Prepare X
                Xint = apply_interaction_toggles(grp, cfg["causal_cols"], ["G_VOL", "G_TREND"], drop_raw=cfg["drop_raw_causal"])

                # Align cols
                # Filter to fcols
                for c in fcols:
                    if c not in Xint.columns: Xint[c] = 0.0
                Xpred = Xint[fcols].fillna(0.0).astype(np.float32)

                p_mr = model_mr.predict(Xpred)
                p_tf, _ = model_tf.predict(Xpred)

                score = p_tf - p_mr # Regime Score

                # Trend direction scaling
                # If d="up", trend > 0. Score * 1.
                # If d="down", trend <= 0. Score * -1.
                sign = 1.0 if d == "up" else -1.0
                final_score = score * sign

                # Collect
                for i, sym in enumerate(grp["symbol"]):
                    picks.append((sym, final_score[i]))

        # Sort and Pick
        picks.sort(key=lambda x: x[1], reverse=True) # Descending score

        # Top K Longs (Score > thr)
        longs = [(s, sc) for s, sc in picks if sc > cfg["thr_long"]]
        # Bottom K Shorts (Score < thr) -> Actually Score < -thr
        shorts = [(s, sc) for s, sc in picks if sc < cfg["thr_short"]]

        longs = longs[:cfg["k_long"]]
        shorts = sorted(shorts, key=lambda x: x[1])[:cfg["k_short"]] # Lowest negative

        # Execute
        final_orders = []
        for s, sc in longs: final_orders.append((s, "long", sc))
        for s, sc in shorts: final_orders.append((s, "short", sc))

        for sym, side, score in final_orders:
            atr = float(feats["atr_pct"].loc[ts_sig, sym])
            entry_px = float(c.loc[ts_sig, sym])
            ts_risk = TrailingStop(
                entry_px=entry_px, side=side, atr_val=atr,
                k_sl=cfg["risk_k_sl"], k_trail_start=cfg["risk_k_trail_start"], k_trail_dist=cfg["risk_k_trail_dist"]
            )
            pos = {
                "symbol": sym, "side": side, "entry_px": entry_px,
                "entry_ts": ts_sig.isoformat(), "score": float(score),
                "risk_state": ts_risk.to_dict(), "run_id": run_id
            }
            state.set_position(sym, pos)
            tprint(f"ENTRY {side} {sym} @ {entry_px} (score={score:.4f})")
            logger.log(ts_sig, {"event": "entry", "symbol": sym, "side": side, "score": score})

    state.set_last_ts_sig(ts_sig)
    logger.log(ts_sig, {"n_candidates": len(candidates), "run_id": run_id})
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
