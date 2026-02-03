import numpy as np
import pandas as pd

from extreme_price_movements.utils import tprint
from extreme_price_movements.models import map_pred_to_score
from extreme_price_movements.risk import TrailingStop
from extreme_price_movements.training import (
    compute_p_exhaustion_at_t,
    select_best_horizon,
)
from extreme_price_movements.feature_transforms import apply_interaction_toggles
from extreme_price_movements.simulation import simulate_trade_hourly, entry_price_next_hour_open

def select_trade_candidates_hourly(feats, ts, syms, pct=0.05, min_n=10, max_n=60, metric="dist_ema_fast"):
    if ts not in feats[metric].index:
        return [], []
    s = feats[metric].loc[ts, syms].dropna()
    if s.empty:
        return [], []
    n = len(s)
    k = max(min_n, int(n * pct))
    k = min(k, max_n)
    top = s.sort_values(ascending=False).head(k).index.tolist()
    bot = s.sort_values(ascending=True).head(k).index.tolist()
    return top, bot

def generate_signals(ts, candidates, panel, feats, mkt_gates, cfg, alpha_models, meta_models, p_exh_hist, p_exh_ts, active_positions):
    """
    Generates desired entries based on models.
    """
    if not candidates:
        return []

    if ts not in mkt_gates.index:
        return []
    mrk = mkt_gates.loc[ts]
    ts_lag = ts - pd.Timedelta(hours=1)
    trend_df = feats.get("trend_pct")

    rows = []
    for sym in candidates:
        if sym in active_positions:
            continue

        try:
            t_val = 0.0
            if trend_df is not None and sym in trend_df.columns: t_val = float(trend_df.loc[ts, sym])
            direction = "up" if t_val > 0 else "down"
            m_bundle = alpha_models.get(direction)
            if not m_bundle or not m_bundle["mr"] or not m_bundle["tf"]: continue

            model_mr = m_bundle["mr"]["model"]; model_tf = m_bundle["tf"]["model"]
            feat_cols = m_bundle["mr"]["feat_cols"]; meta_model = meta_models.get(direction)

            p_lag = 0.5
            if ts_lag in p_exh_hist.index and sym in p_exh_hist.columns: p_lag = float(p_exh_hist.loc[ts_lag, sym])

            rec = {
                "symbol": sym, "direction": direction, "model_mr": model_mr, "model_tf": model_tf, "meta_model": meta_model, "feat_cols": feat_cols,
                "mkt_ret24h": float(mrk["mkt_ret24h"]), "mkt_ret6h": float(mrk["mkt_ret6h"]), "mkt_trend": float(mrk["mkt_trend"]), "mkt_rv": float(mrk["mkt_rv"]),
                "G_VOL": int(mrk["G_VOL"]), "G_TREND": int(mrk["G_TREND"]), "p_exh_lag1": p_lag
            }
            for k in feat_cols:
                if k in feats: rec[k] = float(feats[k].loc[ts, sym])
            # Meta Feats
            for mk in ["a_rv24", "a_volz", "a_rsi", "dist_ema_fast", "atr_slope", "dist_vwap_norm", "mom_accel"]:
                if mk in feats: rec[mk] = float(feats[mk].loc[ts, sym])

            rows.append(rec)
        except Exception: continue

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
                sym = grp.loc[idx, "symbol"]
                s_score = score[i]
                dom = "mr" if p_mr[i] > p_tf[i] else "tf"
                score_raw_list.append((sym, s_score, dom))

    score_raw_list.sort(key=lambda x: x[1], reverse=True)
    longs = [x for x in score_raw_list if x[1] > cfg["thr_long"]][:cfg["k_long"]]
    shorts = sorted([x for x in score_raw_list if x[1] < cfg["thr_short"]], key=lambda x: x[1])[:cfg["k_short"]]

    final_signals = []
    for s, sc, dom in longs: final_signals.append({"symbol": s, "side": "long", "score": sc, "dom": dom})
    for s, sc, dom in shorts: final_signals.append({"symbol": s, "side": "short", "score": sc, "dom": dom})

    return final_signals

def execute_signals(signals, state, exchange, panel, feats, ts_sig, cfg, logger, risk_conf):
    """
    Executes entry orders for signals.
    """
    if not signals:
        return

    active_syms = state.get_positions().keys()
    granular_risk = risk_conf.get("granular_risk", {}) if risk_conf else {}
    c = panel["close"]
    gross_cap = float(cfg["wallet_gross_cap"])

    total_wt = sum(abs(x["score"]) for x in signals)
    if total_wt <= 0: return

    for sig in signals:
        sym = sig["symbol"]
        side = sig["side"]
        score = sig["score"]
        dom = sig["dom"]

        if sym in active_syms: continue

        w_alloc = gross_cap * (abs(score) / total_wt)
        atr = float(feats["atr_pct"].loc[ts_sig, sym])
        entry_px = float(c.loc[ts_sig, sym])

        risk_key = f"risk_{side}_{dom}"
        rp = granular_risk.get(risk_key, {})

        k_sl = rp.get("k_sl", cfg["risk_k_sl"])
        k_ts = rp.get("k_trail_start", cfg["risk_k_trail_start"])
        k_td = rp.get("k_trail_dist", cfg["risk_k_trail_dist"])
        sc_scale = rp.get("score_scale", 0.0)

        adj = (1.0 + sc_scale * abs(score))
        k_sl_adj = k_sl * adj

        ts_risk = TrailingStop(
            entry_px=entry_px, side=side, atr_val=atr,
            k_sl=k_sl_adj, k_trail_start=k_ts, k_trail_dist=k_td
        )
        pos = {
            "symbol": sym, "side": side, "entry_px": entry_px, "entry_ts": ts_sig.isoformat(),
            "score": float(score), "weight": float(w_alloc), "risk_state": ts_risk.to_dict(),
            "run_id": state.state.get("run_id")
        }

        # Here we would send order to exchange...
        # For now just update state
        state.set_position(sym, pos)
        tprint(f"ENTRY {side} {sym} @ {entry_px} (score={score:.4f}, w={w_alloc:.4f}, dom={dom})")
        logger.log(ts_sig, {"event": "entry", "symbol": sym, "side": side, "score": score, "weight": w_alloc, "dom": dom})


def hourly_engine_backtest(panel, feats, mkt_gates, cfg, symbols_all):
    o, h, l, c = panel["open"], panel["high"], panel["low"], panel["close"]
    idx = c.index

    equity = 1.0
    eq_curve = []
    trades = []

    fee_rt = cfg["fee_bps"] / 1e4
    borrow_hourly = (cfg["borrow_apr"] / 365.0) / 24.0

    p_exh_hist = pd.DataFrame(index=idx, columns=symbols_all, dtype=np.float32)

    alpha_models = None
    meta_models = None
    last_train_day = None

    warm = max(cfg["train_lookback_hours"], cfg["exh_train_lookback_hours"]) + max(cfg["label_horizons_hours"]) + 48
    start_ts = idx.min() + pd.Timedelta(hours=warm)
    start_ts = idx[idx.get_indexer([start_ts], method="backfill")[0]]

    tprint(f"Engine: start_ts={start_ts}  idx={len(idx)}  symbols={len(symbols_all)}")

    for ts in idx[idx >= start_ts]:
        ts_entry = ts + pd.Timedelta(hours=1)
        if ts_entry not in idx:
            break

        p_exh_ts = compute_p_exhaustion_at_t(panel, feats, mkt_gates, cfg, ts, symbols_all)
        p_exh_hist.loc[ts, symbols_all] = p_exh_ts.values.astype(np.float32)

        ts_day = ts.floor("D")
        if last_train_day is None or ts_day != last_train_day:
            tprint(f"TRAIN day-roll: {ts_day}")
            bundle = select_best_horizon(panel, feats, mkt_gates, cfg, symbols_all, ts, p_exh_hist)
            alpha_models = bundle["alpha_models"]
            meta_models = bundle["meta_models"]
            last_train_day = ts_day

        if not alpha_models:
            eq_curve.append((ts, equity))
            continue

        # Selection
        top_syms, bot_syms = select_trade_candidates_hourly(
            feats=feats, ts=ts, syms=symbols_all,
            pct=cfg["trade_extreme_pct"], min_n=cfg["trade_extreme_min"], max_n=cfg["trade_extreme_max"],
            metric=cfg["trade_deviation_metric"]
        )
        trade_syms = list(set(top_syms) | set(bot_syms))
        if not trade_syms:
            eq_curve.append((ts, equity))
            continue

        if ts not in mkt_gates.index:
            eq_curve.append((ts, equity))
            continue
        mrk = mkt_gates.loc[ts]

        t_exh_lag = ts - pd.Timedelta(hours=1)

        # Build Rows
        rows = []
        trend_df = feats.get("trend_pct")

        for sym in trade_syms:
            try:
                # Trend Dir
                t_val = 0.0
                if trend_df is not None and sym in trend_df.columns:
                    t_val = float(trend_df.loc[ts, sym])
                direction = "up" if t_val > 0 else "down"

                # Get Models
                m_bundle = alpha_models.get(direction)
                if not m_bundle or not m_bundle["mr"] or not m_bundle["tf"]: continue

                model_mr = m_bundle["mr"]["model"]
                model_tf = m_bundle["tf"]["model"]
                feat_cols = m_bundle["mr"]["feat_cols"]
                meta_model = meta_models.get(direction) # Could be None

                p_lag = 0.5
                if t_exh_lag in p_exh_hist.index and sym in p_exh_hist.columns:
                    p_lag = float(p_exh_hist.loc[t_exh_lag, sym])
                p_out = 0.0
                if sym in p_exh_ts.index: p_out = float(p_exh_ts[sym])

                rec = {
                    "symbol": sym,
                    "direction": direction,
                    "model_mr": model_mr,
                    "model_tf": model_tf,
                    "meta_model": meta_model,
                    "feat_cols": feat_cols,
                    "mkt_ret24h": float(mrk["mkt_ret24h"]),
                    "mkt_ret6h": float(mrk["mkt_ret6h"]),
                    "mkt_trend": float(mrk["mkt_trend"]),
                    "mkt_rv": float(mrk["mkt_rv"]),
                    "G_VOL": int(mrk["G_VOL"]),
                    "G_TREND": int(mrk["G_TREND"]),
                    "p_exh_lag1": p_lag
                }

                # Feats
                for k in feat_cols:
                    if k in feats:
                        rec[k] = float(feats[k].loc[ts, sym])

                # Feats for Meta (Vol etc)
                # MetaModel needs specific names.
                # X passed to MetaModel is constructed from features.
                # We need to ensure we have them in `rec` or lookup.
                if "a_rv24" in feats: rec["a_rv24"] = float(feats["a_rv24"].loc[ts, sym])
                if "a_volz" in feats: rec["a_volz"] = float(feats["a_volz"].loc[ts, sym])
                if "a_rsi" in feats: rec["a_rsi"] = float(feats["a_rsi"].loc[ts, sym])
                if "dist_ema_fast" in feats: rec["dist_ema_fast"] = float(feats["dist_ema_fast"].loc[ts, sym])

                rows.append(rec)
            except Exception:
                continue

        if not rows:
             eq_curve.append((ts, equity))
             continue

        df_all = pd.DataFrame(rows)
        score_raw_list = []

        # Group by direction
        for d, grp in df_all.groupby("direction"):
            first = grp.iloc[0]
            model_mr = first["model_mr"]
            model_tf = first["model_tf"]
            meta_model = first["meta_model"]
            fcols = first["feat_cols"]

            Xint = apply_interaction_toggles(grp, cfg["causal_cols"], ["G_VOL","G_TREND"], drop_raw=cfg["drop_raw_causal"])

            for c in fcols:
                if c not in Xint.columns: Xint[c] = 0.0
            Xpred = Xint[fcols].fillna(0.0).astype(np.float32)

            p_mr = model_mr.predict(Xpred)
            p_tf = model_tf.predict(Xpred)

            # Meta Model Scoring
            if meta_model:
                X_meta = meta_model.prepare_meta_features(p_tf, p_mr, grp)
                score = meta_model.predict(X_meta)
            else:
                score = p_tf - p_mr

            # Trend direction scaling
            # Meta Model output is "Position". If it learned correctly, it handles sign.
            # But let's verify if MetaModel logic assumes directional return.
            # If we trained Meta on Y = Return, then Output = E[Return].
            # If Trend Up, Y_TF predicts +ve. Y_MR predicts -ve (reversion).
            # If Price continues Up, Return > 0.
            # So Score > 0 -> Long.
            # If MetaModel is present, we use its output directly.
            # If not, use (TF - MR) * sign.

            if not meta_model:
                sign = 1.0 if d == "up" else -1.0
                score = score * sign

            for i, idx in enumerate(grp.index):
                score_raw_list.append((grp.loc[idx, "symbol"], score[i]))

        # Sorting
        score_raw_list.sort(key=lambda x: x[1], reverse=True)

        longs = [x for x in score_raw_list if x[1] > cfg["thr_long"]]
        shorts = [x for x in score_raw_list if x[1] < cfg["thr_short"]]

        # Sizing
        # Share capacity based on relative scores (Req 9)
        # We pick top K.
        # Then weight by abs(score) / sum(abs(scores)).

        picks_long = longs[:cfg["k_long"]]
        picks_short = sorted(shorts, key=lambda x: x[1])[:cfg["k_short"]]

        final_orders = []
        for s, sc in picks_long: final_orders.append((s, "long", sc))
        for s, sc in picks_short: final_orders.append((s, "short", sc))

        total_wt = sum(abs(x[2]) for x in final_orders)
        if total_wt == 0:
            eq_curve.append((ts, equity))
            continue

        gross_cap = float(cfg["wallet_gross_cap"])

        pnl = 0.0
        for sym, side, raw_score in final_orders:
            w = gross_cap * (abs(raw_score) / total_wt)

            entry_px = entry_price_next_hour_open(o, ts_entry, sym)
            if np.isnan(entry_px) or entry_px <= 0: continue

            rr, exit_ts, why = simulate_trade_hourly(
                o_s=o[sym], h_s=h[sym], l_s=l[sym], c_s=c[sym],
                feats_s=feats["atr_pct"].loc[:, sym],
                entry_ts=ts_entry,
                entry_px=entry_px,
                side=side,
                cfg=cfg,
                max_hold_hours=cfg["hold_hours"]
            )

            if side == "short":
                rr -= borrow_hourly * float(cfg["hold_hours"])
            rr -= 2.0 * fee_rt

            pnl += w * rr
            trades.append({
                "ts_sig": ts,
                "symbol": sym,
                "side": side,
                "weight": w,
                "score_raw": raw_score,
                "ret": float(rr),
            })

        equity *= (1.0 + pnl)
        eq_curve.append((ts, equity))

    eq = pd.Series({t: e for t, e in eq_curve}).sort_index()
    trades_df = pd.DataFrame(trades)

    if len(eq) > 2:
        dr = eq.pct_change().dropna()
        sharpe = (dr.mean() / (dr.std(ddof=0) + 1e-12)) * np.sqrt(365.0 * 24.0)
        max_dd = (eq / eq.cummax() - 1.0).min()
    else:
        sharpe = np.nan
        max_dd = np.nan

    stats = {
        "total_return": float(eq.iloc[-1] - 1.0) if len(eq) else np.nan,
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "n_trades": int(len(trades_df)) if not trades_df.empty else 0,
    }
    return eq, trades_df, stats
