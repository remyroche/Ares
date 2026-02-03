import numpy as np
import pandas as pd

from extreme_price_movements.utils import tprint
from extreme_price_movements.models import map_pred_to_score
from extreme_price_movements.risk import TrailingStop
from extreme_price_movements.training import (
    compute_p_exhaustion_at_t,
    select_best_horizon,
    apply_interaction_toggles,
    # select_best_horizon likely needs to be updated to return MRModel/TFModel
)

def entry_price_next_hour_open(panel_open, ts_entry, symbol):
    try:
        px = panel_open.loc[ts_entry, symbol]
        return float(px) if pd.notna(px) and px > 0 else np.nan
    except Exception:
        return np.nan

def simulate_trade_hourly(o_s, h_s, l_s, c_s, feats_s, ts_entry, entry_px, side, cfg, max_hold_hours):
    """
    Simulates trade using TrailingStop (vol-adjusted).
    """
    if np.isnan(entry_px) or entry_px <= 0:
        return 0.0, ts_entry, "no_entry"

    # Get ATR at entry time (ts_sig = ts_entry - 1h)
    ts_sig = ts_entry - pd.Timedelta(hours=1)
    if ts_sig not in feats_s.index:
        # Fallback if missing data?
        atr = 0.02 # default 2%
    else:
        atr = float(feats_s.loc[ts_sig]) # feats_s is atr_pct series

    # Trailing Stop Manager
    ts_manager = TrailingStop(
        entry_px=entry_px,
        side=side,
        atr_val=atr,
        k_sl=cfg["risk_k_sl"],
        k_trail_start=cfg["risk_k_trail_start"],
        k_trail_dist=cfg["risk_k_trail_dist"]
    )

    end_ts = ts_entry + pd.Timedelta(hours=max_hold_hours)
    path = o_s.loc[ts_entry:end_ts].index
    if len(path) == 0:
        return 0.0, ts_entry, "no_path"

    for ts in path:
        hh = h_s.loc[ts]; ll = l_s.loc[ts]; cc = c_s.loc[ts]
        if np.isnan(hh) or np.isnan(ll) or np.isnan(cc):
            continue

        stopped, exit_px, reason = ts_manager.update(hh, ll, cc)
        if stopped:
            if side == "long":
                return (exit_px / entry_px) - 1.0, ts, reason
            else:
                return (entry_px / exit_px) - 1.0, ts, reason

    last_ts = path[-1]
    last_close = c_s.loc[last_ts]
    if side == "long":
        return (last_close / entry_px) - 1.0, last_ts, "time_exit"
    else:
        return (entry_px / last_close) - 1.0, last_ts, "time_exit"

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

def hourly_engine_backtest(panel, feats, mkt_gates, cfg, symbols_all):
    o, h, l, c = panel["open"], panel["high"], panel["low"], panel["close"]
    idx = c.index

    equity = 1.0
    eq_curve = []
    trades = []

    fee_rt = cfg["fee_bps"] / 1e4
    borrow_hourly = (cfg["borrow_apr"] / 365.0) / 24.0

    # exhaustion history
    p_exh_hist = pd.DataFrame(index=idx, columns=symbols_all, dtype=np.float32)

    # fitted state (retrained daily)
    best_mr = None
    best_tf = None
    last_train_day = None

    # warmup start
    warm = max(cfg["train_lookback_hours"], cfg["exh_train_lookback_hours"]) + max(cfg["label_horizons_hours"]) + 48
    start_ts = idx.min() + pd.Timedelta(hours=warm)
    start_ts = idx[idx.get_indexer([start_ts], method="backfill")[0]]

    tprint(f"Engine: start_ts={start_ts}  idx={len(idx)}  symbols={len(symbols_all)}")

    for ts in idx[idx >= start_ts]:
        ts_entry = ts + pd.Timedelta(hours=1)
        if ts_entry not in idx:
            break

        # (A) exhaustion sensor first (p_exh_out)
        p_exh_ts = compute_p_exhaustion_at_t(panel, feats, mkt_gates, cfg, ts, symbols_all)
        p_exh_hist.loc[ts, symbols_all] = p_exh_ts.values.astype(np.float32)

        # (B) daily retrain + horizon select (1)
        ts_day = ts.floor("D")
        if last_train_day is None or ts_day != last_train_day:
            tprint(f"TRAIN day-roll: {ts_day}")
            # Ensure training.py returns new model instances
            best_mr = select_best_horizon(panel, feats, mkt_gates, cfg, symbols_all, ts, p_exh_hist, model_kind="mr")
            best_tf = select_best_horizon(panel, feats, mkt_gates, cfg, symbols_all, ts, p_exh_hist, model_kind="tf")
            last_train_day = ts_day

        if best_mr is None or best_tf is None:
            eq_curve.append((ts, equity))
            continue

        model_mr = best_mr["model"]; feat_cols = best_mr["feat_cols"]
        model_tf = best_tf["model"]

        # RuleCleaner/Stability were internal to MRModel/TFModel now, hopefully?
        # If select_best_horizon returns our new classes, they have internal logic.
        # But we need to make sure we construct X properly.

        # (C) hourly candidate selection
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

        # build prediction frame
        t_exh_lag = ts - pd.Timedelta(hours=1)
        p_lag = (p_exh_hist.loc[t_exh_lag, trade_syms] if t_exh_lag in p_exh_hist.index else pd.Series(index=trade_syms, data=np.nan)).astype(np.float32)
        p_out = p_exh_ts.reindex(trade_syms).astype(np.float32)

        rows = []
        for sym in trade_syms:
            try:
                rows.append({
                    "symbol": sym,
                    # Base features need to match what models expect
                    # Assuming feat_cols are consistent
                    **{k: float(feats[k].loc[ts, sym]) for k in feat_cols if k in feats},
                    # Add market/gates/exh
                    "mkt_ret24h": float(mrk["mkt_ret24h"]),
                    "mkt_ret6h": float(mrk["mkt_ret6h"]),
                    "mkt_trend": float(mrk["mkt_trend"]),
                    "mkt_rv": float(mrk["mkt_rv"]),
                    "G_VOL": int(mrk["G_VOL"]),
                    "G_TREND": int(mrk["G_TREND"]),
                    "p_exh_lag1": float(p_lag.loc[sym]) if pd.notna(p_lag.loc[sym]) else np.nan,
                    "p_exh_out": float(p_out.loc[sym]) if pd.notna(p_out.loc[sym]) else np.nan,
                })
            except Exception:
                continue

        if not rows:
             eq_curve.append((ts, equity))
             continue

        Xdf = pd.DataFrame(rows).dropna(subset=["p_exh_lag1"]) # p_exh_out not needed for prediction, only for weighting?
        # Actually p_exh_out is current hour prediction.
        if Xdf.empty:
            eq_curve.append((ts, equity))
            continue

        # interactionize
        Xint = apply_interaction_toggles(
            Xdf,
            causal_cols=cfg["causal_cols"],
            gate_cols=["G_VOL","G_TREND"],
            drop_raw=cfg["drop_raw_causal"]
        )

        # align columns
        for col in feat_cols:
            if col not in Xint.columns:
                Xint[col] = 0.0 # or nan

        # New Model Prediction
        # Assuming model_mr is MRModel instance
        Xpred = Xint[feat_cols].fillna(0.0).astype(np.float32)

        pred_mr = model_mr.predict(Xpred)
        pred_tf, disp_tf = model_tf.predict(Xpred)

        # SCORE = TF - MR
        # TF is continuation (same sign as trend?), MR is reversion (opposite sign?)
        # Wait, targets are returns.
        # If trend is UP:
        # TF predicts +ve return (Continuation).
        # MR predicts -ve return (Reversion).
        # Score = +ve - (-ve) = Large +ve (Long).
        # If trend is DOWN:
        # TF predicts -ve return.
        # MR predicts +ve return.
        # Score = -ve - (+ve) = Large -ve (Short).
        # This logic holds.

        score_raw = pred_tf - pred_mr

        # Long/Short selection based on thresholds
        # score_map is likely tanh.
        # But here we threshold on raw diff?
        # User: "Backtest the thresholds above/below which we should open short/long"

        syms_pred = Xint["symbol"].tolist()
        res = pd.DataFrame({
            "symbol": syms_pred,
            "score": score_raw,
            "pred_tf": pred_tf,
            "pred_mr": pred_mr,
            "disp_tf": disp_tf
        })

        longs = res[res["score"] > cfg["thr_long"]].sort_values("score", ascending=False).head(cfg["k_long"])
        shorts = res[res["score"] < cfg["thr_short"]].sort_values("score", ascending=True).head(cfg["k_short"])

        picks = []
        for _, row in longs.iterrows():
            # Normalized score for sizing
            # user: "score = TF - MR (both of which are normalised)"
            # They are likely small returns (e.g. 0.01).
            # map_pred_to_score scales them up.
            s = map_pred_to_score(row["score"], cfg["score_map"], cfg["score_scale"])
            picks.append((row["symbol"], "long", s, row["score"]))

        for _, row in shorts.iterrows():
            s = map_pred_to_score(-row["score"], cfg["score_map"], cfg["score_scale"])
            picks.append((row["symbol"], "short", s, row["score"]))

        total_score = sum(p[2] for p in picks)
        if total_score <= 0:
            eq_curve.append((ts, equity))
            continue

        gross_cap = float(cfg["wallet_gross_cap"])
        weights = [(sym, side, gross_cap * (score / total_score), raw_score) for sym, side, score, raw_score in picks]

        pnl = 0.0
        for sym, side, w, raw_score in weights:
            entry_px = entry_price_next_hour_open(o, ts_entry, sym)
            if np.isnan(entry_px) or entry_px <= 0:
                continue

            rr, exit_ts, why = simulate_trade_hourly(
                o_s=o[sym], h_s=h[sym], l_s=l[sym], c_s=c[sym],
                feats_s=feats["atr_pct"].loc[:, sym], # pass atr series
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
                "entry_ts": ts_entry,
                "exit_ts": exit_ts,
                "symbol": sym,
                "side": side,
                "weight": w,
                "score_raw": raw_score,
                "ret": float(rr),
                "pnl_contrib": float(w * rr),
                "exit_reason": why,
                "disp_tf": float(res.loc[res["symbol"]==sym, "disp_tf"].values[0])
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
