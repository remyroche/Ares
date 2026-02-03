import numpy as np
import pandas as pd

from utils import tprint
from models import map_pred_to_score
from training import (
    compute_p_exhaustion_at_t,
    select_best_horizon,
)
from training import apply_interaction_toggles

def entry_price_next_hour_open(panel_open, ts_entry, symbol):
    try:
        px = panel_open.loc[ts_entry, symbol]
        return float(px) if pd.notna(px) and px > 0 else np.nan
    except Exception:
        return np.nan

def simulate_trade_hourly(o_s, h_s, l_s, c_s, entry_ts, entry_px, side, tp, sl, max_hold_hours):
    if np.isnan(entry_px) or entry_px <= 0:
        return 0.0, entry_ts, "no_entry"

    if side == "long":
        tp_px = entry_px * (1 + tp)
        sl_px = entry_px * (1 - sl)
    else:
        tp_px = entry_px * (1 - tp)
        sl_px = entry_px * (1 + sl)

    end_ts = entry_ts + pd.Timedelta(hours=max_hold_hours)
    path = o_s.loc[entry_ts:end_ts].index
    if len(path) == 0:
        return 0.0, entry_ts, "no_path"

    for ts in path:
        hh = h_s.loc[ts]; ll = l_s.loc[ts]; cc = c_s.loc[ts]
        if np.isnan(hh) or np.isnan(ll) or np.isnan(cc):
            continue

        if side == "long":
            hit_tp = hh >= tp_px
            hit_sl = ll <= sl_px
            if hit_tp and hit_sl:
                return (sl_px / entry_px) - 1.0, ts, "sl_same_hour"
            if hit_tp:
                return (tp_px / entry_px) - 1.0, ts, "tp"
            if hit_sl:
                return (sl_px / entry_px) - 1.0, ts, "sl"
        else:
            hit_tp = ll <= tp_px
            hit_sl = hh >= sl_px
            if hit_tp and hit_sl:
                return (entry_px / sl_px) - 1.0, ts, "sl_same_hour"
            if hit_tp:
                return (entry_px / tp_px) - 1.0, ts, "tp"
            if hit_sl:
                return (entry_px / sl_px) - 1.0, ts, "sl"

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
            best_mr = select_best_horizon(panel, feats, mkt_gates, cfg, symbols_all, ts, p_exh_hist, model_kind="mr")
            best_tf = select_best_horizon(panel, feats, mkt_gates, cfg, symbols_all, ts, p_exh_hist, model_kind="tf")
            last_train_day = ts_day

        if best_mr is None or best_tf is None:
            eq_curve.append((ts, equity))
            continue

        model_mr = best_mr["model"]; feat_cols = best_mr["feat_cols"]; cleaner_mr = best_mr["cleaner"]; stable_mr = best_mr["stable_mask"]
        model_tf = best_tf["model"]; cleaner_tf = best_tf["cleaner"]; stable_tf = best_tf["stable_mask"]

        # (C) hourly candidate selection (compute only on selected symbols)
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

        # build prediction frame for selected symbols only
        t_exh_lag = ts - pd.Timedelta(hours=1)
        p_lag = (p_exh_hist.loc[t_exh_lag, trade_syms] if t_exh_lag in p_exh_hist.index else pd.Series(index=trade_syms, data=np.nan)).astype(np.float32)
        p_out = p_exh_ts.reindex(trade_syms).astype(np.float32)

        rows = []
        for sym in trade_syms:
            try:
                rows.append({
                    "symbol": sym,

                    # same set as training (must match)
                    "a_ret12h": float(feats["ret12h"].loc[ts, sym]),
                    "a_ret16h": float(feats["ret16h"].loc[ts, sym]),
                    "a_ret20h": float(feats["ret20h"].loc[ts, sym]),
                    "a_ret24h": float(feats["ret24h"].loc[ts, sym]),
                    "a_ret28h": float(feats["ret28h"].loc[ts, sym]),

                    "a_ret6h": float(feats["ret6h"].loc[ts, sym]),
                    "a_ret1h_z": float(feats["ret1h_z"].loc[ts, sym]),
                    "a_atr": float(feats["atr_pct"].loc[ts, sym]),
                    "a_rsi": float(feats["rsi"].loc[ts, sym]),
                    "a_volz": float(feats["vol_z"].loc[ts, sym]),
                    "a_trend": float(feats["trend_pct"].loc[ts, sym]),
                    "a_rv24": float(feats["rv_24h"].loc[ts, sym]),
                    "a_range": float(feats["range_pct"].loc[ts, sym]),
                    "a_gap": float(feats["gap_pct"].loc[ts, sym]),
                    "a_body": float(feats["body_pct"].loc[ts, sym]),
                    "a_dist_ema_fast": float(feats["dist_ema_fast"].loc[ts, sym]),
                    "a_dist_ema_slow": float(feats["dist_ema_slow"].loc[ts, sym]),
                    "a_roc_div": float(feats["roc_div"].loc[ts, sym]),
                    "a_vol_price_spread": float(feats["vol_price_spread"].loc[ts, sym]),
                    "a_funding_proxy": float(feats["a_funding_proxy"].loc[ts, sym]),

                    "sin_hod": float(feats["sin_hod"].loc[ts, sym]),
                    "cos_hod": float(feats["cos_hod"].loc[ts, sym]),
                    "sin_dow": float(feats["sin_dow"].loc[ts, sym]),
                    "cos_dow": float(feats["cos_dow"].loc[ts, sym]),

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

        Xdf = pd.DataFrame(rows).dropna(subset=["p_exh_lag1","p_exh_out"])
        if Xdf.empty:
            eq_curve.append((ts, equity))
            continue

        # interactionize
        Xint = apply_interaction_toggles(
            Xdf.drop(columns=["p_exh_out"]),
            causal_cols=cfg["causal_cols"],
            gate_cols=["G_VOL","G_TREND"],
            drop_raw=cfg["drop_raw_causal"]
        )

        # align columns to feat_cols
        for col in feat_cols:
            if col not in Xint.columns:
                Xint[col] = np.nan
        Xint = Xint[["symbol"] + feat_cols].dropna()
        if Xint.empty:
            eq_curve.append((ts, equity))
            continue

        syms_pred = Xint["symbol"].tolist()
        p_out_vec = Xdf.set_index("symbol").reindex(syms_pred)["p_exh_out"].to_numpy(dtype=np.float32)

        # RuleCleaner + stability masking
        Xm = Xint[feat_cols].astype(np.float32)

        Xm_mr = cleaner_mr.transform(Xm).astype(np.float32)
        Xm_tf = cleaner_tf.transform(Xm).astype(np.float32)

        Xmr = Xm_mr.to_numpy(dtype=np.float32, copy=False)
        Xtf = Xm_tf.to_numpy(dtype=np.float32, copy=False)

        # mask unstable features globally (per model)
        if stable_mr is not None:
            kept = list(Xm_mr.columns)
            idx_mr = [feat_cols.index(c) for c in kept if c in feat_cols]
            mask = stable_mr[idx_mr]
            Xmr[:, ~mask] = 0.0
        if stable_tf is not None:
            kept = list(Xm_tf.columns)
            idx_tf = [feat_cols.index(c) for c in kept if c in feat_cols]
            mask = stable_tf[idx_tf]
            Xtf[:, ~mask] = 0.0

        pred_mr = model_mr.predict(Xmr).astype(np.float32)
        pred_tf = model_tf.predict(Xtf).astype(np.float32)

        # Tactical blend (exhaustion as output regime selector)
        mixed = (p_out_vec * pred_mr) + ((1.0 - p_out_vec) * pred_tf)

        # Long/Short selection
        pred_ser = pd.Series(mixed, index=syms_pred)
        long_syms = pred_ser[pred_ser >= cfg["thr_long"]].sort_values(ascending=False).head(cfg["k_long"]).index.tolist()
        short_syms = pred_ser[pred_ser <= cfg["thr_short"]].sort_values(ascending=True).head(cfg["k_short"]).index.tolist()
        if not long_syms and not short_syms:
            eq_curve.append((ts, equity))
            continue

        picks = []
        for sym in long_syms:
            score = map_pred_to_score(pred_ser[sym], cfg["score_map"], cfg["score_scale"])
            picks.append((sym, "long", score, float(pred_ser[sym])))
        for sym in short_syms:
            score = map_pred_to_score(-pred_ser[sym], cfg["score_map"], cfg["score_scale"])
            picks.append((sym, "short", score, float(pred_ser[sym])))

        total_score = sum(p[2] for p in picks)
        if total_score <= 0:
            eq_curve.append((ts, equity))
            continue

        gross_cap = float(cfg["wallet_gross_cap"])
        weights = [(sym, side, gross_cap * (score / total_score), pred) for sym, side, score, pred in picks]

        pnl = 0.0
        for sym, side, w, pred in weights:
            entry_px = entry_price_next_hour_open(o, ts_entry, sym)
            if np.isnan(entry_px) or entry_px <= 0:
                continue

            rr, exit_ts, why = simulate_trade_hourly(
                o_s=o[sym], h_s=h[sym], l_s=l[sym], c_s=c[sym],
                entry_ts=ts_entry,
                entry_px=entry_px,
                side=side,
                tp=cfg["tp"],
                sl=cfg["sl"],
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
                "pred": pred,
                "p_exh_out": float(p_out.loc[sym]) if sym in p_out.index else np.nan,
                "ret": float(rr),
                "pnl_contrib": float(w * rr),
                "exit_reason": why,
                "H_mr": best_mr["H"],
                "H_tf": best_tf["H"],
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
