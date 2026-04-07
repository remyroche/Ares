from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from extreme_price_movements.path_utils import resolve_reports_dir
from extreme_price_movements.purged_cv import PurgedKFold
from extreme_price_movements.utils import tprint
from extreme_price_movements.metrics import _stable_equity_and_drawdown

_NS_PER_DAY = 86_400_000_000_000 # DEPRECATED, kept for diff compat. Correct is 86400 * 1e9


@dataclass(frozen=True)
class LabelPolicy:
    sl_atr_mult: float
    tp_sl_ratio: float
    max_hold_bars: int
    trail_activate_atr: float
    giveback_pct: float
    early_exit_deadline_bars: int
    early_exit_mfe_atr: float


def _policy_to_prices(entry: float, atr: float, is_long: bool, p: LabelPolicy) -> Tuple[float, float]:
    sl_dist = max(float(p.sl_atr_mult) * max(float(atr), 1e-9), 1e-9)
    tp_dist = float(p.tp_sl_ratio) * sl_dist
    if is_long:
        return entry + tp_dist, entry - sl_dist
    return entry - tp_dist, entry + sl_dist


def _simulate_with_policy(
    simulate_trade_exit_fn,
    entry_price: float,
    atr_entry: float,
    is_long: bool,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    policy: LabelPolicy,
    cost_pct: float,
) -> Tuple[float, str, int, int]:
    tp_price, sl_price = _policy_to_prices(entry_price, atr_entry, is_long, policy)
    peak = float(entry_price)
    trough = float(entry_price)
    max_bars = min(len(highs), max(1, int(policy.max_hold_bars)))

    early_idx = max_bars + 1
    for i in range(max_bars):
        h = float(highs[i])
        l = float(lows[i])
        if is_long:
            peak = max(peak, h)
            mfe_atr = (peak - entry_price) / max(atr_entry, 1e-9)
        else:
            trough = min(trough, l)
            mfe_atr = (entry_price - trough) / max(atr_entry, 1e-9)
        if (i + 1) == int(policy.early_exit_deadline_bars) and mfe_atr < float(policy.early_exit_mfe_atr):
            early_idx = i
            break

    if early_idx <= max_bars - 1:
        exit_price = float(closes[early_idx])
        exit_bar = int(early_idx)
        reason = "early"
    else:
        trailing_pct_eff = float(policy.giveback_pct) if float(policy.trail_activate_atr) <= 1.0 else 0.0
        exit_price, exit_bar, reason_i = simulate_trade_exit_fn(
            highs=np.asarray(highs[:max_bars], dtype=np.float64),
            lows=np.asarray(lows[:max_bars], dtype=np.float64),
            opens=np.asarray(closes[:max_bars], dtype=np.float64),
            closes=np.asarray(closes[:max_bars], dtype=np.float64),
            entry_price=float(entry_price),
            is_long=bool(is_long),
            tp_price=float(tp_price),
            sl_price=float(sl_price),
            trailing_pct=float(trailing_pct_eff),
            max_bars=int(max_bars),
        )
        reason = {0: "tp", 1: "sl", 2: "trail", 3: "timeout"}.get(int(reason_i), "timeout")

    if is_long:
        u = np.log(max(exit_price, 1e-12) / max(entry_price, 1e-12))
    else:
        u = np.log(max(entry_price, 1e-12) / max(exit_price, 1e-12))
    return float(u - cost_pct), reason, int(exit_bar), int(max_bars)


def _topq_select_indices(ts: np.ndarray, symbols: np.ndarray, score: np.ndarray, q: float) -> np.ndarray:
    out: List[int] = []
    df = pd.DataFrame({"ts": ts, "symbol": symbols.astype(str), "score": score, "i": np.arange(len(score))})
    for _, g in df.groupby("ts", sort=True):
        g2 = g.sort_values(["score", "symbol"], ascending=[False, True])
        k = max(1, int(np.ceil(float(q) * len(g2))))
        out.extend(g2.head(k)["i"].tolist())
    return np.asarray(out, dtype=int)


def _prepare_grouped_topq_inputs(ts: np.ndarray, symbols: np.ndarray) -> Dict[str, np.ndarray]:
    ts_ns = pd.to_datetime(ts, utc=True, errors="coerce").view("i8")
    sym_codes, _ = pd.factorize(np.asarray(symbols).astype(str), sort=True)
    valid = np.isfinite(ts_ns.astype(np.float64))
    order = np.lexsort((sym_codes[valid], ts_ns[valid]))
    valid_idx = np.flatnonzero(valid)[order]
    ts_sorted = ts_ns[valid_idx]
    starts = np.flatnonzero(np.r_[True, ts_sorted[1:] != ts_sorted[:-1]])
    ends = np.r_[starts[1:], len(valid_idx)]
    return {
        "valid_idx": valid_idx,
        "starts": starts,
        "ends": ends,
        "ts_ns": ts_ns,
    }


def _topq_select_indices_fast(score: np.ndarray, q: float, grouped: Dict[str, np.ndarray]) -> np.ndarray:
    valid_idx = grouped["valid_idx"]
    starts = grouped["starts"]
    ends = grouped["ends"]
    if len(valid_idx) == 0:
        return np.asarray([], dtype=np.int64)
    score_valid = np.asarray(score, dtype=np.float64)[valid_idx]
    chosen: List[np.ndarray] = []
    for start, end in zip(starts, ends):
        n = int(end - start)
        if n <= 0:
            continue
        k = max(1, int(np.ceil(float(q) * n)))
        local = score_valid[start:end]
        if k >= n:
            chosen.append(valid_idx[start:end])
            continue
        sel = np.argpartition(local, -k)[-k:]
        chosen.append(valid_idx[start + sel])
    if not chosen:
        return np.asarray([], dtype=np.int64)
    return np.concatenate(chosen).astype(np.int64, copy=False)


def _daily_metrics_from_u(ts: np.ndarray, u_vals: np.ndarray, fee_roundtrip: float = 0.002) -> Tuple[float, float]:
    if len(u_vals) == 0:
        return 0.0, 0.0
    # `u_vals` are already net log-returns from `_simulate_policy_batch`.
    r_trade = np.expm1(np.asarray(u_vals, dtype=np.float64))
    df = pd.DataFrame({"ts": pd.to_datetime(ts), "r": r_trade})
    r_ts = df.groupby("ts", sort=True)["r"].mean()
    r_day = r_ts.groupby(r_ts.index.floor("D")).sum()
    eq, _ = _stable_equity_and_drawdown(np.asarray(r_day.values, dtype=np.float64))
    pnl = float(eq[-1] - 1.0) if eq.size else 0.0
    neg = np.minimum(r_day.values, 0.0)
    neg_days = int(np.count_nonzero(neg < 0.0))
    total_dev = float(np.nanstd(r_day.values, ddof=1)) if len(r_day) > 1 else 0.0
    downside_dev = max(float(np.sqrt(np.mean(np.square(neg)))) if len(r_day) else 0.0, 0.25 * total_dev, 1e-3)
    if neg_days >= 3 and np.isfinite(downside_dev) and downside_dev > 0.0:
        sortino = float(np.clip((np.mean(r_day.values) / downside_dev) * np.sqrt(365.0), -25.0, 25.0))
    else:
        sortino = 0.0
    return pnl, sortino


def _daily_metrics_from_u_fast(ts: np.ndarray, u_vals: np.ndarray, fee_roundtrip: float = 0.002) -> Tuple[float, float]:
    if len(u_vals) == 0:
        return 0.0, 0.0
    ts_dt = pd.to_datetime(ts, utc=True, errors="coerce")
    ts_ns = ts_dt.view("i8")
    valid = np.isfinite(ts_ns.astype(np.float64)) & np.isfinite(u_vals)
    if not np.any(valid):
        return 0.0, 0.0

    ts_ns = ts_ns[valid]
    # Fast path for robust daily flooring
    days = ts_dt[valid].floor("D").view("i8")

    # `u_vals` are already net log-returns from `_simulate_policy_batch`.
    r_trade = np.expm1(np.asarray(u_vals, dtype=np.float64)[valid])
    uniq_ts, inv_ts = np.unique(ts_ns, return_inverse=True)
    ts_sum = np.bincount(inv_ts, weights=r_trade)
    ts_cnt = np.bincount(inv_ts)
    ts_mean = ts_sum / np.maximum(ts_cnt, 1)

    # Re-extract the days for the unique timestamps.
    uniq_days_for_ts = days[np.unique(inv_ts, return_index=True)[1]]
    uniq_days, inv_days = np.unique(uniq_days_for_ts, return_inverse=True)
    day_sum = np.bincount(inv_days, weights=ts_mean)
    r_day = day_sum.astype(np.float64, copy=False)
    if r_day.size == 0:
        return 0.0, 0.0
    r_day = np.clip(r_day, -0.999999, None)
    eq, _ = _stable_equity_and_drawdown(r_day)
    pnl = float(eq[-1] - 1.0) if eq.size else 0.0
    neg = np.minimum(r_day, 0.0)
    neg_days = int(np.count_nonzero(neg < 0.0))
    total_dev = float(np.nanstd(r_day, ddof=1)) if r_day.size > 1 else 0.0
    downside_dev = max(float(np.sqrt(np.mean(np.square(neg)))), 0.25 * total_dev, 1e-3)
    if neg_days >= 3 and np.isfinite(downside_dev) and downside_dev > 0.0:
        sortino = float(np.clip((np.mean(r_day) / downside_dev) * np.sqrt(365.0), -25.0, 25.0))
    else:
        sortino = 0.0
    return pnl, sortino


def _selection_metrics_from_u(ts: np.ndarray, u_vals: np.ndarray, fee_roundtrip: float = 0.002) -> Tuple[float, float]:
    pnl, sortino = _daily_metrics_from_u_fast(ts, u_vals, fee_roundtrip=fee_roundtrip)
    if len(u_vals) == 0:
        return pnl, sortino
    if abs(float(pnl)) > 1e-12 or abs(float(sortino)) > 1e-12:
        return float(pnl), float(sortino)
    # Fallback when calendar aggregation collapses or produces a flat-zero score.
    u_arr = np.asarray(u_vals, dtype=np.float64)
    pnl_trade = float(np.expm1(np.clip(np.sum(u_arr), -20.0, 20.0)))
    downside = np.minimum(np.expm1(u_arr), 0.0)
    if np.count_nonzero(downside < 0.0) >= 3:
        downside_dev = max(float(np.sqrt(np.mean(np.square(downside)))), 1e-3)
        sortino_trade = float(np.clip((np.mean(np.expm1(u_arr)) / downside_dev) * np.sqrt(365.0), -25.0, 25.0))
    else:
        sortino_trade = 0.0
    return pnl_trade, sortino_trade


def _financial_summary_from_u(u_vals: np.ndarray) -> Dict[str, float]:
    if len(u_vals) == 0:
        return {
            "mean_trade_pnl": 0.0,
            "median_trade_pnl": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
        }
    r = np.expm1(np.asarray(u_vals, dtype=np.float64))
    pos = r[r > 0.0]
    neg = r[r < 0.0]
    avg_win = float(np.mean(pos)) if len(pos) else 0.0
    avg_loss = float(np.mean(neg)) if len(neg) else 0.0
    win_rate = float(np.mean(r > 0.0))
    profit_factor = float(np.sum(pos) / abs(np.sum(neg))) if len(neg) and abs(np.sum(neg)) > 1e-12 else (float("inf") if len(pos) else 0.0)
    expectancy = float(np.mean(r))
    return {
        "mean_trade_pnl": expectancy,
        "median_trade_pnl": float(np.median(r)),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": float(profit_factor),
        "expectancy": expectancy,
    }


def _format_policy_progress(row: Dict[str, Any]) -> str:
    return (
        f"j_stable={float(row['j_stable']):.6f} "
        f"(j_mean={float(row['j_mean']):.6f}, j_std={float(row['j_std']):.6f}; "
        f"q05 pnl={float(row['pnl_q05']):.6f} sortino={float(row['sortino_q05']):.3f} j={float(row['j_q05']):.6f}; "
        f"q10 pnl={float(row['pnl_q10']):.6f} sortino={float(row['sortino_q10']):.3f} j={float(row['j_q10']):.6f}; "
        f"q30 pnl={float(row['pnl_q30']):.6f} sortino={float(row['sortino_q30']):.3f} j={float(row['j_q30']):.6f}) "
        f"params=[sl_atr={float(row['sl_atr_mult']):.2f}, tp_sl={float(row['tp_sl_ratio']):.2f}, "
        f"hold={int(row['max_hold_bars'])}, trail_atr={float(row['trail_activate_atr']):.2f}, "
        f"giveback={float(row['giveback_pct']):.2f}, early_deadline={int(row['early_exit_deadline_bars'])}, "
        f"early_mfe={float(row['early_exit_mfe_atr']):.2f}] "
        f"rates=[tp={float(row['pct_TP']):.2%}, sl={float(row['pct_SL']):.2%}, trail={float(row['pct_TRAIL']):.2%}, "
        f"early={float(row['pct_EARLY']):.2%}, timeout={float(row['pct_TIMEOUT']):.2%}] "
        f"proxy_pnl_q30={float(row.get('proxy_pnl_q30', 0.0)):.6f} "
        f"hard_reject={bool(row['hard_reject'])} "
        f"prefilter={str(row.get('prefilter_reason', '')) or 'none'}"
    )


def _build_proxy_anchor(X: np.ndarray) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.asarray([], dtype=np.float64)
    n, p = arr.shape
    ranks = np.empty((n, p), dtype=np.float64)
    for j in range(p):
        col = arr[:, j]
        order = np.argsort(col, kind="mergesort")
        rank = np.empty(n, dtype=np.float64)
        rank[order] = (np.arange(n, dtype=np.float64) + 0.5) / max(n, 1)
        ranks[:, j] = rank
    return np.nanmean(ranks, axis=1)


def _policy_prefilter_reasons(
    *,
    policy: LabelPolicy,
    pct_timeout: float,
    pct_sl: float,
    pct_trail: float,
    pct_early: float,
    frac_near0: float,
    u_mean: float,
    proxy_pnl_q30: float,
    dynamic_u_floor: float,
    dynamic_pnl_floor: float,
) -> List[str]:
    reasons: List[str] = []
    if pct_timeout > 0.60:
        reasons.append("timeout>60%")
    if pct_sl > 0.80:
        reasons.append("sl>80%")
    if frac_near0 > 0.60:
        reasons.append("near0>60%")
    if pct_early < 0.01 or pct_early > 0.80:
        reasons.append("early_outside_[1%,80%]")
    if pct_trail < 0.01 or pct_trail > 0.80:
        reasons.append("trail_outside_[1%,80%]")
    if float(policy.giveback_pct) > 0.5 * float(policy.trail_activate_atr):
        reasons.append("giveback>0.5x_trail_activate")
    if float(policy.trail_activate_atr) > 5.0 * float(policy.giveback_pct):
        reasons.append("trail_activate>5x_giveback")
    if float(policy.trail_activate_atr) < 2.0 * float(policy.giveback_pct):
        reasons.append("trail_activate<2x_giveback")
    if (u_mean <= dynamic_u_floor) and (proxy_pnl_q30 <= dynamic_pnl_floor):
        reasons.append("cheap_proxy_below_dynamic_floor")
    return reasons


def _stack_object_paths(values: Sequence[object], max_bars: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(values)
    out = np.full((n, max_bars), np.nan, dtype=np.float64)
    lengths = np.zeros(n, dtype=np.int64)
    for i, val in enumerate(values):
        arr = np.asarray(val, dtype=np.float64)
        use = min(len(arr), max_bars)
        if use <= 0:
            continue
        out[i, :use] = arr[:use]
        lengths[i] = use
    return out, lengths


def _simulate_policy_batch(
    entry_prices: np.ndarray,
    atr_entries: np.ndarray,
    is_longs: np.ndarray,
    opens_2d: np.ndarray,
    highs_2d: np.ndarray,
    lows_2d: np.ndarray,
    closes_2d: np.ndarray,
    path_lengths: np.ndarray,
    policy: LabelPolicy,
    cost_pct: float,
) -> Tuple[np.ndarray, np.ndarray]:
    from extreme_price_movements.ridge_position_sizer import simulate_trade_exit_batch

    n = len(entry_prices)
    if n == 0:
        return np.asarray([], dtype=np.float32), np.zeros(5, dtype=np.int64)

    atr_safe = np.maximum(np.asarray(atr_entries, dtype=np.float64), 1e-9)
    is_long_bool = np.asarray(is_longs, dtype=bool)
    max_bars = int(policy.max_hold_bars)
    active_lens = np.minimum(np.asarray(path_lengths, dtype=np.int64), max_bars)

    tp_dist = float(policy.tp_sl_ratio) * float(policy.sl_atr_mult) * atr_safe
    sl_dist = float(policy.sl_atr_mult) * atr_safe
    tp_prices = np.where(is_long_bool, entry_prices + tp_dist, entry_prices - tp_dist)
    sl_prices = np.where(is_long_bool, entry_prices - sl_dist, entry_prices + sl_dist)

    deadline = int(policy.early_exit_deadline_bars)
    early_mask = np.zeros(n, dtype=bool)
    if deadline > 0:
        eligible = active_lens >= deadline
        if np.any(eligible):
            peak = np.nanmax(highs_2d[eligible, :deadline], axis=1)
            trough = np.nanmin(lows_2d[eligible, :deadline], axis=1)
            mfe = np.where(
                is_long_bool[eligible],
                (peak - entry_prices[eligible]) / atr_safe[eligible],
                (entry_prices[eligible] - trough) / atr_safe[eligible],
            )
            early_mask[eligible] = mfe < float(policy.early_exit_mfe_atr)

    trailing_pcts = np.full(n, float(policy.giveback_pct) if float(policy.trail_activate_atr) <= 1.0 else 0.0, dtype=np.float64)
    exit_prices, exit_bars, exit_reasons = simulate_trade_exit_batch(
        highs_2d,
        lows_2d,
        opens_2d,
        closes_2d,
        entry_prices.astype(np.float64, copy=False),
        is_long_bool.astype(np.int64, copy=False),
        tp_prices.astype(np.float64, copy=False),
        sl_prices.astype(np.float64, copy=False),
        trailing_pcts,
        max_bars,
    )

    if np.any(early_mask):
        early_idx = deadline - 1
        early_rows = np.flatnonzero(early_mask)
        exit_prices[early_rows] = closes_2d[early_rows, early_idx]
        exit_bars[early_rows] = early_idx
        exit_reasons[early_rows] = 4

    log_ret = np.where(
        is_long_bool,
        np.log(np.maximum(exit_prices, 1e-12) / np.maximum(entry_prices, 1e-12)),
        np.log(np.maximum(entry_prices, 1e-12) / np.maximum(exit_prices, 1e-12)),
    )
    u = np.nan_to_num(log_ret - cost_pct, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    counts = np.bincount(exit_reasons.astype(np.int64, copy=False), minlength=5)
    return u, counts


def _simulate_simple_tp_sl_batch(
    entry_prices: np.ndarray,
    is_longs: np.ndarray,
    opens_2d: np.ndarray,
    highs_2d: np.ndarray,
    lows_2d: np.ndarray,
    closes_2d: np.ndarray,
    path_lengths: np.ndarray,
    tp_pct: float,
    sl_pct: float,
    max_hold_bars: int,
    cost_pct: float,
) -> Tuple[np.ndarray, np.ndarray]:
    from extreme_price_movements.ridge_position_sizer import simulate_trade_exit_batch

    n = len(entry_prices)
    if n == 0:
        return np.asarray([], dtype=np.float32), np.zeros(5, dtype=np.int64)
    is_long_bool = np.asarray(is_longs, dtype=bool)
    max_bars = int(max_hold_bars)
    active_lens = np.minimum(np.asarray(path_lengths, dtype=np.int64), max_bars)
    tp_prices = np.where(is_long_bool, entry_prices * (1.0 + float(tp_pct)), entry_prices * (1.0 - float(tp_pct)))
    sl_prices = np.where(is_long_bool, entry_prices * (1.0 - float(sl_pct)), entry_prices * (1.0 + float(sl_pct)))
    trailing_pcts = np.zeros(n, dtype=np.float64)
    exit_prices, exit_bars, exit_reasons = simulate_trade_exit_batch(
        highs_2d,
        lows_2d,
        opens_2d,
        closes_2d,
        entry_prices.astype(np.float64, copy=False),
        is_long_bool.astype(np.int64, copy=False),
        tp_prices.astype(np.float64, copy=False),
        sl_prices.astype(np.float64, copy=False),
        trailing_pcts,
        max_bars,
    )
    timeout_mask = active_lens < max_bars
    if np.any(timeout_mask):
        idx = np.flatnonzero(timeout_mask)
        last_bar = np.maximum(active_lens[idx] - 1, 0)
        exit_prices[idx] = closes_2d[idx, last_bar]
        exit_bars[idx] = last_bar
        exit_reasons[idx] = 3
    log_ret = np.where(
        is_long_bool,
        np.log(np.maximum(exit_prices, 1e-12) / np.maximum(entry_prices, 1e-12)),
        np.log(np.maximum(entry_prices, 1e-12) / np.maximum(exit_prices, 1e-12)),
    )
    u = np.nan_to_num(log_ret - cost_pct, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    counts = np.bincount(exit_reasons.astype(np.int64, copy=False), minlength=5)
    return u, counts


def _ridge_probe_oof(
    X: np.ndarray,
    y: np.ndarray,
    ts: np.ndarray,
    groups: Optional[np.ndarray],
    alpha: float = 1.0,
    winsor_q_low: float = 0.01,
    winsor_q_high: float = 0.99,
) -> np.ndarray:
    oof = np.full(len(y), np.nan, dtype=np.float32)
    ts_norm = None
    if ts is not None:
        ts_dt = pd.to_datetime(ts, utc=True, errors="coerce")
        ts_i8 = np.asarray(getattr(ts_dt, "view", lambda *_: ts_dt)("i8"), dtype=np.int64)
        if np.any(np.isfinite(ts_i8.astype(np.float64))):
            ts_norm = ts_i8 // 10**9
    if ts_norm is not None:
        pkf = PurgedKFold(n_splits=3, purge=43200, embargo=43200, min_train_size=max(50, len(y) // 6), times=ts_norm)
    else:
        pkf = PurgedKFold(n_splits=3, purge=12, embargo=12, min_train_size=max(50, len(y) // 6))
    split_args: List[Any] = [X]
    if groups is not None:
        split_args.append(groups)
    n_done = 0
    for tr, va in pkf.split(*split_args):
        if len(tr) < 50 or len(va) == 0:
            continue
        ytr = y[tr]
        lo = float(np.quantile(ytr, winsor_q_low))
        hi = float(np.quantile(ytr, winsor_q_high))
        ytr = np.clip(ytr, lo, hi)
        scl = StandardScaler()
        xtr = scl.fit_transform(X[tr])
        xva = scl.transform(X[va])
        mdl = Ridge(alpha=float(alpha), fit_intercept=True, solver="auto")
        mdl.fit(xtr, ytr)
        oof[va] = mdl.predict(xva).astype(np.float32)
        n_done += 1
    if n_done == 0 or int(np.isfinite(oof).sum()) < max(20, len(y) // 10):
        fold_size = max(1, len(y) // 3)
        for fold in range(3):
            va_start = fold * fold_size
            va_end = len(y) if fold == 2 else min(len(y), va_start + fold_size)
            train_end = max(0, va_start - 12)
            if train_end < 50 or va_end <= va_start:
                continue
            tr = np.arange(0, train_end, dtype=np.int64)
            va = np.arange(va_start, va_end, dtype=np.int64)
            ytr = y[tr]
            lo = float(np.quantile(ytr, winsor_q_low))
            hi = float(np.quantile(ytr, winsor_q_high))
            ytr = np.clip(ytr, lo, hi)
            scl = StandardScaler()
            xtr = scl.fit_transform(X[tr])
            xva = scl.transform(X[va])
            mdl = Ridge(alpha=float(alpha), fit_intercept=True, solver="auto")
            mdl.fit(xtr, ytr)
            oof[va] = mdl.predict(xva).astype(np.float32)
    finite = np.isfinite(oof)
    if np.any(finite):
        fill = float(np.nanmin(oof[finite]) - max(np.nanstd(oof[finite]), 1e-3))
        oof = np.where(finite, oof, fill).astype(np.float32, copy=False)
    else:
        oof = _build_proxy_anchor(X).astype(np.float32, copy=False)
    return oof


def optimize_label_policy(
    trade_outcomes: pd.DataFrame,
    oof_preds: pd.DataFrame,
    timestamps: Optional[np.ndarray],
    symbols: Optional[np.ndarray],
    groups: Optional[np.ndarray],
    cfg: Dict[str, Any],
    simulate_trade_exit_fn,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Select stable labeling policy via coarse grid + fast Ridge probe objective."""
    req = {"entry_price", "is_long", "future_highs", "future_lows", "future_closes", "atr_12_15m"}
    if not req.issubset(set(trade_outcomes.columns)):
        return trade_outcomes, {"status": "skipped", "reason": "missing_path_columns"}

    full_trade_outcomes = trade_outcomes.reset_index(drop=True).copy()
    full_oof_preds = oof_preds.reset_index(drop=True).copy() if isinstance(oof_preds, pd.DataFrame) else oof_preds
    full_timestamps = np.asarray(timestamps).copy() if timestamps is not None else None
    full_symbols = np.asarray(symbols).copy() if symbols is not None else None
    full_groups = np.asarray(groups).copy() if groups is not None else None

    search_trade_outcomes = full_trade_outcomes
    search_oof_preds = full_oof_preds
    search_timestamps = full_timestamps
    search_symbols = full_symbols
    search_groups = full_groups

    max_policy_samples = int(cfg.get("label_policy_max_samples", 10_000))
    orig_n = len(full_trade_outcomes)
    if orig_n > max_policy_samples > 0:
        rng = np.random.default_rng(int(cfg.get("label_policy_sample_seed", 42)))
        keep_idx = np.sort(rng.choice(orig_n, size=max_policy_samples, replace=False))
        search_trade_outcomes = full_trade_outcomes.iloc[keep_idx].reset_index(drop=True)
        if full_timestamps is not None:
            search_timestamps = full_timestamps[keep_idx]
        if full_symbols is not None:
            search_symbols = full_symbols[keep_idx]
        if full_groups is not None:
            search_groups = full_groups[keep_idx]
        if isinstance(full_oof_preds, pd.DataFrame):
            search_oof_preds = full_oof_preds.iloc[keep_idx].reset_index(drop=True)
        tprint(
            f"Policy optimization subsampled to {max_policy_samples}/{orig_n} rows "
            f"(seed={int(cfg.get('label_policy_sample_seed', 42))})"
        )

    ts = np.asarray(
        search_timestamps if search_timestamps is not None else search_trade_outcomes.get("timestamp", np.arange(len(search_trade_outcomes)))
    )
    sy = np.asarray(
        search_symbols if search_symbols is not None else search_trade_outcomes.get("symbol", np.array(["ALL"] * len(search_trade_outcomes)))
    )
    grouped_topq = _prepare_grouped_topq_inputs(ts, sy)

    # Keep feature extraction aligned with RidgePositionSizer.fit semantics.
    if 'model_name' in search_oof_preds.columns and 'pred' in search_oof_preds.columns:
        pred_wide = search_oof_preds.pivot(columns='model_name', values='pred')
        X_cols = list(pred_wide.columns)
        X = np.nan_to_num(pred_wide.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    else:
        X_cols = list(search_oof_preds.columns)
        X = np.nan_to_num(search_oof_preds[X_cols].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    grid = list(itertools.product(
        [0.8, 1.2, 1.6, 2.0],
        [1.5, 2.0, 2.5],
        [24],
        [0.8, 1.2],
        [0.35, 0.50],
        [8, 12],
        [0.3, 0.5],
    ))

    rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    entry_prices = np.asarray(search_trade_outcomes["entry_price"].to_numpy(), dtype=np.float64)
    is_longs = np.asarray(search_trade_outcomes["is_long"].to_numpy(), dtype=bool)
    atr_entries = np.asarray(search_trade_outcomes["atr_12_15m"].to_numpy(), dtype=np.float64)
    closes_2d, close_lens = _stack_object_paths(search_trade_outcomes["future_closes"].values, max_bars=24)
    highs_2d, high_lens = _stack_object_paths(search_trade_outcomes["future_highs"].values, max_bars=24)
    lows_2d, low_lens = _stack_object_paths(search_trade_outcomes["future_lows"].values, max_bars=24)
    if "future_opens" in search_trade_outcomes.columns:
        opens_2d, open_lens = _stack_object_paths(search_trade_outcomes["future_opens"].values, max_bars=24)
    else:
        opens_2d = closes_2d.copy()
        open_lens = close_lens.copy()
    path_lengths = np.minimum(np.minimum(open_lens, high_lens), np.minimum(low_lens, close_lens))
    fee_rt = float(cfg.get("policy_fee_rt", 0.002))
    default_tbm_grid = [
        {"name": "tbm_50_25_h8", "tp_pct": 0.0050, "sl_pct": 0.0025, "max_hold_bars": 8},
        {"name": "tbm_50_25_h16", "tp_pct": 0.0050, "sl_pct": 0.0025, "max_hold_bars": 16},
        {"name": "tbm_50_25_h24", "tp_pct": 0.0050, "sl_pct": 0.0025, "max_hold_bars": 24},
        {"name": "tbm_100_50_h8", "tp_pct": 0.0100, "sl_pct": 0.0050, "max_hold_bars": 8},
        {"name": "tbm_100_50_h16", "tp_pct": 0.0100, "sl_pct": 0.0050, "max_hold_bars": 16},
        {"name": "tbm_100_50_h24", "tp_pct": 0.0100, "sl_pct": 0.0050, "max_hold_bars": 24},
    ]
    tbm_grid = cfg.get("label_policy_ab_tbm_grid", default_tbm_grid)
    proxy_anchor = _build_proxy_anchor(X)
    proxy_top30_idx = _topq_select_indices_fast(proxy_anchor, 0.30, grouped_topq)
    past_u_means: List[float] = []
    past_proxy_pnl_q30: List[float] = []

    tprint(f"Starting label policy optimization over grid of {len(grid)} candidate policies...")
    for idx_val, vals in enumerate(grid):
        pol = LabelPolicy(*vals)
        u, reason_code_counts = _simulate_policy_batch(
            entry_prices=entry_prices,
            atr_entries=atr_entries,
            is_longs=is_longs,
            opens_2d=opens_2d,
            highs_2d=highs_2d,
            lows_2d=lows_2d,
            closes_2d=closes_2d,
            path_lengths=path_lengths,
            policy=pol,
            cost_pct=fee_rt,
        )
        reason_counts = {
            "tp": int(reason_code_counts[0]),
            "sl": int(reason_code_counts[1]),
            "trail": int(reason_code_counts[2]),
            "timeout": int(reason_code_counts[3]),
            "early": int(reason_code_counts[4]),
        }

        n = len(u)
        pct_timeout = reason_counts["timeout"] / max(n, 1)
        pct_sl = reason_counts["sl"] / max(n, 1)
        pct_trail = reason_counts["trail"] / max(n, 1)
        pct_early = reason_counts["early"] / max(n, 1)
        frac_near0 = float(np.mean(np.abs(u) < 1e-4))
        u_mean = float(np.mean(u))
        proxy_pnl_q30, _ = _selection_metrics_from_u(ts[proxy_top30_idx], u[proxy_top30_idx], fee_roundtrip=fee_rt)
        if idx_val >= 20 and past_u_means and past_proxy_pnl_q30:
            dynamic_u_floor = 0.8 * float(np.median(np.asarray(past_u_means, dtype=np.float64)))
            dynamic_pnl_floor = 0.8 * float(np.median(np.asarray(past_proxy_pnl_q30, dtype=np.float64)))
        else:
            dynamic_u_floor = 0.0
            dynamic_pnl_floor = 0.0
        prefilter_reasons = _policy_prefilter_reasons(
            policy=pol,
            pct_timeout=pct_timeout,
            pct_sl=pct_sl,
            pct_trail=pct_trail,
            pct_early=pct_early,
            frac_near0=frac_near0,
            u_mean=u_mean,
            proxy_pnl_q30=float(proxy_pnl_q30),
            dynamic_u_floor=dynamic_u_floor,
            dynamic_pnl_floor=dynamic_pnl_floor,
        )
        prefilter_reject = bool(prefilter_reasons)
        past_u_means.append(u_mean)
        past_proxy_pnl_q30.append(float(proxy_pnl_q30))

        q_stats: Dict[float, Dict[str, float]] = {
            0.05: {"pnl": 0.0, "sortino": 0.0, "j": -1e9},
            0.10: {"pnl": 0.0, "sortino": 0.0, "j": -1e9},
            0.30: {"pnl": 0.0, "sortino": 0.0, "j": -1e9},
        }
        if not prefilter_reject:
            score_oof = _ridge_probe_oof(
                X=X,
                y=u.astype(np.float32),
                ts=ts,
                groups=search_groups,
                alpha=float(cfg.get("label_policy_probe_alpha", 1.0)),
                winsor_q_low=float(cfg.get("sizer_winsor_q_low", 0.01)),
                winsor_q_high=float(cfg.get("sizer_winsor_q_high", 0.99)),
            )

            for q in (0.05, 0.10, 0.30):
                idx = _topq_select_indices_fast(score_oof, q, grouped_topq)
                pnl, sortino = _selection_metrics_from_u(ts[idx], u[idx], fee_roundtrip=fee_rt)
                beta = float(cfg.get("label_policy_sortino_beta", 0.01))
                q_stats[q] = {
                    "pnl": float(pnl),
                    "sortino": float(sortino),
                    "j": float(pnl + beta * sortino),
                }

        fold_js = np.asarray([q_stats[0.05]["j"], q_stats[0.10]["j"], q_stats[0.30]["j"]], dtype=float)
        j_mean = float(np.mean(fold_js))
        j_std = float(np.std(fold_js))
        j_stable = float(j_mean - float(cfg.get("label_policy_lambda", 0.5)) * j_std)
        hard_reject = bool(
            prefilter_reject
            or (pct_timeout > float(cfg.get("label_policy_max_timeout", 0.80)))
            or (pct_sl > 0.80)
            or (frac_near0 > 0.70)
        )

        row = {
            **asdict(pol),
            "u_mean": u_mean,
            "u_std": float(np.std(u)),
            "frac_pos": float(np.mean(u > 0.0)),
            "frac_near0": frac_near0,
            "pct_TP": reason_counts["tp"] / max(n, 1),
            "pct_SL": pct_sl,
            "pct_TRAIL": pct_trail,
            "pct_EARLY": pct_early,
            "pct_TIMEOUT": pct_timeout,
            "proxy_pnl_q30": float(proxy_pnl_q30),
            "dynamic_u_floor": dynamic_u_floor,
            "dynamic_pnl_floor": dynamic_pnl_floor,
            "prefilter_reject": prefilter_reject,
            "prefilter_reason": "|".join(prefilter_reasons),
            "pnl_q05": q_stats[0.05]["pnl"],
            "pnl_q10": q_stats[0.10]["pnl"],
            "pnl_q30": q_stats[0.30]["pnl"],
            "sortino_q05": q_stats[0.05]["sortino"],
            "sortino_q10": q_stats[0.10]["sortino"],
            "sortino_q30": q_stats[0.30]["sortino"],
            "j_q05": q_stats[0.05]["j"],
            "j_q10": q_stats[0.10]["j"],
            "j_q30": q_stats[0.30]["j"],
            "j_mean": j_mean,
            "j_std": j_std,
            "j_stable": j_stable,
            "hard_reject": hard_reject,
            "u_policy": u,
        }
        rows.append(row)

        if not hard_reject and (best is None or row["j_stable"] > best["j_stable"]):
            best = row

        if (idx_val + 1) % 10 == 0 or idx_val == len(grid) - 1:
            if best is not None:
                tprint(
                    f"  Optimized {idx_val + 1}/{len(grid)} policies. "
                    f"Best so far: {_format_policy_progress(best)}"
                )
            else:
                tprint(
                    f"  Optimized {idx_val + 1}/{len(grid)} policies. "
                    f"No non-rejected policy yet. Current: {_format_policy_progress(row)}"
                )

    if best is None:
        best = max(rows, key=lambda r: r["j_stable"])
        tprint(f"Warning: All policies were hard rejected. Selecting best rejected policy with j_stable={best['j_stable']:.6f}")

    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != "u_policy"} for r in rows]).sort_values("j_stable", ascending=False)

    eps = float(cfg.get("label_policy_plateau_eps", 0.02))
    plateau = results_df[results_df["j_stable"] >= float(results_df["j_stable"].max()) - eps]
    chosen = plateau.sort_values(["j_std", "j_stable"], ascending=[True, False]).iloc[0]
    chosen_key = (
        float(chosen["sl_atr_mult"]),
        float(chosen["tp_sl_ratio"]),
        int(chosen["max_hold_bars"]),
        float(chosen["trail_activate_atr"]),
        float(chosen["giveback_pct"]),
        int(chosen["early_exit_deadline_bars"]),
        float(chosen["early_exit_mfe_atr"]),
    )
    chosen_row = next(r for r in rows if (
        float(r["sl_atr_mult"]), float(r["tp_sl_ratio"]), int(r["max_hold_bars"]),
        float(r["trail_activate_atr"]), float(r["giveback_pct"]),
        int(r["early_exit_deadline_bars"]), float(r["early_exit_mfe_atr"]),
    ) == chosen_key)

    full_entry_prices = np.asarray(full_trade_outcomes["entry_price"].to_numpy(), dtype=np.float64)
    full_is_longs = np.asarray(full_trade_outcomes["is_long"].to_numpy(), dtype=bool)
    full_atr_entries = np.asarray(full_trade_outcomes["atr_12_15m"].to_numpy(), dtype=np.float64)
    full_closes_2d, full_close_lens = _stack_object_paths(full_trade_outcomes["future_closes"].values, max_bars=24)
    full_highs_2d, full_high_lens = _stack_object_paths(full_trade_outcomes["future_highs"].values, max_bars=24)
    full_lows_2d, full_low_lens = _stack_object_paths(full_trade_outcomes["future_lows"].values, max_bars=24)
    if "future_opens" in full_trade_outcomes.columns:
        full_opens_2d, full_open_lens = _stack_object_paths(full_trade_outcomes["future_opens"].values, max_bars=24)
    else:
        full_opens_2d = full_closes_2d.copy()
        full_open_lens = full_close_lens.copy()
    full_path_lengths = np.minimum(
        np.minimum(full_open_lens, full_high_lens),
        np.minimum(full_low_lens, full_close_lens),
    )
    full_u, full_reason_code_counts = _simulate_policy_batch(
        entry_prices=full_entry_prices,
        atr_entries=full_atr_entries,
        is_longs=full_is_longs,
        opens_2d=full_opens_2d,
        highs_2d=full_highs_2d,
        lows_2d=full_lows_2d,
        closes_2d=full_closes_2d,
        path_lengths=full_path_lengths,
        policy=LabelPolicy(*chosen_key),
        cost_pct=fee_rt,
    )
    chosen_u_search = np.asarray(chosen_row["u_policy"], dtype=np.float32)
    chosen_score_oof = _ridge_probe_oof(
        X=X,
        y=chosen_u_search,
        ts=ts,
        groups=search_groups,
        alpha=float(cfg.get("label_policy_probe_alpha", 1.0)),
        winsor_q_low=float(cfg.get("sizer_winsor_q_low", 0.01)),
        winsor_q_high=float(cfg.get("sizer_winsor_q_high", 0.99)),
    )
    chosen_probe_q_stats: Dict[float, Dict[str, float]] = {}
    for q in (0.05, 0.10, 0.30):
        idx = _topq_select_indices_fast(chosen_score_oof, q, grouped_topq)
        pnl, sortino = _selection_metrics_from_u(ts[idx], chosen_u_search[idx], fee_roundtrip=fee_rt)
        beta = float(cfg.get("label_policy_sortino_beta", 0.01))
        chosen_probe_q_stats[q] = {
            "pnl": float(pnl),
            "sortino": float(sortino),
            "j": float(pnl + beta * sortino),
        }
    chosen_probe_fold_js = np.asarray([chosen_probe_q_stats[0.05]["j"], chosen_probe_q_stats[0.10]["j"], chosen_probe_q_stats[0.30]["j"]], dtype=float)
    chosen_probe_j_mean = float(np.mean(chosen_probe_fold_js))
    chosen_probe_j_std = float(np.std(chosen_probe_fold_js))
    chosen_probe_j_stable = float(chosen_probe_j_mean - float(cfg.get("label_policy_lambda", 0.5)) * chosen_probe_j_std)

    def _evaluate_tbm_baseline(name: str, tp_pct: float, sl_pct: float, max_hold_bars: int) -> Dict[str, Any]:
        search_u, reason_counts_search = _simulate_simple_tp_sl_batch(
            entry_prices=entry_prices,
            is_longs=is_longs,
            opens_2d=opens_2d,
            highs_2d=highs_2d,
            lows_2d=lows_2d,
            closes_2d=closes_2d,
            path_lengths=path_lengths,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            max_hold_bars=max_hold_bars,
            cost_pct=fee_rt,
        )
        score_oof = _ridge_probe_oof(
            X=X,
            y=search_u.astype(np.float32),
            ts=ts,
            groups=search_groups,
            alpha=float(cfg.get("label_policy_probe_alpha", 1.0)),
            winsor_q_low=float(cfg.get("sizer_winsor_q_low", 0.01)),
            winsor_q_high=float(cfg.get("sizer_winsor_q_high", 0.99)),
        )
        q_stats: Dict[float, Dict[str, float]] = {}
        for q in (0.05, 0.10, 0.30):
            idx = _topq_select_indices_fast(score_oof, q, grouped_topq)
            pnl, sortino = _selection_metrics_from_u(ts[idx], search_u[idx], fee_roundtrip=fee_rt)
            beta = float(cfg.get("label_policy_sortino_beta", 0.01))
            q_stats[q] = {"pnl": float(pnl), "sortino": float(sortino), "j": float(pnl + beta * sortino)}
        fold_js = np.asarray([q_stats[0.05]["j"], q_stats[0.10]["j"], q_stats[0.30]["j"]], dtype=float)
        full_u_local, reason_counts_full = _simulate_simple_tp_sl_batch(
            entry_prices=full_entry_prices,
            is_longs=full_is_longs,
            opens_2d=full_opens_2d,
            highs_2d=full_highs_2d,
            lows_2d=full_lows_2d,
            closes_2d=full_closes_2d,
            path_lengths=full_path_lengths,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            max_hold_bars=max_hold_bars,
            cost_pct=fee_rt,
        )
        return {
            "name": str(name),
            "tp_pct": float(tp_pct),
            "sl_pct": float(sl_pct),
            "max_hold_bars": int(max_hold_bars),
            "j_stable": float(np.mean(fold_js) - float(cfg.get("label_policy_lambda", 0.5)) * np.std(fold_js)),
            "j_mean": float(np.mean(fold_js)),
            "j_std": float(np.std(fold_js)),
            "q05_j": float(q_stats[0.05]["j"]),
            "q10_j": float(q_stats[0.10]["j"]),
            "q30_j": float(q_stats[0.30]["j"]),
            "q05_pnl": float(q_stats[0.05]["pnl"]),
            "q10_pnl": float(q_stats[0.10]["pnl"]),
            "q30_pnl": float(q_stats[0.30]["pnl"]),
            "financials_search": _financial_summary_from_u(search_u),
            "financials_full": _financial_summary_from_u(full_u_local),
            "pct_timeout_search": float(reason_counts_search[3] / max(len(search_u), 1)),
            "pct_sl_search": float(reason_counts_search[1] / max(len(search_u), 1)),
            "pct_timeout_full": float(reason_counts_full[3] / max(len(full_u_local), 1)),
            "pct_sl_full": float(reason_counts_full[1] / max(len(full_u_local), 1)),
            "full_u": np.asarray(full_u_local, dtype=np.float32),
        }

    tbm_ab_rows: List[Dict[str, Any]] = []
    for row_cfg in tbm_grid:
        try:
            tbm_ab_rows.append(
                _evaluate_tbm_baseline(
                    name=row_cfg.get("name", f"tbm_{row_cfg.get('tp_pct', 0)}_{row_cfg.get('sl_pct', 0)}_{row_cfg.get('max_hold_bars', 24)}"),
                    tp_pct=float(row_cfg.get("tp_pct", 0.005)),
                    sl_pct=float(row_cfg.get("sl_pct", 0.0025)),
                    max_hold_bars=int(row_cfg.get("max_hold_bars", 24)),
                )
            )
        except Exception:
            continue
    best_tbm = max(tbm_ab_rows, key=lambda r: r["j_stable"]) if tbm_ab_rows else None

    out = full_trade_outcomes.copy()
    out["u_policy"] = np.asarray(full_u, dtype=np.float32)
    out["u_policy_net"] = out["u_policy"]
    if best_tbm is not None:
        out["u_simple_tp_sl"] = np.asarray(best_tbm["full_u"], dtype=np.float32)
        out["u_simple_tp_sl_net"] = out["u_simple_tp_sl"]
    for tbm_row in tbm_ab_rows:
        tbm_name = str(tbm_row.get("name", "")).strip()
        if not tbm_name:
            continue
        out[f"u_{tbm_name}"] = np.asarray(tbm_row["full_u"], dtype=np.float32)
    # Persist selected policy params onto rows so downstream Ridge models can consume
    # the exact same policy configuration (no hidden defaults divergence).
    out["label_policy_sl_atr_mult"] = float(chosen_row["sl_atr_mult"])
    out["label_policy_tp_sl_ratio"] = float(chosen_row["tp_sl_ratio"])
    out["label_policy_max_hold_bars"] = int(chosen_row["max_hold_bars"])
    out["label_policy_trail_activate_atr"] = float(chosen_row["trail_activate_atr"])
    out["label_policy_giveback_pct"] = float(chosen_row["giveback_pct"])
    out["label_policy_early_exit_deadline_bars"] = int(chosen_row["early_exit_deadline_bars"])
    out["label_policy_early_exit_mfe_atr"] = float(chosen_row["early_exit_mfe_atr"])

    optimized_fin_search = _financial_summary_from_u(chosen_u_search)
    optimized_fin_full = _financial_summary_from_u(full_u)

    reports_dir = resolve_reports_dir(cfg.get("reports_root") if cfg else None)
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_path = reports_dir / "policy_grid_results.csv"
    sel_path = reports_dir / "selected_policy.json"
    results_df.to_csv(results_path, index=False)
    with open(sel_path, "w") as f:
        json.dump({
            "selected_policy": {k: v for k, v in chosen_row.items() if k != "u_policy"},
            "acceptance": {
                "search_pct_TIMEOUT": float(chosen_row["pct_TIMEOUT"]),
                "search_pct_SL": float(chosen_row["pct_SL"]),
                "search_frac_near0": float(chosen_row["frac_near0"]),
                "search_hard_reject": bool(chosen_row["hard_reject"]),
                "full_pct_TIMEOUT": float(full_reason_code_counts[3] / max(len(full_u), 1)),
                "full_pct_SL": float(full_reason_code_counts[1] / max(len(full_u), 1)),
                "full_frac_near0": float(np.mean(np.abs(full_u) < 1e-4)) if len(full_u) else 0.0,
            },
            "provenance": {
                "grid_size": len(grid),
                "x_cols": X_cols,
                "search_rows": int(len(search_trade_outcomes)),
                "full_rows": int(len(full_trade_outcomes)),
            },
            "ab_test": {
                "optimized_policy_target": {
                    "j_stable": chosen_probe_j_stable,
                    "j_mean": chosen_probe_j_mean,
                    "j_std": chosen_probe_j_std,
                    "q05_j": float(chosen_probe_q_stats[0.05]["j"]),
                    "q10_j": float(chosen_probe_q_stats[0.10]["j"]),
                    "q30_j": float(chosen_probe_q_stats[0.30]["j"]),
                    "financials_search": optimized_fin_search,
                    "financials_full": optimized_fin_full,
                },
                "tbm_ridge_only_candidates": [{k: v for k, v in row.items() if k != "full_u"} for row in tbm_ab_rows],
                "best_tbm_ridge_only": {k: v for k, v in best_tbm.items() if k != "full_u"} if best_tbm is not None else None,
                "winner": "optimized_policy_target" if (best_tbm is None or chosen_probe_j_stable >= best_tbm["j_stable"]) else str(best_tbm["name"]),
                "delta_j_stable": float(chosen_probe_j_stable - (best_tbm["j_stable"] if best_tbm is not None else 0.0)),
            },
        }, f, indent=2)

    tprint(
        "Label policy optimizer selected policy with "
        f"j_stable={float(chosen_row['j_stable']):.6f} using {len(X_cols)} features "
        "and target=u_policy_net"
    )
    meta = {
        "status": "ok",
        "results_path": str(results_path),
        "selected_policy_path": str(sel_path),
        "feature_columns": X_cols,
        "target_column": "u_policy_net",
        "selected": {k: v for k, v in chosen_row.items() if k != "u_policy"},
        "search_rows": int(len(search_trade_outcomes)),
        "full_rows": int(len(full_trade_outcomes)),
        "ab_test": {
            "optimized_policy_target": {
                "j_stable": chosen_probe_j_stable,
                "j_mean": chosen_probe_j_mean,
                "j_std": chosen_probe_j_std,
                "financials_search": optimized_fin_search,
                "financials_full": optimized_fin_full,
            },
            "tbm_ridge_only_candidates": [{k: v for k, v in row.items() if k != "full_u"} for row in tbm_ab_rows],
            "best_tbm_ridge_only": {k: v for k, v in best_tbm.items() if k != "full_u"} if best_tbm is not None else None,
            "winner": "optimized_policy_target" if (best_tbm is None or chosen_probe_j_stable >= best_tbm["j_stable"]) else str(best_tbm["name"]),
            "delta_j_stable": float(chosen_probe_j_stable - (best_tbm["j_stable"] if best_tbm is not None else 0.0)),
        },
    }
    return out, meta
