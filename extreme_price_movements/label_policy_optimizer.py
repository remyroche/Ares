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


def _daily_metrics_from_u(ts: np.ndarray, u_vals: np.ndarray, fee_roundtrip: float = 0.002) -> Tuple[float, float]:
    if len(u_vals) == 0:
        return 0.0, 0.0
    r_trade = np.exp(u_vals - fee_roundtrip) - 1.0
    df = pd.DataFrame({"ts": pd.to_datetime(ts), "r": r_trade})
    r_ts = df.groupby("ts", sort=True)["r"].mean()
    r_day = r_ts.groupby(r_ts.index.floor("D")).sum()
    eq = (1.0 + r_day).cumprod()
    pnl = float(eq.iloc[-1] - 1.0) if len(eq) else 0.0
    neg = np.minimum(r_day.values, 0.0)
    sortino = float((np.mean(r_day.values) / (np.std(neg) + 1e-9)) * np.sqrt(365.0)) if len(r_day) else 0.0
    return pnl, sortino


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
    if ts is not None:
        pkf = PurgedKFold(n_splits=3, purge=43200, embargo=43200, times=ts)
    else:
        pkf = PurgedKFold(n_splits=3, purge=12, embargo=12)
    split_args: List[Any] = [X]
    if groups is not None:
        split_args.append(groups)
    for tr, va in pkf.split(*split_args):
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

    ts = np.asarray(timestamps if timestamps is not None else trade_outcomes.get("timestamp", np.arange(len(trade_outcomes))))
    sy = np.asarray(symbols if symbols is not None else trade_outcomes.get("symbol", np.array(["ALL"] * len(trade_outcomes))))

    # Keep feature extraction aligned with RidgePositionSizer.fit semantics.
    if 'model_name' in oof_preds.columns and 'pred' in oof_preds.columns:
        pred_wide = oof_preds.pivot(columns='model_name', values='pred')
        X_cols = list(pred_wide.columns)
        X = np.nan_to_num(pred_wide.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    else:
        X_cols = list(oof_preds.columns)
        X = np.nan_to_num(oof_preds[X_cols].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

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

    for vals in grid:
        pol = LabelPolicy(*vals)
        u = np.zeros(len(trade_outcomes), dtype=np.float32)
        reason_counts = {"tp": 0, "sl": 0, "trail": 0, "early": 0, "timeout": 0}
        for i, row in enumerate(trade_outcomes.itertuples(index=False)):
            entry = float(getattr(row, "entry_price"))
            is_long = bool(getattr(row, "is_long"))
            atr = float(getattr(row, "atr_12_15m"))
            highs = np.asarray(getattr(row, "future_highs"), dtype=np.float64)
            lows = np.asarray(getattr(row, "future_lows"), dtype=np.float64)
            closes = np.asarray(getattr(row, "future_closes"), dtype=np.float64)
            if len(highs) == 0 or len(lows) == 0 or len(closes) == 0:
                u[i] = 0.0
                reason_counts["timeout"] += 1
                continue
            ui, reason, _, _ = _simulate_with_policy(
                simulate_trade_exit_fn,
                entry_price=entry,
                atr_entry=max(atr, 1e-9),
                is_long=is_long,
                highs=highs,
                lows=lows,
                closes=closes,
                policy=pol,
                cost_pct=float(cfg.get("policy_fee_rt", 0.002)),
            )
            u[i] = ui
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        score_oof = _ridge_probe_oof(
            X=X,
            y=u.astype(np.float32),
            ts=ts,
            groups=groups,
            alpha=float(cfg.get("label_policy_probe_alpha", 1.0)),
            winsor_q_low=float(cfg.get("sizer_winsor_q_low", 0.01)),
            winsor_q_high=float(cfg.get("sizer_winsor_q_high", 0.99)),
        )

        q_stats: Dict[float, Dict[str, float]] = {}
        for q in (0.05, 0.10, 0.30):
            idx = _topq_select_indices(ts, sy, score_oof, q)
            pnl, sortino = _daily_metrics_from_u(ts[idx], u[idx], fee_roundtrip=float(cfg.get("policy_fee_rt", 0.002)))
            beta = float(cfg.get("label_policy_sortino_beta", 0.01))
            q_stats[q] = {
                "pnl": float(pnl),
                "sortino": float(sortino),
                "j": float(pnl + beta * sortino),
            }

        n = len(u)
        pct_timeout = reason_counts["timeout"] / max(n, 1)
        pct_sl = reason_counts["sl"] / max(n, 1)
        frac_near0 = float(np.mean(np.abs(u) < 1e-4))
        fold_js = np.asarray([q_stats[0.05]["j"], q_stats[0.10]["j"], q_stats[0.30]["j"]], dtype=float)
        j_mean = float(np.mean(fold_js))
        j_std = float(np.std(fold_js))
        j_stable = float(j_mean - float(cfg.get("label_policy_lambda", 0.5)) * j_std)
        hard_reject = bool((pct_timeout > float(cfg.get("label_policy_max_timeout", 0.80))) or (pct_sl > 0.80) or (frac_near0 > 0.70))

        row = {
            **asdict(pol),
            "u_mean": float(np.mean(u)),
            "u_std": float(np.std(u)),
            "frac_pos": float(np.mean(u > 0.0)),
            "frac_near0": frac_near0,
            "pct_TP": reason_counts["tp"] / max(n, 1),
            "pct_SL": pct_sl,
            "pct_TRAIL": reason_counts["trail"] / max(n, 1),
            "pct_EARLY": reason_counts["early"] / max(n, 1),
            "pct_TIMEOUT": pct_timeout,
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

    if best is None:
        best = max(rows, key=lambda r: r["j_stable"])

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

    out = trade_outcomes.copy()
    out["u_policy"] = np.asarray(chosen_row["u_policy"], dtype=np.float32)
    out["u_policy_net"] = out["u_policy"]
    # Persist selected policy params onto rows so downstream Ridge models can consume
    # the exact same policy configuration (no hidden defaults divergence).
    out["label_policy_sl_atr_mult"] = float(chosen_row["sl_atr_mult"])
    out["label_policy_tp_sl_ratio"] = float(chosen_row["tp_sl_ratio"])
    out["label_policy_max_hold_bars"] = int(chosen_row["max_hold_bars"])
    out["label_policy_trail_activate_atr"] = float(chosen_row["trail_activate_atr"])
    out["label_policy_giveback_pct"] = float(chosen_row["giveback_pct"])
    out["label_policy_early_exit_deadline_bars"] = int(chosen_row["early_exit_deadline_bars"])
    out["label_policy_early_exit_mfe_atr"] = float(chosen_row["early_exit_mfe_atr"])

    reports_dir = resolve_reports_dir(cfg.get("reports_root") if cfg else None)
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_path = reports_dir / "policy_grid_results.csv"
    sel_path = reports_dir / "selected_policy.json"
    results_df.to_csv(results_path, index=False)
    with open(sel_path, "w") as f:
        json.dump({
            "selected_policy": {k: v for k, v in chosen_row.items() if k != "u_policy"},
            "acceptance": {
                "pct_TIMEOUT": float(chosen_row["pct_TIMEOUT"]),
                "pct_SL": float(chosen_row["pct_SL"]),
                "frac_near0": float(chosen_row["frac_near0"]),
                "hard_reject": bool(chosen_row["hard_reject"]),
            },
            "provenance": {
                "grid_size": len(grid),
                "x_cols": X_cols,
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
    }
    return out, meta
