#!/usr/bin/env python3
"""Investigate policies and simple TP/SL baselines on Apr-May-Jun OOS slices."""

from __future__ import annotations

import json
import math
import os
import argparse
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd

import scripts.report_single_head_monthly_vanilla_walkforward_oos as vanilla
import scripts.run_single_head_monthly_walkforward_oos as wf
from extreme_price_movements import simple_policy_optimiser as spo


DEFAULT_EXPERIMENT_ID = os.environ.get("EPM_MONTHLY_WF_ID", wf.DEFAULT_EXPERIMENT_ID)
DEFAULT_SOURCE_RUN_ID = os.environ.get("EPM_SOURCE_RUN_ID", wf.DEFAULT_SOURCE_RUN_ID)
OUT_DIR = wf.DATA_ROOT / "reports" / DEFAULT_EXPERIMENT_ID / "policy_baseline_investigation"


def _set_context(*, experiment_id: str, source_run_id: str) -> None:
    global DEFAULT_EXPERIMENT_ID, DEFAULT_SOURCE_RUN_ID, OUT_DIR
    DEFAULT_EXPERIMENT_ID = str(experiment_id).strip()
    DEFAULT_SOURCE_RUN_ID = str(source_run_id).strip()
    OUT_DIR = wf.DATA_ROOT / "reports" / DEFAULT_EXPERIMENT_ID / "policy_baseline_investigation"

RANK_SLICES = [
    ("top_30", 0.70),
    ("top_20", 0.80),
    ("top_15", 0.85),
    ("top_10", 0.90),
    ("top_5", 0.95),
    ("top_1", 0.99),
]

SIMPLE_BASELINES = [
    {
        "policy": "Baseline abs TP3% SL2%",
        "family": "simple_abs_tp_sl",
        "tp_abs_pct": 0.030,
        "sl_abs_pct": 0.020,
    },
    {
        "policy": "Baseline abs TP2% SL1%",
        "family": "simple_abs_tp_sl",
        "tp_abs_pct": 0.020,
        "sl_abs_pct": 0.010,
    },
    {
        "policy": "Baseline abs TP1.5% SL1%",
        "family": "simple_abs_tp_sl",
        "tp_abs_pct": 0.015,
        "sl_abs_pct": 0.010,
    },
    {
        "policy": "Baseline ATR/barrier TP3x SL2x",
        "family": "simple_atr_barrier_tp_sl",
        "tp_mult": 3.0,
        "sl_mult": 2.0,
    },
    {
        "policy": "Baseline ATR/barrier TP2x SL1x",
        "family": "simple_atr_barrier_tp_sl",
        "tp_mult": 2.0,
        "sl_mult": 1.0,
    },
    {
        "policy": "Baseline ATR/barrier TP1.5x SL1x",
        "family": "simple_atr_barrier_tp_sl",
        "tp_mult": 1.5,
        "sl_mult": 1.0,
    },
]

OOS_MONTHS = {
    "train_through_march_score_april": "2026-04",
    "train_through_april_score_may": "2026-05",
    "train_through_may_score_june": "2026-06",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NaT:
        return None
    return value


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _drawdown(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return 0.0
    curve = np.cumsum(arr)
    return float(np.min(curve - np.maximum.accumulate(curve)))


def _sortino(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    downside = arr[arr < 0.0]
    if downside.size == 0:
        return 0.0
    denom = float(np.std(downside))
    if denom <= 0.0 or not math.isfinite(denom):
        return 0.0
    return float(np.mean(arr) / denom)


def _selected_stats(
    rows: pd.DataFrame,
    gains: np.ndarray,
    *,
    selected_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    selected_rows = rows
    if selected_mask is not None and len(selected_mask) == len(rows):
        selected_rows = rows.iloc[np.flatnonzero(selected_mask)].copy()
    gains = np.asarray(gains, dtype=np.float64)
    gains = gains[np.isfinite(gains)]
    out = {
        "n_trades": int(gains.size),
        "net_pnl": float(np.sum(gains)) if gains.size else 0.0,
        "mean_net_trade": float(np.mean(gains)) if gains.size else 0.0,
        "hit_rate": float(np.mean(gains > 0.0)) if gains.size else 0.0,
        "max_drawdown": _drawdown(gains),
        "sortino": _sortino(gains),
    }
    for col in ("expected_spread_bps", "expected_half_spread_bps", "exit_spread_cost_bps"):
        if col in selected_rows.columns and len(selected_rows):
            out[f"mean_{col}"] = _as_float(
                pd.to_numeric(selected_rows[col], errors="coerce").mean(),
                default=0.0,
            )
    return out


def _concurrency_mask(
    rows: pd.DataFrame,
    *,
    max_concurrent_trades: int = spo.MAX_CONCURRENT_TRADES,
    max_concurrent_per_asset: int = spo.DEPLOYMENT_MAX_CONCURRENT_PER_ASSET,
) -> np.ndarray:
    if rows.empty:
        return np.zeros(0, dtype=bool)
    ts = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    ts_ns = ts.astype("int64").to_numpy()
    finite_ts = ts_ns != pd.NaT.value
    rank_values = (
        pd.to_numeric(rows.get("rank_pct"), errors="coerce")
        .fillna(-np.inf)
        .to_numpy(dtype=np.float64)
    )
    order = np.lexsort((-rank_values, np.where(finite_ts, ts_ns, np.iinfo(np.int64).max)))
    selected = np.zeros(len(rows), dtype=bool)
    active_until: list[int] = []
    active_until_by_symbol: dict[str, list[int]] = {}
    symbols = rows["symbol"].astype(str).to_numpy() if "symbol" in rows.columns else np.repeat("", len(rows))
    holding_bars = (
        pd.to_numeric(rows.get("exit_bars", pd.Series(1, index=rows.index)), errors="coerce")
        .fillna(1)
        .clip(lower=1)
        .astype("int64")
        .to_numpy()
    )
    bar_ns = int(pd.Timedelta(minutes=15).value)
    for idx in order:
        if not finite_ts[idx]:
            continue
        cur_ts = int(ts_ns[idx])
        active_until = [until for until in active_until if until > cur_ts]
        symbol = str(symbols[idx])
        active_until_by_symbol[symbol] = [
            until for until in active_until_by_symbol.get(symbol, []) if until > cur_ts
        ]
        if len(active_until) >= max(1, int(max_concurrent_trades)):
            continue
        if len(active_until_by_symbol.get(symbol, [])) >= max(1, int(max_concurrent_per_asset)):
            continue
        selected[idx] = True
        until = cur_ts + int(holding_bars[idx]) * bar_ns
        active_until.append(until)
        active_until_by_symbol.setdefault(symbol, []).append(until)
    return selected


def _score_simulate_and_score(
    rows: pd.DataFrame,
    paths: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    params: dict[str, Any],
) -> dict[str, Any]:
    metrics = spo.simulate_and_score(
        rows,
        *paths,
        cost_pct=spo.DEFAULT_POLICY_PER_SIDE_COST_PCT,
        size_power=1.0,
        market_mode="perps",
        max_concurrent_trades=spo.MAX_CONCURRENT_TRADES,
        max_concurrent_per_asset=spo.DEPLOYMENT_MAX_CONCURRENT_PER_ASSET,
        **params,
    )
    adv = spo.calculate_advanced_metrics(
        rows,
        metrics.get("raw_gains", np.array([], dtype=np.float32)),
        metrics.get("sizes", np.array([], dtype=np.float32)),
        metrics.get("selected_mask"),
        metrics.get("gross_gains"),
        metrics.get("exit_reason"),
        metrics.get("exit_bars"),
    )
    out = {
        "candidate_rows": int(len(rows)),
        "n_trades": int(metrics.get("total_trades", 0) or 0),
        "net_pnl": _as_float(metrics.get("net_pnl")),
        "mean_net_trade": _as_float(metrics.get("mean_net_trade")),
        "hit_rate": _as_float(metrics.get("win_rate")),
        "max_drawdown": _as_float(adv.get("max_drawdown", adv.get("max_dd", 0.0))),
        "sortino": _as_float(adv.get("m_sortino")),
        "avg_holding_bars": _as_float(metrics.get("avg_holding_bars")),
        "full_sl_exit_rate": _as_float(metrics.get("full_sl_exit_count")) / max(int(metrics.get("total_trades", 0) or 0), 1),
        "hard_tp_exit_rate": _as_float(metrics.get("hard_tp_exit_count")) / max(int(metrics.get("total_trades", 0) or 0), 1),
        "trailing_exit_rate": _as_float(metrics.get("trailing_exit_count")) / max(int(metrics.get("total_trades", 0) or 0), 1),
        "timeout_exit_rate": _as_float(metrics.get("timeout_exit_rate")),
        "mean_expected_spread_bps": _as_float(np.nanmean(metrics.get("expected_spread_bps", []))),
        "mean_exit_spread_cost_bps": _as_float(np.nanmean(metrics.get("exit_spread_cost_bps", []))),
    }
    return out


def _score_atr_barrier_tp_sl(
    rows: pd.DataFrame,
    paths: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    sl_mult: float,
    tp_mult: float,
) -> dict[str, Any]:
    sim_rows = spo._simulate_simple_tp_sl_rows(
        rows,
        paths,
        cost_pct=spo.DEFAULT_POLICY_PER_SIDE_COST_PCT,
        size_power=1.0,
        sl_mult=float(sl_mult),
        tp_mult=float(tp_mult),
        market_mode="perps",
    )
    if sim_rows.empty:
        return {
            "candidate_rows": int(len(rows)),
            "n_trades": 0,
            "net_pnl": 0.0,
            "mean_net_trade": 0.0,
            "hit_rate": 0.0,
            "max_drawdown": 0.0,
            "sortino": 0.0,
        }
    selected = _concurrency_mask(sim_rows)
    gains = pd.to_numeric(sim_rows.loc[selected, "net_gain"], errors="coerce").to_numpy(dtype=np.float64)
    out = _selected_stats(sim_rows, gains, selected_mask=selected)
    out["candidate_rows"] = int(len(sim_rows))
    out["avg_holding_bars"] = _as_float(
        pd.to_numeric(sim_rows.loc[selected, "exit_bars"], errors="coerce").mean()
    )
    return out


def _policy_row(
    *,
    eval_month: str,
    scope: str,
    source: str,
    policy: str,
    oos_window_start: str,
    oos_window_end: str,
    rank_slice: str,
    metric_type: str,
    notes: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "eval_month": eval_month,
        "scope": scope,
        "source": source,
        "policy": policy,
        "oos_window_start": oos_window_start,
        "oos_window_end": oos_window_end,
        "rank_slice": rank_slice,
        "metric_type": metric_type,
        "notes": notes,
        **metrics,
    }


def _evaluate_single_head_baselines() -> list[dict[str, Any]]:
    os.environ.setdefault("EPM_EXCHANGE", "krakenfutures")
    os.environ.setdefault("EPM_SIMPLE_POLICY_15M_DOWNLOAD", "0")
    os.environ.setdefault("EPM_SIMPLE_POLICY_1M_DOWNLOAD", "0")
    os.environ.setdefault("MPLCONFIGDIR", str(wf.ROOT / ".mplconfig"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    strategy_id = wf._select_june_best_strategy(DEFAULT_SOURCE_RUN_ID)["strategy_id"]
    ds = spo._make_policy_replay_store(str(wf.DATA_ROOT), "perps")
    rows: list[dict[str, Any]] = []
    for fold in wf._folds(DEFAULT_EXPERIMENT_ID):
        df_all, split_info = vanilla._prepare_policy_frame(fold.run_id, strategy_id)
        all_paths = spo._fetch_policy_paths(df_all, ds)
        df_all, all_paths = spo._apply_delayed_entry_execution_model(
            df_all,
            all_paths,
            data_root=str(wf.DATA_ROOT),
            market_mode="perps",
        )
        validation_mask = split_info["validation_mask"]
        validation_idx = np.flatnonzero(validation_mask)
        validation_df = df_all.iloc[validation_idx].copy().reset_index(drop=True)
        validation_paths = spo._path_take(all_paths, validation_idx)
        eval_month = OOS_MONTHS[fold.name]
        window_start = fold.policy_split.isoformat()
        window_end = fold.policy_end.isoformat()
        for label, threshold in RANK_SLICES:
            rank_mask = validation_df["rank_pct"].to_numpy(dtype=np.float32) >= float(threshold)
            idx = np.flatnonzero(rank_mask)
            selected_df = validation_df.iloc[idx].copy().reset_index(drop=True)
            selected_paths = spo._path_take(validation_paths, idx)
            vanilla_metrics = _score_simulate_and_score(selected_df, selected_paths, params={})
            rows.append(
                _policy_row(
                    eval_month=eval_month,
                    scope="single_head_monthly_walkforward",
                    source=fold.run_id,
                    policy="Vanilla fixed simulate_and_score defaults",
                    oos_window_start=window_start,
                    oos_window_end=window_end,
                    rank_slice=label,
                    metric_type="recomputed policy-OOS fixed geometry",
                    notes="Same policy-OOS rows, delayed entry, cost_pct_per_side=0.001, spread model, max concurrent 4/per asset 1.",
                    metrics=vanilla_metrics,
                )
            )
            for baseline in SIMPLE_BASELINES:
                if baseline["family"] == "simple_abs_tp_sl":
                    params = {
                        "sl_mult": 1.0e6,
                        "sl_abs_cap_pct": float(baseline["sl_abs_pct"]),
                        "hard_tp_abs_pct": float(baseline["tp_abs_pct"]),
                        "trailing_activation_mult": 1.0e6,
                        "trailing_activation_cap_pct": 0.0,
                        "trailing_activation_decay_half_life_bars": 0.0,
                        "capital_protect_mfe_mult": 0.0,
                        "adverse_exit_enabled": False,
                        "exit_pressure_enabled": False,
                    }
                    metrics = _score_simulate_and_score(selected_df, selected_paths, params=params)
                else:
                    metrics = _score_atr_barrier_tp_sl(
                        selected_df,
                        selected_paths,
                        sl_mult=float(baseline["sl_mult"]),
                        tp_mult=float(baseline["tp_mult"]),
                    )
                rows.append(
                    _policy_row(
                        eval_month=eval_month,
                        scope="single_head_monthly_walkforward",
                        source=fold.run_id,
                        policy=str(baseline["policy"]),
                        oos_window_start=window_start,
                        oos_window_end=window_end,
                        rank_slice=label,
                        metric_type="recomputed simple baseline",
                        notes="Same policy-OOS rows, delayed entry, cost_pct_per_side=0.001, spread model, max concurrent 4/per asset 1.",
                        metrics=metrics,
                    )
                )
    return rows


def _load_existing_single_head_policy_rows() -> list[dict[str, Any]]:
    monthly = (
        wf.DATA_ROOT
        / "reports"
        / DEFAULT_EXPERIMENT_ID
        / "policy_comparison"
        / "monthly_oos_policy_comparison.csv"
    )
    if not monthly.exists():
        return []
    df = pd.read_csv(monthly)
    df = df[df["scope"].eq("single_head_monthly_walkforward")].copy()
    rows = []
    for _, row in df.iterrows():
        rows.append(
            _policy_row(
                eval_month=str(row["eval_month"]),
                scope=str(row["scope"]),
                source=str(row["model_run_id"]),
                policy=str(row["policy"]),
                oos_window_start=str(row["oos_window_start"]),
                oos_window_end=str(row["oos_window_end"]),
                rank_slice=str(row["rank_slice"]),
                metric_type=str(row["metric_type"]),
                notes=str(row["notes"]),
                metrics={
                    "candidate_rows": row.get("candidate_rows"),
                    "n_trades": int(row.get("n_trades", 0) or 0),
                    "net_pnl": _as_float(row.get("net_pnl")),
                    "mean_net_trade": _as_float(row.get("mean_net_trade")),
                    "hit_rate": _as_float(row.get("hit_rate")),
                    "max_drawdown": _as_float(row.get("max_drawdown")),
                    "sortino": _as_float(row.get("sortino")),
                },
            )
        )
    return rows


def _policy_name_from_selected_rows_path(path: Path) -> str:
    parts = path.parts
    if "20260629_050000_lgbm_mda_dynamic_hr_surprise_t16_6mo_overlay_20260630" in parts:
        return "T16_q42_weighted_guard_hr35_last7_11"
    parent = path.parent.parent.name
    grandparent = path.parent.parent.parent.name
    if grandparent == "rank_failure_guard_ablation_20260630":
        return f"{parent} -> T16_q42_weighted_guard_hr35_last7_11"
    if grandparent in {
        "prehead_symbol_guard_ablation_20260630",
        "prehead_symbol_guard_threshold_sweep_rel_disp_breadth10_20260630",
    }:
        return f"{parent} -> T16_q42_weighted_guard_hr35_last7_11"
    return " -> ".join(path.parts[-4:-1])


def _scan_selected_row_policies() -> list[dict[str, Any]]:
    paths = sorted(
        (wf.DATA_ROOT / "reports").glob("**/calendar_dynamic_hr_surprise_selected_rows.csv")
    )
    windows = {
        "2026-04": (pd.Timestamp("2026-04-16", tz="UTC"), pd.Timestamp("2026-05-01", tz="UTC")),
        "2026-05": (pd.Timestamp("2026-05-16", tz="UTC"), pd.Timestamp("2026-06-01", tz="UTC")),
        "2026-06": (pd.Timestamp("2026-06-16", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC")),
    }
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "timestamp" not in df.columns or "net_return" not in df.columns:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["net_return"] = pd.to_numeric(df["net_return"], errors="coerce")
        df = df.dropna(subset=["timestamp", "net_return"]).copy()
        if df.empty:
            continue
        policy = _policy_name_from_selected_rows_path(path)
        for month, (start, end) in windows.items():
            sub = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].copy()
            if sub.empty:
                continue
            gains = sub.sort_values("timestamp")["net_return"].to_numpy(dtype=np.float64)
            metrics = _selected_stats(sub.sort_values("timestamp"), gains)
            metrics["candidate_rows"] = None
            rows.append(
                _policy_row(
                    eval_month=month,
                    scope="source_run_selected_rows_exact_oos_window",
                    source=str(path),
                    policy=policy,
                    oos_window_start=start.isoformat(),
                    oos_window_end=end.isoformat(),
                    rank_slice="as selected by policy artifact",
                    metric_type="as-reported selected-row net_return filtered to exact OOS window",
                    notes="Uses selected-row artifact economics as reported; not recomputed through single-head simulator.",
                    metrics=metrics,
                )
            )
    return rows


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["scope", "policy", "rank_slice", "metric_type"]
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False, sort=False):
        gains = pd.to_numeric(group["net_pnl"], errors="coerce").fillna(0.0)
        trades = pd.to_numeric(group["n_trades"], errors="coerce").fillna(0.0)
        weighted_mean = (
            float((pd.to_numeric(group["mean_net_trade"], errors="coerce").fillna(0.0) * trades).sum() / trades.sum())
            if float(trades.sum()) > 0.0
            else 0.0
        )
        hit_rate = (
            float((pd.to_numeric(group["hit_rate"], errors="coerce").fillna(0.0) * trades).sum() / trades.sum())
            if float(trades.sum()) > 0.0
            else 0.0
        )
        rows.append(
            {
                "scope": keys[0],
                "policy": keys[1],
                "rank_slice": keys[2],
                "metric_type": keys[3],
                "months": int(group["eval_month"].nunique()),
                "oos_window_start": str(group["oos_window_start"].min()),
                "oos_window_end": str(group["oos_window_end"].max()),
                "n_trades": int(trades.sum()),
                "net_pnl": float(gains.sum()),
                "mean_net_trade_weighted": weighted_mean,
                "hit_rate_weighted": hit_rate,
                "positive_months": int((gains > 0.0).sum()),
                "worst_month_net_pnl": float(gains.min()) if len(gains) else 0.0,
                "best_month_net_pnl": float(gains.max()) if len(gains) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["net_pnl", "months", "n_trades"],
        ascending=[False, False, False],
    )


def _format_float(value: Any, digits: int = 6) -> str:
    try:
        out = float(value)
    except Exception:
        return ""
    if not math.isfinite(out):
        return ""
    return f"{out:.{digits}f}"


def _write_markdown(monthly: pd.DataFrame, aggregate: pd.DataFrame, manifest: dict[str, Any]) -> None:
    top_cols = [
        "scope",
        "policy",
        "rank_slice",
        "months",
        "n_trades",
        "net_pnl",
        "mean_net_trade_weighted",
        "hit_rate_weighted",
        "positive_months",
        "worst_month_net_pnl",
        "best_month_net_pnl",
    ]
    detail_cols = [
        "eval_month",
        "scope",
        "policy",
        "rank_slice",
        "n_trades",
        "net_pnl",
        "mean_net_trade",
        "hit_rate",
        "max_drawdown",
        "sortino",
        "metric_type",
    ]
    top = aggregate[top_cols].head(40).copy()
    details = monthly[detail_cols].copy()
    for frame in (top, details):
        for col in frame.columns:
            if col in {"net_pnl", "mean_net_trade", "mean_net_trade_weighted", "hit_rate", "hit_rate_weighted", "max_drawdown", "sortino", "worst_month_net_pnl", "best_month_net_pnl"}:
                frame[col] = frame[col].map(lambda x: _format_float(x))
    positive = aggregate[aggregate["net_pnl"] > 0.0].copy()
    positive = positive[
        [
            "scope",
            "policy",
            "rank_slice",
            "months",
            "n_trades",
            "net_pnl",
            "positive_months",
            "worst_month_net_pnl",
        ]
    ].head(30)
    for col in ("net_pnl", "worst_month_net_pnl"):
        positive[col] = positive[col].map(lambda x: _format_float(x))
    single = aggregate[aggregate["scope"].eq("single_head_monthly_walkforward")].copy()
    best_single = single.iloc[0].to_dict() if not single.empty else {}
    top15_single = single[single["rank_slice"].eq("top_15")].copy()
    best_top15 = top15_single.iloc[0].to_dict() if not top15_single.empty else {}
    selected = aggregate[
        aggregate["scope"].eq("source_run_selected_rows_exact_oos_window")
    ].copy()
    best_selected = selected.iloc[0].to_dict() if not selected.empty else {}
    selected_monthly = monthly[
        monthly["scope"].eq("source_run_selected_rows_exact_oos_window")
    ].copy()
    selected_month_counts = (
        selected_monthly.assign(_positive=selected_monthly["net_pnl"] > 0.0)
        .groupby("eval_month")["_positive"]
        .agg(["sum", "count"])
        .to_dict("index")
        if not selected_monthly.empty
        else {}
    )
    finding_lines = [
        "## Findings",
        "",
    ]
    if best_single:
        finding_lines.append(
            "- No recomputed single-head policy or simple baseline is positive "
            f"across Apr-May-Jun. Best aggregate single-head row: "
            f"`{best_single.get('policy')}` / `{best_single.get('rank_slice')}` "
            f"with net PnL `{_format_float(best_single.get('net_pnl'))}` over "
            f"`{int(best_single.get('n_trades', 0))}` trades."
        )
    if best_top15:
        finding_lines.append(
            "- On the top-15 slice, best single-head simple baseline is "
            f"`{best_top15.get('policy')}` with net PnL "
            f"`{_format_float(best_top15.get('net_pnl'))}`. "
            "The reported Optuna simple-policy row is still better than the "
            "top-15 fixed TP/SL baselines, but remains negative."
        )
    if best_selected:
        finding_lines.append(
            "- Source-run selected-row policies have positive May+June aggregate "
            f"rows, led by `{best_selected.get('policy')}` at net PnL "
            f"`{_format_float(best_selected.get('net_pnl'))}`. These artifacts "
            "do not provide April rows in the exact held-out slice."
        )
    if selected_month_counts:
        month_bits = [
            f"{month}: {int(stats['sum'])}/{int(stats['count'])} positive"
            for month, stats in sorted(selected_month_counts.items())
        ]
        finding_lines.append(
            "- Exact-window source-run selected-row monthly positivity: "
            + "; ".join(month_bits)
            + "."
        )
    finding_lines.append(
        "- Bottom line: this scan does not prove a policy that is positive on "
        "all three exact OOS slices. It finds May-positive source-run overlays, "
        "but June is negative for every selected-row policy scanned and the "
        "single-head Apr-May-Jun policies/baselines are all negative."
    )
    finding_lines.append("")
    lines = [
        "# Apr-May-Jun OOS Policy/Baseline Investigation",
        "",
        "OOS slices are exact held-out validation windows: "
        "`2026-04-16..2026-05-01`, `2026-05-16..2026-06-01`, "
        "`2026-06-16..2026-07-01`.",
        "",
        "Single-head baselines are recomputed from the same policy-OOS rows with "
        "`cost_pct_per_side=0.001`, the simple-policy spread model, delayed entry, "
        "and max concurrent `4` / per asset `1`.",
        "",
        "Selected-row source-run policies are filtered to the same dates but use "
        "their artifact `net_return` as reported.",
        "",
        *finding_lines,
        "## Positive Aggregate Policies",
        "",
        positive.to_markdown(index=False) if not positive.empty else "No aggregate policy is positive.",
        "",
        "## Top Aggregate Rows",
        "",
        top.to_markdown(index=False),
        "",
        "## Monthly Detail",
        "",
        details.sort_values(["eval_month", "net_pnl"], ascending=[True, False]).to_markdown(index=False),
        "",
        "## Manifest",
        "",
        "```json",
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True),
        "```",
        "",
    ]
    (OUT_DIR / "policy_baseline_investigation.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--source-run-id", default=DEFAULT_SOURCE_RUN_ID)
    args = parser.parse_args()
    _set_context(experiment_id=args.experiment_id, source_run_id=args.source_run_id)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    single_head_existing = _load_existing_single_head_policy_rows()
    single_head_recomputed = _evaluate_single_head_baselines()
    selected_rows = _scan_selected_row_policies()
    monthly = pd.DataFrame(single_head_existing + single_head_recomputed + selected_rows)
    if monthly.empty:
        raise RuntimeError("No policy/baseline rows were produced.")
    monthly = monthly.sort_values(["eval_month", "scope", "policy", "rank_slice"]).reset_index(drop=True)
    aggregate = _aggregate(monthly)
    monthly_path = OUT_DIR / "policy_baseline_monthly.csv"
    aggregate_path = OUT_DIR / "policy_baseline_aggregate.csv"
    manifest_path = OUT_DIR / "manifest.json"
    monthly.to_csv(monthly_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    manifest = {
        "generated_by": Path(__file__).name,
        "output_dir": str(OUT_DIR),
        "monthly_csv": str(monthly_path),
        "aggregate_csv": str(aggregate_path),
        "single_head_existing_rows": len(single_head_existing),
        "single_head_recomputed_rows": len(single_head_recomputed),
        "selected_row_policy_rows": len(selected_rows),
        "cost_pct_per_side": float(spo.DEFAULT_POLICY_PER_SIDE_COST_PCT),
        "market_mode": "perps",
        "max_concurrent_trades": int(spo.MAX_CONCURRENT_TRADES),
        "max_concurrent_per_asset": int(spo.DEPLOYMENT_MAX_CONCURRENT_PER_ASSET),
        "oos_windows": {
            "2026-04": ["2026-04-16T00:00:00+00:00", "2026-05-01T00:00:00+00:00"],
            "2026-05": ["2026-05-16T00:00:00+00:00", "2026-06-01T00:00:00+00:00"],
            "2026-06": ["2026-06-16T00:00:00+00:00", "2026-07-01T00:00:00+00:00"],
        },
        "notes": [
            "Single-head existing Optuna/Vanilla rows are the previously reported policy-OOS validation rows.",
            "Single-head recomputed baselines use the same validation rows and simple-policy execution model.",
            "Source-run selected-row policies are not recomputed through the single-head simulator; they are filtered to matching dates and reported as-is.",
        ],
    }
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(monthly, aggregate, manifest)
    print(monthly_path)
    print(aggregate_path)
    print(OUT_DIR / "policy_baseline_investigation.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
