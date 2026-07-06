#!/usr/bin/env python3
"""Counterfactual marginal-utility ablation for portfolio sizing.

This script keeps scores, rank contracts, thresholds, and auction ordering fixed.
It compares:

N0: baseline replay with no multiplier.
N1: current global risk model benchmark (G4-style combined risk).
N2: long-format marginal utility model selecting one global multiplier.
N3: shared timestamp encoder with strategy-specific size multipliers.
N4: shared timestamp encoder with strategy-specific threshold uplifts.
N5: strategy threshold uplifts plus the global emergency cap.
N6: Optuna-tuned pressure reallocation via priority, size, and weak-row thresholds.
N7: path-dependent replay oracle over chunked multiplier schedules.
N8: constrained Optuna pressure reallocation, targeting a lower-risk middle
    ground between N6 and N7.

The N2 target is not generic risk.  For each timestamp and candidate multiplier,
it estimates the realized forward utility difference versus m=1.0 using frozen
policy replays inside the training fold.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    fit_hierarchical_ev_curves,
    normalise_candidate_table,
    replay_candidates,
)
from scripts.run_global_portfolio_period_multiplier import (  # noqa: E402
    DEFAULT_POLICY_MANIFEST,
    DEFAULT_TRAIN_BROAD,
    DEFAULT_TRAIN_DEPLOYABLE,
    _accepted_trades,
    _add_open_position_concentration_features,
    _add_portfolio_state_features,
    _add_trailing_performance,
    _apply_multiplier,
    _feature_columns,
    _fit_models,
    _forward_labels,
    _json_safe,
    _load_candidates,
    _load_policy_params,
    _map_risk_to_multiplier,
    _metrics_row as _global_metrics_row,
    _period_proxy,
    _predict_models,
    _timestamp_feature_fill_values,
    _timestamp_features,
)
from scripts.run_global_portfolio_period_multiplier_walkforward import (  # noqa: E402
    _build_folds,
    _timestamp_mask,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/portfolio_marginal_utility_ablation_20260625")
MULTIPLIERS = (0.25, 0.50, 0.75, 1.00)
THRESHOLD_UPLIFTS = {1.0: 0.0, 0.75: 0.025, 0.50: 0.05, 0.25: 0.10}
REALLOCATION_SIZE_MULTIPLIERS = (0.50, 0.75, 1.00, 1.25)
RECENT_HR_FEATURE_COLS = (
    "strategy_recent_hit_rate",
    "strategy_recent_hit_rate_long",
    "strategy_recent_hr_surprise",
    "strategy_recent_trade_count",
    "head_recent_hit_rate",
    "head_recent_hit_rate_long",
    "head_recent_hr_surprise",
    "head_recent_trade_count",
    "recent_hr_ev_adjustment",
    "recent_hr_priority_adjustment",
)
RECENT_HR_COUNT_SCALE = 8.0
PORTFOLIO_HEADS = ("long_bars", "long_dist", "short_asset", "short_bollinger")


try:
    from numba import njit as _numba_njit
except Exception:  # pragma: no cover - optional speed path
    _numba_njit = None


def _maybe_njit(fn: Callable[..., Any]) -> Callable[..., Any]:
    if _numba_njit is None:
        return fn
    return _numba_njit(cache=True)(fn)


def _schedule_for_timestamps(timestamps: pd.Series, multiplier: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, utc=True, errors="coerce"),
            "multiplier": float(multiplier),
        }
    ).dropna(subset=["timestamp"]).drop_duplicates("timestamp")


def _canonical_head_name(value: Any) -> str:
    text = str(value)
    if text.startswith("short_boll"):
        return "short_bollinger"
    for head in PORTFOLIO_HEADS:
        if text == head or text.startswith(f"{head}_"):
            return head
    return text


def _head_name(strategy_id: Any) -> str:
    text = str(strategy_id)
    if text.startswith("short_boll"):
        return "short_bollinger"
    parts = text.split("_")
    head = "_".join(parts[:2]) if len(parts) >= 2 else text
    return _canonical_head_name(head)


def _candidate_head_values(candidates: pd.DataFrame) -> np.ndarray:
    if "strategy_head" in candidates.columns:
        values = candidates["strategy_head"].map(_canonical_head_name)
    elif "head" in candidates.columns:
        values = candidates["head"].map(_canonical_head_name)
    elif "strategy_id" in candidates.columns:
        values = candidates["strategy_id"].map(_head_name)
    else:
        values = pd.Series("", index=candidates.index, dtype=object)
    return values.fillna("").astype(str).to_numpy(dtype=object)


def _head_param_keys(param_name: str, head: str) -> tuple[str, ...]:
    if head == "short_bollinger":
        return (f"{param_name}__short_bollinger", f"{param_name}__short_boll")
    return (f"{param_name}__{head}",)


def _per_head_param_values(
    params_dict: dict[str, Any],
    param_name: str,
    head_values: np.ndarray,
    *,
    default: float,
    minimum: float | None = None,
) -> np.ndarray:
    values = np.full(len(head_values), float(params_dict.get(param_name, default)), dtype=float)
    by_head = params_dict.get(f"{param_name}_by_head")
    if isinstance(by_head, dict):
        for raw_head, raw_value in by_head.items():
            head = _canonical_head_name(raw_head)
            values[head_values == head] = float(raw_value)
    for raw_head in np.unique(head_values):
        head = _canonical_head_name(raw_head)
        for key in _head_param_keys(param_name, head):
            if key in params_dict:
                values[head_values == raw_head] = float(params_dict[key])
                break
    if minimum is not None:
        values = np.maximum(values, float(minimum))
    return values


@_maybe_njit
def _recent_hit_rate_for_rows_numba(
    row_ts_ns: np.ndarray,
    event_ts_ns: np.ndarray,
    event_hits: np.ndarray,
    short_window_ns: int,
    long_window_ns: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = row_ts_ns.shape[0]
    short_hr = np.full(n, np.nan, dtype=np.float64)
    long_hr = np.full(n, np.nan, dtype=np.float64)
    short_count = np.zeros(n, dtype=np.float64)
    long_count = np.zeros(n, dtype=np.float64)
    for i in range(n):
        ts = row_ts_ns[i]
        s_left = np.searchsorted(event_ts_ns, ts - short_window_ns, side="left")
        l_left = np.searchsorted(event_ts_ns, ts - long_window_ns, side="left")
        right = np.searchsorted(event_ts_ns, ts, side="left")
        s_count = right - s_left
        l_count = right - l_left
        short_count[i] = float(s_count)
        long_count[i] = float(l_count)
        if s_count > 0:
            total = 0.0
            for j in range(s_left, right):
                total += event_hits[j]
            short_hr[i] = total / float(s_count)
        if l_count > 0:
            total = 0.0
            for j in range(l_left, right):
                total += event_hits[j]
            long_hr[i] = total / float(l_count)
    return short_hr, long_hr, short_count, long_count


def _recent_hit_rate_by_key(
    rows: pd.DataFrame,
    events: pd.DataFrame,
    key_col: str,
    *,
    short_window_hours: int,
    long_window_hours: int,
    min_count: int,
    prefix: str,
) -> pd.DataFrame:
    out = pd.DataFrame(index=rows.index)
    for col in (
        f"{prefix}_recent_hit_rate",
        f"{prefix}_recent_hit_rate_long",
        f"{prefix}_recent_hr_surprise",
        f"{prefix}_recent_trade_count",
    ):
        out[col] = 0.0
    if rows.empty or events.empty or key_col not in rows.columns or key_col not in events.columns:
        return out
    row_keys = rows[key_col].astype(str)
    event_keys = events[key_col].astype(str)
    short_ns = int(pd.Timedelta(hours=int(short_window_hours)).value)
    long_ns = int(pd.Timedelta(hours=int(long_window_hours)).value)
    row_ts = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    event_ts = pd.to_datetime(events["event_timestamp"], utc=True, errors="coerce")
    event_hits = pd.to_numeric(events["hit"], errors="coerce").fillna(0.0)
    for key in sorted(set(row_keys.dropna()) & set(event_keys.dropna())):
        row_idx = rows.index[row_keys.eq(key)].to_numpy()
        event_mask = event_keys.eq(key) & event_ts.notna()
        if not bool(event_mask.any()):
            continue
        event_order = np.argsort(event_ts.loc[event_mask].astype("int64").to_numpy(dtype=np.int64))
        event_ns = event_ts.loc[event_mask].astype("int64").to_numpy(dtype=np.int64)[event_order]
        hits = event_hits.loc[event_mask].to_numpy(dtype=np.float64)[event_order]
        short_hr, long_hr, short_count, long_count = _recent_hit_rate_for_rows_numba(
            row_ts.loc[row_idx].astype("int64").to_numpy(dtype=np.int64),
            event_ns,
            hits,
            short_ns,
            long_ns,
        )
        shrink = short_count / np.maximum(short_count + float(max(int(min_count), 1)), 1.0)
        surprise = (np.nan_to_num(short_hr, nan=0.5) - np.nan_to_num(long_hr, nan=0.5)) * shrink
        out.loc[row_idx, f"{prefix}_recent_hit_rate"] = np.nan_to_num(short_hr, nan=0.5)
        out.loc[row_idx, f"{prefix}_recent_hit_rate_long"] = np.nan_to_num(long_hr, nan=0.5)
        out.loc[row_idx, f"{prefix}_recent_hr_surprise"] = surprise
        out.loc[row_idx, f"{prefix}_recent_trade_count"] = short_count
    return out


def _add_recent_hit_rate_context(
    candidates: pd.DataFrame,
    *,
    short_window_hours: int,
    long_window_hours: int,
    min_count: int,
    rank_weight: float,
    priority_weight: float,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    out = candidates.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if "strategy_id" not in out.columns:
        out["strategy_id"] = ""
    out["strategy_head"] = out["strategy_id"].map(_head_name)
    event_ts = (
        pd.to_datetime(out["exit_timestamp"], utc=True, errors="coerce")
        if "exit_timestamp" in out.columns
        else pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    )
    events = pd.DataFrame(
        {
            "event_timestamp": event_ts.fillna(out["timestamp"]),
            "strategy_id": out["strategy_id"].astype(str),
            "strategy_head": out["strategy_head"].astype(str),
            "hit": (
                pd.to_numeric(
                    out["net_return"] if "net_return" in out.columns else pd.Series(0.0, index=out.index),
                    errors="coerce",
                ).fillna(0.0)
                > 0.0
            ).astype(float),
        },
        index=out.index,
    ).dropna(subset=["event_timestamp"])
    strategy_ctx = _recent_hit_rate_by_key(
        out,
        events,
        "strategy_id",
        short_window_hours=int(short_window_hours),
        long_window_hours=int(long_window_hours),
        min_count=int(min_count),
        prefix="strategy",
    )
    head_ctx = _recent_hit_rate_by_key(
        out,
        events,
        "strategy_head",
        short_window_hours=int(short_window_hours),
        long_window_hours=int(long_window_hours),
        min_count=int(min_count),
        prefix="head",
    )
    out = pd.concat([out, strategy_ctx, head_ctx], axis=1)
    surprise = (
        pd.to_numeric(out["strategy_recent_hr_surprise"], errors="coerce").fillna(0.0)
        + 0.5 * pd.to_numeric(out["head_recent_hr_surprise"], errors="coerce").fillna(0.0)
    )
    for col in ("normalized_rank_score", "calibrated_score"):
        if col in out.columns:
            out[col] = (
                pd.to_numeric(out[col], errors="coerce").fillna(0.0)
                + float(rank_weight) * surprise
            ).clip(0.0, 1.0)
    base_priority = (
        pd.to_numeric(out.get("portfolio_priority_adjustment"), errors="coerce").fillna(0.0)
        if "portfolio_priority_adjustment" in out.columns
        else pd.Series(0.0, index=out.index)
    )
    out["portfolio_priority_adjustment"] = base_priority + float(priority_weight) * surprise
    out["recent_hr_ev_adjustment"] = float(rank_weight) * surprise
    out["recent_hr_priority_adjustment"] = float(priority_weight) * surprise
    return normalise_candidate_table(out)


def _add_recent_hr_timestamp_features(features: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in RECENT_HR_FEATURE_COLS if col in candidates.columns]
    if not cols or features.empty:
        return features
    work = candidates[["timestamp", *cols]].copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.dropna(subset=["timestamp"])
    if work.empty:
        return features
    frames: list[pd.DataFrame] = []
    for col in cols:
        values = pd.to_numeric(work[col], errors="coerce")
        if values.notna().sum() == 0:
            continue
        tmp = work[["timestamp"]].copy()
        tmp[col] = values
        agg = tmp.groupby("timestamp")[col].agg(["mean", "std", "min", "max"])
        agg.columns = [f"{col}__{stat}" for stat in agg.columns]
        frames.append(agg)
    if not frames:
        return features
    recent = pd.concat(frames, axis=1).replace([np.inf, -np.inf], np.nan).reset_index()
    out = features.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    return out.merge(recent, on="timestamp", how="left")


@_maybe_njit
def _forward_window_stats_numba(
    ts_ns: np.ndarray,
    acc_ts_ns: np.ndarray,
    position_size: np.ndarray,
    net_pnl: np.ndarray,
    gross_abs_pnl: np.ndarray,
    cost_pnl: np.ndarray,
    net_return: np.ndarray,
    full_sl_weight: np.ndarray,
    timeout_weight: np.ndarray,
    horizon_ns: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_ts = ts_ns.shape[0]
    n_acc = acc_ts_ns.shape[0]
    future_notional = np.zeros(n_ts, dtype=np.float64)
    future_utility = np.full(n_ts, np.nan, dtype=np.float64)
    future_full_sl_rate = np.full(n_ts, np.nan, dtype=np.float64)
    future_timeout_rate = np.full(n_ts, np.nan, dtype=np.float64)
    future_cost_to_gross = np.full(n_ts, np.nan, dtype=np.float64)
    future_worst_trade_return = np.full(n_ts, np.nan, dtype=np.float64)
    left = 0
    right = 0
    for i in range(n_ts):
        ts = ts_ns[i]
        end = ts + horizon_ns
        while left < n_acc and acc_ts_ns[left] <= ts:
            left += 1
        if right < left:
            right = left
        while right < n_acc and acc_ts_ns[right] <= end:
            right += 1
        notional = 0.0
        pnl = 0.0
        gross_abs = 0.0
        cost = 0.0
        full_sl = 0.0
        timeout = 0.0
        worst = np.inf
        for j in range(left, right):
            size = position_size[j]
            notional += size
            pnl += net_pnl[j]
            gross_abs += gross_abs_pnl[j]
            cost += cost_pnl[j]
            full_sl += full_sl_weight[j]
            timeout += timeout_weight[j]
            ret = net_return[j]
            if ret < worst:
                worst = ret
        future_notional[i] = notional
        if notional > 1e-9:
            future_utility[i] = pnl / notional
            future_full_sl_rate[i] = full_sl / notional
            future_timeout_rate[i] = timeout / notional
            future_cost_to_gross[i] = cost / max(gross_abs, 1e-9)
            future_worst_trade_return[i] = worst
    return (
        future_notional,
        future_utility,
        future_full_sl_rate,
        future_timeout_rate,
        future_cost_to_gross,
        future_worst_trade_return,
    )


def _future_j_labels_fast(
    timestamps: pd.Series,
    accepted: pd.DataFrame,
    *,
    horizon_hours: int,
    lambda_cost: float,
    lambda_dd: float,
) -> pd.DataFrame:
    ts_values = (
        pd.to_datetime(timestamps, utc=True, errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
    )
    if accepted.empty:
        return pd.DataFrame({"timestamp": ts_values})
    acc = accepted.copy()
    acc["timestamp"] = pd.to_datetime(acc["timestamp"], utc=True, errors="coerce")
    acc = acc.dropna(subset=["timestamp"]).sort_values("timestamp")
    if acc.empty:
        return pd.DataFrame({"timestamp": ts_values})
    position_size = pd.to_numeric(acc["position_size"], errors="coerce").fillna(0.0)
    net_return = pd.to_numeric(acc["net_return"], errors="coerce").fillna(0.0)
    gross_return = pd.to_numeric(acc["gross_return"], errors="coerce").fillna(0.0)
    net_pnl = (
        pd.to_numeric(acc.get("net_pnl"), errors="coerce")
        if "net_pnl" in acc.columns
        else position_size * net_return
    ).fillna(0.0)
    gross_pnl = (
        pd.to_numeric(acc.get("gross_pnl"), errors="coerce")
        if "gross_pnl" in acc.columns
        else position_size * gross_return
    ).fillna(0.0)
    cost_pnl = (
        pd.to_numeric(acc.get("cost_pnl"), errors="coerce")
        if "cost_pnl" in acc.columns
        else gross_pnl - net_pnl
    ).fillna(0.0)
    reason = acc["simple_policy_exit_reason"].astype(str).str.lower()
    full_sl = reason.isin(["sl", "full_sl", "stop", "stop_loss"]).astype(float)
    timeout = reason.str.contains("timeout", regex=False).astype(float)
    ts_ns = ts_values.astype("int64").to_numpy(dtype=np.int64)
    acc_ts_ns = acc["timestamp"].astype("int64").to_numpy(dtype=np.int64)
    horizon_ns = int(pd.Timedelta(hours=int(horizon_hours)).value)
    (
        future_notional,
        future_utility,
        future_full_sl_rate,
        future_timeout_rate,
        future_cost_to_gross,
        future_worst_trade_return,
    ) = _forward_window_stats_numba(
        ts_ns,
        acc_ts_ns,
        position_size.to_numpy(dtype=np.float64),
        net_pnl.to_numpy(dtype=np.float64),
        gross_pnl.abs().to_numpy(dtype=np.float64),
        cost_pnl.to_numpy(dtype=np.float64),
        net_return.to_numpy(dtype=np.float64),
        (full_sl * position_size).to_numpy(dtype=np.float64),
        (timeout * position_size).to_numpy(dtype=np.float64),
        horizon_ns,
    )
    labels = pd.DataFrame(
        {
            "timestamp": ts_values.to_numpy(),
            "future_notional": future_notional,
            "future_utility": future_utility,
            "future_full_sl_rate": future_full_sl_rate,
            "future_timeout_rate": future_timeout_rate,
            "future_cost_to_gross": future_cost_to_gross,
            "future_worst_trade_return": future_worst_trade_return,
        }
    )
    labels["future_low_opportunity"] = (
        pd.to_numeric(labels["future_timeout_rate"], errors="coerce").fillna(0.0)
        + pd.to_numeric(labels["future_cost_to_gross"], errors="coerce")
        .clip(lower=0.0, upper=3.0)
        .fillna(0.0)
    )
    utility = pd.to_numeric(labels["future_utility"], errors="coerce")
    cost = pd.to_numeric(labels["future_cost_to_gross"], errors="coerce").clip(lower=0.0).fillna(0.0)
    worst = pd.to_numeric(labels["future_worst_trade_return"], errors="coerce").fillna(0.0)
    labels["J"] = utility - float(lambda_cost) * cost - float(lambda_dd) * (-worst).clip(lower=0.0)
    return labels


def _replay_objective(metrics: dict[str, Any], accepted: pd.DataFrame) -> float:
    net_pnl = float(metrics.get("net_pnl", 0.0) or 0.0)
    worst_24h = float(metrics.get("worst_24h_net_pnl", 0.0) or 0.0)
    max_drawdown = float(metrics.get("max_drawdown", 0.0) or 0.0)
    cost_pnl = float(metrics.get("cost_pnl", 0.0) or 0.0)
    full_sl_rate = float(metrics.get("full_sl_rate", 0.0) or 0.0)
    trade_count = int(metrics.get("trade_count", len(accepted)) or 0)
    return float(
        net_pnl
        + 0.20 * min(0.0, worst_24h)
        + 5000.0 * min(0.0, max_drawdown)
        - 0.05 * cost_pnl
        - 25.0 * full_sl_rate
        - 0.02 * max(0, trade_count)
    )


def _replay_with_multiplier(
    candidates: pd.DataFrame,
    params: Any,
    ev_curve: dict[str, Any],
    *,
    multiplier: float,
    market_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    schedule = _schedule_for_timestamps(candidates["timestamp"], multiplier)
    arm_candidates = _apply_multiplier(
        candidates,
        schedule,
        scale_entries=False,
        max_entries=int(params.max_new_entries_per_bar),
    )
    decisions, equity, metrics = replay_candidates(
        arm_candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )
    accepted = _accepted_trades(arm_candidates, decisions)
    return decisions, equity, metrics, accepted


def _forward_j_labels(
    timestamps: pd.Series,
    accepted: pd.DataFrame,
    *,
    horizon_hours: int,
    lambda_cost: float,
    lambda_dd: float,
) -> pd.DataFrame:
    out = _future_j_labels_fast(
        timestamps,
        accepted,
        horizon_hours=int(horizon_hours),
        lambda_cost=float(lambda_cost),
        lambda_dd=float(lambda_dd),
    )
    if "J" not in out.columns:
        out["J"] = np.nan
    return out


def _counterfactual_label_panel(
    candidates: pd.DataFrame,
    params: Any,
    ev_curve: dict[str, Any],
    *,
    horizon_hours: int,
    lambda_cost: float,
    lambda_dd: float,
    market_mode: str,
) -> tuple[pd.DataFrame, dict[float, dict[str, Any]], dict[float, pd.DataFrame]]:
    timestamps = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce").dropna().drop_duplicates().sort_values()
    label_frames: list[pd.DataFrame] = []
    metrics_by_m: dict[float, dict[str, Any]] = {}
    accepted_by_m: dict[float, pd.DataFrame] = {}
    base_j: pd.Series | None = None
    for multiplier in MULTIPLIERS:
        _, _, metrics, accepted = _replay_with_multiplier(
            candidates,
            params,
            ev_curve,
            multiplier=float(multiplier),
            market_mode=market_mode,
        )
        metrics_by_m[float(multiplier)] = metrics
        accepted_by_m[float(multiplier)] = accepted
        labels = _forward_j_labels(
            timestamps,
            accepted,
            horizon_hours=int(horizon_hours),
            lambda_cost=float(lambda_cost),
            lambda_dd=float(lambda_dd),
        )
        labels["multiplier"] = float(multiplier)
        label_frames.append(labels)
        if abs(float(multiplier) - 1.0) < 1e-12:
            base_j = labels.set_index("timestamp")["J"]
    panel = pd.concat(label_frames, ignore_index=True)
    if base_j is None:
        raise RuntimeError("Missing baseline multiplier labels")
    panel["delta_J"] = panel["J"] - panel["timestamp"].map(base_j)
    return panel, metrics_by_m, accepted_by_m


def _build_timestamp_features(
    candidates: pd.DataFrame,
    accepted: pd.DataFrame,
    equity: pd.DataFrame,
    *,
    feature_cols_raw: list[str],
    max_feature_cols: int,
    fill_values: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    features = _timestamp_features(
        candidates,
        feature_cols=feature_cols_raw,
        max_cols=int(max_feature_cols),
        fill_values=fill_values,
    )
    features = _add_recent_hr_timestamp_features(features, candidates)
    if fill_values is None:
        fill_values = _timestamp_feature_fill_values(features)
    features = _add_trailing_performance(features, accepted)
    features = _add_portfolio_state_features(features, equity)
    features = _add_open_position_concentration_features(features, accepted)
    features["period_proxy"] = _period_proxy(features)
    features = features.fillna(fill_values).fillna(0.0)
    return features, fill_values


def _long_format_training_frame(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    base = features.copy()
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True, errors="coerce")
    lab = labels[["timestamp", "multiplier", "delta_J"]].copy()
    lab["timestamp"] = pd.to_datetime(lab["timestamp"], utc=True, errors="coerce")
    out = lab.merge(base, on="timestamp", how="left")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["delta_J"])
    return out


def _select_model_features(
    frame: pd.DataFrame,
    *,
    target_col: str,
    excluded: set[str],
    max_cols: int,
    min_non_null_frac: float,
) -> tuple[list[str], pd.DataFrame]:
    numeric_cols = [
        col
        for col in frame.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(frame[col])
    ]
    target = pd.to_numeric(frame[target_col], errors="coerce")
    rows: list[dict[str, Any]] = []
    protected = {"multiplier", "strategy_code", "global_cap_multiplier"}
    for col in numeric_cols:
        values = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        non_null_frac = float(values.notna().mean()) if len(values) else 0.0
        std = float(values.std(skipna=True)) if values.notna().any() else 0.0
        if col not in protected and (non_null_frac < float(min_non_null_frac) or not np.isfinite(std) or std <= 1e-12):
            continue
        corr = values.corr(target)
        rows.append(
            {
                "feature": col,
                "non_null_frac": non_null_frac,
                "std": std,
                "abs_corr": abs(float(corr)) if np.isfinite(corr) else 0.0,
                "protected": col in protected,
            }
        )
    diagnostics = pd.DataFrame(rows).sort_values(
        ["protected", "abs_corr", "non_null_frac"],
        ascending=[False, False, False],
    )
    if diagnostics.empty:
        return [], diagnostics
    selected = diagnostics.head(max(1, int(max_cols)))["feature"].astype(str).tolist()
    return selected, diagnostics


def _fit_marginal_utility_model(train: pd.DataFrame, feature_cols: list[str]):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(train) < 100:
        raise RuntimeError(f"Not enough marginal utility rows: {len(train)}")
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        GradientBoostingRegressor(
            random_state=41,
            max_depth=2,
            n_estimators=160,
            learning_rate=0.035,
            subsample=0.85,
        ),
    )
    model.fit(train[feature_cols], pd.to_numeric(train["delta_J"], errors="coerce").fillna(0.0))
    return model


def _predict_multiplier_schedule(
    model: Any,
    features: pd.DataFrame,
    feature_cols: list[str],
    *,
    min_positive_edge: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for multiplier in MULTIPLIERS:
        frame = features.copy()
        frame["multiplier"] = float(multiplier)
        for col in feature_cols:
            if col not in frame.columns:
                frame[col] = 0.0
        pred = np.asarray(model.predict(frame[feature_cols]), dtype=float)
        rows.append(
            pd.DataFrame(
                {
                    "timestamp": frame["timestamp"].to_numpy(),
                    "multiplier": float(multiplier),
                    "pred_delta_J": pred,
                }
            )
        )
    pred = pd.concat(rows, ignore_index=True)
    pred = pred.sort_values(["timestamp", "pred_delta_J", "multiplier"], ascending=[True, False, False])
    best = pred.drop_duplicates("timestamp", keep="first").copy()
    best.loc[pd.to_numeric(best["pred_delta_J"], errors="coerce") < float(min_positive_edge), "multiplier"] = 1.0
    return best[["timestamp", "multiplier", "pred_delta_J"]].sort_values("timestamp")


def _oracle_schedule(labels: pd.DataFrame) -> pd.DataFrame:
    pred = labels[["timestamp", "multiplier", "delta_J"]].dropna(subset=["delta_J"]).copy()
    pred = pred.sort_values(["timestamp", "delta_J", "multiplier"], ascending=[True, False, False])
    best = pred.drop_duplicates("timestamp", keep="first").rename(columns={"delta_J": "oracle_delta_J"})
    return best[["timestamp", "multiplier", "oracle_delta_J"]].sort_values("timestamp")


def _path_dependent_oracle_schedule(
    candidates: pd.DataFrame,
    params: Any,
    ev_curve: dict[str, Any],
    *,
    market_mode: str,
    chunk_hours: int,
    beam_width: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Find an oracle multiplier schedule by replaying complete schedules.

    The previous oracle selected each timestamp from labels created by constant
    whole-fold replays. That is not path-consistent. This oracle evaluates full
    replay schedules as it searches, so open positions, cooldowns, and wallet
    use are respected. The search is chunked/beam-limited to keep replay counts
    bounded.
    """
    timestamps = (
        pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
    )
    if timestamps.empty:
        return pd.DataFrame(columns=["timestamp", "multiplier", "oracle_objective"]), {}
    chunk_hours = max(1, int(chunk_hours))
    beam_width = max(1, int(beam_width))
    first_ts = timestamps.min()
    chunk_ids = ((timestamps - first_ts) // pd.Timedelta(hours=chunk_hours)).astype(int)
    unique_chunks = list(dict.fromkeys(int(x) for x in chunk_ids))
    ts_frame = pd.DataFrame({"timestamp": timestamps, "chunk_id": chunk_ids})
    beams: list[tuple[float, dict[int, float], dict[str, Any]]] = [(float("-inf"), {}, {})]
    replay_count = 0
    for chunk_id in unique_chunks:
        expanded: list[tuple[float, dict[int, float], dict[str, Any]]] = []
        for _score, assignments, _metrics in beams:
            for multiplier in MULTIPLIERS:
                cand_assignments = dict(assignments)
                cand_assignments[int(chunk_id)] = float(multiplier)
                schedule = ts_frame.copy()
                schedule["multiplier"] = [
                    cand_assignments.get(int(cid), 1.0) for cid in schedule["chunk_id"]
                ]
                _, _, metrics, accepted = _replay_schedule(
                    candidates,
                    schedule[["timestamp", "multiplier"]],
                    params,
                    ev_curve,
                    market_mode=market_mode,
                )
                replay_count += 1
                objective = _replay_objective(metrics, accepted)
                expanded.append((objective, cand_assignments, metrics))
        expanded.sort(key=lambda item: item[0], reverse=True)
        beams = expanded[:beam_width]
    best_score, best_assignments, best_metrics = beams[0]
    schedule = ts_frame.copy()
    schedule["multiplier"] = [best_assignments.get(int(cid), 1.0) for cid in schedule["chunk_id"]]
    schedule["oracle_objective"] = float(best_score)
    return schedule[["timestamp", "multiplier", "oracle_objective"]], {
        "oracle_replay_count": int(replay_count),
        "oracle_chunk_hours": int(chunk_hours),
        "oracle_beam_width": int(beam_width),
        "oracle_objective": float(best_score),
        "oracle_metrics": best_metrics,
    }


def _replay_schedule(
    candidates: pd.DataFrame,
    schedule: pd.DataFrame,
    params: Any,
    ev_curve: dict[str, Any],
    *,
    market_mode: str,
    scale_entries: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    arm_candidates = _apply_multiplier(
        candidates,
        schedule[["timestamp", "multiplier"]],
        scale_entries=bool(scale_entries),
        max_entries=int(params.max_new_entries_per_bar),
    )
    decisions, equity, metrics = replay_candidates(
        arm_candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )
    accepted = _accepted_trades(arm_candidates, decisions)
    return decisions, equity, metrics, accepted


def _strategy_ids(candidates: pd.DataFrame) -> list[str]:
    if "strategy_id" not in candidates.columns:
        return []
    return sorted(str(x) for x in candidates["strategy_id"].dropna().astype(str).unique())


def _apply_strategy_schedule(
    candidates: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    action: str,
) -> pd.DataFrame:
    work = candidates.copy()
    if "strategy_id" not in work.columns or schedule.empty:
        return work
    keys = ["timestamp", "strategy_id"]
    sched = schedule.copy()
    sched["timestamp"] = pd.to_datetime(sched["timestamp"], utc=True, errors="coerce")
    sched["strategy_id"] = sched["strategy_id"].astype(str)
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work["strategy_id"] = work["strategy_id"].astype(str)
    work = work.merge(sched[keys + ["multiplier"]], on=keys, how="left")
    multiplier = pd.to_numeric(work.pop("multiplier"), errors="coerce").fillna(1.0).clip(lower=0.0, upper=1.0)
    if action == "size":
        base = (
            pd.to_numeric(work.get("portfolio_size_multiplier"), errors="coerce").fillna(1.0)
            if "portfolio_size_multiplier" in work.columns
            else pd.Series(1.0, index=work.index)
        )
        work["portfolio_size_multiplier"] = (base * multiplier).clip(lower=0.0, upper=1.0)
    elif action == "threshold":
        base = pd.to_numeric(work["base_strategy_threshold"], errors="coerce").fillna(1.0)
        uplift = multiplier.map(THRESHOLD_UPLIFTS).fillna(0.0)
        work["base_strategy_threshold"] = np.maximum(base, (base + uplift).clip(upper=0.999))
    else:
        raise ValueError(f"Unknown strategy action: {action}")
    return work


def _pressure_feature_by_timestamp(features: pd.DataFrame, timestamps: pd.Series) -> pd.Series:
    if features.empty:
        return pd.Series(0.0, index=pd.to_datetime(timestamps, utc=True, errors="coerce"))
    feat = features.copy()
    feat["timestamp"] = pd.to_datetime(feat["timestamp"], utc=True, errors="coerce")
    pressure_cols = [
        "portfolio_state_open_capital_pct",
        "portfolio_state_open_positions",
        "portfolio_open_max_head_share",
        "portfolio_open_max_symbol_share",
    ]
    values = pd.Series(0.0, index=feat.index, dtype=float)
    if "portfolio_state_open_capital_pct" in feat.columns:
        values = values.combine(
            pd.to_numeric(feat["portfolio_state_open_capital_pct"], errors="coerce").fillna(0.0),
            max,
        )
    if "portfolio_state_open_positions" in feat.columns:
        values = values.combine(
            pd.to_numeric(feat["portfolio_state_open_positions"], errors="coerce").fillna(0.0) / 7.0,
            max,
        )
    for col in ("portfolio_open_max_head_share", "portfolio_open_max_symbol_share"):
        if col in feat.columns:
            values = values.combine(
                pd.to_numeric(feat[col], errors="coerce").fillna(0.0),
                max,
            )
    tmp = pd.DataFrame({"timestamp": feat["timestamp"], "feature_pressure": values.clip(0.0, 2.0)})
    query = pd.DataFrame({"timestamp": pd.to_datetime(timestamps, utc=True, errors="coerce")})
    merged = query.merge(tmp.dropna(subset=["timestamp"]), on="timestamp", how="left")
    return pd.to_numeric(merged["feature_pressure"], errors="coerce").fillna(0.0)


def _apply_pressure_reallocation(
    candidates: pd.DataFrame,
    timestamp_features: pd.DataFrame,
    *,
    params_dict: dict[str, float],
    max_entries_per_bar: int,
) -> pd.DataFrame:
    """Reallocate scarce capital by changing priority and per-row sizing.

    This intentionally does not lower ``portfolio_wallet_cap_multiplier``. Under
    pressure it moves capital toward the strongest rows by auction priority and
    size, while weak rows get lower size/priority or a threshold uplift.
    """
    out = candidates.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    rank_col = "normalized_rank_score" if "normalized_rank_score" in out.columns else "rank_pct"
    if rank_col not in out.columns:
        out[rank_col] = 0.0
    if "base_strategy_threshold" not in out.columns:
        out["base_strategy_threshold"] = 0.70
    rank = pd.to_numeric(out[rank_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    threshold = pd.to_numeric(out["base_strategy_threshold"], errors="coerce").fillna(0.70).clip(0.0, 0.999)
    counts = out.groupby("timestamp", observed=True)[rank_col].transform("size").astype(float)
    rank_order = out.groupby("timestamp", observed=True)[rank_col].rank(method="first", ascending=False)
    bar_strength = 1.0 - (rank_order - 1.0) / (counts - 1.0).replace(0.0, np.nan)
    bar_strength = bar_strength.fillna(1.0).clip(0.0, 1.0)
    feature_pressure = _pressure_feature_by_timestamp(timestamp_features, out["timestamp"]).to_numpy(dtype=float)
    entry_pressure = np.clip(
        (counts.to_numpy(dtype=float) / max(float(max_entries_per_bar), 1.0))
        - float(params_dict["pressure_start"]),
        0.0,
        2.0,
    )
    rank_gap = np.maximum(rank.to_numpy(dtype=float) - threshold.to_numpy(dtype=float), 0.0)
    pressure = np.clip(
        np.maximum(entry_pressure, feature_pressure)
        * float(params_dict["pressure_scale"]),
        0.0,
        1.0,
    )
    top_cut = float(params_dict["top_cut"])
    weak_cut = float(params_dict["weak_cut"])
    top_score = np.clip((bar_strength.to_numpy(dtype=float) - top_cut) / max(1.0 - top_cut, 1e-6), 0.0, 1.0)
    weak_score = np.clip((weak_cut - bar_strength.to_numpy(dtype=float)) / max(weak_cut, 1e-6), 0.0, 1.0)
    rank_gap_score = np.clip(rank_gap / max(float(params_dict["rank_gap_scale"]), 1e-6), 0.0, 1.0)
    strategy_surprise = (
        pd.to_numeric(out.get("strategy_recent_hr_surprise"), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=float)
        if "strategy_recent_hr_surprise" in out.columns
        else np.zeros(len(out), dtype=float)
    )
    head_surprise = (
        pd.to_numeric(out.get("head_recent_hr_surprise"), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=float)
        if "head_recent_hr_surprise" in out.columns
        else np.zeros(len(out), dtype=float)
    )
    strategy_hr = (
        pd.to_numeric(out.get("strategy_recent_hit_rate"), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.5)
        .clip(0.0, 1.0)
        .to_numpy(dtype=float)
        if "strategy_recent_hit_rate" in out.columns
        else np.full(len(out), 0.5, dtype=float)
    )
    head_hr = (
        pd.to_numeric(out.get("head_recent_hit_rate"), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.5)
        .clip(0.0, 1.0)
        .to_numpy(dtype=float)
        if "head_recent_hit_rate" in out.columns
        else np.full(len(out), 0.5, dtype=float)
    )
    strategy_count = (
        pd.to_numeric(out.get("strategy_recent_trade_count"), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=float)
        if "strategy_recent_trade_count" in out.columns
        else np.zeros(len(out), dtype=float)
    )
    head_count = (
        pd.to_numeric(out.get("head_recent_trade_count"), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=float)
        if "head_recent_trade_count" in out.columns
        else np.zeros(len(out), dtype=float)
    )
    head_values = _candidate_head_values(out)
    count_scale = _per_head_param_values(
        params_dict,
        "recent_hr_count_scale",
        head_values,
        default=RECENT_HR_COUNT_SCALE,
        minimum=1.0,
    )
    strategy_shrink = strategy_count / np.maximum(strategy_count + count_scale, 1.0)
    head_shrink = head_count / np.maximum(head_count + count_scale, 1.0)
    strategy_level = (strategy_hr - 0.5) * strategy_shrink
    head_level = (head_hr - 0.5) * head_shrink
    head_weight = _per_head_param_values(params_dict, "recent_hr_head_weight", head_values, default=0.5)
    level_weight = _per_head_param_values(params_dict, "recent_hr_level_weight", head_values, default=0.0)
    hr_signal = np.clip(
        strategy_surprise
        + head_weight * head_surprise
        + level_weight * (strategy_level + head_weight * head_level),
        -0.75,
        0.75,
    )
    hr_positive = np.maximum(hr_signal, 0.0)
    hr_negative = np.maximum(-hr_signal, 0.0)
    hr_ev_boost = _per_head_param_values(params_dict, "recent_hr_ev_boost", head_values, default=0.0)
    hr_ev_penalty = _per_head_param_values(params_dict, "recent_hr_ev_penalty", head_values, default=0.0)
    hr_ev_delta = pressure * (
        hr_ev_boost * hr_positive
        - hr_ev_penalty * hr_negative
    )
    priority_delta = pressure * (
        float(params_dict["priority_boost"]) * top_score * (0.5 + 0.5 * rank_gap_score)
        - float(params_dict["priority_penalty"]) * weak_score
    ) + hr_ev_delta
    base_priority_adjustment = (
        pd.to_numeric(out.get("portfolio_priority_adjustment"), errors="coerce").fillna(0.0)
        if "portfolio_priority_adjustment" in out.columns
        else pd.Series(0.0, index=out.index)
    )
    out["portfolio_priority_adjustment"] = base_priority_adjustment + priority_delta
    base_size_multiplier = (
        pd.to_numeric(out.get("portfolio_size_multiplier"), errors="coerce").fillna(1.0)
        if "portfolio_size_multiplier" in out.columns
        else pd.Series(1.0, index=out.index)
    )
    size_multiplier = base_size_multiplier.to_numpy(dtype=float) * (
        1.0
        + pressure * float(params_dict["top_size_boost"]) * top_score
        - pressure * float(params_dict["weak_size_cut"]) * weak_score
    )
    out["portfolio_size_multiplier"] = np.clip(
        size_multiplier,
        float(params_dict["min_size_multiplier"]),
        float(params_dict["max_size_multiplier"]),
    )
    threshold_uplift = pressure * weak_score * float(params_dict["weak_threshold_uplift"])
    out["base_strategy_threshold"] = np.maximum(
        threshold,
        np.minimum(threshold + threshold_uplift, 0.999),
    )
    out["portfolio_reallocation_pressure"] = pressure
    out["portfolio_reallocation_strength"] = bar_strength.to_numpy(dtype=float)
    out["portfolio_reallocation_top_score"] = top_score
    out["portfolio_reallocation_weak_score"] = weak_score
    out["portfolio_recent_hr_head"] = head_values
    out["portfolio_recent_hr_head_weight"] = head_weight
    out["portfolio_recent_hr_level_weight"] = level_weight
    out["portfolio_recent_hr_ev_boost"] = hr_ev_boost
    out["portfolio_recent_hr_ev_penalty"] = hr_ev_penalty
    out["portfolio_recent_hr_count_scale"] = count_scale
    out["portfolio_recent_hr_ev_signal"] = hr_signal
    out["portfolio_recent_hr_ev_priority_delta"] = hr_ev_delta
    return normalise_candidate_table(out)


def _replay_pressure_reallocation(
    candidates: pd.DataFrame,
    timestamp_features: pd.DataFrame,
    params: Any,
    ev_curve: dict[str, Any],
    *,
    reallocation_params: dict[str, float],
    market_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    arm_candidates = _apply_pressure_reallocation(
        candidates,
        timestamp_features,
        params_dict=reallocation_params,
        max_entries_per_bar=int(params.max_new_entries_per_bar),
    )
    decisions, equity, metrics = replay_candidates(
        arm_candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )
    accepted = _accepted_trades(arm_candidates, decisions)
    return decisions, equity, metrics, accepted


def _suggest_recent_hr_ev_params(trial: Any, *, constrained: bool) -> dict[str, float]:
    if constrained:
        head_weight_bounds = (0.0, 1.50)
        level_weight_bounds = (0.0, 1.50)
        boost_bounds = (0.0, 1.25)
        penalty_bounds = (0.0, 2.00)
        count_scale_choices = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    else:
        head_weight_bounds = (0.0, 2.00)
        level_weight_bounds = (0.0, 2.00)
        boost_bounds = (0.0, 1.75)
        penalty_bounds = (0.0, 2.50)
        count_scale_choices = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    params: dict[str, float] = {}
    for head in PORTFOLIO_HEADS:
        suffix = f"__{head}"
        params[f"recent_hr_head_weight{suffix}"] = trial.suggest_float(
            f"recent_hr_head_weight{suffix}",
            *head_weight_bounds,
        )
        params[f"recent_hr_level_weight{suffix}"] = trial.suggest_float(
            f"recent_hr_level_weight{suffix}",
            *level_weight_bounds,
        )
        params[f"recent_hr_ev_boost{suffix}"] = trial.suggest_float(
            f"recent_hr_ev_boost{suffix}",
            *boost_bounds,
        )
        params[f"recent_hr_ev_penalty{suffix}"] = trial.suggest_float(
            f"recent_hr_ev_penalty{suffix}",
            *penalty_bounds,
        )
        params[f"recent_hr_count_scale{suffix}"] = trial.suggest_categorical(
            f"recent_hr_count_scale{suffix}",
            count_scale_choices,
        )
    return params


def _suggest_reallocation_params(trial: Any, *, constrained: bool = False) -> dict[str, float]:
    if constrained:
        return {
            "pressure_start": trial.suggest_float("pressure_start", 0.75, 1.75),
            "pressure_scale": trial.suggest_float("pressure_scale", 0.20, 1.20),
            "top_cut": trial.suggest_float("top_cut", 0.65, 0.90),
            "weak_cut": trial.suggest_float("weak_cut", 0.10, 0.35),
            "rank_gap_scale": trial.suggest_float("rank_gap_scale", 0.05, 0.25),
            "priority_boost": trial.suggest_float("priority_boost", 0.0, 0.80),
            "priority_penalty": trial.suggest_float("priority_penalty", 0.30, 1.50),
            **_suggest_recent_hr_ev_params(trial, constrained=True),
            "top_size_boost": trial.suggest_float("top_size_boost", 0.0, 0.55),
            "weak_size_cut": trial.suggest_float("weak_size_cut", 0.20, 0.95),
            "weak_threshold_uplift": trial.suggest_float("weak_threshold_uplift", 0.0, 0.06),
            "min_size_multiplier": trial.suggest_categorical("min_size_multiplier", [0.25, 0.50, 0.75]),
            "max_size_multiplier": trial.suggest_categorical("max_size_multiplier", [1.0, 1.25, 1.50]),
        }
    return {
        "pressure_start": trial.suggest_float("pressure_start", 0.45, 1.50),
        "pressure_scale": trial.suggest_float("pressure_scale", 0.30, 1.75),
        "top_cut": trial.suggest_float("top_cut", 0.55, 0.88),
        "weak_cut": trial.suggest_float("weak_cut", 0.12, 0.45),
        "rank_gap_scale": trial.suggest_float("rank_gap_scale", 0.02, 0.20),
        "priority_boost": trial.suggest_float("priority_boost", 0.0, 1.25),
        "priority_penalty": trial.suggest_float("priority_penalty", 0.0, 1.25),
        **_suggest_recent_hr_ev_params(trial, constrained=False),
        "top_size_boost": trial.suggest_float("top_size_boost", 0.0, 1.00),
        "weak_size_cut": trial.suggest_float("weak_size_cut", 0.0, 0.95),
        "weak_threshold_uplift": trial.suggest_float("weak_threshold_uplift", 0.0, 0.12),
        "min_size_multiplier": trial.suggest_categorical("min_size_multiplier", [0.0, 0.10, 0.25, 0.50]),
        "max_size_multiplier": trial.suggest_categorical("max_size_multiplier", [1.0, 1.25, 1.50, 2.0]),
    }


def _default_reallocation_params() -> dict[str, float]:
    return {
        "pressure_start": 0.75,
        "pressure_scale": 1.0,
        "top_cut": 0.70,
        "weak_cut": 0.30,
        "rank_gap_scale": 0.08,
        "priority_boost": 0.50,
        "priority_penalty": 0.50,
        "recent_hr_head_weight": 0.5,
        "recent_hr_level_weight": 0.0,
        "recent_hr_ev_boost": 0.0,
        "recent_hr_ev_penalty": 0.0,
        "recent_hr_count_scale": RECENT_HR_COUNT_SCALE,
        "top_size_boost": 0.35,
        "weak_size_cut": 0.50,
        "weak_threshold_uplift": 0.03,
        "min_size_multiplier": 0.25,
        "max_size_multiplier": 1.50,
    }


def _default_constrained_reallocation_params() -> dict[str, float]:
    return {
        "pressure_start": 1.0,
        "pressure_scale": 0.75,
        "top_cut": 0.76,
        "weak_cut": 0.24,
        "rank_gap_scale": 0.12,
        "priority_boost": 0.35,
        "priority_penalty": 0.75,
        "recent_hr_head_weight": 0.5,
        "recent_hr_level_weight": 0.0,
        "recent_hr_ev_boost": 0.0,
        "recent_hr_ev_penalty": 0.0,
        "recent_hr_count_scale": RECENT_HR_COUNT_SCALE,
        "top_size_boost": 0.25,
        "weak_size_cut": 0.55,
        "weak_threshold_uplift": 0.03,
        "min_size_multiplier": 0.50,
        "max_size_multiplier": 1.25,
    }


def _optimise_reallocation_params(
    train_candidates: pd.DataFrame,
    train_features: pd.DataFrame,
    params: Any,
    ev_curve: dict[str, Any],
    *,
    market_mode: str,
    validation_hours: int,
    n_trials: int,
    seed: int,
    constrained: bool = False,
    excess_cost_penalty: float = 0.0,
    excess_notional_penalty: float = 0.0,
) -> tuple[dict[str, float], pd.DataFrame]:
    import optuna

    ts = pd.to_datetime(train_candidates["timestamp"], utc=True, errors="coerce")
    val_start = ts.max() - pd.Timedelta(hours=int(validation_hours))
    val_candidates = train_candidates.loc[ts >= val_start].copy()
    if len(val_candidates) < 20:
        fallback = _default_constrained_reallocation_params() if constrained else _default_reallocation_params()
        return fallback, pd.DataFrame()
    val_ts = pd.to_datetime(val_candidates["timestamp"], utc=True, errors="coerce")
    feature_ts = pd.to_datetime(train_features["timestamp"], utc=True, errors="coerce")
    val_features = train_features.loc[feature_ts.isin(set(val_ts.dropna().unique()))].copy()
    n0_schedule = _schedule_for_timestamps(val_candidates["timestamp"], 1.0)
    _, _, baseline_metrics, baseline_accepted = _replay_schedule(
        val_candidates,
        n0_schedule,
        params,
        ev_curve,
        market_mode=market_mode,
    )
    baseline_objective = _replay_objective(baseline_metrics, baseline_accepted)
    baseline_cost = float(baseline_metrics.get("cost_pnl", 0.0) or 0.0)
    baseline_notional = float(baseline_metrics.get("notional_turnover", 0.0) or 0.0)
    records: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        reallocation_params = _suggest_reallocation_params(trial, constrained=bool(constrained))
        _, _, metrics, accepted = _replay_pressure_reallocation(
            val_candidates,
            val_features,
            params,
            ev_curve,
            reallocation_params=reallocation_params,
            market_mode=market_mode,
        )
        score = _replay_objective(metrics, accepted) - baseline_objective
        excess_cost = max(0.0, float(metrics.get("cost_pnl", 0.0) or 0.0) - baseline_cost)
        excess_notional = max(0.0, float(metrics.get("notional_turnover", 0.0) or 0.0) - baseline_notional)
        score -= float(excess_cost_penalty) * excess_cost
        score -= float(excess_notional_penalty) * excess_notional
        records.append(
            {
                "trial": int(trial.number),
                "objective": float(score),
                "net_pnl": float(metrics.get("net_pnl", 0.0) or 0.0),
                "baseline_net_pnl": float(baseline_metrics.get("net_pnl", 0.0) or 0.0),
                "cost_pnl": float(metrics.get("cost_pnl", 0.0) or 0.0),
                "baseline_cost_pnl": baseline_cost,
                "notional_turnover": float(metrics.get("notional_turnover", 0.0) or 0.0),
                "baseline_notional_turnover": baseline_notional,
                "constrained": bool(constrained),
                **{f"param_{k}": v for k, v in reallocation_params.items()},
            }
        )
        return float(score)

    sampler = optuna.samplers.TPESampler(seed=int(seed))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=max(1, int(n_trials)), show_progress_bar=False)
    best = dict(_default_constrained_reallocation_params() if constrained else _default_reallocation_params())
    best.update({str(k): float(v) for k, v in study.best_params.items()})
    return best, pd.DataFrame(records).sort_values("objective", ascending=False)


def _replay_strategy_schedule(
    candidates: pd.DataFrame,
    schedule: pd.DataFrame,
    params: Any,
    ev_curve: dict[str, Any],
    *,
    action: str,
    market_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    arm_candidates = _apply_strategy_schedule(candidates, schedule, action=action)
    decisions, equity, metrics = replay_candidates(
        arm_candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )
    accepted = _accepted_trades(arm_candidates, decisions)
    return decisions, equity, metrics, accepted


def _replay_strategy_threshold_with_global_cap(
    candidates: pd.DataFrame,
    strategy_schedule: pd.DataFrame,
    global_schedule: pd.DataFrame,
    params: Any,
    ev_curve: dict[str, Any],
    *,
    market_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    arm_candidates = _apply_strategy_schedule(candidates, strategy_schedule, action="threshold")
    if not global_schedule.empty:
        arm_candidates = _apply_multiplier(
            arm_candidates,
            global_schedule[["timestamp", "multiplier"]],
            scale_entries=False,
            max_entries=int(params.max_new_entries_per_bar),
        )
    decisions, equity, metrics = replay_candidates(
        arm_candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )
    accepted = _accepted_trades(arm_candidates, decisions)
    return decisions, equity, metrics, accepted


def _strategy_feature_frame(candidates: pd.DataFrame, timestamp_features: pd.DataFrame) -> pd.DataFrame:
    if "strategy_id" not in candidates.columns:
        return pd.DataFrame(columns=["timestamp", "strategy_id"])
    work = candidates.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work["strategy_id"] = work["strategy_id"].astype(str)
    rank_col = "strategy_rank_pct" if "strategy_rank_pct" in work.columns else "rank_pct"
    score_col = "calibrated_score" if "calibrated_score" in work.columns else "normalized_rank_score"
    for col in (rank_col, score_col, "base_strategy_threshold"):
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")
    grouped = work.groupby(["timestamp", "strategy_id"], observed=True)
    rows = grouped.agg(
        strategy_candidate_count=(rank_col, "size"),
        strategy_rank_mean=(rank_col, "mean"),
        strategy_rank_max=(rank_col, "max"),
        strategy_rank_q75=(rank_col, lambda s: float(np.nanquantile(s, 0.75)) if len(s) else np.nan),
        strategy_score_mean=(score_col, "mean"),
        strategy_score_max=(score_col, "max"),
        strategy_threshold_mean=("base_strategy_threshold", "mean"),
    ).reset_index()
    ts = timestamp_features.copy()
    ts["timestamp"] = pd.to_datetime(ts["timestamp"], utc=True, errors="coerce")
    out = rows.merge(ts, on="timestamp", how="left")
    codes, _ = pd.factorize(out["strategy_id"].astype(str), sort=True)
    out["strategy_code"] = codes.astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


def _constant_strategy_schedule(
    timestamps: pd.Series,
    strategy_ids: list[str],
    strategy_id: str,
    multiplier: float,
) -> pd.DataFrame:
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").dropna().drop_duplicates()
    rows = pd.MultiIndex.from_product([ts, strategy_ids], names=["timestamp", "strategy_id"]).to_frame(index=False)
    rows["multiplier"] = 1.0
    rows.loc[rows["strategy_id"].astype(str).eq(str(strategy_id)), "multiplier"] = float(multiplier)
    return rows


def _strategy_counterfactual_label_panel(
    candidates: pd.DataFrame,
    params: Any,
    ev_curve: dict[str, Any],
    *,
    action: str,
    horizon_hours: int,
    lambda_cost: float,
    lambda_dd: float,
    market_mode: str,
) -> pd.DataFrame:
    timestamps = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce").dropna().drop_duplicates().sort_values()
    strategy_ids = _strategy_ids(candidates)
    if not strategy_ids:
        return pd.DataFrame()
    base_schedule = pd.MultiIndex.from_product([timestamps, strategy_ids], names=["timestamp", "strategy_id"]).to_frame(index=False)
    base_schedule["multiplier"] = 1.0
    _, _, _, base_accepted = _replay_strategy_schedule(
        candidates,
        base_schedule,
        params,
        ev_curve,
        action=action,
        market_mode=market_mode,
    )
    base_j = _forward_j_labels(
        timestamps,
        base_accepted,
        horizon_hours=int(horizon_hours),
        lambda_cost=float(lambda_cost),
        lambda_dd=float(lambda_dd),
    ).set_index("timestamp")["J"]
    frames: list[pd.DataFrame] = []
    for strategy_id in strategy_ids:
        for multiplier in MULTIPLIERS:
            schedule = _constant_strategy_schedule(timestamps, strategy_ids, strategy_id, float(multiplier))
            _, _, _, accepted = _replay_strategy_schedule(
                candidates,
                schedule,
                params,
                ev_curve,
                action=action,
                market_mode=market_mode,
            )
            labels = _forward_j_labels(
                timestamps,
                accepted,
                horizon_hours=int(horizon_hours),
                lambda_cost=float(lambda_cost),
                lambda_dd=float(lambda_dd),
            )
            labels["strategy_id"] = str(strategy_id)
            labels["multiplier"] = float(multiplier)
            labels["delta_J"] = labels["J"] - labels["timestamp"].map(base_j)
            frames.append(labels[["timestamp", "strategy_id", "multiplier", "delta_J"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _strategy_long_training_frame(strategy_features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    if strategy_features.empty or labels.empty:
        return pd.DataFrame()
    base = strategy_features.copy()
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True, errors="coerce")
    base["strategy_id"] = base["strategy_id"].astype(str)
    lab = labels.copy()
    lab["timestamp"] = pd.to_datetime(lab["timestamp"], utc=True, errors="coerce")
    lab["strategy_id"] = lab["strategy_id"].astype(str)
    out = lab.merge(base, on=["timestamp", "strategy_id"], how="left")
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["delta_J"])


def _predict_strategy_schedule(
    model: Any,
    strategy_features: pd.DataFrame,
    feature_cols: list[str],
    *,
    min_positive_edge: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for multiplier in MULTIPLIERS:
        frame = strategy_features.copy()
        frame["multiplier"] = float(multiplier)
        for col in feature_cols:
            if col not in frame.columns:
                frame[col] = 0.0
        pred = np.asarray(model.predict(frame[feature_cols]), dtype=float)
        rows.append(
            pd.DataFrame(
                {
                    "timestamp": frame["timestamp"].to_numpy(),
                    "strategy_id": frame["strategy_id"].astype(str).to_numpy(),
                    "multiplier": float(multiplier),
                    "pred_delta_J": pred,
                }
            )
        )
    pred = pd.concat(rows, ignore_index=True)
    pred = pred.sort_values(["timestamp", "strategy_id", "pred_delta_J", "multiplier"], ascending=[True, True, False, False])
    best = pred.drop_duplicates(["timestamp", "strategy_id"], keep="first").copy()
    best.loc[pd.to_numeric(best["pred_delta_J"], errors="coerce") < float(min_positive_edge), "multiplier"] = 1.0
    return best[["timestamp", "strategy_id", "multiplier", "pred_delta_J"]].sort_values(["timestamp", "strategy_id"])


def _promotion_summary(summary: pd.DataFrame) -> pd.DataFrame:
    base = summary.loc[summary["arm"].eq("N0_baseline"), ["fold_id", "net_pnl", "cost_pnl", "max_drawdown", "worst_24h_net_pnl", "notional_turnover"]]
    base = base.rename(
        columns={
            "net_pnl": "base_net_pnl",
            "cost_pnl": "base_cost_pnl",
            "max_drawdown": "base_max_drawdown",
            "worst_24h_net_pnl": "base_worst_24h_net_pnl",
            "notional_turnover": "base_notional_turnover",
        }
    )
    work = summary.merge(base, on="fold_id", how="left")
    work["delta_net_pnl"] = work["net_pnl"] - work["base_net_pnl"]
    work["delta_cost_pnl"] = work["cost_pnl"] - work["base_cost_pnl"]
    work["delta_max_drawdown"] = work["max_drawdown"] - work["base_max_drawdown"]
    work["delta_worst_24h_net_pnl"] = work["worst_24h_net_pnl"] - work["base_worst_24h_net_pnl"]
    work["exposure_ratio"] = work["notional_turnover"] / work["base_notional_turnover"].replace(0.0, np.nan)
    rows: list[dict[str, Any]] = []
    for arm, g in work.groupby("arm", sort=True):
        rows.append(
            {
                "arm": arm,
                "folds": int(g["fold_id"].nunique()),
                "median_delta_net_pnl": float(g["delta_net_pnl"].median()),
                "q25_delta_net_pnl": float(g["delta_net_pnl"].quantile(0.25)),
                "mean_delta_net_pnl": float(g["delta_net_pnl"].mean()),
                "positive_delta_net_pnl_share": float((g["delta_net_pnl"] > 0).mean()),
                "median_delta_cost_pnl": float(g["delta_cost_pnl"].median()),
                "median_delta_max_drawdown": float(g["delta_max_drawdown"].median()),
                "median_delta_worst_24h_net_pnl": float(g["delta_worst_24h_net_pnl"].median()),
                "median_exposure_ratio": float(g["exposure_ratio"].median()),
                "median_multiplier": float(g["mean_multiplier"].median()),
            }
        )
    return pd.DataFrame(rows)


def _accepted_delta_metric_frames(accepted_summaries: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if accepted_summaries.empty:
        return {
            "weekly": pd.DataFrame(),
            "strategy": pd.DataFrame(),
            "weekly_strategy": pd.DataFrame(),
        }

    def _delta_for(keys: list[str], scope: str) -> pd.DataFrame:
        value_cols = [
            "trade_count",
            "net_pnl",
            "gross_pnl",
            "cost_pnl",
            "notional",
            "mean_net_return",
            "hit_rate",
            "full_sl_rate",
            "timeout_rate",
        ]
        source = accepted_summaries.loc[accepted_summaries["scope"].eq(scope)].copy()
        if source.empty:
            return pd.DataFrame()
        group_cols = ["fold_id", "arm"] + keys
        grouped = (
            source.groupby(group_cols, dropna=False, observed=True)[value_cols]
            .sum(numeric_only=True)
            .reset_index()
        )
        base = grouped.loc[grouped["arm"].eq("N0_baseline")].copy()
        base = base.drop(columns=["arm"]).rename(
            columns={col: f"baseline_{col}" for col in value_cols}
        )
        merged = grouped.merge(base, on=["fold_id"] + keys, how="left")
        for col in value_cols:
            merged[f"baseline_{col}"] = pd.to_numeric(
                merged.get(f"baseline_{col}"), errors="coerce"
            ).fillna(0.0)
            merged[f"delta_{col}"] = (
                pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
                - merged[f"baseline_{col}"]
            )
        return merged.sort_values(["fold_id", "arm"] + keys).reset_index(drop=True)

    return {
        "weekly": _delta_for(["week"], "weekly"),
        "strategy": _delta_for(["strategy_id"], "strategy"),
        "weekly_strategy": _delta_for(["week", "strategy_id"], "weekly_strategy"),
    }


def _accepted_summary_rows(
    arm: str,
    fold_id: int,
    accepted: pd.DataFrame,
) -> pd.DataFrame:
    if accepted.empty:
        return pd.DataFrame(
            columns=[
                "arm",
                "fold_id",
                "week",
                "strategy_id",
                "scope",
                "trade_count",
                "net_pnl",
                "gross_pnl",
                "cost_pnl",
                "notional",
                "mean_net_return",
                "hit_rate",
                "full_sl_rate",
                "timeout_rate",
            ]
        )
    work = accepted.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work["week"] = work["timestamp"].dt.to_period("W").astype(str)
    work["strategy_id"] = (
        work["strategy_id"].astype(str)
        if "strategy_id" in work.columns
        else pd.Series("UNKNOWN", index=work.index, dtype=object)
    )
    for col in ("position_size", "net_return", "gross_return"):
        work[col] = pd.to_numeric(
            work[col] if col in work.columns else pd.Series(0.0, index=work.index),
            errors="coerce",
        ).fillna(0.0)
    work["net_pnl"] = (
        pd.to_numeric(work.get("net_pnl"), errors="coerce")
        if "net_pnl" in work.columns
        else work["position_size"] * work["net_return"]
    ).fillna(0.0)
    work["gross_pnl"] = (
        pd.to_numeric(work.get("gross_pnl"), errors="coerce")
        if "gross_pnl" in work.columns
        else work["position_size"] * work["gross_return"]
    ).fillna(0.0)
    work["cost_pnl"] = (
        pd.to_numeric(work.get("cost_pnl"), errors="coerce")
        if "cost_pnl" in work.columns
        else work["gross_pnl"] - work["net_pnl"]
    ).fillna(0.0)
    reason = (
        work["simple_policy_exit_reason"].astype(str)
        if "simple_policy_exit_reason" in work.columns
        else pd.Series("", index=work.index, dtype=object)
    ).str.lower()
    work["_hit"] = (work["net_return"] > 0.0).astype(float)
    work["_full_sl"] = reason.isin(["sl", "full_sl", "stop", "stop_loss"]).astype(float)
    work["_timeout"] = reason.str.contains("timeout", regex=False).astype(float)
    rows: list[pd.DataFrame] = []
    for scope, keys in (
        ("weekly_strategy", ["week", "strategy_id"]),
        ("weekly", ["week"]),
        ("strategy", ["strategy_id"]),
    ):
        grouped = work.groupby(keys, dropna=False, observed=True)
        frame = grouped.agg(
            trade_count=("net_pnl", "size"),
            net_pnl=("net_pnl", "sum"),
            gross_pnl=("gross_pnl", "sum"),
            cost_pnl=("cost_pnl", "sum"),
            notional=("position_size", "sum"),
            mean_net_return=("net_return", "mean"),
            hit_rate=("_hit", "mean"),
            full_sl_rate=("_full_sl", "mean"),
            timeout_rate=("_timeout", "mean"),
        ).reset_index()
        if "week" not in frame.columns:
            frame["week"] = "ALL"
        if "strategy_id" not in frame.columns:
            frame["strategy_id"] = "ALL"
        frame["scope"] = scope
        rows.append(frame)
    out = pd.concat(rows, ignore_index=True)
    out.insert(0, "fold_id", int(fold_id))
    out.insert(0, "arm", str(arm))
    return out


def _metric_row_for_arm(arm: str, fold_id: int, metrics: dict[str, Any], schedule: pd.DataFrame, accepted: pd.DataFrame) -> dict[str, Any]:
    row = _global_metrics_row(arm, metrics, schedule, accepted)
    row["fold_id"] = int(fold_id)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broad-candidates", type=Path, default=DEFAULT_TRAIN_BROAD)
    parser.add_argument("--deployable-candidates", type=Path, default=DEFAULT_TRAIN_DEPLOYABLE)
    parser.add_argument("--policy-manifest", type=Path, default=DEFAULT_POLICY_MANIFEST)
    parser.add_argument("--policy-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizon-hours", type=int, default=72)
    parser.add_argument("--embargo-hours", type=int, default=96)
    parser.add_argument("--min-train-hours", type=int, default=336)
    parser.add_argument("--fold-hours", type=int, default=168)
    parser.add_argument("--max-folds", type=int, default=6)
    parser.add_argument("--eval-start", default=None)
    parser.add_argument("--eval-end", default=None)
    parser.add_argument("--train-start", default=None)
    parser.add_argument("--max-feature-cols", type=int, default=96)
    parser.add_argument("--model-max-feature-cols", type=int, default=96)
    parser.add_argument("--feature-min-non-null-frac", type=float, default=0.60)
    parser.add_argument("--lambda-cost", type=float, default=0.001)
    parser.add_argument("--lambda-dd", type=float, default=0.25)
    parser.add_argument("--min-positive-edge", type=float, default=0.0)
    parser.add_argument("--optuna-trials", type=int, default=24)
    parser.add_argument("--optuna-validation-hours", type=int, default=168)
    parser.add_argument("--middle-excess-cost-penalty", type=float, default=0.20)
    parser.add_argument("--middle-excess-notional-penalty", type=float, default=0.0025)
    parser.add_argument("--oracle-chunk-hours", type=int, default=24)
    parser.add_argument("--oracle-beam-width", type=int, default=2)
    parser.add_argument("--skip-path-oracle", action="store_true")
    parser.add_argument("--recent-hr-window-hours", type=int, default=72)
    parser.add_argument("--recent-hr-long-window-hours", type=int, default=336)
    parser.add_argument("--recent-hr-min-count", type=int, default=8)
    parser.add_argument("--recent-hr-rank-weight", type=float, default=0.08)
    parser.add_argument("--recent-hr-priority-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=104729)
    parser.add_argument("--market-mode", default="perps")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    params, policy_payload = _load_policy_params(args.policy_manifest, args.policy_variant)
    broad = _add_recent_hit_rate_context(
        _load_candidates(args.broad_candidates),
        short_window_hours=int(args.recent_hr_window_hours),
        long_window_hours=int(args.recent_hr_long_window_hours),
        min_count=int(args.recent_hr_min_count),
        rank_weight=float(args.recent_hr_rank_weight),
        priority_weight=float(args.recent_hr_priority_weight),
    )
    deployable = _add_recent_hit_rate_context(
        _load_candidates(args.deployable_candidates),
        short_window_hours=int(args.recent_hr_window_hours),
        long_window_hours=int(args.recent_hr_long_window_hours),
        min_count=int(args.recent_hr_min_count),
        rank_weight=float(args.recent_hr_rank_weight),
        priority_weight=float(args.recent_hr_priority_weight),
    )
    if args.eval_start or args.eval_end:
        if not (args.eval_start and args.eval_end):
            raise RuntimeError("--eval-start and --eval-end must be supplied together")
        eval_start_arg = pd.Timestamp(str(args.eval_start))
        eval_end_arg = pd.Timestamp(str(args.eval_end))
        if eval_start_arg.tzinfo is None:
            eval_start_arg = eval_start_arg.tz_localize("UTC")
        else:
            eval_start_arg = eval_start_arg.tz_convert("UTC")
        if eval_end_arg.tzinfo is None:
            eval_end_arg = eval_end_arg.tz_localize("UTC")
        else:
            eval_end_arg = eval_end_arg.tz_convert("UTC")
        if eval_end_arg <= eval_start_arg:
            raise RuntimeError("--eval-end must be after --eval-start")
        ts = pd.to_datetime(broad["timestamp"], utc=True, errors="coerce").dropna().drop_duplicates().sort_values()
        train_cutoff = eval_start_arg - pd.Timedelta(hours=int(args.embargo_hours))
        train_start_arg = None
        if args.train_start:
            train_start_arg = pd.Timestamp(str(args.train_start))
            if train_start_arg.tzinfo is None:
                train_start_arg = train_start_arg.tz_localize("UTC")
            else:
                train_start_arg = train_start_arg.tz_convert("UTC")
        train_ts = ts.loc[ts < train_cutoff]
        if train_start_arg is not None:
            train_ts = train_ts.loc[train_ts >= train_start_arg]
        eval_ts = ts.loc[(ts >= eval_start_arg) & (ts < eval_end_arg)]
        if len(train_ts) < 120 or len(eval_ts) < 12:
            raise RuntimeError(
                "Exact eval window does not have enough timestamps: "
                f"train={len(train_ts)} eval={len(eval_ts)}"
            )
        folds = [
            {
                "fold_id": 0,
                "train_start": train_ts.min(),
                "train_end": train_ts.max(),
                "embargo_start": train_ts.max(),
                "eval_start": eval_ts.min(),
                "eval_end": eval_ts.max(),
                "train_timestamp_count": int(len(train_ts)),
                "eval_timestamp_count": int(len(eval_ts)),
            }
        ]
    else:
        folds = _build_folds(
            broad["timestamp"],
            min_train_hours=int(args.min_train_hours),
            fold_hours=int(args.fold_hours),
            embargo_hours=int(args.embargo_hours),
            max_folds=args.max_folds,
        )
    if not folds:
        raise RuntimeError("No folds available")

    summary_rows: list[dict[str, Any]] = []
    schedule_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    oracle_frames: list[pd.DataFrame] = []
    feature_diagnostic_frames: list[pd.DataFrame] = []
    optuna_frames: list[pd.DataFrame] = []
    accepted_summary_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []

    for fold in folds:
        fold_id = int(fold["fold_id"])
        train_start = pd.Timestamp(fold["train_start"])
        train_end = pd.Timestamp(fold["train_end"])
        eval_start = pd.Timestamp(fold["eval_start"])
        eval_end = pd.Timestamp(fold["eval_end"]) + pd.Timedelta(nanoseconds=1)
        train_broad = broad.loc[
            _timestamp_mask(broad, start=train_start, end=train_end + pd.Timedelta(nanoseconds=1))
        ].copy()
        eval_candidates = broad.loc[_timestamp_mask(broad, start=eval_start, end=eval_end)].copy()
        train_deployable = deployable.loc[
            _timestamp_mask(deployable, start=train_start, end=train_end + pd.Timedelta(nanoseconds=1))
        ].copy()
        if len(train_broad) < 200 or len(train_deployable) < 50 or len(eval_candidates) < 20:
            continue

        ev_curve = fit_hierarchical_ev_curves(train_deployable)
        train_decisions, train_equity, _ = replay_candidates(
            train_broad,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            market_mode=args.market_mode,
        )
        train_accepted = _accepted_trades(train_broad, train_decisions)
        feature_cols_raw = _feature_columns(train_broad, max_cols=int(args.max_feature_cols))
        train_features, fill_values = _build_timestamp_features(
            train_broad,
            train_accepted,
            train_equity,
            feature_cols_raw=feature_cols_raw,
            max_feature_cols=int(args.max_feature_cols),
        )
        label_panel, train_metrics_by_m, _ = _counterfactual_label_panel(
            train_broad,
            params,
            ev_curve,
            horizon_hours=int(args.horizon_hours),
            lambda_cost=float(args.lambda_cost),
            lambda_dd=float(args.lambda_dd),
            market_mode=args.market_mode,
        )
        label_cutoff = train_end - pd.Timedelta(hours=int(args.horizon_hours))
        label_panel = label_panel.loc[pd.to_datetime(label_panel["timestamp"], utc=True, errors="coerce") <= label_cutoff].copy()
        train_long = _long_format_training_frame(train_features, label_panel)
        model_feature_cols = [
            col
            for col in train_long.columns
            if col not in {"timestamp", "delta_J", "J"}
            and pd.api.types.is_numeric_dtype(train_long[col])
        ]
        raw_model_feature_count = len(model_feature_cols)
        model_feature_cols, model_feature_diag = _select_model_features(
            train_long,
            target_col="delta_J",
            excluded={"timestamp", "delta_J", "J"},
            max_cols=int(args.model_max_feature_cols),
            min_non_null_frac=float(args.feature_min_non_null_frac),
        )
        if not model_feature_cols:
            raise RuntimeError("No usable global marginal-utility features after pruning")
        model_feature_diag["fold_id"] = fold_id
        model_feature_diag["model"] = "global_marginal_utility"
        model_feature_diag["raw_feature_count"] = int(raw_model_feature_count)
        feature_diagnostic_frames.append(model_feature_diag)
        mu_model = _fit_marginal_utility_model(train_long, model_feature_cols)
        train_strategy_features = _strategy_feature_frame(train_broad, train_features)
        strategy_models: dict[str, Any] = {}
        strategy_feature_cols_by_action: dict[str, list[str]] = {}
        for strategy_action in ("size", "threshold"):
            strategy_labels = _strategy_counterfactual_label_panel(
                train_broad,
                params,
                ev_curve,
                action=strategy_action,
                horizon_hours=int(args.horizon_hours),
                lambda_cost=float(args.lambda_cost),
                lambda_dd=float(args.lambda_dd),
                market_mode=args.market_mode,
            )
            if not strategy_labels.empty:
                strategy_labels = strategy_labels.loc[
                    pd.to_datetime(strategy_labels["timestamp"], utc=True, errors="coerce") <= label_cutoff
                ].copy()
            strategy_long = _strategy_long_training_frame(train_strategy_features, strategy_labels)
            strategy_feature_cols = [
                col
                for col in strategy_long.columns
                if col not in {"timestamp", "strategy_id", "delta_J", "J"}
                and pd.api.types.is_numeric_dtype(strategy_long[col])
            ]
            raw_strategy_feature_count = len(strategy_feature_cols)
            strategy_feature_cols, strategy_feature_diag = _select_model_features(
                strategy_long,
                target_col="delta_J",
                excluded={"timestamp", "strategy_id", "delta_J", "J"},
                max_cols=int(args.model_max_feature_cols),
                min_non_null_frac=float(args.feature_min_non_null_frac),
            )
            if not strategy_feature_diag.empty:
                strategy_feature_diag["fold_id"] = fold_id
                strategy_feature_diag["model"] = f"strategy_{strategy_action}_marginal_utility"
                strategy_feature_diag["raw_feature_count"] = int(raw_strategy_feature_count)
                feature_diagnostic_frames.append(strategy_feature_diag)
            if len(strategy_long) >= 100 and strategy_feature_cols:
                strategy_models[strategy_action] = _fit_marginal_utility_model(strategy_long, strategy_feature_cols)
                strategy_feature_cols_by_action[strategy_action] = strategy_feature_cols

        # Current global risk model benchmark for N1, fitted on the same training fold.
        risk_labels = _forward_labels(train_features["timestamp"], train_accepted, int(args.horizon_hours))
        risk_frame = train_features.merge(risk_labels, on="timestamp", how="left")
        risk_frame = risk_frame.loc[
            pd.to_datetime(risk_frame["timestamp"], utc=True, errors="coerce") <= label_cutoff
        ].copy()
        risk_feature_cols = [c for c in train_features.columns if c != "timestamp" and pd.api.types.is_numeric_dtype(train_features[c])]
        risk_models, risk_cutoffs, _ = _fit_models(risk_frame, risk_feature_cols)
        reallocation_params, optuna_trials = _optimise_reallocation_params(
            train_broad,
            train_features,
            params,
            ev_curve,
            market_mode=args.market_mode,
            validation_hours=int(args.optuna_validation_hours),
            n_trials=int(args.optuna_trials),
            seed=int(args.seed) + fold_id,
        )
        if not optuna_trials.empty:
            optuna_trials["fold_id"] = fold_id
            optuna_trials["arm"] = "N6_optuna_pressure_reallocation"
            optuna_frames.append(optuna_trials)
        middle_reallocation_params, middle_optuna_trials = _optimise_reallocation_params(
            train_broad,
            train_features,
            params,
            ev_curve,
            market_mode=args.market_mode,
            validation_hours=int(args.optuna_validation_hours),
            n_trials=int(args.optuna_trials),
            seed=int(args.seed) + fold_id + 1009,
            constrained=True,
            excess_cost_penalty=float(args.middle_excess_cost_penalty),
            excess_notional_penalty=float(args.middle_excess_notional_penalty),
        )
        if not middle_optuna_trials.empty:
            middle_optuna_trials["fold_id"] = fold_id
            middle_optuna_trials["arm"] = "N8_constrained_pressure_reallocation"
            optuna_frames.append(middle_optuna_trials)

        # Eval features use baseline replay through the fold end for causal current-state context.
        history_candidates = broad.loc[_timestamp_mask(broad, end=eval_end)].copy()
        history_decisions, history_equity, _ = replay_candidates(
            history_candidates,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            market_mode=args.market_mode,
        )
        history_accepted = _accepted_trades(history_candidates, history_decisions)
        eval_features, _ = _build_timestamp_features(
            eval_candidates,
            history_accepted,
            history_equity,
            feature_cols_raw=feature_cols_raw,
            max_feature_cols=int(args.max_feature_cols),
            fill_values=fill_values,
        )
        for col in model_feature_cols:
            if col not in eval_features.columns:
                eval_features[col] = 0.0
        for col in risk_feature_cols:
            if col not in eval_features.columns:
                eval_features[col] = 0.0
        eval_strategy_features = _strategy_feature_frame(eval_candidates, eval_features)

        n0_schedule = _schedule_for_timestamps(eval_candidates["timestamp"], 1.0)
        _, _, n0_metrics, n0_accepted = _replay_schedule(
            eval_candidates,
            n0_schedule,
            params,
            ev_curve,
            market_mode=args.market_mode,
        )
        summary_rows.append(_metric_row_for_arm("N0_baseline", fold_id, n0_metrics, n0_schedule, n0_accepted))
        accepted_summary_frames.append(_accepted_summary_rows("N0_baseline", fold_id, n0_accepted))

        risk_pred = _predict_models(risk_models, eval_features, risk_feature_cols)
        g4_raw = _map_risk_to_multiplier(risk_pred["combined_risk"], risk_cutoffs)
        g4_raw = g4_raw.where(pd.to_numeric(risk_pred["pred_utility_q10"], errors="coerce") >= 0.0, 0.25)
        n1_schedule = pd.DataFrame({"timestamp": eval_features["timestamp"], "multiplier": g4_raw.to_numpy(dtype=float)})
        _, _, n1_metrics, n1_accepted = _replay_schedule(
            eval_candidates,
            n1_schedule,
            params,
            ev_curve,
            market_mode=args.market_mode,
        )
        summary_rows.append(_metric_row_for_arm("N1_global_risk_G4", fold_id, n1_metrics, n1_schedule, n1_accepted))
        accepted_summary_frames.append(_accepted_summary_rows("N1_global_risk_G4", fold_id, n1_accepted))

        n2_schedule = _predict_multiplier_schedule(
            mu_model,
            eval_features,
            model_feature_cols,
            min_positive_edge=float(args.min_positive_edge),
        )
        _, _, n2_metrics, n2_accepted = _replay_schedule(
            eval_candidates,
            n2_schedule,
            params,
            ev_curve,
            market_mode=args.market_mode,
        )
        summary_rows.append(_metric_row_for_arm("N2_global_marginal_utility", fold_id, n2_metrics, n2_schedule, n2_accepted))
        accepted_summary_frames.append(_accepted_summary_rows("N2_global_marginal_utility", fold_id, n2_accepted))

        if "size" in strategy_models:
            n3_schedule = _predict_strategy_schedule(
                strategy_models["size"],
                eval_strategy_features,
                strategy_feature_cols_by_action["size"],
                min_positive_edge=float(args.min_positive_edge),
            )
            _, _, n3_metrics, n3_accepted = _replay_strategy_schedule(
                eval_candidates,
                n3_schedule,
                params,
                ev_curve,
                action="size",
                market_mode=args.market_mode,
            )
            summary_rows.append(_metric_row_for_arm("N3_strategy_size_marginal_utility", fold_id, n3_metrics, n3_schedule, n3_accepted))
            accepted_summary_frames.append(_accepted_summary_rows("N3_strategy_size_marginal_utility", fold_id, n3_accepted))
        else:
            n3_schedule = pd.DataFrame()

        if "threshold" in strategy_models:
            n4_schedule = _predict_strategy_schedule(
                strategy_models["threshold"],
                eval_strategy_features,
                strategy_feature_cols_by_action["threshold"],
                min_positive_edge=float(args.min_positive_edge),
            )
            _, _, n4_metrics, n4_accepted = _replay_strategy_schedule(
                eval_candidates,
                n4_schedule,
                params,
                ev_curve,
                action="threshold",
                market_mode=args.market_mode,
            )
            summary_rows.append(_metric_row_for_arm("N4_strategy_threshold_marginal_utility", fold_id, n4_metrics, n4_schedule, n4_accepted))
            accepted_summary_frames.append(_accepted_summary_rows("N4_strategy_threshold_marginal_utility", fold_id, n4_accepted))
        else:
            n4_schedule = pd.DataFrame()

        if not n4_schedule.empty:
            _, _, n5_metrics, n5_accepted = _replay_strategy_threshold_with_global_cap(
                eval_candidates,
                n4_schedule,
                n1_schedule,
                params,
                ev_curve,
                market_mode=args.market_mode,
            )
            n5_schedule = n4_schedule.copy()
            n5_schedule = n5_schedule.merge(
                n1_schedule.rename(columns={"multiplier": "global_cap_multiplier"}),
                on="timestamp",
                how="left",
            )
            n5_schedule["multiplier"] = pd.to_numeric(n5_schedule["multiplier"], errors="coerce").fillna(1.0) * pd.to_numeric(
                n5_schedule["global_cap_multiplier"], errors="coerce"
            ).fillna(1.0)
            summary_rows.append(_metric_row_for_arm("N5_strategy_threshold_plus_emergency_cap", fold_id, n5_metrics, n5_schedule, n5_accepted))
            accepted_summary_frames.append(_accepted_summary_rows("N5_strategy_threshold_plus_emergency_cap", fold_id, n5_accepted))
        else:
            n5_schedule = pd.DataFrame()

        _, _, n6_metrics, n6_accepted = _replay_pressure_reallocation(
            eval_candidates,
            eval_features,
            params,
            ev_curve,
            reallocation_params=reallocation_params,
            market_mode=args.market_mode,
        )
        n6_schedule = _schedule_for_timestamps(eval_candidates["timestamp"], 1.0)
        for key, value in reallocation_params.items():
            n6_schedule[f"reallocation_{key}"] = float(value)
        summary_rows.append(_metric_row_for_arm("N6_optuna_pressure_reallocation", fold_id, n6_metrics, n6_schedule, n6_accepted))
        accepted_summary_frames.append(_accepted_summary_rows("N6_optuna_pressure_reallocation", fold_id, n6_accepted))

        _, _, n8_metrics, n8_accepted = _replay_pressure_reallocation(
            eval_candidates,
            eval_features,
            params,
            ev_curve,
            reallocation_params=middle_reallocation_params,
            market_mode=args.market_mode,
        )
        n8_schedule = _schedule_for_timestamps(eval_candidates["timestamp"], 1.0)
        for key, value in middle_reallocation_params.items():
            n8_schedule[f"reallocation_{key}"] = float(value)
        summary_rows.append(_metric_row_for_arm("N8_constrained_pressure_reallocation", fold_id, n8_metrics, n8_schedule, n8_accepted))
        accepted_summary_frames.append(_accepted_summary_rows("N8_constrained_pressure_reallocation", fold_id, n8_accepted))

        eval_label_panel, _, _ = _counterfactual_label_panel(
            eval_candidates,
            params,
            ev_curve,
            horizon_hours=int(args.horizon_hours),
            lambda_cost=float(args.lambda_cost),
            lambda_dd=float(args.lambda_dd),
            market_mode=args.market_mode,
        )
        oracle_schedule = _oracle_schedule(eval_label_panel)
        _, _, oracle_metrics, oracle_accepted = _replay_schedule(
            eval_candidates,
            oracle_schedule,
            params,
            ev_curve,
            market_mode=args.market_mode,
        )
        summary_rows.append(_metric_row_for_arm("N2_oracle_multiplier", fold_id, oracle_metrics, oracle_schedule, oracle_accepted))
        accepted_summary_frames.append(_accepted_summary_rows("N2_oracle_multiplier", fold_id, oracle_accepted))
        path_oracle_schedule = pd.DataFrame()
        path_oracle_diag: dict[str, Any] = {}
        if not args.skip_path_oracle:
            path_oracle_schedule, path_oracle_diag = _path_dependent_oracle_schedule(
                eval_candidates,
                params,
                ev_curve,
                market_mode=args.market_mode,
                chunk_hours=int(args.oracle_chunk_hours),
                beam_width=int(args.oracle_beam_width),
            )
            _, _, path_oracle_metrics, path_oracle_accepted = _replay_schedule(
                eval_candidates,
                path_oracle_schedule,
                params,
                ev_curve,
                market_mode=args.market_mode,
            )
            summary_rows.append(_metric_row_for_arm("N7_path_dependent_oracle_multiplier", fold_id, path_oracle_metrics, path_oracle_schedule, path_oracle_accepted))
            accepted_summary_frames.append(_accepted_summary_rows("N7_path_dependent_oracle_multiplier", fold_id, path_oracle_accepted))

        for arm, sched in (
            ("N0_baseline", n0_schedule),
            ("N1_global_risk_G4", n1_schedule),
            ("N2_global_marginal_utility", n2_schedule),
            ("N3_strategy_size_marginal_utility", n3_schedule),
            ("N4_strategy_threshold_marginal_utility", n4_schedule),
            ("N5_strategy_threshold_plus_emergency_cap", n5_schedule),
            ("N6_optuna_pressure_reallocation", n6_schedule),
            ("N8_constrained_pressure_reallocation", n8_schedule),
            ("N2_oracle_multiplier", oracle_schedule.rename(columns={"oracle_delta_J": "pred_delta_J"})),
            ("N7_path_dependent_oracle_multiplier", path_oracle_schedule.rename(columns={"oracle_objective": "pred_delta_J"})),
        ):
            if sched.empty:
                continue
            tmp = sched.copy()
            tmp["fold_id"] = fold_id
            tmp["arm"] = arm
            schedule_frames.append(tmp)
        pred = n2_schedule.copy()
        pred["fold_id"] = fold_id
        prediction_frames.append(pred)
        oracle = oracle_schedule.copy()
        oracle["fold_id"] = fold_id
        oracle_frames.append(oracle)
        if not path_oracle_schedule.empty:
            path_oracle = path_oracle_schedule.copy()
            path_oracle["fold_id"] = fold_id
            for key, value in path_oracle_diag.items():
                if key != "oracle_metrics":
                    path_oracle[key] = value
            oracle_frames.append(path_oracle)
        fold_rows.append(
            {
                **fold,
                "train_rows_long": int(len(train_long)),
                "raw_model_feature_count": int(raw_model_feature_count),
                "model_feature_count": int(len(model_feature_cols)),
                "strategy_size_model_feature_count": int(len(strategy_feature_cols_by_action.get("size", []))),
                "strategy_threshold_model_feature_count": int(len(strategy_feature_cols_by_action.get("threshold", []))),
                "optuna_trials": int(args.optuna_trials),
                "oracle_chunk_hours": int(args.oracle_chunk_hours),
                "oracle_beam_width": int(args.oracle_beam_width),
                "path_oracle_replay_count": int(path_oracle_diag.get("oracle_replay_count", 0)),
                "train_baseline_net_pnl_m1": float(train_metrics_by_m[1.0].get("net_pnl", 0.0)),
            }
        )

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        raise RuntimeError("No fold summaries generated")
    promotion = _promotion_summary(summary)
    accepted_summaries = (
        pd.concat(accepted_summary_frames, ignore_index=True)
        if accepted_summary_frames
        else pd.DataFrame()
    )
    delta_metric_frames = _accepted_delta_metric_frames(accepted_summaries)
    summary.to_csv(args.output_dir / "marginal_utility_fold_summary.csv", index=False)
    promotion.to_csv(args.output_dir / "marginal_utility_promotion_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "marginal_utility_folds.csv", index=False)
    if not accepted_summaries.empty:
        accepted_summaries.to_csv(args.output_dir / "marginal_utility_accepted_summaries.csv", index=False)
    for name, frame in delta_metric_frames.items():
        if not frame.empty:
            frame.to_csv(args.output_dir / f"marginal_utility_{name}_delta_vs_baseline.csv", index=False)
    if schedule_frames:
        pd.concat(schedule_frames, ignore_index=True).to_csv(args.output_dir / "marginal_utility_schedules.csv", index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(args.output_dir / "marginal_utility_predictions.csv", index=False)
    if oracle_frames:
        pd.concat(oracle_frames, ignore_index=True).to_csv(args.output_dir / "marginal_utility_oracle.csv", index=False)
    if feature_diagnostic_frames:
        pd.concat(feature_diagnostic_frames, ignore_index=True).to_csv(
            args.output_dir / "marginal_utility_feature_diagnostics.csv",
            index=False,
        )
    if optuna_frames:
        pd.concat(optuna_frames, ignore_index=True).to_csv(
            args.output_dir / "marginal_utility_optuna_trials.csv",
            index=False,
        )

    manifest = {
        "generated_by": "run_portfolio_marginal_utility_ablation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "broad_candidates": str(args.broad_candidates),
        "deployable_candidates": str(args.deployable_candidates),
        "policy_manifest": str(args.policy_manifest),
        "policy_variant": str(args.policy_variant),
        "policy_manifest_run_id": policy_payload.get("run_id"),
        "policy_params": asdict(params),
        "multipliers": list(MULTIPLIERS),
        "horizon_hours": int(args.horizon_hours),
        "embargo_hours": int(args.embargo_hours),
        "eval_start": args.eval_start,
        "eval_end": args.eval_end,
        "train_start": args.train_start,
        "model_max_feature_cols": int(args.model_max_feature_cols),
        "feature_min_non_null_frac": float(args.feature_min_non_null_frac),
        "optuna_trials": int(args.optuna_trials),
        "optuna_validation_hours": int(args.optuna_validation_hours),
        "middle_excess_cost_penalty": float(args.middle_excess_cost_penalty),
        "middle_excess_notional_penalty": float(args.middle_excess_notional_penalty),
        "oracle_chunk_hours": int(args.oracle_chunk_hours),
        "oracle_beam_width": int(args.oracle_beam_width),
        "skip_path_oracle": bool(args.skip_path_oracle),
        "recent_hr_window_hours": int(args.recent_hr_window_hours),
        "recent_hr_long_window_hours": int(args.recent_hr_long_window_hours),
        "recent_hr_min_count": int(args.recent_hr_min_count),
        "recent_hr_rank_weight": float(args.recent_hr_rank_weight),
        "recent_hr_priority_weight": float(args.recent_hr_priority_weight),
        "numba_available": bool(_numba_njit is not None),
        "lambda_cost": float(args.lambda_cost),
        "lambda_dd": float(args.lambda_dd),
        "fold_count": int(summary["fold_id"].nunique()),
        "limitations": [
            "Counterfactual labels are generated by replaying each multiplier through the fold, not by cloning an intra-replay state object at every timestamp.",
            "Strategy-response labels replay one strategy action at a time through the fold, so they are architecture diagnostics rather than exact single-timestamp intervention labels.",
            "N2_oracle_multiplier is retained as the old label oracle for comparison only; N7_path_dependent_oracle_multiplier is the path-consistent replay oracle.",
            "The path-dependent oracle is chunked and beam-limited, so it is a tractable lower-bound on exact timestamp-level search.",
        ],
        "outputs": {
            "fold_summary": str(args.output_dir / "marginal_utility_fold_summary.csv"),
            "promotion_summary": str(args.output_dir / "marginal_utility_promotion_summary.csv"),
            "schedules": str(args.output_dir / "marginal_utility_schedules.csv"),
            "predictions": str(args.output_dir / "marginal_utility_predictions.csv"),
            "oracle": str(args.output_dir / "marginal_utility_oracle.csv"),
            "feature_diagnostics": str(args.output_dir / "marginal_utility_feature_diagnostics.csv"),
            "optuna_trials": str(args.output_dir / "marginal_utility_optuna_trials.csv"),
            "accepted_summaries": str(args.output_dir / "marginal_utility_accepted_summaries.csv"),
            "weekly_delta_vs_baseline": str(args.output_dir / "marginal_utility_weekly_delta_vs_baseline.csv"),
            "strategy_delta_vs_baseline": str(args.output_dir / "marginal_utility_strategy_delta_vs_baseline.csv"),
            "weekly_strategy_delta_vs_baseline": str(args.output_dir / "marginal_utility_weekly_strategy_delta_vs_baseline.csv"),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n")
    print(json.dumps(_json_safe(manifest), indent=2)[:6000])
    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()
