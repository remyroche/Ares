#!/usr/bin/env python3
"""Ablate per-archetype recent hit-rate surprise threshold modulation.

This is intentionally replay-only: it reuses a saved side-parent execution
geometry, estimates recent side x archetype hit-rate surprise on the
optimisation sample, then applies frozen rank-threshold adjustments to the
holdout sample.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Match the side-archetype optimiser runner's replay defaults before importing
# modules that read environment-backed constants.
os.environ.setdefault("EPM_SIMPLE_POLICY_PER_ARCHETYPE_OPTIMISATION", "1")
os.environ.setdefault("EPM_SIMPLE_POLICY_PER_ARCHETYPE_FULL_OPTIMISATION", "1")
os.environ.setdefault("EPM_SIMPLE_POLICY_1M_DOWNLOAD", "1")
os.environ.setdefault("EPM_SIMPLE_POLICY_15M_DOWNLOAD", "1")

from scripts.ablate_simple_policy_exit_geometry import (  # noqa: E402
    _load_bundles,
    _prepare_rows,
)
from scripts.run_s52_side_archetype_simple_policy_optimiser import (  # noqa: E402
    _params_from_parent_summary_row,
    _split_optimisation_holdout_rows,
)
from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    PortfolioPolicyParams,
    fit_hierarchical_ev_curves,
    replay_candidates,
)
from extreme_price_movements.regime_ev_calibration import (  # noqa: E402
    CALIBRATION_POLICY_ID,
    default_regime_ev_calibration_artifact,
    default_regime_ev_feature_handoff,
)
from extreme_price_movements.simple_policy_optimiser import (  # noqa: E402
    calculate_advanced_metrics,
    simulate_and_score,
)


TOP_THRESHOLDS: dict[str, float] = {
    "top30": 0.70,
    "top20": 0.80,
    "top10": 0.90,
    "top5": 0.95,
}

PROMOTED_POLICY_NAME = "s52_archetype_hit_surprise_portfolio_top10_v1"
PROMOTED_TOP_SLICE = "top10"
PROMOTED_HIT_SURPRISE_MODE = "hit_surprise_priority_rank_50"
PROMOTED_HALF_LIFE_DAYS = 14.0
PROMOTED_ALPHA = 0.25
PROMOTED_MAX_ADJUST = 0.05
PROMOTED_MAX_CONCURRENT_POSITIONS = 10
PROMOTED_MAX_CONCURRENT_PER_SIDE: int | None = None
PROMOTED_MAX_NEW_ENTRIES_PER_BAR = 2


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _path_take(paths: tuple[np.ndarray, ...], idx: np.ndarray) -> tuple[np.ndarray, ...]:
    return tuple(arr[idx] for arr in paths)


def _load_parent_summary(path: Path) -> dict[str, dict[str, Any]]:
    df = pd.read_csv(path)
    if "strategy_id" not in df.columns:
        raise ValueError(f"Parent summary missing strategy_id: {path}")
    return {str(row["strategy_id"]): row.to_dict() for _, row in df.iterrows()}


def _expected_probability(rows: pd.DataFrame) -> np.ndarray:
    for col in ("calibrated_score", "meta_score_oof", "exec_guard_score_oof", "base_score_oof"):
        if col in rows.columns:
            values = pd.to_numeric(rows[col], errors="coerce").to_numpy(dtype=np.float64)
            if np.isfinite(values).any():
                return np.clip(np.nan_to_num(values, nan=np.nanmean(values)), 1e-4, 1.0 - 1e-4)
    return np.full(len(rows), 0.5, dtype=np.float64)


def _simulate_selected_rows(
    rows: pd.DataFrame,
    paths: tuple[np.ndarray, ...],
    *,
    rank_threshold: float,
    params: Mapping[str, Any],
    size_power: float,
    cost_pct: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if rows.empty or "rank_pct" not in rows.columns:
        return rows.iloc[0:0].copy(), {}
    rank = pd.to_numeric(rows["rank_pct"], errors="coerce").to_numpy(dtype=np.float64)
    idx = np.flatnonzero(np.isfinite(rank) & (rank >= float(rank_threshold)))
    if idx.size == 0:
        return rows.iloc[0:0].copy(), {}
    sub = rows.iloc[idx].copy().reset_index(drop=True)
    sub_paths = _path_take(paths, idx)
    metrics = simulate_and_score(
        sub,
        *sub_paths,
        cost_pct=float(cost_pct),
        size_power=float(size_power),
        **dict(params),
    )
    adv = calculate_advanced_metrics(
        sub,
        metrics.get("raw_gains", np.array([], dtype=np.float32)),
        metrics.get("sizes", np.array([], dtype=np.float32)),
        metrics.get("selected_mask"),
        metrics.get("gross_gains"),
        metrics.get("exit_reason"),
        metrics.get("exit_bars"),
    )
    selected_mask = np.asarray(metrics.get("selected_mask", np.zeros(len(sub), dtype=bool)), dtype=bool)
    selected = sub.loc[selected_mask].copy().reset_index(drop=True)
    raw = np.asarray(metrics.get("raw_gains", np.array([], dtype=np.float32)), dtype=np.float64)
    gross = np.asarray(metrics.get("gross_gains", np.array([], dtype=np.float32)), dtype=np.float64)
    sizes = np.asarray(metrics.get("sizes", np.array([], dtype=np.float32)), dtype=np.float64)
    exit_reason = np.asarray(metrics.get("exit_reason", np.array([], dtype=object))).astype(str)
    exit_bars = np.asarray(metrics.get("exit_bars", np.array([], dtype=np.int16)), dtype=np.float64)
    n = min(len(selected), len(raw), len(sizes), len(exit_reason), len(exit_bars))
    selected = selected.iloc[:n].copy()
    if n:
        selected["net_gain"] = raw[:n]
        selected["gross_gain"] = gross[:n] if len(gross) >= n else np.nan
        selected["position_size"] = sizes[:n]
        with np.errstate(divide="ignore", invalid="ignore"):
            selected["ret_net_notional"] = selected["net_gain"].to_numpy(dtype=np.float64) / np.maximum(sizes[:n], 1e-12)
            selected["ret_gross_notional"] = selected["gross_gain"].to_numpy(dtype=np.float64) / np.maximum(sizes[:n], 1e-12)
        selected["exit_reason"] = exit_reason[:n]
        selected["exit_bars"] = np.maximum(exit_bars[:n], 1.0)
        selected["expected_probability"] = _expected_probability(selected)
    return selected, adv


def _stamp_selection(
    selected: pd.DataFrame,
    *,
    mode: str,
    top_label: str,
    base_threshold: float,
    threshold_by_archetype: Mapping[str, float] | None = None,
    half_life_days: float | None = None,
    alpha: float | None = None,
    max_adjust: float | None = None,
) -> pd.DataFrame:
    out = selected.copy()
    out["mode"] = str(mode)
    out["top_slice"] = str(top_label)
    out["base_rank_threshold"] = float(base_threshold)
    out["half_life_days"] = np.nan if half_life_days is None else float(half_life_days)
    out["alpha"] = np.nan if alpha is None else float(alpha)
    out["max_adjust"] = np.nan if max_adjust is None else float(max_adjust)
    if threshold_by_archetype:
        if "policy_archetype" not in out.columns:
            out["policy_archetype"] = "missing"
        mapped = out["policy_archetype"].astype(str).map(dict(threshold_by_archetype))
        out["applied_rank_threshold"] = pd.to_numeric(mapped, errors="coerce").fillna(float(base_threshold))
    else:
        out["applied_rank_threshold"] = float(base_threshold)
    return out


def _portfolio_config_key(
    *,
    mode: str,
    top_label: str,
    half_life_days: float | None = None,
    alpha: float | None = None,
    max_adjust: float | None = None,
) -> tuple[Any, ...]:
    if mode == "baseline":
        return (str(top_label), "baseline", None, None, None)
    return (
        str(top_label),
        str(mode),
        float(half_life_days),
        float(alpha),
        float(max_adjust),
    )


def _surprise_quality_map(
    surprise: pd.DataFrame,
    *,
    alpha: float,
    max_adjust: float,
) -> dict[str, float]:
    """Return signed side/archetype HR quality adjustment.

    Positive values mean recent realized hit rate exceeded expected hit rate;
    negative values mean the cell is underperforming. The support confidence
    keeps low-sample cells close to neutral.
    """
    out: dict[str, float] = {}
    if surprise.empty:
        return out
    for _, row in surprise.iterrows():
        delta = _safe_float(row.get("hit_rate_delta"), 0.0)
        conf = _safe_float(row.get("support_confidence"), 0.0)
        adjust = float(np.clip(float(alpha) * delta * conf, -float(max_adjust), float(max_adjust)))
        out[str(row["policy_archetype"])] = adjust
    return out


def _apply_portfolio_hr_adjustments(
    selected: pd.DataFrame,
    *,
    mode: str,
    quality_by_archetype: Mapping[str, float],
    max_adjust: float,
) -> pd.DataFrame:
    """Attach HR fields consumed by the global portfolio replay.

    Modes are intentionally soft. They preserve the candidate set and let the
    auction reallocate priority/size, unlike the hard-threshold variant.
    """
    out = selected.copy()
    if out.empty:
        return out
    if "policy_archetype" not in out.columns:
        out["policy_archetype"] = "missing"
    q = (
        out["policy_archetype"]
        .astype(str)
        .map(dict(quality_by_archetype))
        .fillna(0.0)
        .astype(float)
    )
    denom = max(float(max_adjust), 1e-12)
    q_unit = (q / denom).clip(-1.0, 1.0)
    out["hr_quality_adjustment"] = q
    out["hr_quality_unit"] = q_unit
    if mode == "hit_surprise_priority":
        # Reorder within the auction while preserving rank thresholds.
        out["portfolio_priority_multiplier"] = (1.0 + q_unit).clip(0.25, 1.75)
    elif mode == "hit_surprise_priority_mild":
        out["portfolio_priority_multiplier"] = (1.0 + 0.5 * q_unit).clip(0.50, 1.50)
    elif mode == "hit_surprise_rank":
        # Small effective-rank nudge changes both auction priority and sizing.
        out["portfolio_rank_adjustment"] = q.clip(-float(max_adjust), float(max_adjust))
    elif mode == "hit_surprise_size":
        # Keep admission/ranking stable, but reduce/increase allocation.
        out["portfolio_size_multiplier"] = (1.0 + q_unit).clip(0.25, 1.75)
    elif mode == "hit_surprise_combined":
        out["portfolio_priority_multiplier"] = (1.0 + 0.5 * q_unit).clip(0.50, 1.50)
        out["portfolio_rank_adjustment"] = (0.5 * q).clip(-0.5 * float(max_adjust), 0.5 * float(max_adjust))
        out["portfolio_size_multiplier"] = (1.0 + 0.5 * q_unit).clip(0.50, 1.50)
    elif mode in {
        "hit_surprise_priority_rank_50",
        "hit_surprise_threshold_priority_rank_50",
    }:
        # 50/50 blend between priority reallocation and rank nudge; no sizing change.
        out["portfolio_priority_multiplier"] = (1.0 + 0.5 * q_unit).clip(0.50, 1.50)
        out["portfolio_rank_adjustment"] = (0.5 * q).clip(-0.5 * float(max_adjust), 0.5 * float(max_adjust))
    elif mode == "hit_surprise_downweight_only":
        # Only penalize weak cells; do not boost recently hot cells.
        penalty = q_unit.clip(-1.0, 0.0)
        out["portfolio_priority_multiplier"] = (1.0 + penalty).clip(0.25, 1.0)
        out["portfolio_size_multiplier"] = (1.0 + penalty).clip(0.25, 1.0)
    else:
        raise ValueError(f"Unknown HR portfolio adjustment mode: {mode}")
    return out


def _portfolio_candidate_table(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    out = selected.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if "policy_archetype" not in out.columns:
        out["policy_archetype"] = "missing"
    rank = pd.to_numeric(out.get("rank_pct"), errors="coerce")
    if rank.isna().all() and "normalized_rank_score" in out.columns:
        rank = pd.to_numeric(out["normalized_rank_score"], errors="coerce")
    out["normalized_rank_score"] = rank
    out["strategy_rank_pct"] = rank
    out["base_strategy_threshold"] = pd.to_numeric(
        out.get("applied_rank_threshold", out.get("base_rank_threshold", 0.0)),
        errors="coerce",
    ).fillna(pd.to_numeric(out.get("base_rank_threshold", 0.0), errors="coerce")).fillna(0.0)
    if "calibrated_score" not in out.columns or pd.to_numeric(out["calibrated_score"], errors="coerce").isna().all():
        out["calibrated_score"] = out.get("expected_probability", rank)
    out["entry_price"] = 1.0
    out["net_return"] = pd.to_numeric(out["ret_net_notional"], errors="coerce")
    out["gross_return"] = pd.to_numeric(out["ret_gross_notional"], errors="coerce")
    out["exit_price"] = 1.0 + out["gross_return"].fillna(0.0)
    bars = pd.to_numeric(out.get("exit_bars"), errors="coerce").fillna(1.0).clip(lower=1.0)
    out["holding_bars"] = bars
    out["exit_timestamp"] = out["timestamp"] + pd.to_timedelta((bars * 15.0).round().astype(int), unit="m")
    out["simple_policy_exit_reason"] = out.get("exit_reason", "").astype(str)
    out["fees_bps"] = 100.0
    out["slippage_bps"] = 0.0
    out["expected_friction_bps"] = 100.0
    out["price_gap_bps"] = 0.0
    out["liquidity_capacity_weight"] = 1.0
    keep = [
        "timestamp",
        "symbol",
        "side",
        "strategy_id",
        "policy_archetype",
        "candidate_uid",
        "mode",
        "top_slice",
        "base_rank_threshold",
        "half_life_days",
        "alpha",
        "max_adjust",
        "normalized_rank_score",
        "strategy_rank_pct",
        "base_strategy_threshold",
        "calibrated_score",
        "entry_price_real",
        "barrier_pct",
        "expected_spread_bps",
        "expected_half_spread_bps",
        "spread_cost_bps",
        "entry_price",
        "exit_timestamp",
        "exit_price",
        "net_return",
        "gross_return",
        "holding_bars",
        "simple_policy_exit_reason",
        "fees_bps",
        "slippage_bps",
        "expected_friction_bps",
        "price_gap_bps",
        "liquidity_capacity_weight",
        "portfolio_priority_multiplier",
        "portfolio_priority_adjustment",
        "portfolio_rank_adjustment",
        "portfolio_size_multiplier",
        "portfolio_wallet_cap_multiplier",
        "hr_quality_adjustment",
        "hr_quality_unit",
    ]
    present = [col for col in keep if col in out.columns]
    result = out[present].dropna(
        subset=["timestamp", "normalized_rank_score", "net_return", "gross_return"]
    ).copy()
    decision_keys = [col for col in ("timestamp", "symbol", "strategy_id") if col in result.columns]
    if len(decision_keys) == 3:
        sort_cols = [
            col
            for col in (
                "normalized_rank_score",
                "calibrated_score",
                "portfolio_priority_multiplier",
                "portfolio_size_multiplier",
                "net_return",
            )
            if col in result.columns
        ]
        if sort_cols:
            result = result.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        result = result.drop_duplicates(decision_keys, keep="first")
    return result.copy()


def _portfolio_metrics_rows(
    selected_frames: Mapping[tuple[Any, ...], list[pd.DataFrame]],
    *,
    portfolio_grid: list[PortfolioPolicyParams],
    market_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    accepted_frames: list[pd.DataFrame] = []
    rejection_rows: list[dict[str, Any]] = []
    for key, frames in selected_frames.items():
        top_label, mode, half_life, alpha, max_adjust = key
        candidates = _portfolio_candidate_table(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame()
        if candidates.empty:
            continue
        ev_curve = fit_hierarchical_ev_curves(candidates)
        for params in portfolio_grid:
            decisions, _equity, metrics = replay_candidates(
                candidates,
                params,
                mode="global_auction",
                ev_curve=ev_curve,
                market_mode=market_mode,
            )
            accepted = decisions[decisions["accepted"]].copy()
            if not accepted.empty:
                accepted["top_slice"] = str(top_label)
                accepted["mode"] = str(mode)
                accepted["half_life_days"] = np.nan if half_life is None else float(half_life)
                accepted["alpha"] = np.nan if alpha is None else float(alpha)
                accepted["max_adjust"] = np.nan if max_adjust is None else float(max_adjust)
                accepted["portfolio_max_concurrent_positions"] = int(params.max_concurrent_positions)
                accepted["portfolio_max_concurrent_per_side"] = (
                    np.nan if params.max_concurrent_per_side is None else int(params.max_concurrent_per_side)
                )
                accepted_frames.append(accepted)
            rejection = decisions["rejection_reason"].astype(str).value_counts() if not decisions.empty else pd.Series(dtype=int)
            for reason, count in rejection.items():
                rejection_rows.append(
                    {
                        "top_slice": str(top_label),
                        "mode": str(mode),
                        "half_life_days": np.nan if half_life is None else float(half_life),
                        "alpha": np.nan if alpha is None else float(alpha),
                        "max_adjust": np.nan if max_adjust is None else float(max_adjust),
                        "max_concurrent_positions": int(params.max_concurrent_positions),
                        "max_concurrent_per_side": np.nan
                        if params.max_concurrent_per_side is None
                        else int(params.max_concurrent_per_side),
                        "rejection_reason": str(reason),
                        "count": int(count),
                    }
                )
            metric_rows.append(
                {
                    "top_slice": str(top_label),
                    "mode": str(mode),
                    "half_life_days": np.nan if half_life is None else float(half_life),
                    "alpha": np.nan if alpha is None else float(alpha),
                    "max_adjust": np.nan if max_adjust is None else float(max_adjust),
                    "max_concurrent_positions": int(params.max_concurrent_positions),
                    "max_concurrent_per_side": np.nan
                    if params.max_concurrent_per_side is None
                    else int(params.max_concurrent_per_side),
                    "max_new_entries_per_bar": int(params.max_new_entries_per_bar),
                    "trade_count": int(metrics.get("trade_count", 0)),
                    "trades_per_day": _safe_float(metrics.get("trades_per_day")),
                    "net_pnl": _safe_float(metrics.get("net_pnl")),
                    "gross_pnl": _safe_float(metrics.get("gross_pnl")),
                    "compounded_return": _safe_float(metrics.get("compounded_return")),
                    "max_drawdown": _safe_float(metrics.get("max_drawdown")),
                    "worst_week": _safe_float(metrics.get("worst_week")),
                    "notional_weighted_net_return": _safe_float(metrics.get("notional_weighted_net_return")),
                    "mean_net_return_per_trade": _safe_float(metrics.get("mean_net_return_per_trade")),
                    "mean_gross_return_per_trade": _safe_float(metrics.get("mean_gross_return_per_trade")),
                    "mean_position_pct_entry_wallet": _safe_float(metrics.get("mean_position_pct_entry_wallet")),
                    "full_sl_rate": _safe_float(metrics.get("full_sl_rate")),
                    "timeout_rate": _safe_float(metrics.get("timeout_rate")),
                    "avg_open_positions": _safe_float(metrics.get("avg_open_positions")),
                    "position_utilization": _safe_float(metrics.get("position_utilization")),
                    "missed_high_confidence_trades": int(metrics.get("missed_high_confidence_trades", 0)),
                    "objective": _safe_float(metrics.get("objective")),
                }
            )
    metrics_df = pd.DataFrame(metric_rows)
    accepted_df = pd.concat(accepted_frames, ignore_index=True) if accepted_frames else pd.DataFrame()
    rejection_df = pd.DataFrame(rejection_rows)
    return metrics_df, accepted_df, rejection_df


def _weighted_surprise(
    selected: pd.DataFrame,
    *,
    holdout_start: pd.Timestamp,
    half_life_days: float,
    base_threshold: float,
    min_effective_n: float,
) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    work = selected.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work[work["timestamp"].lt(holdout_start)].copy()
    if work.empty:
        return pd.DataFrame()
    age_days = (holdout_start - work["timestamp"]).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
    max_age = max(1.0, float(half_life_days) * 4.0)
    keep = np.isfinite(age_days) & (age_days >= 0.0) & (age_days <= max_age)
    work = work.loc[keep].copy()
    age_days = age_days[keep]
    if work.empty:
        return pd.DataFrame()
    weights = np.exp(-math.log(2.0) * age_days / max(float(half_life_days), 1e-6))
    work["_w"] = weights
    work["_actual"] = (pd.to_numeric(work["net_gain"], errors="coerce") > 0.0).astype(float)
    work["_expected"] = pd.to_numeric(work["expected_probability"], errors="coerce").clip(1e-4, 1.0 - 1e-4)
    work["_var"] = work["_expected"] * (1.0 - work["_expected"])
    if "policy_archetype" not in work.columns:
        work["policy_archetype"] = "missing"
    rows: list[dict[str, Any]] = []
    for archetype, group in work.groupby("policy_archetype", dropna=False):
        w = group["_w"].to_numpy(dtype=np.float64)
        actual = group["_actual"].to_numpy(dtype=np.float64)
        expected = group["_expected"].to_numpy(dtype=np.float64)
        var = group["_var"].to_numpy(dtype=np.float64)
        w_sum = float(np.sum(w))
        if w_sum <= 0.0:
            continue
        n_eff = float((w_sum * w_sum) / max(np.sum(w * w), 1e-12))
        actual_rate = float(np.sum(w * actual) / w_sum)
        expected_rate = float(np.sum(w * expected) / w_sum)
        delta = actual_rate - expected_rate
        surprise = float(np.sum(w * (actual - expected)))
        surprise_var = float(np.sum((w * w) * np.clip(var, 1e-6, np.inf)))
        z = float(surprise / math.sqrt(max(surprise_var, 1e-12)))
        support_conf = float(np.clip(n_eff / max(float(min_effective_n), 1.0), 0.0, 1.0))
        rows.append(
            {
                "policy_archetype": str(archetype),
                "base_rank_threshold": float(base_threshold),
                "half_life_days": float(half_life_days),
                "lookback_max_age_days": float(max_age),
                "rows": int(len(group)),
                "n_eff": n_eff,
                "support_confidence": support_conf,
                "actual_hit_rate": actual_rate,
                "expected_hit_rate": expected_rate,
                "hit_rate_delta": delta,
                "hit_rate_surprise_z": z,
                "mean_recent_ret_net_notional": _safe_float(group["ret_net_notional"].mean()),
                "last_timestamp": str(group["timestamp"].max()),
            }
        )
    return pd.DataFrame(rows)


def _threshold_map(
    surprise: pd.DataFrame,
    *,
    base_threshold: float,
    alpha: float,
    max_adjust: float,
    min_threshold: float,
    max_threshold: float,
) -> dict[str, float]:
    out: dict[str, float] = {}
    if surprise.empty:
        return out
    for _, row in surprise.iterrows():
        delta = _safe_float(row.get("hit_rate_delta"), 0.0)
        conf = _safe_float(row.get("support_confidence"), 0.0)
        # Negative surprise tightens threshold; positive surprise loosens it.
        raw_adjust = -float(alpha) * delta * conf
        adjust = float(np.clip(raw_adjust, -float(max_adjust), float(max_adjust)))
        threshold = float(np.clip(float(base_threshold) + adjust, float(min_threshold), float(max_threshold)))
        out[str(row["policy_archetype"])] = threshold
    return out


def _evaluate_dynamic_thresholds(
    rows: pd.DataFrame,
    paths: tuple[np.ndarray, ...],
    *,
    threshold_by_archetype: Mapping[str, float],
    base_threshold: float,
    params: Mapping[str, Any],
    size_power: float,
    cost_pct: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = rows.copy().reset_index(drop=True)
    if "policy_archetype" not in work.columns:
        work["policy_archetype"] = "missing"
    thresholds = work["policy_archetype"].astype(str).map(threshold_by_archetype).fillna(float(base_threshold))
    rank = pd.to_numeric(work["rank_pct"], errors="coerce")
    idx = np.flatnonzero(rank.ge(pd.to_numeric(thresholds, errors="coerce")).fillna(False).to_numpy())
    if idx.size == 0:
        return work.iloc[0:0].copy(), {}
    return _simulate_selected_rows(
        work.iloc[idx].copy().reset_index(drop=True),
        _path_take(paths, idx),
        rank_threshold=0.0,
        params=params,
        size_power=size_power,
        cost_pct=cost_pct,
    )


def _record_metrics(
    *,
    side: str,
    top_label: str,
    mode: str,
    selected: pd.DataFrame,
    adv: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = dict(config or {})
    n = int(len(selected))
    return {
        "side": side,
        "top_slice": top_label,
        "mode": mode,
        **config,
        "n_trades": n,
        "avg_pnl_notional": _safe_float(adv.get("avg_pnl_notional", adv.get("avg_pnl_sized"))),
        "avg_pnl_bankroll": _safe_float(adv.get("avg_pnl_bankroll")),
        "avg_gross_return_per_trade": _safe_float(adv.get("avg_gross_return_per_trade")),
        "hit_rate": _safe_float(adv.get("hit_rate")),
        "pnl_positive_rate": _safe_float(adv.get("pnl_positive_rate")),
        "full_sl_exit_rate": _safe_float(adv.get("full_sl_exit_rate")),
        "timeout_exit_rate": _safe_float(adv.get("timeout_exit_rate")),
        "worst_week": _safe_float(adv.get("worst_week")),
        "max_dd": _safe_float(adv.get("max_dd")),
    }


def _parse_top_thresholds(text: str) -> dict[str, float]:
    values = [item.strip() for item in str(text or "").split(",") if item.strip()]
    if not values:
        values = [PROMOTED_TOP_SLICE]
    unknown = [item for item in values if item not in TOP_THRESHOLDS]
    if unknown:
        raise ValueError(f"Unknown --top-slices values {unknown}; expected one of {sorted(TOP_THRESHOLDS)}")
    return {item: TOP_THRESHOLDS[item] for item in values}


def _write_promoted_policy_artifacts(
    out_dir: Path,
    *,
    thresholds: pd.DataFrame,
    portfolio_metrics: pd.DataFrame,
    portfolio_best: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    policy_dir = out_dir / "policy_params"
    policy_dir.mkdir(parents=True, exist_ok=True)
    regime_ev_manifest = dict(manifest.get("regime_ev_calibration") or {})
    allocation_pressure = dict(manifest.get("portfolio_allocation_pressure") or {})
    selected_top_slice = PROMOTED_TOP_SLICE
    selected_mode = PROMOTED_HIT_SURPRISE_MODE
    selected_half_life = float(PROMOTED_HALF_LIFE_DAYS)
    selected_alpha = float(PROMOTED_ALPHA)
    selected_max_adjust = float(PROMOTED_MAX_ADJUST)
    selected_max_concurrent = int(PROMOTED_MAX_CONCURRENT_POSITIONS)
    selected_max_concurrent_per_side = PROMOTED_MAX_CONCURRENT_PER_SIDE
    if portfolio_best is not None and not portfolio_best.empty:
        best_rows = portfolio_best.loc[
            portfolio_best["top_slice"].astype(str).eq(PROMOTED_TOP_SLICE)
        ].copy()
        if not best_rows.empty:
            best_row = best_rows.iloc[0]
            selected_top_slice = str(best_row.get("top_slice", selected_top_slice))
            selected_mode = str(best_row.get("mode", selected_mode))
            selected_half_life = _safe_float(
                best_row.get("half_life_days"),
                selected_half_life,
            )
            selected_alpha = _safe_float(best_row.get("alpha"), selected_alpha)
            selected_max_adjust = _safe_float(
                best_row.get("max_adjust"),
                selected_max_adjust,
            )
            selected_max_concurrent = int(
                _safe_float(
                    best_row.get("max_concurrent_positions"),
                    selected_max_concurrent,
                )
            )
            max_side_value = best_row.get("max_concurrent_per_side")
            if pd.notna(max_side_value):
                selected_max_concurrent_per_side = int(float(max_side_value))
    promoted = {
        "schema_version": "s52_archetype_hit_surprise_portfolio_policy_v1",
        "policy_name": PROMOTED_POLICY_NAME,
        "status": "promoted_default",
        "rank_source": {
            "regime_ev_calibration_enabled": bool(
                regime_ev_manifest.get("enabled", True)
            ),
            "regime_ev_calibration_policy_id": str(
                regime_ev_manifest.get(
                    "policy_id", "per_regime_archetype_calibration_v1"
                )
            ),
            "regime_ev_calibration_artifact_path": str(
                regime_ev_manifest.get("artifact_path", "")
            ),
            "regime_ev_feature_handoff_path": str(
                regime_ev_manifest.get("feature_handoff_path", "")
            ),
            "hit_surprise_expected_probability_source": (
                "calibrated_score_after_regime_ev_calibration"
            ),
            "rank_score_col": "rank_pct",
            "rank_scope": str(regime_ev_manifest.get("rank_scope", "per_strategy")),
            "regime_ev_protect_admission_rank": bool(
                regime_ev_manifest.get("used_for_admission_rank", True)
            ),
            "regime_ev_protected_admission_floor": float(
                regime_ev_manifest.get(
                    "protected_admission_floor",
                    TOP_THRESHOLDS[PROMOTED_TOP_SLICE],
                )
                or TOP_THRESHOLDS[PROMOTED_TOP_SLICE]
            ),
            "regime_ev_retained_surplus_frac": float(
                regime_ev_manifest.get("retained_surplus_frac", 0.5) or 0.5
            ),
        },
        "selection": {
            "top_slice": selected_top_slice,
            "base_rank_threshold": float(TOP_THRESHOLDS[selected_top_slice]),
            "hit_surprise_mode": selected_mode,
            "hit_surprise_threshold_enabled": selected_mode == "hit_surprise_threshold",
            "hit_surprise_priority_rank_enabled": selected_mode
            in {
                "hit_surprise_priority_rank_50",
                "hit_surprise_threshold_priority_rank_50",
            },
            "hit_surprise_threshold_priority_rank_enabled": selected_mode
            == "hit_surprise_threshold_priority_rank_50",
            "group_key": "side_x_policy_archetype",
            "half_life_days": float(selected_half_life),
            "alpha": float(selected_alpha),
            "max_adjust": float(selected_max_adjust),
            "min_effective_n": float(manifest.get("min_effective_n", 20.0)),
            "recent_window_days": float(selected_half_life * 4.0),
            "threshold_rule": (
                "adjusted_rank_threshold = clip(base_rank_threshold - alpha * "
                "hit_rate_delta * support_confidence, base-max_adjust, base+max_adjust)"
            ),
            "priority_rank_rule": (
                "quality_adjustment = clip(alpha * hit_rate_delta * support_confidence, "
                "-max_adjust, max_adjust); priority_multiplier = clip(1 + 0.5 * "
                "quality_adjustment / max_adjust, 0.50, 1.50); rank_adjustment = "
                "0.5 * quality_adjustment"
            ),
            "threshold_priority_rank_rule": (
                "adjusted_rank_threshold is applied first, then the same 50/50 "
                "priority/rank adjustment is applied inside the portfolio auction"
            ),
        },
        "portfolio": {
            "portfolio_policy_version": "global_auction_v1",
            "max_concurrent_positions": int(selected_max_concurrent),
            "max_concurrent_per_side": selected_max_concurrent_per_side,
            "max_concurrent_per_strategy": None,
            "max_concurrent_per_symbol": 1,
            "max_new_entries_per_bar": int(PROMOTED_MAX_NEW_ENTRIES_PER_BAR),
            "max_new_entries_per_strategy_per_bar": None,
            "max_total_wallet_allocation_pct": 0.75,
            "global_threshold_floor": 0.0,
            "occupancy_threshold_alpha": 0.30,
            "occupancy_threshold_power": 1.50,
            "rank_size_power": 1.50,
            "rank_multiplier_min": 0.50,
            "rank_multiplier_max": 1.50,
        },
        "costs": {
            "round_trip_cost_pct": float(manifest.get("round_trip_cost_pct", 0.01)),
            "cost_pct_per_side": float(manifest.get("cost_pct_per_side", 0.005)),
            "spread_included": True,
        },
        "source": {
            "generated_by": manifest.get("generated_by"),
            "candidates": manifest.get("candidates"),
            "parent_policy_summary": manifest.get("parent_policy_summary"),
            "holdout": manifest.get("holdout"),
            "report_dir": str(out_dir),
        },
    }
    if thresholds is not None and not thresholds.empty:
        mask = (
            thresholds["top_slice"].astype(str).eq(selected_top_slice)
            & thresholds["half_life_days"].astype(float).eq(float(selected_half_life))
            & thresholds["alpha"].astype(float).eq(float(selected_alpha))
            & thresholds["max_adjust"].astype(float).eq(float(selected_max_adjust))
        )
        promoted_thresholds = thresholds.loc[mask].copy()
        if not promoted_thresholds.empty:
            promoted["archetype_thresholds"] = promoted_thresholds[
                [
                    "side",
                    "strategy_id",
                    "policy_archetype",
                    "base_rank_threshold",
                    "adjusted_rank_threshold",
                    "threshold_delta",
                    "actual_hit_rate",
                    "expected_hit_rate",
                    "hit_rate_delta",
                    "hit_rate_surprise_z",
                    "support_confidence",
                    "n_eff",
                    "rows",
                ]
            ].to_dict("records")
            quality = (
                pd.to_numeric(promoted_thresholds["hit_rate_delta"], errors="coerce").fillna(0.0)
                * pd.to_numeric(promoted_thresholds["support_confidence"], errors="coerce").fillna(0.0)
                * float(selected_alpha)
            ).clip(-float(selected_max_adjust), float(selected_max_adjust))
            q_unit = (
                quality / float(selected_max_adjust)
                if float(selected_max_adjust) > 0.0
                else quality * 0.0
            )
            adjustments = promoted_thresholds.copy()
            adjustments["quality_adjustment"] = quality.astype(float)
            adjustments["portfolio_priority_multiplier"] = (1.0 + 0.5 * q_unit).clip(0.50, 1.50)
            adjustments["portfolio_priority_adjustment"] = 0.0
            adjustments["portfolio_rank_adjustment"] = (0.5 * quality).clip(
                -0.5 * float(selected_max_adjust),
                0.5 * float(selected_max_adjust),
            )
            promoted["archetype_adjustments"] = adjustments[
                [
                    "side",
                    "strategy_id",
                    "policy_archetype",
                    "base_rank_threshold",
                    "adjusted_rank_threshold",
                    "threshold_delta",
                    "quality_adjustment",
                    "portfolio_priority_multiplier",
                    "portfolio_priority_adjustment",
                    "portfolio_rank_adjustment",
                    "actual_hit_rate",
                    "expected_hit_rate",
                    "hit_rate_delta",
                    "hit_rate_surprise_z",
                    "support_confidence",
                    "n_eff",
                    "rows",
                ]
            ].to_dict("records")
    promoted_metrics = pd.DataFrame()
    if portfolio_metrics is not None and not portfolio_metrics.empty:
        max_side = portfolio_metrics["max_concurrent_per_side"]
        max_side_match = max_side.isna() if selected_max_concurrent_per_side is None else max_side.astype(int).eq(int(selected_max_concurrent_per_side))
        promoted_metrics = portfolio_metrics.loc[
            portfolio_metrics["top_slice"].astype(str).eq(selected_top_slice)
            & portfolio_metrics["mode"].astype(str).eq(selected_mode)
            & portfolio_metrics["half_life_days"].astype(float).eq(float(selected_half_life))
            & portfolio_metrics["alpha"].astype(float).eq(float(selected_alpha))
            & portfolio_metrics["max_adjust"].astype(float).eq(float(selected_max_adjust))
            & portfolio_metrics["max_concurrent_positions"].astype(int).eq(int(selected_max_concurrent))
            & max_side_match
        ]
    if promoted_metrics.empty and portfolio_best is not None and not portfolio_best.empty:
        promoted_metrics = portfolio_best.loc[
            portfolio_best["top_slice"].astype(str).eq(selected_top_slice)
        ]
    if not promoted_metrics.empty:
        promoted["promotion_metrics"] = promoted_metrics.iloc[0].to_dict()

    policy_path = policy_dir / "hit_surprise_archetype_portfolio_policy.json"
    policy_path.write_text(json.dumps(_json_safe(promoted), indent=2, sort_keys=True), encoding="utf-8")

    portfolio_payload = {
        "schema_version": "portfolio_policy_v1",
        "policy_name": PROMOTED_POLICY_NAME,
        "selection": {
            "global_threshold_floor": 0.0,
            "initial_rank_threshold": float(TOP_THRESHOLDS[selected_top_slice]),
            "initial_rank_threshold_floor": float(TOP_THRESHOLDS[selected_top_slice]),
            "threshold_viability_margin": 0.0,
            "occupancy_threshold_alpha": 0.30,
            "occupancy_threshold_power": 1.50,
            "allocation_threshold_alpha": float(
                allocation_pressure.get("allocation_threshold_alpha", 0.0)
            ),
            "allocation_threshold_power": float(
                allocation_pressure.get("allocation_threshold_power", 1.0)
            ),
            "archetype_hit_surprise_enabled": True,
            "archetype_hit_surprise_policy_path": str(policy_path),
            "archetype_hit_surprise_mode": selected_mode,
            "regime_ev_calibration_enabled": bool(
                regime_ev_manifest.get("enabled", True)
            ),
            "regime_ev_calibration_policy_id": str(
                regime_ev_manifest.get(
                    "policy_id", "per_regime_archetype_calibration_v1"
                )
            ),
            "regime_ev_calibration_artifact_path": str(
                regime_ev_manifest.get("artifact_path", "")
            ),
            "regime_ev_calibration_rank_source": str(
                regime_ev_manifest.get(
                    "policy_id", "per_regime_archetype_calibration_v1"
                )
            ),
            "regime_ev_protect_admission_rank": bool(
                regime_ev_manifest.get("used_for_admission_rank", True)
            ),
            "regime_ev_protected_admission_floor": float(
                regime_ev_manifest.get(
                    "protected_admission_floor",
                    TOP_THRESHOLDS[PROMOTED_TOP_SLICE],
                )
                or TOP_THRESHOLDS[PROMOTED_TOP_SLICE]
            ),
            "regime_ev_retained_surplus_frac": float(
                regime_ev_manifest.get("retained_surplus_frac", 0.5) or 0.5
            ),
        },
        "concurrency": {
            "max_concurrent_positions": int(selected_max_concurrent),
            "max_concurrent_per_side": selected_max_concurrent_per_side,
            "max_concurrent_per_strategy": None,
            "max_concurrent_per_symbol": 1,
            "max_new_entries_per_bar": int(PROMOTED_MAX_NEW_ENTRIES_PER_BAR),
            "max_new_entries_per_strategy_per_bar": None,
        },
        "allocation": {
            "max_total_wallet_allocation_pct": 0.75,
        },
        "regime_ev_calibration_enabled": bool(
            regime_ev_manifest.get("enabled", True)
        ),
        "regime_ev_calibration_policy_id": str(
            regime_ev_manifest.get("policy_id", "per_regime_archetype_calibration_v1")
        ),
        "regime_ev_calibration_artifact_path": str(
            regime_ev_manifest.get("artifact_path", "")
        ),
        "regime_ev_calibration_rank_source": str(
            regime_ev_manifest.get("policy_id", "per_regime_archetype_calibration_v1")
        ),
        "regime_ev_protect_admission_rank": bool(
            regime_ev_manifest.get("used_for_admission_rank", True)
        ),
        "regime_ev_protected_admission_floor": float(
            regime_ev_manifest.get(
                "protected_admission_floor",
                TOP_THRESHOLDS[PROMOTED_TOP_SLICE],
            )
            or TOP_THRESHOLDS[PROMOTED_TOP_SLICE]
        ),
        "regime_ev_retained_surplus_frac": float(
            regime_ev_manifest.get("retained_surplus_frac", 0.5) or 0.5
        ),
        "sizing": {
            "rank_multiplier_min": 0.50,
            "rank_multiplier_max": 1.50,
            "rank_size_power": 1.50,
        },
        "friction": {
            "min_liquidity_capacity_weight": None,
            "offline_default_price_gap_bps": 50.0,
        },
        "strategy_contract": {
            "strategy_ids": [
                "long_s52_meta_threshold_handoff",
                "short_s52_meta_threshold_handoff",
            ],
            "strategy_cores": [
                "s52_meta_threshold_handoff",
            ],
        },
    }
    portfolio_path = policy_dir / "optimized_portfolio_policy_config.json"
    portfolio_path.write_text(
        json.dumps(_json_safe(portfolio_payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest_path = out_dir / "promoted_policy_manifest.json"
    manifest_path.write_text(
        json.dumps(
            _json_safe(
                {
                    "policy_name": PROMOTED_POLICY_NAME,
                    "status": "promoted_default",
                    "policy_path": str(policy_path),
                    "portfolio_policy_path": str(portfolio_path),
                    "top_slice": selected_top_slice,
                    "hit_surprise_mode": selected_mode,
                    "half_life_days": selected_half_life,
                    "alpha": selected_alpha,
                    "max_adjust": selected_max_adjust,
                    "max_concurrent_positions": selected_max_concurrent,
                    "max_concurrent_per_side": selected_max_concurrent_per_side,
                    "regime_ev_protect_admission_rank": bool(
                        regime_ev_manifest.get("used_for_admission_rank", True)
                    ),
                    "regime_ev_protected_admission_floor": float(
                        regime_ev_manifest.get(
                            "protected_admission_floor",
                            TOP_THRESHOLDS[selected_top_slice],
                        )
                        or TOP_THRESHOLDS[selected_top_slice]
                    ),
                    "regime_ev_retained_surplus_frac": float(
                        regime_ev_manifest.get("retained_surplus_frac", 0.5) or 0.5
                    ),
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "promoted_policy": str(policy_path),
        "optimized_portfolio_policy_config": str(portfolio_path),
        "promoted_policy_manifest": str(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--parent-policy-summary", type=Path, required=True)
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--market-mode", choices=["spot", "perps"], default="perps")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--holdout-start", default="2026-06-16")
    parser.add_argument("--holdout-end", default="2026-07-01")
    parser.add_argument("--min-rank", type=float, default=0.0)
    parser.add_argument("--path-len", type=int, default=96)
    parser.add_argument("--round-trip-cost-pct", type=float, default=0.01)
    parser.add_argument("--top-slices", default=PROMOTED_TOP_SLICE)
    parser.add_argument(
        "--regime-ev-rerank-admission",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Recompute rank_pct from per_regime_archetype_calibration_v1 before "
            "top-k selection. Enabled by default with an admission floor so raw "
            "top10 rows cannot be pushed below the top10 threshold."
        ),
    )
    parser.add_argument(
        "--regime-ev-protected-admission-floor",
        type=float,
        default=TOP_THRESHOLDS[PROMOTED_TOP_SLICE],
        help=(
            "Rows above this original rank_pct are protected when regime EV "
            "reranking is applied. Defaults to top10."
        ),
    )
    parser.add_argument(
        "--regime-ev-retained-surplus-frac",
        type=float,
        default=0.5,
        help=(
            "Fraction of raw rank surplus above the protected admission floor "
            "retained when clipping negative regime EV rank adjustments."
        ),
    )
    parser.add_argument("--half-lives", default=str(PROMOTED_HALF_LIFE_DAYS))
    parser.add_argument("--alphas", default=str(PROMOTED_ALPHA))
    parser.add_argument("--max-adjusts", default=str(PROMOTED_MAX_ADJUST))
    parser.add_argument(
        "--hr-modes",
        default="threshold,priority_mild,priority,rank,size,combined,downweight_only",
        help=(
            "Comma-separated HR uses: threshold, priority_mild, priority, rank, "
            "size, combined, priority_rank_50, downweight_only. Soft modes preserve the base "
            "candidate set and feed portfolio_* adjustment columns to the "
            "global auction manager."
        ),
    )
    parser.add_argument("--min-effective-n", type=float, default=20.0)
    parser.add_argument("--min-survival", type=float, default=0.80)
    parser.add_argument(
        "--portfolio-max-concurrent",
        default=str(PROMOTED_MAX_CONCURRENT_POSITIONS),
        help="Comma-separated global max concurrent position caps for portfolio replay.",
    )
    parser.add_argument(
        "--portfolio-max-per-side",
        default="none",
        help="Comma-separated max concurrent per-side caps, or none.",
    )
    parser.add_argument("--portfolio-max-new-entries-per-bar", type=int, default=PROMOTED_MAX_NEW_ENTRIES_PER_BAR)
    parser.add_argument("--portfolio-allocation-threshold-alpha", type=float, default=0.30)
    parser.add_argument("--portfolio-allocation-threshold-power", type=float, default=1.0)
    parser.add_argument("--portfolio-save-accepted", action="store_true")
    parser.add_argument("--save-preportfolio-selected", action="store_true")
    parser.add_argument(
        "--force-sl-abs-cap-pct",
        type=float,
        default=np.nan,
        help="If finite and >0, override replay geometry sl_abs_cap_pct for all sides/archetypes.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cost_pct = float(args.round_trip_cost_pct) / 2.0
    holdout_start = pd.Timestamp(args.holdout_start, tz="UTC")
    half_lives = [float(x) for x in str(args.half_lives).split(",") if str(x).strip()]
    alphas = [float(x) for x in str(args.alphas).split(",") if str(x).strip()]
    max_adjusts = [float(x) for x in str(args.max_adjusts).split(",") if str(x).strip()]
    hr_mode_tokens = [str(x).strip().lower() for x in str(args.hr_modes).split(",") if str(x).strip()]
    hr_mode_map = {
        "threshold": "hit_surprise_threshold",
        "priority_mild": "hit_surprise_priority_mild",
        "priority": "hit_surprise_priority",
        "rank": "hit_surprise_rank",
        "size": "hit_surprise_size",
        "combined": "hit_surprise_combined",
        "priority_rank_50": "hit_surprise_priority_rank_50",
        "threshold_priority_rank_50": "hit_surprise_threshold_priority_rank_50",
        "downweight_only": "hit_surprise_downweight_only",
    }
    unknown_hr_modes = [mode for mode in hr_mode_tokens if mode not in hr_mode_map]
    if unknown_hr_modes:
        raise ValueError(f"Unknown --hr-modes values {unknown_hr_modes}; expected one of {sorted(hr_mode_map)}")
    hr_modes = [hr_mode_map[mode] for mode in hr_mode_tokens]
    top_thresholds = _parse_top_thresholds(str(args.top_slices))
    portfolio_max_concurrent = [
        int(x) for x in str(args.portfolio_max_concurrent).split(",") if str(x).strip()
    ]
    portfolio_max_per_side: list[int | None] = []
    for raw in str(args.portfolio_max_per_side).split(","):
        text = raw.strip().lower()
        if not text:
            continue
        portfolio_max_per_side.append(None if text in {"none", "null", "all"} else int(text))
    portfolio_grid = [
        PortfolioPolicyParams(
            max_concurrent_positions=int(max_pos),
            max_concurrent_per_side=max_side,
            max_concurrent_per_strategy=None,
            max_concurrent_per_symbol=1,
            max_new_entries_per_bar=int(args.portfolio_max_new_entries_per_bar),
            max_total_wallet_allocation_pct=0.75,
            global_threshold_floor=0.0,
            allocation_threshold_alpha=float(args.portfolio_allocation_threshold_alpha),
            allocation_threshold_power=float(args.portfolio_allocation_threshold_power),
            cooldown_hours_after_loss=0.0,
        )
        for max_pos in portfolio_max_concurrent
        for max_side in portfolio_max_per_side
        if max_side is None or int(max_side) <= int(max_pos)
    ]
    max_lookback_days = max(1.0, max(half_lives or [14.0]) * 4.0)
    surprise_train_start = holdout_start - pd.Timedelta(days=max_lookback_days)
    rows = _prepare_rows(
        args.candidates,
        min_rank=float(args.min_rank),
        rank_scope="per_strategy",
        regime_ev_rerank_admission=bool(args.regime_ev_rerank_admission),
        regime_ev_protected_admission_floor=(
            float(args.regime_ev_protected_admission_floor)
            if np.isfinite(float(args.regime_ev_protected_admission_floor))
            else None
        ),
        regime_ev_retained_surplus_frac=float(args.regime_ev_retained_surplus_frac),
    )
    optimisation_rows, holdout_rows, holdout_diag = _split_optimisation_holdout_rows(
        rows,
        holdout_start=args.holdout_start,
        holdout_end=args.holdout_end,
    )
    opt_ts = pd.to_datetime(optimisation_rows["timestamp"], utc=True, errors="coerce")
    optimisation_rows_all = optimisation_rows
    optimisation_rows = optimisation_rows.loc[opt_ts.ge(surprise_train_start) & opt_ts.lt(holdout_start)].copy()
    if optimisation_rows.empty:
        raise ValueError(
            f"No optimisation rows in recent surprise window {surprise_train_start} -> {holdout_start}"
        )
    holdout_diag["surprise_train_start"] = str(surprise_train_start)
    holdout_diag["surprise_train_end"] = str(holdout_start)
    holdout_diag["surprise_train_max_lookback_days"] = float(max_lookback_days)
    holdout_diag["optimisation_rows_all_before_recent_filter"] = int(len(optimisation_rows_all))
    holdout_diag["optimisation_rows_recent_surprise_window"] = int(len(optimisation_rows))
    regime_ev_artifact = default_regime_ev_calibration_artifact()
    regime_ev_handoff = default_regime_ev_feature_handoff()
    parent_summary = _load_parent_summary(args.parent_policy_summary)
    opt_bundles = _load_bundles(
        optimisation_rows,
        data_root=str(args.data_root),
        market_mode=str(args.market_mode),
        path_len=int(args.path_len),
        min_rows_per_strategy=5,
    )
    holdout_bundles = _load_bundles(
        holdout_rows,
        data_root=str(args.data_root),
        market_mode=str(args.market_mode),
        path_len=int(args.path_len),
        min_rows_per_strategy=5,
    )
    raw_opt_counts = optimisation_rows.groupby("strategy_id").size().to_dict()
    raw_holdout_counts = holdout_rows.groupby("strategy_id").size().to_dict()
    survival_rows: list[dict[str, Any]] = []
    for scope, bundles, raw_counts in (
        ("optimisation", opt_bundles, raw_opt_counts),
        ("holdout", holdout_bundles, raw_holdout_counts),
    ):
        for bundle in bundles:
            raw_n = int(raw_counts.get(str(bundle.strategy_id), 0))
            finite_n = int(len(bundle.rows))
            survival = float(finite_n / max(raw_n, 1))
            survival_rows.append(
                {
                    "scope": scope,
                    "strategy_id": str(bundle.strategy_id),
                    "raw_rows": raw_n,
                    "finite_rows": finite_n,
                    "survival": survival,
                }
            )
    survival_df = pd.DataFrame(survival_rows)
    survival_df.to_csv(args.out_dir / "path_survival.csv", index=False)
    weak = survival_df[survival_df["survival"].lt(float(args.min_survival))]
    if not weak.empty:
        raise RuntimeError(
            "Replay path survival below threshold; refusing biased threshold ablation: "
            + weak.to_json(orient="records")
        )
    opt_by_strategy = {str(bundle.strategy_id): bundle for bundle in opt_bundles}

    metric_rows: list[dict[str, Any]] = []
    surprise_rows: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, Any]] = []
    archetype_metric_rows: list[dict[str, Any]] = []
    portfolio_selection_frames: dict[tuple[Any, ...], list[pd.DataFrame]] = {}

    for hold_bundle in holdout_bundles:
        strategy_id = str(hold_bundle.strategy_id)
        side = "short" if strategy_id.startswith("short") else "long"
        if strategy_id not in parent_summary or strategy_id not in opt_by_strategy:
            continue
        params, size_power = _params_from_parent_summary_row(parent_summary[strategy_id])
        if np.isfinite(float(args.force_sl_abs_cap_pct)) and float(args.force_sl_abs_cap_pct) > 0.0:
            params["sl_abs_cap_pct"] = float(args.force_sl_abs_cap_pct)
        opt_bundle = opt_by_strategy[strategy_id]

        for top_label, base_threshold in top_thresholds.items():
            baseline_selected, baseline_adv = _simulate_selected_rows(
                hold_bundle.rows,
                hold_bundle.paths,
                rank_threshold=float(base_threshold),
                params=params,
                size_power=size_power,
                cost_pct=cost_pct,
            )
            metric_rows.append(
                _record_metrics(
                    side=side,
                    top_label=top_label,
                    mode="baseline",
                    selected=baseline_selected,
                    adv=baseline_adv,
                    config={"base_rank_threshold": float(base_threshold)},
                )
            )
            baseline_key = _portfolio_config_key(mode="baseline", top_label=top_label)
            portfolio_selection_frames.setdefault(baseline_key, []).append(
                _stamp_selection(
                    baseline_selected,
                    mode="baseline",
                    top_label=top_label,
                    base_threshold=float(base_threshold),
                )
            )

            fit_selected, _fit_adv = _simulate_selected_rows(
                opt_bundle.rows,
                opt_bundle.paths,
                rank_threshold=float(base_threshold),
                params=params,
                size_power=size_power,
                cost_pct=cost_pct,
            )
            for half_life in half_lives:
                surprise = _weighted_surprise(
                    fit_selected,
                    holdout_start=holdout_start,
                    half_life_days=float(half_life),
                    base_threshold=float(base_threshold),
                    min_effective_n=float(args.min_effective_n),
                )
                if not surprise.empty:
                    tmp = surprise.copy()
                    tmp["side"] = side
                    tmp["strategy_id"] = strategy_id
                    tmp["top_slice"] = top_label
                    surprise_rows.append(tmp)
                for alpha in alphas:
                    for max_adjust in max_adjusts:
                        quality = _surprise_quality_map(
                            surprise,
                            alpha=float(alpha),
                            max_adjust=float(max_adjust),
                        )
                        thresholds = _threshold_map(
                            surprise,
                            base_threshold=float(base_threshold),
                            alpha=float(alpha),
                            max_adjust=float(max_adjust),
                            min_threshold=max(0.0, float(base_threshold) - float(max_adjust)),
                            max_threshold=min(0.995, float(base_threshold) + float(max_adjust)),
                        )
                        dyn_selected, dyn_adv = _evaluate_dynamic_thresholds(
                            hold_bundle.rows,
                            hold_bundle.paths,
                            threshold_by_archetype=thresholds,
                            base_threshold=float(base_threshold),
                            params=params,
                            size_power=size_power,
                            cost_pct=cost_pct,
                        )
                        config = {
                            "base_rank_threshold": float(base_threshold),
                            "half_life_days": float(half_life),
                            "alpha": float(alpha),
                            "max_adjust": float(max_adjust),
                            "threshold_count": int(len(thresholds)),
                        }
                        if "hit_surprise_threshold" in hr_modes:
                            dyn_key = _portfolio_config_key(
                                mode="hit_surprise_threshold",
                                top_label=top_label,
                                half_life_days=float(half_life),
                                alpha=float(alpha),
                                max_adjust=float(max_adjust),
                            )
                            portfolio_selection_frames.setdefault(dyn_key, []).append(
                                _stamp_selection(
                                    dyn_selected,
                                    mode="hit_surprise_threshold",
                                    top_label=top_label,
                                    base_threshold=float(base_threshold),
                                    threshold_by_archetype=thresholds,
                                    half_life_days=float(half_life),
                                    alpha=float(alpha),
                                    max_adjust=float(max_adjust),
                                )
                            )
                            metric_rows.append(
                                _record_metrics(
                                    side=side,
                                    top_label=top_label,
                                    mode="hit_surprise_threshold",
                                    selected=dyn_selected,
                                    adv=dyn_adv,
                                    config=config,
                                )
                            )
                        if "hit_surprise_threshold_priority_rank_50" in hr_modes:
                            dyn_soft_selected = _apply_portfolio_hr_adjustments(
                                dyn_selected,
                                mode="hit_surprise_threshold_priority_rank_50",
                                quality_by_archetype=quality,
                                max_adjust=float(max_adjust),
                            )
                            dyn_soft_key = _portfolio_config_key(
                                mode="hit_surprise_threshold_priority_rank_50",
                                top_label=top_label,
                                half_life_days=float(half_life),
                                alpha=float(alpha),
                                max_adjust=float(max_adjust),
                            )
                            portfolio_selection_frames.setdefault(dyn_soft_key, []).append(
                                _stamp_selection(
                                    dyn_soft_selected,
                                    mode="hit_surprise_threshold_priority_rank_50",
                                    top_label=top_label,
                                    base_threshold=float(base_threshold),
                                    threshold_by_archetype=thresholds,
                                    half_life_days=float(half_life),
                                    alpha=float(alpha),
                                    max_adjust=float(max_adjust),
                                )
                            )
                            metric_rows.append(
                                _record_metrics(
                                    side=side,
                                    top_label=top_label,
                                    mode="hit_surprise_threshold_priority_rank_50",
                                    selected=dyn_soft_selected,
                                    adv=dyn_adv,
                                    config=config,
                                )
                            )
                        for soft_mode in [
                            mode
                            for mode in hr_modes
                            if mode
                            not in {
                                "hit_surprise_threshold",
                                "hit_surprise_threshold_priority_rank_50",
                            }
                        ]:
                            soft_selected = _apply_portfolio_hr_adjustments(
                                baseline_selected,
                                mode=soft_mode,
                                quality_by_archetype=quality,
                                max_adjust=float(max_adjust),
                            )
                            soft_key = _portfolio_config_key(
                                mode=soft_mode,
                                top_label=top_label,
                                half_life_days=float(half_life),
                                alpha=float(alpha),
                                max_adjust=float(max_adjust),
                            )
                            portfolio_selection_frames.setdefault(soft_key, []).append(
                                _stamp_selection(
                                    soft_selected,
                                    mode=soft_mode,
                                    top_label=top_label,
                                    base_threshold=float(base_threshold),
                                    half_life_days=float(half_life),
                                    alpha=float(alpha),
                                    max_adjust=float(max_adjust),
                                )
                            )
                            metric_rows.append(
                                _record_metrics(
                                    side=side,
                                    top_label=top_label,
                                    mode=soft_mode,
                                    selected=soft_selected,
                                    adv=baseline_adv,
                                    config=config,
                                )
                            )
                        for archetype, group in dyn_selected.groupby("policy_archetype", dropna=False):
                            archetype_metric_rows.append(
                                _record_metrics(
                                    side=side,
                                    top_label=top_label,
                                    mode="hit_surprise_threshold",
                                    selected=group,
                                    adv={
                                        "avg_pnl_notional": group["ret_net_notional"].mean(),
                                        "avg_pnl_bankroll": group["net_gain"].mean(),
                                        "avg_gross_return_per_trade": group["ret_gross_notional"].mean(),
                                        "hit_rate": (group["exit_reason"].astype(str) == "trailing").mean(),
                                        "pnl_positive_rate": (group["net_gain"] > 0.0).mean(),
                                        "full_sl_exit_rate": (group["exit_reason"].astype(str) == "full_sl").mean(),
                                        "timeout_exit_rate": (group["exit_reason"].astype(str) == "timeout").mean(),
                                    },
                                    config={**config, "policy_archetype": str(archetype)},
                                )
                            )
                        surprise_by_archetype = (
                            surprise.set_index(surprise["policy_archetype"].astype(str), drop=False)
                            if not surprise.empty and "policy_archetype" in surprise.columns
                            else pd.DataFrame()
                        )
                        for archetype, threshold in thresholds.items():
                            source_row = (
                                surprise_by_archetype.loc[str(archetype)]
                                if not surprise_by_archetype.empty
                                and str(archetype) in surprise_by_archetype.index
                                else {}
                            )
                            if isinstance(source_row, pd.DataFrame):
                                source_row = source_row.iloc[-1]
                            threshold_rows.append(
                                {
                                    "side": side,
                                    "strategy_id": strategy_id,
                                    "top_slice": top_label,
                                    "policy_archetype": archetype,
                                    **config,
                                    "adjusted_rank_threshold": float(threshold),
                                    "threshold_delta": float(threshold - float(base_threshold)),
                                    "actual_hit_rate": _safe_float(
                                        source_row.get("actual_hit_rate", np.nan)
                                    ),
                                    "expected_hit_rate": _safe_float(
                                        source_row.get("expected_hit_rate", np.nan)
                                    ),
                                    "hit_rate_delta": _safe_float(
                                        source_row.get("hit_rate_delta", np.nan)
                                    ),
                                    "hit_rate_surprise_z": _safe_float(
                                        source_row.get("hit_rate_surprise_z", np.nan)
                                    ),
                                    "support_confidence": _safe_float(
                                        source_row.get("support_confidence", np.nan)
                                    ),
                                    "n_eff": _safe_float(source_row.get("n_eff", np.nan)),
                                    "rows": _safe_float(source_row.get("rows", np.nan)),
                                }
                            )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.out_dir / "hit_surprise_threshold_metrics.csv", index=False)
    if surprise_rows:
        pd.concat(surprise_rows, ignore_index=True).to_csv(
            args.out_dir / "hit_surprise_by_archetype.csv",
            index=False,
        )
    threshold_df = pd.DataFrame(threshold_rows)
    if threshold_rows:
        threshold_df.to_csv(
            args.out_dir / "hit_surprise_thresholds_by_archetype.csv",
            index=False,
        )
    if archetype_metric_rows:
        pd.DataFrame(archetype_metric_rows).to_csv(
            args.out_dir / "hit_surprise_threshold_metrics_by_archetype.csv",
            index=False,
        )
    if args.save_preportfolio_selected and portfolio_selection_frames:
        selected_parts: list[pd.DataFrame] = []
        for key, frames in portfolio_selection_frames.items():
            top_label, mode, half_life, alpha, max_adjust = key
            for frame in frames:
                if frame.empty:
                    continue
                part = frame.copy()
                part["top_slice"] = str(top_label)
                part["mode"] = str(mode)
                part["half_life_days"] = np.nan if half_life is None else float(half_life)
                part["alpha"] = np.nan if alpha is None else float(alpha)
                part["max_adjust"] = np.nan if max_adjust is None else float(max_adjust)
                selected_parts.append(part)
        if selected_parts:
            pd.concat(selected_parts, ignore_index=True).to_parquet(
                args.out_dir / "hit_surprise_preportfolio_selected.parquet",
                index=False,
            )

    portfolio_metrics = pd.DataFrame()
    portfolio_best = pd.DataFrame()
    if portfolio_grid:
        portfolio_metrics, portfolio_accepted, portfolio_rejections = _portfolio_metrics_rows(
            portfolio_selection_frames,
            portfolio_grid=portfolio_grid,
            market_mode=str(args.market_mode),
        )
        portfolio_metrics.to_csv(args.out_dir / "hit_surprise_portfolio_metrics.csv", index=False)
        if not portfolio_rejections.empty:
            portfolio_rejections.to_csv(
                args.out_dir / "hit_surprise_portfolio_rejections.csv",
                index=False,
            )
        if args.portfolio_save_accepted and not portfolio_accepted.empty:
            portfolio_accepted.to_parquet(
                args.out_dir / "hit_surprise_portfolio_accepted.parquet",
                index=False,
            )
        if not portfolio_metrics.empty:
            scored_portfolio = portfolio_metrics.copy()
            scored_portfolio["portfolio_selection_objective"] = (
                100.0 * scored_portfolio["notional_weighted_net_return"].fillna(-1.0)
                + 10.0 * scored_portfolio["compounded_return"].fillna(-1.0)
                - 25.0 * scored_portfolio["full_sl_rate"].fillna(1.0)
                - 10.0 * scored_portfolio["timeout_rate"].fillna(1.0)
                - 5.0 * scored_portfolio["max_drawdown"].abs().fillna(1.0)
            )
            portfolio_best = (
                scored_portfolio.sort_values(
                    ["top_slice", "portfolio_selection_objective"],
                    ascending=[True, False],
                )
                .groupby("top_slice", as_index=False)
                .head(1)
            )
            portfolio_best.to_csv(args.out_dir / "hit_surprise_portfolio_best_by_top.csv", index=False)

    if not metrics.empty:
        scored = metrics.copy()
        scored["objective"] = (
            100.0 * scored["avg_pnl_notional"].fillna(-1.0)
            - 10.0 * scored["full_sl_exit_rate"].fillna(1.0)
            - 5.0 * scored["timeout_exit_rate"].fillna(1.0)
            + 0.01 * scored["n_trades"].fillna(0.0).clip(upper=500.0)
        )
        best = (
            scored.sort_values(["side", "top_slice", "objective"], ascending=[True, True, False])
            .groupby(["side", "top_slice"], as_index=False)
            .head(1)
        )
        best.to_csv(args.out_dir / "hit_surprise_threshold_best_by_side_top.csv", index=False)
    else:
        best = pd.DataFrame()

    manifest = {
        "generated_by": "ablate_s52_archetype_hit_surprise_thresholds",
        "candidates": str(args.candidates),
        "parent_policy_summary": str(args.parent_policy_summary),
        "holdout": holdout_diag,
        "round_trip_cost_pct": float(args.round_trip_cost_pct),
        "cost_pct_per_side": float(cost_pct),
        "regime_ev_calibration": {
            "enabled": regime_ev_artifact is not None,
            "policy_id": CALIBRATION_POLICY_ID,
            "artifact_path": str(regime_ev_artifact) if regime_ev_artifact is not None else "",
            "feature_handoff_path": str(regime_ev_handoff) if regime_ev_handoff is not None else "",
            "rank_score_col_after_calibration": "score_regime_calibrated",
            "rank_scope": "per_strategy",
            "applied_before_hit_surprise": regime_ev_artifact is not None,
            "used_for_admission_rank": bool(args.regime_ev_rerank_admission),
            "protected_admission_floor": (
                float(args.regime_ev_protected_admission_floor)
                if np.isfinite(float(args.regime_ev_protected_admission_floor))
                else None
            ),
            "retained_surplus_frac": float(args.regime_ev_retained_surplus_frac),
        },
        "forced_sl_abs_cap_pct": (
            float(args.force_sl_abs_cap_pct)
            if np.isfinite(float(args.force_sl_abs_cap_pct)) and float(args.force_sl_abs_cap_pct) > 0.0
            else None
        ),
        "top_thresholds": TOP_THRESHOLDS,
        "active_top_thresholds": top_thresholds,
        "half_lives": half_lives,
        "alphas": alphas,
        "max_adjusts": max_adjusts,
        "hr_modes": hr_modes,
        "min_effective_n": float(args.min_effective_n),
        "min_survival": float(args.min_survival),
        "path_survival": survival_rows,
        "outputs": {
            "metrics": str(args.out_dir / "hit_surprise_threshold_metrics.csv"),
            "best_by_side_top": str(args.out_dir / "hit_surprise_threshold_best_by_side_top.csv"),
            "surprise_by_archetype": str(args.out_dir / "hit_surprise_by_archetype.csv"),
            "thresholds_by_archetype": str(args.out_dir / "hit_surprise_thresholds_by_archetype.csv"),
            "metrics_by_archetype": str(args.out_dir / "hit_surprise_threshold_metrics_by_archetype.csv"),
            "path_survival": str(args.out_dir / "path_survival.csv"),
            "portfolio_metrics": str(args.out_dir / "hit_surprise_portfolio_metrics.csv"),
            "portfolio_best_by_top": str(args.out_dir / "hit_surprise_portfolio_best_by_top.csv"),
            "portfolio_rejections": str(args.out_dir / "hit_surprise_portfolio_rejections.csv"),
            "preportfolio_selected": str(args.out_dir / "hit_surprise_preportfolio_selected.parquet"),
            "portfolio_accepted": str(args.out_dir / "hit_surprise_portfolio_accepted.parquet"),
        },
        "portfolio_grid": [params.to_live_config() for params in portfolio_grid],
        "portfolio_allocation_pressure": {
            "allocation_threshold_alpha": float(args.portfolio_allocation_threshold_alpha),
            "allocation_threshold_power": float(args.portfolio_allocation_threshold_power),
            "applies_to_dynamic_threshold": True,
            "allocation_limit_pct": 0.75,
        },
    }
    promoted_outputs = _write_promoted_policy_artifacts(
        args.out_dir,
        thresholds=threshold_df,
        portfolio_metrics=portfolio_metrics,
        portfolio_best=portfolio_best,
        manifest=manifest,
    )
    manifest["outputs"].update(promoted_outputs)
    manifest["promoted_policy"] = {
        "policy_name": PROMOTED_POLICY_NAME,
        "top_slice": PROMOTED_TOP_SLICE,
        "base_rank_threshold": float(TOP_THRESHOLDS[PROMOTED_TOP_SLICE]),
        "half_life_days": float(PROMOTED_HALF_LIFE_DAYS),
        "alpha": float(PROMOTED_ALPHA),
        "max_adjust": float(PROMOTED_MAX_ADJUST),
        "max_concurrent_positions": int(PROMOTED_MAX_CONCURRENT_POSITIONS),
        "max_concurrent_per_side": PROMOTED_MAX_CONCURRENT_PER_SIDE,
        "max_new_entries_per_bar": int(PROMOTED_MAX_NEW_ENTRIES_PER_BAR),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"event": "hit_surprise_threshold_ablation_done", **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
