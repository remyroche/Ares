#!/usr/bin/env python3
"""Optuna-tuned gradual recent-metric head activation overlay.

This is a research ablation, not the active production path.  It keeps model
scores, labels, exit outcomes, and portfolio policy fixed, then learns a causal
per-head overlay from recent metrics only:

* raise per-row thresholds gradually when a head is degrading;
* shrink size before reducing per-head entry capacity;
* hard-stop only under extreme recent degradation;
* never lower the existing base threshold.

The Optuna objective is deliberately head-identity invariant.  It may reward
portfolio PnL, downside robustness, and diversification, but it must not encode
period-specific facts such as protecting one named head or promoting another.

At timestamp ``t`` the overlay uses only rows for the same head with
``timestamp < t - embargo``.  The optimization target is portfolio utility on
the pre-June reference period; June is reported as a forward replay.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    fit_hierarchical_ev_curves,
    normalise_candidate_table,
    replay_candidates,
)
from scripts import run_market_state_threshold_controller as mstc  # noqa: E402
from scripts.reliability_blend_rank_reference import apply_frozen_policy_rank_reference  # noqa: E402


DEFAULT_TRAIN_BROAD = Path(
    "data_perp/artifacts/reliability_blend_native_simple_policy_replay_20260624_floor070"
    "/simple_policy_optimiser/simple_policy_candidates_broad.parquet"
)
DEFAULT_TRAIN_DEPLOYABLE = Path(
    "data_perp/artifacts/reliability_blend_native_simple_policy_replay_20260624_floor070"
    "/simple_policy_optimiser/simple_policy_candidates.parquet"
)
DEFAULT_EVAL_CANDIDATES = Path(
    "data_perp/artifacts/reliability_blend_arm_A0_anchor_only_20260625_jun15_22"
    "/simple_policy_optimiser/simple_policy_candidates_broad.parquet"
)
DEFAULT_POLICY_MANIFEST = mstc.DEFAULT_POLICY_MANIFEST
DEFAULT_RANK_REFERENCE_RUN_ID = "reliability_blend_anchor_rank_reference_20260625_prejune"
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/recent_head_activation_optuna_20260625")
DEFAULT_RANK_CONTRACT = "short_boll_timestamp_rank"
HEADS = ("long_bars", "long_dist", "short_asset", "short_boll")
ACTION_METRIC_SUFFIXES = (
    "candidate_share_hard_stop",
    "candidate_mean_threshold_delta",
    "candidate_mean_size_multiplier",
    "candidate_share_capacity_reduced",
)


@dataclass(frozen=True)
class RecentHeadParams:
    lookback_hours: float
    embargo_hours: float
    min_samples: int
    shrink_samples: float
    decay_halflife_hours: float
    health_clip: float
    net_weight: float
    hr_weight: float
    weighted_hr_weight: float
    ic_weight: float
    full_sl_weight: float
    cost_drag_weight: float
    worst_return_weight: float
    weighted_hr_power: float
    head_control_strength: float
    threshold_start: float
    threshold_scale: float
    threshold_power: float
    max_threshold_shift: float
    size_start: float
    size_scale: float
    min_size_multiplier: float
    cap_start: float
    cap_scale: float
    hard_stop_health: float
    hard_stop_threshold: float
    objective_q_low_weight: float
    objective_q_mid_deterioration_weight: float
    objective_defensive_success_weight: float
    objective_full_sl_penalty: float
    objective_q_low: float
    objective_q_mid: float
    objective_short_horizon_hours: int
    objective_long_horizon_hours: int
    objective_hard_stop_start: float
    objective_hard_stop_weight: float
    objective_head_action_weight: float
    objective_head_hard_stop_start: float
    objective_head_threshold_start: float
    objective_head_size_floor: float
    objective_head_capacity_start: float
    objective_max_head_trade_share: float
    objective_global_balance_weight: float
    objective_weekly_balance_weight: float


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _parse_heads(value: str) -> set[str]:
    if not value:
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def _load_candidates(path: Path) -> pd.DataFrame:
    df = mstc._load_candidates(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for col in ("head", "strategy_id", "symbol"):
        df[col] = df[col].astype(str)
    return normalise_candidate_table(df)


def _period_payload(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty or "timestamp" not in df.columns:
        return {
            "row_count": int(len(df)),
            "timestamp_min": None,
            "timestamp_max": None,
            "timestamp_count": 0,
            "head_counts": {},
        }
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    finite = ts.notna()
    head_counts: dict[str, int] = {}
    if "head" in df.columns:
        head_counts = {
            str(head): int(count)
            for head, count in df.loc[finite, "head"].astype(str).value_counts(sort=False).sort_index().items()
        }
    return {
        "row_count": int(len(df)),
        "timestamp_min": ts.loc[finite].min().isoformat() if bool(finite.any()) else None,
        "timestamp_max": ts.loc[finite].max().isoformat() if bool(finite.any()) else None,
        "timestamp_count": int(ts.loc[finite].nunique()) if bool(finite.any()) else 0,
        "head_counts": head_counts,
    }


def _chronological_selection_split(
    df: pd.DataFrame,
    *,
    validation_frac: float,
    min_validation_timestamps: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split by complete timestamps for Optuna selection.

    The earlier slice defines long-run per-head baselines. The later slice is
    the pre-June replay objective. During that replay, earlier matured outcomes
    from the same validation slice may enter recent-health state through the
    normal causal outcome-availability path, but no future validation rows are
    visible to earlier timestamps.
    """

    if df.empty or "timestamp" not in df.columns:
        return df.copy(), df.copy(), {
            "mode": "full_reference_replay_empty_or_missing_timestamp",
            "validation_frac": float(validation_frac),
            "min_validation_timestamps": int(min_validation_timestamps),
            "cutoff_timestamp": None,
            "reference": _period_payload(df),
            "objective": _period_payload(df),
        }
    work = df.copy()
    ts = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    finite_ts = pd.Index(ts.dropna().drop_duplicates().sort_values())
    if float(validation_frac) <= 0.0 or len(finite_ts) < 2:
        return work, work.copy(), {
            "mode": "full_reference_replay",
            "validation_frac": float(validation_frac),
            "min_validation_timestamps": int(min_validation_timestamps),
            "cutoff_timestamp": None,
            "reference": _period_payload(work),
            "objective": _period_payload(work),
        }

    requested = int(np.ceil(len(finite_ts) * float(validation_frac)))
    min_requested = max(int(min_validation_timestamps), 1)
    valid_count = min(max(requested, min_requested), len(finite_ts) - 1)
    cutoff = finite_ts[-valid_count]
    reference_mask = ts.notna() & (ts < cutoff)
    objective_mask = ts.notna() & (ts >= cutoff)
    if not bool(reference_mask.any()) or not bool(objective_mask.any()):
        return work, work.copy(), {
            "mode": "full_reference_replay_split_fallback",
            "validation_frac": float(validation_frac),
            "min_validation_timestamps": int(min_validation_timestamps),
            "cutoff_timestamp": None,
            "reference": _period_payload(work),
            "objective": _period_payload(work),
        }

    reference = work.loc[reference_mask].copy()
    objective = work.loc[objective_mask].copy()
    return reference, objective, {
        "mode": "chronological_holdout",
        "validation_frac": float(validation_frac),
        "min_validation_timestamps": int(min_validation_timestamps),
        "cutoff_timestamp": pd.Timestamp(cutoff).isoformat(),
        "reference": _period_payload(reference),
        "objective": _period_payload(objective),
        "reference_max_timestamp": pd.to_datetime(reference["timestamp"], utc=True, errors="coerce").max().isoformat(),
        "objective_min_timestamp": pd.to_datetime(objective["timestamp"], utc=True, errors="coerce").min().isoformat(),
        "complete_timestamp_split": True,
    }


def _selection_ev_reference(
    train_deployable: pd.DataFrame,
    selection_split: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Use only pre-selection-objective rows to fit Optuna replay EV curves."""

    if (
        train_deployable.empty
        or "timestamp" not in train_deployable.columns
        or str(selection_split.get("mode") or "") != "chronological_holdout"
    ):
        ref = train_deployable.copy()
        return ref, {
            "mode": "full_train_deployable_ev_reference",
            "objective_min_timestamp": None,
            "reference": _period_payload(ref),
        }

    objective_min = pd.to_datetime(selection_split.get("objective_min_timestamp"), utc=True, errors="coerce")
    if pd.isna(objective_min):
        ref = train_deployable.copy()
        return ref, {
            "mode": "full_train_deployable_ev_reference_missing_objective_min",
            "objective_min_timestamp": None,
            "reference": _period_payload(ref),
        }

    ts = pd.to_datetime(train_deployable["timestamp"], utc=True, errors="coerce")
    ref = train_deployable.loc[ts.notna() & (ts < objective_min)].copy()
    if ref.empty:
        fallback = train_deployable.copy()
        return fallback, {
            "mode": "full_train_deployable_ev_reference_empty_pre_objective_fallback",
            "objective_min_timestamp": objective_min.isoformat(),
            "reference": _period_payload(fallback),
        }

    return ref, {
        "mode": "chronological_pre_selection_objective_ev_reference",
        "objective_min_timestamp": objective_min.isoformat(),
        "reference_max_timestamp": pd.to_datetime(ref["timestamp"], utc=True, errors="coerce").max().isoformat(),
        "complete_timestamp_split": True,
        "reference": _period_payload(ref),
    }


def _rank_scope(rank_contract: str) -> str:
    if rank_contract == "anchor_global_policy_rank_reference":
        return "global_over_time"
    if rank_contract == "short_boll_timestamp_rank":
        return "within_timestamp"
    raise ValueError(f"Unknown rank contract: {rank_contract}")


def _ranked_candidate_label(rank_contract: str) -> str:
    if rank_contract == "anchor_global_policy_rank_reference":
        return "global_rank"
    if rank_contract == "short_boll_timestamp_rank":
        return "t1_timestamp_rank"
    raise ValueError(f"Unknown rank contract: {rank_contract}")


def _apply_ablation_rank_contract(
    df: pd.DataFrame,
    *,
    rank_contract: str,
    data_root: Path,
    rank_reference_run_id: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if rank_contract == "short_boll_timestamp_rank":
        out = mstc._apply_rank_contract(df, "short_boll_timestamp_rank")
        rank_col = next(
            (
                col
                for col in ("policy_rank_pct", "normalized_rank_score", "strategy_rank_pct", "rank_pct")
                if col in out.columns
            ),
            None,
        )
        missing = 0
        ranked = int(len(out))
        if rank_col is not None:
            rank_values = pd.to_numeric(out[rank_col], errors="coerce")
            missing = int((~np.isfinite(rank_values.to_numpy(dtype=float))).sum())
            ranked = int(np.isfinite(rank_values.to_numpy(dtype=float)).sum())
        return normalise_candidate_table(out), {
            "rank_contract": "short_boll_timestamp_rank",
            "rank_scope": "within_timestamp",
            "rank_reference_run_id": None,
            "rank_source": "input_policy_rank_with_short_boll_head_timestamp_repair",
            "ranked_rows": ranked,
            "missing_rank_rows": missing,
            "window_rank_debug_used": False,
        }
    if rank_contract != "anchor_global_policy_rank_reference":
        raise ValueError(f"Unknown rank contract: {rank_contract}")
    out, diag = apply_frozen_policy_rank_reference(
        df,
        data_root=data_root,
        run_id=rank_reference_run_id,
        score_col="calibrated_score",
        allow_window_rank_debug=False,
    )
    out["rank_contract_source"] = "anchor_global_policy_rank_reference"
    diag["rank_contract"] = "anchor_global_policy_rank_reference"
    diag["rank_scope"] = "global_over_time"
    return normalise_candidate_table(out), diag


def _exit_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if "exit_timestamp" in out.columns:
        available_ts = pd.to_datetime(out["exit_timestamp"], utc=True, errors="coerce")
        out["_outcome_available_timestamp"] = available_ts.fillna(out["timestamp"])
    else:
        out["_outcome_available_timestamp"] = out["timestamp"]
    ret = pd.to_numeric(out.get("net_return"), errors="coerce").fillna(0.0)
    gross = pd.to_numeric(out.get("gross_return", ret), errors="coerce").fillna(0.0)
    reason = out.get("simple_policy_exit_reason", pd.Series("", index=out.index)).astype(str).str.lower()
    out["_ret"] = ret
    out["_gross"] = gross
    out["_win"] = (ret > 0.0).astype(float)
    out["_full_sl"] = reason.isin(["sl", "full_sl", "stop", "stop_loss"]).astype(float)
    out["_rank"] = pd.to_numeric(out.get("normalized_rank_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["_cost"] = gross - ret
    out["_abs_gross"] = gross.abs()
    return out


def _weighted_mean(values: np.ndarray, weights: np.ndarray, default: float = 0.0) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
    if int(mask.sum()) == 0:
        return float(default)
    denom = float(w[mask].sum())
    if denom <= 1e-12:
        return float(default)
    return float(np.dot(v[mask], w[mask]) / denom)


def _corr_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 5:
        return 0.0
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _rank01(values: list[float], *, higher_is_better: bool = True) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, 0.5, dtype=float)
    mask = np.isfinite(arr)
    if int(mask.sum()) <= 1:
        return out
    order = np.argsort(arr[mask], kind="mergesort")
    ranks = np.empty(int(mask.sum()), dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, int(mask.sum()))
    if not higher_is_better:
        ranks = 1.0 - ranks
    out[mask] = ranks
    return out


def _add_head_control_scales(
    baselines: dict[str, dict[str, float]],
    *,
    strength: float,
) -> dict[str, dict[str, float]]:
    if not baselines:
        return baselines
    heads = list(baselines.keys())
    quality = (
        _rank01([baselines[h].get("net_mean", 0.0) for h in heads], higher_is_better=True)
        + _rank01([baselines[h].get("weighted_hr", 0.0) for h in heads], higher_is_better=True)
        + _rank01([baselines[h].get("rank_ic", 0.0) for h in heads], higher_is_better=True)
        + _rank01([baselines[h].get("q10_ret", 0.0) for h in heads], higher_is_better=True)
        + _rank01([baselines[h].get("full_sl", 0.0) for h in heads], higher_is_better=False)
        + _rank01([baselines[h].get("cost_drag", 0.0) for h in heads], higher_is_better=False)
    ) / 6.0
    scale = np.clip(1.0 + float(strength) * (0.5 - quality), 0.35, 2.25)
    out = {head: dict(values) for head, values in baselines.items()}
    for head, q, s in zip(heads, quality, scale):
        out[head]["head_control_quality"] = float(q)
        out[head]["head_control_scale"] = float(s)
    return out


def _baseline_by_head(
    df: pd.DataFrame,
    weighted_hr_power: float,
    *,
    head_control_strength: float = 0.0,
) -> dict[str, dict[str, float]]:
    work = _exit_flags(df)
    baselines: dict[str, dict[str, float]] = {}
    for head, g in work.groupby("head", sort=True):
        rank = g["_rank"].to_numpy(dtype=float)
        w = np.power(np.clip(rank, 0.0, 1.0), float(weighted_hr_power))
        gross_abs = float(g["_abs_gross"].sum())
        baselines[str(head)] = {
            "net_mean": float(g["_ret"].mean()) if len(g) else 0.0,
            "hr": float(g["_win"].mean()) if len(g) else 0.0,
            "weighted_hr": _weighted_mean(g["_win"].to_numpy(dtype=float), w, default=0.0),
            "full_sl": float(g["_full_sl"].mean()) if len(g) else 0.0,
            "cost_drag": float(g["_cost"].sum() / max(gross_abs, 1e-9)),
            "rank_ic": _corr_safe(rank, g["_ret"].to_numpy(dtype=float)),
            "q10_ret": float(g["_ret"].quantile(0.10)) if len(g) else 0.0,
        }
    return _add_head_control_scales(baselines, strength=float(head_control_strength))


def _recent_health_schedule(
    reference: pd.DataFrame,
    timestamps: pd.Series,
    *,
    params: RecentHeadParams,
    baselines: dict[str, dict[str, float]],
) -> pd.DataFrame:
    ref = _exit_flags(reference)
    ref = ref.sort_values(["head", "_outcome_available_timestamp", "timestamp"], kind="mergesort").reset_index(drop=True)
    target_ts = pd.to_datetime(pd.Series(timestamps).dropna().drop_duplicates(), utc=True).sort_values()
    rows: list[dict[str, Any]] = []
    half_life = max(float(params.decay_halflife_hours), 1e-6)
    lookback = pd.Timedelta(hours=float(params.lookback_hours))
    embargo = pd.Timedelta(hours=float(params.embargo_hours))
    for head in HEADS:
        h = ref.loc[ref["head"].eq(head)].copy()
        base = baselines.get(head, {})
        for ts in target_ts:
            end = ts - embargo
            start = end - lookback
            available_ts = pd.to_datetime(h["_outcome_available_timestamp"], utc=True, errors="coerce")
            g = h.loc[(available_ts >= start) & (available_ts < end)]
            if g.empty:
                rows.append(
                    {
                        "timestamp": ts,
                        "head": head,
                        "recent_rows": 0,
                        "health_raw": 0.0,
                        "health": 0.0,
                        "badness": 0.0,
                        "recent_net_mean": np.nan,
                        "recent_hr": np.nan,
                        "recent_weighted_hr": np.nan,
                        "recent_full_sl": np.nan,
                        "recent_cost_drag": np.nan,
                        "recent_rank_ic": np.nan,
                        "recent_q10_ret": np.nan,
                        "head_control_quality": float(base.get("head_control_quality", 0.5)),
                        "head_control_scale": float(base.get("head_control_scale", 1.0)),
                    }
                )
                continue
            g_available_ts = pd.to_datetime(g["_outcome_available_timestamp"], utc=True, errors="coerce")
            age_hours = (end - g_available_ts).dt.total_seconds().to_numpy(dtype=float) / 3600.0
            time_w = np.power(0.5, np.maximum(age_hours, 0.0) / half_life)
            rank = g["_rank"].to_numpy(dtype=float)
            conf_w = np.power(np.clip(rank, 0.0, 1.0), float(params.weighted_hr_power))
            w = time_w * np.maximum(conf_w, 1e-6)
            n_eff = float(np.square(w.sum()) / max(float(np.dot(w, w)), 1e-12))
            shrink = float(np.clip(n_eff / max(float(params.shrink_samples), 1.0), 0.0, 1.0))
            net_mean = _weighted_mean(g["_ret"].to_numpy(dtype=float), w, default=float(base.get("net_mean", 0.0)))
            hr = _weighted_mean(g["_win"].to_numpy(dtype=float), w, default=float(base.get("hr", 0.0)))
            weighted_hr = _weighted_mean(g["_win"].to_numpy(dtype=float), w * conf_w, default=float(base.get("weighted_hr", 0.0)))
            full_sl = _weighted_mean(g["_full_sl"].to_numpy(dtype=float), w, default=float(base.get("full_sl", 0.0)))
            gross_abs = _weighted_mean(g["_abs_gross"].to_numpy(dtype=float), w, default=0.0)
            cost_drag = _weighted_mean(g["_cost"].to_numpy(dtype=float), w, default=0.0) / max(gross_abs, 1e-9)
            rank_ic = _corr_safe(rank, g["_ret"].to_numpy(dtype=float))
            q10_ret = float(g["_ret"].quantile(0.10))
            raw = (
                float(params.net_weight) * (net_mean - float(base.get("net_mean", 0.0))) / 0.01
                + float(params.hr_weight) * (hr - float(base.get("hr", 0.0)))
                + float(params.weighted_hr_weight) * (weighted_hr - float(base.get("weighted_hr", 0.0)))
                + float(params.ic_weight) * (rank_ic - float(base.get("rank_ic", 0.0)))
                - float(params.full_sl_weight) * (full_sl - float(base.get("full_sl", 0.0)))
                - float(params.cost_drag_weight) * (cost_drag - float(base.get("cost_drag", 0.0)))
                + float(params.worst_return_weight) * (q10_ret - float(base.get("q10_ret", 0.0))) / 0.01
            )
            raw *= shrink
            if int(len(g)) < int(params.min_samples):
                raw = 0.0
            health = float(np.clip(raw, -float(params.health_clip), float(params.health_clip)))
            rows.append(
                {
                    "timestamp": ts,
                    "head": head,
                    "recent_rows": int(len(g)),
                    "effective_rows": n_eff,
                    "health_raw": raw,
                    "health": health,
                    "badness": max(0.0, -health),
                    "recent_net_mean": net_mean,
                    "recent_hr": hr,
                    "recent_weighted_hr": weighted_hr,
                    "recent_full_sl": full_sl,
                    "recent_cost_drag": cost_drag,
                    "recent_rank_ic": rank_ic,
                    "recent_q10_ret": q10_ret,
                    "head_control_quality": float(base.get("head_control_quality", 0.5)),
                    "head_control_scale": float(base.get("head_control_scale", 1.0)),
                }
            )
    return pd.DataFrame(rows)


def _apply_overlay(candidates: pd.DataFrame, schedule: pd.DataFrame, params: RecentHeadParams) -> pd.DataFrame:
    out = candidates.copy()
    sched = schedule[["timestamp", "head", "health_raw", "health", "badness", "head_control_scale"]].copy()
    sched["timestamp"] = pd.to_datetime(sched["timestamp"], utc=True, errors="coerce")
    out = out.merge(sched, on=["timestamp", "head"], how="left", validate="many_to_one")
    out["health_raw"] = pd.to_numeric(out["health_raw"], errors="coerce").fillna(0.0)
    out["health"] = pd.to_numeric(out["health"], errors="coerce").fillna(0.0)
    raw_bad = pd.to_numeric(out["badness"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    control_scale = pd.to_numeric(out.get("head_control_scale"), errors="coerce").fillna(1.0).to_numpy(dtype=float)
    bad = np.clip(raw_bad * np.clip(control_scale, 0.35, 2.25), 0.0, None)
    thresh_pressure = np.maximum(bad - float(params.threshold_start), 0.0)
    threshold_delta = np.minimum(
        float(params.max_threshold_shift),
        float(params.threshold_scale) * np.power(thresh_pressure, float(params.threshold_power)),
    )
    base = pd.to_numeric(out["base_strategy_threshold"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    # Hard stops use the unclipped health estimate, while gradual threshold/size/cap
    # controls use clipped badness. This keeps hard stops reserved for truly
    # extreme recent degradation instead of firing whenever health hits the clip.
    health_raw = out["health_raw"].to_numpy(dtype=float)
    hard = health_raw <= float(params.hard_stop_health)
    adjusted = np.where(hard, float(params.hard_stop_threshold), np.clip(base + threshold_delta, base, 1.01))
    out["recent_head_base_threshold"] = base
    out["recent_head_threshold_delta"] = adjusted - base
    out["base_strategy_threshold"] = adjusted
    out["deployment_rank_threshold"] = adjusted
    size_pressure = np.maximum(bad - float(params.size_start), 0.0)
    size_mult = np.clip(
        1.0 - float(params.size_scale) * size_pressure,
        float(params.min_size_multiplier),
        1.0,
    )
    size_mult = np.where(hard, 0.0, size_mult)
    out["portfolio_size_multiplier"] = size_mult
    cap_pressure = np.maximum(bad - float(params.cap_start), 0.0)
    strat_cap = np.ceil(np.clip(2.0 * (1.0 - float(params.cap_scale) * cap_pressure), 1.0, 2.0))
    strat_cap = np.where(hard, 0.0, strat_cap)
    out["portfolio_max_new_entries_per_strategy_per_bar"] = strat_cap
    out["recent_head_hard_stop"] = hard
    return normalise_candidate_table(out.drop(columns=["badness"]))


def _accepted_trades(candidates: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    return mstc._accepted_trades(candidates, decisions)


def _replay(
    candidates: pd.DataFrame,
    *,
    params: Any,
    ev_curve: dict[str, Any],
    arm: str,
    market_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    decisions, equity, metrics = replay_candidates(
        candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )
    accepted = _accepted_trades(candidates, decisions)
    row = mstc._metrics_row(arm, metrics, accepted, None)
    row.update(_rolling_downside_metrics(accepted))
    row.update(_overlay_metrics(candidates, accepted))
    summary = pd.DataFrame([row])
    return accepted, decisions, summary


def _rolling_window_pnl_quantile(accepted: pd.DataFrame, *, hours: int, quantile: float) -> float:
    if accepted.empty or "timestamp" not in accepted.columns or "net_pnl" not in accepted.columns:
        return 0.0
    work = accepted[["timestamp", "net_pnl"]].copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work["net_pnl"] = pd.to_numeric(work["net_pnl"], errors="coerce").fillna(0.0)
    work = work.dropna(subset=["timestamp"]).sort_values("timestamp", kind="mergesort")
    if work.empty:
        return 0.0
    by_ts = work.groupby("timestamp", sort=True)["net_pnl"].sum()
    ts_ns = by_ts.index.view("int64")
    values = by_ts.to_numpy(dtype=float)
    csum = np.concatenate([[0.0], np.cumsum(values)])
    horizon_ns = int(pd.Timedelta(hours=int(hours)).value)
    starts = np.searchsorted(ts_ns, ts_ns - horizon_ns, side="right")
    rolling = csum[np.arange(1, len(values) + 1)] - csum[starts]
    if rolling.size == 0:
        return 0.0
    return float(np.nanquantile(rolling, float(quantile)))


def _rolling_downside_metrics(accepted: pd.DataFrame) -> dict[str, float]:
    q5_120 = _rolling_window_pnl_quantile(accepted, hours=120, quantile=0.05)
    q5_48 = _rolling_window_pnl_quantile(accepted, hours=48, quantile=0.05)
    q15_120 = _rolling_window_pnl_quantile(accepted, hours=120, quantile=0.15)
    q15_48 = _rolling_window_pnl_quantile(accepted, hours=48, quantile=0.15)
    return {
        "q5_120h_net_pnl": q5_120,
        "q5_48h_net_pnl": q5_48,
        "q15_120h_net_pnl": q15_120,
        "q15_48h_net_pnl": q15_48,
        "robust_downside_pnl": q5_120 + q5_48 + q15_120 + q15_48,
    }


def _overlay_metrics(candidates: pd.DataFrame, accepted: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not candidates.empty and "recent_head_threshold_delta" in candidates.columns:
        delta = pd.to_numeric(candidates["recent_head_threshold_delta"], errors="coerce").fillna(0.0)
        size = pd.to_numeric(candidates.get("portfolio_size_multiplier"), errors="coerce").fillna(1.0)
        cap = pd.to_numeric(candidates.get("portfolio_max_new_entries_per_strategy_per_bar"), errors="coerce").fillna(2.0)
        hard = candidates.get("recent_head_hard_stop", pd.Series(False, index=candidates.index)).astype(bool)
        candidate_heads = (
            sorted(candidates.get("head", pd.Series([], dtype=str)).dropna().astype(str).unique().tolist())
            if "head" in candidates.columns
            else []
        )
        metrics.update(
            {
                "candidate_mean_threshold_delta": float(delta.mean()),
                "candidate_p75_threshold_delta": float(delta.quantile(0.75)),
                "candidate_max_threshold_delta": float(delta.max()),
                "candidate_share_threshold_raised": float((delta > 1e-9).mean()),
                "candidate_mean_size_multiplier": float(size.mean()),
                "candidate_share_size_shrunk": float((size < 1.0 - 1e-9).mean()),
                "candidate_mean_strategy_cap": float(cap.mean()),
                "candidate_share_capacity_reduced": float((cap < 2.0 - 1e-9).mean()),
                "candidate_share_hard_stop": float(hard.mean()),
            }
        )
        for head in candidate_heads:
            mask = candidates.get("head", pd.Series("", index=candidates.index)).astype(str).eq(head)
            if bool(mask.any()):
                metrics.update(
                    {
                        f"{head}_candidate_mean_threshold_delta": float(delta.loc[mask].mean()),
                        f"{head}_candidate_share_threshold_raised": float((delta.loc[mask] > 1e-9).mean()),
                        f"{head}_candidate_mean_size_multiplier": float(size.loc[mask].mean()),
                        f"{head}_candidate_share_size_shrunk": float((size.loc[mask] < 1.0 - 1e-9).mean()),
                        f"{head}_candidate_mean_strategy_cap": float(cap.loc[mask].mean()),
                        f"{head}_candidate_share_capacity_reduced": float((cap.loc[mask] < 2.0 - 1e-9).mean()),
                        f"{head}_candidate_share_hard_stop": float(hard.loc[mask].mean()),
                    }
                )
        # Keep current-head columns stable for old reports/tests, but do not
        # restrict objective penalties to this fixed list.
        for head in HEADS:
            if head in candidate_heads:
                continue
            metrics.update(
                {
                    f"{head}_candidate_mean_threshold_delta": 0.0,
                    f"{head}_candidate_share_threshold_raised": 0.0,
                    f"{head}_candidate_mean_size_multiplier": 1.0,
                    f"{head}_candidate_share_size_shrunk": 0.0,
                    f"{head}_candidate_mean_strategy_cap": 2.0,
                    f"{head}_candidate_share_capacity_reduced": 0.0,
                    f"{head}_candidate_share_hard_stop": 0.0,
                }
            )
    else:
        metrics.update(
            {
                "candidate_mean_threshold_delta": 0.0,
                "candidate_p75_threshold_delta": 0.0,
                "candidate_max_threshold_delta": 0.0,
                "candidate_share_threshold_raised": 0.0,
                "candidate_mean_size_multiplier": 1.0,
                "candidate_share_size_shrunk": 0.0,
                "candidate_mean_strategy_cap": 2.0,
                "candidate_share_capacity_reduced": 0.0,
                "candidate_share_hard_stop": 0.0,
            }
        )
        for head in HEADS:
            metrics.update(
                {
                    f"{head}_candidate_mean_threshold_delta": 0.0,
                    f"{head}_candidate_share_threshold_raised": 0.0,
                    f"{head}_candidate_mean_size_multiplier": 1.0,
                    f"{head}_candidate_share_size_shrunk": 0.0,
                    f"{head}_candidate_mean_strategy_cap": 2.0,
                    f"{head}_candidate_share_capacity_reduced": 0.0,
                    f"{head}_candidate_share_hard_stop": 0.0,
                }
            )
    if not accepted.empty and {"base_threshold", "dynamic_threshold"}.issubset(accepted.columns):
        accepted_delta = (
            pd.to_numeric(accepted["dynamic_threshold"], errors="coerce")
            - pd.to_numeric(accepted["base_threshold"], errors="coerce")
        ).fillna(0.0)
        metrics.update(
            {
                "mean_threshold_delta": float(accepted_delta.mean()),
                "p75_threshold_delta": float(accepted_delta.quantile(0.75)),
                "max_threshold_delta": float(accepted_delta.max()),
                "share_threshold_raised": float((accepted_delta > 1e-9).mean()),
            }
        )
    return metrics


def _weekly(accepted: pd.DataFrame, arm: str) -> pd.DataFrame:
    if accepted.empty:
        return pd.DataFrame()
    work = accepted.copy()
    ts = pd.to_datetime(work["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    work["week"] = ts.dt.to_period("W").dt.start_time.astype(str)
    rows = []
    for keys, g in work.groupby(["week", "head"], sort=True):
        week, head = keys
        rec = mstc._by_head(arm, g).iloc[0].to_dict()
        rec["week"] = week
        rec["head"] = head
        rows.append(rec)
    return pd.DataFrame(rows)


def _head_stats(accepted: pd.DataFrame) -> dict[str, dict[str, float]]:
    if accepted.empty or "head" not in accepted.columns:
        return {}
    rows: dict[str, dict[str, float]] = {}
    reason = accepted.get("simple_policy_exit_reason", pd.Series("", index=accepted.index)).astype(str).str.lower()
    work = accepted.copy()
    work["_full_sl"] = reason.isin(["sl", "full_sl", "stop", "stop_loss"]).astype(float)
    work["_win"] = (pd.to_numeric(work.get("net_return"), errors="coerce").fillna(0.0) > 0.0).astype(float)
    for head, g in work.groupby("head", sort=True):
        net = float(pd.to_numeric(g.get("net_pnl"), errors="coerce").fillna(0.0).sum())
        trades = int(len(g))
        rows[str(head)] = {
            "trade_count": float(trades),
            "net_pnl": net,
            "mean_net_pnl": float(net / max(trades, 1)),
            "win_rate": float(g["_win"].mean()) if trades else 0.0,
            "full_sl_rate": float(g["_full_sl"].mean()) if trades else 0.0,
        }
    return rows


def _accepted_key_series(df: pd.DataFrame) -> pd.Series:
    if "candidate_index" in df.columns:
        return df["candidate_index"].astype(str)
    cols = [col for col in ("timestamp", "symbol", "side", "strategy_id", "head") if col in df.columns]
    if not cols:
        return pd.Series(np.arange(len(df), dtype=int), index=df.index).astype(str)
    parts = []
    for col in cols:
        if col == "timestamp":
            value = pd.to_datetime(df[col], utc=True, errors="coerce").astype(str)
        else:
            value = df[col].astype(str)
        parts.append(value)
    out = parts[0]
    for value in parts[1:]:
        out = out.str.cat(value, sep="|")
    return out


def _defensive_success_metrics(baseline_accepted: pd.DataFrame, overlay_accepted: pd.DataFrame) -> dict[str, float]:
    if baseline_accepted.empty:
        return {
            "removed_trade_count": 0.0,
            "loss_avoided_pnl": 0.0,
            "winner_pnl_sacrificed": 0.0,
            "defensive_success_pnl": 0.0,
        }
    baseline = baseline_accepted.copy()
    baseline["_key"] = _accepted_key_series(baseline)
    overlay_keys = set(_accepted_key_series(overlay_accepted)) if not overlay_accepted.empty else set()
    removed = baseline.loc[~baseline["_key"].isin(overlay_keys)].copy()
    if removed.empty:
        return {
            "removed_trade_count": 0.0,
            "loss_avoided_pnl": 0.0,
            "winner_pnl_sacrificed": 0.0,
            "defensive_success_pnl": 0.0,
        }
    if "net_pnl" in removed.columns:
        pnl_source = removed["net_pnl"]
    elif "net_return" in removed.columns:
        pnl_source = removed["net_return"]
    else:
        pnl_source = pd.Series(0.0, index=removed.index)
    pnl = pd.to_numeric(pnl_source, errors="coerce").fillna(0.0)
    loss_avoided = float((-pnl.clip(upper=0.0)).sum())
    winner_sacrificed = float(pnl.clip(lower=0.0).sum())
    return {
        "removed_trade_count": float(len(removed)),
        "loss_avoided_pnl": loss_avoided,
        "winner_pnl_sacrificed": winner_sacrificed,
        "defensive_success_pnl": loss_avoided - winner_sacrificed,
    }


def _annotate_summary(summary: pd.DataFrame, metrics: dict[str, float]) -> pd.DataFrame:
    out = summary.copy()
    for key, value in metrics.items():
        out.loc[out.index[0], key] = float(value)
    return out


def _head_activity_concentration_penalty(accepted: pd.DataFrame, *, max_head_share: float) -> tuple[float, float]:
    """Head-agnostic concentration penalty.

    This intentionally does not reference head names or per-head PnL. Net PnL
    is already the dominant objective term; this term only prevents the overlay
    from learning brittle monoculture solutions on the fit period.
    """
    if accepted.empty or "head" not in accepted.columns:
        return 0.0, 0.0
    work = accepted.copy()
    work["head"] = work["head"].astype(str)
    observed_head_count = max(int(work["head"].nunique(dropna=True)), 1)
    cap = float(np.clip(max_head_share, 1.0 / observed_head_count, 1.0))

    def _concentration(frame: pd.DataFrame) -> float:
        if frame.empty:
            return 0.0
        counts = frame.groupby("head", sort=False).size().to_numpy(dtype=float)
        if float(counts.sum()) <= 0.0:
            return 0.0
        share = counts / float(counts.sum())
        max_share = float(np.max(share))
        # Smoothly penalize only excess concentration above the Optuna-selected
        # cap. This avoids rewarding any specific head for being profitable in
        # the sampled period.
        return float(max(0.0, max_share - cap) ** 2)

    global_concentration = _concentration(work)
    if "timestamp" not in work.columns:
        return global_concentration, 0.0
    ts = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.loc[ts.notna()].copy()
    if work.empty:
        return global_concentration, 0.0
    work["week"] = ts.loc[work.index].dt.to_period("W").dt.start_time.astype(str)
    weekly = [_concentration(g) for _, g in work.groupby("week", sort=True) if len(g) >= 5]
    return global_concentration, float(np.mean(weekly)) if weekly else 0.0


def _objective_downside_metrics(accepted: pd.DataFrame, params: RecentHeadParams) -> dict[str, float]:
    low = float(params.objective_q_low)
    mid = float(params.objective_q_mid)
    short_h = int(params.objective_short_horizon_hours)
    long_h = int(params.objective_long_horizon_hours)
    q_low_long = _rolling_window_pnl_quantile(accepted, hours=long_h, quantile=low)
    q_low_short = _rolling_window_pnl_quantile(accepted, hours=short_h, quantile=low)
    q_mid_long = _rolling_window_pnl_quantile(accepted, hours=long_h, quantile=mid)
    q_mid_short = _rolling_window_pnl_quantile(accepted, hours=short_h, quantile=mid)
    return {
        "objective_q_low_long_pnl": q_low_long,
        "objective_q_low_short_pnl": q_low_short,
        "objective_q_mid_long_pnl": q_mid_long,
        "objective_q_mid_short_pnl": q_mid_short,
        "objective_robust_downside_pnl": q_low_long + q_low_short + q_mid_long + q_mid_short,
    }


def _objective_contract() -> dict[str, Any]:
    return {
        "head_identity_invariant": True,
        "no_named_head_reward": True,
        "no_named_head_suppression_penalty": True,
        "portfolio_pnl_dominant": True,
        "allowed_head_terms": [
            "symmetric action-concentration penalty",
            "head-agnostic accepted-trade concentration penalty",
            "per-head recent health computed with the same formula for every head",
        ],
        "minimum_trade_count_penalty": True,
        "forbidden_head_terms": [
            "named-head PnL reward",
            "named-head suppression penalty",
            "period-specific forced activation/deactivation preference",
        ],
    }


def _candidate_action_metric_heads(summary: pd.Series | dict[str, Any]) -> list[str]:
    """Find heads that have per-head overlay action metrics in a summary row."""

    keys = [str(k) for k in dict(summary).keys()]
    heads: set[str] = set()
    for key in keys:
        for suffix in ACTION_METRIC_SUFFIXES:
            marker = f"_{suffix}"
            if key.endswith(marker):
                head = key[: -len(marker)]
                if head:
                    heads.add(head)
    return sorted(heads)


def _objective_components(
    summary: pd.Series,
    accepted: pd.DataFrame,
    *,
    baseline_summary: pd.Series,
    baseline_accepted: pd.DataFrame,
    min_trades: int,
    params: RecentHeadParams,
) -> dict[str, float]:
    net = float(summary.get("net_pnl", 0.0) or 0.0)
    baseline_net = float(baseline_summary.get("net_pnl", 0.0) or 0.0)
    net_delta = net - baseline_net
    trade_count = float(summary.get("trade_count", 0.0) or 0.0)
    min_trade_count = max(float(min_trades), 0.0)
    min_trade_shortfall = max(0.0, min_trade_count - trade_count)
    min_trade_penalty = min_trade_shortfall * min_trade_shortfall
    full_sl = float(summary.get("full_sl_rate", 0.0) or 0.0)
    baseline_full_sl = float(baseline_summary.get("full_sl_rate", 0.0) or 0.0)
    full_sl_deterioration = max(0.0, full_sl - baseline_full_sl)
    downside = _objective_downside_metrics(accepted, params)
    baseline_downside = _objective_downside_metrics(baseline_accepted, params)
    robust_downside = float(downside["objective_robust_downside_pnl"])
    q_low_delta = (
        float(downside["objective_q_low_long_pnl"] - baseline_downside["objective_q_low_long_pnl"])
        + float(downside["objective_q_low_short_pnl"] - baseline_downside["objective_q_low_short_pnl"])
    )
    q_mid_long_delta = float(downside["objective_q_mid_long_pnl"] - baseline_downside["objective_q_mid_long_pnl"])
    q_mid_short_delta = float(downside["objective_q_mid_short_pnl"] - baseline_downside["objective_q_mid_short_pnl"])
    q_mid_deterioration = max(0.0, -q_mid_long_delta) + max(0.0, -q_mid_short_delta)
    defensive = _defensive_success_metrics(baseline_accepted, accepted)
    hard_stop_share = float(summary.get("candidate_share_hard_stop", 0.0) or 0.0)
    hard_stop_penalty = (
        float(params.objective_hard_stop_weight)
        * max(0.0, hard_stop_share - float(params.objective_hard_stop_start)) ** 2
    )

    head_action_concentration_penalty = 0.0
    for head in _candidate_action_metric_heads(summary):
        head_hard_stop = float(summary.get(f"{head}_candidate_share_hard_stop", 0.0) or 0.0)
        head_threshold_delta = float(summary.get(f"{head}_candidate_mean_threshold_delta", 0.0) or 0.0)
        head_size_mult = float(summary.get(f"{head}_candidate_mean_size_multiplier", 1.0) or 1.0)
        head_capacity_cut = float(summary.get(f"{head}_candidate_share_capacity_reduced", 0.0) or 0.0)
        head_action_concentration_penalty += float(params.objective_head_action_weight) * (
            max(0.0, head_hard_stop - float(params.objective_head_hard_stop_start)) ** 2
            + 0.40 * max(0.0, head_threshold_delta - float(params.objective_head_threshold_start)) ** 2
            + 0.25 * max(0.0, float(params.objective_head_size_floor) - head_size_mult) ** 2
            + 0.20 * max(0.0, head_capacity_cut - float(params.objective_head_capacity_start)) ** 2
        )

    global_concentration, weekly_concentration = _head_activity_concentration_penalty(
        accepted,
        max_head_share=float(params.objective_max_head_trade_share),
    )
    balance_penalty = (
        float(params.objective_global_balance_weight) * global_concentration
        + float(params.objective_weekly_balance_weight) * weekly_concentration
    )

    base_utility = (
        net_delta
        + float(params.objective_q_low_weight) * q_low_delta
        - float(params.objective_q_mid_deterioration_weight) * q_mid_deterioration
        + float(params.objective_defensive_success_weight) * float(defensive["defensive_success_pnl"])
        - float(params.objective_full_sl_penalty) * full_sl_deterioration
    )
    objective = (
        base_utility
        - hard_stop_penalty
        - head_action_concentration_penalty
        - balance_penalty
        - min_trade_penalty
    )
    out = {
        "objective": float(objective),
        "base_utility": float(base_utility),
        "net_delta": float(net_delta),
        "trade_count": float(trade_count),
        "min_trade_count": float(min_trade_count),
        "min_trade_shortfall": float(min_trade_shortfall),
        "min_trade_penalty": float(min_trade_penalty),
        "full_sl_deterioration": float(full_sl_deterioration),
        "robust_downside_pnl": float(robust_downside),
        "q_low_protection_delta": float(q_low_delta),
        "q_mid_deterioration_pnl": float(q_mid_deterioration),
        "objective_baseline_q_low_long_pnl": float(baseline_downside["objective_q_low_long_pnl"]),
        "objective_baseline_q_low_short_pnl": float(baseline_downside["objective_q_low_short_pnl"]),
        "objective_baseline_q_mid_long_pnl": float(baseline_downside["objective_q_mid_long_pnl"]),
        "objective_baseline_q_mid_short_pnl": float(baseline_downside["objective_q_mid_short_pnl"]),
        "head_action_concentration_penalty": float(head_action_concentration_penalty),
        "hard_stop_penalty": float(hard_stop_penalty),
        "global_head_concentration": float(global_concentration),
        "weekly_head_concentration": float(weekly_concentration),
        "head_balance_penalty": float(balance_penalty),
    }
    out.update(defensive)
    out.update(downside)
    return out


def _objective_value(
    summary: pd.Series,
    accepted: pd.DataFrame,
    *,
    baseline_summary: pd.Series,
    baseline_accepted: pd.DataFrame,
    min_trades: int,
    params: RecentHeadParams,
) -> float:
    return _objective_components(
        summary,
        accepted,
        baseline_summary=baseline_summary,
        baseline_accepted=baseline_accepted,
        min_trades=min_trades,
        params=params,
    )["objective"]


def _portfolio_promotion_gate(
    baseline_summary: pd.Series | dict[str, Any],
    overlay_summary: pd.Series | dict[str, Any],
    *,
    min_trade_retention: float = 0.75,
) -> dict[str, Any]:
    """Evaluate conservative promotion gates against the paired static baseline."""

    base = dict(baseline_summary)
    overlay = dict(overlay_summary)
    base_trades = float(base.get("trade_count", 0.0) or 0.0)
    overlay_trades = float(overlay.get("trade_count", 0.0) or 0.0)
    net_delta = float(overlay.get("net_pnl", 0.0) or 0.0) - float(base.get("net_pnl", 0.0) or 0.0)
    robust_downside_delta = (
        float(overlay.get("robust_downside_pnl", 0.0) or 0.0)
        - float(base.get("robust_downside_pnl", 0.0) or 0.0)
    )
    full_sl_delta = float(overlay.get("full_sl_rate", 0.0) or 0.0) - float(base.get("full_sl_rate", 0.0) or 0.0)
    worst_24h_delta = (
        float(overlay.get("worst_24h_net_pnl", 0.0) or 0.0)
        - float(base.get("worst_24h_net_pnl", 0.0) or 0.0)
    )
    defensive_success = float(overlay.get("defensive_success_pnl", 0.0) or 0.0)
    loss_avoided = float(overlay.get("loss_avoided_pnl", 0.0) or 0.0)
    winner_sacrificed = float(overlay.get("winner_pnl_sacrificed", 0.0) or 0.0)
    trade_retention = overlay_trades / base_trades if base_trades > 0 else 0.0
    gates = {
        "net_pnl_improved": net_delta > 0.0,
        "robust_downside_not_worse": robust_downside_delta >= 0.0,
        "full_sl_not_worse": full_sl_delta <= 0.0,
        "worst_24h_not_worse": worst_24h_delta >= 0.0,
        "defensive_success_positive": defensive_success > 0.0,
        "loss_avoided_exceeds_winner_sacrificed": loss_avoided > winner_sacrificed,
        "trade_retention_sufficient": trade_retention >= float(min_trade_retention),
    }
    return {
        "passed": bool(all(gates.values())),
        "gates": gates,
        "net_pnl_delta": float(net_delta),
        "robust_downside_delta": float(robust_downside_delta),
        "full_sl_rate_delta": float(full_sl_delta),
        "worst_24h_net_pnl_delta": float(worst_24h_delta),
        "defensive_success_pnl": float(defensive_success),
        "loss_avoided_pnl": float(loss_avoided),
        "winner_pnl_sacrificed": float(winner_sacrificed),
        "trade_retention": float(trade_retention),
        "min_trade_retention": float(min_trade_retention),
    }


def _suggest_params(trial: optuna.Trial) -> RecentHeadParams:
    health_clip = float(trial.suggest_float("health_clip", 0.15, 1.0))
    return RecentHeadParams(
        lookback_hours=float(trial.suggest_categorical("lookback_hours", [24, 48, 72, 120, 168, 240])),
        embargo_hours=float(trial.suggest_categorical("embargo_hours", [0, 6, 12, 24])),
        min_samples=int(trial.suggest_categorical("min_samples", [5, 10, 20, 40])),
        shrink_samples=float(trial.suggest_float("shrink_samples", 10.0, 120.0, log=True)),
        decay_halflife_hours=float(trial.suggest_categorical("decay_halflife_hours", [24, 48, 96, 168])),
        health_clip=health_clip,
        net_weight=float(trial.suggest_float("net_weight", 0.2, 2.5)),
        hr_weight=float(trial.suggest_float("hr_weight", 0.0, 2.0)),
        weighted_hr_weight=float(trial.suggest_float("weighted_hr_weight", 0.0, 2.0)),
        ic_weight=float(trial.suggest_float("ic_weight", 0.0, 1.5)),
        full_sl_weight=float(trial.suggest_float("full_sl_weight", 0.2, 2.5)),
        cost_drag_weight=float(trial.suggest_float("cost_drag_weight", 0.0, 1.5)),
        worst_return_weight=float(trial.suggest_float("worst_return_weight", 0.0, 1.5)),
        weighted_hr_power=float(trial.suggest_float("weighted_hr_power", 0.5, 3.0)),
        head_control_strength=float(trial.suggest_float("head_control_strength", 0.0, 1.5)),
        threshold_start=float(trial.suggest_float("threshold_start", 0.0, 0.35)),
        threshold_scale=float(trial.suggest_float("threshold_scale", 0.02, 0.40)),
        threshold_power=float(trial.suggest_float("threshold_power", 0.7, 2.5)),
        max_threshold_shift=float(trial.suggest_float("max_threshold_shift", 0.02, 0.18)),
        size_start=float(trial.suggest_float("size_start", 0.0, 0.45)),
        size_scale=float(trial.suggest_float("size_scale", 0.0, 1.5)),
        min_size_multiplier=float(trial.suggest_float("min_size_multiplier", 0.10, 0.80)),
        cap_start=float(trial.suggest_float("cap_start", 0.2, 0.8)),
        cap_scale=float(trial.suggest_float("cap_scale", 0.0, 1.5)),
        hard_stop_health=float(trial.suggest_float("hard_stop_health_raw", -4.0, -1.0)),
        hard_stop_threshold=float(trial.suggest_float("hard_stop_threshold", 0.98, 1.01)),
        objective_q_low_weight=float(trial.suggest_float("objective_q_low_weight", 0.0, 0.30)),
        objective_q_mid_deterioration_weight=float(
            trial.suggest_float("objective_q_mid_deterioration_weight", 0.0, 0.60)
        ),
        objective_defensive_success_weight=float(trial.suggest_float("objective_defensive_success_weight", 0.0, 0.35)),
        objective_full_sl_penalty=float(trial.suggest_float("objective_full_sl_penalty", 0.0, 160.0)),
        objective_q_low=float(trial.suggest_float("objective_q_low", 0.03, 0.10)),
        objective_q_mid=float(trial.suggest_float("objective_q_mid", 0.10, 0.25)),
        objective_short_horizon_hours=int(trial.suggest_categorical("objective_short_horizon_hours", [24, 48, 72])),
        objective_long_horizon_hours=int(trial.suggest_categorical("objective_long_horizon_hours", [96, 120, 168, 240])),
        objective_hard_stop_start=float(trial.suggest_float("objective_hard_stop_start", 0.05, 0.30)),
        objective_hard_stop_weight=float(trial.suggest_float("objective_hard_stop_weight", 5_000.0, 100_000.0, log=True)),
        objective_head_action_weight=float(trial.suggest_float("objective_head_action_weight", 1_000.0, 80_000.0, log=True)),
        objective_head_hard_stop_start=float(trial.suggest_float("objective_head_hard_stop_start", 0.20, 0.65)),
        objective_head_threshold_start=float(trial.suggest_float("objective_head_threshold_start", 0.08, 0.30)),
        objective_head_size_floor=float(trial.suggest_float("objective_head_size_floor", 0.20, 0.75)),
        objective_head_capacity_start=float(trial.suggest_float("objective_head_capacity_start", 0.35, 0.90)),
        objective_max_head_trade_share=float(trial.suggest_float("objective_max_head_trade_share", 0.65, 0.95)),
        objective_global_balance_weight=float(trial.suggest_float("objective_global_balance_weight", 0.0, 1_500.0)),
        objective_weekly_balance_weight=float(trial.suggest_float("objective_weekly_balance_weight", 0.0, 1_500.0)),
    )


HEALTH_PARAM_NAMES = {
    "lookback_hours",
    "embargo_hours",
    "min_samples",
    "shrink_samples",
    "decay_halflife_hours",
    "health_clip",
    "net_weight",
    "hr_weight",
    "weighted_hr_weight",
    "ic_weight",
    "full_sl_weight",
    "cost_drag_weight",
    "worst_return_weight",
    "weighted_hr_power",
}


def _fixed_action_for_health_stage(params: RecentHeadParams) -> RecentHeadParams:
    return replace(
        params,
        head_control_strength=0.5,
        threshold_start=0.20,
        threshold_scale=0.10,
        threshold_power=1.25,
        max_threshold_shift=0.08,
        size_start=0.05,
        size_scale=0.35,
        min_size_multiplier=0.50,
        cap_start=0.80,
        cap_scale=0.25,
        hard_stop_health=-3.0,
        hard_stop_threshold=0.99,
        objective_q_low_weight=0.12,
        objective_q_mid_deterioration_weight=0.30,
        objective_defensive_success_weight=0.12,
        objective_full_sl_penalty=60.0,
        objective_q_low=0.05,
        objective_q_mid=0.15,
        objective_short_horizon_hours=48,
        objective_long_horizon_hours=120,
        objective_hard_stop_start=0.30,
        objective_hard_stop_weight=30_000.0,
        objective_head_action_weight=30_000.0,
        objective_head_hard_stop_start=0.45,
        objective_head_threshold_start=0.20,
        objective_head_size_floor=0.35,
        objective_head_capacity_start=0.75,
        objective_max_head_trade_share=0.85,
        objective_global_balance_weight=500.0,
        objective_weekly_balance_weight=500.0,
    )


def _merge_health_into_action(health_params: RecentHeadParams, action_params: RecentHeadParams) -> RecentHeadParams:
    updates = {name: getattr(health_params, name) for name in HEALTH_PARAM_NAMES}
    return replace(action_params, **updates)


def _add_prefixed_attrs(trial: optuna.Trial, values: dict[str, float], *, prefix: str = "") -> None:
    for key, value in values.items():
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
            trial.set_user_attr(f"{prefix}{key}", float(value))


def _render_report(
    *,
    output_dir: Path,
    best_params: RecentHeadParams,
    selection_split: dict[str, Any],
    selection_ev_reference_diag: dict[str, Any],
    selection_summary: pd.DataFrame,
    baseline_selection_summary: pd.DataFrame,
    selection_gate: dict[str, Any],
    train_summary: pd.DataFrame,
    eval_summary: pd.DataFrame,
    baseline_eval_summary: pd.DataFrame,
    eval_gate: dict[str, Any],
    eval_by_head: pd.DataFrame,
) -> str:
    lines = [
        "# Recent Head Activation Optuna",
        "",
        "This is a research ablation. It does not replace the active T1 stack.",
        "",
        "## Objective Contract",
        "",
        "The objective is head-identity invariant: it can reward portfolio PnL and robust downside behavior, "
        "but it cannot contain named-head rewards or named-head suppression penalties. Per-head controls are "
        "symmetric and use the same recent-health formula for every head.",
        "",
        "## Best Parameters",
        "",
        "```json",
        json.dumps(_json_safe(asdict(best_params)), indent=2),
        "```",
        "",
        "## Selection Split",
        "",
        f"- Mode: `{selection_split.get('mode')}`",
        f"- Reference: `{selection_split.get('reference', {}).get('timestamp_min')}` to "
        f"`{selection_split.get('reference', {}).get('timestamp_max')}` "
        f"({selection_split.get('reference', {}).get('timestamp_count')} timestamps)",
        f"- Objective: `{selection_split.get('objective', {}).get('timestamp_min')}` to "
        f"`{selection_split.get('objective', {}).get('timestamp_max')}` "
        f"({selection_split.get('objective', {}).get('timestamp_count')} timestamps)",
        f"- EV reference mode: `{selection_ev_reference_diag.get('mode')}`",
        f"- EV reference: `{selection_ev_reference_diag.get('reference', {}).get('timestamp_min')}` to "
        f"`{selection_ev_reference_diag.get('reference', {}).get('timestamp_max')}` "
        f"({selection_ev_reference_diag.get('reference', {}).get('timestamp_count')} timestamps)",
        "",
        "## Selection Objective Summary",
        "",
        "| arm | trades | net_pnl | full_sl | q5_120h | robust_downside | defensive_success | min_trade_shortfall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for frame in (baseline_selection_summary, selection_summary):
        row = frame.iloc[0]
        lines.append(
            f"| {row['arm']} | {int(row.get('trade_count', 0))} | {float(row.get('net_pnl', 0.0)):.6f} | "
            f"{float(row.get('full_sl_rate', 0.0)):.6f} | {float(row.get('q5_120h_net_pnl', 0.0)):.6f} | "
            f"{float(row.get('robust_downside_pnl', 0.0)):.6f} | "
            f"{float(row.get('defensive_success_pnl', 0.0)):.6f} | "
            f"{float(row.get('min_trade_shortfall', 0.0)):.0f} |"
        )
    lines.extend(
        [
            "",
        "## Evaluation Summary",
        "",
        "| arm | trades | net_pnl | full_sl | timeout | q5_120h | q5_48h | q15_120h | q15_48h | worst_24h |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for frame in (baseline_eval_summary, eval_summary):
        row = frame.iloc[0]
        lines.append(
            f"| {row['arm']} | {int(row['trade_count'])} | {float(row['net_pnl']):.6f} | "
            f"{float(row['full_sl_rate']):.6f} | {float(row['timeout_rate']):.6f} | "
            f"{float(row.get('q5_120h_net_pnl', 0.0)):.6f} | "
            f"{float(row.get('q5_48h_net_pnl', 0.0)):.6f} | "
            f"{float(row.get('q15_120h_net_pnl', 0.0)):.6f} | "
            f"{float(row.get('q15_48h_net_pnl', 0.0)):.6f} | "
            f"{float(row['worst_24h_net_pnl']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Overlay Controls",
            "",
            "| arm | accepted_mean_threshold_delta | candidate_threshold_raised | candidate_size_shrunk | candidate_capacity_reduced | candidate_hard_stop |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for frame in (baseline_eval_summary, eval_summary):
        row = frame.iloc[0]
        lines.append(
            f"| {row['arm']} | {float(row.get('mean_threshold_delta', 0.0)):.6f} | "
            f"{float(row.get('candidate_share_threshold_raised', 0.0)):.6f} | "
            f"{float(row.get('candidate_share_size_shrunk', 0.0)):.6f} | "
            f"{float(row.get('candidate_share_capacity_reduced', 0.0)):.6f} | "
            f"{float(row.get('candidate_share_hard_stop', 0.0)):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Defensive Success",
            "",
            "| arm | removed_trades | loss_avoided | winner_pnl_sacrificed | defensive_success |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for frame in (baseline_eval_summary, eval_summary):
        row = frame.iloc[0]
        lines.append(
            f"| {row['arm']} | {float(row.get('removed_trade_count', 0.0)):.0f} | "
            f"{float(row.get('loss_avoided_pnl', 0.0)):.6f} | "
            f"{float(row.get('winner_pnl_sacrificed', 0.0)):.6f} | "
            f"{float(row.get('defensive_success_pnl', 0.0)):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Promotion Gate",
            "",
            "| period | passed | net_delta | robust_downside_delta | full_sl_delta | defensive_success | trade_retention | failed_gates |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for period, gate in (("selection", selection_gate), ("eval", eval_gate)):
        failed = ",".join(name for name, passed in gate.get("gates", {}).items() if not passed)
        lines.append(
            f"| {period} | {bool(gate.get('passed'))} | "
            f"{float(gate.get('net_pnl_delta', 0.0)):.6f} | "
            f"{float(gate.get('robust_downside_delta', 0.0)):.6f} | "
            f"{float(gate.get('full_sl_rate_delta', 0.0)):.6f} | "
            f"{float(gate.get('defensive_success_pnl', 0.0)):.6f} | "
            f"{float(gate.get('trade_retention', 0.0)):.6f} | {failed or 'none'} |"
        )
    lines.extend(["", "## Evaluation By Head", ""])
    if eval_by_head.empty:
        lines.append("No accepted trades.")
    else:
        lines.extend(
            [
                "| head | trades | win_rate | net_pnl | full_sl | timeout |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for rec in eval_by_head.to_dict("records"):
            lines.append(
                f"| {rec['head']} | {int(rec['trade_count'])} | {float(rec['win_rate']):.6f} | "
                f"{float(rec['net_pnl']):.6f} | {float(rec['full_sl_rate']):.6f} | "
                f"{float(rec['timeout_rate']):.6f} |"
            )
    lines.extend(
        [
            "",
            "## Train Objective Feasibility",
            "",
            "| arm | trades | min_trade_count | min_trade_shortfall | min_trade_penalty |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    train_row = train_summary.iloc[0]
    lines.append(
        f"| {train_row['arm']} | {int(train_row.get('trade_count', 0))} | "
        f"{float(train_row.get('min_trade_count', 0.0)):.0f} | "
        f"{float(train_row.get('min_trade_shortfall', 0.0)):.0f} | "
        f"{float(train_row.get('min_trade_penalty', 0.0)):.6f} |"
    )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Trial table: `{output_dir / 'optuna_trials.csv'}`",
            f"- Manifest: `{output_dir / 'manifest.json'}`",
            f"- Eval accepted trades: `{output_dir / 'eval_overlay_accepted_trades.parquet'}`",
            f"- Eval schedule: `{output_dir / 'eval_recent_head_schedule.csv'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-broad-candidates", type=Path, default=DEFAULT_TRAIN_BROAD)
    parser.add_argument("--train-deployable-candidates", type=Path, default=DEFAULT_TRAIN_DEPLOYABLE)
    parser.add_argument("--eval-candidates", type=Path, default=DEFAULT_EVAL_CANDIDATES)
    parser.add_argument("--policy-manifest", type=Path, default=DEFAULT_POLICY_MANIFEST)
    parser.add_argument("--policy-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument(
        "--rank-contract",
        choices=("anchor_global_policy_rank_reference", "short_boll_timestamp_rank"),
        default=DEFAULT_RANK_CONTRACT,
        help=(
            "Rank contract for this ablation. The default is the active provisional T1 "
            "short_boll within-timestamp rank repair. Use anchor_global_policy_rank_reference "
            "only for the explicit global-over-time research challenger."
        ),
    )
    parser.add_argument("--rank-reference-run-id", default=DEFAULT_RANK_REFERENCE_RUN_ID)
    parser.add_argument("--data-root", type=Path, default=Path("data_perp"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--health-trials", type=int, default=12)
    parser.add_argument("--seed", type=int, default=719)
    parser.add_argument("--disable-heads", default="")
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument("--min-train-trades", type=int, default=100)
    parser.add_argument(
        "--selection-validation-frac",
        type=float,
        default=0.30,
        help=(
            "Fraction of pre-June train timestamps reserved as a chronological "
            "Optuna-selection objective. Set to 0 for legacy full-train selection."
        ),
    )
    parser.add_argument(
        "--selection-min-validation-timestamps",
        type=int,
        default=24,
        help="Minimum complete timestamps in the pre-June Optuna-selection objective slice.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    disabled_heads = _parse_heads(args.disable_heads)
    train_broad_raw = _load_candidates(args.train_broad_candidates)
    train_deployable_raw = _load_candidates(args.train_deployable_candidates)
    eval_raw = _load_candidates(args.eval_candidates)
    if disabled_heads:
        train_broad_raw = train_broad_raw.loc[~train_broad_raw["head"].isin(disabled_heads)].copy()
        train_deployable_raw = train_deployable_raw.loc[~train_deployable_raw["head"].isin(disabled_heads)].copy()
        eval_raw = eval_raw.loc[~eval_raw["head"].isin(disabled_heads)].copy()
    rank_contract = str(args.rank_contract)
    rank_label = _ranked_candidate_label(rank_contract)
    train_broad, train_rank_diag = _apply_ablation_rank_contract(
        train_broad_raw,
        rank_contract=rank_contract,
        data_root=args.data_root,
        rank_reference_run_id=str(args.rank_reference_run_id),
    )
    train_deployable, train_deployable_rank_diag = _apply_ablation_rank_contract(
        train_deployable_raw,
        rank_contract=rank_contract,
        data_root=args.data_root,
        rank_reference_run_id=str(args.rank_reference_run_id),
    )
    eval_candidates, eval_rank_diag = _apply_ablation_rank_contract(
        eval_raw,
        rank_contract=rank_contract,
        data_root=args.data_root,
        rank_reference_run_id=str(args.rank_reference_run_id),
    )
    baseline_eval_arm = f"baseline_{rank_label}_static"
    baseline_train_arm = f"train_baseline_{rank_label}_static"
    params, policy_payload = mstc._load_policy_params(args.policy_manifest, args.policy_variant)
    ev_curve = fit_hierarchical_ev_curves(train_deployable)

    baseline_eval_accepted, baseline_eval_decisions, baseline_eval_summary = _replay(
        eval_candidates,
        params=params,
        ev_curve=ev_curve,
        arm=baseline_eval_arm,
        market_mode=str(args.market_mode),
    )
    baseline_train_accepted, _, baseline_train_summary = _replay(
        train_broad,
        params=params,
        ev_curve=ev_curve,
        arm=baseline_train_arm,
        market_mode=str(args.market_mode),
    )
    selection_reference, selection_objective, selection_split = _chronological_selection_split(
        train_broad,
        validation_frac=float(args.selection_validation_frac),
        min_validation_timestamps=int(args.selection_min_validation_timestamps),
    )
    selection_ev_candidates, selection_ev_reference_diag = _selection_ev_reference(
        train_deployable,
        selection_split,
    )
    selection_ev_curve = fit_hierarchical_ev_curves(selection_ev_candidates)
    baseline_selection_arm = f"selection_baseline_{rank_label}_static"
    baseline_selection_accepted, _, baseline_selection_summary = _replay(
        selection_objective,
        params=params,
        ev_curve=selection_ev_curve,
        arm=baseline_selection_arm,
        market_mode=str(args.market_mode),
    )
    selection_schedule_reference = pd.concat(
        [selection_reference, selection_objective],
        ignore_index=True,
        copy=False,
    )

    def evaluate_params(hp: RecentHeadParams, trial: optuna.Trial | None = None) -> tuple[float, dict[str, float]]:
        baselines = _baseline_by_head(
            selection_reference,
            hp.weighted_hr_power,
            head_control_strength=hp.head_control_strength,
        )
        schedule = _recent_health_schedule(
            selection_schedule_reference,
            selection_objective["timestamp"],
            params=hp,
            baselines=baselines,
        )
        scored = _apply_overlay(selection_objective, schedule, hp)
        accepted, _, summary = _replay(
            scored,
            params=params,
            ev_curve=selection_ev_curve,
            arm="selection_recent_head_overlay",
            market_mode=str(args.market_mode),
        )
        components = _objective_components(
            summary.iloc[0],
            accepted,
            baseline_summary=baseline_selection_summary.iloc[0],
            baseline_accepted=baseline_selection_accepted,
            min_trades=int(args.min_train_trades),
            params=hp,
        )
        score = components["objective"]
        attrs = {
            "trade_count": float(summary.iloc[0].get("trade_count", 0)),
            "net_pnl": float(summary.iloc[0].get("net_pnl", 0.0)),
            "full_sl_rate": float(summary.iloc[0].get("full_sl_rate", 0.0)),
            "worst_24h_net_pnl": float(summary.iloc[0].get("worst_24h_net_pnl", 0.0)),
        }
        for key in (
            "objective",
            "base_utility",
            "net_delta",
            "full_sl_deterioration",
            "robust_downside_pnl",
            "objective_robust_downside_pnl",
            "objective_q_low_long_pnl",
            "objective_q_low_short_pnl",
            "objective_q_mid_long_pnl",
            "objective_q_mid_short_pnl",
            "q_low_protection_delta",
            "q_mid_deterioration_pnl",
            "trade_count",
            "min_trade_count",
            "min_trade_shortfall",
            "min_trade_penalty",
            "defensive_success_pnl",
            "loss_avoided_pnl",
            "winner_pnl_sacrificed",
            "removed_trade_count",
            "head_action_concentration_penalty",
            "head_balance_penalty",
            "global_head_concentration",
            "weekly_head_concentration",
            "hard_stop_penalty",
            "q5_120h_net_pnl",
            "q5_48h_net_pnl",
            "q15_120h_net_pnl",
            "q15_48h_net_pnl",
        ):
            attrs[key] = float(components.get(key, summary.iloc[0].get(key, 0.0)))
        if trial is not None:
            _add_prefixed_attrs(trial, attrs)
        return float(score), attrs

    health_study: optuna.Study | None = None
    best_health_params: RecentHeadParams | None = None
    if int(args.health_trials) > 0:
        def health_objective(trial: optuna.Trial) -> float:
            hp = _fixed_action_for_health_stage(_suggest_params(trial))
            score, _ = evaluate_params(hp, trial)
            return float(score)

        health_study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=int(args.seed), multivariate=True),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0),
            study_name="recent_head_activation_health_optuna",
        )
        health_study.optimize(health_objective, n_trials=max(1, int(args.health_trials)), show_progress_bar=False)
        best_health_params = _fixed_action_for_health_stage(_suggest_params(health_study.best_trial))

    def action_objective(trial: optuna.Trial) -> float:
        hp = _suggest_params(trial)
        if best_health_params is not None:
            hp = _merge_health_into_action(best_health_params, hp)
        score, _ = evaluate_params(hp, trial)
        return float(score)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(args.seed) + 1, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0),
        study_name="recent_head_activation_action_optuna",
    )
    study.optimize(action_objective, n_trials=max(1, int(args.n_trials)), show_progress_bar=False)
    best_params = _suggest_params(study.best_trial)
    if best_health_params is not None:
        best_params = _merge_health_into_action(best_health_params, best_params)

    selection_baselines = _baseline_by_head(
        selection_reference,
        best_params.weighted_hr_power,
        head_control_strength=best_params.head_control_strength,
    )
    selection_schedule = _recent_health_schedule(
        selection_schedule_reference,
        selection_objective["timestamp"],
        params=best_params,
        baselines=selection_baselines,
    )
    selection_scored = _apply_overlay(selection_objective, selection_schedule, best_params)
    selection_accepted, selection_decisions, selection_summary = _replay(
        selection_scored,
        params=params,
        ev_curve=selection_ev_curve,
        arm="selection_recent_head_overlay",
        market_mode=str(args.market_mode),
    )
    selection_summary = _annotate_summary(
        selection_summary,
        _defensive_success_metrics(baseline_selection_accepted, selection_accepted),
    )
    selected_selection_objective = _objective_components(
        selection_summary.iloc[0],
        selection_accepted,
        baseline_summary=baseline_selection_summary.iloc[0],
        baseline_accepted=baseline_selection_accepted,
        min_trades=int(args.min_train_trades),
        params=best_params,
    )
    selection_summary = _annotate_summary(selection_summary, selected_selection_objective)

    baselines = _baseline_by_head(
        train_broad,
        best_params.weighted_hr_power,
        head_control_strength=best_params.head_control_strength,
    )
    train_schedule = _recent_health_schedule(
        train_broad,
        train_broad["timestamp"],
        params=best_params,
        baselines=baselines,
    )
    train_scored = _apply_overlay(train_broad, train_schedule, best_params)
    train_accepted, train_decisions, train_summary = _replay(
        train_scored,
        params=params,
        ev_curve=ev_curve,
        arm="train_recent_head_overlay",
        market_mode=str(args.market_mode),
    )
    train_summary = _annotate_summary(
        train_summary,
        _defensive_success_metrics(baseline_train_accepted, train_accepted),
    )
    selected_train_objective = _objective_components(
        train_summary.iloc[0],
        train_accepted,
        baseline_summary=baseline_train_summary.iloc[0],
        baseline_accepted=baseline_train_accepted,
        min_trades=int(args.min_train_trades),
        params=best_params,
    )
    train_summary = _annotate_summary(train_summary, selected_train_objective)
    eval_reference = pd.concat([train_broad, eval_candidates], ignore_index=True, copy=False)
    eval_schedule = _recent_health_schedule(
        eval_reference,
        eval_candidates["timestamp"],
        params=best_params,
        baselines=baselines,
    )
    eval_scored = _apply_overlay(eval_candidates, eval_schedule, best_params)
    eval_accepted, eval_decisions, eval_summary = _replay(
        eval_scored,
        params=params,
        ev_curve=ev_curve,
        arm="eval_recent_head_overlay",
        market_mode=str(args.market_mode),
    )
    eval_summary = _annotate_summary(
        eval_summary,
        _defensive_success_metrics(baseline_eval_accepted, eval_accepted),
    )
    baseline_train_summary = _annotate_summary(
        baseline_train_summary,
        {
            "removed_trade_count": 0.0,
            "loss_avoided_pnl": 0.0,
            "winner_pnl_sacrificed": 0.0,
            "defensive_success_pnl": 0.0,
        },
    )
    baseline_selection_summary = _annotate_summary(
        baseline_selection_summary,
        {
            "removed_trade_count": 0.0,
            "loss_avoided_pnl": 0.0,
            "winner_pnl_sacrificed": 0.0,
            "defensive_success_pnl": 0.0,
            "min_trade_count": float(args.min_train_trades),
            "min_trade_shortfall": 0.0,
            "min_trade_penalty": 0.0,
        },
    )
    baseline_eval_summary = _annotate_summary(
        baseline_eval_summary,
        {
            "removed_trade_count": 0.0,
            "loss_avoided_pnl": 0.0,
            "winner_pnl_sacrificed": 0.0,
            "defensive_success_pnl": 0.0,
        },
    )
    selection_gate = _portfolio_promotion_gate(
        baseline_selection_summary.iloc[0],
        selection_summary.iloc[0],
    )
    train_gate = _portfolio_promotion_gate(
        baseline_train_summary.iloc[0],
        train_summary.iloc[0],
    )
    eval_gate = _portfolio_promotion_gate(
        baseline_eval_summary.iloc[0],
        eval_summary.iloc[0],
    )
    eval_by_head = mstc._by_head("eval_recent_head_overlay", eval_accepted)

    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    trials.to_csv(args.output_dir / "optuna_trials.csv", index=False)
    if health_study is not None:
        health_trials = health_study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
        health_trials.to_csv(args.output_dir / "optuna_health_trials.csv", index=False)
    train_broad.to_parquet(args.output_dir / "train_candidates_ranked.parquet", index=False)
    eval_candidates.to_parquet(args.output_dir / "eval_candidates_ranked.parquet", index=False)
    if rank_contract == "anchor_global_policy_rank_reference":
        train_broad.to_parquet(args.output_dir / "train_candidates_global_rank.parquet", index=False)
        eval_candidates.to_parquet(args.output_dir / "eval_candidates_global_rank.parquet", index=False)
    elif rank_contract == "short_boll_timestamp_rank":
        train_broad.to_parquet(args.output_dir / "train_candidates_t1_timestamp_rank.parquet", index=False)
        eval_candidates.to_parquet(args.output_dir / "eval_candidates_t1_timestamp_rank.parquet", index=False)
    selection_ev_candidates.to_parquet(args.output_dir / "selection_ev_curve_candidates.parquet", index=False)
    train_scored.to_parquet(args.output_dir / "train_overlay_candidates.parquet", index=False)
    selection_scored.to_parquet(args.output_dir / "selection_overlay_candidates.parquet", index=False)
    eval_scored.to_parquet(args.output_dir / "eval_overlay_candidates.parquet", index=False)
    train_accepted.to_parquet(args.output_dir / "train_overlay_accepted_trades.parquet", index=False)
    selection_accepted.to_parquet(args.output_dir / "selection_overlay_accepted_trades.parquet", index=False)
    eval_accepted.to_parquet(args.output_dir / "eval_overlay_accepted_trades.parquet", index=False)
    baseline_eval_accepted.to_parquet(args.output_dir / "baseline_eval_accepted_trades.parquet", index=False)
    train_decisions.to_parquet(args.output_dir / "train_overlay_decisions.parquet", index=False)
    selection_decisions.to_parquet(args.output_dir / "selection_overlay_decisions.parquet", index=False)
    eval_decisions.to_parquet(args.output_dir / "eval_overlay_decisions.parquet", index=False)
    baseline_eval_decisions.to_parquet(args.output_dir / "baseline_eval_decisions.parquet", index=False)
    train_schedule.to_csv(args.output_dir / "train_recent_head_schedule.csv", index=False)
    selection_schedule.to_csv(args.output_dir / "selection_recent_head_schedule.csv", index=False)
    eval_schedule.to_csv(args.output_dir / "eval_recent_head_schedule.csv", index=False)
    train_summary.to_csv(args.output_dir / "train_overlay_summary.csv", index=False)
    baseline_train_summary.to_csv(args.output_dir / "train_baseline_summary.csv", index=False)
    selection_summary.to_csv(args.output_dir / "selection_overlay_summary.csv", index=False)
    baseline_selection_summary.to_csv(args.output_dir / "selection_baseline_summary.csv", index=False)
    eval_summary.to_csv(args.output_dir / "eval_overlay_summary.csv", index=False)
    baseline_eval_summary.to_csv(args.output_dir / "baseline_eval_summary.csv", index=False)
    eval_by_head.to_csv(args.output_dir / "eval_overlay_by_head.csv", index=False)
    mstc._by_head(baseline_eval_arm, baseline_eval_accepted).to_csv(
        args.output_dir / "baseline_eval_by_head.csv",
        index=False,
    )
    _weekly(eval_accepted, "eval_recent_head_overlay").to_csv(args.output_dir / "eval_overlay_weekly_by_head.csv", index=False)
    _weekly(baseline_eval_accepted, baseline_eval_arm).to_csv(args.output_dir / "baseline_eval_weekly_by_head.csv", index=False)

    manifest = {
        "generated_by": "run_recent_head_activation_optuna",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_broad_candidates": str(args.train_broad_candidates),
        "train_deployable_candidates": str(args.train_deployable_candidates),
        "eval_candidates": str(args.eval_candidates),
        "periods": {
            "train_broad_raw": _period_payload(train_broad_raw),
            "train_deployable_raw": _period_payload(train_deployable_raw),
            "eval_raw": _period_payload(eval_raw),
            "train_broad_ranked": _period_payload(train_broad),
            "train_deployable_ranked": _period_payload(train_deployable),
            "selection_reference": _period_payload(selection_reference),
            "selection_objective": _period_payload(selection_objective),
            "selection_ev_reference": _period_payload(selection_ev_candidates),
            "eval_ranked": _period_payload(eval_candidates),
        },
        "validation_role": {
            "train_broad_ranked": "final_train_replay_and_recent_metric_reference",
            "train_deployable_ranked": "hierarchical_ev_curve_fit",
            "selection_reference": "recent_health_baseline_fit_for_optuna_selection",
            "selection_objective": "chronological_pre_june_optuna_selection_replay",
            "selection_ev_reference": "hierarchical_ev_curve_fit_for_optuna_selection_replay",
            "eval_ranked": "forward_replay_only_not_used_for_optuna_selection",
        },
        "selection_split": selection_split,
        "selection_ev_reference": selection_ev_reference_diag,
        "policy_manifest": str(args.policy_manifest),
        "policy_variant": str(args.policy_variant),
        "policy_manifest_run_id": policy_payload.get("run_id"),
        "rank_reference_run_id": str(args.rank_reference_run_id)
        if rank_contract == "anchor_global_policy_rank_reference"
        else None,
        "rank_contract": rank_contract,
        "rank_scope": _rank_scope(rank_contract),
        "active_t1_rank_contract": rank_contract == "short_boll_timestamp_rank",
        "active_t1_stack_contract": (
            rank_contract == "short_boll_timestamp_rank"
            and sorted(disabled_heads) == ["long_bars", "long_dist"]
        ),
        "active_t1_contract": (
            rank_contract == "short_boll_timestamp_rank"
            and sorted(disabled_heads) == ["long_bars", "long_dist"]
        ),
        "rank_contract_note": (
            "This recent-head activation ablation applies the frozen global-over-time rank reference. "
            "It is a parallel research challenger and not the active provisional T1 short_boll timestamp-rank path."
            if rank_contract == "anchor_global_policy_rank_reference"
            else "This recent-head activation ablation uses the provisional T1 short_boll within-timestamp rank repair. "
            "It matches the active T1 rank contract only when long_bars and long_dist are disabled."
        ),
        "rank_diagnostics": {
            "train_broad": train_rank_diag,
            "train_deployable": train_deployable_rank_diag,
            "eval": eval_rank_diag,
        },
        "disabled_heads": sorted(disabled_heads),
        "n_trials": int(args.n_trials),
        "health_trials": int(args.health_trials),
        "best_health_trial_number": int(health_study.best_trial.number) if health_study is not None else None,
        "best_health_objective": float(health_study.best_value) if health_study is not None else None,
        "best_trial_number": int(study.best_trial.number),
        "best_objective": float(study.best_value),
        "best_params": asdict(best_params),
        "objective_contract": _objective_contract(),
        "selected_selection_objective_components": selected_selection_objective,
        "selected_train_objective_components": selected_train_objective,
        "recent_health_evidence_contract": {
            "uses_outcome_available_timestamp": True,
            "outcome_available_timestamp_source": "exit_timestamp_if_present_else_entry_timestamp",
            "window_filter": "outcome_available_timestamp in [target_timestamp - embargo - lookback, target_timestamp - embargo)",
            "age_weight_timestamp": "outcome_available_timestamp",
            "entry_timestamp_only_filter": False,
            "uses_unmatured_candidate_outcomes": False,
        },
        "train_baseline_summary": baseline_train_summary.iloc[0].to_dict(),
        "selection_baseline_summary": baseline_selection_summary.iloc[0].to_dict(),
        "selection_overlay_summary": selection_summary.iloc[0].to_dict(),
        "train_overlay_summary": train_summary.iloc[0].to_dict(),
        "eval_baseline_summary": baseline_eval_summary.iloc[0].to_dict(),
        "eval_overlay_summary": eval_summary.iloc[0].to_dict(),
        "promotion_gates": {
            "selection": selection_gate,
            "train": train_gate,
            "eval": eval_gate,
        },
        "outputs": {
            "report": str(args.output_dir / "recent_head_activation_optuna_report.md"),
            "trials": str(args.output_dir / "optuna_trials.csv"),
            "selection_overlay_summary": str(args.output_dir / "selection_overlay_summary.csv"),
            "selection_baseline_summary": str(args.output_dir / "selection_baseline_summary.csv"),
            "selection_ev_curve_candidates": str(args.output_dir / "selection_ev_curve_candidates.parquet"),
            "eval_overlay_summary": str(args.output_dir / "eval_overlay_summary.csv"),
            "baseline_eval_summary": str(args.output_dir / "baseline_eval_summary.csv"),
            "eval_overlay_by_head": str(args.output_dir / "eval_overlay_by_head.csv"),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "recent_head_activation_optuna_report.md").write_text(
        _render_report(
            output_dir=args.output_dir,
            best_params=best_params,
            selection_split=selection_split,
            selection_ev_reference_diag=selection_ev_reference_diag,
            selection_summary=selection_summary,
            baseline_selection_summary=baseline_selection_summary,
            selection_gate=selection_gate,
            train_summary=train_summary,
            eval_summary=eval_summary,
            baseline_eval_summary=baseline_eval_summary,
            eval_gate=eval_gate,
            eval_by_head=eval_by_head,
        ),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe({
        "output_dir": str(args.output_dir),
        "best_objective": study.best_value,
        "eval_baseline": baseline_eval_summary.iloc[0].to_dict(),
        "eval_overlay": eval_summary.iloc[0].to_dict(),
        "best_params": asdict(best_params),
    }), indent=2)[:6000])


if __name__ == "__main__":
    main()
