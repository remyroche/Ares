#!/usr/bin/env python3
"""Compare a causal hit-rate surprise dynamic threshold policy.

The policy is intentionally isolated from the market-state archetype pipeline.
It operates on actual policy candidate rows with realized net returns and a
frozen rank/probability score contract.
"""

from __future__ import annotations

import argparse
import json
import math
from bisect import bisect_right, insort
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import pyarrow.parquet as pq


HEADS = ("long_bars", "long_dist", "short_asset", "short_boll")
DEFAULT_META_DRIFT_COLUMNS = (
    "meta_lgbm_inference_drift_score",
    "meta_lgbm_contribution_drift_score",
)
DEFAULT_META_UNCERTAINTY_COLUMNS = (
    "meta_lgbm_uncertainty_score",
    "meta_lgbm_rare_leaf_low_support_score",
)
ROBUST_SUBWINDOWS_V2_PRESET = "robust_subwindows_v2"
ROBUST_SUBWINDOWS_V3_PRESET = "robust_subwindows_v3"
ROBUST_SUBWINDOWS_V4_PRESET = "robust_subwindows_v4"
ROBUST_SUBWINDOWS_V5_PRESET = "robust_subwindows_v5"
ROBUST_SUBWINDOWS_V6_PRESET = "robust_subwindows_v6"
ROBUST_SUBWINDOW_PRESETS = {
    ROBUST_SUBWINDOWS_V2_PRESET,
    ROBUST_SUBWINDOWS_V3_PRESET,
    ROBUST_SUBWINDOWS_V4_PRESET,
    ROBUST_SUBWINDOWS_V5_PRESET,
    ROBUST_SUBWINDOWS_V6_PRESET,
}


def _apply_policy_preset(args: argparse.Namespace) -> None:
    """Apply named policy presets after CLI parsing.

    Presets intentionally override the relevant flags so the report manifest
    captures a reproducible policy variant instead of a loose collection of
    command-line defaults.
    """
    preset = str(getattr(args, "policy_preset", "default"))
    if preset not in ROBUST_SUBWINDOW_PRESETS:
        return

    # True calendar replay: monthly X/W refresh, daily Y refresh.
    args.calendar_only = True
    args.calendar_replay = True
    args.calendar_xw_min_train_days = 90.0
    args.calendar_xw_max_train_days = 183.0
    args.calendar_y_train_days = 28.0

    # No strict final-fit floor. Deployed thresholds remain available only for
    # the fixed baseline and diagnostics.
    args.use_deployed_threshold_floor = False
    args.fallback_rejected_heads_to_deployed = False
    args.require_dynamic_head_improvement_over_deployed = False
    args.require_dynamic_head_tail_not_worse_than_deployed = False

    # Independent per-head monthly X/W, daily Y grid, with asymmetric response:
    # allow some threshold lowering when surprise is strong, but preserve larger
    # room to raise thresholds after bad surprise.
    args.head_optimization_mode = "independent"
    args.threshold_refresh_mode = "grid"
    args.top_rank_floor = 0.70
    args.trials = 120
    args.threshold_grid_size = 201
    args.x_min_days = 1.0
    args.x_max_days = 28.0
    args.w_lower_min = 0.0
    args.w_lower_max = 0.25
    args.w_raise_min = 0.0
    args.w_raise_max = 0.60
    args.require_raise_sensitivity_at_least_lower = True
    args.y_min = -0.50
    args.y_max = 1.50
    args.z_clip = 5.0
    args.require_lowering_confirmation = True

    # Weekly robust-subwindow scoring, slightly looser than the original
    # robust_subwindows guard so daily Y is not forced into cash by a few
    # borderline weeks.
    args.subwindow_days = 7.0
    args.min_subwindows = 2
    args.min_positive_objective_fraction = 0.55
    args.subwindow_q15_floor = -0.25
    args.subwindow_drawdown_floor = -1.50
    args.lambda_iqr = 0.40
    args.lambda_tail = 0.80
    args.subwindow_constraint_penalty = 75.0
    args.min_threshold_selected_count = 15
    args.min_threshold_active_subwindows = 2
    args.per_head_min_objective = 0.0
    args.per_head_min_robust_objective = -0.25

    if preset == ROBUST_SUBWINDOWS_V3_PRESET:
        args.deployed_threshold_soft_prior_strength = 12.0
        args.deployed_threshold_soft_prior_deadband = 0.03
        args.deployed_threshold_soft_prior_power = 2.0
        args.deployed_threshold_soft_prior_activity_weight = 0.50
    if preset in {ROBUST_SUBWINDOWS_V4_PRESET, ROBUST_SUBWINDOWS_V5_PRESET, ROBUST_SUBWINDOWS_V6_PRESET}:
        args.subwindow_constraints_mode = "penalty"
        args.subwindow_days = 5.0
        args.min_subwindows = 4
        args.calendar_y_train_days = 20.0
        args.min_positive_objective_fraction = 0.25
        args.subwindow_q15_floor = -1.00
        args.subwindow_drawdown_floor = -3.00
        args.lambda_iqr = 0.25
        args.lambda_tail = 0.50
        args.subwindow_constraint_penalty = 10.0
        args.min_threshold_selected_count = 0
        args.min_threshold_active_subwindows = 0
        args.per_head_min_objective = -1.0e18
        args.per_head_min_q05_week_pnl = -1.0e18
        args.per_head_min_q15_week_pnl = -1.0e18
        args.per_head_min_robust_objective = -1.0e18
        args.deployed_threshold_soft_prior_strength = 8.0
        args.deployed_threshold_soft_prior_deadband = 0.03
        args.deployed_threshold_soft_prior_power = 2.0
        args.deployed_threshold_soft_prior_activity_weight = 0.25
    if preset == ROBUST_SUBWINDOWS_V5_PRESET:
        args.recent_validation_guard = True
        args.recent_validation_days = 5.0
        args.recent_validation_min_count = 20
        args.recent_validation_min_total_pnl = 0.0
        args.recent_validation_min_hit_rate = 0.30
        args.recent_validation_step = 0.01
    if preset == ROBUST_SUBWINDOWS_V6_PRESET:
        args.threshold_selection_objective = "recent_daily_quantile"
        args.recent_quantile_days = 20.0


@dataclass(frozen=True)
class HeadParams:
    head: str
    x_days: float
    w: float
    y: float
    guarded_y: float
    guard_shift: float
    local_band_pnl: float
    local_band_count: int
    w_lower: float = np.nan
    w_raise: float = np.nan
    slope_lag: int = 1
    forecast_intercept: float = 0.0
    forecast_rho: float = 1.0
    forecast_beta: float = 0.0
    forecast_count_coef: float = 0.0
    forecast_keep_slope: bool = False
    forecast_loss_edge: float = 0.0
    meta_context_enabled: bool = False
    meta_drift_raise: float = 0.0
    meta_drift_floor: float = np.nan
    meta_uncertainty_raise: float = 0.0
    meta_uncertainty_floor: float = np.nan
    meta_context_removed_count: int = 0
    meta_context_removed_total_pnl: float = np.nan
    meta_context_removed_avg_pnl: float = np.nan
    meta_context_removed_gate_passed: bool = False
    meta_badness_cutoff: float = 0.60
    meta_badness_intercept: float = 0.0
    meta_badness_drift_coef: float = 0.0
    meta_badness_uncertainty_coef: float = 0.0
    meta_badness_zneg_coef: float = 0.0
    meta_badness_score_coef: float = 0.0
    meta_badness_train_auc: float = np.nan
    meta_badness_train_rows: int = 0
    meta_badness_temperature: float = 0.08
    meta_badness_pressure_scale: float = 1.0
    deactivated: bool = False
    deactivation_reason: str = ""
    dynamic_rejected: bool = False
    fallback_to_deployed: bool = False
    fallback_threshold: float = np.nan
    head_objective: float = np.nan
    head_q05_week_pnl: float = np.nan
    head_q15_week_pnl: float = np.nan
    deployed_head_objective: float = np.nan
    deployed_head_q05_week_pnl: float = np.nan
    deployed_head_q15_week_pnl: float = np.nan
    deployed_robust_objective: float = np.nan
    subwindow_count: int = 0
    positive_objective_fraction: float = np.nan
    median_subwindow_objective: float = np.nan
    q15_subwindow_objective: float = np.nan
    iqr_subwindow_objective: float = np.nan
    worst_subwindow_drawdown: float = np.nan
    robust_objective: float = np.nan
    passes_subwindow_constraints: bool = False
    recent_validation_guarded: bool = False
    recent_validation_count: int = 0
    recent_validation_total_pnl: float = np.nan
    recent_validation_hit_rate: float = np.nan
    recent_validation_avg_pnl: float = np.nan
    recent_validation_shift: float = 0.0
    recent_validation_reason: str = ""
    quality_gate_enabled: bool = False
    quality_gate_p_hit_floor: float = np.nan
    quality_gate_train_count: int = 0
    quality_gate_train_total_pnl: float = np.nan
    quality_gate_train_hit_rate: float = np.nan
    quality_gate_train_avg_pnl: float = np.nan
    quality_gate_base_count: int = 0
    quality_gate_base_total_pnl: float = np.nan
    quality_gate_base_hit_rate: float = np.nan
    quality_gate_base_avg_pnl: float = np.nan
    quality_gate_reason: str = ""


@dataclass(frozen=True)
class SubwindowObjectiveCache:
    returns: np.ndarray
    hits: np.ndarray
    scores: np.ndarray
    window_id: np.ndarray
    durations_days: np.ndarray
    n_windows: int


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_default(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_default(v) for v in value]
    return value


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path.with_suffix(".csv"), index=False)
    frame.to_parquet(path, index=False)


def _context_linear_config(args: argparse.Namespace) -> dict[str, float]:
    return {
        "context_linear_density_raise": float(getattr(args, "context_linear_density_raise", 0.0)),
        "context_linear_density_floor": float(getattr(args, "context_linear_density_floor", 0.0)),
        "context_linear_relaxation_dampen": float(getattr(args, "context_linear_relaxation_dampen", 1.0)),
        "context_linear_pressure_raise": float(getattr(args, "context_linear_pressure_raise", 0.0)),
        "context_linear_lowering_penalty_strength": float(
            getattr(args, "context_linear_lowering_penalty_strength", 0.0)
        ),
        "context_linear_z_weight": float(getattr(args, "context_linear_z_weight", 1.0)),
        "context_linear_similarity_weight": float(getattr(args, "context_linear_similarity_weight", 1.0)),
        "context_linear_density_weight": float(getattr(args, "context_linear_density_weight", 1.0)),
        "context_linear_drift_weight": float(getattr(args, "context_linear_drift_weight", 1.0)),
        "context_linear_uncertainty_weight": float(getattr(args, "context_linear_uncertainty_weight", 1.0)),
        "context_linear_z_scale": float(getattr(args, "context_linear_z_scale", 1.0)),
        "context_linear_similarity_scale": float(getattr(args, "context_linear_similarity_scale", 5.0)),
        "context_linear_meta_center": float(getattr(args, "context_linear_meta_center", 0.50)),
    }


def _make_tpe_sampler(args: argparse.Namespace) -> optuna.samplers.TPESampler:
    dynamic_meta_space = bool(getattr(args, "use_meta_context_features", False)) and bool(
        getattr(args, "meta_context_tune_enable", False)
    )
    return optuna.samplers.TPESampler(
        seed=int(args.seed),
        multivariate=not dynamic_meta_space,
        warn_independent_sampling=False,
    )


def _meta_context_columns_for_transform(transform: str) -> tuple[str, str]:
    if str(transform) == "causal_percentile":
        return "meta_context_drift_pct_ts", "meta_context_uncertainty_pct_ts"
    return "meta_context_drift_ts", "meta_context_uncertainty_ts"


def _uses_meta_badness_classifier(action_mode: str) -> bool:
    return str(action_mode) in {"badness_classifier_raise", "badness_classifier_soft_raise"}


def _uses_linear_context(action_mode: str) -> bool:
    return str(action_mode) == "linear_context"


def _sigmoid(values: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(values, dtype=float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _causal_expanding_percentile(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    out = np.full(vals.shape[0], 0.5, dtype="float32")
    seen: list[float] = []
    for idx, value in enumerate(vals):
        if seen and np.isfinite(value):
            out[idx] = np.float32(bisect_right(seen, float(value)) / len(seen))
        if np.isfinite(value):
            insort(seen, float(value))
    return out


def _add_causal_meta_context_percentiles(timestamp_context: pd.DataFrame) -> pd.DataFrame:
    out = timestamp_context.copy()
    for col in ("meta_context_drift_ts", "meta_context_uncertainty_ts"):
        if col not in out.columns:
            out[col] = np.float32(0.0)
    out["meta_context_drift_pct_ts"] = np.float32(0.5)
    out["meta_context_uncertainty_pct_ts"] = np.float32(0.5)
    if out.empty:
        return out
    out = out.sort_values(["head", "timestamp"]).reset_index(drop=True)
    for _head, idx in out.groupby("head", sort=False).groups.items():
        loc = np.asarray(list(idx), dtype=int)
        out.loc[loc, "meta_context_drift_pct_ts"] = _causal_expanding_percentile(
            pd.to_numeric(out.loc[loc, "meta_context_drift_ts"], errors="coerce").to_numpy(dtype=float)
        )
        out.loc[loc, "meta_context_uncertainty_pct_ts"] = _causal_expanding_percentile(
            pd.to_numeric(out.loc[loc, "meta_context_uncertainty_ts"], errors="coerce").to_numpy(dtype=float)
        )
    return out


def _context_pressure_from_frame(
    frame: pd.DataFrame,
    params: dict[str, HeadParams],
    *,
    meta_context_transform: str = "raw",
) -> np.ndarray:
    drift_col, uncertainty_col = _meta_context_columns_for_transform(meta_context_transform)
    if drift_col in frame.columns:
        drift = (
            pd.to_numeric(frame[drift_col], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
            .to_numpy(dtype=float)
        )
    else:
        drift = np.zeros(len(frame), dtype=float)
    if uncertainty_col in frame.columns:
        uncertainty = (
            pd.to_numeric(frame[uncertainty_col], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
            .to_numpy(dtype=float)
        )
    else:
        uncertainty = np.zeros(len(frame), dtype=float)
    drift_raise = (
        frame["head"]
        .map({head: (p.meta_drift_raise if p.meta_context_enabled else 0.0) for head, p in params.items()})
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    uncertainty_raise = (
        frame["head"]
        .map({head: (p.meta_uncertainty_raise if p.meta_context_enabled else 0.0) for head, p in params.items()})
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    drift_floor = (
        frame["head"]
        .map({head: p.meta_drift_floor for head, p in params.items()})
        .fillna(np.inf)
        .to_numpy(dtype=float)
    )
    uncertainty_floor = (
        frame["head"]
        .map({head: p.meta_uncertainty_floor for head, p in params.items()})
        .fillna(np.inf)
        .to_numpy(dtype=float)
    )
    pressure = drift_raise * np.maximum(0.0, drift - drift_floor)
    pressure = pressure + uncertainty_raise * np.maximum(0.0, uncertainty - uncertainty_floor)
    return np.nan_to_num(pressure, nan=0.0, posinf=0.0, neginf=0.0)


def _linear_context_density_pressure(
    count_shrink: np.ndarray,
    *,
    density_raise: float,
    density_floor: float,
) -> np.ndarray:
    density = np.asarray(count_shrink, dtype=float)
    return float(density_raise) * np.maximum(0.0, density - float(density_floor))


def _apply_linear_context_to_offset(
    offset: np.ndarray,
    pressure: np.ndarray,
    *,
    relaxation_dampen: float,
    pressure_raise: float,
) -> np.ndarray:
    offset = np.asarray(offset, dtype=float)
    pressure = np.maximum(0.0, np.asarray(pressure, dtype=float))
    relaxation = np.maximum(0.0, -offset)
    dampen = np.clip(float(relaxation_dampen) * pressure, 0.0, 1.0)
    return offset + relaxation * dampen + float(pressure_raise) * pressure


def _available_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def _pick_column(columns: set[str], requested: str | None, candidates: tuple[str, ...]) -> str:
    if requested:
        if requested not in columns:
            raise ValueError(f"Requested column {requested!r} is missing")
        return requested
    for col in candidates:
        if col in columns:
            return col
    raise ValueError(f"None of the candidate columns exist: {candidates}")


def _pick_column_by_coverage(path: Path, columns: set[str], requested: str | None, candidates: tuple[str, ...]) -> str:
    if requested:
        return _pick_column(columns, requested, candidates)
    available = [col for col in candidates if col in columns]
    if not available:
        raise ValueError(f"None of the candidate columns exist: {candidates}")
    if len(available) == 1:
        return available[0]
    try:
        preview = pd.read_parquet(path, columns=available)
        coverage = {
            col: int(pd.to_numeric(preview[col], errors="coerce").replace([np.inf, -np.inf], np.nan).notna().sum())
            for col in available
        }
    except Exception:
        return available[0]
    return max(available, key=lambda col: (coverage.get(col, 0), -available.index(col)))


def _split_columns(value: str | None, defaults: tuple[str, ...]) -> list[str]:
    if value is None or str(value).strip() == "":
        return list(defaults)
    out = [item.strip() for item in str(value).split(",") if item.strip()]
    return out or list(defaults)


def _available_requested_columns(columns: set[str], requested: list[str]) -> tuple[list[str], list[str]]:
    available = [col for col in requested if col in columns]
    missing = [col for col in requested if col not in columns]
    return available, missing


def _composite_meta_context(frame: pd.DataFrame, columns: list[str], *, aggregation: str) -> pd.Series:
    if not columns:
        return pd.Series(0.0, index=frame.index, dtype="float32")
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    numeric = numeric.clip(lower=0.0)
    if aggregation == "max":
        score = numeric.max(axis=1, skipna=True)
    else:
        score = numeric.mean(axis=1, skipna=True)
    return score.fillna(0.0).astype("float32")


def _timestamp_meta_context(frame: pd.DataFrame, aggregation: str) -> pd.DataFrame:
    cols = [col for col in ("meta_context_drift", "meta_context_uncertainty") if col in frame.columns]
    if not cols:
        return pd.DataFrame(columns=["timestamp", "head", "meta_context_drift", "meta_context_uncertainty"])
    grouped = frame[["timestamp", "head", *cols]].copy()
    for col in ("meta_context_drift", "meta_context_uncertainty"):
        if col not in grouped.columns:
            grouped[col] = 0.0
        grouped[col] = pd.to_numeric(grouped[col], errors="coerce").fillna(0.0).clip(lower=0.0)
    groupby = grouped.groupby(["timestamp", "head"], sort=False)
    if aggregation == "max":
        out = groupby[["meta_context_drift", "meta_context_uncertainty"]].max()
    elif aggregation == "q90":
        out = groupby[["meta_context_drift", "meta_context_uncertainty"]].quantile(0.90)
    else:
        out = groupby[["meta_context_drift", "meta_context_uncertainty"]].mean()
    return out.reset_index()


def _meta_context_feature_diagnostics(frame: pd.DataFrame, contract: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, columns in (
        ("meta_drift", contract.get("meta_context_drift_columns", [])),
        ("meta_uncertainty", contract.get("meta_context_uncertainty_columns", [])),
        ("meta_context_composite", ["meta_context_drift", "meta_context_uncertainty"]),
        ("meta_context_timestamp", ["meta_context_drift_ts", "meta_context_uncertainty_ts"]),
        ("meta_context_causal_percentile", ["meta_context_drift_pct_ts", "meta_context_uncertainty_pct_ts"]),
    ):
        for col in columns:
            if col not in frame.columns:
                rows.append(
                    {
                        "family": family,
                        "feature": col,
                        "available": False,
                        "finite_count": 0,
                        "finite_share": 0.0,
                        "mean": np.nan,
                        "std": np.nan,
                        "min": np.nan,
                        "q50": np.nan,
                        "q90": np.nan,
                        "q99": np.nan,
                        "max": np.nan,
                    }
                )
                continue
            s = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            finite = s.dropna()
            rows.append(
                {
                    "family": family,
                    "feature": col,
                    "available": True,
                    "finite_count": int(finite.size),
                    "finite_share": float(finite.size / max(len(frame), 1)),
                    "mean": float(finite.mean()) if finite.size else np.nan,
                    "std": float(finite.std()) if finite.size else np.nan,
                    "min": float(finite.min()) if finite.size else np.nan,
                    "q50": float(finite.quantile(0.50)) if finite.size else np.nan,
                    "q90": float(finite.quantile(0.90)) if finite.size else np.nan,
                    "q99": float(finite.quantile(0.99)) if finite.size else np.nan,
                    "max": float(finite.max()) if finite.size else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _infer_head(strategy_id: Any) -> str:
    value = str(strategy_id)
    for head in HEADS:
        if value.startswith(head):
            return head
    return "unknown"


def _thresholds_from_policy_params(path: Path | None, heads: set[str]) -> dict[str, float]:
    if path is None or not path.exists():
        return {head: 0.70 for head in heads}
    raw = json.loads(path.read_text())
    rules = raw.get("selection_rules", {}) if isinstance(raw, dict) else {}
    guard = rules.get("local_candidate_hit_rate_guard", {}) if isinstance(rules, dict) else {}
    strategies = guard.get("strategies", {}) if isinstance(guard, dict) else {}
    out: dict[str, float] = {}
    for strategy_id, item in strategies.items():
        head = _infer_head(strategy_id)
        if head == "unknown" or head not in heads:
            continue
        enabled = bool(item.get("enabled", True))
        threshold = item.get("deployment_threshold_after_guard", item.get("applied_threshold", item.get("selected_threshold")))
        if enabled and threshold is not None:
            out[head] = float(threshold)
    for head in heads:
        out.setdefault(head, 1.50)
    return out


def _numeric_optional(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").fillna(default).astype("float64")


def _spread_net_return_from_columns(
    frame: pd.DataFrame,
    *,
    return_col: str,
) -> tuple[pd.Series, dict[str, Any]]:
    base_col = "net_return_before_spread" if "net_return_before_spread" in frame.columns else return_col
    base = pd.to_numeric(frame[base_col], errors="coerce").astype("float64")
    full_spread = _numeric_optional(frame, "expected_spread_bps", 0.0).clip(lower=0.0)
    entry_half = _numeric_optional(frame, "expected_half_spread_bps", np.nan)
    if "spread_cost_bps" in frame.columns:
        spread_cost = _numeric_optional(frame, "spread_cost_bps", np.nan)
        entry_half = entry_half.where(np.isfinite(entry_half), spread_cost)
    exit_half = _numeric_optional(frame, "exit_spread_cost_bps", np.nan)
    if "exit_quote_half_spread_bps" in frame.columns:
        exit_quote = _numeric_optional(frame, "exit_quote_half_spread_bps", np.nan)
        exit_half = exit_half.where(np.isfinite(exit_half), exit_quote)

    full_half = full_spread / 2.0
    entry_half = entry_half.where(np.isfinite(entry_half), full_half)
    exit_half = exit_half.where(np.isfinite(exit_half), full_half)

    # Some historical ledgers exported all explicit half-spread columns as zero
    # while retaining a per-asset full spread. Treat that as missing so old
    # artifacts can be evaluated with the same spread-net semantics as new ones.
    explicit_spread_cols = [
        col
        for col in (
            "expected_half_spread_bps",
            "spread_cost_bps",
            "exit_spread_cost_bps",
            "exit_quote_half_spread_bps",
        )
        if col in frame.columns
    ]
    zero_explicit_share = 0.0
    fallback_to_full_spread = False
    if explicit_spread_cols and full_spread.gt(0.0).any():
        explicit_abs = pd.concat(
            [_numeric_optional(frame, col, 0.0).abs() for col in explicit_spread_cols],
            axis=1,
        )
        zero_explicit = explicit_abs.max(axis=1).le(1e-12) & full_spread.gt(0.0)
        zero_explicit_share = float(zero_explicit.mean()) if len(zero_explicit) else 0.0
        if zero_explicit_share > 0.95:
            entry_half = entry_half.where(~zero_explicit, full_half)
            exit_half = exit_half.where(~zero_explicit, full_half)
            fallback_to_full_spread = True

    entry_half = entry_half.fillna(0.0).clip(lower=0.0)
    exit_half = exit_half.fillna(0.0).clip(lower=0.0)
    spread_bps = entry_half + exit_half
    spread_net = base - spread_bps / 10_000.0
    diagnostics = {
        "return_base_col": base_col,
        "spread_adjustment_bps_mean": float(spread_bps.mean()) if len(spread_bps) else 0.0,
        "spread_adjustment_bps_q50": float(spread_bps.quantile(0.50)) if len(spread_bps) else 0.0,
        "spread_adjustment_bps_q90": float(spread_bps.quantile(0.90)) if len(spread_bps) else 0.0,
        "spread_adjustment_uses_full_spread_fallback": bool(fallback_to_full_spread),
        "zero_explicit_spread_col_share": float(zero_explicit_share),
    }
    return spread_net, diagnostics


def load_candidates(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(args.candidates)
    columns = set(_available_columns(path))
    timestamp_col = _pick_column(columns, "timestamp", ("timestamp",))
    strategy_col = _pick_column(columns, args.strategy_col, ("head", "strategy_id"))
    score_col = _pick_column(
        columns,
        args.score_col,
        ("normalized_rank_score", "policy_rank_pct", "rank_pct", "auction_rank_score", "calibrated_score"),
    )
    rank_col = _pick_column_by_coverage(path, columns, args.rank_col, ("policy_rank_pct", "rank_pct", "strategy_rank_pct", score_col))
    p_hit_col = _pick_column_by_coverage(
        path,
        columns,
        args.p_hit_col,
        ("simple_policy_calibrated_good_trade_prob", "calibrated_score", "reliability_blend_score", score_col),
    )
    return_col = _pick_column(columns, args.return_col, ("net_return", "fixed_return_net_after_cost", "gross_return"))
    weight_col = args.surprise_weight_col if args.surprise_weight_col in columns else None
    ev_col = None
    for col in (
        args.ev_col,
        "uncertainty_adjusted_ev_net_return",
        "simple_policy_calibrated_expected_net_gain",
        "simple_grid_net_ev_bps",
    ):
        if col and col in columns:
            ev_col = col
            break
    requested_meta_drift = _split_columns(args.meta_context_drift_cols, DEFAULT_META_DRIFT_COLUMNS)
    requested_meta_uncertainty = _split_columns(args.meta_context_uncertainty_cols, DEFAULT_META_UNCERTAINTY_COLUMNS)
    meta_drift_cols, missing_meta_drift_cols = _available_requested_columns(columns, requested_meta_drift)
    meta_uncertainty_cols, missing_meta_uncertainty_cols = _available_requested_columns(columns, requested_meta_uncertainty)
    extra = [c for c in ("symbol", "side") if c in columns]
    read_cols = [timestamp_col, strategy_col, score_col, rank_col, p_hit_col, return_col] + extra
    spread_return_cols = [
        col
        for col in (
            "net_return_before_spread",
            "expected_spread_bps",
            "expected_half_spread_bps",
            "spread_cost_bps",
            "exit_spread_cost_bps",
            "exit_quote_half_spread_bps",
        )
        if col in columns
    ]
    if bool(args.spread_adjust_returns):
        read_cols.extend(spread_return_cols)
    if weight_col:
        read_cols.append(weight_col)
    if ev_col:
        read_cols.append(ev_col)
    if bool(args.use_meta_context_features):
        read_cols.extend(meta_drift_cols)
        read_cols.extend(meta_uncertainty_cols)
    read_cols = list(dict.fromkeys(read_cols))
    frame = pd.read_parquet(path, columns=read_cols).copy()
    frame["timestamp"] = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
    frame["strategy_key"] = frame[strategy_col].astype(str)
    frame["head"] = frame["strategy_key"].map(_infer_head) if strategy_col != "head" else frame[strategy_col].astype(str)
    frame["score"] = pd.to_numeric(frame[score_col], errors="coerce")
    frame["rank"] = pd.to_numeric(frame[rank_col], errors="coerce")
    frame["p_hit"] = pd.to_numeric(frame[p_hit_col], errors="coerce").clip(1e-6, 1.0 - 1e-6)
    spread_return_diagnostics: dict[str, Any] = {}
    if bool(args.spread_adjust_returns):
        frame["net_return"], spread_return_diagnostics = _spread_net_return_from_columns(frame, return_col=return_col)
    else:
        frame["net_return"] = pd.to_numeric(frame[return_col], errors="coerce")
    frame["hit"] = frame["net_return"].gt(0.0).astype(float)
    frame["surprise_weight"] = (
        pd.to_numeric(frame[weight_col], errors="coerce").fillna(1.0).clip(0.0, 100.0)
        if weight_col
        else 1.0
    )
    frame["ev"] = pd.to_numeric(frame[ev_col], errors="coerce") if ev_col else np.nan
    if bool(args.use_meta_context_features):
        frame["meta_context_drift"] = _composite_meta_context(
            frame,
            meta_drift_cols,
            aggregation=str(args.meta_context_feature_aggregation),
        )
        frame["meta_context_uncertainty"] = _composite_meta_context(
            frame,
            meta_uncertainty_cols,
            aggregation=str(args.meta_context_feature_aggregation),
        )
    else:
        frame["meta_context_drift"] = np.float32(0.0)
        frame["meta_context_uncertainty"] = np.float32(0.0)
    timestamp_context = _timestamp_meta_context(frame, str(args.meta_context_timestamp_aggregation)).rename(
        columns={
            "meta_context_drift": "meta_context_drift_ts",
            "meta_context_uncertainty": "meta_context_uncertainty_ts",
        }
    )
    timestamp_context = _add_causal_meta_context_percentiles(timestamp_context)
    frame = frame.merge(timestamp_context, on=["timestamp", "head"], how="left", sort=False)
    frame["meta_context_drift_ts"] = (
        pd.to_numeric(frame["meta_context_drift_ts"], errors="coerce").fillna(0.0).clip(lower=0.0).astype("float32")
    )
    frame["meta_context_uncertainty_ts"] = (
        pd.to_numeric(frame["meta_context_uncertainty_ts"], errors="coerce").fillna(0.0).clip(lower=0.0).astype("float32")
    )
    frame["meta_context_drift_pct_ts"] = (
        pd.to_numeric(frame["meta_context_drift_pct_ts"], errors="coerce").fillna(0.5).clip(0.0, 1.0).astype("float32")
    )
    frame["meta_context_uncertainty_pct_ts"] = (
        pd.to_numeric(frame["meta_context_uncertainty_pct_ts"], errors="coerce").fillna(0.5).clip(0.0, 1.0).astype("float32")
    )
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["timestamp", "head", "score", "rank", "p_hit", "net_return"]).copy()
    frame = frame.loc[frame["head"].isin(HEADS)].sort_values(["timestamp", "head", "score"], ascending=[True, True, False])
    frame = frame.reset_index(drop=True)
    contract = {
        "timestamp_col": timestamp_col,
        "strategy_col": strategy_col,
        "score_col": score_col,
        "rank_col": rank_col,
        "p_hit_col": p_hit_col,
        "return_col": return_col,
        "spread_adjust_returns": bool(args.spread_adjust_returns),
        "spread_return_columns": spread_return_cols,
        **spread_return_diagnostics,
        "surprise_weight_col": weight_col or "constant_1",
        "ev_col": ev_col or "",
        "meta_context_enabled": bool(args.use_meta_context_features),
        "meta_context_feature_aggregation": str(args.meta_context_feature_aggregation),
        "meta_context_timestamp_aggregation": str(args.meta_context_timestamp_aggregation),
        "meta_context_transform": str(args.meta_context_transform),
        "meta_context_requested_drift_columns": requested_meta_drift,
        "meta_context_requested_uncertainty_columns": requested_meta_uncertainty,
        "meta_context_drift_columns": meta_drift_cols,
        "meta_context_uncertainty_columns": meta_uncertainty_cols,
        "meta_context_missing_drift_columns": missing_meta_drift_cols,
        "meta_context_missing_uncertainty_columns": missing_meta_uncertainty_cols,
    }
    return frame, contract


def _ewm_shifted(series: pd.Series, halflife_days: float) -> pd.Series:
    series = series.sort_index()
    if series.empty:
        return series
    return (
        series.ewm(
            halflife=pd.Timedelta(days=float(halflife_days)),
            times=series.index,
            adjust=True,
        )
        .mean()
        .shift(1)
    )


def build_surprise(
    frame: pd.DataFrame,
    *,
    halflife_days_by_head: dict[str, float],
    top_rank_floor: float,
    z_clip: float,
    slope_lag_by_head: dict[str, int] | None = None,
    count_shrink_n0: float = 20.0,
) -> pd.DataFrame:
    out_parts: list[pd.DataFrame] = []
    for head, group in frame.groupby("head", sort=True):
        ts_ns = pd.to_datetime(group["timestamp"], utc=True, errors="coerce").astype("int64").to_numpy(dtype=np.int64)
        idx_ns = np.unique(ts_ns)
        idx = pd.DatetimeIndex(pd.to_datetime(idx_ns, utc=True))
        eligible = group.loc[group["rank"].ge(float(top_rank_floor))]
        if eligible.empty:
            agg = pd.DataFrame(index=idx, data={"num": 0.0, "var": 0.0, "count": 0.0})
        else:
            weight = pd.to_numeric(eligible["surprise_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
            p_hit = pd.to_numeric(eligible["p_hit"], errors="coerce").fillna(0.5).clip(1e-6, 1.0 - 1e-6).to_numpy(dtype=float)
            hit = pd.to_numeric(eligible["hit"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            eligible_ts_ns = (
                pd.to_datetime(eligible["timestamp"], utc=True, errors="coerce")
                .astype("int64")
                .to_numpy(dtype=np.int64)
            )
            positions = np.searchsorted(idx_ns, eligible_ts_ns)
            n_idx = len(idx_ns)
            agg = pd.DataFrame(
                index=idx,
                data={
                    "num": np.bincount(positions, weights=weight * (hit - p_hit), minlength=n_idx),
                    "var": np.bincount(positions, weights=np.square(weight) * p_hit * (1.0 - p_hit), minlength=n_idx),
                    "count": np.bincount(positions, weights=np.ones(len(positions), dtype=float), minlength=n_idx),
                },
            )
        x_days = float(halflife_days_by_head.get(head, 7.0))
        ewma_num = _ewm_shifted(agg["num"], x_days).fillna(0.0)
        ewma_var = _ewm_shifted(agg["var"], x_days).fillna(0.0)
        ewma_count = _ewm_shifted(agg["count"], x_days).fillna(0.0)
        z_raw = ewma_num / np.sqrt(ewma_var + 1e-12)
        count_shrink = ewma_count / (ewma_count + max(float(count_shrink_n0), 0.0))
        z_level = (z_raw * count_shrink).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-float(z_clip), float(z_clip))
        slope_lag = max(int((slope_lag_by_head or {}).get(head, 1)), 1)
        slope = (z_level - z_level.shift(slope_lag)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out_parts.append(
            pd.DataFrame(
                {
                    "timestamp": idx,
                    "head": head,
                    "ewma_num": ewma_num.to_numpy(dtype=float),
                    "ewma_var": ewma_var.to_numpy(dtype=float),
                    "ewma_count": ewma_count.to_numpy(dtype=float),
                    "z_raw": z_raw.to_numpy(dtype=float),
                    "count_shrink": count_shrink.to_numpy(dtype=float),
                    "z_eff": z_level.to_numpy(dtype=float),
                    "slope": slope.to_numpy(dtype=float),
                    "slope_lag": int(slope_lag),
                }
            )
        )
    return pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame()


def _fit_surprise_forecast_coefficients(surprise: pd.DataFrame, args: argparse.Namespace) -> dict[str, float | bool]:
    if surprise.empty or str(args.surprise_forecast_mode) != "slope":
        return {
            "intercept": 0.0,
            "rho": 1.0,
            "beta": 0.0,
            "count_coef": 0.0,
            "keep_slope": False,
            "loss_edge": 0.0,
        }
    ordered = surprise.sort_values("timestamp").copy()
    z = pd.to_numeric(ordered["z_eff"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    slope = pd.to_numeric(ordered["slope"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    count = pd.to_numeric(ordered["ewma_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    target = np.roll(z, -1)
    valid = np.arange(len(z) - 1)
    if len(valid) < int(args.forecast_min_rows):
        return {
            "intercept": 0.0,
            "rho": 1.0,
            "beta": 0.0,
            "count_coef": 0.0,
            "keep_slope": False,
            "loss_edge": 0.0,
        }
    X = np.column_stack(
        [
            np.ones(len(valid), dtype=float),
            z[valid],
            slope[valid],
            np.log1p(np.maximum(count[valid], 0.0)),
        ]
    )
    y = target[valid]
    split = max(int(len(valid) * float(args.forecast_train_fraction)), 1)
    split = min(split, len(valid) - 1)
    X_train, y_train = X[:split], y[:split]
    X_valid, y_valid = X[split:], y[split:]
    if len(y_valid) < max(5, int(args.forecast_min_valid_rows)):
        X_train, y_train = X, y
        X_valid, y_valid = X, y
    alpha = float(args.forecast_ridge_alpha)
    penalty = np.eye(X_train.shape[1], dtype=float) * alpha
    penalty[0, 0] = 0.0
    try:
        coef = np.linalg.solve(X_train.T @ X_train + penalty, X_train.T @ y_train)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(X_train.T @ X_train + penalty) @ X_train.T @ y_train
    coef[1] = float(np.clip(coef[1], float(args.forecast_rho_min), float(args.forecast_rho_max)))
    pred = X_valid @ coef
    baseline = X_valid[:, 1]
    slope_loss = float(np.mean(np.square(y_valid - pred)))
    baseline_loss = float(np.mean(np.square(y_valid - baseline)))
    edge = baseline_loss - slope_loss
    keep_slope = bool(edge > float(args.min_forecast_edge))
    if not keep_slope:
        coef = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=float)
    return {
        "intercept": float(coef[0]),
        "rho": float(coef[1]),
        "beta": float(coef[2]),
        "count_coef": float(coef[3]),
        "keep_slope": keep_slope,
        "loss_edge": float(edge),
    }


def _binary_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(score, dtype=float)
    valid = np.isfinite(y) & np.isfinite(s)
    y = y[valid] > 0.5
    s = s[valid]
    pos = y
    neg = ~y
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = pd.Series(s).rank(method="average").to_numpy(dtype=float)
    return float((np.sum(ranks[pos]) - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def _badness_probability_from_frame(
    frame: pd.DataFrame,
    params: dict[str, HeadParams],
    z_eff: np.ndarray,
    *,
    meta_context_transform: str = "raw",
) -> np.ndarray:
    drift_col, uncertainty_col = _meta_context_columns_for_transform(meta_context_transform)
    if drift_col in frame.columns:
        drift = pd.to_numeric(frame[drift_col], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    else:
        drift = np.zeros(len(frame), dtype=float)
    if uncertainty_col in frame.columns:
        uncertainty = pd.to_numeric(frame[uncertainty_col], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    else:
        uncertainty = np.zeros(len(frame), dtype=float)
    zneg = np.maximum(0.0, -np.asarray(z_eff, dtype=float))
    score = pd.to_numeric(frame["score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    head = frame["head"]
    intercept = head.map({h: p.meta_badness_intercept for h, p in params.items()}).fillna(0.0).to_numpy(dtype=float)
    drift_coef = head.map({h: p.meta_badness_drift_coef for h, p in params.items()}).fillna(0.0).to_numpy(dtype=float)
    uncertainty_coef = head.map({h: p.meta_badness_uncertainty_coef for h, p in params.items()}).fillna(0.0).to_numpy(dtype=float)
    zneg_coef = head.map({h: p.meta_badness_zneg_coef for h, p in params.items()}).fillna(0.0).to_numpy(dtype=float)
    score_coef = head.map({h: p.meta_badness_score_coef for h, p in params.items()}).fillna(0.0).to_numpy(dtype=float)
    p_bad = intercept + drift_coef * drift + uncertainty_coef * uncertainty + zneg_coef * zneg + score_coef * score
    return np.clip(np.nan_to_num(p_bad, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _threshold_vector(
    frame: pd.DataFrame,
    params: dict[str, HeadParams],
    surprise: pd.DataFrame,
    *,
    deployed_thresholds: dict[str, float] | None = None,
    use_deployed_threshold_floor: bool = True,
    surprise_forecast_mode: str = "level",
    slope_tolerance: float = 0.0,
    z_cap: float = 5.0,
    require_lowering_confirmation: bool = True,
    meta_context_action_mode: str = "raise",
    meta_context_bad_z_threshold: float = 0.0,
    meta_context_transform: str = "raw",
    context_linear_density_raise: float = 0.0,
    context_linear_density_floor: float = 0.0,
    context_linear_relaxation_dampen: float = 1.0,
    context_linear_pressure_raise: float = 0.0,
) -> np.ndarray:
    n_rows = len(frame)
    single_param = next(iter(params.values())) if len(params) == 1 else None
    single_head = False
    if single_param is not None and n_rows:
        heads = frame["head"].to_numpy()
        single_head = bool(np.all(heads == heads[0]) and str(heads[0]) == str(single_param.head))
    surprise_keyed = surprise.drop_duplicates(["timestamp", "head"], keep="last")
    merged = frame[["timestamp", "head"]].merge(
        surprise_keyed[["timestamp", "head", "z_eff", "slope", "ewma_count", "count_shrink"]],
        on=["timestamp", "head"],
        how="left",
        sort=False,
    )
    z_eff = pd.to_numeric(merged["z_eff"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    slope = pd.to_numeric(merged["slope"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    count = pd.to_numeric(merged["ewma_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    count_shrink = pd.to_numeric(merged["count_shrink"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if single_head and single_param is not None:
        y = np.full(n_rows, float(single_param.guarded_y), dtype=float)
        w = np.full(n_rows, float(single_param.w), dtype=float)
    else:
        y = frame["head"].map({head: p.guarded_y for head, p in params.items()}).to_numpy(dtype=float)
        w = frame["head"].map({head: p.w for head, p in params.items()}).to_numpy(dtype=float)
    if str(surprise_forecast_mode) == "slope":
        if single_head and single_param is not None:
            intercept = np.full(n_rows, float(single_param.forecast_intercept), dtype=float)
            rho = np.full(n_rows, float(single_param.forecast_rho), dtype=float)
            beta = np.full(n_rows, float(single_param.forecast_beta), dtype=float)
            count_coef = np.full(n_rows, float(single_param.forecast_count_coef), dtype=float)
            w_lower = np.full(n_rows, float(single_param.w_lower), dtype=float)
            w_raise = np.full(n_rows, float(single_param.w_raise), dtype=float)
        else:
            intercept = frame["head"].map({head: p.forecast_intercept for head, p in params.items()}).to_numpy(dtype=float)
            rho = frame["head"].map({head: p.forecast_rho for head, p in params.items()}).to_numpy(dtype=float)
            beta = frame["head"].map({head: p.forecast_beta for head, p in params.items()}).to_numpy(dtype=float)
            count_coef = frame["head"].map({head: p.forecast_count_coef for head, p in params.items()}).to_numpy(dtype=float)
            w_lower = frame["head"].map({head: p.w_lower for head, p in params.items()}).fillna(pd.Series(w, index=frame.index)).to_numpy(dtype=float)
            w_raise = frame["head"].map({head: p.w_raise for head, p in params.items()}).fillna(pd.Series(w, index=frame.index)).to_numpy(dtype=float)
        predicted = intercept + rho * z_eff + beta * slope + count_coef * np.log1p(np.maximum(count, 0.0))
        predicted = np.clip(predicted * count_shrink, -float(z_cap), float(z_cap))
        lower_signal = np.maximum(0.0, predicted)
        if require_lowering_confirmation:
            allow_lowering = (predicted > 0.0) & (z_eff > 0.0) & (slope >= -float(slope_tolerance))
            lower_signal = np.where(allow_lowering, lower_signal, 0.0)
        raise_signal = np.minimum(0.0, predicted)
        threshold = y - w_lower * lower_signal - w_raise * raise_signal
    else:
        if single_head and single_param is not None:
            w_lower = np.full(n_rows, float(single_param.w_lower), dtype=float)
            w_raise = np.full(n_rows, float(single_param.w_raise), dtype=float)
        else:
            w_lower = frame["head"].map({head: p.w_lower for head, p in params.items()}).fillna(pd.Series(w, index=frame.index)).to_numpy(dtype=float)
            w_raise = frame["head"].map({head: p.w_raise for head, p in params.items()}).fillna(pd.Series(w, index=frame.index)).to_numpy(dtype=float)
        lower_signal = np.maximum(0.0, z_eff)
        raise_signal = np.minimum(0.0, z_eff)
        threshold = y - w_lower * lower_signal - w_raise * raise_signal
    if use_deployed_threshold_floor and deployed_thresholds:
        if single_head and single_param is not None:
            floor = np.full(n_rows, float(deployed_thresholds.get(str(single_param.head), -0.50)), dtype=float)
        else:
            floor = frame["head"].map(deployed_thresholds).fillna(-0.50).to_numpy(dtype=float)
        threshold = np.maximum(floor, threshold)
    drift_col, uncertainty_col = _meta_context_columns_for_transform(meta_context_transform)
    if drift_col in frame.columns or uncertainty_col in frame.columns:
        pressure = _context_pressure_from_frame(frame, params, meta_context_transform=meta_context_transform)
        mode = str(meta_context_action_mode)
        bad_z_threshold = float(meta_context_bad_z_threshold)
        if _uses_linear_context(mode):
            pressure = pressure + _linear_context_density_pressure(
                count_shrink,
                density_raise=float(context_linear_density_raise),
                density_floor=float(context_linear_density_floor),
            )
            threshold = y + _apply_linear_context_to_offset(
                threshold - y,
                pressure,
                relaxation_dampen=float(context_linear_relaxation_dampen),
                pressure_raise=float(context_linear_pressure_raise),
            )
        elif mode == "bad_surprise_raise":
            pressure = np.where(z_eff < bad_z_threshold, pressure, 0.0)
            threshold = threshold + pressure
        elif mode == "badness_classifier_raise":
            p_bad = _badness_probability_from_frame(
                frame,
                params,
                z_eff,
                meta_context_transform=meta_context_transform,
            )
            cutoff = (
                frame["head"]
                .map({head: p.meta_badness_cutoff for head, p in params.items()})
                .fillna(1.0)
                .to_numpy(dtype=float)
            )
            pressure = np.where(p_bad >= cutoff, pressure, 0.0)
            threshold = threshold + pressure
        elif mode == "badness_classifier_soft_raise":
            p_bad = _badness_probability_from_frame(
                frame,
                params,
                z_eff,
                meta_context_transform=meta_context_transform,
            )
            cutoff = (
                frame["head"]
                .map({head: p.meta_badness_cutoff for head, p in params.items()})
                .fillna(1.0)
                .to_numpy(dtype=float)
            )
            temperature = (
                frame["head"]
                .map({head: p.meta_badness_temperature for head, p in params.items()})
                .fillna(0.08)
                .to_numpy(dtype=float)
            )
            pressure_scale = (
                frame["head"]
                .map({head: p.meta_badness_pressure_scale for head, p in params.items()})
                .fillna(1.0)
                .to_numpy(dtype=float)
            )
            gate = _sigmoid((p_bad - cutoff) / np.maximum(temperature, 1e-6))
            threshold = threshold + pressure * np.clip(pressure_scale, 0.0, 10.0) * gate
        elif mode == "dampen_relaxation":
            relaxation = np.maximum(0.0, y - threshold)
            damp = 1.0 - np.exp(-np.maximum(0.0, pressure))
            threshold = threshold + relaxation * damp
        else:
            threshold = threshold + pressure
    return np.clip(threshold, -0.50, 1.50)


def _mask_from_params(
    frame: pd.DataFrame,
    params: dict[str, HeadParams],
    surprise: pd.DataFrame,
    *,
    deployed_thresholds: dict[str, float] | None = None,
    use_deployed_threshold_floor: bool = True,
    surprise_forecast_mode: str = "level",
    slope_tolerance: float = 0.0,
    z_cap: float = 5.0,
    require_lowering_confirmation: bool = True,
    meta_context_action_mode: str = "raise",
    meta_context_bad_z_threshold: float = 0.0,
    meta_context_transform: str = "raw",
    context_linear_density_raise: float = 0.0,
    context_linear_density_floor: float = 0.0,
    context_linear_relaxation_dampen: float = 1.0,
    context_linear_pressure_raise: float = 0.0,
) -> np.ndarray:
    threshold = _threshold_vector(
        frame,
        params,
        surprise,
        deployed_thresholds=deployed_thresholds,
        use_deployed_threshold_floor=use_deployed_threshold_floor,
        surprise_forecast_mode=surprise_forecast_mode,
        slope_tolerance=slope_tolerance,
        z_cap=z_cap,
        require_lowering_confirmation=require_lowering_confirmation,
        meta_context_action_mode=meta_context_action_mode,
        meta_context_bad_z_threshold=meta_context_bad_z_threshold,
        meta_context_transform=meta_context_transform,
        context_linear_density_raise=context_linear_density_raise,
        context_linear_density_floor=context_linear_density_floor,
        context_linear_relaxation_dampen=context_linear_relaxation_dampen,
        context_linear_pressure_raise=context_linear_pressure_raise,
    )
    mask = frame["score"].to_numpy(dtype=float) >= threshold
    quality_floor_map = {
        head: p.quality_gate_p_hit_floor
        for head, p in params.items()
        if bool(p.quality_gate_enabled) and np.isfinite(p.quality_gate_p_hit_floor)
    }
    if quality_floor_map:
        floor = frame["head"].map(quality_floor_map).fillna(-np.inf).to_numpy(dtype=float)
        p_hit = pd.to_numeric(frame["p_hit"], errors="coerce").fillna(-np.inf).to_numpy(dtype=float)
        mask &= p_hit >= floor
    return mask


def _build_surprise_for_params(frame: pd.DataFrame, params: dict[str, HeadParams], args: argparse.Namespace) -> pd.DataFrame:
    return build_surprise(
        frame,
        halflife_days_by_head={head: param.x_days for head, param in params.items()},
        slope_lag_by_head={head: param.slope_lag for head, param in params.items()},
        top_rank_floor=float(args.top_rank_floor),
        z_clip=float(args.z_clip),
        count_shrink_n0=float(args.surprise_count_shrink_n0),
    )


def _mask_from_params_with_args(
    frame: pd.DataFrame,
    params: dict[str, HeadParams],
    surprise: pd.DataFrame,
    args: argparse.Namespace,
    *,
    deployed_thresholds: dict[str, float] | None = None,
) -> np.ndarray:
    return _mask_from_params(
        frame,
        params,
        surprise,
        deployed_thresholds=deployed_thresholds,
        use_deployed_threshold_floor=bool(args.use_deployed_threshold_floor),
        surprise_forecast_mode=str(args.surprise_forecast_mode),
        slope_tolerance=float(args.slope_tolerance),
        z_cap=float(args.z_clip),
        require_lowering_confirmation=bool(args.require_lowering_confirmation),
        meta_context_action_mode=str(args.meta_context_action_mode),
        meta_context_bad_z_threshold=float(args.meta_context_bad_z_threshold),
        meta_context_transform=str(args.meta_context_transform),
        context_linear_density_raise=float(getattr(args, "context_linear_density_raise", 0.0)),
        context_linear_density_floor=float(getattr(args, "context_linear_density_floor", 0.0)),
        context_linear_relaxation_dampen=float(getattr(args, "context_linear_relaxation_dampen", 1.0)),
        context_linear_pressure_raise=float(getattr(args, "context_linear_pressure_raise", 0.0)),
    )


def _deployed_threshold_soft_prior_penalty(
    frame: pd.DataFrame,
    params: dict[str, HeadParams],
    surprise: pd.DataFrame,
    args: argparse.Namespace,
    *,
    deployed_thresholds: dict[str, float] | None = None,
) -> float:
    strength = float(getattr(args, "deployed_threshold_soft_prior_strength", 0.0))
    if strength <= 0.0 or not deployed_thresholds or frame.empty:
        return 0.0
    thresholds = _threshold_vector(
        frame,
        params,
        surprise,
        deployed_thresholds=deployed_thresholds,
        use_deployed_threshold_floor=False,
        surprise_forecast_mode=str(args.surprise_forecast_mode),
        slope_tolerance=float(args.slope_tolerance),
        z_cap=float(args.z_clip),
        require_lowering_confirmation=bool(args.require_lowering_confirmation),
        meta_context_action_mode=str(args.meta_context_action_mode),
        meta_context_bad_z_threshold=float(args.meta_context_bad_z_threshold),
        meta_context_transform=str(args.meta_context_transform),
        context_linear_density_raise=float(getattr(args, "context_linear_density_raise", 0.0)),
        context_linear_density_floor=float(getattr(args, "context_linear_density_floor", 0.0)),
        context_linear_relaxation_dampen=float(getattr(args, "context_linear_relaxation_dampen", 1.0)),
        context_linear_pressure_raise=float(getattr(args, "context_linear_pressure_raise", 0.0)),
    )
    deployed = frame["head"].map(deployed_thresholds).fillna(np.nan).to_numpy(dtype=float)
    valid = np.isfinite(deployed)
    if not np.any(valid):
        return 0.0
    deadband = max(float(getattr(args, "deployed_threshold_soft_prior_deadband", 0.0)), 0.0)
    power = max(float(getattr(args, "deployed_threshold_soft_prior_power", 2.0)), 1.0)
    below = np.maximum(0.0, deployed[valid] - thresholds[valid] - deadband)
    if not np.any(below > 0.0):
        return 0.0
    active = frame["score"].to_numpy(dtype=float)[valid] >= thresholds[valid]
    activity_weight = max(float(getattr(args, "deployed_threshold_soft_prior_activity_weight", 0.0)), 0.0)
    return float(strength * np.mean(np.power(below, power)) * (1.0 + activity_weight * float(np.mean(active))))


def _attach_forecast_coefficients(param: HeadParams, surprise: pd.DataFrame, args: argparse.Namespace) -> HeadParams:
    coeffs = _fit_surprise_forecast_coefficients(surprise, args)
    return replace(
        param,
        forecast_intercept=float(coeffs["intercept"]),
        forecast_rho=float(coeffs["rho"]),
        forecast_beta=float(coeffs["beta"]),
        forecast_count_coef=float(coeffs["count_coef"]),
        forecast_keep_slope=bool(coeffs["keep_slope"]),
        forecast_loss_edge=float(coeffs["loss_edge"]),
    )


def _fit_meta_badness_classifier(
    head_frame: pd.DataFrame,
    param: HeadParams,
    surprise: pd.DataFrame,
    args: argparse.Namespace,
    *,
    deployed_thresholds: dict[str, float] | None = None,
) -> HeadParams:
    if not bool(param.meta_context_enabled) or not _uses_meta_badness_classifier(str(args.meta_context_action_mode)):
        return param
    surprise_keyed = surprise.drop_duplicates(["timestamp", "head"], keep="last")
    merged = head_frame.merge(
        surprise_keyed[["timestamp", "head", "z_eff"]],
        on=["timestamp", "head"],
        how="left",
        sort=False,
    )
    merged["z_eff"] = pd.to_numeric(merged["z_eff"], errors="coerce").fillna(0.0)
    no_context = replace(
        param,
        meta_context_enabled=False,
        meta_drift_raise=0.0,
        meta_drift_floor=np.nan,
        meta_uncertainty_raise=0.0,
        meta_uncertainty_floor=np.nan,
    )
    base_threshold = _threshold_vector(
        head_frame,
        {param.head: no_context},
        surprise,
        deployed_thresholds=deployed_thresholds,
        use_deployed_threshold_floor=bool(args.use_deployed_threshold_floor),
        surprise_forecast_mode=str(args.surprise_forecast_mode),
        slope_tolerance=float(args.slope_tolerance),
        z_cap=float(args.z_clip),
        require_lowering_confirmation=bool(args.require_lowering_confirmation),
        meta_context_action_mode="raise",
        meta_context_bad_z_threshold=float(args.meta_context_bad_z_threshold),
        meta_context_transform=str(args.meta_context_transform),
    )
    train_mask = head_frame["score"].to_numpy(dtype=float) >= base_threshold
    min_rows = int(args.meta_badness_min_train_rows)
    if int(np.sum(train_mask)) < min_rows:
        train_mask = head_frame["rank"].to_numpy(dtype=float) >= float(args.top_rank_floor)
    if int(np.sum(train_mask)) < min_rows:
        train_mask = np.ones(len(head_frame), dtype=bool)
    fit = merged.loc[np.asarray(train_mask, dtype=bool)].copy()
    if len(fit) < min_rows:
        return replace(param, meta_badness_train_rows=int(len(fit)), meta_badness_train_auc=np.nan)
    y = fit["net_return"].le(0.0).astype(float).to_numpy(dtype=float)
    if np.unique(y).size < 2:
        return replace(param, meta_badness_train_rows=int(len(fit)), meta_badness_train_auc=np.nan)
    drift_col, uncertainty_col = _meta_context_columns_for_transform(str(args.meta_context_transform))
    for col in (drift_col, uncertainty_col):
        if col not in fit.columns:
            fit[col] = 0.0
    raw = np.column_stack(
        [
            pd.to_numeric(fit[drift_col], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float),
            pd.to_numeric(fit[uncertainty_col], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float),
            np.maximum(0.0, -pd.to_numeric(fit["z_eff"], errors="coerce").fillna(0.0).to_numpy(dtype=float)),
            pd.to_numeric(fit["score"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        ]
    )
    mean = np.nanmean(raw, axis=0)
    std = np.nanstd(raw, axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    X = np.column_stack([np.ones(raw.shape[0], dtype=float), (raw - mean) / std])
    p_bad = float(np.mean(y))
    weights = np.where(y > 0.5, 0.5 / max(p_bad, 1e-6), 0.5 / max(1.0 - p_bad, 1e-6))
    root_w = np.sqrt(np.clip(weights, 0.0, 100.0))
    Xw = X * root_w[:, None]
    yw = y * root_w
    penalty = np.eye(X.shape[1], dtype=float) * float(args.meta_badness_ridge_alpha)
    penalty[0, 0] = 0.0
    try:
        coef = np.linalg.solve(Xw.T @ Xw + penalty, Xw.T @ yw)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(Xw.T @ Xw + penalty) @ Xw.T @ yw
    raw_coef = coef[1:] / std
    intercept = float(coef[0] - np.sum(coef[1:] * mean / std))
    pred = np.clip(intercept + raw @ raw_coef, 0.0, 1.0)
    return replace(
        param,
        meta_badness_intercept=intercept,
        meta_badness_drift_coef=float(raw_coef[0]),
        meta_badness_uncertainty_coef=float(raw_coef[1]),
        meta_badness_zneg_coef=float(raw_coef[2]),
        meta_badness_score_coef=float(raw_coef[3]),
        meta_badness_train_auc=_binary_auc(y, pred),
        meta_badness_train_rows=int(len(fit)),
    )


def _rolling_week_pnl(selected: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    if selected.empty:
        daily = pd.Series(0.0, index=pd.date_range(start.floor("D"), end.ceil("D"), freq="D", tz="UTC"))
    else:
        pnl = selected.groupby(selected["timestamp"].dt.floor("D"))["net_return"].sum().sort_index()
        daily = pnl.reindex(pd.date_range(start.floor("D"), end.ceil("D"), freq="D", tz="UTC"), fill_value=0.0)
    return daily.rolling(7, min_periods=1).sum()


def _policy_metrics(frame: pd.DataFrame, mask: np.ndarray, *, policy: str) -> dict[str, Any]:
    selected = frame.loc[np.asarray(mask, dtype=bool)].copy()
    start = pd.Timestamp(frame["timestamp"].min())
    end = pd.Timestamp(frame["timestamp"].max())
    rolling_week = _rolling_week_pnl(selected, start, end)
    total = float(selected["net_return"].sum()) if not selected.empty else 0.0
    q05 = float(rolling_week.quantile(0.05)) if len(rolling_week) else 0.0
    q15 = float(rolling_week.quantile(0.15)) if len(rolling_week) else 0.0
    avg_week = float(rolling_week.mean()) if len(rolling_week) else 0.0
    return {
        "policy": policy,
        "candidate_count": int(len(frame)),
        "selected_count": int(len(selected)),
        "selected_share": float(len(selected) / max(len(frame), 1)),
        "active_head_count": int(selected["head"].nunique()) if not selected.empty else 0,
        "total_net_pnl": total,
        "avg_net_pnl_per_trade": float(selected["net_return"].mean()) if not selected.empty else np.nan,
        "hit_rate": float(selected["net_return"].gt(0.0).mean()) if not selected.empty else np.nan,
        "avg_pnl_per_rolling_week": avg_week,
        "q05_rolling_week_pnl": q05,
        "q15_rolling_week_pnl": q15,
        "objective": avg_week + q05 + q15,
        "mean_selected_ev": float(selected["ev"].mean()) if "ev" in selected and selected["ev"].notna().any() else np.nan,
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
    }


def _policy_metrics_by_fold(frame: pd.DataFrame, mask: np.ndarray, *, policy: str, fold: int) -> dict[str, Any]:
    out = _policy_metrics(frame, mask, policy=policy)
    out["fold"] = int(fold)
    return out


def _by_head_metrics(frame: pd.DataFrame, mask: np.ndarray, *, policy: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected_mask = pd.Series(np.asarray(mask, dtype=bool), index=frame.index)
    for head, group in frame.groupby("head", sort=True):
        selected = group.loc[selected_mask.loc[group.index]]
        rows.append(
            {
                "policy": policy,
                "head": head,
                "candidate_count": int(len(group)),
                "selected_count": int(len(selected)),
                "selected_share": float(len(selected) / max(len(group), 1)),
                "total_net_pnl": float(selected["net_return"].sum()) if len(selected) else 0.0,
                "avg_net_pnl_per_trade": float(selected["net_return"].mean()) if len(selected) else np.nan,
                "hit_rate": float(selected["net_return"].gt(0.0).mean()) if len(selected) else np.nan,
                "mean_threshold_score_selected": float(selected["score"].mean()) if len(selected) else np.nan,
                "mean_ev_selected": float(selected["ev"].mean()) if len(selected) and selected["ev"].notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _monthly_metrics(frame: pd.DataFrame, mask: np.ndarray, *, policy: str) -> pd.DataFrame:
    selected = frame.loc[np.asarray(mask, dtype=bool)].copy()
    if selected.empty:
        return pd.DataFrame(columns=["policy", "month", "selected_count", "total_net_pnl", "hit_rate"])
    selected["month"] = selected["timestamp"].dt.to_period("M").astype(str)
    out = (
        selected.groupby("month", sort=True)
        .agg(
            selected_count=("net_return", "size"),
            total_net_pnl=("net_return", "sum"),
            avg_net_pnl_per_trade=("net_return", "mean"),
            hit_rate=("net_return", lambda s: float(pd.Series(s).gt(0.0).mean())),
        )
        .reset_index()
    )
    out.insert(0, "policy", policy)
    return out


def _monthly_by_head_metrics(frame: pd.DataFrame, mask: np.ndarray, *, policy: str) -> pd.DataFrame:
    selected = frame.loc[np.asarray(mask, dtype=bool)].copy()
    if selected.empty:
        return pd.DataFrame(
            columns=[
                "policy",
                "month",
                "head",
                "selected_count",
                "total_net_pnl",
                "avg_net_pnl_per_trade",
                "hit_rate",
                "mean_threshold_score_selected",
                "mean_ev_selected",
            ]
        )
    selected["month"] = selected["timestamp"].dt.to_period("M").astype(str)
    out = (
        selected.groupby(["month", "head"], sort=True)
        .agg(
            selected_count=("net_return", "size"),
            total_net_pnl=("net_return", "sum"),
            avg_net_pnl_per_trade=("net_return", "mean"),
            hit_rate=("net_return", lambda s: float(pd.Series(s).gt(0.0).mean())),
            mean_threshold_score_selected=("score", "mean"),
            mean_ev_selected=("ev", "mean"),
        )
        .reset_index()
    )
    out.insert(0, "policy", policy)
    return out


def _weekly_frame(frame: pd.DataFrame, mask: np.ndarray, *, policy: str) -> pd.DataFrame:
    selected = frame.loc[np.asarray(mask, dtype=bool)].copy()
    roll = _rolling_week_pnl(selected, pd.Timestamp(frame["timestamp"].min()), pd.Timestamp(frame["timestamp"].max()))
    return pd.DataFrame({"policy": policy, "date": roll.index, "rolling_week_net_pnl": roll.to_numpy(dtype=float)})


def _max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    cumulative = np.cumsum(returns.astype(float, copy=False))
    peaks = np.maximum.accumulate(np.r_[0.0, cumulative])[:-1]
    drawdowns = cumulative - peaks
    return float(np.min(drawdowns)) if drawdowns.size else 0.0


def _build_subwindow_objective_cache(frame: pd.DataFrame, args: argparse.Namespace) -> SubwindowObjectiveCache:
    if frame.empty:
        return SubwindowObjectiveCache(
            returns=np.asarray([], dtype=float),
            hits=np.asarray([], dtype=float),
            scores=np.asarray([], dtype=float),
            window_id=np.asarray([], dtype=np.int32),
            durations_days=np.asarray([float(args.subwindow_days)], dtype=float),
            n_windows=1,
        )
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    start = pd.Timestamp(timestamps.min()).floor("D")
    end = pd.Timestamp(timestamps.max()).ceil("D")
    step_seconds = max(float(args.subwindow_days) * 86400.0, 1e-6)
    elapsed = (timestamps - start).dt.total_seconds().fillna(0.0).to_numpy(dtype=float)
    window_id = np.floor(elapsed / step_seconds).astype(np.int32)
    n_windows = int(window_id.max()) + 1 if window_id.size else 1
    durations = np.empty(n_windows, dtype=float)
    for window in range(n_windows):
        window_start = start + pd.Timedelta(seconds=window * step_seconds)
        window_end = min(start + pd.Timedelta(seconds=(window + 1) * step_seconds), end)
        durations[window] = max((window_end - window_start).total_seconds() / 86400.0, 1e-6)
    return SubwindowObjectiveCache(
        returns=pd.to_numeric(frame["net_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        hits=frame["net_return"].gt(0.0).to_numpy(dtype=float),
        scores=pd.to_numeric(frame["score"], errors="coerce").fillna(-np.inf).to_numpy(dtype=float),
        window_id=window_id,
        durations_days=durations,
        n_windows=n_windows,
    )


def evaluate_policy_by_subwindow(
    frame: pd.DataFrame,
    mask: np.ndarray,
    *,
    subwindow_days: float,
    policy: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "policy",
                "subwindow_start",
                "subwindow_end",
                "candidate_count",
                "selected_count",
                "total_net_pnl",
                "objective",
                "hit_rate",
                "max_drawdown",
            ]
        )
    selected_mask = pd.Series(np.asarray(mask, dtype=bool), index=frame.index)
    start = pd.Timestamp(frame["timestamp"].min()).floor("D")
    end = pd.Timestamp(frame["timestamp"].max()).ceil("D")
    step = pd.Timedelta(days=max(float(subwindow_days), 1e-6))
    rows: list[dict[str, Any]] = []
    window_start = start
    while window_start < end:
        window_end = min(window_start + step, end)
        window_frame = frame.loc[frame["timestamp"].ge(window_start) & frame["timestamp"].lt(window_end)]
        window_selected = window_frame.loc[selected_mask.loc[window_frame.index]]
        duration_days = max((window_end - window_start).total_seconds() / 86400.0, 1e-6)
        returns = window_selected.sort_values("timestamp")["net_return"].to_numpy(dtype=float)
        total = float(np.sum(returns)) if returns.size else 0.0
        rows.append(
            {
                "policy": policy,
                "subwindow_start": window_start.isoformat(),
                "subwindow_end": window_end.isoformat(),
                "candidate_count": int(len(window_frame)),
                "selected_count": int(len(window_selected)),
                "total_net_pnl": total,
                "objective": float(total * 7.0 / duration_days),
                "hit_rate": float(window_selected["net_return"].gt(0.0).mean()) if len(window_selected) else np.nan,
                "max_drawdown": _max_drawdown(returns),
            }
        )
        window_start = window_end
    return pd.DataFrame(rows)


def _subwindow_summary_from_objectives(
    objectives: np.ndarray,
    drawdowns: np.ndarray,
    args: argparse.Namespace,
    *,
    policy: str,
    subwindow_metrics: pd.DataFrame | None = None,
) -> dict[str, Any]:
    objectives = np.asarray(objectives, dtype=float)
    drawdowns = np.asarray(drawdowns, dtype=float)
    if objectives.size == 0:
        objectives = np.asarray([0.0], dtype=float)
    if drawdowns.size == 0:
        drawdowns = np.asarray([0.0], dtype=float)
    q15 = float(np.quantile(objectives, 0.15))
    q25 = float(np.quantile(objectives, 0.25))
    q75 = float(np.quantile(objectives, 0.75))
    iqr = float(q75 - q25)
    median = float(np.median(objectives))
    positive_fraction = float(np.mean(objectives > 0.0))
    worst_drawdown = float(np.min(drawdowns)) if drawdowns.size else 0.0
    robust = float(
        median
        - float(args.lambda_iqr) * iqr
        - float(args.lambda_tail) * abs(min(0.0, q15))
    )
    min_fraction = float(args.min_positive_objective_fraction)
    q15_floor = float(args.subwindow_q15_floor)
    drawdown_floor = float(args.subwindow_drawdown_floor)
    enough_subwindows = len(objectives) >= int(args.min_subwindows)
    passes = (
        enough_subwindows
        and positive_fraction >= min_fraction
        and median > 0.0
        and q15 >= q15_floor
        and worst_drawdown >= drawdown_floor
    )
    constraint_gap = 0.0
    constraint_gap += max(0.0, min_fraction - positive_fraction)
    constraint_gap += max(0.0, -median)
    constraint_gap += max(0.0, q15_floor - q15)
    constraint_gap += max(0.0, drawdown_floor - worst_drawdown)
    if not enough_subwindows:
        constraint_gap += float(int(args.min_subwindows) - len(objectives) + 1)
    optimization_score = robust if passes else robust - float(args.subwindow_constraint_penalty) * constraint_gap
    return {
        "subwindow_count": int(len(objectives)),
        "positive_objective_fraction": positive_fraction,
        "median_subwindow_objective": median,
        "q15_subwindow_objective": q15,
        "iqr_subwindow_objective": iqr,
        "worst_subwindow_drawdown": worst_drawdown,
        "robust_objective": robust,
        "passes_subwindow_constraints": bool(passes),
        "optimization_score": float(optimization_score),
        "subwindow_metrics": subwindow_metrics if subwindow_metrics is not None else pd.DataFrame(),
    }


def _subwindow_summary_from_cache(
    cache: SubwindowObjectiveCache,
    mask: np.ndarray,
    args: argparse.Namespace,
    *,
    policy: str,
    include_metrics: bool = False,
) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    if cache.returns.size == 0:
        return _subwindow_summary_from_objectives(np.asarray([0.0]), np.asarray([0.0]), args, policy=policy)
    selected_returns = np.where(mask, cache.returns, 0.0)
    selected_hits = np.where(mask, cache.hits, 0.0)
    selected_count = np.bincount(cache.window_id, weights=mask.astype(float), minlength=cache.n_windows)
    total_by_window = np.bincount(cache.window_id, weights=selected_returns, minlength=cache.n_windows)
    hit_by_window = np.bincount(cache.window_id, weights=selected_hits, minlength=cache.n_windows)
    objectives = total_by_window * 7.0 / cache.durations_days
    drawdowns = np.zeros(cache.n_windows, dtype=float)
    for window in range(cache.n_windows):
        active = (cache.window_id == window) & mask
        drawdowns[window] = _max_drawdown(cache.returns[active])
    metrics = None
    if include_metrics:
        hit_rate = np.divide(hit_by_window, selected_count, out=np.full(cache.n_windows, np.nan), where=selected_count > 0)
        metrics = pd.DataFrame(
            {
                "policy": policy,
                "subwindow": np.arange(cache.n_windows, dtype=int),
                "selected_count": selected_count.astype(int),
                "total_net_pnl": total_by_window,
                "objective": objectives,
                "hit_rate": hit_rate,
                "max_drawdown": drawdowns,
            }
        )
    return _subwindow_summary_from_objectives(objectives, drawdowns, args, policy=policy, subwindow_metrics=metrics)


def _subwindow_summary(
    frame: pd.DataFrame,
    mask: np.ndarray,
    args: argparse.Namespace,
    *,
    policy: str,
) -> dict[str, Any]:
    cache = _build_subwindow_objective_cache(frame, args)
    return _subwindow_summary_from_cache(cache, mask, args, policy=policy, include_metrics=True)


def _threshold_offset_vector(
    frame: pd.DataFrame,
    param: HeadParams,
    surprise: pd.DataFrame,
    args: argparse.Namespace,
) -> np.ndarray:
    surprise_keyed = surprise.drop_duplicates(["timestamp", "head"], keep="last")
    merged = frame[["timestamp", "head"]].merge(
        surprise_keyed[["timestamp", "head", "z_eff", "slope", "ewma_count", "count_shrink"]],
        on=["timestamp", "head"],
        how="left",
        sort=False,
    )
    z_eff = pd.to_numeric(merged["z_eff"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    slope = pd.to_numeric(merged["slope"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    count = pd.to_numeric(merged["ewma_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    count_shrink = pd.to_numeric(merged["count_shrink"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if str(args.surprise_forecast_mode) != "slope":
        w_lower = float(param.w_lower if np.isfinite(param.w_lower) else param.w)
        w_raise = float(param.w_raise if np.isfinite(param.w_raise) else param.w)
        offset = -w_lower * np.maximum(0.0, z_eff) - w_raise * np.minimum(0.0, z_eff)
    else:
        predicted = (
            float(param.forecast_intercept)
            + float(param.forecast_rho) * z_eff
            + float(param.forecast_beta) * slope
            + float(param.forecast_count_coef) * np.log1p(np.maximum(count, 0.0))
        )
        predicted = np.clip(predicted * count_shrink, -float(args.z_clip), float(args.z_clip))
        lower_signal = np.maximum(0.0, predicted)
        if bool(args.require_lowering_confirmation):
            allow_lowering = (predicted > 0.0) & (z_eff > 0.0) & (slope >= -float(args.slope_tolerance))
            lower_signal = np.where(allow_lowering, lower_signal, 0.0)
        raise_signal = np.minimum(0.0, predicted)
        w_lower = float(param.w_lower if np.isfinite(param.w_lower) else param.w)
        w_raise = float(param.w_raise if np.isfinite(param.w_raise) else param.w)
        offset = -w_lower * lower_signal - w_raise * raise_signal
    drift_col, uncertainty_col = _meta_context_columns_for_transform(str(args.meta_context_transform))
    if bool(param.meta_context_enabled) and (drift_col in frame.columns or uncertainty_col in frame.columns):
        pressure = _context_pressure_from_frame(
            frame,
            {param.head: param},
            meta_context_transform=str(args.meta_context_transform),
        )
        mode = str(args.meta_context_action_mode)
        if _uses_linear_context(mode):
            pressure = pressure + _linear_context_density_pressure(
                count_shrink,
                density_raise=float(getattr(args, "context_linear_density_raise", 0.0)),
                density_floor=float(getattr(args, "context_linear_density_floor", 0.0)),
            )
            offset = _apply_linear_context_to_offset(
                offset,
                pressure,
                relaxation_dampen=float(getattr(args, "context_linear_relaxation_dampen", 1.0)),
                pressure_raise=float(getattr(args, "context_linear_pressure_raise", 0.0)),
            )
        elif mode == "bad_surprise_raise":
            pressure = np.where(z_eff < float(args.meta_context_bad_z_threshold), pressure, 0.0)
            offset = offset + pressure
        elif mode == "badness_classifier_raise":
            p_bad = _badness_probability_from_frame(
                frame,
                {param.head: param},
                z_eff,
                meta_context_transform=str(args.meta_context_transform),
            )
            pressure = np.where(p_bad >= float(param.meta_badness_cutoff), pressure, 0.0)
            offset = offset + pressure
        elif mode == "badness_classifier_soft_raise":
            p_bad = _badness_probability_from_frame(
                frame,
                {param.head: param},
                z_eff,
                meta_context_transform=str(args.meta_context_transform),
            )
            gate = _sigmoid(
                (p_bad - float(param.meta_badness_cutoff))
                / max(float(param.meta_badness_temperature), 1e-6)
            )
            offset = offset + pressure * float(np.clip(param.meta_badness_pressure_scale, 0.0, 10.0)) * gate
        elif mode == "dampen_relaxation":
            relaxation = np.maximum(0.0, -offset)
            damp = 1.0 - np.exp(-np.maximum(0.0, pressure))
            offset = offset + relaxation * damp
        else:
            offset = offset + pressure
    return offset


def _score_y_grid(
    frame: pd.DataFrame,
    cache: SubwindowObjectiveCache,
    param: HeadParams,
    surprise: pd.DataFrame,
    y_values: np.ndarray,
    args: argparse.Namespace,
    *,
    deployed_thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    y_values = np.asarray(y_values, dtype=float)
    offset = _threshold_offset_vector(frame, param, surprise, args)
    floor = -0.50
    if bool(args.use_deployed_threshold_floor) and deployed_thresholds:
        floor = float(deployed_thresholds.get(param.head, -0.50))
    thresholds = np.clip(np.maximum(floor, offset[:, None] + y_values[None, :]), -0.50, 1.50)
    masks = cache.scores[:, None] >= thresholds
    n_grid = y_values.size
    totals = np.zeros((cache.n_windows, n_grid), dtype=float)
    counts = np.zeros((cache.n_windows, n_grid), dtype=float)
    hits = np.zeros((cache.n_windows, n_grid), dtype=float)
    for window in range(cache.n_windows):
        rows = cache.window_id == window
        if not np.any(rows):
            continue
        window_masks = masks[rows, :]
        totals[window, :] = (cache.returns[rows, None] * window_masks).sum(axis=0)
        counts[window, :] = window_masks.sum(axis=0)
        hits[window, :] = (cache.hits[rows, None] * window_masks).sum(axis=0)
    objectives = totals * (7.0 / cache.durations_days[:, None])
    drawdowns = np.zeros((cache.n_windows, n_grid), dtype=float)
    for grid_idx in range(n_grid):
        mask = masks[:, grid_idx]
        for window in range(cache.n_windows):
            drawdowns[window, grid_idx] = _max_drawdown(cache.returns[(cache.window_id == window) & mask])
    q15 = np.quantile(objectives, 0.15, axis=0)
    q25 = np.quantile(objectives, 0.25, axis=0)
    q75 = np.quantile(objectives, 0.75, axis=0)
    iqr = q75 - q25
    median = np.median(objectives, axis=0)
    positive_fraction = np.mean(objectives > 0.0, axis=0)
    worst_drawdown = np.min(drawdowns, axis=0)
    robust = median - float(args.lambda_iqr) * iqr - float(args.lambda_tail) * np.abs(np.minimum(0.0, q15))
    enough_subwindows = cache.n_windows >= int(args.min_subwindows)
    passes = (
        enough_subwindows
        & (positive_fraction >= float(args.min_positive_objective_fraction))
        & (median > 0.0)
        & (q15 >= float(args.subwindow_q15_floor))
        & (worst_drawdown >= float(args.subwindow_drawdown_floor))
    )
    constraint_gap = np.maximum(0.0, float(args.min_positive_objective_fraction) - positive_fraction)
    constraint_gap += np.maximum(0.0, -median)
    constraint_gap += np.maximum(0.0, float(args.subwindow_q15_floor) - q15)
    constraint_gap += np.maximum(0.0, float(args.subwindow_drawdown_floor) - worst_drawdown)
    if not enough_subwindows:
        constraint_gap += float(int(args.min_subwindows) - cache.n_windows + 1)
    optimization_score = np.where(passes, robust, robust - float(args.subwindow_constraint_penalty) * constraint_gap)
    selected_count = counts.sum(axis=0)
    total_pnl = totals.sum(axis=0)
    hit_rate = np.divide(hits.sum(axis=0), selected_count, out=np.full(n_grid, np.nan), where=selected_count > 0)
    active_subwindow_count = np.sum(counts > 0, axis=0)
    min_selected_count = int(getattr(args, "min_threshold_selected_count", 0))
    min_active_subwindows = int(getattr(args, "min_threshold_active_subwindows", 0))
    activity_ok = (selected_count >= min_selected_count) & (active_subwindow_count >= min_active_subwindows)
    if min_selected_count > 0 or min_active_subwindows > 0:
        activity_gap = np.maximum(0.0, min_selected_count - selected_count) / max(float(min_selected_count), 1.0)
        activity_gap += np.maximum(0.0, min_active_subwindows - active_subwindow_count) / max(float(min_active_subwindows), 1.0)
        optimization_score = np.where(
            activity_ok,
            optimization_score,
            optimization_score - float(args.subwindow_constraint_penalty) * activity_gap,
        )
        passes = passes & activity_ok
    soft_prior_penalty = np.zeros(n_grid, dtype=float)
    prior_strength = float(getattr(args, "deployed_threshold_soft_prior_strength", 0.0))
    if prior_strength > 0.0 and deployed_thresholds and param.head in deployed_thresholds:
        deployed_threshold = float(deployed_thresholds[param.head])
        deadband = max(float(getattr(args, "deployed_threshold_soft_prior_deadband", 0.0)), 0.0)
        power = max(float(getattr(args, "deployed_threshold_soft_prior_power", 2.0)), 1.0)
        activity_weight = max(float(getattr(args, "deployed_threshold_soft_prior_activity_weight", 0.0)), 0.0)
        below = np.maximum(0.0, deployed_threshold - thresholds - deadband)
        activity = np.mean(masks, axis=0)
        soft_prior_penalty = prior_strength * np.mean(np.power(below, power), axis=0) * (1.0 + activity_weight * activity)
        optimization_score = optimization_score - soft_prior_penalty
    return pd.DataFrame(
        {
            "y": y_values,
            "value": optimization_score,
            "deployed_threshold_soft_prior_penalty": soft_prior_penalty,
            "robust_objective": robust,
            "passes_subwindow_constraints": passes,
            "positive_objective_fraction": positive_fraction,
            "median_subwindow_objective": median,
            "q15_subwindow_objective": q15,
            "iqr_subwindow_objective": iqr,
            "worst_subwindow_drawdown": worst_drawdown,
            "active_subwindow_count": active_subwindow_count.astype(int),
            "passes_activity_constraints": activity_ok,
            "selected_count": selected_count.astype(int),
            "total_net_pnl": total_pnl,
            "hit_rate": hit_rate,
        }
    )


def _recent_quantile_day_weights(
    day_index: pd.Index,
    end: pd.Timestamp,
    args: argparse.Namespace,
) -> np.ndarray:
    mode = str(getattr(args, "recent_quantile_weight_mode", "uniform"))
    n_days = len(day_index)
    if n_days == 0 or mode == "uniform":
        return np.ones(n_days, dtype=float)
    if mode != "bucket":
        raise ValueError(f"Unsupported recent_quantile_weight_mode={mode!r}")
    days = pd.to_datetime(day_index, utc=True, errors="coerce")
    end_day = pd.Timestamp(end).floor("D")
    age_days = (end_day - days.floor("D")).days.to_numpy(dtype=float)
    weights = np.full(n_days, float(getattr(args, "recent_quantile_weight_older", 1.0)), dtype=float)
    weights = np.where(age_days < 14.0, float(getattr(args, "recent_quantile_weight_prev_7", 1.0)), weights)
    weights = np.where(age_days < 7.0, float(getattr(args, "recent_quantile_weight_last_7", 1.0)), weights)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if float(weights.sum()) <= 0.0:
        weights = np.ones(n_days, dtype=float)
    return weights


def _weighted_quantile_by_column(values: np.ndarray, weights: np.ndarray, q: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be a 2D array")
    n_rows, n_cols = values.shape
    if n_rows == 0:
        return np.zeros(n_cols, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.size != n_rows:
        raise ValueError("weights length must match values rows")
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if float(weights.sum()) <= 0.0:
        weights = np.ones(n_rows, dtype=float)
    q = float(np.clip(q, 0.0, 1.0))
    order = np.argsort(values, axis=0, kind="mergesort")
    sorted_values = np.take_along_axis(values, order, axis=0)
    tiled_weights = np.broadcast_to(weights[:, None], values.shape)
    sorted_weights = np.take_along_axis(tiled_weights, order, axis=0)
    cumulative = np.cumsum(sorted_weights, axis=0)
    cutoffs = q * np.sum(sorted_weights, axis=0)
    idx = np.argmax(cumulative >= cutoffs[None, :], axis=0)
    return sorted_values[idx, np.arange(n_cols)]


def _weighted_mean_by_column(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be a 2D array")
    n_rows, n_cols = values.shape
    if n_rows == 0:
        return np.zeros(n_cols, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.size != n_rows:
        raise ValueError("weights length must match values rows")
    valid_values = np.isfinite(values)
    clean_values = np.where(valid_values, values, 0.0)
    clean_weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    weighted = clean_values * clean_weights[:, None]
    denom = (valid_values * clean_weights[:, None]).sum(axis=0)
    return np.divide(weighted.sum(axis=0), denom, out=np.zeros(n_cols, dtype=float), where=denom > 0.0)


def _daily_similarity_feature_frame(
    recent: pd.DataFrame,
    surprise: pd.DataFrame,
) -> pd.DataFrame:
    if recent.empty:
        return pd.DataFrame()
    keyed = surprise.drop_duplicates(["timestamp", "head"], keep="last")
    cols = ["timestamp", "head", "z_eff", "ewma_count", "count_shrink"]
    available = [col for col in cols if col in keyed.columns]
    merged = recent.merge(
        keyed[available],
        on=["timestamp", "head"],
        how="left",
        sort=False,
    )
    merged["day"] = pd.to_datetime(merged["timestamp"], utc=True, errors="coerce").dt.floor("D")
    spread_col = "spread_cost_bps" if "spread_cost_bps" in merged.columns else None
    optional_context_cols = [
        col
        for col in (
            "meta_context_drift_ts",
            "meta_context_uncertainty_ts",
            "meta_context_drift_pct_ts",
            "meta_context_uncertainty_pct_ts",
        )
        if col in merged.columns
    ]
    rows: list[dict[str, float | pd.Timestamp]] = []
    for day, day_frame in merged.groupby("day", sort=True):
        if pd.isna(day):
            continue
        score = pd.to_numeric(day_frame["score"], errors="coerce")
        row: dict[str, float | pd.Timestamp] = {
            "day": pd.Timestamp(day),
            "candidate_count_log": float(np.log1p(len(day_frame))),
            "score_mean": float(score.mean()) if len(score) else 0.0,
            "score_q90": float(score.quantile(0.90)) if len(score) else 0.0,
        }
        for col in ("z_eff", "ewma_count", "count_shrink"):
            values = pd.to_numeric(day_frame.get(col, pd.Series(index=day_frame.index, dtype=float)), errors="coerce")
            row[f"{col}_mean"] = float(values.mean()) if values.notna().any() else 0.0
            row[f"{col}_last"] = float(values.dropna().iloc[-1]) if values.notna().any() else 0.0
        if spread_col is not None:
            spread = pd.to_numeric(day_frame[spread_col], errors="coerce")
            row["spread_cost_bps_mean"] = float(spread.mean()) if spread.notna().any() else 0.0
        for col in optional_context_cols:
            values = pd.to_numeric(day_frame[col], errors="coerce")
            row[f"{col}_mean"] = float(values.mean()) if values.notna().any() else 0.0
            row[f"{col}_last"] = float(values.dropna().iloc[-1]) if values.notna().any() else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("day").reset_index(drop=True)


def _similar_day_weights(
    context: pd.DataFrame,
    *,
    query_recent_days: int,
    top_k: int,
    min_days: int,
    temperature: float,
) -> np.ndarray:
    n_days = len(context)
    weights = np.zeros(n_days, dtype=float)
    if n_days <= max(int(query_recent_days), 0):
        return weights
    query_recent_days = max(int(query_recent_days), 1)
    top_k = max(int(top_k), 1)
    min_days = max(int(min_days), 1)
    historical_end = max(n_days - query_recent_days, 0)
    if historical_end < min_days:
        return weights
    feature_cols = [col for col in context.columns if col != "day"]
    if not feature_cols:
        return weights
    features = context[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)
    matrix = features.to_numpy(dtype=float)
    hist = matrix[:historical_end, :]
    query = matrix[historical_end:, :].mean(axis=0)
    center = np.nanmedian(hist, axis=0)
    scale = np.nanmedian(np.abs(hist - center[None, :]), axis=0) * 1.4826
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, np.nanstd(hist, axis=0))
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
    hist_norm = (hist - center[None, :]) / scale[None, :]
    query_norm = (query - center) / scale
    dist = np.sqrt(np.nanmean(np.square(hist_norm - query_norm[None, :]), axis=1))
    valid = np.isfinite(dist)
    if int(np.sum(valid)) < min_days:
        return weights
    order = np.argsort(np.where(valid, dist, np.inf), kind="mergesort")
    chosen = order[: min(top_k, int(np.sum(valid)))]
    if chosen.size < min_days:
        return weights
    tau = max(float(temperature), 1e-6)
    chosen_weights = np.exp(-dist[chosen] / tau)
    if float(chosen_weights.sum()) <= 0.0:
        chosen_weights = np.ones_like(chosen_weights, dtype=float)
    weights[chosen] = chosen_weights
    return weights


def _query_context_scalar(context: pd.DataFrame, column: str, *, query_recent_days: int, default: float = 0.0) -> float:
    if context.empty or column not in context.columns:
        return float(default)
    query_recent_days = max(int(query_recent_days), 1)
    values = pd.to_numeric(context[column].tail(query_recent_days), errors="coerce").replace([np.inf, -np.inf], np.nan)
    if not values.notna().any():
        return float(default)
    return float(values.mean())


def _query_context_density_stress(context: pd.DataFrame, *, query_recent_days: int) -> float:
    if context.empty or "candidate_count_log" not in context.columns:
        return 0.0
    query_recent_days = max(int(query_recent_days), 1)
    values = (
        pd.to_numeric(context["candidate_count_log"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(values) <= query_recent_days + 2:
        return 0.0
    historical = values.iloc[: -query_recent_days]
    query = values.iloc[-query_recent_days:].mean()
    if historical.empty or not np.isfinite(query):
        return 0.0
    center = float(historical.median())
    mad = float(np.median(np.abs(historical.to_numpy(dtype=float) - center))) * 1.4826
    scale = mad if np.isfinite(mad) and mad > 1e-9 else float(historical.std(ddof=0))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    return float(max(0.0, (query - center) / scale))


def _linear_context_lowering_penalty(
    *,
    context: pd.DataFrame,
    thresholds: np.ndarray,
    param: HeadParams,
    args: argparse.Namespace,
    similarity_prior_objective: np.ndarray,
    deployed_thresholds: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    n_grid = int(np.asarray(similarity_prior_objective).size)
    zeros = np.zeros(n_grid, dtype=float)
    strength = float(getattr(args, "context_linear_lowering_penalty_strength", 0.0))
    if strength <= 0.0 or n_grid == 0:
        return zeros, {
            "context_linear_score_mean": 0.0,
            "context_linear_z_signal": 0.0,
            "context_linear_density_stress": 0.0,
            "context_linear_drift_stress": 0.0,
            "context_linear_uncertainty_stress": 0.0,
        }
    query_recent_days = int(getattr(args, "similarity_prior_query_recent_days", 1))
    z_scale = max(float(getattr(args, "context_linear_z_scale", 1.0)), 1e-6)
    similarity_scale = max(float(getattr(args, "context_linear_similarity_scale", 5.0)), 1e-6)
    meta_center = float(getattr(args, "context_linear_meta_center", 0.50))
    meta_span = max(1.0 - meta_center, 1e-6)
    z_signal = _query_context_scalar(context, "z_eff_mean", query_recent_days=query_recent_days) / z_scale
    density_stress = _query_context_density_stress(context, query_recent_days=query_recent_days)
    drift_value = _query_context_scalar(
        context,
        "meta_context_drift_pct_ts_mean",
        query_recent_days=query_recent_days,
        default=meta_center,
    )
    uncertainty_value = _query_context_scalar(
        context,
        "meta_context_uncertainty_pct_ts_mean",
        query_recent_days=query_recent_days,
        default=meta_center,
    )
    drift_stress = max(0.0, (drift_value - meta_center) / meta_span)
    uncertainty_stress = max(0.0, (uncertainty_value - meta_center) / meta_span)
    similarity_signal = np.asarray(similarity_prior_objective, dtype=float) / similarity_scale
    context_score = (
        float(getattr(args, "context_linear_z_weight", 1.0)) * z_signal
        + float(getattr(args, "context_linear_similarity_weight", 1.0)) * similarity_signal
        - float(getattr(args, "context_linear_density_weight", 1.0)) * density_stress
        - float(getattr(args, "context_linear_drift_weight", 1.0)) * drift_stress
        - float(getattr(args, "context_linear_uncertainty_weight", 1.0)) * uncertainty_stress
    )
    if deployed_thresholds and param.head in deployed_thresholds:
        reference = float(deployed_thresholds[param.head])
    else:
        reference = float(param.guarded_y if np.isfinite(param.guarded_y) else param.y)
    lowering_exposure = np.mean(np.maximum(0.0, reference - np.asarray(thresholds, dtype=float)), axis=0)
    penalty = strength * lowering_exposure * np.maximum(0.0, -context_score)
    return np.nan_to_num(penalty, nan=0.0, posinf=0.0, neginf=0.0), {
        "context_linear_score_mean": float(np.nanmean(context_score)) if np.size(context_score) else 0.0,
        "context_linear_z_signal": float(z_signal),
        "context_linear_density_stress": float(density_stress),
        "context_linear_drift_stress": float(drift_stress),
        "context_linear_uncertainty_stress": float(uncertainty_stress),
    }


def _score_y_grid_recent_daily_quantile(
    frame: pd.DataFrame,
    param: HeadParams,
    surprise: pd.DataFrame,
    y_values: np.ndarray,
    args: argparse.Namespace,
    *,
    deployed_thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    y_values = np.asarray(y_values, dtype=float)
    if frame.empty:
        return pd.DataFrame(
            {
                "y": y_values,
                "value": np.zeros_like(y_values, dtype=float),
                "deployed_threshold_soft_prior_penalty": np.zeros_like(y_values, dtype=float),
                "robust_objective": np.zeros_like(y_values, dtype=float),
                "passes_subwindow_constraints": np.ones_like(y_values, dtype=bool),
                "positive_objective_fraction": np.zeros_like(y_values, dtype=float),
                "median_subwindow_objective": np.zeros_like(y_values, dtype=float),
                "q15_subwindow_objective": np.zeros_like(y_values, dtype=float),
                "iqr_subwindow_objective": np.zeros_like(y_values, dtype=float),
                "worst_subwindow_drawdown": np.zeros_like(y_values, dtype=float),
                "active_subwindow_count": np.zeros_like(y_values, dtype=int),
                "passes_activity_constraints": np.ones_like(y_values, dtype=bool),
                "selected_count": np.zeros_like(y_values, dtype=int),
                "total_net_pnl": np.zeros_like(y_values, dtype=float),
                "hit_rate": np.full_like(y_values, np.nan, dtype=float),
                "recent_quantile_objective": np.zeros_like(y_values, dtype=float),
                "context_linear_lowering_penalty": np.zeros_like(y_values, dtype=float),
                "context_linear_score": np.zeros_like(y_values, dtype=float),
                "context_linear_z_signal": np.zeros_like(y_values, dtype=float),
                "context_linear_density_stress": np.zeros_like(y_values, dtype=float),
                "context_linear_drift_stress": np.zeros_like(y_values, dtype=float),
                "context_linear_uncertainty_stress": np.zeros_like(y_values, dtype=float),
            }
        )

    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    end = pd.Timestamp(timestamps.max()).ceil("D")
    lookback_days = max(float(getattr(args, "recent_quantile_days", 20.0)), 1.0)
    recent_mask = timestamps.ge(end - pd.Timedelta(days=lookback_days)).to_numpy(dtype=bool)
    recent = frame.loc[recent_mask].copy()
    if recent.empty:
        recent = frame.copy()

    offset = _threshold_offset_vector(recent, param, surprise, args)
    floor = -0.50
    if bool(args.use_deployed_threshold_floor) and deployed_thresholds:
        floor = float(deployed_thresholds.get(param.head, -0.50))
    thresholds = np.clip(np.maximum(floor, offset[:, None] + y_values[None, :]), -0.50, 1.50)
    scores = pd.to_numeric(recent["score"], errors="coerce").fillna(-np.inf).to_numpy(dtype=float)
    returns = pd.to_numeric(recent["net_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    hits_raw = returns > 0.0
    masks = scores[:, None] >= thresholds

    day_codes, day_index = pd.factorize(pd.to_datetime(recent["timestamp"], utc=True).dt.floor("D"), sort=True)
    n_days = int(len(day_index))
    n_grid = int(y_values.size)
    daily_pnl = np.zeros((n_days, n_grid), dtype=float)
    daily_count = np.zeros((n_days, n_grid), dtype=float)
    daily_hits = np.zeros((n_days, n_grid), dtype=float)
    # There are at most ~20 daily groups in this mode. Loop over days and keep
    # each group's threshold scoring vectorized over the full Y grid.
    for day in range(n_days):
        rows = day_codes == day
        if not np.any(rows):
            continue
        day_masks = masks[rows, :]
        daily_pnl[day, :] = (returns[rows, None] * day_masks).sum(axis=0)
        daily_count[day, :] = day_masks.sum(axis=0)
        daily_hits[day, :] = (hits_raw[rows, None] * day_masks).sum(axis=0)

    daily_objective = daily_pnl * 7.0
    day_weights = _recent_quantile_day_weights(day_index, end, args)
    q_level = float(getattr(args, "recent_quantile_level", 0.25))
    q_level = float(np.clip(q_level, 0.01, 0.99))
    if str(getattr(args, "recent_quantile_weight_mode", "uniform")) == "uniform":
        q_objective = np.quantile(daily_objective, q_level, axis=0)
        q15 = np.quantile(daily_objective, 0.15, axis=0)
        q25 = np.quantile(daily_objective, 0.25, axis=0)
        q75 = np.quantile(daily_objective, 0.75, axis=0)
        median = np.median(daily_objective, axis=0)
        mean = np.mean(daily_objective, axis=0)
        positive_fraction = np.mean(daily_objective > 0.0, axis=0)
    else:
        q_objective = _weighted_quantile_by_column(daily_objective, day_weights, q_level)
        q15 = _weighted_quantile_by_column(daily_objective, day_weights, 0.15)
        q25 = _weighted_quantile_by_column(daily_objective, day_weights, 0.25)
        q75 = _weighted_quantile_by_column(daily_objective, day_weights, 0.75)
        median = _weighted_quantile_by_column(daily_objective, day_weights, 0.50)
        weight_sum = max(float(day_weights.sum()), 1e-12)
        mean = (daily_objective * day_weights[:, None]).sum(axis=0) / weight_sum
        positive_fraction = ((daily_objective > 0.0) * day_weights[:, None]).sum(axis=0) / weight_sum
    similarity_prior_objective = np.zeros(n_grid, dtype=float)
    similarity_prior_hit_rate = np.full(n_grid, np.nan, dtype=float)
    similarity_prior_effective_days = np.zeros(n_grid, dtype=float)
    similarity_prior_weight_sum = np.zeros(n_grid, dtype=float)
    context = _daily_similarity_feature_frame(recent, surprise)
    if bool(getattr(args, "similarity_prior_enable", False)) and n_days > 1:
        similarity_weights = _similar_day_weights(
            context,
            query_recent_days=int(getattr(args, "similarity_prior_query_recent_days", 1)),
            top_k=int(getattr(args, "similarity_prior_top_k_days", 5)),
            min_days=int(getattr(args, "similarity_prior_min_days", 3)),
            temperature=float(getattr(args, "similarity_prior_temperature", 1.0)),
        )
        if len(similarity_weights) == n_days and float(similarity_weights.sum()) > 0.0:
            similarity_prior_objective = _weighted_mean_by_column(daily_objective, similarity_weights)
            similarity_hit_rate_by_day = np.divide(
                daily_hits,
                daily_count,
                out=np.full_like(daily_hits, np.nan, dtype=float),
                where=daily_count > 0,
            )
            similarity_prior_hit_rate = _weighted_mean_by_column(similarity_hit_rate_by_day, similarity_weights)
            weight_sum = float(similarity_weights.sum())
            similarity_prior_weight_sum = np.full(n_grid, weight_sum, dtype=float)
            effective_days = weight_sum * weight_sum / max(float(np.sum(np.square(similarity_weights))), 1e-12)
            similarity_prior_effective_days = np.full(n_grid, effective_days, dtype=float)
    iqr = q75 - q25
    selected_count = daily_count.sum(axis=0)
    total_pnl = daily_pnl.sum(axis=0)
    total_hits = daily_hits.sum(axis=0)
    hit_rate = np.divide(total_hits, selected_count, out=np.full(n_grid, np.nan), where=selected_count > 0)
    active_day_count = np.sum(daily_count > 0, axis=0)
    robust = (
        q_objective
        + float(getattr(args, "recent_quantile_median_weight", 0.25)) * median
        + float(getattr(args, "recent_quantile_mean_weight", 0.05)) * mean
        - float(getattr(args, "recent_quantile_iqr_penalty", 0.10)) * iqr
    )
    if bool(getattr(args, "similarity_prior_enable", False)):
        hr_floor = float(getattr(args, "similarity_prior_hr_floor", 0.0))
        hr_weight = float(getattr(args, "similarity_prior_hr_weight", 0.0))
        hr_bonus = np.nan_to_num(similarity_prior_hit_rate - hr_floor, nan=0.0, posinf=0.0, neginf=0.0)
        robust = (
            robust
            + float(getattr(args, "similarity_prior_ev_weight", 0.0)) * similarity_prior_objective
            + hr_weight * hr_bonus
        )
    context_linear_penalty, context_linear_info = _linear_context_lowering_penalty(
        context=context,
        thresholds=thresholds,
        param=param,
        args=args,
        similarity_prior_objective=similarity_prior_objective,
        deployed_thresholds=deployed_thresholds,
    )
    robust = robust - context_linear_penalty
    min_selected_count = int(getattr(args, "min_threshold_selected_count", 0))
    min_active_days = int(getattr(args, "min_threshold_active_subwindows", 0))
    activity_ok = (selected_count >= min_selected_count) & (active_day_count >= min_active_days)
    optimization_score = robust.copy()
    if min_selected_count > 0 or min_active_days > 0:
        activity_gap = np.maximum(0.0, min_selected_count - selected_count) / max(float(min_selected_count), 1.0)
        activity_gap += np.maximum(0.0, min_active_days - active_day_count) / max(float(min_active_days), 1.0)
        optimization_score = np.where(
            activity_ok,
            optimization_score,
            optimization_score - float(args.subwindow_constraint_penalty) * activity_gap,
        )

    soft_prior_penalty = np.zeros(n_grid, dtype=float)
    prior_strength = float(getattr(args, "deployed_threshold_soft_prior_strength", 0.0))
    if prior_strength > 0.0 and deployed_thresholds and param.head in deployed_thresholds:
        deployed_threshold = float(deployed_thresholds[param.head])
        deadband = max(float(getattr(args, "deployed_threshold_soft_prior_deadband", 0.0)), 0.0)
        power = max(float(getattr(args, "deployed_threshold_soft_prior_power", 2.0)), 1.0)
        activity_weight = max(float(getattr(args, "deployed_threshold_soft_prior_activity_weight", 0.0)), 0.0)
        below = np.maximum(0.0, deployed_threshold - thresholds - deadband)
        activity = np.mean(masks, axis=0)
        soft_prior_penalty = prior_strength * np.mean(np.power(below, power), axis=0) * (1.0 + activity_weight * activity)
        optimization_score = optimization_score - soft_prior_penalty

    return pd.DataFrame(
        {
            "y": y_values,
            "value": optimization_score,
            "deployed_threshold_soft_prior_penalty": soft_prior_penalty,
            "robust_objective": robust,
            "passes_subwindow_constraints": activity_ok,
            "positive_objective_fraction": positive_fraction,
            "median_subwindow_objective": median,
            "q15_subwindow_objective": q15,
            "iqr_subwindow_objective": iqr,
            "worst_subwindow_drawdown": np.zeros(n_grid, dtype=float),
            "active_subwindow_count": active_day_count.astype(int),
            "passes_activity_constraints": activity_ok,
            "selected_count": selected_count.astype(int),
            "total_net_pnl": total_pnl,
            "hit_rate": hit_rate,
            "recent_quantile_objective": q_objective,
            "recent_quantile_level": np.full(n_grid, q_level, dtype=float),
            "similarity_prior_objective": similarity_prior_objective,
            "similarity_prior_hit_rate": similarity_prior_hit_rate,
            "similarity_prior_effective_days": similarity_prior_effective_days,
            "similarity_prior_weight_sum": similarity_prior_weight_sum,
            "context_linear_lowering_penalty": context_linear_penalty,
            "context_linear_score": np.full(n_grid, context_linear_info["context_linear_score_mean"], dtype=float),
            "context_linear_z_signal": np.full(n_grid, context_linear_info["context_linear_z_signal"], dtype=float),
            "context_linear_density_stress": np.full(n_grid, context_linear_info["context_linear_density_stress"], dtype=float),
            "context_linear_drift_stress": np.full(n_grid, context_linear_info["context_linear_drift_stress"], dtype=float),
            "context_linear_uncertainty_stress": np.full(n_grid, context_linear_info["context_linear_uncertainty_stress"], dtype=float),
        }
    )


def _local_band_guard(
    frame: pd.DataFrame,
    head: str,
    *,
    x_days: float,
    w: float,
    y: float,
    w_lower: float | None = None,
    w_raise: float | None = None,
    slope_lag: int = 1,
    meta_context_enabled: bool = False,
    meta_drift_raise: float = 0.0,
    meta_drift_floor: float = np.nan,
    meta_uncertainty_raise: float = 0.0,
    meta_uncertainty_floor: float = np.nan,
    meta_badness_cutoff: float = 0.60,
    meta_badness_temperature: float = 0.08,
    meta_badness_pressure_scale: float = 1.0,
    top_rank_floor: float,
    z_clip: float,
    band_width: float,
    step: float,
    min_rows: int,
    deployed_thresholds: dict[str, float] | None = None,
    use_deployed_threshold_floor: bool = True,
    args: argparse.Namespace | None = None,
) -> HeadParams:
    head_frame = frame.loc[frame["head"].eq(head)].copy()
    base_param = HeadParams(
        head=head,
        x_days=float(x_days),
        w=float(w),
        y=float(y),
        guarded_y=float(y),
        guard_shift=0.0,
        local_band_pnl=np.nan,
        local_band_count=0,
        w_lower=float(w if w_lower is None else w_lower),
        w_raise=float(w if w_raise is None else w_raise),
        slope_lag=int(slope_lag),
        meta_context_enabled=bool(meta_context_enabled),
        meta_drift_raise=float(meta_drift_raise),
        meta_drift_floor=float(meta_drift_floor),
        meta_uncertainty_raise=float(meta_uncertainty_raise),
        meta_uncertainty_floor=float(meta_uncertainty_floor),
        meta_badness_cutoff=float(meta_badness_cutoff),
        meta_badness_temperature=float(meta_badness_temperature),
        meta_badness_pressure_scale=float(meta_badness_pressure_scale),
    )
    if args is None:
        surprise = build_surprise(
            head_frame,
            halflife_days_by_head={head: x_days},
            slope_lag_by_head={head: int(slope_lag)},
            top_rank_floor=top_rank_floor,
            z_clip=z_clip,
        )
    else:
        surprise = _build_surprise_for_params(head_frame, {head: base_param}, args)
        base_param = _attach_forecast_coefficients(base_param, surprise, args)
        base_param = _fit_meta_badness_classifier(
            head_frame,
            base_param,
            surprise,
            args,
            deployed_thresholds=deployed_thresholds,
        )
    score = head_frame["score"].to_numpy(dtype=float)
    ret = head_frame["net_return"].to_numpy(dtype=float)
    current = float(y)
    final_band_pnl = np.nan
    final_count = 0
    while current <= 1.5000001:
        loop_param = replace(base_param, guarded_y=float(current))
        threshold = _threshold_vector(
            head_frame,
            {head: loop_param},
            surprise,
            deployed_thresholds=deployed_thresholds,
            use_deployed_threshold_floor=use_deployed_threshold_floor,
            surprise_forecast_mode=str(args.surprise_forecast_mode) if args is not None else "level",
            slope_tolerance=float(args.slope_tolerance) if args is not None else 0.0,
            z_cap=float(args.z_clip) if args is not None else float(z_clip),
            require_lowering_confirmation=bool(args.require_lowering_confirmation) if args is not None else True,
            meta_context_action_mode=str(args.meta_context_action_mode) if args is not None else "raise",
            meta_context_bad_z_threshold=float(args.meta_context_bad_z_threshold) if args is not None else 0.0,
            meta_context_transform=str(args.meta_context_transform) if args is not None else "raw",
            context_linear_density_raise=(
                float(getattr(args, "context_linear_density_raise", 0.0)) if args is not None else 0.0
            ),
            context_linear_density_floor=(
                float(getattr(args, "context_linear_density_floor", 0.0)) if args is not None else 0.0
            ),
            context_linear_relaxation_dampen=(
                float(getattr(args, "context_linear_relaxation_dampen", 1.0)) if args is not None else 1.0
            ),
            context_linear_pressure_raise=(
                float(getattr(args, "context_linear_pressure_raise", 0.0)) if args is not None else 0.0
            ),
        )
        band = (score >= threshold) & (score < (threshold + float(band_width)))
        final_count = int(np.sum(band))
        final_band_pnl = float(np.sum(ret[band])) if final_count else 0.0
        if final_count < int(min_rows) or final_band_pnl > 0.0:
            break
        current += float(step)
    guarded = float(min(current, 1.50))
    return replace(
        base_param,
        guarded_y=guarded,
        guard_shift=float(guarded - float(y)),
        local_band_pnl=float(final_band_pnl),
        local_band_count=int(final_count),
    )


def _recent_validation_grid(
    frame: pd.DataFrame,
    param: HeadParams,
    surprise: pd.DataFrame,
    y_values: np.ndarray,
    args: argparse.Namespace,
    *,
    deployed_thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    y_values = np.asarray(y_values, dtype=float)
    if frame.empty:
        return pd.DataFrame(
            {
                "y": y_values,
                "selected_count": np.zeros_like(y_values, dtype=int),
                "total_net_pnl": np.zeros_like(y_values, dtype=float),
                "avg_net_pnl": np.full_like(y_values, np.nan, dtype=float),
                "hit_rate": np.full_like(y_values, np.nan, dtype=float),
                "passes_recent_validation": np.ones_like(y_values, dtype=bool),
            }
        )
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    end = pd.Timestamp(timestamps.max()).ceil("D")
    start = end - pd.Timedelta(days=max(float(getattr(args, "recent_validation_days", 5.0)), 1e-6))
    recent = frame.loc[timestamps.ge(start)].copy()
    if recent.empty:
        recent = frame.copy()
    offset = _threshold_offset_vector(recent, param, surprise, args)
    floor = -0.50
    if bool(args.use_deployed_threshold_floor) and deployed_thresholds:
        floor = float(deployed_thresholds.get(param.head, -0.50))
    thresholds = np.clip(np.maximum(floor, offset[:, None] + y_values[None, :]), -0.50, 1.50)
    scores = pd.to_numeric(recent["score"], errors="coerce").fillna(-np.inf).to_numpy(dtype=float)
    returns = pd.to_numeric(recent["net_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    masks = scores[:, None] >= thresholds
    selected_count = masks.sum(axis=0).astype(int)
    total_pnl = (returns[:, None] * masks).sum(axis=0)
    hit_count = ((returns > 0.0)[:, None] * masks).sum(axis=0)
    hit_rate = np.divide(hit_count, selected_count, out=np.full(y_values.size, np.nan), where=selected_count > 0)
    avg_pnl = np.divide(total_pnl, selected_count, out=np.full(y_values.size, np.nan), where=selected_count > 0)
    min_count = int(getattr(args, "recent_validation_min_count", 20))
    min_total = float(getattr(args, "recent_validation_min_total_pnl", 0.0))
    min_hr = float(getattr(args, "recent_validation_min_hit_rate", 0.0))
    min_avg = float(getattr(args, "recent_validation_min_avg_pnl", -np.inf))
    too_sparse_to_judge = selected_count < min_count
    passes = (
        too_sparse_to_judge
        | (
            (total_pnl >= min_total)
            & (np.nan_to_num(hit_rate, nan=0.0) >= min_hr)
            & (np.nan_to_num(avg_pnl, nan=-np.inf) >= min_avg)
        )
    )
    return pd.DataFrame(
        {
            "y": y_values,
            "selected_count": selected_count,
            "total_net_pnl": total_pnl,
            "avg_net_pnl": avg_pnl,
            "hit_rate": hit_rate,
            "passes_recent_validation": passes,
        }
    )


def _apply_recent_validation_guard(
    frame: pd.DataFrame,
    param: HeadParams,
    surprise: pd.DataFrame,
    args: argparse.Namespace,
    *,
    deployed_thresholds: dict[str, float] | None = None,
) -> HeadParams:
    if not bool(getattr(args, "recent_validation_guard", False)):
        return param
    step = max(float(getattr(args, "recent_validation_step", 0.01)), 1e-6)
    current_y = float(param.guarded_y if np.isfinite(param.guarded_y) else param.y)
    grid = np.arange(current_y, 1.5000001 + step, step, dtype=float)
    grid = np.unique(np.clip(np.r_[current_y, grid, 1.50], -0.50, 1.50))
    scored = _recent_validation_grid(
        frame,
        param,
        surprise,
        grid,
        args,
        deployed_thresholds=deployed_thresholds,
    ).sort_values("y", kind="mergesort")
    if scored.empty:
        return param
    current_pos = int(np.argmin(np.abs(scored["y"].to_numpy(dtype=float) - current_y)))
    current_row = scored.iloc[current_pos]
    if bool(current_row["passes_recent_validation"]):
        return replace(
            param,
            recent_validation_count=int(current_row["selected_count"]),
            recent_validation_total_pnl=float(current_row["total_net_pnl"]),
            recent_validation_hit_rate=float(current_row["hit_rate"]) if np.isfinite(current_row["hit_rate"]) else np.nan,
            recent_validation_avg_pnl=float(current_row["avg_net_pnl"]) if np.isfinite(current_row["avg_net_pnl"]) else np.nan,
        )
    passing = scored.loc[scored["passes_recent_validation"] & scored["y"].ge(current_y)]
    if not passing.empty:
        chosen = passing.iloc[0]
        reason = "raised_to_recent_validation_pass"
    else:
        # If no threshold satisfies the recent slice, close the head for the day.
        chosen = scored.iloc[-1]
        reason = "raised_to_max_recent_validation_fail"
    new_y = float(chosen["y"])
    return replace(
        param,
        guarded_y=new_y,
        guard_shift=float(new_y - float(param.y)),
        recent_validation_guarded=True,
        recent_validation_count=int(chosen["selected_count"]),
        recent_validation_total_pnl=float(chosen["total_net_pnl"]),
        recent_validation_hit_rate=float(chosen["hit_rate"]) if np.isfinite(chosen["hit_rate"]) else np.nan,
        recent_validation_avg_pnl=float(chosen["avg_net_pnl"]) if np.isfinite(chosen["avg_net_pnl"]) else np.nan,
        recent_validation_shift=float(new_y - current_y),
        recent_validation_reason=reason,
    )


def _candidate_quality_gate_floors(args: argparse.Namespace) -> np.ndarray:
    low = float(getattr(args, "quality_gate_p_hit_min", 0.50))
    high = float(getattr(args, "quality_gate_p_hit_max", 0.90))
    step = max(float(getattr(args, "quality_gate_p_hit_step", 0.01)), 1e-6)
    grid = np.arange(low, high + step * 0.5, step, dtype=float)
    extras = [0.0]
    if bool(getattr(args, "quality_gate_allow_deactivation", False)):
        # p_hit is clipped below 1.0 at load time; 1.01 is an explicit close.
        extras.append(1.01)
    return np.unique(np.clip(np.r_[extras, grid], 0.0, 1.50))


def _quality_gate_metrics(frame: pd.DataFrame, mask: np.ndarray) -> dict[str, float]:
    selected = frame.loc[np.asarray(mask, dtype=bool)]
    count = int(len(selected))
    if count <= 0:
        return {"count": 0, "total_pnl": 0.0, "hit_rate": np.nan, "avg_pnl": np.nan}
    returns = pd.to_numeric(selected["net_return"], errors="coerce").fillna(0.0)
    return {
        "count": count,
        "total_pnl": float(returns.sum()),
        "hit_rate": float(returns.gt(0.0).mean()),
        "avg_pnl": float(returns.mean()),
    }


def _apply_quality_gate(
    frame: pd.DataFrame,
    param: HeadParams,
    surprise: pd.DataFrame,
    args: argparse.Namespace,
    *,
    deployed_thresholds: dict[str, float] | None = None,
) -> HeadParams:
    if not bool(getattr(args, "quality_gate_enable", False)):
        return param

    ungated = replace(param, quality_gate_enabled=False, quality_gate_p_hit_floor=np.nan)
    if frame.empty:
        return replace(
            ungated,
            quality_gate_enabled=True,
            quality_gate_p_hit_floor=1.01 if bool(getattr(args, "quality_gate_allow_deactivation", False)) else 0.0,
            quality_gate_reason="empty_train_window",
        )

    base_mask = _mask_from_params_with_args(
        frame,
        {param.head: ungated},
        surprise,
        args,
        deployed_thresholds=deployed_thresholds,
    )
    base = _quality_gate_metrics(frame, base_mask)
    p_hit = pd.to_numeric(frame["p_hit"], errors="coerce").fillna(-np.inf).to_numpy(dtype=float)

    min_count = int(getattr(args, "quality_gate_min_selected_count", 20))
    min_keep_fraction = max(float(getattr(args, "quality_gate_min_keep_fraction", 0.0)), 0.0)
    required_count = max(min_count, int(math.ceil(min_keep_fraction * int(base["count"]))))
    target_hr = float(getattr(args, "quality_gate_target_hit_rate", 0.0))
    min_total_pnl = float(getattr(args, "quality_gate_min_total_pnl", -1.0e18))
    min_avg_pnl = float(getattr(args, "quality_gate_min_avg_pnl", -1.0e18))

    rows: list[dict[str, Any]] = []
    for floor in _candidate_quality_gate_floors(args):
        mask = np.asarray(base_mask, dtype=bool) & (p_hit >= float(floor))
        metric = _quality_gate_metrics(frame, mask)
        count = int(metric["count"])
        total = float(metric["total_pnl"])
        hr = float(metric["hit_rate"]) if np.isfinite(metric["hit_rate"]) else np.nan
        avg = float(metric["avg_pnl"]) if np.isfinite(metric["avg_pnl"]) else np.nan
        closes_head = float(floor) > 1.0
        count_ok = count >= required_count
        hr_ok = np.isfinite(hr) and hr >= target_hr
        pnl_ok = total >= min_total_pnl
        avg_ok = np.isfinite(avg) and avg >= min_avg_pnl
        rows.append(
            {
                "floor": float(floor),
                "count": count,
                "total_pnl": total,
                "hit_rate": hr,
                "avg_pnl": avg,
                "passes": bool((not closes_head) and count_ok and hr_ok and pnl_ok and avg_ok),
                "closes_head": bool(closes_head),
            }
        )

    scored = pd.DataFrame(rows)
    passing = scored.loc[scored["passes"]].copy()
    if not passing.empty:
        passing = passing.sort_values(["total_pnl", "hit_rate", "count"], ascending=[False, False, False], kind="mergesort")
        chosen = passing.iloc[0]
        reason = "target_pass"
    elif bool(getattr(args, "quality_gate_deactivate_if_no_pass", False)) and bool(getattr(args, "quality_gate_allow_deactivation", False)):
        deactivated = scored.loc[scored["closes_head"]]
        chosen = deactivated.iloc[0] if not deactivated.empty else scored.iloc[-1]
        reason = "deactivated_no_target_pass"
    else:
        chosen = scored.loc[scored["floor"].eq(0.0)].iloc[0] if scored["floor"].eq(0.0).any() else scored.iloc[0]
        reason = "no_target_pass_keep_ungated"

    return replace(
        param,
        quality_gate_enabled=True,
        quality_gate_p_hit_floor=float(chosen["floor"]),
        quality_gate_train_count=int(chosen["count"]),
        quality_gate_train_total_pnl=float(chosen["total_pnl"]),
        quality_gate_train_hit_rate=float(chosen["hit_rate"]) if np.isfinite(chosen["hit_rate"]) else np.nan,
        quality_gate_train_avg_pnl=float(chosen["avg_pnl"]) if np.isfinite(chosen["avg_pnl"]) else np.nan,
        quality_gate_base_count=int(base["count"]),
        quality_gate_base_total_pnl=float(base["total_pnl"]),
        quality_gate_base_hit_rate=float(base["hit_rate"]) if np.isfinite(base["hit_rate"]) else np.nan,
        quality_gate_base_avg_pnl=float(base["avg_pnl"]) if np.isfinite(base["avg_pnl"]) else np.nan,
        quality_gate_reason=reason,
    )


def _tail_adjusted_objective(metrics: dict[str, Any], args: argparse.Namespace) -> float:
    q05 = float(metrics.get("q05_rolling_week_pnl", 0.0))
    q15 = float(metrics.get("q15_rolling_week_pnl", 0.0))
    objective = float(metrics.get("objective", 0.0))
    penalty = float(args.tail_penalty_weight) * (
        max(0.0, float(args.per_head_min_q05_week_pnl) - q05)
        + max(0.0, float(args.per_head_min_q15_week_pnl) - q15)
    )
    return objective - penalty


def _maybe_apply_removed_trade_gate(
    head_frame: pd.DataFrame,
    param: HeadParams,
    surprise: pd.DataFrame,
    args: argparse.Namespace,
    *,
    deployed_thresholds: dict[str, float] | None = None,
) -> HeadParams:
    if not bool(getattr(args, "meta_context_removed_trade_gate", False)) or not bool(param.meta_context_enabled):
        return param
    no_context = replace(
        param,
        meta_context_enabled=False,
        meta_drift_raise=0.0,
        meta_drift_floor=np.nan,
        meta_uncertainty_raise=0.0,
        meta_uncertainty_floor=np.nan,
    )
    base_mask = _mask_from_params_with_args(
        head_frame,
        {param.head: no_context},
        surprise,
        args,
        deployed_thresholds=deployed_thresholds,
    )
    context_mask = _mask_from_params_with_args(
        head_frame,
        {param.head: param},
        surprise,
        args,
        deployed_thresholds=deployed_thresholds,
    )
    removed = np.asarray(base_mask, dtype=bool) & ~np.asarray(context_mask, dtype=bool)
    removed_count = int(np.sum(removed))
    returns = pd.to_numeric(head_frame["net_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    removed_total = float(np.sum(returns[removed])) if removed_count else 0.0
    removed_avg = float(removed_total / removed_count) if removed_count else np.nan
    gate_passed = (
        removed_count >= int(args.meta_context_removed_min_count)
        and removed_total <= float(args.meta_context_removed_max_total_pnl)
        and np.isfinite(removed_avg)
        and removed_avg <= float(args.meta_context_removed_max_avg_pnl)
    )
    if gate_passed:
        return replace(
            param,
            meta_context_removed_count=removed_count,
            meta_context_removed_total_pnl=removed_total,
            meta_context_removed_avg_pnl=removed_avg,
            meta_context_removed_gate_passed=True,
        )
    return replace(
        param,
        meta_context_enabled=False,
        meta_drift_raise=0.0,
        meta_drift_floor=np.nan,
        meta_uncertainty_raise=0.0,
        meta_uncertainty_floor=np.nan,
        meta_context_removed_count=removed_count,
        meta_context_removed_total_pnl=removed_total,
        meta_context_removed_avg_pnl=removed_avg,
        meta_context_removed_gate_passed=False,
    )


def _finalize_head_param(
    head_frame: pd.DataFrame,
    param: HeadParams,
    args: argparse.Namespace,
    *,
    deployed_thresholds: dict[str, float] | None = None,
) -> HeadParams:
    surprise = _build_surprise_for_params(head_frame, {param.head: param}, args)
    param = _attach_forecast_coefficients(param, surprise, args)
    param = _fit_meta_badness_classifier(
        head_frame,
        param,
        surprise,
        args,
        deployed_thresholds=deployed_thresholds,
    )
    param = _maybe_apply_removed_trade_gate(
        head_frame,
        param,
        surprise,
        args,
        deployed_thresholds=deployed_thresholds,
    )
    mask = _mask_from_params_with_args(
        head_frame,
        {param.head: param},
        surprise,
        args,
        deployed_thresholds=deployed_thresholds,
    )
    metrics = _policy_metrics(head_frame, mask, policy=f"{param.head}_candidate")
    subwindow = _subwindow_summary(head_frame, mask, args, policy=f"{param.head}_candidate")
    objective = float(metrics["objective"])
    q05 = float(metrics["q05_rolling_week_pnl"])
    q15 = float(metrics["q15_rolling_week_pnl"])
    deployed_objective = np.nan
    deployed_q05 = np.nan
    deployed_q15 = np.nan
    deployed_robust = np.nan
    if deployed_thresholds and param.head in deployed_thresholds:
        deployed_threshold = float(deployed_thresholds[param.head])
        deployed_mask = (
            pd.to_numeric(head_frame["score"], errors="coerce")
            .fillna(-np.inf)
            .to_numpy(dtype=float)
            >= deployed_threshold
        )
        deployed_metrics = _policy_metrics(
            head_frame,
            deployed_mask,
            policy=f"{param.head}_deployed_fixed",
        )
        deployed_subwindow = _subwindow_summary(
            head_frame,
            deployed_mask,
            args,
            policy=f"{param.head}_deployed_fixed",
        )
        deployed_objective = float(deployed_metrics["objective"])
        deployed_q05 = float(deployed_metrics["q05_rolling_week_pnl"])
        deployed_q15 = float(deployed_metrics["q15_rolling_week_pnl"])
        deployed_robust = float(deployed_subwindow["robust_objective"])
    reasons: list[str] = []
    gate_constraints = str(getattr(args, "subwindow_constraints_mode", "gate")) == "gate"
    if gate_constraints:
        if objective < float(args.per_head_min_objective):
            reasons.append(f"objective<{args.per_head_min_objective:g}")
        if q05 < float(args.per_head_min_q05_week_pnl):
            reasons.append(f"q05<{args.per_head_min_q05_week_pnl:g}")
        if q15 < float(args.per_head_min_q15_week_pnl):
            reasons.append(f"q15<{args.per_head_min_q15_week_pnl:g}")
        if not bool(subwindow["passes_subwindow_constraints"]):
            reasons.append("subwindow_constraints")
        if float(subwindow["robust_objective"]) < float(args.per_head_min_robust_objective):
            reasons.append(f"robust<{args.per_head_min_robust_objective:g}")
    if gate_constraints and bool(getattr(args, "require_dynamic_head_improvement_over_deployed", False)) and np.isfinite(deployed_objective):
        min_delta = float(getattr(args, "min_dynamic_head_objective_delta", 0.0))
        if objective < deployed_objective + min_delta:
            reasons.append("objective<=deployed")
        if bool(getattr(args, "require_dynamic_head_tail_not_worse_than_deployed", True)):
            if np.isfinite(deployed_q05) and q05 < deployed_q05:
                reasons.append("q05<deployed")
            if np.isfinite(deployed_q15) and q15 < deployed_q15:
                reasons.append("q15<deployed")
        if np.isfinite(deployed_robust) and float(subwindow["robust_objective"]) < deployed_robust + min_delta:
            reasons.append("robust<=deployed")
    if reasons:
        if bool(getattr(args, "fallback_rejected_heads_to_deployed", False)) and deployed_thresholds:
            fallback_threshold = float(deployed_thresholds.get(param.head, 1.50))
            return replace(
                param,
                w=0.0,
                w_lower=0.0,
                w_raise=0.0,
                guarded_y=fallback_threshold,
                guard_shift=float(fallback_threshold - param.y),
                deactivated=False,
                dynamic_rejected=True,
                fallback_to_deployed=True,
                fallback_threshold=fallback_threshold,
                deactivation_reason=";".join(reasons),
                head_objective=objective,
                head_q05_week_pnl=q05,
                head_q15_week_pnl=q15,
                deployed_head_objective=deployed_objective,
                deployed_head_q05_week_pnl=deployed_q05,
                deployed_head_q15_week_pnl=deployed_q15,
                deployed_robust_objective=deployed_robust,
                subwindow_count=int(subwindow["subwindow_count"]),
                positive_objective_fraction=float(subwindow["positive_objective_fraction"]),
                median_subwindow_objective=float(subwindow["median_subwindow_objective"]),
                q15_subwindow_objective=float(subwindow["q15_subwindow_objective"]),
                iqr_subwindow_objective=float(subwindow["iqr_subwindow_objective"]),
                worst_subwindow_drawdown=float(subwindow["worst_subwindow_drawdown"]),
                robust_objective=float(subwindow["robust_objective"]),
                passes_subwindow_constraints=bool(subwindow["passes_subwindow_constraints"]),
            )
        return replace(
            param,
            w=0.0,
            w_lower=0.0,
            w_raise=0.0,
            guarded_y=1.50,
            guard_shift=float(1.50 - param.y),
            deactivated=True,
            deactivation_reason=";".join(reasons),
            head_objective=objective,
            head_q05_week_pnl=q05,
            head_q15_week_pnl=q15,
            deployed_head_objective=deployed_objective,
            deployed_head_q05_week_pnl=deployed_q05,
            deployed_head_q15_week_pnl=deployed_q15,
            deployed_robust_objective=deployed_robust,
            subwindow_count=int(subwindow["subwindow_count"]),
            positive_objective_fraction=float(subwindow["positive_objective_fraction"]),
            median_subwindow_objective=float(subwindow["median_subwindow_objective"]),
            q15_subwindow_objective=float(subwindow["q15_subwindow_objective"]),
            iqr_subwindow_objective=float(subwindow["iqr_subwindow_objective"]),
            worst_subwindow_drawdown=float(subwindow["worst_subwindow_drawdown"]),
            robust_objective=float(subwindow["robust_objective"]),
            passes_subwindow_constraints=bool(subwindow["passes_subwindow_constraints"]),
        )
    return replace(
        param,
        deactivated=False,
        dynamic_rejected=False,
        fallback_to_deployed=False,
        fallback_threshold=np.nan,
        deactivation_reason="",
        head_objective=objective,
        head_q05_week_pnl=q05,
        head_q15_week_pnl=q15,
        deployed_head_objective=deployed_objective,
        deployed_head_q05_week_pnl=deployed_q05,
        deployed_head_q15_week_pnl=deployed_q15,
        deployed_robust_objective=deployed_robust,
        subwindow_count=int(subwindow["subwindow_count"]),
        positive_objective_fraction=float(subwindow["positive_objective_fraction"]),
        median_subwindow_objective=float(subwindow["median_subwindow_objective"]),
        q15_subwindow_objective=float(subwindow["q15_subwindow_objective"]),
        iqr_subwindow_objective=float(subwindow["iqr_subwindow_objective"]),
        worst_subwindow_drawdown=float(subwindow["worst_subwindow_drawdown"]),
        robust_objective=float(subwindow["robust_objective"]),
        passes_subwindow_constraints=bool(subwindow["passes_subwindow_constraints"]),
    )


def _optimize_independent_dynamic_policy(
    frame: pd.DataFrame,
    args: argparse.Namespace,
    *,
    deployed_thresholds: dict[str, float] | None = None,
) -> tuple[dict[str, HeadParams], None, pd.DataFrame, pd.DataFrame]:
    params: dict[str, HeadParams] = {}
    trial_frames: list[pd.DataFrame] = []
    for head, head_frame in frame.groupby("head", sort=True):
        head_frame = head_frame.copy()
        subwindow_cache = _build_subwindow_objective_cache(head_frame, args)
        objective_cache: dict[tuple[Any, ...], float] = {}
        surprise_cache: dict[tuple[Any, ...], tuple[pd.DataFrame, HeadParams]] = {}

        def objective(trial: optuna.Trial) -> float:
            x_days = trial.suggest_float(f"{head}__x_days", float(args.x_min_days), float(args.x_max_days), log=True)
            if str(args.surprise_forecast_mode) == "slope":
                w_lower = trial.suggest_float(f"{head}__w_lower", float(args.w_lower_min), float(args.w_lower_max))
                raw_w_raise = trial.suggest_float(f"{head}__w_raise", float(args.w_raise_min), float(args.w_raise_max))
                w_raise = max(w_lower, raw_w_raise)
                w = max(w_lower, w_raise)
                slope_lag = trial.suggest_int(f"{head}__slope_lag", int(args.slope_lag_min), int(args.slope_lag_max))
            else:
                w_lower = trial.suggest_float(f"{head}__w_lower", float(args.w_lower_min), float(args.w_lower_max))
                raw_w_raise = trial.suggest_float(f"{head}__w_raise", float(args.w_raise_min), float(args.w_raise_max))
                w_raise = max(w_lower, raw_w_raise) if bool(args.require_raise_sensitivity_at_least_lower) else raw_w_raise
                w = max(w_lower, w_raise)
                slope_lag = 1
            y = trial.suggest_float(f"{head}__y", float(args.y_min), float(args.y_max))
            use_meta_context_for_head = bool(args.use_meta_context_features)
            if use_meta_context_for_head and bool(args.meta_context_tune_enable):
                use_meta_context_for_head = bool(trial.suggest_categorical(f"{head}__meta_context_enabled", [False, True]))
            if use_meta_context_for_head:
                meta_drift_raise = trial.suggest_float(
                    f"{head}__meta_drift_raise",
                    float(args.meta_drift_raise_min),
                    float(args.meta_drift_raise_max),
                )
                meta_drift_floor = trial.suggest_float(
                    f"{head}__meta_drift_floor",
                    float(args.meta_drift_floor_min),
                    float(args.meta_drift_floor_max),
                )
                meta_uncertainty_raise = trial.suggest_float(
                    f"{head}__meta_uncertainty_raise",
                    float(args.meta_uncertainty_raise_min),
                    float(args.meta_uncertainty_raise_max),
                )
                meta_uncertainty_floor = trial.suggest_float(
                    f"{head}__meta_uncertainty_floor",
                    float(args.meta_uncertainty_floor_min),
                    float(args.meta_uncertainty_floor_max),
                )
                if _uses_meta_badness_classifier(str(args.meta_context_action_mode)):
                    meta_badness_cutoff = trial.suggest_float(
                        f"{head}__meta_badness_cutoff",
                        float(args.meta_badness_cutoff_min),
                        float(args.meta_badness_cutoff_max),
                    )
                else:
                    meta_badness_cutoff = float(args.meta_badness_cutoff_default)
                if str(args.meta_context_action_mode) == "badness_classifier_soft_raise":
                    meta_badness_temperature = trial.suggest_float(
                        f"{head}__meta_badness_temperature",
                        float(args.meta_badness_temperature_min),
                        float(args.meta_badness_temperature_max),
                        log=True,
                    )
                    meta_badness_pressure_scale = trial.suggest_float(
                        f"{head}__meta_badness_pressure_scale",
                        float(args.meta_badness_pressure_scale_min),
                        float(args.meta_badness_pressure_scale_max),
                    )
                else:
                    meta_badness_temperature = float(args.meta_badness_temperature_default)
                    meta_badness_pressure_scale = float(args.meta_badness_pressure_scale_default)
            else:
                meta_drift_raise = 0.0
                meta_drift_floor = np.nan
                meta_uncertainty_raise = 0.0
                meta_uncertainty_floor = np.nan
                meta_badness_cutoff = float(args.meta_badness_cutoff_default)
                meta_badness_temperature = float(args.meta_badness_temperature_default)
                meta_badness_pressure_scale = float(args.meta_badness_pressure_scale_default)
            param = HeadParams(
                head=head,
                x_days=x_days,
                w=w,
                y=y,
                guarded_y=y,
                guard_shift=0.0,
                local_band_pnl=np.nan,
                local_band_count=0,
                w_lower=w_lower,
                w_raise=w_raise,
                slope_lag=slope_lag,
                meta_context_enabled=bool(use_meta_context_for_head),
                meta_drift_raise=meta_drift_raise,
                meta_drift_floor=meta_drift_floor,
                meta_uncertainty_raise=meta_uncertainty_raise,
                meta_uncertainty_floor=meta_uncertainty_floor,
                meta_badness_cutoff=float(meta_badness_cutoff),
                meta_badness_temperature=float(meta_badness_temperature),
                meta_badness_pressure_scale=float(meta_badness_pressure_scale),
            )
            cache_key = (
                round(float(x_days), 8),
                round(float(w), 8),
                round(float(w_lower), 8),
                round(float(w_raise), 8),
                round(float(y), 8),
                int(slope_lag),
                bool(use_meta_context_for_head),
                round(float(meta_drift_raise), 8),
                round(float(meta_drift_floor), 8) if np.isfinite(meta_drift_floor) else "nan",
                round(float(meta_uncertainty_raise), 8),
                round(float(meta_uncertainty_floor), 8) if np.isfinite(meta_uncertainty_floor) else "nan",
                round(float(meta_badness_cutoff), 8),
                round(float(meta_badness_temperature), 8),
                round(float(meta_badness_pressure_scale), 8),
                str(args.surprise_forecast_mode),
            )
            if cache_key in objective_cache:
                return objective_cache[cache_key]
            surprise_key = (round(float(x_days), 8), int(slope_lag), str(args.surprise_forecast_mode))
            if surprise_key not in surprise_cache:
                surprise = _build_surprise_for_params(head_frame, {head: param}, args)
                surprise_cache[surprise_key] = (surprise, _attach_forecast_coefficients(param, surprise, args))
            surprise, forecast_param = surprise_cache[surprise_key]
            param = replace(
                forecast_param,
                w=float(w),
                w_lower=float(w_lower),
                w_raise=float(w_raise),
                y=float(y),
                guarded_y=float(y),
                meta_context_enabled=bool(use_meta_context_for_head),
                meta_drift_raise=float(meta_drift_raise),
                meta_drift_floor=float(meta_drift_floor),
                meta_uncertainty_raise=float(meta_uncertainty_raise),
                meta_uncertainty_floor=float(meta_uncertainty_floor),
                meta_badness_cutoff=float(meta_badness_cutoff),
                meta_badness_temperature=float(meta_badness_temperature),
                meta_badness_pressure_scale=float(meta_badness_pressure_scale),
            )
            param = _fit_meta_badness_classifier(
                head_frame,
                param,
                surprise,
                args,
                deployed_thresholds=deployed_thresholds,
            )
            mask = _mask_from_params_with_args(
                head_frame,
                {head: param},
                surprise,
                args,
                deployed_thresholds=deployed_thresholds,
            )
            subwindow = _subwindow_summary_from_cache(subwindow_cache, mask, args, policy=f"{head}_trial")
            value = float(subwindow["optimization_score"])
            value -= _deployed_threshold_soft_prior_penalty(
                head_frame,
                {head: param},
                surprise,
                args,
                deployed_thresholds=deployed_thresholds,
            )
            objective_cache[cache_key] = value
            return value

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = _make_tpe_sampler(args)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=int(args.trials), show_progress_bar=False)
        best = study.best_params
        if str(args.surprise_forecast_mode) == "slope":
            best_w_lower = float(best[f"{head}__w_lower"])
            best_w_raise = max(best_w_lower, float(best[f"{head}__w_raise"]))
            best_w = max(best_w_lower, best_w_raise)
            best_slope_lag = int(best[f"{head}__slope_lag"])
        else:
            best_w_lower = float(best[f"{head}__w_lower"])
            raw_best_w_raise = float(best[f"{head}__w_raise"])
            best_w_raise = max(best_w_lower, raw_best_w_raise) if bool(args.require_raise_sensitivity_at_least_lower) else raw_best_w_raise
            best_w = max(best_w_lower, best_w_raise)
            best_slope_lag = 1
        if bool(args.use_meta_context_features) and bool(args.meta_context_tune_enable):
            best_meta_context_enabled = bool(best.get(f"{head}__meta_context_enabled", False))
        else:
            best_meta_context_enabled = bool(args.use_meta_context_features)
        if best_meta_context_enabled:
            best_meta_drift_raise = float(best.get(f"{head}__meta_drift_raise", 0.0))
            best_meta_drift_floor = float(best.get(f"{head}__meta_drift_floor", np.nan))
            best_meta_uncertainty_raise = float(best.get(f"{head}__meta_uncertainty_raise", 0.0))
            best_meta_uncertainty_floor = float(best.get(f"{head}__meta_uncertainty_floor", np.nan))
            best_meta_badness_cutoff = float(
                best.get(f"{head}__meta_badness_cutoff", float(args.meta_badness_cutoff_default))
            )
            best_meta_badness_temperature = float(
                best.get(f"{head}__meta_badness_temperature", float(args.meta_badness_temperature_default))
            )
            best_meta_badness_pressure_scale = float(
                best.get(f"{head}__meta_badness_pressure_scale", float(args.meta_badness_pressure_scale_default))
            )
        else:
            best_meta_drift_raise = 0.0
            best_meta_drift_floor = np.nan
            best_meta_uncertainty_raise = 0.0
            best_meta_uncertainty_floor = np.nan
            best_meta_badness_cutoff = float(args.meta_badness_cutoff_default)
            best_meta_badness_temperature = float(args.meta_badness_temperature_default)
            best_meta_badness_pressure_scale = float(args.meta_badness_pressure_scale_default)
        guarded = _local_band_guard(
            frame,
            str(head),
            x_days=float(best[f"{head}__x_days"]),
            w=best_w,
            y=float(best[f"{head}__y"]),
            w_lower=best_w_lower,
            w_raise=best_w_raise,
            slope_lag=best_slope_lag,
            meta_context_enabled=best_meta_context_enabled,
            meta_drift_raise=best_meta_drift_raise,
            meta_drift_floor=best_meta_drift_floor,
            meta_uncertainty_raise=best_meta_uncertainty_raise,
            meta_uncertainty_floor=best_meta_uncertainty_floor,
            meta_badness_cutoff=best_meta_badness_cutoff,
            meta_badness_temperature=best_meta_badness_temperature,
            meta_badness_pressure_scale=best_meta_badness_pressure_scale,
            top_rank_floor=float(args.top_rank_floor),
            z_clip=float(args.z_clip),
            band_width=float(args.local_band_width),
            step=float(args.local_band_step),
            min_rows=int(args.local_band_min_rows),
            deployed_thresholds=deployed_thresholds,
            use_deployed_threshold_floor=bool(args.use_deployed_threshold_floor),
            args=args,
        )
        params[str(head)] = _finalize_head_param(head_frame, guarded, args, deployed_thresholds=deployed_thresholds)
        trials = study.trials_dataframe()
        trials.insert(0, "head", str(head))
        trial_frames.append(trials)
    surprise = build_surprise(
        frame,
        halflife_days_by_head={head: param.x_days for head, param in params.items()},
        slope_lag_by_head={head: param.slope_lag for head, param in params.items()},
        top_rank_floor=float(args.top_rank_floor),
        z_clip=float(args.z_clip),
        count_shrink_n0=float(args.surprise_count_shrink_n0),
    )
    trial_frame = pd.concat(trial_frames, ignore_index=True) if trial_frames else pd.DataFrame()
    return params, None, surprise, trial_frame


def optimize_dynamic_policy(
    frame: pd.DataFrame,
    args: argparse.Namespace,
    *,
    deployed_thresholds: dict[str, float] | None = None,
) -> tuple[dict[str, HeadParams], optuna.study.Study | None, pd.DataFrame, pd.DataFrame]:
    if str(args.head_optimization_mode) == "independent":
        return _optimize_independent_dynamic_policy(frame, args, deployed_thresholds=deployed_thresholds)
    if str(args.surprise_forecast_mode) == "slope":
        raise ValueError("surprise_forecast_mode='slope' requires --head-optimization-mode independent")

    heads = tuple(sorted(frame["head"].dropna().unique()))
    head_frames = {str(head): group.copy() for head, group in frame.groupby("head", sort=True)}
    head_caches = {head: _build_subwindow_objective_cache(group, args) for head, group in head_frames.items()}
    cache: dict[tuple[tuple[tuple[str, float], ...]], pd.DataFrame] = {}

    def surprise_for(x_map: dict[str, float]) -> pd.DataFrame:
        key = tuple(sorted((str(k), round(float(v), 8)) for k, v in x_map.items()))
        if key not in cache:
            cache[key] = build_surprise(
                frame,
                halflife_days_by_head={k: float(v) for k, v in x_map.items()},
                top_rank_floor=float(args.top_rank_floor),
                z_clip=float(args.z_clip),
            )
        return cache[key]

    def objective(trial: optuna.Trial) -> float:
        raw_params: dict[str, HeadParams] = {}
        x_map: dict[str, float] = {}
        for head in heads:
            x_days = trial.suggest_float(f"{head}__x_days", float(args.x_min_days), float(args.x_max_days), log=True)
            w_lower = trial.suggest_float(f"{head}__w_lower", float(args.w_lower_min), float(args.w_lower_max))
            raw_w_raise = trial.suggest_float(f"{head}__w_raise", float(args.w_raise_min), float(args.w_raise_max))
            w_raise = max(w_lower, raw_w_raise) if bool(args.require_raise_sensitivity_at_least_lower) else raw_w_raise
            w = max(w_lower, w_raise)
            y = trial.suggest_float(f"{head}__y", float(args.y_min), float(args.y_max))
            x_map[head] = x_days
            raw_params[head] = HeadParams(
                head=head,
                x_days=x_days,
                w=w,
                y=y,
                guarded_y=y,
                guard_shift=0.0,
                local_band_pnl=np.nan,
                local_band_count=0,
                w_lower=w_lower,
                w_raise=w_raise,
            )
        surprise = surprise_for(x_map)
        mask = _mask_from_params(
            frame,
            raw_params,
            surprise,
            deployed_thresholds=deployed_thresholds,
            use_deployed_threshold_floor=bool(args.use_deployed_threshold_floor),
        )
        score = 0.0
        mask_series = pd.Series(mask, index=frame.index)
        for head, head_frame in head_frames.items():
            head_mask = mask_series.loc[head_frame.index].to_numpy(dtype=bool)
            score += float(
                _subwindow_summary_from_cache(
                    head_caches[head],
                    head_mask,
                    args,
                    policy=f"{head}_trial",
                )["optimization_score"]
            )
        score -= _deployed_threshold_soft_prior_penalty(
            frame,
            raw_params,
            surprise,
            args,
            deployed_thresholds=deployed_thresholds,
        )
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = _make_tpe_sampler(args)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=int(args.trials), show_progress_bar=False)
    best = study.best_params
    guarded_params: dict[str, HeadParams] = {}
    for head in heads:
        guarded_params[head] = _local_band_guard(
            frame,
            head,
            x_days=float(best[f"{head}__x_days"]),
            w=max(
                float(best[f"{head}__w_lower"]),
                (
                    max(float(best[f"{head}__w_lower"]), float(best[f"{head}__w_raise"]))
                    if bool(args.require_raise_sensitivity_at_least_lower)
                    else float(best[f"{head}__w_raise"])
                ),
            ),
            y=float(best[f"{head}__y"]),
            w_lower=float(best[f"{head}__w_lower"]),
            w_raise=(
                max(float(best[f"{head}__w_lower"]), float(best[f"{head}__w_raise"]))
                if bool(args.require_raise_sensitivity_at_least_lower)
                else float(best[f"{head}__w_raise"])
            ),
            top_rank_floor=float(args.top_rank_floor),
            z_clip=float(args.z_clip),
            band_width=float(args.local_band_width),
            step=float(args.local_band_step),
            min_rows=int(args.local_band_min_rows),
            deployed_thresholds=deployed_thresholds,
            use_deployed_threshold_floor=bool(args.use_deployed_threshold_floor),
        )
        guarded_params[head] = _finalize_head_param(
            frame.loc[frame["head"].eq(head)].copy(),
            guarded_params[head],
            args,
            deployed_thresholds=deployed_thresholds,
        )
    surprise = build_surprise(
        frame,
        halflife_days_by_head={head: param.x_days for head, param in guarded_params.items()},
        top_rank_floor=float(args.top_rank_floor),
        z_clip=float(args.z_clip),
    )
    trials = study.trials_dataframe()
    return guarded_params, study, surprise, trials


def optimize_thresholds_with_fixed_xw(
    frame: pd.DataFrame,
    args: argparse.Namespace,
    *,
    fixed_xw_params: dict[str, HeadParams],
    deployed_thresholds: dict[str, float] | None = None,
) -> tuple[dict[str, HeadParams], pd.DataFrame, pd.DataFrame]:
    params: dict[str, HeadParams] = {}
    trial_frames: list[pd.DataFrame] = []
    for head, head_frame in frame.groupby("head", sort=True):
        head = str(head)
        head_frame = head_frame.copy()
        base = fixed_xw_params.get(head)
        if base is None:
            fallback = bool(getattr(args, "fallback_rejected_heads_to_deployed", False))
            fallback_threshold = float((deployed_thresholds or {}).get(head, 1.50))
            params[head] = HeadParams(
                head=head,
                x_days=float(args.x_min_days),
                w=0.0,
                y=fallback_threshold,
                guarded_y=fallback_threshold,
                guard_shift=0.0,
                local_band_pnl=0.0,
                local_band_count=0,
                w_lower=0.0,
                w_raise=0.0,
                deactivated=not fallback,
                dynamic_rejected=fallback,
                fallback_to_deployed=fallback,
                fallback_threshold=fallback_threshold if fallback else np.nan,
                deactivation_reason="missing_fixed_xw",
            )
            continue
        subwindow_cache = _build_subwindow_objective_cache(head_frame, args)
        base_w_lower = base.w_lower if np.isfinite(base.w_lower) else base.w
        base_w_raise = base.w_raise if np.isfinite(base.w_raise) else base.w
        base_param = HeadParams(
            head=head,
            x_days=base.x_days,
            w=base.w,
            y=base.y,
            guarded_y=base.y,
            guard_shift=0.0,
            local_band_pnl=np.nan,
            local_band_count=0,
            w_lower=base_w_lower,
            w_raise=base_w_raise,
            slope_lag=base.slope_lag,
            meta_context_enabled=base.meta_context_enabled,
            meta_drift_raise=base.meta_drift_raise,
            meta_drift_floor=base.meta_drift_floor,
            meta_uncertainty_raise=base.meta_uncertainty_raise,
            meta_uncertainty_floor=base.meta_uncertainty_floor,
            meta_badness_cutoff=base.meta_badness_cutoff,
            meta_badness_intercept=base.meta_badness_intercept,
            meta_badness_drift_coef=base.meta_badness_drift_coef,
            meta_badness_uncertainty_coef=base.meta_badness_uncertainty_coef,
            meta_badness_zneg_coef=base.meta_badness_zneg_coef,
            meta_badness_score_coef=base.meta_badness_score_coef,
            meta_badness_temperature=base.meta_badness_temperature,
            meta_badness_pressure_scale=base.meta_badness_pressure_scale,
        )
        base_surprise = _build_surprise_for_params(head_frame, {head: base_param}, args)
        base_param = _attach_forecast_coefficients(base_param, base_surprise, args)
        base_param = _fit_meta_badness_classifier(
            head_frame,
            base_param,
            base_surprise,
            args,
            deployed_thresholds=deployed_thresholds,
        )

        if str(args.threshold_refresh_mode) == "grid":
            y_grid = np.linspace(float(args.y_min), float(args.y_max), int(args.threshold_grid_size))
            if np.isfinite(base.y):
                y_grid = np.unique(np.r_[y_grid, float(base.y)])
            if deployed_thresholds and head in deployed_thresholds:
                y_grid = np.unique(np.r_[y_grid, float(deployed_thresholds[head])])
            if str(getattr(args, "threshold_selection_objective", "subwindow")) == "recent_daily_quantile":
                grid_scores = _score_y_grid_recent_daily_quantile(
                    head_frame,
                    base_param,
                    base_surprise,
                    y_grid,
                    args,
                    deployed_thresholds=deployed_thresholds,
                )
            else:
                grid_scores = _score_y_grid(
                    head_frame,
                    subwindow_cache,
                    base_param,
                    base_surprise,
                    y_grid,
                    args,
                    deployed_thresholds=deployed_thresholds,
                )
            best_row = grid_scores.sort_values(["value", "robust_objective"], ascending=False).iloc[0]
            best_y = float(best_row["y"])
            trials = grid_scores.copy()
            trials.insert(0, "head", head)
            trials["params_" + head + "__y"] = trials["y"]
            trials["state"] = "COMPLETE"
            trial_frames.append(trials)
        else:
            objective_cache: dict[float, float] = {}

            def objective(trial: optuna.Trial) -> float:
                y = trial.suggest_float(f"{head}__y", float(args.y_min), float(args.y_max))
                cache_key = round(float(y), 8)
                if cache_key in objective_cache:
                    return objective_cache[cache_key]
                param = replace(base_param, y=float(y), guarded_y=float(y))
                mask = _mask_from_params_with_args(
                    head_frame,
                    {head: param},
                    base_surprise,
                    args,
                    deployed_thresholds=deployed_thresholds,
                )
                subwindow = _subwindow_summary_from_cache(subwindow_cache, mask, args, policy=f"{head}_threshold_trial")
                value = float(subwindow["optimization_score"])
                value -= _deployed_threshold_soft_prior_penalty(
                    head_frame,
                    {head: param},
                    base_surprise,
                    args,
                    deployed_thresholds=deployed_thresholds,
                )
                min_selected_count = int(getattr(args, "min_threshold_selected_count", 0))
                min_active_subwindows = int(getattr(args, "min_threshold_active_subwindows", 0))
                if min_selected_count > 0 or min_active_subwindows > 0:
                    selected_count = int(np.sum(mask))
                    active_subwindows = int(np.unique(subwindow_cache.window_id[np.asarray(mask, dtype=bool)]).size) if selected_count else 0
                    if selected_count < min_selected_count or active_subwindows < min_active_subwindows:
                        activity_gap = max(0.0, min_selected_count - selected_count) / max(float(min_selected_count), 1.0)
                        activity_gap += max(0.0, min_active_subwindows - active_subwindows) / max(float(min_active_subwindows), 1.0)
                        value -= float(args.subwindow_constraint_penalty) * activity_gap
                objective_cache[cache_key] = value
                return value

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            sampler = optuna.samplers.TPESampler(seed=int(args.seed), multivariate=True)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(objective, n_trials=int(args.threshold_trials), show_progress_bar=False)
            best_y = float(study.best_params[f"{head}__y"])
            trials = study.trials_dataframe()
            trials.insert(0, "head", head)
            trial_frames.append(trials)
        guarded = _local_band_guard(
            frame,
            head,
            x_days=base.x_days,
            w=base.w,
            y=best_y,
            w_lower=base_w_lower,
            w_raise=base_w_raise,
            slope_lag=base.slope_lag,
            meta_context_enabled=base.meta_context_enabled,
            meta_drift_raise=base.meta_drift_raise,
            meta_drift_floor=base.meta_drift_floor,
            meta_uncertainty_raise=base.meta_uncertainty_raise,
            meta_uncertainty_floor=base.meta_uncertainty_floor,
            meta_badness_cutoff=base.meta_badness_cutoff,
            meta_badness_temperature=base.meta_badness_temperature,
            meta_badness_pressure_scale=base.meta_badness_pressure_scale,
            top_rank_floor=float(args.top_rank_floor),
            z_clip=float(args.z_clip),
            band_width=float(args.local_band_width),
            step=float(args.local_band_step),
            min_rows=int(args.local_band_min_rows),
            deployed_thresholds=deployed_thresholds,
            use_deployed_threshold_floor=bool(args.use_deployed_threshold_floor),
            args=args,
        )
        guarded = _apply_recent_validation_guard(
            head_frame,
            guarded,
            base_surprise,
            args,
            deployed_thresholds=deployed_thresholds,
        )
        guarded = _apply_quality_gate(
            head_frame,
            guarded,
            base_surprise,
            args,
            deployed_thresholds=deployed_thresholds,
        )
        params[head] = _finalize_head_param(head_frame, guarded, args, deployed_thresholds=deployed_thresholds)
    surprise = build_surprise(
        frame,
        halflife_days_by_head={head: param.x_days for head, param in params.items()},
        slope_lag_by_head={head: param.slope_lag for head, param in params.items()},
        top_rank_floor=float(args.top_rank_floor),
        z_clip=float(args.z_clip),
        count_shrink_n0=float(args.surprise_count_shrink_n0),
    )
    return params, surprise, pd.concat(trial_frames, ignore_index=True) if trial_frames else pd.DataFrame()


def _walk_forward_splits(frame: pd.DataFrame, args: argparse.Namespace) -> list[tuple[int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(frame["timestamp"].min()).floor("D")
    end = pd.Timestamp(frame["timestamp"].max()).ceil("D")
    train_delta = pd.Timedelta(days=float(args.train_days))
    valid_delta = pd.Timedelta(days=float(args.valid_days))
    step_delta = pd.Timedelta(days=float(args.step_days))
    out: list[tuple[int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    fold = 1
    train_start = start
    while True:
        train_end = train_start + train_delta
        valid_start = train_end
        valid_end = valid_start + valid_delta
        if valid_start >= end:
            break
        out.append((fold, train_start, train_end, valid_start, min(valid_end, end)))
        fold += 1
        train_start = train_start + step_delta
    return out


def run_walk_forward(
    frame: pd.DataFrame,
    args: argparse.Namespace,
    *,
    deployed_thresholds: dict[str, float],
    output_dir: Path,
) -> None:
    splits = _walk_forward_splits(frame, args)
    fixed_xw_params: dict[str, HeadParams] | None = None
    initial_xw_rows: list[pd.DataFrame] = []
    if splits and str(args.walk_forward_xw_fit_mode) == "initial":
        first_fold, train_start, train_end, valid_start, _valid_end = splits[0]
        xw_fit_end = valid_start
        if args.xw_fit_days is None:
            xw_fit_start = train_start
        else:
            xw_fit_start = max(pd.Timestamp(frame["timestamp"].min()).floor("D"), xw_fit_end - pd.Timedelta(days=float(args.xw_fit_days)))
        xw_fit = frame.loc[frame["timestamp"].ge(xw_fit_start) & frame["timestamp"].lt(xw_fit_end)].copy()
        if len(xw_fit) >= int(args.min_train_rows):
            fixed_xw_params, _study, _surprise, _trials = optimize_dynamic_policy(
                xw_fit,
                args,
                deployed_thresholds=deployed_thresholds,
            )
            pf = pd.DataFrame([asdict(param) for param in fixed_xw_params.values()])
            pf.insert(0, "fold", int(first_fold))
            pf["xw_fit_start"] = xw_fit_start.isoformat()
            pf["xw_fit_end"] = xw_fit_end.isoformat()
            pf["deployed_fixed_threshold"] = pf["head"].map(deployed_thresholds)
            initial_xw_rows.append(pf)

    selected_ids: dict[str, set[int]] = {
        "fixed_deployed_thresholds": set(),
        "all_top30_candidate_pool": set(),
        "dynamic_hr_surprise_guarded_walkforward": set(),
    }
    valid_ids: list[int] = []
    fold_rows: list[dict[str, Any]] = []
    param_rows: list[pd.DataFrame] = []
    threshold_trial_rows: list[pd.DataFrame] = []
    for fold, train_start, train_end, valid_start, valid_end in splits:
        train = frame.loc[frame["timestamp"].ge(train_start) & frame["timestamp"].lt(train_end)].copy()
        valid = frame.loc[frame["timestamp"].ge(valid_start) & frame["timestamp"].lt(valid_end)].copy()
        if len(train) < int(args.min_train_rows) or valid.empty:
            continue
        if fixed_xw_params is not None:
            params, _train_surprise, threshold_trials = optimize_thresholds_with_fixed_xw(
                train,
                args,
                fixed_xw_params=fixed_xw_params,
                deployed_thresholds=deployed_thresholds,
            )
            if not threshold_trials.empty:
                threshold_trials.insert(0, "fold", int(fold))
                threshold_trial_rows.append(threshold_trials)
        else:
            params, _study, _train_surprise, _trials = optimize_dynamic_policy(
                train,
                args,
                deployed_thresholds=deployed_thresholds,
            )
        history = frame.loc[frame["timestamp"].ge(train_start) & frame["timestamp"].lt(valid_end)].copy()
        surprise = _build_surprise_for_params(history, params, args)
        dynamic_mask = _mask_from_params_with_args(
            valid,
            params,
            surprise,
            args,
            deployed_thresholds=deployed_thresholds,
        )
        deployed_mask = valid["score"].to_numpy(dtype=float) >= valid["head"].map(deployed_thresholds).to_numpy(dtype=float)
        top30_mask = valid["rank"].to_numpy(dtype=float) >= float(args.top_rank_floor)
        fold_masks = {
            "fixed_deployed_thresholds": deployed_mask,
            "all_top30_candidate_pool": top30_mask,
            "dynamic_hr_surprise_guarded_walkforward": dynamic_mask,
        }
        valid_ids.extend(valid["row_uid"].astype(int).tolist())
        for policy, mask in fold_masks.items():
            ids = valid.loc[np.asarray(mask, dtype=bool), "row_uid"].astype(int).tolist()
            selected_ids[policy].update(ids)
            row = _policy_metrics_by_fold(valid, mask, policy=policy, fold=fold)
            row.update(
                {
                    "train_start": train_start.isoformat(),
                    "train_end": train_end.isoformat(),
                    "valid_start": valid_start.isoformat(),
                    "valid_end": valid_end.isoformat(),
                    "train_rows": int(len(train)),
                    "valid_rows": int(len(valid)),
                }
            )
            fold_rows.append(row)
        pf = pd.DataFrame([asdict(param) for param in params.values()])
        pf.insert(0, "fold", int(fold))
        pf["train_start"] = train_start.isoformat()
        pf["train_end"] = train_end.isoformat()
        pf["valid_start"] = valid_start.isoformat()
        pf["valid_end"] = valid_end.isoformat()
        pf["deployed_fixed_threshold"] = pf["head"].map(deployed_thresholds)
        param_rows.append(pf)

    if not valid_ids:
        return
    valid_id_set = set(valid_ids)
    wf_frame = frame.loc[frame["row_uid"].astype(int).isin(valid_id_set)].copy()
    wf_frame = wf_frame.sort_values(["timestamp", "head", "score"], ascending=[True, True, False]).reset_index(drop=True)
    aggregate_rows: list[dict[str, Any]] = []
    by_head_frames: list[pd.DataFrame] = []
    monthly_frames: list[pd.DataFrame] = []
    monthly_by_head_frames: list[pd.DataFrame] = []
    weekly_frames: list[pd.DataFrame] = []
    for policy, ids in selected_ids.items():
        mask = wf_frame["row_uid"].astype(int).isin(ids).to_numpy(dtype=bool)
        aggregate_rows.append(_policy_metrics(wf_frame, mask, policy=policy))
        by_head_frames.append(_by_head_metrics(wf_frame, mask, policy=policy))
        monthly_frames.append(_monthly_metrics(wf_frame, mask, policy=policy))
        monthly_by_head_frames.append(_monthly_by_head_metrics(wf_frame, mask, policy=policy))
        weekly_frames.append(_weekly_frame(wf_frame, mask, policy=policy))
    _write_frame(output_dir / "walkforward_dynamic_hr_surprise_policy_summary.parquet", pd.DataFrame(aggregate_rows))
    _write_frame(output_dir / "walkforward_dynamic_hr_surprise_policy_by_head.parquet", pd.concat(by_head_frames, ignore_index=True))
    _write_frame(output_dir / "walkforward_dynamic_hr_surprise_policy_monthly.parquet", pd.concat(monthly_frames, ignore_index=True))
    _write_frame(output_dir / "walkforward_dynamic_hr_surprise_policy_monthly_by_head.parquet", pd.concat(monthly_by_head_frames, ignore_index=True))
    _write_frame(output_dir / "walkforward_dynamic_hr_surprise_policy_rolling_weeks.parquet", pd.concat(weekly_frames, ignore_index=True))
    _write_frame(output_dir / "walkforward_dynamic_hr_surprise_fold_summary.parquet", pd.DataFrame(fold_rows))
    if param_rows:
        _write_frame(output_dir / "walkforward_dynamic_hr_surprise_params.parquet", pd.concat(param_rows, ignore_index=True))
    if initial_xw_rows:
        _write_frame(output_dir / "walkforward_dynamic_hr_surprise_initial_xw_params.parquet", pd.concat(initial_xw_rows, ignore_index=True))
    if threshold_trial_rows:
        _write_frame(output_dir / "walkforward_dynamic_hr_surprise_threshold_trials.parquet", pd.concat(threshold_trial_rows, ignore_index=True))


def _month_starts_between(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    first = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    months = pd.date_range(first, end.ceil("D"), freq="MS", tz="UTC")
    out = [pd.Timestamp(m) for m in months if pd.Timestamp(m) <= end]
    if start not in out:
        out = sorted(set(out + [start.floor("D")]))
    return out


def _daily_starts_between(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    days = pd.date_range(start.floor("D"), end.ceil("D"), freq="D", tz="UTC")
    return [pd.Timestamp(day) for day in days if start <= pd.Timestamp(day) < end]


def _asof_surprise_for_scoring(
    history: pd.DataFrame,
    scoring: pd.DataFrame,
    params: dict[str, HeadParams],
    args: argparse.Namespace,
) -> pd.DataFrame:
    history_surprise = _build_surprise_for_params(history, params, args)
    if scoring.empty:
        return history_surprise.iloc[0:0].copy()
    if history_surprise.empty:
        return pd.DataFrame(
            {
                "timestamp": scoring["timestamp"],
                "head": scoring["head"],
                "ewma_num": 0.0,
                "ewma_var": 0.0,
                "ewma_count": 0.0,
                "z_raw": 0.0,
                "count_shrink": 0.0,
                "z_eff": 0.0,
                "slope": 0.0,
                "slope_lag": 1,
            }
        )
    parts: list[pd.DataFrame] = []
    surprise_cols = ["ewma_num", "ewma_var", "ewma_count", "z_raw", "count_shrink", "z_eff", "slope", "slope_lag"]
    for head, score_group in scoring[["timestamp", "head"]].drop_duplicates().groupby("head", sort=True):
        left = score_group.sort_values("timestamp").copy()
        right = history_surprise.loc[history_surprise["head"].eq(head)].sort_values("timestamp")
        if right.empty:
            for col in surprise_cols:
                left[col] = 0.0
            left["slope_lag"] = 1
            parts.append(left)
            continue
        merged = pd.merge_asof(
            left,
            right[["timestamp", "head", *surprise_cols]].sort_values("timestamp"),
            on="timestamp",
            by="head",
            direction="backward",
        )
        for col in surprise_cols:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
        merged["slope_lag"] = merged["slope_lag"].replace(0, 1).fillna(1).astype(int)
        parts.append(merged)
    return pd.concat(parts, ignore_index=True) if parts else history_surprise.iloc[0:0].copy()


def _inactive_params_for_heads(
    heads: set[str],
    args: argparse.Namespace,
    *,
    deployed_thresholds: dict[str, float] | None = None,
    reason: str = "insufficient_calendar_history",
) -> dict[str, HeadParams]:
    fallback = bool(getattr(args, "fallback_rejected_heads_to_deployed", False))
    out: dict[str, HeadParams] = {}
    for head in sorted(heads):
        threshold = 1.50
        if fallback and deployed_thresholds:
            threshold = float(deployed_thresholds.get(head, 1.50))
        out[head] = HeadParams(
            head=head,
            x_days=float(args.x_min_days),
            w=0.0,
            y=threshold,
            guarded_y=threshold,
            guard_shift=0.0,
            local_band_pnl=0.0,
            local_band_count=0,
            w_lower=0.0,
            w_raise=0.0,
            deactivated=not fallback,
            dynamic_rejected=fallback,
            fallback_to_deployed=fallback,
            fallback_threshold=float(threshold) if fallback else np.nan,
            deactivation_reason=reason,
        )
    return out


def run_calendar_replay(
    frame: pd.DataFrame,
    args: argparse.Namespace,
    *,
    deployed_thresholds: dict[str, float],
    output_dir: Path,
) -> None:
    if frame.empty:
        return
    data_start = pd.Timestamp(frame["timestamp"].min()).floor("D")
    data_end = pd.Timestamp(frame["timestamp"].max()).ceil("D")
    requested_start = pd.to_datetime(args.calendar_eval_start, utc=True) if args.calendar_eval_start else data_start
    requested_end = pd.to_datetime(args.calendar_eval_end, utc=True) if args.calendar_eval_end else data_end
    eval_start = max(requested_start, data_start + pd.Timedelta(days=float(args.calendar_xw_min_train_days)))
    eval_end = min(requested_end, data_end)
    heads = set(frame["head"].dropna().astype(str).unique())
    if eval_start >= eval_end:
        manifest = {
            "status": "skipped",
            "reason": "insufficient_history_for_calendar_replay",
            "data_start": data_start.isoformat(),
            "data_end": data_end.isoformat(),
            "required_min_train_days": float(args.calendar_xw_min_train_days),
            "eval_start": eval_start.isoformat(),
            "eval_end": eval_end.isoformat(),
        }
        (output_dir / "calendar_dynamic_hr_surprise_manifest.json").write_text(json.dumps(_json_default(manifest), indent=2), encoding="utf-8")
        return

    selected_ids: dict[str, set[int]] = {
        "fixed_deployed_thresholds": set(),
        "all_top30_candidate_pool": set(),
        "calendar_dynamic_hr_surprise": set(),
    }
    valid_ids: list[int] = []
    month_param_rows: list[pd.DataFrame] = []
    day_param_rows: list[pd.DataFrame] = []
    day_summary_rows: list[dict[str, Any]] = []
    threshold_trial_rows: list[pd.DataFrame] = []
    month_cache: dict[pd.Timestamp, dict[str, HeadParams]] = {}

    for month_start in _month_starts_between(eval_start, eval_end):
        month_end = min(month_start + pd.offsets.MonthBegin(1), eval_end)
        month_valid_start = max(month_start, eval_start)
        if month_valid_start >= month_end:
            continue
        train_end = month_start
        train_start = max(data_start, train_end - pd.Timedelta(days=float(args.calendar_xw_max_train_days)))
        train_span_days = max((train_end - train_start).total_seconds() / 86400.0, 0.0)
        train = frame.loc[frame["timestamp"].ge(train_start) & frame["timestamp"].lt(train_end)].copy()
        if train_span_days < float(args.calendar_xw_min_train_days) or len(train) < int(args.calendar_min_xw_train_rows):
            xw_params = _inactive_params_for_heads(heads, args, deployed_thresholds=deployed_thresholds)
        else:
            xw_params, _study, _surprise, _trials = optimize_dynamic_policy(
                train,
                args,
                deployed_thresholds=deployed_thresholds,
            )
        month_cache[month_start] = xw_params
        mp = pd.DataFrame([asdict(param) for param in xw_params.values()])
        mp.insert(0, "month_start", month_start.isoformat())
        mp["train_start"] = train_start.isoformat()
        mp["train_end"] = train_end.isoformat()
        mp["train_rows"] = int(len(train))
        mp["train_span_days"] = float(train_span_days)
        mp["deployed_fixed_threshold"] = mp["head"].map(deployed_thresholds)
        month_param_rows.append(mp)

        for day_start in _daily_starts_between(month_valid_start, month_end):
            day_end = min(day_start + pd.Timedelta(days=1), eval_end)
            valid = frame.loc[frame["timestamp"].ge(day_start) & frame["timestamp"].lt(day_end)].copy()
            if valid.empty:
                continue
            y_train_start = max(data_start, day_start - pd.Timedelta(days=float(args.calendar_y_train_days)))
            y_train = frame.loc[frame["timestamp"].ge(y_train_start) & frame["timestamp"].lt(day_start)].copy()
            if len(y_train) < int(args.calendar_min_y_train_rows):
                day_params = _inactive_params_for_heads(heads, args, deployed_thresholds=deployed_thresholds)
                threshold_trials = pd.DataFrame()
            else:
                day_params, _train_surprise, threshold_trials = optimize_thresholds_with_fixed_xw(
                    y_train,
                    args,
                    fixed_xw_params=xw_params,
                    deployed_thresholds=deployed_thresholds,
                )
            if not threshold_trials.empty:
                threshold_trials.insert(0, "day_start", day_start.isoformat())
                threshold_trial_rows.append(threshold_trials)
            history = frame.loc[frame["timestamp"].ge(y_train_start) & frame["timestamp"].lt(day_start)].copy()
            scoring_surprise = _asof_surprise_for_scoring(history, valid, day_params, args)
            dynamic_mask = _mask_from_params_with_args(
                valid,
                day_params,
                scoring_surprise,
                args,
                deployed_thresholds=deployed_thresholds,
            )
            fixed_mask = valid["score"].to_numpy(dtype=float) >= valid["head"].map(deployed_thresholds).to_numpy(dtype=float)
            top30_mask = valid["rank"].to_numpy(dtype=float) >= float(args.top_rank_floor)
            masks = {
                "fixed_deployed_thresholds": fixed_mask,
                "all_top30_candidate_pool": top30_mask,
                "calendar_dynamic_hr_surprise": dynamic_mask,
            }
            valid_ids.extend(valid["row_uid"].astype(int).tolist())
            for policy, mask in masks.items():
                selected_ids[policy].update(valid.loc[np.asarray(mask, dtype=bool), "row_uid"].astype(int).tolist())
                row = _policy_metrics(valid, mask, policy=policy)
                row["day_start"] = day_start.isoformat()
                row["day_end"] = day_end.isoformat()
                row["month_start"] = month_start.isoformat()
                row["y_train_start"] = y_train_start.isoformat()
                row["y_train_end"] = day_start.isoformat()
                row["valid_rows"] = int(len(valid))
                row["y_train_rows"] = int(len(y_train))
                day_summary_rows.append(row)
            dp = pd.DataFrame([asdict(param) for param in day_params.values()])
            dp.insert(0, "day_start", day_start.isoformat())
            dp["day_end"] = day_end.isoformat()
            dp["month_start"] = month_start.isoformat()
            dp["y_train_start"] = y_train_start.isoformat()
            dp["y_train_end"] = day_start.isoformat()
            dp["y_train_rows"] = int(len(y_train))
            dp["deployed_fixed_threshold"] = dp["head"].map(deployed_thresholds)
            day_param_rows.append(dp)

    if not valid_ids:
        return
    calendar_frame = frame.loc[frame["row_uid"].astype(int).isin(set(valid_ids))].copy()
    calendar_frame = calendar_frame.sort_values(["timestamp", "head", "score"], ascending=[True, True, False]).reset_index(drop=True)
    aggregate_rows: list[dict[str, Any]] = []
    by_head_frames: list[pd.DataFrame] = []
    monthly_frames: list[pd.DataFrame] = []
    monthly_by_head_frames: list[pd.DataFrame] = []
    weekly_frames: list[pd.DataFrame] = []
    for policy, ids in selected_ids.items():
        mask = calendar_frame["row_uid"].astype(int).isin(ids).to_numpy(dtype=bool)
        aggregate_rows.append(_policy_metrics(calendar_frame, mask, policy=policy))
        by_head_frames.append(_by_head_metrics(calendar_frame, mask, policy=policy))
        monthly_frames.append(_monthly_metrics(calendar_frame, mask, policy=policy))
        monthly_by_head_frames.append(_monthly_by_head_metrics(calendar_frame, mask, policy=policy))
        weekly_frames.append(_weekly_frame(calendar_frame, mask, policy=policy))
        if policy == "calendar_dynamic_hr_surprise":
            selected_rows = calendar_frame.loc[np.asarray(mask, dtype=bool)].copy()
            _write_frame(output_dir / "calendar_dynamic_hr_surprise_selected_rows.parquet", selected_rows)
    _write_frame(output_dir / "calendar_dynamic_hr_surprise_policy_summary.parquet", pd.DataFrame(aggregate_rows))
    _write_frame(output_dir / "calendar_dynamic_hr_surprise_policy_by_head.parquet", pd.concat(by_head_frames, ignore_index=True))
    _write_frame(output_dir / "calendar_dynamic_hr_surprise_policy_monthly.parquet", pd.concat(monthly_frames, ignore_index=True))
    _write_frame(output_dir / "calendar_dynamic_hr_surprise_policy_monthly_by_head.parquet", pd.concat(monthly_by_head_frames, ignore_index=True))
    _write_frame(output_dir / "calendar_dynamic_hr_surprise_policy_rolling_weeks.parquet", pd.concat(weekly_frames, ignore_index=True))
    _write_frame(output_dir / "calendar_dynamic_hr_surprise_daily_summary.parquet", pd.DataFrame(day_summary_rows))
    if month_param_rows:
        _write_frame(output_dir / "calendar_dynamic_hr_surprise_monthly_xw_params.parquet", pd.concat(month_param_rows, ignore_index=True))
    if day_param_rows:
        _write_frame(output_dir / "calendar_dynamic_hr_surprise_daily_y_params.parquet", pd.concat(day_param_rows, ignore_index=True))
    if threshold_trial_rows:
        _write_frame(output_dir / "calendar_dynamic_hr_surprise_daily_y_grid.parquet", pd.concat(threshold_trial_rows, ignore_index=True))
    manifest = {
        "status": "complete",
        "policy_preset": str(getattr(args, "policy_preset", "default")),
        "data_start": data_start.isoformat(),
        "data_end": data_end.isoformat(),
        "eval_start": eval_start.isoformat(),
        "eval_end": eval_end.isoformat(),
        "xw_min_train_days": float(args.calendar_xw_min_train_days),
        "xw_max_train_days": float(args.calendar_xw_max_train_days),
        "daily_y_train_days": float(args.calendar_y_train_days),
        "monthly_update": "X/W fit once per month on prior growing window",
        "daily_update": "Y fit once per day on prior daily_y_train_days using frozen monthly X/W",
        "fallback_rejected_heads_to_deployed": bool(args.fallback_rejected_heads_to_deployed),
        "use_deployed_threshold_floor": bool(args.use_deployed_threshold_floor),
        "spread_adjust_returns": bool(args.spread_adjust_returns),
        "require_dynamic_head_improvement_over_deployed": bool(args.require_dynamic_head_improvement_over_deployed),
        "min_dynamic_head_objective_delta": float(args.min_dynamic_head_objective_delta),
        "require_dynamic_head_tail_not_worse_than_deployed": bool(args.require_dynamic_head_tail_not_worse_than_deployed),
        "deployed_threshold_soft_prior_strength": float(args.deployed_threshold_soft_prior_strength),
        "deployed_threshold_soft_prior_deadband": float(args.deployed_threshold_soft_prior_deadband),
        "deployed_threshold_soft_prior_power": float(args.deployed_threshold_soft_prior_power),
        "deployed_threshold_soft_prior_activity_weight": float(args.deployed_threshold_soft_prior_activity_weight),
        "subwindow_days": float(args.subwindow_days),
        "subwindow_constraints_mode": str(args.subwindow_constraints_mode),
        "min_subwindows": int(args.min_subwindows),
        "min_positive_objective_fraction": float(args.min_positive_objective_fraction),
        "subwindow_q15_floor": float(args.subwindow_q15_floor),
        "subwindow_drawdown_floor": float(args.subwindow_drawdown_floor),
        "lambda_iqr": float(args.lambda_iqr),
        "lambda_tail": float(args.lambda_tail),
        "min_threshold_selected_count": int(args.min_threshold_selected_count),
        "min_threshold_active_subwindows": int(args.min_threshold_active_subwindows),
        "threshold_selection_objective": str(args.threshold_selection_objective),
        "recent_quantile_days": float(args.recent_quantile_days),
        "recent_quantile_level": float(args.recent_quantile_level),
        "recent_quantile_weight_mode": str(args.recent_quantile_weight_mode),
        "recent_quantile_weight_last_7": float(args.recent_quantile_weight_last_7),
        "recent_quantile_weight_prev_7": float(args.recent_quantile_weight_prev_7),
        "recent_quantile_weight_older": float(args.recent_quantile_weight_older),
        "similarity_prior_enable": bool(args.similarity_prior_enable),
        "similarity_prior_ev_weight": float(args.similarity_prior_ev_weight),
        "similarity_prior_hr_weight": float(args.similarity_prior_hr_weight),
        "similarity_prior_hr_floor": float(args.similarity_prior_hr_floor),
        "similarity_prior_query_recent_days": int(args.similarity_prior_query_recent_days),
        "similarity_prior_top_k_days": int(args.similarity_prior_top_k_days),
        "similarity_prior_min_days": int(args.similarity_prior_min_days),
        "similarity_prior_temperature": float(args.similarity_prior_temperature),
        **_context_linear_config(args),
        "recent_validation_guard": bool(args.recent_validation_guard),
        "recent_validation_days": float(args.recent_validation_days),
        "recent_validation_min_count": int(args.recent_validation_min_count),
        "recent_validation_min_total_pnl": float(args.recent_validation_min_total_pnl),
        "recent_validation_min_hit_rate": float(args.recent_validation_min_hit_rate),
        "quality_gate_enable": bool(args.quality_gate_enable),
        "quality_gate_target_hit_rate": float(args.quality_gate_target_hit_rate),
        "quality_gate_p_hit_min": float(args.quality_gate_p_hit_min),
        "quality_gate_p_hit_max": float(args.quality_gate_p_hit_max),
        "quality_gate_p_hit_step": float(args.quality_gate_p_hit_step),
        "quality_gate_min_selected_count": int(args.quality_gate_min_selected_count),
        "quality_gate_min_keep_fraction": float(args.quality_gate_min_keep_fraction),
        "quality_gate_min_total_pnl": float(args.quality_gate_min_total_pnl),
        "quality_gate_min_avg_pnl": float(args.quality_gate_min_avg_pnl),
        "quality_gate_allow_deactivation": bool(args.quality_gate_allow_deactivation),
        "quality_gate_deactivate_if_no_pass": bool(args.quality_gate_deactivate_if_no_pass),
    }
    (output_dir / "calendar_dynamic_hr_surprise_manifest.json").write_text(json.dumps(_json_default(manifest), indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--policy-params", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--policy-preset",
        choices=(
            "default",
            ROBUST_SUBWINDOWS_V2_PRESET,
            ROBUST_SUBWINDOWS_V3_PRESET,
            ROBUST_SUBWINDOWS_V4_PRESET,
            ROBUST_SUBWINDOWS_V5_PRESET,
            ROBUST_SUBWINDOWS_V6_PRESET,
        ),
        default="default",
        help=(
            "Named dynamic-threshold preset. robust_subwindows_v2 runs true "
            "calendar replay with monthly X/W, daily Y, weekly robust "
            "subwindow constraints, and no deployed-threshold floor. "
            "robust_subwindows_v3 adds a soft prior against moving far below "
            "deployed thresholds. robust_subwindows_v4 makes subwindow checks "
            "a modulation penalty rather than a hard deactivation gate. "
            "robust_subwindows_v5 adds a recent per-head validation guard. "
            "robust_subwindows_v6 replaces subwindow Y scoring with a vectorized "
            "20-day daily-PnL quantile objective."
        ),
    )
    parser.add_argument("--strategy-col", default=None)
    parser.add_argument("--score-col", default="normalized_rank_score")
    parser.add_argument("--rank-col", default=None)
    parser.add_argument("--p-hit-col", default=None)
    parser.add_argument("--return-col", default="net_return")
    parser.add_argument(
        "--disable-spread-adjusted-returns",
        dest="spread_adjust_returns",
        action="store_false",
        help="Use the selected return column as-is instead of debiting entry/exit spread columns.",
    )
    parser.set_defaults(spread_adjust_returns=True)
    parser.add_argument("--ev-col", default=None)
    parser.add_argument("--surprise-weight-col", default=None)
    parser.add_argument("--use-meta-context-features", action="store_true")
    parser.add_argument("--meta-context-drift-cols", default=",".join(DEFAULT_META_DRIFT_COLUMNS))
    parser.add_argument("--meta-context-uncertainty-cols", default=",".join(DEFAULT_META_UNCERTAINTY_COLUMNS))
    parser.add_argument("--meta-context-feature-aggregation", choices=("mean", "max"), default="mean")
    parser.add_argument("--meta-context-timestamp-aggregation", choices=("mean", "max", "q90"), default="mean")
    parser.add_argument("--meta-context-transform", choices=("raw", "causal_percentile"), default="raw")
    parser.add_argument(
        "--meta-context-action-mode",
        choices=(
            "raise",
            "bad_surprise_raise",
            "dampen_relaxation",
            "linear_context",
            "badness_classifier_raise",
            "badness_classifier_soft_raise",
        ),
        default="raise",
    )
    parser.add_argument("--meta-context-bad-z-threshold", type=float, default=0.0)
    parser.add_argument("--meta-context-removed-trade-gate", action="store_true")
    parser.add_argument("--meta-context-removed-min-count", type=int, default=25)
    parser.add_argument("--meta-context-removed-max-total-pnl", type=float, default=0.0)
    parser.add_argument("--meta-context-removed-max-avg-pnl", type=float, default=0.0)
    parser.add_argument("--meta-badness-cutoff-default", type=float, default=0.60)
    parser.add_argument("--meta-badness-cutoff-min", type=float, default=0.50)
    parser.add_argument("--meta-badness-cutoff-max", type=float, default=0.80)
    parser.add_argument("--meta-badness-ridge-alpha", type=float, default=3.0)
    parser.add_argument("--meta-badness-min-train-rows", type=int, default=25)
    parser.add_argument("--meta-badness-temperature-default", type=float, default=0.08)
    parser.add_argument("--meta-badness-temperature-min", type=float, default=0.03)
    parser.add_argument("--meta-badness-temperature-max", type=float, default=0.25)
    parser.add_argument("--meta-badness-pressure-scale-default", type=float, default=1.0)
    parser.add_argument("--meta-badness-pressure-scale-min", type=float, default=0.0)
    parser.add_argument("--meta-badness-pressure-scale-max", type=float, default=1.0)
    parser.add_argument("--disable-meta-context-enable-tuning", dest="meta_context_tune_enable", action="store_false")
    parser.set_defaults(meta_context_tune_enable=True)
    parser.add_argument("--meta-drift-raise-min", type=float, default=0.0)
    parser.add_argument("--meta-drift-raise-max", type=float, default=4.0)
    parser.add_argument("--meta-drift-floor-min", type=float, default=0.0)
    parser.add_argument("--meta-drift-floor-max", type=float, default=0.45)
    parser.add_argument("--meta-uncertainty-raise-min", type=float, default=0.0)
    parser.add_argument("--meta-uncertainty-raise-max", type=float, default=8.0)
    parser.add_argument("--meta-uncertainty-floor-min", type=float, default=0.0)
    parser.add_argument("--meta-uncertainty-floor-max", type=float, default=0.30)
    parser.add_argument("--context-linear-density-raise", type=float, default=0.0)
    parser.add_argument("--context-linear-density-floor", type=float, default=0.0)
    parser.add_argument("--context-linear-relaxation-dampen", type=float, default=1.0)
    parser.add_argument("--context-linear-pressure-raise", type=float, default=0.0)
    parser.add_argument("--context-linear-lowering-penalty-strength", type=float, default=0.0)
    parser.add_argument("--context-linear-z-weight", type=float, default=1.0)
    parser.add_argument("--context-linear-similarity-weight", type=float, default=1.0)
    parser.add_argument("--context-linear-density-weight", type=float, default=1.0)
    parser.add_argument("--context-linear-drift-weight", type=float, default=1.0)
    parser.add_argument("--context-linear-uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--context-linear-z-scale", type=float, default=1.0)
    parser.add_argument("--context-linear-similarity-scale", type=float, default=5.0)
    parser.add_argument("--context-linear-meta-center", type=float, default=0.50)
    parser.add_argument("--top-rank-floor", type=float, default=0.70)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--x-min-days", type=float, default=1.0)
    parser.add_argument("--x-max-days", type=float, default=28.0)
    parser.add_argument("--w-min", type=float, default=0.0)
    parser.add_argument("--w-max", type=float, default=0.40)
    parser.add_argument("--w-lower-min", type=float, default=0.0)
    parser.add_argument("--w-lower-max", type=float, default=0.15)
    parser.add_argument("--w-raise-min", type=float, default=0.0)
    parser.add_argument("--w-raise-max", type=float, default=0.60)
    parser.add_argument("--allow-raise-sensitivity-below-lower", dest="require_raise_sensitivity_at_least_lower", action="store_false")
    parser.set_defaults(require_raise_sensitivity_at_least_lower=True)
    parser.add_argument("--y-min", type=float, default=-0.50)
    parser.add_argument("--y-max", type=float, default=1.50)
    parser.add_argument("--z-clip", type=float, default=5.0)
    parser.add_argument("--surprise-forecast-mode", choices=("level", "slope"), default="level")
    parser.add_argument("--surprise-count-shrink-n0", type=float, default=20.0)
    parser.add_argument("--slope-lag-min", type=int, default=1)
    parser.add_argument("--slope-lag-max", type=int, default=12)
    parser.add_argument("--slope-tolerance", type=float, default=0.0)
    parser.add_argument("--no-lowering-confirmation", dest="require_lowering_confirmation", action="store_false")
    parser.set_defaults(require_lowering_confirmation=True)
    parser.add_argument("--forecast-ridge-alpha", type=float, default=10.0)
    parser.add_argument("--forecast-train-fraction", type=float, default=0.75)
    parser.add_argument("--forecast-min-rows", type=int, default=24)
    parser.add_argument("--forecast-min-valid-rows", type=int, default=6)
    parser.add_argument("--forecast-rho-min", type=float, default=0.0)
    parser.add_argument("--forecast-rho-max", type=float, default=1.0)
    parser.add_argument("--min-forecast-edge", type=float, default=1e-4)
    parser.add_argument("--local-band-width", type=float, default=0.02)
    parser.add_argument("--local-band-step", type=float, default=0.01)
    parser.add_argument("--local-band-min-rows", type=int, default=5)
    parser.add_argument("--disable-deployed-threshold-floor", dest="use_deployed_threshold_floor", action="store_false")
    parser.set_defaults(use_deployed_threshold_floor=True)
    parser.add_argument("--head-optimization-mode", choices=("independent", "joint"), default="independent")
    parser.add_argument("--per-head-min-objective", type=float, default=0.0)
    parser.add_argument("--per-head-min-q05-week-pnl", type=float, default=-1.0e18)
    parser.add_argument("--per-head-min-q15-week-pnl", type=float, default=-1.0e18)
    parser.add_argument("--per-head-min-robust-objective", type=float, default=0.0)
    parser.add_argument(
        "--fallback-rejected-heads-to-deployed",
        action="store_true",
        help="Deployment guard: rejected or insufficient-history dynamic heads replay the deployed fixed threshold.",
    )
    parser.add_argument(
        "--require-dynamic-head-improvement-over-deployed",
        action="store_true",
        help="Reject a dynamic head unless its local objective and robust score improve over the deployed threshold.",
    )
    parser.add_argument("--min-dynamic-head-objective-delta", type=float, default=0.0)
    parser.add_argument(
        "--allow-dynamic-head-tail-worse-than-deployed",
        dest="require_dynamic_head_tail_not_worse_than_deployed",
        action="store_false",
    )
    parser.set_defaults(require_dynamic_head_tail_not_worse_than_deployed=True)
    parser.add_argument("--tail-penalty-weight", type=float, default=10.0)
    parser.add_argument("--subwindow-days", type=float, default=7.0)
    parser.add_argument("--min-subwindows", type=int, default=2)
    parser.add_argument("--min-positive-objective-fraction", type=float, default=0.60)
    parser.add_argument("--subwindow-q15-floor", type=float, default=0.0)
    parser.add_argument("--subwindow-drawdown-floor", type=float, default=-1.0)
    parser.add_argument("--lambda-iqr", type=float, default=0.50)
    parser.add_argument("--lambda-tail", type=float, default=1.0)
    parser.add_argument("--subwindow-constraint-penalty", type=float, default=100.0)
    parser.add_argument("--subwindow-constraints-mode", choices=("gate", "penalty"), default="gate")
    parser.add_argument("--threshold-selection-objective", choices=("subwindow", "recent_daily_quantile"), default="subwindow")
    parser.add_argument("--recent-quantile-days", type=float, default=20.0)
    parser.add_argument("--recent-quantile-level", type=float, default=0.25)
    parser.add_argument("--recent-quantile-median-weight", type=float, default=0.25)
    parser.add_argument("--recent-quantile-mean-weight", type=float, default=0.05)
    parser.add_argument("--recent-quantile-iqr-penalty", type=float, default=0.10)
    parser.add_argument("--recent-quantile-weight-mode", choices=("uniform", "bucket"), default="uniform")
    parser.add_argument("--recent-quantile-weight-last-7", type=float, default=1.0)
    parser.add_argument("--recent-quantile-weight-prev-7", type=float, default=1.0)
    parser.add_argument("--recent-quantile-weight-older", type=float, default=1.0)
    parser.add_argument(
        "--similarity-prior-enable",
        action="store_true",
        help=(
            "Add a causal same-head similar-period prior to the daily Y grid. "
            "The query context uses only the latest prior day(s), and matched "
            "outcome days are strictly earlier than the query context."
        ),
    )
    parser.add_argument("--similarity-prior-ev-weight", type=float, default=0.0)
    parser.add_argument("--similarity-prior-hr-weight", type=float, default=0.0)
    parser.add_argument("--similarity-prior-hr-floor", type=float, default=0.35)
    parser.add_argument("--similarity-prior-query-recent-days", type=int, default=1)
    parser.add_argument("--similarity-prior-top-k-days", type=int, default=5)
    parser.add_argument("--similarity-prior-min-days", type=int, default=3)
    parser.add_argument("--similarity-prior-temperature", type=float, default=1.0)
    parser.add_argument("--threshold-trials", type=int, default=120)
    parser.add_argument("--threshold-refresh-mode", choices=("grid", "optuna"), default="grid")
    parser.add_argument("--threshold-grid-size", type=int, default=201)
    parser.add_argument("--min-threshold-selected-count", type=int, default=20)
    parser.add_argument("--min-threshold-active-subwindows", type=int, default=2)
    parser.add_argument("--recent-validation-guard", action="store_true")
    parser.add_argument("--recent-validation-days", type=float, default=5.0)
    parser.add_argument("--recent-validation-min-count", type=int, default=20)
    parser.add_argument("--recent-validation-min-total-pnl", type=float, default=0.0)
    parser.add_argument("--recent-validation-min-hit-rate", type=float, default=0.30)
    parser.add_argument("--recent-validation-min-avg-pnl", type=float, default=-1.0e18)
    parser.add_argument("--recent-validation-step", type=float, default=0.01)
    parser.add_argument("--quality-gate-enable", action="store_true")
    parser.add_argument("--quality-gate-target-hit-rate", type=float, default=0.45)
    parser.add_argument("--quality-gate-p-hit-min", type=float, default=0.50)
    parser.add_argument("--quality-gate-p-hit-max", type=float, default=0.90)
    parser.add_argument("--quality-gate-p-hit-step", type=float, default=0.01)
    parser.add_argument("--quality-gate-min-selected-count", type=int, default=20)
    parser.add_argument("--quality-gate-min-keep-fraction", type=float, default=0.0)
    parser.add_argument("--quality-gate-min-total-pnl", type=float, default=-1.0e18)
    parser.add_argument("--quality-gate-min-avg-pnl", type=float, default=-1.0e18)
    parser.add_argument("--quality-gate-allow-deactivation", action="store_true")
    parser.add_argument("--quality-gate-deactivate-if-no-pass", action="store_true")
    parser.add_argument("--deployed-threshold-soft-prior-strength", type=float, default=0.0)
    parser.add_argument("--deployed-threshold-soft-prior-deadband", type=float, default=0.0)
    parser.add_argument("--deployed-threshold-soft-prior-power", type=float, default=2.0)
    parser.add_argument("--deployed-threshold-soft-prior-activity-weight", type=float, default=0.0)
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--walk-forward-xw-fit-mode", choices=("initial", "per-fold"), default="initial")
    parser.add_argument("--xw-fit-days", type=float, default=None)
    parser.add_argument("--train-days", type=float, default=28.0)
    parser.add_argument("--valid-days", type=float, default=7.0)
    parser.add_argument("--step-days", type=float, default=7.0)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--calendar-replay", action="store_true")
    parser.add_argument(
        "--calendar-only",
        action="store_true",
        help="Run only the true calendar replay path; skip the in-period diagnostic fit.",
    )
    parser.add_argument("--calendar-xw-min-train-days", type=float, default=90.0)
    parser.add_argument("--calendar-xw-max-train-days", type=float, default=183.0)
    parser.add_argument("--calendar-y-train-days", type=float, default=28.0)
    parser.add_argument("--calendar-eval-start", default=None)
    parser.add_argument("--calendar-eval-end", default=None)
    parser.add_argument("--calendar-min-xw-train-rows", type=int, default=1000)
    parser.add_argument("--calendar-min-y-train-rows", type=int, default=100)
    args = parser.parse_args()
    _apply_policy_preset(args)
    if bool(args.calendar_only):
        args.calendar_replay = True
    if bool(args.use_meta_context_features) and str(args.head_optimization_mode) != "independent":
        raise ValueError("--use-meta-context-features currently requires --head-optimization-mode independent")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, contract = load_candidates(args)
    if frame.empty:
        raise RuntimeError("No usable candidate rows after schema normalization")
    frame["row_uid"] = np.arange(len(frame), dtype=np.int64)
    if bool(args.use_meta_context_features):
        _write_frame(output_dir / "meta_context_feature_diagnostics.parquet", _meta_context_feature_diagnostics(frame, contract))

    heads = set(frame["head"].dropna().astype(str).unique())
    deployed_thresholds = _thresholds_from_policy_params(Path(args.policy_params) if args.policy_params else None, heads)
    deployed_mask = frame["score"].to_numpy(dtype=float) >= frame["head"].map(deployed_thresholds).to_numpy(dtype=float)
    top30_mask = frame["rank"].to_numpy(dtype=float) >= float(args.top_rank_floor)

    if bool(args.calendar_only):
        run_calendar_replay(frame, args, deployed_thresholds=deployed_thresholds, output_dir=output_dir)
        manifest = {
            "candidate_path": str(Path(args.candidates)),
            "policy_params_path": str(args.policy_params) if args.policy_params else None,
            "policy_preset": str(getattr(args, "policy_preset", "default")),
            "period_start": pd.Timestamp(frame["timestamp"].min()).isoformat(),
            "period_end": pd.Timestamp(frame["timestamp"].max()).isoformat(),
            "candidate_count": int(len(frame)),
            "heads": sorted(heads),
            "schema_contract": contract,
            "calendar_only": True,
            "calendar_replay": True,
            "use_deployed_threshold_floor": bool(args.use_deployed_threshold_floor),
            "spread_adjust_returns": bool(args.spread_adjust_returns),
            "use_meta_context_features": bool(args.use_meta_context_features),
            "meta_context_tune_enable": bool(args.meta_context_tune_enable),
            "meta_context_transform": str(args.meta_context_transform),
            "meta_context_action_mode": str(args.meta_context_action_mode),
            "meta_context_bad_z_threshold": float(args.meta_context_bad_z_threshold),
            "meta_context_removed_trade_gate": bool(args.meta_context_removed_trade_gate),
            "meta_context_drift_columns": contract.get("meta_context_drift_columns", []),
            "meta_context_uncertainty_columns": contract.get("meta_context_uncertainty_columns", []),
            "meta_badness_cutoff_default": float(args.meta_badness_cutoff_default),
            "meta_badness_cutoff_range": [float(args.meta_badness_cutoff_min), float(args.meta_badness_cutoff_max)],
            "meta_badness_ridge_alpha": float(args.meta_badness_ridge_alpha),
            "meta_badness_min_train_rows": int(args.meta_badness_min_train_rows),
            "meta_badness_temperature_default": float(args.meta_badness_temperature_default),
            "meta_badness_temperature_range": [
                float(args.meta_badness_temperature_min),
                float(args.meta_badness_temperature_max),
            ],
            "meta_badness_pressure_scale_default": float(args.meta_badness_pressure_scale_default),
            "meta_badness_pressure_scale_range": [
                float(args.meta_badness_pressure_scale_min),
                float(args.meta_badness_pressure_scale_max),
            ],
            "calendar_xw_min_train_days": float(args.calendar_xw_min_train_days),
            "calendar_xw_max_train_days": float(args.calendar_xw_max_train_days),
            "calendar_y_train_days": float(args.calendar_y_train_days),
            "calendar_eval_start": args.calendar_eval_start,
            "calendar_eval_end": args.calendar_eval_end,
            "calendar_min_xw_train_rows": int(args.calendar_min_xw_train_rows),
            "calendar_min_y_train_rows": int(args.calendar_min_y_train_rows),
            "deployed_threshold_soft_prior_strength": float(args.deployed_threshold_soft_prior_strength),
            "deployed_threshold_soft_prior_deadband": float(args.deployed_threshold_soft_prior_deadband),
            "deployed_threshold_soft_prior_power": float(args.deployed_threshold_soft_prior_power),
            "deployed_threshold_soft_prior_activity_weight": float(args.deployed_threshold_soft_prior_activity_weight),
            "subwindow_days": float(args.subwindow_days),
            "subwindow_constraints_mode": str(args.subwindow_constraints_mode),
            "min_subwindows": int(args.min_subwindows),
            "min_positive_objective_fraction": float(args.min_positive_objective_fraction),
            "subwindow_q15_floor": float(args.subwindow_q15_floor),
            "subwindow_drawdown_floor": float(args.subwindow_drawdown_floor),
            "lambda_iqr": float(args.lambda_iqr),
            "lambda_tail": float(args.lambda_tail),
            "min_threshold_selected_count": int(args.min_threshold_selected_count),
            "min_threshold_active_subwindows": int(args.min_threshold_active_subwindows),
            "threshold_selection_objective": str(args.threshold_selection_objective),
            "recent_quantile_days": float(args.recent_quantile_days),
            "recent_quantile_level": float(args.recent_quantile_level),
            "recent_quantile_weight_mode": str(args.recent_quantile_weight_mode),
            "recent_quantile_weight_last_7": float(args.recent_quantile_weight_last_7),
            "recent_quantile_weight_prev_7": float(args.recent_quantile_weight_prev_7),
            "recent_quantile_weight_older": float(args.recent_quantile_weight_older),
            "similarity_prior_enable": bool(args.similarity_prior_enable),
            "similarity_prior_ev_weight": float(args.similarity_prior_ev_weight),
            "similarity_prior_hr_weight": float(args.similarity_prior_hr_weight),
            "similarity_prior_hr_floor": float(args.similarity_prior_hr_floor),
            "similarity_prior_query_recent_days": int(args.similarity_prior_query_recent_days),
            "similarity_prior_top_k_days": int(args.similarity_prior_top_k_days),
            "similarity_prior_min_days": int(args.similarity_prior_min_days),
            "similarity_prior_temperature": float(args.similarity_prior_temperature),
            **_context_linear_config(args),
            "recent_validation_guard": bool(args.recent_validation_guard),
            "recent_validation_days": float(args.recent_validation_days),
            "recent_validation_min_count": int(args.recent_validation_min_count),
            "recent_validation_min_total_pnl": float(args.recent_validation_min_total_pnl),
            "recent_validation_min_hit_rate": float(args.recent_validation_min_hit_rate),
            "quality_gate_enable": bool(args.quality_gate_enable),
            "quality_gate_target_hit_rate": float(args.quality_gate_target_hit_rate),
            "quality_gate_p_hit_min": float(args.quality_gate_p_hit_min),
            "quality_gate_p_hit_max": float(args.quality_gate_p_hit_max),
            "quality_gate_p_hit_step": float(args.quality_gate_p_hit_step),
            "quality_gate_min_selected_count": int(args.quality_gate_min_selected_count),
            "quality_gate_min_keep_fraction": float(args.quality_gate_min_keep_fraction),
            "quality_gate_min_total_pnl": float(args.quality_gate_min_total_pnl),
            "quality_gate_min_avg_pnl": float(args.quality_gate_min_avg_pnl),
            "quality_gate_allow_deactivation": bool(args.quality_gate_allow_deactivation),
            "quality_gate_deactivate_if_no_pass": bool(args.quality_gate_deactivate_if_no_pass),
            "threshold_formula": (
                "clip(max(deployed_threshold_h, "
                "Y_h - W_lower_h * max(0, z_h_t) - W_raise_h * min(0, z_h_t)), "
                "-0.50, 1.50); optional meta context raises threshold above tuned drift/uncertainty floors; "
                "linear_context mode continuously dampens lowering from meta/density pressure"
            ),
            "causal_surprise": "EWMA num/var/count are shifted by one timestamp per head before use",
        }
        (output_dir / "dynamic_hr_surprise_manifest.json").write_text(
            json.dumps(_json_default(manifest), indent=2),
            encoding="utf-8",
        )
        calendar_summary_path = output_dir / "calendar_dynamic_hr_surprise_policy_summary.csv"
        if calendar_summary_path.exists():
            calendar_summary = pd.read_csv(calendar_summary_path)
            print(calendar_summary.sort_values("objective", ascending=False).to_string(index=False))
        else:
            calendar_manifest = output_dir / "calendar_dynamic_hr_surprise_manifest.json"
            if calendar_manifest.exists():
                print(calendar_manifest.read_text(encoding="utf-8"))
        print(f"\nWrote {output_dir}")
        return

    params, study, surprise, trials = optimize_dynamic_policy(frame, args, deployed_thresholds=deployed_thresholds)
    dynamic_mask = _mask_from_params_with_args(
        frame,
        params,
        surprise,
        args,
        deployed_thresholds=deployed_thresholds,
    )

    # Also record the unguarded best trial for transparency.
    if study is not None:
        raw_params = {
            head: HeadParams(
                head=head,
                x_days=float(study.best_params[f"{head}__x_days"]),
                w=max(
                    float(study.best_params[f"{head}__w_lower"]),
                    (
                        max(float(study.best_params[f"{head}__w_lower"]), float(study.best_params[f"{head}__w_raise"]))
                        if bool(args.require_raise_sensitivity_at_least_lower)
                        else float(study.best_params[f"{head}__w_raise"])
                    ),
                ),
                y=float(study.best_params[f"{head}__y"]),
                guarded_y=float(study.best_params[f"{head}__y"]),
                guard_shift=0.0,
                local_band_pnl=np.nan,
                local_band_count=0,
                w_lower=float(study.best_params[f"{head}__w_lower"]),
                w_raise=(
                    max(float(study.best_params[f"{head}__w_lower"]), float(study.best_params[f"{head}__w_raise"]))
                    if bool(args.require_raise_sensitivity_at_least_lower)
                    else float(study.best_params[f"{head}__w_raise"])
                ),
            )
            for head in sorted(heads)
        }
        raw_surprise = build_surprise(
            frame,
            halflife_days_by_head={head: param.x_days for head, param in raw_params.items()},
            top_rank_floor=float(args.top_rank_floor),
            z_clip=float(args.z_clip),
        )
        raw_dynamic_mask = _mask_from_params(
            frame,
            raw_params,
            raw_surprise,
            deployed_thresholds=deployed_thresholds,
            use_deployed_threshold_floor=bool(args.use_deployed_threshold_floor),
        )
    else:
        raw_dynamic_mask = dynamic_mask.copy()

    masks = {
        "fixed_deployed_thresholds": deployed_mask,
        "all_top30_candidate_pool": top30_mask,
        "dynamic_hr_surprise_unguarded": raw_dynamic_mask,
        "dynamic_hr_surprise_guarded": dynamic_mask,
    }
    summary = pd.DataFrame([_policy_metrics(frame, mask, policy=policy) for policy, mask in masks.items()])
    _write_frame(output_dir / "dynamic_hr_surprise_policy_summary.parquet", summary)
    by_head = pd.concat([_by_head_metrics(frame, mask, policy=policy) for policy, mask in masks.items()], ignore_index=True)
    _write_frame(output_dir / "dynamic_hr_surprise_policy_by_head.parquet", by_head)
    monthly = pd.concat([_monthly_metrics(frame, mask, policy=policy) for policy, mask in masks.items()], ignore_index=True)
    _write_frame(output_dir / "dynamic_hr_surprise_policy_monthly.parquet", monthly)
    monthly_by_head = pd.concat([_monthly_by_head_metrics(frame, mask, policy=policy) for policy, mask in masks.items()], ignore_index=True)
    _write_frame(output_dir / "dynamic_hr_surprise_policy_monthly_by_head.parquet", monthly_by_head)
    weekly = pd.concat([_weekly_frame(frame, mask, policy=policy) for policy, mask in masks.items()], ignore_index=True)
    _write_frame(output_dir / "dynamic_hr_surprise_policy_rolling_weeks.parquet", weekly)

    params_frame = pd.DataFrame([asdict(param) for param in params.values()]).sort_values("head")
    params_frame["deployed_fixed_threshold"] = params_frame["head"].map(deployed_thresholds)
    _write_frame(output_dir / "dynamic_hr_surprise_params.parquet", params_frame)
    surprise.to_parquet(output_dir / "dynamic_hr_surprise_series.parquet", index=False)
    selected = frame.loc[dynamic_mask].copy()
    selected["dynamic_threshold"] = _threshold_vector(
        frame,
        params,
        surprise,
        deployed_thresholds=deployed_thresholds,
        use_deployed_threshold_floor=bool(args.use_deployed_threshold_floor),
        surprise_forecast_mode=str(args.surprise_forecast_mode),
        slope_tolerance=float(args.slope_tolerance),
        z_cap=float(args.z_clip),
        require_lowering_confirmation=bool(args.require_lowering_confirmation),
        meta_context_action_mode=str(args.meta_context_action_mode),
        meta_context_bad_z_threshold=float(args.meta_context_bad_z_threshold),
        meta_context_transform=str(args.meta_context_transform),
        context_linear_density_raise=float(getattr(args, "context_linear_density_raise", 0.0)),
        context_linear_density_floor=float(getattr(args, "context_linear_density_floor", 0.0)),
        context_linear_relaxation_dampen=float(getattr(args, "context_linear_relaxation_dampen", 1.0)),
        context_linear_pressure_raise=float(getattr(args, "context_linear_pressure_raise", 0.0)),
    )[dynamic_mask]
    selected.to_parquet(output_dir / "dynamic_hr_surprise_selected_trades.parquet", index=False)
    trials.to_csv(output_dir / "dynamic_hr_surprise_optuna_trials.csv", index=False)
    trials.to_parquet(output_dir / "dynamic_hr_surprise_optuna_trials.parquet", index=False)

    manifest = {
        "candidate_path": str(Path(args.candidates)),
        "policy_params_path": str(args.policy_params) if args.policy_params else None,
        "policy_preset": str(getattr(args, "policy_preset", "default")),
        "period_start": pd.Timestamp(frame["timestamp"].min()).isoformat(),
        "period_end": pd.Timestamp(frame["timestamp"].max()).isoformat(),
        "candidate_count": int(len(frame)),
        "heads": sorted(heads),
        "schema_contract": contract,
        "top_rank_floor": float(args.top_rank_floor),
        "threshold_formula": (
            "level: clip(max(deployed_threshold_h, "
            "Y_h - W_lower_h * max(0, z_eff_h_t) - W_raise_h * min(0, z_eff_h_t)), -0.50, 1.50); "
            "slope: clip(max(deployed_threshold_h, Y_h - W_lower_h * lower_signal - W_raise_h * raise_signal), -0.50, 1.50); "
            "optional meta context: threshold += drift_raise_h * max(0, meta_drift_h_t - drift_floor_h) "
            "+ uncertainty_raise_h * max(0, meta_uncertainty_h_t - uncertainty_floor_h); "
            "linear_context mode dampens relaxation and adds raise pressure continuously from meta/density pressure"
        ),
        "causal_surprise": "EWMA num/var/count are shifted by one timestamp per head before use",
        "objective": "per-head robust subwindow objective: median - lambda_iqr * IQR - lambda_tail * abs(min(0, q15)) with subwindow constraints",
        "use_deployed_threshold_floor": bool(args.use_deployed_threshold_floor),
        "spread_adjust_returns": bool(args.spread_adjust_returns),
        "head_optimization_mode": str(args.head_optimization_mode),
        "surprise_forecast_mode": str(args.surprise_forecast_mode),
        "surprise_count_shrink_n0": float(args.surprise_count_shrink_n0),
        "use_meta_context_features": bool(args.use_meta_context_features),
        "meta_context_tune_enable": bool(args.meta_context_tune_enable),
        "meta_context_transform": str(args.meta_context_transform),
        "meta_context_action_mode": str(args.meta_context_action_mode),
        "meta_context_bad_z_threshold": float(args.meta_context_bad_z_threshold),
        "meta_context_removed_trade_gate": bool(args.meta_context_removed_trade_gate),
        "meta_context_removed_min_count": int(args.meta_context_removed_min_count),
        "meta_context_removed_max_total_pnl": float(args.meta_context_removed_max_total_pnl),
        "meta_context_removed_max_avg_pnl": float(args.meta_context_removed_max_avg_pnl),
        "meta_context_drift_columns": contract.get("meta_context_drift_columns", []),
        "meta_context_uncertainty_columns": contract.get("meta_context_uncertainty_columns", []),
        "meta_context_missing_drift_columns": contract.get("meta_context_missing_drift_columns", []),
        "meta_context_missing_uncertainty_columns": contract.get("meta_context_missing_uncertainty_columns", []),
        "meta_context_feature_aggregation": str(args.meta_context_feature_aggregation),
        "meta_context_timestamp_aggregation": str(args.meta_context_timestamp_aggregation),
        "meta_badness_cutoff_default": float(args.meta_badness_cutoff_default),
        "meta_badness_cutoff_range": [float(args.meta_badness_cutoff_min), float(args.meta_badness_cutoff_max)],
        "meta_badness_ridge_alpha": float(args.meta_badness_ridge_alpha),
        "meta_badness_min_train_rows": int(args.meta_badness_min_train_rows),
        "meta_badness_temperature_default": float(args.meta_badness_temperature_default),
        "meta_badness_temperature_range": [
            float(args.meta_badness_temperature_min),
            float(args.meta_badness_temperature_max),
        ],
        "meta_badness_pressure_scale_default": float(args.meta_badness_pressure_scale_default),
        "meta_badness_pressure_scale_range": [
            float(args.meta_badness_pressure_scale_min),
            float(args.meta_badness_pressure_scale_max),
        ],
        "meta_drift_raise_range": [float(args.meta_drift_raise_min), float(args.meta_drift_raise_max)],
        "meta_drift_floor_range": [float(args.meta_drift_floor_min), float(args.meta_drift_floor_max)],
        "meta_uncertainty_raise_range": [float(args.meta_uncertainty_raise_min), float(args.meta_uncertainty_raise_max)],
        "meta_uncertainty_floor_range": [float(args.meta_uncertainty_floor_min), float(args.meta_uncertainty_floor_max)],
        **_context_linear_config(args),
        "slope_lag_min": int(args.slope_lag_min),
        "slope_lag_max": int(args.slope_lag_max),
        "slope_tolerance": float(args.slope_tolerance),
        "require_lowering_confirmation": bool(args.require_lowering_confirmation),
        "forecast_ridge_alpha": float(args.forecast_ridge_alpha),
        "forecast_train_fraction": float(args.forecast_train_fraction),
        "forecast_min_rows": int(args.forecast_min_rows),
        "forecast_min_valid_rows": int(args.forecast_min_valid_rows),
        "forecast_rho_min": float(args.forecast_rho_min),
        "forecast_rho_max": float(args.forecast_rho_max),
        "min_forecast_edge": float(args.min_forecast_edge),
        "w_lower_min": float(args.w_lower_min),
        "w_lower_max": float(args.w_lower_max),
        "w_raise_min": float(args.w_raise_min),
        "w_raise_max": float(args.w_raise_max),
        "require_raise_sensitivity_at_least_lower": bool(args.require_raise_sensitivity_at_least_lower),
        "deployed_threshold_soft_prior_strength": float(args.deployed_threshold_soft_prior_strength),
        "deployed_threshold_soft_prior_deadband": float(args.deployed_threshold_soft_prior_deadband),
        "deployed_threshold_soft_prior_power": float(args.deployed_threshold_soft_prior_power),
        "deployed_threshold_soft_prior_activity_weight": float(args.deployed_threshold_soft_prior_activity_weight),
        "per_head_min_objective": float(args.per_head_min_objective),
        "per_head_min_q05_week_pnl": float(args.per_head_min_q05_week_pnl),
        "per_head_min_q15_week_pnl": float(args.per_head_min_q15_week_pnl),
        "per_head_min_robust_objective": float(args.per_head_min_robust_objective),
        "tail_penalty_weight": float(args.tail_penalty_weight),
        "subwindow_days": float(args.subwindow_days),
        "subwindow_constraints_mode": str(args.subwindow_constraints_mode),
        "min_subwindows": int(args.min_subwindows),
        "min_positive_objective_fraction": float(args.min_positive_objective_fraction),
        "subwindow_q15_floor": float(args.subwindow_q15_floor),
        "subwindow_drawdown_floor": float(args.subwindow_drawdown_floor),
        "lambda_iqr": float(args.lambda_iqr),
        "lambda_tail": float(args.lambda_tail),
        "subwindow_constraint_penalty": float(args.subwindow_constraint_penalty),
        "threshold_trials": int(args.threshold_trials),
        "threshold_refresh_mode": str(args.threshold_refresh_mode),
        "threshold_grid_size": int(args.threshold_grid_size),
        "min_threshold_selected_count": int(args.min_threshold_selected_count),
        "min_threshold_active_subwindows": int(args.min_threshold_active_subwindows),
        "walk_forward_xw_fit_mode": str(args.walk_forward_xw_fit_mode),
        "xw_fit_days": None if args.xw_fit_days is None else float(args.xw_fit_days),
        "calendar_replay": bool(args.calendar_replay),
        "calendar_xw_min_train_days": float(args.calendar_xw_min_train_days),
        "calendar_xw_max_train_days": float(args.calendar_xw_max_train_days),
        "calendar_y_train_days": float(args.calendar_y_train_days),
        "calendar_eval_start": args.calendar_eval_start,
        "calendar_eval_end": args.calendar_eval_end,
        "calendar_min_xw_train_rows": int(args.calendar_min_xw_train_rows),
        "calendar_min_y_train_rows": int(args.calendar_min_y_train_rows),
        "threshold_selection_objective": str(args.threshold_selection_objective),
        "recent_quantile_days": float(args.recent_quantile_days),
        "recent_quantile_level": float(args.recent_quantile_level),
        "recent_quantile_weight_mode": str(args.recent_quantile_weight_mode),
        "recent_quantile_weight_last_7": float(args.recent_quantile_weight_last_7),
        "recent_quantile_weight_prev_7": float(args.recent_quantile_weight_prev_7),
        "recent_quantile_weight_older": float(args.recent_quantile_weight_older),
        "similarity_prior_enable": bool(args.similarity_prior_enable),
        "similarity_prior_ev_weight": float(args.similarity_prior_ev_weight),
        "similarity_prior_hr_weight": float(args.similarity_prior_hr_weight),
        "similarity_prior_hr_floor": float(args.similarity_prior_hr_floor),
        "similarity_prior_query_recent_days": int(args.similarity_prior_query_recent_days),
        "similarity_prior_top_k_days": int(args.similarity_prior_top_k_days),
        "similarity_prior_min_days": int(args.similarity_prior_min_days),
        "similarity_prior_temperature": float(args.similarity_prior_temperature),
        "best_objective": float(study.best_value) if study is not None else None,
        "params": [asdict(param) for param in params.values()],
        "walk_forward": bool(args.walk_forward),
        "walk_forward_train_days": float(args.train_days),
        "walk_forward_valid_days": float(args.valid_days),
        "walk_forward_step_days": float(args.step_days),
    }
    (output_dir / "dynamic_hr_surprise_manifest.json").write_text(json.dumps(_json_default(manifest), indent=2), encoding="utf-8")

    if args.walk_forward:
        run_walk_forward(frame, args, deployed_thresholds=deployed_thresholds, output_dir=output_dir)
    if args.calendar_replay:
        run_calendar_replay(frame, args, deployed_thresholds=deployed_thresholds, output_dir=output_dir)

    print(summary.sort_values("objective", ascending=False).to_string(index=False))
    if args.walk_forward and (output_dir / "walkforward_dynamic_hr_surprise_policy_summary.csv").exists():
        wf_summary = pd.read_csv(output_dir / "walkforward_dynamic_hr_surprise_policy_summary.csv")
        print("\nWalk-forward")
        print(wf_summary.sort_values("objective", ascending=False).to_string(index=False))
    print("\nParams")
    print(params_frame.to_string(index=False))
    print(f"\nWrote {output_dir}")


if __name__ == "__main__":
    main()
