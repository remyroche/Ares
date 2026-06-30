#!/usr/bin/env python3
"""Run fold-local performance-regime market-state modulation artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.performance_regimes.archetype_experts import (  # noqa: E402
    ArchetypeExpertConfig,
    train_archetype_experts,
)
from extreme_price_movements.performance_regimes.archetypes import (  # noqa: E402
    ArchetypeClusteringConfig,
    MarketStateArchetype,
    build_archetype_activity_targets,
    build_cross_strategy_archetype_features,
    cluster_leaves_into_archetypes,
)
from extreme_price_movements.performance_regimes.artifacts import (  # noqa: E402
    ensure_fold_artifact_dirs,
    write_frame,
    write_joblib,
    write_json,
)
from extreme_price_movements.performance_regimes.diagnostics import (  # noqa: E402
    PipelineStageReporter,
)
from extreme_price_movements.performance_regimes.feature_matrix import (  # noqa: E402
    build_market_state_feature_matrix,
)
from extreme_price_movements.performance_regimes.first_stage_models import (  # noqa: E402
    FirstStageLGBMConfig,
    TimeSeriesSplitSpec,
    train_first_stage_bad_good_models,
    walk_forward_splits,
)
from extreme_price_movements.performance_regimes.gatekeeping import (  # noqa: E402
    QuantStageGateConfig,
    StageGateError,
    gate_config_for_profile,
    evaluate_stage_gate,
)
from extreme_price_movements.performance_regimes.labels import (  # noqa: E402
    build_strategy_performance_labels,
)
from extreme_price_movements.performance_regimes.leaf_extraction import (  # noqa: E402
    extract_model_leaves,
)
from extreme_price_movements.performance_regimes.leaf_interactions import (  # noqa: E402
    extract_leaf_guided_interactions,
)
from extreme_price_movements.performance_regimes.leaf_scoring import prune_leaves  # noqa: E402
from extreme_price_movements.performance_regimes.portfolio_calibration import (  # noqa: E402
    PortfolioCalibratorConfig,
    build_portfolio_action_targets_from_labels,
    threshold_archetype_scores_for_modulation,
    train_portfolio_calibrator,
)
from extreme_price_movements.unsupervised_regime_learning.pipeline import (  # noqa: E402
    generate_operator_features,
)


def _load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _safe_scope_name(value: str) -> str:
    text = str(value).strip() or "strategy"
    out = []
    for char in text:
        if char.isalnum() or char in {"-", "_"}:
            out.append(char)
        else:
            out.append("_")
    return "".join(out).strip("_") or "strategy"


def _parse_feature_family(value: str) -> tuple[str, list[str]]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("feature family must be FAMILY:col1,col2")
    name, cols = value.split(":", 1)
    return name.strip(), [col.strip() for col in cols.split(",") if col.strip()]


def _parse_float_csv(value: str, *, default: tuple[float, ...]) -> tuple[float, ...]:
    text = str(value or "").strip()
    if not text:
        return tuple(float(v) for v in default)
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def _parse_modes(values: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    modes: list[str] = []
    for value in values or ():
        for part in str(value).split(","):
            text = part.strip().lower()
            if text:
                modes.append(text)
    return tuple(dict.fromkeys(modes))


def _parse_strategy_risk_modes(values: list[str] | tuple[str, ...] | None) -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    for value in values or ():
        if ":" not in str(value):
            raise argparse.ArgumentTypeError(
                "strategy risk mode must be STRATEGY:mode1,mode2"
            )
        strategy, raw_modes = str(value).split(":", 1)
        out[strategy.strip()] = _parse_modes([raw_modes])
    return out


def _first_stage_lgbm_config(args: argparse.Namespace) -> FirstStageLGBMConfig:
    """Build the fold-local soft-label LGBM config from CLI args."""

    return FirstStageLGBMConfig(
        max_depth=int(args.first_stage_max_depth),
        num_leaves=int(args.first_stage_num_leaves),
        min_child_samples_fraction=float(args.first_stage_min_child_samples_fraction),
        learning_rate=float(args.first_stage_learning_rate),
        n_estimators=int(args.n_estimators),
        subsample=float(args.first_stage_subsample),
        colsample_bytree=float(args.first_stage_colsample_bytree),
        min_gain_to_split=float(args.first_stage_min_gain_to_split),
        lambda_l1=float(args.first_stage_lambda_l1),
        lambda_l2=float(args.first_stage_lambda_l2),
        early_stopping_rounds=int(args.first_stage_early_stopping_rounds),
        random_state=int(args.first_stage_random_state),
    )


def _is_model_prediction_feature(name: str) -> bool:
    text = str(name)
    lowered = text.lower()
    blocked_prefixes = (
        "pred_",
        "raw_pred",
        "base_pred",
        "meta_pred",
        "mean_pred",
        "mean_tail_pred",
        "tail_pred",
        "oof_pred",
        "pred_h",
        "pred_logit",
        "base_h",
        "base_prob",
        "base_model",
    )
    return lowered.startswith(blocked_prefixes) or "qfail" in lowered


def _sanitize_feature_families(
    feature_families: dict[str, list[str]],
) -> tuple[dict[str, list[str]], list[str]]:
    removed: list[str] = []
    sanitized: dict[str, list[str]] = {}
    for family, columns in feature_families.items():
        kept = []
        for column in columns:
            if _is_model_prediction_feature(str(column)):
                removed.append(str(column))
            else:
                kept.append(str(column))
        sanitized[str(family)] = kept
    return sanitized, sorted(dict.fromkeys(removed))


def _optuna_archetype_score_threshold_range_from_p_active(
    args: argparse.Namespace,
    *,
    min_p_active: float,
) -> tuple[float, float, float, float]:
    """Map the requested p-active certainty range to normalized Optuna scores."""

    base_floor = float(np.clip(min_p_active, 0.0, 1.0))
    raw_score_low = float(
        np.clip(
            min(args.optuna_archetype_modulation_score_low, args.optuna_archetype_modulation_score_high),
            0.0,
            1.0,
        )
    )
    raw_score_high = float(
        np.clip(
            max(args.optuna_archetype_modulation_score_low, args.optuna_archetype_modulation_score_high),
            0.0,
            1.0,
        )
    )
    p_low = float(
        np.clip(
            min(args.optuna_archetype_min_p_active_low, args.optuna_archetype_min_p_active_high),
            0.0,
            1.0,
        )
    )
    p_high = float(
        np.clip(
            max(args.optuna_archetype_min_p_active_low, args.optuna_archetype_min_p_active_high),
            0.0,
            1.0,
        )
    )
    p_low = max(p_low, base_floor)
    p_high = max(p_high, p_low)
    denom = max(1.0 - base_floor, 1e-6)
    p_score_low = float(np.clip((p_low - base_floor) / denom, 0.0, 1.0))
    p_score_high = float(np.clip((p_high - base_floor) / denom, 0.0, 1.0))
    score_low = max(raw_score_low, p_score_low)
    score_high = min(raw_score_high, p_score_high)
    if score_high < score_low:
        score_high = score_low
    effective_p_low = float(np.clip(base_floor + score_low * (1.0 - base_floor), 0.0, 1.0))
    effective_p_high = float(np.clip(base_floor + score_high * (1.0 - base_floor), 0.0, 1.0))
    return float(score_low), float(score_high), effective_p_low, effective_p_high


def _default_feature_families(
    frame: pd.DataFrame,
    *,
    timestamp_col: str,
    strategy_col: str,
    performance_col: str,
) -> dict[str, list[str]]:
    excluded = {timestamp_col, strategy_col, performance_col}
    numeric = [
        col
        for col in frame.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(frame[col])
    ]
    return {"general_market_state": numeric}


def _safe_feature_token(value: str) -> str:
    text = str(value)
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")[:96] or "feature"


def _timestamp_filter(
    frame: pd.DataFrame,
    timestamps: pd.Index,
    *,
    timestamp_col: str,
) -> pd.DataFrame:
    ts = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
    wanted = set(pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True)).tolist())
    return frame.loc[ts.isin(wanted)].copy()


def _strategy_return_matrix(
    frame: pd.DataFrame,
    *,
    timestamp_col: str,
    strategy_col: str,
    performance_col: str,
    timestamps: pd.Index,
    strategies: list[str],
) -> pd.DataFrame:
    local = frame[[timestamp_col, strategy_col, performance_col]].copy()
    local[timestamp_col] = pd.to_datetime(local[timestamp_col], utc=True, errors="coerce")
    index = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    matrix = (
        local.dropna(subset=[timestamp_col])
        .groupby([timestamp_col, strategy_col], sort=True)[performance_col]
        .mean()
        .unstack(strategy_col)
        .reindex(index=index, columns=[str(s) for s in strategies])
        .fillna(0.0)
        .astype(np.float32, copy=False)
    )
    return matrix


def _lagged_head_streak_risk_features(
    labels,
    timestamps: pd.Index,
    *,
    strategies: list[str],
    full_pressure_hours: float,
) -> pd.DataFrame:
    """Build causal per-head path-risk features for the portfolio calibrator."""

    index = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    full_hours = max(float(full_pressure_hours), 1e-9)
    if len(index) > 1:
        diffs = index.sort_values().to_series().diff().dropna().dt.total_seconds().to_numpy(dtype=float)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        step_hours = max(float(np.nanmedian(diffs) / 3600.0), 1e-9) if len(diffs) else 1.0
    else:
        step_hours = 1.0
    columns: dict[str, pd.Series] = {}
    for strategy in [str(s) for s in strategies]:
        label_set = labels.by_strategy.get(strategy)
        if label_set is None:
            continue
        streak = (
            pd.to_numeric(label_set.loss_streak_hours, errors="coerce")
            .reindex(index)
            .fillna(0.0)
            .shift(1)
            .fillna(0.0)
            .clip(lower=0.0)
        )
        pressure = (
            pd.to_numeric(label_set.loss_streak_bad_pressure, errors="coerce")
            .reindex(index)
            .fillna(0.0)
            .shift(1)
            .fillna(0.0)
            .clip(0.0, 1.0)
        )
        prefix = f"{strategy}__head_streak"
        columns[f"{prefix}_hours_lag1"] = streak.astype(np.float32, copy=False)
        columns[f"{prefix}_pressure_lag1"] = pressure.astype(np.float32, copy=False)
        for attr, feature_name in [
            ("loss_density_bad_pressure", "slow_bleed_density_pressure"),
            ("drawdown_bad_pressure", "drawdown_pressure"),
            ("utility_bad_pressure", "utility_lcb_pressure"),
            ("cooldown_bad_pressure", "cooldown_pressure"),
            ("composite_bad_pressure", "composite_bad_pressure"),
        ]:
            series = getattr(label_set, attr, None)
            if series is None:
                continue
            columns[f"{prefix}_{feature_name}_lag1"] = (
                pd.to_numeric(series, errors="coerce")
                .reindex(index)
                .fillna(0.0)
                .shift(1)
                .fillna(0.0)
                .clip(0.0, 1.0)
                .astype(np.float32, copy=False)
            )
        columns[f"{prefix}_active_lag1"] = streak.gt(0.0).astype(np.float32, copy=False)
        columns[f"{prefix}_full_pressure_ratio_lag1"] = (
            streak / full_hours
        ).clip(0.0, 1.0).astype(np.float32, copy=False)
        perf_lag = (
            pd.to_numeric(label_set.strategy_performance, errors="coerce")
            .reindex(index)
            .fillna(0.0)
            .shift(1)
            .fillna(0.0)
            .astype(float)
        )
        ewma_lag = (
            pd.to_numeric(label_set.ewma_performance, errors="coerce")
            .reindex(index)
            .fillna(0.0)
            .shift(1)
            .fillna(0.0)
            .astype(float)
        )
        negative = perf_lag.lt(0.0).astype(np.float32)
        loss = (-perf_lag.clip(upper=0.0)).astype(float)
        gain = perf_lag.clip(lower=0.0).astype(float)
        cumulative = perf_lag.cumsum()
        drawdown_abs = (cumulative.cummax() - cumulative).clip(lower=0.0)
        columns[f"{prefix}_return_lag1"] = perf_lag.astype(np.float32, copy=False)
        columns[f"{prefix}_ewma_perf_lag1"] = ewma_lag.astype(np.float32, copy=False)
        columns[f"{prefix}_ewma_perf_delta_lag1"] = ewma_lag.diff().fillna(0.0).astype(
            np.float32,
            copy=False,
        )
        columns[f"{prefix}_drawdown_abs_lag1"] = drawdown_abs.astype(np.float32, copy=False)
        for window_hours in (24.0, 72.0, 168.0):
            bars = max(1, int(math.ceil(float(window_hours) / step_hours)))
            suffix = f"{int(window_hours)}h_lag1"
            loss_sum = loss.rolling(bars, min_periods=1).sum()
            gain_sum = gain.rolling(bars, min_periods=1).sum()
            return_sum = perf_lag.rolling(bars, min_periods=1).sum()
            columns[f"{prefix}_neg_share_{suffix}"] = (
                negative.rolling(bars, min_periods=1).mean().fillna(0.0).astype(np.float32, copy=False)
            )
            columns[f"{prefix}_loss_sum_{suffix}"] = loss_sum.astype(np.float32, copy=False)
            columns[f"{prefix}_return_sum_{suffix}"] = return_sum.astype(np.float32, copy=False)
            columns[f"{prefix}_loss_to_gain_ratio_{suffix}"] = (
                loss_sum / (gain_sum + 1e-12)
            ).clip(0.0, 1e6).astype(np.float32, copy=False)
            columns[f"{prefix}_ewma_recovery_slope_{suffix}"] = (
                (ewma_lag - ewma_lag.shift(bars).fillna(ewma_lag.iloc[0] if len(ewma_lag) else 0.0))
                / float(max(window_hours, 1e-9))
            ).astype(np.float32, copy=False)
        composite = columns.get(f"{prefix}_composite_bad_pressure_lag1")
        if composite is not None:
            columns[f"{prefix}_hazard_pressure_lag1"] = (
                composite
                * (
                    columns[f"{prefix}_full_pressure_ratio_lag1"]
                    + columns[f"{prefix}_neg_share_24h_lag1"]
                ).clip(0.0, 1.0)
            ).clip(0.0, 1.0).astype(np.float32, copy=False)
    if not columns:
        return pd.DataFrame(index=index)
    return pd.DataFrame(columns, index=index).astype(np.float32, copy=False)


def _execution_regime_feature_frame(
    X: pd.DataFrame,
    cross_features: pd.DataFrame,
    *,
    pattern: str,
    max_features: int,
) -> pd.DataFrame:
    if int(max_features) <= 0:
        return pd.DataFrame(index=X.index)
    regex = re.compile(str(pattern), re.IGNORECASE)
    candidates = [str(col) for col in X.columns if regex.search(str(col))]
    if not candidates:
        return pd.DataFrame(index=X.index)
    variances = (
        X.loc[:, candidates]
        .replace([np.inf, -np.inf], np.nan)
        .apply(pd.to_numeric, errors="coerce")
        .var(axis=0, ddof=0)
        .sort_values(ascending=False)
    )
    selected = [str(col) for col in variances.head(int(max_features)).index]
    columns: dict[str, pd.Series] = {}
    for col in selected:
        series = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = float(series.median()) if series.notna().any() else 0.0
        q75 = float(series.quantile(0.75)) if series.notna().any() else 0.0
        q25 = float(series.quantile(0.25)) if series.notna().any() else 0.0
        scale = max(q75 - q25, float(series.std(ddof=0)) if series.notna().any() else 0.0, 1e-9)
        z = ((series.fillna(median) - median) / scale).clip(-8.0, 8.0).astype(np.float32, copy=False)
        token = _safe_feature_token(col)
        columns[f"exec__{token}__robust_z"] = z
        for cross_col in ["bad_breadth", "good_breadth", "bad_concentration", "good_concentration"]:
            if cross_col in cross_features.columns:
                cross = pd.to_numeric(cross_features[cross_col], errors="coerce").reindex(X.index).fillna(0.0)
                columns[f"exec__{token}__x_{cross_col}"] = (z * cross).astype(np.float32, copy=False)
    if not columns:
        return pd.DataFrame(index=X.index)
    return pd.DataFrame(columns, index=X.index).astype(np.float32, copy=False)


def _labels_to_frame(labels) -> pd.DataFrame:
    rows = []
    for strategy, label_set in labels.by_strategy.items():
        rows.append(
            pd.DataFrame(
                {
                    "strategy": strategy,
                    "timestamp": label_set.timestamps,
                    "strategy_performance": label_set.strategy_performance.to_numpy(dtype=float),
                    "ewma_performance": label_set.ewma_performance.to_numpy(dtype=float),
                    "bad_label": label_set.bad_label.to_numpy(dtype=float),
                    "good_label": label_set.good_label.to_numpy(dtype=float),
                    "bad_sample_weight": label_set.bad_sample_weight.to_numpy(dtype=float),
                    "good_sample_weight": label_set.good_sample_weight.to_numpy(dtype=float),
                    "loss_streak_hours": label_set.loss_streak_hours.to_numpy(dtype=float),
                    "loss_streak_bad_pressure": label_set.loss_streak_bad_pressure.to_numpy(dtype=float),
                    "loss_density_bad_pressure": label_set.loss_density_bad_pressure.to_numpy(dtype=float),
                    "drawdown_bad_pressure": label_set.drawdown_bad_pressure.to_numpy(dtype=float),
                    "utility_bad_pressure": label_set.utility_bad_pressure.to_numpy(dtype=float),
                    "forward_bad_pressure": label_set.forward_bad_pressure.to_numpy(dtype=float),
                    "cooldown_bad_pressure": label_set.cooldown_bad_pressure.to_numpy(dtype=float),
                    "composite_bad_pressure": label_set.composite_bad_pressure.to_numpy(dtype=float),
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _leaves_to_frame(leaves: list) -> pd.DataFrame:
    rows = []
    for leaf in leaves:
        row = {
            key: value
            for key, value in leaf.__dict__.items()
            if key != "timestamp_membership"
        }
        row["active_timestamp_count"] = int(np.asarray(leaf.timestamp_membership, dtype=bool).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _action_targets_to_frame(action_targets) -> pd.DataFrame:
    rows = []
    for strategy, frame in action_targets.by_strategy.items():
        work = frame.reset_index()
        first_col = str(work.columns[0])
        if first_col != "timestamp":
            work = work.rename(columns={first_col: "timestamp"})
        work["strategy"] = strategy
        rows.append(work)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _mean_oof_brier(frame: pd.DataFrame) -> float:
    if frame.empty or "oof_weighted_brier" not in frame.columns:
        return np.inf
    values = pd.to_numeric(frame["oof_weighted_brier"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return float(values.mean()) if values.notna().any() else np.inf


def _operator_frame_from_timestamp_matrix(
    X: pd.DataFrame,
    *,
    timestamp_col: str,
) -> pd.DataFrame:
    out = X.reset_index()
    first_col = str(out.columns[0])
    if first_col != timestamp_col:
        out = out.rename(columns={first_col: timestamp_col})
    out["symbol"] = "__timestamp_market_state__"
    return out


def _archetype_quality(archetype) -> float:
    contribution = max(0.0, float(getattr(archetype, "mean_contribution_share", 0.0)))
    edge = max(0.0, float(getattr(archetype, "mean_edge_mass", 0.0)))
    stability = max(0.0, float(getattr(archetype, "mean_stability", 0.0)))
    coverage = max(0.0, float(getattr(archetype, "total_weighted_coverage", 0.0)))
    return float(np.log1p(contribution) * max(edge, 1e-12) * stability * np.sqrt(coverage))


def _archetype_active_positions(archetype) -> np.ndarray:
    activation = np.asarray(getattr(archetype, "activation_timestamps", []))
    if activation.dtype == bool:
        return np.flatnonzero(activation).astype(np.int32, copy=False)
    return np.asarray(activation, dtype=np.int32)


def _archetype_jaccard(left, right) -> float:
    a = np.unique(_archetype_active_positions(left))
    b = np.unique(_archetype_active_positions(right))
    if a.size == 0 and b.size == 0:
        return 0.0
    return float(np.intersect1d(a, b, assume_unique=True).size / max(np.union1d(a, b).size, 1))


def _tuple_jaccard(left: tuple, right: tuple) -> float:
    a = set(str(v) for v in left)
    b = set(str(v) for v in right)
    if not a and not b:
        return 0.0
    return float(len(a & b) / max(len(a | b), 1))


def _archetype_distance(left, right) -> float:
    quality_left = _archetype_quality(left)
    quality_right = _archetype_quality(right)
    quality_scale = max(abs(quality_left), abs(quality_right), 1e-12)
    edge_delta = abs(float(left.mean_edge_mass) - float(right.mean_edge_mass))
    contrib_delta = abs(float(left.mean_contribution_share) - float(right.mean_contribution_share))
    stability_delta = abs(float(left.mean_stability) - float(right.mean_stability))
    quality_delta = abs(quality_left - quality_right) / quality_scale
    return float(
        0.45 * (1.0 - _archetype_jaccard(left, right))
        + 0.25 * (1.0 - _tuple_jaccard(left.dominant_features, right.dominant_features))
        + 0.10 * edge_delta
        + 0.10 * contrib_delta
        + 0.05 * stability_delta
        + 0.05 * quality_delta
    )


def _allocate_cluster_counts(group_sizes: pd.DataFrame, max_count: int) -> dict[tuple[str, str], int]:
    if group_sizes.empty:
        return {}
    keys = [(str(row.strategy), str(row.direction)) for row in group_sizes.itertuples(index=False)]
    max_count = max(1, int(max_count), len(keys))
    allocation = {key: 1 for key in keys}
    remaining = max_count - len(allocation)
    weights = group_sizes["count"].to_numpy(dtype=float)
    weights = weights / max(float(weights.sum()), 1e-12)
    extras = np.floor(weights * remaining).astype(int)
    for key, extra in zip(keys, extras):
        allocation[key] += int(extra)
    while sum(allocation.values()) < max_count:
        deficits = []
        for i, key in enumerate(keys):
            count = int(group_sizes.iloc[i]["count"])
            deficits.append((count - allocation[key], count, key))
        deficits.sort(reverse=True)
        for deficit, _count, key in deficits:
            if sum(allocation.values()) >= max_count:
                break
            if deficit > 0:
                allocation[key] += 1
    return {key: min(value, int(group_sizes.loc[(group_sizes["strategy"].eq(key[0])) & (group_sizes["direction"].eq(key[1])), "count"].iloc[0])) for key, value in allocation.items()}


def _merge_archetype_cluster(members: list, cluster_id: str) -> MarketStateArchetype:
    if len(members) == 1:
        archetype = members[0]
        return MarketStateArchetype(
            archetype_id=cluster_id,
            strategy=archetype.strategy,
            direction=archetype.direction,
            leaf_ids=archetype.leaf_ids,
            dominant_features=archetype.dominant_features,
            dominant_feature_families=archetype.dominant_feature_families,
            total_weighted_coverage=archetype.total_weighted_coverage,
            mean_edge_mass=archetype.mean_edge_mass,
            mean_contribution_share=archetype.mean_contribution_share,
            mean_stability=archetype.mean_stability,
            activation_timestamps=archetype.activation_timestamps,
            diagnostics={**dict(archetype.diagnostics), "source_archetype_ids": (archetype.archetype_id,), "compression_member_count": 1},
        )
    weights = np.asarray([max(_archetype_quality(member), 1e-12) for member in members], dtype=float)
    weights = weights / max(float(weights.sum()), 1e-12)
    feature_counts = Counter(
        feature
        for member in members
        for feature in tuple(getattr(member, "dominant_features", ()))
    )
    dominant = tuple(name for name, _count in feature_counts.most_common(16))
    family_counts = Counter(
        family
        for member in members
        for family in tuple(getattr(member, "dominant_feature_families", ()))
    )
    max_len = max((len(getattr(member, "activation_timestamps", [])) for member in members), default=0)
    activation = np.zeros(max_len, dtype=bool)
    for member in members:
        positions = _archetype_active_positions(member)
        positions = positions[(positions >= 0) & (positions < max_len)]
        activation[positions] = True
    leaf_ids = tuple(dict.fromkeys(leaf_id for member in members for leaf_id in member.leaf_ids))
    return MarketStateArchetype(
        archetype_id=cluster_id,
        strategy=str(members[0].strategy),
        direction=members[0].direction,
        leaf_ids=leaf_ids,
        dominant_features=dominant,
        dominant_feature_families=tuple(name for name, _count in family_counts.most_common(10)),
        total_weighted_coverage=float(sum(float(member.total_weighted_coverage) for member in members)),
        mean_edge_mass=float(np.average([float(member.mean_edge_mass) for member in members], weights=weights)),
        mean_contribution_share=float(np.average([float(member.mean_contribution_share) for member in members], weights=weights)),
        mean_stability=float(np.average([float(member.mean_stability) for member in members], weights=weights)),
        activation_timestamps=activation,
        diagnostics={
            "source_archetype_ids": tuple(str(member.archetype_id) for member in members),
            "compression_member_count": int(len(members)),
            "compression_quality_sum": float(sum(_archetype_quality(member) for member in members)),
            "source_leaf_count": int(len(leaf_ids)),
        },
    )


def _coefficient_of_variation(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    mean = float(np.mean(arr))
    if abs(mean) <= 1e-12:
        return 0.0
    return float(np.std(arr) / abs(mean))


def _compression_silhouettes(group_archetypes: list, labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=object)
    n = len(group_archetypes)
    if n == 0:
        return np.asarray([], dtype=float)
    unique_labels = pd.unique(labels)
    if n < 2 or len(unique_labels) < 2:
        return np.zeros(n, dtype=float)

    distances = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            value = _archetype_distance(group_archetypes[i], group_archetypes[j])
            distances[i, j] = value
            distances[j, i] = value

    silhouettes = np.zeros(n, dtype=float)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            silhouettes[i] = 0.0
            continue
        a = float(np.mean(distances[i, same]))
        b_values = []
        for label in unique_labels:
            if label == labels[i]:
                continue
            other = labels == label
            if other.any():
                b_values.append(float(np.mean(distances[i, other])))
        if not b_values:
            silhouettes[i] = 0.0
            continue
        b = float(min(b_values))
        denom = max(a, b)
        silhouettes[i] = float(np.clip((b - a) / denom, -1.0, 1.0)) if denom > 1e-12 else 0.0
    return silhouettes


def _build_archetype_compression_diagnostics(
    archetypes: tuple,
    assignment_report: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    report = assignment_report.copy()
    if report.empty:
        diagnostics = pd.DataFrame()
        return report, diagnostics, {
            "compression_group_count": 0.0,
            "compression_source_coverage_min": 1.0,
            "compression_silhouette_mean": 0.0,
            "compression_silhouette_q10": 0.0,
            "compression_member_count_cov_max": 0.0,
            "compression_distance_to_seed_mean": 0.0,
            "compression_distance_to_seed_p95": 0.0,
            "compression_distance_to_seed_max": 0.0,
        }

    by_id = {str(archetype.archetype_id): archetype for archetype in archetypes}
    report["compression_silhouette"] = np.nan
    group_rows: list[dict[str, object]] = []

    for (strategy, direction), group_report in report.groupby(["strategy", "direction"], sort=True):
        raw_ids = group_report["archetype_id"].astype(str).tolist()
        group_archetypes = [by_id[raw_id] for raw_id in raw_ids if raw_id in by_id]
        labels = group_report["compressed_archetype_id"].astype(str).to_numpy()
        silhouettes = _compression_silhouettes(group_archetypes, labels[: len(group_archetypes)])
        if len(silhouettes) == len(group_report):
            report.loc[group_report.index, "compression_silhouette"] = silhouettes

        cluster_counts = group_report.groupby("compressed_archetype_id", sort=False).size().to_numpy(dtype=float)
        distances = pd.to_numeric(group_report["compression_distance_to_seed"], errors="coerce")
        qualities = pd.to_numeric(group_report["quality"], errors="coerce").clip(lower=0.0).fillna(0.0)
        weights = qualities.to_numpy(dtype=float) + 1e-12
        valid_sil = np.asarray(silhouettes, dtype=float)
        valid_sil = valid_sil[np.isfinite(valid_sil)]
        source_coverage = len(group_report) / max(len(group_archetypes), 1)
        group_rows.append(
            {
                "strategy": strategy,
                "direction": direction,
                "raw_archetype_count": int(len(group_archetypes)),
                "compressed_archetype_count": int(len(np.unique(labels))),
                "compression_ratio": float(len(np.unique(labels)) / max(len(group_archetypes), 1)),
                "source_coverage": float(source_coverage),
                "member_count_mean": float(np.mean(cluster_counts)) if cluster_counts.size else 0.0,
                "member_count_median": float(np.median(cluster_counts)) if cluster_counts.size else 0.0,
                "member_count_max": int(np.max(cluster_counts)) if cluster_counts.size else 0,
                "member_count_cov": _coefficient_of_variation(cluster_counts),
                "singleton_cluster_share": float(np.mean(cluster_counts <= 1.0)) if cluster_counts.size else 0.0,
                "distance_to_seed_mean": float(distances.mean()) if distances.notna().any() else 0.0,
                "distance_to_seed_p50": float(distances.quantile(0.50)) if distances.notna().any() else 0.0,
                "distance_to_seed_p95": float(distances.quantile(0.95)) if distances.notna().any() else 0.0,
                "distance_to_seed_max": float(distances.max()) if distances.notna().any() else 0.0,
                "silhouette_mean": float(np.mean(valid_sil)) if valid_sil.size else 0.0,
                "silhouette_q10": float(np.quantile(valid_sil, 0.10)) if valid_sil.size else 0.0,
                "silhouette_min": float(np.min(valid_sil)) if valid_sil.size else 0.0,
                "quality_weighted_silhouette_mean": float(np.average(silhouettes, weights=weights))
                if len(silhouettes) == len(weights)
                else np.nan,
                "quality_weighted_distance_to_seed_mean": float(np.average(distances.fillna(0.0), weights=weights)),
            }
        )

    diagnostics = pd.DataFrame(group_rows)
    all_distances = pd.to_numeric(report["compression_distance_to_seed"], errors="coerce")
    all_silhouettes = pd.to_numeric(report["compression_silhouette"], errors="coerce")
    metrics = {
        "compression_group_count": float(len(diagnostics)),
        "compression_source_coverage_min": float(diagnostics["source_coverage"].min()) if not diagnostics.empty else 1.0,
        "compression_silhouette_mean": float(all_silhouettes.mean()) if all_silhouettes.notna().any() else 0.0,
        "compression_silhouette_q10": float(all_silhouettes.quantile(0.10)) if all_silhouettes.notna().any() else 0.0,
        "compression_silhouette_min": float(all_silhouettes.min()) if all_silhouettes.notna().any() else 0.0,
        "compression_member_count_cov_max": float(diagnostics["member_count_cov"].max()) if not diagnostics.empty else 0.0,
        "compression_member_count_max": float(diagnostics["member_count_max"].max()) if not diagnostics.empty else 0.0,
        "compression_distance_to_seed_mean": float(all_distances.mean()) if all_distances.notna().any() else 0.0,
        "compression_distance_to_seed_p95": float(all_distances.quantile(0.95)) if all_distances.notna().any() else 0.0,
        "compression_distance_to_seed_max": float(all_distances.max()) if all_distances.notna().any() else 0.0,
    }
    return report, diagnostics, metrics


def _select_archetypes_for_experts(archetypes: tuple, max_count: int) -> tuple[tuple, pd.DataFrame]:
    rows = []
    for archetype in archetypes:
        rows.append(
            {
                "archetype_id": archetype.archetype_id,
                "strategy": archetype.strategy,
                "direction": archetype.direction,
                "quality": _archetype_quality(archetype),
                "mean_contribution_share": archetype.mean_contribution_share,
                "mean_edge_mass": archetype.mean_edge_mass,
                "mean_stability": archetype.mean_stability,
                "total_weighted_coverage": archetype.total_weighted_coverage,
                "selected": True,
                "compressed_archetype_id": archetype.archetype_id,
                "compression_member_count": 1,
                "compression_distance_to_seed": 0.0,
            }
        )
    report = pd.DataFrame(rows)
    if len(archetypes) <= int(max_count):
        return tuple(archetypes), report
    by_id = {str(archetype.archetype_id): archetype for archetype in archetypes}
    group_sizes = (
        report.groupby(["strategy", "direction"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["count", "strategy", "direction"], ascending=[False, True, True])
    )
    allocation = _allocate_cluster_counts(group_sizes, int(max_count))
    compressed: list[MarketStateArchetype] = []
    assignment_rows: list[pd.DataFrame] = []
    for (strategy, direction), group_report in report.groupby(["strategy", "direction"], sort=True):
        group_archetypes = [by_id[str(archetype_id)] for archetype_id in group_report["archetype_id"].astype(str)]
        k = int(allocation.get((str(strategy), str(direction)), 0))
        if k <= 0:
            continue
        seeds = sorted(group_archetypes, key=_archetype_quality, reverse=True)[:k]
        clusters: dict[str, list] = {str(seed.archetype_id): [] for seed in seeds}
        distances: dict[str, float] = {}
        for archetype in group_archetypes:
            seed = min(seeds, key=lambda candidate: _archetype_distance(archetype, candidate))
            seed_id = str(seed.archetype_id)
            clusters[seed_id].append(archetype)
            distances[str(archetype.archetype_id)] = _archetype_distance(archetype, seed)
        seed_order = {str(seed.archetype_id): i + 1 for i, seed in enumerate(seeds)}
        for seed_id, members in clusters.items():
            if not members:
                continue
            cluster_id = f"strategy_{strategy}_{direction}_compressed_archetype_{seed_order[seed_id]}"
            compressed_archetype = _merge_archetype_cluster(members, cluster_id)
            compressed.append(compressed_archetype)
            member_ids = {str(member.archetype_id) for member in members}
            chunk = group_report.loc[group_report["archetype_id"].astype(str).isin(member_ids)].copy()
            chunk["compressed_archetype_id"] = cluster_id
            chunk["compression_member_count"] = int(len(members))
            chunk["compression_distance_to_seed"] = chunk["archetype_id"].astype(str).map(distances).fillna(0.0)
            assignment_rows.append(chunk)
    compressed_report = pd.concat(assignment_rows, ignore_index=True) if assignment_rows else report.iloc[0:0].copy()
    compressed_report["selected"] = True
    return tuple(compressed), compressed_report


def _configure_logger(level_name: str) -> logging.Logger:
    logger = logging.getLogger("performance_market_state_modulator")
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=getattr(logging, level_name.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
    logger.setLevel(getattr(logging, level_name.upper(), logging.INFO))
    return logger


def _write_gate_reports(
    output_dir: Path,
    gate_rows: list[dict[str, Any]],
    *,
    fold_id: int | None = None,
    fold_dirs: dict[str, Path] | None = None,
) -> None:
    if not gate_rows:
        return
    gate_report = pd.DataFrame(gate_rows)
    write_frame(output_dir / "performance_market_state_gate_report.parquet", gate_report)
    write_frame(output_dir / "performance_market_state_gate_report.csv", gate_report)
    if fold_id is None or fold_dirs is None or "fold" not in gate_report.columns:
        return
    fold_values = pd.to_numeric(gate_report["fold"], errors="coerce")
    fold_report = gate_report.loc[fold_values == int(fold_id)].copy()
    if not fold_report.empty:
        write_frame(fold_dirs["evaluation"] / "pipeline_gate_report.parquet", fold_report)
        write_frame(fold_dirs["evaluation"] / "pipeline_gate_report.csv", fold_report)


def _record_stage_gate(
    *,
    stage: str,
    metrics: dict[str, Any],
    gate_config: QuantStageGateConfig,
    gate_rows: list[dict[str, Any]],
    reporter: PipelineStageReporter,
    output_dir: Path,
    fold_id: int | None = None,
    fold_dirs: dict[str, Path] | None = None,
) -> None:
    decision = evaluate_stage_gate(stage, metrics, gate_config)
    row = decision.to_row(fold=fold_id)
    gate_rows.append(row)
    reporter.event(
        f"{stage}__gate",
        "pass" if decision.passed else "fail",
        fold=fold_id,
        **{k: v for k, v in row.items() if k not in {"stage", "fold"}},
    )
    _write_gate_reports(output_dir, gate_rows, fold_id=fold_id, fold_dirs=fold_dirs)
    if gate_config.enabled and gate_config.fail_fast and not decision.passed:
        raise StageGateError(decision)


def _write_stage_reports(
    output_dir: Path,
    reporter: PipelineStageReporter,
    *,
    fold_id: int | None = None,
    fold_dirs: dict[str, Path] | None = None,
) -> None:
    stage_report = reporter.to_frame()
    stage_summary = reporter.summary_frame()
    if not stage_report.empty:
        write_frame(output_dir / "performance_market_state_stage_report.parquet", stage_report)
        write_frame(output_dir / "performance_market_state_stage_report.csv", stage_report)
    if not stage_summary.empty:
        write_frame(output_dir / "performance_market_state_stage_summary.parquet", stage_summary)
        write_frame(output_dir / "performance_market_state_stage_summary.csv", stage_summary)
    if fold_id is None or fold_dirs is None or stage_report.empty or "fold" not in stage_report.columns:
        return
    fold_values = pd.to_numeric(stage_report["fold"], errors="coerce")
    fold_report = stage_report.loc[fold_values == int(fold_id)].copy()
    if not fold_report.empty:
        write_frame(fold_dirs["evaluation"] / "pipeline_stage_report.parquet", fold_report)
        write_frame(fold_dirs["evaluation"] / "pipeline_stage_report.csv", fold_report)
    if not stage_summary.empty and "fold" in stage_summary.columns:
        summary_fold_values = pd.to_numeric(stage_summary["fold"], errors="coerce")
        fold_summary = stage_summary.loc[summary_fold_values == int(fold_id)].copy()
        if not fold_summary.empty:
            write_frame(fold_dirs["evaluation"] / "pipeline_stage_summary.parquet", fold_summary)
            write_frame(fold_dirs["evaluation"] / "pipeline_stage_summary.csv", fold_summary)


def _run_single_scope(args: argparse.Namespace) -> dict[str, Any]:
    logger = _configure_logger(getattr(args, "log_level", "INFO"))
    reporter = PipelineStageReporter(logger=logger)
    gate_config = gate_config_for_profile(
        getattr(args, "stage_gate_profile", "standard"),
        enabled=bool(getattr(args, "stage_gates", True)),
        fail_fast=bool(getattr(args, "stage_gate_fail_fast", True)),
    )
    gate_overrides: dict[str, float] = {}
    if getattr(args, "min_compression_silhouette_mean", None) is not None:
        gate_overrides["min_archetype_compression_silhouette_mean"] = float(args.min_compression_silhouette_mean)
    if getattr(args, "max_compression_member_cov", None) is not None:
        gate_overrides["max_archetype_compression_member_cov"] = float(args.max_compression_member_cov)
    if getattr(args, "max_compression_distance_to_seed_p95", None) is not None:
        gate_overrides["max_archetype_compression_distance_to_seed_p95"] = float(
            args.max_compression_distance_to_seed_p95
        )
    if getattr(args, "min_compression_source_coverage", None) is not None:
        gate_overrides["min_archetype_compression_source_coverage"] = float(args.min_compression_source_coverage)
    if gate_overrides:
        gate_config = replace(gate_config, **gate_overrides)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame()
    fold_summaries: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    current_fold_id: int | None = None
    current_fold_dirs: dict[str, Path] | None = None
    try:
        with reporter.stage("load_input", input_path=args.input) as metrics:
            frame = _load_frame(args.input)
            metrics.update(
                {
                    "row_count": int(len(frame)),
                    "column_count": int(frame.shape[1]),
                    "input_suffix": args.input.suffix,
                }
            )
            _record_stage_gate(
                stage="load_input",
                metrics=metrics,
                gate_config=gate_config,
                gate_rows=gate_rows,
                reporter=reporter,
                output_dir=args.output_dir,
            )

        with reporter.stage("resolve_strategies_and_features") as metrics:
            strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
            if not strategies:
                strategies = sorted(frame[args.strategy_col].astype(str).dropna().unique().tolist())
            family_items = [_parse_feature_family(value) for value in args.feature_family]
            feature_families = (
                {name: cols for name, cols in family_items}
                if family_items
                else _default_feature_families(
                    frame,
                    timestamp_col=args.timestamp_col,
                    strategy_col=args.strategy_col,
                    performance_col=args.performance_col,
                )
            )
            feature_families, excluded_model_prediction_features = _sanitize_feature_families(feature_families)
            risk_label_modes = _parse_modes(args.risk_label_mode)
            strategy_risk_label_modes = _parse_strategy_risk_modes(args.strategy_risk_label_mode)
            rolling_bad_regime_windows_hours = _parse_float_csv(
                args.rolling_bad_regime_window_hours,
                default=(24.0, 72.0, 168.0),
            )
            excluded_qfail_features = sorted(
                column for column in excluded_model_prediction_features if "qfail" in str(column).lower()
            )
            remaining_model_prediction_features = sorted(
                {
                    str(column)
                    for columns in feature_families.values()
                    for column in columns
                    if _is_model_prediction_feature(str(column))
                }
            )
            remaining_qfail_features = sorted(
                column for column in remaining_model_prediction_features if "qfail" in str(column).lower()
            )
            metrics.update(
                {
                    "strategy_count": int(len(strategies)),
                    "strategies": strategies,
                    "feature_family_count": int(len(feature_families)),
                    "requested_feature_count": int(sum(len(cols) for cols in feature_families.values())),
                    "feature_families": sorted(feature_families.keys()),
                    "excluded_model_prediction_feature_count": int(len(excluded_model_prediction_features)),
                    "excluded_model_prediction_features": excluded_model_prediction_features,
                    "remaining_model_prediction_feature_count": int(len(remaining_model_prediction_features)),
                    "remaining_model_prediction_features": remaining_model_prediction_features,
                    "excluded_qfail_feature_count": int(len(excluded_qfail_features)),
                    "excluded_qfail_features": excluded_qfail_features,
                    "remaining_qfail_feature_count": int(len(remaining_qfail_features)),
                    "remaining_qfail_features": remaining_qfail_features,
                    "risk_label_modes": risk_label_modes,
                    "strategy_risk_label_modes": {
                        key: list(value) for key, value in strategy_risk_label_modes.items()
                    },
                    "rolling_bad_regime_windows_hours": rolling_bad_regime_windows_hours,
                }
            )
            _record_stage_gate(
                stage="resolve_strategies_and_features",
                metrics=metrics,
                gate_config=gate_config,
                gate_rows=gate_rows,
                reporter=reporter,
                output_dir=args.output_dir,
            )

        with reporter.stage("build_outer_folds") as metrics:
            all_ts = pd.Index(
                sorted(pd.to_datetime(frame[args.timestamp_col], utc=True, errors="coerce").dropna().unique())
            )
            outer_cv = TimeSeriesSplitSpec(
                n_splits=int(args.outer_folds),
                purge_hours=float(args.embargo_hours),
                min_train_size=int(args.min_train_timestamps),
            )
            folds = walk_forward_splits(all_ts, outer_cv)
            metrics.update(
                {
                    "timestamp_count": int(len(all_ts)),
                    "outer_fold_count": int(len(folds)),
                    "min_train_timestamps": int(args.min_train_timestamps),
                    "embargo_hours": float(args.embargo_hours),
                }
            )
            _record_stage_gate(
                stage="build_outer_folds",
                metrics=metrics,
                gate_config=gate_config,
                gate_rows=gate_rows,
                reporter=reporter,
                output_dir=args.output_dir,
            )

        for fold_id, (train_pos, valid_pos) in enumerate(folds, start=1):
            current_fold_id = fold_id
            fold_root = args.output_dir / f"fold_{fold_id:02d}"
            dirs = ensure_fold_artifact_dirs(fold_root)
            current_fold_dirs = dirs
            with reporter.stage(
                "prepare_fold",
                fold=fold_id,
                fold_root=fold_root,
                train_timestamp_count=int(len(train_pos)),
                valid_timestamp_count=int(len(valid_pos)),
            ) as metrics:
                train_df = _timestamp_filter(frame, all_ts[train_pos], timestamp_col=args.timestamp_col)
                valid_df = _timestamp_filter(frame, all_ts[valid_pos], timestamp_col=args.timestamp_col)
                metrics.update(
                    {
                        "train_rows": int(len(train_df)),
                        "valid_rows": int(len(valid_df)),
                    }
                )
                _record_stage_gate(
                    stage="prepare_fold",
                    metrics=metrics,
                    gate_config=gate_config,
                    gate_rows=gate_rows,
                    reporter=reporter,
                    output_dir=args.output_dir,
                    fold_id=fold_id,
                    fold_dirs=dirs,
                )

            with reporter.stage("build_labels", fold=fold_id, strategy_count=len(strategies)) as metrics:
                labels = build_strategy_performance_labels(
                    train_df,
                    strategy_col=args.strategy_col,
                    timestamp_col=args.timestamp_col,
                    performance_col=args.performance_col,
                    strategies=strategies,
                    ewma_halflife=args.ewma_halflife,
                    loss_streak_target_min_hours=float(args.loss_streak_target_min_hours),
                    loss_streak_target_full_hours=float(args.loss_streak_target_full_hours),
                    loss_streak_label_weight=float(args.loss_streak_label_weight),
                    loss_streak_sample_weight_multiplier=float(
                        args.loss_streak_sample_weight_multiplier
                    ),
                    risk_label_modes=risk_label_modes,
                    strategy_risk_label_modes=strategy_risk_label_modes,
                    rolling_bad_regime_windows_hours=rolling_bad_regime_windows_hours,
                    loss_density_label_weight=float(args.loss_density_label_weight),
                    loss_density_min_negative_share=float(args.loss_density_min_negative_share),
                    loss_density_full_negative_share=float(args.loss_density_full_negative_share),
                    drawdown_label_weight=float(args.drawdown_label_weight),
                    drawdown_anchor_quantile=float(args.drawdown_anchor_quantile),
                    utility_label_weight=float(args.utility_label_weight),
                    utility_lcb_z_score=float(args.utility_lcb_z_score),
                    forward_bad_label_weight=float(args.forward_bad_label_weight),
                    forward_bad_window_hours=float(args.forward_bad_window_hours),
                    cooldown_label_weight=float(args.cooldown_label_weight),
                    cooldown_hours=float(args.cooldown_hours),
                    cooldown_trigger=float(args.cooldown_trigger),
                )
                label_frame = _labels_to_frame(labels)
                write_frame(dirs["labels"] / "strategy_label_anchors.parquet", labels.diagnostics)
                write_frame(dirs["labels"] / "strategy_bad_good_labels.parquet", label_frame)
                metrics.update(
                    {
                        "label_rows": int(len(label_frame)),
                        "anchor_rows": int(len(labels.diagnostics)),
                        "min_bad_label_std": float(
                            label_frame.groupby("strategy")["bad_label"].std(ddof=0).min()
                        )
                        if not label_frame.empty
                        else np.nan,
                        "artifact_labels": dirs["labels"],
                        "loss_streak_target_min_hours": float(args.loss_streak_target_min_hours),
                        "loss_streak_target_full_hours": float(args.loss_streak_target_full_hours),
                        "max_loss_streak_hours": float(
                            pd.to_numeric(
                                labels.diagnostics.get("max_loss_streak_hours"),
                                errors="coerce",
                            ).max()
                        )
                        if "max_loss_streak_hours" in labels.diagnostics
                        else np.nan,
                        "loss_streak_pressure_share": float(
                            pd.to_numeric(
                                labels.diagnostics.get("loss_streak_pressure_share"),
                                errors="coerce",
                            ).mean()
                        )
                        if "loss_streak_pressure_share" in labels.diagnostics
                        else np.nan,
                        "loss_density_pressure_share": float(
                            pd.to_numeric(
                                labels.diagnostics.get("loss_density_pressure_share"),
                                errors="coerce",
                            ).mean()
                        )
                        if "loss_density_pressure_share" in labels.diagnostics
                        else np.nan,
                        "drawdown_pressure_share": float(
                            pd.to_numeric(
                                labels.diagnostics.get("drawdown_pressure_share"),
                                errors="coerce",
                            ).mean()
                        )
                        if "drawdown_pressure_share" in labels.diagnostics
                        else np.nan,
                        "utility_pressure_share": float(
                            pd.to_numeric(
                                labels.diagnostics.get("utility_pressure_share"),
                                errors="coerce",
                            ).mean()
                        )
                        if "utility_pressure_share" in labels.diagnostics
                        else np.nan,
                        "forward_bad_pressure_share": float(
                            pd.to_numeric(
                                labels.diagnostics.get("forward_bad_pressure_share"),
                                errors="coerce",
                            ).mean()
                        )
                        if "forward_bad_pressure_share" in labels.diagnostics
                        else np.nan,
                        "cooldown_pressure_share": float(
                            pd.to_numeric(
                                labels.diagnostics.get("cooldown_pressure_share"),
                                errors="coerce",
                            ).mean()
                        )
                        if "cooldown_pressure_share" in labels.diagnostics
                        else np.nan,
                        "composite_bad_pressure_share": float(
                            pd.to_numeric(
                                labels.diagnostics.get("composite_bad_pressure_share"),
                                errors="coerce",
                            ).mean()
                        )
                        if "composite_bad_pressure_share" in labels.diagnostics
                        else np.nan,
                    }
                )
                _record_stage_gate(
                    stage="build_labels",
                    metrics=metrics,
                    gate_config=gate_config,
                    gate_rows=gate_rows,
                    reporter=reporter,
                    output_dir=args.output_dir,
                    fold_id=fold_id,
                    fold_dirs=dirs,
                )

            with reporter.stage("build_feature_matrices", fold=fold_id) as metrics:
                train_matrix = build_market_state_feature_matrix(
                    train_df,
                    timestamp_col=args.timestamp_col,
                    feature_families=feature_families,
                    aggregation_config={"*": ["mean", "std", "fraction_missing"]},
                )
                valid_matrix = build_market_state_feature_matrix(
                    valid_df,
                    timestamp_col=args.timestamp_col,
                    feature_families=feature_families,
                    aggregation_config={"*": ["mean", "std", "fraction_missing"]},
                )
                write_frame(dirs["features"] / "feature_family_coverage.parquet", train_matrix.diagnostics)
                write_frame(
                    dirs["features"] / "timestamp_feature_matrix.parquet",
                    train_matrix.X.reset_index(),
                )
                metrics.update(
                    {
                        "train_timestamp_count": int(train_matrix.X.shape[0]),
                        "valid_timestamp_count": int(valid_matrix.X.shape[0]),
                        "train_feature_count": int(train_matrix.X.shape[1]),
                        "valid_feature_count": int(valid_matrix.X.shape[1]),
                        "feature_family_diagnostic_rows": int(len(train_matrix.diagnostics)),
                        "missing_family_share": float(
                            train_matrix.diagnostics["missing_feature_count"].gt(0).mean()
                        )
                        if not train_matrix.diagnostics.empty
                        else 1.0,
                    }
                )
                _record_stage_gate(
                    stage="build_feature_matrices",
                    metrics=metrics,
                    gate_config=gate_config,
                    gate_rows=gate_rows,
                    reporter=reporter,
                    output_dir=args.output_dir,
                    fold_id=fold_id,
                    fold_dirs=dirs,
                )

            inner_cv = TimeSeriesSplitSpec(
                n_splits=int(args.inner_folds),
                purge=int(args.inner_purge_timestamps),
                min_train_size=max(5, int(args.min_inner_train_timestamps)),
            )
            with reporter.stage(
                "train_first_stage",
                fold=fold_id,
                inner_folds=int(args.inner_folds),
                n_estimators=int(args.n_estimators),
                first_stage_max_depth=int(args.first_stage_max_depth),
                first_stage_num_leaves=int(args.first_stage_num_leaves),
                first_stage_min_child_samples_fraction=float(args.first_stage_min_child_samples_fraction),
            ) as metrics:
                first_stage_config = _first_stage_lgbm_config(args)
                first_stage = train_first_stage_bad_good_models(
                    train_matrix.X,
                    labels,
                    strategies=strategies,
                    cv=inner_cv,
                    lgbm_config=first_stage_config,
                )
                write_joblib(
                    dirs["first_stage_models"] / "first_stage_model_bundle.joblib",
                    first_stage,
                )
                write_frame(dirs["evaluation"] / "first_stage_oof_metrics.parquet", first_stage.diagnostics)
                metrics.update(
                    {
                        "model_count": int(len(first_stage.by_strategy_direction)),
                        "diagnostic_rows": int(len(first_stage.diagnostics)),
                        "mean_oof_weighted_brier": _mean_oof_brier(first_stage.diagnostics),
                        "median_prediction_std": float(
                            pd.to_numeric(first_stage.diagnostics.get("prediction_std"), errors="coerce").median()
                        )
                        if "prediction_std" in first_stage.diagnostics
                        else np.nan,
                    }
                )
                _record_stage_gate(
                    stage="train_first_stage",
                    metrics=metrics,
                    gate_config=gate_config,
                    gate_rows=gate_rows,
                    reporter=reporter,
                    output_dir=args.output_dir,
                    fold_id=fold_id,
                    fold_dirs=dirs,
                )

            with reporter.stage("extract_score_prune_leaves", fold=fold_id) as metrics:
                leaf_table = extract_model_leaves(first_stage, train_matrix.X, labels)
                pruned = prune_leaves(list(leaf_table.leaves), min_stability=float(args.min_leaf_stability))
                write_frame(dirs["leaves"] / "extracted_leaves.parquet", leaf_table.frame)
                write_frame(dirs["leaves"] / "pruned_leaves.parquet", _leaves_to_frame(pruned))
                metrics.update(
                    {
                        "extracted_leaf_count": int(len(leaf_table.leaves)),
                        "pruned_leaf_count": int(len(pruned)),
                        "min_leaf_stability": float(args.min_leaf_stability),
                        "mean_pruned_leaf_stability": float(
                            np.nanmean([float(getattr(leaf, "stability", np.nan)) for leaf in pruned])
                        )
                        if pruned
                        else np.nan,
                        "mean_pruned_contribution_share": float(
                            np.nanmean([float(getattr(leaf, "contribution_share", np.nan)) for leaf in pruned])
                        )
                        if pruned
                        else np.nan,
                    }
                )
                _record_stage_gate(
                    stage="extract_score_prune_leaves",
                    metrics=metrics,
                    gate_config=gate_config,
                    gate_rows=gate_rows,
                    reporter=reporter,
                    output_dir=args.output_dir,
                    fold_id=fold_id,
                    fold_dirs=dirs,
                )

            with reporter.stage("extract_leaf_guided_interactions", fold=fold_id) as metrics:
                interactions = extract_leaf_guided_interactions(pruned)
                write_frame(dirs["interactions"] / "leaf_guided_pairs.parquet", interactions.pairs)
                write_frame(dirs["interactions"] / "leaf_guided_triples.parquet", interactions.triples)
                metrics.update(
                    {
                        "pair_count": int(len(interactions.pairs)),
                        "triple_count": int(len(interactions.triples)),
                        "diagnostic_rows": int(len(interactions.diagnostics)),
                        "interaction_gate_required": bool(int(args.max_feedback_passes) >= 1),
                    }
                )
                _record_stage_gate(
                    stage="extract_leaf_guided_interactions",
                    metrics=metrics,
                    gate_config=gate_config,
                    gate_rows=gate_rows,
                    reporter=reporter,
                    output_dir=args.output_dir,
                    fold_id=fold_id,
                    fold_dirs=dirs,
                )

            final_pruned = pruned
            final_expert_X = train_matrix.X
            composite_features = pd.DataFrame(index=train_matrix.X.index)
            feedback_rows = []
            with reporter.stage(
                "feedback_operator_generation_and_second_pass",
                fold=fold_id,
                max_feedback_passes=int(args.max_feedback_passes),
            ) as metrics:
                metrics["require_oof_improvement"] = bool(args.require_oof_improvement_for_second_pass)
                run_feedback = int(args.max_feedback_passes) >= 1 and (
                    not interactions.pairs.empty or not interactions.triples.empty
                )
                metrics["feedback_run"] = bool(run_feedback)
                if run_feedback:
                    operator_input = _operator_frame_from_timestamp_matrix(
                        train_matrix.X,
                        timestamp_col=args.timestamp_col,
                    )
                    composite_features = generate_operator_features(
                        operator_input,
                        primitive_features=list(train_matrix.X.columns),
                        seeded_pairs=interactions.pairs,
                        seeded_triples=interactions.triples,
                        mode="leaf_guided",
                        cfg={
                            "quality": {
                                "timestamp_col": args.timestamp_col,
                                "symbol_col": "symbol",
                            },
                            "operators": {
                                "pair_window": int(args.operator_window),
                                "quantile_window": int(args.operator_window),
                                "autocorr_window": int(args.operator_window),
                                "eigen_window": int(args.operator_window),
                                "min_periods": int(args.operator_min_periods),
                            },
                        },
                    )
                    composite_features.index = train_matrix.X.index
                    composite_features = composite_features.replace([np.inf, -np.inf], np.nan)
                    second_pass_X = pd.concat([train_matrix.X, composite_features], axis=1)
                    second_stage = train_first_stage_bad_good_models(
                        second_pass_X,
                        labels,
                        strategies=strategies,
                        cv=inner_cv,
                        lgbm_config=_first_stage_lgbm_config(args),
                    )
                    base_brier = _mean_oof_brier(first_stage.diagnostics)
                    second_brier = _mean_oof_brier(second_stage.diagnostics)
                    accept_second = (
                        second_brier < base_brier
                        or not bool(args.require_oof_improvement_for_second_pass)
                    )
                    feedback_rows.append(
                        {
                            "feedback_pass": 1,
                            "base_oof_weighted_brier": base_brier,
                            "second_pass_oof_weighted_brier": second_brier,
                            "accepted": bool(accept_second),
                            "generated_composite_feature_count": int(composite_features.shape[1]),
                        }
                    )
                    metrics.update(
                        {
                            "generated_composite_feature_count": int(composite_features.shape[1]),
                            "base_oof_weighted_brier": base_brier,
                            "second_pass_oof_weighted_brier": second_brier,
                            "second_pass_accepted": bool(accept_second),
                        }
                    )
                    if accept_second:
                        final_expert_X = second_pass_X
                        second_leaf_table = extract_model_leaves(second_stage, second_pass_X, labels)
                        final_pruned = prune_leaves(
                            list(second_leaf_table.leaves),
                            min_stability=float(args.min_leaf_stability),
                        )
                        write_frame(dirs["leaves"] / "second_pass_extracted_leaves.parquet", second_leaf_table.frame)
                        write_frame(dirs["leaves"] / "second_pass_pruned_leaves.parquet", _leaves_to_frame(final_pruned))
                        write_frame(
                            dirs["evaluation"] / "second_pass_first_stage_oof_metrics.parquet",
                            second_stage.diagnostics,
                        )
                        write_joblib(
                            dirs["first_stage_models"] / "second_pass_first_stage_model_bundle.joblib",
                            second_stage,
                        )
                        metrics.update(
                            {
                                "second_pass_extracted_leaf_count": int(len(second_leaf_table.leaves)),
                                "second_pass_pruned_leaf_count": int(len(final_pruned)),
                            }
                        )
                else:
                    metrics.update(
                        {
                            "skip_reason": "no_retained_leaf_guided_pairs_or_triples",
                            "generated_composite_feature_count": 0,
                        }
                    )
                write_frame(
                    dirs["features"] / "leaf_guided_composites.parquet",
                    composite_features.reset_index(),
                )
                manifest = pd.concat(
                    [
                        interactions.diagnostics,
                        pd.DataFrame(feedback_rows),
                    ],
                    axis=1,
                )
                write_frame(dirs["interactions"] / "generated_operator_features_manifest.parquet", manifest)
                _record_stage_gate(
                    stage="feedback_operator_generation_and_second_pass",
                    metrics=metrics,
                    gate_config=gate_config,
                    gate_rows=gate_rows,
                    reporter=reporter,
                    output_dir=args.output_dir,
                    fold_id=fold_id,
                    fold_dirs=dirs,
                )

            with reporter.stage("cluster_archetypes", fold=fold_id) as metrics:
                archetype_bundle = cluster_leaves_into_archetypes(
                    final_pruned,
                    clustering_config=ArchetypeClusteringConfig(),
                )
                selected_archetypes, archetype_selection = _select_archetypes_for_experts(
                    tuple(archetype_bundle.archetypes),
                    int(args.max_archetypes_for_experts),
                )
                archetype_selection, compression_diagnostics, compression_metrics = (
                    _build_archetype_compression_diagnostics(
                        tuple(archetype_bundle.archetypes),
                        archetype_selection,
                    )
                )
                if len(selected_archetypes) < len(archetype_bundle.archetypes):
                    write_json(dirs["archetypes"] / "archetype_definitions_all.json", archetype_bundle.archetypes)
                write_json(dirs["archetypes"] / "archetype_definitions.json", selected_archetypes)
                write_frame(dirs["archetypes"] / "archetype_selection.parquet", archetype_selection)
                write_frame(dirs["archetypes"] / "archetype_compression_diagnostics.parquet", compression_diagnostics)
                write_frame(dirs["archetypes"] / "archetype_similarity.parquet", archetype_bundle.similarity)
                metrics.update(
                    {
                        "input_leaf_count": int(len(final_pruned)),
                        "raw_archetype_count": int(len(archetype_bundle.archetypes)),
                        "archetype_count": int(len(selected_archetypes)),
                        "max_archetypes_for_experts": int(args.max_archetypes_for_experts),
                        "similarity_rows": int(len(archetype_bundle.similarity)),
                    }
                )
                metrics.update(compression_metrics)
                _record_stage_gate(
                    stage="cluster_archetypes",
                    metrics=metrics,
                    gate_config=gate_config,
                    gate_rows=gate_rows,
                    reporter=reporter,
                    output_dir=args.output_dir,
                    fold_id=fold_id,
                    fold_dirs=dirs,
                )

            experts = None
            intensities = pd.DataFrame(index=train_matrix.X.index)
            calibrator_archetypes = tuple(selected_archetypes)
            calibrator_expert_scores = pd.DataFrame(index=final_expert_X.index)
            predictive_archetype_ids: set[str] = set()
            with reporter.stage("train_archetype_experts", fold=fold_id) as metrics:
                if selected_archetypes:
                    targets = build_archetype_activity_targets(
                        selected_archetypes,
                        final_pruned,
                        final_expert_X.index,
                    )
                    intensities = pd.DataFrame(targets.activity, index=train_matrix.X.index)
                    write_frame(
                        dirs["archetypes"] / "archetype_intensities.parquet",
                        intensities.reset_index(),
                    )
                    experts = train_archetype_experts(
                        final_expert_X,
                        targets.activity,
                        targets.sample_weights,
                        cv=inner_cv,
                        config=ArchetypeExpertConfig(n_estimators=int(args.n_estimators)),
                    )
                    write_joblib(dirs["archetype_experts"] / "archetype_expert_bundle.joblib", experts)
                    write_frame(
                        dirs["evaluation"] / "archetype_expert_oof_metrics.parquet",
                        experts.diagnostics,
                    )
                    expert_pred_std = pd.to_numeric(
                        experts.diagnostics.get("prediction_std"),
                        errors="coerce",
                    )
                    predictive_threshold = float(gate_config.min_archetype_expert_prediction_std)
                    predictive_by_archetype = (
                        experts.diagnostics.assign(_prediction_std=expert_pred_std)
                        .groupby("archetype_id")["_prediction_std"]
                        .mean()
                        .gt(predictive_threshold)
                    )
                    predictive_archetype_ids = set(
                        str(idx) for idx, value in predictive_by_archetype.items() if bool(value)
                    )
                    if bool(args.calibrator_use_predictive_archetypes_only):
                        requested_ids = [str(a.archetype_id) for a in selected_archetypes]
                        kept_ids = [archetype_id for archetype_id in requested_ids if archetype_id in predictive_archetype_ids]
                        calibrator_archetypes = tuple(
                            archetype
                            for archetype in selected_archetypes
                            if str(archetype.archetype_id) in set(kept_ids)
                        )
                        calibrator_expert_scores = experts.scores.reindex(columns=kept_ids).fillna(0.0)
                    else:
                        calibrator_archetypes = tuple(selected_archetypes)
                        calibrator_expert_scores = experts.scores.fillna(0.0)
                    metrics.update(
                        {
                            "archetype_count": int(len(selected_archetypes)),
                            "expert_count": int(len(experts.by_archetype)),
                            "diagnostic_rows": int(len(experts.diagnostics)),
                            "activity_feature_count": int(experts.activity_scores.shape[1]),
                            "mean_oof_weighted_brier": _mean_oof_brier(experts.diagnostics),
                            "median_prediction_std": float(expert_pred_std.median()),
                            "mean_prediction_std": float(expert_pred_std.mean()),
                            "predictive_expert_fold_share": float(
                                expert_pred_std.gt(predictive_threshold).mean()
                            ),
                            "predictive_expert_count": int(predictive_by_archetype.sum()),
                            "calibrator_use_predictive_archetypes_only": bool(
                                args.calibrator_use_predictive_archetypes_only
                            ),
                            "calibrator_archetype_count": int(len(calibrator_archetypes)),
                        }
                    )
                else:
                    write_frame(dirs["archetypes"] / "archetype_intensities.parquet", pd.DataFrame())
                    write_frame(dirs["evaluation"] / "archetype_expert_oof_metrics.parquet", pd.DataFrame())
                    metrics.update({"skipped": True, "skip_reason": "no_archetypes"})
                _record_stage_gate(
                    stage="train_archetype_experts",
                    metrics=metrics,
                    gate_config=gate_config,
                    gate_rows=gate_rows,
                    reporter=reporter,
                    output_dir=args.output_dir,
                    fold_id=fold_id,
                    fold_dirs=dirs,
                )

            with reporter.stage("train_portfolio_calibrator", fold=fold_id) as metrics:
                if experts is not None:
                    archetype_metadata = pd.DataFrame(
                        [
                            {
                                "archetype_id": archetype.archetype_id,
                                "strategy": archetype.strategy,
                                "direction": archetype.direction,
                            }
                            for archetype in calibrator_archetypes
                        ]
                    )
                    min_p_active = float(np.clip(args.archetype_modulation_min_p_active, 0.0, 1.0))
                    if (
                        str(args.portfolio_calibrator_backend) == "optuna"
                        and bool(args.optuna_tune_archetype_thresholds)
                    ):
                        min_p_active = max(
                            min_p_active,
                            float(
                                np.clip(
                                    min(
                                        args.optuna_archetype_min_p_active_low,
                                        args.optuna_archetype_min_p_active_high,
                                    ),
                                    0.0,
                                    1.0,
                                )
                            ),
                        )
                    (
                        optuna_score_threshold_low,
                        optuna_score_threshold_high,
                        optuna_effective_p_active_low,
                        optuna_effective_p_active_high,
                    ) = _optuna_archetype_score_threshold_range_from_p_active(
                        args,
                        min_p_active=min_p_active,
                    )
                    thresholded_scores = threshold_archetype_scores_for_modulation(
                        calibrator_expert_scores.fillna(0.0),
                        min_p_active=min_p_active,
                        min_active_share=float(args.archetype_modulation_min_active_share),
                        relax_floor_to_min_active_share=bool(
                            args.archetype_modulation_relax_floor_to_min_active_share
                        ),
                    )
                    effective_p_active_floor_mean = (
                        float(
                            pd.to_numeric(
                                thresholded_scores.diagnostics.get("effective_min_p_active"),
                                errors="coerce",
                            ).mean()
                        )
                        if "effective_min_p_active" in thresholded_scores.diagnostics
                        else min_p_active
                    )
                    cross = build_cross_strategy_archetype_features(
                        thresholded_scores.p_active,
                        archetype_metadata,
                    )
                    if bool(args.head_streak_risk_features):
                        head_streak_risk = _lagged_head_streak_risk_features(
                            labels,
                            final_expert_X.index,
                            strategies=strategies,
                            full_pressure_hours=float(args.loss_streak_target_full_hours),
                        )
                    else:
                        head_streak_risk = pd.DataFrame(index=final_expert_X.index)
                    if bool(args.calibrator_execution_regime_features):
                        execution_regime = _execution_regime_feature_frame(
                            final_expert_X,
                            cross.X,
                            pattern=str(args.execution_feature_pattern),
                            max_features=int(args.max_execution_regime_features),
                        )
                    else:
                        execution_regime = pd.DataFrame(index=final_expert_X.index)
                    calibrator_X = pd.concat(
                        [
                            thresholded_scores.modulation_scores.fillna(0.0),
                            cross.X.fillna(0.0),
                            head_streak_risk.fillna(0.0),
                            execution_regime.fillna(0.0),
                        ],
                        axis=1,
                    )
                    action_targets = build_portfolio_action_targets_from_labels(
                        labels,
                        final_expert_X.index,
                        strategies=strategies,
                        activation_gate_quality_threshold=float(
                            args.action_target_activation_quality_threshold
                        ),
                        bad_regime_threshold_penalty_scale=float(
                            args.action_target_bad_regime_threshold_penalty
                        ),
                        bad_regime_rank_penalty_scale=float(
                            args.action_target_bad_regime_rank_penalty
                        ),
                        bad_regime_weight_penalty_scale=float(
                            args.action_target_bad_regime_weight_penalty
                        ),
                        bad_regime_activation_penalty_scale=float(
                            args.action_target_bad_regime_activation_penalty
                        ),
                        bad_regime_pressure_column=str(args.action_target_bad_regime_pressure_column),
                    )
                    strategy_returns = _strategy_return_matrix(
                        train_df,
                        timestamp_col=args.timestamp_col,
                        strategy_col=args.strategy_col,
                        performance_col=args.performance_col,
                        timestamps=final_expert_X.index,
                        strategies=strategies,
                    )
                    write_frame(
                        dirs["features"] / "cross_strategy_archetype_features.parquet",
                        cross.X.reset_index(),
                    )
                    write_frame(
                        dirs["features"] / "head_streak_risk_features.parquet",
                        head_streak_risk.reset_index(),
                    )
                    write_frame(
                        dirs["features"] / "execution_regime_features.parquet",
                        execution_regime.reset_index(),
                    )
                    write_frame(
                        dirs["portfolio_calibrator"] / "archetype_modulation_threshold_diagnostics.parquet",
                        thresholded_scores.diagnostics,
                    )
                    write_frame(
                        dirs["portfolio_calibrator"] / "portfolio_action_targets.parquet",
                        _action_targets_to_frame(action_targets),
                    )
                    write_frame(
                        dirs["portfolio_calibrator"] / "portfolio_action_target_diagnostics.parquet",
                        action_targets.diagnostics,
                    )
                    calibrator = train_portfolio_calibrator(
                        calibrator_X,
                        strategies=strategies,
                        action_targets=action_targets.by_strategy,
                        strategy_returns=strategy_returns,
                        config=PortfolioCalibratorConfig(
                            backend=str(args.portfolio_calibrator_backend),
                            allow_cash=True,
                            archetype_score_threshold=0.0,
                            archetype_score_ramp_power=float(args.archetype_score_ramp_power),
                            archetype_score_ramp_gain=float(args.archetype_score_ramp_gain),
                            archetype_base_p_active_floor=effective_p_active_floor_mean,
                            optuna_tune_archetype_score_threshold=bool(
                                args.optuna_tune_archetype_thresholds
                            ),
                            optuna_archetype_score_threshold_range=(
                                optuna_score_threshold_low,
                                optuna_score_threshold_high,
                            ),
                            optuna_tune_archetype_score_ramp=bool(args.optuna_tune_archetype_ramp),
                            optuna_archetype_score_ramp_power_range=(
                                float(args.optuna_archetype_ramp_power_low),
                                float(args.optuna_archetype_ramp_power_high),
                            ),
                            optuna_archetype_score_ramp_gain_range=(
                                float(args.optuna_archetype_ramp_gain_low),
                                float(args.optuna_archetype_ramp_gain_high),
                            ),
                            optuna_archetype_nonzero_penalty=float(
                                args.optuna_archetype_nonzero_penalty
                            ),
                            optuna_trials=int(args.optuna_trials),
                            optuna_objective=str(args.optuna_objective),
                            optuna_mse_weight=float(args.optuna_mse_weight),
                            optuna_ev_weight=float(args.optuna_ev_weight),
                            optuna_hit_rate_weight=float(args.optuna_hit_rate_weight),
                            optuna_loss_streak_weight=float(args.optuna_loss_streak_weight),
                            optuna_loss_streak_hours=float(args.optuna_loss_streak_hours),
                            optuna_downside_weight=float(args.optuna_downside_weight),
                            optuna_turnover_weight=float(args.optuna_turnover_weight),
                            optuna_cash_share_target=float(args.optuna_cash_share_target),
                            optuna_cash_share_weight=float(args.optuna_cash_share_weight),
                            optuna_cash_share_excess_power=float(
                                args.optuna_cash_share_excess_power
                            ),
                            optuna_unjustified_deactivation_weight=float(
                                args.optuna_unjustified_deactivation_weight
                            ),
                            optuna_unjustified_deactivation_gate_margin=float(
                                args.optuna_unjustified_deactivation_gate_margin
                            ),
                            optuna_active_utility_lcb_weight=float(
                                args.optuna_active_utility_lcb_weight
                            ),
                            optuna_active_utility_lcb_z=float(args.optuna_active_utility_lcb_z),
                            optuna_loss_density_weight=float(args.optuna_loss_density_weight),
                            optuna_loss_density_window_hours=float(
                                args.optuna_loss_density_window_hours
                            ),
                            optuna_loss_density_target=float(args.optuna_loss_density_target),
                        ),
                    )
                    write_joblib(dirs["portfolio_calibrator"] / "portfolio_calibrator.joblib", calibrator)
                    write_json(dirs["portfolio_calibrator"] / "portfolio_calibrator.json", calibrator)
                    write_frame(
                        dirs["evaluation"] / "portfolio_oof_metrics.parquet",
                        calibrator.diagnostics,
                    )
                    metrics.update(
                        {
                            "strategy_count": int(len(strategies)),
                            "input_score_count": int(calibrator_X.shape[1]),
                            "head_streak_risk_feature_count": int(head_streak_risk.shape[1]),
                            "execution_regime_feature_count": int(execution_regime.shape[1]),
                            "execution_regime_features_enabled": bool(
                                args.calibrator_execution_regime_features
                            ),
                            "head_streak_risk_features_enabled": bool(
                                args.head_streak_risk_features
                            ),
                            "head_streak_risk_nonzero_share": float(
                                head_streak_risk.ne(0.0).mean().mean()
                            )
                            if head_streak_risk.shape[1]
                            else 0.0,
                            "execution_regime_nonzero_share": float(
                                execution_regime.ne(0.0).mean().mean()
                            )
                            if execution_regime.shape[1]
                            else 0.0,
                            "calibrator_use_predictive_archetypes_only": bool(
                                args.calibrator_use_predictive_archetypes_only
                            ),
                            "calibrator_archetype_count": int(len(calibrator_archetypes)),
                            "diagnostic_rows": int(len(calibrator.diagnostics)),
                            "allow_cash": bool(calibrator.config.allow_cash),
                            "portfolio_calibrator_backend": str(calibrator.config.backend),
                            "archetype_modulation_min_p_active": min_p_active,
                            "archetype_modulation_min_active_share": float(
                                args.archetype_modulation_min_active_share
                            ),
                            "archetype_modulation_relax_floor_to_min_active_share": bool(
                                args.archetype_modulation_relax_floor_to_min_active_share
                            ),
                            "archetype_modulation_score_threshold": 0.0,
                            "optuna_archetype_score_threshold_low": float(optuna_score_threshold_low),
                            "optuna_archetype_score_threshold_high": float(optuna_score_threshold_high),
                            "optuna_effective_p_active_threshold_low": float(
                                optuna_effective_p_active_low
                            ),
                            "optuna_effective_p_active_threshold_high": float(
                                optuna_effective_p_active_high
                            ),
                            "archetype_score_ramp_power": float(args.archetype_score_ramp_power),
                            "archetype_score_ramp_gain": float(args.archetype_score_ramp_gain),
                            "optuna_tune_archetype_ramp": bool(args.optuna_tune_archetype_ramp),
                            "optuna_archetype_nonzero_penalty": float(
                                args.optuna_archetype_nonzero_penalty
                            ),
                            "optuna_trials": int(args.optuna_trials),
                            "optuna_objective": str(args.optuna_objective),
                            "optuna_mse_weight": float(args.optuna_mse_weight),
                            "optuna_ev_weight": float(args.optuna_ev_weight),
                            "optuna_hit_rate_weight": float(args.optuna_hit_rate_weight),
                            "optuna_loss_streak_weight": float(args.optuna_loss_streak_weight),
                            "optuna_loss_streak_hours": float(args.optuna_loss_streak_hours),
                            "optuna_downside_weight": float(args.optuna_downside_weight),
                            "optuna_turnover_weight": float(args.optuna_turnover_weight),
                            "optuna_cash_share_target": float(args.optuna_cash_share_target),
                            "optuna_cash_share_weight": float(args.optuna_cash_share_weight),
                            "optuna_cash_share_excess_power": float(
                                args.optuna_cash_share_excess_power
                            ),
                            "optuna_unjustified_deactivation_weight": float(
                                args.optuna_unjustified_deactivation_weight
                            ),
                            "optuna_unjustified_deactivation_gate_margin": float(
                                args.optuna_unjustified_deactivation_gate_margin
                            ),
                            "optuna_active_utility_lcb_weight": float(
                                args.optuna_active_utility_lcb_weight
                            ),
                            "optuna_active_utility_lcb_z": float(args.optuna_active_utility_lcb_z),
                            "optuna_loss_density_weight": float(args.optuna_loss_density_weight),
                            "optuna_loss_density_window_hours": float(
                                args.optuna_loss_density_window_hours
                            ),
                            "optuna_loss_density_target": float(args.optuna_loss_density_target),
                            "optuna_portfolio_hit_rate_mean": float(
                                pd.to_numeric(
                                    calibrator.diagnostics.get("optuna_portfolio_hit_rate"),
                                    errors="coerce",
                                ).mean()
                            )
                            if "optuna_portfolio_hit_rate" in calibrator.diagnostics
                            else np.nan,
                            "optuna_portfolio_max_loss_streak_hours_mean": float(
                                pd.to_numeric(
                                    calibrator.diagnostics.get(
                                        "optuna_portfolio_max_loss_streak_hours"
                                    ),
                                    errors="coerce",
                                ).mean()
                            )
                            if "optuna_portfolio_max_loss_streak_hours" in calibrator.diagnostics
                            else np.nan,
                            "optuna_portfolio_cash_share_mean": float(
                                pd.to_numeric(
                                    calibrator.diagnostics.get("optuna_portfolio_cash_share"),
                                    errors="coerce",
                                ).mean()
                            )
                            if "optuna_portfolio_cash_share" in calibrator.diagnostics
                            else np.nan,
                            "optuna_portfolio_cash_share_excess_mean": float(
                                pd.to_numeric(
                                    calibrator.diagnostics.get(
                                        "optuna_portfolio_cash_share_excess"
                                    ),
                                    errors="coerce",
                                ).mean()
                            )
                            if "optuna_portfolio_cash_share_excess" in calibrator.diagnostics
                            else np.nan,
                            "optuna_portfolio_cash_share_penalty_mean": float(
                                pd.to_numeric(
                                    calibrator.diagnostics.get(
                                        "optuna_portfolio_cash_share_penalty"
                                    ),
                                    errors="coerce",
                                ).mean()
                            )
                            if "optuna_portfolio_cash_share_penalty" in calibrator.diagnostics
                            else np.nan,
                            "optuna_portfolio_unjustified_deactivation_share_mean": float(
                                pd.to_numeric(
                                    calibrator.diagnostics.get(
                                        "optuna_portfolio_unjustified_deactivation_share"
                                    ),
                                    errors="coerce",
                                ).mean()
                            )
                            if "optuna_portfolio_unjustified_deactivation_share"
                            in calibrator.diagnostics
                            else np.nan,
                            "optuna_portfolio_active_utility_lcb_mean": float(
                                pd.to_numeric(
                                    calibrator.diagnostics.get("optuna_portfolio_active_utility_lcb"),
                                    errors="coerce",
                                ).mean()
                            )
                            if "optuna_portfolio_active_utility_lcb" in calibrator.diagnostics
                            else np.nan,
                            "optuna_portfolio_loss_density_mean": float(
                                pd.to_numeric(
                                    calibrator.diagnostics.get("optuna_portfolio_loss_density_mean"),
                                    errors="coerce",
                                ).mean()
                            )
                            if "optuna_portfolio_loss_density_mean" in calibrator.diagnostics
                            else np.nan,
                            "optuna_portfolio_loss_density_excess_mean": float(
                                pd.to_numeric(
                                    calibrator.diagnostics.get(
                                        "optuna_portfolio_loss_density_excess_mean"
                                    ),
                                    errors="coerce",
                                ).mean()
                            )
                            if "optuna_portfolio_loss_density_excess_mean" in calibrator.diagnostics
                            else np.nan,
                            "archetype_modulation_active_share_mean": float(
                                pd.to_numeric(
                                    thresholded_scores.diagnostics.get("active_share_after_threshold"),
                                    errors="coerce",
                                ).mean()
                            )
                            if "active_share_after_threshold" in thresholded_scores.diagnostics
                            else np.nan,
                            "archetype_modulation_effective_min_p_active_mean": float(
                                pd.to_numeric(
                                    thresholded_scores.diagnostics.get("effective_min_p_active"),
                                    errors="coerce",
                                ).mean()
                            )
                            if "effective_min_p_active" in thresholded_scores.diagnostics
                            else np.nan,
                            "archetype_modulation_score_max": float(
                                pd.to_numeric(
                                    thresholded_scores.diagnostics.get("modulation_score_max"),
                                    errors="coerce",
                                ).max()
                            )
                            if "modulation_score_max" in thresholded_scores.diagnostics
                            else np.nan,
                            "archetype_modulation_suppressed_share_mean": float(
                                pd.to_numeric(
                                    thresholded_scores.diagnostics.get("suppressed_share"),
                                    errors="coerce",
                                ).mean()
                            )
                            if "suppressed_share" in thresholded_scores.diagnostics
                            else np.nan,
                            "archetype_modulation_floor_relaxed_share": float(
                                pd.to_numeric(
                                    thresholded_scores.diagnostics.get(
                                        "floor_relaxed_to_min_active_share"
                                    ),
                                    errors="coerce",
                                ).mean()
                            )
                            if "floor_relaxed_to_min_active_share" in thresholded_scores.diagnostics
                            else np.nan,
                            "optuna_selected_effective_p_active_threshold_mean": float(
                                pd.to_numeric(
                                    calibrator.diagnostics.get(
                                        "archetype_effective_p_active_threshold"
                                    ),
                                    errors="coerce",
                                ).mean()
                            )
                            if "archetype_effective_p_active_threshold" in calibrator.diagnostics
                            else np.nan,
                            "mean_action_prediction_std": float(
                                pd.to_numeric(calibrator.diagnostics.get("prediction_std"), errors="coerce").mean()
                            )
                            if "prediction_std" in calibrator.diagnostics
                            else np.nan,
                            "nonzero_coefficients": int(
                                pd.to_numeric(
                                    calibrator.diagnostics.get("nonzero_coefficients"),
                                    errors="coerce",
                                ).sum()
                            )
                            if "nonzero_coefficients" in calibrator.diagnostics
                            else 0,
                            "activation_target_deactivation_share": float(
                                pd.to_numeric(
                                    action_targets.diagnostics.get("activation_target_deactivation_share"),
                                    errors="coerce",
                                ).mean()
                            )
                            if "activation_target_deactivation_share" in action_targets.diagnostics
                            else np.nan,
                            "action_target_activation_quality_threshold": float(
                                args.action_target_activation_quality_threshold
                            ),
                            "action_target_bad_regime_pressure_column": str(
                                args.action_target_bad_regime_pressure_column
                            ),
                            "action_target_bad_regime_activation_penalty": float(
                                args.action_target_bad_regime_activation_penalty
                            ),
                            "action_target_bad_regime_weight_penalty": float(
                                args.action_target_bad_regime_weight_penalty
                            ),
                        }
                    )
                else:
                    write_frame(dirs["evaluation"] / "portfolio_oof_metrics.parquet", pd.DataFrame())
                    metrics.update({"skipped": True, "skip_reason": "no_archetype_experts"})
                _record_stage_gate(
                    stage="train_portfolio_calibrator",
                    metrics=metrics,
                    gate_config=gate_config,
                    gate_rows=gate_rows,
                    reporter=reporter,
                    output_dir=args.output_dir,
                    fold_id=fold_id,
                    fold_dirs=dirs,
                )

            with reporter.stage("write_fold_summary", fold=fold_id) as metrics:
                fold_summary = {
                    "fold": int(fold_id),
                    "train_rows": int(len(train_df)),
                    "valid_rows": int(len(valid_df)),
                    "train_timestamp_count": int(len(train_pos)),
                    "valid_timestamp_count": int(len(valid_pos)),
                    "extracted_leaf_count": int(len(leaf_table.leaves)),
                    "pruned_leaf_count": int(len(final_pruned)),
                    "archetype_count": int(len(selected_archetypes)),
                    "raw_archetype_count": int(len(archetype_bundle.archetypes)),
                }
                fold_summaries.append(fold_summary)
                write_json(dirs["evaluation"] / "fold_summary.json", fold_summary)
                metrics.update(fold_summary)
            _write_stage_reports(args.output_dir, reporter, fold_id=fold_id, fold_dirs=dirs)

        with reporter.stage("write_manifest", fold_count=len(fold_summaries)) as metrics:
            summary = {
                "folds": fold_summaries,
                "strategies": strategies,
                "feature_families": feature_families,
                "stage_report_path": str(args.output_dir / "performance_market_state_stage_report.parquet"),
                "stage_summary_path": str(args.output_dir / "performance_market_state_stage_summary.parquet"),
                "gate_report_path": str(args.output_dir / "performance_market_state_gate_report.parquet"),
                "stage_gate_profile": str(getattr(args, "stage_gate_profile", "standard")),
                "stage_gates_enabled": bool(gate_config.enabled),
                "stage_gate_fail_fast": bool(gate_config.fail_fast),
            }
            write_json(args.output_dir / "performance_market_state_modulator_manifest.json", summary)
            metrics.update(
                {
                    "manifest_path": args.output_dir / "performance_market_state_modulator_manifest.json",
                    "fold_count": int(len(fold_summaries)),
                }
            )
        return summary
    finally:
        _write_stage_reports(args.output_dir, reporter, fold_id=current_fold_id, fold_dirs=current_fold_dirs)
        _write_gate_reports(args.output_dir, gate_rows, fold_id=current_fold_id, fold_dirs=current_fold_dirs)


def _resolve_run_strategies(args: argparse.Namespace) -> list[str]:
    requested = [s.strip() for s in str(getattr(args, "strategies", "")).split(",") if s.strip()]
    if requested:
        return requested
    frame = _load_frame(args.input)
    return sorted(frame[args.strategy_col].astype(str).dropna().unique().tolist())


def _run_per_head(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    strategies = _resolve_run_strategies(args)
    head_summaries: list[dict[str, Any]] = []
    for strategy in strategies:
        scoped_args = argparse.Namespace(**vars(args))
        scoped_args.pipeline_scope = "global"
        scoped_args.strategies = str(strategy)
        scoped_args.output_dir = args.output_dir / f"head_{_safe_scope_name(strategy)}"
        summary = _run_single_scope(scoped_args)
        head_summary = {
            "head": str(strategy),
            "output_dir": str(scoped_args.output_dir),
            "manifest_path": str(scoped_args.output_dir / "performance_market_state_modulator_manifest.json"),
            "folds": summary.get("folds", []),
        }
        head_summaries.append(head_summary)
    manifest = {
        "pipeline_scope": "per_head",
        "head_count": int(len(head_summaries)),
        "heads": head_summaries,
        "stage_gate_profile": str(getattr(args, "stage_gate_profile", "standard")),
        "stage_gates_enabled": bool(getattr(args, "stage_gates", True)),
        "stage_gate_fail_fast": bool(getattr(args, "stage_gate_fail_fast", True)),
    }
    write_json(args.output_dir / "performance_market_state_modulator_manifest.json", manifest)
    return manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    if str(getattr(args, "pipeline_scope", "per_head")) == "per_head":
        return _run_per_head(args)
    return _run_single_scope(args)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strategy-col", default="strategy")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--performance-col", default="performance")
    parser.add_argument("--strategies", default="")
    parser.add_argument("--pipeline-scope", choices=["per_head", "global"], default="per_head")
    parser.add_argument("--feature-family", action="append", default=[])
    parser.add_argument("--outer-folds", type=int, default=3)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--embargo-hours", type=float, default=96.0)
    parser.add_argument("--inner-purge-timestamps", type=int, default=1)
    parser.add_argument("--min-train-timestamps", type=int, default=20)
    parser.add_argument("--min-inner-train-timestamps", type=int, default=8)
    parser.add_argument("--ewma-halflife", default="3D")
    parser.add_argument("--loss-streak-target-min-hours", type=float, default=72.0)
    parser.add_argument("--loss-streak-target-full-hours", type=float, default=168.0)
    parser.add_argument("--loss-streak-label-weight", type=float, default=1.0)
    parser.add_argument("--loss-streak-sample-weight-multiplier", type=float, default=2.0)
    parser.add_argument(
        "--risk-label-mode",
        action="append",
        default=[],
        help="Opt-in bad-regime label modes: streak,density,drawdown,utility,forward,cooldown.",
    )
    parser.add_argument(
        "--strategy-risk-label-mode",
        action="append",
        default=[],
        help="Per-head override like short_boll:streak,density,drawdown,utility,cooldown.",
    )
    parser.add_argument("--rolling-bad-regime-window-hours", default="24,72,168")
    parser.add_argument("--loss-density-label-weight", type=float, default=0.0)
    parser.add_argument("--loss-density-min-negative-share", type=float, default=0.55)
    parser.add_argument("--loss-density-full-negative-share", type=float, default=0.80)
    parser.add_argument("--drawdown-label-weight", type=float, default=0.0)
    parser.add_argument("--drawdown-anchor-quantile", type=float, default=0.90)
    parser.add_argument("--utility-label-weight", type=float, default=0.0)
    parser.add_argument("--utility-lcb-z-score", type=float, default=1.0)
    parser.add_argument("--forward-bad-label-weight", type=float, default=0.0)
    parser.add_argument("--forward-bad-window-hours", type=float, default=72.0)
    parser.add_argument("--cooldown-label-weight", type=float, default=0.0)
    parser.add_argument("--cooldown-hours", type=float, default=24.0)
    parser.add_argument("--cooldown-trigger", type=float, default=0.75)
    parser.add_argument("--head-streak-risk-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action-target-activation-quality-threshold", type=float, default=0.0)
    parser.add_argument("--action-target-bad-regime-pressure-column", default="composite_bad_pressure")
    parser.add_argument("--action-target-bad-regime-threshold-penalty", type=float, default=0.0)
    parser.add_argument("--action-target-bad-regime-rank-penalty", type=float, default=0.0)
    parser.add_argument("--action-target-bad-regime-weight-penalty", type=float, default=0.0)
    parser.add_argument("--action-target-bad-regime-activation-penalty", type=float, default=0.0)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--first-stage-max-depth", type=int, default=4)
    parser.add_argument("--first-stage-num-leaves", type=int, default=16)
    parser.add_argument("--first-stage-min-child-samples-fraction", type=float, default=0.01)
    parser.add_argument("--first-stage-learning-rate", type=float, default=0.03)
    parser.add_argument("--first-stage-subsample", type=float, default=0.85)
    parser.add_argument("--first-stage-colsample-bytree", type=float, default=0.85)
    parser.add_argument("--first-stage-min-gain-to-split", type=float, default=1e-3)
    parser.add_argument("--first-stage-lambda-l1", type=float, default=0.0)
    parser.add_argument("--first-stage-lambda-l2", type=float, default=1e-3)
    parser.add_argument("--first-stage-early-stopping-rounds", type=int, default=50)
    parser.add_argument("--first-stage-random-state", type=int, default=42)
    parser.add_argument("--min-leaf-stability", type=float, default=0.0)
    parser.add_argument("--max-feedback-passes", type=int, default=1)
    parser.add_argument("--require-oof-improvement-for-second-pass", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--operator-window", type=int, default=24)
    parser.add_argument("--operator-min-periods", type=int, default=6)
    parser.add_argument("--max-archetypes-for-experts", type=int, default=64)
    parser.add_argument(
        "--calibrator-use-predictive-archetypes-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--calibrator-execution-regime-features",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--execution-feature-pattern",
        default="spread|slippage|fee|funding|liquidity|volume|depth|bid|ask|volatility|atr|ood|odd|drift|uncertainty",
    )
    parser.add_argument("--max-execution-regime-features", type=int, default=32)
    parser.add_argument("--portfolio-calibrator-backend", choices=["linear", "ebm_gam", "optuna"], default="linear")
    parser.add_argument("--archetype-modulation-min-p-active", type=float, default=0.55)
    parser.add_argument("--archetype-modulation-min-active-share", type=float, default=0.05)
    parser.add_argument(
        "--archetype-modulation-relax-floor-to-min-active-share",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--archetype-score-ramp-power", type=float, default=1.0)
    parser.add_argument("--archetype-score-ramp-gain", type=float, default=1.0)
    parser.add_argument("--optuna-tune-archetype-thresholds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optuna-archetype-min-p-active-low", type=float, default=0.55)
    parser.add_argument("--optuna-archetype-min-p-active-high", type=float, default=0.90)
    parser.add_argument("--optuna-archetype-modulation-score-low", type=float, default=0.0)
    parser.add_argument("--optuna-archetype-modulation-score-high", type=float, default=0.90)
    parser.add_argument("--optuna-tune-archetype-ramp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optuna-archetype-ramp-power-low", type=float, default=0.5)
    parser.add_argument("--optuna-archetype-ramp-power-high", type=float, default=3.0)
    parser.add_argument("--optuna-archetype-ramp-gain-low", type=float, default=0.25)
    parser.add_argument("--optuna-archetype-ramp-gain-high", type=float, default=4.0)
    parser.add_argument("--optuna-archetype-nonzero-penalty", type=float, default=0.0)
    parser.add_argument("--optuna-trials", type=int, default=20)
    parser.add_argument("--optuna-objective", choices=["mse", "hybrid", "portfolio"], default="hybrid")
    parser.add_argument("--optuna-mse-weight", type=float, default=0.25)
    parser.add_argument("--optuna-ev-weight", type=float, default=1.0)
    parser.add_argument("--optuna-hit-rate-weight", type=float, default=1.0)
    parser.add_argument("--optuna-loss-streak-weight", type=float, default=2.0)
    parser.add_argument("--optuna-loss-streak-hours", type=float, default=72.0)
    parser.add_argument("--optuna-downside-weight", type=float, default=1.0)
    parser.add_argument("--optuna-turnover-weight", type=float, default=0.05)
    parser.add_argument("--optuna-cash-share-target", type=float, default=1.0)
    parser.add_argument("--optuna-cash-share-weight", type=float, default=0.0)
    parser.add_argument("--optuna-cash-share-excess-power", type=float, default=1.0)
    parser.add_argument("--optuna-unjustified-deactivation-weight", type=float, default=0.0)
    parser.add_argument("--optuna-unjustified-deactivation-gate-margin", type=float, default=0.0)
    parser.add_argument("--optuna-active-utility-lcb-weight", type=float, default=0.0)
    parser.add_argument("--optuna-active-utility-lcb-z", type=float, default=1.0)
    parser.add_argument("--optuna-loss-density-weight", type=float, default=0.0)
    parser.add_argument("--optuna-loss-density-window-hours", type=float, default=72.0)
    parser.add_argument("--optuna-loss-density-target", type=float, default=0.50)
    parser.add_argument("--min-compression-silhouette-mean", type=float, default=None)
    parser.add_argument("--max-compression-member-cov", type=float, default=None)
    parser.add_argument("--max-compression-distance-to-seed-p95", type=float, default=None)
    parser.add_argument("--min-compression-source-coverage", type=float, default=None)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--stage-gates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stage-gate-profile", choices=["standard", "lenient", "smoke"], default="standard")
    parser.add_argument("--stage-gate-fail-fast", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
