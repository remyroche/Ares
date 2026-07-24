#!/usr/bin/env python3
"""Train a sparse side x archetype residual-error overlay over the V9 champion.

The model target is deliberately narrower than another meta model.  Inside the
globally defined base/meta top-20 stream, it predicts membership in a V9-specific
persistent adverse side x archetype residual-calendar cell.  Realized row EV is
used to select the overlay strength, not as an inference feature or as a second
row-level state label: market-wide context cannot identify which individual row
will win inside a shared market state.

All model inputs are observable pre-entry context.  Calendar membership and
realized outcomes are labels only.  Model/feature selection and overlay
parameters are selected from chronological OOF predictions ending before the
April-June evaluation window.  The resulting overlay can only demote V9 top-10
rows; it cannot create trades.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

from extreme_price_movements.features_negative_residuals import (
    NEGATIVE_RESIDUAL_META_FEATURE_KEYS,
)
from extreme_price_movements.residual_event_archetypes import (
    OUTCOME_COLUMNS,
    RESIDUAL_EVENT_PREFIX,
    RESIDUAL_EVENT_TEMPORAL_SUFFIXES,
    _binned_mi,
    residual_event_distilled_feature_names,
)
from scripts import run_meta_residual_extreme_local_champion_overlay as champion


KEYS = ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]
TARGET = "bad_residual_event_target"
EVENT = "adverse_calendar_cell"
LEGACY_EVENT = "legacy_adverse_calendar_cell"
SIDE_EVENT = "side_adverse_calendar_cell"
RISK_SCORE = "residual_error_risk_score"
RISK_PCT = "residual_error_risk_percentile"
SIDE_RISK_SCORE = "side_residual_error_risk_score"
SIDE_RISK_PCT = "side_residual_error_risk_percentile"


@dataclass(frozen=True)
class Config:
    train_start: str = "2025-04-01"
    train_end: str = "2026-04-01"
    eval_end: str = "2026-07-01"
    top20_floor: float = 0.80
    top10_floor: float = 0.90
    max_features: int = 24
    targeted_temporal_features: int = 0
    targeted_temporal_only: bool = False
    min_train_rows: int = 4_000
    min_positive_rows: int = 80
    min_feature_coverage: float = 0.60
    min_activity_ratio: float = 0.95
    max_normal_ev_degradation: float = 0.00010
    minimum_event_blocks: int = 3
    seed: int = 20260713


FOLD_STARTS = (
    pd.Timestamp("2025-07-01", tz="UTC"),
    pd.Timestamp("2025-10-01", tz="UTC"),
    pd.Timestamp("2026-01-01", tz="UTC"),
)
RISK_THRESHOLDS = (0.80, 0.825, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975, 0.99)
SOFT_ALPHAS = (0.01, 0.02, 0.03, 0.05, 0.10)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _parse_state_group_filter(value: str | None) -> tuple[str, str] | None:
    if not value:
        return None
    parts = [part.strip() for part in str(value).split("::", 1)]
    if len(parts) != 2 or not all(parts):
        raise ValueError("--state-group-filter must be side::archetype")
    return parts[0], parts[1]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _candidate_features(columns: Iterable[str]) -> list[str]:
    available = set(columns)
    temporal = [
        f"{RESIDUAL_EVENT_PREFIX}{suffix}"
        for suffix in RESIDUAL_EVENT_TEMPORAL_SUFFIXES
    ]
    candidates = [
        "score_meta_base_soft_label",
        "hit_probability",
        *NEGATIVE_RESIDUAL_META_FEATURE_KEYS,
        *residual_event_distilled_feature_names(include_market=True),
        *temporal,
    ]
    forbidden = set(OUTCOME_COLUMNS) | {
        TARGET,
        EVENT,
        SIDE_EVENT,
        "resid_event_class",
        "resid_event_persistent",
        "resid_event_negative_large",
        "resid_event_positive_large",
        "assessment_hr8_surprise",
        "assessment_hr8_effective_n",
    }
    return [
        name
        for name in dict.fromkeys(candidates)
        if name in available
        and name not in forbidden
        and not name.startswith("resid_target_")
    ]


def _load_event_cells(primary: Path, extension: Path | None) -> pd.DataFrame:
    primary_frame = pd.read_csv(primary)
    primary_frame["day"] = pd.to_datetime(primary_frame["day"], utc=True).dt.floor("D")
    primary_frame = primary_frame.loc[:, ["day", "side_name", "archetype_policy_key"]]
    frames = [primary_frame]
    if extension is not None and extension.exists():
        extra = pd.read_csv(extension)
        extra["day"] = pd.to_datetime(extra["day"], utc=True).dt.floor("D")
        extra = extra.loc[
            pd.to_numeric(extra["adverse_event_rows"], errors="coerce")
            .fillna(0.0)
            .gt(0.0),
            ["day", "side_name", "archetype_policy_key"],
        ]
        frames.append(extra)
    return pd.concat(frames, ignore_index=True, copy=False).drop_duplicates()


def _attach_event_target(frame: pd.DataFrame, event_cells: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["day"] = result["__ts__"].dt.floor("D")
    result = result.merge(
        event_cells.assign(**{LEGACY_EVENT: np.int8(1)}),
        on=["day", "side_name", "archetype_policy_key"],
        how="left",
        validate="many_to_one",
    )
    result[LEGACY_EVENT] = result[LEGACY_EVENT].fillna(0).astype(np.int8)
    return result


def _fit_expected_clean_baseline(
    train: pd.DataFrame,
    *,
    top10_floor: float,
) -> pd.DataFrame:
    selected = train.loc[train["parent_rank_v9"].ge(top10_floor)].copy()
    global_rate = float(pd.to_numeric(selected["clean_exec"], errors="coerce").mean())
    side = (
        selected.groupby("side_name", observed=True)["clean_exec"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "side_rate", "count": "side_rows"})
    )
    local = (
        selected.groupby(
            ["side_name", "archetype_policy_key"], observed=True
        )["clean_exec"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "local_rate", "count": "local_rows"})
        .reset_index()
        .merge(side.reset_index(), on="side_name", how="left", validate="many_to_one")
    )
    side_weight = local["side_rows"] / (local["side_rows"] + 500.0)
    local["side_expected_clean_rate"] = (
        side_weight * local["side_rate"] + (1.0 - side_weight) * global_rate
    )
    local_weight = local["local_rows"] / (local["local_rows"] + 300.0)
    local["expected_clean_rate"] = (
        local_weight * local["local_rate"]
        + (1.0 - local_weight) * local["side_expected_clean_rate"]
    )
    local["global_expected_clean_rate"] = global_rate
    return local


def _v9_residual_calendar(
    frame: pd.DataFrame,
    *,
    top10_floor: float,
    expected_clean_baseline: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a V9-specific adverse-state target from selected-row outcomes.

    The label is outcome-derived and never exposed as an inference feature.  A
    day is adverse when V9 clean-hit surprise and EV are both negative and the
    damage is either extreme, persists for two adjacent days, or remains weak
    over a causal three-day mean.  The adjacent-day rule is a two-day outcome
    label and is purged at model fold boundaries below.
    """

    result = frame.copy()
    result["day"] = result["__ts__"].dt.floor("D")
    selected = result.loc[result["parent_rank_v9"].ge(top10_floor)].copy()
    selected = selected.merge(
        expected_clean_baseline.loc[
            :, ["side_name", "archetype_policy_key", "expected_clean_rate"]
        ],
        on=["side_name", "archetype_policy_key"],
        how="left",
        validate="many_to_one",
    )
    fallback = float(expected_clean_baseline["global_expected_clean_rate"].iloc[0])
    selected["expected_clean_rate"] = selected["expected_clean_rate"].fillna(fallback)
    selected["row_signed_surprise"] = pd.to_numeric(
        selected["clean_exec"], errors="coerce"
    ) - pd.to_numeric(selected["expected_clean_rate"], errors="coerce")
    daily = (
        selected.groupby(
            ["day", "side_name", "archetype_policy_key"],
            observed=True,
            sort=True,
        )
        .agg(
            selected_rows=("clean_exec", "size"),
            clean_exec_rate=("clean_exec", "mean"),
            expected_clean_rate=("expected_clean_rate", "mean"),
            signed_surprise=("row_signed_surprise", "mean"),
            mean_ev_after_1pct=("ev_after_1pct", "mean"),
        )
        .reset_index()
    )
    outputs: list[pd.DataFrame] = []
    for _, local in daily.groupby(
        ["side_name", "archetype_policy_key"], observed=True, sort=True
    ):
        local = local.sort_values("day", kind="stable").copy()
        surprise = pd.to_numeric(local["signed_surprise"], errors="coerce")
        ev = pd.to_numeric(local["mean_ev_after_1pct"], errors="coerce")
        enough = local["selected_rows"].ge(5)
        adverse = enough & surprise.le(-0.10) & ev.lt(0.0)
        adjacent = adverse & (adverse.shift(1, fill_value=False) | adverse.shift(-1, fill_value=False))
        extreme = enough & surprise.le(-0.20) & ev.lt(0.0)
        long_weak = (
            enough
            & surprise.rolling(3, min_periods=2).mean().le(-0.08)
            & ev.rolling(3, min_periods=2).mean().lt(0.0)
        )
        local["adverse_base"] = adverse.astype(np.int8)
        local["adverse_adjacent_2d"] = adjacent.astype(np.int8)
        local["adverse_extreme"] = extreme.astype(np.int8)
        local["adverse_longer"] = long_weak.astype(np.int8)
        local[EVENT] = (adjacent | extreme | long_weak).astype(np.int8)
        outputs.append(local)
    calendar = pd.concat(outputs, ignore_index=True, copy=False)
    result = result.merge(
        calendar.loc[:, ["day", "side_name", "archetype_policy_key", EVENT]],
        on=["day", "side_name", "archetype_policy_key"],
        how="left",
        validate="many_to_one",
    )
    result[EVENT] = result[EVENT].fillna(0).astype(np.int8)
    result[TARGET] = result[EVENT].astype(np.int8)
    return result, calendar


def _merge_temporal(frame: pd.DataFrame, path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return frame
    names = set(pq.read_schema(path).names)
    temporal = [
        f"{RESIDUAL_EVENT_PREFIX}{suffix}"
        for suffix in RESIDUAL_EVENT_TEMPORAL_SUFFIXES
        if f"{RESIDUAL_EVENT_PREFIX}{suffix}" in names
    ]
    if not temporal:
        return frame
    context = pd.read_parquet(path, columns=KEYS + temporal)
    context["__ts__"] = pd.to_datetime(context["__ts__"], utc=True)
    context = context.drop_duplicates(KEYS, keep="last")
    return frame.merge(context, on=KEYS, how="left", validate="one_to_one")


def _load_frames(args: argparse.Namespace, config: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_start = pd.Timestamp(config.train_start, tz="UTC")
    train_end = pd.Timestamp(config.train_end, tz="UTC")
    eval_end = pd.Timestamp(config.eval_end, tz="UTC")
    train, valid, coverage = champion._load_joined(
        champion_path=args.champion_ledger,
        parent_eval_path=args.parent_eval_predictions,
        state_path=[args.state_artifact, *getattr(args, "additional_state_artifact", [])],
        train_oof_predictions_dir=args.train_oof_predictions_dir,
        train_oof_rank_cache=args.train_oof_rank_cache,
        train_start=train_start,
        train_end=train_end,
        eval_end=eval_end,
        negative_residual_features=args.negative_residual_features,
        state_group_filter=_parse_state_group_filter(
            getattr(args, "state_group_filter", "")
        ),
    )
    train = _merge_temporal(train, args.temporal_state_features)
    valid = _merge_temporal(valid, args.temporal_state_features)
    if "hit_probability" not in train.columns:
        state_names = set(pq.read_schema(args.state_artifact).names)
        if "hit_probability" not in state_names:
            raise KeyError("State artifact has no hit_probability for V9 residual labels")
        probability = pd.read_parquet(
            args.state_artifact,
            columns=KEYS + ["hit_probability"],
        )
        probability["__ts__"] = pd.to_datetime(probability["__ts__"], utc=True)
        probability = probability.drop_duplicates(KEYS, keep="last")
        train = train.merge(probability, on=KEYS, how="left", validate="one_to_one")

    if bool(getattr(args, "direct_parent_rank", False)):
        # Research-only new parent contract. The rank is already causal and
        # frozen in the supplied source; do not apply V9's separately fitted
        # local adjustment to it.
        v9_train_rank = pd.to_numeric(
            train["historical_rank"], errors="coerce"
        ).to_numpy(dtype=np.float32)
        train["parent_rank_v9"] = v9_train_rank
        valid["parent_rank_v9"] = pd.to_numeric(
            valid["historical_rank"], errors="coerce"
        ).astype(np.float32)
        coverage["train_rank_contract"] = "direct_causal_candidate_trailing_window"
        coverage["parent_rank_contract"] = "direct_causal_candidate_parent"
    else:
        manifest = json.loads(args.v9_manifest.read_text())
        catalog = pd.read_csv(args.v9_selected_features)
        v9_train_rank, _, _ = champion._rank_for_params(
            train,
            train,
            catalog,
            manifest["strict_best"],
        )
        train["parent_rank_v9"] = v9_train_rank
        if "historical_rank_strict_extreme_local" not in valid.columns:
            v9_eval = pd.read_parquet(
                args.v9_predictions,
                columns=KEYS + ["historical_rank_strict_extreme_local"],
            )
            v9_eval["__ts__"] = pd.to_datetime(v9_eval["__ts__"], utc=True)
            valid = valid.merge(v9_eval, on=KEYS, how="inner", validate="one_to_one")
        valid["parent_rank_v9"] = pd.to_numeric(
            valid["historical_rank_strict_extreme_local"], errors="coerce"
        ).astype(np.float32)
        coverage["parent_rank_contract"] = "v9_strict_extreme_local"
    coverage["v9_train_selected_rows"] = int((v9_train_rank >= config.top10_floor).sum())
    return train, valid, coverage


def _matrix(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train = train[features].to_numpy(dtype=np.float32, copy=True)
    x_valid = valid[features].to_numpy(dtype=np.float32, copy=True)
    x_train[~np.isfinite(x_train)] = np.nan
    x_valid[~np.isfinite(x_valid)] = np.nan
    medians = np.nanmedian(x_train, axis=0).astype(np.float32)
    medians = np.nan_to_num(medians, nan=0.0)
    for matrix in (x_train, x_valid):
        missing = ~np.isfinite(matrix)
        if missing.any():
            matrix[missing] = np.take(medians, np.nonzero(missing)[1])
    return x_train, x_valid, medians


def _timestamp_training_frame(
    frame: pd.DataFrame,
    features: list[str],
    *,
    target_column: str,
    event_column: str,
) -> pd.DataFrame:
    """Collapse repeated asset rows to the observable timestamp state."""

    numeric = list(dict.fromkeys(features))
    values = frame.loc[:, ["__ts__", *numeric]].copy()
    for feature in numeric:
        values[feature] = pd.to_numeric(values[feature], errors="coerce").astype(
            np.float32
        )
    states = values.groupby("__ts__", observed=True, sort=True)[numeric].median()
    grouped = frame.groupby("__ts__", observed=True, sort=True)
    labels = grouped.agg(
        day=("day", "first"),
        ev_after_1pct=("ev_after_1pct", "mean"),
        clean_exec=("clean_exec", "mean"),
    )
    labels[target_column] = grouped[target_column].max().astype(np.int8)
    if event_column != target_column:
        labels[event_column] = grouped[event_column].max().astype(np.int8)
    return states.join(labels, how="inner").reset_index()


def _fit_timestamp_model(
    train_rows: pd.DataFrame,
    valid_rows: pd.DataFrame,
    features: list[str],
    seed: int,
    *,
    target_column: str,
    event_column: str,
) -> tuple[
    lgb.Booster,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
]:
    train_state = _timestamp_training_frame(
        train_rows,
        features,
        target_column=target_column,
        event_column=event_column,
    )
    valid_state = _timestamp_training_frame(
        valid_rows,
        features,
        target_column=target_column,
        event_column=event_column,
    )
    model, train_score, state_valid_score, medians = _fit_model(
        train_state,
        valid_state,
        features,
        seed,
        target_column=target_column,
        event_column=event_column,
    )
    score_by_timestamp = pd.Series(
        state_valid_score, index=valid_state["__ts__"]
    )
    row_valid_score = valid_rows["__ts__"].map(score_by_timestamp).to_numpy(
        np.float32
    )
    reference = np.sort(train_score[np.isfinite(train_score)])
    return (
        model,
        row_valid_score,
        state_valid_score,
        medians,
        reference,
        train_state,
        valid_state,
    )


def _targeted_temporal_features(side: str | None, archetype: str | None) -> set[str]:
    """Return the narrow observable mechanism family relevant to one local model."""

    key = (str(side or ""), str(archetype or ""))
    compression = {
        "compression_quality_consistency",
        "compression_confirmation_ratio",
        "healthy_compression_score",
        "exhausted_compression_score",
        "fragile_compression_score",
        "compression_duration_72h_norm",
        "compression_integral_72h",
        "compression_onset_shock_24h",
    }
    persistence = {
        "short_default_damage_pressure",
        "short_default_damage_ema_5d",
        "short_default_damage_integral_5d",
        "short_default_damage_max_5d",
        "short_default_adverse_duration_5d_norm",
        "market_state_transition_entropy_5d",
        "market_state_persistence_5d",
        "recovery_failure_score_24h",
    }
    breakout = {
        "breakout_efficiency_4h",
        "breakout_participation_4h",
        "breakout_retention_4h",
        "breakout_confirmation_ratio",
        "breakout_disagreement_score",
        "breakout_bilateral_failure_score",
    }
    if key == ("long", "long_volcompression_wideslow_candidate"):
        return compression
    if key == ("short", "short_default_clean_path"):
        return persistence
    if key[1] in {
        "short_breakout_precision",
        "long_breakout_diagnostic_candidate",
    }:
        return breakout
    return set()


def _screen_features(
    train: pd.DataFrame,
    candidates: list[str],
    config: Config,
    target_column: str = TARGET,
    side: str | None = None,
    archetype: str | None = None,
    min_finite_rows: int = 500,
) -> tuple[list[str], pd.DataFrame]:
    y = train[target_column].to_numpy(np.int8)
    rows: list[dict[str, Any]] = []
    for feature in candidates:
        values = pd.to_numeric(train[feature], errors="coerce").to_numpy(np.float32)
        finite = np.isfinite(values)
        if (
            float(finite.mean()) < config.min_feature_coverage
            or int(finite.sum()) < int(min_finite_rows)
        ):
            continue
        unique = np.unique(values[finite])
        if len(unique) < 4:
            continue
        prevalence = float(y[finite].mean())
        tails: list[tuple[float, float, int]] = []
        for direction in (-1.0, 1.0):
            directed = direction * values[finite]
            cutoff = float(np.quantile(directed, 0.90))
            tail = directed >= cutoff
            tail_rows = int(tail.sum())
            rate = float(y[finite][tail].mean()) if tail_rows else 0.0
            tails.append((rate / max(prevalence, 1e-8), direction, tail_rows))
        lift, direction, tail_rows = max(tails, key=lambda item: item[0])
        mi = _binned_mi(values, y, 10)
        score = float(mi * max(1.0, math.log1p(max(lift - 1.0, 0.0))))
        rows.append(
            {
                "feature": feature,
                "binned_mi": mi,
                "tail_lift": lift,
                "tail_direction": direction,
                "tail_rows": tail_rows,
                "finite_rate": float(finite.mean()),
                "screen_score": score,
            }
        )
    if not rows:
        return [], pd.DataFrame(
            columns=[
                "feature", "binned_mi", "tail_lift", "tail_direction",
                "tail_rows", "finite_rate", "screen_score", "selected",
            ]
        )
    report = pd.DataFrame(rows).sort_values(
        ["screen_score", "tail_lift", "binned_mi", "feature"],
        ascending=[False, False, False, True],
        kind="stable",
    )
    eligible = report.loc[
        report["binned_mi"].gt(0.0) & report["tail_lift"].gt(1.0), "feature"
    ].tolist()
    target_family = _targeted_temporal_features(side, archetype)
    if config.targeted_temporal_only and target_family:
        candidates = [feature for feature in candidates if feature in target_family]
        report = report.loc[report["feature"].isin(target_family)].copy()
        eligible = [feature for feature in eligible if feature in target_family]
    requested_targeted_budget = (
        int(config.max_features)
        if config.targeted_temporal_only and target_family
        else max(int(config.targeted_temporal_features), 0)
    )
    targeted_budget = min(requested_targeted_budget, len(target_family))
    selected_targeted = [
        feature for feature in eligible if feature in target_family
    ][:targeted_budget]
    general_budget = max(int(config.max_features) - len(selected_targeted), 0)
    selected_general = [
        feature for feature in eligible if feature not in target_family
    ][:general_budget]
    selected = selected_general + selected_targeted
    report["targeted_temporal_family"] = report["feature"].isin(target_family)
    report["selected"] = report["feature"].isin(selected)
    return selected, report


def _sample_weights(
    frame: pd.DataFrame,
    target_column: str = TARGET,
    event_column: str = EVENT,
) -> np.ndarray:
    y = frame[target_column].to_numpy(np.int8)
    ev = pd.to_numeric(frame["ev_after_1pct"], errors="coerce").to_numpy(np.float32)
    negative = max(int((y <= 0).sum()), 1)
    weights = np.full(len(frame), 0.5 / negative, dtype=np.float32)
    blocks = _event_block_ids(frame, event_column=event_column)
    positive_blocks = np.unique(blocks[(y > 0) & (blocks >= 0)])
    severity = np.clip(np.nan_to_num(-ev, nan=0.0) / 0.02, 0.0, 3.0)
    if len(positive_blocks):
        block_mass = 0.5 / len(positive_blocks)
        for block in positive_blocks:
            mask = (y > 0) & (blocks == block)
            local = (1.0 + severity[mask]).astype(np.float32)
            weights[mask] = np.float32(block_mass) * local / max(float(local.sum()), 1e-8)
    else:
        positive = max(int((y > 0).sum()), 1)
        weights[y > 0] = np.float32(0.5 / positive)
    weights *= np.float32(len(weights))
    return (weights / max(float(weights.mean()), 1e-8)).astype(np.float32)


def _fit_model(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    seed: int,
    target_column: str = TARGET,
    event_column: str = EVENT,
) -> tuple[lgb.Booster, np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_valid, medians = _matrix(train, valid, features)
    model = lgb.train(
        {
            "objective": "binary",
            "metric": "None",
            "learning_rate": 0.035,
            "max_depth": 3,
            "num_leaves": 7,
            "min_data_in_leaf": 80,
            "min_gain_to_split": 0.02,
            "lambda_l1": 1.0,
            "lambda_l2": 8.0,
            "feature_fraction": 0.80,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "num_threads": -1,
            "verbosity": -1,
            "force_col_wise": True,
        },
        lgb.Dataset(
            x_train,
            label=train[target_column].to_numpy(np.float32),
            weight=_sample_weights(train, target_column, event_column),
            feature_name=features,
            free_raw_data=True,
        ),
        num_boost_round=160,
    )
    train_score = np.asarray(model.predict(x_train), dtype=np.float32)
    valid_score = np.asarray(model.predict(x_valid), dtype=np.float32)
    return model, train_score, valid_score, medians


def _midrank(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    result = np.full(len(values), 0.5, dtype=np.float32)
    finite = np.isfinite(values)
    left = np.searchsorted(reference, values[finite], side="left")
    right = np.searchsorted(reference, values[finite], side="right")
    result[finite] = (left + right) / (2.0 * max(len(reference), 1))
    return result


def _timestamp_state_score(frame: pd.DataFrame, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse local row risk to a causal cross-sectional timestamp state.

    The 90th percentile preserves a strong broad-state warning without letting
    one symbol dominate the whole timestamp.  Both returned arrays are ordered
    to match ``frame``; the second contains one value per unique timestamp for
    fitting the frozen percentile reference without symbol-count weighting.
    """

    payload = pd.DataFrame(
        {
            "__ts__": frame["__ts__"].to_numpy(),
            "score": np.asarray(score, dtype=np.float32),
        }
    )
    by_timestamp = payload.groupby("__ts__", observed=True, sort=True)["score"].quantile(0.90)
    mapped = payload["__ts__"].map(by_timestamp).to_numpy(np.float32)
    return mapped, by_timestamp.to_numpy(np.float32)


def _event_block_ids(
    frame: pd.DataFrame, *, event_column: str = EVENT
) -> np.ndarray:
    days = frame["day"].to_numpy(dtype="datetime64[D]")
    event = frame[event_column].to_numpy(bool)
    result = np.full(len(frame), -1, dtype=np.int32)
    event_days = np.unique(days[event])
    block = -1
    previous: np.datetime64 | None = None
    mapping: dict[np.datetime64, int] = {}
    for day in event_days:
        if previous is None or int((day - previous).astype(int)) > 1:
            block += 1
        mapping[day] = block
        previous = day
    for day, block_id in mapping.items():
        result[(days == day) & event] = block_id
    return result


def _apply_rank(
    rank: np.ndarray,
    risk_pct: np.ndarray,
    threshold: float,
    alpha: float,
    hard_block: bool,
    top10_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    adjusted = rank.astype(np.float32, copy=True)
    parent_selected = adjusted >= top10_floor
    flagged = parent_selected & (risk_pct >= threshold)
    if hard_block:
        adjusted[flagged] = np.nextafter(np.float32(top10_floor), np.float32(0.0))
    else:
        intensity = np.clip(
            (risk_pct - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0
        )
        adjusted[parent_selected] -= np.float32(alpha) * intensity[parent_selected]
    return np.clip(adjusted, 0.0, 1.0), flagged


def _selection_metric_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    month_codes, _ = pd.factorize(
        frame["__ts__"].dt.strftime("%Y-%m"), sort=True
    )
    return {
        "ev": pd.to_numeric(frame["ev_after_1pct"], errors="coerce").to_numpy(
            np.float32
        ),
        "clean": pd.to_numeric(frame["clean_exec"], errors="coerce").to_numpy(
            np.float32
        ),
        "event": frame[EVENT].to_numpy(bool),
        "month_codes": month_codes.astype(np.int16, copy=False),
    }


def _selection_metrics(
    frame: pd.DataFrame,
    rank: np.ndarray,
    top10_floor: float,
    arrays: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    selected = rank >= top10_floor
    values = arrays if arrays is not None else _selection_metric_arrays(frame)
    ev = values["ev"]
    clean = values["clean"]
    event = values["event"]
    month_codes = values["month_codes"]
    finite_month = selected & np.isfinite(ev) & (month_codes >= 0)
    if finite_month.any():
        month_count = int(month_codes.max()) + 1
        sums = np.bincount(
            month_codes[finite_month], weights=ev[finite_month], minlength=month_count
        )
        counts = np.bincount(month_codes[finite_month], minlength=month_count)
        monthly = sums[counts > 0] / counts[counts > 0]
    else:
        monthly = np.empty(0, dtype=np.float64)
    return {
        "selected_rows": int(selected.sum()),
        "mean_ev": float(np.nanmean(ev[selected])) if selected.any() else np.nan,
        "positive_ev_rate": float(np.nanmean(ev[selected] > 0.0)) if selected.any() else np.nan,
        "clean_precision": float(np.nanmean(clean[selected])) if selected.any() else np.nan,
        "event_mean_ev": float(np.nanmean(ev[selected & event])) if (selected & event).any() else np.nan,
        "normal_mean_ev": float(np.nanmean(ev[selected & ~event])) if (selected & ~event).any() else np.nan,
        "mean_month_ev": float(np.mean(monthly)) if len(monthly) else np.nan,
        "std_month_ev": float(np.std(monthly)) if len(monthly) else np.nan,
        "worst_month_ev": float(np.min(monthly)) if len(monthly) else np.nan,
    }


def _risk_metrics(
    frame: pd.DataFrame,
    risk_pct: np.ndarray,
    threshold: float,
    *,
    target: np.ndarray | None = None,
    blocks: np.ndarray | None = None,
) -> dict[str, float]:
    target = frame[TARGET].to_numpy(bool) if target is None else target
    flagged = risk_pct >= threshold
    prevalence = float(target.mean())
    precision = float(target[flagged].mean()) if flagged.any() else np.nan
    fpr = float(flagged[~target].mean()) if (~target).any() else np.nan
    blocks = _event_block_ids(frame) if blocks is None else blocks
    block_ids = np.unique(blocks[blocks >= 0])
    hit_blocks = [
        bool((flagged & target & (blocks == block_id)).any()) for block_id in block_ids
    ]
    return {
        "target_prevalence": prevalence,
        "risk_precision": precision,
        "risk_lift": precision / max(prevalence, 1e-9) if np.isfinite(precision) else np.nan,
        "risk_fpr": fpr,
        "event_blocks": int(len(block_ids)),
        "recognized_event_blocks": int(sum(hit_blocks)),
        "event_block_recall": float(np.mean(hit_blocks)) if hit_blocks else np.nan,
    }


def _search_local_overlay(
    frame: pd.DataFrame,
    config: Config,
    *,
    risk_column: str = RISK_PCT,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    parent = frame["parent_rank_v9"].to_numpy(np.float32)
    risk_pct = frame[risk_column].to_numpy(np.float32)
    metric_arrays = _selection_metric_arrays(frame)
    target = frame[TARGET].to_numpy(bool)
    blocks = _event_block_ids(frame)
    parent_metrics = _selection_metrics(
        frame, parent, config.top10_floor, metric_arrays
    )
    base_count = max(int(parent_metrics["selected_rows"]), 1)
    rows: list[dict[str, Any]] = []
    for threshold in RISK_THRESHOLDS:
        risk_metrics = _risk_metrics(
            frame, risk_pct, threshold, target=target, blocks=blocks
        )
        for hard_block, alphas in ((False, SOFT_ALPHAS), (True, (0.0,))):
            for alpha in alphas:
                adjusted, flagged = _apply_rank(
                    parent,
                    risk_pct,
                    threshold,
                    alpha,
                    hard_block,
                    config.top10_floor,
                )
                metrics = _selection_metrics(
                    frame, adjusted, config.top10_floor, metric_arrays
                )
                activity = float(metrics["selected_rows"]) / base_count
                event_delta = metrics["event_mean_ev"] - parent_metrics["event_mean_ev"]
                normal_delta = metrics["normal_mean_ev"] - parent_metrics["normal_mean_ev"]
                precision_delta = metrics["positive_ev_rate"] - parent_metrics["positive_ev_rate"]
                overall_delta = metrics["mean_ev"] - parent_metrics["mean_ev"]
                objective = (
                    metrics["mean_month_ev"]
                    - 0.5 * metrics["std_month_ev"]
                    + 0.25 * metrics["worst_month_ev"]
                    + 0.25 * max(event_delta, -0.02)
                    - 0.01 * abs(math.log(max(activity, 1e-8)))
                )
                promotable = (
                    risk_metrics["risk_lift"] >= 1.5
                    and risk_metrics["risk_fpr"] <= 0.15
                    and risk_metrics["recognized_event_blocks"] >= config.minimum_event_blocks
                    and precision_delta > 0.0
                    and overall_delta > 0.0
                    and event_delta > 0.0
                    and normal_delta >= -config.max_normal_ev_degradation
                    and activity >= config.min_activity_ratio
                )
                rows.append(
                    {
                        "risk_variant": risk_column,
                        "threshold": threshold,
                        "mode": "hard_block" if hard_block else "soft_nudge",
                        "alpha": alpha,
                        "flagged_parent_rows": int(flagged.sum()),
                        "activity_ratio": activity,
                        "overall_ev_delta": overall_delta,
                        "event_ev_delta": event_delta,
                        "normal_ev_delta": normal_delta,
                        "positive_ev_rate_delta": precision_delta,
                        "objective": objective,
                        "promotable": promotable,
                        **risk_metrics,
                        **{f"adjusted_{key}": value for key, value in metrics.items()},
                    }
                )
    search = pd.DataFrame(rows).sort_values(
        ["promotable", "objective", "activity_ratio"],
        ascending=[False, False, False],
        kind="stable",
    )
    accepted = search.loc[search["promotable"]]
    return search, (accepted.iloc[0].to_dict() if not accepted.empty else None)


def _add_risk_variants(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    local = pd.to_numeric(result[RISK_PCT], errors="coerce").fillna(0.5).to_numpy(np.float32)
    side = pd.to_numeric(result[SIDE_RISK_PCT], errors="coerce").fillna(0.5).to_numpy(np.float32)
    result["residual_risk_local"] = local
    result["residual_risk_side"] = side
    for weight in (0.25, 0.50, 0.75):
        result[f"residual_risk_blend_side_{int(weight * 100):02d}"] = (
            (1.0 - weight) * local + weight * side
        ).astype(np.float32)
    result["residual_risk_max_local_side"] = np.maximum(local, side).astype(np.float32)
    result["residual_risk_geometric_local_side"] = np.sqrt(
        np.clip(local, 0.0, 1.0) * np.clip(side, 0.0, 1.0)
    ).astype(np.float32)
    return result


RISK_VARIANTS = (
    "residual_risk_local",
    "residual_risk_side",
    "residual_risk_blend_side_25",
    "residual_risk_blend_side_50",
    "residual_risk_blend_side_75",
    "residual_risk_max_local_side",
    "residual_risk_geometric_local_side",
)


def _fit_oof_and_final(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    candidates: list[str],
    config: Config,
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[str, str], dict[str, Any]]]:
    oof_parts: list[pd.DataFrame] = []
    feature_reports: list[pd.DataFrame] = []
    model_rows: list[dict[str, Any]] = []
    final_states: dict[tuple[str, str], dict[str, Any]] = {}
    groups = train.groupby(["side_name", "archetype_policy_key"], observed=True, sort=True)
    for group_index, ((side, archetype), local) in enumerate(groups):
        local = local.loc[local["parent_rank_v9"].ge(config.top20_floor)].sort_values("__ts__", kind="stable")
        if len(local) < config.min_train_rows:
            continue
        for fold_index, fold_start in enumerate(FOLD_STARTS):
            fold_end = (
                FOLD_STARTS[fold_index + 1]
                if fold_index + 1 < len(FOLD_STARTS)
                else pd.Timestamp(config.train_end, tz="UTC")
            )
            # V9 adverse-state labels may use the immediately following day to
            # qualify a two-day episode, so purge two full days at boundaries.
            fit = local.loc[local["__ts__"].lt(fold_start - pd.Timedelta(days=2))]
            score = local.loc[local["__ts__"].ge(fold_start) & local["__ts__"].lt(fold_end)]
            if (
                len(fit) < config.min_train_rows
                or int(fit[TARGET].sum()) < config.min_positive_rows
                or score.empty
            ):
                continue
            fit_screen = _timestamp_training_frame(
                fit,
                candidates,
                target_column=TARGET,
                event_column=EVENT,
            )
            selected, report = _screen_features(
                fit_screen,
                candidates,
                config,
                side=str(side),
                archetype=str(archetype),
            )
            if not selected:
                continue
            (
                model,
                valid_score,
                state_valid_score,
                _,
                reference,
                fit_state,
                score_state,
            ) = _fit_timestamp_model(
                fit,
                score,
                selected,
                config.seed + 100 * group_index + fold_index,
                target_column=TARGET,
                event_column=EVENT,
            )
            part = score.loc[:, KEYS + [
                "day", "parent_rank_v9", "ev_after_1pct", "clean_exec", EVENT, TARGET
            ]].copy()
            part[RISK_SCORE] = valid_score
            part[RISK_PCT] = _midrank(valid_score, reference)
            part["fold_start"] = fold_start
            part["fold_end"] = fold_end
            oof_parts.append(part)
            report.insert(0, "fold_start", fold_start)
            report.insert(0, "archetype_policy_key", archetype)
            report.insert(0, "side_name", side)
            feature_reports.append(report)
            model_rows.append(
                {
                    "stage": "oof",
                    "side_name": side,
                    "archetype_policy_key": archetype,
                    "fold_start": fold_start,
                    "train_rows": len(fit),
                    "train_target_rows": int(fit[TARGET].sum()),
                    "score_rows": len(score),
                    "features": len(selected),
                    "selected_features": "|".join(selected),
                    "average_precision": float(
                        average_precision_score(score_state[TARGET], state_valid_score)
                    )
                    if score_state[TARGET].nunique() > 1
                    else np.nan,
                    "train_state_rows": len(fit_state),
                    "score_state_rows": len(score_state),
                }
            )

        local_screen = _timestamp_training_frame(
            local,
            candidates,
            target_column=TARGET,
            event_column=EVENT,
        )
        selected, report = _screen_features(
            local_screen,
            candidates,
            config,
            side=str(side),
            archetype=str(archetype),
        )
        if not selected or int(local[TARGET].sum()) < config.min_positive_rows:
            continue
        local_valid = valid.loc[
            valid["side_name"].astype(str).eq(str(side))
            & valid["archetype_policy_key"].astype(str).eq(str(archetype))
            & valid["parent_rank_v9"].ge(config.top20_floor)
        ].sort_values("__ts__", kind="stable")
        if local_valid.empty:
            continue
        (
            model,
            valid_score,
            state_valid_score,
            medians,
            reference,
            train_state,
            valid_state,
        ) = _fit_timestamp_model(
            local,
            local_valid,
            selected,
            config.seed + 10_000 + group_index,
            target_column=TARGET,
            event_column=EVENT,
        )
        final_states[(str(side), str(archetype))] = {
            "model": model,
            "features": selected,
            "medians": medians,
            "reference": reference,
            "valid_index": local_valid.index.to_numpy(),
            "valid_score": valid_score,
        }
        report.insert(0, "fold_start", "final")
        report.insert(0, "archetype_policy_key", archetype)
        report.insert(0, "side_name", side)
        feature_reports.append(report)
        model_rows.append(
            {
                "stage": "final",
                "side_name": side,
                "archetype_policy_key": archetype,
                "fold_start": config.train_end,
                "train_rows": len(local),
                "train_target_rows": int(local[TARGET].sum()),
                "score_rows": len(local_valid),
                "features": len(selected),
                "selected_features": "|".join(selected),
                "average_precision": float(
                    average_precision_score(valid_state[TARGET], state_valid_score)
                )
                if valid_state[TARGET].nunique() > 1
                else np.nan,
                "train_state_rows": len(train_state),
                "score_state_rows": len(valid_state),
            }
        )
        model.save_model(str(output / f"model__{side}__{archetype}.txt"))
        np.savez_compressed(
            output / f"state__{side}__{archetype}.npz",
            features=np.asarray(selected, dtype=np.str_),
            medians=np.asarray(medians, dtype=np.float32),
            reference=np.asarray(reference, dtype=np.float32),
        )
    if not oof_parts:
        raise RuntimeError("No chronological OOF residual-error predictions were generated")
    return (
        pd.concat(oof_parts, ignore_index=True, copy=False),
        pd.concat(feature_reports, ignore_index=True, copy=False),
        pd.DataFrame(model_rows),
        final_states,
    )


def _fit_side_oof_and_final(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    candidates: list[str],
    config: Config,
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    """Fit support-sharing side parents without changing local admission units."""

    oof_parts: list[pd.DataFrame] = []
    feature_reports: list[pd.DataFrame] = []
    model_rows: list[dict[str, Any]] = []
    final_states: dict[str, dict[str, Any]] = {}
    for side_index, (side, side_rows) in enumerate(
        train.groupby("side_name", observed=True, sort=True)
    ):
        local = side_rows.loc[
            side_rows["parent_rank_v9"].ge(config.top20_floor)
        ].sort_values("__ts__", kind="stable")
        if len(local) < config.min_train_rows:
            continue
        for fold_index, fold_start in enumerate(FOLD_STARTS):
            fold_end = (
                FOLD_STARTS[fold_index + 1]
                if fold_index + 1 < len(FOLD_STARTS)
                else pd.Timestamp(config.train_end, tz="UTC")
            )
            fit = local.loc[
                local["__ts__"].lt(fold_start - pd.Timedelta(days=2))
            ]
            score = local.loc[
                local["__ts__"].ge(fold_start) & local["__ts__"].lt(fold_end)
            ]
            if (
                len(fit) < config.min_train_rows
                or int(fit[SIDE_EVENT].sum()) < config.min_positive_rows
                or score.empty
            ):
                continue
            fit_screen = _timestamp_training_frame(
                fit,
                candidates,
                target_column=SIDE_EVENT,
                event_column=SIDE_EVENT,
            )
            selected, report = _screen_features(
                fit_screen, candidates, config, target_column=SIDE_EVENT
            )
            if not selected:
                continue
            (
                _,
                valid_score,
                state_valid_score,
                _,
                reference,
                fit_state,
                score_state,
            ) = _fit_timestamp_model(
                fit,
                score,
                selected,
                config.seed + 50_000 + 100 * side_index + fold_index,
                target_column=SIDE_EVENT,
                event_column=SIDE_EVENT,
            )
            part = score.loc[
                :,
                KEYS
                + [
                    "day",
                    "parent_rank_v9",
                    "ev_after_1pct",
                    "clean_exec",
                    EVENT,
                    TARGET,
                    SIDE_EVENT,
                ],
            ].copy()
            part[SIDE_RISK_SCORE] = valid_score
            part[SIDE_RISK_PCT] = _midrank(valid_score, reference)
            part["fold_start"] = fold_start
            part["fold_end"] = fold_end
            oof_parts.append(part)
            report.insert(0, "fold_start", fold_start)
            report.insert(0, "archetype_policy_key", "__side_parent__")
            report.insert(0, "side_name", side)
            feature_reports.append(report)
            model_rows.append(
                {
                    "stage": "side_oof",
                    "side_name": side,
                    "archetype_policy_key": "__side_parent__",
                    "fold_start": fold_start,
                    "train_rows": len(fit),
                    "train_target_rows": int(fit[SIDE_EVENT].sum()),
                    "score_rows": len(score),
                    "features": len(selected),
                    "selected_features": "|".join(selected),
                    "average_precision": float(
                        average_precision_score(
                            score_state[SIDE_EVENT], state_valid_score
                        )
                    )
                    if score_state[SIDE_EVENT].nunique() > 1
                    else np.nan,
                    "train_state_rows": len(fit_state),
                    "score_state_rows": len(score_state),
                }
            )

        local_screen = _timestamp_training_frame(
            local,
            candidates,
            target_column=SIDE_EVENT,
            event_column=SIDE_EVENT,
        )
        selected, report = _screen_features(
            local_screen, candidates, config, target_column=SIDE_EVENT
        )
        if not selected or int(local[SIDE_EVENT].sum()) < config.min_positive_rows:
            continue
        side_valid = valid.loc[
            valid["side_name"].astype(str).eq(str(side))
            & valid["parent_rank_v9"].ge(config.top20_floor)
        ].sort_values("__ts__", kind="stable")
        if side_valid.empty:
            continue
        (
            model,
            valid_score,
            state_valid_score,
            medians,
            reference,
            train_state,
            valid_state,
        ) = _fit_timestamp_model(
            local,
            side_valid,
            selected,
            config.seed + 60_000 + side_index,
            target_column=SIDE_EVENT,
            event_column=SIDE_EVENT,
        )
        final_states[str(side)] = {
            "model": model,
            "features": selected,
            "medians": medians,
            "reference": reference,
            "valid_index": side_valid.index.to_numpy(),
            "valid_score": valid_score,
        }
        report.insert(0, "fold_start", "final")
        report.insert(0, "archetype_policy_key", "__side_parent__")
        report.insert(0, "side_name", side)
        feature_reports.append(report)
        model_rows.append(
            {
                "stage": "side_final",
                "side_name": side,
                "archetype_policy_key": "__side_parent__",
                "fold_start": config.train_end,
                "train_rows": len(local),
                "train_target_rows": int(local[SIDE_EVENT].sum()),
                "score_rows": len(side_valid),
                "features": len(selected),
                "selected_features": "|".join(selected),
                "average_precision": float(
                    average_precision_score(valid_state[SIDE_EVENT], state_valid_score)
                )
                if valid_state[SIDE_EVENT].nunique() > 1
                else np.nan,
                "train_state_rows": len(train_state),
                "score_state_rows": len(valid_state),
            }
        )
        model.save_model(str(output / f"model__side_parent__{side}.txt"))
        np.savez_compressed(
            output / f"state__side_parent__{side}.npz",
            features=np.asarray(selected, dtype=np.str_),
            medians=np.asarray(medians, dtype=np.float32),
            reference=np.asarray(reference, dtype=np.float32),
        )
    if not oof_parts:
        raise RuntimeError("No chronological OOF side-parent predictions were generated")
    return (
        pd.concat(oof_parts, ignore_index=True, copy=False),
        pd.concat(feature_reports, ignore_index=True, copy=False),
        pd.DataFrame(model_rows),
        final_states,
    )


def _apply_selected_overlays(
    frame: pd.DataFrame,
    params: dict[tuple[str, str], dict[str, Any]],
    rank_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    adjusted = frame[rank_column].to_numpy(np.float32).copy()
    flagged = np.zeros(len(frame), dtype=bool)
    for (side, archetype), local_params in params.items():
        mask = (
            frame["side_name"].astype(str).eq(side)
            & frame["archetype_policy_key"].astype(str).eq(archetype)
        ).to_numpy()
        idx = np.flatnonzero(mask)
        if not len(idx):
            continue
        risk_column = str(local_params.get("risk_variant") or RISK_PCT)
        local_rank, local_flagged = _apply_rank(
            adjusted[idx],
            frame.iloc[idx][risk_column].to_numpy(np.float32),
            float(local_params["threshold"]),
            float(local_params["alpha"]),
            str(local_params["mode"]) == "hard_block",
            0.90,
        )
        adjusted[idx] = local_rank
        flagged[idx] = local_flagged
    return adjusted, flagged


def _monthly_selected_ev(frame: pd.DataFrame, rank: np.ndarray) -> pd.Series:
    selected = rank >= 0.90
    month = frame["__ts__"].dt.strftime("%Y-%m")
    ev = pd.to_numeric(frame["ev_after_1pct"], errors="coerce")
    return ev.loc[selected].groupby(month.loc[selected], observed=True).mean()


def _breakdown(frame: pd.DataFrame, selector: str, rank: np.ndarray) -> pd.DataFrame:
    selected = frame.loc[rank >= 0.90].copy()
    selected["month"] = selected["__ts__"].dt.strftime("%Y-%m")
    day = selected["__ts__"].dt.floor("D")
    selected["week_start"] = day - pd.to_timedelta(day.dt.weekday, unit="D")
    reports: list[pd.DataFrame] = []
    for scope, groups in (
        ("month", ["month"]),
        ("week", ["week_start"]),
        ("side_archetype", ["side_name", "archetype_policy_key"]),
        ("event_state", [EVENT]),
        ("month_side_archetype", ["month", "side_name", "archetype_policy_key"]),
    ):
        report = (
            selected.groupby(groups, observed=True, dropna=False)
            .agg(
                selected_rows=("ev_after_1pct", "size"),
                mean_ev_after_1pct=("ev_after_1pct", "mean"),
                positive_ev_rate=("ev_after_1pct", lambda values: float((values > 0).mean())),
                clean_exec_precision=("clean_exec", "mean"),
            )
            .reset_index()
        )
        report["scope"] = scope
        report["selector"] = selector
        reports.append(report)
    return pd.concat(reports, ignore_index=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = Config(
        train_start=args.train_start,
        train_end=args.train_end,
        eval_end=args.eval_end,
        max_features=args.max_features,
        targeted_temporal_features=args.targeted_temporal_features,
        targeted_temporal_only=args.targeted_temporal_only,
        seed=args.seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    train, valid, coverage = _load_frames(args, config)
    events = _load_event_cells(args.event_calendar, args.extension_calendar)
    train = _attach_event_target(train, events)
    valid = _attach_event_target(valid, events)
    expected_clean_baseline = _fit_expected_clean_baseline(
        train, top10_floor=config.top10_floor
    )
    train, train_calendar = _v9_residual_calendar(
        train,
        top10_floor=config.top10_floor,
        expected_clean_baseline=expected_clean_baseline,
    )
    valid, valid_calendar = _v9_residual_calendar(
        valid,
        top10_floor=config.top10_floor,
        expected_clean_baseline=expected_clean_baseline,
    )
    for frame in (train, valid):
        frame[SIDE_EVENT] = (
            frame.groupby(["day", "side_name"], observed=True)[EVENT]
            .transform("max")
            .fillna(0)
            .astype(np.int8)
        )
    expected_clean_baseline.to_csv(
        args.output / "v9_train_expected_clean_baseline.csv", index=False
    )
    train_calendar.to_csv(args.output / "v9_train_residual_calendar.csv", index=False)
    valid_calendar.to_csv(args.output / "v9_eval_residual_calendar.csv", index=False)
    candidates = _candidate_features(train.columns)
    if not candidates:
        raise RuntimeError("No leakage-safe residual-error model features are available")
    oof, feature_report, model_report, final_states = _fit_oof_and_final(
        train,
        valid,
        candidates,
        config,
        args.output,
    )
    side_oof, side_feature_report, side_model_report, side_final_states = (
        _fit_side_oof_and_final(
            train,
            valid,
            candidates,
            config,
            args.output,
        )
    )
    oof = oof.merge(
        side_oof.loc[:, KEYS + [SIDE_RISK_SCORE, SIDE_RISK_PCT]].drop_duplicates(
            KEYS, keep="last"
        ),
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    oof[SIDE_RISK_SCORE] = pd.to_numeric(
        oof[SIDE_RISK_SCORE], errors="coerce"
    ).astype(np.float32)
    oof[SIDE_RISK_PCT] = pd.to_numeric(
        oof[SIDE_RISK_PCT], errors="coerce"
    ).fillna(0.5).astype(np.float32)
    oof = _add_risk_variants(oof)
    feature_report = pd.concat(
        [feature_report, side_feature_report], ignore_index=True, copy=False
    )
    model_report = pd.concat(
        [model_report, side_model_report], ignore_index=True, copy=False
    )

    local_searches: list[pd.DataFrame] = []
    accepted: dict[tuple[str, str], dict[str, Any]] = {}
    for (side, archetype), local in oof.groupby(
        ["side_name", "archetype_policy_key"], observed=True, sort=True
    ):
        local_best: list[dict[str, Any]] = []
        for risk_variant in RISK_VARIANTS:
            search, best = _search_local_overlay(
                local, config, risk_column=risk_variant
            )
            search.insert(0, "archetype_policy_key", archetype)
            search.insert(0, "side_name", side)
            local_searches.append(search)
            if best is not None:
                local_best.append(best)
        if local_best:
            accepted[(str(side), str(archetype))] = max(
                local_best, key=lambda row: float(row["objective"])
            )

    valid[RISK_SCORE] = np.float32(np.nan)
    valid[RISK_PCT] = np.float32(0.5)
    for state in final_states.values():
        idx = state["valid_index"]
        score = state["valid_score"]
        valid.loc[idx, RISK_SCORE] = score
        valid.loc[idx, RISK_PCT] = _midrank(score, state["reference"])
    valid[SIDE_RISK_SCORE] = np.float32(np.nan)
    valid[SIDE_RISK_PCT] = np.float32(0.5)
    for state in side_final_states.values():
        idx = state["valid_index"]
        score = state["valid_score"]
        valid.loc[idx, SIDE_RISK_SCORE] = score
        valid.loc[idx, SIDE_RISK_PCT] = _midrank(score, state["reference"])
    valid = _add_risk_variants(valid)
    parent_rank = valid["parent_rank_v9"].to_numpy(np.float32)
    adjusted_rank, flagged = _apply_selected_overlays(valid, accepted, "parent_rank_v9")
    valid["parent_rank_v9_residual_error_overlay"] = adjusted_rank
    valid["residual_error_overlay_flagged"] = flagged
    valid.to_parquet(args.output / "oos_predictions.parquet", index=False, compression="zstd")
    oof.to_parquet(args.output / "train_oof_predictions.parquet", index=False, compression="zstd")
    feature_report.to_csv(args.output / "feature_screening.csv", index=False)
    model_report.to_csv(args.output / "model_report.csv", index=False)
    search_frame = pd.concat(local_searches, ignore_index=True, copy=False)
    search_frame.to_csv(args.output / "local_overlay_search.csv", index=False)
    pd.DataFrame(
        [
            {"side_name": side, "archetype_policy_key": archetype, **params}
            for (side, archetype), params in accepted.items()
        ]
    ).to_csv(args.output / "accepted_local_overlays.csv", index=False)

    parent_metrics = _selection_metrics(valid, parent_rank, config.top10_floor)
    adjusted_metrics = _selection_metrics(valid, adjusted_rank, config.top10_floor)
    parent_month = _monthly_selected_ev(valid, parent_rank)
    adjusted_month = _monthly_selected_ev(valid, adjusted_rank)
    month_delta = adjusted_month.subtract(parent_month).dropna()
    worst_month_delta = float(month_delta.min()) if len(month_delta) else np.nan
    summary = pd.DataFrame(
        [
            {"selector": "v9_parent", **parent_metrics},
            {"selector": "v9_residual_event_error_overlay", **adjusted_metrics},
        ]
    )
    for metric in ("mean_ev", "positive_ev_rate", "clean_precision", "event_mean_ev", "normal_mean_ev"):
        summary[f"delta_{metric}_vs_parent"] = summary[metric] - parent_metrics[metric]
    summary.to_csv(args.output / "summary.csv", index=False)
    pd.concat(
        [
            _breakdown(valid, "v9_parent", parent_rank),
            _breakdown(valid, "v9_residual_event_error_overlay", adjusted_rank),
        ],
        ignore_index=True,
    ).to_csv(args.output / "breakdowns.csv", index=False)

    manifest = {
        "schema": "meta_residual_event_balanced_error_overlay_v2",
        "config": asdict(config),
        "coverage": coverage,
        "candidate_features": candidates,
        "accepted_local_overlays": [
            {"side_name": side, "archetype_policy_key": archetype, **params}
            for (side, archetype), params in accepted.items()
        ],
        "train_oof_rows": len(oof),
        "side_parent_oof_rows": len(side_oof),
        "eval_rows": len(valid),
        "eval_flagged_parent_rows": int(flagged.sum()),
        "eval_month_ev_delta": {
            str(month): float(value) for month, value in month_delta.items()
        },
        "eval_worst_month_ev_delta": worst_month_delta,
        "train_v9_adverse_cells": int(train_calendar[EVENT].sum()),
        "eval_v9_adverse_cells": int(valid_calendar[EVENT].sum()),
        "eval_promotion_pass": bool(
            adjusted_metrics["mean_ev"] > parent_metrics["mean_ev"]
            and adjusted_metrics["positive_ev_rate"] > parent_metrics["positive_ev_rate"]
            and adjusted_metrics["event_mean_ev"] > parent_metrics["event_mean_ev"]
            and adjusted_metrics["normal_mean_ev"]
            >= parent_metrics["normal_mean_ev"] - config.max_normal_ev_degradation
            and adjusted_metrics["selected_rows"]
            >= config.min_activity_ratio * parent_metrics["selected_rows"]
            and worst_month_delta >= -config.max_normal_ev_degradation
        ),
        "training_unit": (
            "One cross-sectional median observable state per timestamp is fitted "
            "within each side x archetype and side parent. Frozen timestamp risk "
            "is broadcast back to candidate rows; realized row outcomes never enter "
            "the inference matrix."
        ),
        "leakage_contract": (
            "Realized EV and the V9-specific clean-hit residual calendar define labels only. "
            "Feature screening, local LGBM fits, score references, and overlay "
            "parameters use chronological OOF rows ending 2026-03-31. Side-parent "
            "models share market-state support across archetypes but every overlay "
            "is still selected and evaluated within one side x archetype. April-June "
            "2026 is untouched evaluation. Inputs exclude outcomes, residual targets, "
            "historical performance overlays, and threshold artifacts. Two days are "
            "purged at every internal fold boundary for the adjacent-day label."
        ),
        "promotion_contract": (
            "Each local overlay requires target lift >=1.5, FPR <=15%, positive "
            "incremental positive-EV precision, support in >=3 adverse event blocks, "
            "positive overall/event EV deltas, >=95% activity, and <=1bp normal-period "
            "EV degradation on train OOF predictions before untouched evaluation."
        ),
    }
    _write_json(args.output / "manifest.json", manifest)
    print(summary.to_string(index=False), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--champion-ledger",
        type=Path,
        default=Path(
            "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
            "champion_frozen_single_source_202501_20260710/"
            "frozen_champion_single_source_ledger.parquet"
        ),
    )
    parser.add_argument(
        "--train-oof-predictions-dir",
        type=Path,
        default=Path(
            "data_perp/reports/s59_h5_2025start_monthly_v4_base_configfull_"
            "mdafs120_hpo150_largestfold_oos15_ae3000_nocrossfit_k34567_"
            "payload300k_20260706/train_meta_regime_handoff_singlehead_base_soft_"
            "lgbmpipeline_auto_hpo150_oos15_top30_hpo45k_20260706_v5/"
            "best_full_oos_fixedfs_streamed_v1/prediction_shards"
        ),
    )
    parser.add_argument(
        "--train-oof-rank-cache",
        type=Path,
        default=Path(
            "data_perp/reports/residual_event_archetype_true_base_oof_compactlocal_"
            "market_20260712_v3/meta_oof_global_rank_202504_202603.parquet"
        ),
    )
    parser.add_argument(
        "--state-artifact",
        type=Path,
        default=Path(
            "data_perp/reports/residual_event_archetype_true_base_oof_compactlocal_"
            "market_20260712_v3/oos_residual_event_states.parquet"
        ),
    )
    parser.add_argument(
        "--additional-state-artifact",
        type=Path,
        action="append",
        default=[],
        help="Additional disjoint OOS state artifact, for example early-2025 coverage.",
    )
    parser.add_argument(
        "--direct-parent-rank",
        action="store_true",
        help=(
            "Research-only: use the supplied causal parent historical_rank directly "
            "rather than applying the V9 strict local rank adjustment."
        ),
    )
    parser.add_argument(
        "--state-group-filter",
        default="",
        help="Optional side::archetype parquet-level state filter for isolated local studies.",
    )
    parser.add_argument(
        "--parent-eval-predictions",
        type=Path,
        default=Path(
            "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
            "lifecycle_residual_aware_ae_gmm_overlay_pca8_clip8_baseline_"
            "globaloverlay_sparse_shock_composite/oos_predictions_historical_rank.parquet"
        ),
    )
    parser.add_argument(
        "--v9-predictions",
        type=Path,
        default=Path(
            "data_perp/reports/meta_residual_extreme_local_champion_overlay_"
            "ooftrain_tieaware_downonly_20260712_v9/oos_predictions.parquet"
        ),
    )
    parser.add_argument(
        "--v9-manifest",
        type=Path,
        default=Path(
            "data_perp/reports/meta_residual_extreme_local_champion_overlay_"
            "ooftrain_tieaware_downonly_20260712_v9/manifest.json"
        ),
    )
    parser.add_argument(
        "--v9-selected-features",
        type=Path,
        default=Path(
            "data_perp/reports/meta_residual_extreme_local_champion_overlay_"
            "ooftrain_tieaware_downonly_20260712_v9/selected_local_features_strict.csv"
        ),
    )
    parser.add_argument(
        "--negative-residual-features",
        type=Path,
        default=Path("data_perp/features/20260712_185800/symbol=BTC_USD:USD.parquet"),
    )
    parser.add_argument(
        "--temporal-state-features",
        type=Path,
        default=Path(
            "data_perp/reports/residual_event_target_transitions_july_oos_20260713_v2_"
            "support_fallback/oos_temporal_state_context_apr2025_july2026.parquet"
        ),
    )
    parser.add_argument(
        "--event-calendar",
        type=Path,
        default=Path(
            "data_perp/reports/residual_episode_recognition_calendar_20260712_v1/"
            "calendar_recognized_vs_ignored.csv"
        ),
    )
    parser.add_argument(
        "--extension-calendar",
        type=Path,
        default=Path(
            "data_perp/reports/residual_event_target_transitions_july_oos_20260713_v2_"
            "support_fallback/residual_event_calendar.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data_perp/reports/meta_residual_event_balanced_error_overlay_20260713_v1"
        ),
    )
    parser.add_argument("--train-start", default="2025-04-01")
    parser.add_argument("--train-end", default="2026-04-01")
    parser.add_argument("--eval-end", default="2026-07-01")
    parser.add_argument("--max-features", type=int, default=24)
    parser.add_argument(
        "--targeted-temporal-features",
        type=int,
        default=0,
        help=(
            "Reserve this many local slots for the matching compression, persistence, "
            "or breakout family; the total remains max-features."
        ),
    )
    parser.add_argument(
        "--targeted-temporal-only",
        action="store_true",
        help="Fit matching local models only on their observable temporal mechanism family.",
    )
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()
    print(json.dumps(_json_safe(run(args)), indent=2))


if __name__ == "__main__":
    main()
