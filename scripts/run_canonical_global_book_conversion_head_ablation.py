#!/usr/bin/env python3
"""Strict chronological learnability test for causal global-book conversion.

This runner fits compact, fixed-geometry CatBoost regression and sign heads on
the immutable global-book and global-EV-band transition artifacts.  It is a
component-learnability experiment only: it does not alter candidate ranking,
admission, execution actions, or portfolio policy.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

try:
    from scripts.run_canonical_economic_conversion_transition_head_ablation import (
        CATBOOST_GEOMETRY,
        _artifact_manifest,
        _catboost_classifier,
        _catboost_regressor,
        _classification_metrics,
        _regression_metrics,
        _safe,
        build_expanding_folds,
        sha256,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from run_canonical_economic_conversion_transition_head_ablation import (
        CATBOOST_GEOMETRY,
        _artifact_manifest,
        _catboost_classifier,
        _catboost_regressor,
        _classification_metrics,
        _regression_metrics,
        _safe,
        build_expanding_folds,
        sha256,
    )


ROOT = Path(__file__).resolve().parents[1]
CONTEXT_SOURCE = (
    ROOT
    / "data_perp/artifacts/canonical_global_book_conversion_context_20260729_v1"
)
LABEL_SOURCE = (
    ROOT
    / "data_perp/artifacts/"
    "canonical_global_book_conversion_transition_labels_20260729_v1"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data_perp/artifacts/"
    "canonical_global_book_conversion_head_ablation_20260729_v1"
)
SCHEMA = "canonical_global_book_conversion_head_ablation_v1"
CONTEXT_SCHEMA = "canonical_global_book_conversion_context_v1"
LABEL_SCHEMA = "canonical_global_book_conversion_transition_labels_v1"
PRIMARY_HORIZON = 12
SENSITIVITY_HORIZON = 3
PRIMARY_BOOK_FRACTION = 0.10


@dataclass(frozen=True)
class TargetSpec:
    name: str
    book_column: str
    band_column: str


TARGETS = (
    TargetSpec(
        "conversion_residual",
        "delta_mean_conversion_residual",
        "delta_mean_conversion_residual",
    ),
    TargetSpec(
        "realized_net",
        "delta_direct_mean_net",
        "delta_mean_realized_net",
    ),
    TargetSpec(
        "mapped_ev",
        "delta_mapped_score_mean",
        "delta_mean_mapped_ev",
    ),
    TargetSpec(
        "opportunity_probability_0bps",
        "delta_opportunity_probability_0bps",
        "delta_opportunity_probability_0bps",
    ),
    TargetSpec(
        "positive_net_contribution",
        "delta_positive_net_contribution",
        "delta_positive_net_contribution",
    ),
    TargetSpec(
        "loss_net_contribution",
        "delta_loss_net_contribution",
        "delta_loss_net_contribution",
    ),
)

AUDIT_COLUMNS = (
    "before_global_hour_complete_flag",
    "after_global_hour_complete_flag",
    "before_target_available_utc",
    "after_target_available_utc",
)


def _source_hashes(
    context_source: Path, label_source: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    context_manifest, context_hashes = _artifact_manifest(
        context_source, CONTEXT_SCHEMA
    )
    label_manifest, label_hashes = _artifact_manifest(
        label_source, LABEL_SCHEMA
    )
    paths = (
        context_source / "global_book_context.parquet",
        context_source / "global_ev_band_context.parquet",
        label_source / "global_book_transition_labels.parquet",
        label_source / "global_ev_band_transition_labels.parquet",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"global-book head source lacks material files: {missing}"
        )
    expected_context = context_manifest.get("source_artifacts_sha256", {})
    for path in paths[2:]:
        if expected_context.get(str(path)) != sha256(path):
            raise ValueError(
                "context artifact is not bound to the supplied label file: "
                f"{path.name}"
            )
    return (
        {"context": context_manifest, "labels": label_manifest},
        {
            **context_hashes,
            **label_hashes,
            **{str(path): sha256(path) for path in paths},
        },
    )


def _features(
    manifest: Mapping[str, Any], architecture: str
) -> tuple[str, ...]:
    key = (
        "global_book_feature_columns"
        if architecture == "book"
        else "global_band_feature_columns"
    )
    values = tuple(map(str, manifest.get(key, [])))
    if not values or len(values) != len(set(values)):
        raise ValueError(
            f"{architecture} context has an empty or duplicate feature contract"
        )
    prohibited = (
        "target",
        "label",
        "outcome",
        "realized",
        "opportunity_",
        "exit",
        "mfe",
        "mae",
        "execution_net",
        "execution_gross",
        "execution_cost",
        "wait_action",
        "target_price",
    )
    bad = [
        column
        for column in values
        if any(token in column.lower() for token in prohibited)
    ]
    if bad:
        raise ValueError(
            f"{architecture} context permits prohibited model features: {bad}"
        )
    forbidden_task_features = {
        "context__book_fraction",
        "context__horizon_hours",
    }
    if forbidden_task_features.intersection(values):
        raise ValueError(
            "book fraction and horizon must be task keys, not model features"
        )
    return values


def _key(architecture: str) -> tuple[str, ...]:
    if architecture == "book":
        return ("cohort_anchor_utc", "horizon_hours", "book_fraction")
    if architecture == "band":
        return (
            "cohort_anchor_utc",
            "horizon_hours",
            "global_common_ev_band",
        )
    raise ValueError(f"unsupported architecture: {architecture}")


def _target_column(target: TargetSpec, architecture: str) -> str:
    return target.book_column if architecture == "book" else target.band_column


def _prepare_population(
    *,
    architecture: str,
    context: pd.DataFrame,
    labels: pd.DataFrame,
    features: Iterable[str],
) -> pd.DataFrame:
    features = tuple(features)
    key = _key(architecture)
    required_context = {
        *key,
        *features,
        *(f"label_audit__{column}" for column in AUDIT_COLUMNS),
    }
    required_labels = {
        *key,
        "horizon_role",
        *AUDIT_COLUMNS,
        *(
            _target_column(target, architecture)
            for target in TARGETS
        ),
    }
    if architecture == "book":
        required_labels.update(
            {
                "before_selected_candidate_support",
                "after_selected_candidate_support",
            }
        )
        support_columns = (
            "before_selected_candidate_support",
            "after_selected_candidate_support",
        )
    else:
        required_labels.update(
            {"before_candidate_support", "after_candidate_support"}
        )
        support_columns = (
            "before_candidate_support",
            "after_candidate_support",
        )
    missing_context = sorted(required_context.difference(context.columns))
    missing_labels = sorted(required_labels.difference(labels.columns))
    if missing_context or missing_labels:
        raise ValueError(
            "global-book head input lacks columns: "
            f"context={missing_context}, labels={missing_labels}"
        )
    context = context.loc[:, [*key, *(f"label_audit__{c}" for c in AUDIT_COLUMNS), *features]].copy()
    labels = labels.loc[:, [*key, "horizon_role", *AUDIT_COLUMNS, *support_columns, *(_target_column(t, architecture) for t in TARGETS)]].copy()
    for frame in (context, labels):
        frame["cohort_anchor_utc"] = pd.to_datetime(
            frame["cohort_anchor_utc"], utc=True, errors="raise"
        )
        frame["horizon_hours"] = pd.to_numeric(
            frame["horizon_hours"], errors="raise"
        ).astype(np.int8)
    if architecture == "book":
        for frame in (context, labels):
            frame["book_fraction"] = pd.to_numeric(
                frame["book_fraction"], errors="raise"
            )
    else:
        for frame in (context, labels):
            frame["global_common_ev_band"] = frame[
                "global_common_ev_band"
            ].astype(str)
    for column in (
        "before_target_available_utc",
        "after_target_available_utc",
    ):
        labels[column] = pd.to_datetime(
            labels[column], utc=True, errors="coerce"
        )
        context[f"label_audit__{column}"] = pd.to_datetime(
            context[f"label_audit__{column}"], utc=True, errors="coerce"
        )
    if context.duplicated(list(key)).any() or labels.duplicated(list(key)).any():
        raise ValueError(
            f"{architecture} context/label identity is not one-to-one"
        )
    joined = labels.merge(
        context, on=list(key), how="left", validate="one_to_one"
    )
    if joined.loc[:, list(features)].isna().all(axis=1).any():
        raise ValueError(
            f"{architecture} labels lack all decision-time context"
        )
    for column in AUDIT_COLUMNS:
        left = joined[column]
        right = joined[f"label_audit__{column}"]
        if "available_utc" in column:
            parity = left.eq(right) | (left.isna() & right.isna())
        else:
            parity = left.astype(bool).eq(right.astype(bool))
        if not parity.all():
            raise ValueError(
                f"{architecture} context/label audit parity failed: {column}"
            )
    for column in features:
        joined[column] = pd.to_numeric(joined[column], errors="coerce")
        if np.isinf(
            joined[column].to_numpy(dtype=float, na_value=np.nan)
        ).any():
            raise ValueError(f"feature contains infinity: {column}")
    joined["label_available_utc"] = pd.concat(
        [
            joined["before_target_available_utc"],
            joined["after_target_available_utc"],
        ],
        axis=1,
    ).max(axis=1)
    complete = (
        joined["before_global_hour_complete_flag"].astype(bool)
        & joined["after_global_hour_complete_flag"].astype(bool)
        & pd.to_numeric(joined[support_columns[0]], errors="coerce").gt(0)
        & pd.to_numeric(joined[support_columns[1]], errors="coerce").gt(0)
        & joined["after_target_available_utc"].notna()
    )
    for target in TARGETS:
        column = _target_column(target, architecture)
        joined[column] = pd.to_numeric(joined[column], errors="coerce")
        joined[f"{target.name}__target_valid"] = complete & np.isfinite(
            joined[column]
        )
    return joined.sort_values(list(key), kind="stable").reset_index(drop=True)


def _spread_cap(frame: pd.DataFrame, budget: int) -> pd.DataFrame:
    ordered = frame.sort_values(
        ["cohort_anchor_utc"], kind="stable"
    )
    if len(ordered) <= int(budget):
        return ordered
    positions = np.linspace(
        0, len(ordered) - 1, int(budget), dtype=np.int64
    )
    return ordered.iloc[positions].copy()


def _last_known(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    target_column: str,
    *,
    architecture: str,
) -> np.ndarray:
    groups = (
        ["global_common_ev_band"] if architecture == "band" else []
    )
    ordered = train.sort_values(
        ["label_available_utc", "cohort_anchor_utc"], kind="stable"
    )
    if not groups:
        value = (
            float(ordered[target_column].iloc[-1])
            if len(ordered)
            else np.nan
        )
        return np.full(len(evaluation), value, dtype=float)
    last = (
        ordered.groupby(groups, sort=False, observed=True)
        .tail(1)
        .loc[:, [*groups, target_column]]
    )
    return (
        evaluation.loc[:, groups]
        .merge(last, on=groups, how="left", validate="many_to_one")[
            target_column
        ]
        .to_numpy(dtype=float)
    )


def _eligible_training_rows(
    population: pd.DataFrame,
    *,
    valid_column: str,
    validation_start_utc: pd.Timestamp,
) -> pd.DataFrame:
    """Return only targets fully resolved before a validation boundary."""

    start = pd.Timestamp(validation_start_utc)
    train = population.loc[
        population[valid_column].astype(bool)
        & population["label_available_utc"].lt(start)
        & population["after_target_available_utc"].lt(start)
    ].copy()
    if len(train) and not (
        train["label_available_utc"].lt(start).all()
        and train["after_target_available_utc"].lt(start).all()
    ):
        raise AssertionError(
            "training targets are not strictly resolved before validation"
        )
    return train


def _metrics(
    y: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
    constant: np.ndarray,
    constant_probability: np.ndarray,
    last: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for prefix, values in (
        ("model", prediction),
        ("constant", constant),
        ("last_known", last),
    ):
        for name, value in _regression_metrics(y, values).items():
            if name != "rows":
                result[f"{prefix}_regression_{name}"] = value
    sign = (y > 0).astype(np.int8)
    last_probability = np.where(
        np.isfinite(last), (last > 0).astype(float), np.nan
    )
    for prefix, values in (
        ("model", probability),
        ("constant", constant_probability),
        ("last_known", last_probability),
    ):
        for name, value in _classification_metrics(sign, values).items():
            if name != "rows":
                result[f"{prefix}_sign_{name}"] = value
    return result


def _fit_target(
    population: pd.DataFrame,
    *,
    architecture: str,
    target: TargetSpec,
    features: Iterable[str],
    folds: Iterable[Mapping[str, Any]],
    min_train_rows: int,
    fit_budget_rows: int,
    random_state: int,
    threads: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = tuple(features)
    column = _target_column(target, architecture)
    key = _key(architecture)
    parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for fold in folds:
        start = pd.Timestamp(fold["validation_start_utc"])
        end = pd.Timestamp(fold["validation_end_utc"])
        evaluation = population.loc[
            population["cohort_anchor_utc"].ge(start)
            & population["cohort_anchor_utc"].lt(end)
        ].copy()
        if evaluation.empty:
            continue
        valid_column = f"{target.name}__target_valid"
        valid_eval = evaluation[valid_column].to_numpy(bool)
        train = _eligible_training_rows(
            population,
            valid_column=valid_column,
            validation_start_utc=start,
        )
        fitted = _spread_cap(train, int(fit_budget_rows))
        y_train = fitted[column].to_numpy(dtype=float)
        constant = float(y_train.mean()) if len(y_train) else 0.0
        constant_probability = (
            float((y_train > 0).mean()) if len(y_train) else 0.0
        )
        last = _last_known(
            train,
            evaluation,
            column,
            architecture=architecture,
        )
        result = evaluation.loc[
            :,
            [
                *key,
                "horizon_role",
                "label_available_utc",
                "after_target_available_utc",
                column,
                valid_column,
            ],
        ].copy()
        result["architecture"] = architecture
        result["target"] = target.name
        result["fold_id"] = int(fold["fold_id"])
        result["validation_start_utc"] = start
        result["validation_end_utc"] = end
        result["delta_prediction"] = np.nan
        result["sign_probability"] = np.nan
        result["constant_delta_prediction"] = np.where(
            valid_eval, constant, np.nan
        )
        result["constant_sign_probability"] = np.where(
            valid_eval, constant_probability, np.nan
        )
        result["last_known_delta_prediction"] = np.where(
            valid_eval, last, np.nan
        )
        status = "constant_fallback_insufficient_prior_resolved_rows"
        if valid_eval.any() and len(fitted) >= int(min_train_rows):
            x_train = fitted.loc[:, list(features)]
            x_eval = evaluation.loc[valid_eval, list(features)]
            regressor = _catboost_regressor(
                seed=(
                    int(random_state)
                    + 10_000 * int(fold["fold_id"])
                    + len(target.name)
                    + (1 if architecture == "band" else 0)
                ),
                threads=int(threads),
            ).fit(x_train, y_train)
            result.loc[valid_eval, "delta_prediction"] = np.asarray(
                regressor.predict(x_eval), dtype=float
            )
            sign_train = (y_train > 0).astype(np.int8)
            if np.unique(sign_train).size == 2:
                classifier = _catboost_classifier(
                    seed=(
                        int(random_state)
                        + 20_000 * int(fold["fold_id"])
                        + len(target.name)
                        + (1 if architecture == "band" else 0)
                    ),
                    threads=int(threads),
                ).fit(x_train, sign_train)
                result.loc[valid_eval, "sign_probability"] = np.asarray(
                    classifier.predict_proba(x_eval)[:, 1], dtype=float
                )
                status = "fixed_compact_catboost_regression_and_sign"
            else:
                result.loc[
                    valid_eval, "sign_probability"
                ] = constant_probability
                status = "fixed_compact_catboost_regression_constant_sign"
        else:
            result.loc[valid_eval, "delta_prediction"] = constant
            result.loc[valid_eval, "sign_probability"] = constant_probability
        result["fit_status"] = status
        result["training_rows"] = int(len(train))
        result["fitted_training_rows"] = int(len(fitted))
        result["training_max_after_target_available_utc"] = (
            train["after_target_available_utc"].max()
            if len(train)
            else pd.NaT
        )
        result = result.rename(
            columns={
                column: "target_delta",
                valid_column: "target_valid",
            }
        )
        parts.append(result)
        valid_result = result.loc[result["target_valid"].astype(bool)]
        metric = {
            "architecture": architecture,
            "target": target.name,
            "horizon_hours": int(
                evaluation["horizon_hours"].iloc[0]
            ),
            **dict(fold),
            "evaluation_rows": int(len(evaluation)),
            "valid_rows": int(len(valid_result)),
            "training_rows": int(len(train)),
            "fitted_training_rows": int(len(fitted)),
            "fit_status": status,
        }
        metric.update(
            _metrics(
                valid_result["target_delta"].to_numpy(dtype=float),
                valid_result["delta_prediction"].to_numpy(dtype=float),
                valid_result["sign_probability"].to_numpy(dtype=float),
                valid_result["constant_delta_prediction"].to_numpy(dtype=float),
                valid_result["constant_sign_probability"].to_numpy(dtype=float),
                valid_result["last_known_delta_prediction"].to_numpy(dtype=float),
            )
        )
        metric_rows.append(metric)
    return (
        pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(),
        pd.DataFrame(metric_rows),
    )


def _aggregate(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if predictions.empty:
        return pd.DataFrame()
    groups = ["architecture", "horizon_hours", "target"]
    for key, group in predictions.groupby(groups, sort=True):
        valid = group.loc[group["target_valid"].astype(bool)]
        row: dict[str, Any] = {
            **dict(zip(groups, key)),
            "oof_rows": int(len(valid)),
            "folds": int(valid["fold_id"].nunique()),
            "latest_fold_id": int(valid["fold_id"].max()),
        }
        row.update(
            _metrics(
                valid["target_delta"].to_numpy(dtype=float),
                valid["delta_prediction"].to_numpy(dtype=float),
                valid["sign_probability"].to_numpy(dtype=float),
                valid["constant_delta_prediction"].to_numpy(dtype=float),
                valid["constant_sign_probability"].to_numpy(dtype=float),
                valid["last_known_delta_prediction"].to_numpy(dtype=float),
            )
        )
        latest = valid.loc[valid["fold_id"].eq(valid["fold_id"].max())]
        latest_metrics = _metrics(
            latest["target_delta"].to_numpy(dtype=float),
            latest["delta_prediction"].to_numpy(dtype=float),
            latest["sign_probability"].to_numpy(dtype=float),
            latest["constant_delta_prediction"].to_numpy(dtype=float),
            latest["constant_sign_probability"].to_numpy(dtype=float),
            latest["last_known_delta_prediction"].to_numpy(dtype=float),
        )
        row.update({f"latest_{name}": value for name, value in latest_metrics.items()})
        row["latest_rows"] = int(len(latest))
        row["latest_coverage"] = float(
            latest["delta_prediction"].notna().mean()
        ) if len(latest) else np.nan
        row["aggregate_learnability_gate"] = bool(
            row.get("model_regression_mae", np.inf)
            < row.get("constant_regression_mae", -np.inf)
            and row.get("model_regression_rank_ic", -np.inf) > 0.10
            and row.get("model_sign_auc", -np.inf) > 0.55
        )
        row["latest_fold_gate"] = bool(
            row.get("latest_model_regression_mae", np.inf)
            <= row.get("latest_constant_regression_mae", -np.inf)
            and row.get("latest_model_regression_rank_ic", -np.inf) > 0.0
            and row.get("latest_model_sign_auc", -np.inf) > 0.50
            and row["latest_coverage"] == 1.0
        )
        rows.append(row)
    return pd.DataFrame(rows)


def plan(
    context_source: Path,
    label_source: Path,
    output: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    manifests, hashes = _source_hashes(context_source, label_source)
    return {
        "action": "PLAN_ONLY_NO_TRAINING_OR_MATERIALIZATION",
        "schema": SCHEMA,
        "context_source": str(context_source),
        "label_source": str(label_source),
        "output": str(output),
        "source_sha256": hashes,
        "features": {
            architecture: list(
                _features(manifests["context"], architecture)
            )
            for architecture in ("book", "band")
        },
        "tasks": {
            "book": {
                "fraction": PRIMARY_BOOK_FRACTION,
                "horizons": [PRIMARY_HORIZON, SENSITIVITY_HORIZON],
            },
            "band": {
                "bands": ["B0", "B1", "B2", "B3", "B4"],
                "horizons": [PRIMARY_HORIZON, SENSITIVITY_HORIZON],
            },
        },
        "targets": [target.name for target in TARGETS],
        "fixed_catboost_geometry": dict(CATBOOST_GEOMETRY),
        "fold_contract": {
            "kind": "chronological expanding",
            "minimum_history_days": int(args.min_train_days),
            "validation_days": int(args.validation_days),
            "availability": "actual after_target_available_utc and combined label availability must both be strictly before validation start",
        },
        "scope": "component learnability only; no score change, admission, action layer, PnL or portfolio claim",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    context_source = Path(args.context_source)
    label_source = Path(args.label_source)
    output = Path(args.output_dir)
    if args.plan_only:
        return plan(context_source, label_source, output, args)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    manifests, hashes = _source_hashes(context_source, label_source)
    prediction_parts: list[pd.DataFrame] = []
    metric_parts: list[pd.DataFrame] = []
    population_rows: dict[str, int] = {}
    for architecture in ("book", "band"):
        features = _features(manifests["context"], architecture)
        key = _key(architecture)
        context_name = (
            "global_book_context.parquet"
            if architecture == "book"
            else "global_ev_band_context.parquet"
        )
        label_name = (
            "global_book_transition_labels.parquet"
            if architecture == "book"
            else "global_ev_band_transition_labels.parquet"
        )
        context = pd.read_parquet(context_source / context_name)
        labels = pd.read_parquet(label_source / label_name)
        population = _prepare_population(
            architecture=architecture,
            context=context,
            labels=labels,
            features=features,
        )
        if architecture == "book":
            population = population.loc[
                np.isclose(
                    population["book_fraction"],
                    PRIMARY_BOOK_FRACTION,
                )
            ].copy()
        population = population.loc[
            population["horizon_hours"].isin(
                (PRIMARY_HORIZON, SENSITIVITY_HORIZON)
            )
        ].copy()
        population_rows[architecture] = int(len(population))
        folds = build_expanding_folds(
            population,
            min_train_days=int(args.min_train_days),
            validation_days=int(args.validation_days),
        )
        for horizon in (PRIMARY_HORIZON, SENSITIVITY_HORIZON):
            horizon_frame = population.loc[
                population["horizon_hours"].eq(horizon)
            ].copy()
            for target in TARGETS:
                predictions, metrics = _fit_target(
                    horizon_frame,
                    architecture=architecture,
                    target=target,
                    features=features,
                    folds=folds,
                    min_train_rows=int(args.min_train_rows),
                    fit_budget_rows=int(args.fit_budget_rows),
                    random_state=int(args.random_state),
                    threads=int(args.threads),
                )
                prediction_parts.append(predictions)
                metric_parts.append(metrics)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    per_fold = pd.concat(metric_parts, ignore_index=True)
    aggregate = _aggregate(predictions)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.")
    )
    predictions.to_parquet(
        temporary / "oof_head_predictions.parquet",
        index=False,
        compression="zstd",
    )
    per_fold.to_parquet(
        temporary / "per_fold_metrics.parquet",
        index=False,
        compression="zstd",
    )
    aggregate.to_parquet(
        temporary / "aggregate_metrics.parquet",
        index=False,
        compression="zstd",
    )
    manifest = {
        "schema": SCHEMA,
        "status": "IMMUTABLE_COMPONENT_LEARNABILITY_NOT_PROMOTION_ELIGIBLE",
        "promotion_eligible": False,
        "source_artifacts_sha256": hashes,
        "source_panel_identity_sha256": manifests["context"].get(
            "source_panel_identity_sha256"
        ),
        "feature_columns": {
            architecture: list(
                _features(manifests["context"], architecture)
            )
            for architecture in ("book", "band")
        },
        "tasks": {
            "book": {
                "fraction": PRIMARY_BOOK_FRACTION,
                "horizons": [PRIMARY_HORIZON, SENSITIVITY_HORIZON],
            },
            "band": {
                "bands": ["B0", "B1", "B2", "B3", "B4"],
                "horizons": [PRIMARY_HORIZON, SENSITIVITY_HORIZON],
            },
        },
        "targets": [
            {
                "name": target.name,
                "book_column": target.book_column,
                "band_column": target.band_column,
                "sign_label": "delta > 0",
            }
            for target in TARGETS
        ],
        "fixed_catboost_geometry": dict(CATBOOST_GEOMETRY),
        "fit_budget": {
            "minimum_prior_resolved_rows": int(args.min_train_rows),
            "maximum_fit_rows_per_target_fold": int(args.fit_budget_rows),
            "threads": int(args.threads),
            "random_state": int(args.random_state),
            "feature_selection": "disabled",
            "hyperparameter_optimization": "disabled",
        },
        "contracts": {
            "global_book": "H12 10% primary and H3 sensitivity are separate fixed tasks; fraction and horizon are not features",
            "global_band": "one shared B0-B4 context architecture with causal band ordinal; horizon is not a feature",
            "training_availability": "every fitted target has actual after_target_available_utc and combined label availability strictly before validation start",
            "baselines": "fold-local resolved constant and last-known target (per band for the band architecture)",
            "metrics": "OOF and latest-fold MAE, rank IC, sign AUC/AP/Brier/ECE, coverage and explicit gates",
            "scope": "no ranking change, admission, timing/MAE/target-price/wait action, PnL or portfolio replay",
        },
        "rows": {
            "population": population_rows,
            "oof_predictions": int(len(predictions)),
        },
        "outputs_sha256": {
            path.name: sha256(path)
            for path in sorted(temporary.glob("*.parquet"))
        },
        "checksum_convention": "manifest.json is verified by detached manifest.sha256",
    }
    (temporary / "manifest.json").write_text(
        json.dumps(_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (temporary / "manifest.sha256").write_text(
        f"{sha256(temporary / 'manifest.json')}  manifest.json\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return {
        "output": str(output),
        "population_rows": population_rows,
        "oof_predictions": int(len(predictions)),
        "aggregate_rows": int(len(aggregate)),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--context-source", type=Path, default=CONTEXT_SOURCE)
    result.add_argument("--label-source", type=Path, default=LABEL_SOURCE)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--min-train-days", type=int, default=28)
    result.add_argument("--validation-days", type=int, default=14)
    result.add_argument("--min-train-rows", type=int, default=240)
    result.add_argument("--fit-budget-rows", type=int, default=20_000)
    result.add_argument("--threads", type=int, default=1)
    result.add_argument("--random-state", type=int, default=20260729)
    result.add_argument("--plan-only", action="store_true")
    return result


def main() -> None:
    print(json.dumps(_safe(run(parser().parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
