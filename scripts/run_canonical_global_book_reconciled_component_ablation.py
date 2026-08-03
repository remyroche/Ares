#!/usr/bin/env python3
"""Compare a global residual head with reconciled B1--B4 book components.

The primary task is the exact H12 pooled-global 10% mapped-EV book.  B0 is
structurally zero and is audited, not fitted.  B1--B4 targets are selected-book
conversion-residual contributions whose sum exactly reconciles to the global
book conversion residual.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

try:
    from scripts.run_canonical_economic_conversion_transition_head_ablation import (
        _classification_metrics,
        _safe,
        build_expanding_folds,
        sha256,
    )
    from scripts.run_canonical_global_book_conversion_head_ablation import (
        CONTEXT_SCHEMA,
        LABEL_SCHEMA,
        PRIMARY_BOOK_FRACTION,
        PRIMARY_HORIZON,
        SENSITIVITY_HORIZON,
        _artifact_manifest,
        _features,
        _source_hashes,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from run_canonical_economic_conversion_transition_head_ablation import (
        _classification_metrics,
        _safe,
        build_expanding_folds,
        sha256,
    )
    from run_canonical_global_book_conversion_head_ablation import (
        CONTEXT_SCHEMA,
        LABEL_SCHEMA,
        PRIMARY_BOOK_FRACTION,
        PRIMARY_HORIZON,
        SENSITIVITY_HORIZON,
        _artifact_manifest,
        _features,
        _source_hashes,
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
DIRECT_SOURCE = (
    ROOT
    / "data_perp/artifacts/"
    "canonical_global_book_conversion_head_ablation_20260729_v1"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data_perp/artifacts/"
    "canonical_global_book_reconciled_component_ablation_20260729_v1"
)
SCHEMA = "canonical_global_book_reconciled_component_ablation_v1"
BANDS = ("B1", "B2", "B3", "B4")
HORIZONS = (PRIMARY_HORIZON, SENSITIVITY_HORIZON)
GEOMETRY: Mapping[str, Any] = {
    "iterations": 64,
    "depth": 3,
    "learning_rate": 0.05,
    "l2_leaf_reg": 16.0,
    "random_strength": 0.0,
    "bootstrap_type": "No",
    "allow_writing_files": False,
    "verbose": False,
}


def _regressor(*, seed: int, threads: int):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        loss_function="MAE",
        random_seed=int(seed),
        thread_count=int(threads),
        **GEOMETRY,
    )


def _classifier(*, seed: int, threads: int):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        loss_function="Logloss",
        random_seed=int(seed),
        thread_count=int(threads),
        **GEOMETRY,
    )


def _numeric_metrics(
    y: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    valid = np.isfinite(y) & np.isfinite(prediction)
    if not valid.any():
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "rank_ic": np.nan,
        }
    actual = y[valid]
    predicted = prediction[valid]
    rank_ic = np.nan
    if (
        len(actual) > 1
        and np.unique(actual).size > 1
        and np.unique(predicted).size > 1
    ):
        rank_ic = float(
            pd.Series(actual).corr(
                pd.Series(predicted), method="spearman"
            )
        )
    return {
        "mae": float(np.abs(actual - predicted).mean()),
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "rank_ic": rank_ic,
    }


def _causal_persistence(
    population: pd.DataFrame, evaluation: pd.DataFrame
) -> np.ndarray:
    """Latest target whose actual availability precedes each decision."""

    resolved = population.loc[
        population["target_valid"].astype(bool)
        & population["label_available_utc"].notna()
    ].sort_values(
        ["label_available_utc", "cohort_anchor_utc"], kind="stable"
    )
    if resolved.empty:
        return np.full(len(evaluation), np.nan)
    availability = resolved["label_available_utc"].array.asi8
    target = resolved["target_delta"].to_numpy(dtype=float)
    anchors = evaluation["cohort_anchor_utc"].array.asi8
    positions = np.searchsorted(availability, anchors, side="left") - 1
    result = np.full(len(evaluation), np.nan)
    valid = positions >= 0
    result[valid] = target[positions[valid]]
    return result


def _source_contract(
    context_source: Path,
    label_source: Path,
    direct_source: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    manifests, hashes = _source_hashes(context_source, label_source)
    direct_manifest, direct_hashes = _artifact_manifest(
        direct_source,
        "canonical_global_book_conversion_head_ablation_v1",
    )
    direct_predictions = direct_source / "oof_head_predictions.parquet"
    if not direct_predictions.is_file():
        raise FileNotFoundError(
            "direct global-book comparator lacks OOF predictions"
        )
    if (
        direct_manifest.get("source_panel_identity_sha256")
        != manifests["context"].get("source_panel_identity_sha256")
    ):
        raise ValueError(
            "direct comparator and component sources do not share panel identity"
        )
    return (
        {**manifests, "direct": direct_manifest},
        {
            **hashes,
            **direct_hashes,
            str(direct_predictions): sha256(direct_predictions),
        },
    )


def _prepare_global(
    labels: pd.DataFrame,
    context: pd.DataFrame,
    features: Iterable[str],
) -> pd.DataFrame:
    features = tuple(features)
    key = ["cohort_anchor_utc", "horizon_hours", "book_fraction"]
    label_columns = [
        *key,
        "horizon_role",
        "before_global_hour_complete_flag",
        "after_global_hour_complete_flag",
        "before_population_candidate_support",
        "after_population_candidate_support",
        "before_selected_candidate_support",
        "after_selected_candidate_support",
        "before_band_contribution_complete_flag",
        "after_band_contribution_complete_flag",
        "before_target_available_utc",
        "after_target_available_utc",
        "delta_mean_conversion_residual",
        "delta_direct_mean_net",
        "delta_mapped_score_mean",
        *(
            column
            for band in ("B0", *BANDS)
            for column in (
                f"before_band_{band}_conversion_residual_contribution",
                f"after_band_{band}_conversion_residual_contribution",
            )
        ),
    ]
    required_context = [
        *key,
        "label_audit__before_global_hour_complete_flag",
        "label_audit__after_global_hour_complete_flag",
        "label_audit__before_target_available_utc",
        "label_audit__after_target_available_utc",
        *features,
    ]
    missing_labels = sorted(set(label_columns).difference(labels.columns))
    missing_context = sorted(set(required_context).difference(context.columns))
    if missing_labels or missing_context:
        raise ValueError(
            "reconciled component source lacks columns: "
            f"labels={missing_labels}, context={missing_context}"
        )
    labels = labels.loc[:, label_columns].copy()
    context = context.loc[:, required_context].copy()
    for frame in (labels, context):
        frame["cohort_anchor_utc"] = pd.to_datetime(
            frame["cohort_anchor_utc"], utc=True, errors="raise"
        )
        frame["horizon_hours"] = pd.to_numeric(
            frame["horizon_hours"], errors="raise"
        ).astype(np.int8)
        frame["book_fraction"] = pd.to_numeric(
            frame["book_fraction"], errors="raise"
        )
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
    joined = labels.merge(
        context, on=key, how="left", validate="one_to_one"
    )
    for column in (
        "before_global_hour_complete_flag",
        "after_global_hour_complete_flag",
        "before_target_available_utc",
        "after_target_available_utc",
    ):
        audit = joined[f"label_audit__{column}"]
        if "available_utc" in column:
            parity = joined[column].eq(audit) | (
                joined[column].isna() & audit.isna()
            )
        else:
            parity = joined[column].astype(bool).eq(audit.astype(bool))
        if not parity.all():
            raise ValueError(
                f"global context label-audit parity failed: {column}"
            )
    joined = joined.loc[
        np.isclose(joined["book_fraction"], PRIMARY_BOOK_FRACTION)
        & joined["horizon_hours"].isin(HORIZONS)
    ].copy()
    for column in features:
        joined[column] = pd.to_numeric(joined[column], errors="coerce")
        if np.isinf(
            joined[column].to_numpy(dtype=float, na_value=np.nan)
        ).any():
            raise ValueError(f"global feature contains infinity: {column}")
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
        & joined["before_band_contribution_complete_flag"].astype(bool)
        & joined["after_band_contribution_complete_flag"].astype(bool)
        & pd.to_numeric(
            joined["before_population_candidate_support"], errors="coerce"
        ).gt(0)
        & pd.to_numeric(
            joined["after_population_candidate_support"], errors="coerce"
        ).gt(0)
        & pd.to_numeric(
            joined["before_selected_candidate_support"], errors="coerce"
        ).gt(0)
        & pd.to_numeric(
            joined["after_selected_candidate_support"], errors="coerce"
        ).gt(0)
        & joined["before_target_available_utc"].notna()
        & joined["after_target_available_utc"].notna()
    )
    joined["target_delta"] = pd.to_numeric(
        joined["delta_mean_conversion_residual"], errors="coerce"
    )
    joined["target_valid"] = complete & np.isfinite(
        joined["target_delta"]
    )
    b0 = (
        joined["after_band_B0_conversion_residual_contribution"]
        - joined["before_band_B0_conversion_residual_contribution"]
    )
    if not np.allclose(
        b0.loc[joined["target_valid"]], 0.0, atol=1e-15, rtol=0.0
    ):
        raise ValueError("B0 is no longer a structural-zero contribution")
    return joined.sort_values(
        ["horizon_hours", "cohort_anchor_utc"], kind="stable"
    ).reset_index(drop=True)


def _prepare_component(
    global_population: pd.DataFrame,
    band_context: pd.DataFrame,
    *,
    band: str,
    book_features: Iterable[str],
    band_features: Iterable[str],
) -> pd.DataFrame:
    book_features = tuple(book_features)
    band_features = tuple(band_features)
    context = band_context.loc[
        band_context["global_common_ev_band"].eq(band),
        [
            "cohort_anchor_utc",
            "horizon_hours",
            "global_common_ev_band",
            *band_features,
        ],
    ].copy()
    context["cohort_anchor_utc"] = pd.to_datetime(
        context["cohort_anchor_utc"], utc=True, errors="raise"
    )
    context["horizon_hours"] = pd.to_numeric(
        context["horizon_hours"], errors="raise"
    ).astype(np.int8)
    if context.duplicated(
        ["cohort_anchor_utc", "horizon_hours"]
    ).any():
        raise ValueError(f"{band} context identity is not one-to-one")
    result = global_population.merge(
        context,
        on=["cohort_anchor_utc", "horizon_hours"],
        how="left",
        validate="one_to_one",
    )
    result["band_context_available"] = ~result.loc[
        :, list(band_features)
    ].isna().all(axis=1)
    for column in band_features:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if np.isinf(
            result[column].to_numpy(dtype=float, na_value=np.nan)
        ).any():
            raise ValueError(f"{band} feature contains infinity: {column}")
    result["target_delta"] = (
        pd.to_numeric(
            result[
                f"after_band_{band}_conversion_residual_contribution"
            ],
            errors="coerce",
        )
        - pd.to_numeric(
            result[
                f"before_band_{band}_conversion_residual_contribution"
            ],
            errors="coerce",
        )
    )
    result["target_valid"] = (
        result["target_valid"].astype(bool)
        & result["band_context_available"].astype(bool)
        & np.isfinite(result["target_delta"])
    )
    result["component_band"] = band
    return result


def _fit_series(
    population: pd.DataFrame,
    *,
    model_name: str,
    features: Iterable[str],
    folds: Iterable[Mapping[str, Any]],
    min_train_rows: int,
    threads: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = tuple(features)
    predictions: list[pd.DataFrame] = []
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
        train = population.loc[
            population["target_valid"].astype(bool)
            & population["label_available_utc"].lt(start)
            & population["after_target_available_utc"].lt(start)
        ].copy()
        valid_eval = evaluation["target_valid"].to_numpy(bool)
        y_train = train["target_delta"].to_numpy(dtype=float)
        constant = float(y_train.mean()) if len(y_train) else 0.0
        constant_probability = (
            float((y_train > 0).mean()) if len(y_train) else 0.0
        )
        result = evaluation.loc[
            :,
            [
                "cohort_anchor_utc",
                "horizon_hours",
                "book_fraction",
                "target_delta",
                "target_valid",
                "delta_direct_mean_net",
                "delta_mapped_score_mean",
                "label_available_utc",
                "after_target_available_utc",
            ],
        ].copy()
        result["model_name"] = model_name
        result["fold_id"] = int(fold["fold_id"])
        result["validation_start_utc"] = start
        result["validation_end_utc"] = end
        result["delta_prediction"] = np.nan
        result["sign_probability"] = np.nan
        result["zero_delta_prediction"] = np.where(valid_eval, 0.0, np.nan)
        result["constant_delta_prediction"] = np.where(
            valid_eval, constant, np.nan
        )
        result["constant_sign_probability"] = np.where(
            valid_eval, constant_probability, np.nan
        )
        persistence = _causal_persistence(population, evaluation)
        result["causal_persistence_prediction"] = np.where(
            valid_eval, persistence, np.nan
        )
        status = "constant_fallback_insufficient_prior_resolved_rows"
        if valid_eval.any() and len(train) >= int(min_train_rows):
            regressor = _regressor(
                seed=(
                    int(random_state)
                    + 10_000 * int(fold["fold_id"])
                    + len(model_name)
                ),
                threads=int(threads),
            ).fit(
                train.loc[:, list(features)],
                y_train,
            )
            result.loc[valid_eval, "delta_prediction"] = np.asarray(
                regressor.predict(
                    evaluation.loc[valid_eval, list(features)]
                ),
                dtype=float,
            )
            sign_train = (y_train > 0).astype(np.int8)
            if np.unique(sign_train).size == 2:
                classifier = _classifier(
                    seed=(
                        int(random_state)
                        + 20_000 * int(fold["fold_id"])
                        + len(model_name)
                    ),
                    threads=int(threads),
                ).fit(train.loc[:, list(features)], sign_train)
                result.loc[valid_eval, "sign_probability"] = np.asarray(
                    classifier.predict_proba(
                        evaluation.loc[valid_eval, list(features)]
                    )[:, 1],
                    dtype=float,
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
        result["training_max_after_target_available_utc"] = (
            train["after_target_available_utc"].max()
            if len(train)
            else pd.NaT
        )
        predictions.append(result)
        valid = result.loc[result["target_valid"].astype(bool)]
        metric = {
            "model_name": model_name,
            "horizon_hours": int(
                evaluation["horizon_hours"].iloc[0]
            ),
            **dict(fold),
            "evaluation_rows": int(len(evaluation)),
            "valid_rows": int(len(valid)),
            "training_rows": int(len(train)),
            "fit_status": status,
        }
        metric.update(_metric_bundle(valid))
        metric_rows.append(metric)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(metric_rows)


def _metric_bundle(frame: pd.DataFrame) -> dict[str, Any]:
    y = frame["target_delta"].to_numpy(dtype=float)
    result: dict[str, Any] = {}
    for prefix, column in (
        ("model", "delta_prediction"),
        ("zero", "zero_delta_prediction"),
        ("constant", "constant_delta_prediction"),
        ("causal_persistence", "causal_persistence_prediction"),
    ):
        metrics = _numeric_metrics(
            y, frame[column].to_numpy(dtype=float)
        )
        result.update(
            {f"{prefix}_regression_{name}": value for name, value in metrics.items()}
        )
    sign = (y > 0).astype(np.int8)
    for prefix, column in (
        ("model", "sign_probability"),
        ("constant", "constant_sign_probability"),
    ):
        metrics = _classification_metrics(
            sign, frame[column].to_numpy(dtype=float)
        )
        result.update(
            {
                f"{prefix}_sign_{name}": value
                for name, value in metrics.items()
                if name != "rows"
            }
        )
    if len(frame):
        ranked = frame.assign(
            __rank=pd.qcut(
                frame["delta_prediction"].rank(method="first"),
                q=min(5, len(frame)),
                labels=False,
                duplicates="drop",
            )
        )
        low = ranked.loc[ranked["__rank"].eq(ranked["__rank"].min())]
        high = ranked.loc[ranked["__rank"].eq(ranked["__rank"].max())]
        result["top_bottom_target_spread"] = float(
            high["target_delta"].mean() - low["target_delta"].mean()
        )
        result["top_bottom_direct_net_spread"] = float(
            high["delta_direct_mean_net"].mean()
            - low["delta_direct_mean_net"].mean()
        )
    return result


def _component_sum(predictions: pd.DataFrame) -> pd.DataFrame:
    key = [
        "cohort_anchor_utc",
        "horizon_hours",
        "book_fraction",
        "fold_id",
        "validation_start_utc",
        "validation_end_utc",
    ]
    summed = (
        predictions.groupby(key, sort=True, observed=True)
        .agg(
            target_delta=("target_delta", "sum"),
            target_valid=("target_valid", "all"),
            delta_prediction=("delta_prediction", "sum"),
            zero_delta_prediction=("zero_delta_prediction", "sum"),
            constant_delta_prediction=("constant_delta_prediction", "sum"),
            causal_persistence_prediction=(
                "causal_persistence_prediction",
                "sum",
            ),
            delta_direct_mean_net=("delta_direct_mean_net", "first"),
            delta_mapped_score_mean=("delta_mapped_score_mean", "first"),
            sign_probability=("sign_probability", "mean"),
            constant_sign_probability=(
                "constant_sign_probability",
                "mean",
            ),
            label_available_utc=("label_available_utc", "first"),
            after_target_available_utc=(
                "after_target_available_utc",
                "first",
            ),
            training_rows=("training_rows", "min"),
            training_max_after_target_available_utc=(
                "training_max_after_target_available_utc",
                "max",
            ),
        )
        .reset_index()
    )
    # Independent component sign probabilities do not define the sign
    # probability of their sum.  Leave this unscored until an explicit
    # reconciled sign head is cross-fitted.
    summed["sign_probability"] = np.nan
    summed["constant_sign_probability"] = np.nan
    summed["model_name"] = "reconciled_component_sum_B1_B4"
    summed["fit_status"] = "sum_of_independent_component_predictions"
    return summed


def _aggregate(
    predictions: pd.DataFrame,
    *,
    model_column: str = "model_name",
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (model, horizon), group in predictions.groupby(
        [model_column, "horizon_hours"], sort=True
    ):
        valid = group.loc[group["target_valid"].astype(bool)]
        record: dict[str, Any] = {
            "model_name": model,
            "horizon_hours": int(horizon),
            "oof_rows": int(len(valid)),
            "folds": int(valid["fold_id"].nunique()),
        }
        record.update(_metric_bundle(valid))
        full_fold_ids = []
        for fold_id, fold in valid.groupby("fold_id", sort=True):
            duration = (
                fold["validation_end_utc"].iloc[0]
                - fold["validation_start_utc"].iloc[0]
            )
            if duration >= pd.Timedelta(days=14):
                full_fold_ids.append(int(fold_id))
        record["latest_full_fold_id"] = (
            max(full_fold_ids) if full_fold_ids else -1
        )
        if full_fold_ids:
            latest = valid.loc[
                valid["fold_id"].eq(max(full_fold_ids))
            ]
            record.update(
                {
                    f"latest_full_{name}": value
                    for name, value in _metric_bundle(latest).items()
                }
            )
            record["latest_full_rows"] = int(len(latest))
        development = valid.loc[valid["fold_id"].isin(full_fold_ids)]
        per_fold_positive_ic = 0
        per_fold_positive_spread = 0
        for _, fold in development.groupby("fold_id", sort=True):
            metrics = _metric_bundle(fold)
            per_fold_positive_ic += int(
                metrics.get("model_regression_rank_ic", -np.inf) > 0
            )
            per_fold_positive_spread += int(
                metrics.get("top_bottom_target_spread", -np.inf) > 0
            )
        record["development_full_folds"] = int(len(full_fold_ids))
        record["development_positive_ic_folds"] = int(
            per_fold_positive_ic
        )
        record["development_positive_spread_folds"] = int(
            per_fold_positive_spread
        )
        records.append(record)
    return pd.DataFrame(records)


def plan(
    context_source: Path,
    label_source: Path,
    direct_source: Path,
    output: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    manifests, hashes = _source_contract(
        context_source, label_source, direct_source
    )
    book_features = _features(manifests["context"], "book")
    band_features = tuple(
        column
        for column in _features(manifests["context"], "band")
        if column != "context__global_common_ev_band_ordinal"
    )
    return {
        "action": "PLAN_ONLY_NO_TRAINING_OR_MATERIALIZATION",
        "schema": SCHEMA,
        "output": str(output),
        "source_sha256": hashes,
        "primary_task": {
            "horizon_hours": PRIMARY_HORIZON,
            "book_fraction": PRIMARY_BOOK_FRACTION,
            "target": "delta_mean_conversion_residual",
        },
        "sensitivity_horizon": SENSITIVITY_HORIZON,
        "components": list(BANDS),
        "structural_zero_component": "B0",
        "features": {
            "global_direct": list(book_features),
            "component": [*book_features, *band_features],
        },
        "fixed_geometry": dict(GEOMETRY),
        "folds": {
            "minimum_history_days": int(args.min_train_days),
            "validation_days": int(args.validation_days),
            "actual_target_availability_purge": True,
        },
        "baselines": [
            "zero correction",
            "fold-local resolved mean",
            "causal latest-resolved persistence",
            "same-geometry global direct residual head",
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    context_source = Path(args.context_source)
    label_source = Path(args.label_source)
    direct_source = Path(args.direct_source)
    output = Path(args.output_dir)
    if args.plan_only:
        return plan(
            context_source, label_source, direct_source, output, args
        )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    manifests, hashes = _source_contract(
        context_source, label_source, direct_source
    )
    book_features = _features(manifests["context"], "book")
    band_features = tuple(
        column
        for column in _features(manifests["context"], "band")
        if column != "context__global_common_ev_band_ordinal"
    )
    labels = pd.read_parquet(
        label_source / "global_book_transition_labels.parquet"
    )
    book_context = pd.read_parquet(
        context_source / "global_book_context.parquet"
    )
    band_context = pd.read_parquet(
        context_source / "global_ev_band_context.parquet"
    )
    global_population = _prepare_global(
        labels, book_context, book_features
    )
    predictions: list[pd.DataFrame] = []
    per_fold: list[pd.DataFrame] = []
    component_predictions: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        horizon_global = global_population.loc[
            global_population["horizon_hours"].eq(horizon)
        ].copy()
        folds = build_expanding_folds(
            horizon_global,
            min_train_days=int(args.min_train_days),
            validation_days=int(args.validation_days),
        )
        direct_prediction, direct_metrics = _fit_series(
            horizon_global,
            model_name="compact_global_direct_residual",
            features=book_features,
            folds=folds,
            min_train_rows=int(args.min_train_rows),
            threads=int(args.threads),
            random_state=int(args.random_state),
        )
        predictions.append(direct_prediction)
        per_fold.append(direct_metrics)
        for band in BANDS:
            component = _prepare_component(
                horizon_global,
                band_context,
                band=band,
                book_features=book_features,
                band_features=band_features,
            )
            component_prediction, component_metrics = _fit_series(
                component,
                model_name=f"component_{band}",
                features=(*book_features, *band_features),
                folds=folds,
                min_train_rows=int(args.min_train_rows),
                threads=int(args.threads),
                random_state=int(args.random_state),
            )
            component_predictions.append(component_prediction)
            predictions.append(component_prediction)
            per_fold.append(component_metrics)
    component_prediction_table = pd.concat(
        component_predictions, ignore_index=True
    )
    component_sum = _component_sum(component_prediction_table)
    predictions.append(component_sum)
    all_predictions = pd.concat(predictions, ignore_index=True)
    per_fold_table = pd.concat(per_fold, ignore_index=True)
    aggregate = _aggregate(all_predictions)
    reconciliation = component_sum.merge(
        global_population.loc[
            :,
            [
                "cohort_anchor_utc",
                "horizon_hours",
                "book_fraction",
                "delta_mean_conversion_residual",
            ],
        ],
        on=["cohort_anchor_utc", "horizon_hours", "book_fraction"],
        how="left",
        validate="one_to_one",
    )
    reconciliation["absolute_label_reconciliation_error"] = (
        reconciliation["target_delta"]
        - reconciliation["delta_mean_conversion_residual"]
    ).abs()
    max_reconciliation_error = float(
        reconciliation["absolute_label_reconciliation_error"].max()
    )
    if max_reconciliation_error > 1e-12:
        raise ValueError(
            "B1-B4 component labels do not reconcile to global residual"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.")
    )
    all_predictions.to_parquet(
        temporary / "oof_predictions.parquet",
        index=False,
        compression="zstd",
    )
    per_fold_table.to_parquet(
        temporary / "per_fold_metrics.parquet",
        index=False,
        compression="zstd",
    )
    aggregate.to_parquet(
        temporary / "aggregate_metrics.parquet",
        index=False,
        compression="zstd",
    )
    reconciliation.loc[
        :,
        [
            "cohort_anchor_utc",
            "horizon_hours",
            "book_fraction",
            "fold_id",
            "target_delta",
            "delta_mean_conversion_residual",
            "absolute_label_reconciliation_error",
        ],
    ].to_parquet(
        temporary / "component_reconciliation_audit.parquet",
        index=False,
        compression="zstd",
    )
    manifest = {
        "schema": SCHEMA,
        "status": "IMMUTABLE_RECONCILED_COMPONENT_LEARNABILITY_NOT_PROMOTION_ELIGIBLE",
        "promotion_eligible": False,
        "source_artifacts_sha256": hashes,
        "source_panel_identity_sha256": manifests["context"].get(
            "source_panel_identity_sha256"
        ),
        "primary_task": {
            "horizon_hours": PRIMARY_HORIZON,
            "book_fraction": PRIMARY_BOOK_FRACTION,
            "target": "delta_mean_conversion_residual",
        },
        "sensitivity_horizon": SENSITIVITY_HORIZON,
        "components": list(BANDS),
        "structural_zero_component": "B0",
        "global_feature_columns": list(book_features),
        "component_band_feature_columns": list(band_features),
        "fixed_geometry": dict(GEOMETRY),
        "fit_contract": {
            "minimum_prior_resolved_rows": int(args.min_train_rows),
            "minimum_history_days": int(args.min_train_days),
            "validation_days": int(args.validation_days),
            "threads": int(args.threads),
            "random_state": int(args.random_state),
            "feature_selection": "disabled",
            "hyperparameter_optimization": "disabled",
        },
        "contracts": {
            "target": "global realized net = mapped EV + conversion residual; learn only the residual correction",
            "component_reconciliation": "B0 is structural zero; B1-B4 selected-book residual contributions sum to the global book residual target",
            "availability": "every fit uses only targets whose actual before/after availability is strictly before validation start",
            "selection": "one pooled global H-window book ranked by causal mapped_direct_net; no timestamp/side/asset quota",
            "baselines": "zero correction, fold-local resolved mean, causal latest-resolved persistence, and same-geometry global direct head",
            "scope": "component learnability only; no admission, action layer, policy PnL or portfolio claim",
        },
        "label_reconciliation_max_absolute_error": max_reconciliation_error,
        "rows": {
            "global_population": int(len(global_population)),
            "oof_predictions": int(len(all_predictions)),
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
        "oof_predictions": int(len(all_predictions)),
        "label_reconciliation_max_absolute_error": max_reconciliation_error,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--context-source", type=Path, default=CONTEXT_SOURCE)
    result.add_argument("--label-source", type=Path, default=LABEL_SOURCE)
    result.add_argument("--direct-source", type=Path, default=DIRECT_SOURCE)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--min-train-days", type=int, default=28)
    result.add_argument("--validation-days", type=int, default=14)
    result.add_argument("--min-train-rows", type=int, default=500)
    result.add_argument("--threads", type=int, default=1)
    result.add_argument("--random-state", type=int, default=20260729)
    result.add_argument("--plan-only", action="store_true")
    return result


def main() -> None:
    print(json.dumps(_safe(run(parser().parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
