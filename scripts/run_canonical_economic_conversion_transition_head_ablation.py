#!/usr/bin/env python3
"""Measure learnability of canonical economic-conversion transition heads.

This is intentionally a component learnability experiment.  It uses the
immutable anchor-time context artifact and the separately immutable transition
labels, has no feature search or HPO, and does not produce a trading score,
admission rule, portfolio replay, or PnL claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONTEXT_SOURCE = ROOT / "data_perp/artifacts/canonical_economic_conversion_transition_context_20260729_v1"
LABEL_SOURCE = ROOT / "data_perp/artifacts/canonical_economic_conversion_transition_labels_20260729_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/canonical_economic_conversion_transition_head_ablation_20260729_v1"

SCHEMA = "canonical_economic_conversion_transition_head_ablation_v1"
CONTEXT_SCHEMA = "canonical_economic_conversion_transition_context_v1"
LABEL_SCHEMA = "canonical_economic_conversion_transition_labels_v1"
COHORT_KEY = ("cohort_anchor_utc", "side_name", "frozen_base_score_decile")
PRIMARY_HORIZON = 12
SENSITIVITY_HORIZON = 3
HORIZONS = (PRIMARY_HORIZON, SENSITIVITY_HORIZON)

# This geometry is deliberately fixed across every component and horizon.  It
# is compact enough for the cohort table and has no validation-set tuning.
CATBOOST_GEOMETRY: Mapping[str, Any] = {
    "iterations": 96,
    "depth": 5,
    "learning_rate": 0.05,
    "l2_leaf_reg": 8.0,
    "random_strength": 0.0,
    "bootstrap_type": "No",
    "allow_writing_files": False,
    "verbose": False,
}


@dataclass(frozen=True)
class TargetSpec:
    name: str
    delta_column: str
    conditional_support_flags: tuple[str, ...] = ()


TARGETS = (
    TargetSpec("opportunity_probability_0bps", "delta_opportunity_probability_0bps"),
    TargetSpec(
        "favorable_payoff_robust_mean",
        "delta_conditional_favorable_net_robust_mean",
        (
            "before_favorable_net_missing_support_flag",
            "after_favorable_net_missing_support_flag",
        ),
    ),
    TargetSpec(
        "adverse_severity_robust_mean",
        "delta_conditional_adverse_loss_robust_mean",
        (
            "before_adverse_loss_missing_support_flag",
            "after_adverse_loss_missing_support_flag",
        ),
    ),
    TargetSpec("exit_mixture_net", "delta_exit_mixture_expected_net"),
    TargetSpec("direct_mean_net", "delta_direct_mean_net"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _artifact_manifest(root: Path, schema: str) -> tuple[dict[str, Any], dict[str, str]]:
    manifest_path = root / "manifest.json"
    sidecar_path = root / "manifest.sha256"
    if not manifest_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"immutable source manifest is incomplete: {root}")
    expected = sidecar_path.read_text(encoding="utf-8").strip().split(maxsplit=1)
    actual = sha256(manifest_path)
    if not expected or expected[0] != actual:
        raise ValueError(f"manifest checksum mismatch: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != schema:
        raise ValueError(f"unexpected immutable schema at {root}: {manifest.get('schema')!r}")
    return manifest, {str(manifest_path): actual, str(sidecar_path): sha256(sidecar_path)}


def _source_hashes(context_source: Path, label_source: Path) -> tuple[dict[str, Any], dict[str, str]]:
    context_manifest, context_hashes = _artifact_manifest(context_source, CONTEXT_SCHEMA)
    label_manifest, label_hashes = _artifact_manifest(label_source, LABEL_SCHEMA)
    context_path = context_source / "cohort_transition_context.parquet"
    label_path = label_source / "cohort_transition_labels.parquet"
    if not context_path.is_file() or not label_path.is_file():
        raise FileNotFoundError("immutable input lacks its material parquet")
    context_hash = sha256(context_path)
    label_hash = sha256(label_path)
    context_source_hashes = context_manifest.get("source_artifacts_sha256", {})
    if context_source_hashes.get(str(label_path)) != label_hash:
        raise ValueError("context artifact is not bound to the exact supplied transition-label parquet")
    return (
        {"context": context_manifest, "labels": label_manifest},
        {
            **context_hashes,
            **label_hashes,
            str(context_path): context_hash,
            str(label_path): label_hash,
        },
    )


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="raise")


def _context_features(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    values = tuple(map(str, manifest.get("context_feature_columns", [])))
    if not values:
        raise ValueError("context manifest has no explicit feature surface")
    forbidden = (
        "mapped_",
        "target",
        "label",
        "outcome",
        "execution_net",
        "execution_gross",
        "opportunity_",
        "exit",
        "mfe",
        "mae",
        "first_touch",
        "realized",
    )
    bad = [column for column in values if any(token in column.lower() for token in forbidden)]
    if bad:
        raise ValueError(f"context manifest permits prohibited model features: {bad}")
    if len(set(values)) != len(values):
        raise ValueError("context feature contract contains duplicate columns")
    return values


def _label_columns() -> tuple[str, ...]:
    return (
        *COHORT_KEY,
        "horizon_hours",
        "horizon_role",
        "before_global_hour_complete_flag",
        "after_global_hour_complete_flag",
        "before_candidate_support",
        "after_candidate_support",
        "before_target_available_utc",
        "after_target_available_utc",
        *(target.delta_column for target in TARGETS),
        *(flag for target in TARGETS for flag in target.conditional_support_flags),
    )


def _normalise_context(context: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    features = tuple(features)
    required = set(COHORT_KEY) | set(features)
    missing = sorted(required.difference(context.columns))
    if missing:
        raise ValueError(f"context table lacks contract columns: {missing}")
    result = context.loc[:, [*COHORT_KEY, *features]].copy()
    result["cohort_anchor_utc"] = _utc(result["cohort_anchor_utc"])
    result["side_name"] = result["side_name"].astype(str).str.lower()
    result["frozen_base_score_decile"] = pd.to_numeric(
        result["frozen_base_score_decile"], errors="raise"
    ).astype(np.int8)
    if not result["side_name"].isin(("long", "short")).all():
        raise ValueError("context contains non-canonical sides")
    if not result["frozen_base_score_decile"].between(0, 9).all():
        raise ValueError("context contains invalid score deciles")
    if result.duplicated(list(COHORT_KEY)).any():
        raise ValueError("context cohort identity is not one-to-one")
    for column in features:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if np.isinf(result[column].to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"context feature contains an infinity: {column}")
    return result.sort_values(list(COHORT_KEY), kind="stable").reset_index(drop=True)


def _normalise_labels(labels: pd.DataFrame) -> pd.DataFrame:
    required = set(_label_columns())
    missing = sorted(required.difference(labels.columns))
    if missing:
        raise ValueError(f"transition labels lack ablation columns: {missing}")
    result = labels.loc[:, list(_label_columns())].copy()
    result["cohort_anchor_utc"] = _utc(result["cohort_anchor_utc"])
    result["side_name"] = result["side_name"].astype(str).str.lower()
    result["frozen_base_score_decile"] = pd.to_numeric(
        result["frozen_base_score_decile"], errors="raise"
    ).astype(np.int8)
    result["horizon_hours"] = pd.to_numeric(result["horizon_hours"], errors="raise").astype(np.int8)
    result["before_target_available_utc"] = _utc(result["before_target_available_utc"])
    # A missing after target is an explicit unresolved/unsupported target, not
    # a timestamp to fill.  ``to_datetime`` preserves NaT while normalising UTC.
    result["after_target_available_utc"] = pd.to_datetime(
        result["after_target_available_utc"], utc=True, errors="coerce"
    )
    for target in TARGETS:
        result[target.delta_column] = pd.to_numeric(result[target.delta_column], errors="coerce")
    for flag in (flag for target in TARGETS for flag in target.conditional_support_flags):
        result[flag] = result[flag].astype(bool)
    key = [*COHORT_KEY, "horizon_hours"]
    if result.duplicated(key).any():
        raise ValueError("transition label identity is not one-to-one")
    if set(result["horizon_hours"].unique()).difference(HORIZONS):
        raise ValueError("transition labels contain an unsupported horizon")
    return result.sort_values(key, kind="stable").reset_index(drop=True)


def _label_available_utc(frame: pd.DataFrame) -> pd.Series:
    before = pd.to_datetime(frame["before_target_available_utc"], utc=True, errors="coerce")
    after = pd.to_datetime(frame["after_target_available_utc"], utc=True, errors="coerce")
    return pd.concat([before, after], axis=1).max(axis=1)


def _target_validity(frame: pd.DataFrame, target: TargetSpec) -> tuple[pd.Series, pd.Series]:
    complete = (
        frame["before_global_hour_complete_flag"].astype(bool)
        & frame["after_global_hour_complete_flag"].astype(bool)
        & pd.to_numeric(frame["before_candidate_support"], errors="coerce").gt(0)
        & pd.to_numeric(frame["after_candidate_support"], errors="coerce").gt(0)
        & frame["after_target_available_utc"].notna()
    )
    conditional = pd.Series(True, index=frame.index)
    for flag in target.conditional_support_flags:
        conditional &= ~frame[flag].astype(bool)
    finite_delta = np.isfinite(pd.to_numeric(frame[target.delta_column], errors="coerce"))
    valid = complete & conditional & finite_delta
    reason = pd.Series("valid", index=frame.index, dtype="object")
    reason.loc[~complete] = "incomplete_window_or_unresolved_after_target"
    reason.loc[complete & ~conditional] = "missing_conditional_support"
    reason.loc[complete & conditional & ~finite_delta] = "non_finite_delta"
    return valid, reason


def prepare_population(context: pd.DataFrame, labels: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    """Join features to labels while preserving absent conditional targets."""

    features = tuple(features)
    contexts = _normalise_context(context, features)
    targets = _normalise_labels(labels)
    population = targets.merge(contexts, on=list(COHORT_KEY), how="left", validate="many_to_one")
    if population.loc[:, list(features)].isna().all(axis=1).any():
        raise ValueError("one or more label cohorts lack all anchor-time context values")
    population["label_available_utc"] = _label_available_utc(population)
    for target in TARGETS:
        valid, reason = _target_validity(population, target)
        population[f"{target.name}__target_valid"] = valid
        population[f"{target.name}__target_status"] = reason
    return population.sort_values(
        ["horizon_hours", "cohort_anchor_utc", "side_name", "frozen_base_score_decile"], kind="stable"
    ).reset_index(drop=True)


def build_expanding_folds(
    frame: pd.DataFrame, *, min_train_days: int, validation_days: int
) -> list[dict[str, Any]]:
    """Return fixed-width chronological validation windows with expanding history."""

    if int(min_train_days) < 1 or int(validation_days) < 1:
        raise ValueError("fold durations must be positive whole days")
    anchors = _utc(frame["cohort_anchor_utc"])
    if anchors.empty:
        return []
    first = anchors.min().floor("D") + pd.Timedelta(days=int(min_train_days))
    end_limit = anchors.max().ceil("D") + pd.Timedelta(days=1)
    folds: list[dict[str, Any]] = []
    index = 0
    start = first
    while start < end_limit:
        end = min(start + pd.Timedelta(days=int(validation_days)), end_limit)
        if (anchors.ge(start) & anchors.lt(end)).any():
            folds.append({"fold_id": index, "validation_start_utc": start, "validation_end_utc": end})
            index += 1
        start = end
    return folds


def _spread_train_subset(frame: pd.DataFrame, budget: int) -> pd.DataFrame:
    """Timestamp-spread cap independent of targets, labels, or feature values."""

    ordered = frame.sort_values(
        ["cohort_anchor_utc", "side_name", "frozen_base_score_decile"], kind="stable"
    )
    if len(ordered) <= int(budget):
        return ordered
    positions = np.linspace(0, len(ordered) - 1, int(budget), dtype=np.int64)
    return ordered.iloc[positions].copy()


def _catboost_regressor(*, seed: int, threads: int):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        loss_function="MAE",
        random_seed=int(seed),
        thread_count=int(threads),
        **CATBOOST_GEOMETRY,
    )


def _catboost_classifier(*, seed: int, threads: int):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        loss_function="Logloss",
        random_seed=int(seed),
        thread_count=int(threads),
        **CATBOOST_GEOMETRY,
    )


def _last_known_baselines(train: pd.DataFrame, evaluation: pd.DataFrame, target_column: str) -> np.ndarray:
    """Fold-start known last delta per side/decile; never uses validation labels."""

    groups = ["side_name", "frozen_base_score_decile"]
    last = (
        train.sort_values(
            ["label_available_utc", "cohort_anchor_utc", *groups], kind="stable"
        )
        .groupby(groups, observed=True, sort=False)
        .tail(1)
        .loc[:, [*groups, target_column]]
    )
    lookup = evaluation.loc[:, groups].merge(last, on=groups, how="left", validate="many_to_one")
    return pd.to_numeric(lookup[target_column], errors="coerce").to_numpy(dtype=float)


def _regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(y_true) & np.isfinite(prediction)
    if not valid.any():
        return {"mae": float("nan"), "rank_ic": float("nan"), "rows": 0.0}
    actual = y_true[valid]
    predicted = prediction[valid]
    ic = float("nan")
    if len(actual) >= 2 and np.unique(actual).size >= 2 and np.unique(predicted).size >= 2:
        ic = float(pd.Series(actual).corr(pd.Series(predicted), method="spearman"))
    return {"mae": float(np.abs(actual - predicted).mean()), "rank_ic": ic, "rows": float(len(actual))}


def _calibration_ece(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    valid = np.isfinite(y_true) & np.isfinite(probability)
    if not valid.any():
        return float("nan")
    actual = y_true[valid]
    predicted = np.clip(probability[valid], 0.0, 1.0)
    bucket = np.minimum((predicted * bins).astype(int), bins - 1)
    total = float(len(actual))
    error = 0.0
    for index in range(bins):
        selected = bucket == index
        if selected.any():
            error += selected.sum() / total * abs(float(actual[selected].mean()) - float(predicted[selected].mean()))
    return float(error)


def _classification_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(y_true) & np.isfinite(probability)
    if not valid.any():
        return {"auc": float("nan"), "ap": float("nan"), "brier": float("nan"), "calibration_ece_10": float("nan"), "rows": 0.0}
    actual = y_true[valid].astype(np.int8)
    predicted = np.clip(probability[valid], 0.0, 1.0)
    try:
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    except ImportError as error:  # pragma: no cover - repository dependency contract
        raise RuntimeError("scikit-learn is required for head-ablation metrics") from error
    auc = float("nan")
    ap = float("nan")
    if np.unique(actual).size >= 2:
        auc = float(roc_auc_score(actual, predicted))
        ap = float(average_precision_score(actual, predicted))
    return {
        "auc": auc,
        "ap": ap,
        "brier": float(brier_score_loss(actual, predicted)),
        "calibration_ece_10": _calibration_ece(actual, predicted),
        "rows": float(len(actual)),
    }


def _metric_row(
    *,
    horizon: int,
    target: str,
    fold: Mapping[str, Any],
    evaluation_rows: int,
    valid_rows: int,
    missing_rows: int,
    train_rows: int,
    fitted_rows: int,
    status: str,
    y: np.ndarray,
    prediction: np.ndarray,
    constant_regression: np.ndarray,
    last_known_regression: np.ndarray,
) -> dict[str, Any]:
    sign = (y > 0.0).astype(np.int8)
    model_sign = np.clip(prediction["sign_probability"], 0.0, 1.0)
    constant_sign = np.clip(prediction["constant_sign_probability"], 0.0, 1.0)
    last_sign = np.where(np.isfinite(last_known_regression), (last_known_regression > 0.0).astype(float), np.nan)
    result: dict[str, Any] = {
        "horizon_hours": int(horizon),
        "target": target,
        **dict(fold),
        "evaluation_cohort_rows": int(evaluation_rows),
        "target_valid_rows": int(valid_rows),
        "target_missing_rows": int(missing_rows),
        "training_rows": int(train_rows),
        "fitted_training_rows": int(fitted_rows),
        "fit_status": status,
        "sign_positive_rate": float(sign.mean()) if len(sign) else float("nan"),
    }
    for prefix, values in (
        ("model", prediction["delta_prediction"]),
        ("constant", constant_regression),
        ("last_known", last_known_regression),
    ):
        for name, value in _regression_metrics(y, values).items():
            if name != "rows":
                result[f"{prefix}_regression_{name}"] = value
    for prefix, values in (("model", model_sign), ("constant", constant_sign), ("last_known", last_sign)):
        for name, value in _classification_metrics(sign, values).items():
            if name != "rows":
                result[f"{prefix}_sign_{name}"] = value
    return result


def fit_target_oof(
    frame: pd.DataFrame,
    *,
    target: TargetSpec,
    features: Iterable[str],
    folds: Iterable[Mapping[str, Any]],
    min_train_rows: int,
    fit_budget_rows: int,
    random_state: int,
    threads: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one compact regressor and sign head per expanding availability fold."""

    features = tuple(features)
    work = frame.copy()
    valid_column = f"{target.name}__target_valid"
    status_column = f"{target.name}__target_status"
    work["__target__"] = pd.to_numeric(work[target.delta_column], errors="coerce")
    work["__target_valid__"] = work[valid_column].astype(bool)
    parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for fold in folds:
        start = pd.Timestamp(fold["validation_start_utc"])
        end = pd.Timestamp(fold["validation_end_utc"])
        evaluation = work.loc[
            work["cohort_anchor_utc"].ge(start) & work["cohort_anchor_utc"].lt(end)
        ].copy()
        if evaluation.empty:
            continue
        valid_eval = evaluation["__target_valid__"].to_numpy(bool)
        train = work.loc[
            work["__target_valid__"]
            & work["label_available_utc"].lt(start)
            & work["after_target_available_utc"].lt(start)
        ].copy()
        # This is deliberately asserted rather than merely described.  It is
        # the strict contract requested for every label used to train a fold.
        if not train.empty and not train["after_target_available_utc"].lt(start).all():
            raise AssertionError("training label availability is not strictly before validation anchor start")
        train_subset = _spread_train_subset(train, int(fit_budget_rows))
        y_train = train_subset["__target__"].to_numpy(dtype=float)
        constant = float(np.mean(y_train)) if len(y_train) else 0.0
        constant_sign = float(np.mean(y_train > 0.0)) if len(y_train) else 0.0
        result = evaluation.loc[
            :,
            [*COHORT_KEY, "horizon_hours", "horizon_role", "label_available_utc", "after_target_available_utc", "__target__", "__target_valid__", status_column],
        ].copy()
        result["target"] = target.name
        result["fold_id"] = int(fold["fold_id"])
        result["validation_start_utc"] = start
        result["validation_end_utc"] = end
        result["delta_prediction"] = np.nan
        result["sign_probability"] = np.nan
        result["constant_delta_prediction"] = np.where(valid_eval, constant, np.nan)
        result["constant_sign_probability"] = np.where(valid_eval, constant_sign, np.nan)
        last_known = _last_known_baselines(train, evaluation, "__target__") if len(train) else np.full(len(evaluation), np.nan)
        result["last_known_delta_prediction"] = np.where(valid_eval, last_known, np.nan)
        result["last_known_sign_probability"] = np.where(
            valid_eval & np.isfinite(last_known), (last_known > 0.0).astype(float), np.nan
        )
        status = "constant_fallback_insufficient_prior_resolved_rows"
        if valid_eval.any() and len(train_subset) >= int(min_train_rows):
            x_train = train_subset.loc[:, list(features)]
            x_eval = evaluation.loc[valid_eval, list(features)]
            unique_sign = np.unique(y_train > 0.0)
            regressor = _catboost_regressor(
                seed=int(random_state + 10_000 * int(fold["fold_id"]) + len(target.name)), threads=threads
            ).fit(x_train, y_train)
            result.loc[valid_eval, "delta_prediction"] = np.asarray(regressor.predict(x_eval), dtype=float)
            if len(unique_sign) == 2:
                classifier = _catboost_classifier(
                    seed=int(random_state + 20_000 * int(fold["fold_id"]) + len(target.name)), threads=threads
                ).fit(x_train, (y_train > 0.0).astype(np.int8))
                result.loc[valid_eval, "sign_probability"] = np.asarray(
                    classifier.predict_proba(x_eval)[:, 1], dtype=float
                )
                status = "fixed_compact_catboost_regression_and_sign_classifier"
            else:
                result.loc[valid_eval, "sign_probability"] = constant_sign
                status = "fixed_compact_catboost_regression_constant_sign_fallback"
        else:
            result.loc[valid_eval, "delta_prediction"] = constant
            result.loc[valid_eval, "sign_probability"] = constant_sign
        result["fit_status"] = status
        result["training_rows"] = int(len(train))
        result["fitted_training_rows"] = int(len(train_subset))
        result["training_max_after_target_available_utc"] = (
            train["after_target_available_utc"].max() if len(train) else pd.NaT
        )
        result = result.rename(
            columns={
                "__target__": "target_delta",
                "__target_valid__": "target_valid",
                status_column: "target_status",
            }
        )
        parts.append(result)
        metric_valid = result["target_valid"].to_numpy(bool)
        metric_frame = result.loc[metric_valid]
        metric_rows.append(
            _metric_row(
                horizon=int(evaluation["horizon_hours"].iloc[0]),
                target=target.name,
                fold=fold,
                evaluation_rows=len(evaluation),
                valid_rows=int(metric_valid.sum()),
                missing_rows=int((~metric_valid).sum()),
                train_rows=len(train),
                fitted_rows=len(train_subset),
                status=status,
                y=metric_frame["target_delta"].to_numpy(dtype=float),
                prediction={
                    "delta_prediction": metric_frame["delta_prediction"].to_numpy(dtype=float),
                    "sign_probability": metric_frame["sign_probability"].to_numpy(dtype=float),
                    "constant_sign_probability": metric_frame["constant_sign_probability"].to_numpy(dtype=float),
                },
                constant_regression=metric_frame["constant_delta_prediction"].to_numpy(dtype=float),
                last_known_regression=metric_frame["last_known_delta_prediction"].to_numpy(dtype=float),
            )
        )
    prediction_columns = [
        *COHORT_KEY, "horizon_hours", "horizon_role", "label_available_utc", "after_target_available_utc",
        "target", "fold_id", "validation_start_utc", "validation_end_utc", "target_delta", "target_valid",
        "target_status", "delta_prediction", "sign_probability", "constant_delta_prediction", "constant_sign_probability",
        "last_known_delta_prediction", "last_known_sign_probability", "fit_status", "training_rows", "fitted_training_rows",
        "training_max_after_target_available_utc",
    ]
    return (
        pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=prediction_columns),
        pd.DataFrame(metric_rows),
    )


def _aggregate_oof_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if predictions.empty:
        return pd.DataFrame(rows)
    for (horizon, target), group in predictions.groupby(["horizon_hours", "target"], sort=True):
        valid = group.loc[group["target_valid"].astype(bool)].copy()
        y = valid["target_delta"].to_numpy(dtype=float)
        result: dict[str, Any] = {
            "horizon_hours": int(horizon),
            "horizon_role": "primary" if int(horizon) == PRIMARY_HORIZON else "sensitivity",
            "target": target,
            "oof_rows": int(len(valid)),
            "missing_target_rows": int((~group["target_valid"].astype(bool)).sum()),
            "folds": int(valid["fold_id"].nunique()),
        }
        for prefix, values in (
            ("model", valid["delta_prediction"].to_numpy(dtype=float)),
            ("constant", valid["constant_delta_prediction"].to_numpy(dtype=float)),
            ("last_known", valid["last_known_delta_prediction"].to_numpy(dtype=float)),
        ):
            for name, value in _regression_metrics(y, values).items():
                if name != "rows":
                    result[f"{prefix}_regression_{name}"] = value
        sign = (y > 0.0).astype(np.int8)
        for prefix, values in (
            ("model", valid["sign_probability"].to_numpy(dtype=float)),
            ("constant", valid["constant_sign_probability"].to_numpy(dtype=float)),
            ("last_known", valid["last_known_sign_probability"].to_numpy(dtype=float)),
        ):
            for name, value in _classification_metrics(sign, values).items():
                if name != "rows":
                    result[f"{prefix}_sign_{name}"] = value
        rows.append(result)
    return pd.DataFrame(rows)


def _fold_stability(per_fold: pd.DataFrame) -> pd.DataFrame:
    if per_fold.empty:
        return pd.DataFrame(columns=["horizon_hours", "target", "metric", "folds", "mean", "std", "minimum", "maximum"])
    metrics = (
        "model_regression_mae",
        "model_regression_rank_ic",
        "model_sign_auc",
        "model_sign_ap",
        "model_sign_brier",
        "model_sign_calibration_ece_10",
    )
    records: list[dict[str, Any]] = []
    for (horizon, target), group in per_fold.groupby(["horizon_hours", "target"], sort=True):
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            records.append(
                {
                    "horizon_hours": int(horizon),
                    "target": target,
                    "metric": metric,
                    "folds": int(len(values)),
                    "mean": float(values.mean()) if len(values) else float("nan"),
                    "std": float(values.std(ddof=0)) if len(values) else float("nan"),
                    "minimum": float(values.min()) if len(values) else float("nan"),
                    "maximum": float(values.max()) if len(values) else float("nan"),
                }
            )
    return pd.DataFrame(records)


def _latest_fold_coverage(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if predictions.empty:
        return pd.DataFrame(records)
    for (horizon, target), group in predictions.groupby(["horizon_hours", "target"], sort=True):
        latest = group.loc[group["fold_id"].eq(group["fold_id"].max())]
        valid = latest["target_valid"].astype(bool)
        model_scored = valid & latest["delta_prediction"].notna() & latest["sign_probability"].notna()
        records.append(
            {
                "horizon_hours": int(horizon),
                "target": target,
                "latest_fold_id": int(latest["fold_id"].max()),
                "validation_start_utc": latest["validation_start_utc"].iloc[0],
                "validation_end_utc": latest["validation_end_utc"].iloc[0],
                "latest_fold_cohort_rows": int(len(latest)),
                "latest_fold_valid_target_rows": int(valid.sum()),
                "latest_fold_missing_target_rows": int((~valid).sum()),
                "latest_fold_model_scored_rows": int(model_scored.sum()),
                "latest_fold_model_coverage": float(model_scored.sum() / valid.sum()) if valid.any() else float("nan"),
            }
        )
    return pd.DataFrame(records)


def plan(context_source: Path, label_source: Path, output: Path, args: argparse.Namespace) -> dict[str, Any]:
    manifests, hashes = _source_hashes(context_source, label_source)
    features = _context_features(manifests["context"])
    return {
        "action": "PLAN_ONLY_NO_TRAINING_OR_MATERIALIZATION",
        "schema": SCHEMA,
        "context_source": str(context_source),
        "label_source": str(label_source),
        "output": str(output),
        "source_sha256": hashes,
        "panel_identity_sha256": manifests["context"].get("source_panel_identity_sha256"),
        "horizons": {"primary": PRIMARY_HORIZON, "sensitivity": SENSITIVITY_HORIZON},
        "targets": [{"name": target.name, "delta_column": target.delta_column} for target in TARGETS],
        "features": list(features),
        "fixed_catboost_geometry": dict(CATBOOST_GEOMETRY),
        "fit_budget": {
            "minimum_prior_resolved_rows": int(args.min_train_rows),
            "maximum_fit_rows_per_target_fold": int(args.fit_budget_rows),
            "threads": int(args.threads),
            "random_state": int(args.random_state),
            "feature_selection": "disabled",
            "hyperparameter_optimization": "disabled",
        },
        "fold_contract": {
            "kind": "chronological expanding",
            "minimum_calendar_history_days": int(args.min_train_days),
            "validation_days": int(args.validation_days),
            "training_rule": "every training row satisfies after_target_available_utc < validation anchor start (and max before/after availability < start)",
            "conditional_targets": "missing conditional support remains missing and is excluded per target; it is never zero-filled or synthesized",
        },
        "scope": "component learnability only; no trading PnL, admission routing, policy selection, or portfolio replay",
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
    features = _context_features(manifests["context"])
    context = pd.read_parquet(context_source / "cohort_transition_context.parquet", columns=[*COHORT_KEY, *features])
    labels = pd.read_parquet(label_source / "cohort_transition_labels.parquet", columns=list(_label_columns()))
    population = prepare_population(context, labels, features)
    folds = build_expanding_folds(
        population, min_train_days=args.min_train_days, validation_days=args.validation_days
    )
    prediction_parts: list[pd.DataFrame] = []
    metric_parts: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        horizon_frame = population.loc[population["horizon_hours"].eq(horizon)].copy()
        for target in TARGETS:
            predictions, metrics = fit_target_oof(
                horizon_frame,
                target=target,
                features=features,
                folds=folds,
                min_train_rows=args.min_train_rows,
                fit_budget_rows=args.fit_budget_rows,
                random_state=args.random_state,
                threads=args.threads,
            )
            prediction_parts.append(predictions)
            metric_parts.append(metrics)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    per_fold = pd.concat(metric_parts, ignore_index=True)
    oof_metrics = _aggregate_oof_metrics(predictions)
    stability = _fold_stability(per_fold)
    latest_coverage = _latest_fold_coverage(predictions)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    predictions.to_parquet(temporary / "oof_head_predictions.parquet", index=False, compression="zstd")
    per_fold.to_parquet(temporary / "per_fold_metrics.parquet", index=False, compression="zstd")
    oof_metrics.to_parquet(temporary / "oof_metrics.parquet", index=False, compression="zstd")
    stability.to_parquet(temporary / "fold_stability.parquet", index=False, compression="zstd")
    latest_coverage.to_parquet(temporary / "latest_fold_coverage.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "status": "IMMUTABLE_COMPONENT_LEARNABILITY_ABLATION_NOT_PROMOTION_ELIGIBLE",
        "promotion_eligible": False,
        "source_artifacts_sha256": hashes,
        "source_panel_identity_sha256": manifests["context"].get("source_panel_identity_sha256"),
        "context_feature_columns": list(features),
        "targets": [{"name": target.name, "delta_column": target.delta_column, "sign_label": "delta > 0"} for target in TARGETS],
        "horizons": {"primary": PRIMARY_HORIZON, "sensitivity": SENSITIVITY_HORIZON},
        "fixed_catboost_geometry": dict(CATBOOST_GEOMETRY),
        "fit_budget": {
            "minimum_prior_resolved_rows": int(args.min_train_rows),
            "maximum_fit_rows_per_target_fold": int(args.fit_budget_rows),
            "threads": int(args.threads),
            "random_state": int(args.random_state),
            "feature_selection": "disabled",
            "hyperparameter_optimization": "disabled",
        },
        "folds": folds,
        "contracts": {
            "utc": "all stored timestamps are timezone-aware UTC",
            "training_availability": "every fit uses actual after_target_available_utc strictly before validation anchor start; label_available_utc is max(before, after) and is also strictly before start",
            "conditional_target_missingness": "missing conditional favorable/adverse values stay unscored with an explicit target-status reason; no zero fill or synthetic fallback",
            "baselines": "constant prior-resolved train mean/prevalence and fold-start last-resolved side/decile target where available",
            "metrics": "OOF MAE/rank IC; sign AUC/AP/Brier/ECE calibration; fold stability and latest-fold coverage",
            "scope": "no trading PnL, admission routing, policy selection, or portfolio replay",
        },
        "rows": {"population": int(len(population)), "oof_predictions": int(len(predictions))},
        "outputs_sha256": {path.name: sha256(path) for path in sorted(temporary.glob("*.parquet"))},
        "checksum_convention": "manifest.json is verified by detached manifest.sha256",
    }
    (temporary / "manifest.json").write_text(
        json.dumps(_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (temporary / "manifest.sha256").write_text(
        f"{sha256(temporary / 'manifest.json')}  manifest.json\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    return {
        "output": str(output),
        "oof_predictions": int(len(predictions)),
        "source_sha256": hashes,
        "primary_horizon": PRIMARY_HORIZON,
        "sensitivity_horizon": SENSITIVITY_HORIZON,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--context-source", type=Path, default=CONTEXT_SOURCE)
    result.add_argument("--label-source", type=Path, default=LABEL_SOURCE)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--min-train-days", type=int, default=28)
    result.add_argument("--validation-days", type=int, default=14)
    result.add_argument("--min-train-rows", type=int, default=1_500)
    result.add_argument("--fit-budget-rows", type=int, default=75_000)
    result.add_argument("--threads", type=int, default=1)
    result.add_argument("--random-state", type=int, default=20260729)
    result.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate immutable hashes and print the fixed learnability contract without reading data rows or fitting heads.",
    )
    return result


def main() -> None:
    print(json.dumps(_safe(run(parser().parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
