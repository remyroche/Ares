"""Chronological, bounded explanation of recurrent leaf-family instability.

This is a diagnostic module for the feature/leaf-reasoning portability funnel.
It deliberately consumes *period-level*, already-resolved family effects rather
than row-level realised outcomes.  That distinction is important: no current
candidate can become part of its own correctness, calibration or covariance
history merely because its family's later diagnostic is reviewed.

The D0--D6 ladder is intentionally linear/ridge based.  It answers whether a
small number of causal support, context and relationship-break summaries
explain a family effect that has already been observed out of sample; it is not
another flexible predictive model and it does not produce inference features by
itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import os
import shutil
import tempfile
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


SCHEMA = "leaf_failure_explainability_v1"
RAW_LEAF_TOKENS = ("leaf_token", "leaf_id", "raw_leaf", "leaf_assignment")


class LeafFailureExplainabilityError(ValueError):
    """Raised when a decomposition cannot prove its causal diagnostic contract."""


@dataclass(frozen=True)
class LeafFailureExplainabilityColumns:
    """Column contract for already-materialised recurrent-family period rows."""

    side: str = "side_name"
    layer: str = "layer"
    head: str = "head_name"
    family: str = "rule_signature"
    period: str = "period_start"
    effect: str = "economic_effect"
    standard_error: str = "effect_standard_error"
    label_available_ts: str = "label_available_ts"
    feature_generation_ts: str = "feature_generation_ts"

    @property
    def keys(self) -> tuple[str, ...]:
        return (self.side, self.layer, self.head, self.family)


@dataclass(frozen=True)
class LeafFailureExplainabilityConfig:
    """Small fixed diagnostic ladder, selected before any result is inspected."""

    ridge_alpha: float = 8.0
    variance_floor: float = 1e-6
    min_train_periods: int = 4
    min_evaluation_periods: int = 3
    regime_conditional_min_explainability: float = 0.40
    partly_conditionable_min_explainability: float = 0.15

    def validate(self) -> None:
        if self.ridge_alpha < 0.0 or not np.isfinite(self.ridge_alpha):
            raise LeafFailureExplainabilityError("ridge_alpha must be finite and non-negative")
        if self.variance_floor <= 0.0 or not np.isfinite(self.variance_floor):
            raise LeafFailureExplainabilityError("variance_floor must be finite and positive")
        if self.min_train_periods < 2 or self.min_evaluation_periods < 2:
            raise LeafFailureExplainabilityError("minimum period support must be at least two")
        if not 0.0 <= self.partly_conditionable_min_explainability <= self.regime_conditional_min_explainability <= 1.0:
            raise LeafFailureExplainabilityError("explainability thresholds must lie in [0, 1] and be ordered")


@dataclass(frozen=True)
class LeafFailureExplainabilityResult:
    """Per-family ladder metrics and chronological held-out predictions."""

    diagnostics: pd.DataFrame
    predictions: pd.DataFrame
    classifications: pd.DataFrame


STEPS: tuple[tuple[str, str], ...] = (
    ("D0", "baseline"),
    ("D1", "global_period"),
    ("D2", "support_composition"),
    ("D3", "marginal_levels_variances"),
    ("D4", "causal_market_context"),
    ("D5", "family_context_interactions"),
    ("D6", "covariance_relationship_breaks"),
)


def _as_utc(value: pd.Series, name: str) -> pd.Series:
    result = pd.to_datetime(value, utc=True, errors="coerce")
    if result.isna().any():
        raise LeafFailureExplainabilityError(f"{name} contains invalid UTC timestamps")
    return result


def _forbid_raw_leaf_columns(frame: pd.DataFrame) -> None:
    bad = [
        name for name in frame
        if not str(name).lower().startswith("base_reasoning__g1_leaf_assignment_count")
        and any(token in str(name).lower() for token in RAW_LEAF_TOKENS)
    ]
    if bad:
        raise LeafFailureExplainabilityError(f"raw leaf identity is forbidden in decomposition input: {bad[:8]}")


def _validate(
    frame: pd.DataFrame,
    *,
    columns: LeafFailureExplainabilityColumns,
    groups: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    _forbid_raw_leaf_columns(frame)
    expected = [*columns.keys, columns.period, columns.effect, columns.standard_error, columns.label_available_ts, columns.feature_generation_ts]
    absent = [name for name in expected if name not in frame]
    if absent:
        raise LeafFailureExplainabilityError(f"period-level input lacks required columns: {absent}")
    unknown = [name for step, names in groups.items() for name in names if name not in frame]
    if unknown:
        raise LeafFailureExplainabilityError(f"declared explanatory fields are absent: {sorted(set(unknown))}")
    if set(groups).difference({name for name, _ in STEPS}):
        raise LeafFailureExplainabilityError("groups contains an unknown D-step")
    work = frame.copy()
    work[columns.period] = _as_utc(work[columns.period], columns.period)
    work[columns.label_available_ts] = _as_utc(work[columns.label_available_ts], columns.label_available_ts)
    work[columns.feature_generation_ts] = _as_utc(work[columns.feature_generation_ts], columns.feature_generation_ts)
    if not work[columns.label_available_ts].lt(work[columns.feature_generation_ts]).all():
        raise LeafFailureExplainabilityError("period diagnostics include a label unresolved at feature generation")
    for name in (columns.effect, columns.standard_error):
        work[name] = pd.to_numeric(work[name], errors="coerce")
    if not np.isfinite(work[columns.effect]).all() or not np.isfinite(work[columns.standard_error]).all():
        raise LeafFailureExplainabilityError("effect and standard error must be finite")
    if work[columns.standard_error].lt(0.0).any():
        raise LeafFailureExplainabilityError("effect standard error cannot be negative")
    if work.duplicated([*columns.keys, columns.period]).any():
        raise LeafFailureExplainabilityError("family-period rows must be unique")
    return work.sort_values([*columns.keys, columns.period], kind="stable").reset_index(drop=True)


def _feature_order(groups: Mapping[str, Sequence[str]], step: str) -> list[str]:
    index = [name for name, _ in STEPS].index(step)
    values: list[str] = []
    for prior, _ in STEPS[: index + 1]:
        values.extend(groups.get(prior, ()))
    return list(dict.fromkeys(values))


def _train_matrix(train: pd.DataFrame, evaluate: pd.DataFrame, fields: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    if not fields:
        return np.empty((len(train), 0), dtype=np.float32), np.empty((len(evaluate), 0), dtype=np.float32)
    train_values = train.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce")
    medians = train_values.median(axis=0).fillna(0.0)
    scale = (train_values.quantile(0.75) - train_values.quantile(0.25)).abs().replace(0.0, 1.0).fillna(1.0)
    x_train = train_values.fillna(medians).sub(medians).div(scale).clip(-8.0, 8.0).to_numpy(np.float32)
    x_eval = evaluate.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").fillna(medians).sub(medians).div(scale).clip(-8.0, 8.0).to_numpy(np.float32)
    return x_train, x_eval


def _weighted_mean(value: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sum(value * weight) / max(np.sum(weight), 1e-12))


def _fit_step(
    train: pd.DataFrame,
    evaluate: pd.DataFrame,
    *,
    fields: Sequence[str],
    columns: LeafFailureExplainabilityColumns,
    config: LeafFailureExplainabilityConfig,
) -> np.ndarray:
    y = train[columns.effect].to_numpy(float)
    weight = 1.0 / (np.square(train[columns.standard_error].to_numpy(float)) + config.variance_floor)
    if not fields:
        return np.full(len(evaluate), _weighted_mean(y, weight), dtype=np.float64)
    x_train, x_eval = _train_matrix(train, evaluate, fields)
    model = Ridge(alpha=config.ridge_alpha, fit_intercept=True, random_state=20260803)
    model.fit(x_train, y, sample_weight=weight)
    return np.asarray(model.predict(x_eval), dtype=np.float64)


def _classification(
    diagnostics: pd.DataFrame,
    *,
    columns: LeafFailureExplainabilityColumns,
    config: LeafFailureExplainabilityConfig,
) -> pd.DataFrame:
    d0 = diagnostics.loc[diagnostics["step"].eq("D0"), [*columns.keys, "weighted_mse"]].rename(columns={"weighted_mse": "d0_weighted_mse"})
    d4 = diagnostics.loc[diagnostics["step"].eq("D4"), [*columns.keys, "weighted_mse", "heldout_r2"]].rename(columns={"weighted_mse": "d4_weighted_mse", "heldout_r2": "d4_heldout_r2"})
    d6 = diagnostics.loc[diagnostics["step"].eq("D6"), [*columns.keys, "weighted_mse", "heldout_r2"]].rename(columns={"weighted_mse": "d6_weighted_mse", "heldout_r2": "d6_heldout_r2"})
    result = d0.merge(d4, on=list(columns.keys), how="outer", validate="one_to_one").merge(d6, on=list(columns.keys), how="outer", validate="one_to_one")
    result["regime_explainability"] = (1.0 - result["d4_weighted_mse"] / result["d0_weighted_mse"].clip(lower=1e-12)).clip(-np.inf, 1.0)
    result["covariance_explainability"] = (1.0 - result["d6_weighted_mse"] / result["d4_weighted_mse"].clip(lower=1e-12)).clip(-np.inf, 1.0)
    result["classification"] = np.select(
        [
            result["d4_heldout_r2"].gt(0.0) & result["regime_explainability"].ge(config.regime_conditional_min_explainability),
            result["d4_heldout_r2"].gt(0.0) & result["regime_explainability"].ge(config.partly_conditionable_min_explainability),
            result["d6_heldout_r2"].gt(0.0) & result["covariance_explainability"].gt(0.0),
        ],
        ["REGIME_CONDITIONAL", "PARTLY_CONDITIONABLE", "COVARIANCE_CONDITIONAL"],
        default="UNEXPLAINED_CONCEPT_BREAK",
    )
    return result


def analyze_leaf_failure_explainability(
    frame: pd.DataFrame,
    *,
    groups: Mapping[str, Sequence[str]],
    columns: LeafFailureExplainabilityColumns = LeafFailureExplainabilityColumns(),
    config: LeafFailureExplainabilityConfig = LeafFailureExplainabilityConfig(),
) -> LeafFailureExplainabilityResult:
    """Run the predeclared chronological D0--D6 ladder for each family.

    The first ``min_train_periods`` periods warm up the history.  Each later
    period is predicted from only earlier period outcomes, so the diagnostic's
    held-out scores are chronological even though it is never an inference
    model.  Empty D-step blocks are legitimate controls and simply repeat the
    preceding fitted representation.
    """
    config.validate()
    work = _validate(frame, columns=columns, groups=groups)
    diagnostic_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []
    for keys, cell in work.groupby(list(columns.keys), observed=True, sort=True):
        cell = cell.sort_values(columns.period, kind="stable").reset_index(drop=True)
        if len(cell) < config.min_train_periods + config.min_evaluation_periods:
            continue
        prefix = dict(zip(columns.keys, keys, strict=True))
        first_eval = config.min_train_periods
        actual = cell.loc[first_eval:, columns.effect].to_numpy(float)
        error = cell.loc[first_eval:, columns.standard_error].to_numpy(float)
        weight = 1.0 / (np.square(error) + config.variance_floor)
        baseline_variance = _weighted_mean(np.square(actual - _weighted_mean(actual, weight)), weight)
        for step, _ in STEPS:
            fields = _feature_order(groups, step)
            predictions: list[float] = []
            for end in range(first_eval, len(cell)):
                predictions.append(float(_fit_step(cell.iloc[:end], cell.iloc[end:end + 1], fields=fields, columns=columns, config=config)[0]))
            pred = np.asarray(predictions, dtype=float)
            squared_error = np.square(actual - pred)
            mse = _weighted_mean(squared_error, weight)
            r2 = float(1.0 - mse / max(baseline_variance, config.variance_floor))
            diagnostic_rows.append({
                **prefix,
                "step": step,
                "field_count": int(len(fields)),
                "evaluation_periods": int(len(actual)),
                "weighted_mse": mse,
                "heldout_r2": r2,
                "unexplained_excess_variance": mse,
                "first_evaluation_period": cell.loc[first_eval, columns.period],
                "last_evaluation_period": cell.loc[len(cell) - 1, columns.period],
            })
            prediction_parts.append(pd.DataFrame({
                **prefix,
                "step": step,
                "period_start": cell.loc[first_eval:, columns.period].to_numpy(),
                "effect": actual,
                "effect_standard_error": error,
                "prediction": pred,
                "residual": actual - pred,
                "feature_count": int(len(fields)),
            }))
    diagnostics = pd.DataFrame(diagnostic_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    classifications = _classification(diagnostics, columns=columns, config=config) if not diagnostics.empty else pd.DataFrame()
    return LeafFailureExplainabilityResult(diagnostics=diagnostics, predictions=predictions, classifications=classifications)


def write_leaf_failure_explainability(result: LeafFailureExplainabilityResult, output_dir: Path) -> Path:
    """Write immutable development-only diagnostic artifacts atomically."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        result.diagnostics.to_parquet(temporary / "leaf_failure_decomposition.parquet", index=False, compression="zstd")
        result.predictions.to_parquet(temporary / "leaf_failure_predictions.parquet", index=False, compression="zstd")
        result.classifications.to_parquet(temporary / "leaf_failure_classification.parquet", index=False, compression="zstd")
        payload = {
            "schema": SCHEMA,
            "status": "COMPLETED_DIAGNOSTIC_ONLY",
            "chronological_protocol": "each evaluated family-period uses only earlier period rows",
            "raw_leaf_ids": "forbidden",
            "classification_counts": result.classifications.get("classification", pd.Series(dtype=str)).value_counts().to_dict(),
        }
        (temporary / "leaf_failure_classification.yaml").write_text(
            "schema: " + SCHEMA + "\nstatus: COMPLETED_DIAGNOSTIC_ONLY\nclassification_counts: " + json.dumps(payload["classification_counts"], sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(temporary, output_dir)
        return output_dir
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
