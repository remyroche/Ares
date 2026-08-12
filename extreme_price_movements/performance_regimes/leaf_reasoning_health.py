"""Fold-local health diagnostics for strict-OOF leaf assignments.

This is an *audit* of already-issued strict-OOF assignments.  It does not fit
rules, calibrate predictions, or infer a relationship between leaf tokens from
different folds.  A raw tree-leaf token is only meaningful inside the model
that produced it, so every summary key contains ``head``, ``side`` and
``fold``.  Cross-fold recurrence belongs to the separate G2 rule-signature
workstream.

The implementation reduces assignments to additive sufficient statistics.
Consequently callers may pass either one frame or an iterator of compatible
chunks without changing the result or materialising a second wide copy of a
large OOF ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


EPSILON = 1e-12


@dataclass(frozen=True)
class LeafReasoningHealthColumns:
    """Explicit input contract for a strict-OOF leaf-assignment ledger."""

    leaf_token: str = "leaf_token"
    head: str = "head"
    side: str = "side_name"
    fold: str = "fold"
    timestamp: str = "__ts__"
    activation: str = "activation"
    prediction: str = "prediction"
    label: str = "label"
    economic: str = "net_bps"
    candidate_id: str | None = "candidate_id"
    # An OOF marker is optional because some artifacts establish strict OOF
    # provenance in their manifest rather than as a row field.  If supplied,
    # it is fail-closed: every row must be true.
    strict_oof: str | None = None
    # If the caller has a coarser, non-overlapping target period, provide it.
    # Otherwise each decision timestamp is treated as its own period.
    period: str | None = None


@dataclass(frozen=True)
class LeafReasoningHealthConfig:
    """Support and conservative within-fold health thresholds.

    ``economic`` is assumed to be in its native units (normally net bps),
    while calibration is normalized by the observed label standard deviation
    so the calibration gate is not tied to an arbitrary label scale.
    """

    activation_threshold: float = 0.60
    minimum_rows: int = 50
    minimum_active_rows: int = 25
    minimum_active_periods: int = 3
    minimum_active_months: int = 1
    minimum_activation_share: float = 0.01
    minimum_score_rows: int = 25
    minimum_economic_rows: int = 25
    minimum_prediction_label_correlation: float = 0.0
    maximum_normalized_calibration_bias: float = 0.50
    minimum_active_economic_mean: float = 0.0


@dataclass(frozen=True)
class LeafReasoningHealthResult:
    """Per-leaf, per-period and per-month strict-OOF health tables."""

    leaf_health: pd.DataFrame
    period_health: pd.DataFrame
    month_health: pd.DataFrame


_STAT_COLUMNS: tuple[str, ...] = (
    "row_support",
    "activation_valid_rows",
    "activation_sum",
    "active_rows",
    "active_score_rows",
    "active_prediction_sum",
    "active_label_sum",
    "active_prediction_sq_sum",
    "active_label_sq_sum",
    "active_prediction_label_sum",
    "active_error_sum",
    "active_abs_error_sum",
    "active_sq_error_sum",
    "active_economic_rows",
    "active_economic_sum",
    "active_economic_sq_sum",
    "active_prediction_economic_rows",
    "active_pair_prediction_sum",
    "active_pair_prediction_sq_sum",
    "active_pair_economic_sum",
    "active_pair_economic_sq_sum",
    "active_prediction_economic_sum",
)


def _required_columns(columns: LeafReasoningHealthColumns) -> set[str]:
    required = {
        columns.leaf_token,
        columns.head,
        columns.side,
        columns.fold,
        columns.timestamp,
        columns.activation,
        columns.prediction,
        columns.label,
        columns.economic,
    }
    if columns.candidate_id:
        required.add(columns.candidate_id)
    if columns.strict_oof:
        required.add(columns.strict_oof)
    if columns.period:
        required.add(columns.period)
    return required


def _to_finite_numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    return np.where(np.isfinite(values), values, np.nan)


def _prepare_chunk(
    frame: pd.DataFrame,
    *,
    columns: LeafReasoningHealthColumns,
    config: LeafReasoningHealthConfig,
) -> pd.DataFrame:
    missing = sorted(_required_columns(columns).difference(frame.columns))
    if missing:
        raise KeyError(f"leaf reasoning health input is missing required columns: {missing}")
    if frame.empty:
        return pd.DataFrame(columns=["head", "side_name", "fold", "leaf_token", "period", "month", *_STAT_COLUMNS])

    if columns.strict_oof and not frame[columns.strict_oof].fillna(False).astype(bool).all():
        raise ValueError("leaf reasoning health accepts strict-OOF assignments only")

    group_columns = [columns.head, columns.side, columns.fold, columns.leaf_token]
    if frame[group_columns].isna().any().any():
        raise ValueError("head, side, fold, and leaf token must be non-null")
    if columns.candidate_id:
        identity = [columns.candidate_id, *group_columns]
        if frame.duplicated(identity).any():
            raise ValueError("duplicate candidate/head/side/fold/leaf assignment")

    timestamp = pd.to_datetime(frame[columns.timestamp], utc=True, errors="raise")
    activation = _to_finite_numeric(frame, columns.activation)
    prediction = _to_finite_numeric(frame, columns.prediction)
    label = _to_finite_numeric(frame, columns.label)
    economic = _to_finite_numeric(frame, columns.economic)
    active = np.isfinite(activation) & (activation >= float(config.activation_threshold))
    score_valid = active & np.isfinite(prediction) & np.isfinite(label)
    economic_valid = active & np.isfinite(economic)
    prediction_economic_valid = active & np.isfinite(prediction) & np.isfinite(economic)

    out = pd.DataFrame(
        {
            "head": frame[columns.head].to_numpy(copy=False),
            "side_name": frame[columns.side].to_numpy(copy=False),
            "fold": frame[columns.fold].to_numpy(copy=False),
            "leaf_token": frame[columns.leaf_token].to_numpy(copy=False),
            "period": (
                frame[columns.period].to_numpy(copy=False)
                if columns.period
                else timestamp.to_numpy(copy=False)
            ),
            # A UTC month label avoids a timezone-dropping Period conversion.
            "month": timestamp.dt.strftime("%Y-%m").to_numpy(copy=False),
            "row_support": np.ones(len(frame), dtype=np.int64),
            "activation_valid_rows": np.isfinite(activation).astype(np.int64),
            "activation_sum": np.nan_to_num(activation, nan=0.0),
            "active_rows": active.astype(np.int64),
            "active_score_rows": score_valid.astype(np.int64),
            "active_prediction_sum": np.where(score_valid, prediction, 0.0),
            "active_label_sum": np.where(score_valid, label, 0.0),
            "active_prediction_sq_sum": np.where(score_valid, prediction * prediction, 0.0),
            "active_label_sq_sum": np.where(score_valid, label * label, 0.0),
            "active_prediction_label_sum": np.where(score_valid, prediction * label, 0.0),
            "active_error_sum": np.where(score_valid, prediction - label, 0.0),
            "active_abs_error_sum": np.where(score_valid, np.abs(prediction - label), 0.0),
            "active_sq_error_sum": np.where(score_valid, (prediction - label) ** 2, 0.0),
            "active_economic_rows": economic_valid.astype(np.int64),
            "active_economic_sum": np.where(economic_valid, economic, 0.0),
            "active_economic_sq_sum": np.where(economic_valid, economic * economic, 0.0),
            "active_prediction_economic_rows": prediction_economic_valid.astype(np.int64),
            "active_pair_prediction_sum": np.where(prediction_economic_valid, prediction, 0.0),
            "active_pair_prediction_sq_sum": np.where(prediction_economic_valid, prediction * prediction, 0.0),
            "active_pair_economic_sum": np.where(prediction_economic_valid, economic, 0.0),
            "active_pair_economic_sq_sum": np.where(prediction_economic_valid, economic * economic, 0.0),
            "active_prediction_economic_sum": np.where(prediction_economic_valid, prediction * economic, 0.0),
        }
    )
    return out


def _sufficient_statistics(work: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if work.empty:
        return pd.DataFrame(columns=[*keys, *_STAT_COLUMNS])
    return work.groupby(keys, observed=True, sort=False)[list(_STAT_COLUMNS)].sum().reset_index()


def _combine_statistics(parts: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    nonempty = [part for part in parts if not part.empty]
    if not nonempty:
        return pd.DataFrame(columns=[*keys, *_STAT_COLUMNS])
    return (
        pd.concat(nonempty, ignore_index=True)
        .groupby(keys, observed=True, sort=False)[list(_STAT_COLUMNS)]
        .sum()
        .reset_index()
    )


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(len(numerator), np.nan, dtype=np.float64)
    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def _finalize_statistics(statistics: pd.DataFrame) -> pd.DataFrame:
    if statistics.empty:
        return statistics.copy()
    out = statistics.copy()
    score_rows = out["active_score_rows"].to_numpy(dtype=np.float64)
    economic_rows = out["active_economic_rows"].to_numpy(dtype=np.float64)
    prediction_economic_rows = out["active_prediction_economic_rows"].to_numpy(dtype=np.float64)
    rows = out["row_support"].to_numpy(dtype=np.float64)
    activation_rows = out["activation_valid_rows"].to_numpy(dtype=np.float64)

    out["activation_mean"] = _safe_divide(out["activation_sum"].to_numpy(float), activation_rows)
    out["activation_missing_rows"] = out["row_support"] - out["activation_valid_rows"]
    out["active_share"] = _safe_divide(out["active_rows"].to_numpy(float), rows)
    out["active_prediction_mean"] = _safe_divide(out["active_prediction_sum"].to_numpy(float), score_rows)
    out["active_label_mean"] = _safe_divide(out["active_label_sum"].to_numpy(float), score_rows)
    out["active_economic_mean"] = _safe_divide(out["active_economic_sum"].to_numpy(float), economic_rows)

    prediction_mean = out["active_prediction_mean"].to_numpy(float)
    label_mean = out["active_label_mean"].to_numpy(float)
    prediction_var = _safe_divide(out["active_prediction_sq_sum"].to_numpy(float), score_rows) - prediction_mean**2
    label_var = _safe_divide(out["active_label_sq_sum"].to_numpy(float), score_rows) - label_mean**2
    covariance = _safe_divide(out["active_prediction_label_sum"].to_numpy(float), score_rows) - prediction_mean * label_mean
    prediction_var = np.maximum(prediction_var, 0.0)
    label_var = np.maximum(label_var, 0.0)
    denominator = np.sqrt(prediction_var * label_var)
    correlation = np.full(len(out), np.nan, dtype=np.float64)
    valid_correlation = denominator > EPSILON
    correlation[valid_correlation] = covariance[valid_correlation] / denominator[valid_correlation]
    slope = np.full(len(out), np.nan, dtype=np.float64)
    valid_slope = prediction_var > EPSILON
    slope[valid_slope] = covariance[valid_slope] / prediction_var[valid_slope]

    out["prediction_label_pearson"] = correlation
    out["calibration_slope"] = slope
    out["calibration_intercept"] = label_mean - slope * prediction_mean
    out["calibration_signed_error"] = _safe_divide(out["active_error_sum"].to_numpy(float), score_rows)
    out["calibration_mae"] = _safe_divide(out["active_abs_error_sum"].to_numpy(float), score_rows)
    out["calibration_rmse"] = np.sqrt(_safe_divide(out["active_sq_error_sum"].to_numpy(float), score_rows))
    out["label_standard_deviation"] = np.sqrt(label_var)
    out["normalized_calibration_bias"] = np.abs(out["calibration_signed_error"].to_numpy(float)) / np.maximum(
        out["label_standard_deviation"].to_numpy(float), EPSILON
    )

    economic_mean = out["active_economic_mean"].to_numpy(float)
    economic_var = _safe_divide(out["active_economic_sq_sum"].to_numpy(float), economic_rows) - economic_mean**2
    economic_var = np.maximum(economic_var, 0.0)
    out["active_economic_standard_deviation"] = np.sqrt(economic_var)
    prediction_economic_covariance = _safe_divide(
        out["active_prediction_economic_sum"].to_numpy(float), prediction_economic_rows
    ) - (
        _safe_divide(out["active_pair_prediction_sum"].to_numpy(float), prediction_economic_rows)
        * _safe_divide(out["active_pair_economic_sum"].to_numpy(float), prediction_economic_rows)
    )
    # The two individual means above use the same pairwise-valid support as
    # the product, so this remains a proper pairwise Pearson diagnostic.
    pair_prediction_sq = _safe_divide(out["active_pair_prediction_sq_sum"].to_numpy(float), prediction_economic_rows)
    pair_prediction_mean = _safe_divide(out["active_pair_prediction_sum"].to_numpy(float), prediction_economic_rows)
    pair_economic_sq = _safe_divide(out["active_pair_economic_sq_sum"].to_numpy(float), prediction_economic_rows)
    pair_economic_mean = _safe_divide(out["active_pair_economic_sum"].to_numpy(float), prediction_economic_rows)
    pair_denominator = np.sqrt(
        np.maximum(pair_prediction_sq - pair_prediction_mean**2, 0.0)
        * np.maximum(pair_economic_sq - pair_economic_mean**2, 0.0)
    )
    prediction_economic_correlation = np.full(len(out), np.nan, dtype=np.float64)
    valid_pair_correlation = pair_denominator > EPSILON
    prediction_economic_correlation[valid_pair_correlation] = (
        prediction_economic_covariance[valid_pair_correlation] / pair_denominator[valid_pair_correlation]
    )
    out["prediction_economic_pearson"] = prediction_economic_correlation
    return out


def _support_breadth(
    summary: pd.DataFrame,
    period_health: pd.DataFrame,
    month_health: pd.DataFrame,
    keys: list[str],
) -> pd.DataFrame:
    out = summary.copy()
    for source, prefix in ((period_health, "period"), (month_health, "month")):
        if source.empty:
            breadth = pd.DataFrame(columns=[*keys, f"{prefix}_support", f"active_{prefix}_support"])
        else:
            local = source.copy()
            local["__active_unit__"] = local["active_rows"].gt(0).astype(np.int64)
            breadth = (
                local.groupby(keys, observed=True, sort=False)
                .agg(
                    **{
                        f"{prefix}_support": (prefix, "size"),
                        f"active_{prefix}_support": ("__active_unit__", "sum"),
                    }
                )
                .reset_index()
            )
        out = out.merge(breadth, on=keys, how="left", validate="one_to_one")
    return out


def _classify_within_fold_health(
    frame: pd.DataFrame,
    config: LeafReasoningHealthConfig,
) -> pd.DataFrame:
    out = frame.copy()
    out["health_row_support_pass"] = out["row_support"].ge(int(config.minimum_rows))
    out["health_activation_pass"] = (
        out["active_rows"].ge(int(config.minimum_active_rows))
        & out["active_share"].ge(float(config.minimum_activation_share))
    )
    out["health_period_support_pass"] = out["active_period_support"].ge(int(config.minimum_active_periods))
    out["health_month_support_pass"] = out["active_month_support"].ge(int(config.minimum_active_months))
    out["health_score_support_pass"] = out["active_score_rows"].ge(int(config.minimum_score_rows))
    out["health_economic_support_pass"] = out["active_economic_rows"].ge(int(config.minimum_economic_rows))
    out["health_discrimination_pass"] = (
        out["prediction_label_pearson"].notna()
        & out["prediction_label_pearson"].ge(float(config.minimum_prediction_label_correlation))
    )
    out["health_calibration_pass"] = (
        out["normalized_calibration_bias"].notna()
        & out["normalized_calibration_bias"].le(float(config.maximum_normalized_calibration_bias))
    )
    out["health_economic_pass"] = (
        out["active_economic_mean"].notna()
        & out["active_economic_mean"].ge(float(config.minimum_active_economic_mean))
    )

    conditions = [
        ~out["health_row_support_pass"],
        ~out["health_activation_pass"],
        ~out["health_period_support_pass"],
        ~out["health_month_support_pass"],
        ~out["health_score_support_pass"],
        ~out["health_economic_support_pass"],
        out["prediction_label_pearson"].lt(0.0),
        ~out["health_discrimination_pass"],
        ~out["health_calibration_pass"],
        ~out["health_economic_pass"],
    ]
    labels = [
        "INSUFFICIENT_ROW_SUPPORT",
        "LOW_ACTIVATION_OR_ACTIVE_SUPPORT",
        "INSUFFICIENT_PERIOD_SUPPORT",
        "INSUFFICIENT_MONTH_SUPPORT",
        "INSUFFICIENT_SCORE_SUPPORT",
        "INSUFFICIENT_ECONOMIC_SUPPORT",
        "PREDICTION_INVERTED",
        "LOW_PREDICTION_LABEL_DISCRIMINATION",
        "CALIBRATION_BIAS",
        "ECONOMICALLY_ADVERSE",
    ]
    out["within_fold_health"] = np.select(conditions, labels, default="HEALTHY")
    out["health_eligible"] = out["within_fold_health"].eq("HEALTHY")
    return out


def analyze_leaf_reasoning_health(
    assignments: pd.DataFrame | Iterable[pd.DataFrame],
    *,
    columns: LeafReasoningHealthColumns = LeafReasoningHealthColumns(),
    config: LeafReasoningHealthConfig = LeafReasoningHealthConfig(),
) -> LeafReasoningHealthResult:
    """Summarize strict-OOF leaf health without cross-fold token alignment.

    ``assignments`` may be a DataFrame or a chunk iterator.  Every row must
    describe one candidate's assignment to one raw leaf for one ``head``,
    ``side`` and ``fold``.  Only the supplied label/prediction/economic values
    are audited; no transform is fitted and no later information is used.
    """

    if not 0.0 <= float(config.activation_threshold) <= 1.0:
        raise ValueError("activation_threshold must be in [0, 1]")
    if config.minimum_rows < 1 or config.minimum_active_rows < 1:
        raise ValueError("minimum row support must be positive")
    chunk_iterator = iter((assignments,)) if isinstance(assignments, pd.DataFrame) else iter(assignments)
    leaf_parts: list[pd.DataFrame] = []
    period_parts: list[pd.DataFrame] = []
    month_parts: list[pd.DataFrame] = []
    leaf_keys = ["head", "side_name", "fold", "leaf_token"]
    seen_chunk = False
    for chunk in chunk_iterator:
        seen_chunk = True
        prepared = _prepare_chunk(chunk, columns=columns, config=config)
        leaf_parts.append(_sufficient_statistics(prepared, leaf_keys))
        period_parts.append(_sufficient_statistics(prepared, [*leaf_keys, "period"]))
        month_parts.append(_sufficient_statistics(prepared, [*leaf_keys, "month"]))
    if not seen_chunk:
        empty = pd.DataFrame()
        return LeafReasoningHealthResult(empty, empty.copy(), empty.copy())
    leaf_statistics = _combine_statistics(leaf_parts, leaf_keys)
    period_statistics = _combine_statistics(period_parts, [*leaf_keys, "period"])
    month_statistics = _combine_statistics(month_parts, [*leaf_keys, "month"])
    period_health = _finalize_statistics(period_statistics)
    month_health = _finalize_statistics(month_statistics)
    leaf_health = _support_breadth(
        _finalize_statistics(leaf_statistics), period_health, month_health, leaf_keys
    )
    leaf_health = _classify_within_fold_health(leaf_health, config)
    return LeafReasoningHealthResult(leaf_health, period_health, month_health)


__all__ = [
    "LeafReasoningHealthColumns",
    "LeafReasoningHealthConfig",
    "LeafReasoningHealthResult",
    "analyze_leaf_reasoning_health",
]
