"""Leakage-explicit outcome labels for breakout path quality.

The inputs are realized post-entry outcomes used only as training labels. They
must never be included in the pre-entry feature matrix.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BreakoutPathQualityThresholds:
    retention_low: float
    efficiency_low: float
    participation_low: float
    reversal_high: float
    fit_rows: int
    lower_quantile: float = 0.25
    upper_quantile: float = 0.75


@dataclass(frozen=True)
class SevereRetentionThreshold:
    """Train-only lower-tail cutoff for economically severe retention failure."""

    capture_net_low: float
    fit_rows: int
    lower_quantile: float = 0.10


OUTCOME_COLUMNS = {
    "retention": "breakout_retention_outcome",
    "efficiency": "breakout_path_efficiency_outcome",
    "participation": "breakout_participation_outcome",
    "reversal": "breakout_reversal_magnitude_outcome",
}


def _finite(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float32)


def fit_breakout_path_quality_thresholds(
    train_outcomes: pd.DataFrame,
    *,
    lower_quantile: float = 0.25,
    upper_quantile: float = 0.75,
) -> BreakoutPathQualityThresholds:
    """Fit outcome cutoffs on an authorized training fold only."""

    missing = [column for column in OUTCOME_COLUMNS.values() if column not in train_outcomes]
    if missing:
        raise KeyError(f"Missing breakout outcome columns: {missing}")
    if not 0.0 < lower_quantile < upper_quantile < 1.0:
        raise ValueError("Expected 0 < lower_quantile < upper_quantile < 1")
    values = {name: _finite(train_outcomes, column) for name, column in OUTCOME_COLUMNS.items()}
    complete = np.logical_and.reduce([np.isfinite(value) for value in values.values()])
    if int(complete.sum()) < 100:
        raise ValueError("At least 100 complete train outcomes are required")
    return BreakoutPathQualityThresholds(
        retention_low=float(np.quantile(values["retention"][complete], lower_quantile)),
        efficiency_low=float(np.quantile(values["efficiency"][complete], lower_quantile)),
        participation_low=float(np.quantile(values["participation"][complete], lower_quantile)),
        reversal_high=float(np.quantile(values["reversal"][complete], upper_quantile)),
        fit_rows=int(complete.sum()),
        lower_quantile=float(lower_quantile),
        upper_quantile=float(upper_quantile),
    )


def materialize_breakout_path_quality_labels(
    outcomes: pd.DataFrame,
    thresholds: BreakoutPathQualityThresholds,
) -> pd.DataFrame:
    """Apply frozen train-derived cutoffs to train, OOF, or OOS outcomes."""

    values = {name: _finite(outcomes, column) for name, column in OUTCOME_COLUMNS.items()}
    valid = np.logical_and.reduce([np.isfinite(value) for value in values.values()])
    retention = valid & (values["retention"] <= thresholds.retention_low)
    efficiency = valid & (values["efficiency"] <= thresholds.efficiency_low)
    participation = valid & (values["participation"] <= thresholds.participation_low)
    reversal = valid & (values["reversal"] >= thresholds.reversal_high)
    output = pd.DataFrame(index=outcomes.index)
    output["breakout_quality_label_valid"] = valid.astype(np.int8)
    output["breakout_retention_failure"] = retention.astype(np.int8)
    output["breakout_low_efficiency"] = efficiency.astype(np.int8)
    output["breakout_participation_failure"] = participation.astype(np.int8)
    output["breakout_rapid_reversal"] = reversal.astype(np.int8)
    output["breakout_path_quality_failure_count"] = (
        retention.astype(np.int8)
        + efficiency.astype(np.int8)
        + participation.astype(np.int8)
        + reversal.astype(np.int8)
    )
    output["breakout_any_path_quality_failure"] = (
        output["breakout_path_quality_failure_count"].gt(0) & valid
    ).astype(np.int8)
    output["breakout_path_quality_soft_risk"] = (
        output["breakout_path_quality_failure_count"].to_numpy(np.float32) / 4.0
    )
    return output


def fit_severe_retention_threshold(
    train_capture_net: pd.Series,
    *,
    lower_quantile: float = 0.10,
) -> SevereRetentionThreshold:
    """Fit an economic lower-tail cutoff on an authorized training fold only."""

    if not 0.0 < lower_quantile < 0.5:
        raise ValueError("Expected 0 < lower_quantile < 0.5")
    values = pd.to_numeric(train_capture_net, errors="coerce").to_numpy(np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 100:
        raise ValueError("At least 100 finite capture-net rows are required")
    return SevereRetentionThreshold(
        capture_net_low=float(np.quantile(values, lower_quantile)),
        fit_rows=int(len(values)),
        lower_quantile=float(lower_quantile),
    )


def materialize_severe_retention_failure(
    trailing_success: pd.Series,
    capture_net: pd.Series,
    threshold: SevereRetentionThreshold,
) -> pd.Series:
    """Mark failed trailing paths in the train-defined worst capture-net tail."""

    retention = pd.to_numeric(trailing_success, errors="coerce")
    capture = pd.to_numeric(capture_net, errors="coerce")
    valid = retention.notna() & capture.notna()
    return (
        valid
        & retention.le(0.0)
        & capture.le(threshold.capture_net_low)
    ).astype(np.int8)


def breakout_path_quality_label_manifest(
    thresholds: BreakoutPathQualityThresholds,
) -> dict[str, object]:
    return {
        "schema": "breakout_path_quality_labels_v1",
        "thresholds": asdict(thresholds),
        "outcome_columns": OUTCOME_COLUMNS,
        "leakage_contract": (
            "All four inputs are realized post-entry outcomes and are labels only. "
            "Thresholds are fitted on the authorized training fold and frozen for OOF/OOS. "
            "No outcome column may enter a pre-entry feature matrix."
        ),
        "deprecated_development_target": "bad_residual_event_target for breakout archetypes",
        "short_breakout_target_status": {
            "rapid_reversal": "primary_candidate",
            "severe_retention": "separate_train_only_tail_candidate",
            "retention_failure": "too_prevalent_for_initial_binary_model",
            "participation_failure": (
                "diagnostic_only_pending_true_breadth_redefinition; do not train alongside "
                "low_efficiency because the short-breakout audit found them redundant."
            ),
        },
    }


__all__ = [
    "BreakoutPathQualityThresholds",
    "SevereRetentionThreshold",
    "OUTCOME_COLUMNS",
    "breakout_path_quality_label_manifest",
    "fit_breakout_path_quality_thresholds",
    "fit_severe_retention_threshold",
    "materialize_breakout_path_quality_labels",
    "materialize_severe_retention_failure",
]
