"""Bounded, time-ordered regime diagnostics for fixed execution-EV models.

This module deliberately separates a valid forward rolling control from an
intentionally invalid reverse-time stress diagnostic.  The latter can be useful
for detecting regime sensitivity, but it is never OOS evidence and never a
promotion input.  Callers supply the fixed model's fit/predict function so this
tool does not silently retune a winning configuration.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from .execution_ev_meta import execution_ev_metrics


EXECUTION_EV_REGIME_DIAGNOSIS_SCHEMA = "execution_ev_regime_diagnosis_v1"
SelectionMode = Literal[
    "forward_rolling",
    "forward_rolling_matched_reverse_size",
    "reversed_month_diagnostic",
]
FitPredictHook = Callable[
    [pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray | None],
    Sequence[float] | np.ndarray,
]
SampleWeightHook = Callable[[pd.DataFrame], Sequence[float] | np.ndarray | None]


@dataclass(frozen=True)
class RegimeDiagnosisConfig:
    """Fixed temporal and economic contracts for a bounded diagnostic run."""

    decision_time_col: str = "execution_decision_utc"
    label_resolution_col: str = "execution_label_end_utc"
    target_col: str = "execution_net_ev_12h"
    train_window_months: int = 3
    purge_hours: float = 12.0
    min_train_rows: int = 100
    top_k_fraction: float = 0.10
    huber_delta: float = 0.01
    max_periods: int | None = 6
    random_state: int = 42


@dataclass(frozen=True)
class RegimeDiagnosisSplit:
    """One fixed-month evaluation with either past or diagnostic future fit rows."""

    mode: SelectionMode
    evaluation_month: str
    evaluation_start: pd.Timestamp
    evaluation_end: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    train_positions: np.ndarray
    evaluation_positions: np.ndarray
    forward_reference_train_rows: int
    matched_size_control: bool

    @property
    def is_oos(self) -> bool:
        return self.mode in (
            "forward_rolling",
            "forward_rolling_matched_reverse_size",
        )

    @property
    def evaluation_status(self) -> str:
        if self.mode == "forward_rolling":
            return "forward_rolling_oos_control"
        if self.mode == "forward_rolling_matched_reverse_size":
            return "forward_rolling_oos_matched_size_diagnostic"
        return "diagnostic_non_oos_reversed_training"


@dataclass
class RegimeDiagnosisResult:
    """Per-period metrics and row-level predictions for audit/review."""

    metrics: pd.DataFrame
    predictions: pd.DataFrame
    splits: pd.DataFrame
    config: RegimeDiagnosisConfig


def _utc(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(f"regime diagnosis is missing required column {column!r}")
    converted = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if converted.isna().any():
        raise ValueError(f"regime diagnosis {column!r} contains invalid timestamps")
    return pd.Series(converted, index=frame.index)


def _month_start(value: str | pd.Period | pd.Timestamp) -> pd.Timestamp:
    if isinstance(value, pd.Period):
        value = value.start_time
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    naive_utc = timestamp.tz_localize(None)
    return naive_utc.to_period("M").start_time.tz_localize("UTC")


def _month_end(start: pd.Timestamp) -> pd.Timestamp:
    return start + pd.offsets.MonthBegin(1)


def _evenly_spaced_positions(positions: np.ndarray, size: int) -> np.ndarray:
    """Deterministically shrink a future control without random sampling."""

    if size <= 0 or size > len(positions):
        raise ValueError("matched control size must be within the source population")
    if size == len(positions):
        return positions.copy()
    indices = np.floor(np.linspace(0, len(positions), num=size, endpoint=False)).astype(
        int
    )
    return positions[indices]


def validate_regime_diagnosis_input(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    config: RegimeDiagnosisConfig = RegimeDiagnosisConfig(),
) -> pd.DataFrame:
    """Validate fixed inputs and return a chronological, positional work frame.

    Forward controls require both a decision timestamp and the true label
    resolution timestamp.  A prior decision whose path is still unresolved at
    an evaluation boundary is not eligible to train that control.
    """

    if config.train_window_months < 1:
        raise ValueError("train_window_months must be at least one")
    if config.purge_hours < 0.0:
        raise ValueError("purge_hours must be non-negative")
    if config.min_train_rows < 1:
        raise ValueError("min_train_rows must be positive")
    if config.max_periods is not None and config.max_periods < 1:
        raise ValueError("max_periods must be positive when supplied")
    if not 0.0 < config.top_k_fraction <= 1.0:
        raise ValueError("top_k_fraction must be in (0, 1]")
    if config.huber_delta <= 0.0:
        raise ValueError("huber_delta must be positive")

    names = tuple(dict.fromkeys(map(str, feature_columns)))
    if not names:
        raise ValueError("regime diagnosis requires at least one fixed feature column")
    required = [
        config.decision_time_col,
        config.label_resolution_col,
        config.target_col,
        *names,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("regime diagnosis is missing columns: " + ", ".join(missing))

    work = frame.copy().reset_index(drop=False).rename(
        columns={"index": "__source_index__"}
    )
    if work["__source_index__"].duplicated().any():
        # The original index is audit metadata only; reset it if callers passed
        # a duplicated index rather than rejecting otherwise valid rows.
        work["__source_index__"] = np.arange(len(work), dtype=np.int64)
    work[config.decision_time_col] = _utc(work, config.decision_time_col)
    work[config.label_resolution_col] = _utc(work, config.label_resolution_col)
    if (work[config.label_resolution_col] < work[config.decision_time_col]).any():
        raise ValueError("label resolution cannot precede the execution decision")

    target = pd.to_numeric(work[config.target_col], errors="coerce")
    if not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError(
            f"regime diagnosis target {config.target_col!r} must be finite"
        )
    work[config.target_col] = target.astype("float64")
    features = work.loc[:, list(names)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError(
            "regime diagnosis feature columns must be finite numeric values"
        )
    work.loc[:, list(names)] = features.astype("float64")
    return work.sort_values(
        [config.decision_time_col, "__source_index__"], kind="mergesort"
    ).reset_index(drop=True)


def build_regime_diagnosis_splits(
    frame: pd.DataFrame,
    *,
    config: RegimeDiagnosisConfig = RegimeDiagnosisConfig(),
    start_month: str | pd.Timestamp | None = None,
    end_month: str | pd.Timestamp | None = None,
) -> list[RegimeDiagnosisSplit]:
    """Build paired forward and future-trained diagnostic month controls.

    The forward split is the only OOS control: decisions and resolved labels are
    both strictly before the evaluation month, with an additional decision-time
    purge.  The reverse split uses only rows after that month and is sampled to
    exactly the forward train-row count; it is deliberately marked non-OOS.
    """

    decision = _utc(frame, config.decision_time_col)
    resolved = _utc(frame, config.label_resolution_col)
    all_months = pd.PeriodIndex(
        decision.dt.tz_localize(None).dt.to_period("M")
    ).unique().sort_values()
    months = all_months
    if start_month is not None:
        start_period = _month_start(start_month).to_period("M")
        months = months[months >= start_period]
    if end_month is not None:
        end_period = _month_start(end_month).to_period("M")
        months = months[months <= end_period]
    if config.max_periods is not None:
        months = months[: int(config.max_periods)]

    purge = pd.Timedelta(hours=float(config.purge_hours))
    splits: list[RegimeDiagnosisSplit] = []
    for period in months:
        required_history = {
            period - offset for offset in range(1, int(config.train_window_months) + 1)
        }
        if not required_history.issubset(set(all_months)):
            # A rolling two-month control should not silently become a one-month
            # control at the leading edge of a bounded calendar.
            continue
        evaluation_start = _month_start(period)
        evaluation_end = _month_end(evaluation_start)
        rolling_start = evaluation_start - pd.DateOffset(
            months=config.train_window_months
        )
        evaluation_positions = np.flatnonzero(
            (decision >= evaluation_start).to_numpy()
            & (decision < evaluation_end).to_numpy()
        )
        forward_positions = np.flatnonzero(
            (decision >= rolling_start).to_numpy()
            & (decision < evaluation_start - purge).to_numpy()
            & (resolved < evaluation_start).to_numpy()
        )
        if (
            len(evaluation_positions) == 0
            or len(forward_positions) < config.min_train_rows
        ):
            continue
        splits.append(
            RegimeDiagnosisSplit(
                mode="forward_rolling",
                evaluation_month=str(period),
                evaluation_start=evaluation_start,
                evaluation_end=evaluation_end,
                train_start=rolling_start,
                train_end=evaluation_start,
                train_positions=forward_positions,
                evaluation_positions=evaluation_positions,
                forward_reference_train_rows=len(forward_positions),
                matched_size_control=False,
            )
        )

        # This population is intentionally future-trained and therefore
        # non-OOS.  Starting at the next month boundary preserves the matched
        # rolling-month span; the forward purge remains the actual safety rule.
        reverse_start = evaluation_end
        reverse_end = evaluation_end + pd.DateOffset(months=config.train_window_months)
        reverse_pool = np.flatnonzero(
            (decision >= reverse_start).to_numpy() & (decision < reverse_end).to_numpy()
        )
        if len(reverse_pool) < config.min_train_rows:
            continue
        matched_size = min(len(reverse_pool), len(forward_positions))
        if matched_size < len(forward_positions):
            matched_forward_positions = _evenly_spaced_positions(
                forward_positions, matched_size
            )
            splits.append(
                RegimeDiagnosisSplit(
                    mode="forward_rolling_matched_reverse_size",
                    evaluation_month=str(period),
                    evaluation_start=evaluation_start,
                    evaluation_end=evaluation_end,
                    train_start=rolling_start,
                    train_end=evaluation_start,
                    train_positions=matched_forward_positions,
                    evaluation_positions=evaluation_positions,
                    forward_reference_train_rows=len(forward_positions),
                    matched_size_control=True,
                )
            )
        reverse_positions = _evenly_spaced_positions(
            reverse_pool, matched_size
        )
        splits.append(
            RegimeDiagnosisSplit(
                mode="reversed_month_diagnostic",
                evaluation_month=str(period),
                evaluation_start=evaluation_start,
                evaluation_end=evaluation_end,
                train_start=reverse_start,
                train_end=reverse_end,
                train_positions=reverse_positions,
                evaluation_positions=evaluation_positions,
                forward_reference_train_rows=len(forward_positions),
                matched_size_control=True,
            )
        )
    return splits


def split_audit_frame(splits: Sequence[RegimeDiagnosisSplit]) -> pd.DataFrame:
    """Render the split-level temporal contracts without any model fitting."""

    rows: list[dict[str, Any]] = []
    for split in splits:
        rows.append(
            {
                "mode": split.mode,
                "evaluation_month": split.evaluation_month,
                "evaluation_status": split.evaluation_status,
                "is_oos": split.is_oos,
                "promotion_eligible": False,
                "training_direction": (
                    "future_to_past"
                    if split.mode == "reversed_month_diagnostic"
                    else "past_to_future"
                ),
                "matched_size_control": split.matched_size_control,
                "train_rows": int(len(split.train_positions)),
                "forward_reference_train_rows": int(split.forward_reference_train_rows),
                "evaluation_rows": int(len(split.evaluation_positions)),
                "train_start_utc": split.train_start.isoformat(),
                "train_end_utc": split.train_end.isoformat(),
                "evaluation_start_utc": split.evaluation_start.isoformat(),
                "evaluation_end_utc": split.evaluation_end.isoformat(),
                "selection_basis": "global_topk_across_full_evaluation_period",
            }
        )
    return pd.DataFrame(rows)


def feature_regime_diagnostics(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    config: RegimeDiagnosisConfig = RegimeDiagnosisConfig(),
) -> pd.DataFrame:
    """Measure month-to-month feature drift and economic relationship changes."""

    names = tuple(dict.fromkeys(map(str, feature_columns)))
    work = validate_regime_diagnosis_input(frame, names, config=config)
    month = _utc(work, config.decision_time_col).dt.strftime("%Y-%m")
    target = work[config.target_col].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    prior: dict[str, dict[str, float | str]] = {}
    for month_name in sorted(month.unique()):
        mask = month.eq(month_name).to_numpy()
        y = target[mask]
        for feature in names:
            x = work.loc[mask, feature].to_numpy(dtype=float)
            std = float(np.std(x, ddof=0))
            rho = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
            if not np.isfinite(rho):
                rho = 0.0
            current = {
                "month": str(month_name),
                "mean": float(np.mean(x)),
                "std": std,
                "target_spearman": rho,
            }
            previous = prior.get(feature)
            if previous is None:
                standardized_mean_shift = 0.0
                spearman_delta = 0.0
                sign_flip = False
                previous_month = None
            else:
                scale = np.sqrt(
                    (
                        float(previous["std"]) ** 2
                        + float(current["std"]) ** 2
                    )
                    / 2.0
                )
                standardized_mean_shift = (
                    float(current["mean"]) - float(previous["mean"])
                ) / max(float(scale), 1e-12)
                spearman_delta = float(current["target_spearman"]) - float(
                    previous["target_spearman"]
                )
                sign_flip = (
                    float(current["target_spearman"])
                    * float(previous["target_spearman"])
                    < 0.0
                )
                previous_month = str(previous["month"])
            rows.append(
                {
                    "month": str(month_name),
                    "previous_month": previous_month,
                    "feature": feature,
                    "rows": int(mask.sum()),
                    "mean": float(current["mean"]),
                    "std": float(current["std"]),
                    "target_spearman": float(current["target_spearman"]),
                    "standardized_mean_shift_vs_previous": float(
                        standardized_mean_shift
                    ),
                    "target_spearman_delta_vs_previous": float(spearman_delta),
                    "target_spearman_sign_flip": bool(sign_flip),
                }
            )
            prior[feature] = current
    return pd.DataFrame(rows)


def _training_weights(
    train: pd.DataFrame,
    sample_weight_hook: SampleWeightHook | None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if sample_weight_hook is None:
        return None, {
            "sample_weighting": "none",
            "sample_weight_sum": float(len(train)),
        }
    values = sample_weight_hook(train.copy())
    if values is None:
        return None, {
            "sample_weighting": "hook_returned_none",
            "sample_weight_sum": float(len(train)),
        }
    weights = np.asarray(values, dtype=float).reshape(-1)
    if weights.shape[0] != len(train):
        raise ValueError(
            "sample_weight_hook must return exactly one weight per train row"
        )
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError("sample weights must be finite and non-negative")
    if float(weights.sum()) <= 0.0:
        raise ValueError("sample weights must have positive total weight")
    return weights, {
        "sample_weighting": "training_only_hook",
        "sample_weight_sum": float(weights.sum()),
    }


def _assert_forward_split_is_safe(
    train: pd.DataFrame,
    split: RegimeDiagnosisSplit,
    config: RegimeDiagnosisConfig,
) -> None:
    decision = _utc(train, config.decision_time_col)
    resolved = _utc(train, config.label_resolution_col)
    purge_cutoff = split.evaluation_start - pd.Timedelta(
        hours=float(config.purge_hours)
    )
    if not (
        (decision < purge_cutoff).all()
        and (resolved < split.evaluation_start).all()
    ):
        raise RuntimeError(
            "forward diagnosis split violates its purge/resolution contract"
        )


def evaluate_regime_diagnosis(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    fit_predict: FitPredictHook,
    *,
    config: RegimeDiagnosisConfig = RegimeDiagnosisConfig(),
    sample_weight_hook: SampleWeightHook | None = None,
    start_month: str | pd.Timestamp | None = None,
    end_month: str | pd.Timestamp | None = None,
) -> RegimeDiagnosisResult:
    """Fit fixed models on each control and calculate global-top-k metrics.

    ``fit_predict`` receives train features/targets and evaluation features only.
    Any optional weighting hook receives only the current train frame; realised
    evaluation returns never influence fitting or weights.
    """

    names = tuple(dict.fromkeys(map(str, feature_columns)))
    work = validate_regime_diagnosis_input(frame, names, config=config)
    splits = build_regime_diagnosis_splits(
        work, config=config, start_month=start_month, end_month=end_month
    )
    if not splits:
        raise ValueError(
            "no regime-diagnosis splits satisfy the fixed temporal contract"
        )

    metric_rows: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    for split in splits:
        train = work.iloc[split.train_positions].copy()
        evaluation = work.iloc[split.evaluation_positions].copy()
        if split.is_oos:
            _assert_forward_split_is_safe(train, split, config)
        weights, weight_audit = _training_weights(train, sample_weight_hook)
        prediction = np.asarray(
            fit_predict(
                train.loc[:, list(names)],
                train[config.target_col].to_numpy(dtype=float),
                evaluation.loc[:, list(names)],
                weights,
            ),
            dtype=float,
        ).reshape(-1)
        if len(prediction) != len(evaluation) or not np.isfinite(prediction).all():
            raise ValueError(
                "fit_predict must return one finite prediction per evaluation row"
            )
        metrics = execution_ev_metrics(
            evaluation[config.target_col].to_numpy(dtype=float),
            prediction,
            top_k_fraction=config.top_k_fraction,
            huber_delta=config.huber_delta,
        )
        row = {
            "mode": split.mode,
            "evaluation_month": split.evaluation_month,
            "evaluation_status": split.evaluation_status,
            "is_oos": split.is_oos,
            "promotion_eligible": False,
            "training_direction": (
                "future_to_past"
                if split.mode == "reversed_month_diagnostic"
                else "past_to_future"
            ),
            "matched_size_control": split.matched_size_control,
            "train_rows": int(len(train)),
            "forward_reference_train_rows": int(split.forward_reference_train_rows),
            "evaluation_rows": int(len(evaluation)),
            "train_start_utc": split.train_start.isoformat(),
            "train_end_utc": split.train_end.isoformat(),
            "evaluation_start_utc": split.evaluation_start.isoformat(),
            "evaluation_end_utc": split.evaluation_end.isoformat(),
            "max_train_label_resolution_utc": _utc(
                train, config.label_resolution_col
            ).max().isoformat(),
            "selection_basis": "global_topk_across_full_evaluation_period",
            **weight_audit,
            **metrics,
        }
        metric_rows.append(row)
        prediction_parts.append(
            pd.DataFrame(
                {
                    "source_row": evaluation["__source_index__"].to_numpy(),
                    "mode": split.mode,
                    "evaluation_month": split.evaluation_month,
                    "evaluation_status": split.evaluation_status,
                    "is_oos": split.is_oos,
                    "promotion_eligible": False,
                    "prediction": prediction,
                    "realized_net_ev": evaluation[config.target_col].to_numpy(
                        dtype=float
                    ),
                }
            )
        )
    return RegimeDiagnosisResult(
        metrics=pd.DataFrame(metric_rows),
        predictions=pd.concat(prediction_parts, ignore_index=True),
        splits=split_audit_frame(splits),
        config=config,
    )


def regime_diagnosis_manifest(
    result: RegimeDiagnosisResult | None,
    *,
    config: RegimeDiagnosisConfig,
    feature_columns: Sequence[str],
    configuration_name: str,
    split_count: int,
) -> dict[str, Any]:
    """Return a self-contained audit manifest for dry or executed runs."""

    return {
        "schema": EXECUTION_EV_REGIME_DIAGNOSIS_SCHEMA,
        "configuration_name": str(configuration_name),
        "config": asdict(config),
        "feature_columns": list(feature_columns),
        "split_count": int(split_count),
        "forward_control_contract": (
            "train decision timestamp is before evaluation start minus purge and "
            "train label resolution is strictly before evaluation start"
        ),
        "reversed_training_contract": (
            "diagnostic_non_oos_reversed_training; future-to-past fits are never "
            "promotion evidence and are row-count matched to their forward control"
        ),
        "selection_basis": "global_topk_across_full_evaluation_period",
        "sample_weight_contract": "optional hooks receive training rows only",
        "status": "completed" if result is not None else "dry_run_planned",
        "promotion_eligible": False,
    }
