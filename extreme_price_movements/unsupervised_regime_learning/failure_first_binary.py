"""Direct causal binary detector for model-failure onset and near-term risk.

This challenger deliberately avoids a learned multi-state taxonomy.  A health
bin is either ``stable`` or ``failure`` under the same strict-OOF economic
failure definition used by the failure-first pipeline.  The detector predicts:

* failure onset within the next three fully observed hours; and
* failure active now or reached within those three hours.

All target availability follows the exact resolved health-bin availability.
The model is therefore suitable for purged chronological OOF evaluation but
does not make the historical comparator equivalent to current-model OOF.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Sequence

import numpy as np
import pandas as pd

from extreme_price_movements.unsupervised_regime_learning.failure_first_detector import (
    validate_detector_features,
)
from extreme_price_movements.unsupervised_regime_learning.failure_first_hourly import (
    build_hourly_state_transition_labels,
)


BINARY_TARGET_COLUMNS = (
    "target__failure_onset_within_3h",
    "target__failure_active_or_within_3h",
)


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="raise")


def _slug(value: object) -> str:
    return (
        re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip())
        .strip("_")
        .casefold()
        or "signal"
    )


def build_hourly_binary_failure_targets(
    health: pd.DataFrame,
    *,
    health_bin_hours: int = 6,
    horizon_hours: int = 3,
) -> pd.DataFrame:
    """Create direct binary labels without fitting or consulting a taxonomy."""

    required = (
        "decision_bin_start_utc",
        "bin_available_utc",
        "evaluation_origin",
        "model_failure_bin",
    )
    missing = [name for name in required if name not in health]
    if missing:
        raise KeyError("binary failure targets missing: " + ", ".join(missing))
    if int(health_bin_hours) < 1 or int(horizon_hours) < 1:
        raise ValueError("health_bin_hours and horizon_hours must be positive")
    bins = health.loc[:, required].copy()
    bins["decision_bin_start_utc"] = _utc(
        bins["decision_bin_start_utc"]
    )
    bins["bin_available_utc"] = _utc(bins["bin_available_utc"])
    if bins.duplicated(
        ["decision_bin_start_utc", "evaluation_origin"]
    ).any():
        raise ValueError("binary targets require unique health bins per origin")
    rows: list[dict[str, Any]] = []
    for row in bins.itertuples(index=False):
        state = "failure" if bool(row.model_failure_bin) else "stable"
        for offset in range(int(health_bin_hours)):
            rows.append(
                {
                    "execution_decision_utc": pd.Timestamp(
                        row.decision_bin_start_utc
                    )
                    + pd.Timedelta(hours=offset),
                    "side_name": "global",
                    "evaluation_origin": row.evaluation_origin,
                    "binary_failure_state": state,
                    "state_available_utc": row.bin_available_utc,
                }
            )
    hourly = pd.DataFrame.from_records(rows)
    observed_through = _utc(hourly["state_available_utc"]).max()
    labels = build_hourly_state_transition_labels(
        hourly,
        state_col="binary_failure_state",
        state_available_col="state_available_utc",
        timestamp_col="execution_decision_utc",
        side_col="side_name",
        boundary_columns=("evaluation_origin",),
        horizon_hours=int(horizon_hours),
        observed_through=observed_through,
    )
    destination_col = f"target__destination_state_{int(horizon_hours)}h"
    transition_col = f"target__transition_within_{int(horizon_hours)}h"
    current = labels["target__current_state"]
    destination = labels[destination_col]
    future_known = labels[transition_col].notna() & destination.notna()
    labels["target__failure_active"] = np.where(
        current.notna(), current.eq("failure").astype(float), np.nan
    )
    labels[f"target__failure_onset_within_{int(horizon_hours)}h"] = np.where(
        future_known,
        (current.eq("stable") & destination.eq("failure")).astype(float),
        np.nan,
    )
    labels[
        f"target__failure_active_or_within_{int(horizon_hours)}h"
    ] = np.where(
        future_known,
        (
            current.eq("failure") | destination.eq("failure")
        ).astype(float),
        np.nan,
    )
    labels["binary_failure_label_available_at"] = pd.concat(
        [
            _utc(labels["target__current_state_label_resolution_utc"]),
            _utc(labels[f"target__future_label_resolution_utc"]),
        ],
        axis=1,
    ).max(axis=1)
    return labels


def add_causal_transition_deltas(
    frame: pd.DataFrame,
    *,
    signal_columns: Sequence[str],
    timestamp_col: str = "execution_decision_utc",
    group_columns: Sequence[str] = ("side_name", "evaluation_origin"),
    lags: Sequence[int] = (1, 3),
) -> tuple[pd.DataFrame, list[str]]:
    """Add exact-lag, past-only changes for a small observable signal set."""

    signals = list(dict.fromkeys(str(name) for name in signal_columns))
    required = [timestamp_col, *group_columns, *signals]
    missing = [name for name in required if name not in frame]
    if missing:
        raise KeyError("transition deltas missing: " + ", ".join(missing))
    if not signals:
        raise ValueError("transition deltas require observable signals")
    lag_values = sorted({int(value) for value in lags})
    if not lag_values or lag_values[0] < 1:
        raise ValueError("transition delta lags must be positive")
    output = frame.copy()
    output[timestamp_col] = _utc(output[timestamp_col])
    scope = [*group_columns, timestamp_col]
    if output.duplicated(scope).any():
        raise ValueError(
            "transition deltas require one row per group and timestamp"
        )
    output["__transition_original_order"] = np.arange(len(output))
    output = output.sort_values(
        [*group_columns, timestamp_col], kind="stable"
    )
    generated: list[str] = []
    grouped = output.groupby(list(group_columns), observed=True, sort=False)
    for signal in signals:
        numeric = pd.to_numeric(output[signal], errors="coerce")
        for lag in lag_values:
            previous = grouped[signal].shift(lag)
            previous_time = grouped[timestamp_col].shift(lag)
            exact = (
                output[timestamp_col] - previous_time
            ).eq(pd.Timedelta(hours=lag))
            name = f"failure_transition_delta_{lag}h__{_slug(signal)}"
            output[name] = (numeric - pd.to_numeric(
                previous, errors="coerce"
            )).where(exact)
            generated.append(name)
    output = output.sort_values(
        "__transition_original_order", kind="stable"
    ).drop(columns="__transition_original_order")
    return output, generated


@dataclass(frozen=True)
class BinaryFailureDetectorConfig:
    timestamp_col: str = "execution_decision_utc"
    label_available_col: str = "binary_failure_label_available_at"
    onset_target_col: str = "target__failure_onset_within_3h"
    risk_target_col: str = "target__failure_active_or_within_3h"
    first_eval_time: str = ""
    eval_hours: int = 720
    min_train_rows: int = 1_000
    min_positive_rows: int = 12
    max_features: int = 40
    learning_rate: float = 0.05
    max_iter: int = 120
    depth: int = 5
    l2_regularization: float = 5.0
    auto_class_weights: str | None = "Balanced"
    random_state: int = 20260726


@dataclass
class _BinaryHead:
    model: Any

    def probability(self, values: np.ndarray) -> np.ndarray:
        classes = [str(value).casefold() for value in self.model.classes_]
        positive = next(
            index
            for index, value in enumerate(classes)
            if value in {"1", "1.0", "true"}
        )
        return np.asarray(
            self.model.predict_proba(values)[:, positive], dtype=np.float32
        )


@dataclass
class BinaryFailureDetectorBundle:
    feature_columns: list[str]
    medians: np.ndarray
    onset_head: _BinaryHead
    risk_head: _BinaryHead
    train_rows: int
    train_end_exclusive: str
    train_label_available_max: str
    config: BinaryFailureDetectorConfig

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [
            name for name in self.feature_columns if name not in frame
        ]
        if missing:
            raise KeyError(
                "binary failure detector missing features: "
                + ", ".join(missing[:12])
            )
        values = (
            frame.loc[:, self.feature_columns]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(np.float64)
        )
        values = np.where(np.isfinite(values), values, self.medians)
        return np.clip(values, -1e6, 1e6).astype(np.float32, copy=False)

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        values = self._matrix(frame)
        return pd.DataFrame(
            {
                "p_failure_onset_within_3h": (
                    self.onset_head.probability(values)
                ),
                "p_failure_active_or_within_3h": (
                    self.risk_head.probability(values)
                ),
            },
            index=frame.index,
        )


def _fit_binary_head(
    values: np.ndarray,
    target: pd.Series,
    *,
    config: BinaryFailureDetectorConfig,
    seed_offset: int,
) -> _BinaryHead:
    labels = pd.to_numeric(target, errors="raise").astype(int)
    counts = labels.value_counts()
    if len(counts) != 2 or int(counts.min()) < int(config.min_positive_rows):
        raise ValueError("insufficient binary class support")
    from catboost import CatBoostClassifier

    parameters: dict[str, Any] = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": int(config.max_iter),
        "learning_rate": float(config.learning_rate),
        "depth": int(config.depth),
        "l2_leaf_reg": float(config.l2_regularization),
        "random_seed": int(config.random_state) + int(seed_offset),
        "thread_count": 1,
        "verbose": False,
        "allow_writing_files": False,
    }
    if config.auto_class_weights:
        parameters["auto_class_weights"] = str(config.auto_class_weights)
    model = CatBoostClassifier(**parameters)
    model.fit(values, labels)
    return _BinaryHead(model=model)


def fit_binary_failure_detector(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    train_end_exclusive: pd.Timestamp,
    config: BinaryFailureDetectorConfig,
) -> BinaryFailureDetectorBundle:
    """Fit both direct binary heads on labels resolved before the boundary."""

    features = validate_detector_features(
        feature_columns, max_features=int(config.max_features)
    )
    required = [
        config.timestamp_col,
        config.label_available_col,
        config.onset_target_col,
        config.risk_target_col,
        *features,
    ]
    missing = [name for name in required if name not in frame]
    if missing:
        raise KeyError("binary detector input missing: " + ", ".join(missing))
    timestamp = _utc(frame[config.timestamp_col])
    available = _utc(frame[config.label_available_col])
    boundary = pd.Timestamp(train_end_exclusive)
    if boundary.tzinfo is None:
        raise ValueError("train_end_exclusive must be timezone-aware")
    boundary = boundary.tz_convert("UTC")
    eligible = (
        timestamp.lt(boundary)
        & available.lt(boundary)
        & frame[[config.onset_target_col, config.risk_target_col]]
        .notna()
        .all(axis=1)
    )
    train = frame.loc[eligible].copy()
    if len(train) < int(config.min_train_rows):
        raise ValueError("insufficient label-available training rows")
    values = (
        train.loc[:, features]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(np.float64)
    )
    medians = np.nanmedian(values, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    values = np.where(np.isfinite(values), values, medians)
    values = np.clip(values, -1e6, 1e6).astype(np.float32, copy=False)
    return BinaryFailureDetectorBundle(
        feature_columns=features,
        medians=medians.astype(np.float32),
        onset_head=_fit_binary_head(
            values,
            train[config.onset_target_col],
            config=config,
            seed_offset=1,
        ),
        risk_head=_fit_binary_head(
            values,
            train[config.risk_target_col],
            config=config,
            seed_offset=2,
        ),
        train_rows=int(len(train)),
        train_end_exclusive=str(boundary),
        train_label_available_max=str(available.loc[eligible].max()),
        config=config,
    )


def chronological_binary_failure_oof(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    config: BinaryFailureDetectorConfig,
) -> tuple[pd.DataFrame, list[BinaryFailureDetectorBundle]]:
    """Generate expanding, availability-purged OOF predictions."""

    work = frame.copy()
    work[config.timestamp_col] = _utc(work[config.timestamp_col])
    work = work.sort_values(
        config.timestamp_col, kind="stable"
    ).reset_index(drop=True)
    first = (
        pd.Timestamp(config.first_eval_time)
        if str(config.first_eval_time).strip()
        else work[config.timestamp_col].min()
    )
    if first.tzinfo is None:
        raise ValueError("first_eval_time must be timezone-aware")
    first = first.tz_convert("UTC")
    step = pd.Timedelta(hours=max(1, int(config.eval_hours)))
    last = work[config.timestamp_col].max() + pd.Timedelta("1ns")
    starts = pd.date_range(first, last, freq=step, inclusive="left")
    predictions: list[pd.DataFrame] = []
    bundles: list[BinaryFailureDetectorBundle] = []
    for fold_index, start in enumerate(starts):
        end = min(start + step, last)
        evaluation = work.loc[
            work[config.timestamp_col].ge(start)
            & work[config.timestamp_col].lt(end)
        ]
        if evaluation.empty:
            continue
        try:
            bundle = fit_binary_failure_detector(
                work,
                feature_columns=feature_columns,
                train_end_exclusive=start,
                config=config,
            )
        except ValueError as error:
            if (
                "insufficient label-available training rows" in str(error)
                or "insufficient binary class support" in str(error)
            ):
                continue
            raise
        scored = bundle.score(evaluation)
        identity = [
            name
            for name in (
                config.timestamp_col,
                "side_name",
                "evaluation_origin",
            )
            if name in evaluation
        ]
        fold = evaluation.loc[:, identity].copy()
        for name in scored:
            fold[name] = scored[name].to_numpy()
        for name in (
            config.onset_target_col,
            config.risk_target_col,
            config.label_available_col,
        ):
            fold[name] = evaluation[name].to_numpy()
        fold["fold_index"] = int(fold_index)
        fold["train_end_exclusive"] = start
        fold["evaluation_end_exclusive"] = end
        fold["train_rows"] = int(bundle.train_rows)
        fold["train_label_available_max"] = pd.Timestamp(
            bundle.train_label_available_max
        )
        predictions.append(fold)
        bundles.append(bundle)
    if not predictions:
        raise RuntimeError("no chronological binary failure OOF folds generated")
    return pd.concat(predictions, ignore_index=True), bundles


__all__ = [
    "BINARY_TARGET_COLUMNS",
    "BinaryFailureDetectorBundle",
    "BinaryFailureDetectorConfig",
    "add_causal_transition_deltas",
    "build_hourly_binary_failure_targets",
    "chronological_binary_failure_oof",
    "fit_binary_failure_detector",
]
