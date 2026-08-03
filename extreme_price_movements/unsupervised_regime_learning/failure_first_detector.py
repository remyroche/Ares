"""Causal multi-head detector for failure-first market regimes.

The descriptive taxonomy is allowed to use resolved trading outcomes.  This
module is the inference-safe counterpart: it consumes only decision-time state
and model-health fields and emits chronological OOF probabilities for:

* a state transition within the configured horizon;
* an active state transition;
* the current failure state; and
* the transition destination.

The four fitted heads are one CatBoost detector bundle with one feature
contract and one model family.  They are not independent architecture
candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from extreme_price_movements.bayesian_changepoint import (
    BOCPDConfig,
    bocpd_student_t,
)
from extreme_price_movements.unsupervised_regime_learning.failure_episodes import (
    validate_inference_feature_columns,
)


TARGET_COLUMNS = {
    "target__transition_within_3h",
    "target__active_transition",
    "target__current_failure_state",
    "target__destination_state_3h",
}
FORBIDDEN_FEATURE_TOKENS = (
    "target",
    "label",
    "outcome",
    "future",
    "realized",
    "execution_net_ev",
    "mfe",
    "mae",
    "exit_reason",
)


@dataclass(frozen=True)
class FailureFirstDetectorConfig:
    """Fixed research contract for the compact hourly detector."""

    timestamp_col: str = "execution_decision_utc"
    label_available_col: str = "transition_label_available_at"
    transition_target_col: str = "target__transition_within_3h"
    active_target_col: str = "target__active_transition"
    current_state_col: str = "target__current_failure_state"
    destination_col: str = "target__destination_state_3h"
    first_eval_time: str = ""
    eval_hours: int = 168
    min_train_rows: int = 1_000
    min_class_rows: int = 12
    max_features: int = 40
    learning_rate: float = 0.06
    max_iter: int = 80
    depth: int = 5
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 40
    l2_regularization: float = 2.0
    failure_state_labels: tuple[str, ...] = (
        "volatility_expansion",
        "liquidity_dislocation",
        "funding_transition",
        "leverage_repricing",
        "correlation_fragmentation",
        "directional_transition",
        "mixed_observable_state",
    )
    random_state: int = 20260726


@dataclass
class _Head:
    classes: np.ndarray
    model: Any | None
    constant_probability: np.ndarray | None

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        if self.model is not None:
            return np.asarray(self.model.predict_proba(values), dtype=np.float64)
        assert self.constant_probability is not None
        return np.repeat(
            self.constant_probability.reshape(1, -1), len(values), axis=0
        )


@dataclass
class FailureFirstDetectorBundle:
    """Serializable four-head detector sharing one compact feature contract."""

    feature_columns: list[str]
    medians: np.ndarray
    transition_head: _Head
    active_head: _Head
    current_state_head: _Head
    destination_head: _Head
    train_rows: int
    train_label_available_max: str
    train_end_exclusive: str
    failure_state_labels: tuple[str, ...]

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [name for name in self.feature_columns if name not in frame]
        if missing:
            raise KeyError(
                "Failure-first detector missing required features: "
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
        matrix = self._matrix(frame)
        output = pd.DataFrame(index=frame.index)
        _attach_binary_probability(
            output,
            self.transition_head,
            matrix,
            "p_transition_within_3h",
        )
        _attach_binary_probability(
            output,
            self.active_head,
            matrix,
            "p_active_transition",
        )
        _attach_multiclass_probabilities(
            output,
            self.current_state_head,
            matrix,
            prefix="p_current_state",
            predicted_col="predicted_current_failure_state",
        )
        _attach_multiclass_probabilities(
            output,
            self.destination_head,
            matrix,
            prefix="p_destination",
            predicted_col="predicted_destination_state_3h",
        )
        destination_columns = [
            f"p_destination__{_slug(name)}"
            for name in self.failure_state_labels
            if f"p_destination__{_slug(name)}" in output
        ]
        output["p_failure_destination_3h"] = (
            output[destination_columns].sum(axis=1)
            if destination_columns
            else np.float32(0.0)
        )
        return output


def _slug(value: object) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip()).strip("_")
    return text.casefold() or "missing"


def validate_detector_features(
    feature_columns: Sequence[str],
    *,
    max_features: int = 40,
) -> list[str]:
    """Reject outcome-like/offline fields and enforce a compact contract."""

    columns = list(dict.fromkeys(str(name) for name in feature_columns))
    if not columns:
        raise ValueError("failure-first detector requires observable features")
    if len(columns) > int(max_features):
        raise ValueError(
            f"failure-first detector feature count {len(columns)} exceeds "
            f"the compact {int(max_features)}-feature contract"
        )
    validate_inference_feature_columns(columns)
    forbidden = sorted(
        name
        for name in columns
        if any(token in name.casefold() for token in FORBIDDEN_FEATURE_TOKENS)
    )
    if forbidden:
        raise ValueError(
            "failure-first detector received outcome-like features: "
            + ", ".join(forbidden[:12])
        )
    return columns


def _causal_scale(values: pd.Series, *, min_history: int = 16) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").astype(np.float64)
    history = numeric.shift(1)
    median = history.expanding(min_periods=min_history).median()
    q25 = history.expanding(min_periods=min_history).quantile(0.25)
    q75 = history.expanding(min_periods=min_history).quantile(0.75)
    scale = (q75 - q25).clip(lower=1e-4)
    filled = numeric.where(numeric.notna(), median)
    standardized = ((filled - median) / scale).clip(-8.0, 8.0)
    return standardized.fillna(0.0).to_numpy(np.float32)


def add_causal_bocpd_features(
    frame: pd.DataFrame,
    *,
    signal_columns: Sequence[str],
    timestamp_col: str = "execution_decision_utc",
    group_columns: Sequence[str] = ("side_name",),
    config: BOCPDConfig | None = None,
    threshold_quantile: float = 0.95,
    min_threshold_history: int = 24,
) -> pd.DataFrame:
    """Add one causal multi-signal BOCPD detector as continuous context.

    Scaling and synchronization thresholds use only earlier rows in the same
    group.  Consequently, appending future rows cannot alter an earlier score.
    """

    signals = list(dict.fromkeys(str(name) for name in signal_columns))
    missing = [name for name in [timestamp_col, *group_columns, *signals] if name not in frame]
    if missing:
        raise KeyError("BOCPD input missing columns: " + ", ".join(missing))
    if not signals:
        raise ValueError("BOCPD requires at least one observable signal")
    detector_config = config or BOCPDConfig()
    output = frame.copy()
    output[timestamp_col] = pd.to_datetime(
        output[timestamp_col], utc=True, errors="raise"
    )
    if output.duplicated([*group_columns, timestamp_col]).any():
        raise ValueError(
            "BOCPD requires one pre-aggregated row per group and timestamp"
        )
    output["failure_bocpd_probability_max"] = np.nan
    output["failure_bocpd_break_count"] = np.nan
    output["failure_bocpd_break_intensity"] = np.nan
    group_key: str | list[str]
    group_key = list(group_columns)
    if len(group_key) == 1:
        group_key = group_key[0]
    for _, positions in output.groupby(
        group_key, observed=True, sort=False
    ).indices.items():
        index = np.asarray(positions, dtype=np.int64)
        ordered = index[
            np.argsort(
                output.iloc[index][timestamp_col].to_numpy(),
                kind="stable",
            )
        ]
        probabilities = np.column_stack(
            [
                bocpd_student_t(
                    _causal_scale(output.iloc[ordered][name]),
                    detector_config,
                )
                for name in signals
            ]
        )
        probability_frame = pd.DataFrame(probabilities)
        thresholds = probability_frame.shift(1).expanding(
            min_periods=int(min_threshold_history)
        ).quantile(float(threshold_quantile))
        denominator = (1.0 - thresholds).clip(lower=1e-5)
        excess = ((probability_frame - thresholds) / denominator).clip(lower=0.0)
        count = probability_frame.gt(thresholds).sum(axis=1)
        intensity = excess.mean(axis=1)
        output.iloc[
            ordered,
            output.columns.get_loc("failure_bocpd_probability_max"),
        ] = np.nanmax(probabilities, axis=1)
        output.iloc[
            ordered,
            output.columns.get_loc("failure_bocpd_break_count"),
        ] = count.to_numpy(np.float32)
        output.iloc[
            ordered,
            output.columns.get_loc("failure_bocpd_break_intensity"),
        ] = intensity.to_numpy(np.float32)
    return output


def _fit_head(
    values: np.ndarray,
    target: pd.Series,
    *,
    config: FailureFirstDetectorConfig,
    seed_offset: int,
) -> _Head:
    labels = target.astype(str).to_numpy()
    classes, counts = np.unique(labels, return_counts=True)
    probabilities = counts.astype(np.float64) / max(float(counts.sum()), 1.0)
    if len(classes) < 2 or counts.min() < int(config.min_class_rows):
        return _Head(
            classes=classes,
            model=None,
            constant_probability=probabilities,
        )
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        loss_function="MultiClass" if len(classes) > 2 else "Logloss",
        iterations=int(config.max_iter),
        learning_rate=float(config.learning_rate),
        depth=int(config.depth),
        min_data_in_leaf=int(config.min_samples_leaf),
        l2_leaf_reg=float(config.l2_regularization),
        random_seed=int(config.random_state) + int(seed_offset),
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(values, labels)
    return _Head(
        classes=np.asarray(model.classes_, dtype=object),
        model=model,
        constant_probability=None,
    )


def fit_failure_first_detector(
    train: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    train_end_exclusive: pd.Timestamp,
    config: FailureFirstDetectorConfig,
) -> FailureFirstDetectorBundle:
    """Fit the one-family four-head detector on label-available rows only."""

    features = validate_detector_features(
        feature_columns, max_features=int(config.max_features)
    )
    timestamp = pd.to_datetime(
        train[config.timestamp_col], utc=True, errors="raise"
    )
    available = pd.to_datetime(
        train[config.label_available_col], utc=True, errors="coerce"
    )
    boundary = pd.Timestamp(train_end_exclusive)
    if boundary.tzinfo is None:
        raise ValueError("train_end_exclusive must be timezone-aware")
    boundary = boundary.tz_convert("UTC")
    required_targets = [
        config.transition_target_col,
        config.active_target_col,
        config.current_state_col,
        config.destination_col,
    ]
    missing = [
        name
        for name in [
            config.timestamp_col,
            config.label_available_col,
            *required_targets,
            *features,
        ]
        if name not in train
    ]
    if missing:
        raise KeyError("failure-first training input missing: " + ", ".join(missing))
    eligible = timestamp.lt(boundary) & available.lt(boundary)
    eligible &= train[required_targets].notna().all(axis=1)
    fitted = train.loc[eligible].copy()
    if len(fitted) < int(config.min_train_rows):
        raise ValueError(
            f"insufficient label-available training rows: {len(fitted)} "
            f"< {int(config.min_train_rows)}"
        )
    matrix = (
        fitted.loc[:, features]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(np.float64)
    )
    medians = np.nanmedian(matrix, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    matrix = np.where(np.isfinite(matrix), matrix, medians)
    matrix = np.clip(matrix, -1e6, 1e6).astype(np.float32, copy=False)
    return FailureFirstDetectorBundle(
        feature_columns=features,
        medians=medians.astype(np.float32),
        transition_head=_fit_head(
            matrix,
            fitted[config.transition_target_col],
            config=config,
            seed_offset=1,
        ),
        active_head=_fit_head(
            matrix,
            fitted[config.active_target_col],
            config=config,
            seed_offset=2,
        ),
        current_state_head=_fit_head(
            matrix,
            fitted[config.current_state_col],
            config=config,
            seed_offset=3,
        ),
        destination_head=_fit_head(
            matrix,
            fitted[config.destination_col],
            config=config,
            seed_offset=4,
        ),
        train_rows=int(len(fitted)),
        train_label_available_max=str(available.loc[eligible].max()),
        train_end_exclusive=str(boundary),
        failure_state_labels=tuple(config.failure_state_labels),
    )


def chronological_failure_first_oof(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    config: FailureFirstDetectorConfig,
) -> tuple[pd.DataFrame, list[FailureFirstDetectorBundle]]:
    """Generate expanding, purged OOF predictions for the compact detector."""

    work = frame.copy()
    work[config.timestamp_col] = pd.to_datetime(
        work[config.timestamp_col], utc=True, errors="raise"
    )
    work = work.sort_values(config.timestamp_col, kind="stable").reset_index(drop=True)
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
    bundles: list[FailureFirstDetectorBundle] = []
    for fold_index, start in enumerate(starts):
        end = min(start + step, last)
        evaluation = work.loc[
            work[config.timestamp_col].ge(start)
            & work[config.timestamp_col].lt(end)
        ]
        if evaluation.empty:
            continue
        try:
            bundle = fit_failure_first_detector(
                work,
                feature_columns=feature_columns,
                train_end_exclusive=start,
                config=config,
            )
        except ValueError as error:
            if "insufficient label-available training rows" in str(error):
                continue
            raise
        scored = bundle.score(evaluation)
        identity_columns = [
            name
            for name in (
                "candidate_id",
                config.timestamp_col,
                "side_name",
                "evaluation_origin",
            )
            if name in evaluation
        ]
        fold = evaluation.loc[:, identity_columns].copy()
        for name in scored:
            fold[name] = scored[name].to_numpy()
        for name in (
            config.transition_target_col,
            config.active_target_col,
            config.current_state_col,
            config.destination_col,
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
        raise RuntimeError("no chronological failure-first OOF folds were generated")
    return pd.concat(predictions, ignore_index=True), bundles


def _attach_binary_probability(
    output: pd.DataFrame,
    head: _Head,
    matrix: np.ndarray,
    column: str,
) -> None:
    probabilities = head.predict_proba(matrix)
    classes = [str(value).casefold() for value in head.classes]
    positive_index = next(
        (
            index
            for index, value in enumerate(classes)
            if value in {"1", "1.0", "true"}
        ),
        None,
    )
    output[column] = (
        probabilities[:, positive_index].astype(np.float32)
        if positive_index is not None
        else np.zeros(len(matrix), dtype=np.float32)
    )


def _attach_multiclass_probabilities(
    output: pd.DataFrame,
    head: _Head,
    matrix: np.ndarray,
    *,
    prefix: str,
    predicted_col: str,
) -> None:
    probabilities = head.predict_proba(matrix)
    for index, value in enumerate(head.classes):
        output[f"{prefix}__{_slug(value)}"] = probabilities[:, index].astype(
            np.float32
        )
    winner = probabilities.argmax(axis=1)
    output[predicted_col] = np.asarray(head.classes, dtype=object)[winner]


__all__ = [
    "FailureFirstDetectorBundle",
    "FailureFirstDetectorConfig",
    "TARGET_COLUMNS",
    "add_causal_bocpd_features",
    "chronological_failure_first_oof",
    "fit_failure_first_detector",
    "validate_detector_features",
]
