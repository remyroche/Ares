"""Leakage-safe economic scoring for CatBoost class-balance OOF mini-sweeps.

This module deliberately does *not* fit a model or choose class weights.  It
scores the four already-produced, matched OOF probability matrices and emits
the selection provenance consumed by the final class-weight materialisation.
All class-conditional outcome priors are estimated separately from each
``PurgedFold.train_indices``; validation outcomes are never used to create a
prior, rank cutoff, or smoothing parameter.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ECONOMIC_OOF_SCORE_SCHEMA = "catboost_path_archetype_balance_economic_oof_v1"
CLASS_BALANCE_SELECTION_SCHEMA = (
    "catboost_path_archetype_class_balance_arm_selection_v1"
)
DEFAULT_BALANCE_ARMS = (
    "uniform",
    "frequency_power_0.25",
    "frequency_power_0.50",
    "frequency_power_0.75",
)

_REQUIRED_OUTCOME_COLUMNS = (
    "path_arch_final_return_net_1pct",
    "path_arch_peak_mfe_atr",
    "path_arch_mae_12h_r",
    "path_arch_mae_before_meaningful_mfe_r",
    "path_arch_stop_before_meaningful_mfe",
    "path_arch_reaches_meaningful_mfe",
    "path_arch_time_to_first_meaningful_mfe_h",
    "path_arch_peak_retention_ratio",
    "path_arch_time_to_trailing_h",
    "path_arch_mfe_to_activation_distance",
)

_CANONICAL_OUTCOME_MAPPING = {
    "net_ev": "path_arch_final_return_net_1pct",
    "mfe_atr": "path_arch_peak_mfe_atr",
    "mae_r": "path_arch_mae_12h_r",
    "mae_before_meaningful_r": "path_arch_mae_before_meaningful_mfe_r",
    "stop_before_meaningful": "path_arch_stop_before_meaningful_mfe",
    "reaches_meaningful": "path_arch_reaches_meaningful_mfe",
    "time_to_meaningful_h": "path_arch_time_to_first_meaningful_mfe_h",
    "retention": "path_arch_peak_retention_ratio",
    "trailing_conversion": (
        "path_arch_time_to_trailing_h + path_arch_mfe_to_activation_distance"
    ),
}


@dataclass(frozen=True)
class EconomicOOFConfig:
    """Frozen scoring choices, all of which belong in the mini-sweep fingerprint."""

    timestamp_col: str = "__ts__"
    side_col: str = "__side__"
    label_end_col: str = "__label_end_ts__"
    identity_col: str = "candidate_id"
    embargo: pd.Timedelta = pd.Timedelta(hours=24)
    continuous_prior_kappa: float = 20.0
    binary_prior_alpha: float = 1.0
    binary_prior_beta: float = 1.0
    tail_fraction: float = 0.20
    calibration_bins: int = 10
    minimum_month_rows: int = 20
    expected_arms: tuple[str, ...] = DEFAULT_BALANCE_ARMS
    # Strictly zero by default: a balance arm must be ML non-inferior to
    # uniform rather than trading measurable classification deterioration for
    # an apparently favourable economic tail in this first deployment.
    logloss_tolerance: float = 0.0
    brier_tolerance: float = 0.0
    rps_tolerance: float = 0.0
    f1_tolerance: float = 0.0


@dataclass(frozen=True)
class BalanceArmOOF:
    """One fully materialised fixed-parameter class-balance OOF result.

    ``folds`` uses the small ``PurgedFold`` protocol (``fold_id``,
    ``train_indices``, ``validation_indices``) without importing CatBoost's
    classifier module.  The three fingerprints are mandatory deliberately:
    the scorer must fail rather than compare probabilities produced by a
    different feature, geometry, or structural-HPO contract.
    """

    probabilities: np.ndarray
    fold_ids: np.ndarray
    folds: Sequence[Any]
    classes: Sequence[Any]
    structural_fingerprint: str
    feature_fingerprint: str
    geometry_fingerprint: str
    oof_guard: Mapping[str, Any] | None
    row_ids: Sequence[Any]


def _finite_numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _json_ready(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if np.isfinite(result) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    return value


def _selector_contract(config: EconomicOOFConfig) -> dict[str, Any]:
    """Return every frozen outcome-sensitive selector setting for provenance."""
    return _json_ready(
        {
            "identity_col": config.identity_col,
            "timestamp_col": config.timestamp_col,
            "side_col": config.side_col,
            "label_end_col": config.label_end_col,
            "columns": {
                "identity": config.identity_col,
                "timestamp": config.timestamp_col,
                "side": config.side_col,
                "label_end": config.label_end_col,
                "canonical_v9_outcomes": _CANONICAL_OUTCOME_MAPPING,
            },
            "embargo": config.embargo,
            "continuous_prior_kappa": config.continuous_prior_kappa,
            "binary_prior_alpha": config.binary_prior_alpha,
            "binary_prior_beta": config.binary_prior_beta,
            "tail_fraction": config.tail_fraction,
            "calibration_bins": config.calibration_bins,
            "minimum_month_rows": config.minimum_month_rows,
            "expected_arms": config.expected_arms,
            "logloss_tolerance": config.logloss_tolerance,
            "brier_tolerance": config.brier_tolerance,
            "rps_tolerance": config.rps_tolerance,
            "f1_tolerance": config.f1_tolerance,
        }
    )


def _selector_contract_digest(contract: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_probability_matrix(
    probabilities: np.ndarray,
    rows: int,
    classes: tuple[str, ...],
    scored_rows: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or matrix.shape != (rows, len(classes)):
        raise ValueError(
            "OOF probabilities must align exactly with outcomes and classes"
        )
    # Pre-first-fold rows are intentionally unscored and the ordinary OOF
    # fitter leaves them as NaN.  They must not be validated or consumed as
    # if they were predictions; only validation rows require a probability
    # simplex.
    if (
        not len(scored_rows)
        or not np.isfinite(matrix[scored_rows]).all()
        or (matrix[scored_rows] < 0.0).any()
        or not np.allclose(matrix[scored_rows].sum(axis=1), 1.0, rtol=1e-6, atol=1e-8)
    ):
        raise ValueError(
            "OOF probabilities must be finite non-negative rows summing to 1"
        )
    return matrix


def _fold_signature(
    folds: Sequence[Any],
) -> tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...]:
    signature: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []
    for fold in folds:
        try:
            fold_id = int(fold.fold_id)
            train = tuple(map(int, np.asarray(fold.train_indices, dtype=int)))
            valid = tuple(map(int, np.asarray(fold.validation_indices, dtype=int)))
        except AttributeError as exc:
            raise TypeError(
                "folds must expose fold_id, train_indices, and validation_indices"
            ) from exc
        if not train or not valid:
            raise ValueError(
                "economic OOF scoring requires non-empty train and validation folds"
            )
        signature.append((fold_id, train, valid))
    if not signature or len({row[0] for row in signature}) != len(signature):
        raise ValueError("OOF folds must be non-empty with unique fold ids")
    return tuple(signature)


def _validate_matched_arms(
    outcomes: pd.DataFrame,
    target_codes: Sequence[int],
    arms: Mapping[str, BalanceArmOOF],
    config: EconomicOOFConfig,
) -> tuple[tuple[str, ...], tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...]]:
    names = tuple(arms)
    if set(names) != set(config.expected_arms) or len(names) != len(
        config.expected_arms
    ):
        raise ValueError(
            "class-balance economics requires exactly the predeclared arms: "
            + ", ".join(config.expected_arms)
        )
    if "uniform" not in arms:
        raise ValueError("uniform control arm is mandatory")
    rows = len(outcomes)
    target = np.asarray(target_codes, dtype=int)
    if target.ndim != 1 or len(target) != rows:
        raise ValueError("target_codes must align exactly with outcomes")
    first = arms[names[0]]
    classes = tuple(map(str, first.classes))
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("OOF class order must be non-empty and unique")
    fingerprints = (
        first.structural_fingerprint,
        first.feature_fingerprint,
        first.geometry_fingerprint,
    )
    if any(not isinstance(value, str) or not value for value in fingerprints):
        raise ValueError(
            "matched-arm structural, feature, and geometry fingerprints are required"
        )
    first_folds = _fold_signature(first.folds)
    first_fold_ids = np.asarray(first.fold_ids, dtype=int)
    if first_fold_ids.shape != (rows,):
        raise ValueError("OOF fold_ids must align exactly with outcomes")
    if first.row_ids is None:
        raise ValueError("OOF row_ids are mandatory for matched-arm comparison")
    first_rows = np.asarray(first.row_ids)
    if first_rows.shape != (rows,) or len(pd.unique(first_rows)) != rows:
        raise ValueError("OOF row_ids must be unique and align exactly with outcomes")
    for name, arm in arms.items():
        arm_classes = tuple(map(str, arm.classes))
        if arm_classes != classes:
            raise ValueError(f"{name} has a different frozen OOF class order")
        if (
            arm.structural_fingerprint,
            arm.feature_fingerprint,
            arm.geometry_fingerprint,
        ) != fingerprints:
            raise ValueError(f"{name} is not a matched structural/feature/geometry arm")
        _validate_probability_matrix(
            arm.probabilities,
            rows,
            classes,
            np.flatnonzero(first_fold_ids >= 0),
        )
        if not np.array_equal(np.asarray(arm.fold_ids, dtype=int), first_fold_ids):
            raise ValueError(f"{name} does not use the identical OOF fold-id vector")
        if _fold_signature(arm.folds) != first_folds:
            raise ValueError(f"{name} does not use identical purged OOF folds")
        if arm.row_ids is None:
            raise ValueError(f"{name} is missing mandatory OOF row_ids")
        row_ids = np.asarray(arm.row_ids)
        if not np.array_equal(row_ids, first_rows):
            raise ValueError(f"{name} does not use identical OOF row identities")
    if (target < 0).any() or (target >= len(classes)).any():
        raise ValueError("target_codes fall outside the frozen OOF class order")
    return classes, first_folds


def _validate_outcomes_and_folds(
    outcomes: pd.DataFrame,
    signature: tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...],
    fold_ids: np.ndarray,
    row_ids: np.ndarray,
    config: EconomicOOFConfig,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    missing = sorted(
        set(_REQUIRED_OUTCOME_COLUMNS)
        .union(
            {
                config.identity_col,
                config.timestamp_col,
                config.side_col,
                config.label_end_col,
            }
        )
        .difference(outcomes.columns)
    )
    if missing:
        raise KeyError(
            "canonical v9 economic scorer missing columns: " + ", ".join(missing)
        )
    timestamps = pd.to_datetime(
        outcomes[config.timestamp_col], utc=True, errors="coerce"
    )
    label_end = pd.to_datetime(
        outcomes[config.label_end_col], utc=True, errors="coerce"
    )
    side = outcomes[config.side_col].astype("string")
    identity = outcomes[config.identity_col]
    if (
        timestamps.isna().any()
        or label_end.isna().any()
        or side.isna().any()
        or identity.isna().any()
    ):
        raise ValueError(
            "identity, timestamps, label-end timestamps, and side are required for every row"
        )
    if identity.nunique(dropna=False) != len(outcomes):
        raise ValueError("outcome identities must be unique")
    if not np.array_equal(identity.to_numpy(), row_ids):
        raise ValueError("outcome identities must exactly match OOF row_ids in order")
    side_values = tuple(pd.unique(side))
    if len(side_values) != 1 or side_values[0] not in {"long", "short"}:
        raise ValueError(
            "economic OOF scorer requires exactly one canonical side: long or short"
        )
    for fold_id, train_tuple, valid_tuple in signature:
        train, valid = (
            np.asarray(train_tuple, dtype=int),
            np.asarray(valid_tuple, dtype=int),
        )
        if (
            np.any(train < 0)
            or np.any(valid < 0)
            or np.any(train >= len(outcomes))
            or np.any(valid >= len(outcomes))
        ):
            raise ValueError("fold indices exceed the outcome frame")
        if np.intersect1d(train, valid).size:
            raise ValueError("OOF fold train and validation rows overlap")
        if not np.all(fold_ids[valid] == fold_id):
            raise ValueError("fold_ids do not match declared validation indices")
        validation_start = timestamps.iloc[valid].min()
        if not bool((label_end.iloc[train] < validation_start).all()):
            raise ValueError(
                "economic priors would include unresolved labels at OOF validation start"
            )
        if not bool((timestamps.iloc[train] < validation_start - config.embargo).all()):
            raise ValueError("economic priors violate the configured OOF embargo")
    validation_rows = np.concatenate(
        [np.asarray(valid, dtype=int) for _fold_id, _train, valid in signature]
    )
    if len(np.unique(validation_rows)) != len(validation_rows):
        raise ValueError("OOF validation rows cannot appear in more than one fold")
    scored_rows = np.flatnonzero(fold_ids >= 0)
    if not np.array_equal(np.sort(validation_rows), scored_rows):
        raise ValueError(
            "only declared validation rows may have a non-negative OOF fold id"
        )
    return timestamps, label_end, side


def _continuous_prior(
    values: np.ndarray,
    labels: np.ndarray,
    classes: int,
    kappa: float,
    *,
    allow_empty: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    valid = np.isfinite(values)
    if not valid.any():
        if allow_empty:
            return np.full(classes, np.nan), {
                "type": "continuous_empirical_bayes_no_train_support",
                "global_mean": float("nan"),
                "support": [0] * classes,
                "kappa": float(kappa),
            }
        raise ValueError("train-only economic prior has no finite support")
    global_mean = float(values[valid].mean())
    priors = np.empty(classes, dtype=float)
    supports: list[int] = []
    for class_id in range(classes):
        mask = valid & (labels == class_id)
        support = int(mask.sum())
        supports.append(support)
        mean = float(values[mask].mean()) if support else global_mean
        priors[class_id] = (support * mean + kappa * global_mean) / (support + kappa)
    return priors, {
        "type": "continuous_empirical_bayes",
        "global_mean": global_mean,
        "support": supports,
        "kappa": float(kappa),
    }


def _binary_prior(
    values: np.ndarray, labels: np.ndarray, classes: int, alpha: float, beta: float
) -> tuple[np.ndarray, dict[str, Any]]:
    valid = np.isfinite(values)
    if not valid.any():
        raise ValueError("train-only binary economic prior has no finite support")
    successes = float(values[valid].sum())
    global_mean = (successes + alpha) / (float(valid.sum()) + alpha + beta)
    priors = np.empty(classes, dtype=float)
    supports: list[int] = []
    for class_id in range(classes):
        mask = valid & (labels == class_id)
        support = int(mask.sum())
        supports.append(support)
        if support:
            priors[class_id] = (float(values[mask].sum()) + alpha) / (
                support + alpha + beta
            )
        else:
            priors[class_id] = global_mean
    return priors, {
        "type": "binary_beta",
        "global_mean": global_mean,
        "support": supports,
        "alpha": float(alpha),
        "beta": float(beta),
    }


def _targets(outcomes: pd.DataFrame) -> dict[str, np.ndarray]:
    mae = -_finite_numeric(outcomes, "path_arch_mae_12h_r")
    values = {
        "net_ev": _finite_numeric(outcomes, "path_arch_final_return_net_1pct"),
        "mfe_atr": _finite_numeric(outcomes, "path_arch_peak_mfe_atr"),
        "mae_r": np.where(np.isfinite(mae), np.maximum(mae, 0.0), np.nan),
        "mae_before_meaningful_r": _finite_numeric(
            outcomes, "path_arch_mae_before_meaningful_mfe_r"
        ),
        "stop_before_meaningful": _finite_numeric(
            outcomes, "path_arch_stop_before_meaningful_mfe"
        ),
        "reaches_meaningful": _finite_numeric(
            outcomes, "path_arch_reaches_meaningful_mfe"
        ),
        "time_to_meaningful_h": _finite_numeric(
            outcomes, "path_arch_time_to_first_meaningful_mfe_h"
        ),
        "retention": _finite_numeric(outcomes, "path_arch_peak_retention_ratio"),
    }
    activation = _finite_numeric(outcomes, "path_arch_mfe_to_activation_distance")
    time_to_trail = _finite_numeric(outcomes, "path_arch_time_to_trailing_h")
    # An unavailable activation geometry is excluded from the target rather
    # than silently called a failed trailing conversion.
    values["trailing_conversion"] = np.where(
        np.isfinite(activation) & (activation > 0.0),
        np.isfinite(time_to_trail).astype(float),
        np.nan,
    )
    values["time_to_meaningful_h"] = np.where(
        values["reaches_meaningful"] > 0.5, values["time_to_meaningful_h"], np.nan
    )
    # Retention is meaningful only for an opportunity-bearing path.  The
    # canonical materializer also leaves it NaN when no cost-positive peak
    # exists; this additional reach gate keeps no-opportunity rows from being
    # interpreted as retention failures.
    values["retention"] = np.where(
        values["reaches_meaningful"] > 0.5, values["retention"], np.nan
    )
    for name in ("stop_before_meaningful", "reaches_meaningful", "trailing_conversion"):
        values[name] = np.where(
            np.isfinite(values[name]), np.clip(values[name], 0.0, 1.0), np.nan
        )
    return values


def _regression_metrics(
    actual: np.ndarray, predicted: np.ndarray
) -> dict[str, float | int]:
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if not mask.any():
        return {"rows": 0, "mae": float("nan"), "rmse": float("nan")}
    errors = actual[mask] - predicted[mask]
    return {
        "rows": int(mask.sum()),
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.mean(errors**2))),
    }


def _binary_metrics(
    actual: np.ndarray, predicted: np.ndarray
) -> dict[str, float | int]:
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if not mask.any():
        return {
            "rows": 0,
            "brier": float("nan"),
            "observed_rate": float("nan"),
            "predicted_rate": float("nan"),
        }
    return {
        "rows": int(mask.sum()),
        "brier": float(np.mean((actual[mask] - predicted[mask]) ** 2)),
        "observed_rate": float(actual[mask].mean()),
        "predicted_rate": float(predicted[mask].mean()),
    }


def _multiclass_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    classes = probabilities.shape[1]
    clipped = np.clip(probabilities, 1e-15, 1.0)
    logloss = float(-np.log(clipped[np.arange(len(labels)), labels]).mean())
    one_hot = np.eye(classes, dtype=float)[labels]
    brier = float(np.mean(np.mean((probabilities - one_hot) ** 2, axis=0)))
    cumulative_error = np.cumsum(probabilities - one_hot, axis=1)[:, :-1]
    rps = float(np.mean(np.mean(cumulative_error**2, axis=1))) if classes > 1 else 0.0
    predicted = probabilities.argmax(axis=1)
    f1s: list[float] = []
    for class_id in range(classes):
        true_positive = int(((labels == class_id) & (predicted == class_id)).sum())
        false_positive = int(((labels != class_id) & (predicted == class_id)).sum())
        false_negative = int(((labels == class_id) & (predicted != class_id)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        f1s.append(0.0 if denominator == 0 else (2 * true_positive) / denominator)
    return {
        "logloss": logloss,
        "brier_macro": brier,
        "rps": rps,
        "f1_macro": float(np.mean(f1s)),
    }


def _ev_calibration(
    actual: np.ndarray, predicted: np.ndarray, bins: int
) -> list[dict[str, float | int]]:
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if not mask.any():
        return []
    actual, predicted = actual[mask], predicted[mask]
    edges = np.linspace(float(predicted.min()), float(predicted.max()), bins + 1)
    if edges[0] == edges[-1]:
        edges[-1] += 1e-12
    bucket = np.clip(np.digitize(predicted, edges[1:-1]), 0, bins - 1)
    return [
        {
            "bin": int(index),
            "rows": int(group.sum()),
            "predicted_net_ev": float(predicted[group].mean()),
            "realised_net_ev": float(actual[group].mean()),
        }
        for index in range(bins)
        if (group := bucket == index).any()
    ]


def _tail_metrics(
    actual_ev: np.ndarray, predicted_ev: np.ndarray, rows: np.ndarray, fraction: float
) -> dict[str, float | int]:
    valid = rows[np.isfinite(actual_ev[rows]) & np.isfinite(predicted_ev[rows])]
    if not len(valid):
        return {
            "rows": 0,
            "realised_net_ev": float("nan"),
            "positive_net_ev_fraction": float("nan"),
        }
    selected_count = max(1, int(np.ceil(len(valid) * fraction)))
    # Stable index tie-breaker means two matched arms are never compared using
    # an accidental non-deterministic ordering.
    order = np.lexsort((valid, -predicted_ev[valid]))
    selected = valid[order[:selected_count]]
    return {
        "rows": int(len(selected)),
        "realised_net_ev": float(actual_ev[selected].mean()),
        "positive_net_ev_fraction": float((actual_ev[selected] > 0.0).mean()),
    }


def _fold_priors(
    targets: Mapping[str, np.ndarray],
    labels: np.ndarray,
    train: np.ndarray,
    classes: int,
    config: EconomicOOFConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    priors: dict[str, np.ndarray] = {}
    provenance: dict[str, Any] = {}
    continuous = (
        "net_ev",
        "mfe_atr",
        "mae_r",
        "mae_before_meaningful_r",
        "time_to_meaningful_h",
        "retention",
    )
    for name in continuous:
        priors[name], provenance[name] = _continuous_prior(
            targets[name][train],
            labels[train],
            classes,
            config.continuous_prior_kappa,
            # Conditional time and retention can correctly have no support in
            # an early chronological fold.  Keep them unscored rather than
            # manufacture an OOS-informed value or reject all arm evidence.
            allow_empty=name in {"time_to_meaningful_h", "retention"},
        )
    for name in ("stop_before_meaningful", "reaches_meaningful", "trailing_conversion"):
        priors[name], provenance[name] = _binary_prior(
            targets[name][train],
            labels[train],
            classes,
            config.binary_prior_alpha,
            config.binary_prior_beta,
        )
    return priors, provenance


def _score_arm(
    arm: BalanceArmOOF,
    outcomes: pd.DataFrame,
    labels: np.ndarray,
    targets: Mapping[str, np.ndarray],
    timestamps: pd.Series,
    side: pd.Series,
    signature: tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...],
    config: EconomicOOFConfig,
) -> dict[str, Any]:
    probabilities = np.asarray(arm.probabilities, dtype=float)
    predicted: dict[str, np.ndarray] = {
        name: np.full(len(outcomes), np.nan) for name in targets
    }
    fold_records: list[dict[str, Any]] = []
    prior_provenance: list[dict[str, Any]] = []
    for fold_id, train_tuple, valid_tuple in signature:
        train, valid = (
            np.asarray(train_tuple, dtype=int),
            np.asarray(valid_tuple, dtype=int),
        )
        priors, provenance = _fold_priors(
            targets, labels, train, probabilities.shape[1], config
        )
        for name, values in priors.items():
            predicted[name][valid] = probabilities[valid] @ values
        fold_record: dict[str, Any] = {
            "fold_id": int(fold_id),
            "train_rows": int(len(train)),
            "validation_rows": int(len(valid)),
            "ml": _multiclass_metrics(labels[valid], probabilities[valid]),
            "economic": {
                "net_ev": _regression_metrics(
                    targets["net_ev"][valid], predicted["net_ev"][valid]
                ),
                "mfe_atr": _regression_metrics(
                    targets["mfe_atr"][valid], predicted["mfe_atr"][valid]
                ),
                "mae_r": _regression_metrics(
                    targets["mae_r"][valid], predicted["mae_r"][valid]
                ),
                "mae_before_meaningful_r": _regression_metrics(
                    targets["mae_before_meaningful_r"][valid],
                    predicted["mae_before_meaningful_r"][valid],
                ),
                "stop_before_meaningful": _binary_metrics(
                    targets["stop_before_meaningful"][valid],
                    predicted["stop_before_meaningful"][valid],
                ),
                "reaches_meaningful": _binary_metrics(
                    targets["reaches_meaningful"][valid],
                    predicted["reaches_meaningful"][valid],
                ),
                "trailing_conversion": _binary_metrics(
                    targets["trailing_conversion"][valid],
                    predicted["trailing_conversion"][valid],
                ),
                "time_to_meaningful_h": _regression_metrics(
                    targets["time_to_meaningful_h"][valid],
                    predicted["time_to_meaningful_h"][valid],
                ),
                "top_tail": _tail_metrics(
                    targets["net_ev"], predicted["net_ev"], valid, config.tail_fraction
                ),
            },
        }
        fold_records.append(fold_record)
        prior_provenance.append(
            {
                "fold_id": int(fold_id),
                "source": "exact_purged_fold_train_indices_only",
                "targets": provenance,
            }
        )
    oof = np.flatnonzero(np.asarray(arm.fold_ids, dtype=int) >= 0)
    aggregate = {
        "ml": _multiclass_metrics(labels[oof], probabilities[oof]),
        "economic": {
            "net_ev": _regression_metrics(
                targets["net_ev"][oof], predicted["net_ev"][oof]
            ),
            "mfe_atr": _regression_metrics(
                targets["mfe_atr"][oof], predicted["mfe_atr"][oof]
            ),
            "mae_r": _regression_metrics(
                targets["mae_r"][oof], predicted["mae_r"][oof]
            ),
            "mae_before_meaningful_r": _regression_metrics(
                targets["mae_before_meaningful_r"][oof],
                predicted["mae_before_meaningful_r"][oof],
            ),
            "stop_before_meaningful": _binary_metrics(
                targets["stop_before_meaningful"][oof],
                predicted["stop_before_meaningful"][oof],
            ),
            "reaches_meaningful": _binary_metrics(
                targets["reaches_meaningful"][oof], predicted["reaches_meaningful"][oof]
            ),
            "trailing_conversion": _binary_metrics(
                targets["trailing_conversion"][oof],
                predicted["trailing_conversion"][oof],
            ),
            "time_to_meaningful_h": _regression_metrics(
                targets["time_to_meaningful_h"][oof],
                predicted["time_to_meaningful_h"][oof],
            ),
            "top_tail": _tail_metrics(
                targets["net_ev"], predicted["net_ev"], oof, config.tail_fraction
            ),
            "net_ev_calibration": _ev_calibration(
                targets["net_ev"][oof],
                predicted["net_ev"][oof],
                config.calibration_bins,
            ),
        },
    }
    month_records: list[dict[str, Any]] = []
    # The timestamps are already explicitly normalised to UTC.  Formatting
    # avoids pandas' timezone-dropping Period conversion warning while keeping
    # the grouping semantics unambiguous.
    month_key = timestamps.dt.strftime("%Y-%m")
    for (fold_id, side_value, month), positions in (
        pd.DataFrame({"fold": arm.fold_ids, "side": side, "month": month_key})
        .iloc[oof]
        .groupby(["fold", "side", "month"], observed=True)
        .indices.items()
    ):
        rows = oof[np.asarray(positions, dtype=int)]
        supported = len(rows) >= config.minimum_month_rows
        month_records.append(
            {
                "fold_id": int(fold_id),
                "side": str(side_value),
                "month_utc": str(month),
                "rows": int(len(rows)),
                "supported": bool(supported),
                "top_tail": _tail_metrics(
                    targets["net_ev"], predicted["net_ev"], rows, config.tail_fraction
                )
                if supported
                else None,
                "net_ev": _regression_metrics(
                    targets["net_ev"][rows], predicted["net_ev"][rows]
                )
                if supported
                else None,
                "stop_before_meaningful": _binary_metrics(
                    targets["stop_before_meaningful"][rows],
                    predicted["stop_before_meaningful"][rows],
                )
                if supported
                else None,
            }
        )
    return {
        "aggregate": aggregate,
        "folds": fold_records,
        "months": month_records,
        "train_only_priors": prior_provenance,
    }


def _mean_metric(records: Sequence[Mapping[str, Any]], path: Sequence[str]) -> float:
    values: list[float] = []
    for record in records:
        value: Any = record
        for key in path:
            value = value[key]
        if value is not None and np.isfinite(float(value)):
            values.append(float(value))
    return float(np.mean(values)) if values else float("nan")


def _select_arm(
    per_arm: Mapping[str, Mapping[str, Any]],
    arms: Mapping[str, BalanceArmOOF],
    classes: tuple[str, ...],
    config: EconomicOOFConfig,
) -> dict[str, Any]:
    uniform = per_arm["uniform"]
    all_guard_pass = all(
        bool((arm.oof_guard or {}).get("passed", False)) for arm in arms.values()
    )
    all_months_supported = all(
        bool(row["supported"])
        for report in per_arm.values()
        for row in report["months"]
    )
    common = {
        "schema": CLASS_BALANCE_SELECTION_SCHEMA,
        "class_order": list(classes),
        "selection_evidence": "purged_chronological_oof_validation_only",
        "final_refit_used_for_selection": False,
        "mandatory_initial_coverage_complete": bool(all_guard_pass),
        "economic_oof_schema": ECONOMIC_OOF_SCORE_SCHEMA,
        "economic_month_support_complete": bool(all_months_supported),
        "promotion_eligible": False,
    }
    if not all_guard_pass or not all_months_supported:
        reason = (
            "oof_guard_failed"
            if not all_guard_pass
            else "insufficient_supported_oof_month"
        )
        return {
            **common,
            "arm": "uniform",
            "selection_status": f"uniform_default_{reason}",
            "promotion_reason": reason,
        }

    candidates: list[tuple[tuple[float, float, float, float], str, dict[str, Any]]] = []
    base_ml = uniform["aggregate"]["ml"]
    base_econ = uniform["aggregate"]["economic"]
    base_fold_tail = [
        record["economic"]["top_tail"]["realised_net_ev"] for record in uniform["folds"]
    ]
    base_month_tail = {
        (row["fold_id"], row["side"], row["month_utc"]): row["top_tail"][
            "realised_net_ev"
        ]
        for row in uniform["months"]
        if row["supported"]
    }
    diagnostics: dict[str, Any] = {}
    for name, report in per_arm.items():
        if name == "uniform":
            continue
        ml, econ = report["aggregate"]["ml"], report["aggregate"]["economic"]
        fold_tail = [
            record["economic"]["top_tail"]["realised_net_ev"]
            for record in report["folds"]
        ]
        month_tail = {
            (row["fold_id"], row["side"], row["month_utc"]): row["top_tail"][
                "realised_net_ev"
            ]
            for row in report["months"]
            if row["supported"]
        }
        fold_nonnegative = int(sum(a >= b for a, b in zip(fold_tail, base_fold_tail)))
        month_deltas = [
            month_tail[key] - value for key, value in base_month_tail.items()
        ]
        ml_ok = (
            ml["logloss"] <= base_ml["logloss"] + config.logloss_tolerance
            and ml["brier_macro"] <= base_ml["brier_macro"] + config.brier_tolerance
            and ml["rps"] <= base_ml["rps"] + config.rps_tolerance
            and ml["f1_macro"] >= base_ml["f1_macro"] - config.f1_tolerance
        )
        econ_ok = (
            econ["top_tail"]["realised_net_ev"]
            > base_econ["top_tail"]["realised_net_ev"]
            and econ["net_ev"]["mae"] <= base_econ["net_ev"]["mae"]
            and fold_nonnegative >= int(np.ceil(0.75 * len(fold_tail)))
            and sum(delta >= 0.0 for delta in month_deltas)
            >= int(np.ceil(0.50 * len(month_deltas)))
            and min(month_tail.values()) >= min(base_month_tail.values())
            and min(month_deltas) >= 0.0
        )
        diagnostics[name] = {
            "ml_noninferior": bool(ml_ok),
            "economic_gate": bool(econ_ok),
            "fold_nonnegative_tail_count": fold_nonnegative,
            "month_deltas": month_deltas,
        }
        if ml_ok and econ_ok:
            exponent = float(name.rsplit("_", 1)[-1])
            candidates.append(
                (
                    (
                        econ["top_tail"]["realised_net_ev"],
                        -econ["net_ev"]["mae"],
                        -ml["logloss"],
                        -exponent,
                    ),
                    name,
                    diagnostics[name],
                )
            )
    if not candidates:
        return {
            **common,
            "arm": "uniform",
            "selection_status": "uniform_default_no_nonuniform_arm_passed_lexicographic_gates",
            "promotion_reason": "no_candidate_passed",
            # The control is an OOF-selected arm too.  A fully covered,
            # guarded uniform decision must reach final refit exactly as a
            # promoted non-uniform arm would.
            "promotion_eligible": True,
            "candidate_diagnostics": diagnostics,
        }
    _, selected, selected_diagnostics = max(candidates)
    return {
        **common,
        "arm": selected,
        "selection_status": "economic_oof_promoted",
        "promotion_reason": "passed_strict_ml_and_economic_lexicographic_gates",
        "promotion_eligible": True,
        "candidate_diagnostics": diagnostics,
        "selected_arm_diagnostics": selected_diagnostics,
    }


def score_class_balance_oof_economics(
    outcomes: pd.DataFrame,
    target_codes: Sequence[int],
    arms: Mapping[str, BalanceArmOOF],
    *,
    config: EconomicOOFConfig = EconomicOOFConfig(),
) -> dict[str, Any]:
    """Score matched balance-arm probabilities using canonical v9 OOF outcomes.

    The result's ``selection_provenance`` has the required schema/fields for
    ``rematerialize_final_class_balance_params``.  It remains intentionally
    separate from the classifier while pipeline integration is in flux.
    """
    if not 0.0 < config.tail_fraction <= 1.0 or config.minimum_month_rows < 1:
        raise ValueError("tail_fraction must be in (0, 1] and minimum_month_rows >= 1")
    classes, signature = _validate_matched_arms(outcomes, target_codes, arms, config)
    anchor = arms["uniform"]
    fold_ids = np.asarray(anchor.fold_ids, dtype=int)
    timestamps, _label_end, side = _validate_outcomes_and_folds(
        outcomes,
        signature,
        fold_ids,
        np.asarray(anchor.row_ids),
        config,
    )
    targets = _targets(outcomes)
    labels = np.asarray(target_codes, dtype=int)
    per_arm = {
        name: _score_arm(
            arm, outcomes, labels, targets, timestamps, side, signature, config
        )
        for name, arm in arms.items()
    }
    selection = _select_arm(per_arm, arms, classes, config)
    selector_contract = _selector_contract(config)
    return _json_ready(
        {
            "schema": ECONOMIC_OOF_SCORE_SCHEMA,
            "contract": {
                "outcome_source": "canonical_path_archetype_v9",
                "net_ev": "path_arch_final_return_net_1pct_already_cost_aware_no_second_cost_deduction",
                "train_prior_scope": "exact_purged_fold_train_indices_only",
                "selection_scope": "purged_chronological_oof_validation_only",
                "tail_fraction": config.tail_fraction,
                "minimum_month_rows": config.minimum_month_rows,
                "selector_config": selector_contract,
                "selector_config_sha256": _selector_contract_digest(selector_contract),
            },
            "per_arm": per_arm,
            "selection_provenance": selection,
        }
    )
