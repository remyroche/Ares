"""Leakage-safe training and diagnostics for a 12-hour execution-EV meta head.

The module compares direct and residual side-aware LightGBM regressors, creates
purged chronological OOF predictions, and fits frozen hierarchical EV maps. It
does not select or deploy an execution policy.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .path_archetype_labels import PATH_SHAPE_TYPES

TargetMode = Literal["direct", "residual"]

EXECUTION_EV_BUNDLE_SCHEMA = "execution_ev_side_aware_lgbm_bundle_v2"

# Families are intentionally semantic rather than tied to a particular model
# implementation.  Callers register the actual materialized column names.
EXECUTION_EV_FEATURE_FAMILIES: tuple[str, ...] = (
    "time_to_mfe",
    "peak_mfe",
    "mae_before_meaningful_mfe",
    "adverse_turn_timing",
    "favorable_path_slope",
    "time_to_meaningful_mfe_cdf_probability",
    "catboost_probabilities",
    "catboost_entropy",
    "catboost_probability_confidence",
    "catboost_probability_uncertainty",
    "catboost_path_role_mass",
    "prediction_uncertainty",
    "leaf_support",
    "alpha_score",
    "base_archetype_labels",
)
PREDICTED_PATH_ARCHETYPE_FAMILY = "predicted_path_archetype"
BASE_ARCHETYPE_FEATURE_PREFIX = "base_archetype_label__"

# These names are targets/realized path measures in their unprefixed form.  A
# model prediction such as ``pred_time_to_mfe_12h`` is valid only when it has a
# FeatureProvenance declaration proving it was available at entry time.
FORBIDDEN_OUTCOME_TOKENS: tuple[str, ...] = (
    "realized",
    "future_",
    "label",
    "target",
    "outcome",
    "y_exec",
    "ev_after",
    "ret_net",
    "exec_margin",
    "mfe_before",
    "mae_before",
    "actual_mfe",
    "actual_time_to_mfe",
)


@dataclass(frozen=True)
class FeatureProvenance:
    """Pre-entry availability declaration for one auxiliary input.

    ``available_at_col`` is optional.  When supplied, every finite input must
    have an availability timestamp at or before its decision timestamp.
    """

    family: str
    source: str
    pre_entry: bool = True
    available_at_col: str | None = None
    oof_or_frozen: bool = True
    # Context can be required for calibration without being a numeric model
    # input, as is the case for the predicted path-archetype assignment.
    model_input: bool = True
    # CatBoost path-taxonomy identity.  Legacy handoffs omitted this and are
    # interpreted with ``PATH_SHAPE_TYPES``; new handoffs must carry both.
    class_order: Sequence[str] | None = None
    class_order_sha256: str | None = None


@dataclass(frozen=True)
class ExecutionEVTargetSpec:
    """Definition of the net 12-hour execution target and its residual form."""

    net_ev_col: str = "execution_net_ev_12h"
    alpha_ev_col: str = "existing_alpha_ev"
    mode: TargetMode = "direct"
    target_col: str = "execution_ev_target"
    horizon_hours: float = 12.0


@dataclass(frozen=True)
class ChronologicalPurgedSplit:
    """An expanding chronological split with an explicit pre-validation gap."""

    fold: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    purge_hours: float
    embargo_hours: float


def _utc(values: pd.Series | Sequence[Any], *, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if pd.isna(parsed).any():
        raise ValueError(f"{name} contains invalid timestamps")
    return pd.Series(parsed, index=getattr(values, "index", None))


def _numeric(frame: pd.DataFrame, column: str, *, role: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"Execution-EV {role} is missing required column {column!r}")
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)


def _is_outcome_like(column: str) -> bool:
    name = str(column).lower()
    if name.startswith(("pred_", "oof_", "frozen_", "score_")):
        return False
    return any(token in name for token in FORBIDDEN_OUTCOME_TOKENS)


def validate_execution_ev_feature_provenance(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    provenance: Mapping[str, FeatureProvenance],
    *,
    decision_time_col: str = "__ts__",
    require_model_input: bool = True,
) -> list[str]:
    """Validate an exact, observable auxiliary feature set.

    No feature is inferred from a name alone: every input must have a declared
    source, be marked pre-entry and OOF/frozen, and (where supplied) have an
    availability timestamp no later than the decision time.  Outcome-like raw
    names are rejected even if mistakenly declared safe.
    """

    if decision_time_col not in frame.columns:
        raise ValueError(
            f"Execution-EV provenance requires decision time {decision_time_col!r}"
        )
    decisions = _utc(frame[decision_time_col], name=decision_time_col)
    requested = list(dict.fromkeys(map(str, feature_names)))
    missing = [name for name in requested if name not in frame.columns]
    if missing:
        raise ValueError("Execution-EV inputs missing columns: " + ", ".join(missing))
    undeclared = [name for name in requested if name not in provenance]
    if undeclared:
        raise ValueError(
            "Execution-EV inputs lack provenance declarations: " + ", ".join(undeclared)
        )
    for name in requested:
        spec = provenance[name]
        if spec.family not in (
            *EXECUTION_EV_FEATURE_FAMILIES,
            PREDICTED_PATH_ARCHETYPE_FAMILY,
        ):
            raise ValueError(
                f"Unsupported Execution-EV feature family for {name!r}: {spec.family!r}"
            )
        if not spec.pre_entry or not spec.oof_or_frozen:
            raise ValueError(
                f"Execution-EV input {name!r} must be pre-entry and OOF/frozen"
            )
        if require_model_input and not spec.model_input:
            raise ValueError(
                f"Execution-EV input {name!r} is calibration context, not a model feature"
            )
        is_frozen_base_archetype = (
            spec.family == "base_archetype_labels"
            and name.startswith(BASE_ARCHETYPE_FEATURE_PREFIX)
        )
        if _is_outcome_like(name) and not is_frozen_base_archetype:
            raise ValueError(
                f"Execution-EV input {name!r} appears outcome-derived; use a declared pre-entry prediction output instead"
            )
        if spec.available_at_col is None:
            continue
        if spec.available_at_col not in frame.columns:
            raise ValueError(
                f"Execution-EV input {name!r} references missing availability column {spec.available_at_col!r}"
            )
        available = _utc(frame[spec.available_at_col], name=spec.available_at_col)
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        late = np.isfinite(values) & (available.to_numpy() > decisions.to_numpy())
        if late.any():
            raise ValueError(
                f"Execution-EV input {name!r} has {int(late.sum())} values available after entry"
            )
    return requested


def execution_ev_feature_columns(
    frame: pd.DataFrame,
    provenance: Mapping[str, FeatureProvenance],
    *,
    families: Iterable[str] = EXECUTION_EV_FEATURE_FAMILIES,
    decision_time_col: str = "__ts__",
) -> list[str]:
    """Return exact, validated inputs for the requested semantic families."""

    requested_families = tuple(dict.fromkeys(map(str, families)))
    unknown = sorted(set(requested_families) - set(EXECUTION_EV_FEATURE_FAMILIES))
    if unknown:
        raise ValueError("Unknown Execution-EV feature families: " + ", ".join(unknown))
    names = [
        name
        for name, spec in provenance.items()
        if spec.model_input and spec.family in requested_families
    ]
    return validate_execution_ev_feature_provenance(
        frame, names, provenance, decision_time_col=decision_time_col
    )


def build_execution_ev_target(
    frame: pd.DataFrame,
    spec: ExecutionEVTargetSpec = ExecutionEVTargetSpec(),
) -> pd.Series:
    """Create direct ``Y_exec`` or ``Y_exec - existing_alpha_EV`` target.

    ``net_ev_col`` must already reconcile all execution costs exactly once and
    represent the intended 12-hour policy outcome.  The helper deliberately
    does not derive it from MFE/path columns.
    """

    net_ev = _numeric(frame, spec.net_ev_col, role="net EV target")
    if spec.mode == "direct":
        target = net_ev
    elif spec.mode == "residual":
        alpha = _numeric(frame, spec.alpha_ev_col, role="existing alpha EV")
        target = net_ev - alpha
    else:
        raise ValueError(f"Unsupported Execution-EV target mode: {spec.mode!r}")
    return pd.Series(target, index=frame.index, name=spec.target_col, dtype="float64")


def chronological_purged_splits(
    frame: pd.DataFrame,
    *,
    n_splits: int,
    min_train_size: int,
    min_train_group_col: str | None = None,
    required_train_groups: Sequence[str] | None = None,
    decision_time_col: str = "__ts__",
    label_end_time_col: str | None = None,
    horizon_hours: float = 12.0,
    embargo_hours: float = 12.0,
) -> list[ChronologicalPurgedSplit]:
    """Build expanding OOS folds and purge rows with overlapping label paths.

    Validation blocks are complete decision-timestamp groups.  The embargo is
    an explicit additional gap before each validation block; combined with the
    label-interval purge it prevents the 12-hour policy outcomes near the
    boundary from training the fold.  When ``min_train_group_col`` is supplied,
    every required group must independently meet ``min_train_size`` before a
    validation boundary is eligible.
    """

    if n_splits < 1 or min_train_size < 1:
        raise ValueError("n_splits and min_train_size must both be positive")
    if horizon_hours < 0.0 or embargo_hours < 0.0:
        raise ValueError("horizon_hours and embargo_hours must be non-negative")
    decision = _utc(frame[decision_time_col], name=decision_time_col)
    if label_end_time_col is None:
        label_end = decision + pd.Timedelta(hours=float(horizon_hours))
    else:
        label_end = _utc(frame[label_end_time_col], name=label_end_time_col)
    order = np.argsort(decision.astype("int64").to_numpy(), kind="stable")
    # Use UTC nanoseconds internally.  ``Series.to_numpy`` may otherwise
    # produce object Timestamp arrays whose comparisons differ by pandas
    # version when combined with numpy datetime64 values.
    ordered_decisions = decision.iloc[order].astype("int64").to_numpy()
    unique_times, first = np.unique(ordered_decisions, return_index=True)
    if len(unique_times) <= 1:
        return []
    embargo = pd.Timedelta(hours=float(embargo_hours))
    # A fold boundary starts on a whole timestamp; min_train_size is row based
    # so dense cross-sections are not split apart.
    candidate_blocks: list[int] = []
    label_end_ordered = label_end.iloc[order].astype("int64").to_numpy()
    ordered_groups: np.ndarray | None = None
    groups: tuple[str, ...] = ()
    if min_train_group_col is not None:
        if min_train_group_col not in frame.columns:
            raise ValueError(f"min_train_group_col is missing: {min_train_group_col!r}")
        ordered_groups = frame[min_train_group_col].astype(str).to_numpy()[order]
        groups = tuple(
            map(
                str,
                required_train_groups
                if required_train_groups is not None
                else sorted(pd.unique(ordered_groups)),
            )
        )
        if not groups:
            raise ValueError("required_train_groups cannot be empty")
        missing_groups = sorted(set(groups).difference(ordered_groups))
        if missing_groups:
            raise ValueError(
                f"required train groups are absent from the frame: {missing_groups}"
            )
    for block, validation_start in enumerate(unique_times):
        retained_train = (
            (ordered_decisions < validation_start)
            & (label_end_ordered <= validation_start)
            & (ordered_decisions <= int(validation_start) - int(embargo.value))
        )
        group_ready = ordered_groups is None or all(
            int(np.sum(retained_train & (ordered_groups == group)))
            >= int(min_train_size)
            for group in groups
        )
        if int(retained_train.sum()) >= int(min_train_size) and group_ready:
            candidate_blocks.append(block)
    eligible = np.asarray(candidate_blocks, dtype=int)
    if len(eligible) < n_splits:
        raise ValueError("Not enough complete timestamp blocks after min_train_size")
    # Leave a real validation interval after every start.  Selecting the last
    # eligible boundary would create a one-timestamp final fold rather than an
    # expanding OOS block.
    starts = eligible[np.linspace(0, len(eligible), n_splits + 1, dtype=int)[:-1]]
    starts = np.unique(starts)
    splits: list[ChronologicalPurgedSplit] = []
    for fold, start_block in enumerate(starts):
        end_block = starts[fold + 1] if fold + 1 < len(starts) else len(unique_times)
        val_start_ns = int(unique_times[start_block])
        val_end_ns = int(unique_times[end_block - 1])
        val_start = pd.Timestamp(val_start_ns, tz="UTC")
        val_end = pd.Timestamp(val_end_ns, tz="UTC")
        validation = order[
            (ordered_decisions >= val_start_ns) & (ordered_decisions <= val_end_ns)
        ]
        train = order[
            (ordered_decisions < val_start_ns)
            & (label_end_ordered <= val_start_ns)
            & (ordered_decisions <= val_start_ns - int(embargo.value))
        ]
        group_ready = ordered_groups is None or all(
            int(np.sum(ordered_groups[np.isin(order, train)] == group))
            >= int(min_train_size)
            for group in groups
        )
        if len(train) < min_train_size or not group_ready or not len(validation):
            continue
        splits.append(
            ChronologicalPurgedSplit(
                fold=fold,
                train_indices=np.sort(train),
                validation_indices=np.sort(validation),
                validation_start=val_start,
                validation_end=val_end,
                purge_hours=float(horizon_hours),
                embargo_hours=float(embargo_hours),
            )
        )
    if not splits:
        raise ValueError("Purging and embargo left no valid chronological folds")
    return splits


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 2:
        return float("nan")
    lhs = pd.Series(left[valid]).rank(method="average").to_numpy(dtype=float)
    rhs = pd.Series(right[valid]).rank(method="average").to_numpy(dtype=float)
    if np.std(lhs) <= 1e-12 or np.std(rhs) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(lhs, rhs)[0, 1])


def _huber_loss(residual: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(residual)
    return np.where(
        absolute <= delta, 0.5 * residual**2, delta * (absolute - 0.5 * delta)
    )


def execution_ev_metrics(
    realized_net_ev: Sequence[float] | np.ndarray,
    predicted_net_ev: Sequence[float] | np.ndarray,
    *,
    top_k_fraction: float = 0.10,
    huber_delta: float = 0.01,
) -> dict[str, float | int]:
    """Return regression and traded-tail net-EV metrics on identical rows."""

    actual = np.asarray(realized_net_ev, dtype=float)
    predicted = np.asarray(predicted_net_ev, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError(
            "Execution-EV realized and predicted arrays must have identical shape"
        )
    if not 0.0 < top_k_fraction <= 1.0 or huber_delta <= 0.0:
        raise ValueError(
            "top_k_fraction must be in (0, 1] and huber_delta must be positive"
        )
    valid = np.isfinite(actual) & np.isfinite(predicted)
    y = actual[valid]
    p = predicted[valid]
    if not len(y):
        return {
            "rows": 0,
            "mae": float("nan"),
            "huber": float("nan"),
            "rmse": float("nan"),
            "spearman": float("nan"),
            "top_k_rows": 0,
            "top_k_mean_net_ev": float("nan"),
            "top_k_sum_net_ev": float("nan"),
            "top_k_predicted_net_ev": float("nan"),
            "positive_ev_rate": float("nan"),
            "positive_ev_auc": float("nan"),
            "top_k_positive_ev_rate": float("nan"),
            "prediction_bias": float("nan"),
        }
    residual = p - y
    top_k = max(1, int(np.ceil(len(y) * top_k_fraction)))
    top = np.argsort(p, kind="stable")[-top_k:]
    positive = y > 0.0
    positive_auc = (
        float(roc_auc_score(positive.astype(np.int8), p))
        if np.unique(positive).size == 2
        else float("nan")
    )
    return {
        "rows": int(len(y)),
        "mae": float(np.mean(np.abs(residual))),
        "huber": float(np.mean(_huber_loss(residual, float(huber_delta)))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "spearman": _spearman(p, y),
        "top_k_rows": int(top_k),
        "top_k_mean_net_ev": float(np.mean(y[top])),
        "top_k_sum_net_ev": float(np.sum(y[top])),
        "top_k_predicted_net_ev": float(np.mean(p[top])),
        "positive_ev_rate": float(np.mean(positive)),
        "positive_ev_auc": positive_auc,
        "top_k_positive_ev_rate": float(np.mean(positive[top])),
        "prediction_bias": float(np.mean(p - y)),
    }


def compare_direct_and_residual(
    frame: pd.DataFrame,
    *,
    direct_prediction_col: str,
    residual_prediction_col: str,
    target_spec: ExecutionEVTargetSpec = ExecutionEVTargetSpec(),
    top_k_fraction: float = 0.10,
    huber_delta: float = 0.01,
) -> dict[str, float | int]:
    """Evaluate direct ``Y_exec`` and residual predictions in net-EV units."""

    actual = _numeric(frame, target_spec.net_ev_col, role="net EV target")
    direct = _numeric(frame, direct_prediction_col, role="direct prediction")
    residual = _numeric(frame, residual_prediction_col, role="residual prediction")
    alpha = _numeric(frame, target_spec.alpha_ev_col, role="existing alpha EV")
    direct_metrics = execution_ev_metrics(
        actual, direct, top_k_fraction=top_k_fraction, huber_delta=huber_delta
    )
    residual_metrics = execution_ev_metrics(
        actual, residual + alpha, top_k_fraction=top_k_fraction, huber_delta=huber_delta
    )
    report: dict[str, float | int] = {
        **{f"direct__{key}": value for key, value in direct_metrics.items()},
        **{f"residual__{key}": value for key, value in residual_metrics.items()},
    }
    for key in ("mae", "huber", "rmse", "top_k_mean_net_ev", "top_k_sum_net_ev"):
        report[f"residual_minus_direct__{key}"] = float(
            residual_metrics[key] - direct_metrics[key]
        )
    return report


def execution_ev_ablation_plan(
    provenance: Mapping[str, FeatureProvenance],
    *,
    include_leave_one_family_out: bool = True,
) -> dict[str, tuple[str, ...]]:
    """Create a reproducible baseline/full/leave-one-family-out input matrix."""

    by_family = {
        family: tuple(
            name
            for name, spec in provenance.items()
            if spec.family == family and spec.model_input
        )
        for family in EXECUTION_EV_FEATURE_FAMILIES
    }
    alpha = by_family["alpha_score"]
    if not alpha:
        raise ValueError(
            "Execution-EV ablations require an alpha_score baseline feature"
        )
    full = tuple(
        name for family in EXECUTION_EV_FEATURE_FAMILIES for name in by_family[family]
    )
    context_families = (
        "alpha_score",
        "prediction_uncertainty",
        "leaf_support",
        "base_archetype_labels",
    )
    auxiliary_families = (
        "peak_mfe",
        "time_to_mfe",
        "time_to_meaningful_mfe_cdf_probability",
    )
    catboost_families = (
        "catboost_probabilities",
        "catboost_entropy",
        "catboost_probability_confidence",
        "catboost_probability_uncertainty",
        "catboost_path_role_mass",
    )

    def features(families: Sequence[str]) -> tuple[str, ...]:
        return tuple(name for family in families for name in by_family[family])

    alpha_context = features(context_families)
    plan: dict[str, tuple[str, ...]] = {
        "alpha_only": alpha,
        "alpha_context": alpha_context,
        "alpha_context_plus_aux": tuple(
            dict.fromkeys((*alpha_context, *features(auxiliary_families)))
        ),
        "alpha_context_plus_catboost": tuple(
            dict.fromkeys((*alpha_context, *features(catboost_families)))
        ),
        "all_features": full,
    }
    if include_leave_one_family_out:
        for family in EXECUTION_EV_FEATURE_FAMILIES:
            if family == "alpha_score" or not by_family[family]:
                continue
            plan[f"without_{family}"] = tuple(
                name for name in full if name not in by_family[family]
            )

    return plan


def timing_slope_ablation_comparison(
    ablation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare timing and slope without fitting duplicate ablation arms."""

    indexed = {str(row.get("arm")): row for row in ablation_rows}
    timing_context = indexed.get("without_favorable_path_slope")
    slope_context = indexed.get("without_time_to_mfe")
    if timing_context is None or slope_context is None:
        return {"status": "unavailable"}
    timing_ev = float(timing_context["top_k_mean_net_ev"])
    slope_ev = float(slope_context["top_k_mean_net_ev"])
    timing_mae = float(timing_context["mae"])
    slope_mae = float(slope_context["mae"])
    return {
        "status": "available",
        "comparison_contract": (
            "same OOF folds and all other feature families held fixed; "
            "timing context removes slope, slope context removes timing"
        ),
        "timing_context_arm": "without_favorable_path_slope",
        "slope_context_arm": "without_time_to_mfe",
        "timing_top10_mean_net_ev": timing_ev,
        "slope_top10_mean_net_ev": slope_ev,
        "slope_minus_timing_top10_mean_net_ev": slope_ev - timing_ev,
        "timing_mae": timing_mae,
        "slope_mae": slope_mae,
        "preferred_by_top10_net_ev": "slope" if slope_ev > timing_ev else "timing",
        "preferred_by_mae": "slope" if slope_mae < timing_mae else "timing",
    }


def execution_ev_ablation_metrics(
    realized_net_ev: Sequence[float] | np.ndarray,
    predictions_by_arm: Mapping[str, Sequence[float] | np.ndarray],
    *,
    top_k_fraction: float = 0.10,
    huber_delta: float = 0.01,
) -> pd.DataFrame:
    """Score precomputed OOS predictions for every ablation arm.

    Prediction fitting is intentionally external so every arm can be trained
    with the same chronological/purged folds and frozen transforms.
    """

    rows: list[dict[str, Any]] = []
    for arm, prediction in predictions_by_arm.items():
        rows.append(
            {
                "arm": str(arm),
                **execution_ev_metrics(
                    realized_net_ev,
                    prediction,
                    top_k_fraction=top_k_fraction,
                    huber_delta=huber_delta,
                ),
            }
        )
    report = pd.DataFrame(rows).sort_values("arm", kind="stable").reset_index(drop=True)
    if "all_features" not in set(report["arm"]):
        return report

    full = report.loc[report["arm"] == "all_features"].iloc[0]
    # Positive values always favor retaining the input group in all_features.
    # This makes leave-one-family-out output directly usable rather than
    # requiring downstream consumers to remember metric directionality.
    for metric in ("mae", "huber", "rmse"):
        report[f"all_features_advantage__{metric}"] = report[metric] - float(
            full[metric]
        )
    for metric in ("spearman", "top_k_mean_net_ev", "top_k_sum_net_ev"):
        report[f"all_features_advantage__{metric}"] = (
            float(full[metric]) - report[metric]
        )

    groups: list[str] = []
    verdicts: list[str] = []
    for _, row in report.iterrows():
        arm = str(row["arm"])
        group = (
            arm.removeprefix("without_")
            if arm.startswith("without_")
            else ("all_non_alpha_features" if arm == "alpha_only" else "all_features")
        )
        groups.append(group)
        if arm == "all_features":
            verdicts.append("reference")
        elif (
            row["all_features_advantage__top_k_mean_net_ev"] > 0.0
            and row["all_features_advantage__mae"] >= 0.0
        ):
            verdicts.append("helps")
        elif (
            row["all_features_advantage__top_k_mean_net_ev"] < 0.0
            and row["all_features_advantage__mae"] <= 0.0
        ):
            verdicts.append("hurts")
        else:
            verdicts.append("mixed")
    report.insert(1, "input_group", groups)
    report["all_features_contribution"] = verdicts
    return report


def fit_side_archetype_monotonic_ev_mapping(*args: Any, **kwargs: Any) -> Any:
    """Hook to the canonical frozen side x archetype hierarchical EV mapper.

    Call this only with authorized chronological OOF rows.  It is imported
    lazily to keep this auxiliary contract usable without training dependencies.
    """

    from .supervised_market_state_calibration import fit_hierarchical_ev_calibrator

    return fit_hierarchical_ev_calibrator(*args, **kwargs)


def predict_side_archetype_monotonic_ev_mapping(
    *args: Any, **kwargs: Any
) -> np.ndarray:
    """Predict with the canonical side/archetype-aware monotonic EV mapping."""

    from .supervised_market_state_calibration import predict_hierarchical_ev

    return np.asarray(predict_hierarchical_ev(*args, **kwargs), dtype=np.float64)


@dataclass(frozen=True)
class ExecutionEVTrainerConfig:
    """Configuration for the leakage-safe side-aware execution-EV trainer.

    ``catboost_archetype_col`` is the *pre-entry CatBoost assignment*, not a
    realized-path archetype.  The complete CatBoost probability vector remains
    in ``provenance`` as numeric ``catboost_probabilities`` inputs.
    """

    n_splits: int = 3
    min_train_rows: int = 500
    purge_hours: float = 12.0
    embargo_hours: float = 12.0
    hpo_trials: int = 20
    inner_n_splits: int = 2
    # Selection is deliberately an explicit, side-local stage rather than an
    # incidental by-product of the fitted booster.  The selector is fit only
    # on the relevant outer-train rows, uses its own train-only inner OOF
    # permutation scores, then freezes the resulting side feature set before
    # the side HPO/model fit begins.
    feature_selection_enabled: bool = True
    feature_selection_max_features: int = 12
    feature_selection_min_features: int = 6
    feature_selection_n_estimators: int = 400
    feature_selection_min_importance: float = 0.0
    early_stopping_rounds: int = 150
    n_estimators: int = 3000
    random_state: int = 42
    side_col: str = "side_name"
    catboost_archetype_col: str = "catboost_archetype"
    decision_time_col: str = "__ts__"
    label_end_time_col: str | None = None
    huber_delta: float = 0.01
    run_ablations: bool = True
    calibration_min_rows: int = 100
    calibration_min_local_rows: int = 400
    calibration_shrink_rows: float = 2_000.0
    n_jobs: int = 3


@dataclass
class ExecutionEVModelBundle:
    """Persistable final models and train-derived transforms for live scoring."""

    schema: str
    config: dict[str, Any]
    provenance: dict[str, FeatureProvenance]
    feature_plan: dict[str, tuple[str, ...]]
    models: dict[str, dict[str, Any]]
    calibration: dict[str, dict[str, Any]]
    report: dict[str, Any]
    oof_predictions: pd.DataFrame = field(repr=False)
    # This maps each OOF score back to its outer-fold decision and training
    # cutoff. Identity columns remain with the caller's joined handoff.
    oof_provenance: pd.DataFrame = field(repr=False)


def _side_values(frame: pd.DataFrame, side_col: str) -> np.ndarray:
    if side_col not in frame.columns:
        raise ValueError(f"Execution-EV trainer requires side column {side_col!r}")
    sides = frame[side_col].astype(str).str.lower().to_numpy()
    unknown = sorted(set(sides) - {"long", "short"})
    if unknown:
        raise ValueError(
            "Execution-EV side values must be long/short; got "
            + ", ".join(unknown[:10])
        )
    return sides


def _catboost_archetypes(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(
            "Execution-EV trainer requires pre-entry CatBoost archetype column "
            f"{column!r} for the hierarchical side x archetype EV map"
        )
    return frame[column].fillna("missing").astype(str).to_numpy()


def _finite_target_rows(frame: pd.DataFrame, spec: ExecutionEVTargetSpec) -> np.ndarray:
    target = build_execution_ev_target(frame, spec).to_numpy(dtype=float)
    return np.isfinite(target)


def catboost_class_order_sha256(class_order: Sequence[str]) -> str:
    """Return the canonical digest for an ordered CatBoost class taxonomy."""

    normalized = tuple(str(name).strip() for name in class_order)
    if not normalized or any(not name for name in normalized):
        raise ValueError("CatBoost class order must contain non-empty class names")
    if len(set(normalized)) != len(normalized):
        raise ValueError("CatBoost class order must not contain duplicate class names")
    encoded = json.dumps(list(normalized), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _catboost_class_order(
    provenance: Mapping[str, FeatureProvenance],
    *,
    probability_columns: Sequence[str],
    entropy_columns: Sequence[str],
    archetype_column: str,
) -> tuple[str, ...]:
    """Resolve the declared CatBoost taxonomy, with an eight-class fallback."""

    records = [*probability_columns, *entropy_columns, archetype_column]
    declared = [name for name in records if provenance[name].class_order is not None]
    declared_hashes = [
        name for name in records if provenance[name].class_order_sha256 is not None
    ]
    if not declared and not declared_hashes:
        return tuple(PATH_SHAPE_TYPES)
    missing = [
        name
        for name in records
        if provenance[name].class_order is None
        or not str(provenance[name].class_order_sha256 or "").strip()
    ]
    if missing:
        raise ValueError(
            "Execution-EV CatBoost class contract is incomplete for: "
            + ", ".join(missing)
        )
    orders = {
        tuple(str(value).strip() for value in provenance[name].class_order or ())
        for name in records
    }
    if len(orders) != 1:
        raise ValueError(
            "Execution-EV CatBoost class order disagrees across provenance"
        )
    order = next(iter(orders))
    expected_hash = catboost_class_order_sha256(order)
    mismatched_hashes = [
        name
        for name in records
        if not hmac.compare_digest(
            str(provenance[name].class_order_sha256), expected_hash
        )
    ]
    if mismatched_hashes:
        raise ValueError(
            "Execution-EV CatBoost class-order hash does not match the declared order for: "
            + ", ".join(mismatched_hashes)
        )
    return order


def validate_execution_ev_training_contract(
    frame: pd.DataFrame,
    provenance: Mapping[str, FeatureProvenance],
    *,
    decision_time_col: str = "__ts__",
    predicted_path_archetype_col: str = "catboost_archetype",
) -> list[str]:
    """Require the complete frozen alpha/path/CatBoost support handoff.

    A signed handoff may declare a promoted taxonomy; older handoffs retain the
    legacy eight-class order. The provenance declaration is intentionally
    explicit: source descriptions provide audit context while ``oof_or_frozen``
    prevents same-fold alpha-stack leakage.
    """
    names = execution_ev_feature_columns(
        frame, provenance, decision_time_col=decision_time_col
    )
    by_family = {
        family: [name for name in names if provenance[name].family == family]
        for family in EXECUTION_EV_FEATURE_FAMILIES
    }
    required = (
        "peak_mfe",
        "catboost_probabilities",
        "catboost_entropy",
        "prediction_uncertainty",
        "leaf_support",
        "alpha_score",
        "base_archetype_labels",
    )
    missing = [family for family in required if not by_family[family]]
    if missing:
        raise ValueError(
            "Execution-EV trainer missing required feature families: "
            + ", ".join(missing)
        )
    timing_families = {
        provenance[name].family
        for name in names
        if provenance[name].family
        in {"time_to_mfe", "time_to_meaningful_mfe_cdf_probability"}
    }
    if not timing_families:
        raise ValueError(
            "Execution-EV trainer requires an OOF timing input: either the scalar "
            "timing head or promotion-audited timing-CDF probabilities"
        )
    archetype_spec = provenance.get(predicted_path_archetype_col)
    if (
        archetype_spec is None
        or archetype_spec.family != PREDICTED_PATH_ARCHETYPE_FAMILY
        or archetype_spec.model_input
    ):
        raise ValueError(
            "Execution-EV trainer requires the predicted pre-entry path-archetype "
            f"provenance declaration for {predicted_path_archetype_col!r}"
        )
    probability_columns = by_family["catboost_probabilities"]
    entropy_columns = by_family["catboost_entropy"]
    class_order = _catboost_class_order(
        provenance,
        probability_columns=probability_columns,
        entropy_columns=entropy_columns,
        archetype_column=predicted_path_archetype_col,
    )
    if len(probability_columns) != len(class_order):
        raise ValueError(
            "Execution-EV trainer requires the full CatBoost probability vector "
            f"({len(class_order)} catboost_probabilities features)"
        )
    active = _finite_target_rows(frame, ExecutionEVTargetSpec())
    probabilities = (
        frame.loc[:, probability_columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )
    if active.any() and not np.isfinite(probabilities[active]).all():
        raise ValueError(
            "Execution-EV CatBoost probability vector is incomplete on finite-target rows"
        )
    active_probabilities = probabilities[active]
    if active_probabilities.size:
        if (active_probabilities < -1e-6).any() or (
            active_probabilities > 1.0 + 1e-6
        ).any():
            raise ValueError("Execution-EV CatBoost probabilities must lie in [0, 1]")
        probability_sum = active_probabilities.sum(axis=1)
        if not np.allclose(probability_sum, 1.0, atol=1e-4, rtol=1e-4):
            worst = float(np.max(np.abs(probability_sum - 1.0)))
            raise ValueError(
                "Execution-EV CatBoost probability vector is not normalized; "
                f"max_abs_sum_error={worst:.6g}"
            )
        entropy = pd.to_numeric(
            frame.loc[:, entropy_columns[0]], errors="coerce"
        ).to_numpy(dtype=np.float64)[active]
        expected_entropy = -np.sum(
            np.clip(active_probabilities, 1e-12, 1.0)
            * np.log(np.clip(active_probabilities, 1e-12, 1.0)),
            axis=1,
        )
        if not np.isfinite(entropy).all() or not np.allclose(
            entropy, expected_entropy, atol=1e-4, rtol=1e-4
        ):
            raise ValueError(
                "Execution-EV CatBoost entropy does not match the declared full probability vector"
            )
        predicted = (
            frame.loc[active, predicted_path_archetype_col].astype(str).to_numpy()
        )
        expected = np.asarray(class_order, dtype=object)[
            np.argmax(active_probabilities, axis=1)
        ]
        if not np.array_equal(predicted, expected):
            raise ValueError(
                "Execution-EV predicted path archetype does not match argmax of the full probability vector"
            )
    empty_source = [name for name in names if not str(provenance[name].source).strip()]
    if empty_source:
        raise ValueError(
            "Execution-EV provenance requires non-empty sources: "
            + ", ".join(empty_source)
        )
    validate_execution_ev_feature_provenance(
        frame,
        [predicted_path_archetype_col],
        provenance,
        decision_time_col=decision_time_col,
        require_model_input=False,
    )
    return names


def _lgbm_params(
    config: ExecutionEVTrainerConfig, trial: Any | None = None
) -> dict[str, Any]:
    if trial is None:
        return {
            "objective": "huber",
            "n_estimators": int(config.n_estimators),
            "learning_rate": 0.03,
            "max_depth": 5,
            "num_leaves": 24,
            "min_child_samples": 100,
            "min_split_gain": 1e-3,
            "reg_alpha": 0.1,
            "reg_lambda": 5.0,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "max_bin": 127,
            "random_state": int(config.random_state),
            "n_jobs": int(config.n_jobs),
            "verbosity": -1,
        }
    return {
        "objective": trial.suggest_categorical(
            "objective", ["regression", "huber", "fair"]
        ),
        "n_estimators": int(config.n_estimators),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.06, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "num_leaves": trial.suggest_categorical("num_leaves", [8, 16, 24, 32, 48, 64]),
        "min_child_samples": trial.suggest_int("min_child_samples", 30, 800, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 1e-4, 0.05, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 40.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.60, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 1.0),
        "max_bin": trial.suggest_categorical("max_bin", [63, 127, 255]),
        "random_state": int(config.random_state),
        "n_jobs": int(config.n_jobs),
        "verbosity": -1,
    }


def _hpo_score(
    y: np.ndarray, prediction: np.ndarray, config: ExecutionEVTrainerConfig
) -> float:
    metrics = execution_ev_metrics(
        y, prediction, top_k_fraction=0.10, huber_delta=config.huber_delta
    )
    ic = metrics["spearman"] if np.isfinite(metrics["spearman"]) else -1.0
    tail = (
        metrics["top_k_mean_net_ev"]
        if np.isfinite(metrics["top_k_mean_net_ev"])
        else -1.0
    )
    return float(-metrics["huber"] + 0.25 * ic + tail)


def _inner_splits(
    frame: pd.DataFrame, config: ExecutionEVTrainerConfig
) -> list[ChronologicalPurgedSplit]:
    if len(frame) < max(8, config.min_train_rows):
        return []
    try:
        return chronological_purged_splits(
            frame,
            n_splits=config.inner_n_splits,
            min_train_size=max(4, min(config.min_train_rows, len(frame) // 3)),
            decision_time_col=config.decision_time_col,
            label_end_time_col=config.label_end_time_col,
            horizon_hours=config.purge_hours,
            embargo_hours=config.embargo_hours,
        )
    except ValueError:
        return []


def _fit_lgbm(
    x: pd.DataFrame,
    y: np.ndarray,
    *,
    fit_indices: np.ndarray,
    early_stop: ChronologicalPurgedSplit | None,
    params: Mapping[str, Any],
    config: ExecutionEVTrainerConfig,
) -> tuple[Any, int]:
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("LightGBM is required for execution-EV training") from exc
    model = lgb.LGBMRegressor(**dict(params))
    if early_stop is None:
        model.fit(x.iloc[fit_indices], y[fit_indices])
    else:
        train = np.asarray(early_stop.train_indices, dtype=int)
        valid = np.asarray(early_stop.validation_indices, dtype=int)
        # The supplied inner split indexes the already-authorized outer-train
        # frame, so no outer validation outcome can influence early stopping.
        model.fit(
            x.iloc[train],
            y[train],
            eval_set=[(x.iloc[valid], y[valid])],
            callbacks=[
                lgb.early_stopping(int(config.early_stopping_rounds), verbose=False)
            ],
        )
    return model, int(model.best_iteration_ or params["n_estimators"])


def _tune_lgbm(
    x: pd.DataFrame,
    y: np.ndarray,
    inner: list[ChronologicalPurgedSplit],
    config: ExecutionEVTrainerConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fallback = _lgbm_params(config)
    if not inner or config.hpo_trials <= 0:
        return fallback, {"status": "default_no_inner_hpo", "trials": 0}
    try:
        import lightgbm as lgb
        import optuna
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise RuntimeError(
            "LightGBM and Optuna are required for execution-EV HPO"
        ) from exc

    def objective(trial: Any) -> float:
        params = _lgbm_params(config, trial)
        scores: list[float] = []
        for fold_i, fold in enumerate(inner):
            model = lgb.LGBMRegressor(**params)
            model.fit(
                x.iloc[fold.train_indices],
                y[fold.train_indices],
                eval_set=[
                    (x.iloc[fold.validation_indices], y[fold.validation_indices])
                ],
                callbacks=[
                    lgb.early_stopping(int(config.early_stopping_rounds), verbose=False)
                ],
            )
            pred = model.predict(x.iloc[fold.validation_indices])
            scores.append(_hpo_score(y[fold.validation_indices], pred, config))
            trial.report(float(np.mean(scores)), step=fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores) - 0.25 * np.std(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(config.random_state)),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=min(8, config.hpo_trials), n_warmup_steps=1
        ),
    )
    study.optimize(objective, n_trials=int(config.hpo_trials), show_progress_bar=False)
    params = _lgbm_params(config)
    params.update(study.best_params)
    return params, {
        "status": "tuned",
        "trials": int(len(study.trials)),
        "best_value": float(study.best_value),
    }


def _feature_list_sha256(features: Sequence[str]) -> str:
    """Return a stable audit digest for an ordered feature contract."""

    payload = json.dumps(list(map(str, features)), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _selection_rank_fallback(
    x: pd.DataFrame,
    y: np.ndarray,
) -> dict[str, float]:
    """Rank inputs from authorized rows when no inner OOF slice is viable.

    This is intentionally only a low-support fallback.  It is still fitted on
    the side's outer-training rows, never outer validation rows, and records
    its weaker contract in the selector audit rather than pretending to be
    inner-OOF MDA.
    """

    scores: dict[str, float] = {}
    for column in x.columns:
        value = pd.to_numeric(x[column], errors="coerce").to_numpy(dtype=float)
        association = _spearman(value, y)
        scores[str(column)] = (
            float(abs(association)) if np.isfinite(association) else 0.0
        )
    return scores


def _select_side_train_only_features(
    x: pd.DataFrame,
    y: np.ndarray,
    inner: Sequence[ChronologicalPurgedSplit],
    *,
    side: str,
    protected_features: Sequence[str],
    config: ExecutionEVTrainerConfig,
) -> dict[str, Any]:
    """Select an independent side-local contract using authorized rows only.

    The normal path fits a fixed-parameter selector model on each chronological
    inner-training slice and measures the loss of each feature by permuting it
    in that inner validation slice.  Those validation outcomes are part of the
    *outer training* interval, so neither selection nor its feature ranking can
    observe the outer OOF rows.  The selected list is frozen before HPO.

    No side is ever mixed into this selector: callers pass only one side's
    outer-train matrix.  ``protected_features`` keeps the alpha baseline in
    every fitted contract, which makes the named ablations comparable.
    """

    candidates = list(map(str, x.columns))
    if not candidates:
        raise ValueError(
            "Execution-EV feature selection requires at least one candidate"
        )
    if len(y) != len(x):
        raise ValueError("Execution-EV feature selection inputs have inconsistent rows")
    canonical_side = str(side).lower()
    if canonical_side not in {"long", "short"}:
        raise ValueError(
            f"Execution-EV feature selection requires long/short side, got {side!r}"
        )
    protected = [
        feature
        for feature in dict.fromkeys(map(str, protected_features))
        if feature in candidates
    ]
    if not config.feature_selection_enabled:
        selected = candidates
        return {
            "schema": "execution_ev_side_local_feature_selection_v1",
            "status": "disabled_all_candidates",
            "method": "disabled",
            "side": canonical_side,
            "train_rows": int(len(x)),
            "inner_folds": 0,
            "candidate_features": candidates,
            "protected_features": protected,
            "selected_features": selected,
            "selected_features_sha256": _feature_list_sha256(selected),
            "feature_importance_mean": {},
        }

    max_features = int(config.feature_selection_max_features)
    if max_features <= 0:
        max_features = len(candidates)
    max_features = min(len(candidates), max(max_features, len(protected)))
    min_features = min(
        max_features,
        max(
            len(protected),
            min(int(config.feature_selection_min_features), len(candidates)),
        ),
    )
    # A deterministic side-specific seed makes the permutation ledger
    # reproducible without sharing any fit state across long/short.
    seed_bytes = hashlib.sha256(
        f"execution-ev-selector:{config.random_state}:{canonical_side}".encode("utf-8")
    ).digest()
    seed = int.from_bytes(seed_bytes[:4], byteorder="little", signed=False)
    scores: dict[str, list[float]] = {feature: [] for feature in candidates}
    selector_params = _lgbm_params(config)
    selector_params["n_estimators"] = max(
        1,
        min(
            int(selector_params["n_estimators"]),
            int(config.feature_selection_n_estimators),
        ),
    )

    method = "inner_oof_permutation_mda"
    for inner_fold in inner:
        train = np.asarray(inner_fold.train_indices, dtype=int)
        valid = np.asarray(inner_fold.validation_indices, dtype=int)
        if len(train) < 4 or not len(valid):
            continue
        model, _ = _fit_lgbm(
            x,
            y,
            fit_indices=train,
            early_stop=None,
            params=selector_params,
            config=config,
        )
        valid_x = x.iloc[valid]
        baseline = model.predict(valid_x)
        baseline_score = _hpo_score(y[valid], baseline, config)
        # Fold is included so permutation is stable but not accidentally
        # identical across validation intervals.
        fold_rng = np.random.default_rng(seed + int(inner_fold.fold))
        for feature in candidates:
            permuted = valid_x.copy()
            values = permuted[feature].to_numpy(copy=True)
            permuted[feature] = values[fold_rng.permutation(len(values))]
            permuted_score = _hpo_score(y[valid], model.predict(permuted), config)
            scores[feature].append(float(baseline_score - permuted_score))

    usable_inner = sum(bool(values) for values in scores.values())
    if not usable_inner:
        method = "train_only_abs_spearman_fallback"
        importance = _selection_rank_fallback(x, y)
        inner_folds = 0
    else:
        importance = {
            feature: float(np.mean(values)) if values else float("-inf")
            for feature, values in scores.items()
        }
        inner_folds = len(inner)

    # Stable sorting makes selection reproducible even for ties.  Negative MDA
    # inputs are retained only when required to meet declared minimum support.
    ranked = sorted(
        (feature for feature in candidates if feature not in protected),
        key=lambda feature: (-importance[feature], candidates.index(feature)),
    )
    selected = list(protected)
    for feature in ranked:
        if len(selected) >= max_features:
            break
        if importance[feature] >= float(config.feature_selection_min_importance):
            selected.append(feature)
    for feature in ranked:
        if len(selected) >= min_features:
            break
        if feature not in selected:
            selected.append(feature)
    # The requested feature order, not importance order, is the model matrix
    # contract.  This keeps final inference deterministic and audit-friendly.
    selected = [feature for feature in candidates if feature in set(selected)]
    if not selected:
        raise ValueError("Execution-EV feature selection selected no inputs")
    return {
        "schema": "execution_ev_side_local_feature_selection_v1",
        "status": "selected",
        "method": method,
        "side": canonical_side,
        "train_rows": int(len(x)),
        "inner_folds": int(inner_folds),
        "candidate_features": candidates,
        "protected_features": protected,
        "selected_features": selected,
        "selected_features_sha256": _feature_list_sha256(selected),
        "feature_importance_mean": {
            feature: float(importance[feature]) for feature in candidates
        },
        "selector_params": dict(selector_params),
    }


def _derived_ablation_selection(
    source: Mapping[str, Any],
    candidates: Sequence[str],
    *,
    protected_features: Sequence[str],
    side: str,
) -> dict[str, Any]:
    """Derive a fair ablation contract from an all-feature train-only choice.

    Re-running selection for every leave-one-family-out arm would turn an
    attribution exercise into a broad feature-search and dramatically increase
    compute.  Instead every ablation uses the matching direct/residual,
    side-local all-feature selector that was already frozen for that same outer
    train interval; the arm removes only its named candidates.
    """

    source_selected = list(map(str, source.get("selected_features", ())))
    if not source_selected:
        raise ValueError(
            "Execution-EV ablation selection source has no selected features"
        )
    candidate_list = list(map(str, candidates))
    protected = [
        feature
        for feature in dict.fromkeys(map(str, protected_features))
        if feature in candidate_list
    ]
    selected_set = set(source_selected) | set(protected)
    selected = [feature for feature in candidate_list if feature in selected_set]
    if not selected:
        # This can only happen for a malformed named arm that drops every
        # protected feature.  Retaining its own candidates preserves the arm
        # definition while keeping the failure explicit in the audit.
        selected = candidate_list
    return {
        "schema": "execution_ev_side_local_feature_selection_v1",
        "status": "derived_from_all_features_train_only_selection",
        "method": "frozen_all_features_intersection",
        "side": str(side).lower(),
        "train_rows": int(source.get("train_rows", 0)),
        "inner_folds": int(source.get("inner_folds", 0)),
        "candidate_features": candidate_list,
        "protected_features": protected,
        "selected_features": selected,
        "selected_features_sha256": _feature_list_sha256(selected),
        "source_selected_features_sha256": str(
            source.get("selected_features_sha256", "")
        ),
        "source_selector_method": str(source.get("method", "")),
    }


def _train_only_calibrator(
    frame: pd.DataFrame,
    raw_oof: np.ndarray,
    target: np.ndarray,
    *,
    side_col: str,
    archetype_col: str,
    config: ExecutionEVTrainerConfig,
) -> Any | None:
    valid = np.isfinite(raw_oof) & np.isfinite(target)
    # The canonical mapper has its own 100-row minimum.  Keep this explicit
    # rather than catching its validation error and accidentally hiding it.
    if int(valid.sum()) < max(100, int(config.calibration_min_rows)):
        return None
    calibration_frame = pd.DataFrame(
        {
            "side_name": frame[side_col].astype(str).to_numpy(),
            "archetype_policy_key": frame[archetype_col]
            .fillna("missing")
            .astype(str)
            .to_numpy(),
        },
        index=frame.index,
    ).iloc[np.flatnonzero(valid)]
    return fit_side_archetype_monotonic_ev_mapping(
        calibration_frame,
        raw_oof[valid],
        target[valid],
        shrink_rows=float(config.calibration_shrink_rows),
        min_local_rows=int(config.calibration_min_local_rows),
        tail_weight_by_score_quantile=True,
    )


def _metric_scopes(
    frame: pd.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    config: ExecutionEVTrainerConfig,
    mode: str,
    arm: str,
) -> pd.DataFrame:
    ts = _utc(frame[config.decision_time_col], name=config.decision_time_col)
    work = pd.DataFrame(
        {
            "actual": actual,
            "prediction": predicted,
            "side": frame[config.side_col].astype(str).to_numpy(),
            "archetype": frame[config.catboost_archetype_col]
            .fillna("missing")
            .astype(str)
            .to_numpy(),
            "week": (
                ts.dt.floor("D") - pd.to_timedelta(ts.dt.weekday, unit="D")
            ).astype(str),
            "month": ts.dt.tz_localize(None).dt.to_period("M").astype(str),
        },
        index=frame.index,
    )
    rows: list[dict[str, Any]] = []
    scopes: tuple[tuple[str, list[str]], ...] = (
        ("overall", []),
        ("week", ["week"]),
        ("month", ["month"]),
        ("side", ["side"]),
        ("archetype", ["archetype"]),
        ("side_archetype", ["side", "archetype"]),
    )
    for scope, keys in scopes:
        groups: Iterable[tuple[Any, pd.DataFrame]]
        groups = (
            [((), work)] if not keys else work.groupby(keys, observed=True, sort=True)
        )
        for key, part in groups:
            values = key if isinstance(key, tuple) else (key,)
            row: dict[str, Any] = {"mode": mode, "arm": arm, "scope": scope}
            row.update(dict(zip(keys, values)))
            base = execution_ev_metrics(
                part["actual"],
                part["prediction"],
                top_k_fraction=0.10,
                huber_delta=config.huber_delta,
            )
            row.update(
                {
                    key: base[key]
                    for key in (
                        "rows",
                        "mae",
                        "huber",
                        "rmse",
                        "spearman",
                        "positive_ev_rate",
                        "positive_ev_auc",
                        "prediction_bias",
                    )
                }
            )
            row["ic"] = base["spearman"]
            for fraction, label in (
                (0.01, "1"),
                (0.05, "5"),
                (0.10, "10"),
                (0.20, "20"),
                (0.30, "30"),
            ):
                tail = execution_ev_metrics(
                    part["actual"],
                    part["prediction"],
                    top_k_fraction=fraction,
                    huber_delta=config.huber_delta,
                )
                row[f"top_{label}pct_rows"] = tail["top_k_rows"]
                row[f"top_{label}pct_mean_net_ev"] = tail["top_k_mean_net_ev"]
                row[f"top_{label}pct_sum_net_ev"] = tail["top_k_sum_net_ev"]
                row[f"top_{label}pct_positive_ev_rate"] = tail["top_k_positive_ev_rate"]
            rows.append(row)
    return pd.DataFrame(rows)


def _oof_fold_provenance(
    frame: pd.DataFrame,
    folds: Sequence[ChronologicalPurgedSplit],
    *,
    decision_time_col: str,
) -> pd.DataFrame:
    """Record the exact outer-fold provenance for every eligible OOF row."""

    decision = _utc(frame[decision_time_col], name=decision_time_col)
    output = pd.DataFrame(
        {
            "execution_ev_oof_fold": pd.Series(pd.NA, index=frame.index, dtype="Int64"),
            "execution_ev_oof_validation_start_utc": pd.Series(
                pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]"
            ),
            "execution_ev_oof_train_decision_cutoff_utc": pd.Series(
                pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]"
            ),
        }
    )
    for fold in folds:
        valid = np.asarray(fold.validation_indices, dtype=int)
        train = np.asarray(fold.train_indices, dtype=int)
        if not len(valid) or not len(train):
            continue
        output.iloc[valid, output.columns.get_loc("execution_ev_oof_fold")] = int(
            fold.fold
        )
        output.iloc[
            valid, output.columns.get_loc("execution_ev_oof_validation_start_utc")
        ] = fold.validation_start
        output.iloc[
            valid, output.columns.get_loc("execution_ev_oof_train_decision_cutoff_utc")
        ] = decision.iloc[train].max()
    return output


def _fit_arm_mode(
    frame: pd.DataFrame,
    features: Sequence[str],
    target_spec: ExecutionEVTargetSpec,
    folds: Sequence[ChronologicalPurgedSplit],
    *,
    config: ExecutionEVTrainerConfig,
    tune: bool,
    protected_features: Sequence[str] = (),
    selection_overrides: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Return outer-fold OOF net-EV predictions, final side models, and audit."""
    target = build_execution_ev_target(frame, target_spec).to_numpy(dtype=float)
    sides = _side_values(frame, config.side_col)
    x = frame.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    raw_oof = np.full(len(frame), np.nan, dtype=float)
    output = np.full(len(frame), np.nan, dtype=float)
    audit: dict[str, Any] = {
        "folds": [],
        "calibration": [],
        "feature_selection": {"outer": {}, "final": {}},
    }
    params_by_side: dict[str, list[dict[str, Any]]] = {"long": [], "short": []}

    for fold in folds:
        outer_selection = audit["feature_selection"]["outer"].setdefault(
            str(fold.fold), {}
        )
        pending_predictions: list[tuple[str, np.ndarray, np.ndarray, Any | None]] = []
        for side in ("long", "short"):
            train = np.asarray(
                [
                    i
                    for i in fold.train_indices
                    if sides[i] == side and np.isfinite(target[i])
                ],
                dtype=int,
            )
            valid = np.asarray(
                [
                    i
                    for i in fold.validation_indices
                    if sides[i] == side and np.isfinite(target[i])
                ],
                dtype=int,
            )
            if len(train) < 4 or not len(valid):
                audit["folds"].append(
                    {
                        "fold": fold.fold,
                        "side": side,
                        "status": "insufficient_side_rows",
                        "train_rows": int(len(train)),
                        "valid_rows": int(len(valid)),
                    }
                )
                continue
            outer_train = frame.iloc[train].reset_index(drop=True)
            inner = _inner_splits(outer_train, config)
            local_x = x.iloc[train].reset_index(drop=True)
            local_y = target[train]
            source_selection = None
            if selection_overrides is not None:
                source_selection = (
                    selection_overrides.get("outer", {})
                    .get(str(fold.fold), {})
                    .get(side)
                )
            selection = (
                _derived_ablation_selection(
                    source_selection,
                    local_x.columns,
                    protected_features=protected_features,
                    side=side,
                )
                if source_selection is not None
                else _select_side_train_only_features(
                    local_x,
                    local_y,
                    inner,
                    side=side,
                    protected_features=protected_features,
                    config=config,
                )
            )
            outer_selection[side] = selection
            selected_features = list(selection["selected_features"])
            selected_x = local_x.loc[:, selected_features]
            params, hpo = (
                _tune_lgbm(selected_x, local_y, inner, config)
                if tune
                else (
                    _lgbm_params(config),
                    {"status": "reused_default_for_ablation", "trials": 0},
                )
            )
            params_by_side[side].append(params)
            # Generate inner OOF predictions solely from outer-train rows for
            # this side's fold calibration map.  Validation targets never
            # enter it, and long/short never share a calibration sample.
            inner_oof = np.full(len(train), np.nan, dtype=float)
            for inner_fold in inner:
                model, _ = _fit_lgbm(
                    selected_x,
                    local_y,
                    fit_indices=inner_fold.train_indices,
                    early_stop=None,
                    params=params,
                    config=config,
                )
                inner_oof[inner_fold.validation_indices] = model.predict(
                    selected_x.iloc[inner_fold.validation_indices]
                )
            model, best_iteration = _fit_lgbm(
                selected_x,
                local_y,
                fit_indices=np.arange(len(train), dtype=int),
                early_stop=inner[-1] if inner else None,
                params=params,
                config=config,
            )
            if inner:
                # Early stopping above estimates model capacity without seeing
                # the outer validation fold. Refit that capacity on every
                # authorized outer-training row before producing outer OOF.
                refit_params = dict(params)
                refit_params["n_estimators"] = max(1, int(best_iteration))
                model, _ = _fit_lgbm(
                    selected_x,
                    local_y,
                    fit_indices=np.arange(len(train), dtype=int),
                    early_stop=None,
                    params=refit_params,
                    config=config,
                )
            raw = model.predict(x.iloc[valid].loc[:, selected_features])
            raw_oof[valid] = raw
            calibrator = _train_only_calibrator(
                outer_train,
                inner_oof,
                local_y,
                side_col=config.side_col,
                archetype_col=config.catboost_archetype_col,
                config=config,
            )
            pending_predictions.append((side, valid, raw, calibrator))
            audit["folds"].append(
                {
                    "fold": fold.fold,
                    "side": side,
                    "status": "ok",
                    "train_rows": int(len(train)),
                    "valid_rows": int(len(valid)),
                    "best_iteration": best_iteration,
                    "hpo": hpo,
                    "selected_features": selected_features,
                    "selected_features_sha256": selection["selected_features_sha256"],
                }
            )
            audit["calibration"].append(
                {
                    "fold": fold.fold,
                    "side": side,
                    "scope": "side_local",
                    "status": "fit"
                    if calibrator is not None
                    else "skipped_insufficient_inner_oof",
                    "train_only_oof_rows": int(np.isfinite(inner_oof).sum()),
                }
            )
        for side, valid, raw, calibrator in pending_predictions:
            if calibrator is not None:
                mapping_frame = pd.DataFrame(
                    {
                        "side_name": frame.iloc[valid][config.side_col]
                        .astype(str)
                        .to_numpy(),
                        "archetype_policy_key": frame.iloc[valid][
                            config.catboost_archetype_col
                        ]
                        .fillna("missing")
                        .astype(str)
                        .to_numpy(),
                    }
                )
                predicted_target = predict_side_archetype_monotonic_ev_mapping(
                    calibrator, mapping_frame, raw
                )
            else:
                predicted_target = np.asarray(raw, dtype=float)
            alpha = _numeric(
                frame.iloc[valid], target_spec.alpha_ev_col, role="existing alpha EV"
            )
            output[valid] = (
                predicted_target
                if target_spec.mode == "direct"
                else predicted_target + alpha
            )

    final_models: dict[str, Any] = {}
    final_calibration: dict[str, Any] = {}
    for side in ("long", "short"):
        pos = np.flatnonzero((sides == side) & np.isfinite(target))
        if len(pos) < 4:
            continue
        final_frame = frame.iloc[pos].reset_index(drop=True)
        final_x = x.iloc[pos].reset_index(drop=True)
        final_y = target[pos]
        final_inner = _inner_splits(final_frame, config)
        source_selection = None
        if selection_overrides is not None:
            source_selection = selection_overrides.get("final", {}).get(side)
        selection = (
            _derived_ablation_selection(
                source_selection,
                final_x.columns,
                protected_features=protected_features,
                side=side,
            )
            if source_selection is not None
            else _select_side_train_only_features(
                final_x,
                final_y,
                final_inner,
                side=side,
                protected_features=protected_features,
                config=config,
            )
        )
        audit["feature_selection"]["final"][side] = selection
        selected_features = list(selection["selected_features"])
        params = (
            params_by_side[side][-1] if params_by_side[side] else _lgbm_params(config)
        )
        model, best_iteration = _fit_lgbm(
            final_x.loc[:, selected_features],
            final_y,
            fit_indices=np.arange(len(pos), dtype=int),
            early_stop=None,
            params=params,
            config=config,
        )
        final_models[side] = {
            "model": model,
            "features": selected_features,
            "params": params,
            "best_iteration": best_iteration,
            "feature_selection": selection,
        }
        # The final live map is fit only on this side's outer-OOF raw scores;
        # it does not pool long/short outcomes and is never used to revise the
        # OOF predictions above.
        final_calibration[side] = _train_only_calibrator(
            final_frame,
            raw_oof[pos],
            final_y,
            side_col=config.side_col,
            archetype_col=config.catboost_archetype_col,
            config=config,
        )
        audit["calibration"].append(
            {
                "phase": "final_refit",
                "side": side,
                "scope": "side_local_outer_oof",
                "status": "fit"
                if final_calibration[side] is not None
                else "skipped_insufficient_outer_oof",
                "train_only_oof_rows": int(np.isfinite(raw_oof[pos]).sum()),
            }
        )
    audit["raw_oof_rows"] = int(np.isfinite(raw_oof).sum())
    return output, final_models, {**audit, "final_calibration": final_calibration}


def train_execution_ev_meta(
    frame: pd.DataFrame,
    provenance: Mapping[str, FeatureProvenance],
    *,
    config: ExecutionEVTrainerConfig = ExecutionEVTrainerConfig(),
) -> ExecutionEVModelBundle:
    """Train direct and residual side-aware LGBM heads on strict OOF inputs.

    The returned predictions are outer-fold OOF only.  For every outer fold,
    HPO, early stopping, and the monotonic side x CatBoost-archetype EV map use
    only that fold's chronological training rows.  The bundle's final models
    are fitted after evaluation and must not be presented as untouched OOS.
    """
    _side_values(frame, config.side_col)
    _catboost_archetypes(frame, config.catboost_archetype_col)
    validate_execution_ev_training_contract(
        frame,
        provenance,
        decision_time_col=config.decision_time_col,
        predicted_path_archetype_col=config.catboost_archetype_col,
    )
    plan = execution_ev_ablation_plan(provenance)
    for arm_features in plan.values():
        validate_execution_ev_feature_provenance(
            frame, arm_features, provenance, decision_time_col=config.decision_time_col
        )
    if not _finite_target_rows(frame, ExecutionEVTargetSpec()).any():
        raise ValueError("Execution-EV trainer has no finite direct target rows")
    folds = chronological_purged_splits(
        frame,
        n_splits=config.n_splits,
        min_train_size=config.min_train_rows,
        min_train_group_col=config.side_col,
        required_train_groups=("long", "short"),
        decision_time_col=config.decision_time_col,
        label_end_time_col=config.label_end_time_col,
        horizon_hours=config.purge_hours,
        embargo_hours=config.embargo_hours,
    )
    oof_provenance = _oof_fold_provenance(
        frame, folds, decision_time_col=config.decision_time_col
    )
    arms = plan if config.run_ablations else {"all_features": plan["all_features"]}
    prediction_table = pd.DataFrame(index=frame.index)
    models: dict[str, dict[str, Any]] = {}
    calibration: dict[str, dict[str, Any]] = {}
    audits: dict[str, Any] = {}
    diagnostic_parts: list[pd.DataFrame] = []
    net_ev = _numeric(frame, ExecutionEVTargetSpec().net_ev_col, role="net EV target")
    alpha_ev = _numeric(
        frame, ExecutionEVTargetSpec().alpha_ev_col, role="existing alpha EV"
    )
    prediction_table["baseline__existing_alpha"] = alpha_ev
    diagnostic_parts.append(
        _metric_scopes(
            frame,
            net_ev,
            alpha_ev,
            config=config,
            mode="baseline",
            arm="existing_alpha",
        )
    )
    protected_features = plan["alpha_only"]
    for mode in ("direct", "residual"):
        spec = ExecutionEVTargetSpec(mode=mode)
        # The all-feature arm is the sole feature-selection search for this
        # target mode.  It independently freezes a long/short contract inside
        # each outer fold and again for final refit *before* that side's HPO.
        # The named ablations below inherit the matching frozen contract and
        # remove only their declared families, avoiding a new selection search
        # that would make their attribution unfair and needlessly expensive.
        prediction, fitted, audit = _fit_arm_mode(
            frame,
            arms["all_features"],
            spec,
            folds,
            config=config,
            tune=True,
            protected_features=protected_features,
        )
        key = f"{mode}__all_features"
        prediction_table[key] = prediction
        models[key] = fitted
        calibration[key] = audit.pop("final_calibration")
        audits[key] = audit
        diagnostic_parts.append(
            _metric_scopes(
                frame,
                net_ev,
                prediction,
                config=config,
                mode=mode,
                arm="all_features",
            )
        )
        frozen_selection = audit["feature_selection"]
        for arm, features in arms.items():
            if arm == "all_features":
                continue
            prediction, fitted, audit = _fit_arm_mode(
                frame,
                features,
                spec,
                folds,
                config=config,
                tune=False,
                protected_features=protected_features,
                selection_overrides=frozen_selection,
            )
            key = f"{mode}__{arm}"
            prediction_table[key] = prediction
            models[key] = fitted
            calibration[key] = audit.pop("final_calibration")
            audits[key] = audit
            diagnostic_parts.append(
                _metric_scopes(
                    frame,
                    net_ev,
                    prediction,
                    config=config,
                    mode=mode,
                    arm=arm,
                )
            )
    diagnostics = (
        pd.concat(diagnostic_parts, ignore_index=True)
        if diagnostic_parts
        else pd.DataFrame()
    )
    ablations = {
        mode: execution_ev_ablation_metrics(
            net_ev,
            {
                arm: prediction_table[f"{mode}__{arm}"].to_numpy(dtype=float)
                for arm in arms
            },
            top_k_fraction=0.10,
            huber_delta=config.huber_delta,
        ).to_dict(orient="records")
        for mode in ("direct", "residual")
    }
    timing_vs_slope = {
        mode: timing_slope_ablation_comparison(ablations[mode])
        for mode in ("direct", "residual")
    }
    comparison = compare_direct_and_residual(
        pd.DataFrame(
            {
                ExecutionEVTargetSpec().net_ev_col: net_ev,
                ExecutionEVTargetSpec().alpha_ev_col: alpha_ev,
                "direct": prediction_table.get(
                    "direct__all_features", pd.Series(np.nan, index=frame.index)
                ),
                "residual": prediction_table.get(
                    "residual__all_features", pd.Series(np.nan, index=frame.index)
                )
                - alpha_ev,
            }
        ),
        direct_prediction_col="direct",
        residual_prediction_col="residual",
        top_k_fraction=0.10,
        huber_delta=config.huber_delta,
    )
    report = {
        "schema": EXECUTION_EV_BUNDLE_SCHEMA,
        "provenance_contract": "all model inputs declared pre_entry and oof_or_frozen; availability checked at decision timestamp",
        "oof_contract": "outer expanding purged folds; per-side feature selection, HPO, early-stop, and calibration are training-only",
        "feature_selection_contract": "per-side inner-OOF permutation MDA on outer-train rows, frozen before that side's HPO; named ablations intersect the matching frozen all-feature contract",
        "calibration_contract": "separate long/short monotonic maps fit only on same-side inner OOF rows per outer fold and same-side outer OOF rows for final refit; no pooled map",
        "folds": [
            {
                "fold": split.fold,
                "validation_start": split.validation_start.isoformat(),
                "validation_end": split.validation_end.isoformat(),
                "purge_hours": split.purge_hours,
                "embargo_hours": split.embargo_hours,
            }
            for split in folds
        ],
        "direct_vs_residual": comparison,
        "existing_alpha_baseline": execution_ev_metrics(
            net_ev,
            alpha_ev,
            top_k_fraction=0.10,
            huber_delta=config.huber_delta,
        ),
        "ablations": ablations,
        "timing_vs_slope": timing_vs_slope,
        "diagnostics": diagnostics,
        "audits": audits,
    }
    return ExecutionEVModelBundle(
        EXECUTION_EV_BUNDLE_SCHEMA,
        asdict(config),
        dict(provenance),
        plan,
        models,
        calibration,
        report,
        prediction_table,
        oof_provenance,
    )


def predict_execution_ev_bundle(
    bundle: ExecutionEVModelBundle, frame: pd.DataFrame, *, arm: str = "all_features"
) -> pd.DataFrame:
    """Score direct/residual final heads and their frozen side calibration maps."""
    config = ExecutionEVTrainerConfig(**bundle.config)
    _side_values(frame, config.side_col)
    _catboost_archetypes(frame, config.catboost_archetype_col)
    result = pd.DataFrame(index=frame.index)
    for mode in ("direct", "residual"):
        key = f"{mode}__{arm}"
        if key not in bundle.models:
            continue
        output = np.full(len(frame), np.nan, dtype=float)
        for side, state in bundle.models[key].items():
            pos = np.flatnonzero(
                frame[config.side_col].astype(str).str.lower().to_numpy() == side
            )
            if not len(pos):
                continue
            features = state["features"]
            validate_execution_ev_feature_provenance(
                frame,
                features,
                bundle.provenance,
                decision_time_col=config.decision_time_col,
            )
            raw = state["model"].predict(
                frame.iloc[pos].loc[:, features].apply(pd.to_numeric, errors="coerce")
            )
            calibrator = bundle.calibration.get(key, {}).get(side)
            if calibrator is not None:
                mapper = pd.DataFrame(
                    {
                        "side_name": frame.iloc[pos][config.side_col]
                        .astype(str)
                        .to_numpy(),
                        "archetype_policy_key": frame.iloc[pos][
                            config.catboost_archetype_col
                        ]
                        .fillna("missing")
                        .astype(str)
                        .to_numpy(),
                    }
                )
                raw = predict_side_archetype_monotonic_ev_mapping(
                    calibrator, mapper, raw
                )
            if mode == "residual":
                raw = raw + _numeric(
                    frame.iloc[pos],
                    ExecutionEVTargetSpec().alpha_ev_col,
                    role="existing alpha EV",
                )
            output[pos] = raw
        result[f"execution_ev_{mode}"] = output
    return result


def save_execution_ev_bundle(bundle: ExecutionEVModelBundle, path: str | Path) -> Path:
    """Persist models plus the OOF ledger with joblib; caller owns retention."""
    import joblib

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, target, compress=3)
    return target


def load_execution_ev_bundle(path: str | Path) -> ExecutionEVModelBundle:
    import joblib

    bundle = joblib.load(path)
    if (
        not isinstance(bundle, ExecutionEVModelBundle)
        or bundle.schema != EXECUTION_EV_BUNDLE_SCHEMA
    ):
        raise ValueError("not an execution-EV side-aware LGBM bundle")
    return bundle


def write_execution_ev_report(
    bundle: ExecutionEVModelBundle, output_dir: str | Path
) -> dict[str, Path]:
    """Write a JSON audit summary and flat diagnostics/OOF tables."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    diagnostics = bundle.report["diagnostics"]
    diagnostic_path = root / "execution_ev_diagnostics.csv"
    oof_path = root / "execution_ev_oof_predictions.parquet"
    report_path = root / "execution_ev_report.json"
    diagnostics.to_csv(diagnostic_path, index=False)
    oof = bundle.oof_predictions.join(bundle.oof_provenance, how="left")
    try:
        oof.to_parquet(oof_path, index=True)
    except (ImportError, ValueError):
        oof_path = root / "execution_ev_oof_predictions.pkl"
        oof.to_pickle(oof_path)
    payload = {
        key: value for key, value in bundle.report.items() if key != "diagnostics"
    }
    payload["diagnostics_path"] = diagnostic_path.name
    payload["oof_predictions_path"] = oof_path.name
    report_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return {
        "report": report_path,
        "diagnostics": diagnostic_path,
        "oof_predictions": oof_path,
    }
