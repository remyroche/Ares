"""Leakage-safe downstream execution-EV model and feature ablations.

This module is deliberately independent from :mod:`execution_ev_meta`.  It
uses the same provenance, target, chronological-purge, and metric contracts,
but compares three regressors and a nested permutation-MDA feature-selection
arm.  It is evaluation tooling only; it neither selects a trading policy nor
changes live execution behaviour.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .execution_ev_meta import (
    ChronologicalPurgedSplit,
    ExecutionEVTargetSpec,
    FeatureProvenance,
    TargetMode,
    build_execution_ev_target,
    chronological_purged_splits,
    execution_ev_metrics,
    validate_execution_ev_feature_provenance,
)
from .path_archetype_labels import PATH_SHAPE_TYPES

EXECUTION_EV_MODEL_ABLATION_SCHEMA = "execution_ev_model_ablation_v1"
ALGORITHM_NAMES: tuple[str, ...] = ("lgbm", "catboost", "extra_trees")
FIVE_AUXILIARY_FAMILIES: tuple[str, ...] = (
    "time_to_mfe",
    "peak_mfe",
    "mae_before_meaningful_mfe",
    "adverse_turn_timing",
    "favorable_path_slope",
)
REQUIRED_INPUT_FAMILIES: tuple[str, ...] = (
    "alpha_score",
    *FIVE_AUXILIARY_FAMILIES,
    "catboost_probabilities",
    "catboost_entropy",
    "prediction_uncertainty",
    "leaf_support",
    "base_archetype_labels",
)
CATBOOST_ARGMAX_CONTEXT_FAMILY = "predicted_path_archetype"
BASE_ARCHETYPE_FEATURE_PREFIX = "base_archetype_label__"
DOWNSTREAM_FORBIDDEN_OUTCOME_TOKENS: tuple[str, ...] = (
    "actual_",
    "realized_",
    "execution_net_ev",
    "execution_gross_ev",
    "execution_cost",
    "future_",
    "label",
    "target",
    "outcome",
)
FeatureArm = Literal["all_features", "mda_1se"]
HPOArm = Literal["without_hpo", "with_hpo"]


@dataclass(frozen=True)
class ExecutionEVModelAblationConfig:
    """Configuration for the strict, side-local downstream model ablation."""

    n_splits: int = 3
    min_train_rows: int = 500
    purge_hours: float = 12.0
    embargo_hours: float = 12.0
    inner_n_splits: int = 2
    min_fit_rows: int = 32
    hpo_trials: int = 12
    n_estimators: int = 400
    early_stopping_rounds: int = 50
    random_state: int = 42
    n_jobs: int = 3
    side_col: str = "side_name"
    decision_time_col: str = "__ts__"
    label_end_time_col: str | None = "execution_label_end_utc"
    catboost_archetype_col: str = "catboost_archetype"
    target_spec: ExecutionEVTargetSpec = field(default_factory=ExecutionEVTargetSpec)
    target_modes: tuple[TargetMode, ...] = ("direct", "residual")
    gross_ev_col: str = "execution_gross_ev_12h"
    top_k_fraction: float = 0.10
    mda_min_features: int = 8
    mda_max_steps: int = 24
    mda_repeats: int = 1
    isotonic_min_rows: int = 24
    recent_ev_correction_enabled: bool = True
    recent_ev_correction_routes: tuple[str, ...] = (
        "catboost_predicted_archetype",
        "gmm_archetype",
    )
    gmm_archetype_col: str = "gmm_cluster_id"
    recent_ev_window_days: int = 21
    recent_ev_trim_fraction: float = 0.10
    recent_ev_side_support_target: float = 320.0
    recent_ev_local_support_target: float = 160.0
    recent_ev_correction_cap: float = 0.03
    algorithms: tuple[str, ...] = ALGORITHM_NAMES
    additional_input_families: tuple[str, ...] = ()
    feature_arms: tuple[FeatureArm, ...] = ("all_features", "mda_1se")
    require_latest_period_stability: bool = True
    min_latest_period_selected_rows: int = 100
    min_latest_period_selection_share: float = 0.01
    min_latest_period_top_k_net_ev: float = 0.0


@dataclass
class IsotonicEVMapping:
    """Train-only isotonic post-map, with an explicit identity fallback."""

    model: IsotonicRegression | None
    status: str
    train_rows: int

    def predict(self, raw_prediction: Sequence[float] | np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_prediction, dtype=np.float64)
        if self.model is None:
            return raw
        return np.asarray(self.model.predict(raw), dtype=np.float64)


@dataclass
class ExecutionEVModelAblationBundle:
    """Persistable per-side final fits and frozen inference feature contract."""

    schema: str
    config: dict[str, Any]
    provenance: dict[str, FeatureProvenance]
    raw_feature_columns: tuple[str, ...]
    expanded_feature_columns: tuple[str, ...]
    archetype_levels: tuple[str, ...]
    final_feature_sets: dict[str, dict[str, dict[str, dict[str, dict[str, list[str]]]]]]
    models: dict[str, dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]]]
    report: dict[str, Any]
    oof_predictions: pd.DataFrame = field(repr=False)
    oof_provenance: pd.DataFrame = field(repr=False)


def _utc(values: pd.Series | Sequence[Any], *, name: str) -> pd.Series:
    converted = pd.to_datetime(values, utc=True, errors="coerce")
    if pd.isna(converted).any():
        raise ValueError(f"{name} contains invalid timestamps")
    return pd.Series(converted, index=getattr(values, "index", None))


def _side_values(frame: pd.DataFrame, side_col: str) -> np.ndarray:
    if side_col not in frame.columns:
        raise ValueError(f"Execution-EV ablation requires side column {side_col!r}")
    sides = frame[side_col].astype(str).str.lower().to_numpy()
    unknown = sorted(set(sides) - {"long", "short"})
    if unknown:
        raise ValueError(
            "side values must be long/short; got " + ", ".join(unknown[:10])
        )
    return sides


def _finite_numeric(frame: pd.DataFrame, column: str, *, role: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"Execution-EV ablation is missing {role} column {column!r}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Execution-EV ablation {role} {column!r} must be finite")
    return values


def _family_columns(
    provenance: Mapping[str, FeatureProvenance],
    family: str,
) -> list[str]:
    return [name for name, spec in provenance.items() if spec.family == family]


def _is_forbidden_downstream_input(name: str, spec: FeatureProvenance) -> bool:
    """Reject raw outcomes while permitting proven pre-entry OOF predictions.

    ``future_slope`` is a valid auxiliary prediction when its declared
    provenance proves it is OOF/frozen and available at entry.  The same name
    without that evidence remains a forbidden raw future-derived input.
    """

    lowered = name.lower()
    if spec.family == "base_archetype_labels" and name.startswith(
        BASE_ARCHETYPE_FEATURE_PREFIX
    ):
        return False
    if lowered.startswith(("pred_", "oof_", "frozen_", "score_")):
        return False
    for token in DOWNSTREAM_FORBIDDEN_OUTCOME_TOKENS:
        if token not in lowered:
            continue
        if token == "future_" and spec.pre_entry and spec.oof_or_frozen:
            continue
        return True
    return False


def _validate_downstream_feature_provenance(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    provenance: Mapping[str, FeatureProvenance],
    *,
    decision_time_col: str,
) -> None:
    """Apply the shared guard plus the narrow proven-``future_`` exception."""

    declared_future_predictions = [
        name
        for name in feature_names
        if "future_" in name.lower()
        and not _is_forbidden_downstream_input(name, provenance[name])
        and not name.lower().startswith(("pred_", "oof_", "frozen_", "score_"))
    ]
    standard = [
        name for name in feature_names if name not in declared_future_predictions
    ]
    if standard:
        validate_execution_ev_feature_provenance(
            frame,
            standard,
            provenance,
            decision_time_col=decision_time_col,
        )
    if not declared_future_predictions:
        return
    decision = _utc(frame[decision_time_col], name=decision_time_col)
    for name in declared_future_predictions:
        spec = provenance[name]
        if not spec.pre_entry or not spec.oof_or_frozen or not spec.model_input:
            raise ValueError(
                f"Execution-EV input {name!r} must be a pre-entry OOF/frozen model input"
            )
        if spec.available_at_col is None:
            continue
        available = _utc(frame[spec.available_at_col], name=spec.available_at_col)
        if (available > decision).any():
            raise ValueError(
                f"Execution-EV input {name!r} has availability after entry"
            )


def validate_execution_ev_model_ablation_contract(
    frame: pd.DataFrame,
    provenance: Mapping[str, FeatureProvenance],
    *,
    decision_time_col: str = "__ts__",
    side_col: str = "side_name",
    catboost_archetype_col: str = "catboost_archetype",
    additional_input_families: Sequence[str] = (),
) -> tuple[list[str], tuple[str, ...]]:
    """Validate the exact frozen alpha, five-auxiliary, CatBoost handoff.

    The archetype assignment is expanded into fixed one-hot inputs by this
    module.  It remains declared as a non-numeric pre-entry context in the
    upstream provenance so it cannot silently be confused with a realized path
    label.
    """

    _side_values(frame, side_col)
    if decision_time_col not in frame.columns:
        raise ValueError(f"Execution-EV ablation requires {decision_time_col!r}")
    if catboost_archetype_col not in frame.columns:
        raise ValueError(
            "Execution-EV ablation requires the pre-entry CatBoost archetype "
            f"column {catboost_archetype_col!r}"
        )

    extra_families = tuple(dict.fromkeys(map(str, additional_input_families)))
    overlap = sorted(set(extra_families) & set(REQUIRED_INPUT_FAMILIES))
    if overlap:
        raise ValueError(
            "additional input families duplicate required families: "
            + ", ".join(overlap)
        )
    input_families = (*REQUIRED_INPUT_FAMILIES, *extra_families)
    by_family = {
        family: _family_columns(provenance, family)
        for family in input_families
    }
    missing = [family for family, names in by_family.items() if not names]
    if missing:
        raise ValueError(
            "Execution-EV ablation missing required frozen feature families: "
            + ", ".join(missing)
        )
    probability_columns = by_family["catboost_probabilities"]
    declared_orders = {
        tuple(map(str, provenance[name].class_order))
        for name in [*probability_columns, catboost_archetype_col]
        if provenance[name].class_order is not None
    }
    if len(declared_orders) > 1:
        raise ValueError("CatBoost probability/archetype class orders disagree")
    archetype_levels = (
        next(iter(declared_orders))
        if declared_orders
        else tuple(map(str, PATH_SHAPE_TYPES))
    )
    if len(probability_columns) != len(archetype_levels):
        raise ValueError(
            "Execution-EV ablation requires the complete ordered CatBoost "
            f"probability vector ({len(archetype_levels)} columns)"
        )
    raw_feature_columns = [
        name
        for family in input_families
        for name in by_family[family]
        if provenance[name].model_input
    ]
    disabled_probability_features = [
        name for name in probability_columns if not provenance[name].model_input
    ]
    if disabled_probability_features:
        raise ValueError(
            "CatBoost probability-vector columns cannot be disabled model inputs: "
            + ", ".join(disabled_probability_features)
        )
    empty_sources = [
        name for name in raw_feature_columns if not str(provenance[name].source).strip()
    ]
    if empty_sources:
        raise ValueError(
            "Execution-EV ablation provenance requires non-empty sources: "
            + ", ".join(empty_sources)
        )
    invalid_base_archetype_columns = [
        name
        for name in by_family["base_archetype_labels"]
        if not name.startswith(BASE_ARCHETYPE_FEATURE_PREFIX)
    ]
    if invalid_base_archetype_columns:
        raise ValueError(
            "Execution-EV ablation base_archetype_labels must use the frozen "
            f"{BASE_ARCHETYPE_FEATURE_PREFIX!r} prefix: "
            + ", ".join(invalid_base_archetype_columns)
        )
    outcome_like = [
        name
        for name in raw_feature_columns
        if _is_forbidden_downstream_input(name, provenance[name])
    ]
    if outcome_like:
        raise ValueError(
            "Execution-EV ablation inputs appear outcome-derived: "
            + ", ".join(outcome_like)
        )
    _validate_downstream_feature_provenance(
        frame,
        raw_feature_columns,
        provenance,
        decision_time_col=decision_time_col,
    )
    archetype_spec = provenance.get(catboost_archetype_col)
    if (
        archetype_spec is None
        or archetype_spec.family != CATBOOST_ARGMAX_CONTEXT_FAMILY
        or archetype_spec.model_input
        or not archetype_spec.pre_entry
        or not archetype_spec.oof_or_frozen
        or not str(archetype_spec.source).strip()
    ):
        raise ValueError(
            "Execution-EV ablation requires a non-model-input pre-entry frozen "
            f"CatBoost argmax context provenance declaration for {catboost_archetype_col!r}"
        )
    validate_execution_ev_feature_provenance(
        frame,
        [catboost_archetype_col],
        provenance,
        decision_time_col=decision_time_col,
        require_model_input=False,
    )

    probabilities = (
        frame.loc[:, probability_columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )
    if not np.isfinite(probabilities).all():
        raise ValueError("CatBoost probability inputs must be finite")
    if (probabilities < -1e-6).any() or (probabilities > 1.0 + 1e-6).any():
        raise ValueError("CatBoost probability inputs must lie in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4, rtol=1e-4):
        raise ValueError("CatBoost probability inputs must sum to one per row")
    entropy = (
        frame.loc[:, by_family["catboost_entropy"]]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )
    expected_entropy = -np.sum(
        np.clip(probabilities, 1e-12, 1.0) * np.log(np.clip(probabilities, 1e-12, 1.0)),
        axis=1,
    )
    if not np.isfinite(entropy).all() or not np.allclose(
        entropy[:, 0], expected_entropy, atol=1e-4, rtol=1e-4
    ):
        raise ValueError("CatBoost entropy input does not match the probability vector")
    expected_archetype = np.asarray(archetype_levels, dtype=object)[
        np.argmax(probabilities, axis=1)
    ]
    observed_archetype = frame[catboost_archetype_col].astype(str).to_numpy()
    if not np.array_equal(observed_archetype, expected_archetype):
        raise ValueError(
            "pre-entry CatBoost archetype must equal the argmax of the declared "
            "ordered probability vector"
        )
    return list(dict.fromkeys(raw_feature_columns)), archetype_levels


def _materialize_feature_matrix(
    frame: pd.DataFrame,
    raw_feature_columns: Sequence[str],
    *,
    catboost_archetype_col: str,
    archetype_levels: Sequence[str],
) -> pd.DataFrame:
    """Return a float32 matrix with a fixed pre-entry archetype encoding."""

    numeric = frame.loc[:, list(raw_feature_columns)].apply(
        pd.to_numeric, errors="coerce"
    )
    values = numeric.to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Execution-EV ablation model inputs must be finite")
    output = pd.DataFrame(values, columns=list(raw_feature_columns), index=frame.index)
    archetype = frame[catboost_archetype_col].astype(str)
    for level in archetype_levels:
        output[f"catboost_archetype__{level}"] = (archetype == level).astype("float32")
    return output.astype("float32", copy=False)


def _fixed_params(
    algorithm: str,
    config: ExecutionEVModelAblationConfig,
) -> dict[str, Any]:
    """Return the declared fixed-parameter reference for one regressor."""

    if algorithm == "lgbm":
        return {
            "objective": "huber",
            "n_estimators": int(config.n_estimators),
            "learning_rate": 0.03,
            "max_depth": -1,
            "num_leaves": 24,
            "min_child_samples": 32,
            "min_split_gain": 1e-3,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "max_bin": 127,
            "reg_alpha": 0.1,
            "reg_lambda": 5.0,
            "random_state": int(config.random_state),
            "n_jobs": int(config.n_jobs),
            "verbosity": -1,
        }
    if algorithm == "catboost":
        return {
            "loss_function": "MAE",
            "iterations": int(config.n_estimators),
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 6.0,
            "random_seed": int(config.random_state),
            "thread_count": int(config.n_jobs),
            "verbose": False,
            "allow_writing_files": False,
            "random_strength": 0.5,
            "bagging_temperature": 1.0,
            "bootstrap_type": "Bayesian",
        }
    if algorithm == "extra_trees":
        return {
            "n_estimators": int(config.n_estimators),
            "max_features": 0.8,
            "min_samples_leaf": 4,
            "max_depth": 10,
            "min_samples_split": 8,
            "bootstrap": False,
            "random_state": int(config.random_state),
            "n_jobs": int(config.n_jobs),
        }
    raise ValueError(f"Unsupported execution-EV ablation algorithm {algorithm!r}")


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _randomized_hpo_params(
    algorithm: str,
    config: ExecutionEVModelAblationConfig,
    *,
    trial_number: int,
) -> dict[str, Any]:
    """Sample one deterministic, model-appropriate regularized recipe.

    This intentionally does not use a small recycled recipe list.  The seed is
    stable across processes and the bounded spaces avoid unregularized trees.
    """

    algorithm_seed = ALGORITHM_NAMES.index(algorithm) * 10_007
    rng = np.random.default_rng(
        int(config.random_state) + algorithm_seed + int(trial_number)
    )
    if algorithm == "lgbm":
        return {
            "objective": "huber",
            "n_estimators": int(config.n_estimators),
            "learning_rate": _log_uniform(rng, 0.01, 0.08),
            "max_depth": int(rng.integers(3, 8)),
            "num_leaves": int(rng.choice([8, 12, 16, 24, 32, 48])),
            "min_child_samples": int(np.exp(rng.uniform(np.log(16), np.log(128)))),
            "min_split_gain": _log_uniform(rng, 1e-4, 0.05),
            "reg_alpha": _log_uniform(rng, 1e-4, 5.0),
            "reg_lambda": _log_uniform(rng, 0.5, 30.0),
            "subsample": float(rng.uniform(0.60, 1.0)),
            "subsample_freq": 1,
            "colsample_bytree": float(rng.uniform(0.55, 1.0)),
            "max_bin": int(rng.choice([63, 127, 255])),
            "random_state": int(config.random_state),
            "n_jobs": int(config.n_jobs),
            "verbosity": -1,
        }
    if algorithm == "catboost":
        return {
            "loss_function": "MAE",
            "iterations": int(config.n_estimators),
            "learning_rate": _log_uniform(rng, 0.01, 0.08),
            "depth": int(rng.integers(4, 9)),
            "l2_leaf_reg": _log_uniform(rng, 2.0, 40.0),
            "random_strength": _log_uniform(rng, 0.05, 3.0),
            "bagging_temperature": float(rng.uniform(0.2, 3.0)),
            "bootstrap_type": "Bayesian",
            "random_seed": int(config.random_state),
            "thread_count": int(config.n_jobs),
            "verbose": False,
            "allow_writing_files": False,
        }
    if algorithm == "extra_trees":
        return {
            "n_estimators": int(config.n_estimators),
            "max_features": float(rng.uniform(0.35, 1.0)),
            "min_samples_leaf": int(rng.integers(2, 13)),
            "min_samples_split": int(rng.integers(4, 25)),
            "max_depth": int(rng.integers(5, 17)),
            "max_leaf_nodes": int(rng.choice([32, 64, 128, 256])),
            "bootstrap": False,
            "random_state": int(config.random_state),
            "n_jobs": int(config.n_jobs),
        }
    raise ValueError(f"Unsupported execution-EV ablation algorithm {algorithm!r}")


def _fit_regressor(
    algorithm: str,
    x: pd.DataFrame,
    y: np.ndarray,
    *,
    params: Mapping[str, Any],
    early_stop_x: pd.DataFrame | None = None,
    early_stop_y: np.ndarray | None = None,
    early_stopping_rounds: int | None = None,
) -> Any:
    if algorithm == "lgbm":
        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover - dependency boundary
            raise RuntimeError(
                "LightGBM is required for the lgbm ablation arm"
            ) from exc
        model = lgb.LGBMRegressor(**dict(params))
    elif algorithm == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:  # pragma: no cover - dependency boundary
            raise RuntimeError(
                "CatBoost is required for the catboost ablation arm"
            ) from exc
        model = CatBoostRegressor(**dict(params))
    elif algorithm == "extra_trees":
        try:
            from sklearn.ensemble import ExtraTreesRegressor
        except ImportError as exc:  # pragma: no cover - dependency boundary
            raise RuntimeError(
                "scikit-learn is required for the ExtraTrees ablation arm"
            ) from exc
        model = ExtraTreesRegressor(**dict(params))
    else:
        raise ValueError(f"Unsupported execution-EV ablation algorithm {algorithm!r}")
    use_early_stopping = (
        early_stop_x is not None
        and early_stop_y is not None
        and len(early_stop_x) > 0
        and int(early_stopping_rounds or 0) > 0
    )
    if algorithm == "lgbm" and use_early_stopping:
        model.fit(
            x,
            y,
            eval_set=[(early_stop_x, early_stop_y)],
            callbacks=[lgb.early_stopping(int(early_stopping_rounds), verbose=False)],
        )
    elif algorithm == "catboost" and use_early_stopping:
        model.fit(
            x,
            y,
            eval_set=(early_stop_x, early_stop_y),
            early_stopping_rounds=int(early_stopping_rounds),
            verbose=False,
        )
    else:
        model.fit(x, y)
    return model


def _ranking_metrics(
    net_ev: np.ndarray,
    gross_ev: np.ndarray,
    prediction: np.ndarray,
    *,
    top_k_fraction: float,
) -> dict[str, float | int]:
    """Report tail EV and rank first; calibration is deliberately secondary."""

    base = execution_ev_metrics(net_ev, prediction, top_k_fraction=top_k_fraction)
    valid = np.isfinite(net_ev) & np.isfinite(gross_ev) & np.isfinite(prediction)
    if not valid.any():
        return {
            **base,
            "top_k_mean_gross_ev": float("nan"),
            "top_k_sum_gross_ev": float("nan"),
            "ranking_objective": float("nan"),
        }
    net = net_ev[valid]
    gross = gross_ev[valid]
    score = prediction[valid]
    count = max(1, int(np.ceil(len(net) * top_k_fraction)))
    top = np.argsort(score, kind="stable")[-count:]
    gross_mean = float(np.mean(gross[top]))
    net_mean = float(np.mean(net[top]))
    rank_correlation = float(base["spearman"])
    if not np.isfinite(rank_correlation):
        rank_correlation = -1.0
    scale = max(float(np.nanmedian(np.abs(net))), 1e-4)
    # Model and feature selection optimize only observable rank and economic
    # tail ordering. Calibration loss remains a diagnostic, never an objective.
    objective = 2.0 * gross_mean / scale + 2.0 * net_mean / scale + rank_correlation
    return {
        **base,
        "top_k_mean_gross_ev": gross_mean,
        "top_k_sum_gross_ev": float(np.sum(gross[top])),
        "ranking_objective": float(objective),
    }


def _absolute_net_ev_prediction(
    prediction: np.ndarray,
    alpha_ev: np.ndarray,
    *,
    target_mode: TargetMode,
) -> np.ndarray:
    """Translate a residual head back to the common absolute net-EV scale."""

    raw = np.asarray(prediction, dtype=np.float64)
    if target_mode == "direct":
        return raw
    if target_mode == "residual":
        return raw + np.asarray(alpha_ev, dtype=np.float64)
    raise ValueError(f"Unsupported execution-EV target mode: {target_mode!r}")


def _aggregate_fold_metrics(
    rows: Sequence[Mapping[str, float | int]],
) -> dict[str, Any]:
    if not rows:
        return {
            "status": "no_inner_folds",
            "objective_mean": float("nan"),
            "objective_se": float("nan"),
            "folds": [],
        }
    objectives = np.asarray(
        [float(row["ranking_objective"]) for row in rows], dtype=float
    )
    mean = float(np.nanmean(objectives))
    if len(objectives) > 1:
        se = float(np.nanstd(objectives, ddof=1) / np.sqrt(len(objectives)))
    else:
        se = 0.0
    return {
        "status": "ok",
        "objective_mean": mean,
        "objective_se": se,
        "top_k_mean_gross_ev": float(
            np.nanmean([float(row["top_k_mean_gross_ev"]) for row in rows])
        ),
        "top_k_mean_net_ev": float(
            np.nanmean([float(row["top_k_mean_net_ev"]) for row in rows])
        ),
        "spearman": float(np.nanmean([float(row["spearman"]) for row in rows])),
        "huber": float(np.nanmean([float(row["huber"]) for row in rows])),
        "folds": [dict(row) for row in rows],
    }


def _evaluate_inner_feature_set(
    algorithm: str,
    x: pd.DataFrame,
    target: np.ndarray,
    gross_ev: np.ndarray,
    features: Sequence[str],
    folds: Sequence[ChronologicalPurgedSplit],
    *,
    params: Mapping[str, Any],
    top_k_fraction: float,
    frame: pd.DataFrame | None = None,
    config: ExecutionEVModelAblationConfig | None = None,
    use_nested_early_stopping: bool = False,
    absolute_net_ev: np.ndarray | None = None,
    alpha_ev: np.ndarray | None = None,
    target_mode: TargetMode = "direct",
) -> dict[str, Any]:
    actual_net_ev = target if absolute_net_ev is None else absolute_net_ev
    alpha = np.zeros(len(target), dtype=np.float64) if alpha_ev is None else alpha_ev
    rows: list[dict[str, float | int]] = []
    for split in folds:
        train = np.asarray(split.train_indices, dtype=int)
        valid = np.asarray(split.validation_indices, dtype=int)
        fit_train = train
        early_stop_x: pd.DataFrame | None = None
        early_stop_y: np.ndarray | None = None
        if use_nested_early_stopping and frame is not None and config is not None:
            fit_train, early_valid = _nested_early_stopping_indices(
                frame, train, config=config
            )
            if early_valid is not None:
                early_stop_x = x.iloc[early_valid].loc[:, list(features)]
                early_stop_y = target[early_valid]
        model = _fit_regressor(
            algorithm,
            x.iloc[fit_train].loc[:, list(features)],
            target[fit_train],
            params=params,
            early_stop_x=early_stop_x,
            early_stop_y=early_stop_y,
            early_stopping_rounds=(config.early_stopping_rounds if config else None),
        )
        prediction = np.asarray(
            model.predict(x.iloc[valid].loc[:, list(features)]), dtype=np.float64
        )
        rows.append(
            _ranking_metrics(
                actual_net_ev[valid],
                gross_ev[valid],
                _absolute_net_ev_prediction(
                    prediction, alpha[valid], target_mode=target_mode
                ),
                top_k_fraction=top_k_fraction,
            )
        )
    return _aggregate_fold_metrics(rows)


def _permutation_mda(
    algorithm: str,
    x: pd.DataFrame,
    target: np.ndarray,
    gross_ev: np.ndarray,
    features: Sequence[str],
    folds: Sequence[ChronologicalPurgedSplit],
    *,
    params: Mapping[str, Any],
    top_k_fraction: float,
    repeats: int,
    random_state: int,
    absolute_net_ev: np.ndarray | None = None,
    alpha_ev: np.ndarray | None = None,
    target_mode: TargetMode = "direct",
) -> dict[str, float]:
    """Return validation MDA drops for the ranking-first objective only."""

    drops: dict[str, list[float]] = {name: [] for name in features}
    actual_net_ev = target if absolute_net_ev is None else absolute_net_ev
    alpha = np.zeros(len(target), dtype=np.float64) if alpha_ev is None else alpha_ev
    for fold_number, split in enumerate(folds):
        train = np.asarray(split.train_indices, dtype=int)
        valid = np.asarray(split.validation_indices, dtype=int)
        train_x = x.iloc[train].loc[:, list(features)]
        valid_x = x.iloc[valid].loc[:, list(features)]
        model = _fit_regressor(algorithm, train_x, target[train], params=params)
        baseline = _ranking_metrics(
            actual_net_ev[valid],
            gross_ev[valid],
            _absolute_net_ev_prediction(
                model.predict(valid_x), alpha[valid], target_mode=target_mode
            ),
            top_k_fraction=top_k_fraction,
        )
        baseline_objective = float(baseline["ranking_objective"])
        for feature_number, name in enumerate(features):
            for repeat in range(max(1, int(repeats))):
                rng = np.random.default_rng(
                    int(random_state)
                    + 1009 * fold_number
                    + 37 * feature_number
                    + repeat
                )
                permuted = valid_x.copy()
                permuted[name] = rng.permutation(permuted[name].to_numpy())
                score = _ranking_metrics(
                    actual_net_ev[valid],
                    gross_ev[valid],
                    _absolute_net_ev_prediction(
                        model.predict(permuted), alpha[valid], target_mode=target_mode
                    ),
                    top_k_fraction=top_k_fraction,
                )
                drops[name].append(
                    baseline_objective - float(score["ranking_objective"])
                )
    return {
        name: float(np.mean(values)) if values else float("nan")
        for name, values in drops.items()
    }


def select_features_by_mda_one_se(
    algorithm: str,
    x: pd.DataFrame,
    target: np.ndarray,
    gross_ev: np.ndarray,
    inner_folds: Sequence[ChronologicalPurgedSplit],
    *,
    params: Mapping[str, Any],
    config: ExecutionEVModelAblationConfig,
    absolute_net_ev: np.ndarray | None = None,
    alpha_ev: np.ndarray | None = None,
    target_mode: TargetMode = "direct",
) -> tuple[list[str], dict[str, Any]]:
    """Iteratively remove the weakest MDA feature and apply the one-SE rule.

    Every score and permutation is computed on chronological inner validation
    blocks.  The caller supplies only an outer-fold training interval, so this
    selection cannot inspect that outer fold's evaluation outcomes.
    """

    current = list(x.columns)
    if not inner_folds:
        return current, {
            "status": "skipped_no_inner_folds",
            "selected_features": current,
            "steps": [],
        }
    min_features = max(1, min(int(config.mda_min_features), len(current)))
    max_steps = max(0, min(int(config.mda_max_steps), len(current) - min_features))
    steps: list[dict[str, Any]] = []
    for step in range(max_steps + 1):
        score = _evaluate_inner_feature_set(
            algorithm,
            x,
            target,
            gross_ev,
            current,
            inner_folds,
            params=params,
            top_k_fraction=config.top_k_fraction,
            absolute_net_ev=absolute_net_ev,
            alpha_ev=alpha_ev,
            target_mode=target_mode,
        )
        row: dict[str, Any] = {
            "step": int(step),
            "features": list(current),
            "feature_count": int(len(current)),
            **{key: value for key, value in score.items() if key != "folds"},
        }
        if step >= max_steps or len(current) <= min_features:
            steps.append(row)
            break
        importance = _permutation_mda(
            algorithm,
            x,
            target,
            gross_ev,
            current,
            inner_folds,
            params=params,
            top_k_fraction=config.top_k_fraction,
            repeats=config.mda_repeats,
            random_state=config.random_state + step,
            absolute_net_ev=absolute_net_ev,
            alpha_ev=alpha_ev,
            target_mode=target_mode,
        )
        # Stable ordering makes ties reproducible and retains the original
        # feature order as the final tie-breaker.
        removed = min(current, key=lambda name: (importance[name], current.index(name)))
        row["mda"] = importance
        row["removed_feature"] = removed
        steps.append(row)
        current.remove(removed)

    finite = [row for row in steps if np.isfinite(float(row["objective_mean"]))]
    if not finite:
        return list(x.columns), {
            "status": "fallback_no_finite_inner_objective",
            "selected_features": list(x.columns),
            "steps": steps,
        }
    best = max(finite, key=lambda row: float(row["objective_mean"]))
    threshold = float(best["objective_mean"]) - max(0.0, float(best["objective_se"]))
    eligible = [row for row in finite if float(row["objective_mean"]) >= threshold]
    # The one-SE rule chooses the smallest model within one standard error of
    # the best inner ranking result.  A stable step tie-break keeps it explicit.
    selected = min(
        eligible, key=lambda row: (int(row["feature_count"]), -int(row["step"]))
    )
    selected_features = list(selected["features"])
    return selected_features, {
        "status": "one_se_selected",
        "best_step": int(best["step"]),
        "best_objective": float(best["objective_mean"]),
        "best_objective_se": float(best["objective_se"]),
        "one_se_threshold": threshold,
        "selected_step": int(selected["step"]),
        "selected_features": selected_features,
        "steps": steps,
    }


def fit_train_only_isotonic_ev_mapping(
    raw_oof_prediction: Sequence[float] | np.ndarray,
    net_ev: Sequence[float] | np.ndarray,
    *,
    min_rows: int,
) -> IsotonicEVMapping:
    """Fit an EV map on authorized train-only OOF predictions, never eval rows."""

    raw = np.asarray(raw_oof_prediction, dtype=np.float64)
    target = np.asarray(net_ev, dtype=np.float64)
    valid = np.isfinite(raw) & np.isfinite(target)
    rows = int(valid.sum())
    if rows < max(2, int(min_rows)):
        return IsotonicEVMapping(None, "identity_insufficient_train_oof", rows)
    if np.unique(raw[valid]).size < 2:
        return IsotonicEVMapping(None, "identity_constant_train_oof", rows)
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(raw[valid], target[valid])
    return IsotonicEVMapping(model, "isotonic_train_oof", rows)


def _route_archetypes(
    frame: pd.DataFrame,
    provenance: Mapping[str, FeatureProvenance],
    *,
    route: str,
    config: ExecutionEVModelAblationConfig,
) -> tuple[pd.Series | None, dict[str, Any]]:
    """Return an observable routing label, never a CatBoost prevalence map."""

    if route == "catboost_predicted_archetype":
        return (
            frame[config.catboost_archetype_col].astype(str),
            {"status": "available", "source": config.catboost_archetype_col},
        )
    if route != "gmm_archetype":
        return None, {"status": "unavailable_unknown_route", "route": route}
    column = config.gmm_archetype_col
    if column not in frame.columns:
        return None, {"status": "unavailable_missing_column", "source": column}
    spec = provenance.get(column)
    if spec is None or not spec.pre_entry or not spec.oof_or_frozen:
        raise ValueError(
            "GMM archetype routing requires declared pre-entry OOF/frozen provenance "
            f"for {column!r}"
        )
    if spec.available_at_col:
        if spec.available_at_col not in frame.columns:
            raise ValueError(
                f"GMM archetype routing is missing availability column {spec.available_at_col!r}"
            )
        available = _utc(frame[spec.available_at_col], name=spec.available_at_col)
        decision = _utc(frame[config.decision_time_col], name=config.decision_time_col)
        if (available > decision).any():
            raise ValueError(f"GMM archetype {column!r} is available after entry")
    numeric = pd.to_numeric(frame[column], errors="coerce")
    if numeric.notna().all() and np.allclose(numeric, np.round(numeric), atol=1e-6):
        labels = numeric.round().astype("int64").astype(str).radd("gmm_")
    else:
        labels = frame[column].fillna("missing").astype(str).radd("gmm_")
    return labels, {"status": "available", "source": column}


def apply_execution_ev_causal_recent_ev_correction(
    frame: pd.DataFrame,
    mapped_prediction: Sequence[float] | np.ndarray,
    realized_net_ev: Sequence[float] | np.ndarray,
    provenance: Mapping[str, FeatureProvenance],
    *,
    route: str,
    config: ExecutionEVModelAblationConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a daily causal side x archetype residual correction to mapped EV.

    Each UTC-day snapshot uses only prior OOF rows whose outcome has resolved
    before the snapshot.  The live threshold-basis helper supplies the robust
    daily trimming and local -> side -> global shrinkage; CatBoost class
    probabilities are not used as a prevalence-calibration shortcut here.
    """

    archetypes, route_report = _route_archetypes(
        frame, provenance, route=route, config=config
    )
    mapped = np.asarray(mapped_prediction, dtype=np.float64)
    realized = np.asarray(realized_net_ev, dtype=np.float64)
    corrected = mapped.copy()
    if archetypes is None:
        return corrected, {
            "route": route,
            **route_report,
            "days": 0,
            "corrected_rows": 0,
        }
    if config.label_end_time_col is None:
        raise ValueError("causal recent-EV correction requires label_end_time_col")
    decision = _utc(frame[config.decision_time_col], name=config.decision_time_col)
    resolved = _utc(frame[config.label_end_time_col], name=config.label_end_time_col)
    if (resolved < decision).any():
        raise ValueError("outcome resolution time cannot precede the decision time")
    work = pd.DataFrame(
        {
            "__position__": np.arange(len(frame), dtype=int),
            "decision_day": decision.dt.floor("D").to_numpy(),
            "outcome_resolved_at": resolved.to_numpy(),
            "side_name": frame[config.side_col].astype(str).str.lower().to_numpy(),
            "policy_archetype": archetypes.astype(str).to_numpy(),
            "mapped_expected_ev": mapped,
            "realized_net_ev": realized,
        },
        index=frame.index,
    )
    policy = {
        "reference_mapped_expected_ev_col": "mapped_expected_ev",
        "return_col": "realized_net_ev",
        "window_days": int(config.recent_ev_window_days),
        "robust_daily_residual_trim_fraction": float(config.recent_ev_trim_fraction),
        "side_support_target": float(config.recent_ev_side_support_target),
        "local_support_target": float(config.recent_ev_local_support_target),
        "recent_ev_correction_cap": float(config.recent_ev_correction_cap),
        "top_fraction": float(config.top_k_fraction),
        "selection_mode": "top_fraction_rank",
    }
    # Keep the implementation aligned with live policy semantics without
    # importing the inference stack unless this optional evaluation is enabled.
    from .inference.threshold_basis_policy import (
        _select_side_archetype_expected_ev_batch,
    )

    days: list[dict[str, Any]] = []
    for day, batch in work.groupby("decision_day", sort=True, observed=True):
        snapshot = pd.Timestamp(day)
        eligible = (
            work["mapped_expected_ev"].notna()
            & work["realized_net_ev"].notna()
            & work["outcome_resolved_at"].lt(snapshot)
            & work["outcome_resolved_at"].ge(
                snapshot - pd.Timedelta(days=int(config.recent_ev_window_days))
            )
        )
        reference = work.loc[eligible].copy()
        current = batch.loc[batch["mapped_expected_ev"].notna()].copy()
        if current.empty:
            continue
        # The helper returns full-batch corrected EV in metadata even though
        # its primary live interface returns only selected decisions.
        _, meta = _select_side_archetype_expected_ev_batch(
            current,
            recent_ref=reference,
            all_prior=reference,
            policy=policy,
        )
        values = pd.to_numeric(meta["corrected_expected_ev"], errors="coerce")
        corrected[current["__position__"].to_numpy(dtype=int)] = values.to_numpy(
            dtype=np.float64
        )
        days.append(
            {
                "snapshot_utc": snapshot.isoformat(),
                "reference_rows": int(len(reference)),
                "corrected_rows": int(len(current)),
                "global_days_retained": int(meta.get("global_days_retained", 0)),
            }
        )
    return corrected, {
        "route": route,
        **route_report,
        "contract": (
            "daily UTC snapshots; resolved-before-snapshot OOF outcomes only; "
            "realized-minus-mapped EV; symmetric daily decile trimming; "
            "local-to-side-to-global support shrinkage"
        ),
        "days": len(days),
        "corrected_rows": int(np.isfinite(mapped).sum()),
        "daily_snapshots": days,
    }


def _inner_splits(
    frame: pd.DataFrame,
    config: ExecutionEVModelAblationConfig,
) -> list[ChronologicalPurgedSplit]:
    if len(frame) < max(2 * int(config.min_fit_rows), 8):
        return []
    try:
        return chronological_purged_splits(
            frame,
            n_splits=int(config.inner_n_splits),
            min_train_size=max(int(config.min_fit_rows), len(frame) // 4),
            decision_time_col=config.decision_time_col,
            label_end_time_col=config.label_end_time_col,
            horizon_hours=config.purge_hours,
            embargo_hours=config.embargo_hours,
        )
    except ValueError:
        return []


def _nested_early_stopping_indices(
    frame: pd.DataFrame,
    parent_train: np.ndarray,
    *,
    config: ExecutionEVModelAblationConfig,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Create a purge-safe early-stop block strictly inside one HPO train fold."""

    if len(parent_train) < max(2 * int(config.min_fit_rows), 8):
        return parent_train, None
    local_frame = frame.iloc[parent_train].reset_index(drop=True)
    try:
        splits = chronological_purged_splits(
            local_frame,
            n_splits=1,
            min_train_size=max(int(config.min_fit_rows), len(local_frame) // 3),
            decision_time_col=config.decision_time_col,
            label_end_time_col=config.label_end_time_col,
            horizon_hours=config.purge_hours,
            embargo_hours=config.embargo_hours,
        )
    except ValueError:
        return parent_train, None
    if not splits:
        return parent_train, None
    split = splits[-1]
    fit_train = parent_train[np.asarray(split.train_indices, dtype=int)]
    early_valid = parent_train[np.asarray(split.validation_indices, dtype=int)]
    if len(fit_train) < int(config.min_fit_rows) or not len(early_valid):
        return parent_train, None
    return fit_train, early_valid


def _tune_params(
    algorithm: str,
    x: pd.DataFrame,
    target: np.ndarray,
    gross_ev: np.ndarray,
    inner_folds: Sequence[ChronologicalPurgedSplit],
    *,
    features: Sequence[str],
    frame: pd.DataFrame,
    config: ExecutionEVModelAblationConfig,
    absolute_net_ev: np.ndarray | None = None,
    alpha_ev: np.ndarray | None = None,
    target_mode: TargetMode = "direct",
) -> tuple[dict[str, Any], dict[str, Any]]:
    fallback = _fixed_params(algorithm, config)
    if int(config.hpo_trials) <= 0 or not inner_folds:
        return fallback, {"status": "fixed_params", "trials": 0, "params": fallback}
    trials: list[dict[str, Any]] = []
    for number in range(int(config.hpo_trials)):
        params = _randomized_hpo_params(algorithm, config, trial_number=number)
        score = _evaluate_inner_feature_set(
            algorithm,
            x,
            target,
            gross_ev,
            features,
            inner_folds,
            params=params,
            top_k_fraction=config.top_k_fraction,
            frame=frame,
            config=config,
            use_nested_early_stopping=algorithm in {"lgbm", "catboost"},
            absolute_net_ev=absolute_net_ev,
            alpha_ev=alpha_ev,
            target_mode=target_mode,
        )
        trials.append(
            {
                "trial": number,
                "params": params,
                "early_stopping": (
                    "nested_purged_inner_train"
                    if algorithm in {"lgbm", "catboost"}
                    else "not_available"
                ),
                **{key: value for key, value in score.items() if key != "folds"},
            }
        )
    valid_trials = [row for row in trials if np.isfinite(float(row["objective_mean"]))]
    if not valid_trials:
        return fallback, {
            "status": "randomized_hpo_no_score",
            "trials": trials,
            "params": fallback,
        }
    best = max(valid_trials, key=lambda row: float(row["objective_mean"]))
    return dict(best["params"]), {
        "status": "deterministic_randomized_purged_oof_hpo",
        "objective": "ranking_objective_only",
        "search_space": "model_specific_regularized_bounded",
        "trials": trials,
        "selected_trial": int(best["trial"]),
        "params": dict(best["params"]),
    }


def _inner_raw_oof(
    algorithm: str,
    x: pd.DataFrame,
    target: np.ndarray,
    features: Sequence[str],
    inner_folds: Sequence[ChronologicalPurgedSplit],
    *,
    params: Mapping[str, Any],
) -> np.ndarray:
    raw = np.full(len(x), np.nan, dtype=np.float64)
    for split in inner_folds:
        train = np.asarray(split.train_indices, dtype=int)
        valid = np.asarray(split.validation_indices, dtype=int)
        model = _fit_regressor(
            algorithm,
            x.iloc[train].loc[:, list(features)],
            target[train],
            params=params,
        )
        raw[valid] = np.asarray(
            model.predict(x.iloc[valid].loc[:, list(features)]), dtype=np.float64
        )
    return raw


def _oof_provenance(
    frame: pd.DataFrame,
    folds: Sequence[ChronologicalPurgedSplit],
    *,
    decision_time_col: str,
) -> pd.DataFrame:
    decision = _utc(frame[decision_time_col], name=decision_time_col)
    result = pd.DataFrame(
        {
            "execution_ev_model_ablation_oof_fold": pd.Series(
                pd.NA, index=frame.index, dtype="Int64"
            ),
            "execution_ev_model_ablation_oof_validation_start_utc": pd.Series(
                pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]"
            ),
            "execution_ev_model_ablation_oof_train_decision_cutoff_utc": pd.Series(
                pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]"
            ),
        }
    )
    for split in folds:
        train = np.asarray(split.train_indices, dtype=int)
        valid = np.asarray(split.validation_indices, dtype=int)
        result.iloc[
            valid, result.columns.get_loc("execution_ev_model_ablation_oof_fold")
        ] = int(split.fold)
        result.iloc[
            valid,
            result.columns.get_loc(
                "execution_ev_model_ablation_oof_validation_start_utc"
            ),
        ] = split.validation_start
        result.iloc[
            valid,
            result.columns.get_loc(
                "execution_ev_model_ablation_oof_train_decision_cutoff_utc"
            ),
        ] = decision.iloc[train].max()
    return result


def _fit_outer_side_arm(
    algorithm: str,
    hpo_arm: HPOArm,
    arm: FeatureArm,
    x: pd.DataFrame,
    outer_train: np.ndarray,
    outer_valid: np.ndarray,
    target: np.ndarray,
    gross_ev: np.ndarray,
    absolute_net_ev: np.ndarray,
    alpha_ev: np.ndarray,
    target_mode: TargetMode,
    frame: pd.DataFrame,
    *,
    config: ExecutionEVModelAblationConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one side and one outer fold without exposing evaluation outcomes."""

    local_frame = frame.iloc[outer_train].reset_index(drop=True)
    local_x = x.iloc[outer_train].reset_index(drop=True)
    local_target = target[outer_train]
    local_absolute_net_ev = absolute_net_ev[outer_train]
    local_alpha_ev = alpha_ev[outer_train]
    local_gross = gross_ev[outer_train]
    inner = _inner_splits(local_frame, config)
    if arm == "all_features":
        selected = list(local_x.columns)
        selection = {
            "status": "all_features_reference",
            "selected_features": selected,
            "steps": [],
        }
    else:
        # MDA is a feature-selection operation.  Its inner scores use only
        # the declared fixed reference; the subsequent HPO operates on the
        # selected subset and remains entirely inside the outer-train span.
        selected, selection = select_features_by_mda_one_se(
            algorithm,
            local_x,
            local_target,
            local_gross,
            inner,
            params=_fixed_params(algorithm, config),
            config=config,
            absolute_net_ev=local_absolute_net_ev,
            alpha_ev=local_alpha_ev,
            target_mode=target_mode,
        )
    if hpo_arm == "with_hpo":
        params, hpo = _tune_params(
            algorithm,
            local_x,
            local_target,
            local_gross,
            inner,
            features=selected,
            frame=local_frame,
            config=config,
            absolute_net_ev=local_absolute_net_ev,
            alpha_ev=local_alpha_ev,
            target_mode=target_mode,
        )
    else:
        params = _fixed_params(algorithm, config)
        hpo = {"status": "fixed_params_reference", "trials": 0, "params": params}
    raw_inner_oof = _inner_raw_oof(
        algorithm, local_x, local_target, selected, inner, params=params
    )
    mapper = fit_train_only_isotonic_ev_mapping(
        _absolute_net_ev_prediction(
            raw_inner_oof, local_alpha_ev, target_mode=target_mode
        ),
        local_absolute_net_ev,
        min_rows=config.isotonic_min_rows,
    )
    model = _fit_regressor(
        algorithm, local_x.loc[:, selected], local_target, params=params
    )
    raw_valid = np.asarray(
        model.predict(x.iloc[outer_valid].loc[:, selected]), dtype=np.float64
    )
    prediction = mapper.predict(
        _absolute_net_ev_prediction(
            raw_valid, alpha_ev[outer_valid], target_mode=target_mode
        )
    )
    decision = _utc(frame[config.decision_time_col], name=config.decision_time_col)
    return prediction, {
        "train_rows": int(len(outer_train)),
        "valid_rows": int(len(outer_valid)),
        "train_sides": sorted(
            set(frame.iloc[outer_train][config.side_col].astype(str).str.lower())
        ),
        "validation_sides": sorted(
            set(frame.iloc[outer_valid][config.side_col].astype(str).str.lower())
        ),
        "train_decision_cutoff_utc": decision.iloc[outer_train].max().isoformat(),
        "validation_start_utc": decision.iloc[outer_valid].min().isoformat(),
        "inner_fold_count": int(len(inner)),
        "params": dict(params),
        "hpo": hpo,
        "feature_selection": selection,
        "isotonic": {"status": mapper.status, "train_rows": mapper.train_rows},
        "target_mode": target_mode,
    }


def _fit_final_side_arm(
    algorithm: str,
    hpo_arm: HPOArm,
    arm: FeatureArm,
    x: pd.DataFrame,
    target: np.ndarray,
    gross_ev: np.ndarray,
    absolute_net_ev: np.ndarray,
    alpha_ev: np.ndarray,
    target_mode: TargetMode,
    side_positions: np.ndarray,
    frame: pd.DataFrame,
    *,
    config: ExecutionEVModelAblationConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    local_frame = frame.iloc[side_positions].reset_index(drop=True)
    local_x = x.iloc[side_positions].reset_index(drop=True)
    local_target = target[side_positions]
    local_absolute_net_ev = absolute_net_ev[side_positions]
    local_alpha_ev = alpha_ev[side_positions]
    local_gross = gross_ev[side_positions]
    inner = _inner_splits(local_frame, config)
    if arm == "all_features":
        selected = list(local_x.columns)
        selection = {
            "status": "all_features_reference",
            "selected_features": selected,
            "steps": [],
        }
    else:
        selected, selection = select_features_by_mda_one_se(
            algorithm,
            local_x,
            local_target,
            local_gross,
            inner,
            params=_fixed_params(algorithm, config),
            config=config,
            absolute_net_ev=local_absolute_net_ev,
            alpha_ev=local_alpha_ev,
            target_mode=target_mode,
        )
    if hpo_arm == "with_hpo":
        params, hpo = _tune_params(
            algorithm,
            local_x,
            local_target,
            local_gross,
            inner,
            features=selected,
            frame=local_frame,
            config=config,
            absolute_net_ev=local_absolute_net_ev,
            alpha_ev=local_alpha_ev,
            target_mode=target_mode,
        )
    else:
        params = _fixed_params(algorithm, config)
        hpo = {"status": "fixed_params_reference", "trials": 0, "params": params}
    raw_inner_oof = _inner_raw_oof(
        algorithm, local_x, local_target, selected, inner, params=params
    )
    mapper = fit_train_only_isotonic_ev_mapping(
        _absolute_net_ev_prediction(
            raw_inner_oof, local_alpha_ev, target_mode=target_mode
        ),
        local_absolute_net_ev,
        min_rows=config.isotonic_min_rows,
    )
    model = _fit_regressor(
        algorithm, local_x.loc[:, selected], local_target, params=params
    )
    return {
        "model": model,
        "features": list(selected),
        "params": dict(params),
        "isotonic": mapper,
        "target_mode": target_mode,
    }, {
        "rows": int(len(side_positions)),
        "inner_fold_count": int(len(inner)),
        "hpo": hpo,
        "feature_selection": selection,
        "isotonic": {"status": mapper.status, "train_rows": mapper.train_rows},
        "target_mode": target_mode,
    }


def _leaderboard(
    net_ev: np.ndarray,
    gross_ev: np.ndarray,
    predictions: pd.DataFrame,
    *,
    top_k_fraction: float,
    evaluation_mask: np.ndarray,
    frame: pd.DataFrame | None = None,
    oof_provenance: pd.DataFrame | None = None,
    config: ExecutionEVModelAblationConfig | None = None,
    post_calibrator_columns: frozenset[str] = frozenset(),
    promotion_eligible_columns: frozenset[str] | None = None,
) -> pd.DataFrame:
    if promotion_eligible_columns is not None:
        post_calibrator_columns = frozenset(promotion_eligible_columns)
    stability_available = (
        frame is not None and oof_provenance is not None and config is not None
    )
    if stability_available:
        decision = _utc(
            frame[config.decision_time_col], name=config.decision_time_col
        )
        fold_values = oof_provenance[
            "execution_ev_model_ablation_oof_fold"
        ]
        valid_folds = fold_values.loc[evaluation_mask].dropna().astype(int)
        latest_fold = int(valid_folds.max())
        latest_month = decision.loc[evaluation_mask].dt.strftime("%Y-%m").max()
        latest_fold_mask = (
            evaluation_mask
            & fold_values.fillna(-1).astype(int).eq(latest_fold).to_numpy()
        )
        latest_month_mask = (
            evaluation_mask
            & decision.dt.strftime("%Y-%m").eq(latest_month).to_numpy()
        )
    else:
        latest_fold = -1
        latest_month = "not_available"
        latest_fold_mask = evaluation_mask.copy()
        latest_month_mask = evaluation_mask.copy()
    rows: list[dict[str, Any]] = []
    for column in predictions.columns:
        full_prediction = predictions[column].to_numpy(dtype=np.float64)
        prediction = full_prediction[evaluation_mask]
        metrics = _ranking_metrics(
            net_ev[evaluation_mask],
            gross_ev[evaluation_mask],
            prediction,
            top_k_fraction=top_k_fraction,
        )
        parts = column.split("__")
        is_model_arm = len(parts) >= 4 and parts[2] in {"without_hpo", "with_hpo"}
        latest_fold_metrics = _ranking_metrics(
            net_ev[latest_fold_mask],
            gross_ev[latest_fold_mask],
            full_prediction[latest_fold_mask],
            top_k_fraction=top_k_fraction,
        )
        latest_month_metrics = _ranking_metrics(
            net_ev[latest_month_mask],
            gross_ev[latest_month_mask],
            full_prediction[latest_month_mask],
            top_k_fraction=top_k_fraction,
        )
        top_count = max(1, int(np.ceil(len(prediction) * top_k_fraction)))
        evaluation_positions = np.flatnonzero(evaluation_mask)
        global_top_positions = evaluation_positions[
            np.argsort(prediction, kind="stable")[-top_count:]
        ]
        pooled_latest_fold_rows = int(latest_fold_mask[global_top_positions].sum())
        pooled_latest_month_rows = int(latest_month_mask[global_top_positions].sum())
        minimum_rows = (
            int(config.min_latest_period_selected_rows)
            if config is not None
            else 0
        )
        minimum_share = (
            float(config.min_latest_period_selection_share)
            if config is not None
            else 0.0
        )
        minimum_ev = (
            float(config.min_latest_period_top_k_net_ev)
            if config is not None
            else -np.inf
        )
        stability_gate = (
            not stability_available
            or (
                float(latest_fold_metrics["top_k_mean_net_ev"]) >= minimum_ev
                and float(latest_month_metrics["top_k_mean_net_ev"]) >= minimum_ev
                and pooled_latest_fold_rows >= minimum_rows
                and pooled_latest_month_rows >= minimum_rows
                and pooled_latest_fold_rows / top_count >= minimum_share
                and pooled_latest_month_rows / top_count >= minimum_share
            )
        )
        post_calibrator = column in post_calibrator_columns
        promotion_eligible = post_calibrator and (
            stability_gate
            or config is None
            or not bool(config.require_latest_period_stability)
        )
        rows.append(
            {
                "prediction": column,
                "algorithm": parts[0],
                "target_mode": parts[1] if is_model_arm else "baseline_alpha",
                "hpo_arm": parts[2] if is_model_arm else "baseline",
                "feature_arm": parts[3] if is_model_arm else "baseline",
                "recent_ev_route": "__".join(parts[4:]) if len(parts) > 4 else "none",
                "arm": "__".join(parts[1:]) if is_model_arm else "baseline",
                "oof_rows": int(evaluation_mask.sum()),
                "ranking_scope": "global_shared_outer_oof",
                "ranking_stage": (
                    "after_causal_21d_admission_calibrator"
                    if post_calibrator
                    else "before_admission_calibrator_diagnostic_only"
                ),
                "post_calibrator_route_available": post_calibrator,
                "promotion_eligible": promotion_eligible,
                "latest_fold": latest_fold,
                "latest_fold_rows": int(latest_fold_mask.sum()),
                "latest_fold_top_k_rows": int(latest_fold_metrics["top_k_rows"]),
                "latest_fold_top_k_mean_net_ev": float(
                    latest_fold_metrics["top_k_mean_net_ev"]
                ),
                "latest_month": latest_month,
                "latest_month_rows": int(latest_month_mask.sum()),
                "latest_month_top_k_rows": int(latest_month_metrics["top_k_rows"]),
                "latest_month_top_k_mean_net_ev": float(
                    latest_month_metrics["top_k_mean_net_ev"]
                ),
                "pooled_global_top_k_latest_fold_rows": pooled_latest_fold_rows,
                "pooled_global_top_k_latest_month_rows": pooled_latest_month_rows,
                "pooled_global_top_k_latest_fold_share": (
                    pooled_latest_fold_rows / top_count
                ),
                "pooled_global_top_k_latest_month_share": (
                    pooled_latest_month_rows / top_count
                ),
                "latest_period_stability_gate": bool(stability_gate),
                **metrics,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "promotion_eligible",
                "ranking_objective",
                "top_k_mean_gross_ev",
                "top_k_mean_net_ev",
                "spearman",
            ],
            ascending=[False, False, False, False, False],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def train_execution_ev_model_ablation(
    frame: pd.DataFrame,
    provenance: Mapping[str, FeatureProvenance],
    *,
    config: ExecutionEVModelAblationConfig = ExecutionEVModelAblationConfig(),
) -> ExecutionEVModelAblationBundle:
    """Train all three models with strict side-local chronological OOF evidence."""

    algorithms = tuple(dict.fromkeys(map(str, config.algorithms)))
    unknown = sorted(set(algorithms) - set(ALGORITHM_NAMES))
    if unknown:
        raise ValueError("Unsupported algorithms: " + ", ".join(unknown))
    if not algorithms:
        raise ValueError("At least one ablation algorithm is required")
    target_modes = tuple(dict.fromkeys(config.target_modes))
    if not target_modes or not set(target_modes) <= {"direct", "residual"}:
        raise ValueError(
            "Execution-EV model ablation target_modes must contain direct and/or residual"
        )
    feature_arms = tuple(dict.fromkeys(config.feature_arms))
    if not feature_arms or not set(feature_arms) <= {"all_features", "mda_1se"}:
        raise ValueError("feature_arms must contain all_features and/or mda_1se")
    if not 0.0 < float(config.top_k_fraction) <= 1.0:
        raise ValueError("top_k_fraction must be in (0, 1]")
    unknown_routes = sorted(
        set(config.recent_ev_correction_routes)
        - {"catboost_predicted_archetype", "gmm_archetype"}
    )
    if unknown_routes:
        raise ValueError(
            "Unsupported recent-EV correction routes: " + ", ".join(unknown_routes)
        )

    raw_columns, archetype_levels = validate_execution_ev_model_ablation_contract(
        frame,
        provenance,
        decision_time_col=config.decision_time_col,
        side_col=config.side_col,
        catboost_archetype_col=config.catboost_archetype_col,
        additional_input_families=config.additional_input_families,
    )
    if (
        config.target_spec.alpha_ev_col not in raw_columns
        or provenance[config.target_spec.alpha_ev_col].family != "alpha_score"
    ):
        raise ValueError(
            "Execution-EV ablation residual target requires the configured existing "
            "alpha EV column as a declared alpha_score model input"
        )
    absolute_net_ev = build_execution_ev_target(
        frame,
        ExecutionEVTargetSpec(
            net_ev_col=config.target_spec.net_ev_col,
            alpha_ev_col=config.target_spec.alpha_ev_col,
            mode="direct",
            target_col=config.target_spec.target_col,
            horizon_hours=config.target_spec.horizon_hours,
        ),
    ).to_numpy(dtype=np.float64)
    alpha_ev = _finite_numeric(
        frame, config.target_spec.alpha_ev_col, role="existing alpha EV"
    )
    targets = {
        mode: build_execution_ev_target(
            frame,
            ExecutionEVTargetSpec(
                net_ev_col=config.target_spec.net_ev_col,
                alpha_ev_col=config.target_spec.alpha_ev_col,
                mode=mode,
                target_col=config.target_spec.target_col,
                horizon_hours=config.target_spec.horizon_hours,
            ),
        ).to_numpy(dtype=np.float64)
        for mode in target_modes
    }
    if not np.isfinite(absolute_net_ev).all() or any(
        not np.isfinite(target).all() for target in targets.values()
    ):
        raise ValueError("Execution-EV ablation targets must be finite")
    gross_ev = _finite_numeric(frame, config.gross_ev_col, role="gross EV")
    if config.label_end_time_col is not None:
        decision = _utc(frame[config.decision_time_col], name=config.decision_time_col)
        resolved = _utc(
            frame[config.label_end_time_col], name=config.label_end_time_col
        )
        if (resolved < decision).any():
            raise ValueError(
                "execution label end time cannot precede the decision time"
            )
    sides = _side_values(frame, config.side_col)
    x = _materialize_feature_matrix(
        frame,
        raw_columns,
        catboost_archetype_col=config.catboost_archetype_col,
        archetype_levels=archetype_levels,
    )
    folds = chronological_purged_splits(
        frame,
        n_splits=config.n_splits,
        min_train_size=config.min_train_rows,
        decision_time_col=config.decision_time_col,
        label_end_time_col=config.label_end_time_col,
        horizon_hours=config.purge_hours,
        embargo_hours=config.embargo_hours,
    )
    oof_provenance = _oof_provenance(
        frame, folds, decision_time_col=config.decision_time_col
    )
    predictions = pd.DataFrame(index=frame.index)
    predictions["baseline__frozen_alpha"] = alpha_ev
    hpo_arms: tuple[HPOArm, ...] = (
        ("without_hpo", "with_hpo") if int(config.hpo_trials) > 0 else ("without_hpo",)
    )
    audits: dict[str, dict[str, dict[str, dict[str, list[dict[str, Any]]]]]] = {}
    for algorithm in algorithms:
        audits[algorithm] = {
            target_mode: {
                hpo_arm: {arm: [] for arm in feature_arms} for hpo_arm in hpo_arms
            }
            for target_mode in target_modes
        }
        for target_mode in target_modes:
            for hpo_arm in hpo_arms:
                for arm in feature_arms:
                    output = np.full(len(frame), np.nan, dtype=np.float64)
                    for split in folds:
                        for side in ("long", "short"):
                            train = np.asarray(
                                [
                                    index
                                    for index in split.train_indices
                                    if sides[index] == side
                                ],
                                dtype=int,
                            )
                            valid = np.asarray(
                                [
                                    index
                                    for index in split.validation_indices
                                    if sides[index] == side
                                ],
                                dtype=int,
                            )
                            audit: dict[str, Any] = {
                                "fold": int(split.fold),
                                "side": side,
                                "target_mode": target_mode,
                            }
                            if len(train) < int(config.min_fit_rows) or not len(valid):
                                audit.update(
                                    {
                                        "status": "insufficient_side_rows",
                                        "train_rows": int(len(train)),
                                        "valid_rows": int(len(valid)),
                                    }
                                )
                                audits[algorithm][target_mode][hpo_arm][arm].append(
                                    audit
                                )
                                continue
                            prediction, detail = _fit_outer_side_arm(
                                algorithm,
                                hpo_arm,
                                arm,
                                x,
                                train,
                                valid,
                                targets[target_mode],
                                gross_ev,
                                absolute_net_ev,
                                alpha_ev,
                                target_mode,
                                frame,
                                config=config,
                            )
                            output[valid] = prediction
                            audit.update({"status": "ok", **detail})
                            audits[algorithm][target_mode][hpo_arm][arm].append(audit)
                    predictions[f"{algorithm}__{target_mode}__{hpo_arm}__{arm}"] = (
                        output
                    )

    model_prediction_columns = [
        column for column in predictions.columns if column.count("__") == 3
    ]
    shared_oof_mask = np.ones(len(frame), dtype=bool)
    for column in model_prediction_columns:
        shared_oof_mask &= np.isfinite(predictions[column].to_numpy(dtype=np.float64))
    if not shared_oof_mask.any():
        raise ValueError("Execution-EV model ablation has no shared outer-OOF rows")
    mismatched_oof = [
        column
        for column in model_prediction_columns
        if not np.array_equal(
            np.isfinite(predictions[column].to_numpy(dtype=np.float64)), shared_oof_mask
        )
    ]
    if mismatched_oof:
        raise ValueError(
            "Execution-EV model arms must use identical outer-OOF rows: "
            + ", ".join(mismatched_oof)
        )

    recent_ev_correction: dict[str, Any] = {}
    promotion_eligible_columns: set[str] = set()
    if bool(config.recent_ev_correction_enabled):
        for column in model_prediction_columns:
            recent_ev_correction[column] = {}
            for route in config.recent_ev_correction_routes:
                corrected, route_report = (
                    apply_execution_ev_causal_recent_ev_correction(
                        frame,
                        predictions[column].to_numpy(dtype=np.float64),
                        absolute_net_ev,
                        provenance,
                        route=str(route),
                        config=config,
                    )
                )
                prediction_name = f"{column}__recent_ev_{route}"
                predictions[prediction_name] = corrected
                recent_ev_correction[column][str(route)] = {
                    "prediction": prediction_name,
                    **route_report,
                }
                if route_report.get("status") == "available":
                    promotion_eligible_columns.add(prediction_name)

    final_models: dict[
        str, dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]]
    ] = {}
    final_feature_sets: dict[
        str, dict[str, dict[str, dict[str, dict[str, list[str]]]]]
    ] = {}
    final_audit: dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]] = {}
    for algorithm in algorithms:
        final_models[algorithm] = {}
        final_feature_sets[algorithm] = {}
        final_audit[algorithm] = {}
        for target_mode in target_modes:
            final_models[algorithm][target_mode] = {
                hpo_arm: {arm: {} for arm in feature_arms} for hpo_arm in hpo_arms
            }
            final_feature_sets[algorithm][target_mode] = {
                hpo_arm: {arm: {} for arm in feature_arms} for hpo_arm in hpo_arms
            }
            final_audit[algorithm][target_mode] = {
                hpo_arm: {arm: {} for arm in feature_arms} for hpo_arm in hpo_arms
            }
            for hpo_arm in hpo_arms:
                for arm in feature_arms:
                    for side in ("long", "short"):
                        positions = np.flatnonzero(sides == side)
                        if len(positions) < int(config.min_fit_rows):
                            final_audit[algorithm][target_mode][hpo_arm][arm][side] = {
                                "status": "insufficient_side_rows",
                                "rows": int(len(positions)),
                            }
                            continue
                        fitted, detail = _fit_final_side_arm(
                            algorithm,
                            hpo_arm,
                            arm,
                            x,
                            targets[target_mode],
                            gross_ev,
                            absolute_net_ev,
                            alpha_ev,
                            target_mode,
                            positions,
                            frame,
                            config=config,
                        )
                        final_models[algorithm][target_mode][hpo_arm][arm][side] = (
                            fitted
                        )
                        final_feature_sets[algorithm][target_mode][hpo_arm][arm][
                            side
                        ] = list(fitted["features"])
                        final_audit[algorithm][target_mode][hpo_arm][arm][side] = {
                            "status": "ok",
                            **detail,
                        }

    leaderboard = _leaderboard(
        absolute_net_ev,
        gross_ev,
        predictions,
        top_k_fraction=config.top_k_fraction,
        evaluation_mask=shared_oof_mask,
        frame=frame,
        oof_provenance=oof_provenance,
        config=config,
        post_calibrator_columns=frozenset(promotion_eligible_columns),
    )
    promotion_leaderboard = leaderboard.loc[leaderboard["promotion_eligible"]].copy()
    promotion_status = (
        "eligible_post_causal_21d_admission_calibrator"
        if not promotion_leaderboard.empty
        else "blocked_no_available_post_calibrator_route"
    )
    report = {
        "schema": EXECUTION_EV_MODEL_ABLATION_SCHEMA,
        "evaluation_status": "outer_oof_only_not_policy_selection",
        "leakage_contract": (
            "exact frozen pre-entry feature provenance; side-local outer expanding "
            "chronological folds; label-overlap purge and embargo; HPO, MDA, and "
            "isotonic mapping restricted to each outer fold's training rows"
        ),
        "ranking_objective": (
            "top-k gross EV and net EV plus rank correlation only; calibration "
            "metrics are diagnostic and never used for model or feature selection"
        ),
        "promotion_metric_contract": {
            "status": promotion_status,
            "top_fraction": float(config.top_k_fraction),
            "ranking_scope": "global_shared_outer_oof",
            "ranking_stage": "after_causal_21d_admission_calibrator",
            "selection_unit": "rows pooled across timestamps and sides",
            "raw_model_scores": "diagnostic_only_not_promotion_eligible",
            "post_calibrator_route_predictions": sorted(promotion_eligible_columns),
            "promotion_eligible_predictions": promotion_leaderboard[
                "prediction"
            ].astype(str).tolist(),
            "latest_period_gate": {
                "required": bool(config.require_latest_period_stability),
                "minimum_pooled_selected_rows": int(
                    config.min_latest_period_selected_rows
                ),
                "minimum_pooled_selection_share": float(
                    config.min_latest_period_selection_share
                ),
                "minimum_local_top_k_mean_net_ev": float(
                    config.min_latest_period_top_k_net_ev
                ),
                "periods": ["latest_outer_fold", "latest_calendar_month"],
            },
        },
        "feature_manifest": {
            "raw_frozen_columns": raw_columns,
            "expanded_model_columns": list(x.columns),
            "catboost_archetype_levels": list(archetype_levels),
            "five_auxiliary_families": list(FIVE_AUXILIARY_FAMILIES),
            "required_input_families": list(REQUIRED_INPUT_FAMILIES),
            "additional_input_families": list(config.additional_input_families),
            "catboost_argmax_context": {
                "column": config.catboost_archetype_col,
                "family": CATBOOST_ARGMAX_CONTEXT_FAMILY,
            },
            "probability_vector_size": len(
                _family_columns(provenance, "catboost_probabilities")
            ),
        },
        "fixed_params": {
            algorithm: _fixed_params(algorithm, config) for algorithm in algorithms
        },
        "hpo_arms": list(hpo_arms),
        "target_modes": list(target_modes),
        "shared_outer_oof_rows": int(shared_oof_mask.sum()),
        "recent_ev_correction": recent_ev_correction,
        "config": asdict(config),
        "folds": [
            {
                "fold": int(split.fold),
                "validation_start": split.validation_start.isoformat(),
                "validation_end": split.validation_end.isoformat(),
                "purge_hours": float(split.purge_hours),
                "embargo_hours": float(split.embargo_hours),
            }
            for split in folds
        ],
        "oof_audit": audits,
        "final_fit_audit": final_audit,
        "leaderboard": leaderboard.to_dict(orient="records"),
        "promotion_leaderboard": promotion_leaderboard.to_dict(orient="records"),
    }
    return ExecutionEVModelAblationBundle(
        schema=EXECUTION_EV_MODEL_ABLATION_SCHEMA,
        config=asdict(config),
        provenance=dict(provenance),
        raw_feature_columns=tuple(raw_columns),
        expanded_feature_columns=tuple(x.columns),
        archetype_levels=tuple(archetype_levels),
        final_feature_sets=final_feature_sets,
        models=final_models,
        report=report,
        oof_predictions=predictions,
        oof_provenance=oof_provenance,
    )


def predict_execution_ev_model_ablation_bundle(
    bundle: ExecutionEVModelAblationBundle,
    frame: pd.DataFrame,
    *,
    algorithms: Sequence[str] | None = None,
    target_modes: Sequence[TargetMode] | None = None,
    hpo_arms: Sequence[HPOArm] | None = None,
    arms: Sequence[FeatureArm] = ("all_features", "mda_1se"),
) -> pd.DataFrame:
    """Score final fits from pre-entry features only; targets are never read."""

    if bundle.schema != EXECUTION_EV_MODEL_ABLATION_SCHEMA:
        raise ValueError(
            f"Unsupported execution-EV ablation bundle schema {bundle.schema!r}"
        )
    bundle_config = dict(bundle.config)
    target_spec = bundle_config.get("target_spec")
    if isinstance(target_spec, Mapping):
        bundle_config["target_spec"] = ExecutionEVTargetSpec(**target_spec)
    if "target_modes" in bundle_config:
        bundle_config["target_modes"] = tuple(bundle_config["target_modes"])
    if "additional_input_families" in bundle_config:
        bundle_config["additional_input_families"] = tuple(
            bundle_config["additional_input_families"]
        )
    if "feature_arms" in bundle_config:
        bundle_config["feature_arms"] = tuple(bundle_config["feature_arms"])
    config = ExecutionEVModelAblationConfig(**bundle_config)
    validate_execution_ev_model_ablation_contract(
        frame,
        bundle.provenance,
        decision_time_col=config.decision_time_col,
        side_col=config.side_col,
        catboost_archetype_col=config.catboost_archetype_col,
        additional_input_families=config.additional_input_families,
    )
    if (
        config.target_spec.alpha_ev_col not in bundle.raw_feature_columns
        or bundle.provenance[config.target_spec.alpha_ev_col].family != "alpha_score"
    ):
        raise ValueError(
            "Execution-EV ablation bundle lacks the declared existing alpha EV input"
        )
    x = _materialize_feature_matrix(
        frame,
        bundle.raw_feature_columns,
        catboost_archetype_col=config.catboost_archetype_col,
        archetype_levels=bundle.archetype_levels,
    )
    if tuple(x.columns) != bundle.expanded_feature_columns:
        raise ValueError(
            "Inference expanded feature order does not match the frozen bundle"
        )
    selected_algorithms = tuple(algorithms or bundle.models.keys())
    unknown = sorted(set(selected_algorithms) - set(bundle.models))
    if unknown:
        raise ValueError("Bundle does not contain algorithms: " + ", ".join(unknown))
    sides = _side_values(frame, config.side_col)
    alpha_ev = _finite_numeric(
        frame, config.target_spec.alpha_ev_col, role="existing alpha EV"
    )
    output = pd.DataFrame(index=frame.index)
    for algorithm in selected_algorithms:
        selected_target_modes = tuple(target_modes or bundle.models[algorithm].keys())
        missing_modes = sorted(
            set(selected_target_modes) - set(bundle.models[algorithm])
        )
        if missing_modes:
            raise ValueError(
                f"Bundle has no {algorithm!r} target modes: " + ", ".join(missing_modes)
            )
        for target_mode in selected_target_modes:
            selected_hpo_arms = tuple(
                hpo_arms or bundle.models[algorithm][target_mode].keys()
            )
            missing_hpo = sorted(
                set(selected_hpo_arms) - set(bundle.models[algorithm][target_mode])
            )
            if missing_hpo:
                raise ValueError(
                    f"Bundle has no {algorithm!r} {target_mode!r} HPO arms: "
                    + ", ".join(missing_hpo)
                )
            for hpo_arm in selected_hpo_arms:
                for arm in arms:
                    if arm not in bundle.models[algorithm][target_mode][hpo_arm]:
                        raise ValueError(
                            f"Bundle has no {algorithm!r} {target_mode!r} {hpo_arm!r} {arm!r} arm"
                        )
                    prediction = np.full(len(frame), np.nan, dtype=np.float64)
                    for side in ("long", "short"):
                        fitted = bundle.models[algorithm][target_mode][hpo_arm][
                            arm
                        ].get(side)
                        positions = np.flatnonzero(sides == side)
                        if fitted is None:
                            continue
                        features = list(fitted["features"])
                        raw = np.asarray(
                            fitted["model"].predict(x.iloc[positions].loc[:, features]),
                            dtype=np.float64,
                        )
                        absolute = _absolute_net_ev_prediction(
                            raw, alpha_ev[positions], target_mode=target_mode
                        )
                        prediction[positions] = fitted["isotonic"].predict(absolute)
                    output[f"{algorithm}__{target_mode}__{hpo_arm}__{arm}"] = prediction
    return output


def save_execution_ev_model_ablation_bundle(
    bundle: ExecutionEVModelAblationBundle,
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    return output


def load_execution_ev_model_ablation_bundle(
    path: str | Path,
) -> ExecutionEVModelAblationBundle:
    bundle = joblib.load(Path(path))
    if not isinstance(bundle, ExecutionEVModelAblationBundle):
        raise ValueError("Artifact is not an execution-EV model-ablation bundle")
    if bundle.schema != EXECUTION_EV_MODEL_ABLATION_SCHEMA:
        raise ValueError(
            f"Unsupported execution-EV ablation bundle schema {bundle.schema!r}"
        )
    return bundle


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_execution_ev_model_ablation_report(
    bundle: ExecutionEVModelAblationBundle,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write JSON manifest, OOF predictions, provenance, and ranking table."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "execution_ev_model_ablation_report.json"
    report_path.write_text(
        json.dumps(_json_safe(bundle.report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    leaderboard_path = root / "execution_ev_model_ablation_leaderboard.csv"
    pd.DataFrame(bundle.report["leaderboard"]).to_csv(leaderboard_path, index=False)
    oof_path = root / "execution_ev_model_ablation_oof.parquet"
    bundle.oof_predictions.join(bundle.oof_provenance).to_parquet(
        oof_path, index=False, compression="zstd"
    )
    features_path = root / "execution_ev_model_ablation_features.json"
    features_path.write_text(
        json.dumps(
            _json_safe(
                {
                    "raw_feature_columns": bundle.raw_feature_columns,
                    "expanded_feature_columns": bundle.expanded_feature_columns,
                    "archetype_levels": bundle.archetype_levels,
                    "final_feature_sets": bundle.final_feature_sets,
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "report": report_path,
        "leaderboard": leaderboard_path,
        "oof": oof_path,
        "features": features_path,
    }
