#!/usr/bin/env python3
"""Run second-generation CatBoost meaningful-MFE event ablations.

The study separates clean-event probability from conditional path quality,
adds a three-way competing-risk model, performs side-local task-specific
feature selection, expands CatBoost geometry HPO, and applies causal rolling
Platt calibration. Model and feature choices use only the purged April split;
May, June, and partial July are expanding resolved-label OOF folds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.exact_policy_soft_binary_ablation import (  # noqa: E402
    economic_metrics,
)
from extreme_price_movements.meaningful_mfe_event_ablation import (  # noqa: E402
    TripleBarrierSoftLabel,
    atr_soft_triple_barrier_labels,
    competing_risk_targets,
    event_quality_scores,
    expanding_resolved_month_folds,
    first_21d_admission,
)
from extreme_price_movements.path_auxiliary_lgbm import (  # noqa: E402
    fit_base_archetype_label_feature_contract,
    transform_base_archetype_label_features,
)
from scripts.run_meaningful_mfe_event_classifier_ablation import (  # noqa: E402
    DEFAULT_CONTEXT,
    DEFAULT_EXACT_POLICY,
    DEFAULT_FEATURE_DIR,
    DEFAULT_INCUMBENT,
    DEFAULT_LABELS,
    DEFAULT_LABEL_RESOLUTION_COLUMN,
    DEFAULT_SELECTION,
    IDENTITY,
    SIDES,
    _classification_metrics,
)
from scripts.run_path_auxiliary_lgbm_models import (  # noqa: E402
    ARCHETYPE_COLUMNS,
    MANDATORY_HANDOFF_MODEL_FEATURES,
    REPRESENTATION_AVAILABLE_FEATURE,
    _complete_archetype_source,
    _file_sha256,
    _join_archetype_context,
    _load_labels,
    _load_static_features,
    _overlay_handoff_model_features,
)

SCHEMA = "meaningful_mfe_catboost_v2_ablation_v1"
DEFAULT_V1 = (
    ROOT
    / "data_perp/artifacts/meaningful_mfe_event_classifier_ablation_20260725_v1/oof_predictions.parquet"
)
DEFAULT_OUTPUT = (
    ROOT / "data_perp/artifacts/meaningful_mfe_catboost_v2_ablation_20260725_v1"
)
NATIVE_CATEGORICAL = (
    "archetype_label_family",
    "archetype_policy_key",
    "local_side_archetype",
)
GEOMETRIES: tuple[dict[str, Any], ...] = (
    {
        "depth": 4,
        "l2_leaf_reg": 8.0,
        "random_strength": 0.5,
        "bagging_temperature": 0.5,
        "rsm": 0.85,
    },
    {
        "depth": 6,
        "l2_leaf_reg": 12.0,
        "random_strength": 1.0,
        "bagging_temperature": 1.0,
        "rsm": 0.75,
    },
    {
        "depth": 8,
        "l2_leaf_reg": 20.0,
        "random_strength": 1.5,
        "bagging_temperature": 1.0,
        "rsm": 0.70,
    },
    {
        "grow_policy": "Lossguide",
        "depth": 8,
        "max_leaves": 31,
        "min_data_in_leaf": 150,
        "l2_leaf_reg": 12.0,
        "random_strength": 1.0,
        "bagging_temperature": 0.5,
        "rsm": 0.75,
    },
)


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fit_catboost(
    task: str,
    X: pd.DataFrame,
    target: np.ndarray,
    params: Mapping[str, Any],
    *,
    seed: int,
    sample_weight: np.ndarray | None = None,
    eval_set: tuple[pd.DataFrame, np.ndarray] | None = None,
    cat_features: Sequence[str] = (),
) -> Any:
    common = {
        "iterations": int(params.get("iterations", 700)),
        "learning_rate": float(params.get("learning_rate", 0.03)),
        "random_seed": int(seed),
        "thread_count": 6,
        "verbose": False,
        "allow_writing_files": False,
        "bootstrap_type": "Bayesian",
        **{
            key: value
            for key, value in params.items()
            if key not in {"iterations", "learning_rate"}
        },
    }
    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    if eval_set is not None:
        fit_kwargs.update(
            {
                "eval_set": eval_set,
                "early_stopping_rounds": 60,
                "use_best_model": True,
            }
        )
    if cat_features:
        fit_kwargs["cat_features"] = list(cat_features)
    if task == "binary":
        model = CatBoostClassifier(loss_function="Logloss", **common)
    elif task == "soft":
        model = CatBoostClassifier(loss_function="CrossEntropy", **common)
    elif task == "multiclass":
        model = CatBoostClassifier(loss_function="MultiClass", **common)
    elif task == "quality":
        model = CatBoostRegressor(loss_function="RMSE", **common)
    else:
        raise ValueError(f"unsupported CatBoost task: {task}")
    model.fit(X, target, **fit_kwargs)
    return model


def _predict(model: Any, task: str, X: pd.DataFrame) -> np.ndarray:
    if task in {"binary", "soft"}:
        return np.clip(model.predict_proba(X)[:, 1], 1e-6, 1.0 - 1e-6)
    if task == "quality":
        return np.clip(np.asarray(model.predict(X), dtype=np.float64), 0.0, 1.0)
    if task == "multiclass":
        probabilities = np.asarray(model.predict_proba(X), dtype=np.float64)
        classes = list(map(int, model.classes_))
        return probabilities[:, [classes.index(index) for index in (0, 1, 2)]]
    raise ValueError(f"unsupported CatBoost task: {task}")


def _objective(
    hard: np.ndarray,
    soft: np.ndarray,
    prediction: np.ndarray,
    *,
    task: str,
) -> tuple[float, dict[str, Any]]:
    metrics = _classification_metrics(hard, soft, prediction)
    if task == "binary":
        value = (
            metrics["log_loss_hard"]
            + metrics["brier_hard"]
            - 0.20 * metrics["roc_auc"]
            - 0.05 * metrics["top10_precision"]
        )
    else:
        value = (
            metrics["brier_soft"]
            - 0.10 * metrics["spearman_soft"]
            - 0.03 * metrics["top10_precision"]
        )
    return float(value), metrics


def _univariate_prescreen(
    X: pd.DataFrame,
    target: np.ndarray,
    *,
    mandatory: Sequence[str],
    limit: int = 120,
) -> list[str]:
    values = X.to_numpy(np.float32, copy=False)
    y = np.asarray(target, dtype=np.float64)
    y_centered = y - np.nanmean(y)
    scores: list[tuple[float, str]] = []
    for position, column in enumerate(X.columns):
        x = values[:, position].astype(np.float64, copy=False)
        finite = np.isfinite(x) & np.isfinite(y_centered)
        if finite.sum() < 200:
            score = 0.0
        else:
            local = x[finite]
            local = np.where(np.isfinite(local), local, np.nan)
            local = local - np.nanmean(local)
            denominator = np.sqrt(
                np.nansum(local * local)
                * np.nansum(y_centered[finite] * y_centered[finite])
            )
            score = (
                abs(float(np.nansum(local * y_centered[finite]) / denominator))
                if denominator > 0.0
                else 0.0
            )
        scores.append((score, str(column)))
    ranked = [column for _, column in sorted(scores, reverse=True)]
    return list(
        dict.fromkeys(
            [
                *[column for column in mandatory if column in X],
                *ranked[: max(1, int(limit))],
            ]
        )
    )


def _task_feature_sets(
    X: pd.DataFrame,
    target: np.ndarray,
    *,
    frozen: Sequence[str],
    mandatory: Sequence[str],
    seed: int,
    task: str,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    prescreen = _univariate_prescreen(X, target, mandatory=mandatory)
    selector_task = "binary" if task == "binary" else "quality"
    model = _fit_catboost(
        selector_task,
        X[prescreen],
        target,
        {
            "iterations": 300,
            "learning_rate": 0.04,
            "depth": 6,
            "l2_leaf_reg": 12.0,
            "random_strength": 1.0,
            "bagging_temperature": 1.0,
            "rsm": 0.80,
        },
        seed=seed,
    )
    importance = sorted(
        zip(model.get_feature_importance(), prescreen),
        reverse=True,
    )

    def top(count: int) -> list[str]:
        return list(
            dict.fromkeys(
                [
                    *[column for column in mandatory if column in X],
                    *[column for _, column in importance[:count]],
                ]
            )
        )

    feature_sets = {
        "frozen": [column for column in frozen if column in X],
        "task_top40": top(40),
        "task_top80": top(80),
    }
    return feature_sets, {
        "prescreen_rows": int(len(X)),
        "prescreen_features": int(len(prescreen)),
        "importance": [
            {"feature": column, "importance": float(value)}
            for value, column in importance
        ],
        "feature_sets": feature_sets,
    }


def _load_matrix(
    labels: pd.DataFrame,
    *,
    context_path: Path,
    feature_dir: Path,
    selection_path: Path,
    archetype_contract_override: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    labels, context_report = _join_archetype_context(
        labels, context_path, labels_are_canonical_top40=False
    )
    payload = joblib.load(selection_path)
    contracts = payload["selection_contracts"]
    pool_by_side = {
        side: list(
            dict.fromkeys(
                feature
                for contract in contracts.values()
                for feature in contract["selected_features_by_side"][side]
            )
        )
        for side in SIDES
    }
    frozen_by_side = contracts["meaningful_mfe_event"]["selected_features_by_side"]
    mandatory_by_side = contracts["meaningful_mfe_event"]["mandatory_features_by_side"]
    union = list(
        dict.fromkeys(
            feature for side in SIDES for feature in pool_by_side[side]
        )
    )
    complete_archetypes = [
        column
        for column in ARCHETYPE_COLUMNS
        if _complete_archetype_source(labels, column)
    ]
    if archetype_contract_override is None:
        reference = labels.loc[
            labels["__ts__"].lt(pd.Timestamp("2026-04-22T00:00:00Z"))
            & labels[DEFAULT_LABEL_RESOLUTION_COLUMN].lt(
                pd.Timestamp("2026-04-22T00:00:00Z")
            )
        ]
        archetype_contract = fit_base_archetype_label_feature_contract(
            reference,
            source_columns=complete_archetypes,
            canonical_source=complete_archetypes[0],
        )
    else:
        archetype_contract = dict(archetype_contract_override)
    archetype = transform_base_archetype_label_features(labels, archetype_contract)
    matrix, load_report = _load_static_features(
        labels,
        feature_dir=feature_dir,
        requested_features=union,
        read_cache=None,
    )
    matrix, load_report = _overlay_handoff_model_features(
        matrix,
        labels,
        requested_features=union,
        static_report=load_report,
        handoff_feature_columns=context_report["handoff_model_feature_columns"],
    )
    generated = [column for column in union if column in archetype]
    if generated:
        matrix.loc[:, generated] = archetype.loc[:, generated].to_numpy(
            dtype=np.float32, copy=False
        )
    matrix = matrix.reindex(columns=union).astype(np.float32)
    unavailable = [column for column in union if matrix[column].isna().all()]
    if unavailable:
        for side in SIDES:
            pool_by_side[side] = [
                column for column in pool_by_side[side] if column not in unavailable
            ]
            frozen_by_side[side] = [
                column for column in frozen_by_side[side] if column not in unavailable
            ]
            mandatory_by_side[side] = [
                column for column in mandatory_by_side[side] if column not in unavailable
            ]
        matrix = matrix.drop(columns=unavailable)
    return labels, matrix, {
        "pool_by_side": pool_by_side,
        "frozen_by_side": frozen_by_side,
        "mandatory_by_side": mandatory_by_side,
        "archetype_contract": archetype_contract,
        "feature_load": load_report,
        "selection_sha256": _sha256(selection_path),
        "unavailable_features": unavailable,
    }


def _april_indices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    start = pd.Timestamp("2026-04-22T00:00:00Z")
    stop = pd.Timestamp("2026-05-01T00:00:00Z")
    train = np.flatnonzero(
        frame["__ts__"].lt(start).to_numpy()
        & frame[DEFAULT_LABEL_RESOLUTION_COLUMN].lt(start).to_numpy()
    )
    valid = np.flatnonzero(
        frame["__ts__"].ge(start).to_numpy()
        & frame["__ts__"].lt(stop).to_numpy()
    )
    return train, valid


def _hpo_task(
    task: str,
    X: pd.DataFrame,
    target: np.ndarray,
    hard: np.ndarray,
    soft: np.ndarray,
    frame: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
    *,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    train, valid = _april_indices(frame)
    trials: list[dict[str, Any]] = []
    predictions: list[np.ndarray] = []
    for feature_name, features in feature_sets.items():
        for geometry_index, geometry in enumerate(GEOMETRIES):
            model = _fit_catboost(
                task,
                X.iloc[train][list(features)],
                target[train],
                geometry,
                seed=seed + 100 * geometry_index + len(trials),
                eval_set=(X.iloc[valid][list(features)], target[valid]),
            )
            prediction = _predict(model, task, X.iloc[valid][list(features)])
            value, metrics = _objective(
                hard[valid], soft[valid], prediction, task=task
            )
            best_iteration = max(100, int(model.get_best_iteration()) + 1)
            trials.append(
                {
                    "feature_set": feature_name,
                    "features": list(features),
                    "geometry": dict(geometry),
                    "best_iteration": best_iteration,
                    "objective": value,
                    **metrics,
                }
            )
            predictions.append(prediction)
    winner_index = min(
        range(len(trials)),
        key=lambda index: (
            trials[index]["objective"],
            trials[index]["feature_set"],
            str(trials[index]["geometry"]),
        ),
    )
    winner = dict(trials[winner_index])
    winner["geometry"] = {
        **winner["geometry"],
        "iterations": winner["best_iteration"],
    }
    return {
        "train_rows": int(len(train)),
        "validation_rows": int(len(valid)),
        "trials": trials,
        "winner": winner,
    }, predictions[winner_index]


def _native_matrix(
    numeric: pd.DataFrame,
    frame: pd.DataFrame,
    features: Sequence[str],
) -> pd.DataFrame:
    output = numeric[list(features)].copy()
    for column in NATIVE_CATEGORICAL:
        output[f"__cat_{column}"] = (
            frame[column].fillna("unknown").astype(str).to_numpy()
        )
    return output


def _rolling_platt(
    prediction: np.ndarray,
    hard: np.ndarray,
    frame: pd.DataFrame,
    *,
    side: str,
    april_prediction: np.ndarray,
    april_hard: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    calibrated = np.full(len(prediction), np.nan, dtype=np.float64)
    reference_p = np.asarray(april_prediction, dtype=np.float64)
    reference_y = np.asarray(april_hard, dtype=np.float64)
    reports = []
    for month in ("2026-05", "2026-06", "2026-07"):
        mask = (
            frame["side_name"].eq(side)
            & frame["__ts__"].dt.strftime("%Y-%m").eq(month)
            & np.isfinite(prediction)
        ).to_numpy()
        model = LogisticRegression(C=1.0, solver="lbfgs")
        logits = np.log(
            np.clip(reference_p, 1e-6, 1 - 1e-6)
            / np.clip(1.0 - reference_p, 1e-6, 1 - 1e-6)
        ).reshape(-1, 1)
        model.fit(logits, reference_y)
        current = prediction[mask]
        current_logits = np.log(
            np.clip(current, 1e-6, 1 - 1e-6)
            / np.clip(1.0 - current, 1e-6, 1 - 1e-6)
        ).reshape(-1, 1)
        calibrated[mask] = model.predict_proba(current_logits)[:, 1]
        reports.append(
            {
                "side": side,
                "month": month,
                "calibration_rows": int(len(reference_y)),
                "coefficient": float(model.coef_[0, 0]),
                "intercept": float(model.intercept_[0]),
            }
        )
        reference_p = np.r_[reference_p, current]
        reference_y = np.r_[reference_y, hard[mask]]
    return calibrated, reports


def run(
    *,
    labels_path: Path = DEFAULT_LABELS,
    context_path: Path = DEFAULT_CONTEXT,
    feature_dir: Path = DEFAULT_FEATURE_DIR,
    selection_path: Path = DEFAULT_SELECTION,
    incumbent_path: Path = DEFAULT_INCUMBENT,
    exact_policy_path: Path = DEFAULT_EXACT_POLICY,
    v1_predictions_path: Path = DEFAULT_V1,
    output_dir: Path = DEFAULT_OUTPUT,
    seed: int = 20260725,
    training_target: str = "clean",
) -> dict[str, Any]:
    if training_target not in {"clean", "literal"}:
        raise ValueError("training_target must be clean or literal")
    required_sources = {
        "labels": labels_path,
        "context": context_path,
        "selection": selection_path,
        "incumbent": incumbent_path,
        "v1_predictions": v1_predictions_path,
        "exact_policy": exact_policy_path,
    }
    missing_sources = {
        name: str(path)
        for name, path in required_sources.items()
        if not path.is_file()
    }
    if missing_sources:
        raise FileNotFoundError(
            "missing required ablation sources before model fitting: "
            + json.dumps(missing_sources, sort_keys=True)
        )
    if not feature_dir.is_dir():
        raise FileNotFoundError(
            f"missing required feature directory before model fitting: {feature_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    labels, label_report = _load_labels(
        labels_path,
        label_resolution_column=DEFAULT_LABEL_RESOLUTION_COLUMN,
    )
    if "__path_auxiliary_atr_fraction__" not in labels:
        atr = pd.read_parquet(
            labels_path,
            columns=[
                "__ts__",
                "__symbol__",
                "side",
                "candidate_id",
                "__path_auxiliary_atr_fraction__",
            ],
        )
        atr["__ts__"] = pd.to_datetime(atr["__ts__"], utc=True, errors="raise")
        labels = labels.merge(
            atr,
            on=["__ts__", "__symbol__", "side", "candidate_id"],
            how="left",
            validate="one_to_one",
        )
    labels["side_name"] = labels["side"].astype(str)
    labels, matrix, feature_payload = _load_matrix(
        labels,
        context_path=context_path,
        feature_dir=feature_dir,
        selection_path=selection_path,
    )
    labels["side_name"] = labels["side"].astype(str)
    valid = labels["__path_auxiliary_target_valid__"].eq(1).to_numpy()
    labels = labels.loc[valid].reset_index(drop=True)
    matrix = matrix.loc[valid].reset_index(drop=True)
    baseline = atr_soft_triple_barrier_labels(labels, TripleBarrierSoftLabel())
    baseline["meaningful_mfe_reached"] = labels[
        "__meaningful_mfe_reached_12h__"
    ].to_numpy(np.float32)
    risks = competing_risk_targets(baseline)
    clean_hard = risks["favorable_first"].to_numpy(np.float32)
    meaningful = baseline["meaningful_mfe_reached"].to_numpy(np.float32)
    hard = clean_hard if training_target == "clean" else meaningful
    soft = baseline["tb_soft_label"].to_numpy(np.float32)

    feature_selection: dict[str, Any] = {}
    hpo: dict[str, Any] = {}
    april_predictions: dict[str, np.ndarray] = {}
    winners: dict[str, dict[str, Any]] = {}
    for side_index, side in enumerate(SIDES):
        positions = np.flatnonzero(labels["side_name"].eq(side).to_numpy())
        local_frame = labels.iloc[positions].reset_index(drop=True)
        local_X = matrix.iloc[positions][
            feature_payload["pool_by_side"][side]
        ].reset_index(drop=True)
        train, valid_april = _april_indices(local_frame)
        feature_selection[side] = {}
        hard_sets, hard_report = _task_feature_sets(
            local_X.iloc[train],
            hard[positions][train],
            frozen=feature_payload["frozen_by_side"][side],
            mandatory=feature_payload["mandatory_by_side"][side],
            seed=seed + 1000 * side_index,
            task="binary",
        )
        soft_sets, soft_report = _task_feature_sets(
            local_X.iloc[train],
            soft[positions][train],
            frozen=feature_payload["frozen_by_side"][side],
            mandatory=feature_payload["mandatory_by_side"][side],
            seed=seed + 1000 * side_index + 100,
            task="soft",
        )
        feature_selection[side]["hard"] = hard_report
        feature_selection[side]["soft"] = soft_report
        hpo[side] = {}
        hpo[side]["hard"], april_predictions[f"{side}__hard"] = _hpo_task(
            "binary",
            local_X,
            hard[positions],
            hard[positions],
            (
                soft[positions]
                if training_target == "clean"
                else hard[positions]
            ),
            local_frame,
            hard_sets,
            seed=seed + 10_000 * side_index,
        )
        hpo[side]["soft"], april_predictions[f"{side}__soft"] = _hpo_task(
            "soft",
            local_X,
            soft[positions],
            clean_hard[positions],
            soft[positions],
            local_frame,
            soft_sets,
            seed=seed + 10_000 * side_index + 1000,
        )
        winners[side] = {
            "hard": hpo[side]["hard"]["winner"],
            "soft": hpo[side]["soft"]["winner"],
            "april_hard": hard[positions][valid_april],
        }
        print(
            f"selected side={side} hard={winners[side]['hard']['feature_set']} "
            f"soft={winners[side]['soft']['feature_set']}",
            flush=True,
        )

    components = {
        name: np.full(len(labels), np.nan, dtype=np.float64)
        for name in (
            "catboost_hard_single",
            "catboost_hard_ensemble",
            "catboost_hard_lcb",
            "catboost_hard_ambiguity_weighted",
            "catboost_soft_unconditional",
            "catboost_competing_p_favorable",
            "catboost_competing_net_probability",
            "catboost_conditional_quality",
            "catboost_hard_native_context",
        )
    }
    fold_reports: list[dict[str, Any]] = []
    for side_index, side in enumerate(SIDES):
        positions = np.flatnonzero(labels["side_name"].eq(side).to_numpy())
        local_frame = labels.iloc[positions].reset_index(drop=True)
        local_X = matrix.iloc[positions].reset_index(drop=True)
        local_hard = hard[positions]
        local_soft = soft[positions]
        local_risk = risks.iloc[positions].reset_index(drop=True)
        hard_features = winners[side]["hard"]["features"]
        hard_params = winners[side]["hard"]["geometry"]
        soft_features = winners[side]["soft"]["features"]
        soft_params = winners[side]["soft"]["geometry"]
        for fold in expanding_resolved_month_folds(
            local_frame["__ts__"],
            local_frame[DEFAULT_LABEL_RESOLUTION_COLUMN],
        ):
            train = np.asarray(fold["train_indices"], dtype=np.int64)
            validation = np.asarray(fold["validation_indices"], dtype=np.int64)
            global_validation = positions[validation]
            hard_seed_predictions = []
            for seed_index in range(3):
                model = _fit_catboost(
                    "binary",
                    local_X.iloc[train][hard_features],
                    local_hard[train],
                    hard_params,
                    seed=seed
                    + 100_000 * side_index
                    + 1000 * int(fold["fold"])
                    + seed_index,
                )
                hard_seed_predictions.append(
                    _predict(
                        model,
                        "binary",
                        local_X.iloc[validation][hard_features],
                    )
                )
            seed_matrix = np.vstack(hard_seed_predictions)
            hard_mean = seed_matrix.mean(axis=0)
            components["catboost_hard_single"][global_validation] = seed_matrix[0]
            components["catboost_hard_ensemble"][global_validation] = hard_mean
            components["catboost_hard_lcb"][global_validation] = np.clip(
                hard_mean - seed_matrix.std(axis=0), 0.0, 1.0
            )

            ambiguity_weights = np.where(
                local_risk["order_ambiguous"].to_numpy()[train], 0.35, 1.0
            )
            ambiguity_model = _fit_catboost(
                "binary",
                local_X.iloc[train][hard_features],
                local_hard[train],
                hard_params,
                seed=seed + 200_000 + 1000 * side_index + int(fold["fold"]),
                sample_weight=ambiguity_weights,
            )
            components["catboost_hard_ambiguity_weighted"][
                global_validation
            ] = _predict(
                ambiguity_model,
                "binary",
                local_X.iloc[validation][hard_features],
            )

            soft_model = _fit_catboost(
                "soft",
                local_X.iloc[train][soft_features],
                local_soft[train],
                soft_params,
                seed=seed + 300_000 + 1000 * side_index + int(fold["fold"]),
            )
            components["catboost_soft_unconditional"][global_validation] = _predict(
                soft_model,
                "soft",
                local_X.iloc[validation][soft_features],
            )

            competing_model = _fit_catboost(
                "multiclass",
                local_X.iloc[train][hard_features],
                local_risk["risk_class"].to_numpy(np.int8)[train],
                hard_params,
                seed=seed + 400_000 + 1000 * side_index + int(fold["fold"]),
            )
            competing = _predict(
                competing_model,
                "multiclass",
                local_X.iloc[validation][hard_features],
            )
            components["catboost_competing_p_favorable"][
                global_validation
            ] = competing[:, 2]
            components["catboost_competing_net_probability"][
                global_validation
            ] = np.clip((1.0 + competing[:, 2] - competing[:, 1]) / 2.0, 0.0, 1.0)

            favorable_train = train[
                local_risk["favorable_first"].to_numpy()[train] > 0.5
            ]
            quality_model = _fit_catboost(
                "quality",
                local_X.iloc[favorable_train][soft_features],
                local_risk["conditional_quality"].to_numpy(np.float32)[
                    favorable_train
                ],
                soft_params,
                seed=seed + 500_000 + 1000 * side_index + int(fold["fold"]),
            )
            components["catboost_conditional_quality"][
                global_validation
            ] = _predict(
                quality_model,
                "quality",
                local_X.iloc[validation][soft_features],
            )

            native_train = _native_matrix(
                local_X.iloc[train], local_frame.iloc[train], hard_features
            )
            native_valid = _native_matrix(
                local_X.iloc[validation], local_frame.iloc[validation], hard_features
            )
            native_columns = [
                column for column in native_train if column.startswith("__cat_")
            ]
            native_model = _fit_catboost(
                "binary",
                native_train,
                local_hard[train],
                hard_params,
                seed=seed + 600_000 + 1000 * side_index + int(fold["fold"]),
                cat_features=native_columns,
            )
            components["catboost_hard_native_context"][
                global_validation
            ] = _predict(native_model, "binary", native_valid)
            fold_reports.append(
                {
                    "side": side,
                    "month": fold["month"],
                    "train_rows": int(len(train)),
                    "validation_rows": int(len(validation)),
                    "training_label_resolved_max": fold[
                        "training_label_resolved_max"
                    ],
                }
            )
            print(
                f"oof side={side} month={fold['month']} rows={len(validation)}",
                flush=True,
            )

    calibration_reports = []
    calibrated = np.full(len(labels), np.nan, dtype=np.float64)
    for side in SIDES:
        side_calibrated, reports = _rolling_platt(
            components["catboost_hard_ensemble"],
            hard,
            labels,
            side=side,
            april_prediction=april_predictions[f"{side}__hard"],
            april_hard=winners[side]["april_hard"],
        )
        finite = np.isfinite(side_calibrated)
        calibrated[finite] = side_calibrated[finite]
        calibration_reports.extend(reports)
    components["catboost_hard_ensemble_platt"] = calibrated
    composed = event_quality_scores(
        calibrated,
        components["catboost_conditional_quality"],
    )
    components["catboost_probability_x_quality"] = composed[
        "probability_x_quality"
    ]
    components["catboost_probability_gated_quality"] = composed[
        "probability_gated_quality"
    ]

    v1 = pd.read_parquet(v1_predictions_path)
    comparators = labels.loc[:, list(IDENTITY)].merge(
        v1.loc[
            :,
            [
                *IDENTITY,
                "catboost_soft_tb_baseline",
                "incumbent_hard_meaningful",
            ],
        ],
        on=list(IDENTITY),
        how="left",
        validate="one_to_one",
    )
    components["v1_catboost_soft_tb"] = pd.to_numeric(
        comparators["catboost_soft_tb_baseline"], errors="coerce"
    ).to_numpy(np.float64)
    components["incumbent_literal_event"] = pd.to_numeric(
        comparators["incumbent_hard_meaningful"], errors="coerce"
    ).to_numpy(np.float64)

    predictions = labels.loc[:, list(IDENTITY)].copy()
    predictions["tb_hard_label"] = clean_hard
    predictions["tb_soft_label"] = soft
    predictions["meaningful_mfe_reached"] = baseline[
        "meaningful_mfe_reached"
    ].to_numpy(np.float32)
    predictions["risk_class"] = risks["risk_class"].to_numpy(np.int8)
    predictions["order_ambiguous"] = risks["order_ambiguous"].to_numpy(bool)
    for name, values in components.items():
        predictions[name] = values

    exact = pd.read_parquet(exact_policy_path)
    paired = exact.merge(
        predictions,
        on=list(IDENTITY),
        how="inner",
        validate="one_to_one",
    )
    reports: dict[str, Any] = {}
    meaningful = predictions["meaningful_mfe_reached"].to_numpy(np.float32)
    for name, values in components.items():
        finite = np.isfinite(values)
        report = {
            "oof_rows": int(finite.sum()),
            "clean_event": _classification_metrics(
                clean_hard[finite], soft[finite], values[finite]
            ),
            "literal_event": _classification_metrics(
                meaningful[finite], meaningful[finite], values[finite]
            ),
            "clean_event_by_side_month": [],
        }
        for (side, month), group in predictions.loc[finite].groupby(
            [
                "side_name",
                predictions.loc[finite, "__ts__"].dt.strftime("%Y-%m"),
            ]
        ):
            local = group[name].to_numpy(np.float64)
            report["clean_event_by_side_month"].append(
                {
                    "side": side,
                    "month": month,
                    **_classification_metrics(
                        group["tb_hard_label"].to_numpy(np.float32),
                        group["tb_soft_label"].to_numpy(np.float32),
                        local,
                    ),
                }
            )
        paired_score = pd.to_numeric(paired[name], errors="coerce").to_numpy(
            np.float64
        )
        paired_finite = np.isfinite(paired_score)
        report["exact_policy"] = economic_metrics(
            paired.loc[paired_finite].reset_index(drop=True),
            paired_score[paired_finite],
        )
        admission = first_21d_admission(
            paired["__ts__"],
            paired_score,
            pd.to_numeric(
                paired["execution_net_ev_12h"], errors="coerce"
            ).to_numpy(np.float64),
        )
        evaluation = np.asarray(admission["evaluation_mask"], dtype=bool)
        admitted = np.asarray(admission["admitted_mask"], dtype=bool)
        report["post_21d_admission"] = {
            "contract": {
                key: value
                for key, value in admission.items()
                if key
                not in {
                    "evaluation_mask",
                    "admitted_mask",
                    "calibrated_expected_net_return",
                }
            },
            "raw_after_fit_window": economic_metrics(
                paired.loc[evaluation].reset_index(drop=True),
                paired_score[evaluation],
            ),
            "admitted": economic_metrics(
                paired,
                paired_score,
                admitted=admitted,
            ),
        }
        report["exact_policy_by_side_month"] = [
            {
                "side": side,
                "month": month,
                **economic_metrics(
                    group.reset_index(drop=True),
                    group[name].to_numpy(np.float64),
                ),
            }
            for (side, month), group in paired.loc[paired_finite].groupby(
                [
                    "side_name",
                    paired.loc[paired_finite, "__ts__"].dt.strftime("%Y-%m"),
                ]
            )
        ]
        report["conditional_quality_ic_on_favorable"] = float(
            spearmanr(
                values[finite & (clean_hard > 0.5)],
                risks.loc[
                    finite & (clean_hard > 0.5), "conditional_quality"
                ],
            ).statistic
        )
        reports[name] = report

    output_predictions = output_dir / "oof_predictions.parquet"
    predictions.to_parquet(output_predictions, index=False)
    output_paired = output_dir / "exact_policy_paired.parquet"
    paired.to_parquet(output_paired, index=False)
    summary = {
        "schema": SCHEMA,
        "training_target": training_target,
        "status": "research_oof_not_untouched_final_test",
        "chronology": {
            "feature_selection_and_hpo": (
                "train labels resolved before 2026-04-22; "
                "validate 2026-04-22 through 2026-04-30"
            ),
            "outer_oof": (
                "expanding May, June, and July folds; every training label "
                "resolves strictly before validation"
            ),
            "calibration": (
                "May uses April held-out predictions; June uses April+May OOF; "
                "July uses April+May+June OOF"
            ),
            "final_test_disclosure": (
                "July 1-10 was previously inspected and is not untouched"
            ),
        },
        "feature_contract": {
            **feature_payload,
            "selection": feature_selection,
            "selection_scope": (
                "side-local task-specific screen and CatBoost importance on "
                "pre-April-22 training rows only"
            ),
            "mandatory_handoff_features": list(MANDATORY_HANDOFF_MODEL_FEATURES),
            "representation_feature": REPRESENTATION_AVAILABLE_FEATURE,
        },
        "hpo": hpo,
        "winners": winners,
        "calibration": calibration_reports,
        "folds": fold_reports,
        "reports": reports,
        "rows": {
            "valid_labels": int(len(labels)),
            "outer_oof": int(
                np.isfinite(components["catboost_hard_ensemble"]).sum()
            ),
            "exact_policy_paired": int(len(paired)),
            "order_ambiguous": int(risks["order_ambiguous"].sum()),
        },
        "unsupported_ablation": (
            "alternative upper barriers and shorter timeouts require exact "
            "intrabar path/order targets not present in the current store"
        ),
        "sources": {
            "labels": {
                "path": str(labels_path),
                "sha256": _file_sha256(labels_path),
                "report": label_report,
            },
            "context": {"path": str(context_path), "sha256": _sha256(context_path)},
            "selection": {
                "path": str(selection_path),
                "sha256": _sha256(selection_path),
            },
            "incumbent": {
                "path": str(incumbent_path),
                "sha256": _sha256(incumbent_path),
            },
            "v1_predictions": {
                "path": str(v1_predictions_path),
                "sha256": _sha256(v1_predictions_path),
            },
            "exact_policy": {
                "path": str(exact_policy_path),
                "sha256": _sha256(exact_policy_path),
            },
        },
        "outputs": {
            "predictions": str(output_predictions),
            "exact_policy_paired": str(output_paired),
        },
    }
    # The April hard labels are arrays used only for causal calibration, not
    # part of the serializable winner contract.
    for side in SIDES:
        summary["winners"][side] = {
            key: value
            for key, value in summary["winners"][side].items()
            if key != "april_hard"
        }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--context-path", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--selection-path", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--incumbent-path", type=Path, default=DEFAULT_INCUMBENT)
    parser.add_argument("--exact-policy-path", type=Path, default=DEFAULT_EXACT_POLICY)
    parser.add_argument("--v1-predictions-path", type=Path, default=DEFAULT_V1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--training-target", choices=("clean", "literal"), default="clean"
    )
    args = parser.parse_args(argv)
    summary = run(**vars(args))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "rows": summary["rows"],
                "output": str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
