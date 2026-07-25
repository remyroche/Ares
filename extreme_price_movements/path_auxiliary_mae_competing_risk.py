"""Side-local competing-risk challenger for pre-MFE adverse excursion."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.path_auxiliary_lgbm import auxiliary_hpo_sample_indices
from extreme_price_movements.path_auxiliary_role_training import (
    FIXED_MAY_JULY_OOF_MONTHS,
    _binary_metrics,
    _bounded_n_jobs,
    _build_purged_role_folds,
    _fit_with_inner_validation,
    _fixed_calendar_oof_folds,
    _model_sha256,
    _predict_role_model,
    _require_utc_cutoff,
    _sanitize_params,
    _suggest_params,
    _utc_series,
)
from extreme_price_movements.path_auxiliary_timing_training import (
    _as_mask,
    _normalized_side_values,
)

MAE_COMPETING_RISK_SCHEMA = "path_auxiliary_mae_competing_risk_side_local_v1_strict_oof"
RISK_CLASS_NAMES: tuple[str, ...] = (
    "favorable_before_0_5r",
    "adverse_0_5r_before_mfe",
    "neither_before_horizon",
)


def build_mae_competing_risk_targets(
    mae_atr: Sequence[Any],
    meaningful_hit: Sequence[Any],
    sides: Sequence[Any],
    train_mask: Sequence[Any],
    *,
    stop_atr_by_side: Mapping[str, float],
) -> dict[str, np.ndarray]:
    """Build mutually exclusive first-outcome and conditional stop targets."""

    mae = pd.to_numeric(pd.Series(mae_atr), errors="coerce").to_numpy(np.float32)
    hit = pd.to_numeric(pd.Series(meaningful_hit), errors="coerce").to_numpy(np.float32)
    rows = len(mae)
    mask = _as_mask(train_mask, name="train_mask", rows=rows)
    side_values = _normalized_side_values(sides, rows=rows)
    if len(hit) != rows:
        raise ValueError("meaningful_hit must align to mae_atr")
    if np.any(mask & (~np.isfinite(mae) | (mae < 0.0))):
        raise ValueError("mae_atr must be finite and non-negative on train rows")
    if np.any(mask & (~np.isfinite(hit) | ~np.isin(hit, (0.0, 1.0)))):
        raise ValueError("meaningful_hit must be binary on train rows")
    stop = np.asarray(
        [float(stop_atr_by_side.get(side, np.nan)) for side in side_values],
        dtype=np.float32,
    )
    if np.any(mask & (~np.isfinite(stop) | (stop <= 0.0))):
        raise ValueError("every train row requires a finite positive side-local stop")
    adverse = mae >= 0.5 * stop
    stopped = mae >= stop
    favorable = (hit > 0.5) & ~adverse
    neither = ~adverse & ~favorable
    risk_class = np.full(rows, -1, dtype=np.int8)
    risk_class[favorable] = 0
    risk_class[adverse] = 1
    risk_class[neither] = 2
    risk_class[~mask] = -1
    stop_if_adverse = np.full(rows, np.nan, dtype=np.float32)
    stop_if_adverse[mask & adverse] = stopped[mask & adverse].astype(np.float32)
    return {
        "risk_class": risk_class,
        "adverse_0_5r": adverse.astype(np.float32),
        "stop_1r": stopped.astype(np.float32),
        "stop_if_adverse": stop_if_adverse,
        "stop_atr": stop,
    }


def _union_features(
    selected: Mapping[str, Sequence[str] | Mapping[str, Sequence[str]]],
    columns: pd.Index,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for raw_side, value in selected.items():
        side = str(raw_side).strip().lower()
        streams = value.values() if isinstance(value, Mapping) else (value,)
        features = list(
            dict.fromkeys(str(item) for stream in streams for item in stream)
        )
        if not features:
            raise ValueError(f"empty MAE competing-risk features for {side}")
        missing = [feature for feature in features if feature not in columns]
        if missing:
            raise ValueError(
                f"MAE competing-risk features missing for {side}: {missing[:20]}"
            )
        result[side] = features
    return result


def _multiclass_params(trial: Any, *, random_state: int, n_jobs: int) -> dict[str, Any]:
    params = _suggest_params(
        trial,
        task_kind="binary",
        quantile_alpha=0.8,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    params.update({"objective": "multiclass", "num_class": len(RISK_CLASS_NAMES)})
    return params


def _fit_multiclass(
    train_X: pd.DataFrame,
    train_y: np.ndarray,
    valid_X: pd.DataFrame | None,
    valid_y: np.ndarray | None,
    params: Mapping[str, Any],
) -> tuple[Any, int]:
    import lightgbm as lgb

    model = lgb.LGBMClassifier(**dict(params))
    kwargs: dict[str, Any] = {}
    if valid_X is not None and valid_y is not None:
        kwargs["eval_set"] = [(valid_X, valid_y)]
        kwargs["callbacks"] = [lgb.early_stopping(75, verbose=False)]
    model.fit(train_X, train_y, **kwargs)
    return model, int(model.best_iteration_ or params.get("n_estimators", 1))


def _predict_multiclass(model: Any, matrix: pd.DataFrame) -> np.ndarray:
    probability = np.asarray(model.predict_proba(matrix), dtype=np.float32)
    if probability.ndim != 2 or probability.shape[1] != len(RISK_CLASS_NAMES):
        raise ValueError(
            "competing-risk model did not return the canonical class vector"
        )
    return probability


def _multiclass_score(observed: np.ndarray, probability: np.ndarray) -> float:
    clipped = np.clip(probability, 1e-7, 1.0)
    logloss = -float(np.mean(np.log(clipped[np.arange(len(observed)), observed])))
    one_hot = np.eye(len(RISK_CLASS_NAMES), dtype=np.float32)[observed]
    brier = float(np.mean(np.sum((probability - one_hot) ** 2, axis=1)))
    return -logloss - 0.25 * brier


def _hpo_multiclass(
    matrix: pd.DataFrame,
    target: np.ndarray,
    base_rows: np.ndarray,
    decision: pd.Series,
    resolved: pd.Series,
    *,
    n_trials: int,
    hpo_rows: int,
    random_state: int,
    n_jobs: int,
    side: str,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sampled = auxiliary_hpo_sample_indices(
        decision.iloc[base_rows].to_numpy(),
        max_rows=max(1, int(hpo_rows)),
        random_state=random_state,
    ).astype(np.int32)
    hpo_base = base_rows[sampled]
    inner = _build_purged_role_folds(
        decision.iloc[hpo_base].to_numpy(),
        resolved.iloc[hpo_base].to_numpy(),
        purge_hours=13.0,
    )
    import optuna

    iterations: dict[int, list[int]] = {}

    def objective(trial: Any) -> float:
        params = _multiclass_params(trial, random_state=random_state, n_jobs=n_jobs)
        observed: list[np.ndarray] = []
        predicted: list[np.ndarray] = []
        trial_iterations: list[int] = []
        for fold in inner:
            train_idx = hpo_base[fold.train_idx]
            valid_idx = hpo_base[fold.valid_idx]
            model, best_iteration = _fit_multiclass(
                matrix.iloc[train_idx],
                target[train_idx],
                matrix.iloc[valid_idx],
                target[valid_idx],
                params,
            )
            observed.append(target[valid_idx])
            predicted.append(_predict_multiclass(model, matrix.iloc[valid_idx]))
            trial_iterations.append(best_iteration)
        iterations[int(trial.number)] = trial_iterations
        score = _multiclass_score(np.concatenate(observed), np.concatenate(predicted))
        if progress_callback:
            progress_callback(
                "mae_competing_hpo_trial_complete",
                {"side": side, "trial": int(trial.number), "score": score},
            )
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    params = _multiclass_params(
        study.best_trial, random_state=random_state, n_jobs=n_jobs
    )
    params["n_estimators"] = max(
        25, int(np.median(iterations[int(study.best_trial.number)]))
    )
    return params, {
        "trial_count": int(len(study.trials)),
        "best_trial": int(study.best_trial.number),
        "best_value": float(study.best_value),
        "reference_rows": int(len(base_rows)),
        "sampled_rows": int(len(hpo_base)),
        "contract": "side-local three-outcome competing-risk multiclass likelihood",
    }


def _hpo_stop_given_adverse(
    matrix: pd.DataFrame,
    target: np.ndarray,
    reference_rows: np.ndarray,
    decision: pd.Series,
    resolved: pd.Series,
    *,
    n_trials: int,
    hpo_rows: int,
    random_state: int,
    n_jobs: int,
    side: str,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sampled = auxiliary_hpo_sample_indices(
        decision.iloc[reference_rows].to_numpy(),
        max_rows=max(1, int(hpo_rows)),
        random_state=random_state,
    ).astype(np.int32)
    hpo_base = reference_rows[sampled]
    inner = _build_purged_role_folds(
        decision.iloc[hpo_base].to_numpy(),
        resolved.iloc[hpo_base].to_numpy(),
        purge_hours=13.0,
    )
    import optuna

    iterations: dict[int, list[int]] = {}

    def objective(trial: Any) -> float:
        params = _suggest_params(
            trial,
            task_kind="binary",
            quantile_alpha=0.8,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        observed: list[np.ndarray] = []
        predicted: list[np.ndarray] = []
        trial_iterations: list[int] = []
        for fold in inner:
            train_idx = hpo_base[fold.train_idx]
            valid_idx = hpo_base[fold.valid_idx]
            model, best_iteration = _fit_with_inner_validation(
                matrix.iloc[train_idx],
                target[train_idx],
                np.ones(len(train_idx), np.float32),
                matrix.iloc[valid_idx],
                target[valid_idx],
                task_kind="binary",
                params=params,
            )
            observed.append(target[valid_idx])
            predicted.append(
                _predict_role_model(model, matrix.iloc[valid_idx], task_kind="binary")
            )
            trial_iterations.append(best_iteration)
        iterations[int(trial.number)] = trial_iterations
        metrics = _binary_metrics(np.concatenate(observed), np.concatenate(predicted))
        score = (
            -float(metrics["binary_logloss"])
            - 0.25 * float(metrics["brier"])
            - 0.10 * float(metrics["ece_10bin"])
        )
        if progress_callback:
            progress_callback(
                "mae_stop_hpo_trial_complete",
                {"side": side, "trial": int(trial.number), "score": score},
            )
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    params = _sanitize_params(
        study.best_params,
        task_kind="binary",
        quantile_alpha=0.8,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    params["n_estimators"] = max(
        25, int(np.median(iterations[int(study.best_trial.number)]))
    )
    return params, {
        "trial_count": int(len(study.trials)),
        "best_trial": int(study.best_trial.number),
        "best_value": float(study.best_value),
        "reference_rows": int(len(reference_rows)),
        "sampled_rows": int(len(hpo_base)),
        "contract": "side-local conditional P(stop | adverse 0.5R first) likelihood",
    }


def fit_side_local_mae_competing_risk_family(
    X: pd.DataFrame,
    mae_atr: Sequence[Any],
    meaningful_hit: Sequence[Any],
    *,
    train_mask: Sequence[Any],
    sides: Sequence[Any],
    selected_features: Mapping[str, Sequence[str] | Mapping[str, Sequence[str]]],
    timestamps: Sequence[Any],
    label_resolved_at: Sequence[Any],
    selection_hpo_reference_end: Any,
    stop_atr_by_side: Mapping[str, float],
    n_trials: int = 12,
    hpo_rows: int = 45_000,
    random_state: int = 42,
    n_jobs: int | None = None,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Fit competing first-outcome and conditional severity models per side."""

    rows = len(X)
    if not rows or not 1 <= int(n_trials) <= 40:
        raise ValueError("non-empty X and 1..40 HPO trials are required")
    mask = _as_mask(train_mask, name="train_mask", rows=rows)
    side_values = _normalized_side_values(sides, rows=rows)
    decision = _utc_series(timestamps, name="timestamps")
    resolved = _utc_series(label_resolved_at, name="label_resolved_at")
    cutoff = _require_utc_cutoff(selection_hpo_reference_end)
    features = _union_features(selected_features, X.columns)
    targets = build_mae_competing_risk_targets(
        mae_atr,
        meaningful_hit,
        side_values,
        mask,
        stop_atr_by_side=stop_atr_by_side,
    )
    folds = _fixed_calendar_oof_folds(
        decision,
        resolved,
        reference_end=cutoff,
        oof_months=FIXED_MAY_JULY_OOF_MONTHS,
    )
    jobs = _bounded_n_jobs(n_jobs)
    probability = {name: np.full(rows, np.nan, np.float32) for name in RISK_CLASS_NAMES}
    probability["stop_1r_before_mfe"] = np.full(rows, np.nan, np.float32)
    probability["stop_given_adverse_0_5r"] = np.full(rows, np.nan, np.float32)
    fold_ids = np.full(rows, -1, np.int16)
    states: dict[str, Any] = {}
    provenance: list[dict[str, Any]] = []

    for side_i, side in enumerate(sorted(set(side_values))):
        seed = int(random_state) + 1009 * side_i
        side_mask = side_values == side
        matrix = X.loc[:, features[side]].astype(np.float32, copy=False)
        reference = np.flatnonzero(
            side_mask
            & mask
            & decision.lt(cutoff).to_numpy()
            & resolved.lt(cutoff).to_numpy()
        )
        adverse_reference = reference[targets["adverse_0_5r"][reference] > 0.5]
        multiclass_params, multiclass_hpo = _hpo_multiclass(
            matrix,
            targets["risk_class"],
            reference,
            decision,
            resolved,
            n_trials=int(n_trials),
            hpo_rows=int(hpo_rows),
            random_state=seed,
            n_jobs=jobs,
            side=side,
            progress_callback=progress_callback,
        )
        stop_params, stop_hpo = _hpo_stop_given_adverse(
            matrix,
            targets["stop_if_adverse"],
            adverse_reference,
            decision,
            resolved,
            n_trials=int(n_trials),
            hpo_rows=int(hpo_rows),
            random_state=seed + 101,
            n_jobs=jobs,
            side=side,
            progress_callback=progress_callback,
        )
        oof_models: list[dict[str, Any]] = []
        for fold_i, fold in enumerate(folds):
            train_idx = fold.base_train_idx[
                side_mask[fold.base_train_idx] & mask[fold.base_train_idx]
            ]
            valid_idx = fold.valid_idx[side_mask[fold.valid_idx]]
            adverse_train = train_idx[targets["adverse_0_5r"][train_idx] > 0.5]
            risk_model, _ = _fit_multiclass(
                matrix.iloc[train_idx],
                targets["risk_class"][train_idx],
                None,
                None,
                multiclass_params,
            )
            stop_model = _make_binary_model(stop_params)
            stop_model.fit(
                matrix.iloc[adverse_train],
                targets["stop_if_adverse"][adverse_train],
            )
            risk_probability = _predict_multiclass(risk_model, matrix.iloc[valid_idx])
            conditional_stop = _predict_role_model(
                stop_model, matrix.iloc[valid_idx], task_kind="binary"
            )
            for class_i, name in enumerate(RISK_CLASS_NAMES):
                probability[name][valid_idx] = risk_probability[:, class_i]
            probability["stop_given_adverse_0_5r"][valid_idx] = conditional_stop
            probability["stop_1r_before_mfe"][valid_idx] = (
                risk_probability[:, 1] * conditional_stop
            )
            fold_ids[valid_idx] = fold_i
            oof_models.append({"risk": risk_model, "stop": stop_model})
            provenance.append(
                {
                    "side": side,
                    "fold_month": fold.fold_month,
                    "training_rows": int(len(train_idx)),
                    "adverse_training_rows": int(len(adverse_train)),
                    "validation_rows": int(len(valid_idx)),
                    "valid_start": fold.valid_start.isoformat(),
                    "train_decision_cutoff": resolved.iloc[train_idx].max().isoformat(),
                    "training_label_resolved_max": resolved.iloc[train_idx]
                    .max()
                    .isoformat(),
                    "risk_model_sha256": _model_sha256(risk_model),
                    "stop_model_sha256": _model_sha256(stop_model),
                }
            )
        final_idx = np.flatnonzero(side_mask & mask & resolved.notna().to_numpy())
        final_adverse = final_idx[targets["adverse_0_5r"][final_idx] > 0.5]
        final_risk, _ = _fit_multiclass(
            matrix.iloc[final_idx],
            targets["risk_class"][final_idx],
            None,
            None,
            multiclass_params,
        )
        final_stop = _make_binary_model(stop_params)
        final_stop.fit(
            matrix.iloc[final_adverse],
            targets["stop_if_adverse"][final_adverse],
        )
        states[side] = {
            "selected_features": features[side],
            "stop_atr": float(stop_atr_by_side[side]),
            "multiclass_params": multiclass_params,
            "stop_params": stop_params,
            "multiclass_hpo": multiclass_hpo,
            "stop_hpo": stop_hpo,
            "oof_models": oof_models,
            "final_models": {"risk": final_risk, "stop": final_stop},
        }
    oof_mask = np.logical_and.reduce(
        [np.isfinite(values) for values in probability.values()]
    )
    risk_matrix = np.column_stack(
        [probability[name][oof_mask] for name in RISK_CLASS_NAMES]
    )
    if not np.allclose(risk_matrix.sum(axis=1), 1.0, atol=1e-5):
        raise AssertionError("competing-risk probabilities must sum to one")
    if np.any(
        probability["stop_1r_before_mfe"][oof_mask]
        > probability["adverse_0_5r_before_mfe"][oof_mask] + 1e-6
    ):
        raise AssertionError("P(stop) cannot exceed P(adverse 0.5R)")
    return {
        "schema": MAE_COMPETING_RISK_SCHEMA,
        "risk_class_names": RISK_CLASS_NAMES,
        "side_models": states,
        "oof_predictions": probability,
        "oof_prediction_mask": oof_mask,
        "oof_fold_ids": fold_ids,
        "fold_provenance": provenance,
        "targets": targets,
        "constraint_contract": (
            "three first-outcome probabilities share one multiclass model and sum "
            "to one; P(stop 1R)=P(adverse 0.5R first)*P(stop|adverse), so stop "
            "probability cannot exceed adverse-first probability"
        ),
    }


def _make_binary_model(params: Mapping[str, Any]) -> Any:
    import lightgbm as lgb

    return lgb.LGBMClassifier(**dict(params))


__all__ = [
    "MAE_COMPETING_RISK_SCHEMA",
    "RISK_CLASS_NAMES",
    "build_mae_competing_risk_targets",
    "fit_side_local_mae_competing_risk_family",
]
