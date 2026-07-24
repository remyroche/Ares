"""Strict side-local trainer for the four-bin meaningful-MFE timing CDF.

The timing target is a right-censored 12-hour clock.  Rather than regressing
that clock directly, this module fits one binary classifier for each of
``P(hit by 2h)``, ``P(hit by 4h)``, ``P(hit by 8h)``, and ``P(hit by 12h)``.
All four classifiers on a side share one HPO-selected parameter set.  Their
independent probabilities are isotonic-projected across horizons only after
outer-fold prediction, so no OOF label is used to select the projection.

This is intentionally a narrowly scoped trainer.  Feature selection is an
upstream responsibility: callers provide frozen, side-local feature lists.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.path_auxiliary_lgbm import auxiliary_hpo_sample_indices
from extreme_price_movements.path_auxiliary_model_families import (
    TIMING_HORIZONS_HOURS,
    compose_timing_cdf_predictions,
    project_monotone_timing_cdf,
)
from extreme_price_movements.path_auxiliary_role_training import (
    FIXED_MAY_JULY_OOF_MONTHS,
    _as_mask,
    _binary_metrics,
    _bounded_n_jobs,
    _build_purged_role_folds,
    _fit_with_inner_validation,
    _fixed_calendar_oof_folds,
    _hpo_score,
    _make_model,
    _model_sha256,
    _predict_role_model,
    _require_utc_cutoff,
    _sanitize_params,
    _suggest_params,
    _timestamp_summary,
    _utc_series,
)

TIMING_CDF_TRAINER_SCHEMA = "path_auxiliary_timing_cdf_side_local_v1_strict_oof"
TIMING_CDF_HORIZONS: tuple[int, ...] = tuple(
    int(value) for value in TIMING_HORIZONS_HOURS
)
TIMING_CDF_PURGE_HOURS = 13.0
TIMING_CDF_HPO_MAX_TRIALS = 40
TIMING_CDF_HPO_STALE_STOP = 12


def _normalized_side_values(values: Sequence[Any], *, rows: int) -> np.ndarray:
    """Return non-empty lower-case side labels without silently coercing nulls."""

    series = pd.Series(values, dtype="string")
    if len(series) != rows:
        raise ValueError("sides must be a one-dimensional vector aligned to X")
    normalized = series.str.strip().str.lower()
    if normalized.isna().any() or normalized.eq("").any():
        raise ValueError("sides must contain non-empty labels for every row")
    return normalized.to_numpy(dtype=str)


def _feature_list(
    values: Sequence[str],
    *,
    side: str,
    horizon: int,
    columns: pd.Index,
) -> list[str]:
    features = list(dict.fromkeys(map(str, values)))
    if not features:
        raise ValueError(
            f"selected_features for side {side!r}, horizon {horizon}h must be non-empty"
        )
    missing = [feature for feature in features if feature not in columns]
    if missing:
        raise ValueError(
            "selected timing-CDF features missing for "
            f"side {side!r}, horizon {horizon}h: {missing[:20]}"
        )
    return features


def _horizon_key(value: Any) -> int:
    text = str(value).strip().lower().removesuffix("h")
    try:
        numeric = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid timing-CDF horizon feature key: {value!r}") from exc
    if not numeric.is_integer() or int(numeric) not in TIMING_CDF_HORIZONS:
        raise ValueError(
            "timing-CDF horizon feature keys must be exactly one of "
            f"{list(TIMING_CDF_HORIZONS)}"
        )
    return int(numeric)


def _features_by_side_and_horizon(
    selected_features: Sequence[str]
    | Mapping[str, Sequence[str] | Mapping[int | str, Sequence[str]]],
    *,
    sides: Sequence[str],
    columns: pd.Index,
) -> dict[str, dict[int, list[str]]]:
    """Resolve shared, side-local, or side/horizon-local frozen feature lists."""

    if isinstance(selected_features, Mapping):
        by_side = {
            str(side).strip().lower(): features
            for side, features in selected_features.items()
        }
        missing_sides = [side for side in sides if side not in by_side]
        if missing_sides:
            raise ValueError(
                "selected_features is missing side-local lists for: "
                + ", ".join(missing_sides)
            )
        result: dict[str, dict[int, list[str]]] = {}
        for side in sides:
            side_spec = by_side[side]
            if isinstance(side_spec, Mapping):
                by_horizon: dict[int, Sequence[str]] = {}
                for raw_horizon, features in side_spec.items():
                    horizon = _horizon_key(raw_horizon)
                    if horizon in by_horizon:
                        raise ValueError(
                            f"duplicate timing-CDF horizon feature key for side {side!r}: "
                            f"{horizon}h"
                        )
                    by_horizon[horizon] = features
                missing_horizons = [
                    horizon
                    for horizon in TIMING_CDF_HORIZONS
                    if horizon not in by_horizon
                ]
                if missing_horizons:
                    raise ValueError(
                        f"selected_features is missing horizon feature lists for side "
                        f"{side!r}: {missing_horizons}"
                    )
                result[side] = {
                    horizon: _feature_list(
                        by_horizon[horizon],
                        side=side,
                        horizon=horizon,
                        columns=columns,
                    )
                    for horizon in TIMING_CDF_HORIZONS
                }
            else:
                result[side] = {
                    horizon: _feature_list(
                        side_spec,
                        side=side,
                        horizon=horizon,
                        columns=columns,
                    )
                    for horizon in TIMING_CDF_HORIZONS
                }
    else:
        result = {
            side: {
                horizon: _feature_list(
                    selected_features,
                    side=side,
                    horizon=horizon,
                    columns=columns,
                )
                for horizon in TIMING_CDF_HORIZONS
            }
            for side in sides
        }
    return result


def _preset_by_side(
    preset_params_by_side: Mapping[str, Mapping[str, Any]] | None,
    *,
    sides: Sequence[str],
) -> dict[str, Mapping[str, Any] | None]:
    if preset_params_by_side is None:
        return {side: None for side in sides}
    normalized = {
        str(side).strip().lower(): params
        for side, params in preset_params_by_side.items()
    }
    missing_sides = [side for side in sides if side not in normalized]
    if missing_sides:
        raise ValueError(
            "preset_params_by_side is missing side-local parameters for: "
            + ", ".join(missing_sides)
        )
    for side in sides:
        if not normalized[side]:
            raise ValueError(
                f"preset_params_by_side[{side!r}] must be non-empty when supplied"
            )
    return {side: normalized[side] for side in sides}


def _validate_timing_targets(
    timing_hours: Sequence[Any],
    meaningful_hit: Sequence[Any],
    timing_train_mask: np.ndarray,
    *,
    rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    timing = pd.to_numeric(pd.Series(timing_hours), errors="coerce").to_numpy(
        dtype=np.float32
    )
    hit = pd.to_numeric(pd.Series(meaningful_hit), errors="coerce").to_numpy(
        dtype=np.float32
    )
    if timing.ndim != 1 or len(timing) != rows:
        raise ValueError("timing_hours must be a one-dimensional vector aligned to X")
    if hit.ndim != 1 or len(hit) != rows:
        raise ValueError("meaningful_hit must be a one-dimensional vector aligned to X")
    valid = timing_train_mask
    if not bool(valid.any()):
        raise ValueError("timing_train_mask has no trainable canonical timing rows")
    invalid_time = valid & (
        ~np.isfinite(timing)
        | (timing < 0.0)
        | (timing > float(TIMING_CDF_HORIZONS[-1]))
    )
    if invalid_time.any():
        raise ValueError(
            "timing_hours must be finite inside [0, 12] on timing_train_mask rows"
        )
    invalid_hit = valid & (~np.isfinite(hit) | ~np.isin(hit, (0.0, 1.0)))
    if invalid_hit.any():
        raise ValueError("meaningful_hit must be exactly 0/1 on timing_train_mask rows")
    impossible_hit = valid & (hit > 0.5) & (timing <= 0.0)
    if impossible_hit.any():
        raise ValueError("meaningful hits require a strictly positive timing clock")
    return timing, hit


def _horizon_targets(
    timing: np.ndarray, meaningful_hit: np.ndarray, train_mask: np.ndarray
) -> dict[int, np.ndarray]:
    """Return aligned binary labels; a censored 12h clock is not a 12h hit."""

    result: dict[int, np.ndarray] = {}
    for hours in TIMING_CDF_HORIZONS:
        target = np.full(len(timing), np.nan, dtype=np.float32)
        target[train_mask] = (
            (meaningful_hit[train_mask] > 0.5) & (timing[train_mask] <= float(hours))
        ).astype(np.float32)
        result[hours] = target
    return result


def _reference_mask(
    decision: pd.Series,
    resolved: pd.Series,
    timing_train_mask: np.ndarray,
    cutoff: pd.Timestamp,
    side_mask: np.ndarray,
) -> np.ndarray:
    return (
        side_mask
        & timing_train_mask
        & decision.lt(cutoff).to_numpy()
        & resolved.lt(cutoff).to_numpy()
    )


def _hpo_fold_provenance(
    hpo_resolved: np.ndarray,
    inner_folds: Sequence[Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for fold_i, fold in enumerate(inner_folds):
        train_resolution = pd.to_datetime(
            pd.Series(hpo_resolved[fold.train_idx]), utc=True, errors="coerce"
        )
        assert bool(train_resolution.max() < fold.valid_start), (
            "HPO timing-CDF fold must resolve before its validation decision window"
        )
        records.append(
            {
                "fold": int(fold_i),
                "training_rows": int(len(fold.train_idx)),
                "validation_rows": int(len(fold.valid_idx)),
                "valid_start": fold.valid_start.isoformat(),
                "valid_end": fold.valid_end.isoformat(),
                "training_label_resolved_max": train_resolution.max().isoformat(),
                "resolution_before_valid_start_assertion": True,
            }
        )
    return records


def _fit_shared_hpo_params(
    matrices: Mapping[int, pd.DataFrame],
    targets: Mapping[int, np.ndarray],
    weights: np.ndarray,
    decision: pd.Series,
    resolved: pd.Series,
    reference_rows: np.ndarray,
    *,
    random_state: int,
    n_jobs: int,
    n_trials: int,
    hpo_rows: int,
    side: str,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Tune one parameter vector jointly across every timing-CDF horizon."""

    hpo_local_idx = auxiliary_hpo_sample_indices(
        decision.iloc[reference_rows].to_numpy(),
        max_rows=max(1, int(hpo_rows)),
        random_state=int(random_state),
    ).astype(np.int32)
    hpo_rows_global = reference_rows[hpo_local_idx]
    hpo_timestamps = decision.iloc[hpo_rows_global].to_numpy()
    hpo_resolved = resolved.iloc[hpo_rows_global].to_numpy()
    inner_folds = _build_purged_role_folds(
        hpo_timestamps,
        hpo_resolved,
        purge_hours=TIMING_CDF_PURGE_HOURS,
    )
    fold_provenance = _hpo_fold_provenance(hpo_resolved, inner_folds)
    hpo_X = {
        hours: matrices[hours].iloc[hpo_rows_global].reset_index(drop=True)
        for hours in TIMING_CDF_HORIZONS
    }
    hpo_weight = weights[hpo_rows_global]
    hpo_targets = {hours: target[hpo_rows_global] for hours, target in targets.items()}

    import optuna

    trial_iterations: dict[int, list[int]] = {}
    trial_horizon_scores: dict[int, dict[int, float]] = {}

    def objective(trial: Any) -> float:
        if progress_callback is not None:
            progress_callback(
                "hpo_trial_start", {"side": side, "trial": int(trial.number)}
            )
        params = _suggest_params(
            trial,
            task_kind="binary",
            quantile_alpha=0.80,
            random_state=int(random_state),
            n_jobs=n_jobs,
        )
        horizon_scores: dict[int, float] = {}
        iterations: list[int] = []
        for hours in TIMING_CDF_HORIZONS:
            if progress_callback is not None:
                progress_callback(
                    "hpo_horizon_start",
                    {
                        "side": side,
                        "trial": int(trial.number),
                        "horizon_hours": int(hours),
                    },
                )
            observed: list[np.ndarray] = []
            predicted: list[np.ndarray] = []
            for fold_i, fold in enumerate(inner_folds):
                if progress_callback is not None:
                    progress_callback(
                        "hpo_fold_start",
                        {
                            "side": side,
                            "trial": int(trial.number),
                            "horizon_hours": int(hours),
                            "fold": int(fold_i),
                            "training_rows": int(len(fold.train_idx)),
                            "validation_rows": int(len(fold.valid_idx)),
                        },
                    )
                model, best_iteration = _fit_with_inner_validation(
                    hpo_X[hours].iloc[fold.train_idx],
                    hpo_targets[hours][fold.train_idx],
                    hpo_weight[fold.train_idx],
                    hpo_X[hours].iloc[fold.valid_idx],
                    hpo_targets[hours][fold.valid_idx],
                    task_kind="binary",
                    params=params,
                )
                observed.append(hpo_targets[hours][fold.valid_idx])
                predicted.append(
                    _predict_role_model(
                        model, hpo_X[hours].iloc[fold.valid_idx], task_kind="binary"
                    )
                )
                iterations.append(int(best_iteration))
                if progress_callback is not None:
                    progress_callback(
                        "hpo_fold_complete",
                        {
                            "side": side,
                            "trial": int(trial.number),
                            "horizon_hours": int(hours),
                            "fold": int(fold_i),
                            "best_iteration": int(best_iteration),
                        },
                    )
            metrics = _binary_metrics(
                np.concatenate(observed), np.concatenate(predicted)
            )
            horizon_scores[hours] = _hpo_score(metrics, task_kind="binary")
            if progress_callback is not None:
                progress_callback(
                    "hpo_horizon_complete",
                    {
                        "side": side,
                        "trial": int(trial.number),
                        "horizon_hours": int(hours),
                        "score": float(horizon_scores[hours]),
                    },
                )
        trial_iterations[int(trial.number)] = iterations
        trial_horizon_scores[int(trial.number)] = horizon_scores
        values = np.asarray(list(horizon_scores.values()), dtype=np.float64)
        score = float(np.mean(values) - 0.25 * np.std(values))
        if progress_callback is not None:
            progress_callback(
                "hpo_trial_complete",
                {"side": side, "trial": int(trial.number), "score": score},
            )
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(random_state)),
        pruner=optuna.pruners.NopPruner(),
    )
    best_seen = -np.inf
    stale_trials = 0

    def stop_after_twelve_stale_trials(study: Any, trial: Any) -> None:
        nonlocal best_seen, stale_trials
        value = trial.value
        if value is not None and np.isfinite(value) and float(value) > best_seen:
            best_seen = float(value)
            stale_trials = 0
        else:
            stale_trials += 1
        if (
            len(study.trials) >= TIMING_CDF_HPO_STALE_STOP
            and stale_trials >= TIMING_CDF_HPO_STALE_STOP
        ):
            study.stop()

    study.optimize(
        objective,
        n_trials=int(n_trials),
        n_jobs=1,
        show_progress_bar=False,
        callbacks=[stop_after_twelve_stale_trials],
    )
    best_params = _sanitize_params(
        study.best_params,
        task_kind="binary",
        quantile_alpha=0.80,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    best_params["n_estimators"] = max(
        25,
        int(np.median(trial_iterations[int(study.best_trial.number)])),
    )
    best_params["subsample_freq"] = 1
    return best_params, {
        "reused_preset_params": False,
        "trial_count": int(len(study.trials)),
        "maximum_trials": TIMING_CDF_HPO_MAX_TRIALS,
        "stale_trial_patience": TIMING_CDF_HPO_STALE_STOP,
        "best_value": float(study.best_value),
        "best_trial_horizon_scores": {
            str(hours): float(value)
            for hours, value in trial_horizon_scores[
                int(study.best_trial.number)
            ].items()
        },
        "reference_rows": int(len(reference_rows)),
        "hpo_rows": int(len(hpo_rows_global)),
        "purged_fold_provenance": fold_provenance,
        "contract": (
            "one side-local parameter vector is selected by the equally weighted "
            "joint score across 2/4/8/12-hour binary CDF horizons; all HPO rows "
            "resolve strictly before each inner validation decision window"
        ),
    }


def _project_oof_cdf(
    raw_predictions: Mapping[int, np.ndarray], prediction_mask: np.ndarray
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Project prefix bins while preserving the 12-hour event-owner exactly."""

    projected = {
        hours: np.full(len(prediction_mask), np.nan, dtype=np.float32)
        for hours in TIMING_CDF_HORIZONS
    }
    if bool(prediction_mask.any()):
        final_horizon = TIMING_CDF_HORIZONS[-1]
        prefix_horizons = TIMING_CDF_HORIZONS[:-1]
        prefix = project_monotone_timing_cdf(
            {
                hours: raw_predictions[hours][prediction_mask]
                for hours in prefix_horizons
            },
            horizons=prefix_horizons,
            preserve_final_horizon=False,
        )
        final_probability = raw_predictions[final_horizon][prediction_mask]
        for hours in prefix_horizons:
            # The prefix PAV result is additionally bounded by the independently
            # trained 12h event owner.  Clipping block levels at that fixed bound
            # is the least-squares constrained isotonic solution for the prefix.
            projected[hours][prediction_mask] = np.minimum(
                prefix[float(hours)], final_probability
            ).astype(np.float32)
        projected[final_horizon][prediction_mask] = final_probability.astype(np.float32)
    adjustable_horizons = TIMING_CDF_HORIZONS[:-1]
    corrections = (
        np.column_stack(
            [
                np.abs(
                    projected[hours][prediction_mask]
                    - raw_predictions[hours][prediction_mask]
                )
                for hours in adjustable_horizons
            ]
        )
        if bool(prediction_mask.any())
        else np.empty((0, len(adjustable_horizons)))
    )
    return projected, {
        "method": "per-row prefix pool-adjacent-violators isotonic projection",
        "applied_after_outer_fold_prediction": True,
        "fixed_horizon_hours": int(TIMING_CDF_HORIZONS[-1]),
        "fixed_final_horizon_probability": True,
        "rows": int(prediction_mask.sum()),
        "rows_changed": int(np.any(corrections > 1e-8, axis=1).sum())
        if len(corrections)
        else 0,
        "max_absolute_adjustment": float(np.max(corrections))
        if len(corrections)
        else 0.0,
    }


def _final_mask(
    side_mask: np.ndarray,
    timing_train_mask: np.ndarray,
    decision: pd.Series,
    resolved: pd.Series,
) -> np.ndarray:
    return (
        side_mask
        & timing_train_mask
        & decision.notna().to_numpy()
        & resolved.notna().to_numpy()
    )


def fit_side_local_timing_cdf_family(
    X: pd.DataFrame,
    timing_hours: Sequence[Any],
    meaningful_hit: Sequence[Any],
    *,
    timing_train_mask: Sequence[Any],
    sides: Sequence[Any],
    selected_features: Sequence[str]
    | Mapping[str, Sequence[str] | Mapping[int | str, Sequence[str]]],
    timestamps: Sequence[Any],
    label_resolved_at: Sequence[Any],
    selection_hpo_reference_end: Any,
    sample_weight: Sequence[Any] | None = None,
    n_trials: int = TIMING_CDF_HPO_MAX_TRIALS,
    preset_params_by_side: Mapping[str, Mapping[str, Any]] | None = None,
    hpo_rows: int = 45_000,
    random_state: int = 42,
    n_jobs: int | None = None,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train side-local 2/4/8/12h timing CDFs under a fixed OOF contract.

    ``timing_hours`` is the canonical right-censored clock and ``meaningful_hit``
    distinguishes a genuine 12h hit from an unreached path censored at 12h.
    ``timing_train_mask`` controls fitting and metrics, never the outer
    validation rows that receive predictions.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    rows = len(X)
    if not rows:
        raise ValueError("X must contain at least one row")
    if (
        preset_params_by_side is None
        and not 1 <= int(n_trials) <= TIMING_CDF_HPO_MAX_TRIALS
    ):
        raise ValueError(
            f"n_trials must be between 1 and the production cap of {TIMING_CDF_HPO_MAX_TRIALS}"
        )
    train_mask = _as_mask(timing_train_mask, name="timing_train_mask", rows=rows)
    decision = _utc_series(timestamps, name="timestamps")
    resolved = _utc_series(label_resolved_at, name="label_resolved_at")
    if len(decision) != rows or len(resolved) != rows:
        raise ValueError("timestamps and label_resolved_at must align to X")
    cutoff = _require_utc_cutoff(selection_hpo_reference_end)
    timing, hit = _validate_timing_targets(
        timing_hours, meaningful_hit, train_mask, rows=rows
    )
    side_values = _normalized_side_values(sides, rows=rows)
    unique_sides = tuple(sorted(set(side_values.tolist())))
    feature_names = _features_by_side_and_horizon(
        selected_features, sides=unique_sides, columns=X.columns
    )
    preset_by_side = _preset_by_side(preset_params_by_side, sides=unique_sides)
    weights = (
        np.ones(rows, dtype=np.float32)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float32)
    )
    if (
        weights.ndim != 1
        or weights.shape != timing.shape
        or not np.isfinite(weights).all()
        or (weights <= 0.0).any()
    ):
        raise ValueError(
            "sample_weight must be finite, positive, and aligned to timing_hours"
        )
    bounded_jobs = _bounded_n_jobs(n_jobs)
    targets = _horizon_targets(timing, hit, train_mask)
    outer_folds = _fixed_calendar_oof_folds(
        decision,
        resolved,
        reference_end=cutoff,
        oof_months=FIXED_MAY_JULY_OOF_MONTHS,
    )
    raw_oof = {
        hours: np.full(rows, np.nan, dtype=np.float32) for hours in TIMING_CDF_HORIZONS
    }
    oof_fold_ids = np.full(rows, -1, dtype=np.int16)
    side_states: dict[str, dict[str, Any]] = {}
    fold_provenance: list[dict[str, Any]] = []

    if progress_callback is not None:
        progress_callback(
            "timing_cdf_training_start",
            {
                "sides": list(unique_sides),
                "horizons_hours": list(TIMING_CDF_HORIZONS),
                "rows": int(rows),
            },
        )

    for side_i, side in enumerate(unique_sides):
        side_seed = int(random_state) + side_i
        side_mask = side_values == side
        reference_mask = _reference_mask(
            decision, resolved, train_mask, cutoff, side_mask
        )
        reference_rows = np.flatnonzero(reference_mask)
        if not len(reference_rows):
            raise ValueError(
                f"side {side!r} has no timing rows resolved before the HPO reference cutoff"
            )
        matrices = {
            hours: X.loc[:, feature_names[side][hours]].astype(np.float32, copy=False)
            for hours in TIMING_CDF_HORIZONS
        }
        if preset_by_side[side] is None:
            if progress_callback is not None:
                progress_callback(
                    "hpo_start",
                    {
                        "side": side,
                        "reference_rows": int(len(reference_rows)),
                        "horizons_hours": list(TIMING_CDF_HORIZONS),
                    },
                )
            best_params, hpo = _fit_shared_hpo_params(
                matrices,
                targets,
                weights,
                decision,
                resolved,
                reference_rows,
                random_state=side_seed,
                n_jobs=bounded_jobs,
                n_trials=int(n_trials),
                hpo_rows=int(hpo_rows),
                side=side,
                progress_callback=progress_callback,
            )
            if progress_callback is not None:
                progress_callback(
                    "hpo_complete",
                    {
                        "side": side,
                        "trial_count": int(hpo["trial_count"]),
                        "best_value": hpo["best_value"],
                    },
                )
        else:
            best_params = _sanitize_params(
                preset_by_side[side] or {},
                task_kind="binary",
                quantile_alpha=0.80,
                random_state=side_seed,
                n_jobs=bounded_jobs,
            )
            hpo = {
                "reused_preset_params": True,
                "trial_count": 0,
                "maximum_trials": TIMING_CDF_HPO_MAX_TRIALS,
                "stale_trial_patience": TIMING_CDF_HPO_STALE_STOP,
                "best_value": None,
                "best_trial_horizon_scores": None,
                "reference_rows": int(len(reference_rows)),
                "hpo_rows": 0,
                "purged_fold_provenance": [],
                "contract": "preset side-local parameters are reused jointly for every 2/4/8/12-hour CDF horizon",
            }
            if progress_callback is not None:
                progress_callback(
                    "hpo_reused",
                    {"side": side, "reference_rows": int(len(reference_rows))},
                )
        oof_models = {hours: [] for hours in TIMING_CDF_HORIZONS}
        for fold_i, fold in enumerate(outer_folds):
            if progress_callback is not None:
                progress_callback(
                    "oof_fold_start",
                    {"side": side, "fold": int(fold_i), "fold_month": fold.fold_month},
                )
            train_idx = fold.base_train_idx[
                side_mask[fold.base_train_idx] & train_mask[fold.base_train_idx]
            ]
            valid_idx = fold.valid_idx[side_mask[fold.valid_idx]]
            if not len(train_idx):
                raise ValueError(
                    f"outer OOF fold {fold.fold_month} has no side-local timing training rows for {side!r}"
                )
            if not len(valid_idx):
                if progress_callback is not None:
                    progress_callback(
                        "oof_fold_skipped",
                        {
                            "side": side,
                            "fold": int(fold_i),
                            "fold_month": fold.fold_month,
                            "reason": "no_side_local_validation_rows",
                        },
                    )
                continue
            train_resolution = resolved.iloc[train_idx]
            if train_resolution.isna().any() or not bool(
                train_resolution.max() < fold.valid_start
            ):
                raise AssertionError(
                    "outer timing-CDF fold violates max(train label resolution) < valid start"
                )
            model_hashes: dict[str, str] = {}
            for hours in TIMING_CDF_HORIZONS:
                if progress_callback is not None:
                    progress_callback(
                        "oof_horizon_start",
                        {
                            "side": side,
                            "fold": int(fold_i),
                            "fold_month": fold.fold_month,
                            "horizon_hours": int(hours),
                            "training_rows": int(len(train_idx)),
                            "validation_rows": int(len(valid_idx)),
                        },
                    )
                model = _make_model("binary", best_params)
                # Do not pass outer validation labels to LightGBM.  HPO selected
                # both the parameter vector and its fixed tree count beforehand.
                model.fit(
                    matrices[hours].iloc[train_idx],
                    targets[hours][train_idx],
                    sample_weight=weights[train_idx],
                )
                raw_oof[hours][valid_idx] = _predict_role_model(
                    model, matrices[hours].iloc[valid_idx], task_kind="binary"
                )
                model_hashes[str(hours)] = _model_sha256(model)
                oof_models[hours].append(model)
                if progress_callback is not None:
                    progress_callback(
                        "oof_horizon_complete",
                        {
                            "side": side,
                            "fold": int(fold_i),
                            "fold_month": fold.fold_month,
                            "horizon_hours": int(hours),
                        },
                    )
            oof_fold_ids[valid_idx] = int(fold_i)
            fold_provenance.append(
                {
                    "fold": int(fold_i),
                    "fold_month": fold.fold_month,
                    "side": side,
                    "train_start": fold.train_start.isoformat()
                    if fold.train_start
                    else None,
                    "train_end": fold.train_end.isoformat() if fold.train_end else None,
                    "valid_start": fold.valid_start.isoformat(),
                    "valid_end": fold.valid_end.isoformat(),
                    "training_rows": int(len(train_idx)),
                    "validation_rows": int(len(valid_idx)),
                    "predicted_validation_rows": int(len(valid_idx)),
                    "conditional_validation_rows": int(train_mask[valid_idx].sum()),
                    "training_label_resolved_max": train_resolution.max().isoformat(),
                    "resolution_before_valid_start_assertion": True,
                    "prediction_contract": (
                        "every side-local decision row in the fixed calendar validation "
                        "month receives all four raw CDF predictions; timing_train_mask "
                        "affects fitting and metrics only"
                    ),
                    "outer_fit_contract": (
                        "fit uses only causally resolved side-local timing rows and no "
                        "outer validation labels reach LightGBM"
                    ),
                    "model_sha256_by_horizon": model_hashes,
                }
            )
            if progress_callback is not None:
                progress_callback(
                    "oof_fold_complete",
                    {"side": side, "fold": int(fold_i), "fold_month": fold.fold_month},
                )
        final_mask = _final_mask(side_mask, train_mask, decision, resolved)
        final_idx = np.flatnonzero(final_mask)
        if not len(final_idx):
            raise ValueError(
                f"side {side!r} has no resolved timing rows for separate final refits"
            )
        final_models: dict[int, Any] = {}
        final_hashes: dict[str, str] = {}
        for hours in TIMING_CDF_HORIZONS:
            if progress_callback is not None:
                progress_callback(
                    "final_model_start",
                    {
                        "side": side,
                        "horizon_hours": int(hours),
                        "rows": int(len(final_idx)),
                    },
                )
            model = _make_model("binary", best_params)
            model.fit(
                matrices[hours].iloc[final_idx],
                targets[hours][final_idx],
                sample_weight=weights[final_idx],
            )
            final_models[hours] = model
            final_hashes[str(hours)] = _model_sha256(model)
            if progress_callback is not None:
                progress_callback(
                    "final_model_complete",
                    {
                        "side": side,
                        "horizon_hours": int(hours),
                        "rows": int(len(final_idx)),
                    },
                )
        side_states[side] = {
            "selected_features": feature_names[side][TIMING_CDF_HORIZONS[-1]],
            "selected_features_by_horizon": feature_names[side],
            "best_params": best_params,
            "hpo": hpo,
            "reference_split_contract": {
                "selection_hpo_reference_end": cutoff.isoformat(),
                "row_rule": (
                    "side matches AND timing_train_mask AND decision_timestamp < "
                    "selection_hpo_reference_end AND label_resolved_at < "
                    "selection_hpo_reference_end"
                ),
                "reference_rows": int(len(reference_rows)),
                "decision_bounds": _timestamp_summary(decision.iloc[reference_rows]),
                "label_resolved_bounds": _timestamp_summary(
                    resolved.iloc[reference_rows]
                ),
            },
            "oof_models": oof_models,
            "final_models": final_models,
            "final_refit_contract": {
                "rows": int(len(final_idx)),
                "row_rule": "all resolved side-local timing rows; separate from and excluded from OOF metrics",
                "decision_bounds": _timestamp_summary(decision.iloc[final_idx]),
                "label_resolved_bounds": _timestamp_summary(resolved.iloc[final_idx]),
                "model_sha256_by_horizon": final_hashes,
            },
        }

    oof_prediction_mask = np.logical_and.reduce(
        [np.isfinite(raw_oof[hours]) for hours in TIMING_CDF_HORIZONS]
    )
    if not bool(oof_prediction_mask.any()):
        raise ValueError("timing-CDF trainer has no complete outer-OOF predictions")
    projected_oof, projection = _project_oof_cdf(raw_oof, oof_prediction_mask)
    oof_metrics = {
        hours: _binary_metrics(
            targets[hours][train_mask & oof_prediction_mask],
            projected_oof[hours][train_mask & oof_prediction_mask],
        )
        for hours in TIMING_CDF_HORIZONS
    }
    raw_oof_metrics = {
        hours: _binary_metrics(
            targets[hours][train_mask & oof_prediction_mask],
            raw_oof[hours][train_mask & oof_prediction_mask],
        )
        for hours in TIMING_CDF_HORIZONS
    }
    expected_oof_mask = np.zeros(rows, dtype=bool)
    for month_text in FIXED_MAY_JULY_OOF_MONTHS:
        month_start = pd.Timestamp(pd.Period(month_text, freq="M").start_time, tz="UTC")
        month_stop = month_start + pd.offsets.MonthBegin(1)
        expected_oof_mask |= (
            decision.ge(month_start) & decision.lt(month_stop)
        ).to_numpy()
    if not np.array_equal(oof_prediction_mask, expected_oof_mask):
        raise AssertionError(
            "complete timing-CDF OOF coverage must equal the fixed May/June/July decision rows"
        )
    result = {
        "schema": TIMING_CDF_TRAINER_SCHEMA,
        "horizons_hours": TIMING_CDF_HORIZONS,
        "sides": unique_sides,
        "side_models": side_states,
        "selected_features_by_side": {
            side: list(
                dict.fromkeys(
                    feature
                    for hours in TIMING_CDF_HORIZONS
                    for feature in feature_names[side][hours]
                )
            )
            for side in unique_sides
        },
        "selected_features_by_side_and_horizon": feature_names,
        "selected_features_contract": (
            "selected_features_by_side is a backward-compatible per-side union; "
            "selected_features_by_side_and_horizon is the exact frozen model contract"
        ),
        "raw_oof_predictions_by_horizon": raw_oof,
        "oof_predictions_by_horizon": projected_oof,
        "oof_prediction_mask": oof_prediction_mask,
        "oof_fold_ids": oof_fold_ids,
        "oof_metrics_by_horizon": oof_metrics,
        "raw_oof_metrics_by_horizon": raw_oof_metrics,
        "fold_provenance": fold_provenance,
        "monotone_projection": projection,
        "models": {
            "oof": {side: state["oof_models"] for side, state in side_states.items()},
            "final": {
                side: state["final_models"] for side, state in side_states.items()
            },
        },
        "final_models": {
            side: state["final_models"] for side, state in side_states.items()
        },
        "oof_contract": (
            "fixed expanding May/June/July calendar OOF, with 13-hour resolution "
            "purge; every validation row is predicted before an outcome-independent "
            "monotone CDF projection"
        ),
        "final_refit_contract": {
            "row_rule": "per-side all resolved timing rows; separate from and excluded from OOF metrics",
            "final_models_are_not_oof_models": True,
        },
        "sample_weight_contract": "sample weights apply to fitting loss only; HPO and OOF metrics are unweighted",
    }
    if progress_callback is not None:
        progress_callback(
            "timing_cdf_training_complete",
            {
                "sides": list(unique_sides),
                "oof_rows": int(oof_prediction_mask.sum()),
                "monotone_projection_rows_changed": int(projection["rows_changed"]),
            },
        )
    return result


def predict_side_local_timing_cdf_family(
    family: Mapping[str, Any], X: pd.DataFrame, *, sides: Sequence[Any]
) -> dict[str, Any]:
    """Score final side-local timing-CDF models and project their probabilities."""

    if family.get("schema") != TIMING_CDF_TRAINER_SCHEMA:
        raise ValueError("not a side-local timing-CDF family bundle")
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    rows = len(X)
    side_values = _normalized_side_values(sides, rows=rows)
    known_sides = tuple(map(str, family["sides"]))
    unknown = sorted(set(side_values).difference(known_sides))
    if unknown:
        raise ValueError(f"timing-CDF family has no final model for sides: {unknown}")
    raw = {
        hours: np.full(rows, np.nan, dtype=np.float32) for hours in TIMING_CDF_HORIZONS
    }
    for side in known_sides:
        positions = np.flatnonzero(side_values == side)
        if not len(positions):
            continue
        state = family["side_models"][side]
        by_horizon = state.get("selected_features_by_horizon")
        for hours in TIMING_CDF_HORIZONS:
            features = list(
                state["selected_features"] if by_horizon is None else by_horizon[hours]
            )
            missing = [feature for feature in features if feature not in X.columns]
            if missing:
                raise ValueError(
                    "timing-CDF scoring features missing for "
                    f"side {side!r}, horizon {hours}h: {missing[:20]}"
                )
            matrix = X.iloc[positions].loc[:, features].astype(np.float32, copy=False)
            raw[hours][positions] = _predict_role_model(
                state["final_models"][hours], matrix, task_kind="binary"
            )
    prediction_mask = np.logical_and.reduce(
        [np.isfinite(raw[hours]) for hours in TIMING_CDF_HORIZONS]
    )
    projected, projection = _project_oof_cdf(raw, prediction_mask)
    composed: dict[str, np.ndarray] = {
        "p_hit_by_2h": np.full(rows, np.nan, dtype=np.float32),
        "p_hit_by_4h": np.full(rows, np.nan, dtype=np.float32),
        "p_hit_by_8h": np.full(rows, np.nan, dtype=np.float32),
        "p_hit_by_12h": np.full(rows, np.nan, dtype=np.float32),
        "p_hit_12h": np.full(rows, np.nan, dtype=np.float32),
        "expected_censored_time_hours": np.full(rows, np.nan, dtype=np.float32),
    }
    if bool(prediction_mask.any()):
        local = compose_timing_cdf_predictions(
            {hours: projected[hours][prediction_mask] for hours in TIMING_CDF_HORIZONS},
            horizons=TIMING_CDF_HORIZONS,
        )
        for name, values in local.items():
            composed[name][prediction_mask] = np.asarray(values, dtype=np.float32)
    return {
        "raw_predictions_by_horizon": raw,
        "predictions_by_horizon": projected,
        "prediction_mask": prediction_mask,
        "monotone_projection": projection,
        **composed,
    }


# The short alias matches the role trainer's public surface without obscuring
# that this function fits all four horizons as one side-local family.
train_side_local_timing_cdf_family = fit_side_local_timing_cdf_family


__all__ = [
    "TIMING_CDF_HORIZONS",
    "TIMING_CDF_HPO_MAX_TRIALS",
    "TIMING_CDF_HPO_STALE_STOP",
    "TIMING_CDF_PURGE_HOURS",
    "TIMING_CDF_TRAINER_SCHEMA",
    "fit_side_local_timing_cdf_family",
    "predict_side_local_timing_cdf_family",
    "train_side_local_timing_cdf_family",
]
