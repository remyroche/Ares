"""Side-local discrete-time hazard challenger for meaningful-MFE timing.

Unlike the incumbent timing family, this trainer fits one pooled at-risk
classifier per side.  Its conditional interval hazards are converted to a CDF
by ``1 - cumulative_survival``, so monotonicity is structural rather than a
post-hoc repair.
"""

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
    _hpo_score,
    _make_model,
    _model_sha256,
    _predict_role_model,
    _require_utc_cutoff,
    _sanitize_params,
    _suggest_params,
    _utc_series,
)
from extreme_price_movements.path_auxiliary_timing_training import (
    TIMING_CDF_HORIZONS,
    _as_mask,
    _normalized_side_values,
    _validate_timing_targets,
)

TIMING_HAZARD_SCHEMA = "path_auxiliary_timing_discrete_hazard_v1_strict_oof"
_LOWER = np.asarray((0.0, 2.0, 4.0, 8.0), dtype=np.float32)
_UPPER = np.asarray(TIMING_CDF_HORIZONS, dtype=np.float32)


def _union_features(
    selected: Mapping[str, Sequence[str] | Mapping[int | str, Sequence[str]]],
    columns: pd.Index,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for raw_side, value in selected.items():
        side = str(raw_side).strip().lower()
        streams = value.values() if isinstance(value, Mapping) else (value,)
        features = list(dict.fromkeys(str(f) for stream in streams for f in stream))
        if not features:
            raise ValueError(f"empty timing hazard feature contract for {side}")
        missing = [feature for feature in features if feature not in columns]
        if missing:
            raise ValueError(
                f"timing hazard features missing for {side}: {missing[:20]}"
            )
        result[side] = features
    return result


def _event_interval(timing: np.ndarray, hit: np.ndarray) -> np.ndarray:
    interval = np.full(len(timing), -1, dtype=np.int8)
    reached = hit > 0.5
    interval[reached] = np.searchsorted(_UPPER, timing[reached], side="left").astype(
        np.int8
    )
    return interval


def _expand_at_risk(
    matrix: pd.DataFrame,
    base_idx: np.ndarray,
    event_interval: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Expand base rows only through their event interval (or all bins if censored)."""

    base_parts: list[np.ndarray] = []
    bin_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for bin_index in range(len(_UPPER)):
        at_risk = base_idx[
            (event_interval[base_idx] < 0) | (event_interval[base_idx] >= bin_index)
        ]
        base_parts.append(at_risk)
        bin_parts.append(np.full(len(at_risk), bin_index, dtype=np.int8))
        target_parts.append((event_interval[at_risk] == bin_index).astype(np.float32))
    row_idx = np.concatenate(base_parts)
    bins = np.concatenate(bin_parts)
    expanded = matrix.iloc[row_idx].reset_index(drop=True).copy()
    expanded["__hazard_bin_2h"] = (bins == 0).astype(np.float32)
    expanded["__hazard_bin_4h"] = (bins == 1).astype(np.float32)
    expanded["__hazard_bin_8h"] = (bins == 2).astype(np.float32)
    expanded["__hazard_bin_12h"] = (bins == 3).astype(np.float32)
    expanded["__hazard_interval_width_h"] = (_UPPER[bins] - _LOWER[bins]).astype(
        np.float32
    )
    return expanded, np.concatenate(target_parts), row_idx


def _predict_cdf(model: Any, matrix: pd.DataFrame, base_idx: np.ndarray) -> np.ndarray:
    """Score every interval for every row and convert hazards to a monotone CDF."""

    blocks: list[np.ndarray] = []
    for bin_index in range(len(_UPPER)):
        local = matrix.iloc[base_idx].reset_index(drop=True).copy()
        for index, hours in enumerate(TIMING_CDF_HORIZONS):
            local[f"__hazard_bin_{hours}h"] = np.float32(index == bin_index)
        local["__hazard_interval_width_h"] = np.float32(
            _UPPER[bin_index] - _LOWER[bin_index]
        )
        blocks.append(
            np.clip(_predict_role_model(model, local, task_kind="binary"), 0.0, 1.0)
        )
    hazard = np.column_stack(blocks)
    return 1.0 - np.cumprod(1.0 - hazard, axis=1)


def _joint_score(targets: Mapping[int, np.ndarray], cdf: np.ndarray) -> float:
    scores = [
        _hpo_score(_binary_metrics(targets[hours], cdf[:, index]), task_kind="binary")
        for index, hours in enumerate(TIMING_CDF_HORIZONS)
    ]
    values = np.asarray(scores, dtype=np.float64)
    return float(values.mean() - 0.25 * values.std())


def fit_side_local_timing_hazard_family(
    X: pd.DataFrame,
    timing_hours: Sequence[Any],
    meaningful_hit: Sequence[Any],
    *,
    timing_train_mask: Sequence[Any],
    sides: Sequence[Any],
    selected_features: Mapping[str, Sequence[str] | Mapping[int | str, Sequence[str]]],
    timestamps: Sequence[Any],
    label_resolved_at: Sequence[Any],
    selection_hpo_reference_end: Any,
    n_trials: int = 12,
    hpo_rows: int = 45_000,
    random_state: int = 42,
    n_jobs: int | None = None,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Fit a pooled at-risk hazard model independently for long and short."""

    rows = len(X)
    if not rows or not 1 <= int(n_trials) <= 40:
        raise ValueError("non-empty X and 1..40 HPO trials are required")
    mask = _as_mask(timing_train_mask, name="timing_train_mask", rows=rows)
    timing, hit = _validate_timing_targets(
        timing_hours, meaningful_hit, mask, rows=rows
    )
    side_values = _normalized_side_values(sides, rows=rows)
    decision = _utc_series(timestamps, name="timestamps")
    resolved = _utc_series(label_resolved_at, name="label_resolved_at")
    cutoff = _require_utc_cutoff(selection_hpo_reference_end)
    features = _union_features(selected_features, X.columns)
    event_interval = _event_interval(timing, hit)
    targets = {
        hours: ((hit > 0.5) & (timing <= hours)).astype(np.float32)
        for hours in TIMING_CDF_HORIZONS
    }
    folds = _fixed_calendar_oof_folds(
        decision, resolved, reference_end=cutoff, oof_months=FIXED_MAY_JULY_OOF_MONTHS
    )
    jobs = _bounded_n_jobs(n_jobs)
    oof = {hours: np.full(rows, np.nan, np.float32) for hours in TIMING_CDF_HORIZONS}
    fold_ids = np.full(rows, -1, np.int16)
    states: dict[str, Any] = {}
    provenance: list[dict[str, Any]] = []

    for side_i, side in enumerate(sorted(set(side_values))):
        if side not in features:
            raise ValueError(f"missing selected features for {side}")
        seed = int(random_state) + 1009 * side_i
        side_mask = side_values == side
        matrix = X.loc[:, features[side]].astype(np.float32, copy=False)
        reference = np.flatnonzero(
            side_mask
            & mask
            & decision.lt(cutoff).to_numpy()
            & resolved.lt(cutoff).to_numpy()
        )
        sampled = auxiliary_hpo_sample_indices(
            decision.iloc[reference].to_numpy(),
            max_rows=max(1, int(hpo_rows)),
            random_state=seed,
        ).astype(np.int32)
        hpo_base = reference[sampled]
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
                quantile_alpha=0.80,
                random_state=seed,
                n_jobs=jobs,
            )
            observed = {hours: [] for hours in TIMING_CDF_HORIZONS}
            predicted = {hours: [] for hours in TIMING_CDF_HORIZONS}
            trial_iters: list[int] = []
            for fold in inner:
                train_base = hpo_base[fold.train_idx]
                valid_base = hpo_base[fold.valid_idx]
                train_X, train_y, _ = _expand_at_risk(
                    matrix, train_base, event_interval
                )
                valid_X, valid_y, _ = _expand_at_risk(
                    matrix, valid_base, event_interval
                )
                model, best_iteration = _fit_with_inner_validation(
                    train_X,
                    train_y,
                    np.ones(len(train_y), np.float32),
                    valid_X,
                    valid_y,
                    task_kind="binary",
                    params=params,
                )
                cdf = _predict_cdf(model, matrix, valid_base)
                for index, hours in enumerate(TIMING_CDF_HORIZONS):
                    observed[hours].append(targets[hours][valid_base])
                    predicted[hours].append(cdf[:, index])
                trial_iters.append(best_iteration)
            iterations[int(trial.number)] = trial_iters
            score = _joint_score(
                {
                    hours: np.concatenate(observed[hours])
                    for hours in TIMING_CDF_HORIZONS
                },
                np.column_stack(
                    [np.concatenate(predicted[h]) for h in TIMING_CDF_HORIZONS]
                ),
            )
            if progress_callback:
                progress_callback(
                    "hazard_hpo_trial_complete",
                    {"side": side, "trial": int(trial.number), "score": score},
                )
            return score

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(
            objective, n_trials=int(n_trials), n_jobs=1, show_progress_bar=False
        )
        params = _sanitize_params(
            study.best_params,
            task_kind="binary",
            quantile_alpha=0.80,
            random_state=seed,
            n_jobs=jobs,
        )
        params["n_estimators"] = max(
            25, int(np.median(iterations[int(study.best_trial.number)]))
        )
        models: list[Any] = []
        for fold_i, fold in enumerate(folds):
            train_idx = fold.base_train_idx[
                side_mask[fold.base_train_idx] & mask[fold.base_train_idx]
            ]
            valid_idx = fold.valid_idx[side_mask[fold.valid_idx]]
            train_X, train_y, _ = _expand_at_risk(matrix, train_idx, event_interval)
            model = _make_model("binary", params)
            model.fit(train_X, train_y)
            cdf = _predict_cdf(model, matrix, valid_idx)
            for index, hours in enumerate(TIMING_CDF_HORIZONS):
                oof[hours][valid_idx] = cdf[:, index]
            fold_ids[valid_idx] = fold_i
            models.append(model)
            provenance.append(
                {
                    "side": side,
                    "fold_month": fold.fold_month,
                    "training_rows": int(len(train_idx)),
                    "expanded_at_risk_rows": int(len(train_X)),
                    "validation_rows": int(len(valid_idx)),
                    "train_decision_cutoff": decision.iloc[train_idx].max().isoformat(),
                    "training_label_resolved_max": resolved.iloc[train_idx]
                    .max()
                    .isoformat(),
                    "valid_start": fold.valid_start.isoformat(),
                    "model_sha256": _model_sha256(model),
                }
            )
        final_idx = np.flatnonzero(side_mask & mask & resolved.notna().to_numpy())
        final_X, final_y, _ = _expand_at_risk(matrix, final_idx, event_interval)
        final_model = _make_model("binary", params)
        final_model.fit(final_X, final_y)
        states[side] = {
            "selected_features": features[side],
            "best_params": params,
            "hpo": {
                "trial_count": len(study.trials),
                "best_trial": int(study.best_trial.number),
                "best_value": float(study.best_value),
                "reference_rows": int(len(reference)),
                "sampled_rows": int(len(hpo_base)),
                "contract": "side-local pooled at-risk discrete-time likelihood",
            },
            "oof_models": models,
            "final_model": final_model,
            "final_model_sha256": _model_sha256(final_model),
        }
    oof_mask = np.logical_and.reduce(
        [np.isfinite(oof[hours]) for hours in TIMING_CDF_HORIZONS]
    )
    metrics = {
        hours: _binary_metrics(
            targets[hours][mask & oof_mask], oof[hours][mask & oof_mask]
        )
        for hours in TIMING_CDF_HORIZONS
    }
    return {
        "schema": TIMING_HAZARD_SCHEMA,
        "horizons_hours": TIMING_CDF_HORIZONS,
        "side_models": states,
        "oof_predictions_by_horizon": oof,
        "oof_prediction_mask": oof_mask,
        "oof_fold_ids": fold_ids,
        "oof_metrics_by_horizon": metrics,
        "fold_provenance": provenance,
        "constraint_contract": (
            "CDF is 1-cumulative-product(1-conditional hazard); monotonicity is "
            "structural and no post-hoc projection or OOF labels are used"
        ),
    }


__all__ = [
    "TIMING_HAZARD_SCHEMA",
    "fit_side_local_timing_hazard_family",
]
