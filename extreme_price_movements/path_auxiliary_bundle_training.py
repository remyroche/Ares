"""Orchestration primitives for the five decomposed path-auxiliary bundles.

The runner owns canonical file loading and persistence.  This module owns the
model semantics between those boundaries: which roles share targets, which
feature-selection studies may be reused, side isolation, and composition of
full-row OOF predictions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.path_auxiliary_model_families import (
    HEAD_SPECS_BY_NAME,
    RoleTargets,
    build_role_targets,
    compose_adverse_timing_predictions,
    compose_mae_predictions,
    compose_peak_predictions,
    compose_slope_diagnostic_predictions,
    compose_timing_cdf_predictions,
)
from extreme_price_movements.path_auxiliary_role_training import (
    fit_auxiliary_role_model,
    select_auxiliary_role_features,
)

BUNDLE_TRAINING_SCHEMA = "path_auxiliary_bundle_training_v1"
SIDES: tuple[str, ...] = ("long", "short")
MEANINGFUL_EVENT_ROLE = "meaningful_mfe_event"

# A selection may be reused only when both the target vector and conditioning
# mask are identical.  In particular, the peak mean and q80 share selection,
# but retain separate task-specific HPO and fitted models.
SELECTION_ROLE_SOURCES: Mapping[str, str] = {
    MEANINGFUL_EVENT_ROLE: "peak_mfe_12h_atr.p_hit",
    "peak_conditional_magnitude": "peak_mfe_12h_atr.conditional_mean",
    "timing_hit_by_2h": "time_to_first_meaningful_mfe.hit_by_2h",
    "timing_hit_by_4h": "time_to_first_meaningful_mfe.hit_by_4h",
    "timing_hit_by_8h": "time_to_first_meaningful_mfe.hit_by_8h",
    "mae_if_hit": "mae_before_meaningful_mfe_atr.if_hit",
    "mae_if_no_hit": "mae_before_meaningful_mfe_atr.if_no_hit",
    "legacy_adverse_extreme": (
        "bars_before_price_stops_decreasing.legacy_adverse_extreme"
    ),
    "confirmed_adverse_trough": (
        "bars_before_price_stops_decreasing.confirmed_adverse_trough"
    ),
    "slope_diagnostic": "future_slope_atr_per_hour.diagnostic",
}

SELECTION_GROUP_BY_ROLE: Mapping[str, str] = {
    "peak_mfe_12h_atr.p_hit": MEANINGFUL_EVENT_ROLE,
    "mae_before_meaningful_mfe_atr.p_hit": MEANINGFUL_EVENT_ROLE,
    "time_to_first_meaningful_mfe.hit_by_12h": MEANINGFUL_EVENT_ROLE,
    "peak_mfe_12h_atr.conditional_mean": "peak_conditional_magnitude",
    "peak_mfe_12h_atr.conditional_q80": "peak_conditional_magnitude",
    "time_to_first_meaningful_mfe.hit_by_2h": "timing_hit_by_2h",
    "time_to_first_meaningful_mfe.hit_by_4h": "timing_hit_by_4h",
    "time_to_first_meaningful_mfe.hit_by_8h": "timing_hit_by_8h",
    "mae_before_meaningful_mfe_atr.if_hit": "mae_if_hit",
    "mae_before_meaningful_mfe_atr.if_no_hit": "mae_if_no_hit",
    "bars_before_price_stops_decreasing.legacy_adverse_extreme": (
        "legacy_adverse_extreme"
    ),
    "bars_before_price_stops_decreasing.confirmed_adverse_trough": (
        "confirmed_adverse_trough"
    ),
    "future_slope_atr_per_hour.diagnostic": "slope_diagnostic",
}

ROLE_TASKS: Mapping[str, str] = {
    MEANINGFUL_EVENT_ROLE: "binary",
    "peak_mfe_12h_atr.conditional_mean": "regression",
    "peak_mfe_12h_atr.conditional_q80": "quantile",
    "time_to_first_meaningful_mfe.hit_by_2h": "binary",
    "time_to_first_meaningful_mfe.hit_by_4h": "binary",
    "time_to_first_meaningful_mfe.hit_by_8h": "binary",
    "mae_before_meaningful_mfe_atr.if_hit": "regression",
    "mae_before_meaningful_mfe_atr.if_no_hit": "regression",
    "bars_before_price_stops_decreasing.legacy_adverse_extreme": "regression",
    "bars_before_price_stops_decreasing.confirmed_adverse_trough": "regression",
    "future_slope_atr_per_hour.diagnostic": "regression",
}

HEAD_ROLE_KEYS: Mapping[str, tuple[str, ...]] = {
    "peak_mfe_12h_atr": (
        MEANINGFUL_EVENT_ROLE,
        "peak_mfe_12h_atr.conditional_mean",
        "peak_mfe_12h_atr.conditional_q80",
    ),
    "time_to_first_meaningful_mfe": (
        "time_to_first_meaningful_mfe.hit_by_2h",
        "time_to_first_meaningful_mfe.hit_by_4h",
        "time_to_first_meaningful_mfe.hit_by_8h",
        MEANINGFUL_EVENT_ROLE,
    ),
    "mae_before_meaningful_mfe_atr": (
        MEANINGFUL_EVENT_ROLE,
        "mae_before_meaningful_mfe_atr.if_hit",
        "mae_before_meaningful_mfe_atr.if_no_hit",
    ),
    "bars_before_price_stops_decreasing": (
        "bars_before_price_stops_decreasing.legacy_adverse_extreme",
        "bars_before_price_stops_decreasing.confirmed_adverse_trough",
    ),
    "future_slope_atr_per_hour": ("future_slope_atr_per_hour.diagnostic",),
}


def canonical_role_targets(frame: pd.DataFrame) -> dict[str, RoleTargets]:
    """Return one aligned target per actually fitted canonical role."""

    raw = build_role_targets(frame)
    result = {
        role_name: raw[role_name]
        for role_name in ROLE_TASKS
        if role_name != MEANINGFUL_EVENT_ROLE
    }
    result[MEANINGFUL_EVENT_ROLE] = raw["peak_mfe_12h_atr.p_hit"]
    return result


def select_bundle_feature_contracts(
    X: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    timestamps: Sequence[Any],
    assets: Sequence[Any],
    sides: Sequence[Any],
    archetypes: Sequence[Any],
    mandatory_features_by_side: Mapping[str, Sequence[str]] | None = None,
    random_state: int = 42,
    purge_hours: float = 13.0,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, dict[str, Any]]:
    """Select features once per unique target-and-conditioning contract."""

    targets = canonical_role_targets(labels)
    result: dict[str, dict[str, Any]] = {}
    for group_index, (group, source_role) in enumerate(SELECTION_ROLE_SOURCES.items()):
        role = targets[
            MEANINGFUL_EVENT_ROLE
            if source_role == "peak_mfe_12h_atr.p_hit"
            else source_role
        ]
        eligible = role.train_mask & np.isfinite(role.target)
        if progress_callback is not None:
            progress_callback(
                "selection_start",
                {
                    "selection_group": group,
                    "source_role": source_role,
                    "rows": int(eligible.sum()),
                },
            )
        selection = select_auxiliary_role_features(
            X.loc[eligible].reset_index(drop=True),
            role.target[eligible],
            task_kind=(
                "binary"
                if source_role.startswith("time_to_first_meaningful_mfe")
                or source_role.endswith(".p_hit")
                else "regression"
            ),
            timestamps=np.asarray(timestamps)[eligible],
            assets=np.asarray(assets)[eligible],
            sides=np.asarray(sides)[eligible],
            archetypes=np.asarray(archetypes)[eligible],
            role_name=group,
            sample_weight=np.ones(int(eligible.sum()), dtype=np.float32),
            mandatory_features_by_side=mandatory_features_by_side,
            random_state=int(random_state) + 10007 * group_index,
            purge_hours=float(purge_hours),
        )
        selection["source_role"] = source_role
        selection["eligible_rows"] = int(eligible.sum())
        selection["reuse_contract"] = "same target vector and conditioning mask only"
        result[group] = selection
        if progress_callback is not None:
            progress_callback(
                "selection_complete",
                {
                    "selection_group": group,
                    "selected_features_by_side": {
                        side: len(selection["selected_features_by_side"][side])
                        for side in SIDES
                    },
                },
            )
    return result


def selected_features_for_role(
    selections: Mapping[str, Mapping[str, Any]],
    role_name: str,
    side: str,
) -> list[str]:
    """Resolve a role to its frozen side-local feature contract."""

    if side not in SIDES:
        raise ValueError(f"unknown side: {side}")
    group = (
        MEANINGFUL_EVENT_ROLE
        if role_name == MEANINGFUL_EVENT_ROLE
        else SELECTION_GROUP_BY_ROLE[role_name]
    )
    features = list(map(str, selections[group]["selected_features_by_side"][side]))
    if not features:
        raise ValueError(f"empty selected feature contract for {role_name}/{side}")
    return features


def fit_role_by_side(
    X: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    role_name: str,
    selection_contracts: Mapping[str, Mapping[str, Any]],
    timestamps: Sequence[Any],
    label_resolved_at: Sequence[Any],
    sides: Sequence[Any],
    selection_hpo_reference_end: Any,
    n_trials: int = 40,
    hpo_rows: int = 45_000,
    hpo_patience: int = 12,
    purge_hours: float = 13.0,
    random_state: int = 42,
    preset_params_by_side: Mapping[str, Mapping[str, Any]] | None = None,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Fit one role independently per side and scatter full-row OOF output."""

    if role_name not in ROLE_TASKS:
        raise ValueError(f"unknown fitted auxiliary role: {role_name}")
    targets = canonical_role_targets(labels)[role_name]
    side_values = np.asarray(sides).astype(str)
    if len(side_values) != len(labels) or len(X) != len(labels):
        raise ValueError("X, labels, and side values must be exactly aligned")
    full_prediction = np.full(len(labels), np.nan, dtype=np.float32)
    full_fold_ids = np.full(len(labels), -1, dtype=np.int16)
    side_results: dict[str, Any] = {}
    for side_index, side in enumerate(SIDES):
        side_idx = np.flatnonzero(side_values == side)
        if not len(side_idx):
            raise ValueError(f"role {role_name} has no {side} rows")
        if progress_callback is not None:
            progress_callback(
                "role_side_start",
                {"role": role_name, "side": side, "rows": int(len(side_idx))},
            )
        fitted = fit_role_for_side(
            X.iloc[side_idx],
            labels.iloc[side_idx].reset_index(drop=True),
            role_name=role_name,
            side=side,
            selection_contracts=selection_contracts,
            timestamps=np.asarray(timestamps)[side_idx],
            label_resolved_at=np.asarray(label_resolved_at)[side_idx],
            selection_hpo_reference_end=selection_hpo_reference_end,
            n_trials=int(n_trials),
            hpo_rows=int(hpo_rows),
            hpo_patience=int(hpo_patience),
            random_state=int(random_state) + 1009 * (side_index + 1),
            purge_hours=float(purge_hours),
            preset_params=(
                dict(preset_params_by_side[side])
                if preset_params_by_side is not None
                else None
            ),
            progress_callback=progress_callback,
        )
        full_prediction[side_idx] = fitted["oof_predictions"]
        full_fold_ids[side_idx] = fitted["oof_fold_ids"]
        side_results[side] = fitted
        if progress_callback is not None:
            progress_callback(
                "role_side_complete",
                {
                    "role": role_name,
                    "side": side,
                    "oof_rows": int(fitted["oof_prediction_mask"].sum()),
                },
            )
    return {
        "schema": BUNDLE_TRAINING_SCHEMA,
        "role_name": role_name,
        "task_kind": ROLE_TASKS[role_name],
        "target_source_column": targets.source_column,
        "role_train_mask": targets.train_mask,
        "valid_mask": targets.valid_mask,
        "target": targets.target,
        "oof_predictions": full_prediction,
        "oof_fold_ids": full_fold_ids,
        "oof_prediction_mask": np.isfinite(full_prediction),
        "side_results": side_results,
        "sample_weight_contract": (
            "unit weights preserve the declared probability/conditional-mean "
            "estimand; outcome-weighted variants require a separate OOF ablation"
        ),
    }


def fit_role_for_side(
    X: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    role_name: str,
    side: str,
    selection_contracts: Mapping[str, Mapping[str, Any]],
    timestamps: Sequence[Any],
    label_resolved_at: Sequence[Any],
    selection_hpo_reference_end: Any,
    n_trials: int = 40,
    hpo_rows: int = 45_000,
    hpo_patience: int = 12,
    purge_hours: float = 13.0,
    random_state: int = 42,
    preset_params: Mapping[str, Any] | None = None,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Fit one role for one side, suitable for role/side checkpointing."""

    if role_name not in ROLE_TASKS:
        raise ValueError(f"unknown fitted auxiliary role: {role_name}")
    if side not in SIDES:
        raise ValueError(f"unknown side: {side}")
    if len(X) != len(labels):
        raise ValueError("X and labels must be exactly aligned")
    targets = canonical_role_targets(labels)[role_name]
    features = selected_features_for_role(selection_contracts, role_name, side)
    return fit_auxiliary_role_model(
        X,
        targets.target,
        role_train_mask=targets.train_mask,
        task_kind=ROLE_TASKS[role_name],  # type: ignore[arg-type]
        selected_features=features,
        timestamps=timestamps,
        label_resolved_at=label_resolved_at,
        selection_hpo_reference_end=selection_hpo_reference_end,
        sample_weight=np.ones(len(labels), dtype=np.float32),
        n_trials=int(n_trials),
        hpo_patience=int(hpo_patience),
        random_state=int(random_state),
        purge_hours=float(purge_hours),
        preset_params=dict(preset_params) if preset_params is not None else None,
        hpo_rows=int(hpo_rows),
        role_name=role_name,
        progress_callback=progress_callback,
    )


def _common_oof_mask(
    role_results: Mapping[str, Mapping[str, Any]], role_names: Sequence[str]
) -> np.ndarray:
    masks = [
        np.asarray(role_results[name]["oof_prediction_mask"], dtype=bool)
        for name in role_names
    ]
    if not masks:
        raise ValueError("at least one role is required for head composition")
    if len({len(mask) for mask in masks}) != 1:
        raise ValueError("role OOF masks are not aligned")
    return np.logical_and.reduce(masks)


def compose_head_oof(
    head_name: str,
    role_results: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Compose one head's natural-unit outputs on the common OOF population."""

    if head_name not in HEAD_ROLE_KEYS or head_name not in HEAD_SPECS_BY_NAME:
        raise ValueError(f"unknown auxiliary head: {head_name}")
    roles = HEAD_ROLE_KEYS[head_name]
    missing = [role for role in roles if role not in role_results]
    if missing:
        raise ValueError(f"head {head_name} is missing fitted roles: {missing}")
    mask = _common_oof_mask(role_results, roles)
    output = pd.DataFrame(
        {"oof_prediction_available": mask},
        index=np.arange(len(mask)),
    )

    def prediction(role: str) -> np.ndarray:
        return np.asarray(role_results[role]["oof_predictions"], dtype=np.float64)[mask]

    composed: Mapping[str, np.ndarray]
    if head_name == "peak_mfe_12h_atr":
        composed = compose_peak_predictions(
            np.clip(prediction(MEANINGFUL_EVENT_ROLE), 0.0, 1.0),
            np.clip(prediction("peak_mfe_12h_atr.conditional_mean"), 0.0, 10.0),
            np.clip(prediction("peak_mfe_12h_atr.conditional_q80"), 0.0, 10.0),
        )
    elif head_name == "time_to_first_meaningful_mfe":
        composed = compose_timing_cdf_predictions(
            {
                2: np.clip(
                    prediction("time_to_first_meaningful_mfe.hit_by_2h"),
                    0.0,
                    1.0,
                ),
                4: np.clip(
                    prediction("time_to_first_meaningful_mfe.hit_by_4h"),
                    0.0,
                    1.0,
                ),
                8: np.clip(
                    prediction("time_to_first_meaningful_mfe.hit_by_8h"),
                    0.0,
                    1.0,
                ),
                12: np.clip(prediction(MEANINGFUL_EVENT_ROLE), 0.0, 1.0),
            }
        )
    elif head_name == "mae_before_meaningful_mfe_atr":
        composed = compose_mae_predictions(
            np.clip(prediction(MEANINGFUL_EVENT_ROLE), 0.0, 1.0),
            np.clip(prediction("mae_before_meaningful_mfe_atr.if_hit"), 0.0, 10.0),
            np.clip(
                prediction("mae_before_meaningful_mfe_atr.if_no_hit"),
                0.0,
                10.0,
            ),
        )
    elif head_name == "bars_before_price_stops_decreasing":
        composed = compose_adverse_timing_predictions(
            np.clip(
                prediction("bars_before_price_stops_decreasing.legacy_adverse_extreme"),
                0.0,
                12.0,
            ),
            np.clip(
                prediction(
                    "bars_before_price_stops_decreasing.confirmed_adverse_trough"
                ),
                0.0,
                12.0,
            ),
        )
    else:
        composed = compose_slope_diagnostic_predictions(
            np.clip(prediction("future_slope_atr_per_hour.diagnostic"), 0.0, 8.0)
        )
    for column, values in composed.items():
        full = np.full(
            len(mask),
            np.nan,
            dtype=object if values.dtype.kind in "OUS" else np.float64,
        )
        full[mask] = values
        output[column] = full
    return output
