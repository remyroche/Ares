#!/usr/bin/env python3
"""Fit and persist the five side-local decomposed path-auxiliary bundles."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.base_candidate_population import (  # noqa: E402
    candidate_identity_sha256,
)
from extreme_price_movements.path_auxiliary_bundle_training import (  # noqa: E402
    BUNDLE_TRAINING_SCHEMA,
    HEAD_ROLE_KEYS,
    MEANINGFUL_EVENT_ROLE,
    ROLE_TASKS,
    SELECTION_GROUP_BY_ROLE,
    SELECTION_ROLE_SOURCES,
    canonical_role_targets,
    compose_head_oof,
    fit_role_for_side,
    select_bundle_feature_contracts,
    selected_features_for_role,
)
from extreme_price_movements.path_auxiliary_lgbm import (  # noqa: E402
    configured_auxiliary_feature_universe,
    fit_base_archetype_label_feature_contract,
    transform_base_archetype_label_features,
)
from extreme_price_movements.path_auxiliary_model_families import (  # noqa: E402
    HEAD_SPECS,
    MEANINGFUL_HIT_COLUMN,
    MODEL_FAMILY_SCHEMA,
    TIMING_COLUMN,
    probability_calibration_metrics,
    regression_metrics,
)
from extreme_price_movements.path_auxiliary_role_training import (  # noqa: E402
    FIXED_MAY_JULY_OOF_MONTHS,
    ROLE_TRAINER_SCHEMA,
)
from extreme_price_movements.path_auxiliary_timing_training import (  # noqa: E402
    TIMING_CDF_TRAINER_SCHEMA,
    fit_side_local_timing_cdf_family,
)
from extreme_price_movements.training_resource_guard import (  # noqa: E402
    TrainingResourceGuard,
)
from scripts.run_path_auxiliary_lgbm_models import (  # noqa: E402
    ARCHETYPE_COLUMNS,
    DEFAULT_LABEL_RESOLUTION_COLUMN,
    MANDATORY_HANDOFF_MODEL_FEATURES,
    REPRESENTATION_AVAILABLE_FEATURE,
    STRICT_IDENTITY_COLUMNS,
    _archetype_context,
    _atomic_joblib_dump,
    _atomic_to_parquet,
    _build_resource_guard,
    _complete_archetype_source,
    _file_sha256,
    _join_archetype_context,
    _load_labels,
    _load_static_features,
    _overlay_handoff_model_features,
    _resource_guard_contract,
    _stable_sha256,
    _static_feature_columns,
    _timestamp_bounds,
    _tree_stat_signature,
    _write_json,
)

RUNNER_SCHEMA = "run_path_auxiliary_role_bundles_v1"
CHECKPOINT_SCHEMA = "path_auxiliary_role_bundle_checkpoint_v1"
PROMOTION_GATE_SCHEMA = "path_auxiliary_promotion_gate_v2"
CANONICAL_REFERENCE_END = pd.Timestamp("2026-05-01T00:00:00Z")
SIDES: tuple[str, ...] = ("long", "short")

# The q80 component is an optional upside diagnostic, not an input required to
# form the peak hurdle mean.  A broad tolerance is deliberate: it allows a
# finite OOF sample some sampling error while refusing a clearly saturated or
# materially under-covering quantile.  Only well-supported side x
# representation slices participate; a tiny slice cannot promote a component.
Q80_COVERAGE_TARGET = 0.80
Q80_COVERAGE_TOLERANCE = 0.10
Q80_MIN_SUPPORTED_SLICE_ROWS = 100

DEFAULT_LABELS = (
    ROOT
    / "data_perp/artifacts/packb_path_auxiliary_targets_20260725_v1_31_8/targets.parquet"
)
DEFAULT_CONTEXT = (
    ROOT
    / "data_perp/artifacts/packb_downstream_context_20260725_v2_31_8_frozen_ae_gmm/context.parquet"
)
DEFAULT_FEATURE_DIR = ROOT / "data_perp/features/20260711_070000"
DEFAULT_OUTPUT = (
    ROOT / "data_perp/artifacts/packb_path_auxiliary_role_bundles_20260725_v1_31_8"
)

HEAD_PREDICTION_RENAMES: Mapping[str, Mapping[str, str]] = {
    "peak_mfe_12h_atr": {
        "p_hit": "pred_p_meaningful_mfe_12h",
        "conditional_mean_atr": "pred_peak_mfe_if_hit_mean_atr",
        "conditional_q80_atr": "pred_peak_mfe_if_hit_q80_atr",
        "expected_peak_mfe_atr": "pred_expected_peak_mfe_atr",
    },
    "time_to_first_meaningful_mfe": {
        "p_hit_by_2h": "pred_p_hit_by_2h",
        "p_hit_by_4h": "pred_p_hit_by_4h",
        "p_hit_by_8h": "pred_p_hit_by_8h",
        "p_hit_by_12h": "pred_p_hit_by_12h",
        "expected_censored_time_hours": ("pred_expected_censored_time_hours"),
    },
    "mae_before_meaningful_mfe_atr": {
        "p_hit": "pred_p_meaningful_mfe_12h",
        "mae_if_hit_atr": "pred_mae_if_hit_atr",
        "mae_if_no_hit_atr": "pred_mae_if_no_hit_atr",
        "expected_mae_atr": "pred_expected_mae_atr",
    },
    "bars_before_price_stops_decreasing": {
        "legacy_adverse_extreme_bars": ("pred_legacy_adverse_extreme_bars"),
        "confirmed_adverse_trough_bars": ("pred_confirmed_adverse_trough_bars"),
        "confirmed_minus_legacy_bars": ("diag_confirmed_minus_legacy_bars"),
    },
    "future_slope_atr_per_hour": {
        "future_slope_atr_per_hour": ("diag_pred_future_slope_atr_per_hour"),
    },
}


def _role_path_name(role_name: str) -> str:
    return role_name.replace(".", "__")


def _checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "checkpoint.json"


def _load_checkpoint(
    output_dir: Path,
    *,
    run_fingerprint: Mapping[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    path = _checkpoint_path(output_dir)
    existing = (
        [
            entry
            for entry in output_dir.iterdir()
            if entry.name != "training_resource_telemetry.jsonl"
        ]
        if output_dir.exists()
        else []
    )
    if existing:
        if path.is_file():
            checkpoint = json.loads(path.read_text(encoding="utf-8"))
            if (
                checkpoint.get("schema") != CHECKPOINT_SCHEMA
                or checkpoint.get("run_fingerprint", {}).get("sha256")
                != run_fingerprint["sha256"]
            ):
                raise ValueError(
                    "existing role-bundle checkpoint has a mismatched contract"
                )
            return checkpoint
        if not overwrite:
            raise FileExistsError(
                f"non-empty output has no resumable checkpoint: {output_dir}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "run_fingerprint": dict(run_fingerprint),
        "selection": None,
        "roles": {},
        "heads": {},
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    _write_json(path, checkpoint)
    return checkpoint


def _save_checkpoint(output_dir: Path, checkpoint: Mapping[str, Any]) -> None:
    payload = dict(checkpoint)
    payload["updated_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    _write_json(_checkpoint_path(output_dir), payload)


def _artifact_record(path: Path, *, kind: str) -> dict[str, str]:
    return {
        "kind": kind,
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
    }


def _validate_record(record: Mapping[str, Any], *, kind: str) -> Path:
    if record.get("kind") != kind:
        raise ValueError(f"checkpoint artifact kind mismatch: expected {kind}")
    path = Path(str(record.get("path")))
    if not path.is_file() or record.get("sha256") != _file_sha256(path):
        raise ValueError(f"checkpoint artifact is missing or corrupt: {path}")
    return path


def _guard_callback(
    guard: TrainingResourceGuard,
    *,
    prefix: str,
):
    def callback(event: str, payload: Mapping[str, Any]) -> None:
        suffix = ":".join(
            str(payload[key])
            for key in ("role", "trial", "fold", "fold_month")
            if key in payload
        )
        stage = f"{prefix}:{event}" + (f":{suffix}" if suffix else "")
        guard.checkpoint(stage)

    return callback


def _full_selected_matrix(
    labels: pd.DataFrame,
    *,
    selected_features: Sequence[str],
    feature_dir: Path,
    handoff_feature_columns: Sequence[str],
    archetype_features: pd.DataFrame,
    guard: TrainingResourceGuard,
    stage: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    features = list(dict.fromkeys(map(str, selected_features)))
    guard.preflight(f"{stage}:feature_load")
    matrix, report = _load_static_features(
        labels,
        feature_dir=feature_dir,
        requested_features=features,
        read_cache=None,
    )
    matrix, report = _overlay_handoff_model_features(
        matrix,
        labels,
        requested_features=features,
        static_report=report,
        handoff_feature_columns=handoff_feature_columns,
    )
    archetype_selected = [
        feature for feature in features if feature in archetype_features.columns
    ]
    if archetype_selected:
        matrix.loc[:, archetype_selected] = archetype_features.loc[
            :, archetype_selected
        ].to_numpy(dtype=np.float32, copy=False)
    matrix = matrix.reindex(columns=features).astype(np.float32, copy=False)
    if REPRESENTATION_AVAILABLE_FEATURE not in matrix:
        raise ValueError(
            f"{REPRESENTATION_AVAILABLE_FEATURE} is a mandatory role input"
        )
    availability = matrix[REPRESENTATION_AVAILABLE_FEATURE].to_numpy(dtype=np.float32)
    if (
        not np.isfinite(availability).all()
        or not np.isin(availability, (0.0, 1.0)).all()
    ):
        raise ValueError(f"{REPRESENTATION_AVAILABLE_FEATURE} must be finite binary")
    guard.checkpoint(f"{stage}:feature_load_complete")
    return matrix, report


def _selection_contracts(
    labels: pd.DataFrame,
    *,
    reference_mask: np.ndarray,
    selection_rows: int,
    requested_features: Sequence[str],
    feature_dir: Path,
    handoff_feature_columns: Sequence[str],
    archetype_features: pd.DataFrame,
    mandatory_features_by_side: Mapping[str, Sequence[str]],
    seed: int,
    output_dir: Path,
    checkpoint: dict[str, Any],
    guard: TrainingResourceGuard,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    record = checkpoint.get("selection")
    if isinstance(record, Mapping):
        path = _validate_record(record, kind="selection_contracts")
        payload = joblib.load(path)
        return dict(payload["selection_contracts"]), dict(payload["report"])

    from extreme_price_movements.lgbm_pipeline import (
        _time_spread_subsample_indices,
    )

    reference_positions = np.flatnonzero(reference_mask)
    relative = _time_spread_subsample_indices(
        np.arange(len(reference_positions), dtype=np.float32),
        max_n=max(300, min(int(selection_rows), len(reference_positions))),
        random_state=int(seed),
        classifier=False,
        timestamps=labels.iloc[reference_positions]["__ts__"].to_numpy(),
    )
    selection_idx = reference_positions[relative]
    selection_labels = labels.iloc[selection_idx].reset_index(drop=True)
    guard.preflight("role_selection:wide_feature_load")
    matrix, static_report = _load_static_features(
        selection_labels,
        feature_dir=feature_dir,
        requested_features=requested_features,
        read_cache=None,
        sampled_periods=True,
    )
    matrix, static_report = _overlay_handoff_model_features(
        matrix,
        selection_labels,
        requested_features=requested_features,
        static_report=static_report,
        handoff_feature_columns=handoff_feature_columns,
    )
    archetype_available = [
        feature
        for feature in archetype_features.columns
        if feature in requested_features
    ]
    if archetype_available:
        matrix.loc[:, archetype_available] = (
            archetype_features.iloc[selection_idx]
            .loc[:, archetype_available]
            .to_numpy(dtype=np.float32, copy=False)
        )
    source_available = sorted(
        set(map(str, static_report.get("available_feature_names", []))).union(
            archetype_available
        )
    )
    static_report = dict(static_report)
    static_report["available_feature_names"] = source_available
    static_report["available_features"] = int(len(source_available))
    static_report["missing_features"] = sorted(
        set(map(str, requested_features)) - set(source_available)
    )
    static_report["archetype_overlay_features"] = sorted(archetype_available)
    available, universe_report = configured_auxiliary_feature_universe(source_available)
    matrix = matrix.reindex(columns=available).astype(np.float32, copy=False)
    partial_path = output_dir / "shared/selection_contracts.partial.joblib"
    partial_contracts: dict[str, dict[str, Any]] = {}
    partial_record = checkpoint.get("selection_partial")
    if isinstance(partial_record, Mapping):
        loaded_partial_path = _validate_record(
            partial_record, kind="selection_contracts_partial"
        )
        partial_payload = joblib.load(loaded_partial_path)
        partial_contracts = {
            str(group): dict(contract)
            for group, contract in partial_payload["selection_contracts"].items()
        }

    def persist_partial_selection(
        group: str,
        _contract: Mapping[str, Any],
        current_contracts: Mapping[str, Mapping[str, Any]],
    ) -> None:
        _atomic_joblib_dump(
            {
                "selection_contracts": dict(current_contracts),
                "completed_groups": list(current_contracts),
            },
            partial_path,
        )
        checkpoint["selection_partial"] = _artifact_record(
            partial_path, kind="selection_contracts_partial"
        )
        _save_checkpoint(output_dir, checkpoint)
        guard.checkpoint(f"role_selection:{group}:partial_persisted")

    contracts = select_bundle_feature_contracts(
        matrix,
        selection_labels,
        timestamps=selection_labels["__ts__"].to_numpy(),
        assets=selection_labels["__symbol__"].to_numpy(),
        sides=selection_labels["side"].to_numpy(),
        archetypes=_archetype_context(selection_labels).to_numpy(),
        mandatory_features_by_side=mandatory_features_by_side,
        random_state=int(seed),
        purge_hours=13.0,
        progress_callback=_guard_callback(guard, prefix="role_selection"),
        existing_contracts=partial_contracts,
        contract_callback=persist_partial_selection,
    )
    report = {
        "selection_rows": int(len(selection_labels)),
        "selection_indices_sha256": _stable_sha256({"indices": selection_idx.tolist()}),
        "selection_decision_bounds": _timestamp_bounds(selection_labels["__ts__"]),
        "static_feature_report": static_report,
        "loaded_universe": universe_report,
        "selection_groups": {
            group: {
                "source_role": contract["source_role"],
                "eligible_rows": contract["eligible_rows"],
                "selected_features_by_side": contract["selected_features_by_side"],
            }
            for group, contract in contracts.items()
        },
    }
    path = output_dir / "shared/selection_contracts.joblib"
    _atomic_joblib_dump({"selection_contracts": contracts, "report": report}, path)
    checkpoint["selection"] = _artifact_record(path, kind="selection_contracts")
    checkpoint.pop("selection_partial", None)
    if partial_path.exists():
        partial_path.unlink()
    _save_checkpoint(output_dir, checkpoint)
    del matrix
    gc.collect()
    return contracts, report


def _role_artifact_path(output_dir: Path, role_name: str, side: str) -> Path:
    owner = (
        output_dir / "shared/meaningful_mfe_event"
        if role_name == MEANINGFUL_EVENT_ROLE
        else output_dir / "roles" / _role_path_name(role_name)
    )
    return owner / side / "role_bundle.joblib"


def _fit_or_load_role_side(
    X_side: pd.DataFrame,
    labels_side: pd.DataFrame,
    *,
    role_name: str,
    side: str,
    selection_contracts: Mapping[str, Mapping[str, Any]],
    reference_end: pd.Timestamp,
    n_trials: int,
    hpo_rows: int,
    hpo_patience: int,
    purge_hours: float,
    seed: int,
    output_dir: Path,
    checkpoint: dict[str, Any],
    guard: TrainingResourceGuard,
) -> dict[str, Any]:
    state = checkpoint.setdefault("roles", {}).setdefault(role_name, {}).get(side)
    if isinstance(state, Mapping):
        path = _validate_record(state, kind="role_side_bundle")
        return joblib.load(path)
    guard.preflight(f"role:{role_name}:{side}:fit")
    fitted = fit_role_for_side(
        X_side,
        labels_side,
        role_name=role_name,
        side=side,
        selection_contracts=selection_contracts,
        timestamps=labels_side["__ts__"].to_numpy(),
        label_resolved_at=labels_side[DEFAULT_LABEL_RESOLUTION_COLUMN].to_numpy(),
        selection_hpo_reference_end=reference_end,
        n_trials=int(n_trials),
        hpo_rows=int(hpo_rows),
        hpo_patience=int(hpo_patience),
        purge_hours=float(purge_hours),
        random_state=int(seed),
        progress_callback=_guard_callback(guard, prefix=f"role:{role_name}:{side}"),
    )
    path = _role_artifact_path(output_dir, role_name, side)
    _atomic_joblib_dump(fitted, path)
    checkpoint.setdefault("roles", {}).setdefault(role_name, {})[side] = (
        _artifact_record(path, kind="role_side_bundle")
    )
    _save_checkpoint(output_dir, checkpoint)
    guard.checkpoint(f"role:{role_name}:{side}:persisted")
    return fitted


def _timing_family_artifact_path(output_dir: Path, side: str) -> Path:
    return (
        output_dir / "shared/meaningful_mfe_event" / side / "timing_cdf_family.joblib"
    )


def _fit_or_load_timing_side(
    X_side: pd.DataFrame,
    labels_side: pd.DataFrame,
    *,
    side: str,
    selection_contracts: Mapping[str, Mapping[str, Any]],
    reference_end: pd.Timestamp,
    n_trials: int,
    hpo_rows: int,
    seed: int,
    output_dir: Path,
    checkpoint: dict[str, Any],
    guard: TrainingResourceGuard,
) -> dict[str, Any]:
    state = checkpoint.setdefault("timing_family", {}).get(side)
    if isinstance(state, Mapping):
        path = _validate_record(state, kind="timing_cdf_family")
        return joblib.load(path)
    timing_target = canonical_role_targets(labels_side)[
        "time_to_first_meaningful_mfe.hit_by_2h"
    ]
    features = {
        side: {
            2: selected_features_for_role(
                selection_contracts,
                "time_to_first_meaningful_mfe.hit_by_2h",
                side,
            ),
            4: selected_features_for_role(
                selection_contracts,
                "time_to_first_meaningful_mfe.hit_by_4h",
                side,
            ),
            8: selected_features_for_role(
                selection_contracts,
                "time_to_first_meaningful_mfe.hit_by_8h",
                side,
            ),
            12: selected_features_for_role(
                selection_contracts, MEANINGFUL_EVENT_ROLE, side
            ),
        }
    }
    guard.preflight(f"timing_cdf:{side}:fit")
    family = fit_side_local_timing_cdf_family(
        X_side,
        labels_side[TIMING_COLUMN].to_numpy(dtype=np.float32, copy=False),
        labels_side[MEANINGFUL_HIT_COLUMN].to_numpy(dtype=np.float32, copy=False),
        timing_train_mask=timing_target.train_mask,
        sides=np.repeat(side, len(labels_side)),
        selected_features=features,
        timestamps=labels_side["__ts__"].to_numpy(),
        label_resolved_at=labels_side[DEFAULT_LABEL_RESOLUTION_COLUMN].to_numpy(),
        selection_hpo_reference_end=reference_end,
        sample_weight=np.ones(len(labels_side), dtype=np.float32),
        n_trials=int(n_trials),
        hpo_rows=int(hpo_rows),
        random_state=int(seed),
        progress_callback=_guard_callback(guard, prefix=f"timing_cdf:{side}"),
    )
    path = _timing_family_artifact_path(output_dir, side)
    _atomic_joblib_dump(family, path)
    record = _artifact_record(path, kind="timing_cdf_family")
    checkpoint.setdefault("timing_family", {})[side] = record
    for role_name in (
        MEANINGFUL_EVENT_ROLE,
        "time_to_first_meaningful_mfe.hit_by_2h",
        "time_to_first_meaningful_mfe.hit_by_4h",
        "time_to_first_meaningful_mfe.hit_by_8h",
    ):
        checkpoint.setdefault("roles", {}).setdefault(role_name, {})[side] = record
    _save_checkpoint(output_dir, checkpoint)
    guard.checkpoint(f"timing_cdf:{side}:persisted")
    return family


def _timing_role_results(
    labels: pd.DataFrame,
    *,
    families_by_side: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    role_by_horizon = {
        2: "time_to_first_meaningful_mfe.hit_by_2h",
        4: "time_to_first_meaningful_mfe.hit_by_4h",
        8: "time_to_first_meaningful_mfe.hit_by_8h",
        12: MEANINGFUL_EVENT_ROLE,
    }
    targets = canonical_role_targets(labels)
    side_values = labels["side"].astype(str).to_numpy()
    results: dict[str, dict[str, Any]] = {}
    for hours, role_name in role_by_horizon.items():
        prediction = np.full(len(labels), np.nan, dtype=np.float32)
        fold_ids = np.full(len(labels), -1, dtype=np.int16)
        side_results: dict[str, Any] = {}
        for side in SIDES:
            rows = np.flatnonzero(side_values == side)
            family = families_by_side[side]
            state = family["side_models"][side]
            prediction[rows] = family["oof_predictions_by_horizon"][hours]
            fold_ids[rows] = family["oof_fold_ids"]
            folds = []
            for fold in family["fold_provenance"]:
                if fold["side"] != side:
                    continue
                local = dict(fold)
                local["model_sha256"] = local["model_sha256_by_horizon"][str(hours)]
                folds.append(local)
            final_contract = dict(state["final_refit_contract"])
            final_contract["model_sha256"] = final_contract["model_sha256_by_horizon"][
                str(hours)
            ]
            side_results[side] = {
                "best_params": state["best_params"],
                "hpo": state["hpo"],
                "selected_features": state["selected_features_by_horizon"][hours],
                "oof_predictions": family["oof_predictions_by_horizon"][hours],
                "oof_fold_ids": family["oof_fold_ids"],
                "oof_prediction_mask": family["oof_prediction_mask"],
                "oof_metrics": family["oof_metrics_by_horizon"][hours],
                "fold_provenance": folds,
                "final_refit_contract": final_contract,
                "hpo_group_id": "timing_cdf_shared_2_4_8_12",
                "timing_family_schema": TIMING_CDF_TRAINER_SCHEMA,
            }
        target = targets[role_name]
        results[role_name] = {
            "role_name": role_name,
            "task_kind": "binary",
            "target_source_column": target.source_column,
            "target": target.target,
            "role_train_mask": target.train_mask,
            "valid_mask": target.valid_mask,
            "oof_predictions": prediction,
            "oof_fold_ids": fold_ids,
            "oof_prediction_mask": np.isfinite(prediction),
            "side_results": side_results,
            "hpo_group_id": "timing_cdf_shared_2_4_8_12",
        }
    event = results[MEANINGFUL_EVENT_ROLE]["oof_predictions"]
    timing_12 = np.concatenate(
        [
            np.asarray(
                families_by_side[side]["oof_predictions_by_horizon"][12],
                dtype=np.float32,
            )
            for side in SIDES
        ]
    )
    # The concatenated order is side-major while canonical rows are timestamp
    # major, so equality is asserted within each side above and again by index.
    del timing_12
    for side in SIDES:
        rows = np.flatnonzero(side_values == side)
        if not np.array_equal(
            event[rows],
            families_by_side[side]["oof_predictions_by_horizon"][12],
            equal_nan=True,
        ):
            raise AssertionError(f"shared event OOF differs from timing 12h for {side}")
    return results


def _scatter_role_result(
    labels: pd.DataFrame,
    *,
    role_name: str,
    side_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    side_values = labels["side"].astype(str).to_numpy()
    predictions = np.full(len(labels), np.nan, dtype=np.float32)
    fold_ids = np.full(len(labels), -1, dtype=np.int16)
    target = canonical_role_targets(labels)[role_name]
    for side in SIDES:
        rows = np.flatnonzero(side_values == side)
        result = side_results[side]
        if len(result["oof_predictions"]) != len(rows):
            raise ValueError(f"{role_name}/{side} result is misaligned")
        predictions[rows] = result["oof_predictions"]
        fold_ids[rows] = result["oof_fold_ids"]
    return {
        "role_name": role_name,
        "task_kind": ROLE_TASKS[role_name],
        "target_source_column": target.source_column,
        "target": target.target,
        "role_train_mask": target.train_mask,
        "valid_mask": target.valid_mask,
        "oof_predictions": predictions,
        "oof_fold_ids": fold_ids,
        "oof_prediction_mask": np.isfinite(predictions),
        "side_results": dict(side_results),
    }


def _representation_role_metrics(
    labels: pd.DataFrame,
    role_result: Mapping[str, Any],
) -> dict[str, Any]:
    availability = labels[REPRESENTATION_AVAILABLE_FEATURE].to_numpy(dtype=np.float32)
    sides = labels["side"].astype(str).to_numpy()
    target = np.asarray(role_result["target"], dtype=np.float64)
    prediction = np.asarray(role_result["oof_predictions"], dtype=np.float64)
    eligible = np.asarray(role_result["role_train_mask"], dtype=bool)
    task = str(role_result["task_kind"])
    report: dict[str, Any] = {}
    for side in SIDES:
        report[side] = {}
        for label, value in (("available", 1.0), ("missing", 0.0)):
            canonical = (sides == side) & (availability == value)
            metric_mask = (
                canonical & eligible & np.isfinite(target) & np.isfinite(prediction)
            )
            support = int(metric_mask.sum())
            if task == "binary":
                metrics = probability_calibration_metrics(
                    target, np.clip(prediction, 0.0, 1.0), mask=metric_mask
                )
            else:
                metrics = regression_metrics(target, prediction, mask=metric_mask)
                if task == "quantile" and support:
                    observed = target[metric_mask]
                    estimate = prediction[metric_mask]
                    residual = observed - estimate
                    metrics["pinball_loss_alpha_0_8"] = float(
                        np.mean(np.maximum(0.8 * residual, (0.8 - 1.0) * residual))
                    )
                    metrics["empirical_coverage_alpha_0_8"] = float(
                        np.mean(observed <= estimate)
                    )
            report[side][label] = {
                "canonical_rows": int(canonical.sum()),
                "oof_prediction_rows": int((canonical & np.isfinite(prediction)).sum()),
                "target_eligible_rows": int((canonical & eligible).sum()),
                "metric_rows": support,
                "status": (
                    "evaluated" if support else "not_evaluable_zero_missing_support"
                ),
                "metrics": metrics,
            }
    return report


def _metric_slice_evidence(
    report: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    metric_names: Sequence[str],
) -> list[dict[str, Any]]:
    """Return compact, JSON-safe per-side representation evidence.

    Promotion must not silently collapse a material representation-missing
    population into an aggregate score.  This deliberately preserves every
    evaluated side x availability slice alongside the particular metrics used
    by the component gate.
    """

    evidence: list[dict[str, Any]] = []
    for side in SIDES:
        for availability in ("available", "missing"):
            slice_report = report.get(side, {}).get(availability, {})
            metrics = slice_report.get("metrics", {})
            values = {
                name: metrics.get(name) for name in metric_names if name in metrics
            }
            evidence.append(
                {
                    "side": side,
                    "representation": availability,
                    "metric_rows": int(slice_report.get("metric_rows", 0)),
                    "status": str(slice_report.get("status", "missing")),
                    "metrics": values,
                }
            )
    return evidence


def _q80_component_quality(
    report: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Gate peak q80 on its declared empirical-coverage estimand.

    This is intentionally a component gate rather than a whole-head gate:
    the hurdle probability and conditional mean can still be investigated
    while a miscalibrated optional q80 column is withheld.
    """

    evidence = _metric_slice_evidence(
        report,
        metric_names=(
            "empirical_coverage_alpha_0_8",
            "pinball_loss_alpha_0_8",
            "bias",
            "spearman_ic",
        ),
    )
    lower = Q80_COVERAGE_TARGET - Q80_COVERAGE_TOLERANCE
    upper = Q80_COVERAGE_TARGET + Q80_COVERAGE_TOLERANCE
    supported: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in evidence:
        coverage = row["metrics"].get("empirical_coverage_alpha_0_8")
        if row["metric_rows"] < Q80_MIN_SUPPORTED_SLICE_ROWS:
            continue
        if coverage is None or not np.isfinite(float(coverage)):
            failures.append({**row, "failure": "non_finite_coverage"})
            continue
        checked = {**row, "coverage": float(coverage)}
        supported.append(checked)
        if not lower <= float(coverage) <= upper:
            failures.append({**checked, "failure": "coverage_outside_tolerance"})
    if not supported:
        status = "WITHHELD_INSUFFICIENT_SUPPORTED_Q80_EVIDENCE"
    elif failures:
        status = "WITHHELD_MISCALIBRATED_Q80"
    else:
        status = "RESEARCH_ABLATION_CANDIDATE_Q80"
    return {
        "status": status,
        "estimand": "conditional_q80_peak_mfe_atr",
        "target_coverage": Q80_COVERAGE_TARGET,
        "accepted_coverage_interval": [lower, upper],
        "minimum_supported_slice_rows": Q80_MIN_SUPPORTED_SLICE_ROWS,
        "supported_slices": supported,
        "failing_slices": failures,
        "all_slice_evidence": evidence,
    }


def _promotion_contract(
    head_name: str,
    role_metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Build a fail-closed research-vs-production promotion contract.

    Auxiliary OOF columns may be consumed by an execution-EV *research*
    ablation.  That is not production promotion.  All production paths remain
    pending an identical-row, per-side, cost-aware execution-EV comparison.
    """

    if head_name == "peak_mfe_12h_atr":
        q80 = _q80_component_quality(role_metrics["peak_mfe_12h_atr.conditional_q80"])
        research_columns = [
            "pred_p_meaningful_mfe_12h",
            "pred_peak_mfe_if_hit_mean_atr",
            "pred_expected_peak_mfe_atr",
        ]
        components = {
            "meaningful_mfe_probability": {
                "status": "RESEARCH_ABLATION_CANDIDATE",
                "evidence": _metric_slice_evidence(
                    role_metrics[MEANINGFUL_EVENT_ROLE],
                    metric_names=("roc_auc", "brier", "log_loss", "ece", "mce"),
                ),
            },
            "conditional_mean": {
                "status": "RESEARCH_ABLATION_CANDIDATE",
                "evidence": _metric_slice_evidence(
                    role_metrics["peak_mfe_12h_atr.conditional_mean"],
                    metric_names=("bias", "mae", "rmse", "spearman_ic"),
                ),
            },
            "conditional_q80": q80,
        }
        production_status = (
            "PRODUCTION_PENDING_Q80_CALIBRATION_AND_EXECUTION_EV_ABLATION"
        )
        production_required = [
            "Recalibrate or retrain q80 with train-only calibration until every supported side x representation slice has empirical coverage inside the declared interval.",
            "Show an identical-row, per-side cost-aware execution-EV ablation with positive aggregate and non-negative worst complete month before production promotion.",
        ]
    elif head_name == "time_to_first_meaningful_mfe":
        research_columns = [
            "pred_p_hit_by_2h",
            "pred_p_hit_by_4h",
            "pred_p_hit_by_8h",
            "pred_p_hit_by_12h",
            "pred_expected_censored_time_hours",
        ]
        components = {
            "early_cdf_probabilities": {
                "status": "RESEARCH_ABLATION_CANDIDATE",
                "columns": ["pred_p_hit_by_2h", "pred_p_hit_by_4h", "pred_p_hit_by_8h"],
                "evidence": {
                    role: _metric_slice_evidence(
                        role_metrics[role],
                        metric_names=("roc_auc", "brier", "log_loss", "ece", "mce"),
                    )
                    for role in (
                        "time_to_first_meaningful_mfe.hit_by_2h",
                        "time_to_first_meaningful_mfe.hit_by_4h",
                        "time_to_first_meaningful_mfe.hit_by_8h",
                    )
                },
            },
            "expected_censored_time": {
                "status": "WEAK_RESEARCH_ONLY",
                "column": "pred_expected_censored_time_hours",
                "reason": "A composed censored-time estimate has no direct production quality pass; retain only for research ablation alongside the separately materialized CDF probabilities.",
            },
        }
        production_status = (
            "PRODUCTION_PENDING_TIMING_CALIBRATION_AND_EXECUTION_EV_ABLATION"
        )
        production_required = [
            "Keep the monotone CDF horizons separately in the identical-row execution-EV ablation; do not treat expected censored time as a validated production feature by itself.",
            "Demonstrate positive aggregate and non-negative worst complete-month cost-aware execution EV per side before production promotion.",
        ]
    elif head_name == "mae_before_meaningful_mfe_atr":
        research_columns = [
            "pred_p_meaningful_mfe_12h",
            "pred_mae_if_hit_atr",
            "pred_mae_if_no_hit_atr",
            "pred_expected_mae_atr",
        ]
        components = {
            "conditional_mae_mixture": {
                "status": "WEAK_RESEARCH_ONLY",
                "evidence": {
                    role: _metric_slice_evidence(
                        role_metrics[role],
                        metric_names=("bias", "mae", "rmse", "spearman_ic"),
                    )
                    for role in (
                        "mae_before_meaningful_mfe_atr.if_hit",
                        "mae_before_meaningful_mfe_atr.if_no_hit",
                    )
                },
                "reason": "Conditional-risk learnability and incremental execution value must be demonstrated; raw OOF availability alone is not a production gate.",
            }
        }
        production_status = (
            "PRODUCTION_PENDING_MAE_LEARNABILITY_AND_EXECUTION_EV_ABLATION"
        )
        production_required = [
            "Demonstrate stable side-local conditional-risk learnability and an identical-row cost-aware execution-EV increment before production promotion.",
        ]
    elif head_name == "bars_before_price_stops_decreasing":
        research_columns = [
            "pred_legacy_adverse_extreme_bars",
            "pred_confirmed_adverse_trough_bars",
            "diag_confirmed_minus_legacy_bars",
        ]
        components = {
            "confirmed_adverse_trough": {
                "status": "TARGET_COMPLETION_AND_LEARNABILITY_REQUIRED",
                "evidence": _metric_slice_evidence(
                    role_metrics[
                        "bars_before_price_stops_decreasing.confirmed_adverse_trough"
                    ],
                    metric_names=("bias", "mae", "rmse", "spearman_ic"),
                ),
                "reason": "The confirmed-trough timing is conditionally missing; model occurrence/censoring before asking a timing value to improve execution EV.",
            }
        }
        production_status = "PRODUCTION_PENDING_TARGET_COMPLETION_LEARNABILITY_AND_EXECUTION_EV_ABLATION"
        production_required = [
            "Add an explicit confirmed-trough occurrence/censoring model, pass a learnability gate, then show an identical-row cost-aware execution-EV ablation per side.",
        ]
    elif head_name == "future_slope_atr_per_hour":
        research_columns = ["diag_pred_future_slope_atr_per_hour"]
        components = {
            "future_slope": {
                "status": "WEAK_DIAGNOSTIC_ONLY",
                "evidence": _metric_slice_evidence(
                    role_metrics["future_slope_atr_per_hour.diagnostic"],
                    metric_names=("bias", "mae", "rmse", "spearman_ic"),
                ),
                "reason": "A diagnostic path-shape target has no demonstrated incremental economic value.",
            }
        }
        production_status = "PRODUCTION_PENDING_SLOPE_LEARNABILITY_AND_INCREMENTAL_EXECUTION_EV_ABLATION"
        production_required = [
            "Pass an incremental identical-row ablation of peak+timing versus peak+timing+slope with positive aggregate and non-negative worst complete month per side.",
        ]
    else:  # pragma: no cover - callers enumerate HEAD_ROLE_KEYS.
        raise ValueError(f"unknown auxiliary head for promotion contract: {head_name}")

    research_status = (
        "ELIGIBLE_FOR_EXECUTION_EV_RESEARCH_ABLATION"
        if head_name
        in {
            "peak_mfe_12h_atr",
            "time_to_first_meaningful_mfe",
            "mae_before_meaningful_mfe_atr",
        }
        else "EXPLICIT_RESEARCH_ABLATION_OVERRIDE_REQUIRED"
    )
    return {
        "schema": PROMOTION_GATE_SCHEMA,
        "status": production_status,
        # Retain this legacy-shaped field as production-empty.  New consumers
        # must use research_ablation explicitly; old artifacts are interpreted
        # in the adapter as research-only rather than production-ready.
        "deployable_prediction_columns": [],
        "research_ablation": {
            "status": research_status,
            "prediction_columns": research_columns,
            "purpose": "execution_ev_research_ablation_only",
        },
        "production_promotion": {
            "status": production_status,
            "production_ready": False,
            "prediction_columns": [],
            "required": production_required,
        },
        "component_quality": components,
    }


def _head_fold_provenance(
    labels: pd.DataFrame,
    role_result: Mapping[str, Any],
) -> pd.DataFrame:
    fold_ids = np.asarray(role_result["oof_fold_ids"], dtype=np.int16)
    available = fold_ids >= 0
    output = pd.DataFrame(index=labels.index)
    output["oof_fold"] = fold_ids
    output["oof_fold_month"] = ""
    output.loc[available, "oof_fold_month"] = pd.to_datetime(
        labels.loc[available, "__ts__"], utc=True
    ).dt.strftime("%Y-%m")
    for column in (
        "validation_start",
        "validation_end",
        "train_decision_cutoff",
        "train_label_resolution_max",
    ):
        output[column] = pd.Series(
            pd.NaT, index=output.index, dtype="datetime64[ns, UTC]"
        )
    sides = labels["side"].astype(str).to_numpy()
    for side in SIDES:
        rows = np.flatnonzero(sides == side)
        side_result = role_result["side_results"][side]
        for fold in side_result["fold_provenance"]:
            local = np.flatnonzero(
                np.asarray(side_result["oof_fold_ids"]) == int(fold["fold"])
            )
            global_rows = rows[local]
            output.loc[global_rows, "validation_start"] = pd.Timestamp(
                fold["valid_start"]
            )
            output.loc[global_rows, "validation_end"] = pd.Timestamp(fold["valid_end"])
            output.loc[global_rows, "train_decision_cutoff"] = pd.Timestamp(
                fold["valid_start"]
            )
            output.loc[global_rows, "train_label_resolution_max"] = pd.Timestamp(
                fold["training_label_resolved_max"]
            )
    output["prediction_available_at"] = output["validation_start"]
    return output


def _persist_head(
    labels: pd.DataFrame,
    *,
    head_name: str,
    role_results: Mapping[str, Mapping[str, Any]],
    reference_end: pd.Timestamp,
    output_dir: Path,
) -> dict[str, Any]:
    roles = HEAD_ROLE_KEYS[head_name]
    composed = compose_head_oof(head_name, role_results)
    composed = composed.rename(columns=HEAD_PREDICTION_RENAMES[head_name])
    first_role = role_results[roles[0]]
    fold = _head_fold_provenance(labels, first_role)
    for role in roles[1:]:
        if not np.array_equal(
            first_role["oof_fold_ids"], role_results[role]["oof_fold_ids"]
        ):
            raise ValueError(f"{head_name} role fold identities do not match exactly")
    common_columns = [
        *STRICT_IDENTITY_COLUMNS,
        *[
            column
            for column in (
                "archetype",
                REPRESENTATION_AVAILABLE_FEATURE,
                "__path_auxiliary_target_valid__",
                DEFAULT_LABEL_RESOLUTION_COLUMN,
            )
            if column in labels
        ],
    ]
    bundle = labels.loc[:, common_columns].copy()
    bundle["oof_available"] = composed["oof_prediction_available"].to_numpy(dtype=bool)
    bundle = pd.concat(
        [
            bundle.reset_index(drop=True),
            fold.reset_index(drop=True),
            composed.drop(
                columns=["oof_prediction_available", "deployment_status"],
                errors="ignore",
            ).reset_index(drop=True),
        ],
        axis=1,
        copy=False,
    )
    bundle["selection_hpo_reference_end"] = reference_end
    bundle_path = output_dir / head_name / "oof_bundle.parquet"
    _atomic_to_parquet(bundle, bundle_path)
    role_metrics = {
        role: _representation_role_metrics(labels, role_results[role]) for role in roles
    }
    availability = labels[REPRESENTATION_AVAILABLE_FEATURE].to_numpy(dtype=np.float32)
    side_values = labels["side"].astype(str).to_numpy()
    head_output_slices = {
        side: {
            availability_name: {
                "canonical_rows": int(
                    ((side_values == side) & (availability == availability_value)).sum()
                ),
                "oof_prediction_rows": int(
                    (
                        (side_values == side)
                        & (availability == availability_value)
                        & bundle["oof_available"].to_numpy(dtype=bool)
                    ).sum()
                ),
                "status": (
                    "evaluated"
                    if bool(
                        (
                            (side_values == side)
                            & (availability == availability_value)
                            & bundle["oof_available"].to_numpy(dtype=bool)
                        ).any()
                    )
                    else "not_evaluable_zero_missing_support"
                ),
            }
            for availability_name, availability_value in (
                ("available", 1.0),
                ("missing", 0.0),
            )
        }
        for side in SIDES
    }
    role_metrics["derived_head_output"] = head_output_slices
    metrics_path = output_dir / head_name / "representation_metrics.json"
    _write_json(metrics_path, role_metrics)
    promotion_gate = _promotion_contract(head_name, role_metrics)
    research_ablation_columns = list(
        promotion_gate["research_ablation"]["prediction_columns"]
    )
    production_columns = list(
        promotion_gate["production_promotion"]["prediction_columns"]
    )
    promotion_path = output_dir / head_name / "promotion_gate.json"
    _write_json(promotion_path, promotion_gate)
    head_manifest = {
        "head_name": head_name,
        "roles": list(roles),
        "rows": int(len(bundle)),
        "candidate_identity_sha256": candidate_identity_sha256(
            bundle, columns=STRICT_IDENTITY_COLUMNS
        ),
        "oof_rows": int(bundle["oof_available"].sum()),
        "oof_months": list(FIXED_MAY_JULY_OOF_MONTHS),
        "oof_bundle": _artifact_record(bundle_path, kind="head_oof_bundle"),
        "representation_metrics": _artifact_record(
            metrics_path, kind="representation_metrics"
        ),
        "promotion_gate": _artifact_record(promotion_path, kind="promotion_gate"),
        "prediction_columns": list(HEAD_PREDICTION_RENAMES[head_name].values()),
        "deployable_prediction_columns": production_columns,
        "research_ablation_prediction_columns": research_ablation_columns,
        "production_prediction_columns": production_columns,
        "target_columns_are_audit_only": True,
        "final_refit_excluded_from_oof": True,
        "promotion_status": promotion_gate["status"],
        "production_status": promotion_gate["production_promotion"]["status"],
        "production_ready": False,
    }
    manifest_path = output_dir / head_name / "manifest.json"
    _write_json(manifest_path, head_manifest)
    head_manifest["manifest"] = _artifact_record(manifest_path, kind="head_manifest")
    return head_manifest


def _persist_role_manifest(
    labels: pd.DataFrame,
    *,
    role_name: str,
    role_result: Mapping[str, Any],
    selection_contracts: Mapping[str, Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
    selection_artifact: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    target = canonical_role_targets(labels)[role_name]
    group = (
        MEANINGFUL_EVENT_ROLE
        if role_name == MEANINGFUL_EVENT_ROLE
        else SELECTION_GROUP_BY_ROLE[role_name]
    )
    side_contracts: dict[str, Any] = {}
    for side in SIDES:
        fitted = role_result["side_results"][side]
        record = checkpoint["roles"][role_name][side]
        side_contracts[side] = {
            "bundle": dict(record),
            "selected_features": selected_features_for_role(
                selection_contracts, role_name, side
            ),
            "best_params_sha256": _stable_sha256(dict(fitted["best_params"])),
            "hpo": fitted["hpo"],
            "hpo_group_id": fitted.get(
                "hpo_group_id", role_result.get("hpo_group_id", role_name)
            ),
            "oof_fold_model_sha256": [
                fold["model_sha256"] for fold in fitted["fold_provenance"]
            ],
            "final_model_sha256": fitted["final_refit_contract"]["model_sha256"],
            "final_refit_excluded_from_oof": True,
        }
    role_spec = target.role
    manifest = {
        "role_name": role_name,
        "owner_role": role_name,
        "aliases": (
            [
                "peak_mfe_12h_atr.p_hit",
                "mae_before_meaningful_mfe_atr.p_hit",
                "time_to_first_meaningful_mfe.hit_by_12h",
            ]
            if role_name == MEANINGFUL_EVENT_ROLE
            else []
        ),
        "task_kind": ROLE_TASKS[role_name],
        "quantile_alpha": (0.8 if ROLE_TASKS[role_name] == "quantile" else None),
        "target_source_column": target.source_column,
        "target_condition": role_spec.target_condition,
        "target_units": (
            "probability"
            if ROLE_TASKS[role_name] == "binary"
            else role_spec.description
        ),
        "selection_group": group,
        "selection_source_role": SELECTION_ROLE_SOURCES[group],
        "hpo_group_id": role_result.get("hpo_group_id", role_name),
        "selection_artifact": dict(selection_artifact),
        "sample_weight_contract": ("unit weights preserve the declared estimand"),
        "oof_rows": int(
            np.asarray(role_result["oof_prediction_mask"], dtype=bool).sum()
        ),
        "side_contracts": side_contracts,
        "deployment_status": role_spec.deployment_status,
    }
    root = (
        output_dir / "shared/meaningful_mfe_event"
        if role_name == MEANINGFUL_EVENT_ROLE
        else output_dir / "roles" / _role_path_name(role_name)
    )
    path = root / "manifest.json"
    _write_json(path, manifest)
    manifest["manifest"] = _artifact_record(path, kind="role_manifest")
    return manifest


def run(
    *,
    labels_path: Path = DEFAULT_LABELS,
    context_path: Path = DEFAULT_CONTEXT,
    feature_dir: Path = DEFAULT_FEATURE_DIR,
    output_dir: Path = DEFAULT_OUTPUT,
    selection_hpo_reference_end: Any = CANONICAL_REFERENCE_END,
    selection_rows: int = 45_000,
    hpo_rows: int = 45_000,
    n_trials: int = 40,
    hpo_patience: int = 12,
    purge_hours: float = 13.0,
    seed: int = 42,
    max_rows: int = 0,
    overwrite: bool = False,
    resource_min_free_ram_gib: float = 2.0,
    resource_max_process_rss_gib: float = 12.0,
    resource_min_free_disk_gib: float = 10.0,
    resource_check_interval_seconds: float = 1.0,
    resource_guard: TrainingResourceGuard | None = None,
) -> dict[str, Any]:
    reference_end = pd.Timestamp(selection_hpo_reference_end)
    if reference_end.tzinfo is None:
        raise ValueError("selection_hpo_reference_end must be timezone-aware UTC")
    reference_end = reference_end.tz_convert("UTC")
    if max_rows == 0 and reference_end != CANONICAL_REFERENCE_END:
        raise ValueError(
            "canonical production role bundles require the fixed May-1 cutoff"
        )
    if max_rows == 0 and not np.isclose(float(purge_hours), 13.0):
        raise ValueError(
            "canonical production role bundles require the fixed 13-hour purge"
        )
    if not 1 <= int(n_trials) <= 40:
        raise ValueError("n_trials must be between 1 and 40")
    guard = resource_guard or _build_resource_guard(
        output_dir=output_dir,
        min_free_ram_gib=resource_min_free_ram_gib,
        max_process_rss_gib=resource_max_process_rss_gib,
        min_free_disk_gib=resource_min_free_disk_gib,
        check_interval_seconds=resource_check_interval_seconds,
        telemetry_path=output_dir / "training_resource_telemetry.jsonl",
    )
    guard.preflight("role_bundle:input_load")
    labels, label_report = _load_labels(
        labels_path,
        label_resolution_column=DEFAULT_LABEL_RESOLUTION_COLUMN,
        max_rows=int(max_rows),
    )
    labels, context_report = _join_archetype_context(
        labels, context_path, labels_are_canonical_top40=False
    )
    if REPRESENTATION_AVAILABLE_FEATURE not in labels:
        raise ValueError(
            f"canonical context is missing {REPRESENTATION_AVAILABLE_FEATURE}"
        )
    representation = labels[REPRESENTATION_AVAILABLE_FEATURE].to_numpy(dtype=np.float32)
    if (
        not np.isfinite(representation).all()
        or not np.isin(representation, (0.0, 1.0)).all()
    ):
        raise ValueError(f"{REPRESENTATION_AVAILABLE_FEATURE} must be finite binary")
    reference_mask = (
        labels["__ts__"].lt(reference_end)
        & labels[DEFAULT_LABEL_RESOLUTION_COLUMN].lt(reference_end)
    ).to_numpy()
    if not reference_mask.any():
        raise ValueError("no resolved reference rows precede the May-1 cutoff")
    reference_labels = labels.loc[reference_mask].reset_index(drop=True)
    complete_archetypes = [
        column
        for column in ARCHETYPE_COLUMNS
        if _complete_archetype_source(reference_labels, column)
    ]
    if not complete_archetypes:
        raise ValueError("complete frozen base-archetype identity is required")
    archetype_contract = fit_base_archetype_label_feature_contract(
        reference_labels,
        source_columns=complete_archetypes,
        canonical_source=complete_archetypes[0],
    )
    archetype_features = transform_base_archetype_label_features(
        labels, archetype_contract
    )
    handoff_features = list(map(str, context_report["handoff_model_feature_columns"]))
    static_columns = _static_feature_columns(feature_dir, labels["__symbol__"])
    requested_features, universe_report = configured_auxiliary_feature_universe(
        [*static_columns, *handoff_features, *archetype_features.columns]
    )
    mandatory = list(
        dict.fromkeys(
            [
                *MANDATORY_HANDOFF_MODEL_FEATURES,
                REPRESENTATION_AVAILABLE_FEATURE,
                *archetype_contract["canonical_features"],
            ]
        )
    )
    mandatory_by_side = {
        side: [feature for feature in mandatory if feature in requested_features]
        for side in SIDES
    }
    if any(
        REPRESENTATION_AVAILABLE_FEATURE not in mandatory_by_side[side]
        for side in SIDES
    ):
        raise ValueError(
            f"{REPRESENTATION_AVAILABLE_FEATURE} is unavailable to selection"
        )
    from extreme_price_movements import lgbm_pipeline

    selection_cv_contract = {
        "mode": str(lgbm_pipeline.LGBM_CV_MODE),
        "splits": int(lgbm_pipeline.LGBM_CV_SPLITS),
        "purge_hours": 13.0,
        "min_train_rows": int(lgbm_pipeline.LGBM_FORWARD_MIN_TRAIN_ROWS),
        "min_binary_validation_rows": int(lgbm_pipeline.LGBM_FORWARD_MIN_VALID_ROWS),
        "min_regression_validation_rows": int(
            min(lgbm_pipeline.LGBM_AUX_FORWARD_MIN_VALID_ROWS, 200)
        ),
        "binary_short_history_fallback_fraction": float(
            lgbm_pipeline.LGBM_FORWARD_SHORT_HISTORY_FALLBACK_FRAC
        ),
        "regression_validation_months": 1,
        "train_before_validation_only": True,
        "shuffled_fallback_forbidden": True,
    }
    if max_rows == 0:
        production_selector_requirements = {
            "mode": "forward_burnin",
            "splits": 3,
            "min_train_rows": 200,
            "min_binary_validation_rows": 20,
            "min_regression_validation_rows": 200,
        }
        mismatches = {
            key: {
                "expected": expected,
                "actual": selection_cv_contract[key],
            }
            for key, expected in production_selector_requirements.items()
            if selection_cv_contract[key] != expected
        }
        if mismatches:
            raise ValueError(
                f"canonical production selector runtime is noncanonical: {mismatches}"
            )

    fingerprint_payload = {
        "schema": RUNNER_SCHEMA,
        "labels": {
            "path": str(labels_path.resolve()),
            "sha256": _file_sha256(labels_path),
        },
        "context": {
            "path": str(context_path.resolve()),
            "sha256": _file_sha256(context_path),
        },
        "feature_store_signature": _tree_stat_signature(feature_dir),
        "candidate_identity_sha256": candidate_identity_sha256(
            labels, columns=STRICT_IDENTITY_COLUMNS
        ),
        "selected_population_identity_sha256": context_report[
            "selected_population_identity_sha256"
        ],
        "model_family_schema": MODEL_FAMILY_SCHEMA,
        "role_trainer_schema": ROLE_TRAINER_SCHEMA,
        "bundle_training_schema": BUNDLE_TRAINING_SCHEMA,
        "timing_cdf_trainer_schema": TIMING_CDF_TRAINER_SCHEMA,
        "head_specs": [asdict(spec) for spec in HEAD_SPECS],
        "source_sha256": {
            "runner": _file_sha256(Path(__file__).resolve()),
            "model_families": _file_sha256(
                ROOT / "extreme_price_movements/path_auxiliary_model_families.py"
            ),
            "role_training": _file_sha256(
                ROOT / "extreme_price_movements/path_auxiliary_role_training.py"
            ),
            "bundle_training": _file_sha256(
                ROOT / "extreme_price_movements/path_auxiliary_bundle_training.py"
            ),
            "timing_training": _file_sha256(
                ROOT / "extreme_price_movements/path_auxiliary_timing_training.py"
            ),
        },
        "selection_groups": dict(SELECTION_ROLE_SOURCES),
        "selection_cv_contract": selection_cv_contract,
        "reference_end": reference_end.isoformat(),
        "fixed_oof_months": list(FIXED_MAY_JULY_OOF_MONTHS),
        "purge_hours": float(purge_hours),
        "selection_rows": int(selection_rows),
        "hpo_rows": int(hpo_rows),
        "n_trials": int(n_trials),
        "hpo_patience": int(hpo_patience),
        "seed": int(seed),
        "max_rows": int(max_rows),
    }
    run_fingerprint = {
        "payload": fingerprint_payload,
        "sha256": _stable_sha256(fingerprint_payload),
    }
    checkpoint = _load_checkpoint(
        output_dir,
        run_fingerprint=run_fingerprint,
        overwrite=overwrite,
    )
    selections, selection_report = _selection_contracts(
        labels,
        reference_mask=reference_mask,
        selection_rows=int(selection_rows),
        requested_features=requested_features,
        feature_dir=feature_dir,
        handoff_feature_columns=handoff_features,
        archetype_features=archetype_features,
        mandatory_features_by_side=mandatory_by_side,
        seed=int(seed),
        output_dir=output_dir,
        checkpoint=checkpoint,
        guard=guard,
    )
    role_results: dict[str, dict[str, Any]] = {}
    phase_roles: Mapping[str, tuple[str, ...]] = {
        "time_to_first_meaningful_mfe": (
            MEANINGFUL_EVENT_ROLE,
            "time_to_first_meaningful_mfe.hit_by_2h",
            "time_to_first_meaningful_mfe.hit_by_4h",
            "time_to_first_meaningful_mfe.hit_by_8h",
        ),
        "peak_mfe_12h_atr": (
            "peak_mfe_12h_atr.conditional_mean",
            "peak_mfe_12h_atr.conditional_q80",
        ),
        "mae_before_meaningful_mfe_atr": (
            "mae_before_meaningful_mfe_atr.if_hit",
            "mae_before_meaningful_mfe_atr.if_no_hit",
        ),
        "bars_before_price_stops_decreasing": (
            "bars_before_price_stops_decreasing.legacy_adverse_extreme",
            "bars_before_price_stops_decreasing.confirmed_adverse_trough",
        ),
        "future_slope_atr_per_hour": ("future_slope_atr_per_hour.diagnostic",),
    }
    for phase_index, (phase, roles) in enumerate(phase_roles.items()):
        selected_union = list(
            dict.fromkeys(
                feature
                for role in roles
                for side in SIDES
                for feature in selected_features_for_role(selections, role, side)
            )
        )
        matrix, feature_report = _full_selected_matrix(
            labels,
            selected_features=selected_union,
            feature_dir=feature_dir,
            handoff_feature_columns=handoff_features,
            archetype_features=archetype_features,
            guard=guard,
            stage=phase,
        )
        _write_json(output_dir / phase / "feature_availability.json", feature_report)
        if phase == "time_to_first_meaningful_mfe":
            families: dict[str, Any] = {}
            for side_index, side in enumerate(SIDES):
                side_mask = labels["side"].astype(str).eq(side).to_numpy()
                families[side] = _fit_or_load_timing_side(
                    matrix.loc[side_mask].reset_index(drop=True),
                    labels.loc[side_mask].reset_index(drop=True),
                    side=side,
                    selection_contracts=selections,
                    reference_end=reference_end,
                    n_trials=int(n_trials),
                    hpo_rows=int(hpo_rows),
                    seed=int(seed) + 1_009 * side_index,
                    output_dir=output_dir,
                    checkpoint=checkpoint,
                    guard=guard,
                )
            role_results.update(_timing_role_results(labels, families_by_side=families))
            del matrix, families
            gc.collect()
            guard.checkpoint(f"{phase}:released")
            continue
        for role_index, role in enumerate(roles):
            side_results: dict[str, Any] = {}
            for side_index, side in enumerate(SIDES):
                side_mask = labels["side"].astype(str).eq(side).to_numpy()
                side_results[side] = _fit_or_load_role_side(
                    matrix.loc[side_mask].reset_index(drop=True),
                    labels.loc[side_mask].reset_index(drop=True),
                    role_name=role,
                    side=side,
                    selection_contracts=selections,
                    reference_end=reference_end,
                    n_trials=int(n_trials),
                    hpo_rows=int(hpo_rows),
                    hpo_patience=int(hpo_patience),
                    purge_hours=float(purge_hours),
                    seed=(
                        int(seed)
                        + 100_003 * phase_index
                        + 10_007 * role_index
                        + 1_009 * side_index
                    ),
                    output_dir=output_dir,
                    checkpoint=checkpoint,
                    guard=guard,
                )
            role_results[role] = _scatter_role_result(
                labels, role_name=role, side_results=side_results
            )
        del matrix
        gc.collect()
        guard.checkpoint(f"{phase}:released")

    # The event role is fitted once above and is a bitwise alias in three heads.
    if MEANINGFUL_EVENT_ROLE not in role_results:
        raise AssertionError("shared meaningful-event role was not fitted")
    heads: dict[str, Any] = {}
    for head_name in HEAD_ROLE_KEYS:
        heads[head_name] = _persist_head(
            labels,
            head_name=head_name,
            role_results=role_results,
            reference_end=reference_end,
            output_dir=output_dir,
        )
        checkpoint.setdefault("heads", {})[head_name] = heads[head_name]["manifest"]
        _save_checkpoint(output_dir, checkpoint)
    role_manifests = {
        role_name: _persist_role_manifest(
            labels,
            role_name=role_name,
            role_result=role_result,
            selection_contracts=selections,
            checkpoint=checkpoint,
            selection_artifact=checkpoint["selection"],
            output_dir=output_dir,
        )
        for role_name, role_result in role_results.items()
    }
    manifest = {
        "schema": RUNNER_SCHEMA,
        "status": "COMPLETE_STRICT_SIDE_LOCAL_MULTI_ROLE_AUXILIARY_OOF",
        "run_fingerprint": run_fingerprint,
        "labels": fingerprint_payload["labels"],
        "context": fingerprint_payload["context"],
        "candidate_identity_sha256": fingerprint_payload["candidate_identity_sha256"],
        "rows": int(len(labels)),
        "side_rows": {
            side: int(labels["side"].astype(str).eq(side).sum()) for side in SIDES
        },
        "reference_end": reference_end.isoformat(),
        "fixed_oof_months": list(FIXED_MAY_JULY_OOF_MONTHS),
        "purge_hours": float(purge_hours),
        "model_family_schema": MODEL_FAMILY_SCHEMA,
        "role_trainer_schema": ROLE_TRAINER_SCHEMA,
        "bundle_training_schema": BUNDLE_TRAINING_SCHEMA,
        "timing_cdf_trainer_schema": TIMING_CDF_TRAINER_SCHEMA,
        "shared_role_ownership": {
            "owner": MEANINGFUL_EVENT_ROLE,
            "aliases": [
                "peak_mfe_12h_atr.p_hit",
                "mae_before_meaningful_mfe_atr.p_hit",
                "time_to_first_meaningful_mfe.hit_by_12h",
            ],
            "contract": "one side-local model and bitwise-identical OOF vector",
        },
        "selection_report": selection_report,
        "selection_cv_contract": selection_cv_contract,
        "feature_universe_report": universe_report,
        "mandatory_features_by_side": mandatory_by_side,
        "sample_weight_contract": (
            "unit weights preserve probability and conditional-mean estimands"
        ),
        "training_resource_guard": _resource_guard_contract(guard),
        "checkpoint": str(_checkpoint_path(output_dir).resolve()),
        "roles": role_manifests,
        "heads": heads,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    checkpoint["manifest"] = _artifact_record(manifest_path, kind="root_manifest")
    _save_checkpoint(output_dir, checkpoint)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--context-path", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--selection-hpo-reference-end",
        default=CANONICAL_REFERENCE_END.isoformat(),
    )
    parser.add_argument("--selection-rows", type=int, default=45_000)
    parser.add_argument("--hpo-rows", type=int, default=45_000)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--hpo-patience", type=int, default=12)
    parser.add_argument("--purge-hours", type=float, default=13.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resource-min-free-ram-gib", type=float, default=2.0)
    parser.add_argument("--resource-max-process-rss-gib", type=float, default=12.0)
    parser.add_argument("--resource-min-free-disk-gib", type=float, default=10.0)
    parser.add_argument("--resource-check-interval-seconds", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run(
        labels_path=args.labels_path,
        context_path=args.context_path,
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        selection_hpo_reference_end=args.selection_hpo_reference_end,
        selection_rows=args.selection_rows,
        hpo_rows=args.hpo_rows,
        n_trials=args.n_trials,
        hpo_patience=args.hpo_patience,
        purge_hours=args.purge_hours,
        seed=args.seed,
        max_rows=args.max_rows,
        overwrite=args.overwrite,
        resource_min_free_ram_gib=args.resource_min_free_ram_gib,
        resource_max_process_rss_gib=args.resource_max_process_rss_gib,
        resource_min_free_disk_gib=args.resource_min_free_disk_gib,
        resource_check_interval_seconds=args.resource_check_interval_seconds,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "status": manifest["status"],
                "heads": list(manifest["heads"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
