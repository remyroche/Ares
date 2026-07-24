#!/usr/bin/env python3
"""Run the bounded CatBoost path-archetype geometry target search."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.path_archetype_geometry_search import (
    DEFAULT_MAX_TRAIN_ROWS_PER_FOLD,
    GEOMETRY_NESTED_MONTHS,
    GEOMETRY_OOS_MONTHS,
    GEOMETRY_TRAIN_MONTHS,
    PATH_GEOMETRY_CLASSES,
    PathGeometryColumns,
    PathGeometryConfig,
    ensure_risk_fraction,
    export_checkpoint_geometry,
    staged_geometry_search,
)
from extreme_price_movements.static_feature_store import read_static_features

FEATURE_SELECTION_HPO_CONTRACT_FILENAME = "feature_selection_hpo_contract.json"
FEATURE_SELECTION_HPO_CONTRACT_SCHEMA = (
    "catboost_path_archetype_feature_selection_hpo_contract_v1"
)


def _read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _feature_columns(path: Path) -> list[str]:
    value = _read_json(path)
    if isinstance(value, Mapping):
        value = value.get("selected_features", value.get("feature_columns"))
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("features JSON must be a string list or selected_features mapping")
    if len(value) != len(set(value)):
        raise ValueError("features JSON must not contain duplicate selected features")
    return list(value)


def _classifier_contract_file(path: Path) -> Path:
    path = Path(path)
    return path / FEATURE_SELECTION_HPO_CONTRACT_FILENAME if path.is_dir() else path


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _prediction_identity_hash(frame: pd.DataFrame, identity_columns: list[str]) -> str:
    identity = frame.loc[:, identity_columns].copy()
    for column in ("__ts__", "train_cutoff_utc", "oos_start_utc", "oos_end_utc"):
        if column in identity:
            identity[column] = pd.to_datetime(identity[column], utc=True).astype(str)
    hashed = pd.util.hash_pandas_object(identity, index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prepared_frame_identity(frame: pd.DataFrame) -> str:
    """Hash the exact post-filter/post-join matrix used by the search."""
    digest = hashlib.sha256()
    digest.update(json.dumps(list(frame.columns), separators=(",", ":")).encode("utf-8"))
    for column in frame.columns:
        digest.update(
            pd.util.hash_pandas_object(frame[column], index=True)
            .to_numpy(dtype=np.uint64)
            .tobytes()
        )
    return digest.hexdigest()


def _geometry_progress(event: str, details: Mapping[str, Any]) -> None:
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC")
    rendered = json.dumps(_json_safe(details), sort_keys=True, separators=(",", ":"))
    print(f"[{stamp}] geometry_search {event} {rendered}", flush=True)


def _parquet_ready(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize mixed restored/new temporal values before Arrow conversion."""

    result = frame.copy()
    explicit_temporal = {
        "train_end",
        "oos_start",
        "oos_end",
        "validation_start",
        "validation_end",
    }
    for column in result.columns:
        if (
            column not in explicit_temporal
            and not column.endswith("_utc")
            and not column.endswith("_ts")
        ):
            continue
        values = result[column]
        parsed = pd.to_datetime(values, utc=True, errors="coerce")
        if int(parsed.notna().sum()) == int(values.notna().sum()):
            result[column] = parsed
    return result


def _completed_classifier_selection_hpo_contract(
    contract_path: Path,
) -> tuple[Path, Mapping[str, Any], list[str], dict[str, Any]]:
    """Read the only completed classifier contract accepted by canonical geometry runs."""
    resolved_contract_path = _classifier_contract_file(contract_path)
    if not resolved_contract_path.is_file():
        raise FileNotFoundError(
            "completed classifier feature-selection/HPO contract does not exist: "
            f"{resolved_contract_path}"
        )
    contract = _read_json(resolved_contract_path)
    if not isinstance(contract, Mapping):
        raise ValueError("classifier feature-selection/HPO contract must contain an object")
    required = {
        "schema",
        "status",
        "fingerprint",
        "selected_features",
        "effective_model_params",
    }
    missing = sorted(required.difference(contract))
    if missing:
        raise ValueError(
            "classifier feature-selection/HPO contract is incomplete: "
            + ", ".join(missing)
        )
    if contract["schema"] != FEATURE_SELECTION_HPO_CONTRACT_SCHEMA:
        raise ValueError("classifier feature-selection/HPO contract has an unsupported schema")
    if contract["status"] != "feature_selection_hpo_complete":
        raise ValueError("classifier feature-selection/HPO contract is not complete")
    selected_features = contract["selected_features"]
    effective_model_params = contract["effective_model_params"]
    if not isinstance(selected_features, list) or not all(
        isinstance(value, str) for value in selected_features
    ):
        raise ValueError("classifier feature-selection/HPO contract has invalid selected_features")
    if len(selected_features) != len(set(selected_features)):
        raise ValueError(
            "classifier feature-selection/HPO contract selected_features contain duplicates"
        )
    if not isinstance(effective_model_params, Mapping):
        raise ValueError(
            "classifier feature-selection/HPO contract has invalid effective_model_params"
        )
    return (
        resolved_contract_path,
        contract,
        list(selected_features),
        dict(effective_model_params),
    )


def _verify_classifier_selection_hpo_contract(
    contract_path: Path,
    *,
    feature_columns: list[str] | None = None,
    model_params: Mapping[str, Any] | None = None,
    features_json_path: Path | None = None,
    catboost_params_json_path: Path | None = None,
) -> dict[str, Any]:
    """Bind geometry inputs to one completed classifier selection/HPO run."""
    (
        resolved_contract_path,
        contract,
        selected_features,
        effective_model_params,
    ) = _completed_classifier_selection_hpo_contract(contract_path)
    if feature_columns is not None and feature_columns != selected_features:
        raise ValueError(
            "features JSON must exactly match classifier contract selected_features"
        )
    if model_params is not None and _json_sha256(dict(model_params)) != _json_sha256(
        effective_model_params
    ):
        raise ValueError(
            "CatBoost params JSON must exactly match classifier contract "
            "effective_model_params; raw HPO params are not accepted"
        )
    if features_json_path is not None and _feature_columns(features_json_path) != selected_features:
        raise ValueError(
            "features JSON must exactly match classifier contract selected_features"
        )
    if catboost_params_json_path is not None:
        compatibility_params = _read_json(catboost_params_json_path)
        if not isinstance(compatibility_params, Mapping):
            raise ValueError("catboost params JSON must contain an object")
        if _json_sha256(dict(compatibility_params)) != _json_sha256(
            effective_model_params
        ):
            raise ValueError(
                "CatBoost params JSON must exactly match classifier contract "
                "effective_model_params; raw HPO params are not accepted"
            )
    return {
        "verification": "strict_completed_classifier_selection_hpo_contract",
        "input_source": "classifier_selection_hpo_contract",
        "contract_path": str(resolved_contract_path),
        "contract_sha256": _file_sha256(resolved_contract_path),
        "contract_schema": contract["schema"],
        "contract_fingerprint": str(contract["fingerprint"]),
        "selected_features_source": "classifier_selection_hpo_contract",
        "features_sha256": _json_sha256(selected_features),
        "effective_model_params_source": "classifier_selection_hpo_contract",
        "effective_model_params_sha256": _json_sha256(effective_model_params),
        "features_json_path": str(features_json_path) if features_json_path else None,
        "features_json_sha256": (
            _file_sha256(features_json_path) if features_json_path else None
        ),
        "catboost_params_json_path": (
            str(catboost_params_json_path) if catboost_params_json_path else None
        ),
        "catboost_params_json_sha256": (
            _file_sha256(catboost_params_json_path)
            if catboost_params_json_path
            else None
        ),
    }


def _unsafe_input_provenance(
    *, features_json_path: Path, catboost_params_json_path: Path
) -> dict[str, Any]:
    """Mark an explicitly non-canonical geometry invocation for compatibility."""
    return {
        "verification": "unsafe_unverified_inputs",
        "contract_path": None,
        "contract_sha256": None,
        "features_json_path": str(features_json_path),
        "features_json_sha256": _file_sha256(features_json_path),
        "catboost_params_json_path": str(catboost_params_json_path),
        "catboost_params_json_sha256": _file_sha256(catboost_params_json_path),
    }


def _signed_manifest_hash(payload: Mapping[str, Any]) -> str:
    canonical = {
        str(key): _json_safe(value)
        for key, value in payload.items()
        if key != "prediction_role_manifest_sha256"
    }
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_finalist_diagnostics(
    output_dir: Path,
    rank: int,
    config_id: str,
    diagnostics: Mapping[str, Any],
) -> dict[str, str]:
    """Persist the leakage/audit diagnostics accompanying one finalist's OOS rows."""
    expected = {
        "folds",
        "probability_reliability_bins",
        "economic_confusion",
        "economic_confusion_priors",
        "side_diagnostics",
        "month_diagnostics",
    }
    missing = sorted(expected.difference(diagnostics))
    if missing:
        raise ValueError(f"finalist {config_id} is missing diagnostic tables: {missing}")
    directory = output_dir / "finalist_geometry_diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for key in sorted(expected):
        table = diagnostics[key]
        if not isinstance(table, pd.DataFrame):
            raise TypeError(f"finalist {config_id} diagnostic {key} is not a dataframe")
        path = directory / f"finalist_{rank:02d}_{config_id}_{key}.parquet"
        table.to_parquet(path, index=False)
        paths[key] = str(path)
    return paths


def _write_finalist_predictions(
    output_dir: Path,
    finalists: list[dict[str, Any]],
    feature_columns: list[str],
) -> tuple[Path, dict[str, Any]]:
    if len(finalists) != 5:
        raise ValueError("strict OOS prediction persistence requires exactly five finalists")
    prediction_dir = output_dir / "finalist_oos_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for finalist in finalists:
        predictions = finalist["predictions"].copy()
        rank, config_id = int(finalist["rank"]), str(finalist["config_id"])
        required = {
            "source_row_position", "__ts__", "__symbol__", "side",
            "candidate_id", "available_at", "validation_start",
            "train_decision_cutoff", "label_resolution_available_at",
            "true_dynamic_label", "predicted_class", "probability_vector",
            "probability_entropy", "fold_id", "train_cutoff_utc",
            "oos_start_utc", "oos_end_utc", "config_id",
        }
        missing = sorted(required.difference(predictions.columns))
        if missing:
            raise ValueError(f"finalist {config_id} predictions are missing fields: {missing}")
        if set(predictions["config_id"].astype(str)) != {config_id}:
            raise ValueError(f"finalist {config_id} prediction rows have a mismatched config id")
        for column in (
            "__ts__",
            "train_cutoff_utc",
            "available_at",
            "validation_start",
            "train_decision_cutoff",
            "label_resolution_available_at",
            "oos_start_utc",
            "oos_end_utc",
        ):
            predictions[column] = pd.to_datetime(predictions[column], utc=True, errors="coerce")
        if predictions[list((
            "__ts__", "train_cutoff_utc", "available_at", "validation_start",
            "train_decision_cutoff", "label_resolution_available_at",
            "oos_start_utc", "oos_end_utc",
        ))].isna().any().any():
            raise ValueError(f"finalist {config_id} has invalid UTC prediction keys")
        if not (
            (predictions["__ts__"] >= predictions["oos_start_utc"])
            & (predictions["__ts__"] < predictions["oos_end_utc"])
            & (predictions["train_cutoff_utc"] <= predictions["oos_start_utc"])
        ).all():
            raise ValueError(f"finalist {config_id} contains non-OOS prediction timestamps")
        if not (
            (predictions["available_at"] <= predictions["__ts__"])
            & (
                predictions["label_resolution_available_at"]
                <= predictions["train_decision_cutoff"]
            )
            & (predictions["train_decision_cutoff"] < predictions["validation_start"])
            & (predictions["validation_start"] <= predictions["__ts__"])
        ).all():
            raise ValueError(f"finalist {config_id} violates strict OOS provenance")
        probability_columns = [f"probability_{name}" for name in PATH_GEOMETRY_CLASSES]
        if sorted(set(probability_columns).difference(predictions.columns)):
            raise ValueError(f"finalist {config_id} lacks the full aligned probability vector")
        probability_matrix = predictions.loc[:, probability_columns].to_numpy(dtype=float)
        if not np.isfinite(probability_matrix).all() or not np.allclose(probability_matrix.sum(axis=1), 1.0):
            raise ValueError(f"finalist {config_id} probabilities are invalid or unnormalised")
        identity_columns = ["source_row_position", "__ts__", "__symbol__", "side", "fold_id"]
        if "candidate_id" in predictions:
            identity_columns.insert(1, "candidate_id")
        if predictions.duplicated(identity_columns).any():
            raise ValueError(f"finalist {config_id} contains duplicate strict-OOS identities")
        path = prediction_dir / f"finalist_{rank:02d}_{config_id}.parquet"
        predictions.to_parquet(path, index=False)
        role_manifest_path = path.with_suffix(".role_manifest.json")
        role_manifest = {
            "schema": "path_archetype_oof_prediction_role_v1",
            "prediction_role": "path_archetype_oof",
            "source_artifact": str(path),
            "source_artifact_sha256": _file_sha256(path),
            "prediction_columns": {
                **{
                    column: {
                        "role": "pre_entry_path_archetype_oof_prediction",
                        "target": False,
                    }
                    for column in probability_columns
                },
                "predicted_class": {
                    "role": "pre_entry_path_archetype_oof_prediction",
                    "target": False,
                },
                "probability_entropy": {
                    "role": "pre_entry_path_archetype_oof_prediction",
                    "target": False,
                },
            },
            "identity_columns": ["__ts__", "__symbol__", "side", "candidate_id"],
            "fold_provenance_columns": {
                "fold": "fold_id",
                "validation_start": "validation_start",
                "training_information_cutoff": "train_decision_cutoff",
                "latest_resolved_training_label": "label_resolution_available_at",
                "prediction_available_at": "available_at",
            },
        }
        role_manifest["prediction_role_manifest_sha256"] = _signed_manifest_hash(
            role_manifest
        )
        role_manifest_path.write_text(
            json.dumps(_json_safe(role_manifest), indent=2, sort_keys=True) + "\n"
        )
        diagnostic_paths = _write_finalist_diagnostics(
            output_dir,
            rank,
            config_id,
            finalist.get("diagnostics", {}),
        )
        entries.append(
            {
                "rank": rank,
                "config_id": config_id,
                "path": str(path),
                "rows": int(len(predictions)),
                "folds": sorted(map(int, predictions["fold_id"].unique())),
                "utc_start": str(predictions["__ts__"].min()),
                "utc_end": str(predictions["__ts__"].max()),
                "identity_columns": identity_columns,
                "identity_sha256": _prediction_identity_hash(predictions, identity_columns),
                "prediction_role_manifest": str(role_manifest_path),
                "prediction_role_manifest_sha256": _file_sha256(role_manifest_path),
                "diagnostic_paths": diagnostic_paths,
                "config": finalist["config"],
                "summary": finalist["summary"],
            }
        )
    manifest = {
        "artifact_contract": "strict_purged_oos_top5_geometry_probabilities_v1",
        "selection_provenance": "row-level model OOS; finalist geometries were selected on these OOS metrics, so this is geometry-selection OOS rather than an untouched final test",
        "class_order": list(PATH_GEOMETRY_CLASSES),
        "feature_columns": list(feature_columns),
        "target_or_path_feature_columns": [name for name in feature_columns if "path" in name.lower() or "target" in name.lower()],
        "finalist_count": len(entries),
        "finalists": entries,
    }
    manifest_path = output_dir / "top5_finalist_oos_predictions_manifest.json"
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n")
    return manifest_path, manifest


def _write_exact_geometry_export(
    output_dir: Path,
    export: Mapping[str, Any],
    feature_columns: list[str],
) -> tuple[Path, dict[str, Any]]:
    """Persist the raw merged-seven-class output of one checkpointed refit."""
    config_id = str(export["config_id"])
    predictions = export["predictions"].copy()
    required = {
        "source_row_position", "__ts__", "__symbol__", "side", "candidate_id",
        "true_merged_dynamic_label", "predicted_class", "probability_vector",
        "probability_entropy", "fold_id", "train_cutoff_utc", "available_at",
        "validation_start", "label_resolution_available_at", "train_decision_cutoff",
        "oos_start_utc", "oos_end_utc", "config_id", "raw_max_probability",
        "raw_normalized_entropy", "raw_top1_top2_probability_margin",
        "raw_adverse_probability_mass", "raw_favorable_probability_mass",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"exact geometry export is missing raw prediction fields: {missing}")
    if set(predictions["config_id"].astype(str)) != {config_id}:
        raise ValueError("exact geometry export prediction rows have a mismatched config id")
    class_order = list(export["class_order"])
    if len(class_order) != 7 or "fast_realization_winner" not in class_order:
        raise ValueError("exact geometry export must declare the merged seven-class order")
    probability_columns = [f"probability_{name}" for name in class_order]
    missing_probabilities = sorted(set(probability_columns).difference(predictions.columns))
    if missing_probabilities:
        raise ValueError(
            "exact geometry export lacks the full raw merged seven-class probability vector: "
            f"{missing_probabilities}"
        )
    probability_matrix = predictions.loc[:, probability_columns].to_numpy(dtype=float)
    if not np.isfinite(probability_matrix).all() or not np.allclose(
        probability_matrix.sum(axis=1), 1.0
    ):
        raise ValueError("exact geometry export raw probabilities are invalid or unnormalised")
    for column in (
        "__ts__", "train_cutoff_utc", "available_at", "validation_start",
        "label_resolution_available_at", "train_decision_cutoff", "oos_start_utc", "oos_end_utc",
    ):
        predictions[column] = pd.to_datetime(predictions[column], utc=True, errors="coerce")
    if predictions[[
        "__ts__", "train_cutoff_utc", "available_at", "validation_start",
        "label_resolution_available_at", "train_decision_cutoff", "oos_start_utc", "oos_end_utc",
    ]].isna().any().any():
        raise ValueError("exact geometry export has invalid UTC prediction keys")
    if not (
        (predictions["__ts__"] >= predictions["oos_start_utc"])
        & (predictions["__ts__"] < predictions["oos_end_utc"])
        & (predictions["train_cutoff_utc"] <= predictions["oos_start_utc"])
        & (predictions["label_resolution_available_at"] <= predictions["train_decision_cutoff"])
        & (predictions["train_decision_cutoff"] < predictions["validation_start"])
    ).all():
        raise ValueError("exact geometry export violates strict OOS provenance")
    identity_columns = ["source_row_position", "candidate_id", "__ts__", "__symbol__", "side", "fold_id"]
    if predictions.duplicated(identity_columns).any():
        raise ValueError("exact geometry export contains duplicate strict-OOS identities")
    prediction_dir = output_dir / "exact_geometry_oos_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    path = prediction_dir / f"{config_id}.parquet"
    predictions.to_parquet(path, index=False)
    role_manifest = {
        "schema": "path_archetype_oof_prediction_role_v1",
        "prediction_role": "pre_refinement_path_archetype_oof_raw",
        "source_artifact": str(path),
        "source_artifact_sha256": _file_sha256(path),
        "class_order": class_order,
        "prediction_columns": {
            column: {
                "role": "pre_refinement_path_archetype_oof_raw_probability",
                "target": False,
            }
            for column in probability_columns
        },
        "identity_columns": ["__ts__", "__symbol__", "side", "candidate_id"],
        "fold_provenance_columns": {
            "fold": "fold_id",
            "validation_start": "validation_start",
            "training_information_cutoff": "train_decision_cutoff",
            "latest_resolved_training_label": "label_resolution_available_at",
            "prediction_available_at": "available_at",
        },
    }
    role_manifest["prediction_role_manifest_sha256"] = _signed_manifest_hash(role_manifest)
    role_manifest_path = path.with_suffix(".role_manifest.json")
    role_manifest_path.write_text(
        json.dumps(_json_safe(role_manifest), indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        "schema": "catboost_path_archetype_exact_geometry_raw_oos_export_v1",
        "config_id": config_id,
        "config": export["config"],
        "checkpoint_path": export["checkpoint_path"],
        "checkpoint_fingerprint": export["checkpoint_fingerprint"],
        "checkpoint_contract": export["checkpoint_contract"],
        "export_fingerprint": export["export_fingerprint"],
        "checkpoint_capture_reused": bool(export["reused_checkpoint_capture"]),
        "feature_columns": list(feature_columns),
        "hard_label_target": export["hard_label_target"],
        "class_merge": export["class_merge"],
        "class_order": export["class_order"],
        "sample_weight_contract": export["sample_weight_contract"],
        "probability_output": export["probability_output"],
        "raw_scoring_contract": export["raw_scoring_contract"],
        "model_persistence": export["model_persistence"],
        "prediction_path": str(path),
        "prediction_sha256": _file_sha256(path),
        "prediction_role_manifest": str(role_manifest_path),
        "prediction_role_manifest_sha256": _file_sha256(role_manifest_path),
        "identity_columns": identity_columns,
        "identity_sha256": _prediction_identity_hash(predictions, identity_columns),
        "rows": int(len(predictions)),
        "folds": sorted(map(int, predictions["fold_id"].unique())),
        "summary": export["summary"],
        "diagnostics": "not_fitted_for_raw_seven_class_export",
    }
    manifest_path = output_dir / "exact_geometry_oos_predictions_manifest.json"
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n")
    return manifest_path, manifest


def _join_frozen_features(
    labels: pd.DataFrame,
    feature_path: Path | None,
    feature_columns: list[str],
    join_columns: list[str],
    columns: PathGeometryColumns,
) -> pd.DataFrame:
    missing = sorted(set(feature_columns).difference(labels.columns))
    if not missing:
        return labels
    if feature_path is None:
        raise ValueError(
            "path label artifact does not contain frozen CatBoost features; "
            "provide --features-parquet from the canonical feature store"
        )
    required = set(join_columns).union(feature_columns)
    feature_frame = pd.read_parquet(feature_path, columns=sorted(required))
    absent = sorted(required.difference(feature_frame.columns))
    if absent:
        raise ValueError(f"features parquet is missing join/model columns: {absent}")
    label_absent = sorted(set(join_columns).difference(labels.columns))
    if label_absent:
        raise ValueError(f"path label artifact is missing feature join columns: {label_absent}")
    left, right = labels.copy(), feature_frame.loc[:, sorted(required)].copy()
    if columns.timestamp in join_columns:
        left[columns.timestamp] = pd.to_datetime(left[columns.timestamp], utc=True, errors="coerce")
        right[columns.timestamp] = pd.to_datetime(right[columns.timestamp], utc=True, errors="coerce")
    if right.duplicated(join_columns).any():
        raise ValueError("features parquet has duplicate frozen-feature identity keys")
    merged = left.merge(right, on=join_columns, how="left", validate="m:1", sort=False)
    unresolved = sorted(set(feature_columns).difference(merged.columns))
    if unresolved or merged.loc[:, feature_columns].isna().all(axis=1).any():
        raise ValueError("frozen feature join did not cover every path-label row")
    return merged


def _feature_store_location(feature_dir: Path) -> tuple[pd.Timestamp, Path]:
    try:
        timestamp = pd.to_datetime(feature_dir.name, format="%Y%m%d_%H%M%S", utc=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("--feature-dir must be a timestamped canonical shared feature store") from exc
    if feature_dir.parent.name != "features" or not feature_dir.is_dir():
        raise ValueError("--feature-dir must be an existing data_root/features/<timestamp> directory")
    return pd.Timestamp(timestamp), feature_dir.parent.parent


def _canonical_side(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    names = values.astype(str).str.lower().str.strip()
    return numeric.where(numeric.notna(), np.where(names.isin(("short", "sell", "-1", "s")), -1, 1)).astype("int8")


def _static_schema(feature_dir: Path) -> set[str]:
    """Read feature-store metadata only, to avoid requesting sidecar fields."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - production dependency.
        raise ImportError("pyarrow is required to inspect the canonical feature store") from exc
    fields: set[str] = set()
    for path in feature_dir.glob("symbol=*.parquet"):
        fields.update(map(str, pq.read_schema(path).names))
    return fields


def _load_static_matrix(frame: pd.DataFrame, feature_columns: list[str], feature_dir: Path, columns: PathGeometryColumns) -> pd.DataFrame:
    timestamp, data_root = _feature_store_location(feature_dir)
    matrix = pd.DataFrame(index=frame.index, columns=feature_columns, dtype=np.float32)
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame[columns.timestamp], utc=True))
    static = read_static_features(
        feature_store_ts=timestamp, data_root=data_root, feature_keys=feature_columns,
        symbols=[str(value) for value in frame[columns.symbol].unique()], start_ts=timestamps.min(), end_ts=timestamps.max(),
    )
    if static is None:
        return matrix
    for symbol, positions in frame.groupby(columns.symbol, sort=False).indices.items():
        rows = np.asarray(positions, dtype=np.int64)
        if not hasattr(static, "symbol_frame"):
            continue
        source = static.symbol_frame(str(symbol), keys=feature_columns).reindex(columns=feature_columns)
        source.index = pd.DatetimeIndex(pd.to_datetime(source.index, utc=True, errors="coerce"))
        if not source.index.is_unique:
            raise ValueError("canonical static feature store has duplicate symbol timestamps")
        matrix.iloc[rows] = source.reindex(timestamps[rows]).to_numpy(dtype=np.float32, copy=False)
    return matrix


def _load_sidecar_matrix(frame: pd.DataFrame, feature_columns: list[str], sidecar: Path, columns: PathGeometryColumns) -> pd.DataFrame:
    """Column-pruned exact-key frozen AE/GMM sidecar read using DuckDB."""
    try:
        import duckdb
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - production dependency.
        raise ImportError("duckdb and pyarrow are required for selective AE/GMM sidecar reads") from exc
    available = set(pq.read_schema(sidecar).names)
    requested = [name for name in feature_columns if name in available]
    matrix = pd.DataFrame(index=frame.index, columns=feature_columns, dtype=np.float32)
    if not requested:
        return matrix
    identity = pd.DataFrame({
        "__row_id__": np.arange(len(frame), dtype=np.int64),
        "__ts__": pd.to_datetime(frame[columns.timestamp], utc=True),
        "__symbol__": frame[columns.symbol].astype(str).to_numpy(),
        "side": _canonical_side(frame[columns.side]).to_numpy(),
    })
    quote = lambda value: '"' + str(value).replace('"', '""') + '"'
    projection = ", ".join(f"s.{quote(name)} AS {quote(name)}" for name in requested)
    con = duckdb.connect()
    try:
        con.execute("SET TimeZone='UTC'")
        con.register("label_keys", identity)
        loaded = con.execute(
            f"SELECT l.__row_id__, {projection} FROM label_keys l JOIN read_parquet(?) s "
            "ON epoch_ns(l.__ts__) = epoch_ns(s.__ts__) AND l.__symbol__ = s.__symbol__ "
            "AND l.side = CAST(s.side AS TINYINT) ORDER BY l.__row_id__",
            [str(sidecar)],
        ).fetchdf()
    finally:
        con.close()
    if len(loaded) != len(frame):
        raise ValueError("frozen AE/GMM sidecar does not cover every label identity")
    matrix.loc[:, requested] = loaded.loc[:, requested].to_numpy(dtype=np.float32, copy=False)
    return matrix


def _load_canonical_features(frame: pd.DataFrame, feature_columns: list[str], feature_dir: Path, sidecar: Path | None, columns: PathGeometryColumns) -> pd.DataFrame:
    """Reuse the canonical static-store endpoint and fill frozen AE/GMM fields selectively."""
    static_columns = [name for name in feature_columns if name in _static_schema(feature_dir)]
    matrix = pd.DataFrame(index=frame.index, columns=feature_columns, dtype=np.float32)
    if static_columns:
        matrix.loc[:, static_columns] = _load_static_matrix(frame, static_columns, feature_dir, columns)
    missing = [name for name in feature_columns if matrix[name].isna().all()]
    if missing and sidecar is not None:
        sidecar_matrix = _load_sidecar_matrix(frame, missing, sidecar, columns)
        matrix.loc[:, missing] = sidecar_matrix.loc[:, missing]
    if matrix.isna().all(axis=1).any() or matrix.loc[:, feature_columns].isna().all().any():
        raise ValueError("canonical feature store/sidecar did not cover every frozen model feature")
    result = frame.copy()
    result.loc[:, feature_columns] = matrix.loc[:, feature_columns]
    return result


def run(
    input_path: Path,
    output_dir: Path,
    feature_columns: list[str] | None = None,
    model_params: Mapping[str, Any] | None = None,
    *,
    columns: PathGeometryColumns = PathGeometryColumns(),
    incumbent: PathGeometryConfig = PathGeometryConfig(),
    nested_oof: bool = False,
    run_post_search_refits: bool = False,
    max_rows: int = 0,
    features_parquet: Path | None = None,
    feature_dir: Path | None = None,
    frozen_ae_gmm_sidecar: Path | None = None,
    feature_join_columns: list[str] | None = None,
    max_joint_trials: int = 24,
    max_train_rows_per_fold: int = DEFAULT_MAX_TRAIN_ROWS_PER_FOLD,
    ablation_start_date: str | None = None,
    classifier_selection_hpo_contract: Path | None = None,
    features_json_path: Path | None = None,
    catboost_params_json_path: Path | None = None,
    unsafe_allow_unverified_inputs: bool = False,
    checkpoint_path: Path | None = None,
    exact_checkpoint_geometry_id: str | None = None,
) -> dict[str, Any]:
    if classifier_selection_hpo_contract is not None:
        (
            _,
            _,
            contract_feature_columns,
            contract_model_params,
        ) = _completed_classifier_selection_hpo_contract(
            classifier_selection_hpo_contract
        )
        input_provenance = _verify_classifier_selection_hpo_contract(
            classifier_selection_hpo_contract,
            feature_columns=feature_columns,
            model_params=model_params,
            features_json_path=features_json_path,
            catboost_params_json_path=catboost_params_json_path,
        )
        feature_columns = contract_feature_columns
        model_params = contract_model_params
    else:
        if not unsafe_allow_unverified_inputs:
            raise ValueError(
                "canonical geometry search requires verified classifier selection/HPO input provenance"
            )
        if feature_columns is None or model_params is None:
            raise ValueError(
                "unverified geometry search requires explicit feature columns and CatBoost params"
            )
        input_provenance = (
            _unsafe_input_provenance(
                features_json_path=features_json_path,
                catboost_params_json_path=catboost_params_json_path,
            )
            if features_json_path is not None and catboost_params_json_path is not None
            else {"verification": "unsafe_unverified_inputs"}
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_path or output_dir / "geometry_search_checkpoint.json"
    frame = pd.read_parquet(input_path)
    rows_before_path_validity = int(len(frame))
    if "path_arch_complete_24h" in frame.columns:
        frame = frame.loc[frame["path_arch_complete_24h"].fillna(False).astype(bool)].copy()
    for required_positive in (columns.atr_fraction, columns.risk_fraction):
        if required_positive in frame.columns:
            values = pd.to_numeric(frame[required_positive], errors="coerce")
            frame = frame.loc[np.isfinite(values.to_numpy(dtype=float)) & values.gt(0.0)].copy()
    if frame.empty:
        raise ValueError("no complete finite path rows remain for geometry search")
    if max_rows > 0:
        frame = frame.iloc[:max_rows].copy()
    join_columns = feature_join_columns or [columns.timestamp, columns.symbol, columns.side]
    if feature_dir is not None:
        frame = _load_canonical_features(frame, feature_columns, feature_dir, frozen_ae_gmm_sidecar, columns)
    else:
        frame = _join_frozen_features(frame, features_parquet, feature_columns, join_columns, columns)
    frame = ensure_risk_fraction(frame, columns)
    checkpoint_input_identity = {
        "input_path": str(input_path.resolve()),
        "input_sha256": _file_sha256(input_path),
        "prepared_frame_sha256": _prepared_frame_identity(frame),
        "prepared_rows": int(len(frame)),
        "classifier_selection_hpo_provenance": dict(input_provenance),
        "features_parquet_sha256": _file_sha256(features_parquet) if features_parquet is not None else None,
        "feature_dir": str(feature_dir.resolve()) if feature_dir is not None else None,
        "frozen_ae_gmm_sidecar_sha256": _file_sha256(frozen_ae_gmm_sidecar) if frozen_ae_gmm_sidecar is not None else None,
    }
    if exact_checkpoint_geometry_id is not None:
        export = export_checkpoint_geometry(
            frame,
            feature_columns,
            model_params,
            checkpoint_path=checkpoint_path,
            config_id=exact_checkpoint_geometry_id,
            columns=columns,
            max_train_rows_per_fold=max_train_rows_per_fold,
            checkpoint_input_identity=checkpoint_input_identity,
            progress_reporter=_geometry_progress,
        )
        manifest_path, manifest = _write_exact_geometry_export(
            output_dir,
            export,
            feature_columns,
        )
        (output_dir / "geometry_search_manifest.json").write_text(
            json.dumps(
                _json_safe(
                    {
                        "schema": "catboost_path_archetype_exact_geometry_export_runner_v1",
                        "exact_geometry_oos_predictions_manifest": str(manifest_path),
                        "exact_geometry_oos_predictions": manifest,
                        "classifier_selection_hpo_provenance": dict(input_provenance),
                        "input_path": str(input_path),
                        "rows_before_path_validity": rows_before_path_validity,
                        "rows_after_path_validity": int(len(frame)),
                        "future_refit_contract": {
                            "class_merge": dict(export["class_merge"]),
                            "class_order": export["class_order"],
                            "sample_weight_contract": export["sample_weight_contract"],
                            "probability_output": export["probability_output"],
                            "raw_scoring_contract": export["raw_scoring_contract"],
                        },
                    }
                ),
                indent=2,
                sort_keys=True,
            ) + "\n"
        )
        return export
    report = staged_geometry_search(
        frame,
        feature_columns,
        model_params,
        columns=columns,
        incumbent=incumbent,
        max_joint_trials=max_joint_trials,
        ablation_start_date=ablation_start_date,
        nested_oof=nested_oof,
        capture_predictions=True,
        run_post_search_refits=run_post_search_refits,
        max_train_rows_per_fold=max_train_rows_per_fold,
        checkpoint_path=checkpoint_path,
        checkpoint_input_identity=checkpoint_input_identity,
        progress_reporter=_geometry_progress,
    )
    _parquet_ready(report["sweep_results"]).to_parquet(
        output_dir / "geometry_sweeps.parquet", index=False
    )
    _parquet_ready(report["fold_reports"]).to_parquet(
        output_dir / "geometry_fold_reports.parquet", index=False
    )
    _parquet_ready(report["boundary"]).to_parquet(
        output_dir / "selected_boundary_diagnostics.parquet", index=False
    )
    selected_tables = (
        "temporal_month_stability",
        "side_stability",
        "symbol_stability",
        "side_support",
        "symbol_support",
        "selected_side_diagnostics",
        "selected_month_diagnostics",
        "selected_probability_reliability_bins",
        "selected_economic_confusion",
        "selected_economic_confusion_priors",
    )
    for key in selected_tables:
        table = report.get(key)
        if isinstance(table, pd.DataFrame):
            _parquet_ready(table).to_parquet(
                output_dir / f"selected_{key}.parquet", index=False
            )
    if run_post_search_refits:
        finalist_manifest_path, finalist_manifest = _write_finalist_predictions(
            output_dir,
            report["finalist_oos_predictions"],
            feature_columns,
        )
    else:
        finalist_manifest_path = output_dir / "finalist_oos_predictions_manifest.json"
        finalist_manifest = {
            "schema": "catboost_path_geometry_finalist_oos_predictions_v1",
            "status": "skipped_by_contract",
            "reason": "run_post_search_refits_false",
            "finalists": [],
            "feature_columns": list(feature_columns),
        }
        finalist_manifest_path.write_text(
            json.dumps(_json_safe(finalist_manifest), indent=2, sort_keys=True) + "\n"
        )
    tabular = {
        "sweep_results", "fold_reports", "selected_fold_reports", "boundary", "temporal_month_stability", "side_stability",
        "symbol_stability", "side_support", "symbol_support", "finalist_oos_predictions", *selected_tables,
    }
    manifest = {key: value for key, value in report.items() if key not in tabular}
    checkpoint_details = dict(report.get("search_contract", {}).get("checkpoint", {}))
    manifest.update({
        "input_path": str(input_path), "features_parquet": str(features_parquet) if features_parquet else None,
        "feature_dir": str(feature_dir) if feature_dir else None,
        "frozen_ae_gmm_sidecar": str(frozen_ae_gmm_sidecar) if frozen_ae_gmm_sidecar else None,
        "feature_join_columns": join_columns, "columns": asdict(columns), "incumbent": asdict(incumbent),
        "rows_before_path_validity": rows_before_path_validity,
        "rows_after_path_validity": int(len(frame)),
        "geometry_evaluation_contract": {
            "name": "4_month_train_4_month_oos",
            "train_months": GEOMETRY_TRAIN_MONTHS,
            "oos_months": GEOMETRY_OOS_MONTHS,
            "walk_forward_cadence_months": GEOMETRY_TRAIN_MONTHS,
            "nested_minimum_months": GEOMETRY_NESTED_MONTHS,
            "max_train_rows_per_fold": int(max_train_rows_per_fold),
            "default_max_train_rows_per_fold": DEFAULT_MAX_TRAIN_ROWS_PER_FOLD,
            "oos_row_contract": "all_labelled_oos_rows",
        },
        "fold_reports_path": str(output_dir / "geometry_fold_reports.parquet"),
        "selected_fold_row_usage": report["selected_fold_reports"].to_dict(orient="records"),
        "path_validity_contract": "complete 24h path when available; finite positive ATR/risk fractions",
        "top5_finalist_predictions_manifest": str(finalist_manifest_path),
        "top5_finalist_prediction_paths": [item["path"] for item in finalist_manifest["finalists"]],
        "prediction_identity_contract": "source row position + UTC timestamp + symbol + side + fold; candidate_id included when available",
        "classifier_selection_hpo_provenance": dict(input_provenance),
        "geometry_search_checkpoint": str(checkpoint_path),
        "geometry_search_checkpoint_fingerprint": checkpoint_details.get("fingerprint"),
        "geometry_search_checkpoint_status": "complete" if checkpoint_details else "not_reported",
        "future_refit_contract": (
            {
                "class_merge": {
                    "merged_class": "fast_realization_winner",
                    "source_classes": [
                        "fast_clean_winner",
                        "fast_winner_early_drawdown",
                    ],
                },
                "class_order": [
                    "immediate_adverse_path",
                    "early_mfe_full_reversal",
                    "fast_realization_winner",
                    "late_breakout",
                    "slow_grinder",
                    "noisy_timeout_usable_mfe",
                    "dead_timeout",
                ],
                "sample_weight_contract": "uniform_weights_v1",
                "probability_output": "raw_catboost_predict_proba",
                "raw_scoring_contract": {
                    "max_probability": "max(all_7_raw_probabilities)",
                    "normalized_entropy": "-sum(p_i * log(p_i)) / log(7)",
                    "top2_probability_margin": "largest_raw_probability - second_largest_raw_probability",
                    "adverse_probability_mass": "sum(immediate_adverse_path, early_mfe_full_reversal, dead_timeout)",
                    "favorable_probability_mass": "sum(fast_realization_winner, late_breakout, slow_grinder)",
                    "neutral_classes": ["noisy_timeout_usable_mfe"],
                },
                "refit_invocation": "--export-checkpoint-geometry-id <geometry_id>",
            }
        ),
    })
    (output_dir / "geometry_search_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path-summary parquet with raw 1h..12h path columns.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--features-json",
        type=Path,
        help="Optional compatibility feature list; must exactly match the classifier contract.",
    )
    parser.add_argument(
        "--catboost-params-json",
        type=Path,
        help="Optional compatibility effective params; must exactly match the classifier contract.",
    )
    parser.add_argument(
        "--classifier-selection-hpo-contract",
        type=Path,
        help="Completed classifier feature_selection_hpo_contract.json or its artifact directory.",
    )
    parser.add_argument(
        "--unsafe-allow-unverified-inputs",
        action="store_true",
        help="Explicitly allow unverified feature/parameter JSONs for non-canonical local compatibility only.",
    )
    parser.add_argument("--features-parquet", type=Path, help="Canonical frozen-feature parquet to join when labels do not include model features.")
    parser.add_argument("--feature-dir", type=Path, help="Timestamped canonical static feature store; reads only frozen requested columns.")
    parser.add_argument("--frozen-ae-gmm-sidecar", type=Path, help="Optional exact-key frozen AE/GMM sidecar for requested representation features.")
    parser.add_argument("--feature-join-columns", default="__ts__,__symbol__,side", help="Comma-separated exact identity keys for --features-parquet.")
    parser.add_argument("--columns-json", type=Path, help="Optional PathGeometryColumns overrides.")
    parser.add_argument("--incumbent-json", type=Path, help="Optional PathGeometryConfig overrides.")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-joint-trials", type=int, default=24)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        help="Optional resumable geometry-search checkpoint (default: <output-dir>/geometry_search_checkpoint.json).",
    )
    parser.add_argument(
        "--export-checkpoint-geometry-id",
        help=(
            "Refit and export raw OOS probabilities for exactly one completed geometry "
            "from --checkpoint-path. This never reruns the geometry sweep."
        ),
    )
    parser.add_argument(
        "--max-train-rows-per-fold",
        type=int,
        default=DEFAULT_MAX_TRAIN_ROWS_PER_FOLD,
        help=(
            "Deterministic side x geometry-class chronological training cap per fold "
            f"(default: {DEFAULT_MAX_TRAIN_ROWS_PER_FOLD}; 0 keeps all training rows)."
        ),
    )
    parser.add_argument(
        "--ablation-start-date",
        help="UTC start of the fixed 4m train -> next 4m OOS target-selection split.",
    )
    parser.add_argument(
        "--nested-oof",
        action="store_true",
        help="Run nested 4m/4m finalist validation when at least 12 months are available.",
    )
    parser.add_argument(
        "--run-finalist-refits",
        action="store_true",
        help="Explicitly refit the top-five raw seven-class finalists after the sweep.",
    )
    args = parser.parse_args()
    columns = PathGeometryColumns(**(_read_json(args.columns_json) if args.columns_json else {}))
    incumbent = PathGeometryConfig(**(_read_json(args.incumbent_json) if args.incumbent_json else {}))
    if (args.features_json is None) != (args.catboost_params_json is None):
        parser.error(
            "--features-json and --catboost-params-json must be supplied together"
        )
    params = None
    feature_columns = None
    if args.catboost_params_json is not None:
        params = _read_json(args.catboost_params_json)
        if not isinstance(params, Mapping):
            raise ValueError("catboost params JSON must contain an object")
        feature_columns = _feature_columns(args.features_json)
    if args.classifier_selection_hpo_contract is None and not args.unsafe_allow_unverified_inputs:
        parser.error(
            "--classifier-selection-hpo-contract is required for canonical geometry search; "
            "use --unsafe-allow-unverified-inputs only for an explicitly non-canonical run"
        )
    if args.classifier_selection_hpo_contract is None and feature_columns is None:
        parser.error(
            "unverified geometry search requires --features-json and --catboost-params-json"
        )
    if args.export_checkpoint_geometry_id is not None and args.checkpoint_path is None:
        parser.error("--export-checkpoint-geometry-id requires --checkpoint-path")
    run(
        args.input, args.output_dir, feature_columns, params,
        columns=columns, incumbent=incumbent, max_rows=args.max_rows,
        features_parquet=args.features_parquet,
        feature_dir=args.feature_dir,
        frozen_ae_gmm_sidecar=args.frozen_ae_gmm_sidecar,
        feature_join_columns=[value for value in args.feature_join_columns.split(",") if value],
        max_joint_trials=args.max_joint_trials,
        max_train_rows_per_fold=args.max_train_rows_per_fold,
        ablation_start_date=args.ablation_start_date,
        nested_oof=args.nested_oof,
        run_post_search_refits=args.run_finalist_refits,
        classifier_selection_hpo_contract=args.classifier_selection_hpo_contract,
        features_json_path=args.features_json,
        catboost_params_json_path=args.catboost_params_json,
        unsafe_allow_unverified_inputs=args.unsafe_allow_unverified_inputs,
        checkpoint_path=args.checkpoint_path,
        exact_checkpoint_geometry_id=args.export_checkpoint_geometry_id,
    )


if __name__ == "__main__":
    main()
