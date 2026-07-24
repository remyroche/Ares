#!/usr/bin/env python3
"""Normalize retained raw CatBoost path-archetype OOF probabilities for execution EV."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

SCHEMA_VERSION = "execution_ev_catboost_raw_oof_adapter_v1"
PREDICTION_ROLE = "path_archetype_oof"
RAW_SOURCE_ROLE = "pre_refinement_path_archetype_oof_raw"
RAW_SOURCE_PROBABILITY_ROLE = "pre_refinement_path_archetype_oof_raw_probability"
EXACT_GEOMETRY_SCHEMA = "catboost_path_archetype_exact_geometry_raw_oos_export_v1"
RETAINED_GEOMETRY_ID = "geometry_e33b290e324f3182"
RAW_HARD_LABEL_TARGET = "seven_class_path_geometry"
RAW_SAMPLE_WEIGHT_CONTRACT = "uniform_weights_v1"
RAW_PROBABILITY_OUTPUT = "raw_catboost_predict_proba"
JOIN_KEYS = ("__ts__", "__symbol__", "side_name")
EVIDENCE_COLUMNS = (
    "oof_fold",
    "available_at",
    "validation_start",
    "train_decision_cutoff",
    "label_resolution_available_at",
)
PROBABILITY_PREFIX = "probability__"
RAW_CLASSES = (
    "immediate_adverse_path",
    "early_mfe_full_reversal",
    "fast_realization_winner",
    "late_breakout",
    "slow_grinder",
    "noisy_timeout_usable_mfe",
    "dead_timeout",
)
FAST_WINNER_CLASS_MERGE = {
    "merged_class": "fast_realization_winner",
    "source_classes": ["fast_clean_winner", "fast_winner_early_drawdown"],
}
ADVERSE_CLASSES = (
    "immediate_adverse_path",
    "early_mfe_full_reversal",
    "dead_timeout",
)
FAVORABLE_CLASSES = (
    "fast_realization_winner",
    "late_breakout",
    "slow_grinder",
)
NEUTRAL_CLASSES = ("noisy_timeout_usable_mfe",)
RAW_DERIVED_COLUMNS = {
    "max_probability": "raw_max_probability",
    "normalized_entropy": "raw_normalized_entropy",
    "top2_probability_margin": "raw_top1_top2_probability_margin",
    "adverse_probability_mass": "raw_adverse_probability_mass",
    "favorable_probability_mass": "raw_favorable_probability_mass",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_hash(
    payload: Mapping[str, Any], *, excluded: Sequence[str] = ()
) -> str:
    canonical = {key: value for key, value in payload.items() if key not in excluded}
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _class_order_sha256(class_names: Sequence[str]) -> str:
    encoded = json.dumps(
        list(class_names), separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc(values: pd.Series, *, source: str, column: str) -> pd.Series:
    converted = pd.to_datetime(values, utc=True, errors="coerce")
    if converted.isna().any():
        raise ValueError(f"{source}: {column!r} contains null or invalid timestamps")
    return converted


def _identity_strings(values: pd.Series, *, source: str, column: str) -> pd.Series:
    converted = values.astype("string").str.strip()
    if converted.isna().any() or converted.eq("").any():
        raise ValueError(f"{source}: {column!r} contains null or blank identities")
    return converted.astype(str)


def _canonical_side(values: pd.Series, *, source: str) -> pd.Series:
    normalized = values.astype("string").str.strip().str.lower()
    mapped = normalized.map(
        {
            "1": "long",
            "1.0": "long",
            "long": "long",
            "-1": "short",
            "-1.0": "short",
            "short": "short",
        }
    )
    if mapped.isna().any():
        invalid = sorted(normalized[mapped.isna()].dropna().unique().tolist())
        raise ValueError(
            f"{source}: side must be long/short or +1/-1; invalid={invalid[:5]!r}"
        )
    return mapped.astype(str)


def _require_unique(frame: pd.DataFrame, keys: Sequence[str], *, source: str) -> None:
    duplicate_rows = int(frame.duplicated(list(keys), keep=False).sum())
    if duplicate_rows:
        raise ValueError(
            f"{source}: duplicate exact identities on {list(keys)!r}; "
            f"duplicate_rows={duplicate_rows}"
        )


def _load_json(path: Path, *, source: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{source} is not readable JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must be a JSON object")
    return payload


def _require_exact(value: Any, expected: Any, *, field: str) -> None:
    if value != expected:
        raise ValueError(f"raw source contract requires {field}={expected!r}")


def _validate_signed_raw_source_manifest(
    path: Path, predictions_path: Path
) -> dict[str, Any]:
    payload = _load_json(path, source="raw source role manifest")
    signed_hash = payload.get("prediction_role_manifest_sha256")
    expected_hash = _canonical_json_hash(
        payload, excluded=("prediction_role_manifest_sha256",)
    )
    if not isinstance(signed_hash, str) or not hmac.compare_digest(
        signed_hash, expected_hash
    ):
        raise ValueError("raw source role manifest signature does not verify")
    _require_exact(
        payload.get("prediction_role"), RAW_SOURCE_ROLE, field="prediction_role"
    )
    artifact_hash = payload.get("source_artifact_sha256")
    if not isinstance(artifact_hash, str) or not hmac.compare_digest(
        artifact_hash, _sha256(predictions_path)
    ):
        raise ValueError("raw source role manifest does not bind the OOS parquet")
    _require_exact(payload.get("class_order"), list(RAW_CLASSES), field="class_order")
    declarations = payload.get("prediction_columns")
    if not isinstance(declarations, Mapping):
        raise ValueError("raw source role manifest requires prediction_columns")
    expected_columns = {f"probability_{name}" for name in RAW_CLASSES}
    if set(declarations) != expected_columns:
        raise ValueError(
            "raw source role manifest must declare exactly the raw seven-class "
            "probability vector"
        )
    for column in expected_columns:
        declaration = declarations[column]
        if not isinstance(declaration, Mapping) or declaration.get("role") != (
            RAW_SOURCE_PROBABILITY_ROLE
        ) or declaration.get("target") is not False:
            raise ValueError(
                "raw source role manifest contains calibrated, soft, refinement, "
                "or target probability declarations"
            )
    return payload


def _load_exact_geometry_manifest(
    path: Path, predictions_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _load_json(path, source="exact geometry manifest")
    _require_exact(payload.get("schema"), EXACT_GEOMETRY_SCHEMA, field="schema")
    _require_exact(payload.get("config_id"), RETAINED_GEOMETRY_ID, field="config_id")
    _require_exact(
        payload.get("hard_label_target"),
        RAW_HARD_LABEL_TARGET,
        field="hard_label_target",
    )
    _require_exact(payload.get("class_order"), list(RAW_CLASSES), field="class_order")
    _require_exact(
        payload.get("class_merge"), FAST_WINNER_CLASS_MERGE, field="class_merge"
    )
    _require_exact(
        payload.get("sample_weight_contract"),
        RAW_SAMPLE_WEIGHT_CONTRACT,
        field="sample_weight_contract",
    )
    _require_exact(
        payload.get("probability_output"),
        RAW_PROBABILITY_OUTPUT,
        field="probability_output",
    )
    _require_exact(
        payload.get("calibration_required_for_raw_output"),
        False,
        field="calibration_required_for_raw_output",
    )
    _require_exact(
        payload.get("centroid_required_for_raw_output"),
        False,
        field="centroid_required_for_raw_output",
    )
    prediction_hash = payload.get("prediction_sha256")
    if not isinstance(prediction_hash, str) or not hmac.compare_digest(
        prediction_hash, _sha256(predictions_path)
    ):
        raise ValueError("exact geometry manifest does not bind the OOS parquet")
    source_role_path = payload.get("prediction_role_manifest")
    if not isinstance(source_role_path, str) or not source_role_path:
        raise ValueError("exact geometry manifest requires a raw source role manifest")
    role_path = Path(source_role_path)
    if not role_path.is_absolute():
        role_path = path.parent / role_path
    if not role_path.is_file():
        raise ValueError("exact geometry manifest raw source role manifest is absent")
    expected_role_hash = payload.get("prediction_role_manifest_sha256")
    if not isinstance(expected_role_hash, str) or not hmac.compare_digest(
        expected_role_hash, _sha256(role_path)
    ):
        raise ValueError("exact geometry manifest does not bind its raw source role manifest")
    return payload, _validate_signed_raw_source_manifest(role_path, predictions_path)


def _validate_raw_probabilities(
    predictions: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    source_columns = [f"probability_{name}" for name in RAW_CLASSES]
    missing = sorted(set(source_columns).difference(predictions.columns))
    if missing:
        raise ValueError(f"raw predictions are missing probabilities: {missing!r}")
    probabilities = predictions.loc[:, source_columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError("raw probabilities contain non-finite values")
    if (probabilities < 0.0).any() or (probabilities > 1.0).any():
        raise ValueError("raw probabilities must be in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("raw probability vectors are not normalized")

    vectors = predictions["probability_vector"].tolist()
    try:
        vector_matrix = np.asarray(vectors, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("raw probability_vector is not a numeric seven-class vector") from exc
    if vector_matrix.shape != probabilities.shape or not np.allclose(
        vector_matrix, probabilities, rtol=0.0, atol=1e-12
    ):
        raise ValueError("raw probability_vector does not match the ordered probability columns")

    entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, 1e-15, 1.0)), axis=1
    )
    source_entropy = pd.to_numeric(
        predictions["probability_entropy"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(source_entropy).all() or not np.allclose(
        entropy, source_entropy, rtol=0.0, atol=1e-10
    ):
        raise ValueError("raw entropy does not match the probability vector")
    expected_class = np.asarray(RAW_CLASSES, dtype=object)[probabilities.argmax(axis=1)]
    source_class = _identity_strings(
        predictions["predicted_class"],
        source="raw predictions",
        column="predicted_class",
    ).to_numpy()
    if not np.array_equal(expected_class, source_class):
        raise ValueError("raw predicted class does not match probability argmax")
    return probabilities, entropy


def _validate_raw_probability_derivatives(
    predictions: pd.DataFrame, probabilities: np.ndarray, entropy: np.ndarray
) -> dict[str, np.ndarray]:
    missing = sorted(set(RAW_DERIVED_COLUMNS.values()).difference(predictions.columns))
    if missing:
        raise ValueError(f"raw predictions are missing derived outputs: {missing!r}")
    class_index = {name: index for index, name in enumerate(RAW_CLASSES)}
    ordered = np.sort(probabilities, axis=1)
    expected = {
        "max_probability": probabilities.max(axis=1),
        "normalized_entropy": entropy / np.log(len(RAW_CLASSES)),
        "top2_probability_margin": ordered[:, -1] - ordered[:, -2],
        "adverse_probability_mass": probabilities[
            :, [class_index[name] for name in ADVERSE_CLASSES]
        ].sum(axis=1),
        "favorable_probability_mass": probabilities[
            :, [class_index[name] for name in FAVORABLE_CLASSES]
        ].sum(axis=1),
    }
    for output_column, source_column in RAW_DERIVED_COLUMNS.items():
        observed = pd.to_numeric(
            predictions[source_column], errors="coerce"
        ).to_numpy(dtype=np.float64)
        if not np.isfinite(observed).all() or not np.allclose(
            observed, expected[output_column], rtol=0.0, atol=1e-10
        ):
            raise ValueError(
                f"raw {output_column} does not match the probability vector"
            )
    return expected


def _validate_output_columns(
    frame: pd.DataFrame, probability_columns: Sequence[str]
) -> None:
    expected = [
        *JOIN_KEYS,
        "candidate_id",
        *probability_columns,
        "probability_entropy",
        *RAW_DERIVED_COLUMNS,
        "predicted_path_archetype",
        *EVIDENCE_COLUMNS,
    ]
    if frame.columns.tolist() != expected:
        raise ValueError("adapter output contains unexpected or target columns")
    forbidden = ("true_", "target", "label", "outcome", "realized", "calibrated")
    unexpected = [
        column
        for column in frame.columns
        if column not in EVIDENCE_COLUMNS
        and any(token in column.lower() for token in forbidden)
    ]
    if unexpected:
        raise ValueError(f"adapter output contains target columns: {unexpected!r}")


def run(
    oos_predictions_path: Path,
    exact_geometry_manifest_path: Path,
    output_path: Path,
    manifest_path: Path | None = None,
) -> dict[str, Path]:
    for name, path in (
        ("raw OOS predictions", oos_predictions_path),
        ("exact geometry manifest", exact_geometry_manifest_path),
    ):
        if not path.is_file():
            raise ValueError(f"{name} does not exist: {path}")
    manifest_path = manifest_path or output_path.with_suffix(".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise ValueError("refusing to overwrite an existing adapter artifact")

    source_manifest, source_role_manifest = _load_exact_geometry_manifest(
        exact_geometry_manifest_path, oos_predictions_path
    )
    predictions = pd.read_parquet(oos_predictions_path)
    required = {
        "__ts__",
        "__symbol__",
        "side",
        "candidate_id",
        "config_id",
        "fold_id",
        "available_at",
        "validation_start",
        "train_cutoff_utc",
        "train_decision_cutoff",
        "label_resolution_available_at",
        "oos_start_utc",
        "oos_end_utc",
        "predicted_class",
        "probability_vector",
        "probability_entropy",
        *RAW_DERIVED_COLUMNS.values(),
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"raw predictions are missing columns: {missing!r}")
    if not predictions["config_id"].astype(str).eq(RETAINED_GEOMETRY_ID).all():
        raise ValueError("raw predictions do not belong to the retained geometry")

    predictions["__ts__"] = _utc(
        predictions["__ts__"], source="raw predictions", column="__ts__"
    )
    predictions["__symbol__"] = _identity_strings(
        predictions["__symbol__"], source="raw predictions", column="__symbol__"
    )
    predictions["candidate_id"] = _identity_strings(
        predictions["candidate_id"],
        source="raw predictions",
        column="candidate_id",
    )
    predictions["side_name"] = _canonical_side(
        predictions["side"], source="raw predictions"
    )
    for column in (
        "available_at",
        "validation_start",
        "train_cutoff_utc",
        "train_decision_cutoff",
        "label_resolution_available_at",
        "oos_start_utc",
        "oos_end_utc",
    ):
        predictions[column] = _utc(
            predictions[column], source="raw predictions", column=column
        )
    _require_unique(
        predictions, [*JOIN_KEYS, "candidate_id"], source="raw predictions"
    )
    if not (
        (predictions["available_at"] <= predictions["__ts__"])
        & (predictions["label_resolution_available_at"] <= predictions["train_decision_cutoff"])
        & (predictions["train_decision_cutoff"] < predictions["validation_start"])
        & (predictions["validation_start"] <= predictions["__ts__"])
        & (predictions["__ts__"] >= predictions["oos_start_utc"])
        & (predictions["__ts__"] < predictions["oos_end_utc"])
        & (predictions["train_cutoff_utc"] == predictions["train_decision_cutoff"])
    ).all():
        raise ValueError("raw predictions violate strict OOS provenance")

    probabilities, entropy = _validate_raw_probabilities(predictions)
    derived = _validate_raw_probability_derivatives(predictions, probabilities, entropy)
    probability_columns = [f"{PROBABILITY_PREFIX}{name}" for name in RAW_CLASSES]
    output = predictions.loc[:, list(JOIN_KEYS) + ["candidate_id"]].copy()
    for index, column in enumerate(probability_columns):
        output[column] = probabilities[:, index]
    output["probability_entropy"] = entropy
    for column in RAW_DERIVED_COLUMNS:
        output[column] = derived[column]
    output["predicted_path_archetype"] = np.asarray(RAW_CLASSES, dtype=object)[
        probabilities.argmax(axis=1)
    ]
    output["oof_fold"] = _identity_strings(
        predictions["fold_id"], source="raw predictions", column="fold_id"
    )
    for column in EVIDENCE_COLUMNS[1:]:
        output[column] = predictions[column].to_numpy()
    _validate_output_columns(output, probability_columns)
    _require_unique(output, [*JOIN_KEYS, "candidate_id"], source="adapter output")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False, compression="zstd")
    role_manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "prediction_role": PREDICTION_ROLE,
        "source_artifact_sha256": _sha256(output_path),
        "source_hashes": {
            "raw_oos_predictions_sha256": _sha256(oos_predictions_path),
            "exact_geometry_manifest_sha256": _sha256(exact_geometry_manifest_path),
            "raw_source_role_manifest_sha256": _sha256(
                Path(source_manifest["prediction_role_manifest"])
                if Path(source_manifest["prediction_role_manifest"]).is_absolute()
                else exact_geometry_manifest_path.parent
                / str(source_manifest["prediction_role_manifest"])
            ),
        },
        "rows": int(len(output)),
        "identity_columns": list(JOIN_KEYS) + ["candidate_id"],
        "prediction_columns": {
            **{
                column: {
                    "role": "pre_entry_path_archetype_oof_raw_probability",
                    "target": False,
                }
                for column in probability_columns
            },
            "probability_entropy": {
                "role": "pre_entry_path_archetype_oof_raw_entropy",
                "target": False,
            },
            **{
                column: {
                    "role": "pre_entry_path_archetype_oof_raw_probability_summary",
                    "target": False,
                }
                for column in RAW_DERIVED_COLUMNS
            },
            "predicted_path_archetype": {
                "role": "pre_entry_path_archetype_oof_raw_prediction",
                "target": False,
            },
        },
        "path_shape_types": list(RAW_CLASSES),
        "class_names": list(RAW_CLASSES),
        "class_order_sha256": _class_order_sha256(RAW_CLASSES),
        "raw_probability_contract": {
            "source_prediction_role": source_role_manifest["prediction_role"],
            "retained_geometry_id": RETAINED_GEOMETRY_ID,
            "hard_label_target": RAW_HARD_LABEL_TARGET,
            "class_merge": FAST_WINNER_CLASS_MERGE,
            "sample_weight_contract": RAW_SAMPLE_WEIGHT_CONTRACT,
            "probability_output": RAW_PROBABILITY_OUTPUT,
            "calibration_required_for_raw_output": False,
            "centroid_required_for_raw_output": False,
        },
        "raw_probability_derivations": {
            "max_probability": "max(ordered seven-class raw probability vector)",
            "normalized_entropy": "probability_entropy / log(7)",
            "top2_probability_margin": "largest raw probability - second_largest raw probability",
            "adverse_probability_mass": {
                "classes": list(ADVERSE_CLASSES),
                "formula": "sum(raw probability over adverse classes)",
            },
            "favorable_probability_mass": {
                "classes": list(FAVORABLE_CLASSES),
                "formula": "sum(raw probability over favorable classes)",
            },
            "neutral_classes": list(NEUTRAL_CLASSES),
        },
        "fold_provenance_columns": {
            "fold": "oof_fold",
            "validation_start": "validation_start",
            "training_information_cutoff": "train_decision_cutoff",
            "latest_resolved_training_label": "label_resolution_available_at",
            "prediction_available_at": "available_at",
        },
        "target_columns_emitted": [],
    }
    role_manifest["prediction_role_manifest_sha256"] = _canonical_json_hash(
        role_manifest
    )
    manifest_path.write_text(
        json.dumps(role_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"oof": output_path, "manifest": manifest_path}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oos-predictions", required=True, type=Path)
    parser.add_argument("--exact-geometry-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        paths = run(
            args.oos_predictions,
            args.exact_geometry_manifest,
            args.output,
            args.manifest,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"CatBoost raw OOF adaptation failed: {exc}") from exc
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
