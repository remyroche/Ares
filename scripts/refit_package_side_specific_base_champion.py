#!/usr/bin/env python3
"""Final-refit and package a validated sliding-365 per-side base champion.

This is intentionally a packaging path, not an experiment runner.  It consumes
the completed base report and its latest fold cache, reuses its frozen
parameters, target/weight payloads, and AE/GMM state, and writes a promotion
candidate that is explicitly excluded from every OOS metric.

Some completed runs retain only compact fixed-training payloads.  Those caches
deliberately omit the wide ``train`` frame and its timestamps.  Recovery mode
reconstructs *only* row identities from the immutable label store, proves their
alignment with the cached target/side payloads, and then applies the exact
trailing-window trim to the cached feature matrix.  It never recomputes labels,
weights, features, or AE/GMM outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from pyarrow import parquet as pq

from extreme_price_movements.base_side_target_contract import (
    TARGET_MODE as PROMOTED_TARGET_MODE,
    WEIGHT_ARM as PROMOTED_WEIGHT_ARM,
    promoted_side_target_provenance,
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SIDES = ("long", "short")
SCHEMA = "side_specific_sliding365_base_champion_final_refit_v1"
REQUIRED_PARAMS = (
    "n_estimators",
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "loss_function",
    "min_split_gain",
    "target_mode",
    "weight_arm",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        _json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _feature_contract_hash(features: Sequence[str]) -> str:
    return _sha256_json([str(feature) for feature in features])


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _resolve_path(value: Any, *, report_dir: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    for candidate in (report_dir / path, ROOT / path):
        if candidate.exists():
            return candidate.resolve()
    return (ROOT / path).resolve()


def _normalise_hash(value: Any) -> str:
    text = str(value or "")
    return text if text.startswith("sha256:") else f"sha256:{text}"


def _require_exact_hash(*, name: str, actual: str, expected: Any) -> None:
    if not expected:
        raise ValueError(f"{name} hash is missing")
    if _normalise_hash(actual) != _normalise_hash(expected):
        raise ValueError(
            f"{name} hash mismatch: expected={_normalise_hash(expected)} "
            f"actual={_normalise_hash(actual)}"
        )


def _normalise_params(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = payload.get("params", payload)
    if not isinstance(raw, Mapping):
        raise ValueError("Stage-C params payload must be an object or contain params")
    missing = [name for name in REQUIRED_PARAMS if name not in raw]
    if missing:
        raise ValueError(f"Stage-C params missing required fields: {missing}")
    params = {name: raw[name] for name in REQUIRED_PARAMS}
    for name in ("n_estimators", "num_leaves", "max_depth", "min_child_samples"):
        params[name] = int(params[name])
    for name in (
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "min_split_gain",
    ):
        params[name] = float(params[name])
    for name in ("loss_function", "target_mode", "weight_arm"):
        params[name] = str(params[name])
    return params


def _side_features(payload: Mapping[str, Any], *, source: str) -> dict[str, list[str]]:
    raw = payload.get("selected_features_by_side")
    if not isinstance(raw, Mapping):
        raise ValueError(f"{source} is missing selected_features_by_side")
    result: dict[str, list[str]] = {}
    for side in SIDES:
        values = raw.get(side)
        if not isinstance(values, list):
            raise ValueError(f"{source} is missing {side} selected features")
        features = [str(value) for value in values if str(value).strip()]
        if not features or len(set(features)) != len(features):
            raise ValueError(f"{source} has invalid {side} selected features")
        result[side] = features
    return result


def _latest_fold_from_cache(cache_dir: Path) -> tuple[Path, dict[str, Any]]:
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Fold cache directory not found: {cache_dir}")
    candidates: list[tuple[pd.Timestamp, Path, dict[str, Any]]] = []
    for path in cache_dir.glob("*/fold_manifest.json"):
        manifest = _read_json(path)
        valid_end = pd.to_datetime(manifest.get("valid_end"), utc=True, errors="coerce")
        if pd.isna(valid_end):
            raise ValueError(f"Fold cache has invalid valid_end: {path}")
        candidates.append((valid_end, path.parent, manifest))
    if not candidates:
        raise ValueError(f"No completed fold manifests found in {cache_dir}")
    _, fold_dir, fold = max(candidates, key=lambda item: item[0])
    return fold_dir, fold


def _latest_fold(report_dir: Path, report: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    cache_dir = _resolve_path(report.get("fold_cache_dir", ""), report_dir=report_dir)
    return _latest_fold_from_cache(cache_dir)


def _state_manifest_path(state_path: Path) -> Path:
    if state_path.name.endswith("_state.pkl"):
        return state_path.with_name(state_path.name.replace("_state.pkl", "_manifest.json"))
    return state_path.with_suffix(".json")


def _validate_ae_gmm_contract(
    *, report: Mapping[str, Any], report_dir: Path
) -> dict[str, Any]:
    state_path = _resolve_path(report.get("fixed_ae_gmm_state_pkl", ""), report_dir=report_dir)
    if not state_path.is_file():
        raise FileNotFoundError(f"Frozen AE/GMM state not found: {state_path}")
    report_state_path = _resolve_path(
        report.get("ae_gmm_state_reference_state_path", state_path), report_dir=report_dir
    )
    if report_state_path != state_path:
        raise ValueError("State hash mismatch: report state and reference state paths differ")
    source_state_hash = _sha256_file(state_path)
    transform_manifest_path = _state_manifest_path(state_path)
    transform_manifest = _read_json(transform_manifest_path)
    inputs = [str(value) for value in transform_manifest.get("input_feature_columns", [])]
    if not inputs:
        raise ValueError("Frozen AE/GMM input transform contract is missing input_feature_columns")
    input_hash = _feature_contract_hash(inputs)
    _require_exact_hash(
        name="AE/GMM input transform", actual=input_hash,
        expected=transform_manifest.get("input_feature_order_hash"),
    )
    report_inputs = [str(value) for value in report.get("ae_gmm_input_features", [])]
    if inputs != report_inputs:
        raise ValueError("AE/GMM input transform hash mismatch: report input order differs")
    state = joblib.load(state_path)
    if not isinstance(state, Mapping):
        raise ValueError("Frozen AE/GMM state must deserialize to a mapping")
    state_inputs = [str(value) for value in state.get("feature_columns", [])]
    if state_inputs != inputs:
        raise ValueError("AE/GMM input transform hash mismatch: serialized state input order differs")
    _require_exact_hash(
        name="Serialized AE/GMM state input transform",
        actual=input_hash,
        expected=state.get("input_feature_order_hash"),
    )
    sidecar_path = _resolve_path(
        report.get("frozen_ae_gmm_output_sidecar_path", ""), report_dir=report_dir
    )
    sidecar_manifest_path = sidecar_path.with_name(sidecar_path.stem + ".manifest.json")
    sidecar_manifest = _read_json(sidecar_manifest_path)
    _require_exact_hash(
        name="Serialized AE/GMM state", actual=source_state_hash,
        expected=sidecar_manifest.get("state_sha256"),
    )
    return {
        "state_path": state_path,
        "state_sha256": source_state_hash,
        "state_manifest_path": transform_manifest_path,
        "state_manifest_sha256": _sha256_file(transform_manifest_path),
        "input_features": inputs,
        "input_feature_order_hash": input_hash,
        "sidecar_manifest_path": sidecar_manifest_path,
        "sidecar_manifest_sha256": _sha256_file(sidecar_manifest_path),
        "sidecar_output_feature_hash": _normalise_hash(sidecar_manifest.get("output_feature_hash")),
        "sidecar_output_features": [
            str(value) for value in sidecar_manifest.get("output_features", [])
        ],
    }


def _normalise_side(values: pd.Series | np.ndarray) -> np.ndarray:
    series = pd.Series(values, copy=False).astype(str).str.lower()
    return np.where(series.eq("short").to_numpy(), "short", "long")


def _compact_recovery_report(
    *, report_dir: Path, compact_recovery: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a minimal report contract when the interrupted runner wrote no manifest."""

    required = {
        "labels_path",
        "fixed_params_json",
        "fixed_ae_gmm_state_pkl",
        "frozen_ae_gmm_output_sidecar_path",
    }
    missing = sorted(name for name in required if not compact_recovery.get(name))
    if missing:
        raise ValueError(
            "Compact recovery requires explicit immutable source paths: "
            f"{missing}"
        )
    fold_dir, fold = _latest_fold_from_cache(report_dir / "_fold_cache")
    if not bool(fold.get("compact_fixed_training_payload")):
        raise ValueError("Compact recovery requires compact_fixed_training_payload=true")
    if str(fold.get("payload_train_sampling", "full_train_rows")) != "full_train_rows":
        raise ValueError(
            "Compact recovery refuses sampled training payloads; exact trailing-window "
            "packaging requires full_train_rows"
        )
    selected_by_side = _side_features(fold, source="Latest compact fold cache")
    state_path = _resolve_path(
        compact_recovery["fixed_ae_gmm_state_pkl"], report_dir=report_dir
    )
    state_manifest = _read_json(_state_manifest_path(state_path))
    params_path = _resolve_path(compact_recovery["fixed_params_json"], report_dir=report_dir)
    params = _normalise_params(_read_json(params_path))
    if str(fold.get("fixed_training_target_mode")) != params["target_mode"]:
        raise ValueError("Target contract hash mismatch: compact fold and recovery params differ")
    if str(fold.get("fixed_training_weight_arm")) != params["weight_arm"]:
        raise ValueError("Target contract hash mismatch: compact fold and recovery params differ")
    if params["target_mode"] != "target_soft" or params["weight_arm"] != "W7_timestamp_balanced":
        raise ValueError(
            "Compact recovery is restricted to the corrected Pack-B incumbent "
            "target_soft/W7_timestamp_balanced contract"
        )
    labels_path = _resolve_path(compact_recovery["labels_path"], report_dir=report_dir)
    if not labels_path.exists():
        raise FileNotFoundError(f"Compact recovery label source not found: {labels_path}")
    sidecar_path = _resolve_path(
        compact_recovery["frozen_ae_gmm_output_sidecar_path"], report_dir=report_dir
    )
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"Compact recovery AE/GMM sidecar not found: {sidecar_path}")
    recovery = {
        "schema": "compact_fixed_base_refit_recovery_v1",
        "labels_path": str(labels_path),
        "labels_identity_columns": [
            "__ts__",
            "__symbol__",
            "side_name",
            "__first_touch_target_soft__",
        ],
        "fold_cache_dir": str(report_dir / "_fold_cache"),
        "latest_fold": str(fold.get("fold")),
        "params_path": str(params_path),
        "params_sha256": _sha256_file(params_path),
        "ae_gmm_state_path": str(state_path),
        "ae_gmm_state_sha256": _sha256_file(state_path),
        "ae_gmm_sidecar_path": str(sidecar_path),
        "ae_gmm_sidecar_sha256": _sha256_file(sidecar_path),
        "sidecar_path": str(sidecar_path),
        "sidecar_required_features": [
            str(value)
            for value in _read_json(
                sidecar_path.with_name(sidecar_path.stem + ".manifest.json")
            ).get("output_features", [])
        ],
    }
    report = {
        "model_side_scope": "per_side",
        "train_window_days": 365,
        "compact_fixed_training_payload": True,
        "fold_cache_dir": str(report_dir / "_fold_cache"),
        "selected_features_by_side": selected_by_side,
        "fixed_params_json": str(params_path),
        "outputs": {"best_params": str(params_path)},
        "fixed_ae_gmm_state_pkl": str(state_path),
        "ae_gmm_state_reference_state_path": str(state_path),
        "ae_gmm_input_features": list(state_manifest.get("input_feature_columns", [])),
        "frozen_ae_gmm_output_sidecar_path": str(sidecar_path),
        "_compact_recovery": recovery,
    }
    return report, recovery


def _load_report_or_compact_recovery(
    *, report_dir: Path, compact_recovery: Mapping[str, Any] | None
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    manifest_path = report_dir / "manifest.json"
    if manifest_path.is_file():
        if compact_recovery:
            raise ValueError(
                "Compact recovery arguments are only valid when the source report manifest is absent"
            )
        return _read_json(manifest_path), None
    if compact_recovery is None:
        raise FileNotFoundError(
            f"{manifest_path} is missing; provide compact recovery source paths"
        )
    return _compact_recovery_report(report_dir=report_dir, compact_recovery=compact_recovery)


def _validate_source_contract(
    *, report_dir: Path, compact_recovery: Mapping[str, Any] | None = None
) -> tuple[dict[str, Any], Path, dict[str, Any], dict[str, list[str]], dict[str, Any], dict[str, Any]]:
    report, recovery = _load_report_or_compact_recovery(
        report_dir=report_dir, compact_recovery=compact_recovery
    )
    if str(report.get("model_side_scope", "")).lower() != "per_side":
        raise ValueError("Refit requires model_side_scope=per_side")
    if int(report.get("train_window_days", 0)) != 365:
        raise ValueError("Refit requires the validated sliding365 report")
    if not bool(report.get("compact_fixed_training_payload")):
        raise ValueError("Refit requires compact fixed Stage-C target/weight payloads")
    selected_by_side = _side_features(report, source="Base report")
    fold_dir, fold = _latest_fold(report_dir, report)
    if isinstance(fold.get("selected_features_by_side"), Mapping):
        fold_features = _side_features(fold, source="Latest fold cache")
    else:
        # Reused compact payloads intentionally omit duplicate selection metadata.
        # Their Parquet schemas remain a cheap, authoritative proof of the stored
        # matrix contract without materialising multi-million-row feature frames.
        expected_union = list(
            dict.fromkeys(
                feature
                for side in SIDES
                for feature in selected_by_side[side]
            )
        )
        expected_set = set(expected_union)
        report_matrix_columns = report.get("selected_features")
        expected_matrix_columns = (
            [feature for feature in report_matrix_columns if feature in expected_set]
            if isinstance(report_matrix_columns, list)
            else sorted(expected_set)
        )
        if set(expected_matrix_columns) != expected_set:
            expected_matrix_columns = sorted(expected_set)
        for payload_name in ("x_train", "x_valid"):
            columns = list(pq.ParquetFile(_payload_path(fold_dir, fold, payload_name)).schema.names)
            if columns != expected_matrix_columns:
                raise ValueError(
                    "Feature contract hash mismatch: reused compact "
                    f"{payload_name} schema differs from the report side contracts"
                )
        fold_features = selected_by_side
    for side in SIDES:
        _require_exact_hash(
            name=f"{side} feature contract",
            actual=_feature_contract_hash(fold_features[side]),
            expected=_feature_contract_hash(selected_by_side[side]),
        )
    params_path = _resolve_path(report.get("fixed_params_json", ""), report_dir=report_dir)
    params = _normalise_params(_read_json(params_path))
    best_path = _resolve_path((report.get("outputs") or {}).get("best_params", ""), report_dir=report_dir)
    best = _normalise_params(_read_json(best_path))
    _require_exact_hash(
        name="Stage-C parameter", actual=_sha256_json(params), expected=_sha256_json(best)
    )
    if str(fold.get("fixed_training_target_mode")) != params["target_mode"]:
        raise ValueError("Target contract hash mismatch: latest fold target mode differs")
    if str(fold.get("fixed_training_weight_arm")) != params["weight_arm"]:
        raise ValueError("Target contract hash mismatch: latest fold weight arm differs")
    target_contract: dict[str, Any] = {
        "target_mode": params["target_mode"],
        "target_column": "target_soft",
        "weight_arm": params["weight_arm"],
        "weight_column": "sample_weight",
        "payload_contract": "compact_fixed_training_payload_v1",
    }
    if params["target_mode"] == PROMOTED_TARGET_MODE:
        if params["weight_arm"] != PROMOTED_WEIGHT_ARM:
            raise ValueError("Target contract hash mismatch: promoted target requires promoted weight arm")
        provenance = promoted_side_target_provenance()
        target_contract.update(
            {
                "base_target_contract": provenance["base_target_contract"],
                "base_target_contract_hash": _normalise_hash(
                    provenance["base_target_contract_hash"]
                ),
                "base_sample_weight_spec": provenance["base_sample_weight_spec"],
                "base_sample_weight_spec_hash": _normalise_hash(
                    provenance["base_sample_weight_spec_hash"]
                ),
            }
        )
    target_hash = _sha256_json(target_contract)
    supplied_hash = (fold.get("target_contract_hash") or report.get("target_contract_hash"))
    if supplied_hash:
        _require_exact_hash(name="Target contract", actual=target_hash, expected=supplied_hash)
    ae_gmm = _validate_ae_gmm_contract(report=report, report_dir=report_dir)
    ae_gmm_output_features = _read_json(
        Path(ae_gmm["sidecar_manifest_path"])
    ).get("output_features", [])
    selected_ae_gmm = {
        feature
        for features in selected_by_side.values()
        for feature in features
        if feature.startswith("dae_") or feature.startswith("gmm_")
    }
    missing_ae_gmm = sorted(selected_ae_gmm.difference(map(str, ae_gmm_output_features)))
    if missing_ae_gmm:
        raise ValueError(
            "Feature contract hash mismatch: selected AE/GMM features are absent "
            f"from frozen output sidecar: {missing_ae_gmm}"
        )
    if recovery is None and bool(report.get("compact_fixed_training_payload")):
        labels_path = _resolve_path(report.get("labels_path", ""), report_dir=report_dir)
        if not labels_path.exists():
            raise ValueError(
                "Compact fixed payload omits train timestamps and the report has no "
                "recoverable immutable labels_path"
            )
        recovery = {
            "schema": "report_manifest_compact_identity_recovery_v1",
            "labels_path": str(labels_path),
            "labels_identity_columns": [
                "__ts__",
                "__symbol__",
                "side_name",
                "__first_touch_target_soft__",
            ],
            "source_report_manifest_sha256": _sha256_file(report_dir / "manifest.json"),
            "latest_fold": str(fold.get("fold")),
        }
    if recovery is not None:
        recovery = dict(recovery)
        recovery.setdefault(
            "sidecar_path",
            str(_resolve_path(report.get("frozen_ae_gmm_output_sidecar_path", ""), report_dir=report_dir)),
        )
        recovery.setdefault(
            "sidecar_required_features", list(ae_gmm["sidecar_output_features"])
        )
    return report, fold_dir, fold, selected_by_side, params, {
        "target_contract": target_contract,
        "target_contract_hash": target_hash,
        "ae_gmm": ae_gmm,
        "compact_recovery": recovery,
    }


def _payload_path(fold_dir: Path, fold: Mapping[str, Any], name: str) -> Path:
    paths = fold.get("payload_paths")
    if not isinstance(paths, Mapping) or name not in paths:
        raise ValueError(f"Latest fold cache is missing payload path: {name}")
    path = _resolve_path(paths[name], report_dir=fold_dir)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _identity_hash(frame: pd.DataFrame) -> str:
    """Hash row identities/targets without widening compact feature payloads."""

    hashed = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)
    return "sha256:" + hashlib.sha256(hashed.tobytes()).hexdigest()


def _label_files(labels_path: Path) -> list[Path]:
    files = sorted(labels_path.glob("*.parquet")) if labels_path.is_dir() else [labels_path]
    if not files or not all(path.is_file() for path in files):
        raise FileNotFoundError(f"No parquet label files found under {labels_path}")
    return files


def _load_label_identity(
    *,
    labels_path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    sidecar_support_keys: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load only stable label keys needed to prove compact-cache row alignment."""

    columns = ["__ts__", "__symbol__", "side_name", "__first_touch_target_soft__"]
    frames: list[pd.DataFrame] = []
    for path in _label_files(labels_path):
        frame = pd.read_parquet(path, columns=columns)
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
        frame = frame.loc[frame["__ts__"].ge(start) & frame["__ts__"].lt(end)]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=columns)
    # This is the same stable ordering used by the compact-payload builder.
    identity = (
        pd.concat(frames, ignore_index=True, copy=False)
        .sort_values(["__ts__", "__symbol__"], kind="mergesort")
        .reset_index(drop=True)
    )
    if sidecar_support_keys is not None:
        identity["side"] = np.where(
            identity["side_name"].astype(str).str.lower().eq("short"), -1, 1
        ).astype(np.int8)
        identity["__symbol__"] = identity["__symbol__"].astype(str)
        identity = identity.merge(
            sidecar_support_keys,
            on=["__ts__", "__symbol__", "side"],
            how="inner",
            sort=False,
            validate="many_to_one",
        ).drop(columns="side")
    return identity.reset_index(drop=True)


def _sidecar_supported_keys(
    *,
    sidecar_path: Path,
    required_features: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = list(dict.fromkeys(str(feature) for feature in required_features if str(feature)))
    if not required:
        return pd.DataFrame(columns=["__ts__", "__symbol__", "side"]), {
            "sidecar_support_filter": "not_required",
            "required_feature_count": 0,
        }
    columns = ["__ts__", "__symbol__", "side", *required]
    parquet = pq.ParquetFile(sidecar_path)
    missing = sorted(set(columns).difference(parquet.schema.names))
    if missing:
        raise ValueError(
            "Compact recovery sidecar is missing frozen output columns: "
            f"{missing[:20]}"
        )
    keys = ["__ts__", "__symbol__", "side"]
    parts: list[pd.DataFrame] = []
    scanned_rows = 0
    for batch in parquet.iter_batches(batch_size=250_000, columns=columns):
        sidecar = batch.to_pandas()
        scanned_rows += int(len(sidecar))
        sidecar["__ts__"] = pd.to_datetime(sidecar["__ts__"], utc=True, errors="coerce")
        window = sidecar["__ts__"].ge(start) & sidecar["__ts__"].lt(end)
        if not window.any():
            continue
        sidecar = sidecar.loc[window]
        sidecar["__symbol__"] = sidecar["__symbol__"].astype(str)
        sidecar["side"] = pd.to_numeric(sidecar["side"], errors="coerce").astype("Int8")
        if bool(sidecar[keys].isna().any(axis=None)):
            raise ValueError("Compact recovery sidecar has invalid identity keys")
        supported_mask = sidecar[required].notna().all(axis=1)
        if supported_mask.any():
            parts.append(sidecar.loc[supported_mask, keys].copy())
    supported = (
        pd.concat(parts, ignore_index=True, copy=False)
        if parts
        else pd.DataFrame(columns=keys)
    )
    if bool(supported.duplicated(keys, keep=False).any()):
        raise ValueError("Compact recovery sidecar is not unique by timestamp/symbol/side")
    return supported, {
        "sidecar_support_filter": "frozen_selected_outputs_all_nonnull",
        "required_feature_count": int(len(required)),
        "required_features": required,
        "sidecar_rows": int(parquet.metadata.num_rows),
        "sidecar_rows_scanned": int(scanned_rows),
        "supported_rows": int(len(supported)),
    }


def _validate_compact_identity(
    *,
    identity: pd.DataFrame,
    target: pd.Series,
    sides: pd.Series | np.ndarray,
    source: str,
) -> tuple[pd.Series, dict[str, Any]]:
    if len(identity) != len(target) or len(identity) != len(sides):
        raise ValueError(
            f"Compact recovery identity mismatch for {source}: "
            f"labels={len(identity)} target={len(target)} sides={len(sides)}"
        )
    label_target = pd.to_numeric(identity["__first_touch_target_soft__"], errors="coerce")
    cached_target = pd.to_numeric(target, errors="coerce")
    if not np.allclose(
        label_target.to_numpy(dtype=np.float64),
        cached_target.to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-6,
        equal_nan=True,
    ):
        raise ValueError(f"Compact recovery target alignment mismatch for {source}")
    label_sides = _normalise_side(identity["side_name"])
    cached_sides = _normalise_side(sides)
    if not np.array_equal(label_sides, cached_sides):
        raise ValueError(f"Compact recovery side alignment mismatch for {source}")
    identity_for_hash = identity.loc[:, ["__ts__", "__symbol__", "side_name"]].copy()
    identity_for_hash["target_soft"] = label_target.to_numpy(dtype=np.float32)
    return pd.to_datetime(identity["__ts__"], utc=True, errors="coerce"), {
        "rows": int(len(identity)),
        "identity_target_hash": _identity_hash(identity_for_hash),
        "timestamp_min": identity["__ts__"].min().isoformat() if len(identity) else None,
        "timestamp_max": identity["__ts__"].max().isoformat() if len(identity) else None,
    }


def _recover_compact_gap_rows(
    *,
    cache_dir: Path,
    latest_fold_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    expected_columns: Sequence[str],
    target_mode: str,
    weight_arm: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.DataFrame, dict[str, Any]]:
    """Restore the latest fold's purged tail from prior OOS validation payloads."""

    from scripts.run_first_touch_label_training_smoke import _target_from_frame
    from scripts.run_label_weighted_proxy_ablation import _weight_series

    parts: list[tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.DataFrame]] = []
    source_folds: list[str] = []
    for manifest_path in sorted(cache_dir.glob("*/fold_manifest.json")):
        fold_dir = manifest_path.parent
        if fold_dir == latest_fold_dir:
            continue
        fold = _read_json(manifest_path)
        paths = fold.get("payload_paths")
        if not isinstance(paths, Mapping) or not {"x_valid", "valid", "valid_metrics"}.issubset(paths):
            continue
        valid = pd.read_parquet(_payload_path(fold_dir, fold, "valid"))
        timestamps = pd.to_datetime(valid.get("__ts__"), utc=True, errors="coerce")
        take = timestamps.ge(start) & timestamps.lt(end)
        if not take.any():
            continue
        x_valid_path = _payload_path(fold_dir, fold, "x_valid")
        columns = list(pq.ParquetFile(x_valid_path).schema.names)
        if columns != list(expected_columns):
            raise ValueError(
                "Feature contract hash mismatch: compact gap cache has a different "
                "selected feature schema"
            )
        x_valid = pd.read_parquet(x_valid_path).astype(np.float32, copy=False)
        metrics = pd.read_parquet(_payload_path(fold_dir, fold, "valid_metrics"))
        positions = np.flatnonzero(take.to_numpy(dtype=bool, copy=False))
        source = valid.iloc[positions].reset_index(drop=True)
        source_metrics = metrics.iloc[positions].reset_index(drop=True)
        source_x = x_valid.iloc[positions].reset_index(drop=True)
        source_ts = timestamps.iloc[positions].reset_index(drop=True)
        source_side = source.get("side_name")
        if source_side is None:
            raise ValueError("Compact gap cache is missing valid side labels")
        source_target = _target_from_frame(
            source, source_metrics, target_mode=target_mode
        )["target_soft"]
        source_weight = _weight_series(
            frame=source,
            metrics=source_metrics,
            target=source_target.to_frame("target_soft"),
            arm=weight_arm,
        )
        parts.append((source_x, source_target, source_weight, source_side, source_ts, source))
        source_folds.append(str(fold.get("fold")))
    if not parts:
        raise ValueError(
            "Compact recovery could not restore the purged train-to-valid gap "
            "from prior validation payloads"
        )
    x = pd.concat([part[0] for part in parts], ignore_index=True, copy=False)
    target = pd.concat([part[1] for part in parts], ignore_index=True)
    weight = pd.concat([part[2] for part in parts], ignore_index=True)
    sides = pd.concat([part[3] for part in parts], ignore_index=True)
    timestamps = pd.concat([part[4] for part in parts], ignore_index=True)
    source = pd.concat([part[5] for part in parts], ignore_index=True)
    order = np.lexsort((source["__symbol__"].astype(str).to_numpy(), timestamps.to_numpy()))
    x = x.iloc[order].reset_index(drop=True)
    target = target.iloc[order].reset_index(drop=True)
    weight = weight.iloc[order].reset_index(drop=True)
    sides = sides.iloc[order].reset_index(drop=True)
    timestamps = timestamps.iloc[order].reset_index(drop=True)
    source = source.iloc[order].reset_index(drop=True)
    if bool(source.duplicated(["__ts__", "__symbol__", "side_name"], keep=False).any()):
        raise ValueError("Compact recovery gap payloads contain duplicate row identities")
    return x, target, weight, sides, timestamps, source, {
        "source_folds": source_folds,
        "recovered_rows": int(len(x)),
        "start": start.isoformat(),
        "end": end.isoformat(),
    }


def _load_final_rows(
    fold_dir: Path,
    fold: Mapping[str, Any],
    *,
    compact_recovery: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, np.ndarray, dict[str, Any]]:
    required = (
        "x_train",
        "x_valid",
        "train_target",
        "train_weight",
        "train_side",
        "valid",
        "valid_metrics",
    )
    if "train" in dict(fold.get("payload_paths") or {}):
        required = (*required, "train")
    payload = {name: pd.read_parquet(_payload_path(fold_dir, fold, name)) for name in required}
    x_train = payload["x_train"].astype(np.float32, copy=False)
    x_valid = payload["x_valid"].astype(np.float32, copy=False)
    if list(x_train.columns) != list(x_valid.columns):
        raise ValueError("Feature contract hash mismatch: train/valid cache columns differ")
    train_target = pd.to_numeric(payload["train_target"].get("target_soft"), errors="coerce")
    train_weight = pd.to_numeric(payload["train_weight"].get("sample_weight"), errors="coerce")
    train_side = payload["train_side"].get("side_name")
    if train_side is None:
        raise ValueError("Latest fold cache is missing train side labels")
    from scripts.run_first_touch_label_training_smoke import _target_from_frame
    from scripts.run_label_weighted_proxy_ablation import _weight_series

    valid_target = _target_from_frame(
        payload["valid"], payload["valid_metrics"], target_mode=str(fold["fixed_training_target_mode"])
    )["target_soft"]
    valid_weight = _weight_series(
        frame=payload["valid"], metrics=payload["valid_metrics"], target=valid_target.to_frame("target_soft"), arm=str(fold["fixed_training_weight_arm"])
    )
    valid_side = payload["valid"].get("side_name")
    if valid_side is None:
        raise ValueError("Latest fold cache is missing valid side labels")
    valid_ts = pd.to_datetime(payload["valid"].get("__ts__"), utc=True, errors="coerce")
    if valid_ts is None:
        raise ValueError("Latest fold cache is missing valid timestamps")
    recovery_audit: dict[str, Any] | None = None
    gap_x = pd.DataFrame(columns=x_train.columns)
    gap_target = pd.Series(dtype=np.float64)
    gap_weight = pd.Series(dtype=np.float64)
    gap_side = pd.Series(dtype=object)
    gap_ts = pd.Series(dtype="datetime64[ns, UTC]")
    if "train" in payload:
        train_ts = pd.to_datetime(payload["train"].get("__ts__"), utc=True, errors="coerce")
        if train_ts is None:
            raise ValueError("Latest fold cache is missing train timestamps")
    else:
        if compact_recovery is None:
            raise ValueError(
                "Latest compact fold cache omits train timestamps; provide compact recovery source paths"
            )
        train_start = pd.to_datetime(fold.get("train_start"), utc=True, errors="coerce")
        valid_start = pd.to_datetime(fold.get("valid_start"), utc=True, errors="coerce")
        if pd.isna(train_start) or pd.isna(valid_start):
            raise ValueError(
                "Compact recovery requires finite train_start and valid_start in the fold manifest"
            )
        labels_path = Path(str(compact_recovery["labels_path"]))
        sidecar_required_features = list(
            compact_recovery.get("sidecar_required_features", []) or []
        )
        valid_end = valid_ts.max() + pd.Timedelta(nanoseconds=1)
        support_keys, support_audit = _sidecar_supported_keys(
            sidecar_path=Path(str(compact_recovery["sidecar_path"])),
            required_features=sidecar_required_features,
            start=train_start,
            end=valid_end,
        )
        train_identity_all = _load_label_identity(
            labels_path=labels_path,
            start=train_start,
            end=valid_start,
            sidecar_support_keys=support_keys if sidecar_required_features else None,
        )
        if len(train_identity_all) < len(train_target):
            raise ValueError(
                "Compact recovery has fewer immutable label identities than cached train rows"
            )
        train_identity = train_identity_all.iloc[: len(train_target)].reset_index(drop=True)
        gap_identity = train_identity_all.iloc[len(train_target) :].reset_index(drop=True)
        valid_identity = _load_label_identity(
            labels_path=labels_path,
            start=valid_start,
            end=valid_end,
            sidecar_support_keys=support_keys if sidecar_required_features else None,
        )
        train_ts, train_audit = _validate_compact_identity(
            identity=train_identity,
            target=train_target,
            sides=train_side,
            source="train",
        )
        if not gap_identity.empty:
            gap_start = pd.to_datetime(gap_identity["__ts__"].min(), utc=True)
            gap_x, gap_target, gap_weight, gap_side, gap_ts, gap_source, gap_audit = (
                _recover_compact_gap_rows(
                    cache_dir=fold_dir.parent,
                    latest_fold_dir=fold_dir,
                    start=gap_start,
                    end=valid_start,
                    expected_columns=x_train.columns,
                    target_mode=str(fold["fixed_training_target_mode"]),
                    weight_arm=str(fold["fixed_training_weight_arm"]),
                )
            )
            _, gap_identity_audit = _validate_compact_identity(
                identity=gap_identity,
                target=gap_target,
                sides=gap_side,
                source="purged_gap",
            )
            if not gap_identity["__ts__"].reset_index(drop=True).equals(
                gap_ts.reset_index(drop=True)
            ):
                raise ValueError("Compact recovery timestamp alignment mismatch for purged_gap")
            gap_audit["identity"] = gap_identity_audit
        else:
            gap_audit = {
                "source_folds": [],
                "recovered_rows": 0,
                "start": valid_start.isoformat(),
                "end": valid_start.isoformat(),
            }
        _, valid_audit = _validate_compact_identity(
            identity=valid_identity,
            target=valid_target,
            sides=valid_side,
            source="valid",
        )
        if not valid_identity["__ts__"].reset_index(drop=True).equals(
            valid_ts.reset_index(drop=True)
        ):
            raise ValueError("Compact recovery timestamp alignment mismatch for valid")
        recovery_audit = {
            "schema": "compact_fixed_base_row_identity_validation_v1",
            "labels_path": str(labels_path),
            "sidecar_support": support_audit,
            "train": train_audit,
            "purged_gap": gap_audit,
            "valid": valid_audit,
        }
    x_parts = [x_train, x_valid] if gap_x.empty else [x_train, gap_x, x_valid]
    target_parts = [train_target, valid_target] if gap_target.empty else [train_target, gap_target, valid_target]
    weight_parts = [train_weight, valid_weight] if gap_weight.empty else [train_weight, gap_weight, valid_weight]
    x = pd.concat(x_parts, ignore_index=True, copy=False)
    target = pd.concat(target_parts, ignore_index=True)
    weight = pd.concat(weight_parts, ignore_index=True)
    sides = np.concatenate(
        [
            train_side.astype(str).to_numpy(),
            gap_side.astype(str).to_numpy(),
            valid_side.astype(str).to_numpy(),
        ]
    )
    timestamps = pd.concat([train_ts, gap_ts, valid_ts], ignore_index=True)
    if not (len(x) == len(target) == len(weight) == len(sides) == len(timestamps)):
        raise ValueError("Latest fold cache row alignment mismatch")
    valid = np.isfinite(target.to_numpy(dtype=np.float64)) & np.isfinite(weight.to_numpy(dtype=np.float64))
    valid &= weight.to_numpy(dtype=np.float64) > 0.0
    valid &= timestamps.notna().to_numpy()
    final_end = timestamps.loc[valid].max()
    if pd.isna(final_end):
        raise ValueError("Latest fold cache has no finite resolved timestamp")
    final_start = final_end - pd.Timedelta(days=365)
    trailing_window = timestamps.ge(final_start) & timestamps.le(final_end)
    valid &= trailing_window.to_numpy()
    normalized_sides = _normalise_side(sides)
    accounting = {
        "train_rows": int(len(x_train)),
        "recovered_purged_gap_rows": int(len(gap_x)),
        "latest_labelled_valid_rows": int(len(x_valid)),
        "candidate_rows": int(len(x)),
        "final_refit_start": final_start.isoformat(),
        "final_refit_end": final_end.isoformat(),
        "train_window_days": 365,
        "excluded_outside_trailing_window_rows": int((~trailing_window).sum()),
        "excluded_nonfinite_target_weight_or_timestamp_rows": int(
            (~(
                np.isfinite(target.to_numpy(dtype=np.float64))
                & np.isfinite(weight.to_numpy(dtype=np.float64))
                & (weight.to_numpy(dtype=np.float64) > 0.0)
                & timestamps.notna().to_numpy()
            )).sum()
        ),
        "permitted_rows": int(valid.sum()),
    }
    if recovery_audit is not None:
        accounting["compact_recovery_identity_validation"] = recovery_audit
    if not valid.any():
        raise ValueError("Latest fold cache has no permitted labelled rows")
    return x.loc[valid].reset_index(drop=True), target.loc[valid].reset_index(drop=True), weight.loc[valid].reset_index(drop=True), normalized_sides[valid], accounting


def _fit_side_model(*, matrix: pd.DataFrame, target: pd.Series, weight: pd.Series, params: Mapping[str, Any], seed: int) -> Any:
    try:
        from lightgbm import LGBMRegressor
    except Exception as exc:  # pragma: no cover - dependency is present in production.
        raise RuntimeError("lightgbm is required for final refit") from exc
    model = LGBMRegressor(
        objective=str(params["loss_function"]), n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]), num_leaves=int(params["num_leaves"]),
        max_depth=int(params["max_depth"]), min_child_samples=int(params["min_child_samples"]),
        subsample=float(params["subsample"]), colsample_bytree=float(params["colsample_bytree"]),
        reg_alpha=float(params["reg_alpha"]), reg_lambda=float(params["reg_lambda"]),
        min_split_gain=float(params["min_split_gain"]), random_state=int(seed), n_jobs=2, verbosity=-1,
    )
    model.fit(matrix, target.to_numpy(dtype=np.float32), sample_weight=weight.to_numpy(dtype=np.float32))
    return model


def _copy_ae_gmm_bundle(*, staging: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    source = Path(contract["state_path"])
    state_dir = staging / "ae_gmm_state"
    state_dir.mkdir()
    copied_state = state_dir / "ae_gmm_state.pkl"
    copied_manifest = state_dir / "input_transform_manifest.json"
    shutil.copy2(source, copied_state)
    shutil.copy2(contract["state_manifest_path"], copied_manifest)
    _require_exact_hash(name="Packaged AE/GMM state", actual=_sha256_file(copied_state), expected=contract["state_sha256"])
    bundle = {
        "schema": "frozen_ae_gmm_state_and_input_transform_v1",
        "state_path": str(copied_state.relative_to(staging)), "state_sha256": _sha256_file(copied_state),
        "input_transform_manifest_path": str(copied_manifest.relative_to(staging)),
        "input_transform_manifest_sha256": _sha256_file(copied_manifest),
        "input_features": list(contract["input_features"]),
        "input_feature_order_hash": str(contract["input_feature_order_hash"]),
        "source_state_path": str(source), "source_state_manifest_path": str(contract["state_manifest_path"]),
        "source_sidecar_manifest_sha256": str(contract["sidecar_manifest_sha256"]),
        "source_sidecar_output_feature_hash": str(contract["sidecar_output_feature_hash"]),
    }
    (state_dir / "manifest.json").write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle


def run_final_refit(
    *,
    report_dir: Path,
    output_dir: Path,
    seed: int = 20260721,
    compact_recovery: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit frozen long/short models from the report's latest labelled fold cache."""
    report_dir = Path(report_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite package: {output_dir}")
    report, fold_dir, fold, selected_by_side, params, contract = _validate_source_contract(
        report_dir=report_dir, compact_recovery=compact_recovery
    )
    x, target, weight, sides, accounting = _load_final_rows(
        fold_dir, fold, compact_recovery=contract.get("compact_recovery")
    )
    expected_union = sorted({feature for values in selected_by_side.values() for feature in values})
    _require_exact_hash(name="Latest-cache feature", actual=_feature_contract_hash(list(x.columns)), expected=_feature_contract_hash(expected_union))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}_", dir=output_dir.parent))
    staging = staging_parent / output_dir.name
    staging.mkdir()
    try:
        ae_gmm_bundle = _copy_ae_gmm_bundle(staging=staging, contract=contract["ae_gmm"])
        side_artifacts: dict[str, Any] = {}
        for side_index, side in enumerate(SIDES, start=1):
            mask = sides == side
            if not mask.any():
                raise ValueError(f"No permitted final-refit rows for side={side}")
            features = selected_by_side[side]
            missing = sorted(set(features).difference(x.columns))
            if missing:
                raise ValueError(f"Feature contract hash mismatch: {side} cache is missing {missing}")
            matrix = x.loc[mask, features].reset_index(drop=True).astype(np.float32, copy=False)
            model = _fit_side_model(matrix=matrix, target=target.loc[mask].reset_index(drop=True), weight=weight.loc[mask].reset_index(drop=True), params=params, seed=int(seed) + side_index * 10_003)
            side_dir = staging / side
            side_dir.mkdir()
            model_path = side_dir / "base_model.joblib"
            features_path = side_dir / "features.json"
            metadata_path = side_dir / "metadata.json"
            joblib.dump(model, model_path, compress=3)
            feature_payload = {"schema": "side_specific_base_feature_contract_v1", "side": side, "feature_names": features, "feature_count": len(features), "feature_contract_hash": _feature_contract_hash(features)}
            features_path.write_text(json.dumps(feature_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            metadata = {
                "schema": SCHEMA, "side": side, "model_path": model_path.name, "features_path": features_path.name,
                "model_sha256": _sha256_file(model_path), "train_rows": int(mask.sum()), "params": params,
                "feature_contract_hash": feature_payload["feature_contract_hash"], "target_contract": contract["target_contract"],
                "target_contract_hash": contract["target_contract_hash"],
                "sample_weight_contract": (
                    "frozen compact train weights plus the same declared target/weight "
                    "arm for latest labelled rows"
                ),
                "leakage_contract": {"excluded_from_oos_metrics": True, "feature_selection_and_hpo": "frozen from completed base report", "fit_scope": "exact trailing 365 days through the latest resolved labelled row"},
            }
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            side_artifacts[side] = {"model": str(model_path.relative_to(staging)), "features": str(features_path.relative_to(staging)), "metadata": str(metadata_path.relative_to(staging)), "model_sha256": metadata["model_sha256"], "feature_contract_hash": feature_payload["feature_contract_hash"], "train_rows": int(mask.sum())}
        manifest = {
            "schema": SCHEMA, "generated_by": "refit_package_side_specific_base_champion", "status": "final_refit_non_oos_pending_promotion",
            "source_report": str(report_dir),
            "source_report_manifest_sha256": (
                _sha256_file(report_dir / "manifest.json")
                if (report_dir / "manifest.json").is_file()
                else None
            ),
            "source_latest_fold": str(fold.get("fold")), "source_latest_fold_manifest_sha256": _sha256_file(fold_dir / "fold_manifest.json"),
            "model_side_scope": "per_side", "train_window_days": 365, "excluded_from_oos_metrics": True, "all_permitted_labelled_rows_refit": True,
            "permitted_row_accounting": accounting, "selected_features_by_side": selected_by_side, "params": params,
            "stage_c_params_hash": _sha256_json(params), "target_contract": contract["target_contract"], "target_contract_hash": contract["target_contract_hash"], "ae_gmm_bundle": ae_gmm_bundle, "sides": side_artifacts,
            "leakage_contract": {"oos_claim": "none; final refit is explicitly excluded from OOS metrics", "feature_selection_hpo": "frozen", "ae_gmm": "exact serialized state and ordered input transform copied from validated source", "training_rows": "exact trailing 365 days through the latest resolved labelled row"},
        }
        if contract.get("compact_recovery") is not None:
            manifest["compact_recovery"] = contract["compact_recovery"]
        (staging / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        staging.replace(output_dir)
        staging_parent.rmdir()
        return manifest
    except Exception:
        shutil.rmtree(staging_parent, ignore_errors=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--compact-recovery-labels-path",
        type=Path,
        help="Immutable labels used only to reconstruct and validate compact-cache row identities.",
    )
    parser.add_argument(
        "--compact-recovery-fixed-params-json",
        type=Path,
        help="Frozen fixed-parameter contract for a report whose manifest was not written.",
    )
    parser.add_argument(
        "--compact-recovery-ae-gmm-state-pkl",
        type=Path,
        help="Exact frozen AE/GMM state used by the interrupted compact run.",
    )
    parser.add_argument(
        "--compact-recovery-ae-gmm-output-sidecar",
        type=Path,
        help="Exact frozen AE/GMM output sidecar used by the interrupted compact run.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    recovery_values = {
        "labels_path": args.compact_recovery_labels_path,
        "fixed_params_json": args.compact_recovery_fixed_params_json,
        "fixed_ae_gmm_state_pkl": args.compact_recovery_ae_gmm_state_pkl,
        "frozen_ae_gmm_output_sidecar_path": args.compact_recovery_ae_gmm_output_sidecar,
    }
    supplied = [value is not None for value in recovery_values.values()]
    if any(supplied) and not all(supplied):
        raise SystemExit(
            "All --compact-recovery-* arguments must be supplied together"
        )
    compact_recovery = recovery_values if all(supplied) else None
    print(json.dumps(_json_safe(run_final_refit(
        report_dir=args.report_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        compact_recovery=compact_recovery,
    )), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
