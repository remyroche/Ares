#!/usr/bin/env python3
"""Fit causal, side-local LightGBM heads for 12-hour path timing and peak MFE.

The materialized labels supply only row identity, training archetype context, and
future-path targets.  Every model input is read from the immutable canonical
static feature store at the same UTC decision timestamp and symbol.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import numbers
import os
import sys
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
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
from extreme_price_movements.path_auxiliary_lgbm import (  # noqa: E402
    MODEL_SCHEMA,
    TARGET_COLUMNS,
    auxiliary_hpo_objective,
    build_auxiliary_sample_weights,
    configured_auxiliary_feature_universe,
    default_auxiliary_lgbm_n_jobs,
    fit_base_archetype_label_feature_contract,
    fit_side_aware_auxiliary_models,
    select_features_with_current_pipeline,
    transform_base_archetype_label_features,
)
from extreme_price_movements.path_auxiliary_targets import (  # noqa: E402
    ALL_SUPPORTIVE_LABEL_COLUMNS,
    FUTURE_SLOPE_ATR_PER_HOUR_CLIP,
    MAE_BEFORE_MEANINGFUL_MFE_ATR_CLIP,
    MIN_USABLE_MFE_ATR,
    MIN_USABLE_MFE_RETURN,
    PEAK_MFE_ATR_CLIP,
    TARGET_SCHEMA,
)
from extreme_price_movements.side_aware import candidate_id_series  # noqa: E402
from extreme_price_movements.training_resource_guard import (  # noqa: E402
    GIB,
    TrainingResourceGuard,
    TrainingResourceLimits,
)

RUNNER_SCHEMA = "run_path_auxiliary_lgbm_models_v5_resumable_expanding_oos"
SELECTION_HPO_REUSE_SCHEMA = "path_auxiliary_selection_hpo_reuse_v1"
CHECKPOINT_SCHEMA = "path_auxiliary_lgbm_checkpoint_v1"
STATIC_FEATURE_READ_CACHE_DEFAULT_MAX_BYTES = 192 * 1024 * 1024
STATIC_FEATURE_READ_CACHE_HARD_MAX_BYTES = 512 * 1024 * 1024
STATIC_FEATURE_READ_CACHE_HARD_MAX_ENTRIES = 8
SELECTION_STATIC_READ_COALESCE_GAP = pd.Timedelta(hours=1)
SELECTION_STATIC_READ_MAX_BLOCK = pd.Timedelta(hours=24)
IDENTITY_COLUMNS = ("__ts__", "__symbol__", "side")
CANDIDATE_ID_COLUMN = "candidate_id"
STRICT_IDENTITY_COLUMNS = (*IDENTITY_COLUMNS, CANDIDATE_ID_COLUMN)
SELECTED_TOP40_COLUMN = "selected_top40"
DEFAULT_LABEL_RESOLUTION_COLUMN = "__label_end_ts__"
MANDATORY_HANDOFF_MODEL_FEATURES = (
    "score",
    "base_margin_to_cutoff",
    "base_margin_to_cutoff_z",
    "base_signal_zscore_within_archetype",
)
CONTEXT_COLUMNS = (
    "side_name",
    "archetype",
    "archetype_label_family",
    "archetype_policy_key",
    "policy_archetype",
    "local_side_archetype",
    "gmm_cluster_id",
)
ARCHETYPE_COLUMNS = (
    "archetype_label_family",
    "archetype_policy_key",
    "policy_archetype",
    "local_side_archetype",
    "archetype",
)
PREDICTION_ROLES = {
    "time_to_first_meaningful_mfe": "time_to_mfe_oof",
    "peak_mfe_12h_atr": "peak_mfe_oof",
    "mae_before_meaningful_mfe_atr": "mae_before_mfe_oof",
    "bars_before_price_stops_decreasing": "adverse_turn_oof",
    "future_slope_atr_per_hour": "path_slope_oof",
}
RESOURCE_TELEMETRY_FILENAME = "training_resource_telemetry.jsonl"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        try:
            json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            if temp_path.exists():
                temp_path.unlink()


def _gib_to_bytes(value: float, *, name: str) -> int:
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative GiB value")
    return int(value * GIB)


def _resource_disk_path(output_dir: Path) -> Path:
    """Use an existing ancestor on the output filesystem for disk telemetry."""

    path = Path(output_dir)
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def _build_resource_guard(
    *,
    output_dir: Path,
    min_free_ram_gib: float,
    max_process_rss_gib: float,
    min_free_disk_gib: float,
    check_interval_seconds: float,
    telemetry_path: Path | None,
) -> TrainingResourceGuard:
    if not np.isfinite(check_interval_seconds) or check_interval_seconds < 0:
        raise ValueError(
            "resource_check_interval_seconds must be finite and non-negative"
        )
    limits = TrainingResourceLimits(
        min_free_ram_bytes=_gib_to_bytes(
            min_free_ram_gib, name="resource_min_free_ram_gib"
        ),
        max_process_rss_bytes=_gib_to_bytes(
            max_process_rss_gib, name="resource_max_process_rss_gib"
        ),
        min_free_disk_bytes=_gib_to_bytes(
            min_free_disk_gib, name="resource_min_free_disk_gib"
        ),
        check_interval_seconds=float(check_interval_seconds),
    )
    return TrainingResourceGuard(
        limits=limits,
        disk_path=_resource_disk_path(output_dir),
        telemetry_path=telemetry_path or output_dir / RESOURCE_TELEMETRY_FILENAME,
    )


def _resource_guard_contract(guard: TrainingResourceGuard) -> dict[str, Any]:
    return {
        "limits": {
            "min_free_ram_bytes": guard.limits.min_free_ram_bytes,
            "max_process_rss_bytes": guard.limits.max_process_rss_bytes,
            "min_free_disk_bytes": guard.limits.min_free_disk_bytes,
            "check_interval_seconds": guard.limits.check_interval_seconds,
        },
        "disk_path": str(guard.disk_path),
        "telemetry_path": (
            str(guard.telemetry_path) if guard.telemetry_path is not None else None
        ),
        "contract": "fail_closed_preflight_and_boundary_checkpoints_v1",
    }


def _atomic_joblib_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temp_path = Path(handle.name)
    try:
        joblib.dump(payload, temp_path)
        with temp_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _atomic_to_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temp_path, index=False)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _progress(
    stage: str, *, head: str | None = None, side: str | None = None, **detail: Any
) -> None:
    """Emit machine-readable, UTC progress without mixing it into artifact JSON."""

    payload: dict[str, Any] = {
        "ts_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "stage": stage,
    }
    if head is not None:
        payload["head"] = head
    if side is not None:
        payload["side"] = side
    payload.update(_json_safe(detail))
    print(json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)


def _stable_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "checkpoint.json"


def _checkpoint_artifact_path(
    output_dir: Path, head: str, side: str, name: str
) -> Path:
    return output_dir / ".checkpoints" / head / side / name


def _artifact_record(path: Path, *, stage: str, fingerprint: str) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "stage": stage,
        "fingerprint_sha256": fingerprint,
    }


def _load_checkpoint_artifact(
    record: Mapping[str, Any], *, stage: str, fingerprint: str
) -> Any:
    if record.get("stage") != stage or record.get("fingerprint_sha256") != fingerprint:
        raise ValueError(f"checkpoint {stage} fingerprint mismatch")
    path = Path(str(record.get("path", "")))
    if not path.is_file() or _file_sha256(path) != record.get("sha256"):
        raise ValueError(f"checkpoint {stage} artifact is missing or corrupt")
    payload = joblib.load(path)
    if (
        not isinstance(payload, Mapping)
        or payload.get("fingerprint_sha256") != fingerprint
    ):
        raise ValueError(f"checkpoint {stage} payload fingerprint mismatch")
    return payload.get("payload")


def _load_or_initialize_checkpoint(
    output_dir: Path,
    *,
    fingerprint: Mapping[str, Any],
    overwrite: bool,
    allowed_preflight_paths: Sequence[Path] = (),
) -> tuple[dict[str, Any], bool]:
    """Return a complete exact run or a validated in-progress checkpoint.

    A nonempty directory without this ledger is intentionally rejected: it has no
    provenance that can establish which portions are safe to reuse.
    """

    manifest_path = output_dir / "manifest.json"
    checkpoint_path = _checkpoint_path(output_dir)
    allowed = {Path(path).resolve() for path in allowed_preflight_paths}
    existing_entries = (
        [path for path in output_dir.iterdir() if path.resolve() not in allowed]
        if output_dir.exists()
        else []
    )
    if existing_entries:
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"unreadable existing auxiliary manifest: {manifest_path}"
                ) from exc
            if (
                manifest.get("run_fingerprint", {}).get("sha256")
                == fingerprint["sha256"]
            ):
                if not checkpoint_path.is_file():
                    raise ValueError(
                        "completed auxiliary manifest is missing its checkpoint ledger"
                    )
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                for target_name in TARGET_COLUMNS:
                    record = (
                        checkpoint.get("heads", {}).get(target_name, {}).get("complete")
                    )
                    if record is None:
                        raise ValueError(
                            f"completed auxiliary manifest has no complete checkpoint for {target_name}"
                        )
                    payload = _load_checkpoint_artifact(
                        record, stage="head_complete", fingerprint=fingerprint["sha256"]
                    )
                    paths = (
                        payload.get("paths", {}) if isinstance(payload, Mapping) else {}
                    )
                    required_paths = [
                        paths.get("oof_predictions"),
                        paths.get("oof_prediction_manifest"),
                    ]
                    required_paths.extend((paths.get("bundles") or {}).values())
                    if not all(
                        value and Path(str(value)).is_file() for value in required_paths
                    ):
                        raise ValueError(
                            f"completed auxiliary head {target_name} has missing artifacts"
                        )
                    oof_path = Path(str(paths["oof_predictions"]))
                    oof_manifest = json.loads(
                        Path(str(paths["oof_prediction_manifest"])).read_text(
                            encoding="utf-8"
                        )
                    )
                    if (
                        oof_manifest.get("source_artifact_sha256")
                        != _file_sha256(oof_path)
                        or oof_manifest.get("prediction_role")
                        != PREDICTION_ROLES[target_name]
                    ):
                        raise ValueError(
                            f"completed auxiliary head {target_name} has mismatched OOF artifacts"
                        )
                    for side, bundle_path in (paths.get("bundles") or {}).items():
                        bundle = joblib.load(bundle_path)
                        if (
                            bundle.get("target_name") != target_name
                            or bundle.get("side") != side
                        ):
                            raise ValueError(
                                f"completed auxiliary head {target_name} has mismatched {side} bundle"
                            )
                return manifest, True
            if not overwrite:
                raise ValueError(
                    "existing auxiliary manifest has a mismatched run fingerprint"
                )
        if checkpoint_path.is_file():
            try:
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"unreadable auxiliary checkpoint: {checkpoint_path}"
                ) from exc
            if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
                raise ValueError("legacy or invalid auxiliary checkpoint schema")
            if (
                checkpoint.get("run_fingerprint", {}).get("sha256")
                != fingerprint["sha256"]
            ):
                raise ValueError(
                    "existing auxiliary checkpoint has a mismatched run fingerprint"
                )
            return checkpoint, False
        if not overwrite:
            raise FileExistsError(
                f"Output directory is non-empty and has no resumable checkpoint: {output_dir}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "run_fingerprint": dict(fingerprint),
        "heads": {},
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    _write_json(checkpoint_path, checkpoint)
    return checkpoint, False


def _save_checkpoint(output_dir: Path, checkpoint: Mapping[str, Any]) -> None:
    payload = dict(checkpoint)
    payload["updated_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    _write_json(_checkpoint_path(output_dir), payload)


def _require_explicit_utc_timestamp(
    value: str | pd.Timestamp | None, *, name: str
) -> pd.Timestamp:
    """Reject implicit/local cutoffs; this audit boundary must be explicit UTC."""

    if value is None:
        raise ValueError(f"{name} must be declared explicitly")
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{name} must be an explicit timezone-aware UTC timestamp")
    return timestamp.tz_convert("UTC")


def _candidate_ids(values: pd.Series, *, source: str) -> pd.Series:
    """Preserve the canonical candidate stream identity without synthesizing IDs."""

    result = values.astype("string").str.strip()
    if result.isna().any() or result.eq("").any():
        raise ValueError(f"{source} has null or blank {CANDIDATE_ID_COLUMN} values")
    return result.astype(str)


def _validated_selected_top40(values: pd.Series, *, source: str) -> pd.Series:
    """Accept only explicit boolean encodings and reject nullable/truthy values."""

    def parse(value: Any) -> bool:
        if value is None or pd.isna(value):
            raise ValueError(f"{source} has null {SELECTED_TOP40_COLUMN} values")
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, numbers.Integral) and int(value) in (0, 1):
            return bool(int(value))
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized == "true":
                return True
            if normalized == "false":
                return False
        raise ValueError(
            f"{source} has non-boolean {SELECTED_TOP40_COLUMN} value {value!r}; "
            "accepted encodings are bool, 0/1 integer, or true/false text"
        )

    if values.empty:
        raise ValueError(f"{source} has no {SELECTED_TOP40_COLUMN} rows")
    return values.map(parse).astype(bool)


def _timestamp_bounds(values: pd.Series | Sequence[Any]) -> dict[str, Any]:
    timestamp = pd.to_datetime(pd.Series(values), utc=True, errors="coerce")
    valid = timestamp.dropna()
    return {
        "rows": int(len(timestamp)),
        "valid_rows": int(len(valid)),
        "min_utc": valid.min().isoformat() if not valid.empty else None,
        "max_utc": valid.max().isoformat() if not valid.empty else None,
    }


def _canonical_label_files(labels_path: Path) -> list[Path]:
    """Use month partitions when present, matching the materialized-label contract."""

    if labels_path.is_file():
        files = [labels_path]
    else:
        all_files = sorted(labels_path.glob("*.parquet"))
        monthly = [
            file
            for file in all_files
            if file.name.startswith("train_global_") and file.suffix == ".parquet"
        ]
        files = monthly or all_files
    if not files:
        raise FileNotFoundError(f"No parquet label files found under {labels_path}")
    return files


def _parquet_columns(path: Path) -> set[str]:
    try:
        import pyarrow.parquet as pq

        return set(map(str, pq.read_schema(path).names))
    except Exception:
        return set(map(str, pd.read_parquet(path).columns))


def _label_source_signature(files: Sequence[Path]) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_rows: list[dict[str, Any]] = []
    for path in files:
        stat = path.stat()
        item = {
            "name": path.name,
            "bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
        digest.update(json.dumps(item, sort_keys=True).encode("utf-8"))
        file_rows.append(item)
    return {"files": file_rows, "signature_sha256": digest.hexdigest()}


def _tree_stat_signature(path: Path) -> dict[str, Any]:
    """Cheap immutable-input guard for the static feature-store tree.

    The full feature matrix is intentionally not re-hashed in memory.  Paths,
    byte sizes, and nanosecond mtimes still ensure a replacement or mutation
    invalidates checkpoint reuse before model code is entered.
    """

    if not path.exists():
        return {"exists": False, "signature_sha256": _stable_sha256({"exists": False})}
    digest = hashlib.sha256()
    rows = 0
    for item in sorted(
        (value for value in path.rglob("*") if value.is_file()),
        key=lambda value: str(value),
    ):
        stat = item.stat()
        payload = {
            "relative_path": str(item.relative_to(path)),
            "bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
        digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
        rows += 1
    return {"exists": True, "files": rows, "signature_sha256": digest.hexdigest()}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_hpo_fingerprint(
    *,
    selected_population_identity_sha256: str,
    selection_hpo_reference_contract: Mapping[str, Any],
    archetype_feature_contract: Mapping[str, Any],
    feature_dir: Path,
    static_columns: Sequence[str],
    handoff_feature_columns: Sequence[str],
    configured_universe: Mapping[str, Any],
    selection_rows: int,
    hpo_rows: int,
    n_trials: int,
    seed: int,
    purge_hours: float,
) -> dict[str, Any]:
    """Fingerprint every input that may alter selected features or HPO parameters."""

    payload = {
        "schema": SELECTION_HPO_REUSE_SCHEMA,
        "runner_schema": RUNNER_SCHEMA,
        "model_schema": MODEL_SCHEMA,
        "target_schema": TARGET_SCHEMA,
        "target_columns": TARGET_COLUMNS,
        "supportive_label_columns": list(ALL_SUPPORTIVE_LABEL_COLUMNS),
        "supportive_weight_contract": "head_specific_supportive_labels_clipped_0.5_2.0_v1",
        "selected_population_identity_sha256": selected_population_identity_sha256,
        "selection_hpo_reference_contract_sha256": selection_hpo_reference_contract[
            "contract_sha256"
        ],
        "selection_hpo_reference_end": selection_hpo_reference_contract[
            "selection_hpo_reference_end"
        ],
        "base_archetype_label_feature_contract_sha256": _stable_sha256(
            dict(archetype_feature_contract)
        ),
        "code_contract": {
            "reuse_schema": SELECTION_HPO_REUSE_SCHEMA,
            "runner_source_sha256": _file_sha256(Path(__file__).resolve()),
            "model_source_sha256": _file_sha256(
                ROOT / "extreme_price_movements" / "path_auxiliary_lgbm.py"
            ),
        },
        "feature_store": {
            "feature_dir": str(feature_dir.resolve()),
            "available_static_columns": sorted(map(str, static_columns)),
            "handoff_feature_columns": sorted(map(str, handoff_feature_columns)),
            "configured_universe": dict(configured_universe),
        },
        "selection_hpo_settings": {
            "selection_rows": int(selection_rows),
            "hpo_rows_per_side": int(hpo_rows),
            "hpo_trials": int(n_trials),
            "seed": int(seed),
            "purge_hours": float(purge_hours),
            "selection_contract": (
                "strict_side_local_full_pipeline_univariate_relief_mda_unweighted_v1"
            ),
            "hpo_validation_contract": "unweighted_purged_reference_folds_v1",
        },
    }
    return {"payload": payload, "sha256": _stable_sha256(payload)}


def _read_reusable_selection_hpo(
    output_dir: Path,
    *,
    fingerprint: Mapping[str, Any],
    force_selection_hpo: bool,
) -> tuple[
    dict[str, dict[str, Any]] | None, dict[str, dict[str, Any]] | None, dict[str, Any]
]:
    """Reuse only complete current-schema sibling artifacts with an exact fingerprint."""

    audit: dict[str, Any] = {
        "schema": SELECTION_HPO_REUSE_SCHEMA,
        "search_scope": "direct_sibling_run_directories",
        "fingerprint_sha256": fingerprint["sha256"],
        "force_selection_hpo": bool(force_selection_hpo),
        "auto_reused": False,
        "candidates_checked": 0,
        "rejection_counts": {},
    }
    if force_selection_hpo:
        audit["reason"] = "force_selection_hpo_requested"
        return None, None, audit

    parent = output_dir.parent
    manifests = sorted(
        path for path in parent.glob("*/manifest.json") if path.parent != output_dir
    )
    for manifest_path in manifests:
        audit["candidates_checked"] += 1
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            reason = "unreadable_manifest"
        else:
            if (
                manifest.get("selection_hpo_reuse_contract_schema")
                != SELECTION_HPO_REUSE_SCHEMA
            ):
                reason = "legacy_or_missing_reuse_contract"
            elif not isinstance(manifest.get("selection_hpo_fingerprint"), Mapping):
                reason = "missing_fingerprint"
            elif (
                manifest["selection_hpo_fingerprint"].get("sha256")
                != fingerprint["sha256"]
            ):
                reason = "fingerprint_mismatch"
            else:
                selections: dict[str, dict[str, Any]] = {}
                params: dict[str, dict[str, Any]] = {}
                try:
                    for target_name in TARGET_COLUMNS:
                        paths = manifest["heads"][target_name]["paths"]
                        selection = json.loads(
                            Path(paths["selected_features"]).read_text(encoding="utf-8")
                        )
                        parameter_payload = json.loads(
                            Path(paths["params"]).read_text(encoding="utf-8")
                        )
                        selected_by_side = selection.get("selected_features_by_side")
                        best_params_by_side = parameter_payload.get(
                            "best_params_by_side"
                        )
                        if (
                            not isinstance(selected_by_side, Mapping)
                            or set(selected_by_side) != {"long", "short"}
                            or not isinstance(best_params_by_side, Mapping)
                            or set(best_params_by_side) != {"long", "short"}
                        ):
                            raise ValueError("incomplete_side_local_selection_or_hpo")
                        for side in ("long", "short"):
                            values = selected_by_side[side]
                            if (
                                not isinstance(values, list)
                                or not values
                                or not all(
                                    isinstance(value, str) and value for value in values
                                )
                                or not isinstance(best_params_by_side[side], Mapping)
                                or not best_params_by_side[side]
                            ):
                                raise ValueError("invalid_side_local_selection_or_hpo")
                        selections[target_name] = selection
                        params[target_name] = {
                            side: dict(best_params_by_side[side])
                            for side in ("long", "short")
                        }
                except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError):
                    reason = "missing_or_invalid_head_artifacts"
                else:
                    audit.update(
                        {
                            "auto_reused": True,
                            "reason": "exact_fingerprint_match",
                            "source_manifest": str(manifest_path.resolve()),
                            "source_manifest_sha256": _file_sha256(manifest_path),
                        }
                    )
                    return selections, params, audit
        audit["rejection_counts"][reason] = (
            int(audit["rejection_counts"].get(reason, 0)) + 1
        )
    audit["reason"] = "no_exact_current_schema_sibling_match"
    return None, None, audit


def _normalize_side(values: pd.Series) -> pd.Series:
    raw = values.astype(str).str.strip().str.lower()
    numeric = pd.to_numeric(values, errors="coerce")
    side = pd.Series(np.where(numeric < 0.0, "short", "long"), index=values.index)
    side.loc[raw.isin(("short", "sell", "-1", "-1.0"))] = "short"
    side.loc[raw.isin(("long", "buy", "1", "1.0"))] = "long"
    invalid = values.isna() | ~(
        raw.isin(("short", "sell", "-1", "-1.0", "long", "buy", "1", "1.0"))
        | numeric.notna()
    )
    side.loc[invalid] = np.nan
    return side


def _read_context_for_label_keys(
    context_path: Path,
    wanted: Sequence[str],
    labels: pd.DataFrame,
    *,
    timestamp_column: str,
    symbol_column: str,
    side_column: str,
) -> tuple[pd.DataFrame, str, int]:
    """Push down small/capped context joins instead of loading the full ledger."""

    if len(labels) > 100_000 or side_column != "side_name":
        context = pd.read_parquet(context_path, columns=list(wanted))
        return context, "full_parquet_read", int(len(context))
    try:
        import duckdb
    except ImportError:
        context = pd.read_parquet(context_path, columns=list(wanted))
        return context, "full_parquet_read_no_duckdb", int(len(context))

    keys = labels.loc[:, list(IDENTITY_COLUMNS)].drop_duplicates().copy()
    keys["__ts__"] = pd.to_datetime(keys["__ts__"], utc=True, errors="raise")
    keys["__symbol__"] = keys["__symbol__"].astype(str)
    keys["side"] = keys["side"].astype(str)

    def quote(value: str) -> str:
        return '"' + str(value).replace('"', '""') + '"'

    # Alias every projection back to the exact requested spelling. DuckDB
    # resolves source identifiers case-insensitively, while pandas/config
    # contracts are case-sensitive.
    selected = ", ".join(f"c.{quote(column)} AS {quote(column)}" for column in wanted)
    connection = duckdb.connect()
    try:
        # DuckDB otherwise renders/compares TIMESTAMPTZ values in the host
        # timezone.  The persisted handoff timestamps are naive UTC, while the
        # registered pandas keys are timezone-aware UTC.  Comparing epoch
        # nanoseconds under an explicit UTC session preserves the repository's
        # storage contract even when the worker runs in Europe/Paris.
        connection.execute("SET TimeZone='UTC'")
        connection.register("requested_candidate_keys", keys)
        source_rows = int(
            connection.execute(
                "SELECT count(*) FROM read_parquet(?)", [str(context_path)]
            ).fetchone()[0]
        )
        query = f"""
            SELECT {selected}
            FROM read_parquet(?) AS c
            INNER JOIN requested_candidate_keys AS k
              ON epoch_ns(c.{quote(timestamp_column)}) = epoch_ns(k.__ts__)
             AND CAST(c.{quote(symbol_column)} AS VARCHAR) = k.__symbol__
             AND lower(CAST(c.{quote(side_column)} AS VARCHAR)) = k.side
            WHERE CAST(c.{quote(SELECTED_TOP40_COLUMN)} AS BOOLEAN)
        """
        context = connection.execute(query, [str(context_path)]).fetch_df()
    finally:
        connection.close()
    return context, "duckdb_exact_candidate_key_pushdown", source_rows


def _casefold_unique_feature_columns(columns: Sequence[str]) -> list[str]:
    """Choose one canonical spelling when Parquet has case-only aliases.

    DuckDB follows SQL's case-insensitive identifier rules, so requesting both
    ``AE_reconstruction_error`` and ``ae_reconstruction_error`` silently
    renames one projection.  The shared handoff stores those two columns as
    equivalent compatibility aliases; prefer the all-lowercase config key.
    """

    grouped: dict[str, list[str]] = {}
    order: list[str] = []
    for column in map(str, columns):
        key = column.casefold()
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(column)
    selected: list[str] = []
    for key in order:
        choices = grouped[key]
        selected.append(key if key in choices else choices[0])
    return selected


def _load_labels(
    labels_path: Path,
    *,
    label_resolution_column: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    max_rows: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read only identity, context, and materialized path-target columns."""

    files = _canonical_label_files(labels_path)
    required = {
        "__ts__",
        "__symbol__",
        "side",
        CANDIDATE_ID_COLUMN,
        label_resolution_column,
        *TARGET_COLUMNS.values(),
        *ALL_SUPPORTIVE_LABEL_COLUMNS,
    }
    frames: list[pd.DataFrame] = []
    source_columns: set[str] = set()
    wanted_by_file: dict[Path, list[str]] = {}
    for file in files:
        columns = _parquet_columns(file)
        source_columns.update(columns)
        missing = required.difference(columns)
        if missing:
            raise ValueError(
                f"{file} is missing path auxiliary columns: {sorted(missing)}"
            )
        wanted = [
            column
            for column in [
                *IDENTITY_COLUMNS,
                CANDIDATE_ID_COLUMN,
                label_resolution_column,
                SELECTED_TOP40_COLUMN,
                *CONTEXT_COLUMNS,
                *TARGET_COLUMNS.values(),
                *ALL_SUPPORTIVE_LABEL_COLUMNS,
            ]
            if column in columns
        ]
        wanted_by_file[file] = list(dict.fromkeys(wanted))

    read_contract = "full_parquet_read"
    if max_rows > 0:
        # A smoke cap must bound memory before the wide supportive-label payload
        # is materialized.  First read only canonical identity, choose a stable
        # beginning/middle/end spread, then use DuckDB to fetch those exact rows.
        identity_frames = [
            pd.read_parquet(
                file,
                columns=[
                    *IDENTITY_COLUMNS,
                    CANDIDATE_ID_COLUMN,
                    *(
                        [SELECTED_TOP40_COLUMN]
                        if SELECTED_TOP40_COLUMN in _parquet_columns(file)
                        else []
                    ),
                ],
            )
            for file in files
        ]
        identity = pd.concat(identity_frames, ignore_index=True, copy=False)
        identity["__ts__"] = pd.to_datetime(
            identity["__ts__"], format="mixed", utc=True, errors="coerce"
        )
        identity["__symbol__"] = identity["__symbol__"].astype(str)
        identity["side"] = _normalize_side(identity["side"])
        identity[CANDIDATE_ID_COLUMN] = identity[CANDIDATE_ID_COLUMN].astype(str)
        if SELECTED_TOP40_COLUMN in identity:
            identity = identity.loc[
                _validated_selected_top40(
                    identity[SELECTED_TOP40_COLUMN], source="labels"
                )
            ]
        valid = (
            identity["__ts__"].notna()
            & identity["__symbol__"].ne("")
            & identity["side"].isin(("long", "short"))
        )
        identity = identity.loc[valid]
        if start is not None:
            identity = identity.loc[identity["__ts__"] >= start]
        if end is not None:
            identity = identity.loc[identity["__ts__"] <= end]
        identity = identity.sort_values(
            ["__ts__", "__symbol__", "side"], kind="mergesort"
        ).drop_duplicates(list(STRICT_IDENTITY_COLUMNS), keep="last")
        cap = min(int(max_rows), len(identity))
        positions = np.linspace(0, max(len(identity) - 1, 0), num=cap, dtype=np.int64)
        requested = identity.iloc[np.unique(positions)][
            [CANDIDATE_ID_COLUMN]
        ].drop_duplicates()
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover - production dependency
            raise RuntimeError(
                "DuckDB is required for memory-bounded --max-rows label reads"
            ) from exc
        connection = duckdb.connect()
        try:
            connection.register("requested_label_ids", requested)
            for file in files:
                wanted = wanted_by_file[file]
                quoted = ", ".join(
                    'p."' + str(column).replace('"', '""') + '"' for column in wanted
                )
                query = f"""
                    SELECT {quoted}
                    FROM read_parquet(?) AS p
                    INNER JOIN requested_label_ids AS r
                      ON CAST(p.{CANDIDATE_ID_COLUMN} AS VARCHAR)
                       = r.{CANDIDATE_ID_COLUMN}
                """
                frames.append(connection.execute(query, [str(file)]).fetch_df())
        finally:
            connection.close()
        read_contract = "identity_first_time_spread_then_duckdb_exact_id_pushdown"
    else:
        for file in files:
            frames.append(pd.read_parquet(file, columns=wanted_by_file[file]))
    frame = pd.concat(frames, ignore_index=True, copy=False)
    frame["__ts__"] = pd.to_datetime(
        frame["__ts__"], format="mixed", utc=True, errors="coerce"
    )
    frame[label_resolution_column] = pd.to_datetime(
        frame[label_resolution_column], format="mixed", utc=True, errors="coerce"
    )
    frame["__symbol__"] = frame["__symbol__"].astype(str)
    frame["side"] = _normalize_side(frame["side"])
    frame[CANDIDATE_ID_COLUMN] = frame[CANDIDATE_ID_COLUMN].astype(str)
    expected_candidate_id = candidate_id_series(
        frame["__ts__"], frame["__symbol__"], "1h", frame["side"]
    ).to_numpy()
    if not np.array_equal(frame[CANDIDATE_ID_COLUMN].to_numpy(), expected_candidate_id):
        raise ValueError(
            "label candidate_id does not match canonical UTC/symbol/1h/side identity"
        )
    for column in [*TARGET_COLUMNS.values(), *ALL_SUPPORTIVE_LABEL_COLUMNS]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    for column in CONTEXT_COLUMNS:
        if column not in frame:
            frame[column] = "unknown"
        frame[column] = frame[column].fillna("unknown").astype(str)
    valid_identity = (
        frame["__ts__"].notna()
        & frame["__symbol__"].ne("")
        & frame["side"].isin(("long", "short"))
    )
    frame = frame.loc[valid_identity].copy()
    if start is not None:
        frame = frame.loc[frame["__ts__"] >= start]
    if end is not None:
        frame = frame.loc[frame["__ts__"] <= end]
    frame = frame.sort_values(
        ["__ts__", "__symbol__", "side"], kind="mergesort"
    ).drop_duplicates(list(STRICT_IDENTITY_COLUMNS), keep="last")
    if max_rows > 0:
        cap = min(int(max_rows), len(frame))
        positions = np.linspace(0, max(len(frame) - 1, 0), num=cap, dtype=np.int64)
        frame = frame.iloc[np.unique(positions)]
    frame = frame.reset_index(drop=True)
    return frame, {
        "source": _label_source_signature(files),
        "source_columns": sorted(source_columns),
        "rows_after_identity_and_caps": int(len(frame)),
        "row_cap_sampling": "deterministic_time_spread" if max_rows > 0 else "none",
        "read_contract": read_contract,
        "utc_timestamp_contract": "naive_label_timestamps_interpreted_as_utc",
    }


def _join_archetype_context(
    labels: pd.DataFrame,
    context_path: Path | None,
    *,
    labels_are_canonical_top40: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach canonical base archetypes on the exact UTC row identity.

    The canonical handoff is a candidate stream, so an explicit context source
    also defines the rows on which these downstream auxiliary heads are fit.
    """

    if context_path is None:
        if not labels_are_canonical_top40:
            raise ValueError(
                "Canonical auxiliary-head population requires --archetype-context-path; "
                "use --labels-are-canonical-top40 only for a labels file that "
                "explicitly carries the canonical selected_top40 population"
            )
        rows_before = int(len(labels))
        if SELECTED_TOP40_COLUMN in labels:
            label_selected = _validated_selected_top40(
                labels[SELECTED_TOP40_COLUMN], source="labels"
            )
            selected = labels.loc[label_selected].copy()
            selection_column = SELECTED_TOP40_COLUMN
            rows_selected_before_identity_validation = int(label_selected.sum())
        else:
            # This is the deliberately explicit escape hatch for a labels
            # artifact already materialized from the canonical top-40 handoff.
            # It is never selected implicitly from archetype completeness alone.
            selected = labels.copy()
            selection_column = "declared_all_rows_canonical_top40"
            rows_selected_before_identity_validation = rows_before
        existing = _archetype_context(selected)
        existing_valid = existing.ne("unknown") & existing.ne("")
        if not bool(existing_valid.all()):
            raise ValueError(
                "labels-only canonical population requires complete archetype identity"
            )
        if selected.empty:
            raise ValueError("labels selected_top40 filter produced no rows")
        if selected.duplicated(list(STRICT_IDENTITY_COLUMNS), keep=False).any():
            raise ValueError(
                "labels canonical selected_top40 population has duplicate UTC keys"
            )
        if CANDIDATE_ID_COLUMN not in selected:
            raise ValueError(
                "labels-only canonical population requires preserved candidate_id values"
            )
        selected[CANDIDATE_ID_COLUMN] = _candidate_ids(
            selected[CANDIDATE_ID_COLUMN], source="labels"
        )
        if selected[CANDIDATE_ID_COLUMN].duplicated(keep=False).any():
            raise ValueError(
                "labels canonical population has duplicate candidate_id values"
            )
        selected = selected.drop(columns=[SELECTED_TOP40_COLUMN], errors="ignore")
        return selected, {
            "source": "labels_explicit_canonical_top40",
            "selection_source": "labels",
            "selection_column": selection_column,
            "selection_boolean_contract": (
                "strict_bool_or_0_1_or_true_false_text"
                if selection_column == SELECTED_TOP40_COLUMN
                else "explicit_labels_are_canonical_top40_declaration"
            ),
            "rows_source": rows_before,
            "rows_selected_before_identity_validation": rows_selected_before_identity_validation,
            "rows_selected_after_identity_validation": int(len(selected)),
            "rows_filtered_out": int(rows_before - len(selected)),
            "selected_population_identity_sha256": candidate_identity_sha256(
                selected, columns=STRICT_IDENTITY_COLUMNS
            ),
            "selected_population_timestamp_bounds": _timestamp_bounds(
                selected["__ts__"]
            ),
            "rows_before": rows_before,
            "rows_matched": int(len(selected)),
            "rows_unmatched": int(rows_before - len(selected)),
            "match_fraction": float(len(selected) / max(rows_before, 1)),
            "key": list(STRICT_IDENTITY_COLUMNS),
        }
    if labels_are_canonical_top40:
        raise ValueError(
            "--labels-are-canonical-top40 is labels-only and cannot be combined "
            "with --archetype-context-path"
        )

    columns = _parquet_columns(context_path)
    ts_column = "__ts__" if "__ts__" in columns else "timestamp"
    symbol_column = "__symbol__" if "__symbol__" in columns else "symbol"
    side_column = "side_name" if "side_name" in columns else "side"
    identity = {ts_column, symbol_column, side_column}
    missing_identity = identity.difference(columns)
    if missing_identity:
        raise ValueError(
            f"{context_path} is missing context identity columns: "
            f"{sorted(missing_identity)}"
        )
    available_context = [column for column in CONTEXT_COLUMNS if column in columns]
    if not any(column in available_context for column in ARCHETYPE_COLUMNS):
        raise ValueError(f"{context_path} contains no supported archetype identity")
    if SELECTED_TOP40_COLUMN not in columns:
        raise ValueError(
            f"{context_path} is missing required canonical population flag "
            f"{SELECTED_TOP40_COLUMN}"
        )
    missing_model_context = set(MANDATORY_HANDOFF_MODEL_FEATURES).difference(columns)
    if missing_model_context:
        raise ValueError(
            f"{context_path} is missing mandatory candidate model context: "
            f"{sorted(missing_model_context)}"
        )
    # The canonical handoff is also the only point-in-time source for generated
    # model context that does not belong in the static feature store (frozen
    # AE/GMM outputs, OOD state, and OOF base/meta margin context). Restrict the
    # read to the same config-driven universe used by the auxiliary selector;
    # realized outcomes and arbitrary ledger columns never enter this join.
    handoff_feature_columns, handoff_universe_report = (
        configured_auxiliary_feature_universe(sorted(columns))
    )
    handoff_feature_columns = _casefold_unique_feature_columns(
        dict.fromkeys(
            [
                *MANDATORY_HANDOFF_MODEL_FEATURES,
                *[
                    column
                    for column in handoff_feature_columns
                    if column
                    not in {
                        ts_column,
                        symbol_column,
                        side_column,
                        *CONTEXT_COLUMNS,
                    }
                ],
            ]
        )
    )
    wanted = list(
        dict.fromkeys(
            [
                ts_column,
                symbol_column,
                side_column,
                *([CANDIDATE_ID_COLUMN] if CANDIDATE_ID_COLUMN in columns else []),
                SELECTED_TOP40_COLUMN,
                *available_context,
                *handoff_feature_columns,
            ]
        )
    )
    context, context_read_contract, rows_source = _read_context_for_label_keys(
        context_path,
        wanted,
        labels,
        timestamp_column=ts_column,
        symbol_column=symbol_column,
        side_column=side_column,
    )
    context = context.rename(
        columns={ts_column: "__ts__", symbol_column: "__symbol__", side_column: "side"}
    )
    selected_top40 = _validated_selected_top40(
        context[SELECTED_TOP40_COLUMN], source=str(context_path)
    )
    context = context.loc[selected_top40].copy()
    rows_selected_before_identity_validation = int(len(context))
    context["__ts__"] = pd.to_datetime(
        context["__ts__"], format="mixed", utc=True, errors="coerce"
    )
    context["__symbol__"] = context["__symbol__"].astype(str)
    context["side"] = _normalize_side(context["side"])
    canonical_candidate_id = candidate_id_series(
        context["__ts__"], context["__symbol__"], "1h", context["side"]
    ).to_numpy()
    if CANDIDATE_ID_COLUMN in context:
        if not np.array_equal(
            context[CANDIDATE_ID_COLUMN].astype(str).to_numpy(),
            canonical_candidate_id,
        ):
            raise ValueError("context candidate_id does not match canonical identity")
    else:
        context[CANDIDATE_ID_COLUMN] = canonical_candidate_id
    context = context.loc[
        context["__ts__"].notna()
        & context["__symbol__"].ne("")
        & context["side"].isin(("long", "short"))
    ].copy()
    if context.empty:
        raise ValueError(
            f"{context_path} selected_top40 filter produced no valid UTC rows"
        )
    context[CANDIDATE_ID_COLUMN] = _candidate_ids(
        context[CANDIDATE_ID_COLUMN], source=str(context_path)
    )
    duplicate = context.duplicated(list(STRICT_IDENTITY_COLUMNS), keep=False)
    if bool(duplicate.any()):
        raise ValueError(
            f"{context_path} has {int(duplicate.sum())} duplicate UTC context keys"
        )
    if context[CANDIDATE_ID_COLUMN].duplicated(keep=False).any():
        raise ValueError(f"{context_path} has duplicate canonical candidate_id values")
    for column in available_context:
        if column not in ("side_name",):
            context[column] = context[column].fillna("unknown").astype(str)
    for column in handoff_feature_columns:
        context[column] = pd.to_numeric(context[column], errors="coerce").astype(
            np.float32
        )
    invalid_model_context = {
        column: int((~np.isfinite(context[column].to_numpy(dtype=np.float32))).sum())
        for column in MANDATORY_HANDOFF_MODEL_FEATURES
        if not np.isfinite(context[column].to_numpy(dtype=np.float32)).all()
    }
    if invalid_model_context:
        raise ValueError(
            "Mandatory candidate model context must be finite on every handoff row: "
            f"{invalid_model_context}"
        )

    selected_population_identity_sha256 = candidate_identity_sha256(
        context, columns=STRICT_IDENTITY_COLUMNS
    )
    label_context = [column for column in CONTEXT_COLUMNS if column in labels.columns]
    labels_candidate_id = (
        _candidate_ids(labels[CANDIDATE_ID_COLUMN], source="labels")
        if CANDIDATE_ID_COLUMN in labels
        else None
    )
    base = labels.drop(columns=[*label_context, SELECTED_TOP40_COLUMN], errors="ignore")
    rows_before = len(base)
    joined = base.merge(
        context,
        on=list(STRICT_IDENTITY_COLUMNS),
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    joined = joined.sort_values(
        list(STRICT_IDENTITY_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)
    if labels_candidate_id is not None:
        expected_candidate_ids = pd.DataFrame(
            {
                "__ts__": labels["__ts__"],
                "__symbol__": labels["__symbol__"],
                "side": labels["side"],
                "__label_candidate_id__": labels_candidate_id,
            }
        )
        expected_candidate_ids = joined.loc[:, list(IDENTITY_COLUMNS)].merge(
            expected_candidate_ids,
            on=list(IDENTITY_COLUMNS),
            how="left",
            validate="one_to_one",
            sort=False,
        )
        if not expected_candidate_ids["__label_candidate_id__"].equals(
            joined[CANDIDATE_ID_COLUMN]
        ):
            raise ValueError("labels and canonical context disagree on candidate_id")
    selected_context = _archetype_context(joined)
    valid = selected_context.ne("unknown") & selected_context.ne("")
    if joined.empty or not bool(valid.all()):
        raise ValueError(
            "Canonical context join produced empty or incomplete archetype identity"
        )
    if set(joined["side"].unique()) != {"long", "short"}:
        raise ValueError("Canonical context join must retain both long and short rows")
    return joined, {
        "source": str(context_path.resolve()),
        "selection_source": str(context_path.resolve()),
        "selection_column": SELECTED_TOP40_COLUMN,
        "selection_boolean_contract": "strict_bool_or_0_1_or_true_false_text",
        "rows_source": rows_source,
        "rows_selected_before_identity_validation": rows_selected_before_identity_validation,
        "rows_selected_after_identity_validation": int(len(context)),
        "rows_filtered_out": int(rows_source - len(context)),
        "selected_population_identity_sha256": selected_population_identity_sha256,
        "selected_population_timestamp_bounds": _timestamp_bounds(context["__ts__"]),
        "rows_before": int(rows_before),
        "rows_matched": int(len(joined)),
        "rows_unmatched": int(rows_before - len(joined)),
        "match_fraction": float(len(joined) / max(rows_before, 1)),
        "key": list(STRICT_IDENTITY_COLUMNS),
        "available_context_columns": available_context,
        "handoff_model_feature_columns": handoff_feature_columns,
        "handoff_model_feature_count": int(len(handoff_feature_columns)),
        "handoff_feature_universe_report": handoff_universe_report,
        "context_read_contract": context_read_contract,
        "selected_archetype_column": next(
            column for column in ARCHETYPE_COLUMNS if column in joined.columns
        ),
    }


def _overlay_handoff_model_features(
    matrix: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    requested_features: Sequence[str],
    static_report: Mapping[str, Any],
    handoff_feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fill non-static generated context from the exact joined handoff rows."""

    available_static = set(map(str, static_report.get("available_feature_names", [])))
    allowed_handoff = set(map(str, handoff_feature_columns))
    overlaid: list[str] = []
    for column in map(str, requested_features):
        if column in available_static or column not in allowed_handoff:
            continue
        if column not in frame:
            continue
        matrix[column] = pd.to_numeric(frame[column], errors="coerce").to_numpy(
            dtype=np.float32, copy=False
        )
        overlaid.append(column)
    available = sorted(available_static.union(overlaid))
    report = dict(static_report)
    report["available_feature_names"] = available
    report["available_features"] = int(len(available))
    report["missing_features"] = sorted(
        set(map(str, requested_features)) - set(available)
    )
    report["handoff_overlay_features"] = sorted(overlaid)
    report["handoff_overlay_feature_count"] = int(len(overlaid))
    report["feature_source_contract"] = (
        "static store wins on duplicate columns; absent generated/model context "
        "is filled only from the exact UTC-keyed OOF/frozen handoff"
    )
    return matrix, report


def _static_feature_columns(feature_dir: Path, symbols: Sequence[str]) -> list[str]:
    """Inspect only relevant static-store schemas before requesting their columns."""

    from extreme_price_movements.data_store import _feature_schema_names

    columns: set[str] = set()
    for symbol in sorted(set(map(str, symbols))):
        path = feature_dir / f"symbol={symbol.replace('/', '_')}.parquet"
        columns.update(map(str, _feature_schema_names(str(path))))
    return sorted(columns)


def _static_feature_read_cache_limits() -> tuple[int, int]:
    """Resolve a strict per-run cache cap without widening loader memory bounds."""

    try:
        max_bytes = int(
            os.getenv(
                "EPM_PATH_AUX_STATIC_FEATURE_CACHE_MAX_BYTES",
                str(STATIC_FEATURE_READ_CACHE_DEFAULT_MAX_BYTES),
            )
        )
    except (TypeError, ValueError):
        max_bytes = STATIC_FEATURE_READ_CACHE_DEFAULT_MAX_BYTES
    try:
        max_entries = int(
            os.getenv("EPM_PATH_AUX_STATIC_FEATURE_CACHE_MAX_ENTRIES", "8")
        )
    except (TypeError, ValueError):
        max_entries = STATIC_FEATURE_READ_CACHE_HARD_MAX_ENTRIES
    return (
        max(0, min(max_bytes, STATIC_FEATURE_READ_CACHE_HARD_MAX_BYTES)),
        max(0, min(max_entries, STATIC_FEATURE_READ_CACHE_HARD_MAX_ENTRIES)),
    )


def _coalesced_selection_static_read_periods(
    timestamps: pd.Series | Sequence[Any],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Bound sparse B/M/E reads to short UTC blocks without widening cache coverage."""

    unique = (
        pd.DatetimeIndex(
            pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce").dropna()
        )
        .unique()
        .sort_values()
    )
    periods: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    block_start: pd.Timestamp | None = None
    block_end: pd.Timestamp | None = None
    for timestamp in unique:
        start = pd.Timestamp(timestamp)
        # The static store is hourly. A one-nanosecond half-open interval can
        # collapse to an empty interval when lower-level storage normalizes
        # timestamps to seconds, especially for a symbol with one sampled row.
        end = start + pd.Timedelta(hours=1)
        if (
            block_start is None
            or block_end is None
            or start >= block_end + SELECTION_STATIC_READ_COALESCE_GAP
            or end - block_start > SELECTION_STATIC_READ_MAX_BLOCK
        ):
            if block_start is not None and block_end is not None:
                periods.append((block_start, block_end))
            block_start, block_end = start, end
        else:
            block_end = end
    if block_start is not None and block_end is not None:
        periods.append((block_start, block_end))
    return periods


def _period_coverage_mask(
    timestamps: pd.Series,
    periods: Sequence[tuple[pd.Timestamp, pd.Timestamp]],
) -> np.ndarray:
    """Return exact UTC membership in reader periods; gaps remain uncovered."""

    values = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True, errors="coerce"))
    covered = np.zeros(len(values), dtype=bool)
    for start, end in periods:
        covered |= (values >= start) & (values < end)
    return covered


def _static_reader_payload_nbytes(value: Any, seen: set[int] | None = None) -> int:
    """Estimate retained reader arrays without traversing arbitrary user objects."""

    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return 0
    seen.add(value_id)
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, pd.DataFrame):
        return int(value.memory_usage(index=True, deep=True).sum())
    if isinstance(value, pd.Series):
        return int(value.memory_usage(index=True, deep=True))
    if isinstance(value, Mapping):
        return sum(_static_reader_payload_nbytes(item, seen) for item in value.values())
    if isinstance(value, tuple):
        return sum(_static_reader_payload_nbytes(item, seen) for item in value)
    return 0


def _static_reader_loaded_nbytes(loaded: Any) -> int:
    """Count only the raw/assembled buffers retained by the canonical reader."""

    explicit = getattr(loaded, "__static_feature_cache_nbytes__", None)
    if explicit is not None:
        try:
            return max(0, int(explicit))
        except (TypeError, ValueError):
            return 0
    seen: set[int] = set()
    return (
        _static_reader_payload_nbytes(getattr(loaded, "_raw", None), seen)
        + _static_reader_payload_nbytes(getattr(loaded, "_symbol_indices", None), seen)
        + _static_reader_payload_nbytes(getattr(loaded, "_assembled", None), seen)
    )


@dataclass(frozen=True)
class _StaticFeatureReadCacheEntry:
    symbols: tuple[str, ...]
    feature_names: frozenset[str]
    periods: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]
    loaded: Any
    retained_bytes: int


class _StaticFeatureReadCache:
    """Bounded LRU reuse of canonical reader buffers within one runner invocation."""

    def __init__(self, *, max_bytes: int, max_entries: int):
        self.max_bytes = int(max_bytes)
        self.max_entries = int(max_entries)
        self._entries: OrderedDict[tuple[str, ...], _StaticFeatureReadCacheEntry] = (
            OrderedDict()
        )
        self.retained_bytes = 0
        self.hits = 0
        self.misses = 0
        self.admissions = 0
        self.rejected_entries = 0
        self.evictions = 0
        self.reused_rows = 0

    def get(
        self,
        *,
        symbols: Sequence[str],
        requested_features: Sequence[str],
    ) -> _StaticFeatureReadCacheEntry | None:
        key = tuple(map(str, symbols))
        entry = self._entries.get(key)
        if entry is None or not set(map(str, requested_features)).issubset(
            entry.feature_names
        ):
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return entry

    def put(
        self,
        *,
        symbols: Sequence[str],
        requested_features: Sequence[str],
        periods: Sequence[tuple[pd.Timestamp, pd.Timestamp]],
        loaded: Any,
    ) -> None:
        retained_bytes = _static_reader_loaded_nbytes(loaded)
        if (
            self.max_bytes <= 0
            or self.max_entries <= 0
            or retained_bytes <= 0
            or retained_bytes > self.max_bytes
        ):
            self.rejected_entries += 1
            return
        key = tuple(map(str, symbols))
        existing = self._entries.pop(key, None)
        if existing is not None:
            self.retained_bytes -= existing.retained_bytes
        while self._entries and (
            len(self._entries) >= self.max_entries
            or self.retained_bytes + retained_bytes > self.max_bytes
        ):
            _, evicted = self._entries.popitem(last=False)
            self.retained_bytes -= evicted.retained_bytes
            self.evictions += 1
        if self.retained_bytes + retained_bytes > self.max_bytes:
            self.rejected_entries += 1
            return
        self._entries[key] = _StaticFeatureReadCacheEntry(
            symbols=key,
            feature_names=frozenset(map(str, requested_features)),
            periods=tuple(
                (pd.Timestamp(start), pd.Timestamp(end)) for start, end in periods
            ),
            loaded=loaded,
            retained_bytes=retained_bytes,
        )
        self.retained_bytes += retained_bytes
        self.admissions += 1

    def report(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.max_bytes and self.max_entries),
            "max_bytes": int(self.max_bytes),
            "max_entries": int(self.max_entries),
            "retained_bytes": int(self.retained_bytes),
            "retained_entries": int(len(self._entries)),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "admissions": int(self.admissions),
            "rejected_entries": int(self.rejected_entries),
            "evictions": int(self.evictions),
            "reused_rows": int(self.reused_rows),
        }


def _load_static_features(
    frame: pd.DataFrame,
    *,
    feature_dir: Path,
    requested_features: Sequence[str],
    read_cache: _StaticFeatureReadCache | None = None,
    sampled_periods: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read exact UTC timestamp/symbol rows through the canonical static endpoint."""

    from extreme_price_movements.static_feature_store import read_static_features

    try:
        feature_store_ts = pd.to_datetime(
            feature_dir.name, format="%Y%m%d_%H%M%S", utc=True
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "feature_dir must end in the canonical YYYYMMDD_HHMMSS store id"
        ) from exc
    if feature_dir.parent.name != "features":
        raise ValueError("feature_dir must be <data_root>/features/<YYYYMMDD_HHMMSS>")
    data_root = feature_dir.parents[1]
    requested = list(dict.fromkeys(map(str, requested_features)))
    out = pd.DataFrame(
        np.nan,
        index=frame.index,
        columns=requested,
        dtype=np.float32,
    )
    ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    symbols = frame["__symbol__"].astype(str)
    grouped = {
        str(symbol): np.asarray(index, dtype=np.int64)
        for symbol, index in frame.groupby("__symbol__", sort=False).indices.items()
    }
    # Candidate selection can request the entire config universe (1,000+
    # columns).  The canonical reader must inspect complete symbol files before
    # applying sparse B/M/E periods, so an eight-symbol batch can transiently
    # retain several GiB even when the requested output has only 30k rows.
    # Once selection is complete, per-head reloads request a small selected
    # contract and can safely retain the wider batch.
    batch_size = 1 if sampled_periods and len(requested) > 512 else 8
    read_errors: list[str] = []
    available: set[str] = set()
    allowed_period_count = 0
    ordered_symbols = sorted(grouped)
    for offset in range(0, len(ordered_symbols), batch_size):
        batch_symbols = ordered_symbols[offset : offset + batch_size]
        positions = np.concatenate([grouped[symbol] for symbol in batch_symbols])
        batch_ts = ts.iloc[positions]
        batch_start = batch_ts.min()
        batch_end = batch_ts.max() + pd.Timedelta(nanoseconds=1)
        cached = (
            read_cache.get(
                symbols=batch_symbols,
                requested_features=requested,
            )
            if read_cache is not None
            else None
        )
        sources: list[tuple[Any, bool]] = []
        batch_periods = (
            _coalesced_selection_static_read_periods(batch_ts)
            if sampled_periods
            else [(batch_start, batch_end)]
        )
        if sampled_periods:
            allowed_period_count += len(batch_periods)
        cached_coverage = np.zeros(len(batch_ts), dtype=bool)
        if cached is not None:
            sources.append((cached.loaded, True))
            cached_coverage = _period_coverage_mask(batch_ts, cached.periods)
            read_cache.reused_rows += int(cached_coverage.sum())
        # Sparse B/M/E selection always preserves its bounded query periods.
        # The full selected-feature read stays contiguous unless every requested
        # row is already backed by a cached period; otherwise cache gaps would be
        # incorrectly treated as source coverage.
        should_read = sampled_periods or not bool(cached_coverage.all())
        if should_read:
            try:
                read_kwargs: dict[str, Any] = {
                    "feature_store_ts": feature_store_ts,
                    "data_root": data_root,
                    "feature_keys": requested,
                    "symbols": batch_symbols,
                    "start_ts": batch_start,
                    "end_ts": batch_end,
                    "output_layout": "panels",
                }
                if sampled_periods:
                    read_kwargs["allowed_periods"] = batch_periods
                loaded = read_static_features(**read_kwargs)
            except Exception as exc:
                read_errors.append(
                    f"symbols={batch_symbols[0]}..{batch_symbols[-1]}: "
                    f"{type(exc).__name__}: {exc}"
                )
                loaded = None
            if loaded is not None and hasattr(loaded, "symbol_frame"):
                sources.append((loaded, False))
                if read_cache is not None and cached is None:
                    read_cache.put(
                        symbols=batch_symbols,
                        requested_features=requested,
                        periods=batch_periods,
                        loaded=loaded,
                    )
        for loaded, _from_cache in sources:
            batch_available = [feature for feature in requested if feature in loaded]
            available.update(batch_available)
            for symbol in batch_symbols:
                rows = grouped[symbol]
                symbol_frame = loaded.symbol_frame(symbol, keys=batch_available)
                if not isinstance(symbol_frame, pd.DataFrame) or symbol_frame.empty:
                    continue
                symbol_available = [
                    feature
                    for feature in batch_available
                    if feature in symbol_frame.columns
                ]
                if not symbol_available:
                    continue
                symbol_frame.index = pd.to_datetime(
                    symbol_frame.index, utc=True, errors="coerce"
                )
                target_index = pd.DatetimeIndex(ts.iloc[rows])
                present = target_index.isin(symbol_frame.index)
                if not bool(present.any()):
                    continue
                aligned = symbol_frame.reindex(target_index[present])
                out.loc[frame.index[rows[present]], symbol_available] = aligned.loc[
                    :, symbol_available
                ].to_numpy(dtype=np.float32, copy=False)
    finite_fraction = {
        feature: float(np.isfinite(out[feature].to_numpy(dtype=np.float32)).mean())
        for feature in requested
    }
    return out, {
        "reader": "extreme_price_movements.static_feature_store.read_static_features",
        "feature_store_ts": feature_store_ts.isoformat(),
        "data_root": str(data_root),
        "rows": int(len(frame)),
        "symbols": int(symbols.nunique()),
        "requested_features": int(len(requested)),
        "available_features": int(len(available)),
        "available_feature_names": sorted(available),
        "missing_features": sorted(set(requested) - available),
        "finite_fraction_by_feature": finite_fraction,
        "read_errors": read_errors,
        "sampled_period_read": bool(sampled_periods),
        "allowed_period_count": int(allowed_period_count),
        "read_cache": read_cache.report()
        if read_cache is not None
        else {"enabled": False},
    }


def _archetype_context(frame: pd.DataFrame) -> pd.Series:
    resolved = pd.Series("unknown", index=frame.index, dtype=object)
    for column in ARCHETYPE_COLUMNS:
        if column in frame.columns:
            values = frame[column].fillna("unknown").astype(str).str.strip()
            usable = resolved.eq("unknown") & values.ne("") & values.ne("unknown")
            resolved.loc[usable] = values.loc[usable]
    return resolved


def _complete_archetype_source(frame: pd.DataFrame, column: str) -> bool:
    if column not in frame.columns:
        return False
    values = frame[column].fillna("unknown").astype(str).str.strip()
    return bool((values.ne("") & values.ne("unknown")).all())


def _selection_hpo_reference_contract(
    frame: pd.DataFrame,
    *,
    selection_hpo_reference_end: pd.Timestamp,
    label_resolution_column: str,
    selected_population_identity_sha256: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Partition selected candidates into frozen reference and emitted OOF rows."""

    if label_resolution_column not in frame:
        raise ValueError(
            f"selected population is missing label-resolution column {label_resolution_column!r}"
        )
    decision = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    resolved = pd.to_datetime(frame[label_resolution_column], utc=True, errors="coerce")
    reference_mask = (
        decision.lt(selection_hpo_reference_end)
        & resolved.lt(selection_hpo_reference_end)
    ).to_numpy(dtype=bool)
    oof_mask = (decision.ge(selection_hpo_reference_end) & resolved.notna()).to_numpy(
        dtype=bool
    )
    contract: dict[str, Any] = {
        "schema": "path_auxiliary_selection_hpo_reference_split_v1",
        "selection_hpo_reference_end": selection_hpo_reference_end.isoformat(),
        "timestamp_column": "__ts__",
        "label_resolution_column": label_resolution_column,
        "selected_population_identity_sha256": selected_population_identity_sha256,
        "reference_row_rule": (
            "__ts__ < selection_hpo_reference_end AND "
            f"{label_resolution_column} < selection_hpo_reference_end"
        ),
        "emitted_oof_row_rule": "__ts__ >= selection_hpo_reference_end",
        "decision_bounds": _timestamp_bounds(decision),
        "label_resolved_bounds": _timestamp_bounds(resolved),
        "reference_decision_bounds": _timestamp_bounds(decision.loc[reference_mask]),
        "reference_label_resolved_bounds": _timestamp_bounds(
            resolved.loc[reference_mask]
        ),
        "oof_decision_bounds": _timestamp_bounds(decision.loc[oof_mask]),
        "oof_label_resolved_bounds": _timestamp_bounds(resolved.loc[oof_mask]),
        "reference_rows": int(reference_mask.sum()),
        "oof_candidate_rows": int(oof_mask.sum()),
        "boundary_rows_excluded": int(decision.eq(selection_hpo_reference_end).sum()),
        "unresolved_rows_excluded": int(resolved.isna().sum()),
    }
    contract["contract_sha256"] = _stable_sha256(contract)
    return reference_mask, oof_mask, contract


def _assert_oof_identities_subset_selected_population(
    frame: pd.DataFrame,
    *,
    oof_predictions: np.ndarray,
    oof_fold_ids: np.ndarray,
    selection_hpo_reference_end: pd.Timestamp,
    selected_population_identity_sha256: str,
) -> dict[str, Any]:
    """Fail closed unless emitted OOF rows are selected, unique, and post-cutoff."""

    oof = np.asarray(oof_predictions, dtype=np.float32)
    fold_ids = np.asarray(oof_fold_ids, dtype=np.int16)
    if len(oof) != len(frame) or len(fold_ids) != len(frame):
        raise ValueError(
            "auxiliary OOF arrays must align exactly with selected identities"
        )
    if frame.duplicated(list(STRICT_IDENTITY_COLUMNS), keep=False).any():
        raise ValueError(
            "selected auxiliary population has duplicate strict UTC candidate identities"
        )
    available = np.isfinite(oof)
    if not bool(available.any()):
        raise ValueError("auxiliary OOF emission produced no finite predictions")
    decision = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    if bool((decision.loc[available] < selection_hpo_reference_end).any()):
        raise ValueError(
            "auxiliary OOF identities must be at or after the reference end"
        )
    if bool((fold_ids[available] < 0).any()):
        raise ValueError(
            "finite auxiliary OOF predictions require a non-negative fold id"
        )
    selected_identity_hash = candidate_identity_sha256(
        frame, columns=STRICT_IDENTITY_COLUMNS
    )
    oof_identity = frame.loc[available, list(STRICT_IDENTITY_COLUMNS)]
    # ``frame`` is the post-filter inner join; this explicit subset assertion
    # guards future changes that might construct OOF rows from another source.
    selected_index = pd.MultiIndex.from_frame(
        frame.loc[:, list(STRICT_IDENTITY_COLUMNS)]
    )
    oof_index = pd.MultiIndex.from_frame(oof_identity)
    if not bool(oof_index.isin(selected_index).all()):
        raise ValueError(
            "auxiliary OOF identities are outside the selected_top40 population"
        )
    return {
        "selected_joined_population_identity_sha256": selected_identity_hash,
        "canonical_selected_population_identity_sha256": selected_population_identity_sha256,
        "oof_identity_sha256": candidate_identity_sha256(
            oof_identity, columns=STRICT_IDENTITY_COLUMNS
        ),
        "selected_joined_population_rows": int(len(frame)),
        "oof_available_rows": int(available.sum()),
        "oof_identity_subset_of_selected_top40": True,
        "oof_decision_bounds": _timestamp_bounds(decision.loc[available]),
    }


def _strict_oof_fold_evidence(
    frame: pd.DataFrame,
    *,
    fitted: Mapping[str, Any],
    label_resolution_column: str,
) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    """Bind each finite OOF row to the exact fitted side-local fold evidence.

    The model implementation records the actual train-decision maximum and
    validation interval in ``fold_metrics``.  This reconstructs the same
    resolved training membership from the immutable target frame, verifies that
    recorded decision maximum, and persists the latest resolved-label
    availability that made the fold trainable.  No validation-window inference
    is permitted downstream.
    """

    required = set(IDENTITY_COLUMNS) | {CANDIDATE_ID_COLUMN, label_resolution_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"strict auxiliary OOF provenance is missing columns: {missing}"
        )
    oof = np.asarray(fitted["oof_predictions"], dtype=np.float32)
    fold_ids = np.asarray(fitted["oof_fold_ids"], dtype=np.int64)
    if len(oof) != len(frame) or len(fold_ids) != len(frame):
        raise ValueError(
            "strict auxiliary OOF provenance arrays must align with target rows"
        )
    decision = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    resolved = pd.to_datetime(frame[label_resolution_column], utc=True, errors="raise")
    side_values = frame["side"].astype(str)
    available = np.isfinite(oof)
    evidence = pd.DataFrame(
        {
            "validation_start": pd.Series(
                pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]"
            ),
            "train_decision_cutoff": pd.Series(
                pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]"
            ),
            "label_resolution_available_at": pd.Series(
                pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]"
            ),
        }
    )
    persisted_metrics: dict[str, list[dict[str, Any]]] = {}
    for side in ("long", "short"):
        bundle = fitted["models_by_side"].get(side)
        if not isinstance(bundle, Mapping):
            raise ValueError(
                f"strict auxiliary OOF provenance lacks fitted {side!r} bundle"
            )
        metrics = bundle.get("fold_metrics")
        if not isinstance(metrics, list) or not metrics:
            raise ValueError(
                f"strict auxiliary OOF provenance lacks actual fitted fold metrics for {side!r}"
            )
        side_mask = side_values.eq(side)
        by_fold: dict[
            int, tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]
        ] = {}
        persisted: list[dict[str, Any]] = []
        for raw_metric in metrics:
            if not isinstance(raw_metric, Mapping):
                raise ValueError(
                    f"strict auxiliary OOF provenance has invalid {side!r} fold metric"
                )
            try:
                fold = int(raw_metric["fold"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"strict auxiliary OOF provenance has invalid {side!r} fold ID"
                ) from exc
            if fold < 0 or fold in by_fold:
                raise ValueError(
                    f"strict auxiliary OOF provenance has duplicate/invalid {side!r} fold ID"
                )
            valid_start = _require_explicit_utc_timestamp(
                raw_metric.get("valid_start"), name=f"{side} fold {fold} valid_start"
            )
            valid_end = _require_explicit_utc_timestamp(
                raw_metric.get("valid_end"), name=f"{side} fold {fold} valid_end"
            )
            recorded_train_end = _require_explicit_utc_timestamp(
                raw_metric.get("train_end"), name=f"{side} fold {fold} train_end"
            )
            if valid_end < valid_start or recorded_train_end >= valid_start:
                raise ValueError(
                    f"strict auxiliary OOF provenance has invalid {side!r} fold interval"
                )
            train_mask = side_mask & decision.lt(valid_start) & resolved.lt(valid_start)
            train_decisions = decision.loc[train_mask]
            train_resolved = resolved.loc[train_mask]
            if train_decisions.empty or train_resolved.empty:
                raise ValueError(
                    f"strict auxiliary OOF provenance found no resolved training rows for {side!r} fold {fold}"
                )
            actual_train_end = train_decisions.max()
            if actual_train_end != recorded_train_end:
                raise ValueError(
                    f"strict auxiliary OOF provenance train_end disagrees with actual fitted {side!r} fold {fold} training rows"
                )
            resolution_cutoff = train_resolved.max()
            if resolution_cutoff >= valid_start:
                raise ValueError(
                    f"strict auxiliary OOF provenance has unresolved training labels for {side!r} fold {fold}"
                )
            by_fold[fold] = (
                valid_start,
                valid_end,
                resolution_cutoff,
                actual_train_end,
            )
            metric = dict(raw_metric)
            metric.update(
                {
                    "train_max_decision_timestamp": actual_train_end.isoformat(),
                    "train_decision_cutoff": resolution_cutoff.isoformat(),
                    "label_resolution_available_at": resolution_cutoff.isoformat(),
                    "strict_execution_ev_provenance": "actual_fitted_fold_training_membership_v1",
                }
            )
            persisted.append(metric)
        persisted_metrics[side] = persisted
        positions = np.flatnonzero(available & side_mask.to_numpy())
        for position in positions:
            fold = int(fold_ids[position])
            if fold not in by_fold:
                raise ValueError(
                    f"strict auxiliary OOF provenance has no actual {side!r} fold evidence for fold {fold}"
                )
            valid_start, valid_end, resolution_cutoff, _ = by_fold[fold]
            if not (valid_start <= decision.iloc[position] <= valid_end):
                raise ValueError(
                    f"strict auxiliary OOF row is outside its actual {side!r} fitted validation fold"
                )
            evidence.iloc[position] = (
                valid_start,
                resolution_cutoff,
                resolution_cutoff,
            )
    if available.any() and evidence.loc[available].isna().any(axis=None):
        raise ValueError(
            "strict auxiliary OOF provenance failed to bind every finite prediction"
        )
    return evidence, persisted_metrics


def _metric_slices(
    target_name: str,
    frame: pd.DataFrame,
    target: np.ndarray,
    oof: np.ndarray,
) -> dict[str, Any]:
    valid = np.isfinite(target) & np.isfinite(oof)

    def metrics(mask: np.ndarray) -> dict[str, Any]:
        if not bool(mask.any()):
            return {"rows": 0}
        _, values = auxiliary_hpo_objective(target_name, target[mask], oof[mask])
        return values

    out: dict[str, Any] = {"overall": metrics(valid), "by_side": {}, "by_archetype": {}}
    for side in ("long", "short"):
        out["by_side"][side] = metrics(valid & frame["side"].eq(side).to_numpy())
    archetypes = _archetype_context(frame)
    for archetype in sorted(archetypes.unique()):
        out["by_archetype"][str(archetype)] = metrics(
            valid & archetypes.eq(archetype).to_numpy()
        )
    return out


def _persist_head(
    output_dir: Path,
    *,
    target_name: str,
    frame: pd.DataFrame,
    selection: Mapping[str, Any],
    fitted: Mapping[str, Any],
    sample_weight: np.ndarray,
    label_resolution_column: str,
) -> dict[str, Any]:
    target = frame[TARGET_COLUMNS[target_name]].to_numpy(dtype=np.float32, copy=False)
    oof = np.asarray(fitted["oof_predictions"], dtype=np.float32)
    oof_fold_ids = np.asarray(fitted["oof_fold_ids"], dtype=np.int16)
    metrics = _metric_slices(target_name, frame, target, oof)
    strict_evidence, strict_fold_metrics = _strict_oof_fold_evidence(
        frame,
        fitted=fitted,
        label_resolution_column=label_resolution_column,
    )
    head_dir = output_dir / target_name
    bundles_dir = head_dir / "bundles"
    bundles_dir.mkdir(parents=True, exist_ok=True)
    selection_path = head_dir / "selected_features_by_side.json"
    params_path = head_dir / "params_by_side.json"
    metrics_path = head_dir / "metrics.json"
    oof_path = head_dir / "oof_predictions.parquet"
    oof_manifest_path = head_dir / "oof_predictions.manifest.json"
    _write_json(selection_path, dict(selection))
    _write_json(
        params_path,
        {
            "target": target_name,
            "hpo_trial_count_by_side": {
                side: bundle.get("hpo_trial_count")
                for side, bundle in fitted["models_by_side"].items()
            },
            "hpo_best_value_by_side": {
                side: bundle.get("hpo_best_value")
                for side, bundle in fitted["models_by_side"].items()
            },
            "hpo_rows_by_side": {
                side: bundle.get("hpo_rows")
                for side, bundle in fitted["models_by_side"].items()
            },
            "hpo_sampling_contract_by_side": {
                side: bundle.get("hpo_sampling_contract")
                for side, bundle in fitted["models_by_side"].items()
            },
            "hpo_reused_by_side": {
                side: bool(bundle.get("hpo_reused"))
                for side, bundle in fitted["models_by_side"].items()
            },
            "oos_fold_contract_by_side": {
                side: bundle.get("oos_fold_contract")
                for side, bundle in fitted["models_by_side"].items()
            },
            "final_inference_fit_contract_by_side": {
                side: bundle.get("final_inference_fit_contract")
                for side, bundle in fitted["models_by_side"].items()
            },
            "best_params_by_side": {
                side: bundle["best_params"]
                for side, bundle in fitted["models_by_side"].items()
            },
            "sample_weight_contract": fitted.get("sample_weight_contract"),
            "sample_weight_summary": fitted.get("sample_weight_summary"),
            "reference_split_contract": fitted.get("selection_hpo_reference_contract")
            or fitted.get("reference_split_contract"),
            "oof_population_report": fitted.get("oof_population_report"),
            "selection_hpo_reuse": fitted.get("selection_hpo_reuse"),
        },
    )
    _write_json(
        metrics_path,
        {"target": target_name, **metrics, "fold_metrics": strict_fold_metrics},
    )
    oof_frame = frame.loc[:, [*IDENTITY_COLUMNS, CANDIDATE_ID_COLUMN]].copy()
    oof_frame["archetype"] = _archetype_context(frame)
    oof_frame["target"] = target
    oof_frame["sample_weight"] = np.asarray(sample_weight, dtype=np.float32)
    oof_frame["oof_prediction_log1p"] = oof
    if target_name == "time_to_first_meaningful_mfe":
        natural = np.clip(np.expm1(oof), 0.0, 12.0)
        oof_frame["pred_time_to_first_meaningful_mfe_12h"] = natural
    elif target_name == "peak_mfe_12h_atr":
        natural = np.clip(np.expm1(oof), 0.0, PEAK_MFE_ATR_CLIP)
        oof_frame["pred_peak_mfe_12h_atr"] = natural
    elif target_name == "mae_before_meaningful_mfe_atr":
        natural = np.clip(np.expm1(oof), 0.0, MAE_BEFORE_MEANINGFUL_MFE_ATR_CLIP)
        oof_frame["pred_mae_before_meaningful_mfe_atr_12h"] = natural
    elif target_name == "bars_before_price_stops_decreasing":
        natural = np.clip(np.expm1(oof), 0.0, 12.0)
        oof_frame["pred_bars_before_price_stops_decreasing_12h"] = natural
    elif target_name == "future_slope_atr_per_hour":
        natural = np.clip(np.expm1(oof), 0.0, FUTURE_SLOPE_ATR_PER_HOUR_CLIP)
        oof_frame["pred_future_slope_atr_per_hour_12h"] = natural
    else:  # pragma: no cover - guarded by the fixed TARGET_COLUMNS contract.
        raise ValueError(f"unknown auxiliary target {target_name!r}")
    oof_frame["oof_fold"] = oof_fold_ids
    for column in strict_evidence:
        oof_frame[column] = strict_evidence[column].to_numpy()
    oof_frame["available_at"] = pd.to_datetime(
        oof_frame["__ts__"], utc=True, errors="raise"
    )
    oof_frame["oof_available"] = np.isfinite(oof).astype(np.int8)
    reference_contract = dict(
        fitted.get("selection_hpo_reference_contract")
        or fitted.get("reference_split_contract")
        or {}
    )
    if reference_contract:
        oof_frame["selection_hpo_reference_end"] = reference_contract.get(
            "selection_hpo_reference_end"
        )
        oof_frame["oof_after_selection_hpo_reference_end"] = (
            oof_frame["__ts__"]
            >= pd.Timestamp(reference_contract["selection_hpo_reference_end"])
        ).astype(np.int8)
    _atomic_to_parquet(oof_frame, oof_path)
    prediction_columns = [
        column for column in oof_frame.columns if column.startswith("pred_")
    ]
    if len(prediction_columns) != 1:
        raise ValueError(
            f"strict auxiliary OOF requires exactly one natural prediction column for {target_name}"
        )
    oof_manifest = {
        "schema": "path_auxiliary_oof_prediction_role_v1",
        "prediction_role": PREDICTION_ROLES[target_name],
        "source_artifact_sha256": _file_sha256(oof_path),
        "prediction_columns": {
            prediction_columns[0]: {
                "role": "pre_entry_auxiliary_oof_prediction",
                "target": False,
                "head": target_name,
            }
        },
        "identity_columns": [*IDENTITY_COLUMNS, CANDIDATE_ID_COLUMN],
        "strict_execution_ev_evidence_columns": [
            "oof_fold",
            "validation_start",
            "train_decision_cutoff",
            "label_resolution_available_at",
        ],
    }
    oof_manifest["prediction_role_manifest_sha256"] = _stable_sha256(oof_manifest)
    _write_json(oof_manifest_path, oof_manifest)
    bundle_paths: dict[str, str] = {}
    for side, bundle in fitted["models_by_side"].items():
        bundle_path = bundles_dir / f"{side}.joblib"
        _atomic_joblib_dump(
            {
                "schema": MODEL_SCHEMA,
                "target_name": target_name,
                "target_column": TARGET_COLUMNS[target_name],
                "side": side,
                "selected_features": list(bundle["selected_features"]),
                "base_archetype_label_feature_contract": selection.get(
                    "base_archetype_label_feature_contract"
                ),
                "best_params": dict(bundle["best_params"]),
                "purge_hours": fitted["purge_hours"],
                "sample_weight_contract": fitted.get("sample_weight_contract"),
                "sample_weight_summary": bundle.get("sample_weight_summary"),
                "reference_split_contract": fitted.get(
                    "selection_hpo_reference_contract"
                )
                or fitted.get("reference_split_contract"),
                "oof_population_report": fitted.get("oof_population_report"),
                "hpo_reused": bool(bundle.get("hpo_reused")),
                "oos_fold_contract": bundle.get("oos_fold_contract"),
                "final_inference_fit_contract": bundle.get(
                    "final_inference_fit_contract"
                ),
                "model_role": bundle.get(
                    "model_role",
                    "all_resolved_final_inference_excluded_from_oos_metrics",
                ),
                "final_inference_model": bundle.get(
                    "final_inference_model", bundle["model"]
                ),
                "selection_hpo_reuse": fitted.get("selection_hpo_reuse"),
                "model": bundle["model"],
            },
            bundle_path,
        )
        bundle_paths[side] = str(bundle_path)
    return {
        "rows": int(len(frame)),
        "selected_features_by_side": selection["selected_features_by_side"],
        "metrics": metrics,
        "reference_split_contract": fitted.get("selection_hpo_reference_contract")
        or fitted.get("reference_split_contract"),
        "oof_population_report": fitted.get("oof_population_report"),
        "selection_hpo_reuse": fitted.get("selection_hpo_reuse"),
        "paths": {
            "selected_features": str(selection_path),
            "params": str(params_path),
            "metrics": str(metrics_path),
            "oof_predictions": str(oof_path),
            "oof_prediction_manifest": str(oof_manifest_path),
            "bundles": bundle_paths,
        },
    }


def run(
    *,
    labels_path: Path,
    feature_dir: Path,
    output_dir: Path,
    archetype_context_path: Path | None = None,
    labels_are_canonical_top40: bool = False,
    selection_hpo_reference_end: str | pd.Timestamp | None = None,
    label_resolution_column: str = DEFAULT_LABEL_RESOLUTION_COLUMN,
    n_trials: int = 75,
    seed: int = 42,
    purge_hours: float = 13.0,
    start: str | None = None,
    end: str | None = None,
    max_rows: int = 0,
    selection_rows: int = 45_000,
    hpo_rows: int = 45_000,
    force_selection_hpo: bool = False,
    overwrite: bool = False,
    resource_min_free_ram_gib: float = 2.0,
    resource_max_process_rss_gib: float = 12.0,
    resource_min_free_disk_gib: float = 10.0,
    resource_check_interval_seconds: float = 60.0,
    resource_telemetry_path: Path | None = None,
    resource_guard: TrainingResourceGuard | None = None,
) -> dict[str, Any]:
    """Materialize the causal head inputs once, then select and fit each target once."""

    resource_guard = resource_guard or _build_resource_guard(
        output_dir=output_dir,
        min_free_ram_gib=resource_min_free_ram_gib,
        max_process_rss_gib=resource_max_process_rss_gib,
        min_free_disk_gib=resource_min_free_disk_gib,
        check_interval_seconds=resource_check_interval_seconds,
        telemetry_path=resource_telemetry_path,
    )
    guard_telemetry_path = getattr(resource_guard, "telemetry_path", None)
    allowed_preflight_paths = (
        [Path(guard_telemetry_path)]
        if (
            guard_telemetry_path is not None
            and Path(guard_telemetry_path).parent.resolve() == output_dir.resolve()
        )
        else []
    )
    existing_entries = (
        [
            path
            for path in output_dir.iterdir()
            if path.resolve()
            not in {allowed.resolve() for allowed in allowed_preflight_paths}
        ]
        if output_dir.exists()
        else []
    )
    if (
        existing_entries
        and not overwrite
        and not _checkpoint_path(output_dir).is_file()
        and not (output_dir / "manifest.json").is_file()
    ):
        raise FileExistsError(
            f"Output directory is non-empty and has no resumable checkpoint; pass --overwrite explicitly: {output_dir}"
        )
    reference_end_ts = _require_explicit_utc_timestamp(
        selection_hpo_reference_end, name="selection_hpo_reference_end"
    )
    resource_guard.preflight("labels_load")
    start_ts = pd.Timestamp(start) if start else None
    end_ts = pd.Timestamp(end) if end else None
    if start_ts is not None:
        start_ts = (
            start_ts.tz_localize("UTC")
            if start_ts.tzinfo is None
            else start_ts.tz_convert("UTC")
        )
    if end_ts is not None:
        end_ts = (
            end_ts.tz_localize("UTC")
            if end_ts.tzinfo is None
            else end_ts.tz_convert("UTC")
        )
    labels, label_report = _load_labels(
        labels_path,
        label_resolution_column=label_resolution_column,
        start=start_ts,
        end=end_ts,
        max_rows=max_rows,
    )
    labels, archetype_context_report = _join_archetype_context(
        labels,
        archetype_context_path,
        labels_are_canonical_top40=labels_are_canonical_top40,
    )
    if labels.empty:
        raise ValueError("No label rows remain after UTC date and row caps")
    reference_mask, oof_mask, selection_hpo_reference_contract = (
        _selection_hpo_reference_contract(
            labels,
            selection_hpo_reference_end=reference_end_ts,
            label_resolution_column=label_resolution_column,
            selected_population_identity_sha256=archetype_context_report[
                "selected_population_identity_sha256"
            ],
        )
    )
    if not bool(reference_mask.any()):
        raise ValueError("No selected rows satisfy the selection/HPO reference cutoff")
    if not bool(oof_mask.any()):
        raise ValueError(
            "No selected rows at or after the reference end are available for OOF emission"
        )
    reference_labels = labels.loc[reference_mask].reset_index(drop=True)
    complete_archetype_sources = [
        column
        for column in ARCHETYPE_COLUMNS
        if _complete_archetype_source(reference_labels, column)
    ]
    if not complete_archetype_sources:
        raise ValueError(
            "A persisted, complete base-archetype source is required for the "
            "inference feature contract"
        )
    canonical_archetype_source = complete_archetype_sources[0]
    archetype_sources = [
        canonical_archetype_source,
        *[
            column
            for column in ARCHETYPE_COLUMNS
            if column in labels.columns and column != canonical_archetype_source
        ],
    ]
    archetype_feature_contract = fit_base_archetype_label_feature_contract(
        reference_labels,
        source_columns=archetype_sources,
        canonical_source=canonical_archetype_source,
    )
    archetype_features = transform_base_archetype_label_features(
        labels, archetype_feature_contract
    )
    handoff_feature_columns = list(
        map(
            str,
            archetype_context_report.get("handoff_model_feature_columns", []),
        )
    )
    static_columns = _static_feature_columns(feature_dir, labels["__symbol__"])
    requested_features, static_universe = configured_auxiliary_feature_universe(
        [*static_columns, *handoff_feature_columns]
    )
    if not requested_features:
        raise RuntimeError(
            "No config-driven base/meta features are available in the canonical static store"
        )
    selection_hpo_fingerprint = _selection_hpo_fingerprint(
        selected_population_identity_sha256=archetype_context_report[
            "selected_population_identity_sha256"
        ],
        selection_hpo_reference_contract=selection_hpo_reference_contract,
        archetype_feature_contract=archetype_feature_contract,
        feature_dir=feature_dir,
        static_columns=static_columns,
        handoff_feature_columns=handoff_feature_columns,
        configured_universe=static_universe,
        selection_rows=selection_rows,
        hpo_rows=hpo_rows,
        n_trials=n_trials,
        seed=seed,
        purge_hours=purge_hours,
    )
    run_fingerprint_payload = {
        "schema": CHECKPOINT_SCHEMA,
        "selection_hpo_fingerprint_sha256": selection_hpo_fingerprint["sha256"],
        "label_source_signature_sha256": label_report.get("source", {}).get(
            "signature_sha256"
        ),
        "feature_store_tree_signature": _tree_stat_signature(feature_dir),
        "candidate_identity_sha256": candidate_identity_sha256(
            labels, columns=IDENTITY_COLUMNS
        ),
        "selected_population_identity_sha256": archetype_context_report[
            "selected_population_identity_sha256"
        ],
        "caps": {"start": start, "end": end, "max_rows": int(max_rows)},
        "labels_are_canonical_top40": bool(labels_are_canonical_top40),
        "label_resolution_column": label_resolution_column,
    }
    run_fingerprint = {
        "payload": run_fingerprint_payload,
        "sha256": _stable_sha256(run_fingerprint_payload),
    }
    checkpoint, already_complete = _load_or_initialize_checkpoint(
        output_dir,
        fingerprint=run_fingerprint,
        overwrite=overwrite,
        allowed_preflight_paths=allowed_preflight_paths,
    )
    if already_complete:
        _progress("run_reused", fingerprint_sha256=run_fingerprint["sha256"])
        return checkpoint
    _progress("run_resumed", fingerprint_sha256=run_fingerprint["sha256"])
    selections, reused_hpo_params, selection_hpo_reuse = _read_reusable_selection_hpo(
        output_dir,
        fingerprint=selection_hpo_fingerprint,
        force_selection_hpo=force_selection_hpo,
    )
    local_selections: dict[str, dict[str, Any]] = {}
    for target_name in TARGET_COLUMNS:
        record = checkpoint.get("heads", {}).get(target_name, {}).get("selection")
        if record is None:
            continue
        try:
            payload = _load_checkpoint_artifact(
                record, stage="selection", fingerprint=run_fingerprint["sha256"]
            )
            selected_by_side = payload.get("selected_features_by_side")
            if not isinstance(selected_by_side, Mapping) or set(selected_by_side) != {
                "long",
                "short",
            }:
                raise ValueError("missing long/short selections")
            local_selections[target_name] = dict(payload)
            _progress("selection_reused", head=target_name)
        except (OSError, ValueError, TypeError, KeyError):
            raise ValueError(
                f"invalid selection checkpoint for auxiliary head {target_name}"
            )
    if local_selections:
        sibling_selections = selections or {}
        selections = {**sibling_selections, **local_selections}
        selection_hpo_reuse["local_checkpoint_heads"] = sorted(local_selections)
    reusable_feature_universe = set(requested_features).union(
        archetype_feature_contract["features"]
    )
    static_cache_max_bytes, static_cache_max_entries = (
        _static_feature_read_cache_limits()
    )
    static_feature_read_cache = _StaticFeatureReadCache(
        max_bytes=static_cache_max_bytes,
        max_entries=static_cache_max_entries,
    )
    if selections is not None:
        reused_features = {
            feature
            for selection in selections.values()
            for features in selection["selected_features_by_side"].values()
            for feature in features
        }
        unavailable = sorted(reused_features.difference(reusable_feature_universe))
        if unavailable:
            selection_hpo_reuse.update(
                {
                    "auto_reused": False,
                    "reason": "reused_feature_unavailable_in_current_universe",
                    "unavailable_features": unavailable,
                }
            )
            selections = None
            reused_hpo_params = None
    availability = {
        "requested_configured_universe": static_universe,
        "exact_alignment": "static feature values are reindexed by UTC __ts__ and __symbol__; side is label context only",
        "selection_hpo_reference_contract": selection_hpo_reference_contract,
        "selection_hpo_fingerprint": selection_hpo_fingerprint,
        "selection_hpo_reuse": selection_hpo_reuse,
    }
    if selections is None or len(selections) != len(TARGET_COLUMNS):
        from extreme_price_movements.lgbm_pipeline import _time_spread_subsample_indices

        reference_positions = np.flatnonzero(reference_mask)
        selection_relative_idx = _time_spread_subsample_indices(
            np.arange(len(reference_labels), dtype=np.float32),
            max_n=max(300, min(int(selection_rows), len(reference_labels))),
            random_state=int(seed),
            classifier=False,
            timestamps=reference_labels["__ts__"].to_numpy(),
        )
        selection_idx = reference_positions[selection_relative_idx]
        selection_labels = labels.iloc[selection_idx].reset_index(drop=True)
        resource_guard.checkpoint("selection_feature_load")
        selection_matrix, static_report = _load_static_features(
            selection_labels,
            feature_dir=feature_dir,
            requested_features=requested_features,
            read_cache=static_feature_read_cache,
            sampled_periods=True,
        )
        selection_matrix, static_report = _overlay_handoff_model_features(
            selection_matrix,
            selection_labels,
            requested_features=requested_features,
            static_report=static_report,
            handoff_feature_columns=handoff_feature_columns,
        )
        selection_matrix = pd.concat(
            [
                selection_matrix.reset_index(drop=True),
                archetype_features.iloc[selection_idx].reset_index(drop=True),
            ],
            axis=1,
            copy=False,
        )
        found_columns = list(static_report.get("available_feature_names", []))
        available_features, loaded_universe = configured_auxiliary_feature_universe(
            [*found_columns, *archetype_feature_contract["features"]]
        )
        if not available_features:
            raise RuntimeError(
                "Canonical static-store load retained no config-driven auxiliary features"
            )
        selection_matrix = selection_matrix.reindex(columns=available_features).astype(
            np.float32, copy=False
        )
        availability["loaded_configured_universe"] = loaded_universe
        availability["static_store_read"] = static_report
        selections = dict(selections or {})
        for target_name, target_column in TARGET_COLUMNS.items():
            if target_name in selections:
                continue
            _progress("selection_start", head=target_name)
            eligible = np.isfinite(
                selection_labels[target_column].to_numpy(dtype=np.float32, copy=False)
            )
            target_frame = selection_labels.loc[eligible].reset_index(drop=True)
            target_matrix = selection_matrix.loc[eligible].reset_index(drop=True)
            if target_frame.empty:
                raise ValueError(f"No finite target rows for {target_name}")
            selection_weight = build_auxiliary_sample_weights(target_frame, target_name)
            selections[target_name] = select_features_with_current_pipeline(
                target_matrix,
                target_frame[target_column].to_numpy(dtype=np.float32, copy=False),
                timestamps=target_frame["__ts__"].to_numpy(),
                assets=target_frame["__symbol__"].to_numpy(),
                sides=target_frame["side"].to_numpy(),
                archetypes=_archetype_context(target_frame).to_numpy(),
                mandatory_features_by_side={
                    side: [
                        feature
                        for feature in archetype_feature_contract["canonical_features"]
                        if bool(
                            target_matrix.loc[target_frame["side"].eq(side), feature]
                            .to_numpy(dtype=np.float32)
                            .any()
                        )
                    ]
                    for side in ("long", "short")
                },
                target_name=target_name,
                sample_weight=selection_weight,
                random_state=seed,
            )
            selections[target_name]["base_archetype_label_feature_contract"] = (
                archetype_feature_contract
            )
            selections[target_name]["selection_reference_bounds"] = {
                "decision": _timestamp_bounds(target_frame["__ts__"]),
                "label_resolved": _timestamp_bounds(
                    target_frame[label_resolution_column]
                ),
                "rows": int(len(target_frame)),
                "selection_hpo_reference_end": reference_end_ts.isoformat(),
            }
            selections[target_name]["selection_hpo_reference_contract_hash"] = (
                selection_hpo_reference_contract["contract_sha256"]
            )
            selection_checkpoint_path = _checkpoint_artifact_path(
                output_dir, target_name, "shared", "selection.joblib"
            )
            _atomic_joblib_dump(
                {
                    "fingerprint_sha256": run_fingerprint["sha256"],
                    "payload": selections[target_name],
                },
                selection_checkpoint_path,
            )
            checkpoint.setdefault("heads", {}).setdefault(target_name, {})[
                "selection"
            ] = _artifact_record(
                selection_checkpoint_path,
                stage="selection",
                fingerprint=run_fingerprint["sha256"],
            )
            _save_checkpoint(output_dir, checkpoint)
            for side in ("long", "short"):
                _progress("selection_complete", head=target_name, side=side)
    else:
        selection_labels = reference_labels
        availability["loaded_configured_universe"] = {
            "contract": "reused_selection_hpo_exact_fingerprint",
            "available_selected_features": sorted(reusable_feature_universe),
        }
        availability["static_store_read"] = {
            "skipped": True,
            "reason": "exact_selection_hpo_reuse",
        }
        for selection in selections.values():
            selection["selection_hpo_reused"] = True
            selection["selection_hpo_reference_contract_hash"] = (
                selection_hpo_reference_contract["contract_sha256"]
            )
    # The wide selection matrix and its canonical-reader buffers are no longer
    # needed. Retaining them while materializing the full 1M+ row model matrix
    # can multiply peak RSS by several GiB.
    if "selection_matrix" in locals():
        del selection_matrix
    if "target_matrix" in locals():
        del target_matrix
    static_feature_read_cache = _StaticFeatureReadCache(max_bytes=0, max_entries=0)
    gc.collect()
    _write_json(output_dir / "input_universe_availability.json", availability)
    availability["selection_rows"] = int(len(selection_labels))
    availability["selection_sampling_contract"] = (
        "shared target-neutral beginning/middle/end time-spread sample"
    )
    availability["full_selected_static_store_read_by_head"] = {}
    _write_json(output_dir / "input_universe_availability.json", availability)
    results: dict[str, Any] = {}
    for target_name, target_column in TARGET_COLUMNS.items():
        head_checkpoint = checkpoint.setdefault("heads", {}).setdefault(target_name, {})
        completed_record = head_checkpoint.get("complete")
        if completed_record is not None:
            try:
                results[target_name] = _load_checkpoint_artifact(
                    completed_record,
                    stage="head_complete",
                    fingerprint=run_fingerprint["sha256"],
                )
                _progress("head_reused", head=target_name)
                continue
            except (OSError, ValueError, TypeError, KeyError):
                raise ValueError(
                    f"invalid completed checkpoint for auxiliary head {target_name}"
                )
        eligible = np.isfinite(
            labels[target_column].to_numpy(dtype=np.float32, copy=False)
        )
        target_frame = labels.loc[eligible].reset_index(drop=True)
        selection = selections[target_name]
        selected_union = list(
            dict.fromkeys(
                feature
                for side in ("long", "short")
                for feature in selection["selected_features_by_side"][side]
            )
        )
        _progress(
            "head_feature_load_start",
            head=target_name,
            rows=int(len(target_frame)),
            selected_features=int(len(selected_union)),
        )
        resource_guard.checkpoint(f"head_feature_load:{target_name}")
        target_matrix, full_static_report = _load_static_features(
            target_frame,
            feature_dir=feature_dir,
            requested_features=selected_union,
            read_cache=None,
        )
        target_matrix, full_static_report = _overlay_handoff_model_features(
            target_matrix,
            target_frame,
            requested_features=selected_union,
            static_report=full_static_report,
            handoff_feature_columns=handoff_feature_columns,
        )
        selected_archetype_features = [
            feature
            for feature in selected_union
            if feature in archetype_features.columns
        ]
        if selected_archetype_features:
            target_matrix.loc[:, selected_archetype_features] = archetype_features.loc[
                eligible, selected_archetype_features
            ].to_numpy(dtype=np.float32, copy=False)
        target_matrix = target_matrix.reindex(columns=selected_union).astype(
            np.float32, copy=False
        )
        availability["full_selected_static_store_read_by_head"][target_name] = (
            full_static_report
        )
        _write_json(output_dir / "input_universe_availability.json", availability)
        _progress(
            "head_feature_load_complete",
            head=target_name,
            rows=int(len(target_matrix)),
            selected_features=int(len(selected_union)),
        )
        sample_weight = build_auxiliary_sample_weights(target_frame, target_name)

        resume_by_side: dict[str, dict[str, Any]] = {}
        for side in ("long", "short"):
            side_state = head_checkpoint.get("sides", {}).get(side, {})
            resumed: dict[str, Any] = {}
            for stage, key in (("hpo", "hpo"), ("final_model", "final_model")):
                record = side_state.get(key)
                if record is not None:
                    resumed[key] = _load_checkpoint_artifact(
                        record, stage=stage, fingerprint=run_fingerprint["sha256"]
                    )
            fold_payloads: dict[int, Mapping[str, Any]] = {}
            for fold_id, record in (side_state.get("oof_folds") or {}).items():
                fold_payloads[int(fold_id)] = _load_checkpoint_artifact(
                    record, stage="oof_fold", fingerprint=run_fingerprint["sha256"]
                )
            if fold_payloads:
                resumed["oof_folds"] = fold_payloads
            if resumed:
                resume_by_side[side] = resumed

        def checkpoint_progress(
            event: str, side: str, payload: Mapping[str, Any]
        ) -> None:
            resource_guard.checkpoint(f"{target_name}:{side}:{event}")
            _progress(
                event,
                head=target_name,
                side=side,
                **{
                    key: value
                    for key, value in payload.items()
                    if key
                    not in {"model", "prediction", "valid_idx", "metric", "best_params"}
                },
            )
            side_state = head_checkpoint.setdefault("sides", {}).setdefault(side, {})
            stage_map = {
                "hpo_complete": ("hpo", "hpo.joblib"),
                "oof_fold_complete": (
                    "oof_fold",
                    f"oof_fold_{int(payload['fold']):03d}.joblib",
                ),
                "final_model_complete": ("final_model", "final_model.joblib"),
            }
            if event not in stage_map:
                return
            stage, filename = stage_map[event]
            checkpoint_payload: Mapping[str, Any]
            if event == "hpo_complete":
                checkpoint_payload = {
                    "best_params": dict(payload["best_params"]),
                    "hpo_best_value": payload.get("hpo_best_value"),
                    "hpo_trial_count": payload.get("hpo_trial_count"),
                }
            elif event == "oof_fold_complete":
                checkpoint_payload = {
                    "valid_idx": np.asarray(payload["valid_idx"], dtype=np.int64),
                    "prediction": np.asarray(payload["prediction"], dtype=np.float32),
                    "metric": dict(payload["metric"]),
                }
            else:
                checkpoint_payload = {
                    "model": payload["model"],
                    "contract": dict(payload["contract"]),
                }
            artifact_path = _checkpoint_artifact_path(
                output_dir, target_name, side, filename
            )
            _atomic_joblib_dump(
                {
                    "fingerprint_sha256": run_fingerprint["sha256"],
                    "payload": checkpoint_payload,
                },
                artifact_path,
            )
            record = _artifact_record(
                artifact_path, stage=stage, fingerprint=run_fingerprint["sha256"]
            )
            if event == "oof_fold_complete":
                side_state.setdefault("oof_folds", {})[str(payload["fold"])] = record
            else:
                side_state[stage] = record
            _save_checkpoint(output_dir, checkpoint)

        _progress("head_start", head=target_name)
        resource_guard.checkpoint(f"head_fit:{target_name}")
        fitted = fit_side_aware_auxiliary_models(
            target_matrix,
            target_frame[target_column].to_numpy(dtype=np.float32, copy=False),
            selected_features_by_side=selection["selected_features_by_side"],
            timestamps=target_frame["__ts__"].to_numpy(),
            label_resolved_at=target_frame[label_resolution_column].to_numpy(),
            selection_hpo_reference_end=reference_end_ts,
            sides=target_frame["side"].to_numpy(),
            target_name=target_name,
            sample_weight=sample_weight,
            n_trials=n_trials,
            hpo_rows=hpo_rows,
            random_state=seed,
            purge_hours=purge_hours,
            preset_hpo_params_by_side=(
                reused_hpo_params.get(target_name)
                if reused_hpo_params is not None
                else None
            ),
            resume_by_side=resume_by_side,
            progress_callback=checkpoint_progress,
        )
        resource_guard.checkpoint(f"head_fit_complete:{target_name}")
        fitted["sample_weight_contract"] = (
            "head_specific_supportive_labels_clipped_0.5_2.0_v1; training_loss_only; "
            "validation_mda_hpo_early_stopping_oof_metrics_unweighted"
        )
        fitted["selection_hpo_reference_contract"] = selection_hpo_reference_contract
        fitted["selection_hpo_reuse"] = selection_hpo_reuse
        fitted["oof_population_report"] = (
            _assert_oof_identities_subset_selected_population(
                target_frame,
                oof_predictions=np.asarray(fitted["oof_predictions"]),
                oof_fold_ids=np.asarray(fitted["oof_fold_ids"]),
                selection_hpo_reference_end=reference_end_ts,
                selected_population_identity_sha256=archetype_context_report[
                    "selected_population_identity_sha256"
                ],
            )
        )
        results[target_name] = _persist_head(
            output_dir,
            target_name=target_name,
            frame=target_frame,
            selection=selection,
            fitted=fitted,
            sample_weight=sample_weight,
            label_resolution_column=label_resolution_column,
        )
        head_complete_path = _checkpoint_artifact_path(
            output_dir, target_name, "shared", "head_complete.joblib"
        )
        _atomic_joblib_dump(
            {
                "fingerprint_sha256": run_fingerprint["sha256"],
                "payload": results[target_name],
            },
            head_complete_path,
        )
        head_checkpoint["complete"] = _artifact_record(
            head_complete_path,
            stage="head_complete",
            fingerprint=run_fingerprint["sha256"],
        )
        _save_checkpoint(output_dir, checkpoint)
        _progress("head_complete", head=target_name)
        del target_matrix, target_frame, sample_weight, fitted
        gc.collect()
    manifest = {
        "schema": RUNNER_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "model_schema": MODEL_SCHEMA,
        "target_schema": TARGET_SCHEMA,
        "peak_mfe_atr_clip": float(PEAK_MFE_ATR_CLIP),
        "mae_before_meaningful_mfe_atr_clip": float(MAE_BEFORE_MEANINGFUL_MFE_ATR_CLIP),
        "future_slope_atr_per_hour_clip": float(FUTURE_SLOPE_ATR_PER_HOUR_CLIP),
        "usable_mfe_floor": {
            "atr_multiple": float(MIN_USABLE_MFE_ATR),
            "minimum_return": float(MIN_USABLE_MFE_RETURN),
            "contract": "max(atr_multiple * atr_fraction, minimum_return)",
        },
        "labels_path": str(labels_path.resolve()),
        "archetype_context_path": (
            str(archetype_context_path.resolve())
            if archetype_context_path is not None
            else None
        ),
        "labels_are_canonical_top40": bool(labels_are_canonical_top40),
        "label_resolution_column": label_resolution_column,
        "feature_dir": str(feature_dir.resolve()),
        "row_identity": list(IDENTITY_COLUMNS),
        "candidate_identity_sha256": candidate_identity_sha256(
            labels,
            columns=IDENTITY_COLUMNS,
        ),
        "candidate_population_contract": (
            "selected_top40 is validated and filtered before the exact UTC inner "
            "join to the canonical archetype handoff; emitted OOF identities must "
            "be a subset of that selected population"
        ),
        "timestamp_contract": "UTC only; naive label timestamps are interpreted as UTC",
        "model_input_contract": "config base+meta universe plus frozen pre-entry base-archetype labels; CatBoost path labels excluded",
        "base_archetype_label_feature_contract": archetype_feature_contract,
        "selection_contract": (
            "target-specific full selector run independently inside long and short; "
            "0.88 redundancy pruning, side-local univariate/Relief/MDA, then "
            "side-local HPO"
        ),
        "supportive_label_columns": list(ALL_SUPPORTIVE_LABEL_COLUMNS),
        "sample_weight_contract": (
            "head_specific_supportive_labels_clipped_0.5_2.0_v1; training_loss_only; "
            "validation_mda_hpo_early_stopping_oof_metrics_unweighted"
        ),
        "selection_hpo_reference_contract": selection_hpo_reference_contract,
        "selection_hpo_reference_contract_sha256": selection_hpo_reference_contract[
            "contract_sha256"
        ],
        "selection_hpo_reuse_contract_schema": SELECTION_HPO_REUSE_SCHEMA,
        "selection_hpo_fingerprint": selection_hpo_fingerprint,
        "selection_hpo_reuse": selection_hpo_reuse,
        "force_selection_hpo": bool(force_selection_hpo),
        "hpo_oof_contract": (
            "feature selection and HPO use only the frozen pre-reference rows; "
            "persisted OOF is expanding monthly OOS emitted at or after the "
            "reference cutoff; the all-resolved inference fit is excluded from OOS metrics"
        ),
        "purge_hours": float(purge_hours),
        "n_trials": int(n_trials),
        "seed": int(seed),
        "caps": {"start": start, "end": end, "max_rows": int(max_rows)},
        "selection_rows": int(selection_rows),
        "hpo_rows_per_side": int(hpo_rows),
        "lgbm_n_jobs": default_auxiliary_lgbm_n_jobs(),
        "lgbm_resource_contract": "RAM-capped default: reserve 4 GiB and budget 4 GiB per worker, maximum 3 workers",
        "hpo_sampling_contract": (
            "target-neutral beginning/middle/end time-spread sample; default "
            "15k rows per temporal third"
        ),
        "label_report": label_report,
        "archetype_context_report": archetype_context_report,
        "availability_report": str(output_dir / "input_universe_availability.json"),
        "checkpoint": str(_checkpoint_path(output_dir)),
        "training_resource_guard": _resource_guard_contract(resource_guard),
        "heads": results,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", required=True, type=Path)
    parser.add_argument(
        "--archetype-context-path",
        type=Path,
        help=(
            "Canonical UTC-keyed base/meta handoff supplying side x archetype "
            "identity; required when labels do not contain complete archetypes"
        ),
    )
    parser.add_argument(
        "--labels-are-canonical-top40",
        action="store_true",
        help=(
            "Allow labels-only population input only when labels explicitly carry "
            "the canonical selected_top40 flag"
        ),
    )
    parser.add_argument(
        "--selection-hpo-reference-end",
        required=True,
        help=(
            "Required timezone-aware UTC cutoff: selection/HPO rows and their "
            "resolved labels must be strictly before it; emitted OOF starts at or after it"
        ),
    )
    parser.add_argument(
        "--label-resolution-column",
        default=DEFAULT_LABEL_RESOLUTION_COLUMN,
        help="UTC label-resolution timestamp column required for the reference split",
    )
    parser.add_argument("--feature-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-trials", type=int, default=75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--purge-hours", type=float, default=13.0)
    parser.add_argument("--start", help="Inclusive UTC date/timestamp cap")
    parser.add_argument("--end", help="Inclusive UTC date/timestamp cap")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Deterministic smoke cap after UTC sorting; 0 disables it",
    )
    parser.add_argument(
        "--selection-rows",
        type=int,
        default=45_000,
        help="Beginning/middle/end wide feature-selection row cap",
    )
    parser.add_argument(
        "--hpo-rows",
        type=int,
        default=45_000,
        help="Per-side Optuna population, sampled evenly across beginning/middle/end",
    )
    parser.add_argument(
        "--force-selection-hpo",
        action="store_true",
        help="Ignore exact sibling selection/HPO reuse and rerun both high-CPU stages",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacing files under --output-dir",
    )
    parser.add_argument(
        "--resource-min-free-ram-gib",
        type=float,
        default=2.0,
        help="Fail closed when available RAM is below this threshold (default: 2 GiB).",
    )
    parser.add_argument(
        "--resource-max-process-rss-gib",
        type=float,
        default=12.0,
        help="Fail closed when this process exceeds this RSS threshold (default: 12 GiB).",
    )
    parser.add_argument(
        "--resource-min-free-disk-gib",
        type=float,
        default=10.0,
        help="Fail closed when free output-filesystem disk is below this threshold (default: 10 GiB).",
    )
    parser.add_argument(
        "--resource-check-interval-seconds",
        type=float,
        default=60.0,
        help="Minimum interval between boundary resource samples (default: 60 seconds).",
    )
    parser.add_argument(
        "--resource-telemetry-path",
        type=Path,
        default=None,
        help=(
            "Append guard events as JSONL here; defaults to "
            "OUTPUT_DIR/training_resource_telemetry.jsonl."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run(
        labels_path=args.labels_path,
        archetype_context_path=args.archetype_context_path,
        labels_are_canonical_top40=args.labels_are_canonical_top40,
        selection_hpo_reference_end=args.selection_hpo_reference_end,
        label_resolution_column=args.label_resolution_column,
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        seed=args.seed,
        purge_hours=args.purge_hours,
        start=args.start,
        end=args.end,
        max_rows=args.max_rows,
        selection_rows=args.selection_rows,
        hpo_rows=args.hpo_rows,
        force_selection_hpo=args.force_selection_hpo,
        overwrite=args.overwrite,
        resource_min_free_ram_gib=args.resource_min_free_ram_gib,
        resource_max_process_rss_gib=args.resource_max_process_rss_gib,
        resource_min_free_disk_gib=args.resource_min_free_disk_gib,
        resource_check_interval_seconds=args.resource_check_interval_seconds,
        resource_telemetry_path=args.resource_telemetry_path,
    )
    print(
        json.dumps(
            {"output_dir": str(args.output_dir), "heads": list(manifest["heads"])},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
