"""Policy-OOS artifact provenance checks.

The policy optimiser may only use predictions as executable OOS evidence when
the scoring base and meta artifacts are proven to have been fit before the
policy prediction window.  This module intentionally relies on small sidecar
manifests rather than loading large model pickle files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


BASE_ALLOWED_ROLES = {"base_model_fit", "train_base", "train_base_pre_policy_tail"}
META_ALLOWED_ROLES = {"meta_model_fit", "train_meta", "train_meta_pre_policy_tail"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance_manifest_path(artifact_path: Path) -> Path:
    return artifact_path.with_suffix(".manifest.json")


def _first_present(payload: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in payload and payload[key] not in (None, ""):
            return payload[key]
    return None


def _normalise_manifest(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    nested = payload.get("policy_oos_provenance")
    if isinstance(nested, dict):
        out = dict(payload)
        out.update(nested)
        return out
    return dict(payload)


def load_provenance_manifest(artifact_path: Path) -> Dict[str, Any]:
    manifest_path = provenance_manifest_path(artifact_path)
    if not manifest_path.exists():
        return {
            "manifest_path": str(manifest_path),
            "manifest_present": False,
        }
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "manifest_path": str(manifest_path),
            "manifest_present": True,
            "manifest_read_error": str(exc),
        }
    payload = _normalise_manifest(payload)
    payload["manifest_path"] = str(manifest_path)
    payload["manifest_present"] = True
    return payload


def _validate_one_artifact(
    *,
    name: str,
    artifact_path: Path,
    allowed_roles: set[str],
    current_slice_plan_sha256: str,
    policy_fit_end: pd.Timestamp,
    policy_predict_start: pd.Timestamp,
    row_disjoint_policy_oos: bool = False,
) -> Dict[str, Any]:
    diag: Dict[str, Any] = {
        "name": name,
        "artifact_path": str(artifact_path),
        "artifact_present": artifact_path.exists(),
        "valid": False,
        "errors": [],
    }
    if not artifact_path.exists():
        diag["errors"].append("missing_artifact")
        return diag

    manifest = load_provenance_manifest(artifact_path)
    diag["manifest"] = manifest
    if not bool(manifest.get("manifest_present")):
        diag["errors"].append("missing_artifact_provenance_manifest")
        return diag
    if manifest.get("manifest_read_error"):
        diag["errors"].append("unreadable_artifact_provenance_manifest")
        return diag

    fit_start_raw = _first_present(
        manifest,
        (
            "source_model_fit_start",
            "model_fit_start",
            "fit_start",
            "training_fit_start",
            "fit_window_start",
        ),
    )
    fit_end_raw = _first_present(
        manifest,
        (
            "source_model_fit_end",
            "model_fit_end",
            "fit_end",
            "training_fit_end",
            "fit_window_end",
        ),
    )
    source_role = str(
        _first_present(
            manifest,
            ("source_slice_role", "fit_role", "training_slice_role", "slice_role"),
        )
        or ""
    )
    slice_plan_sha256 = str(
        _first_present(manifest, ("slice_plan_sha256", "slice_plan_hash")) or ""
    )
    feature_contract_hash = str(
        _first_present(
            manifest,
            ("feature_contract_hash", "feature_schema_hash", "feature_list_hash"),
        )
        or ""
    )
    generated_from_final_fit = bool(
        manifest.get("generated_from_final_fit_bundle", False)
    ) or source_role == "full_inference_fit"

    fit_start = pd.to_datetime(fit_start_raw, utc=True, errors="coerce")
    fit_end = pd.to_datetime(fit_end_raw, utc=True, errors="coerce")
    diag.update(
        {
            "source_model_fit_start": fit_start.isoformat()
            if pd.notna(fit_start)
            else None,
            "source_model_fit_end": fit_end.isoformat()
            if pd.notna(fit_end)
            else None,
            "source_slice_role": source_role or None,
            "generated_from_final_fit_bundle": bool(generated_from_final_fit),
            "slice_plan_sha256": slice_plan_sha256 or None,
            "feature_contract_hash": feature_contract_hash or None,
        }
    )

    if pd.isna(fit_start):
        diag["errors"].append("missing_or_invalid_source_model_fit_start")
    if pd.isna(fit_end):
        diag["errors"].append("missing_or_invalid_source_model_fit_end")
    if source_role not in allowed_roles:
        diag["errors"].append("unexpected_source_slice_role")
    if generated_from_final_fit:
        diag["errors"].append("generated_from_final_fit_bundle")
    if not slice_plan_sha256:
        diag["errors"].append("missing_slice_plan_sha256")
    elif slice_plan_sha256 != current_slice_plan_sha256:
        diag["errors"].append("slice_plan_sha256_mismatch")
    if not feature_contract_hash:
        diag["errors"].append("missing_feature_contract_hash")
    if pd.notna(fit_end):
        if fit_end > policy_fit_end:
            diag["errors"].append("source_model_fit_end_after_policy_fit_end")
        if fit_end >= policy_predict_start and not row_disjoint_policy_oos:
            diag["errors"].append("source_model_fit_end_not_before_policy_predict_start")

    diag["valid"] = not diag["errors"]
    return diag


def validate_policy_oos_source_artifacts(
    *,
    run_root: Path,
    slice_plan_path: Path,
    source_validation: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate that current base/meta artifacts are safe policy-OOS scorers."""

    policy_fit_end = pd.to_datetime(
        source_validation.get("policy_optimiser_fit_end"), utc=True, errors="coerce"
    )
    policy_predict_start = pd.to_datetime(
        source_validation.get("policy_optimiser_predict_start"),
        utc=True,
        errors="coerce",
    )
    report: Dict[str, Any] = {
        "schema_version": "policy_oos_source_artifact_preflight_v1",
        "run_root": str(run_root),
        "slice_plan_path": str(slice_plan_path),
        "valid": False,
        "errors": [],
        "source_validation": dict(source_validation),
        "artifacts": {},
    }
    if not slice_plan_path.exists():
        report["errors"].append("missing_slice_plan")
        return report
    if pd.isna(policy_fit_end) or pd.isna(policy_predict_start):
        report["errors"].append("missing_policy_fit_or_predict_window")
        return report
    row_disjoint_policy_oos = bool(
        source_validation.get("policy_holdout_fit_predict_disjoint", False)
    )
    temporal_policy_oos = bool(
        source_validation.get("policy_holdout_temporal_disjoint", False)
    )
    if policy_fit_end >= policy_predict_start and not row_disjoint_policy_oos:
        report["errors"].append("policy_fit_end_not_before_predict_start")
        return report
    if not temporal_policy_oos and not row_disjoint_policy_oos:
        report["errors"].append("policy_slice_not_temporal_or_row_disjoint")
        return report

    slice_plan_sha256 = sha256_file(slice_plan_path)
    report["slice_plan_sha256"] = slice_plan_sha256
    report["policy_fit_end"] = policy_fit_end.isoformat()
    report["policy_predict_start"] = policy_predict_start.isoformat()

    checks = {
        "base": (
            run_root / "base_models_intermediate.pkl",
            BASE_ALLOWED_ROLES,
        ),
        "meta": (
            run_root / "models" / "model_state_meta.pkl",
            META_ALLOWED_ROLES,
        ),
    }
    for name, (artifact_path, allowed_roles) in checks.items():
        report["artifacts"][name] = _validate_one_artifact(
            name=name,
            artifact_path=artifact_path,
            allowed_roles=allowed_roles,
            current_slice_plan_sha256=slice_plan_sha256,
            policy_fit_end=policy_fit_end,
            policy_predict_start=policy_predict_start,
            row_disjoint_policy_oos=row_disjoint_policy_oos,
        )

    for name, diag in report["artifacts"].items():
        if not bool(diag.get("valid")):
            report["errors"].append(f"{name}_artifact_not_policy_oos_safe")
    fit_ends = [
        pd.to_datetime(diag.get("source_model_fit_end"), utc=True, errors="coerce")
        for diag in report["artifacts"].values()
        if diag.get("source_model_fit_end")
    ]
    if fit_ends:
        report["source_model_fit_end"] = max(fit_ends).isoformat()
    report["valid"] = not report["errors"]
    return report


def write_policy_oos_preflight_report(report: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def write_source_artifact_provenance_manifest(
    *,
    artifact_path: Path,
    run_root: Path,
    slice_plan_path: Path,
    source_slice_role: str,
    source_model_fit_start: Any,
    source_model_fit_end: Any,
    feature_contract_hash: str,
    generated_from_final_fit_bundle: bool,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write the sidecar consumed by policy-OOS preflight checks."""

    fit_start = pd.to_datetime(source_model_fit_start, utc=True, errors="coerce")
    fit_end = pd.to_datetime(source_model_fit_end, utc=True, errors="coerce")
    payload: Dict[str, Any] = {
        "schema_version": "policy_oos_source_artifact_provenance_v1",
        "artifact_path": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path) if artifact_path.exists() else None,
        "run_root": str(run_root),
        "slice_plan_path": str(slice_plan_path),
        "slice_plan_sha256": sha256_file(slice_plan_path)
        if slice_plan_path.exists()
        else None,
        "source_slice_role": str(source_slice_role),
        "source_model_fit_start": fit_start.isoformat()
        if pd.notna(fit_start)
        else None,
        "source_model_fit_end": fit_end.isoformat() if pd.notna(fit_end) else None,
        "feature_contract_hash": str(feature_contract_hash or ""),
        "generated_from_final_fit_bundle": bool(generated_from_final_fit_bundle),
    }
    if extra:
        payload.update(dict(extra))
    path = provenance_manifest_path(artifact_path)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def parquet_timestamp_bounds(paths: Iterable[Path], *, column: str = "timestamp") -> tuple[Any, Any]:
    """Return min/max timestamp across parquet handoff files.

    Empty/missing files return ``(None, None)`` so callers can fall back to
    stage-level metadata.
    """

    mins: list[pd.Timestamp] = []
    maxs: list[pd.Timestamp] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path, columns=[column])
        except Exception:
            continue
        if column not in df.columns or df.empty:
            continue
        ts = pd.to_datetime(df[column], utc=True, errors="coerce").dropna()
        if ts.empty:
            continue
        mins.append(ts.min())
        maxs.append(ts.max())
    if not mins or not maxs:
        return None, None
    return min(mins), max(maxs)
