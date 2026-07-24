"""Immutable post-fit evidence for the canonical side-local Pack-B stages.

This module validates the small JSON manifests emitted after the independent
``long`` and ``short`` AE/GMM, feature-selection, and HPO stages.  It never
fits a model or reads label data.  Its job is deliberately narrower: bind each
learned artifact to the frozen pre-March source evidence that the runner
already materialized, and fail closed when that evidence is incomplete,
post-cutoff, pooled, or changed on disk.

The manifest is an audit record, not a trust-me API.  Callers pass manifest
paths to the validator; timing, source, scope, and artifact hashes are read
from those files and the artifact hash is recomputed from disk.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

SIDE_STAGE_MANIFEST_SCHEMA = "packb_side_stage_manifest_v1"
SIDE_STAGE_BUNDLE_SCHEMA = "packb_side_stage_manifest_bundle_v1"
CANONICAL_SIDES = ("long", "short")
CANONICAL_STAGES = ("ae_gmm", "feature_selection", "hpo")
DEFAULT_RESOLUTION_CUTOFF_UTC = pd.Timestamp("2026-03-01T00:00:00Z")
DECISION_LAG = pd.Timedelta(hours=1)
LABEL_RESOLUTION_HORIZON = pd.Timedelta(hours=24)
ACTUAL_LABEL_RESOLUTION_CONTRACT = (
    "__decision_ts__ must equal __ts__ + 1h; authorized only when "
    "__decision_ts__ + 24h < resolution_cutoff_utc"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SOURCE_HASH_FIELDS = (
    "dec09_decisions_sha256",
    "canonical_shard_inventory_sha256",
    "causal_audit_sha256",
    "population_preflight_sha256",
    "authorized_population_ledger_sha256",
    "feature_store_inventory_sha256",
    "feature_store_inventory_evidence_sha256",
)
_TIMING_FIELDS = (
    "signal_min_utc",
    "signal_max_utc",
    "decision_min_utc",
    "decision_max_utc",
    "label_resolution_min_utc",
    "label_resolution_max_utc",
)
_ARTIFACT_KIND_BY_STAGE = {
    "ae_gmm": "ae_gmm_state",
    "feature_selection": "feature_contract",
    "hpo": "parameter",
}


class PackBSideStageManifestError(ValueError):
    """Raised when post-fit Pack-B stage evidence is incomplete or invalid."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 for a file without reading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def canonical_json_sha256(value: Any) -> str:
    """Hash JSON evidence in a stable, representation-independent form."""

    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _require_sha256(value: Any, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if _SHA256_RE.fullmatch(normalized) is None:
        raise PackBSideStageManifestError(
            f"{name} must be a 64-character lowercase SHA-256 digest"
        )
    return normalized


def _require_source_revision(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if _SOURCE_REVISION_RE.fullmatch(normalized) is None:
        raise PackBSideStageManifestError(
            "source_revision must be a 40-character lowercase Git commit SHA"
        )
    return normalized


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PackBSideStageManifestError(f"{name} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], *, name: str, keys: tuple[str, ...]
) -> None:
    actual = set(value)
    expected = set(keys)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing=" + ", ".join(missing))
        if extra:
            details.append("unexpected=" + ", ".join(extra))
        raise PackBSideStageManifestError(
            f"{name} keys are invalid: {'; '.join(details)}"
        )


def _utc(value: Any, *, name: str) -> pd.Timestamp:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise PackBSideStageManifestError(f"{name} must be a non-empty UTC timestamp")
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise PackBSideStageManifestError(f"{name} is not a timestamp") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise PackBSideStageManifestError(f"{name} must include an explicit UTC offset")
    return timestamp.tz_convert("UTC")


def _iso(timestamp: pd.Timestamp) -> str:
    return timestamp.isoformat()


def _resolve_artifact_path(manifest_path: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PackBSideStageManifestError("artifact.path must be a non-empty path")
    declared = Path(value)
    path = declared if declared.is_absolute() else manifest_path.parent / declared
    if not path.is_file():
        raise PackBSideStageManifestError(f"artifact path does not exist: {path}")
    return path


def _validate_bound_file(
    value: Any,
    *,
    manifest_path: Path,
    name: str,
) -> dict[str, str]:
    bound = _require_mapping(value, name=name)
    _require_exact_keys(bound, name=name, keys=("path", "sha256"))
    path = _resolve_artifact_path(manifest_path, bound["path"])
    declared = _require_sha256(bound["sha256"], name=f"{name}.sha256")
    actual = sha256_file(path)
    if declared != actual:
        raise PackBSideStageManifestError(
            f"{name} SHA-256 does not match its source file"
        )
    return {"path": str(path), "sha256": actual}


def _validate_source_hashes(
    value: Any, *, expected: Mapping[str, str] | None
) -> dict[str, str]:
    source_hashes = _require_mapping(value, name="source_hashes")
    _require_exact_keys(
        source_hashes,
        name="source_hashes",
        keys=_SOURCE_HASH_FIELDS,
    )
    normalized = {
        name: _require_sha256(source_hashes[name], name=f"source_hashes.{name}")
        for name in _SOURCE_HASH_FIELDS
    }
    if expected is not None:
        expected_normalized = _normalize_expected_source_hashes(expected)
        mismatch = [
            name
            for name in _SOURCE_HASH_FIELDS
            if normalized[name] != expected_normalized[name]
        ]
        if mismatch:
            raise PackBSideStageManifestError(
                "manifest source hashes do not match the authorized preflight: "
                + ", ".join(mismatch)
            )
    return normalized


def _normalize_expected_source_hashes(value: Mapping[str, str]) -> dict[str, str]:
    _require_exact_keys(value, name="expected_source_hashes", keys=_SOURCE_HASH_FIELDS)
    return {
        name: _require_sha256(value[name], name=f"expected_source_hashes.{name}")
        for name in _SOURCE_HASH_FIELDS
    }


def _validate_candidate_stream(
    value: Any,
    *,
    manifest_path: Path,
    population_ledger: Mapping[str, str],
    side: str,
    cutoff: pd.Timestamp,
) -> dict[str, Any]:
    stream = _require_mapping(value, name="candidate_stream")
    required = ("path", "count", "sha256", *_TIMING_FIELDS)
    _require_exact_keys(stream, name="candidate_stream", keys=required)
    declared_count = stream["count"]
    if (
        isinstance(declared_count, bool)
        or not isinstance(declared_count, int)
        or declared_count < 1
    ):
        raise PackBSideStageManifestError(
            "candidate_stream.count must be a positive integer"
        )
    candidate_file = _validate_bound_file(
        {"path": stream["path"], "sha256": stream["sha256"]},
        manifest_path=manifest_path,
        name="candidate_stream",
    )
    try:
        import duckdb

        candidate_path = candidate_file["path"]
        population_path = str(population_ledger["path"])
        summary = duckdb.execute(
            """
            SELECT
                count(*)::BIGINT,
                count(DISTINCT candidate_id)::BIGINT,
                count_if(side_name <> ?)::BIGINT,
                count_if(__decision_ts__ <> __ts__ + INTERVAL '1 hour')::BIGINT,
                count_if(
                    __label_resolution_ts__
                    <> __decision_ts__ + INTERVAL '24 hours'
                )::BIGINT,
                count_if(__label_resolution_ts__ >= ?::TIMESTAMPTZ)::BIGINT,
                min(__ts__), max(__ts__),
                min(__decision_ts__), max(__decision_ts__),
                min(__label_resolution_ts__), max(__label_resolution_ts__)
            FROM read_parquet(?)
            """,
            [side, cutoff.isoformat(), candidate_path],
        ).fetchone()
        if summary is None:  # pragma: no cover - aggregate always returns one row.
            raise PackBSideStageManifestError("candidate stream summary is empty")
        (
            actual_count,
            distinct_count,
            wrong_side,
            bad_decision,
            bad_resolution,
            post_cutoff,
            *timing_values,
        ) = summary
        if int(actual_count) < 1:
            raise PackBSideStageManifestError("candidate stream is empty")
        if int(distinct_count) != int(actual_count):
            raise PackBSideStageManifestError(
                "candidate stream has duplicate candidate_id"
            )
        if int(wrong_side):
            raise PackBSideStageManifestError(
                f"candidate stream contains rows outside side {side!r}"
            )
        if int(bad_decision):
            raise PackBSideStageManifestError(
                "candidate stream contains decision timestamps not equal to signal + 1h"
            )
        if int(bad_resolution):
            raise PackBSideStageManifestError(
                "candidate stream contains label resolutions not equal to decision + 24h"
            )
        if int(post_cutoff):
            raise PackBSideStageManifestError(
                "candidate stream contains a label resolved at/after the pre-March cutoff"
            )
        missing_population = duckdb.execute(
            """
            SELECT count(*)::BIGINT
            FROM read_parquet(?) AS candidate
            LEFT JOIN read_parquet(?) AS population
            USING (
                candidate_id,
                side_name,
                __ts__,
                __decision_ts__,
                __label_resolution_ts__
            )
            WHERE population.candidate_id IS NULL
            """,
            [candidate_path, population_path],
        ).fetchone()
    except PackBSideStageManifestError:
        raise
    except Exception as exc:
        raise PackBSideStageManifestError(
            f"cannot validate candidate evidence ledger: {exc}"
        ) from exc
    if missing_population is None or int(missing_population[0]):
        raise PackBSideStageManifestError(
            "candidate stream contains rows absent from the authorized population ledger"
        )
    actual_timing = {
        name: _utc(value, name=f"candidate_stream actual {name}")
        for name, value in zip(_TIMING_FIELDS, timing_values, strict=True)
    }
    declared_timing = {
        name: _utc(stream[name], name=f"candidate_stream.{name}")
        for name in _TIMING_FIELDS
    }
    if int(actual_count) != declared_count or actual_timing != declared_timing:
        raise PackBSideStageManifestError(
            "candidate stream declared count/timing does not match its evidence file"
        )
    return {
        "path": candidate_file["path"],
        "count": int(actual_count),
        "sha256": candidate_file["sha256"],
        **{name: _iso(actual_timing[name]) for name in _TIMING_FIELDS},
    }


def _validate_artifact(
    value: Any, *, manifest_path: Path, side: str, stage: str
) -> dict[str, str]:
    artifact = _require_mapping(value, name="artifact")
    _require_exact_keys(
        artifact,
        name="artifact",
        keys=("kind", "path", "sha256", "scope"),
    )
    expected_kind = _ARTIFACT_KIND_BY_STAGE[stage]
    if artifact["kind"] != expected_kind:
        raise PackBSideStageManifestError(
            f"{stage} manifest artifact.kind must equal {expected_kind!r}"
        )
    scope = str(artifact["scope"] or "").strip().lower()
    if scope != side:
        raise PackBSideStageManifestError(
            f"{stage} artifact scope must explicitly equal {side!r}, got {scope!r}"
        )
    path = _resolve_artifact_path(manifest_path, artifact["path"])
    declared_hash = _require_sha256(artifact["sha256"], name="artifact.sha256")
    actual_hash = sha256_file(path)
    if actual_hash != declared_hash:
        raise PackBSideStageManifestError(
            f"{stage} artifact SHA-256 does not match its source file"
        )
    return {
        "kind": expected_kind,
        "path": str(path),
        "sha256": actual_hash,
        "scope": scope,
    }


def _read_manifest(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise PackBSideStageManifestError(f"side-stage manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackBSideStageManifestError(
            f"cannot read side-stage manifest: {path}"
        ) from exc
    return _require_mapping(payload, name="side-stage manifest")


def validate_side_stage_manifest(
    manifest_path: Path,
    *,
    expected_side: str | None = None,
    expected_stage: str | None = None,
    expected_source_revision: str | None = None,
    expected_source_hashes: Mapping[str, str] | None = None,
    expected_fixed_calendar_sha256: str | None = None,
) -> dict[str, Any]:
    """Read and fail-closed validate one immutable post-fit stage manifest.

    ``expected_*`` values are optional binding checks.  Importantly, label
    timing and artifact scope are not function arguments: both are taken from
    the manifest and validated against the locked DEC-09 contract.
    """

    path = Path(manifest_path)
    manifest = _read_manifest(path)
    _require_exact_keys(
        manifest,
        name="side-stage manifest",
        keys=(
            "schema",
            "source_revision",
            "side",
            "stage",
            "resolution_cutoff_utc",
            "actual_label_resolution_contract",
            "source_hashes",
            "authorized_population_ledger",
            "candidate_stream",
            "fixed_calendar_sha256",
            "stage_config",
            "artifact",
        ),
    )
    if manifest["schema"] != SIDE_STAGE_MANIFEST_SCHEMA:
        raise PackBSideStageManifestError(
            f"unexpected side-stage manifest schema: {manifest['schema']!r}"
        )
    source_revision = _require_source_revision(manifest["source_revision"])
    if (
        expected_source_revision is not None
        and source_revision != _require_source_revision(expected_source_revision)
    ):
        raise PackBSideStageManifestError(
            "manifest source revision does not match the authorized runner revision"
        )
    side = str(manifest["side"] or "").strip().lower()
    if side not in CANONICAL_SIDES:
        raise PackBSideStageManifestError("manifest side must be exactly long or short")
    if expected_side is not None and side != str(expected_side).strip().lower():
        raise PackBSideStageManifestError(
            f"manifest side {side!r} does not match expected side {expected_side!r}"
        )
    stage = str(manifest["stage"] or "").strip().lower()
    if stage not in CANONICAL_STAGES:
        raise PackBSideStageManifestError(
            "manifest stage must be ae_gmm, feature_selection, or hpo"
        )
    if expected_stage is not None and stage != str(expected_stage).strip().lower():
        raise PackBSideStageManifestError(
            f"manifest stage {stage!r} does not match expected stage {expected_stage!r}"
        )
    cutoff = _utc(manifest["resolution_cutoff_utc"], name="resolution_cutoff_utc")
    if cutoff != DEFAULT_RESOLUTION_CUTOFF_UTC:
        raise PackBSideStageManifestError(
            "side-stage manifest requires the locked 2026-03-01T00:00:00Z cutoff"
        )
    if manifest["actual_label_resolution_contract"] != ACTUAL_LABEL_RESOLUTION_CONTRACT:
        raise PackBSideStageManifestError(
            "side-stage manifest has an invalid actual label-resolution contract"
        )
    source_hashes = _validate_source_hashes(
        manifest["source_hashes"], expected=expected_source_hashes
    )
    population_ledger = _validate_bound_file(
        manifest["authorized_population_ledger"],
        manifest_path=path,
        name="authorized_population_ledger",
    )
    if (
        population_ledger["sha256"]
        != source_hashes["authorized_population_ledger_sha256"]
    ):
        raise PackBSideStageManifestError(
            "authorized population ledger hash does not match source_hashes"
        )
    fixed_calendar_sha256 = _require_sha256(
        manifest["fixed_calendar_sha256"], name="fixed_calendar_sha256"
    )
    stage_config = _validate_bound_file(
        manifest["stage_config"],
        manifest_path=path,
        name="stage_config",
    )
    if expected_fixed_calendar_sha256 is not None:
        expected_calendar = _require_sha256(
            expected_fixed_calendar_sha256,
            name="expected_fixed_calendar_sha256",
        )
        if fixed_calendar_sha256 != expected_calendar:
            raise PackBSideStageManifestError(
                "manifest fixed calendar hash does not match the locked calendar"
            )
    candidate_stream = _validate_candidate_stream(
        manifest["candidate_stream"],
        manifest_path=path,
        population_ledger=population_ledger,
        side=side,
        cutoff=cutoff,
    )
    artifact = _validate_artifact(
        manifest["artifact"], manifest_path=path, side=side, stage=stage
    )
    return {
        "schema": SIDE_STAGE_MANIFEST_SCHEMA,
        "manifest_path": str(path),
        "manifest_sha256": sha256_file(path),
        "source_revision": source_revision,
        "side": side,
        "stage": stage,
        "resolution_cutoff_utc": _iso(cutoff),
        "source_hashes": source_hashes,
        "authorized_population_ledger": population_ledger,
        "candidate_stream": candidate_stream,
        "fixed_calendar_sha256": fixed_calendar_sha256,
        "stage_config": stage_config,
        "artifact": artifact,
    }


def validate_side_stage_manifest_bundle(
    manifest_paths: Mapping[str, Mapping[str, Path]],
    *,
    expected_source_revision: str | None = None,
    expected_source_hashes: Mapping[str, str] | None = None,
    expected_fixed_calendar_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate all six side/stage manifests and their cross-side separation."""

    if set(manifest_paths) != set(CANONICAL_SIDES):
        raise PackBSideStageManifestError(
            "manifest_paths must contain exactly long and short"
        )
    results: dict[str, dict[str, Any]] = {side: {} for side in CANONICAL_SIDES}
    discovered_sources: dict[str, str] | None = None
    discovered_calendar: str | None = None
    discovered_revision: str | None = None
    for side in CANONICAL_SIDES:
        per_stage = _require_mapping(
            manifest_paths[side], name=f"{side} manifest_paths"
        )
        _require_exact_keys(
            per_stage, name=f"{side} manifest_paths", keys=CANONICAL_STAGES
        )
        for stage in CANONICAL_STAGES:
            result = validate_side_stage_manifest(
                Path(per_stage[stage]),
                expected_side=side,
                expected_stage=stage,
                expected_source_revision=expected_source_revision,
                expected_source_hashes=expected_source_hashes,
                expected_fixed_calendar_sha256=expected_fixed_calendar_sha256,
            )
            if discovered_sources is None:
                discovered_sources = dict(result["source_hashes"])
            elif result["source_hashes"] != discovered_sources:
                raise PackBSideStageManifestError(
                    "all side-stage manifests must bind to identical source hashes"
                )
            if discovered_calendar is None:
                discovered_calendar = str(result["fixed_calendar_sha256"])
            elif result["fixed_calendar_sha256"] != discovered_calendar:
                raise PackBSideStageManifestError(
                    "all side-stage manifests must bind to the same fixed calendar hash"
                )
            if discovered_revision is None:
                discovered_revision = str(result["source_revision"])
            elif result["source_revision"] != discovered_revision:
                raise PackBSideStageManifestError(
                    "all side-stage manifests must bind to the same source revision"
                )
            results[side][stage] = result

    for stage in CANONICAL_STAGES:
        long_stream_hash = results["long"][stage]["candidate_stream"]["sha256"]
        short_stream_hash = results["short"][stage]["candidate_stream"]["sha256"]
        if long_stream_hash == short_stream_hash:
            raise PackBSideStageManifestError(
                f"long and short must have distinct {stage} candidate stream SHA-256 values"
            )
        long_hash = results["long"][stage]["artifact"]["sha256"]
        short_hash = results["short"][stage]["artifact"]["sha256"]
        if long_hash == short_hash:
            raise PackBSideStageManifestError(
                f"long and short must have distinct learned {stage} artifact SHA-256 values"
            )
    if (
        discovered_sources is None
        or discovered_calendar is None
        or discovered_revision is None
    ):  # pragma: no cover
        raise PackBSideStageManifestError("side-stage bundle has no manifests")
    return {
        "schema": SIDE_STAGE_BUNDLE_SCHEMA,
        "status": "VALIDATED_POST_FIT_PRE_MARCH_SIDE_STAGES",
        "resolution_cutoff_utc": _iso(DEFAULT_RESOLUTION_CUTOFF_UTC),
        "source_revision": discovered_revision,
        "source_hashes": discovered_sources,
        "fixed_calendar_sha256": discovered_calendar,
        "by_side": results,
    }


def write_immutable_side_stage_manifest(path: Path, manifest: Mapping[str, Any]) -> str:
    """Validate then create one manifest exactly once, returning its file hash.

    The exclusive create prevents a successful stage from silently overwriting
    its provenance.  Artifact paths are resolved relative to ``path`` while
    validating, so a file cannot be written until its bound artifact exists.
    """

    destination = Path(path)
    if destination.exists():
        raise PackBSideStageManifestError(
            f"refusing to overwrite immutable side-stage manifest: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    try:
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    except FileExistsError as exc:  # pragma: no cover - race guard.
        raise PackBSideStageManifestError(
            f"refusing to overwrite immutable side-stage manifest: {destination}"
        ) from exc
    try:
        validate_side_stage_manifest(destination)
    except Exception:
        # The manifest exists by design even after a failed validation; removing
        # it would make a failed evidence attempt disappear.  Callers must use a
        # distinct run directory for a corrected post-fit attempt.
        raise
    return sha256_file(destination)
