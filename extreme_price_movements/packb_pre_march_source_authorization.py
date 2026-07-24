"""Fail-closed source authorization for the canonical DEC-09 Pack-B recovery.

This module is deliberately independent of a model runner.  It proves the
inputs that may be used for the pre-March feature-selection/HPO cycle before a
runner opens a training matrix:

* the label directory is exactly the causal-audit shard inventory;
* candidate identities are globally unique, checked batch-by-batch;
* label timing is read from the source decision timestamp, not synthesized;
* the authorised pre-March population resolves strictly before the cutoff; and
* long and short use separately scoped, separately hashed learned artifacts.

The audit is streaming.  It creates only a short-lived SQLite index of
candidate IDs so a multi-million-row ledger never has to be loaded into RAM.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

AUTHORIZATION_SCHEMA = "packb_pre_march_source_authorization_v1"
POPULATION_PREFLIGHT_SCHEMA = "packb_pre_march_population_preflight_v1"
CANONICAL_SIDES = ("long", "short")
DECISION_LAG = pd.Timedelta(hours=1)
BASE_LABEL_RESOLUTION_HORIZON = pd.Timedelta(hours=24)
DEFAULT_RESOLUTION_CUTOFF_UTC = pd.Timestamp("2026-03-01T00:00:00Z")
DEFAULT_BATCH_ROWS = 65_536
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_LABEL_COLUMNS = ("candidate_id", "__ts__", "__decision_ts__", "side_name")


class PackBSourceAuthorizationError(ValueError):
    """Raised when a Pack-B source cannot be authorised for canonical use."""


@dataclass(frozen=True)
class SideSourceAuthorization:
    """The three fitted artifacts and their pre-cutoff evidence for one side.

    The declared hashes are checked against the supplied files.  Each
    ``*_scope`` must be the literal canonical side (``long`` or ``short``),
    rather than a pooled/per-side assertion.  The three timestamps are the
    maximum *actual label-resolution* timestamp used while fitting the stated
    stage, recorded by the source selection/HPO run.
    """

    ae_gmm_state_path: Path
    ae_gmm_state_sha256: str
    ae_gmm_state_scope: str
    ae_gmm_reference_label_resolution_max_utc: Any
    feature_contract_path: Path
    feature_contract_sha256: str
    feature_contract_scope: str
    feature_selection_label_resolution_max_utc: Any
    parameter_path: Path
    parameter_sha256: str
    parameter_scope: str
    hpo_label_resolution_max_utc: Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _utc(value: Any, *, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    if pd.isna(timestamp):
        raise PackBSourceAuthorizationError(f"{name} is not a valid UTC timestamp")
    return timestamp


def _validate_sha256(value: str, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if _SHA256_RE.fullmatch(normalized) is None:
        raise PackBSourceAuthorizationError(
            f"{name} must be a 64-character SHA-256 digest"
        )
    return normalized


def _load_canonical_shards(
    labels_dir: Path, causal_audit_path: Path
) -> tuple[list[Path], dict[str, Any]]:
    """Resolve one exact shard inventory and reject a dirty label directory."""

    if not labels_dir.is_dir():
        raise PackBSourceAuthorizationError(
            f"labels directory does not exist: {labels_dir}"
        )
    if not causal_audit_path.is_file():
        raise PackBSourceAuthorizationError(
            f"causal audit does not exist: {causal_audit_path}"
        )
    try:
        audit = json.loads(causal_audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackBSourceAuthorizationError(
            f"cannot read causal audit: {causal_audit_path}"
        ) from exc

    per_file = audit.get("per_file")
    if not isinstance(per_file, list) or not per_file:
        raise PackBSourceAuthorizationError(
            "causal audit has no per_file canonical shard inventory"
        )
    raw_names: list[str] = []
    for item in per_file:
        if not isinstance(item, Mapping) or not isinstance(item.get("file"), str):
            raise PackBSourceAuthorizationError(
                "causal audit per_file entries must name a shard"
            )
        name = str(item["file"])
        if Path(name).name != name or not name.endswith(".parquet"):
            raise PackBSourceAuthorizationError(
                f"unsafe or non-Parquet audit shard name: {name!r}"
            )
        raw_names.append(name)
    expected = set(raw_names)
    if len(expected) != len(raw_names):
        raise PackBSourceAuthorizationError(
            "causal audit contains a duplicate canonical shard name"
        )
    is_current_audit = (
        audit.get("schema") == "packb_current_canonical_label_inventory_audit_v1"
    )
    if is_current_audit:
        if audit.get("status") != "PASS" or audit.get("mode") != "streaming_full_audit":
            raise PackBSourceAuthorizationError(
                "current canonical label audit must be a passing full streaming audit"
            )
        inventory = audit.get("inventory")
        if not isinstance(inventory, Mapping):
            raise PackBSourceAuthorizationError(
                "current canonical label audit has no inventory"
            )
        declared_raw = inventory.get("canonical_monthly_files")
        exclusions_raw = inventory.get("excluded_unlisted_monolithic_files", [])
        if not isinstance(exclusions_raw, list) or any(
            not isinstance(value, str) for value in exclusions_raw
        ):
            raise PackBSourceAuthorizationError(
                "current canonical label audit exclusions are invalid"
            )
        allowed_exclusions = set(exclusions_raw)
    else:
        declared_raw = audit.get("files")
        allowed_exclusions = set()
    try:
        declared_count = int(declared_raw)
    except (TypeError, ValueError) as exc:
        raise PackBSourceAuthorizationError(
            "causal audit files count is invalid"
        ) from exc
    if declared_count != len(expected):
        raise PackBSourceAuthorizationError(
            "causal audit files count does not match its per_file inventory"
        )

    actual = {path.name for path in labels_dir.glob("*.parquet") if path.is_file()}
    missing = sorted(expected - actual)
    extras = sorted(actual - expected)
    unexpected_extras = sorted(set(extras) - allowed_exclusions)
    missing_exclusions = sorted(allowed_exclusions - actual)
    if missing or unexpected_extras or missing_exclusions:
        details: list[str] = []
        if missing:
            details.append("missing canonical shards=" + ", ".join(missing[:8]))
        if unexpected_extras:
            details.append(
                "unlisted parquet shards=" + ", ".join(unexpected_extras[:8])
            )
        if missing_exclusions:
            details.append(
                "declared exclusions missing=" + ", ".join(missing_exclusions[:8])
            )
        raise PackBSourceAuthorizationError(
            "label shard inventory is not exact: " + "; ".join(details)
        )

    names = sorted(expected)
    return [labels_dir / name for name in names], {
        "causal_audit_path": str(causal_audit_path),
        "causal_audit_sha256": _sha256_file(causal_audit_path),
        "canonical_shard_count": len(names),
        "canonical_shard_inventory_sha256": _canonical_json_sha256({"shards": names}),
        "explicitly_excluded_shards": sorted(allowed_exclusions),
    }


def _validate_side_artifacts(
    side_sources: Mapping[str, SideSourceAuthorization], *, cutoff: pd.Timestamp
) -> dict[str, dict[str, Any]]:
    if set(side_sources) != set(CANONICAL_SIDES):
        raise PackBSourceAuthorizationError(
            "side_sources must contain exactly long and short"
        )

    artifact_fields = (
        (
            "ae_gmm_state",
            "ae_gmm_state_path",
            "ae_gmm_state_sha256",
            "ae_gmm_state_scope",
        ),
        (
            "feature_contract",
            "feature_contract_path",
            "feature_contract_sha256",
            "feature_contract_scope",
        ),
        ("parameter", "parameter_path", "parameter_sha256", "parameter_scope"),
    )
    timing_fields = (
        ("ae_gmm_reference", "ae_gmm_reference_label_resolution_max_utc"),
        ("feature_selection", "feature_selection_label_resolution_max_utc"),
        ("hpo", "hpo_label_resolution_max_utc"),
    )
    report: dict[str, dict[str, Any]] = {}
    for side in CANONICAL_SIDES:
        source = side_sources[side]
        current: dict[str, Any] = {
            "side": side,
            "artifacts": {},
            "stage_max_label_resolution_utc": {},
        }
        for artifact_name, path_name, digest_name, scope_name in artifact_fields:
            path = Path(getattr(source, path_name))
            if not path.is_file():
                raise PackBSourceAuthorizationError(
                    f"{side} {artifact_name} path does not exist: {path}"
                )
            declared = _validate_sha256(
                getattr(source, digest_name), name=f"{side} {artifact_name} sha256"
            )
            actual = _sha256_file(path)
            if declared != actual:
                raise PackBSourceAuthorizationError(
                    f"{side} {artifact_name} SHA-256 does not match its source file"
                )
            scope = str(getattr(source, scope_name) or "").strip().lower()
            if scope != side:
                raise PackBSourceAuthorizationError(
                    f"{side} {artifact_name} scope must explicitly equal {side!r}, got {scope!r}"
                )
            current["artifacts"][artifact_name] = {
                "path": str(path),
                "sha256": actual,
                "scope": scope,
            }
        for stage, field in timing_fields:
            maximum = _utc(
                getattr(source, field), name=f"{side} {stage} label-resolution maximum"
            )
            if maximum >= cutoff:
                raise PackBSourceAuthorizationError(
                    f"{side} {stage} uses a label resolved at/after the pre-March cutoff"
                )
            current["stage_max_label_resolution_utc"][stage] = maximum.isoformat()
        report[side] = current

    for artifact_name, _path_name, _digest_name, _scope_name in artifact_fields:
        long_hash = str(report["long"]["artifacts"][artifact_name]["sha256"])
        short_hash = str(report["short"]["artifacts"][artifact_name]["sha256"])
        if long_hash == short_hash:
            raise PackBSourceAuthorizationError(
                f"long and short must have distinct {artifact_name} SHA-256 values"
            )
    return report


def _open_candidate_index(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("CREATE TABLE candidate_ids (candidate_id TEXT PRIMARY KEY)")
    return connection


def _scan_labels(
    shards: Sequence[Path],
    *,
    cutoff: pd.Timestamp,
    batch_rows: int,
    checkpoint: Callable[[str], None] | None = None,
    scratch_dir: Path | None = None,
) -> dict[str, Any]:
    """Check IDs and timing in bounded Parquet batches.

    The temporary SQLite index contains only the candidate-ID key.  Feature and
    target columns never enter memory here.
    """

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - project test/runtime dependency.
        raise PackBSourceAuthorizationError(
            "pyarrow is required for source authorization"
        ) from exc

    if not isinstance(batch_rows, int) or batch_rows < 1:
        raise PackBSourceAuthorizationError("batch_rows must be a positive integer")
    eligible_rows = {side: 0 for side in CANONICAL_SIDES}
    excluded_rows = {side: 0 for side in CANONICAL_SIDES}
    eligible_min: dict[str, pd.Timestamp | None] = {
        side: None for side in CANONICAL_SIDES
    }
    eligible_max: dict[str, pd.Timestamp | None] = {
        side: None for side in CANONICAL_SIDES
    }
    stream_hash = {side: hashlib.sha256() for side in CANONICAL_SIDES}
    total_rows = 0

    with tempfile.TemporaryDirectory(
        prefix="packb-source-authorization-",
        dir=str(scratch_dir) if scratch_dir is not None else None,
    ) as temporary_directory:
        if checkpoint is not None:
            checkpoint("before_duplicate_index")
        index_path = Path(temporary_directory) / "candidate_ids.sqlite"
        connection = _open_candidate_index(index_path)
        try:
            for shard in shards:
                if checkpoint is not None:
                    checkpoint(f"before_label_shard:{shard.name}")
                parquet = pq.ParquetFile(shard)
                schema = set(parquet.schema.names)
                missing = sorted(set(_REQUIRED_LABEL_COLUMNS) - schema)
                if missing:
                    raise PackBSourceAuthorizationError(
                        f"canonical label shard misses required columns: {shard.name}: {missing}"
                    )
                for batch in parquet.iter_batches(
                    batch_size=batch_rows, columns=list(_REQUIRED_LABEL_COLUMNS)
                ):
                    if checkpoint is not None:
                        checkpoint(f"before_label_batch:{shard.name}")
                    frame = batch.to_pandas()
                    total_rows += len(frame)
                    ids = frame["candidate_id"].astype("string")
                    invalid_id = (
                        ids.isna() | ids.str.strip().eq("") | ids.ne(ids.str.strip())
                    )
                    if invalid_id.any():
                        raise PackBSourceAuthorizationError(
                            f"canonical label shard has null, blank, or whitespace-padded candidate_id: {shard.name}"
                        )
                    try:
                        connection.executemany(
                            "INSERT INTO candidate_ids(candidate_id) VALUES (?)",
                            ((str(candidate_id),) for candidate_id in ids.tolist()),
                        )
                    except sqlite3.IntegrityError as exc:
                        raise PackBSourceAuthorizationError(
                            f"duplicate candidate_id across canonical label shards: {shard.name}"
                        ) from exc

                    signal = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
                    decision = pd.to_datetime(
                        frame["__decision_ts__"], utc=True, errors="coerce"
                    )
                    if signal.isna().any() or decision.isna().any():
                        raise PackBSourceAuthorizationError(
                            f"canonical label shard has non-UTC/non-parseable decision timing: {shard.name}"
                        )
                    expected_decision = signal + DECISION_LAG
                    if not decision.eq(expected_decision).all():
                        raise PackBSourceAuthorizationError(
                            f"canonical label shard violates decision_timestamp = signal_timestamp + 1h: {shard.name}"
                        )
                    sides = frame["side_name"].astype("string").str.strip().str.lower()
                    invalid_side = sides.isna() | ~sides.isin(CANONICAL_SIDES)
                    if invalid_side.any():
                        raise PackBSourceAuthorizationError(
                            f"canonical label shard has a non-canonical side_name: {shard.name}"
                        )
                    resolution = decision + BASE_LABEL_RESOLUTION_HORIZON
                    eligible = resolution.lt(cutoff)
                    for side in CANONICAL_SIDES:
                        side_mask = sides.eq(side)
                        eligible_side = side_mask & eligible
                        excluded_side = side_mask & ~eligible
                        selected_resolution = resolution.loc[eligible_side]
                        selected_ids = ids.loc[eligible_side]
                        count = int(eligible_side.sum())
                        eligible_rows[side] += count
                        excluded_rows[side] += int(excluded_side.sum())
                        if count:
                            current_min = selected_resolution.min()
                            current_max = selected_resolution.max()
                            previous_min = eligible_min[side]
                            previous_max = eligible_max[side]
                            eligible_min[side] = (
                                current_min
                                if previous_min is None
                                else min(previous_min, current_min)
                            )
                            eligible_max[side] = (
                                current_max
                                if previous_max is None
                                else max(previous_max, current_max)
                            )
                            for candidate_id, label_resolution in zip(
                                selected_ids.tolist(),
                                selected_resolution.tolist(),
                                strict=True,
                            ):
                                stream_hash[side].update(shard.name.encode("utf-8"))
                                stream_hash[side].update(b"\x1f")
                                stream_hash[side].update(
                                    str(candidate_id).encode("utf-8")
                                )
                                stream_hash[side].update(b"\x1f")
                                stream_hash[side].update(
                                    pd.Timestamp(label_resolution)
                                    .isoformat()
                                    .encode("ascii")
                                )
                                stream_hash[side].update(b"\n")
            connection.commit()
        finally:
            connection.close()

    by_side: dict[str, dict[str, Any]] = {}
    for side in CANONICAL_SIDES:
        maximum = eligible_max[side]
        if eligible_rows[side] == 0 or maximum is None or eligible_min[side] is None:
            raise PackBSourceAuthorizationError(
                f"no pre-March authorized label rows for {side}"
            )
        if maximum >= cutoff:
            raise PackBSourceAuthorizationError(
                f"{side} authorized label resolution reaches the cutoff"
            )
        by_side[side] = {
            "authorized_rows": eligible_rows[side],
            "excluded_rows_at_or_after_cutoff": excluded_rows[side],
            "authorized_label_resolution_min_utc": eligible_min[side].isoformat(),
            "authorized_label_resolution_max_utc": maximum.isoformat(),
            "authorized_candidate_stream_sha256": stream_hash[side].hexdigest(),
        }
    return {"label_rows_scanned": total_rows, "authorized_population_by_side": by_side}


def _locked_resolution_cutoff(value: Any) -> pd.Timestamp:
    cutoff = _utc(value, name="pre-March resolution cutoff")
    if cutoff != DEFAULT_RESOLUTION_CUTOFF_UTC:
        raise PackBSourceAuthorizationError(
            "Pack-B canonical source authorization requires the locked "
            "2026-03-01T00:00:00Z cutoff"
        )
    return cutoff


def preflight_pre_march_packb_population(
    *,
    labels_dir: Path,
    causal_audit_path: Path,
    resolution_cutoff_utc: Any = DEFAULT_RESOLUTION_CUTOFF_UTC,
    batch_rows: int = DEFAULT_BATCH_ROWS,
    checkpoint: Callable[[str], None] | None = None,
    scratch_dir: Path | None = None,
) -> dict[str, Any]:
    """Authorize exact label populations before any learned artifact exists."""

    cutoff = _locked_resolution_cutoff(resolution_cutoff_utc)
    shards, inventory = _load_canonical_shards(
        Path(labels_dir),
        Path(causal_audit_path),
    )
    label_report = _scan_labels(
        shards,
        cutoff=cutoff,
        batch_rows=batch_rows,
        checkpoint=checkpoint,
        scratch_dir=Path(scratch_dir) if scratch_dir is not None else None,
    )
    return {
        "schema": POPULATION_PREFLIGHT_SCHEMA,
        "status": "AUTHORIZED_PRE_MARCH_POPULATION",
        "resolution_cutoff_utc": cutoff.isoformat(),
        "actual_label_resolution_contract": (
            "__decision_ts__ must equal __ts__ + 1h; authorized only when "
            "__decision_ts__ + 24h < resolution_cutoff_utc"
        ),
        "label_inventory": inventory,
        **label_report,
        "streaming_contract": {
            "batch_rows": batch_rows,
            "duplicate_id_index": "short_lived_sqlite_candidate_id_primary_key",
            "feature_or_target_columns_loaded": False,
        },
    }


def verify_pre_march_side_artifacts(
    *,
    side_sources: Mapping[str, SideSourceAuthorization],
    resolution_cutoff_utc: Any = DEFAULT_RESOLUTION_CUTOFF_UTC,
) -> dict[str, dict[str, Any]]:
    """Verify distinct side-local AE, feature, and HPO artifacts after fitting."""

    cutoff = _locked_resolution_cutoff(resolution_cutoff_utc)
    return _validate_side_artifacts(side_sources, cutoff=cutoff)


def authorize_pre_march_packb_sources(
    *,
    labels_dir: Path,
    causal_audit_path: Path,
    side_sources: Mapping[str, SideSourceAuthorization],
    resolution_cutoff_utc: Any = DEFAULT_RESOLUTION_CUTOFF_UTC,
    batch_rows: int = DEFAULT_BATCH_ROWS,
    checkpoint: Callable[[str], None] | None = None,
    scratch_dir: Path | None = None,
) -> dict[str, Any]:
    """Return a compact, fail-closed authorization report for Pack-B sources.

    Canonical runners must use the report's strict label predicate
    ``actual_decision_timestamp + 24h < resolution_cutoff_utc`` for every
    feature-selection, AE/GMM reference, and HPO input row.  The report does
    not authorise model fitting by itself; it authorises the fixed sources that
    such a fit may consume.
    """

    population = preflight_pre_march_packb_population(
        labels_dir=labels_dir,
        causal_audit_path=causal_audit_path,
        resolution_cutoff_utc=resolution_cutoff_utc,
        batch_rows=batch_rows,
        checkpoint=checkpoint,
        scratch_dir=scratch_dir,
    )
    side_report = verify_pre_march_side_artifacts(
        side_sources=side_sources,
        resolution_cutoff_utc=resolution_cutoff_utc,
    )
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "status": "AUTHORIZED_PRE_MARCH_SOURCES",
        "resolution_cutoff_utc": population["resolution_cutoff_utc"],
        "actual_label_resolution_contract": population[
            "actual_label_resolution_contract"
        ],
        "label_inventory": population["label_inventory"],
        "label_rows_scanned": population["label_rows_scanned"],
        "authorized_population_by_side": population["authorized_population_by_side"],
        "side_source_artifacts": side_report,
        "streaming_contract": population["streaming_contract"],
    }
