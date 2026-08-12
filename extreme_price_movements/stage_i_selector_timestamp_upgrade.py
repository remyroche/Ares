"""Immutable upgrade of a completed Stage-I selector to explicit decision time.

The store-backed selector matrix is expensive to materialise, but the timing
repair changes only ledger lineage: ``__ts__`` remains the signal-close
identity and ``decision_ts`` is exactly one hour later.  This helper copies a
completed artifact into a new destination, verifies every copied payload, and
rewrites only the ledger and manifest.  The source artifact is never mutated.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import pandas as pd
import pyarrow.parquet as pq

from .stage_i_timestamp_contract import (
    DECISION_COLUMN,
    attach_stage_i_decision_timestamp,
    resolve_stage_i_timestamp_contract,
)
from .stage_i_r3_contract import r3_label_economics_contract


SCHEMA = "stage_i_selector_timestamp_upgrade_v1"
_REQUIRED_PAYLOADS = (
    "selector_features.parquet",
    "selector_feature_contract.json",
    "selector_exact_feature_coverage_audit.parquet",
    "selector_exact_feature_month_side_coverage.parquet",
    "population_summary.parquet",
)


class StageISelectorTimestampUpgradeError(ValueError):
    """Raised when a selector cannot be upgraded without contract drift."""


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StageISelectorTimestampUpgradeError(f"JSON object expected: {path}")
    return value


def _identity_digest(frame: pd.DataFrame) -> str:
    required = ["candidate_id", "__ts__", "__symbol__"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise StageISelectorTimestampUpgradeError(
            f"selector ledger lacks immutable identity fields: {missing}"
        )
    return sha256(
        pd.util.hash_pandas_object(frame.loc[:, required], index=False)
        .to_numpy(dtype="uint64")
        .tobytes()
    ).hexdigest()


def _artifact_integrity(root: Path, ledger: pd.DataFrame) -> dict[str, Any] | None:
    """Rebuild selector integrity when the source has the new frozen contract."""

    manifest = _json(root / "manifest.json")
    prior = manifest.get("artifact_integrity")
    if prior is None:
        return None
    if not isinstance(prior, Mapping) or prior.get("schema") != "stage_i_selector_artifact_integrity_v1":
        raise StageISelectorTimestampUpgradeError("selector artifact-integrity contract is malformed")
    coverage = root / "selector_exact_feature_coverage_audit.parquet"
    detail = root / "selector_exact_feature_month_side_coverage.parquet"
    required = {
        "selector_ledger_sha256": root / "selector_ledger.parquet",
        "selector_features_sha256": root / "selector_features.parquet",
        "exact_coverage_audit_sha256": coverage,
    }
    for key, path in required.items():
        if not path.is_file() or prior.get(key) != _file_sha256(path):
            raise StageISelectorTimestampUpgradeError(f"selector artifact integrity drift: {key}")
    detail_sha = _file_sha256(detail) if detail.is_file() else None
    if prior.get("exact_coverage_month_side_audit_sha256") != detail_sha:
        raise StageISelectorTimestampUpgradeError("selector artifact integrity drift: detail coverage")
    contract = r3_label_economics_contract(ledger)
    if prior.get("r3_label_economics_contract") != contract or prior.get("r3_label_economics_contract_sha256") != contract["contract_sha256"]:
        raise StageISelectorTimestampUpgradeError("selector artifact integrity drift: R3 label/economics")
    return dict(prior)


def _verify_completed_source(source: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest_path = source / "manifest.json"
    ledger_path = source / "selector_ledger.parquet"
    if not manifest_path.is_file() or not ledger_path.is_file():
        raise StageISelectorTimestampUpgradeError(
            f"completed selector manifest/ledger missing under {source}"
        )
    manifest = _json(manifest_path)
    if (
        manifest.get("schema") != "stage_i_selector_sample_v1"
        or manifest.get("status") != "complete"
    ):
        raise StageISelectorTimestampUpgradeError(
            f"source is not a completed Stage-I selector sample: {source}"
        )
    missing = [name for name in _REQUIRED_PAYLOADS if not (source / name).is_file()]
    if missing:
        raise StageISelectorTimestampUpgradeError(
            f"completed selector payloads are missing: {missing}"
        )
    ledger = pd.read_parquet(ledger_path)
    if len(ledger) != int(manifest.get("rows", -1)):
        raise StageISelectorTimestampUpgradeError(
            "selector ledger row count differs from its completed manifest"
        )
    feature_rows = int(
        pq.ParquetFile(source / "selector_features.parquet").metadata.num_rows
    )
    if feature_rows != len(ledger):
        raise StageISelectorTimestampUpgradeError(
            "selector feature matrix and ledger are not row aligned"
        )
    _artifact_integrity(source, ledger)
    return manifest, ledger


def _upgrade_manifest(
    source: Path,
    source_manifest: Mapping[str, Any],
    source_ledger: pd.DataFrame,
    upgraded_ledger: pd.DataFrame,
) -> dict[str, Any]:
    timing = resolve_stage_i_timestamp_contract(upgraded_ledger)
    source_identity = _identity_digest(source_ledger)
    upgraded_identity = _identity_digest(upgraded_ledger)
    if source_identity != upgraded_identity:
        raise StageISelectorTimestampUpgradeError(
            "timestamp upgrade changed immutable selector identity"
        )
    value = dict(source_manifest)
    value["timestamp_contract"] = dict(timing.audit)
    value["timestamp_upgrade"] = {
        "schema": SCHEMA,
        "source_path": str(source.resolve()),
        "source_manifest_sha256": _file_sha256(source / "manifest.json"),
        "source_ledger_sha256": _file_sha256(source / "selector_ledger.parquet"),
        "source_identity_sha256": source_identity,
        "upgraded_identity_sha256": upgraded_identity,
        "source_feature_payload_sha256": {
            name: _file_sha256(source / name) for name in _REQUIRED_PAYLOADS
        },
        "source_artifact_preserved": True,
        "feature_payload_semantics": "byte_for_byte_copy",
        "only_material_change": f"append_or_validate_{DECISION_COLUMN}",
    }
    source_integrity = _artifact_integrity(source, source_ledger)
    if source_integrity is not None:
        # The copied feature/audit payloads are immutable, while the ledger
        # gains only explicit decision time. Rebind every selector artifact to
        # the newly published bytes so later base selection cannot mix source
        # and upgraded lineage.
        destination_like = dict(source_integrity)
        destination_like["selector_ledger_sha256"] = "__recomputed_after_write__"
        destination_like["r3_label_economics_contract"] = r3_label_economics_contract(upgraded_ledger)
        destination_like["r3_label_economics_contract_sha256"] = destination_like["r3_label_economics_contract"]["contract_sha256"]
        value["artifact_integrity"] = destination_like
    return value


def _verify_destination(destination: Path) -> dict[str, Any]:
    manifest = _json(destination / "manifest.json")
    provenance = manifest.get("timestamp_upgrade")
    if not isinstance(provenance, Mapping) or provenance.get("schema") != SCHEMA:
        raise StageISelectorTimestampUpgradeError(
            f"destination lacks the timestamp-upgrade provenance: {destination}"
        )
    ledger = pd.read_parquet(destination / "selector_ledger.parquet")
    resolve_stage_i_timestamp_contract(ledger)
    if _identity_digest(ledger) != provenance.get("upgraded_identity_sha256"):
        raise StageISelectorTimestampUpgradeError("destination identity digest drift")
    expected = provenance.get("source_feature_payload_sha256")
    if not isinstance(expected, Mapping):
        raise StageISelectorTimestampUpgradeError("destination feature digests missing")
    observed = {name: _file_sha256(destination / name) for name in _REQUIRED_PAYLOADS}
    if observed != dict(expected):
        raise StageISelectorTimestampUpgradeError(
            "destination feature payload differs from the source selector"
        )
    integrity = manifest.get("artifact_integrity")
    if integrity is not None:
        if not isinstance(integrity, Mapping) or integrity.get("schema") != "stage_i_selector_artifact_integrity_v1":
            raise StageISelectorTimestampUpgradeError("destination artifact-integrity contract is malformed")
        if integrity.get("selector_ledger_sha256") != _file_sha256(destination / "selector_ledger.parquet"):
            raise StageISelectorTimestampUpgradeError("destination selector ledger integrity drift")
        contract = r3_label_economics_contract(ledger)
        if integrity.get("r3_label_economics_contract") != contract or integrity.get("r3_label_economics_contract_sha256") != contract["contract_sha256"]:
            raise StageISelectorTimestampUpgradeError("destination R3 label/economics integrity drift")
    return manifest


def upgrade_stage_i_selector_timestamp_contract(
    source_dir: str | Path,
    destination_dir: str | Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Publish a new selector artifact with explicit decision-time lineage."""
    source = Path(source_dir).resolve()
    destination = Path(destination_dir).resolve()
    if source == destination:
        raise StageISelectorTimestampUpgradeError(
            "timestamp upgrade destination must differ from the immutable source"
        )
    if destination.exists():
        if resume:
            return _verify_destination(destination)
        raise FileExistsError(f"timestamp-upgrade destination exists: {destination}")
    source_manifest, source_ledger = _verify_completed_source(source)
    upgraded_ledger = attach_stage_i_decision_timestamp(source_ledger)
    manifest = _upgrade_manifest(
        source, source_manifest, source_ledger, upgraded_ledger
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-", dir=str(destination.parent)
        )
    )
    try:
        for name in _REQUIRED_PAYLOADS:
            shutil.copy2(source / name, temporary / name)
        upgraded_ledger.to_parquet(
            temporary / "selector_ledger.parquet", index=False, compression="zstd"
        )
        if isinstance(manifest.get("artifact_integrity"), dict):
            manifest["artifact_integrity"]["selector_ledger_sha256"] = _file_sha256(
                temporary / "selector_ledger.parquet"
            )
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return _verify_destination(destination)


__all__ = [
    "SCHEMA",
    "StageISelectorTimestampUpgradeError",
    "upgrade_stage_i_selector_timestamp_contract",
]
