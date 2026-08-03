"""Fail-closed supersession registry for research artifacts.

The execution roadmap deliberately keeps invalidated research outputs on disk
for forensic reproducibility.  This module makes their status executable:
callers that intend to train, score, replay, map, or promote an artifact must
assert that it is not superseded.  Audit-only readers may still open a revoked
artifact when they explicitly request that purpose.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "pipeline_supersession_manifest_v1"
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "data_perp"
    / "artifacts"
    / "pipeline_supersession_manifest_20260801_v1"
    / "supersession_manifest.json"
)


class SupersededArtifactError(RuntimeError):
    """Raised when a revoked/non-promotable artifact is used operationally."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _normalise(path: str | Path) -> str:
    value = Path(path)
    if not value.is_absolute():
        value = _repo_root() / value
    try:
        value = value.resolve()
    except OSError:
        value = value.absolute()
    try:
        return value.relative_to(_repo_root()).as_posix()
    except ValueError:
        return value.as_posix()


def load_supersession_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the immutable supersession manifest."""

    manifest_path = Path(path) if path is not None else DEFAULT_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(
            "pipeline supersession manifest is missing; refusing operational use: "
            f"{manifest_path}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"unexpected supersession manifest schema: {payload.get('schema')!r}")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("supersession manifest has no entries")
    for entry in entries:
        if not isinstance(entry, Mapping) or not entry.get("artifact"):
            raise ValueError("supersession manifest contains an invalid entry")
        if entry.get("status") not in {"REVOKED", "SUPERSEDED", "NON_PROMOTABLE_DIAGNOSTIC"}:
            raise ValueError(f"invalid supersession status: {entry.get('status')!r}")
    return payload


def _matching_entry(path: str | Path, manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the most specific entry covering ``path``."""

    normalised = _normalise(path)
    matches: list[dict[str, Any]] = []
    for raw in manifest.get("entries", []):
        entry = dict(raw)
        artifact = _normalise(str(entry["artifact"]))
        if normalised == artifact or normalised.startswith(artifact.rstrip("/") + "/"):
            entry["artifact"] = artifact
            matches.append(entry)
    if not matches:
        return None
    return max(matches, key=lambda item: len(str(item["artifact"])))


def artifact_status(path: str | Path, *, manifest: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """Return the covering status record, or ``None`` for an unlisted path."""

    return _matching_entry(path, manifest or load_supersession_manifest())


def assert_artifact_usable(
    path: str | Path,
    *,
    purpose: str = "training",
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Fail closed for revoked artifacts used outside an explicit audit.

    ``purpose='audit'`` and purposes listed in ``allowed_uses`` are permitted
    because the roadmap requires retaining invalid outputs for diagnosis.  All
    other uses are rejected, including inference, policy replay, score maps,
    and promotion.
    """

    record = artifact_status(path, manifest=manifest)
    if record is None:
        return None
    allowed = {str(value) for value in record.get("allowed_uses", [])}
    blocked = {str(value) for value in record.get("blocked_uses", [])}
    status = str(record.get("status"))
    if purpose in allowed or purpose == "audit":
        return record
    if purpose in blocked or status in {"REVOKED", "SUPERSEDED"}:
        raise SupersededArtifactError(
            f"{purpose} use is forbidden for {record['artifact']} ({status}): "
            f"{record.get('reason', 'superseded artifact')}"
        )
    return record
