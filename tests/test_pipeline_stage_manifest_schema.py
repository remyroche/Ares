from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "pipeline_stage_manifest_v1.schema.json"
MANIFEST_PATH = (
    ROOT / "config" / "pipeline_stage_manifest_repository_20260724.json"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_schema_and_repository_manifest_are_valid_json() -> None:
    schema = _load(SCHEMA_PATH)
    manifest = _load(MANIFEST_PATH)

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert manifest["schema_version"] == "pipeline_stage_manifest_v1"


def test_repository_manifest_covers_every_required_stage_identity() -> None:
    schema = _load(SCHEMA_PATH)
    manifest = _load(MANIFEST_PATH)

    assert set(schema["required"]).issubset(manifest)
    assert set(schema["properties"]["source"]["required"]).issubset(
        manifest["source"]
    )
    assert set(schema["properties"]["contracts"]["required"]).issubset(
        manifest["contracts"]
    )
    assert set(schema["properties"]["fitted_state"]["required"]).issubset(
        manifest["fitted_state"]
    )


def test_repository_manifest_uses_locked_status_and_hash_formats() -> None:
    schema = _load(SCHEMA_PATH)
    manifest = _load(MANIFEST_PATH)

    assert manifest["status"] in schema["properties"]["status"]["enum"]
    assert GIT_SHA_RE.fullmatch(manifest["source"]["revision"])
    assert SHA256_RE.fullmatch(manifest["source"]["tree_sha256"])
    assert SHA256_RE.fullmatch(manifest["source"]["archive_sha256"])
    assert SHA256_RE.fullmatch(manifest["source"]["dirty_diff_sha256"])
    assert manifest["created_at_utc"].endswith("Z")


def test_non_applicable_contracts_are_explicit() -> None:
    manifest = _load(MANIFEST_PATH)

    feature_store = manifest["inputs"]["feature_store"]
    assert feature_store["sha256"] is None
    assert feature_store["not_applicable_reason"]

    for contract in manifest["contracts"].values():
        assert contract["id"] is None
        assert contract["sha256"] is None
        assert contract["not_applicable_reason"]


def test_dirty_source_cannot_claim_verified_remote_recovery() -> None:
    manifest = _load(MANIFEST_PATH)
    source = manifest["source"]

    assert source["dirty"] is True
    assert source["dirty_paths"]
    assert source["remote_recovery"]["state"] != "VERIFIED"
