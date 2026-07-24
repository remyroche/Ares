from __future__ import annotations

import json
import pickle
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_full_pipeline_migration import (
    compare_checksums,
    hash_path,
    run_audit,
)


def _write_inventory(path: Path, entries: list[object]) -> Path:
    path.write_text(json.dumps({"p0": entries}), encoding="utf-8")
    return path


def test_hash_path_is_deterministic_and_covers_nested_content(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "b.txt").write_text("b", encoding="utf-8")
    (artifact / "a.txt").write_text("a", encoding="utf-8")

    first = hash_path(artifact)
    second = hash_path(artifact)

    assert first == second
    assert first["files"] == 2
    assert first["directories"] == 1
    (artifact / "a.txt").write_text("changed", encoding="utf-8")
    assert hash_path(artifact)["sha256"] != first["sha256"]


def test_run_audit_generates_r0_deliverables_and_safe_smoke_evidence(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "checkpoint.json").write_text(
        json.dumps({"previous_root": "/does/not/exist"}), encoding="utf-8"
    )
    (repo / "model.pkl").write_bytes(pickle.dumps({"safe": "inspection only"}))
    database = repo / "state.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE evidence (id INTEGER PRIMARY KEY)")
    connection.close()
    parquet = repo / "rows.parquet"
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"value": [1, 2]}), parquet)
    inventory = _write_inventory(
        tmp_path / "p0.json",
        [
            {"id": "checkpoint", "path": "checkpoint.json"},
            {"id": "pickle", "path": "model.pkl", "kind": "model_bundle"},
            {"id": "sqlite", "path": "state.sqlite"},
            {"id": "parquet", "path": "rows.parquet"},
        ],
    )

    payload = run_audit(repo, inventory, tmp_path / "audit", max_smoke_files=2)

    for name in (
        "migration_inventory.json",
        "migration_checksums.sha256",
        "migration_verification.md",
        "read-only_smoke.log",
    ):
        assert (tmp_path / "audit" / name).is_file()
    assert payload["comparison"]["status"] == "not_requested"
    records = {
        Path(record["path"]).name: record for record in payload["smoke"]["records"]
    }
    assert records["checkpoint.json"]["status"] == "opened"
    assert records["rows.parquet"]["method"] == "pyarrow.footer"
    assert records["state.sqlite"]["method"] == "sqlite.read_only"
    assert records["model.pkl"]["status"] == "inspected_without_deserialization"
    assert any(
        row.get("absolute_path") == "/does/not/exist" and row["exists"] is False
        for row in payload["absolute_path_findings"]
    )
    markdown = (tmp_path / "audit" / "migration_verification.md").read_text(
        encoding="utf-8"
    )
    assert "not as a successful migration comparison" in markdown


def test_comparison_reports_missing_baseline_without_claiming_a_match(
    tmp_path: Path,
) -> None:
    result = compare_checksums({"P0": "a" * 64}, tmp_path / "missing.sha256")

    assert result["status"] == "baseline_missing"
    assert result["matched"] is None
    assert "no hash-match claim" in result["note"]


def test_run_audit_compares_inventory_checksums_by_explicit_id(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "one.txt").write_text("one", encoding="utf-8")
    inventory = _write_inventory(
        tmp_path / "p0.json", [{"id": "source-tree", "path": "one.txt"}]
    )
    initial = run_audit(repo, inventory, tmp_path / "first")

    compared = run_audit(
        repo,
        inventory,
        tmp_path / "second",
        tmp_path / "first" / "migration_checksums.sha256",
    )
    assert compared["comparison"]["status"] == "compared"
    assert compared["comparison"]["matched"] is True

    (repo / "one.txt").write_text("two", encoding="utf-8")
    changed = run_audit(
        repo,
        inventory,
        tmp_path / "third",
        tmp_path / "first" / "migration_checksums.sha256",
    )
    assert changed["comparison"]["matched"] is False
    assert changed["comparison"]["changed"] == ["source-tree"]


def test_missing_required_p0_is_recorded_without_hashing_an_unrelated_path(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    inventory = _write_inventory(
        tmp_path / "p0.json", [{"id": "missing", "path": "not-here"}]
    )

    payload = run_audit(repo, inventory, tmp_path / "audit")

    item = payload["inventory"]["items"][0]
    assert item["id"] == "missing"
    assert item["exists"] is False
    assert item["sha256"] is None
    assert (tmp_path / "audit" / "migration_checksums.sha256").read_text(
        encoding="utf-8"
    ) == ""
