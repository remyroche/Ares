#!/usr/bin/env python3
"""Archive then prune explicitly unreferenced heavyweight research artifacts.

This utility is deliberately conservative.  It considers only direct children
of ``data_perp/artifacts`` which are not mentioned anywhere in current
``config``, ``docs``, ``data_perp/live`` or ``scripts`` text sources and whose
pre-prune size is at least the configured floor.  It preserves a compact tar
archive per run containing the manifest/contract/configuration, aggregate
metric files, reports, and an index with checksums before it removes a source
directory.  A frozen manifest is required for execution.

It never touches live state, configs, docs, referenced artifacts, or any path
outside the artifacts root.  This is a maintenance tool, not model logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "data_perp/artifacts"
REFERENCE_ROOTS = (ROOT / "config", ROOT / "docs", ROOT / "data_perp/live", ROOT / "scripts")
DEFAULT_ARCHIVE = ROOT / "data_perp/research_archives/cleanup_unreferenced_20260829_v1"
MIN_BYTES = 180_000 * 1024
TEXT_SUFFIXES = {".json", ".md", ".py", ".yaml", ".yml", ".txt"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _allocated_bytes(path: Path) -> int:
    """Return allocated disk space, matching the cleanup-budget audit.

    APFS snapshots and sparse/clone-backed files can have a much larger
    logical length than their present on-disk allocation.  The user approved
    the audited disk-space set, so target selection and pre-prune verification
    must use allocated blocks rather than logical byte length.
    """
    return sum(file.stat().st_blocks * 512 for file in path.rglob("*") if file.is_file() and not file.is_symlink())


def _du_bytes_map(paths: list[Path]) -> dict[Path, int]:
    """Return one bulk physical-space scan, avoiding per-directory process cost."""
    result = subprocess.run(["du", "-sk", *(str(path) for path in paths)], check=True, capture_output=True, text=True)
    output: dict[Path, int] = {}
    for line in result.stdout.splitlines():
        kib, raw_path = line.split("\t", 1)
        output[Path(raw_path).resolve()] = int(kib) * 1024
    return output


def _reference_text() -> str:
    pieces: list[str] = []
    for base in REFERENCE_ROOTS:
        if not base.exists():
            continue
        for file in base.rglob("*"):
            if not file.is_file() or file.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if file.stat().st_size > 10_000_000:
                continue
            try:
                pieces.append(file.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                continue
    return "\n".join(pieces)


def _selected_members(root: Path) -> list[Path]:
    """Return compact reproducibility evidence, never raw panels/models."""
    selected: list[Path] = []
    for file in root.rglob("*"):
        if not file.is_file() or file.is_symlink():
            continue
        lower = file.name.lower()
        keep = (
            lower in {"run_manifest.json", "correctness_report.json"}
            or file.suffix.lower() == ".md"
            or (file.suffix.lower() == ".json" and any(token in lower for token in ("config", "contract", "manifest", "spec")))
            or (file.suffix.lower() in {".parquet", ".csv", ".json"} and any(token in lower for token in ("metrics", "summary", "results", "comparison")))
        )
        if keep:
            selected.append(file)
    return sorted(selected)


def _plan(*, selection: str) -> dict[str, object]:
    references = _reference_text()
    targets: list[dict[str, object]] = []
    candidates = [candidate for candidate in sorted(ARTIFACTS.iterdir()) if candidate.is_dir() and not candidate.is_symlink()]
    disk_bytes = _du_bytes_map(candidates)
    for candidate in candidates:
        allocated = _allocated_bytes(candidate)
        disk = disk_bytes[candidate.resolve()]
        selected_size = disk if selection == "du" else allocated
        if selected_size < MIN_BYTES or candidate.name in references:
            continue
        targets.append({
            "name": candidate.name,
            "bytes": allocated,
            "disk_bytes": disk,
            "members": [str(item.relative_to(candidate)) for item in _selected_members(candidate)],
        })
    return {
        "schema": "archive_prune_unreferenced_research_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "artifacts_root": str(ARTIFACTS.resolve()),
        "reference_roots": [str(path.resolve()) for path in REFERENCE_ROOTS],
        "minimum_bytes": MIN_BYTES,
        "selection": selection,
        "target_count": len(targets),
        "target_bytes": sum(int(item["disk_bytes"] if selection == "du" else item["bytes"]) for item in targets),
        "targets": targets,
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _archive_one(source: Path, archive_root: Path, expected_members: Iterable[str], expected_size: int) -> dict[str, object]:
    if source.parent.resolve() != ARTIFACTS.resolve() or not source.is_dir() or source.is_symlink():
        raise RuntimeError(f"unsafe cleanup target: {source}")
    observed_size = _allocated_bytes(source)
    if observed_size != expected_size:
        raise RuntimeError(f"size changed before prune for {source.name}: {observed_size} != {expected_size}")
    members = _selected_members(source)
    names = [str(file.relative_to(source)) for file in members]
    if names != list(expected_members):
        raise RuntimeError(f"archive member set changed before prune for {source.name}")
    member_index = [{"path": name, "bytes": file.stat().st_size, "sha256": _sha256(file)} for name, file in zip(names, members)]
    index = {
        "source": source.name,
        "source_bytes": observed_size,
        "members": member_index,
        "archived_at": datetime.now(UTC).isoformat(),
    }
    index_path = archive_root / f"{source.name}.index.json"
    archive_path = archive_root / f"{source.name}.tar.gz"
    _write_json(index_path, index)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(index_path, arcname=f"{source.name}/ARCHIVE_INDEX.json", recursive=False)
        for file, name in zip(members, names):
            archive.add(file, arcname=f"{source.name}/{name}", recursive=False)
    # Verify archive membership and content checksums before deletion.
    with tarfile.open(archive_path, "r:gz") as archive:
        archive_names = set(archive.getnames())
        required = {f"{source.name}/ARCHIVE_INDEX.json", *(f"{source.name}/{name}" for name in names)}
        if archive_names != required:
            raise RuntimeError(f"archive membership verification failed for {source.name}")
        for record in member_index:
            handle = archive.extractfile(f"{source.name}/{record['path']}")
            if handle is None:
                raise RuntimeError(f"missing archived member {record['path']}")
            digest = hashlib.sha256(handle.read()).hexdigest()
            if digest != record["sha256"]:
                raise RuntimeError(f"archive checksum verification failed for {source.name}/{record['path']}")
    shutil.rmtree(source)
    return {
        "name": source.name,
        "pruned_bytes": observed_size,
        "archive": str(archive_path.resolve()),
        "archive_bytes": archive_path.stat().st_size,
        "index": str(index_path.resolve()),
        "member_count": len(members),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--selection", choices=("du", "allocated"), default="du", help="physical-space measure for target selection")
    parser.add_argument("--write-plan", action="store_true")
    parser.add_argument("--execute", type=Path, help="frozen cleanup manifest written by --write-plan")
    args = parser.parse_args()
    if args.write_plan == bool(args.execute):
        parser.error("use exactly one of --write-plan or --execute")
    archive_root = args.archive_root.resolve()
    if args.write_plan:
        archive_root.mkdir(parents=True, exist_ok=False)
        plan = _plan(selection=args.selection)
        manifest = archive_root / "cleanup_manifest.json"
        _write_json(manifest, plan)
        print(json.dumps({"manifest": str(manifest), "target_count": plan["target_count"], "target_bytes": plan["target_bytes"]}, sort_keys=True))
        return
    manifest = args.execute.resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if Path(payload["artifacts_root"]).resolve() != ARTIFACTS.resolve():
        raise RuntimeError("manifest does not target this artifact root")
    if manifest.parent.resolve() != archive_root:
        raise RuntimeError("manifest must live inside the declared archive root")
    receipts = []
    total = len(payload["targets"])
    for position, item in enumerate(payload["targets"], start=1):
        print(f"[cleanup] {position}/{total} {item['name']}", flush=True)
        receipts.append(_archive_one(
            ARTIFACTS / str(item["name"]), archive_root,
            [str(member) for member in item["members"]], int(item["bytes"]),
        ))
    _write_json(archive_root / "prune_receipt.json", {
        "schema": "archive_prune_unreferenced_research_receipt_v1",
        "completed_at": datetime.now(UTC).isoformat(),
        "source_manifest": str(manifest),
        "pruned_count": len(receipts),
        "pruned_bytes": sum(int(item["pruned_bytes"]) for item in receipts),
        "archive_bytes": sum(int(item["archive_bytes"]) for item in receipts),
        "receipts": receipts,
    })
    print(json.dumps({"pruned_count": len(receipts), "pruned_bytes": sum(int(item["pruned_bytes"]) for item in receipts)}, sort_keys=True))


if __name__ == "__main__":
    main()
