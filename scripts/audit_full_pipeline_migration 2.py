#!/usr/bin/env python3
"""Produce a read-only, deterministic R0 migration audit.

The R0 handover deliberately contains binary model state.  This tool never
deserializes pickle/joblib input: loading either can execute arbitrary code.
It records a non-executing structural inspection instead, so a successful
report does not claim that an untrusted model was loaded.  Parquet is inspected
through its footer and SQLite is opened in immutable read-only mode.

The input inventory is intentionally explicit.  It is JSON (or YAML when
PyYAML is installed) containing a list under ``entries``/``p0``/``p0_paths``/``paths``;
each item is either a path string or an object with ``path``, optional ``id``,
``kind`` and ``required`` fields.  Relative paths are rooted at ``--repo-root``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import pickletools
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".txt",
    ".md",
    ".ini",
    ".cfg",
}
PICKLE_SUFFIXES = {".pkl", ".pickle", ".joblib"}
ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9_])(/[A-Za-z0-9._+@%:=,~\-/]+)")
PROCESS_HINT = re.compile(
    r"(?:python|lightgbm|catboost|optuna|train|run_pipeline)", re.IGNORECASE
)
VERSION_MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "scikit-learn": "sklearn",
    "optuna": "optuna",
    "joblib": "joblib",
    "numba": "numba",
}


@dataclass(frozen=True)
class P0Item:
    identifier: str
    configured_path: str
    resolved_path: Path
    kind: str
    required: bool


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _path_type(path: Path) -> str:
    if path.is_symlink():
        return "symlink"
    if path.is_file():
        return "file"
    if path.is_dir():
        return "directory"
    return "missing"


def _walk_tree(path: Path) -> Iterator[Path]:
    """Yield path then descendants in lexical order, never following links."""

    yield path
    if not path.is_dir() or path.is_symlink():
        return
    with os.scandir(path) as entries:
        for entry in sorted(entries, key=lambda item: item.name):
            child = Path(entry.path)
            yield from _walk_tree(child)


def hash_path(path: Path) -> dict[str, Any]:
    """Return a deterministic content tree digest and bounded metadata."""

    path = Path(path)
    kind = _path_type(path)
    if kind == "missing":
        return {
            "path_type": "missing",
            "sha256": None,
            "bytes": 0,
            "files": 0,
            "directories": 0,
        }
    if kind == "file":
        return {
            "path_type": "file",
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
            "files": 1,
            "directories": 0,
        }
    if kind == "symlink":
        target = os.readlink(path)
        return {
            "path_type": "symlink",
            "sha256": hashlib.sha256(
                ("symlink\\0" + target).encode("utf-8")
            ).hexdigest(),
            "bytes": 0,
            "files": 0,
            "directories": 0,
            "link_target": target,
        }

    digest = hashlib.sha256()
    total_bytes = 0
    files = 0
    directories = 0
    root = path.parent
    for child in _walk_tree(path):
        relative = child.relative_to(root).as_posix()
        child_kind = _path_type(child)
        if child_kind == "file":
            size = child.stat().st_size
            child_hash = _sha256_file(child)
            digest.update(
                f"file\\0{relative}\\0{size}\\0{child_hash}\\n".encode("utf-8")
            )
            total_bytes += size
            files += 1
        elif child_kind == "directory":
            digest.update(f"directory\\0{relative}\\n".encode("utf-8"))
            directories += 1
        elif child_kind == "symlink":
            target = os.readlink(child)
            digest.update(f"symlink\\0{relative}\\0{target}\\n".encode("utf-8"))
    return {
        "path_type": "directory",
        "sha256": digest.hexdigest(),
        "bytes": total_bytes,
        "files": files,
        "directories": directories,
    }


def _load_inventory(path: Path, repository_root: Path) -> list[P0Item]:
    raw_text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ValueError(
                f"inventory is not JSON and PyYAML is unavailable: {path}"
            ) from exc
        payload = yaml.safe_load(raw_text)
    if isinstance(payload, Mapping):
        entries = next(
            (
                payload[key]
                for key in ("entries", "p0", "p0_paths", "paths")
                if key in payload
            ),
            None,
        )
    else:
        entries = payload
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            "inventory must contain a non-empty entries/p0/p0_paths/paths list"
        )
    items: list[P0Item] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if isinstance(entry, str):
            details: Mapping[str, Any] = {"path": entry}
        elif isinstance(entry, Mapping):
            details = entry
        else:
            raise ValueError(f"inventory item {index} is not a path string or object")
        configured = details.get("path")
        if not isinstance(configured, str) or not configured.strip():
            raise ValueError(f"inventory item {index} has no non-empty path")
        identifier = str(details.get("id") or configured)
        if identifier in seen:
            raise ValueError(f"duplicate inventory id: {identifier}")
        seen.add(identifier)
        raw_path = Path(configured).expanduser()
        resolved = raw_path if raw_path.is_absolute() else repository_root / raw_path
        items.append(
            P0Item(
                identifier=identifier,
                configured_path=configured,
                resolved_path=resolved.resolve(strict=False),
                kind=str(details.get("kind") or "auto"),
                required=bool(details.get("required", True)),
            )
        )
    return sorted(items, key=lambda item: (item.identifier, item.configured_path))


def _safe_pickle_inspection(path: Path) -> dict[str, Any]:
    """Inspect pickle opcodes without constructing objects or importing classes."""

    try:
        with path.open("rb") as handle:
            prefix = handle.read(16 * 1024 * 1024)
        opcodes = 0
        globals_seen = 0
        for opcode, _argument, _position in pickletools.genops(prefix):
            opcodes += 1
            if opcode.name in {
                "GLOBAL",
                "STACK_GLOBAL",
                "REDUCE",
                "NEWOBJ",
                "NEWOBJ_EX",
            }:
                globals_seen += 1
        return {
            "status": "inspected_without_deserialization",
            "bytes_inspected": len(prefix),
            "opcode_count": opcodes,
            "potential_object_construction_opcodes": globals_seen,
            "note": "pickle/joblib was not loaded because deserialization can execute code",
        }
    except Exception as exc:  # joblib may be compressed or contain out-of-band blocks
        return {
            "status": "not_deserialized",
            "error": f"{type(exc).__name__}: {exc}",
            "note": "pickle/joblib was not loaded because deserialization can execute code",
        }


def _smoke_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    record: dict[str, Any] = {"path": str(path), "type": suffix or "unknown"}
    try:
        if suffix == ".json":
            with path.open("r", encoding="utf-8") as handle:
                json.load(handle)
            record.update(status="opened", method="json.parse")
        elif suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise RuntimeError(
                    "PyArrow is unavailable for Parquet footer smoke"
                ) from exc
            metadata = pq.ParquetFile(path).metadata
            record.update(
                status="opened",
                method="pyarrow.footer",
                rows=int(metadata.num_rows),
                row_groups=int(metadata.num_row_groups),
            )
        elif suffix in PICKLE_SUFFIXES:
            record.update(_safe_pickle_inspection(path))
        elif suffix in {".sqlite", ".sqlite3", ".db"}:
            uri = f"file:{path.resolve().as_posix()}?mode=ro&immutable=1"
            connection = sqlite3.connect(uri, uri=True)
            try:
                schema_version = connection.execute("PRAGMA schema_version").fetchone()[
                    0
                ]
            finally:
                connection.close()
            record.update(
                status="opened",
                method="sqlite.read_only",
                schema_version=int(schema_version),
            )
        else:
            record.update(status="not_applicable", method="none")
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    return record


def _smoke_candidates(item: P0Item, *, max_files: int) -> list[Path]:
    path = item.resolved_path
    if path.is_file() or path.is_symlink():
        return [path]
    if not path.is_dir() or max_files <= 0:
        return []
    candidates = [
        child
        for child in _walk_tree(path)
        if child.is_file()
        and child.suffix.lower()
        in {
            ".json",
            ".parquet",
            ".pkl",
            ".pickle",
            ".joblib",
            ".sqlite",
            ".sqlite3",
            ".db",
        }
    ]
    return sorted(candidates, key=lambda child: child.relative_to(path).as_posix())[
        :max_files
    ]


def _scan_absolute_paths(
    item: P0Item, *, max_files: int = 1_000, max_bytes: int = 16 * 1024 * 1024
) -> list[dict[str, Any]]:
    path = item.resolved_path
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = [
            child
            for child in _walk_tree(path)
            if child.is_file() and child.suffix.lower() in TEXT_SUFFIXES
        ]
    else:
        return []
    findings: list[dict[str, Any]] = []
    for text_file in sorted(files, key=str)[:max_files]:
        try:
            if text_file.stat().st_size > max_bytes:
                findings.append(
                    {
                        "file": str(text_file),
                        "status": "skipped_too_large",
                        "bytes": text_file.stat().st_size,
                    }
                )
                continue
            content = text_file.read_text(encoding="utf-8", errors="replace")
            values = sorted(
                set(
                    match.group(1).rstrip(".,:;)]}")
                    for match in ABSOLUTE_PATH.finditer(content)
                )
            )
            for value in values:
                findings.append(
                    {
                        "file": str(text_file),
                        "absolute_path": value,
                        "exists": Path(value).exists(),
                    }
                )
        except OSError as exc:
            findings.append(
                {
                    "file": str(text_file),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if len(files) > max_files:
        findings.append(
            {
                "status": "truncated",
                "text_files_seen": len(files),
                "max_files": max_files,
            }
        )
    return findings


def _environment() -> dict[str, Any]:
    versions: dict[str, Any] = {}
    for label, module_name in VERSION_MODULES.items():
        try:
            module = importlib.import_module(module_name)
            versions[label] = {
                "available": True,
                "version": str(getattr(module, "__version__", "unknown")),
            }
        except Exception as exc:
            versions[label] = {
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "modules": versions,
    }


def _disk_evidence(paths: Iterable[Path]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    evidence: list[dict[str, Any]] = []
    for path in paths:
        probe = path if path.exists() else path.parent
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        key = str(probe.resolve())
        if key in seen:
            continue
        seen.add(key)
        usage = shutil.disk_usage(probe)
        evidence.append(
            {
                "path": key,
                "free_bytes": usage.free,
                "total_bytes": usage.total,
                "used_bytes": usage.used,
            }
        )
    return sorted(evidence, key=lambda row: row["path"])


def _process_evidence() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,state=,command="],
            check=True,
            text=True,
            capture_output=True,
        )
        candidates = []
        for line in result.stdout.splitlines():
            if PROCESS_HINT.search(line):
                candidates.append(line.strip())
        return {
            "method": "ps -axo pid,ppid,state,command",
            "candidate_processes": candidates,
            "no_training_processes_observed": not candidates,
            "note": "heuristic process-name evidence; it cannot prove a remote or differently named job is absent",
        }
    except Exception as exc:
        return {
            "method": "ps",
            "error": f"{type(exc).__name__}: {exc}",
            "no_training_processes_observed": None,
        }


def _parse_checksum_file(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([0-9a-fA-F]{64})\s+[* ]?(.*?)$", line)
        if not match or not match.group(2):
            raise ValueError(f"invalid checksum line {number} in {path}")
        checksums[match.group(2)] = match.group(1).lower()
    return checksums


def compare_checksums(
    current: Mapping[str, str], baseline_path: Path | None
) -> dict[str, Any]:
    if baseline_path is None:
        return {
            "status": "not_requested",
            "matched": None,
            "note": "no comparison checksum file was supplied",
        }
    if not baseline_path.exists():
        return {
            "status": "baseline_missing",
            "matched": None,
            "baseline": str(baseline_path),
            "note": "comparison baseline is absent; no hash-match claim is made",
        }
    baseline = _parse_checksum_file(baseline_path)
    current_keys, baseline_keys = set(current), set(baseline)
    changed = sorted(
        key for key in current_keys & baseline_keys if current[key] != baseline[key]
    )
    return {
        "status": "compared",
        "baseline": str(baseline_path),
        "matched": not changed and current_keys == baseline_keys,
        "changed": changed,
        "missing_from_current": sorted(baseline_keys - current_keys),
        "missing_from_baseline": sorted(current_keys - baseline_keys),
    }


def _markdown_report(payload: Mapping[str, Any]) -> str:
    comparison = payload["comparison"]
    missing = [
        item["id"]
        for item in payload["inventory"]["items"]
        if item["required"] and not item["exists"]
    ]
    smoke_failures = [
        row["path"]
        for row in payload["smoke"]["records"]
        if row.get("status") == "failed"
    ]
    unresolved = [
        row for row in payload["absolute_path_findings"] if row.get("exists") is False
    ]
    lines = [
        "# R0 migration verification",
        "",
        "This report is read-only. Pickle and joblib files are not deserialized because loading them can execute code; their status is inspection-only.",
        "",
        "## Inventory",
        "",
        f"- P0 items: {len(payload['inventory']['items'])}",
        f"- Missing required items: {len(missing)}"
        + (f" (`{', '.join(missing)}`)" if missing else ""),
        f"- Checksums file: `{payload['outputs']['checksums']}`",
        "",
        "## Comparison",
        "",
        f"- Status: `{comparison['status']}`",
        f"- Hashes match: `{comparison.get('matched')}`",
        f"- Note: {comparison.get('note', 'comparison completed')}",
        "",
        "## Read-only smoke",
        "",
        f"- Smoke records: {len(payload['smoke']['records'])}",
        f"- Failures: {len(smoke_failures)}"
        + (f" (`{', '.join(smoke_failures)}`)" if smoke_failures else ""),
        "",
        "## Checkpoint paths",
        "",
        f"- Absolute-path findings: {len(payload['absolute_path_findings'])}",
        f"- Missing referenced paths: {len(unresolved)}",
        "",
        "## Environment and host evidence",
        "",
        f"- Python: `{payload['environment']['python']}`",
        f"- Training-process heuristic: `{payload['processes'].get('no_training_processes_observed')}`",
        "",
        "## Gate evidence",
        "",
        "The audit records evidence only. A missing comparison baseline is reported as incomplete, not as a successful migration comparison.",
    ]
    return "\n".join(lines) + "\n"


def run_audit(
    repository_root: Path,
    inventory_config: Path,
    output_dir: Path,
    comparison_checksums: Path | None = None,
    *,
    max_smoke_files: int = 5,
) -> dict[str, Any]:
    """Run the audit and write the four R0 deliverables; does not mutate inputs."""

    repository_root = Path(repository_root).resolve()
    output_dir = Path(output_dir).resolve()
    items = _load_inventory(Path(inventory_config), repository_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory_items: list[dict[str, Any]] = []
    checksum_lines: list[tuple[str, str]] = []
    smoke_records: list[dict[str, Any]] = []
    absolute_findings: list[dict[str, Any]] = []
    for item in items:
        digest = hash_path(item.resolved_path)
        exists = digest["path_type"] != "missing"
        inventory_items.append(
            {
                "id": item.identifier,
                "configured_path": item.configured_path,
                "resolved_path": str(item.resolved_path),
                "kind": item.kind,
                "required": item.required,
                "exists": exists,
                **digest,
            }
        )
        if digest["sha256"]:
            checksum_lines.append((item.identifier, digest["sha256"]))
        if exists:
            for candidate in _smoke_candidates(item, max_files=max_smoke_files):
                record = _smoke_file(candidate)
                record["p0_id"] = item.identifier
                smoke_records.append(record)
            absolute_findings.extend(_scan_absolute_paths(item))

    checksums_path = output_dir / "migration_checksums.sha256"
    checksums_path.write_text(
        "".join(
            f"{digest}  {identifier}\n" for identifier, digest in sorted(checksum_lines)
        ),
        encoding="utf-8",
    )
    comparison = compare_checksums(dict(checksum_lines), comparison_checksums)
    smoke_records.sort(key=lambda row: (row["p0_id"], row["path"]))
    absolute_findings.sort(
        key=lambda row: (row.get("file", ""), row.get("absolute_path", ""))
    )
    outputs = {
        "inventory": str(output_dir / "migration_inventory.json"),
        "checksums": str(checksums_path),
        "verification": str(output_dir / "migration_verification.md"),
        "smoke_log": str(output_dir / "read-only_smoke.log"),
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "read_only": True,
        "repository_root": str(repository_root),
        "inventory": {
            "source_config": str(Path(inventory_config).resolve()),
            "items": inventory_items,
        },
        "comparison": comparison,
        "smoke": {"max_files_per_p0_item": max_smoke_files, "records": smoke_records},
        "absolute_path_findings": absolute_findings,
        "environment": _environment(),
        "disk": _disk_evidence(
            [repository_root, output_dir, *(item.resolved_path for item in items)]
        ),
        "processes": _process_evidence(),
        "outputs": outputs,
    }
    (output_dir / "migration_inventory.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "read-only_smoke.log").write_text(
        json.dumps({"records": smoke_records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "migration_verification.md").write_text(
        _markdown_report(payload), encoding="utf-8"
    )
    return payload


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only deterministic R0 migration audit"
    )
    parser.add_argument(
        "--repo-root",
        "--repository-root",
        required=True,
        type=Path,
        dest="repository_root",
    )
    parser.add_argument(
        "--p0-inventory",
        "--inventory-config",
        required=True,
        type=Path,
        dest="inventory_config",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--comparison-checksums", "--comparison-checksum-file", type=Path, default=None
    )
    parser.add_argument("--max-smoke-files", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    if args.max_smoke_files < 0:
        raise SystemExit("--max-smoke-files must be non-negative")
    payload = run_audit(
        args.repository_root,
        args.inventory_config,
        args.output_dir,
        args.comparison_checksums,
        max_smoke_files=args.max_smoke_files,
    )
    missing = [
        item
        for item in payload["inventory"]["items"]
        if item["required"] and not item["exists"]
    ]
    smoke_failures = [
        record
        for record in payload["smoke"]["records"]
        if record.get("status") == "failed"
    ]
    return 2 if missing or smoke_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
