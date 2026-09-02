#!/usr/bin/env python3
"""Create an audited feature-state implementation successor.

This utility does not transform state values.  It is intentionally restricted
to a single reviewed source-file hash transition, copies all persisted state
files byte-for-byte, and updates only the implementation lineage in the
successor manifest.  It exists for a stateful producer whose materializer was
changed after a verified state checkpoint but before the next causal advance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHANGED_SOURCE = "scripts/materialize_strict_r3_forward_features_incremental_v13.py"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_inventory(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "state_bundle_manifest.json"
    }


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--out-bundle", type=Path, required=True)
    parser.add_argument(
        "--changed-source",
        action="append",
        default=None,
        help=(
            "Reviewed repository-relative implementation path to re-receipt. "
            "Repeat together with --expected-old-sha256 for a deliberately "
            "reviewed multi-file transition."
        ),
    )
    parser.add_argument(
        "--expected-old-sha256",
        action="append",
        default=None,
        help="Expected prior hash for the corresponding --changed-source.",
    )
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()

    source = args.source_bundle.resolve()
    out = args.out_bundle.resolve()
    changed_sources = [str(value) for value in (args.changed_source or [DEFAULT_CHANGED_SOURCE])]
    old_hashes = [str(value) for value in (args.expected_old_sha256 or [])]
    if not old_hashes or len(changed_sources) != len(old_hashes):
        raise ValueError(
            "supply exactly one --expected-old-sha256 for each --changed-source"
        )
    if len(set(changed_sources)) != len(changed_sources):
        raise ValueError("changed implementation paths must be unique")
    changed = dict(zip(changed_sources, old_hashes, strict=True))
    changed_paths = {relative: (ROOT / relative).resolve() for relative in changed}
    if not source.is_dir() or not (source / "state_bundle_manifest.json").is_file():
        raise FileNotFoundError("source bundle lacks state_bundle_manifest.json")
    if out.exists():
        raise FileExistsError("successor bundle already exists")
    if ROOT.resolve() not in source.parents or ROOT.resolve() not in out.parents:
        raise ValueError("state bundles must remain inside the Ares repository")
    for relative, changed_path in changed_paths.items():
        if ROOT.resolve() not in changed_path.parents or not changed_path.is_file():
            raise ValueError(f"changed source is not one repository file: {relative}")

    manifest_path = source / "state_bundle_manifest.json"
    manifest = _json(manifest_path)
    if manifest.get("schema") != "strict_r3_causal_feature_state_bundle_v2":
        raise ValueError("unsupported state-bundle schema")
    implementation = dict(manifest.get("implementation_sha256") or {})
    transitions = []
    for relative, expected_old in changed.items():
        if implementation.get(relative) != expected_old:
            raise ValueError(
                f"source bundle does not carry declared old implementation hash: {relative}"
            )
        current = _sha(changed_paths[relative])
        if current == expected_old:
            raise ValueError(f"implementation has not changed: {relative}")
        transitions.append({
            "changed_source": relative,
            "old_sha256": expected_old,
            "new_sha256": current,
        })
    for relative, expected in implementation.items():
        if relative in changed:
            continue
        candidate = (ROOT / str(relative)).resolve()
        if ROOT.resolve() not in candidate.parents or not candidate.is_file():
            raise ValueError(f"implementation source is invalid: {relative}")
        if _sha(candidate) != str(expected):
            raise ValueError(f"unreviewed implementation change: {relative}")

    before = _relative_inventory(source)
    shutil.copytree(source, out, copy_function=shutil.copy2)
    after = _relative_inventory(out)
    if before != after:
        raise AssertionError("state payload changed while creating successor")

    successor = dict(manifest)
    successor_implementation = dict(implementation)
    for transition in transitions:
        successor_implementation[transition["changed_source"]] = transition["new_sha256"]
    successor["implementation_sha256"] = successor_implementation
    successor["implementation_reseal"] = {
        "schema": "strict_r3_feature_state_materializer_reseal_v1",
        "source_bundle": str(source.relative_to(ROOT)),
        "source_manifest_sha256": _sha(manifest_path),
        "transitions": transitions,
        "reason": str(args.reason),
        "state_payload_files": len(before),
        "state_payloads_preserved_byte_exact": True,
    }
    out_manifest = out / "state_bundle_manifest.json"
    out_manifest.write_text(json.dumps(successor, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema": "strict_r3_feature_state_materializer_reseal_receipt_v1",
        "source_bundle": str(source.relative_to(ROOT)),
        "source_manifest_sha256": _sha(manifest_path),
        "successor_bundle": str(out.relative_to(ROOT)),
        "successor_manifest_sha256": _sha(out_manifest),
        "transitions": transitions,
        "state_payload_files": len(before),
        "state_payloads_preserved_byte_exact": True,
    }
    (out.parent / "materializer_reseal_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
