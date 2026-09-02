#!/usr/bin/env python3
"""Seal strict-R3 feature state into a content-addressed immutable store.

The hourly feature materializer requires a writable private directory while it
advances causal state.  Retaining that directory *and* a copied immutable
bundle doubles the storage for every completed hour.  This post-snapshot stage
keeps the existing logical bundle layout, replaces its payloads with symlinks
to SHA-256-addressed immutable objects, records the delta versus the preceding
bundle, and then retires the private writable overlay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd


SCHEMA = "strict_r3_feature_state_content_store_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _store_object(
    source: Path,
    object_store: Path,
    digest: str,
    *,
    trust_sealed_inventory: bool = False,
) -> tuple[Path, bool]:
    """Publish one immutable object atomically and return (path, reused)."""
    target = object_store / "objects" / digest[:2] / digest
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file():
            raise ValueError(f"content-addressed object is corrupt: {target}")
        # One-time historical backfill can rely on the immutable v2 inventory
        # receipt.  New/live publication always re-hashes the object.
        if not trust_sealed_inventory and _sha256(target) != digest:
            raise ValueError(f"content-addressed object is corrupt: {target}")
        return target, True
    if trust_sealed_inventory:
        # Move the original bytes atomically into the immutable object store
        # rather than reading a slow historical volume merely to make a second
        # copy.  The source hash is the sealed v2 inventory value; a later
        # full-store audit will re-verify every object independently.  Bundle
        # callers immediately replace the moved logical path with a relative
        # object symlink, so its public layout remains unchanged.
        try:
            if source.is_symlink():
                raise ValueError(
                    f"sealed inventory points to a missing content object: {source}"
                )
            os.replace(source, target)
        except FileExistsError:
            if not target.is_file():
                raise ValueError(f"racing object is corrupt: {target}")
            return target, True
        return target, False
    descriptor, name = tempfile.mkstemp(prefix=f".{digest}.", suffix=".tmp", dir=target.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        shutil.copy2(source, temporary)
        if _sha256(temporary) != digest:
            raise ValueError(f"state changed during object publication: {source}")
        try:
            os.link(temporary, target)
        except FileExistsError:
            if _sha256(target) != digest:
                raise ValueError(f"racing object is corrupt: {target}")
        return target, False
    finally:
        temporary.unlink(missing_ok=True)


def _symlink_to_object(target: Path, source_object: Path) -> None:
    """Replace a local payload with a relative read-only object link."""
    temporary = target.with_name(f".{target.name}.link.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(os.path.relpath(source_object, start=target.parent))
    temporary.replace(target)


def _inventory(bundle: Path) -> pd.DataFrame:
    path = bundle / "operator_state_inventory.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"state bundle inventory is absent: {path}")
    table = pd.read_parquet(path)
    required = {"relative_path", "sha256", "bytes"}
    if not required.issubset(table.columns) or table["relative_path"].duplicated().any():
        raise ValueError("invalid feature-state inventory")
    return table


def _base_hashes(base_bundle: Path | None) -> tuple[dict[str, str], str | None]:
    if base_bundle is None:
        return {}, None
    manifest = base_bundle / "state_bundle_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"base state manifest is absent: {manifest}")
    table = _inventory(base_bundle)
    return (
        dict(zip(table["relative_path"].astype(str), table["sha256"].astype(str))),
        _sha256(manifest),
    )


def _retire_overlay(path: Path) -> None:
    if not path.exists():
        return
    if any(path.rglob("*.sqlite-wal")):
        raise RuntimeError(f"refusing to retire active SQLite overlay: {path}")
    shutil.rmtree(path)


def compact_bundle(
    *,
    bundle: Path,
    object_store: Path,
    base_bundle: Path | None,
    cache_dir: Path | None,
    panel_update_dir: Path | None,
    retire_private_overlay: bool,
    trust_sealed_inventory: bool = False,
) -> dict[str, object]:
    bundle = bundle.resolve()
    object_store = object_store.resolve()
    manifest_path = bundle / "state_bundle_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"state bundle manifest is absent: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "strict_r3_causal_feature_state_bundle_v2":
        raise ValueError("content-addressed sealing requires a v2 state bundle")
    table = _inventory(bundle)
    base, base_manifest_hash = _base_hashes(base_bundle.resolve() if base_bundle else None)
    states = bundle / "states"
    changed: list[str] = []
    referenced: list[str] = []
    reused = 0
    created = 0
    logical_bytes = 0
    for row in table.sort_values("relative_path", kind="stable").itertuples(index=False):
        relative = str(row.relative_path)
        digest = str(row.sha256)
        payload = states / relative
        if not payload.is_file():
            raise ValueError(f"state payload is absent: {relative}")
        if not trust_sealed_inventory and _sha256(payload) != digest:
            raise ValueError(f"state payload mismatch: {relative}")
        object_path, was_reused = _store_object(
            payload,
            object_store,
            digest,
            trust_sealed_inventory=trust_sealed_inventory,
        )
        _symlink_to_object(payload, object_path)
        reused += int(was_reused)
        created += int(not was_reused)
        logical_bytes += int(row.bytes)
        (referenced if base.get(relative) == digest else changed).append(relative)

    panel_relative = str(manifest.get("panel_state") or "")
    panel_digest = str(manifest.get("panel_state_sha256") or "")
    panel = bundle / panel_relative
    if not panel_relative or not panel_digest or not panel.is_file():
        raise ValueError("embedded source-panel state is absent or hash-mismatched")
    if not trust_sealed_inventory and _sha256(panel) != panel_digest:
        raise ValueError("embedded source-panel state is absent or hash-mismatched")
    panel_object, panel_reused = _store_object(
        panel,
        object_store,
        panel_digest,
        trust_sealed_inventory=trust_sealed_inventory,
    )
    _symlink_to_object(panel, panel_object)
    reused += int(panel_reused)
    created += int(not panel_reused)

    receipt = {
        "schema": SCHEMA,
        "status": "pass",
        "bundle": str(bundle),
        "bundle_manifest_sha256": _sha256(manifest_path),
        "object_store": str(object_store),
        "base_bundle": str(base_bundle.resolve()) if base_bundle else None,
        "base_bundle_manifest_sha256": base_manifest_hash,
        "state_files": int(len(table)),
        "logical_state_bytes": int(logical_bytes),
        "changed_state_files": int(len(changed)),
        "referenced_state_files": int(len(referenced)),
        "changed_relative_paths": changed,
        "referenced_relative_paths": referenced,
        "panel_state_sha256": panel_digest,
        "objects_created": int(created),
        "objects_reused": int(reused),
        "verification_mode": (
            "sealed_inventory_trusted_hardlink_pending_full_audit"
            if trust_sealed_inventory
            else "full_sha256_verified_copy"
        ),
        "logical_bundle_layout": "v2_bundle_symlinks_to_content_addressed_objects",
        "private_overlay_retired": False,
    }
    if retire_private_overlay:
        if cache_dir is not None:
            _retire_overlay(cache_dir.resolve())
        if panel_update_dir is not None:
            _retire_overlay(panel_update_dir.resolve())
        receipt.update({
            "private_overlay_retired": True,
            "retired_cache_dir": str(cache_dir) if cache_dir else None,
            "retired_panel_update_dir": str(panel_update_dir) if panel_update_dir else None,
        })
    receipt_path = bundle.parent / "content_addressed_storage_receipt.json"
    if receipt_path.exists():
        raise FileExistsError(f"content-addressed receipt already exists: {receipt_path}")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--object-store", type=Path, required=True)
    parser.add_argument("--base-bundle", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--panel-update-dir", type=Path)
    parser.add_argument("--retire-private-overlay", action="store_true")
    parser.add_argument(
        "--trust-sealed-inventory",
        action="store_true",
        help="Historical migration only: use v2 inventory hashes and hard-link immutable payloads; requires a later full audit.",
    )
    args = parser.parse_args()
    print(json.dumps(compact_bundle(
        bundle=args.bundle,
        object_store=args.object_store,
        base_bundle=args.base_bundle,
        cache_dir=args.cache_dir,
        panel_update_dir=args.panel_update_dir,
        retire_private_overlay=bool(args.retire_private_overlay),
        trust_sealed_inventory=bool(args.trust_sealed_inventory),
    ), sort_keys=True))


if __name__ == "__main__":
    main()
