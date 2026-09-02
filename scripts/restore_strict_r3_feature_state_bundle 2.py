#!/usr/bin/env python3
"""Restore an immutable strict-R3 feature-state bundle into a writable cache.

The same snapshot can seed hourly inference or a chronological training chunk.
Restoration verifies every content hash before copying; restored state is always
independent of the immutable bundle and may be advanced append-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


SUPPORTED_SCHEMAS = {
    "strict_r3_causal_feature_state_bundle_v1",
    "strict_r3_causal_feature_state_bundle_v2",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore_bundle(
    *,
    bundle_dir: Path,
    cache_dir: Path,
    expected_contract_hash: str | None = None,
    expected_panel_hash: str | None = None,
    panel_state_out: Path | None = None,
) -> dict:
    manifest_path = bundle_dir / "state_bundle_manifest.json"
    inventory_path = bundle_dir / "operator_state_inventory.parquet"
    if not manifest_path.is_file() or not inventory_path.is_file():
        raise FileNotFoundError("incomplete feature-state bundle")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") not in SUPPORTED_SCHEMAS:
        raise ValueError("unsupported feature-state bundle schema")
    if (
        expected_contract_hash
        and manifest.get("feature_contract_sha256") != expected_contract_hash
    ):
        raise ValueError("feature contract hash does not match state bundle")
    if (
        expected_panel_hash
        and manifest.get("panel_state_sha256") != expected_panel_hash
    ):
        raise ValueError("panel hash does not match state bundle")
    if cache_dir.exists():
        raise FileExistsError(f"restore target already exists: {cache_dir}")
    bundled_panel = bundle_dir / str(manifest.get("panel_state", ""))
    if manifest.get("schema") == "strict_r3_causal_feature_state_bundle_v2":
        if not bundled_panel.is_file():
            raise FileNotFoundError("v2 feature-state bundle lacks source-panel state")
        if _sha256(bundled_panel) != str(manifest.get("panel_state_sha256")):
            raise ValueError("bundled source-panel state hash mismatch")
    if panel_state_out is not None and panel_state_out.exists():
        raise FileExistsError(f"panel-state restore target exists: {panel_state_out}")

    inventory = pd.read_parquet(inventory_path)
    if len(inventory) != int(manifest.get("state_files", -1)):
        raise ValueError("state inventory count does not match manifest")
    temporary = cache_dir.with_name(cache_dir.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale restore directory exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        for row in inventory.itertuples(index=False):
            source = bundle_dir / "states" / str(row.relative_path)
            if not source.is_file():
                raise FileNotFoundError(source)
            if _sha256(source) != str(row.sha256):
                raise ValueError(f"state hash mismatch: {row.relative_path}")
            target = temporary / str(row.relative_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        temporary.rename(cache_dir)
        if panel_state_out is not None:
            if manifest.get("schema") != "strict_r3_causal_feature_state_bundle_v2":
                raise ValueError("v1 bundle cannot restore an embedded panel state")
            panel_state_out.parent.mkdir(parents=True, exist_ok=True)
            panel_temporary = panel_state_out.with_name(panel_state_out.name + ".tmp")
            shutil.copy2(bundled_panel, panel_temporary)
            panel_temporary.rename(panel_state_out)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    receipt = {
        "schema": "strict_r3_causal_feature_state_restore_v1",
        "bundle_dir": str(bundle_dir),
        "cache_dir": str(cache_dir),
        "feature_contract_sha256": manifest["feature_contract_sha256"],
        "panel_state_sha256": manifest["panel_state_sha256"],
        "state_files": int(len(inventory)),
        "state_bytes": int(inventory["bytes"].sum()),
        "latest_state_timestamp": manifest.get("latest_state_timestamp"),
        "panel_state_out": str(panel_state_out) if panel_state_out is not None else None,
        "panel_tail_hours": manifest.get("panel_tail_hours"),
    }
    (cache_dir / "state_restore_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--expected-contract-hash")
    parser.add_argument("--expected-panel-hash")
    parser.add_argument("--panel-state-out", type=Path)
    args = parser.parse_args()
    receipt = restore_bundle(
        bundle_dir=args.bundle_dir,
        cache_dir=args.cache_dir,
        expected_contract_hash=args.expected_contract_hash,
        expected_panel_hash=args.expected_panel_hash,
        panel_state_out=args.panel_state_out,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
