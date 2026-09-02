#!/usr/bin/env python3
"""Bind the recovered frozen 170-symbol universe witness into a test overlay.

This is an artifact-only, no-execution successor.  The recovered manifest is
derived from two preserved candidate-grid receipts that attest the lost
manifest's hash and agree on all ordered source-map entries.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_v155_v184_"
    "embedded_geometry_content_rebind.json"
)
UNIVERSE_MANIFEST = ROOT / (
    "data_perp/artifacts/strict_r3_schema_v2_target_free_long_aug1_7_"
    "current_spread_embedded_recovery_20260823_v1/run_manifest.json"
)
OUT_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_v156_v185_"
    "geometry_universe_content_rebind.json"
)
OUT_RECEIPT = ROOT / (
    "data_perp/artifacts/strict_r3_runtime_reseal_v185_"
    "geometry_universe_content_rebind_20260823_v1/receipt.json"
)
LOST_UNIVERSE_SHA256 = "ceaa143bfaa0c22e7f45ead0d874b8394780b5608e57145a38f7307294c8edc5"
EXPECTED_SCHEMA = "strict_r3_canonical_forward_v2_target_free_hourly_grid"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    if not SOURCE_OVERLAY.is_file() or not UNIVERSE_MANIFEST.is_file():
        raise FileNotFoundError("source overlay or recovered universe manifest missing")
    if OUT_OVERLAY.exists() or OUT_RECEIPT.exists():
        raise FileExistsError("v185 outputs are immutable")
    universe = _read(UNIVERSE_MANIFEST)
    source_map = universe.get("source_map")
    if (
        universe.get("schema") != EXPECTED_SCHEMA
        or universe.get("original_missing_manifest_sha256") != LOST_UNIVERSE_SHA256
        or int(universe.get("universe_rows", -1)) != 170
        or not isinstance(source_map, dict)
        or len(source_map) != 170
        or len(universe.get("source_receipts") or []) < 2
    ):
        raise ValueError("recovered universe witness does not meet the frozen 170 contract")
    overlay = copy.deepcopy(_read(SOURCE_OVERLAY))
    overrides = overlay.setdefault("overrides", {})
    paths = dict(overrides.get("paths") or {})
    hashes = dict(overrides.get("sha256") or {})
    paths["frozen_universe_manifest"] = str(UNIVERSE_MANIFEST.relative_to(ROOT))
    hashes["frozen_universe_manifest"] = _sha(UNIVERSE_MANIFEST)
    overrides["paths"] = paths
    overrides["sha256"] = hashes
    overlay["purpose"] = (
        "v156: bind a recovered 170-symbol frozen-universe source-map witness "
        "after its original historical manifest became unavailable. Geometry, "
        "models, calibration, admission, portfolio, sizing and exits are unchanged."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_artifact_rebind_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_artifact_paths": ["frozen_universe_manifest"],
        "economic_contract_changed": False,
        "original_missing_manifest_sha256": LOST_UNIVERSE_SHA256,
        "recovered_source_map_sha256": universe["source_map_sha256"],
        "reason": (
            "The candidate materializer consumes the ordered source_map keys and "
            "spread limit only. Two immutable historical grid receipts attest the "
            "lost original hash and retain the identical 170-member source map."
        ),
    }
    _write_new(OUT_OVERLAY, overlay)
    receipt = {
        "schema": "strict_r3_universe_witness_content_rebind_v1",
        "status": "pass",
        "source_overlay": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "source_overlay_sha256": _sha(SOURCE_OVERLAY),
        "successor_overlay": str(OUT_OVERLAY.relative_to(ROOT)),
        "successor_overlay_sha256": _sha(OUT_OVERLAY),
        "recovered_universe_manifest": str(UNIVERSE_MANIFEST.relative_to(ROOT)),
        "recovered_universe_manifest_sha256": _sha(UNIVERSE_MANIFEST),
        "original_missing_manifest_sha256": LOST_UNIVERSE_SHA256,
        "source_map_sha256": universe["source_map_sha256"],
        "universe_rows": 170,
        "semantics": {
            "membership_changed": False,
            "geometry_changed": False,
            "models_changed": False,
            "calibration_changed": False,
            "admission_changed": False,
            "portfolio_changed": False,
            "sizing_changed": False,
            "exit_policy_changed": False,
            "execution_authority_granted": False,
        },
    }
    _write_new(OUT_RECEIPT, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
