#!/usr/bin/env python3
"""Seal a test-only inference successor for the embedded Geometry/K9 recovery.

The original Geometry/K9 object is preserved inside the frozen conversion
bundle.  Its original semantic parent and K9-view identities remain unchanged;
only the missing standalone identity-witness payload is rebound to a recovered
content-addressed file.  This utility writes an inference overlay and receipt
only.  It neither starts a service nor grants execution authority.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_v154_v182_"
    "shadow_recovery_hash_rebind.json"
)
GEOMETRY_DIR = ROOT / (
    "data_perp/artifacts/strict_r3_schema_v2_geometry_k9_long_octdec2024_"
    "k9weighted_embedded_recovery_20260823_v2"
)
OUT_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_v155_v184_"
    "embedded_geometry_content_rebind.json"
)
OUT_RECEIPT = ROOT / (
    "data_perp/artifacts/strict_r3_runtime_reseal_v184_"
    "embedded_geometry_content_rebind_20260823_v1/receipt.json"
)
EXPECTED_PARENT = "ad7eae631da909feddee7349d07fd8ef377db173067d971bee33d24d82f20eb4"
EXPECTED_VIEW = "dbf7de6da6bad6927bcbe577d7ad2d2118ecc24a6bdfd35fb7fa190be13d7638"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    manifest_path = GEOMETRY_DIR / "run_manifest.json"
    payload_path = GEOMETRY_DIR / "frozen_geometry_k9.joblib"
    for path in (SOURCE_OVERLAY, manifest_path, payload_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if OUT_OVERLAY.exists() or OUT_RECEIPT.exists():
        raise FileExistsError("v183 recovery outputs are immutable")

    manifest = _read(manifest_path)
    if manifest.get("recovery_mode") != (
        "sealed_conversion_bundle_embedded_parent_extraction_no_refit"
    ):
        raise ValueError("geometry artifact is not an embedded no-refit recovery")
    if manifest.get("semantic_parent_bundle_sha256") != EXPECTED_PARENT:
        raise ValueError("geometry recovery has the wrong original parent identity")
    if manifest.get("semantic_k9_view_sha256") != EXPECTED_VIEW:
        raise ValueError("geometry recovery has the wrong K9-view identity")
    verification = dict(manifest.get("verification") or {})
    if verification.get("maximum_absolute_delta") != 0.0:
        raise ValueError("geometry recovery lacks exact output parity")
    if verification.get("source_output_sha256_float32") != verification.get(
        "extracted_output_sha256_float32"
    ):
        raise ValueError("geometry recovery output hashes differ")
    payload_sha = _sha(payload_path)
    manifest_sha = _sha(manifest_path)
    if payload_sha != manifest.get("bundle_sha256"):
        raise ValueError("geometry recovery payload does not match its manifest")

    overlay = copy.deepcopy(_read(SOURCE_OVERLAY))
    overrides = overlay.setdefault("overrides", {})
    paths = dict(overrides.get("paths") or {})
    hashes = dict(overrides.get("sha256") or {})
    paths["frozen_geometry_bundle"] = str(payload_path.relative_to(ROOT))
    hashes["frozen_geometry_bundle"] = payload_sha
    hashes["frozen_geometry_manifest"] = manifest_sha
    overrides["paths"] = paths
    overrides["sha256"] = hashes
    overlay["purpose"] = (
        "v154: content-addressed rebind of the missing standalone Geometry/K9 "
        "identity witness. The frozen Oct-Dec 2024 parent and K9-view identities "
        "remain ad7e… and dbf7… respectively; no model is refit."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_artifact_rebind_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_artifact_paths": ["frozen_geometry_bundle", "frozen_geometry_manifest"],
        "economic_contract_changed": False,
        "semantic_geometry_parent_sha256": EXPECTED_PARENT,
        "semantic_geometry_view_sha256": EXPECTED_VIEW,
        "reason": (
            "The original raw standalone payload is unavailable locally. Its exact "
            "fitted parent survives embedded in the sealed conversion bundle; the "
            "recovered payload has exact 170-row/57-output parity with that object."
        ),
    }
    _write_new(OUT_OVERLAY, overlay)
    overlay_sha = _sha(OUT_OVERLAY)

    receipt = {
        "schema": "strict_r3_embedded_geometry_content_rebind_v1",
        "status": "pass",
        "source_overlay": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "source_overlay_sha256": _sha(SOURCE_OVERLAY),
        "successor_overlay": str(OUT_OVERLAY.relative_to(ROOT)),
        "successor_overlay_sha256": overlay_sha,
        "recovered_geometry_dir": str(GEOMETRY_DIR.relative_to(ROOT)),
        "recovered_geometry_content_sha256": payload_sha,
        "recovered_geometry_manifest_sha256": manifest_sha,
        "semantic_geometry_parent_sha256": EXPECTED_PARENT,
        "semantic_geometry_view_sha256": EXPECTED_VIEW,
        "output_parity": verification,
        "semantics": {
            "geometry_refit": False,
            "conversion_bundle_changed": False,
            "upstream_bundle_changed": False,
            "base_models_changed": False,
            "residual_models_changed": False,
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
