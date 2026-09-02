#!/usr/bin/env python3
"""Seal the reviewed current-runtime successor without changing live economics."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v107_"
    "bcf_current_dual_runtime_bridge_chainfix.json"
)
OUT_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v108_"
    "bcf_current_dual_full_runtime_review.json"
)
SOURCE_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260820_v35_"
    "v107_runtime_bridge_chainfix.json"
)
OUT_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260820_v36_"
    "v108_full_runtime_review.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v108_v132_"
    "runtime_bridge_chainfix.json"
)
OUT_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v109_v133_"
    "full_runtime_review.json"
)
REVIEW = ROOT / (
    "data_perp/artifacts/strict_r3_full_runtime_review_reseal_20260820_v1/"
    "runtime_review.json"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def resolved_runtime_hashes(overlay: dict) -> dict[str, str]:
    base_path = (ROOT / str(overlay["base_bundle"])).resolve()
    if ROOT not in base_path.parents:
        raise ValueError("base bundle escapes repository root")
    base = json.loads(base_path.read_text())
    expected = dict(base.get("runtime_code_sha256") or {})
    expected.update(dict((overlay.get("overrides") or {}).get("runtime_code_sha256") or {}))
    if not expected:
        raise ValueError("source contract has no runtime hashes")
    return expected


def main() -> None:
    source_overlay = json.loads(SOURCE_OVERLAY.read_text())
    expected = resolved_runtime_hashes(source_overlay)
    actual = {}
    for relative in sorted(expected):
        path = (ROOT / relative).resolve()
        if ROOT not in path.parents or not path.is_file():
            raise FileNotFoundError(f"sealed runtime source is unavailable: {relative}")
        actual[relative] = sha(path)
    changed = sorted(key for key in expected if expected[key] != actual[key])
    if not changed:
        raise ValueError("no current-runtime delta to reseal")

    # The economic and model artifacts are deliberately not reselected here.
    source_resolved = copy.deepcopy(source_overlay)
    source_resolved["runtime_code_sha256"] = expected
    static_before = copy.deepcopy(source_resolved)
    static_before.pop("runtime_code_sha256", None)
    static_before.pop("runtime_reseal", None)
    static_before.pop("purpose", None)

    overlay = copy.deepcopy(source_overlay)
    code_hashes = dict((overlay.get("overrides") or {}).get("runtime_code_sha256") or {})
    for relative in changed:
        code_hashes[relative] = actual[relative]
    overlay["overrides"]["runtime_code_sha256"] = code_hashes
    overlay["purpose"] = (
        "v108: reviewed current-runtime reseal after v131/v132 producer "
        "fail-closed on stale source hashes. Frozen model, feature-contract, "
        "Geometry/K9, EV-map, admission, portfolio and policy artifact hashes "
        "are unchanged; this successor binds the exact reviewed local sources."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": changed,
        "reason": (
            "Complete current-source integrity reseal. It restores fail-closed "
            "producer startup after runtime files changed outside the v131 seal."
        ),
    }
    write_new(OUT_OVERLAY, overlay)

    # Reconstruct the successor's resolved static identity and prove it has
    # no non-runtime deltas against the reviewed source overlay.
    successor_resolved = copy.deepcopy(source_overlay)
    successor_resolved["runtime_code_sha256"] = {
        **expected, **{key: actual[key] for key in changed},
    }
    static_after = copy.deepcopy(successor_resolved)
    static_after.pop("runtime_code_sha256", None)
    static_after.pop("runtime_reseal", None)
    static_after.pop("purpose", None)
    if static_before != static_after:
        raise AssertionError("full-runtime reseal changed a non-runtime contract field")

    authorization = copy.deepcopy(json.loads(SOURCE_AUTHORIZATION.read_text()))
    authorization.update({
        "authorization_source": (
            "User-authorized 2026-08-20 reviewed full-runtime integrity reseal. "
            "It binds current sources after a fail-closed source-hash mismatch; "
            "models, features contract, calibration, admission, portfolio and "
            "parent policy artifacts are unchanged."
        ),
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
    })
    write_new(OUT_AUTHORIZATION, authorization)

    execution = copy.deepcopy(json.loads(SOURCE_EXECUTION.read_text()))
    execution.update({
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "activation_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "activation_authorization_sha256": sha(OUT_AUTHORIZATION),
        "version_note": (
            "v133: reviewed current-runtime integrity reseal. No frozen model, "
            "feature contract, Geometry/K9, EV map, admission, portfolio, entry "
            "or parent-policy parameter change."
        ),
    })
    execution_hashes = dict(execution.get("runtime_code_sha256") or {})
    for relative in execution_hashes:
        execution_hashes[relative] = sha(ROOT / relative)
    execution["runtime_code_sha256"] = execution_hashes
    execution["runtime_reseal_predecessors"] = list(
        execution.get("runtime_reseal_predecessors") or []
    ) + [{
        "successor_execution_semantics": "full_runtime_review_integrity_reseal_v1",
        "current_inference_bundle_sha256": sha(OUT_OVERLAY),
        "predecessor_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "predecessor_execution_sha256": sha(SOURCE_EXECUTION),
        "predecessor_inference_bundle": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "predecessor_inference_bundle_sha256": sha(SOURCE_OVERLAY),
        "allowed_runtime_code_paths": changed,
        "added_runtime_code_paths": [],
        "changed_execution_fields": [],
        "reviewed_current_runtime": True,
    }]
    write_new(OUT_EXECUTION, execution)

    review = {
        "schema": "strict_r3_full_runtime_review_v1",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "source_overlay": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "source_overlay_sha256": sha(SOURCE_OVERLAY),
        "changed_runtime_paths": changed,
        "expected_runtime_hashes": {key: expected[key] for key in changed},
        "actual_runtime_hashes": {key: actual[key] for key in changed},
        "non_runtime_contract_exact": static_before == static_after,
        "successor_overlay": str(OUT_OVERLAY.relative_to(ROOT)),
        "successor_overlay_sha256": sha(OUT_OVERLAY),
        "successor_execution": str(OUT_EXECUTION.relative_to(ROOT)),
        "successor_execution_sha256": sha(OUT_EXECUTION),
    }
    write_new(REVIEW, review)
    print(json.dumps(review, sort_keys=True))


if __name__ == "__main__":
    main()
