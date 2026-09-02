#!/usr/bin/env python3
"""Seal the reviewed OI-worker and current-runtime live successor.

This is intentionally a runtime-only successor.  It binds the verified local
sources after the hourly producer failed closed on stale source hashes and
adds a bounded OI/funding worker lifecycle.  It cannot alter any frozen model,
feature, Geometry/K9, mapper, admission, portfolio or parent-policy artifact.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v111_"
    "bcf_current_dual_feature_state_session_output_schema_fix.json"
)
OUT_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v112_"
    "bcf_current_dual_oi_timeout_runtime_integrity.json"
)
SOURCE_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260820_v39_"
    "v111_feature_state_session_output_schema_fix.json"
)
OUT_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260821_v40_"
    "v112_oi_timeout_runtime_integrity.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v112_v136_"
    "feature_state_session_output_schema_fix.json"
)
OUT_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v113_v137_"
    "oi_timeout_runtime_integrity.json"
)
REVIEW = ROOT / (
    "data_perp/artifacts/strict_r3_v112_v137_oi_timeout_runtime_reseal_"
    "20260821_v1/runtime_review.json"
)

EXPECTED_RUNTIME_DELTA = {
    "extreme_price_movements/inference/strict_r3_live_execution.py",
    "extreme_price_movements/strict_r3_bcf_mc1_mapper.py",
    "extreme_price_movements/strict_r3_canonical_current.py",
    "extreme_price_movements/strict_r3_cell_day_trust.py",
    "extreme_price_movements/strict_r3_mc1_mapper.py",
    "scripts/build_strict_r3_mc1_d2_canonical_bundle.py",
    "scripts/replay_strict_r3_forward_portfolio.py",
    "scripts/run_strict_r3_live_hourly_entry_producer.py",
}


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
    hashes = dict(base.get("runtime_code_sha256") or {})
    hashes.update(dict((overlay.get("overrides") or {}).get("runtime_code_sha256") or {}))
    if not hashes:
        raise ValueError("source contract has no runtime hashes")
    return hashes


def _static_contract(payload: dict) -> dict:
    value = copy.deepcopy(payload)
    value.pop("purpose", None)
    value.pop("runtime_reseal", None)
    value.get("overrides", {}).pop("runtime_code_sha256", None)
    return value


def main() -> None:
    source_overlay = json.loads(SOURCE_OVERLAY.read_text())
    expected = resolved_runtime_hashes(source_overlay)
    actual = {relative: sha(ROOT / relative) for relative in sorted(expected)}
    changed = {key for key in expected if expected[key] != actual[key]}
    if changed != EXPECTED_RUNTIME_DELTA:
        raise ValueError(
            "reviewed runtime delta changed unexpectedly: "
            f"expected={sorted(EXPECTED_RUNTIME_DELTA)} actual={sorted(changed)}"
        )

    overlay = copy.deepcopy(source_overlay)
    runtime_hashes = dict(overlay["overrides"].get("runtime_code_sha256") or {})
    for relative in changed:
        runtime_hashes[relative] = actual[relative]
    overlay["overrides"]["runtime_code_sha256"] = runtime_hashes
    overlay["purpose"] = (
        "v112: reviewed runtime-integrity successor. It seals the complete "
        "current source set and bounds the OI/funding partition lifecycle: a "
        "verified COMPLETE manifest is accepted after clean worker termination; "
        "all other slow workers fail closed at the declared deadline. Frozen "
        "models, features, Geometry/K9, EV maps, dual admission, portfolio and "
        "rich parent policy remain unchanged."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": sorted(changed),
        "reason": (
            "Restore a fail-closed hourly producer after reviewed local runtime "
            "sources changed and prevent a post-COMPLETE OI worker from blocking "
            "all later decisions."
        ),
    }
    if _static_contract(source_overlay) != _static_contract(overlay):
        raise AssertionError("runtime successor changed a static inference contract")
    write_new(OUT_OVERLAY, overlay)

    authorization = copy.deepcopy(json.loads(SOURCE_AUTHORIZATION.read_text()))
    authorization.update({
        "authorization_source": (
            "User-authorized 2026-08-21 runtime-only recovery. It reseals the "
            "reviewed source set and a bounded fail-closed OI worker lifecycle; "
            "no model, mapping, admission, portfolio, or policy parameter changed."
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
            "v137: reviewed OI-worker bounded-runtime successor. No frozen "
            "model, feature, Geometry/K9, EV map, admission, portfolio, entry "
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
        "successor_execution_semantics": "bounded_oi_worker_runtime_integrity_v1",
        "current_inference_bundle_sha256": sha(OUT_OVERLAY),
        "predecessor_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "predecessor_execution_sha256": sha(SOURCE_EXECUTION),
        "predecessor_inference_bundle": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "predecessor_inference_bundle_sha256": sha(SOURCE_OVERLAY),
        "allowed_runtime_code_paths": sorted(changed),
        "added_runtime_code_paths": [],
        "changed_execution_fields": [],
        "reviewed_current_runtime": True,
    }]
    write_new(OUT_EXECUTION, execution)

    review = {
        "schema": "strict_r3_runtime_integrity_reseal_v1",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "source_overlay": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "source_overlay_sha256": sha(SOURCE_OVERLAY),
        "changed_runtime_paths": sorted(changed),
        "expected_runtime_hashes": {key: expected[key] for key in sorted(changed)},
        "actual_runtime_hashes": {key: actual[key] for key in sorted(changed)},
        "non_runtime_contract_exact": _static_contract(source_overlay) == _static_contract(overlay),
        "successor_overlay": str(OUT_OVERLAY.relative_to(ROOT)),
        "successor_overlay_sha256": sha(OUT_OVERLAY),
        "successor_execution": str(OUT_EXECUTION.relative_to(ROOT)),
        "successor_execution_sha256": sha(OUT_EXECUTION),
    }
    write_new(REVIEW, review)
    print(json.dumps(review, sort_keys=True))


if __name__ == "__main__":
    main()
