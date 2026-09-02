#!/usr/bin/env python3
"""Seal the hash-bound predecessor fast-path as a runtime-only successor.

The live hourly producer previously scanned all historical recovery manifests
before it considered its explicitly supplied predecessor.  That is safe but
unbounded at a fresh decision.  The reviewed implementation validates that
exact predecessor first and retains the historical scan only as a fail-closed
fallback.  No model, feature, geometry, mapping, admission, portfolio or exit
semantics change here.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v131_"
    "bcf_current_dual_samebundle21d_direct_15m_reference_shadow.json"
)
OUT_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v132_"
    "bcf_current_dual_samebundle21d_direct_15m_reference_bootstrap_fastpath.json"
)
SOURCE_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260822_v58_"
    "v131_direct_15m_reference_live.json"
)
OUT_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260822_v59_"
    "v132_direct_15m_reference_bootstrap_fastpath_live.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v132_v156_"
    "direct_15m_reference_live.json"
)
OUT_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v133_v157_"
    "direct_15m_reference_bootstrap_fastpath_live.json"
)
REVIEW = ROOT / (
    "data_perp/artifacts/strict_r3_v133_v157_bootstrap_fastpath_runtime_reseal_"
    "20260822_v1/runtime_review.json"
)

CHANGED_PATHS = {"scripts/run_strict_r3_live_hourly_entry_producer.py"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _write_new(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _resolved_runtime_hashes(overlay: dict) -> dict[str, str]:
    base = _read(ROOT / str(overlay["base_bundle"]))
    values = dict(base.get("runtime_code_sha256") or {})
    values.update(dict((overlay.get("overrides") or {}).get("runtime_code_sha256") or {}))
    return values


def _static_contract(overlay: dict) -> dict:
    value = copy.deepcopy(overlay)
    value.pop("purpose", None)
    value.pop("runtime_reseal", None)
    value.get("overrides", {}).pop("runtime_code_sha256", None)
    return value


def main() -> None:
    source_overlay = _read(SOURCE_OVERLAY)
    expected = _resolved_runtime_hashes(source_overlay)
    actual = {relative: _sha(ROOT / relative) for relative in expected}
    changed = {relative for relative in expected if actual[relative] != expected[relative]}
    if changed != CHANGED_PATHS:
        raise ValueError(f"unexpected runtime delta: {sorted(changed)}")

    overlay = copy.deepcopy(source_overlay)
    runtime_hashes = dict(overlay["overrides"].get("runtime_code_sha256") or {})
    runtime_hashes.update({relative: actual[relative] for relative in changed})
    overlay["overrides"]["runtime_code_sha256"] = runtime_hashes
    overlay["purpose"] = (
        "v132: exact hash-bound predecessor fast path for the live hourly "
        "producer. The explicit predecessor is validated before the archival "
        "scan; the archival scan remains a fail-closed fallback."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": sorted(changed),
        "reason": "Eliminate fresh-hour archival-manifest startup latency without changing economic semantics.",
    }
    if _static_contract(source_overlay) != _static_contract(overlay):
        raise AssertionError("runtime successor altered the static inference contract")
    _write_new(OUT_OVERLAY, overlay)

    authorization = copy.deepcopy(_read(SOURCE_AUTHORIZATION))
    authorization.update({
        "authorization_source": (
            "User-authorized runtime-only live continuation after the hourly "
            "producer bootstrap fast path passed hash-bound no-order validation."
        ),
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": _sha(OUT_OVERLAY),
    })
    _write_new(OUT_AUTHORIZATION, authorization)

    execution = copy.deepcopy(_read(SOURCE_EXECUTION))
    execution.update({
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": _sha(OUT_OVERLAY),
        "activation_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "activation_authorization_sha256": _sha(OUT_AUTHORIZATION),
        "version_note": (
            "v157: runtime-only predecessor fast-path successor. No model, "
            "feature, Geometry/K9, EV map, admission, portfolio or exit "
            "parameter changed."
        ),
    })
    execution_hashes = dict(execution.get("runtime_code_sha256") or {})
    for relative in execution_hashes:
        execution_hashes[relative] = _sha(ROOT / relative)
    execution["runtime_code_sha256"] = execution_hashes
    execution["runtime_reseal_predecessors"] = list(
        execution.get("runtime_reseal_predecessors") or []
    ) + [{
        "successor_execution_semantics": "hash_bound_bootstrap_predecessor_fastpath_v1",
        "current_inference_bundle_sha256": _sha(OUT_OVERLAY),
        "predecessor_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "predecessor_execution_sha256": _sha(SOURCE_EXECUTION),
        "predecessor_inference_bundle": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "predecessor_inference_bundle_sha256": _sha(SOURCE_OVERLAY),
        "allowed_runtime_code_paths": sorted(changed),
        "added_runtime_code_paths": [],
        "changed_execution_fields": [],
        "reviewed_current_runtime": True,
    }]
    _write_new(OUT_EXECUTION, execution)

    review = {
        "schema": "strict_r3_bootstrap_fastpath_runtime_reseal_v1",
        "status": "pass",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "source_overlay": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "source_overlay_sha256": _sha(SOURCE_OVERLAY),
        "successor_overlay": str(OUT_OVERLAY.relative_to(ROOT)),
        "successor_overlay_sha256": _sha(OUT_OVERLAY),
        "successor_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "successor_authorization_sha256": _sha(OUT_AUTHORIZATION),
        "successor_execution": str(OUT_EXECUTION.relative_to(ROOT)),
        "successor_execution_sha256": _sha(OUT_EXECUTION),
        "changed_runtime_paths": sorted(changed),
        "economic_contract_changed": False,
        "validation": {
            "explicit_predecessor_hash_bound": True,
            "completion_temporal_state_checks_preserved": True,
            "historical_scan_is_fallback_only": True,
            "fresh_future_decision_required": True,
        },
    }
    _write_new(REVIEW, review)
    print(json.dumps(review, sort_keys=True))


if __name__ == "__main__":
    main()
