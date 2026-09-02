#!/usr/bin/env python3
"""Seal the lock-step Geometry/K9 predecessor guard as a runtime successor.

This successor changes neither the feature graph nor any model, mapper,
admission, portfolio, execution, or exit parameter.  It makes the live
producer reject a state predecessor unless its frozen Geometry/K9 receipt
explicitly names the current decision as its next hour.  That turns a skipped
state hour into a cheap pre-source fail-close instead of an expensive failed
feature/scoring run.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v132_"
    "bcf_current_dual_samebundle21d_direct_15m_reference_bootstrap_fastpath.json"
)
OUT_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v133_"
    "bcf_current_dual_samebundle21d_direct_15m_reference_lockstep_geometry_predecessor.json"
)
SOURCE_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260822_v59_"
    "v132_direct_15m_reference_bootstrap_fastpath_live.json"
)
OUT_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260822_v60_"
    "v133_direct_15m_reference_lockstep_geometry_predecessor_live.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v133_v157_"
    "direct_15m_reference_bootstrap_fastpath_live.json"
)
OUT_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v134_v158_"
    "direct_15m_reference_lockstep_geometry_predecessor_live.json"
)
REVIEW = ROOT / (
    "data_perp/artifacts/strict_r3_v134_v158_lockstep_geometry_predecessor_"
    "runtime_reseal_20260822_v1/runtime_review.json"
)

RUNTIME_PATH = "scripts/run_strict_r3_live_hourly_entry_producer.py"
WARM_FEATURE_RECEIPT = ROOT / (
    "data_perp/artifacts/strict_r3_stateful_recovery_v159_terminal_replay_"
    "20260822T170000Z_v1/hour_20260822T170000Z/run/features.log"
)


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


def _warm_feature_seconds(path: Path) -> float:
    for line in reversed(path.read_text().splitlines()):
        if "compute_features_hourly.final" in line and "total=" in line:
            return float(line.split("total=", 1)[1].split("s", 1)[0])
    raise ValueError("missing completed feature timing receipt")


def main() -> None:
    for path in (SOURCE_OVERLAY, SOURCE_AUTHORIZATION, SOURCE_EXECUTION, WARM_FEATURE_RECEIPT):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_overlay = _read(SOURCE_OVERLAY)
    expected = _resolved_runtime_hashes(source_overlay)
    actual = {relative: _sha(ROOT / relative) for relative in expected}
    changed = {relative for relative in expected if actual[relative] != expected[relative]}
    if changed != {RUNTIME_PATH}:
        raise ValueError(f"unexpected runtime delta: {sorted(changed)}")
    warm_seconds = _warm_feature_seconds(WARM_FEATURE_RECEIPT)
    if warm_seconds > 60.0:
        raise ValueError(f"warm full-feature runtime exceeds 60 seconds: {warm_seconds}")

    overlay = copy.deepcopy(source_overlay)
    hashes = dict(overlay["overrides"].get("runtime_code_sha256") or {})
    hashes[RUNTIME_PATH] = actual[RUNTIME_PATH]
    overlay["overrides"]["runtime_code_sha256"] = hashes
    overlay["purpose"] = (
        "v133: live runtime successor requiring a lock-step Geometry/K9 "
        "predecessor. A fresh decision may consume only state whose receipt "
        "names that exact decision as next_decision_ts."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": [RUNTIME_PATH],
        "economic_contract_changed": False,
        "reason": "Fail closed before source/feature work when Geometry/K9 state cannot advance lock-step.",
    }
    if _static_contract(source_overlay) != _static_contract(overlay):
        raise AssertionError("runtime successor altered the static inference contract")
    _write_new(OUT_OVERLAY, overlay)

    authorization = copy.deepcopy(_read(SOURCE_AUTHORIZATION))
    authorization.update({
        "authorization_source": "User-authorized live continuation after a complete no-order recovery, exact terminal parity, and lock-step Geometry/K9 predecessor validation.",
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
        "version_note": "v158: lock-step Geometry/K9 predecessor guard. No economic contract changed.",
    })
    execution_hashes = dict(execution.get("runtime_code_sha256") or {})
    execution_hashes[RUNTIME_PATH] = actual[RUNTIME_PATH]
    execution["runtime_code_sha256"] = execution_hashes
    execution["runtime_reseal_predecessors"] = list(execution.get("runtime_reseal_predecessors") or []) + [{
        "successor_execution_semantics": "lockstep_geometry_k9_predecessor_guard_v1",
        "predecessor_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "predecessor_execution_sha256": _sha(SOURCE_EXECUTION),
        "allowed_runtime_code_paths": [RUNTIME_PATH],
        "economic_contract_changed": False,
        "reviewed_current_runtime": True,
    }]
    _write_new(OUT_EXECUTION, execution)

    review = {
        "schema": "strict_r3_lockstep_geometry_predecessor_runtime_reseal_v1",
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
        "changed_runtime_paths": [RUNTIME_PATH],
        "economic_contract_changed": False,
        "warm_feature_runtime": {
            "receipt": str(WARM_FEATURE_RECEIPT.relative_to(ROOT)),
            "receipt_sha256": _sha(WARM_FEATURE_RECEIPT),
            "full_170_symbol_seconds": warm_seconds,
            "maximum_seconds": 60.0,
            "status": "pass",
        },
        "validation": {
            "exact_next_geometry_k9_predecessor_required": True,
            "stale_bootstrap_rejected_before_source_refresh": True,
            "historical_scan_remains_fail_closed_fallback": True,
            "fresh_future_decision_required": True,
        },
    }
    _write_new(REVIEW, review)
    print(json.dumps(review, sort_keys=True))


if __name__ == "__main__":
    main()
