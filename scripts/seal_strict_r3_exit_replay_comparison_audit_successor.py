#!/usr/bin/env python3
"""Seal the audit-only live-versus-replay exit-comparison successor.

This successor changes no model, feature, calibration, admission, portfolio,
entry, or exit-policy parameter.  It only makes each confirmed exit retain the
three execution references needed for later research: live executable VWAP,
the frozen parent-policy replay threshold, and the same-minute completed close.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v112_"
    "bcf_current_dual_oi_timeout_runtime_integrity.json"
)
OUT_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v113_"
    "bcf_current_dual_exit_replay_comparison_audit.json"
)
SOURCE_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260821_v40_"
    "v112_oi_timeout_runtime_integrity.json"
)
OUT_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260821_v41_"
    "v113_exit_replay_comparison_audit.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v113_v137_"
    "oi_timeout_runtime_integrity.json"
)
OUT_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v114_v138_"
    "exit_replay_comparison_audit.json"
)
OUT_RECEIPT = ROOT / (
    "data_perp/artifacts/strict_r3_v113_v138_exit_replay_comparison_audit_"
    "20260821_v1/seal_receipt.json"
)
CHANGED_RUNTIME = (
    "extreme_price_movements/inference/run_inference.py",
    "extreme_price_movements/inference/strict_r3_live_execution.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _static_contract(payload: dict) -> dict:
    result = copy.deepcopy(payload)
    result.pop("purpose", None)
    result.pop("runtime_reseal", None)
    result.get("overrides", {}).pop("runtime_code_sha256", None)
    return result


def main() -> None:
    source_overlay = json.loads(SOURCE_OVERLAY.read_text())
    overlay = copy.deepcopy(source_overlay)
    hashes = dict(overlay["overrides"].get("runtime_code_sha256") or {})
    for relative in CHANGED_RUNTIME:
        hashes[relative] = sha(ROOT / relative)
    overlay["overrides"]["runtime_code_sha256"] = hashes
    overlay["purpose"] = (
        "v113: audit-only exit comparison. Each live confirmed exit records "
        "the actual fill and executable VWAP, the frozen parent-policy replay "
        "threshold, and the exact same-minute completed close. No frozen model, "
        "feature, Geometry/K9, mapper, admission, auction, entry or exit "
        "policy parameter changes."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": list(CHANGED_RUNTIME),
        "reason": (
            "Persist and email the live-VWAP / replay-threshold / completed-"
            "close exit comparison. This is telemetry only."
        ),
    }
    if _static_contract(source_overlay) != _static_contract(overlay):
        raise AssertionError("runtime successor changed static inference semantics")
    write_new(OUT_OVERLAY, overlay)

    authorization = copy.deepcopy(json.loads(SOURCE_AUTHORIZATION.read_text()))
    authorization.update({
        "authorization_source": (
            "User-authorized 2026-08-21 audit-only successor: persist the "
            "live VWAP, policy-replay threshold and completed-minute close for "
            "each exit. Trading authority and all economics remain unchanged."
        ),
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
    })
    write_new(OUT_AUTHORIZATION, authorization)

    source_execution = json.loads(SOURCE_EXECUTION.read_text())
    execution = copy.deepcopy(source_execution)
    execution.update({
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "activation_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "activation_authorization_sha256": sha(OUT_AUTHORIZATION),
        "version_note": (
            "v138: audit-only exit-comparison telemetry. Records actual fill, "
            "live executable VWAP, parent-policy replay threshold and same-minute "
            "completed close; no execution authority or rich parent policy change."
        ),
    })
    execution_hashes = dict(execution.get("runtime_code_sha256") or {})
    for relative in execution_hashes:
        execution_hashes[relative] = sha(ROOT / relative)
    execution["runtime_code_sha256"] = execution_hashes
    execution["runtime_reseal_predecessors"] = list(
        execution.get("runtime_reseal_predecessors") or []
    ) + [{
        "successor_execution_semantics": "exit_replay_comparison_audit_v1",
        "predecessor_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "predecessor_execution_sha256": sha(SOURCE_EXECUTION),
        "predecessor_inference_bundle": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "predecessor_inference_bundle_sha256": sha(SOURCE_OVERLAY),
        "allowed_runtime_code_paths": list(CHANGED_RUNTIME),
        "added_runtime_code_paths": [],
        "changed_execution_fields": [],
        "report_only": True,
    }]
    write_new(OUT_EXECUTION, execution)

    receipt = {
        "schema": "strict_r3_exit_replay_comparison_audit_reseal_v1",
        "source_overlay": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "source_overlay_sha256": sha(SOURCE_OVERLAY),
        "source_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "source_execution_sha256": sha(SOURCE_EXECUTION),
        "changed_runtime_paths": list(CHANGED_RUNTIME),
        "runtime_code_sha256": {relative: sha(ROOT / relative) for relative in CHANGED_RUNTIME},
        "static_inference_contract_exact": _static_contract(source_overlay) == _static_contract(overlay),
        "successor_overlay": str(OUT_OVERLAY.relative_to(ROOT)),
        "successor_overlay_sha256": sha(OUT_OVERLAY),
        "successor_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "successor_authorization_sha256": sha(OUT_AUTHORIZATION),
        "successor_execution": str(OUT_EXECUTION.relative_to(ROOT)),
        "successor_execution_sha256": sha(OUT_EXECUTION),
    }
    write_new(OUT_RECEIPT, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
