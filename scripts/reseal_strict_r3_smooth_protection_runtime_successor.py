#!/usr/bin/env python3
"""Reseal the smooth-protection successor after its audited runtime update."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD_OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v60_bcf_current_dual_mc1.json"
OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v61_bcf_current_dual_mc1_smooth_runtime.json"
POLICY = ROOT / "data_perp/artifacts/strict_r3_rich_policy_smooth_protection_long_20260817_v1/frozen_policy.json"
BASE_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v58_v68_bcf_current_dual_rich_policy.json"
EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v60_v70_bcf_current_dual_smooth_protection.json"
BASE_AUTHORIZATION = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260817_v23_bcf_current_dual.json"
AUTHORIZATION = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260817_v25_bcf_current_dual_smooth_protection.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    overlay = copy.deepcopy(json.loads(OLD_OVERLAY.read_text()))
    hashes = dict(overlay["overrides"]["runtime_code_sha256"])
    for relative in (
        "extreme_price_movements/inference/run_inference.py",
        "extreme_price_movements/inference/strict_r3_live_execution.py",
        "extreme_price_movements/strict_r3_rich_policy.py",
        "scripts/run_strict_r3_live_hourly_entry_producer.py",
    ):
        hashes[relative] = sha(ROOT / relative)
    overlay["purpose"] = (
        "sealed shadow-only successor: unchanged dual BCF/current MC1 admission "
        "with audited smooth-capital-protection runtime/reporting lineage"
    )
    overlay["overrides"]["runtime_code_sha256"] = hashes
    write_new(OVERLAY, overlay)
    execution = copy.deepcopy(json.loads(BASE_EXECUTION.read_text()))
    execution.update({
        "version_note": (
            "v70: sealed smooth capital-protection successor after runtime "
            "reporting reseal. Model/admission/portfolio semantics unchanged."
        ),
        "inference_bundle": str(OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OVERLAY),
        "exit_policy": str(POLICY.relative_to(ROOT)),
        "exit_policy_sha256": sha(POLICY),
        "activation_authorization": str(AUTHORIZATION.relative_to(ROOT)),
        "position_monitor": (
            "persistent one-minute controller: hard stop -> prior smooth lock -> "
            "trailing -> fast adverse -> next-bar MFE/arm; entry ATR immutable; "
            "Adaptive Exit V1 only modulates parent trailing activation."
        ),
        "runtime_code_sha256": {
            relative: sha(ROOT / relative)
            for relative in dict(execution["runtime_code_sha256"])
        },
    })
    auth = copy.deepcopy(json.loads(BASE_AUTHORIZATION.read_text()))
    auth.update({
        "authorization_source": (
            "Explicit user approval on 2026-08-17 to promote the sealed smooth "
            "capital-protection parent-policy successor. Model/admission/portfolio "
            "semantics are unchanged."
        ),
        "inference_bundle": str(OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OVERLAY),
        "exit_policy": str(POLICY.relative_to(ROOT)),
        "exit_policy_sha256": sha(POLICY),
        "preserved_gates": list(auth.get("preserved_gates") or []) + [
            "raw_decision_time_ATR_smooth_protection",
            "same_bar_smooth_arm_and_trigger_prohibited",
            "upward_only_reduce_only_stop_amendments",
        ],
    })
    write_new(AUTHORIZATION, auth)
    execution["activation_authorization_sha256"] = sha(AUTHORIZATION)
    write_new(EXECUTION, execution)
    print(json.dumps({
        "overlay": str(OVERLAY), "overlay_sha256": sha(OVERLAY),
        "execution": str(EXECUTION), "execution_sha256": sha(EXECUTION),
        "authorization": str(AUTHORIZATION), "authorization_sha256": sha(AUTHORIZATION),
        "policy_sha256": sha(POLICY),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
