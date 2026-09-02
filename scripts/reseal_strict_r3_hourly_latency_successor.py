#!/usr/bin/env python3
"""Seal the runtime-only hourly latency successor of smooth policy v70."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# v62/v71/v26 are immutable, unactivated receipts produced before the BCF
# scorer hash was audited.  This successor binds the approved scorer hash.
OLD_OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v62_bcf_current_dual_mc1_latency_runtime.json"
OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v63_bcf_current_dual_mc1_latency_bcf_resealed.json"
OLD_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v61_v71_bcf_current_dual_smooth_latency.json"
EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v62_v72_bcf_current_dual_smooth_latency_bcf_resealed.json"
OLD_AUTHORIZATION = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260817_v26_bcf_current_dual_smooth_latency.json"
AUTHORIZATION = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260817_v27_bcf_current_dual_smooth_latency_bcf_resealed.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    overlay = copy.deepcopy(json.loads(OLD_OVERLAY.read_text()))
    overlay["purpose"] = (
        "approved BCF-scorer reseal plus runtime-only hourly-latency successor: "
        "warmed sealed bundle, boundary-aware scheduler, and concurrent independent "
        "15m/funding/analytics refreshes"
    )
    overlay["overrides"]["runtime_code_sha256"][
        "scripts/run_strict_r3_live_hourly_entry_producer.py"
    ] = sha(ROOT / "scripts/run_strict_r3_live_hourly_entry_producer.py")
    overlay["overrides"]["runtime_code_sha256"][
        "scripts/score_strict_r3_bcf_forward.py"
    ] = sha(ROOT / "scripts/score_strict_r3_bcf_forward.py")
    write_new(OVERLAY, overlay)
    execution = copy.deepcopy(json.loads(OLD_EXECUTION.read_text()))
    execution.update({
        "version_note": (
            "v72: approved BCF-scorer reseal and runtime-only hourly latency successor. The sealed model, dual "
            "admission, portfolio and smooth exit-policy contracts are unchanged; "
            "independent source refreshes execute concurrently behind an all-pass gate."
        ),
        "inference_bundle": str(OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OVERLAY),
        "activation_authorization": str(AUTHORIZATION.relative_to(ROOT)),
        "runtime_code_sha256": {
            relative: sha(ROOT / relative)
            for relative in dict(execution["runtime_code_sha256"])
        },
    })
    auth = copy.deepcopy(json.loads(OLD_AUTHORIZATION.read_text()))
    auth.update({
        "authorization_source": (
            "Explicit user approval on 2026-08-17 after behavioral parity audit to "
            "reseal the changed BCF scorer and activate the runtime-only hourly "
            "latency successor: warm sealed contract cache, boundary-aware scheduler "
            "and concurrent independent source branches. No score, admission, "
            "portfolio or exit-policy parameter changed."
        ),
        "inference_bundle": str(OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OVERLAY),
        "exit_policy": execution["exit_policy"],
        "exit_policy_sha256": execution["exit_policy_sha256"],
    })
    write_new(AUTHORIZATION, auth)
    execution["activation_authorization_sha256"] = sha(AUTHORIZATION)
    write_new(EXECUTION, execution)
    print(json.dumps({
        "overlay": str(OVERLAY), "overlay_sha256": sha(OVERLAY),
        "execution": str(EXECUTION), "execution_sha256": sha(EXECUTION),
        "authorization": str(AUTHORIZATION), "authorization_sha256": sha(AUTHORIZATION),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
