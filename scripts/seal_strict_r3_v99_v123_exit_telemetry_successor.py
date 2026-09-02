#!/usr/bin/env python3
"""Seal v123: immutable live exit-telemetry successor of v122."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v98_"
    "bcf_current_dual_side_specific_runtime.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v98_v122_"
    "side_specific_executable_vwap_sentinel.json"
)
SOURCE_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260819_v26_"
    "v98_side_specific_runtime.json"
)
OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v99_"
    "bcf_current_dual_exit_telemetry.json"
)
AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260819_v27_"
    "v99_exit_telemetry.json"
)
EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v99_v123_"
    "side_specific_exit_telemetry.json"
)
LIVE_EXECUTION = "extreme_price_movements/inference/strict_r3_live_execution.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"immutable successor already exists: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    overlay = copy.deepcopy(json.loads(SOURCE_OVERLAY.read_text()))
    hashes = dict(overlay["overrides"].get("runtime_code_sha256") or {})
    hashes[LIVE_EXECUTION] = sha(ROOT / LIVE_EXECUTION)
    overlay["overrides"]["runtime_code_sha256"] = hashes
    overlay["purpose"] = (
        "v99: runtime-only telemetry successor of v98. It persistently records "
        "each directional executable-VWAP sentinel sample and each consumed "
        "completed one-minute policy bar in immutable monitor receipts; no "
        "model, feature/Geometry state, calibration, admission, auction, entry "
        "economics, exit threshold, or rich-policy parameter changes."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": [LIVE_EXECUTION],
        "reason": (
            "Analysis-only persistence of already-consumed live sentinel and "
            "completed-policy-bar inputs; zero decision authority change."
        ),
    }
    write_new(OVERLAY, overlay)

    authorization = copy.deepcopy(json.loads(SOURCE_AUTHORIZATION.read_text()))
    authorization.update({
        "authorization_source": (
            "User-approved runtime-only v123 telemetry successor: preserve full "
            "live entry/exit evidence for analysis while leaving the sealed "
            "strategy, gates, policy and order authority unchanged."
        ),
        "inference_bundle": str(OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OVERLAY),
    })
    gates = list(authorization.get("preserved_gates") or [])
    gates.append("immutable_30_second_vwap_and_completed_bar_telemetry")
    authorization["preserved_gates"] = gates
    write_new(AUTHORIZATION, authorization)

    execution = copy.deepcopy(json.loads(SOURCE_EXECUTION.read_text()))
    execution.update({
        "inference_bundle": str(OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OVERLAY),
        "activation_authorization": str(AUTHORIZATION.relative_to(ROOT)),
        "activation_authorization_sha256": sha(AUTHORIZATION),
        "version_note": (
            "v123: v99 analysis-only live-exit telemetry successor. Immutable "
            "monitor receipts now retain each directional executable-VWAP check "
            "and consumed completed-1m policy OHLCV/state update. No scoring, "
            "admission, auction, execution gate or rich-policy threshold changes."
        ),
    })
    execution_hashes = dict(execution.get("runtime_code_sha256") or {})
    execution_hashes[LIVE_EXECUTION] = sha(ROOT / LIVE_EXECUTION)
    execution["runtime_code_sha256"] = execution_hashes
    execution["runtime_reseal_predecessors"] = [{
        "predecessor_inference_bundle": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "predecessor_inference_bundle_sha256": sha(SOURCE_OVERLAY),
        "current_inference_bundle_sha256": sha(OVERLAY),
        "allowed_runtime_code_paths": [LIVE_EXECUTION],
        "added_runtime_code_paths": [],
        "reason": (
            "Immutable telemetry-only runtime bridge: static model, feature, "
            "policy, admission, portfolio and execution semantics are identical."
        ),
    }]
    write_new(EXECUTION, execution)
    print(json.dumps({
        "overlay": str(OVERLAY.relative_to(ROOT)),
        "overlay_sha256": sha(OVERLAY),
        "authorization": str(AUTHORIZATION.relative_to(ROOT)),
        "authorization_sha256": sha(AUTHORIZATION),
        "execution": str(EXECUTION.relative_to(ROOT)),
        "execution_sha256": sha(EXECUTION),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
