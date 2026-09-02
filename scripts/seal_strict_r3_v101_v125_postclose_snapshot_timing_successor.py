#!/usr/bin/env python3
"""Seal the v125 telemetry-only successor for native-close snapshot timing."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v100_"
    "bcf_current_dual_exit_telemetry_receipts.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v100_v124_"
    "side_specific_exit_telemetry_receipts.json"
)
SOURCE_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260819_v28_"
    "v100_exit_telemetry_receipts.json"
)
OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v101_"
    "bcf_current_dual_exit_postclose_snapshot_timing.json"
)
AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260819_v29_"
    "v101_postclose_snapshot_timing.json"
)
EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v101_v125_"
    "side_specific_exit_postclose_snapshot_timing.json"
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
        "v101: telemetry-only runtime successor. After an exchange-originated "
        "close is detected, persist the immediate public ticker/full-book "
        "capture with detection time, capture start/end, lag and duration. "
        "This remains post-close public evidence, not an exchange private "
        "trigger-time order book; no model, feature, policy, calibration, "
        "admission, auction or order-decision semantics change."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": [LIVE_EXECUTION],
        "reason": (
            "Make native-stop post-close public order-book capture latency "
            "explicit and auditable without changing trading semantics."
        ),
    }
    write_new(OVERLAY, overlay)

    authorization = copy.deepcopy(json.loads(SOURCE_AUTHORIZATION.read_text()))
    authorization.update({
        "authorization_source": (
            "User-approved v125 telemetry correction: record immediate "
            "post-exchange-close public book/ticker capture timing; strategy, "
            "gates, policy and order authority remain unchanged."
        ),
        "inference_bundle": str(OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OVERLAY),
    })
    write_new(AUTHORIZATION, authorization)

    execution = copy.deepcopy(json.loads(SOURCE_EXECUTION.read_text()))
    execution.update({
        "inference_bundle": str(OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OVERLAY),
        "activation_authorization": str(AUTHORIZATION.relative_to(ROOT)),
        "activation_authorization_sha256": sha(AUTHORIZATION),
        "version_note": (
            "v125: telemetry-only native-close snapshot timing successor. "
            "Receipts record detection-to-public-book capture lag and duration; "
            "no trading decision or rich-policy parameter changes."
        ),
    })
    hashes = dict(execution.get("runtime_code_sha256") or {})
    hashes[LIVE_EXECUTION] = sha(ROOT / LIVE_EXECUTION)
    execution["runtime_code_sha256"] = hashes
    execution["runtime_reseal_predecessors"] = [{
        "predecessor_inference_bundle": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "predecessor_inference_bundle_sha256": sha(SOURCE_OVERLAY),
        "current_inference_bundle_sha256": sha(OVERLAY),
        "allowed_runtime_code_paths": [LIVE_EXECUTION],
        "added_runtime_code_paths": [],
        "reason": (
            "Telemetry-only runtime bridge: no static model, feature, policy, "
            "admission, portfolio or execution-gate semantics change."
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
