#!/usr/bin/env python3
"""Seal v124 after the validated immutable exit-telemetry implementation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v99_"
    "bcf_current_dual_exit_telemetry.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v99_v123_"
    "side_specific_exit_telemetry.json"
)
SOURCE_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260819_v27_"
    "v99_exit_telemetry.json"
)
OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v100_"
    "bcf_current_dual_exit_telemetry_receipts.json"
)
AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260819_v28_"
    "v100_exit_telemetry_receipts.json"
)
EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v100_v124_"
    "side_specific_exit_telemetry_receipts.json"
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
        "v100: corrected runtime-only telemetry successor. Immutable monitor "
        "receipts retain every directional executable-VWAP observation and every "
        "completed one-minute OHLCV/policy-state input. Directional close reporting "
        "is also side-correct. No model, feature/Geometry state, calibration, "
        "admission, auction, entry economics, exit threshold, or rich-policy "
        "parameter changes."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": [LIVE_EXECUTION],
        "reason": (
            "Corrected, analysis-only persistence of already consumed live exit "
            "inputs and side-correct terminal reporting."
        ),
    }
    write_new(OVERLAY, overlay)

    authorization = copy.deepcopy(json.loads(SOURCE_AUTHORIZATION.read_text()))
    authorization.update({
        "authorization_source": (
            "User-approved v124 runtime-only telemetry correction: persist exact "
            "live exit inputs for analysis; strategy, gates, policy and order "
            "authority remain unchanged."
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
            "v124: corrected immutable exit-telemetry successor. It records "
            "every 30-second directional executable-VWAP observation and each "
            "completed one-minute policy input/state update; no trading decision "
            "or rich-policy parameter changed."
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
