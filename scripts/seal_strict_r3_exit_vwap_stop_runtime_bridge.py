#!/usr/bin/env python3
"""Bridge the stop-VWAP runtime successor to the latest v84 state receipt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREDECESSOR = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v84_"
    "bcf_current_dual_recovered_terminal_1000.json"
)
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v85_"
    "bcf_current_dual_exit_vwap_stop.json"
)
OUT_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v86_"
    "bcf_current_dual_exit_vwap_stop_bridge.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v83_v105_"
    "bcf_current_dual_exit_vwap_stop.json"
)
OUT_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v84_v106_"
    "bcf_current_dual_exit_vwap_stop_bridge.json"
)
SOURCE_AUTH = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260818_v13_"
    "bcf_current_dual_exit_vwap_stop.json"
)
OUT_AUTH = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260818_v14_"
    "bcf_current_dual_exit_vwap_stop_bridge.json"
)
CHANGED = "extreme_price_movements/inference/strict_r3_live_execution.py"
ADDED = "extreme_price_movements/inference/trade_executor.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    overlay = json.loads(SOURCE_OVERLAY.read_text())
    overlay["purpose"] = (
        "v86: v85 full-size exit-VWAP native-stop successor with a declared "
        "runtime-only bridge from the completed v84 hourly state receipts."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": [],
        "reason": "Add only the verified v84 predecessor bridge; no strategy or runtime source change from v85.",
    }
    write_new(OUT_OVERLAY, overlay)

    authorization = json.loads(SOURCE_AUTH.read_text())
    authorization.update({
        "authorization_source": (
            "User-approved full-size exit-VWAP protective-stop successor, "
            "continued only from a verified v84 predecessor receipt through "
            "the explicit runtime-only bridge."
        ),
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
    })
    write_new(OUT_AUTH, authorization)

    execution = json.loads(SOURCE_EXECUTION.read_text())
    execution.update({
        "version_note": (
            "v106: v105 exit-VWAP stop protection plus verified v84 runtime "
            "bridge; all model, feature, admission, portfolio and policy semantics unchanged."
        ),
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "activation_authorization": str(OUT_AUTH.relative_to(ROOT)),
        "activation_authorization_sha256": sha(OUT_AUTH),
        "runtime_reseal_predecessors": [{
            "current_inference_bundle_sha256": sha(OUT_OVERLAY),
            "predecessor_inference_bundle": str(PREDECESSOR.relative_to(ROOT)),
            "predecessor_inference_bundle_sha256": sha(PREDECESSOR),
            "allowed_runtime_code_paths": [CHANGED],
            "added_runtime_code_paths": [ADDED],
        }],
    })
    write_new(OUT_EXECUTION, execution)
    print(json.dumps({
        "overlay": str(OUT_OVERLAY.relative_to(ROOT)), "overlay_sha256": sha(OUT_OVERLAY),
        "authorization": str(OUT_AUTH.relative_to(ROOT)), "authorization_sha256": sha(OUT_AUTH),
        "execution": str(OUT_EXECUTION.relative_to(ROOT)), "execution_sha256": sha(OUT_EXECUTION),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
