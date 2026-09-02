#!/usr/bin/env python3
"""Correct the producer-only runtime bridge for the v119 sentinel contract.

The v119 execution artifact is valid for the monitor, but it retained the
v96 bridge record whose ``current_inference_bundle_sha256`` points to the
prior overlay.  The producer correctly fail-closes on that mismatch.  This
immutable successor retains v119's sole semantic change and declares the
one valid runtime-only v96 -> v97 bridge.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / (
    "config/strict_r3_kraken_live_execution_v97_v119_"
    "executable_vwap_sentinel.json"
)
PREDECESSOR_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v96_"
    "bcf_current_dual_prefix_schema_runtime_reseal.json"
)
OUTPUT = ROOT / (
    "config/strict_r3_kraken_live_execution_v97_v120_"
    "executable_vwap_sentinel_bridgefix.json"
)
LIVE_EXECUTION = "extreme_price_movements/inference/strict_r3_live_execution.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"immutable successor already exists: {OUTPUT}")
    payload = copy.deepcopy(json.loads(SOURCE.read_text()))
    current_hash = str(payload["inference_bundle_sha256"])
    payload["runtime_reseal_predecessors"] = [{
        "current_inference_bundle_sha256": current_hash,
        "predecessor_inference_bundle": str(PREDECESSOR_OVERLAY.relative_to(ROOT)),
        "predecessor_inference_bundle_sha256": sha(PREDECESSOR_OVERLAY),
        "allowed_runtime_code_paths": [LIVE_EXECUTION],
        "added_runtime_code_paths": [],
    }]
    payload["version_note"] = (
        "v120: v119 executable-VWAP frozen-threshold sentinel with its "
        "producer runtime-reseal bridge corrected to name the v97 overlay. "
        "No model, feature, Geometry/K9, calibration, admission, auction, "
        "entry, exit-policy or sentinel semantic changes from v119."
    )
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "execution": str(OUTPUT.relative_to(ROOT)),
        "execution_sha256": sha(OUTPUT),
        "current_inference_bundle_sha256": current_hash,
        "predecessor_overlay_sha256": sha(PREDECESSOR_OVERLAY),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
