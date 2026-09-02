#!/usr/bin/env python3
"""Seal the combined feature-state and side-specific-exit runtime successor.

This is intentionally a runtime-only reseal.  It adds the already-reviewed
feature-state snapshot implementation to the inference overlay and the
directional executable-VWAP exit implementation to the execution contract.
All frozen model, feature, Geometry/K9, calibration, admission, auction,
entry-economics, and rich-policy parameters remain identical to v97/v120.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v97_"
    "bcf_current_dual_executable_vwap_sentinel.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v97_v120_"
    "executable_vwap_sentinel_bridgefix.json"
)
SOURCE_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260819_v25_"
    "v97_executable_vwap_sentinel.json"
)
OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v98_"
    "bcf_current_dual_side_specific_runtime.json"
)
AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260819_v26_"
    "v98_side_specific_runtime.json"
)
EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v98_v122_"
    "side_specific_executable_vwap_sentinel.json"
)
CHANGED = (
    "scripts/snapshot_strict_r3_feature_state_bundle.py",
    "extreme_price_movements/inference/strict_r3_live_execution.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"immutable successor already exists: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    overlay = copy.deepcopy(json.loads(SOURCE_OVERLAY.read_text()))
    overlay_hashes = dict(overlay["overrides"].get("runtime_code_sha256") or {})
    for relative in CHANGED:
        overlay_hashes[relative] = sha(ROOT / relative)
    overlay["overrides"]["runtime_code_sha256"] = overlay_hashes
    overlay["purpose"] = (
        "v98: reviewed runtime-only successor of v97. It hash-binds the "
        "append-only feature-state snapshot duplicate-state guard and the "
        "side-specific executable-VWAP exit implementation; no frozen model, "
        "feature contract, Geometry/K9 state, calibration, admission, auction, "
        "entry economics, or rich-policy parameter changes."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": list(CHANGED),
        "reason": (
            "Hash-bind two already reviewed runtime-only repairs: feature-state "
            "snapshot stale duplicate suppression and directional exit handling."
        ),
    }
    write_new(OVERLAY, overlay)

    authorization = copy.deepcopy(json.loads(SOURCE_AUTHORIZATION.read_text()))
    authorization.update({
        "authorization_source": (
            "Explicit user approval on 2026-08-19 for the reviewed runtime-only "
            "side-specific executable-VWAP successor; all live strategy, model, "
            "admission, auction, and rich-exit policy parameters remain frozen."
        ),
        "inference_bundle": str(OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OVERLAY),
    })
    preserved = list(authorization.get("preserved_gates") or [])
    preserved.append("side_specific_executable_vwap_long_bids_short_asks")
    authorization["preserved_gates"] = preserved
    write_new(AUTHORIZATION, authorization)

    execution = copy.deepcopy(json.loads(SOURCE_EXECUTION.read_text()))
    execution.update({
        "inference_bundle": str(OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OVERLAY),
        "activation_authorization": str(AUTHORIZATION.relative_to(ROOT)),
        "activation_authorization_sha256": sha(AUTHORIZATION),
        "position_monitor": (
            "every 30 seconds: evaluate the prior completed-1m frozen policy "
            "threshold against remaining-size fresh directional executable VWAP "
            "(long sell-through-bids; short buy-through-asks), after recorded "
            "entry half-spread and directional adverse entry-slippage allowance; "
            "the completed-bar favourable extreme (high for long, low for short) "
            "updates MFE/trailing/smooth state only for the next interval; the "
            "50-bps directionally farther native last stop is the catastrophe "
            "backstop. Production authority remains long-only.",
        ),
        "version_note": (
            "v122: v98 combined runtime-only successor. It seals side-specific "
            "executable-VWAP exit primitives and the reviewed append-only "
            "feature-state snapshot guard. No model, feature/Geometry state, "
            "calibration, admission, auction, entry-economics or rich-policy "
            "parameter changed; production remains Kraken Futures long-only."
        ),
    })
    execution_hashes = dict(execution.get("runtime_code_sha256") or {})
    execution_hashes[
        "extreme_price_movements/inference/strict_r3_live_execution.py"
    ] = sha(ROOT / "extreme_price_movements/inference/strict_r3_live_execution.py")
    execution["runtime_code_sha256"] = execution_hashes
    execution["runtime_reseal_predecessors"] = [{
        "predecessor_inference_bundle": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "predecessor_inference_bundle_sha256": sha(SOURCE_OVERLAY),
        "current_inference_bundle_sha256": sha(OVERLAY),
        "allowed_runtime_code_paths": list(CHANGED),
        "added_runtime_code_paths": [],
        "reason": (
            "Reviewed runtime-only bridge from v97: no static stack contract "
            "changes; only the sealed feature-state snapshot and directional "
            "exit implementation hashes differ."
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
