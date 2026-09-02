#!/usr/bin/env python3
"""Seal the narrow `/openpositions` 503 → `/fills` monitor fallback.

The successor changes no model, feature, Geometry/K9, calibration, admission,
auction, entry, or parent exit-policy parameter.  It only permits the live
position monitor to reconstruct *already tracked* inventory from Kraken's
authenticated `/fills` response when the primary `/openpositions` endpoint
returns HTTP 503.  Any incomplete entry-fill coverage or amount mismatch
remains fail-closed.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v104_"
    "bcf_current_dual_feature_state_snapshot_reseal.json"
)
OUT_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v105_"
    "bcf_current_dual_openpositions503_fills_fallback.json"
)
SOURCE_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260820_v32_"
    "v103_feature_state_snapshot_reseal.json"
)
OUT_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260820_v33_"
    "v105_openpositions503_fills_fallback.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v104_v128_"
    "feature_state_snapshot_reseal.json"
)
OUT_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v106_v130_"
    "openpositions503_fills_fallback.json"
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
    runtime_hashes = dict(overlay["overrides"].get("runtime_code_sha256") or {})
    runtime_hashes[LIVE_EXECUTION] = sha(ROOT / LIVE_EXECUTION)
    overlay["overrides"]["runtime_code_sha256"] = runtime_hashes
    overlay["purpose"] = (
        "v105: user-approved Kraken /openpositions HTTP-503 fail-closed "+
        "/fills fallback for already-tracked live positions only. No model, "
        "feature, Geometry/K9, calibration, admission, portfolio, entry, or "
        "parent-policy semantics change."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": [LIVE_EXECUTION],
        "reason": (
            "Hash-bind the narrow position-monitor availability fallback. It is "
            "available only after an /openpositions 503 and only reconstructs "
            "persisted tracked inventory from complete authenticated fill evidence."
        ),
    }
    write_new(OUT_OVERLAY, overlay)

    authorization = copy.deepcopy(json.loads(SOURCE_AUTHORIZATION.read_text()))
    preserved = list(authorization.get("preserved_gates") or [])
    gate = "openpositions_503_tracked_fills_only_fail_closed_fallback"
    if gate not in preserved:
        preserved.append(gate)
    authorization.update({
        "authorization_source": (
            "Explicit user approval on 2026-08-20 for a narrow availability "
            "fallback: only the live minute monitor may use authenticated Kraken "
            "/fills after an /openpositions HTTP 503, and only to verify "
            "already-tracked positions with complete entry-fill coverage."
        ),
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "preserved_gates": preserved,
    })
    write_new(OUT_AUTHORIZATION, authorization)

    execution = copy.deepcopy(json.loads(SOURCE_EXECUTION.read_text()))
    execution.update({
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "activation_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "activation_authorization_sha256": sha(OUT_AUTHORIZATION),
        "openpositions_503_fills_fallback_enabled": True,
        "position_monitor": list(execution.get("position_monitor") or []) + [
            "When Kraken /openpositions specifically returns HTTP 503, the "
            "minute monitor may reconstruct only already-tracked position "
            "inventory from authenticated /fills. Every tracked entry fill must "
            "be present and its signed post-entry amount must be exactly open or "
            "zero; anything else fails closed. The fallback never discovers "
            "positions and never grants entry authority."
        ],
        "version_note": (
            "v130: sealed fail-closed Kraken /openpositions 503 fallback to "
            "authenticated /fills for the minute monitor's existing tracked "
            "positions only. All model, feature, calibration, admission, "
            "portfolio, entry, and rich-policy semantics are unchanged."
        ),
    })
    execution_hashes = dict(execution.get("runtime_code_sha256") or {})
    execution_hashes[LIVE_EXECUTION] = sha(ROOT / LIVE_EXECUTION)
    execution["runtime_code_sha256"] = execution_hashes
    execution["runtime_reseal_predecessors"] = list(
        execution.get("runtime_reseal_predecessors") or []
    ) + [{
        "successor_execution_semantics": "openpositions_503_fills_monitor_fallback_v1",
        "predecessor_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "predecessor_execution_sha256": sha(SOURCE_EXECUTION),
        "predecessor_inference_bundle": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "predecessor_inference_bundle_sha256": sha(SOURCE_OVERLAY),
        "allowed_runtime_code_paths": [LIVE_EXECUTION],
        "changed_execution_fields": ["openpositions_503_fills_fallback_enabled"],
        "fallback_scope": "minute_monitor_existing_tracked_positions_only",
        "fallback_trigger": "kraken_openpositions_http_503_only",
        "fallback_failure_mode": "fail_closed",
    }]
    write_new(OUT_EXECUTION, execution)

    print(json.dumps({
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "activation_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "activation_authorization_sha256": sha(OUT_AUTHORIZATION),
        "execution": str(OUT_EXECUTION.relative_to(ROOT)),
        "execution_sha256": sha(OUT_EXECUTION),
        "live_execution_runtime_sha256": execution_hashes[LIVE_EXECUTION],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
