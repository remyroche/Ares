#!/usr/bin/env python3
"""Seal the fixed 5x Flex/tier-data fallback successor."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v119_bcf_current_dual_liquidation_headroom_5x_fallback.json"
OUT_OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v120_bcf_current_dual_liquidation_headroom_fixed5x_fallback.json"
SOURCE_AUTHORIZATION = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260821_v47_v119_liquidation_headroom_5x_fallback.json"
OUT_AUTHORIZATION = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260821_v48_v120_liquidation_headroom_fixed5x_fallback.json"
SOURCE_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v121_v145_liquidation_headroom_5x_fallback.json"
OUT_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v122_v146_liquidation_headroom_fixed5x_fallback.json"
RECEIPT = ROOT / "data_perp/artifacts/strict_r3_v122_v146_liquidation_headroom_fixed5x_fallback_20260821_v1/seal_receipt.json"
RUNTIME_PATH = "extreme_price_movements/inference/strict_r3_live_execution.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    source_overlay = json.loads(SOURCE_OVERLAY.read_text())
    overlay = copy.deepcopy(source_overlay)
    overrides = dict(overlay.get("overrides") or {})
    hashes = dict(overrides.get("runtime_code_sha256") or {})
    hashes[RUNTIME_PATH] = sha(ROOT / RUNTIME_PATH)
    overrides["runtime_code_sha256"] = hashes
    overlay["overrides"] = overrides
    overlay["purpose"] = (
        "v120: preserve v119 and simplify the missing maintenance-margin or "
        "retail-tier contingency to a direct fixed 5x request. Model, admission, "
        "auction and parent-exit semantics are unchanged."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": [RUNTIME_PATH],
        "reason": "Direct fixed-5x fallback for unavailable maintenance margin or contract tier.",
    }
    write_new(OUT_OVERLAY, overlay)

    authorization = copy.deepcopy(json.loads(SOURCE_AUTHORIZATION.read_text()))
    authorization.update({
        "authorization_source": (
            "User-authorized 2026-08-21 successor: retain liquidation-headroom "
            "when inputs exist, otherwise request fixed 5x when maintenance-margin "
            "or retail-tier data are unavailable. No selection or exit change."
        ),
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
    })
    write_new(OUT_AUTHORIZATION, authorization)

    source_execution = json.loads(SOURCE_EXECUTION.read_text())
    execution = copy.deepcopy(source_execution)
    sizing = dict(execution.get("leverage_sizing") or {})
    headroom = dict(sizing.get("liquidation_headroom") or {})
    headroom["fallback"] = {
        "enabled": True,
        "maximum_leverage": 5.0,
        "maximum_snapshot_age_seconds": 300.0,
        "equity_haircut": 0.75,
        "maintenance_uplift": 1.25,
        "initial_margin_rate": 0.20,
        "maintenance_margin_rate": 0.10,
        "scope": (
            "new_entries_only; direct fixed 5x only when maintenance margin or "
            "retail tier is unavailable; still requires executable price, quantity, "
            "market metadata and positive slot margin. Cached Flex data are used only "
            "to keep portfolio-capacity allocation continuous."
        ),
    }
    sizing["liquidation_headroom"] = headroom
    execution.update({
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "activation_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "activation_authorization_sha256": sha(OUT_AUTHORIZATION),
        "leverage_sizing": sizing,
        "version_note": (
            "v146: v145 liquidation-headroom leverage remains primary. If maintenance "
            "margin or retail-tier data are unavailable, request exactly 5x; do not "
            "derive leverage from cached headroom data. Missing price, quantity, "
            "market metadata or slot margin still fails closed."
        ),
    })
    runtime_hashes = dict(execution.get("runtime_code_sha256") or {})
    runtime_hashes[RUNTIME_PATH] = sha(ROOT / RUNTIME_PATH)
    execution["runtime_code_sha256"] = runtime_hashes
    predecessors = list(execution.get("runtime_reseal_predecessors") or [])
    predecessors.append({
        "successor_execution_semantics": "liquidation_headroom_fixed_5x_data_fallback_v2",
        "predecessor_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "predecessor_execution_sha256": sha(SOURCE_EXECUTION),
        "predecessor_inference_bundle": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "predecessor_inference_bundle_sha256": sha(SOURCE_OVERLAY),
        "current_inference_bundle_sha256": sha(OUT_OVERLAY),
        "allowed_runtime_code_paths": [RUNTIME_PATH],
        "changed_execution_fields": ["leverage_sizing.liquidation_headroom.fallback"],
        "selection_semantics_unchanged": True,
        "existing_position_policy": "retain_confirmed_fill_leverage",
    })
    execution["runtime_reseal_predecessors"] = predecessors
    write_new(OUT_EXECUTION, execution)

    receipt = {
        "schema": "strict_r3_liquidation_headroom_fixed_5x_fallback_reseal_v2",
        "source_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "source_execution_sha256": sha(SOURCE_EXECUTION),
        "successor_execution": str(OUT_EXECUTION.relative_to(ROOT)),
        "successor_execution_sha256": sha(OUT_EXECUTION),
        "source_overlay": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "source_overlay_sha256": sha(SOURCE_OVERLAY),
        "successor_overlay": str(OUT_OVERLAY.relative_to(ROOT)),
        "successor_overlay_sha256": sha(OUT_OVERLAY),
        "successor_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "successor_authorization_sha256": sha(OUT_AUTHORIZATION),
        "fallback": headroom["fallback"],
        "selection_semantics_unchanged": True,
    }
    write_new(RECEIPT, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
