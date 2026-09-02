#!/usr/bin/env python3
"""Complete the hash-bound live successor for the VWAP exit sentinel.

The first execution-only sentinel receipt intentionally failed contract loading:
the canonical inference overlay also seals the live-execution runtime module.
This successor reseals that overlay, its existing manual authorization, and
the execution contract together.  It does not change any frozen model or
trading semantics beyond the user-approved exit sentinel.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v96_"
    "bcf_current_dual_prefix_schema_runtime_reseal.json"
)
OUT_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v97_"
    "bcf_current_dual_executable_vwap_sentinel.json"
)
SOURCE_AUTH = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260819_v24_"
    "v96_prefix_schema_runtime_reseal.json"
)
OUT_AUTH = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260819_v25_"
    "v97_executable_vwap_sentinel.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v95_v117_"
    "bcf_current_dual_v96_runtime_bridgefix.json"
)
OUT_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v97_v119_"
    "executable_vwap_sentinel.json"
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
    overlay_hashes = dict(overlay["overrides"]["runtime_code_sha256"])
    overlay_hashes[LIVE_EXECUTION] = sha(ROOT / LIVE_EXECUTION)
    overlay["overrides"]["runtime_code_sha256"] = overlay_hashes
    overlay["purpose"] = (
        "v97: user-approved frozen-threshold executable-VWAP exit sentinel. "
        "The only runtime change is the separately sealed live-exit module; "
        "no frozen model, feature, Geometry/K9, calibration, admission, "
        "auction, entry, or rich-policy parameter changes."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": [LIVE_EXECUTION],
        "reason": (
            "Hash-bind the approved 30-second executable-VWAP frozen-threshold "
            "exit sentinel. Its book polling has no model, feature, score, "
            "admission, portfolio, or entry authority."
        ),
    }
    write_new(OUT_OVERLAY, overlay)

    authorization = copy.deepcopy(json.loads(SOURCE_AUTH.read_text()))
    preserved = list(authorization.get("preserved_gates") or [])
    for gate in [
        "prior_completed_1m_frozen_threshold_only",
        "30_second_remaining_size_bid_vwap_exit_sentinel",
        "entry_half_spread_and_adverse_entry_slippage_allowance",
        "native_last_stop_50bps_lower_catastrophe_backstop",
    ]:
        if gate not in preserved:
            preserved.append(gate)
    authorization.update({
        "authorization_source": (
            "Explicit user approval on 2026-08-19 to add only the sealed "
            "30-second executable-VWAP frozen-threshold exit sentinel. It "
            "uses the prior completed 1m parent-policy threshold and recorded "
            "entry friction; the native Kraken last stop remains a lower "
            "catastrophe backstop."
        ),
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "preserved_gates": preserved,
    })
    write_new(OUT_AUTH, authorization)

    execution = copy.deepcopy(json.loads(SOURCE_EXECUTION.read_text()))
    if bool(execution.get("full_vwap_hard_stop_monitor_enabled", False)):
        raise ValueError("sentinel successor requires full-VWAP hard-stop disabled")
    if not bool(execution.get("close_based_hard_stop_monitor_enabled", False)):
        raise ValueError("sentinel successor requires the parent close-based controller")
    execution.update({
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "activation_authorization": str(OUT_AUTH.relative_to(ROOT)),
        "activation_authorization_sha256": sha(OUT_AUTH),
        "executable_vwap_frozen_threshold_sentinel_enabled": True,
        "native_last_stop_backstop_bps": float(
            execution["maximum_exit_slippage_bps"]
        ),
        "position_monitor": (
            "every 30 seconds: evaluate only the prior completed-1m frozen "
            "policy threshold against remaining-size fresh Kraken bid VWAP "
            "after recorded entry half-spread and adverse entry-slippage "
            "allowance; completed-bar high updates MFE/trailing/smooth state "
            "only for the next interval; 50-bps lower native last stop is "
            "the catastrophe backstop."
        ),
        "version_note": (
            "v119: complete hash-bound successor for the approved 30-second "
            "executable-VWAP frozen-threshold exit sentinel. The parent rich "
            "policy remains completed-bar/high based; Kraken native last stop "
            "remains a 50-bps lower catastrophe backstop. Frozen models, "
            "Geometry/K9, calibration, admission, auction, entries, policy "
            "parameters and all entry gates remain unchanged."
        ),
    })
    execution_hashes = dict(execution["runtime_code_sha256"])
    execution_hashes[LIVE_EXECUTION] = sha(ROOT / LIVE_EXECUTION)
    execution["runtime_code_sha256"] = execution_hashes
    execution["runtime_reseal_predecessors"] = list(
        execution.get("runtime_reseal_predecessors") or []
    ) + [{
        "successor_execution_semantics": "frozen_threshold_executable_vwap_sentinel_v1",
        "predecessor_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "predecessor_execution_sha256": sha(SOURCE_EXECUTION),
        "predecessor_inference_bundle": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "predecessor_inference_bundle_sha256": sha(SOURCE_OVERLAY),
        "allowed_runtime_code_paths": [LIVE_EXECUTION],
        "changed_execution_fields": [
            "executable_vwap_frozen_threshold_sentinel_enabled",
            "native_last_stop_backstop_bps",
            "position_monitor",
        ],
        "native_backstop_bps_source": "maximum_exit_slippage_bps",
    }]
    write_new(OUT_EXECUTION, execution)

    print(json.dumps({
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "activation_authorization": str(OUT_AUTH.relative_to(ROOT)),
        "activation_authorization_sha256": sha(OUT_AUTH),
        "execution": str(OUT_EXECUTION.relative_to(ROOT)),
        "execution_sha256": sha(OUT_EXECUTION),
        "runtime_hash": execution_hashes[LIVE_EXECUTION],
        "native_last_stop_backstop_bps": execution["native_last_stop_backstop_bps"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
