#!/usr/bin/env python3
"""Seal an explicit v113-to-current Geometry/K9 runtime bridge.

This repairs successor lineage metadata only.  The live scorer already rejects
non-adjacent Geometry/K9 state correctly; this bridge tells the warm producer
that the completed v113 state is compatible with the reporting-only runtime
successor, rather than falling back to v112.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v114_"
    "bcf_current_dual_close_notification_fill_reporting.json"
)
PREDECESSOR_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v113_"
    "bcf_current_dual_exit_replay_comparison_audit.json"
)
OUT_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v115_"
    "bcf_current_dual_geometry_state_bridge.json"
)
SOURCE_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260821_v42_"
    "v114_close_notification_fill_reporting.json"
)
OUT_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260821_v43_"
    "v115_geometry_state_bridge.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v115_v139_"
    "close_notification_fill_reporting.json"
)
OUT_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v116_v140_"
    "geometry_state_bridge.json"
)
OUT_RECEIPT = ROOT / (
    "data_perp/artifacts/strict_r3_v115_v140_geometry_state_bridge_"
    "20260821_v1/seal_receipt.json"
)
RUNTIME_PATHS = (
    "extreme_price_movements/inference/run_inference.py",
    "extreme_price_movements/inference/strict_r3_live_execution.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def static_contract(payload: dict) -> dict:
    result = copy.deepcopy(payload)
    result.pop("purpose", None)
    result.pop("runtime_reseal", None)
    result.get("overrides", {}).pop("runtime_code_sha256", None)
    return result


def main() -> None:
    source_overlay = json.loads(SOURCE_OVERLAY.read_text())
    predecessor_overlay = json.loads(PREDECESSOR_OVERLAY.read_text())
    if static_contract(source_overlay) != static_contract(predecessor_overlay):
        raise AssertionError("v113 and v114 do not share an identical static contract")

    overlay = copy.deepcopy(source_overlay)
    overlay["purpose"] = (
        "v115: explicit strict runtime bridge from the completed v113 Geometry/K9 "
        "state to the current reporting-only successor. No code, model, feature, "
        "calibration, admission, portfolio, entry or exit-policy change."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": [],
        "reason": "Declare the exact compatible v113 Geometry/K9 predecessor state.",
    }
    if static_contract(source_overlay) != static_contract(overlay):
        raise AssertionError("bridge successor changed static inference semantics")
    write_new(OUT_OVERLAY, overlay)

    authorization = copy.deepcopy(json.loads(SOURCE_AUTHORIZATION.read_text()))
    authorization.update({
        "authorization_source": (
            "User-authorized 2026-08-21 Geometry/K9 state-chain repair: use the "
            "completed v113 predecessor explicitly. Trading authority and every "
            "economic rule remain unchanged."
        ),
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
    })
    write_new(OUT_AUTHORIZATION, authorization)

    source_execution = json.loads(SOURCE_EXECUTION.read_text())
    execution = copy.deepcopy(source_execution)
    execution.update({
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "activation_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "activation_authorization_sha256": sha(OUT_AUTHORIZATION),
        "version_note": (
            "v140: explicit v113 Geometry/K9 state bridge; no execution, "
            "model, policy, admission or portfolio semantics change."
        ),
    })
    execution["runtime_reseal_predecessors"] = list(
        execution.get("runtime_reseal_predecessors") or []
    ) + [{
        "successor_execution_semantics": "geometry_k9_state_bridge_v1",
        "current_inference_bundle_sha256": sha(OUT_OVERLAY),
        "predecessor_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "predecessor_execution_sha256": sha(SOURCE_EXECUTION),
        "predecessor_inference_bundle": str(PREDECESSOR_OVERLAY.relative_to(ROOT)),
        "predecessor_inference_bundle_sha256": sha(PREDECESSOR_OVERLAY),
        "allowed_runtime_code_paths": list(RUNTIME_PATHS),
        "added_runtime_code_paths": [],
        "changed_execution_fields": [],
        "report_only": True,
    }]
    write_new(OUT_EXECUTION, execution)

    receipt = {
        "schema": "strict_r3_geometry_k9_state_bridge_reseal_v1",
        "source_overlay": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "source_overlay_sha256": sha(SOURCE_OVERLAY),
        "compatible_predecessor_overlay": str(PREDECESSOR_OVERLAY.relative_to(ROOT)),
        "compatible_predecessor_overlay_sha256": sha(PREDECESSOR_OVERLAY),
        "source_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "source_execution_sha256": sha(SOURCE_EXECUTION),
        "static_contract_exact": static_contract(source_overlay) == static_contract(overlay),
        "successor_overlay": str(OUT_OVERLAY.relative_to(ROOT)),
        "successor_overlay_sha256": sha(OUT_OVERLAY),
        "successor_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "successor_authorization_sha256": sha(OUT_AUTHORIZATION),
        "successor_execution": str(OUT_EXECUTION.relative_to(ROOT)),
        "successor_execution_sha256": sha(OUT_EXECUTION),
    }
    write_new(OUT_RECEIPT, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
