#!/usr/bin/env python3
"""Seal the long-only inverse-policy-stop leverage live successor.

The scorer, Geometry/K9 state, admission, auction, and exit geometry remain
unchanged.  Only new-entry Kraken leverage and quote notional change:

    leverage = min(10, 66 / policy_stop_absolute_percent)

where the stop percentage comes from the frozen parent policy, immutable
signal ATR, and fresh pre-order executable reference price.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v115_"
    "bcf_current_dual_geometry_state_bridge.json"
)
OUT_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v116_"
    "bcf_current_dual_dynamic_stop_leverage.json"
)
SOURCE_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260821_v43_"
    "v115_geometry_state_bridge.json"
)
OUT_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260821_v44_"
    "v116_dynamic_stop_leverage.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v116_v140_"
    "geometry_state_bridge.json"
)
OUT_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v117_v141_"
    "dynamic_stop_leverage.json"
)
OUT_RECEIPT = ROOT / (
    "data_perp/artifacts/strict_r3_v116_v141_dynamic_stop_leverage_"
    "20260821_v1/seal_receipt.json"
)
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
    runtime_hashes = dict(overrides.get("runtime_code_sha256") or {})
    runtime_hashes[RUNTIME_PATH] = sha(ROOT / RUNTIME_PATH)
    overrides["runtime_code_sha256"] = runtime_hashes
    overlay["overrides"] = overrides
    overlay["purpose"] = (
        "v116: new-entry inverse parent-policy-stop leverage only. Geometry/K9, "
        "models, scoring, dual MC1 admission, BCF auction priority, portfolio "
        "slots, and rich exit policy are unchanged."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": [RUNTIME_PATH],
        "reason": (
            "Use new-entry leverage=min(10, 66 / policy-stop absolute percent), "
            "resolved from immutable signal ATR and fresh pre-order policy geometry."
        ),
    }
    write_new(OUT_OVERLAY, overlay)

    authorization = copy.deepcopy(json.loads(SOURCE_AUTHORIZATION.read_text()))
    authorization.update({
        "authorization_source": (
            "User-authorized 2026-08-21 dynamic leverage successor: new entries "
            "use inverse frozen parent-policy stop distance; no scoring, admission, "
            "portfolio or exit-policy relaxation."
        ),
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
    })
    write_new(OUT_AUTHORIZATION, authorization)

    execution = copy.deepcopy(json.loads(SOURCE_EXECUTION.read_text()))
    execution.update({
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "activation_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "activation_authorization_sha256": sha(OUT_AUTHORIZATION),
        "leverage": 10.0,
        "leverage_sizing": {
            "mode": "inverse_policy_stop_absolute_pct",
            "risk_budget_pct": 66.0,
            "maximum_leverage": 10.0,
            "stop_reference": "fresh_preorder_policy_stop_absolute_pct",
            "formula": "min(10, 66 / policy_stop_absolute_pct)",
            "scope": "new_entries_only; existing_positions_retain_confirmed_fill_leverage",
        },
        "version_note": (
            "v141: new-entry leverage=min(10, 66 / policy-stop absolute percent); "
            "all scoring, admission, auction, state and exit semantics unchanged."
        ),
    })
    runtime_hashes = dict(execution.get("runtime_code_sha256") or {})
    runtime_hashes[RUNTIME_PATH] = sha(ROOT / RUNTIME_PATH)
    execution["runtime_code_sha256"] = runtime_hashes
    execution["runtime_reseal_predecessors"] = list(
        execution.get("runtime_reseal_predecessors") or []
    ) + [{
        "successor_execution_semantics": "inverse_policy_stop_absolute_pct_leverage_v1",
        "predecessor_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "predecessor_execution_sha256": sha(SOURCE_EXECUTION),
        "predecessor_inference_bundle": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "predecessor_inference_bundle_sha256": sha(SOURCE_OVERLAY),
        "current_inference_bundle_sha256": sha(OUT_OVERLAY),
        "allowed_runtime_code_paths": [RUNTIME_PATH],
        "changed_execution_fields": ["leverage", "leverage_sizing"],
        "new_entry_formula": "min(10, 66 / policy_stop_absolute_pct)",
        "existing_position_policy": "retain_confirmed_fill_leverage",
        "selection_semantics_unchanged": True,
    }]
    write_new(OUT_EXECUTION, execution)

    receipt = {
        "schema": "strict_r3_dynamic_stop_leverage_reseal_v1",
        "source_overlay": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "source_overlay_sha256": sha(SOURCE_OVERLAY),
        "source_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "source_execution_sha256": sha(SOURCE_EXECUTION),
        "successor_overlay": str(OUT_OVERLAY.relative_to(ROOT)),
        "successor_overlay_sha256": sha(OUT_OVERLAY),
        "successor_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "successor_authorization_sha256": sha(OUT_AUTHORIZATION),
        "successor_execution": str(OUT_EXECUTION.relative_to(ROOT)),
        "successor_execution_sha256": sha(OUT_EXECUTION),
        "formula": "leverage = min(10, 66 / policy_stop_absolute_pct)",
        "selection_semantics_unchanged": True,
    }
    write_new(OUT_RECEIPT, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
