#!/usr/bin/env python3
"""Seal the strict-R3 new-entry liquidation-headroom leverage successor.

The canonical score, dual admission, portfolio auction, feature/Geometry
state, and rich parent exit are preserved.  This successor only caps the
already-approved inverse policy-stop leverage using fresh Kraken Flex margin
figures and the public retail maintenance tier.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v116_"
    "bcf_current_dual_dynamic_stop_leverage.json"
)
OUT_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v117_"
    "bcf_current_dual_liquidation_headroom_leverage.json"
)
SOURCE_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260821_v44_"
    "v116_dynamic_stop_leverage.json"
)
OUT_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260821_v45_"
    "v117_liquidation_headroom_leverage.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v117_v141_"
    "dynamic_stop_leverage.json"
)
OUT_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v118_v142_"
    "liquidation_headroom_leverage.json"
)
OUT_RECEIPT = ROOT / (
    "data_perp/artifacts/strict_r3_v117_v142_liquidation_headroom_"
    "leverage_20260821_v1/seal_receipt.json"
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
        "v117: new-entry liquidation-headroom cap on the existing inverse "
        "policy-stop leverage. Models, Geometry/K9, scoring, dual MC1 "
        "admission, BCF auction priority, portfolio slots and rich exit remain "
        "unchanged."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": [RUNTIME_PATH],
        "reason": (
            "Cap new-entry leverage by current Kraken Flex maintenance headroom "
            "at the frozen parent-policy stop plus 50 bps stressed exit cost."
        ),
    }
    write_new(OUT_OVERLAY, overlay)

    authorization = copy.deepcopy(json.loads(SOURCE_AUTHORIZATION.read_text()))
    authorization.update({
        "authorization_source": (
            "User-authorized 2026-08-21 liquidation-headroom successor: new "
            "entries retain min(10, 66/SL%) and add only a fresh Kraken Flex "
            "maintenance-safe leverage cap. No scoring, admission, auction or "
            "exit-policy relaxation."
        ),
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
    })
    write_new(OUT_AUTHORIZATION, authorization)

    execution = copy.deepcopy(json.loads(SOURCE_EXECUTION.read_text()))
    sizing = dict(execution.get("leverage_sizing") or {})
    sizing["liquidation_headroom"] = {
        "enabled": True,
        "margin_schedule": "retail_margin_levels",
        "account_source": "kraken_flex_margin_equity_and_maintenance_margin",
        "minimum_margin_level_after_stress": 1.5,
        "stressed_exit_slippage_bps": 50.0,
        "formula": (
            "projected_margin_equity_after_parent_stop_and_stress >= "
            "1.5 * projected_maintenance_margin"
        ),
        "scope": "new_entries_only; fail_closed_if_margin_or_tier_unavailable",
    }
    sizing["formula"] = (
        "min(10, 66 / policy_stop_absolute_pct, liquidation_safe_leverage, "
        "market_initial_margin_max_leverage)"
    )
    execution.update({
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "activation_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "activation_authorization_sha256": sha(OUT_AUTHORIZATION),
        "leverage_sizing": sizing,
        "version_note": (
            "v142: new-entry leverage=min(10, 66/policy-stop%, "
            "liquidation-safe cap) using fresh Kraken Flex margin equity, "
            "maintenance margin, and the retail contract tier. Existing "
            "positions retain confirmed fill leverage; all model and exit "
            "semantics are unchanged."
        ),
    })
    runtime_hashes = dict(execution.get("runtime_code_sha256") or {})
    runtime_hashes[RUNTIME_PATH] = sha(ROOT / RUNTIME_PATH)
    execution["runtime_code_sha256"] = runtime_hashes
    execution["runtime_reseal_predecessors"] = list(
        execution.get("runtime_reseal_predecessors") or []
    ) + [{
        "successor_execution_semantics": (
            "inverse_policy_stop_absolute_pct_with_liquidation_headroom_v1"
        ),
        "predecessor_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "predecessor_execution_sha256": sha(SOURCE_EXECUTION),
        "predecessor_inference_bundle": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "predecessor_inference_bundle_sha256": sha(SOURCE_OVERLAY),
        "current_inference_bundle_sha256": sha(OUT_OVERLAY),
        "allowed_runtime_code_paths": [RUNTIME_PATH],
        "changed_execution_fields": ["leverage_sizing"],
        "new_entry_formula": sizing["formula"],
        "existing_position_policy": "retain_confirmed_fill_leverage",
        "selection_semantics_unchanged": True,
    }]
    write_new(OUT_EXECUTION, execution)

    receipt = {
        "schema": "strict_r3_liquidation_headroom_leverage_reseal_v1",
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
        "new_entry_formula": sizing["formula"],
        "headroom": sizing["liquidation_headroom"],
        "selection_semantics_unchanged": True,
    }
    write_new(OUT_RECEIPT, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
