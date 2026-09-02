#!/usr/bin/env python3
"""Seal the monitor-only mixed Kraken exit-reconciliation successor.

This successor changes neither inference nor order-entry semantics.  It only
allows the position monitor to close a tracked state row when Kraken's private
ledger proves the whole quantity was closed by a *mixed* sequence of ordinary
reduce-only and liquidation fills.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v117_bcf_current_dual_liquidation_headroom_leverage.json"
OUT_OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v118_bcf_current_dual_liquidation_headroom_mixed_exit_reconciliation.json"
SOURCE_AUTHORIZATION = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260821_v45_v117_liquidation_headroom_leverage.json"
OUT_AUTHORIZATION = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260821_v46_v118_liquidation_headroom_mixed_exit_reconciliation.json"
SOURCE = ROOT / "config/strict_r3_kraken_live_execution_v118_v142_liquidation_headroom_leverage.json"
OUT = ROOT / "config/strict_r3_kraken_live_execution_v120_v144_liquidation_headroom_mixed_exit_reconciliation.json"
RECEIPT = ROOT / "data_perp/artifacts/strict_r3_v120_v144_mixed_exit_reconciliation_20260821_v1/seal_receipt.json"
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
    overlay_hashes = dict(overrides.get("runtime_code_sha256") or {})
    overlay_hashes[RUNTIME_PATH] = sha(ROOT / RUNTIME_PATH)
    overrides["runtime_code_sha256"] = overlay_hashes
    overlay["overrides"] = overrides
    overlay["purpose"] = (
        "v118: preserve v117's liquidation-headroom leverage and reseal the "
        "runtime for exact mixed ordinary/liquidation exit reconciliation. "
        "Models, Geometry/K9, scoring, dual MC1 admission, BCF auction "
        "priority, portfolio slots and rich policy are unchanged."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": [RUNTIME_PATH],
        "reason": "Monitor-only exact mixed private-fill reconciliation.",
    }
    write_new(OUT_OVERLAY, overlay)

    authorization = copy.deepcopy(json.loads(SOURCE_AUTHORIZATION.read_text()))
    authorization.update({
        "authorization_source": (
            "User-authorized 2026-08-21 liquidation-headroom successor, "
            "resealed for exact monitor-only mixed ordinary/liquidation fill "
            "reconciliation. No scoring, admission, auction, leverage, or "
            "exit-policy relaxation."
        ),
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
    })
    write_new(OUT_AUTHORIZATION, authorization)

    source = json.loads(SOURCE.read_text())
    successor = copy.deepcopy(source)
    runtime_hashes = dict(successor.get("runtime_code_sha256") or {})
    runtime_hashes[RUNTIME_PATH] = sha(ROOT / RUNTIME_PATH)
    successor["runtime_code_sha256"] = runtime_hashes
    successor.update({
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "activation_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "activation_authorization_sha256": sha(OUT_AUTHORIZATION),
    })
    successor["version_note"] = (
        "v143: preserve the v142 liquidation-headroom new-entry cap and add "
        "only exact private-ledger reconciliation for a fully closed position "
        "whose opposite-side fills mix ordinary reduce-only and liquidation "
        "fills. Inference, admission, auction, leverage formula and exit "
        "thresholds are unchanged."
    )
    predecessors = list(successor.get("runtime_reseal_predecessors") or [])
    predecessors.append({
        "successor_execution_semantics": "mixed_liquidation_external_exit_reconciliation_v1",
        "predecessor_execution": str(SOURCE.relative_to(ROOT)),
        "predecessor_execution_sha256": sha(SOURCE),
        "predecessor_inference_bundle": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "predecessor_inference_bundle_sha256": sha(SOURCE_OVERLAY),
        "current_inference_bundle_sha256": sha(OUT_OVERLAY),
        "allowed_runtime_code_paths": [RUNTIME_PATH],
        "changed_execution_fields": [],
        "monitor_only": True,
        "selection_semantics_unchanged": True,
        "reconciliation_gate": (
            "exact tracked amount must equal all post-entry opposite-side "
            "private fills, with both ordinary and liquidation fill families"
        ),
    })
    successor["runtime_reseal_predecessors"] = predecessors
    write_new(OUT, successor)
    receipt = {
        "schema": "strict_r3_mixed_exit_reconciliation_reseal_v1",
        "source_execution": str(SOURCE.relative_to(ROOT)),
        "source_execution_sha256": sha(SOURCE),
        "source_overlay": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "source_overlay_sha256": sha(SOURCE_OVERLAY),
        "successor_overlay": str(OUT_OVERLAY.relative_to(ROOT)),
        "successor_overlay_sha256": sha(OUT_OVERLAY),
        "successor_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "successor_authorization_sha256": sha(OUT_AUTHORIZATION),
        "successor_execution": str(OUT.relative_to(ROOT)),
        "successor_execution_sha256": sha(OUT),
        "changed_runtime_paths": [RUNTIME_PATH],
        "selection_semantics_unchanged": True,
        "monitor_only": True,
    }
    write_new(RECEIPT, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
