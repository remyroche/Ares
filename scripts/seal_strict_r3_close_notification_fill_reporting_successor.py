#!/usr/bin/env python3
"""Seal the reporting-only successor for terminal close emails.

The successor preserves every model, feature, calibration, admission,
portfolio and exit-policy parameter.  It only permits two runtime changes:
one durable, cross-state notification ledger for confirmed full closes, and
correct rendering of confirmed Kraken-fill gross PnL when fees are absent.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v113_"
    "bcf_current_dual_exit_replay_comparison_audit.json"
)
OUT_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v114_"
    "bcf_current_dual_close_notification_fill_reporting.json"
)
SOURCE_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260821_v41_"
    "v113_exit_replay_comparison_audit.json"
)
OUT_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260821_v42_"
    "v114_close_notification_fill_reporting.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v114_v138_"
    "exit_replay_comparison_audit.json"
)
OUT_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v115_v139_"
    "close_notification_fill_reporting.json"
)
OUT_RECEIPT = ROOT / (
    "data_perp/artifacts/strict_r3_v114_v139_close_notification_fill_reporting_"
    "20260821_v1/seal_receipt.json"
)
CHANGED_RUNTIME = (
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
    overlay = copy.deepcopy(source_overlay)
    hashes = dict(overlay["overrides"].get("runtime_code_sha256") or {})
    for relative in CHANGED_RUNTIME:
        hashes[relative] = sha(ROOT / relative)
    overlay["overrides"]["runtime_code_sha256"] = hashes
    overlay["purpose"] = (
        "v114: terminal close-notification integrity and confirmed-fill gross "
        "PnL rendering. One close email only after a confirmed full closure, "
        "including across live-state migrations; no model or trading semantics change."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": list(CHANGED_RUNTIME),
        "reason": (
            "Persist one notification identity per confirmed full close and "
            "render confirmed Kraken-fill gross PnL when exchange fees are absent."
        ),
    }
    if static_contract(source_overlay) != static_contract(overlay):
        raise AssertionError("runtime successor changed static inference semantics")
    write_new(OUT_OVERLAY, overlay)

    authorization = copy.deepcopy(json.loads(SOURCE_AUTHORIZATION.read_text()))
    authorization.update({
        "authorization_source": (
            "User-authorized 2026-08-21 reporting repair: exactly one terminal "
            "email after a confirmed full close, with confirmed-fill gross PnL. "
            "No trading authority or economic rule changes."
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
            "v139: reporting-only terminal close notification deduplication and "
            "confirmed Kraken-fill gross PnL labels; no execution authority, "
            "entry, exit, portfolio or policy change."
        ),
    })
    execution_hashes = dict(execution.get("runtime_code_sha256") or {})
    for relative in execution_hashes:
        execution_hashes[relative] = sha(ROOT / relative)
    execution["runtime_code_sha256"] = execution_hashes
    execution["runtime_reseal_predecessors"] = list(
        execution.get("runtime_reseal_predecessors") or []
    ) + [{
        "successor_execution_semantics": "close_notification_fill_reporting_v1",
        "predecessor_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "predecessor_execution_sha256": sha(SOURCE_EXECUTION),
        "predecessor_inference_bundle": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "predecessor_inference_bundle_sha256": sha(SOURCE_OVERLAY),
        "allowed_runtime_code_paths": list(CHANGED_RUNTIME),
        "added_runtime_code_paths": [],
        "changed_execution_fields": [],
        "report_only": True,
    }]
    write_new(OUT_EXECUTION, execution)

    receipt = {
        "schema": "strict_r3_close_notification_fill_reporting_reseal_v1",
        "source_overlay": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "source_overlay_sha256": sha(SOURCE_OVERLAY),
        "source_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "source_execution_sha256": sha(SOURCE_EXECUTION),
        "changed_runtime_paths": list(CHANGED_RUNTIME),
        "runtime_code_sha256": {relative: sha(ROOT / relative) for relative in CHANGED_RUNTIME},
        "static_inference_contract_exact": static_contract(source_overlay) == static_contract(overlay),
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
