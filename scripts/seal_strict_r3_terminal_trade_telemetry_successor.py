#!/usr/bin/env python3
"""Seal the runtime-only Strict-R3 terminal-trade telemetry successor.

This successor binds the durable close ledger to the current v146 live stack.
It permits exactly one runtime source change: the execution module that copies
entry/exit microstructure and realised-PnL evidence into an immutable terminal
trade record.  Frozen models, features, Geometry/K9, calibration, admission,
auction, sizing, and exit-policy parameters remain byte-equivalent.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v120_"
    "bcf_current_dual_liquidation_headroom_fixed5x_fallback.json"
)
OUT_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v121_"
    "bcf_current_dual_liquidation_headroom_fixed5x_terminal_trade_telemetry.json"
)
SOURCE_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260821_v48_"
    "v120_liquidation_headroom_fixed5x_fallback.json"
)
OUT_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260822_v49_"
    "v121_terminal_trade_telemetry.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v122_v146_"
    "liquidation_headroom_fixed5x_fallback.json"
)
OUT_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v123_v147_"
    "terminal_trade_telemetry.json"
)
OUT_REVIEW = ROOT / (
    "data_perp/artifacts/strict_r3_terminal_trade_telemetry_reseal_20260822_v1/"
    "runtime_review.json"
)
CHANGED_RUNTIME = "extreme_price_movements/inference/strict_r3_live_execution.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def static_overlay(payload: dict) -> dict:
    output = copy.deepcopy(payload)
    output.pop("purpose", None)
    output.pop("runtime_reseal", None)
    output.get("overrides", {}).pop("runtime_code_sha256", None)
    return output


def main() -> None:
    source_overlay = json.loads(SOURCE_OVERLAY.read_text())
    source_execution = json.loads(SOURCE_EXECUTION.read_text())
    expected_hashes = dict(source_execution.get("runtime_code_sha256") or {})
    if not expected_hashes:
        raise ValueError("source execution contract has no runtime code hashes")
    actual_hashes = {
        relative: sha(ROOT / relative) for relative in sorted(expected_hashes)
    }
    changed = sorted(
        relative for relative in expected_hashes
        if expected_hashes[relative] != actual_hashes[relative]
    )
    if changed != [CHANGED_RUNTIME]:
        raise ValueError(
            "terminal telemetry reseal permits only "
            f"{CHANGED_RUNTIME}; got {changed}"
        )

    overlay = copy.deepcopy(source_overlay)
    overlay.setdefault("overrides", {}).setdefault("runtime_code_sha256", {})[
        CHANGED_RUNTIME
    ] = actual_hashes[CHANGED_RUNTIME]
    overlay["purpose"] = (
        "v121: runtime-only terminal trade telemetry persistence. Each confirmed "
        "full close writes a self-contained entry/exit market, spread, slippage, "
        "book, VWAP and realised-PnL record before notification delivery. No "
        "model, feature, Geometry/K9, calibration, admission, auction, sizing, "
        "or exit-policy parameter changes."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": changed,
        "reason": (
            "Persist self-contained, immutable terminal-trade execution telemetry "
            "rather than relying on mutable state-successor reconstruction."
        ),
    }
    if static_overlay(source_overlay) != static_overlay(overlay):
        raise AssertionError("telemetry successor changed non-runtime inference semantics")
    write_new(OUT_OVERLAY, overlay)

    authorization = copy.deepcopy(json.loads(SOURCE_AUTHORIZATION.read_text()))
    authorization.update({
        "authorization_source": (
            "User-authorized terminal-trade telemetry persistence repair. Frozen "
            "models, feature contracts, EV mapping, admission, auction, sizing and "
            "exit policy are unchanged."
        ),
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
    })
    write_new(OUT_AUTHORIZATION, authorization)

    execution = copy.deepcopy(source_execution)
    execution.update({
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "activation_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "activation_authorization_sha256": sha(OUT_AUTHORIZATION),
        "runtime_code_sha256": actual_hashes,
        "version_note": (
            "v147: terminal trade telemetry persistence only. The ledger stores "
            "entry/exit books, spread, slippage, VWAP, fill and PnL evidence before "
            "close notification. No executable scoring, admission, auction, sizing "
            "or parent-policy semantics changed."
        ),
    })
    execution["runtime_reseal_predecessors"] = list(
        execution.get("runtime_reseal_predecessors") or []
    ) + [{
        "successor_execution_semantics": "terminal_trade_telemetry_ledger_v1",
        "predecessor_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "predecessor_execution_sha256": sha(SOURCE_EXECUTION),
        "predecessor_inference_bundle": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "predecessor_inference_bundle_sha256": sha(SOURCE_OVERLAY),
        "allowed_runtime_code_paths": changed,
        "added_runtime_code_paths": [],
        "changed_execution_fields": [],
        "reviewed_current_runtime": True,
    }]
    write_new(OUT_EXECUTION, execution)

    review = {
        "schema": "strict_r3_terminal_trade_telemetry_runtime_review_v1",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "source_overlay": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "source_overlay_sha256": sha(SOURCE_OVERLAY),
        "source_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "source_execution_sha256": sha(SOURCE_EXECUTION),
        "changed_runtime_paths": changed,
        "previous_runtime_hashes": {
            relative: expected_hashes[relative] for relative in changed
        },
        "current_runtime_hashes": {
            relative: actual_hashes[relative] for relative in changed
        },
        "non_runtime_inference_contract_exact": (
            static_overlay(source_overlay) == static_overlay(overlay)
        ),
        "successor_overlay": str(OUT_OVERLAY.relative_to(ROOT)),
        "successor_overlay_sha256": sha(OUT_OVERLAY),
        "successor_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "successor_authorization_sha256": sha(OUT_AUTHORIZATION),
        "successor_execution": str(OUT_EXECUTION.relative_to(ROOT)),
        "successor_execution_sha256": sha(OUT_EXECUTION),
    }
    write_new(OUT_REVIEW, review)
    print(json.dumps(review, sort_keys=True))


if __name__ == "__main__":
    main()
