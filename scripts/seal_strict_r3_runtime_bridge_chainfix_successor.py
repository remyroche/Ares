#!/usr/bin/env python3
"""Seal the runtime-only live-producer bridge-chain repair successor."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v106_"
    "bcf_current_dual_close_email_dual_mapper_fill_reporting.json"
)
OUT_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v107_"
    "bcf_current_dual_runtime_bridge_chainfix.json"
)
SOURCE_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260820_v34_"
    "v106_close_email_dual_mapper_fill_reporting.json"
)
OUT_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260820_v35_"
    "v107_runtime_bridge_chainfix.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v107_v131_"
    "close_email_dual_mapper_fill_reporting.json"
)
OUT_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v108_v132_"
    "runtime_bridge_chainfix.json"
)
CHANGED_RUNTIME = ("scripts/run_strict_r3_live_hourly_entry_producer.py",)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"immutable successor already exists: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    overlay = copy.deepcopy(json.loads(SOURCE_OVERLAY.read_text()))
    hashes = dict(overlay["overrides"].get("runtime_code_sha256") or {})
    for relative in CHANGED_RUNTIME:
        hashes[relative] = sha(ROOT / relative)
    overlay["overrides"]["runtime_code_sha256"] = hashes
    overlay["purpose"] = (
        "v107: runtime-only live-producer bridge-chain repair. Historical "
        "runtime-reseal provenance is ignored unless it targets the active "
        "bundle; matching bridges remain fully validated. No model, feature, "
        "calibration, admission, portfolio, entry, or exit-policy semantics change."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": list(CHANGED_RUNTIME),
        "reason": (
            "Correct append-only runtime-reseal bridge selection in the live "
            "hourly producer. Earlier historical bridges are provenance only."
        ),
    }
    write_new(OUT_OVERLAY, overlay)

    authorization = copy.deepcopy(json.loads(SOURCE_AUTHORIZATION.read_text()))
    authorization.update({
        "authorization_source": (
            "User-authorized 2026-08-20 runtime-only live-producer bridge-chain "
            "repair after the v131 producer failed closed before source work. "
            "All trading and policy semantics are unchanged."
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
        "version_note": (
            "v132: runtime-only live-producer runtime-reseal bridge-chain repair. "
            "No model, feature, Geometry/K9, calibration, admission, portfolio, "
            "entry, exit threshold, or parent-policy parameter change."
        ),
    })
    execution_hashes = dict(execution.get("runtime_code_sha256") or {})
    for relative in CHANGED_RUNTIME:
        execution_hashes[relative] = sha(ROOT / relative)
    execution["runtime_code_sha256"] = execution_hashes
    execution["runtime_reseal_predecessors"] = list(
        execution.get("runtime_reseal_predecessors") or []
    ) + [{
        "successor_execution_semantics": "runtime_reseal_bridge_chainfix_v1",
        "current_inference_bundle_sha256": sha(OUT_OVERLAY),
        "predecessor_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "predecessor_execution_sha256": sha(SOURCE_EXECUTION),
        "predecessor_inference_bundle": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "predecessor_inference_bundle_sha256": sha(SOURCE_OVERLAY),
        "allowed_runtime_code_paths": list(CHANGED_RUNTIME),
        "added_runtime_code_paths": [],
        "changed_execution_fields": [],
        "report_only": False,
    }]
    write_new(OUT_EXECUTION, execution)
    print(json.dumps({
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "activation_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "activation_authorization_sha256": sha(OUT_AUTHORIZATION),
        "execution": str(OUT_EXECUTION.relative_to(ROOT)),
        "execution_sha256": sha(OUT_EXECUTION),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
