#!/usr/bin/env python3
"""Seal the reporting-only dual-MC1 and confirmed-fill close-email successor."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v105_"
    "bcf_current_dual_openpositions503_fills_fallback.json"
)
OUT_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v106_"
    "bcf_current_dual_close_email_dual_mapper_fill_reporting.json"
)
SOURCE_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260820_v33_"
    "v105_openpositions503_fills_fallback.json"
)
OUT_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260820_v34_"
    "v106_close_email_dual_mapper_fill_reporting.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v106_v130_"
    "openpositions503_fills_fallback.json"
)
OUT_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v107_v131_"
    "close_email_dual_mapper_fill_reporting.json"
)
CHANGED_RUNTIME = (
    "extreme_price_movements/inference/canonical_stack_reporting.py",
    "extreme_price_movements/inference/run_inference.py",
    "extreme_price_movements/inference/strict_r3_live_execution.py",
)


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
        "v106: reporting-only close-email correction. The email now reports "
        "the sealed dual BCF/current-v5 MC1 authority, removes obsolete "
        "corrected-EV presentation, expresses MFE in ATR and price percent, "
        "and labels actual Kraken fill-based PnL. No trading semantics change."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": list(CHANGED_RUNTIME),
        "reason": (
            "Reporting and close-reconciliation presentation only. No model, "
            "feature, Geometry/K9, calibration, admission, auction, entry, "
            "exit threshold, or policy parameter changes."
        ),
    }
    write_new(OUT_OVERLAY, overlay)

    authorization = copy.deepcopy(json.loads(SOURCE_AUTHORIZATION.read_text()))
    authorization.update({
        "authorization_source": (
            "User-approved 2026-08-20 reporting-only successor: close emails "
            "show the active dual BCF/current-v5 MC1 authority and confirmed "
            "Kraken fill PnL labels. Trading authority and all gates are unchanged."
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
            "v131: reporting-only dual-MC1 close-email and confirmed-fill PnL "
            "presentation correction. No model, feature, admission, portfolio, "
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
        "successor_execution_semantics": "close_email_dual_mapper_fill_reporting_v1",
        "predecessor_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "predecessor_execution_sha256": sha(SOURCE_EXECUTION),
        "predecessor_inference_bundle": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "predecessor_inference_bundle_sha256": sha(SOURCE_OVERLAY),
        "allowed_runtime_code_paths": list(CHANGED_RUNTIME),
        "changed_execution_fields": [],
        "report_only": True,
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
