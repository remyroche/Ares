#!/usr/bin/env python3
"""Seal the user-approved live microstructure-plus-10-bps entry revision."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD_OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v63_bcf_current_dual_mc1_latency_bcf_resealed.json"
OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v64_bcf_current_dual_mc1_microstructure_buffer10.json"
OLD_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v62_v72_bcf_current_dual_smooth_latency_bcf_resealed.json"
EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v63_v73_bcf_current_dual_microstructure_buffer10.json"
OLD_AUTHORIZATION = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260817_v27_bcf_current_dual_smooth_latency_bcf_resealed.json"
AUTHORIZATION = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260817_v28_bcf_current_dual_microstructure_buffer10.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    overlay = copy.deepcopy(json.loads(OLD_OVERLAY.read_text()))
    overlay["purpose"] = (
        "approved BCF scorer/latency successor with live entry friction equal to "
        "book-derived microstructure friction plus a fixed 10-bps buffer"
    )
    overlay["overrides"]["runtime_code_sha256"][
        "extreme_price_movements/inference/strict_r3_live_execution.py"
    ] = sha(ROOT / "extreme_price_movements/inference/strict_r3_live_execution.py")
    write_new(OVERLAY, overlay)

    execution = copy.deepcopy(json.loads(OLD_EXECUTION.read_text()))
    execution.update({
        "version_note": (
            "v73: user-approved live entry friction revision. Deduct full observed "
            "microstructure friction, adverse decision-to-quote gap, and a 10-bps "
            "execution buffer; no 80-bps floor. Model, admission, auction and exit "
            "policy semantics are unchanged."
        ),
        "inference_bundle": str(OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OVERLAY),
        "activation_authorization": str(AUTHORIZATION.relative_to(ROOT)),
        "execution_microstructure_buffer_bps": 10.0,
        "execution_adjusted_ev": (
            "raw_expected_gross_bps - (calculated microstructure friction + "
            "adverse delay gap + 10 bps); raw_expected_gross_bps uses the explicit "
            "MC1 gross convention and the historical 100-bps policy cost is not "
            "debited again; adjusted EV must remain >=50 bps at preflight and after fill."
        ),
        "runtime_code_sha256": {
            relative: sha(ROOT / relative)
            for relative in dict(execution["runtime_code_sha256"])
        },
    })

    authorization = copy.deepcopy(json.loads(OLD_AUTHORIZATION.read_text()))
    authorization.update({
        "authorization_source": (
            "Explicit user approval on 2026-08-17 to replace the fixed 80-bps live "
            "execution-friction floor with calculated microstructure friction plus "
            "adverse delay gap plus a sealed 10-bps buffer."
        ),
        "inference_bundle": str(OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OVERLAY),
        "exit_policy": execution["exit_policy"],
        "exit_policy_sha256": execution["exit_policy_sha256"],
    })
    write_new(AUTHORIZATION, authorization)
    execution["activation_authorization_sha256"] = sha(AUTHORIZATION)
    write_new(EXECUTION, execution)
    print(json.dumps({
        "overlay": str(OVERLAY), "overlay_sha256": sha(OVERLAY),
        "execution": str(EXECUTION), "execution_sha256": sha(EXECUTION),
        "authorization": str(AUTHORIZATION), "authorization_sha256": sha(AUTHORIZATION),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
