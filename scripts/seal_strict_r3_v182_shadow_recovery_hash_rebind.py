#!/usr/bin/env python3
"""Seal a runtime-only successor for the shadow-recovery hash rebind.

The v181 overlay already resealed the current runtime except for the shadow
resume orchestrator.  This successor binds that one observed source hash.  It
does not alter a frozen model artifact, feature contract, Geometry/K9 state,
calibration, admission threshold, portfolio rule, sizing, or exit policy.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INFERENCE_SOURCE = ROOT / "config/strict_r3_inference_overlay_long_v153_v181_hash_stability_capacity_fallback.json"
EXECUTION_SOURCE = ROOT / "config/strict_r3_kraken_live_execution_v157_v181_hash_stability_capacity_fallback_live.json"
AUTH_SOURCE = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260823_v83_v181_hash_stability_capacity_fallback_live.json"
INFERENCE_OUT = ROOT / "config/strict_r3_inference_overlay_long_v154_v182_shadow_recovery_hash_rebind.json"
EXECUTION_OUT = ROOT / "config/strict_r3_kraken_live_execution_v158_v182_shadow_recovery_hash_rebind_live.json"
AUTH_OUT = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260823_v84_v182_shadow_recovery_hash_rebind_live.json"
RECEIPT_OUT = ROOT / "data_perp/artifacts/strict_r3_runtime_reseal_v182_shadow_recovery_hash_rebind_20260823_v1/receipt.json"
RUNTIME_PATH = "scripts/run_strict_r3_hourly_shadow_resume_v15.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"immutable successor already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary exists: {temporary}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    for path in (INFERENCE_SOURCE, EXECUTION_SOURCE, AUTH_SOURCE, ROOT / RUNTIME_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (INFERENCE_OUT, EXECUTION_OUT, AUTH_OUT, RECEIPT_OUT):
        if path.exists():
            raise FileExistsError(f"immutable successor already exists: {path}")

    runtime_hash = sha256(ROOT / RUNTIME_PATH)
    inference = copy.deepcopy(json.loads(INFERENCE_SOURCE.read_text()))
    inference["overrides"]["runtime_code_sha256"][RUNTIME_PATH] = runtime_hash
    inference["purpose"] = (
        "v154: bind the observed shadow-resume orchestrator hash required for "
        "no-order state recovery. No model, artifact, feature, Geometry/K9, "
        "calibration, admission, portfolio, sizing or exit-policy semantics change."
    )
    inference["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(INFERENCE_SOURCE.relative_to(ROOT)),
        "changed_runtime_paths": [RUNTIME_PATH],
        "economic_contract_changed": False,
        "reason": "v181 omitted the shadow-resume runtime hash, which made an otherwise valid no-order recovery fail before scoring.",
    }
    write_new(INFERENCE_OUT, inference)
    inference_hash = sha256(INFERENCE_OUT)

    authorization = copy.deepcopy(json.loads(AUTH_SOURCE.read_text()))
    authorization.update({
        "inference_bundle": str(INFERENCE_OUT.relative_to(ROOT)),
        "inference_bundle_sha256": inference_hash,
        "authorization_source": "User-approved v182 recovery-runtime hash rebind; no economic contract change.",
    })
    write_new(AUTH_OUT, authorization)
    authorization_hash = sha256(AUTH_OUT)

    execution = copy.deepcopy(json.loads(EXECUTION_SOURCE.read_text()))
    overrides = execution["overrides"]
    overrides["inference_bundle"] = str(INFERENCE_OUT.relative_to(ROOT))
    overrides["inference_bundle_sha256"] = inference_hash
    overrides["activation_authorization"] = str(AUTH_OUT.relative_to(ROOT))
    overrides["activation_authorization_sha256"] = authorization_hash
    overrides["runtime_code_sha256"][RUNTIME_PATH] = runtime_hash
    execution["overrides"] = overrides
    write_new(EXECUTION_OUT, execution)
    execution_hash = sha256(EXECUTION_OUT)

    receipt = {
        "schema": "strict_r3_runtime_reseal_v182_shadow_recovery_hash_rebind_v1",
        "source": {
            "inference": str(INFERENCE_SOURCE.relative_to(ROOT)),
            "inference_sha256": sha256(INFERENCE_SOURCE),
            "execution": str(EXECUTION_SOURCE.relative_to(ROOT)),
            "execution_sha256": sha256(EXECUTION_SOURCE),
            "authorization": str(AUTH_SOURCE.relative_to(ROOT)),
            "authorization_sha256": sha256(AUTH_SOURCE),
        },
        "successor": {
            "inference": str(INFERENCE_OUT.relative_to(ROOT)),
            "inference_sha256": inference_hash,
            "execution": str(EXECUTION_OUT.relative_to(ROOT)),
            "execution_sha256": execution_hash,
            "authorization": str(AUTH_OUT.relative_to(ROOT)),
            "authorization_sha256": authorization_hash,
        },
        "runtime_hash_rebound": {RUNTIME_PATH: runtime_hash},
        "semantics": {
            "models_changed": False,
            "artifacts_changed": False,
            "features_changed": False,
            "geometry_k9_changed": False,
            "calibration_changed": False,
            "admission_changed": False,
            "portfolio_changed": False,
            "sizing_changed": False,
            "execution_economics_changed": False,
            "exit_thresholds_changed": False,
        },
    }
    write_new(RECEIPT_OUT, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
