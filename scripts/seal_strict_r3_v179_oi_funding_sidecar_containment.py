#!/usr/bin/env python3
"""Seal the reviewed corrupt-local OI/funding source containment successor."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INFERENCE_SOURCE = ROOT / "config/strict_r3_inference_overlay_long_v150_v176_direct_15m_book_guard.json"
EXECUTION_SOURCE = ROOT / "config/strict_r3_kraken_live_execution_v154_v178_direct_15m_book_guard_complete_runtime_live.json"
AUTH_SOURCE = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260823_v80_v178_direct_15m_book_guard_complete_runtime_live.json"
INFERENCE_OUT = ROOT / "config/strict_r3_inference_overlay_long_v151_v179_oi_funding_sidecar_containment.json"
AUTH_OUT = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260823_v81_v179_oi_funding_sidecar_containment_live.json"
EXECUTION_OUT = ROOT / "config/strict_r3_kraken_live_execution_v155_v179_oi_funding_sidecar_containment_live.json"
RECEIPT_OUT = ROOT / "data_perp/artifacts/strict_r3_runtime_reseal_v179_oi_funding_sidecar_containment_20260823_v1/receipt.json"
RUNTIME_PATHS = (
    "scripts/backfill_kraken_oi_funding_sidecars.py",
    "scripts/run_strict_r3_live_hourly_entry_producer.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"immutable successor already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        raise FileExistsError(f"stale successor temporary exists: {tmp}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main() -> None:
    for path in (INFERENCE_SOURCE, EXECUTION_SOURCE, AUTH_SOURCE):
        if not path.is_file():
            raise FileNotFoundError(path)
    for relative in RUNTIME_PATHS:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(relative)
    for path in (INFERENCE_OUT, AUTH_OUT, EXECUTION_OUT, RECEIPT_OUT):
        if path.exists():
            raise FileExistsError(f"immutable successor already exists: {path}")

    inference = copy.deepcopy(json.loads(INFERENCE_SOURCE.read_text()))
    runtime = dict(inference["overrides"]["runtime_code_sha256"])
    for relative in RUNTIME_PATHS:
        runtime[relative] = sha256(ROOT / relative)
    inference["overrides"]["runtime_code_sha256"] = runtime
    inference["purpose"] = (
        "v151: contain an invalid local OI/funding sidecar per symbol. The bad "
        "file is atomically quarantined with an immutable unavailable marker; "
        "partial API history is never substituted and only that candidate may "
        "fail closed. Models, features for source-complete rows, Geometry/K9, "
        "calibration, admission, portfolio, sizing, execution economics and "
        "parent exit policy remain unchanged."
    )
    inference["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(INFERENCE_SOURCE.relative_to(ROOT)),
        "changed_runtime_paths": list(RUNTIME_PATHS),
        "economic_contract_changed": False,
        "reason": (
            "A corrupt local source must fail only the affected candidate closed, "
            "not prevent all symbols from being refreshed or permit partial-history "
            "replacement."
        ),
    }
    write_new(INFERENCE_OUT, inference)
    inference_sha = sha256(INFERENCE_OUT)

    authorization = copy.deepcopy(json.loads(AUTH_SOURCE.read_text()))
    authorization.update({
        "inference_bundle": str(INFERENCE_OUT.relative_to(ROOT)),
        "inference_bundle_sha256": inference_sha,
        "authorization_source": (
            "User-approved v179 corrupt-local OI/funding sidecar containment; "
            "only row-local fail-closed source availability changes."
        ),
    })
    write_new(AUTH_OUT, authorization)
    auth_sha = sha256(AUTH_OUT)

    execution_source = json.loads(EXECUTION_SOURCE.read_text())
    execution = copy.deepcopy(execution_source)
    overrides = dict(execution["overrides"])
    runtime_exec = dict(overrides["runtime_code_sha256"])
    for relative in RUNTIME_PATHS:
        runtime_exec[relative] = sha256(ROOT / relative)
    overrides.update({
        "inference_bundle": str(INFERENCE_OUT.relative_to(ROOT)),
        "inference_bundle_sha256": inference_sha,
        "activation_authorization": str(AUTH_OUT.relative_to(ROOT)),
        "activation_authorization_sha256": auth_sha,
        "runtime_code_sha256": runtime_exec,
    })
    execution["overrides"] = overrides
    write_new(EXECUTION_OUT, execution)
    execution_sha = sha256(EXECUTION_OUT)

    receipt = {
        "schema": "strict_r3_runtime_reseal_v179_oi_funding_sidecar_containment_v1",
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
            "inference_sha256": inference_sha,
            "execution": str(EXECUTION_OUT.relative_to(ROOT)),
            "execution_sha256": execution_sha,
            "authorization": str(AUTH_OUT.relative_to(ROOT)),
            "authorization_sha256": auth_sha,
        },
        "runtime_hashes": {relative: runtime[relative] for relative in RUNTIME_PATHS},
        "semantics": {
            "models_changed": False,
            "features_changed_for_source_complete_rows": False,
            "geometry_k9_changed": False,
            "admission_threshold_changed": False,
            "portfolio_changed": False,
            "sizing_changed": False,
            "exit_changed": False,
            "source_failure_scope": "row_local_fail_closed",
            "partial_api_history_replacement_forbidden": True,
        },
    }
    write_new(RECEIPT_OUT, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
