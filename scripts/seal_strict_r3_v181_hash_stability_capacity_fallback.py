#!/usr/bin/env python3
"""Seal v181: stable BCF validation and tracked-position capacity fallback.

The successor retains every model, feature, Geometry/K9, calibration,
admission, portfolio, sizing and exit-policy parameter.  It only:

* retries a torn local immutable BCF artifact hash observation before failing;
* uses the persisted confirmed entry notional solely for an already-tracked
  position when Kraken returns no usable notional/price fields; and
* prevents an aged decision from reaching execution after a source retry.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INFERENCE_SOURCE = ROOT / "config/strict_r3_inference_overlay_long_v152_v180_compact_execution_1m_range_pruning.json"
EXECUTION_SOURCE = ROOT / "config/strict_r3_kraken_live_execution_v156_v180_compact_execution_1m_range_pruning_live.json"
AUTH_SOURCE = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260823_v82_v180_compact_execution_1m_range_pruning_live.json"
INFERENCE_OUT = ROOT / "config/strict_r3_inference_overlay_long_v153_v181_hash_stability_capacity_fallback.json"
EXECUTION_OUT = ROOT / "config/strict_r3_kraken_live_execution_v157_v181_hash_stability_capacity_fallback_live.json"
AUTH_OUT = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260823_v83_v181_hash_stability_capacity_fallback_live.json"
RECEIPT_OUT = ROOT / "data_perp/artifacts/strict_r3_runtime_reseal_v181_hash_stability_capacity_fallback_20260823_v1/receipt.json"
RUNTIME_PATHS = (
    "extreme_price_movements/strict_r3_inference_bundle.py",
    "extreme_price_movements/inference/strict_r3_live_execution.py",
    "scripts/run_strict_r3_live_hourly_entry_producer.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"immutable successor already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale successor temporary exists: {temporary}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    for path in (INFERENCE_SOURCE, EXECUTION_SOURCE, AUTH_SOURCE):
        if not path.is_file():
            raise FileNotFoundError(path)
    for relative in RUNTIME_PATHS:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(relative)
    for path in (INFERENCE_OUT, EXECUTION_OUT, AUTH_OUT, RECEIPT_OUT):
        if path.exists():
            raise FileExistsError(f"immutable successor already exists: {path}")

    inference = copy.deepcopy(json.loads(INFERENCE_SOURCE.read_text()))
    runtime_hashes = dict(inference["overrides"]["runtime_code_sha256"])
    for relative in RUNTIME_PATHS:
        runtime_hashes[relative] = sha256(ROOT / relative)
    inference["overrides"]["runtime_code_sha256"] = runtime_hashes
    inference["purpose"] = (
        "v153: bounded stable reads of immutable BCF artifacts; conservative "
        "confirmed-entry-notional fallback for malformed Kraken fields on "
        "already-tracked positions; and a 15-minute no-stale execution "
        "deadline after source refresh. No model, feature, Geometry/K9, "
        "calibration, admission, portfolio, sizing or exit-policy parameter changes."
    )
    inference["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(INFERENCE_SOURCE.relative_to(ROOT)),
        "changed_runtime_paths": list(RUNTIME_PATHS),
        "economic_contract_changed": False,
        "reason": "Make immutable artifact reads and live capacity accounting resilient to transient malformed venue responses while keeping all decision semantics fixed.",
    }
    write_new(INFERENCE_OUT, inference)
    inference_sha = sha256(INFERENCE_OUT)

    authorization = copy.deepcopy(json.loads(AUTH_SOURCE.read_text()))
    authorization.update({
        "inference_bundle": str(INFERENCE_OUT.relative_to(ROOT)),
        "inference_bundle_sha256": inference_sha,
        "authorization_source": "User-approved v181 runtime reliability repair; no model or economic-contract change.",
    })
    write_new(AUTH_OUT, authorization)
    auth_sha = sha256(AUTH_OUT)

    execution = copy.deepcopy(json.loads(EXECUTION_SOURCE.read_text()))
    overrides = dict(execution["overrides"])
    execution_hashes = dict(overrides["runtime_code_sha256"])
    for relative in RUNTIME_PATHS:
        execution_hashes[relative] = sha256(ROOT / relative)
    overrides.update({
        "inference_bundle": str(INFERENCE_OUT.relative_to(ROOT)),
        "inference_bundle_sha256": inference_sha,
        "activation_authorization": str(AUTH_OUT.relative_to(ROOT)),
        "activation_authorization_sha256": auth_sha,
        "runtime_code_sha256": execution_hashes,
    })
    execution["overrides"] = overrides
    bridges = list(execution.get("runtime_reseal_predecessors") or [])
    bridges.append({
        "predecessor_inference_bundle": str(INFERENCE_SOURCE.relative_to(ROOT)),
        "predecessor_inference_bundle_sha256": sha256(INFERENCE_SOURCE),
        "current_inference_bundle_sha256": inference_sha,
        "allowed_runtime_code_paths": list(RUNTIME_PATHS),
        "added_runtime_code_paths": [],
    })
    execution["runtime_reseal_predecessors"] = bridges
    write_new(EXECUTION_OUT, execution)
    execution_sha = sha256(EXECUTION_OUT)

    receipt = {
        "schema": "strict_r3_runtime_reseal_v181_hash_stability_capacity_fallback_v1",
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
        "runtime_hashes": {relative: runtime_hashes[relative] for relative in RUNTIME_PATHS},
        "semantics": {
            "models_changed": False,
            "features_changed": False,
            "geometry_k9_changed": False,
            "calibration_changed": False,
            "admission_changed": False,
            "portfolio_changed": False,
            "sizing_changed": False,
            "execution_economics_changed": False,
            "exit_thresholds_changed": False,
            "source_values_changed": False,
            "conservative_existing_position_capacity_fallback_only": True,
            "stale_execution_prevented_after_900_seconds": True,
        },
    }
    write_new(RECEIPT_OUT, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
