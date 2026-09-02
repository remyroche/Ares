#!/usr/bin/env python3
"""Seal the bounded legacy execution-1m compact-range successor.

This is a runtime-only repair.  It recognises the timestamps already encoded
in legacy ``compact-<start>-<end>.parquet`` filenames so a current bounded
position-monitor read does not decode unrelated historical years.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INFERENCE_SOURCE = ROOT / "config/strict_r3_inference_overlay_long_v151_v179_oi_funding_sidecar_containment.json"
EXECUTION_SOURCE = ROOT / "config/strict_r3_kraken_live_execution_v155_v179_oi_funding_sidecar_containment_live.json"
AUTH_SOURCE = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260823_v81_v179_oi_funding_sidecar_containment_live.json"
INFERENCE_OUT = ROOT / "config/strict_r3_inference_overlay_long_v152_v180_compact_execution_1m_range_pruning.json"
EXECUTION_OUT = ROOT / "config/strict_r3_kraken_live_execution_v156_v180_compact_execution_1m_range_pruning_live.json"
AUTH_OUT = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260823_v82_v180_compact_execution_1m_range_pruning_live.json"
RECEIPT_OUT = ROOT / "data_perp/artifacts/strict_r3_runtime_reseal_v180_compact_execution_1m_range_pruning_20260823_v1/receipt.json"
RUNTIME_PATH = "extreme_price_movements/data_store.py"


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
    for path in (INFERENCE_SOURCE, EXECUTION_SOURCE, AUTH_SOURCE, ROOT / RUNTIME_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (INFERENCE_OUT, EXECUTION_OUT, AUTH_OUT, RECEIPT_OUT):
        if path.exists():
            raise FileExistsError(f"immutable successor already exists: {path}")

    inference = copy.deepcopy(json.loads(INFERENCE_SOURCE.read_text()))
    runtime_hashes = dict(inference["overrides"]["runtime_code_sha256"])
    runtime_hashes[RUNTIME_PATH] = sha256(ROOT / RUNTIME_PATH)
    inference["overrides"]["runtime_code_sha256"] = runtime_hashes
    inference["purpose"] = (
        "v152: recognise bounded timestamps in legacy immutable compact "
        "execution-1m part filenames. Current position monitoring therefore "
        "does not decode unrelated historical compact parts. No source values, "
        "models, features, Geometry/K9, calibration, admission, portfolio, "
        "sizing, entry economics, or exit thresholds change."
    )
    inference["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(INFERENCE_SOURCE.relative_to(ROOT)),
        "changed_runtime_paths": [RUNTIME_PATH],
        "economic_contract_changed": False,
        "reason": "Bounded execution-1m reads must prune legacy compact files whose filename timestamps prove they are outside the requested interval.",
    }
    write_new(INFERENCE_OUT, inference)
    inference_sha = sha256(INFERENCE_OUT)

    authorization = copy.deepcopy(json.loads(AUTH_SOURCE.read_text()))
    authorization.update({
        "inference_bundle": str(INFERENCE_OUT.relative_to(ROOT)),
        "inference_bundle_sha256": inference_sha,
        "authorization_source": "User-approved v180 bounded legacy compact one-minute cache-range repair; runtime-only with no economic contract change.",
    })
    write_new(AUTH_OUT, authorization)
    auth_sha = sha256(AUTH_OUT)

    execution = copy.deepcopy(json.loads(EXECUTION_SOURCE.read_text()))
    overrides = dict(execution["overrides"])
    execution_hashes = dict(overrides["runtime_code_sha256"])
    execution_hashes[RUNTIME_PATH] = sha256(ROOT / RUNTIME_PATH)
    overrides.update({
        "inference_bundle": str(INFERENCE_OUT.relative_to(ROOT)),
        "inference_bundle_sha256": inference_sha,
        "activation_authorization": str(AUTH_OUT.relative_to(ROOT)),
        "activation_authorization_sha256": auth_sha,
        "runtime_code_sha256": execution_hashes,
    })
    execution["overrides"] = overrides
    write_new(EXECUTION_OUT, execution)
    execution_sha = sha256(EXECUTION_OUT)

    receipt = {
        "schema": "strict_r3_runtime_reseal_v180_compact_execution_1m_range_pruning_v1",
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
        "runtime_hash": {RUNTIME_PATH: runtime_hashes[RUNTIME_PATH]},
        "semantics": {
            "models_changed": False,
            "features_changed": False,
            "geometry_k9_changed": False,
            "admission_changed": False,
            "portfolio_changed": False,
            "sizing_changed": False,
            "execution_economics_changed": False,
            "exit_thresholds_changed": False,
            "source_values_changed": False,
            "bounded_read_only": True,
        },
    }
    write_new(RECEIPT_OUT, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
