#!/usr/bin/env python3
"""Seal the v176 direct-15m decision-open book-corroboration successor.

This successor preserves the v175 model, Geometry/K9, calibration, portfolio,
sizing and exit contracts.  It changes only target-free candidate eligibility:
a zero/unknown-volume direct 15-minute open must agree with the contemporaneous
decision-time book before it can enter the scored population.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_INFERENCE = ROOT / (
    "config/strict_r3_inference_overlay_long_v149_v175_"
    "indexed_state_snapshot_state_rebind.json"
)
SOURCE_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260823_"
    "v77_v175_indexed_state_snapshot_state_rebind_live.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v151_v175_"
    "indexed_state_snapshot_state_rebind_live.json"
)
INFERENCE_OUT = ROOT / (
    "config/strict_r3_inference_overlay_long_v150_v176_"
    "direct_15m_book_guard.json"
)
AUTHORIZATION_OUT = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260823_"
    "v78_v176_direct_15m_book_guard_live.json"
)
EXECUTION_OUT = ROOT / (
    "config/strict_r3_kraken_live_execution_v152_v176_"
    "direct_15m_book_guard_live.json"
)
RECEIPT_OUT = ROOT / (
    "data_perp/artifacts/strict_r3_runtime_reseal_v176_"
    "direct_15m_book_guard_20260823_v1/receipt.json"
)
CHANGED_PATH = "scripts/materialize_strict_r3_target_free_hourly_grid_v2.py"


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
    for source in (SOURCE_INFERENCE, SOURCE_AUTHORIZATION, SOURCE_EXECUTION):
        if not source.is_file():
            raise FileNotFoundError(source)
    if not (ROOT / CHANGED_PATH).is_file():
        raise FileNotFoundError(ROOT / CHANGED_PATH)
    for target in (INFERENCE_OUT, AUTHORIZATION_OUT, EXECUTION_OUT, RECEIPT_OUT):
        if target.exists():
            raise FileExistsError(f"immutable successor already exists: {target}")

    inference = json.loads(SOURCE_INFERENCE.read_text())
    runtime_hashes = dict(inference["overrides"]["runtime_code_sha256"])
    runtime_hashes[CHANGED_PATH] = sha256(ROOT / CHANGED_PATH)
    inference["overrides"]["runtime_code_sha256"] = runtime_hashes
    inference["purpose"] = (
        "v150: direct 15-minute decision-open eligibility requires causal "
        "book corroboration when final trade volume is unresolved.  The "
        "v175 indexed state snapshot/re-receipt, models, frozen 120 fields, "
        "Geometry/K9, calibration, portfolio, sizing, execution economics "
        "and parent exit policy remain unchanged."
    )
    inference["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_INFERENCE.relative_to(ROOT)),
        "changed_runtime_paths": [CHANGED_PATH],
        "economic_contract_changed": True,
        "reason": (
            "A direct 15-minute open with zero/unknown final volume was "
            "incorrectly allowed despite a contemporaneous book deviation "
            "above the declared 100-bps limit.  The successor rejects that "
            "row before scoring; it does not change model outputs or any "
            "admitted candidate's execution economics."
        ),
    }
    write_new(INFERENCE_OUT, inference)
    inference_sha = sha256(INFERENCE_OUT)

    authorization = json.loads(SOURCE_AUTHORIZATION.read_text())
    authorization["inference_bundle"] = str(INFERENCE_OUT.relative_to(ROOT))
    authorization["inference_bundle_sha256"] = inference_sha
    authorization["authorization_source"] = (
        "User-approved v176 direct-15m causal book-corroboration successor; "
        "only target-free stale-price eligibility was tightened."
    )
    write_new(AUTHORIZATION_OUT, authorization)
    authorization_sha = sha256(AUTHORIZATION_OUT)

    execution = {
        "schema": "strict_r3_kraken_live_execution_overlay_v1",
        "base_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "overrides": {
            "inference_bundle": str(INFERENCE_OUT.relative_to(ROOT)),
            "inference_bundle_sha256": inference_sha,
            "activation_authorization": str(AUTHORIZATION_OUT.relative_to(ROOT)),
            "activation_authorization_sha256": authorization_sha,
        },
    }
    write_new(EXECUTION_OUT, execution)
    execution_sha = sha256(EXECUTION_OUT)

    receipt = {
        "schema": "strict_r3_runtime_reseal_v176_direct_15m_book_guard_v1",
        "source": {
            "inference": str(SOURCE_INFERENCE.relative_to(ROOT)),
            "inference_sha256": sha256(SOURCE_INFERENCE),
            "authorization": str(SOURCE_AUTHORIZATION.relative_to(ROOT)),
            "authorization_sha256": sha256(SOURCE_AUTHORIZATION),
            "execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
            "execution_sha256": sha256(SOURCE_EXECUTION),
        },
        "successor": {
            "inference": str(INFERENCE_OUT.relative_to(ROOT)),
            "inference_sha256": inference_sha,
            "authorization": str(AUTHORIZATION_OUT.relative_to(ROOT)),
            "authorization_sha256": authorization_sha,
            "execution": str(EXECUTION_OUT.relative_to(ROOT)),
            "execution_sha256": execution_sha,
        },
        "changed_runtime": {CHANGED_PATH: sha256(ROOT / CHANGED_PATH)},
        "semantics": {
            "models_changed": False,
            "geometry_k9_changed": False,
            "admission_threshold_changed": False,
            "portfolio_changed": False,
            "sizing_changed": False,
            "exit_changed": False,
            "candidate_eligibility_tightened": True,
        },
    }
    write_new(RECEIPT_OUT, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
