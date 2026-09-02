#!/usr/bin/env python3
"""Repackage v176's validated guard onto the canonical execution base.

The live execution loader deliberately disallows nested execution overlays.
This creates a loader-compatible v177 successor with *identical* v176
inference semantics and authorization, preserving the v175 execution-runtime
hash override while pointing directly to the canonical v172 execution base.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_INFERENCE = ROOT / "config/strict_r3_inference_overlay_long_v150_v176_direct_15m_book_guard.json"
SOURCE_AUTHORIZATION = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260823_v78_v176_direct_15m_book_guard_live.json"
SOURCE_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v151_v175_indexed_state_snapshot_state_rebind_live.json"
CANONICAL_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v148_v172_isolated_margin_hardstop_live.json"
AUTHORIZATION_OUT = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260823_v79_v177_direct_15m_book_guard_flat_execution_live.json"
EXECUTION_OUT = ROOT / "config/strict_r3_kraken_live_execution_v153_v177_direct_15m_book_guard_flat_execution_live.json"
RECEIPT_OUT = ROOT / "data_perp/artifacts/strict_r3_runtime_reseal_v177_direct_15m_book_guard_flat_execution_20260823_v1/receipt.json"


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
    for source in (SOURCE_INFERENCE, SOURCE_AUTHORIZATION, SOURCE_EXECUTION, CANONICAL_EXECUTION):
        if not source.is_file():
            raise FileNotFoundError(source)
    for target in (AUTHORIZATION_OUT, EXECUTION_OUT, RECEIPT_OUT):
        if target.exists():
            raise FileExistsError(f"immutable successor already exists: {target}")

    inference_sha = sha256(SOURCE_INFERENCE)
    authorization = json.loads(SOURCE_AUTHORIZATION.read_text())
    authorization["inference_bundle"] = str(SOURCE_INFERENCE.relative_to(ROOT))
    authorization["inference_bundle_sha256"] = inference_sha
    authorization["authorization_source"] = (
        "User-approved v177 loader-compatible repackaging of the v176 "
        "direct-15m causal book-corroboration candidate-eligibility guard; "
        "no model, policy, sizing, admission, or exit semantics changed."
    )
    write_new(AUTHORIZATION_OUT, authorization)
    authorization_sha = sha256(AUTHORIZATION_OUT)

    v175_execution = json.loads(SOURCE_EXECUTION.read_text())
    runtime_hashes = dict(v175_execution.get("overrides", {}).get("runtime_code_sha256") or {})
    if set(runtime_hashes) != {"extreme_price_movements/inference/strict_r3_live_execution.py"}:
        raise ValueError("unexpected v175 execution runtime override contract")
    execution = {
        "schema": "strict_r3_kraken_live_execution_overlay_v1",
        "base_execution": str(CANONICAL_EXECUTION.relative_to(ROOT)),
        "overrides": {
            "inference_bundle": str(SOURCE_INFERENCE.relative_to(ROOT)),
            "inference_bundle_sha256": inference_sha,
            "activation_authorization": str(AUTHORIZATION_OUT.relative_to(ROOT)),
            "activation_authorization_sha256": authorization_sha,
            "runtime_code_sha256": runtime_hashes,
        },
    }
    write_new(EXECUTION_OUT, execution)
    execution_sha = sha256(EXECUTION_OUT)

    receipt = {
        "schema": "strict_r3_runtime_reseal_v177_direct_15m_book_guard_flat_execution_v1",
        "source": {
            "inference": str(SOURCE_INFERENCE.relative_to(ROOT)),
            "inference_sha256": inference_sha,
            "authorization": str(SOURCE_AUTHORIZATION.relative_to(ROOT)),
            "authorization_sha256": sha256(SOURCE_AUTHORIZATION),
            "execution_overlay": str(SOURCE_EXECUTION.relative_to(ROOT)),
            "execution_overlay_sha256": sha256(SOURCE_EXECUTION),
            "canonical_execution": str(CANONICAL_EXECUTION.relative_to(ROOT)),
            "canonical_execution_sha256": sha256(CANONICAL_EXECUTION),
        },
        "successor": {
            "inference": str(SOURCE_INFERENCE.relative_to(ROOT)),
            "inference_sha256": inference_sha,
            "authorization": str(AUTHORIZATION_OUT.relative_to(ROOT)),
            "authorization_sha256": authorization_sha,
            "execution": str(EXECUTION_OUT.relative_to(ROOT)),
            "execution_sha256": execution_sha,
        },
        "semantics": {
            "inference_identical_to_v176": True,
            "execution_runtime_identical_to_v175": True,
            "candidate_eligibility_tightened": True,
            "models_changed": False,
            "geometry_k9_changed": False,
            "admission_threshold_changed": False,
            "portfolio_changed": False,
            "sizing_changed": False,
            "exit_changed": False,
        },
    }
    write_new(RECEIPT_OUT, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
