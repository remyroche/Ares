#!/usr/bin/env python3
"""Seal the direct-15m book guard with the complete live runtime hash set."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INFERENCE = ROOT / "config/strict_r3_inference_overlay_long_v150_v176_direct_15m_book_guard.json"
AUTH_SOURCE = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260823_v79_v177_direct_15m_book_guard_flat_execution_live.json"
V175_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v151_v175_indexed_state_snapshot_state_rebind_live.json"
CANONICAL_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v148_v172_isolated_margin_hardstop_live.json"
AUTH_OUT = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260823_v80_v178_direct_15m_book_guard_complete_runtime_live.json"
EXECUTION_OUT = ROOT / "config/strict_r3_kraken_live_execution_v154_v178_direct_15m_book_guard_complete_runtime_live.json"
RECEIPT_OUT = ROOT / "data_perp/artifacts/strict_r3_runtime_reseal_v178_direct_15m_book_guard_complete_runtime_20260823_v1/receipt.json"
MATERIALIZER = "scripts/materialize_strict_r3_target_free_hourly_grid_v2.py"


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
    for path in (INFERENCE, AUTH_SOURCE, V175_EXECUTION, CANONICAL_EXECUTION, ROOT / MATERIALIZER):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (AUTH_OUT, EXECUTION_OUT, RECEIPT_OUT):
        if path.exists():
            raise FileExistsError(f"immutable successor already exists: {path}")

    inference_sha = sha256(INFERENCE)
    authorization = json.loads(AUTH_SOURCE.read_text())
    authorization.update({
        "inference_bundle": str(INFERENCE.relative_to(ROOT)),
        "inference_bundle_sha256": inference_sha,
        "authorization_source": (
            "User-approved v178 completion of the v176 direct-15m causal "
            "book-corroboration guard runtime seal; no model or execution "
            "economic semantics changed."
        ),
    })
    write_new(AUTH_OUT, authorization)
    auth_sha = sha256(AUTH_OUT)

    v175 = json.loads(V175_EXECUTION.read_text())
    runtime = dict(v175["overrides"]["runtime_code_sha256"])
    runtime[MATERIALIZER] = sha256(ROOT / MATERIALIZER)
    execution = {
        "schema": "strict_r3_kraken_live_execution_overlay_v1",
        "base_execution": str(CANONICAL_EXECUTION.relative_to(ROOT)),
        "overrides": {
            "inference_bundle": str(INFERENCE.relative_to(ROOT)),
            "inference_bundle_sha256": inference_sha,
            "activation_authorization": str(AUTH_OUT.relative_to(ROOT)),
            "activation_authorization_sha256": auth_sha,
            "runtime_code_sha256": runtime,
        },
    }
    write_new(EXECUTION_OUT, execution)
    execution_sha = sha256(EXECUTION_OUT)

    receipt = {
        "schema": "strict_r3_runtime_reseal_v178_direct_15m_book_guard_complete_runtime_v1",
        "source": {
            "inference": str(INFERENCE.relative_to(ROOT)), "inference_sha256": inference_sha,
            "authorization": str(AUTH_SOURCE.relative_to(ROOT)), "authorization_sha256": sha256(AUTH_SOURCE),
            "canonical_execution": str(CANONICAL_EXECUTION.relative_to(ROOT)), "canonical_execution_sha256": sha256(CANONICAL_EXECUTION),
        },
        "successor": {
            "inference": str(INFERENCE.relative_to(ROOT)), "inference_sha256": inference_sha,
            "authorization": str(AUTH_OUT.relative_to(ROOT)), "authorization_sha256": auth_sha,
            "execution": str(EXECUTION_OUT.relative_to(ROOT)), "execution_sha256": execution_sha,
        },
        "runtime_hashes_added": {MATERIALIZER: runtime[MATERIALIZER]},
        "semantics": {
            "identical_to_v176_guard": True,
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
