#!/usr/bin/env python3
"""Seal the approved indexed-state-snapshot live runtime successor.

The successor is intentionally mechanical: model, feature, Geometry/K9,
admission, portfolio, sizing and exit contracts are copied from the sealed
v172 source.  Only the state-snapshot runtime receipt and the hash-bound
inference/authorization identities change.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_INFERENCE = ROOT / (
    "config/strict_r3_inference_overlay_long_v146_v172_"
    "isolated_margin_hardstop.json"
)
SOURCE_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260823_"
    "v74_v172_isolated_margin_hardstop_live.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v148_v172_"
    "isolated_margin_hardstop_live.json"
)
INFERENCE_OUT = ROOT / (
    "config/strict_r3_inference_overlay_long_v149_v175_"
    "indexed_state_snapshot_state_rebind.json"
)
AUTHORIZATION_OUT = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260823_"
    "v77_v175_indexed_state_snapshot_state_rebind_live.json"
)
EXECUTION_OUT = ROOT / (
    "config/strict_r3_kraken_live_execution_v151_v175_"
    "indexed_state_snapshot_state_rebind_live.json"
)
RECEIPT_OUT = ROOT / (
    "data_perp/artifacts/strict_r3_runtime_reseal_v175_"
    "indexed_state_snapshot_state_rebind_20260823_v1/receipt.json"
)

PREDECESSOR_STATE_BUNDLE = ROOT / (
    "data_perp/artifacts/strict_r3_stateful_recovery_v172_20260823T090000Z_v1/"
    "hour_20260823T090000Z/run/feature_state/bundle"
)
RERECEIPT_STATE_BUNDLE = ROOT / (
    "data_perp/artifacts/strict_r3_runtime_reseal_v174_indexed_state_snapshot_"
    "20260823_v1/bootstrap_0900_feature_state_bundle"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def operator_state_payload_sha256(bundle: Path) -> str:
    """Return the receipt-independent digest required by the rebind guard."""
    import pandas as pd

    inventory = pd.read_parquet(bundle / "operator_state_inventory.parquet")
    rows = inventory.loc[:, ["relative_path", "sha256"]].sort_values(
        "relative_path", kind="stable"
    )
    digest = hashlib.sha256()
    for row in rows.itertuples(index=False):
        digest.update(f"{row.relative_path}\0{row.sha256}\n".encode("utf-8"))
    return digest.hexdigest()


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
    for source in (
        SOURCE_INFERENCE,
        SOURCE_AUTHORIZATION,
        SOURCE_EXECUTION,
        PREDECESSOR_STATE_BUNDLE / "state_bundle_manifest.json",
        RERECEIPT_STATE_BUNDLE / "state_bundle_manifest.json",
    ):
        if not source.is_file():
            raise FileNotFoundError(source)
    for target in (INFERENCE_OUT, AUTHORIZATION_OUT, EXECUTION_OUT, RECEIPT_OUT):
        if target.exists():
            raise FileExistsError(f"immutable successor already exists: {target}")

    inference = json.loads(SOURCE_INFERENCE.read_text())
    runtime_hashes = dict(inference["overrides"]["runtime_code_sha256"])
    runtime_hashes[
        "scripts/snapshot_strict_r3_feature_state_bundle.py"
    ] = sha256(ROOT / "scripts/snapshot_strict_r3_feature_state_bundle.py")
    runtime_hashes[
        "extreme_price_movements/inference/strict_r3_live_execution.py"
    ] = sha256(
        ROOT / "extreme_price_movements/inference/strict_r3_live_execution.py"
    )
    inference["overrides"]["runtime_code_sha256"] = runtime_hashes
    feature_state = dict(inference["overrides"].get("runtime") or {}).get(
        "feature_state"
    )
    if not isinstance(feature_state, dict):
        raise ValueError("source inference lacks a feature-state contract")
    old_payload = operator_state_payload_sha256(PREDECESSOR_STATE_BUNDLE)
    new_payload = operator_state_payload_sha256(RERECEIPT_STATE_BUNDLE)
    if old_payload != new_payload:
        raise ValueError("current-code rereceipt changed an operator-state payload")
    feature_state["one_time_state_reseal"] = {
        "superseded_bundle": str(PREDECESSOR_STATE_BUNDLE.relative_to(ROOT)),
        "superseded_manifest_sha256": sha256(
            PREDECESSOR_STATE_BUNDLE / "state_bundle_manifest.json"
        ),
        "resealed_bundle": str(RERECEIPT_STATE_BUNDLE.relative_to(ROOT)),
        "resealed_manifest_sha256": sha256(
            RERECEIPT_STATE_BUNDLE / "state_bundle_manifest.json"
        ),
        "operator_state_payload_sha256": old_payload,
        "reason": (
            "One-time v175 re-receipt of the verified 09:00 UTC state after "
            "the approved indexed snapshot implementation update; payload "
            "inventory is exactly identical and the bridge is unavailable "
            "outside these two immutable bundles."
        ),
    }
    inference["overrides"]["runtime"]["feature_state"] = feature_state
    inference["purpose"] = (
        "v149: approved indexed SQLite timestamp-bounds state snapshot plus "
        "one-time exact 09:00 feature-state rereceipt. "
        "This removes the noncritical full-table snapshot aggregate from the "
        "fresh-decision path; model, frozen 120 fields, Geometry/K9, admission, "
        "portfolio, sizing, execution economics and parent exit policy remain unchanged."
    )
    inference["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_INFERENCE.relative_to(ROOT)),
        "changed_runtime_paths": [
            "scripts/snapshot_strict_r3_feature_state_bundle.py",
            "extreme_price_movements/inference/strict_r3_live_execution.py",
        ],
        "economic_contract_changed": False,
        "reason": (
            "Nested-derived state publication now validates contract identity "
            "and index-backed first/last causal timestamps.  The removed "
            "row-count/distinct-key aggregate was diagnostic-only and did not "
            "participate in feature, score, admission, portfolio or execution logic."
        ),
    }
    write_new(INFERENCE_OUT, inference)
    inference_sha = sha256(INFERENCE_OUT)

    authorization = json.loads(SOURCE_AUTHORIZATION.read_text())
    authorization["inference_bundle"] = str(INFERENCE_OUT.relative_to(ROOT))
    authorization["inference_bundle_sha256"] = inference_sha
    authorization["authorization_source"] = (
        "User-approved v175 indexed state-snapshot and exact-state-rereceipt "
        "successor; only "
        "the durability receipt implementation changed."
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
            "runtime_code_sha256": {
                "extreme_price_movements/inference/strict_r3_live_execution.py": sha256(
                    ROOT / "extreme_price_movements/inference/strict_r3_live_execution.py"
                ),
            },
        },
    }
    write_new(EXECUTION_OUT, execution)
    execution_sha = sha256(EXECUTION_OUT)

    receipt = {
        "schema": "strict_r3_runtime_reseal_v175_indexed_state_snapshot_state_rebind_v1",
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
        "runtime_code": {
            "snapshot_script_sha256": sha256(
                ROOT / "scripts/snapshot_strict_r3_feature_state_bundle.py"
            ),
            "execution_loader_sha256": sha256(
                ROOT / "extreme_price_movements/inference/strict_r3_live_execution.py"
            ),
        },
        "feature_state_rereceipt": {
            "predecessor_bundle": str(PREDECESSOR_STATE_BUNDLE.relative_to(ROOT)),
            "rereceipt_bundle": str(RERECEIPT_STATE_BUNDLE.relative_to(ROOT)),
            "predecessor_manifest_sha256": sha256(
                PREDECESSOR_STATE_BUNDLE / "state_bundle_manifest.json"
            ),
            "rereceipt_manifest_sha256": sha256(
                RERECEIPT_STATE_BUNDLE / "state_bundle_manifest.json"
            ),
            "operator_state_payload_sha256": old_payload,
            "payload_identical": old_payload == new_payload,
        },
        "semantics": {
            "models_changed": False,
            "geometry_k9_changed": False,
            "admission_changed": False,
            "portfolio_changed": False,
            "sizing_changed": False,
            "exit_changed": False,
            "execution_changed": False,
        },
    }
    write_new(RECEIPT_OUT, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
