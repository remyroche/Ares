#!/usr/bin/env python3
"""Seal the one-time, byte-identical v128 feature-state snapshot bridge."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v102_"
    "bcf_current_dual_exit_postclose_confirmation_timing.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v102_v126_"
    "side_specific_exit_postclose_confirmation_timing.json"
)
SOURCE_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260819_v30_"
    "v102_postclose_confirmation_timing.json"
)
OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v104_"
    "bcf_current_dual_feature_state_snapshot_reseal.json"
)
AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260820_v32_"
    "v103_feature_state_snapshot_reseal.json"
)
EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v104_v128_"
    "feature_state_snapshot_reseal.json"
)
SUPERSEDED = ROOT / (
    "data_perp/artifacts/strict_r3_stateful_recovery_v96_v117_20260819T160000Z_v4/"
    "hour_20260819T160000Z/run/feature_state/bundle"
)
RESEALED = ROOT / (
    "data_perp/artifacts/strict_r3_feature_state_snapshot_reseal_v127_20260819_v1/bundle"
)
CHANGED_RUNTIME = ("scripts/run_strict_r3_live_hourly_entry_producer.py",)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_sha(bundle: Path) -> str:
    inventory = pd.read_parquet(bundle / "operator_state_inventory.parquet")
    digest = hashlib.sha256()
    for row in inventory.loc[:, ["relative_path", "sha256"]].sort_values(
        "relative_path", kind="stable"
    ).itertuples(index=False):
        digest.update(f"{row.relative_path}\0{row.sha256}\n".encode("utf-8"))
    return digest.hexdigest()


def write_new(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"immutable successor already exists: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    if not SUPERSEDED.is_dir() or not RESEALED.is_dir():
        raise FileNotFoundError("feature-state bridge bundles are unavailable")
    preserved = payload_sha(SUPERSEDED)
    if preserved != payload_sha(RESEALED):
        raise ValueError("resealed feature-state payload is not byte-identical")

    overlay = copy.deepcopy(json.loads(SOURCE_OVERLAY.read_text()))
    runtime_hashes = dict(overlay["overrides"].get("runtime_code_sha256") or {})
    for relative in CHANGED_RUNTIME:
        runtime_hashes[relative] = sha(ROOT / relative)
    overlay["overrides"]["runtime_code_sha256"] = runtime_hashes
    feature_state = dict(overlay["overrides"]["runtime"]["feature_state"])
    feature_state["one_time_state_reseal"] = {
        "superseded_bundle": str(SUPERSEDED.relative_to(ROOT)),
        "resealed_bundle": str(RESEALED.relative_to(ROOT)),
        "superseded_manifest_sha256": sha(SUPERSEDED / "state_bundle_manifest.json"),
        "resealed_manifest_sha256": sha(RESEALED / "state_bundle_manifest.json"),
        "operator_state_payload_sha256": preserved,
        "reason": (
            "One-time snapshot-lineage re-receipt. The reviewed current snapshot "
            "utility handles only exact stale duplicate cache-state files; all "
            "feature, Geometry/K9, and operator-state payloads are byte-identical."
        ),
    }
    overlay["overrides"]["runtime"] = dict(overlay["overrides"]["runtime"])
    overlay["overrides"]["runtime"]["feature_state"] = feature_state
    overlay["purpose"] = (
        "v104: one-time byte-identical feature/Geometry/K9 state re-receipt after "
        "the reviewed snapshot-cache duplicate handling update. No model, feature "
        "value, calibration, admission, portfolio, policy, or order-decision "
        "semantics change."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": list(CHANGED_RUNTIME),
        "reason": "Use the sealed byte-identical snapshot state bridge exactly once.",
    }
    write_new(OVERLAY, overlay)

    authorization = copy.deepcopy(json.loads(SOURCE_AUTHORIZATION.read_text()))
    authorization.update({
        "authorization_source": (
            "User-authorized live operational repair: resume only from a sealed, "
            "byte-identical feature/Geometry/K9 state re-receipt after the reviewed "
            "snapshot duplicate-cache handling update. Strategy and order authority "
            "are unchanged."
        ),
        "inference_bundle": str(OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OVERLAY),
    })
    write_new(AUTHORIZATION, authorization)

    execution = copy.deepcopy(json.loads(SOURCE_EXECUTION.read_text()))
    execution.update({
        "inference_bundle": str(OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OVERLAY),
        "activation_authorization": str(AUTHORIZATION.relative_to(ROOT)),
        "activation_authorization_sha256": sha(AUTHORIZATION),
        "version_note": (
            "v128: one-time byte-identical feature-state snapshot lineage bridge. "
            "No trading decision or rich-policy parameter changes."
        ),
    })
    execution_hashes = dict(execution.get("runtime_code_sha256") or {})
    execution_hashes["scripts/run_strict_r3_live_hourly_entry_producer.py"] = sha(
        ROOT / "scripts/run_strict_r3_live_hourly_entry_producer.py"
    )
    execution["runtime_code_sha256"] = execution_hashes
    execution["runtime_reseal_predecessors"] = [{
        "predecessor_inference_bundle": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "predecessor_inference_bundle_sha256": sha(SOURCE_OVERLAY),
        "current_inference_bundle_sha256": sha(OVERLAY),
        "allowed_runtime_code_paths": list(CHANGED_RUNTIME),
        "added_runtime_code_paths": [],
        "reason": (
            "State bridge producer wiring and shadow-state validation only; static "
            "model, feature, admission, portfolio, policy and execution gates are unchanged."
        ),
    }]
    write_new(EXECUTION, execution)
    print(json.dumps({
        "overlay": str(OVERLAY.relative_to(ROOT)), "overlay_sha256": sha(OVERLAY),
        "authorization": str(AUTHORIZATION.relative_to(ROOT)), "authorization_sha256": sha(AUTHORIZATION),
        "execution": str(EXECUTION.relative_to(ROOT)), "execution_sha256": sha(EXECUTION),
        "operator_state_payload_sha256": preserved,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
