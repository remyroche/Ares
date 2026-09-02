#!/usr/bin/env python3
"""Seal the parity-proven BCF plus feature-state runtime successor.

The successor preserves the active BCF/current dual-MC1 economics.  It bridges
the verified v122 state only after an isolated stateful materialization has
reproduced the full 170-row feature vector byte-for-byte under the current
``features.py`` implementation.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v124_"
    "bcf_current_dual_bcf_mc1_samebundle21d_replay_bridge.json"
)
SOURCE_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260822_v52_"
    "v124_bcf_mc1_samebundle21d_replay_bridge.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v126_v150_"
    "bcf_mc1_samebundle21d_replay_bridge.json"
)
V122_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v122_"
    "bcf_current_dual_bcf_mc1_structural_prior_coldstart.json"
)
V122_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v124_v148_"
    "bcf_mc1_structural_prior_coldstart.json"
)
OLD_STATE = ROOT / (
    "data_perp/artifacts/strict_r3_successor_v122_live_20260822T080000Z_v1/"
    "feature_state/bundle"
)
NEW_STATE = ROOT / (
    "data_perp/artifacts/strict_r3_feature_state_features_runtime_reseal_"
    "20260822T080000Z_v1/bundle"
)
PARITY_RECEIPT = ROOT / (
    "data_perp/artifacts/strict_r3_feature_state_features_runtime_reseal_"
    "20260822T080000Z_v1/exact_incremental_clone_parity_20260822T080000Z.json"
)
LEDGER = ROOT / (
    "data_perp/artifacts/strict_r3_bcf_same_bundle_recent_replay_"
    "ledger_20260822T090000Z_v1/bcf_mc1_recent_replay_ledger.parquet"
)
OUT_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v127_"
    "bcf_current_dual_samebundle21d_feature_state_reseal_parallel15m.json"
)
OUT_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260822_v55_"
    "v127_bcf_current_dual_samebundle21d_feature_state_reseal_parallel15m.json"
)
OUT_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v129_v153_"
    "bcf_current_dual_samebundle21d_feature_state_reseal_parallel15m.json"
)
OUT_REVIEW = ROOT / (
    "data_perp/artifacts/strict_r3_bcf_feature_state_reseal_20260822_v3/"
    "runtime_review.json"
)

EXPECTED_V122_RUNTIME_DELTA = {
    "extreme_price_movements/features.py",
    "extreme_price_movements/strict_r3_bcf_mc1_mapper.py",
    "scripts/run_strict_r3_live_hourly_entry_producer.py",
    "scripts/run_tp6_sl4_exact170_canonical_consensus.py",
    "scripts/update_strict_r3_feature_panel_state.py",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_digest(bundle: Path) -> str:
    inventory = pd.read_parquet(bundle / "operator_state_inventory.parquet")
    digest = hashlib.sha256()
    for row in inventory.loc[:, ["relative_path", "sha256"]].sort_values(
        "relative_path", kind="stable"
    ).itertuples(index=False):
        digest.update(f"{row.relative_path}\0{row.sha256}\n".encode())
    return digest.hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def resolved_runtime_hashes(overlay: dict) -> dict[str, str]:
    base = ROOT / str(overlay["base_bundle"])
    payload = json.loads(base.read_text())
    result = dict(payload.get("runtime_code_sha256") or {})
    result.update(dict(overlay["overrides"].get("runtime_code_sha256") or {}))
    return result


def main() -> None:
    for path in (
        SOURCE_OVERLAY,
        SOURCE_AUTHORIZATION,
        SOURCE_EXECUTION,
        V122_OVERLAY,
        V122_EXECUTION,
        OLD_STATE / "state_bundle_manifest.json",
        NEW_STATE / "state_bundle_manifest.json",
        PARITY_RECEIPT,
        LEDGER,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    parity = json.loads(PARITY_RECEIPT.read_text())
    if (
        parity.get("status") != "pass"
        or parity.get("candidate_identity_exact") is not True
        or parity.get("feature_values_exact") is not True
        or int(parity.get("rows", 0)) != 170
    ):
        raise ValueError("feature-state parity receipt is not an exact 170-row pass")
    old_payload = payload_digest(OLD_STATE)
    new_payload = payload_digest(NEW_STATE)
    if old_payload != new_payload:
        raise ValueError("feature-state reseal changed persisted operator payloads")

    source = json.loads(SOURCE_OVERLAY.read_text())
    parent = json.loads(V122_OVERLAY.read_text())
    source_hashes = resolved_runtime_hashes(source)
    parent_hashes = resolved_runtime_hashes(parent)
    current_hashes = {
        relative: sha(ROOT / relative) for relative in source_hashes
    }
    changed = {
        relative for relative in source_hashes
        if current_hashes[relative] != parent_hashes[relative]
    }
    if changed != EXPECTED_V122_RUNTIME_DELTA:
        raise ValueError(
            "unexpected direct-v122 runtime delta: " f"{sorted(changed)}"
        )
    if set(source_hashes) != set(parent_hashes):
        raise ValueError("source and v122 runtime code namespaces differ")

    overlay = copy.deepcopy(source)
    overlay["overrides"]["runtime_code_sha256"] = current_hashes
    feature_state = overlay["overrides"]["runtime"]["feature_state"]
    feature_state["one_time_state_reseal"] = {
        "superseded_bundle": str(OLD_STATE.relative_to(ROOT)),
        "resealed_bundle": str(NEW_STATE.relative_to(ROOT)),
        "superseded_manifest_sha256": sha(OLD_STATE / "state_bundle_manifest.json"),
        "resealed_manifest_sha256": sha(NEW_STATE / "state_bundle_manifest.json"),
        "operator_state_payload_sha256": old_payload,
        "feature_parity_receipt": str(PARITY_RECEIPT.relative_to(ROOT)),
        "feature_parity_receipt_sha256": sha(PARITY_RECEIPT),
        "reason": (
            "The isolated exact incremental materializer reproduced the complete "
            "170-row 2026-08-22 08:00 canonical vector byte-for-byte with the "
            "current features.py implementation. Persisted operator payloads are "
            "unchanged; only the implementation receipt advances."
        ),
    }
    overlay["purpose"] = (
        "v127: parity-proven features.py implementation receipt plus bounded "
        "parallel reads of independent 15-minute source windows and the existing "
        "BCF same-bundle 21-day replay ledger. Models, feature values, Geometry/K9, "
        "admission, auction, sizing, execution, and exit policy are unchanged."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "bridge_parent": str(V122_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": sorted(changed),
        "approved_calibration_artifact": str(LEDGER.relative_to(ROOT)),
        "approved_calibration_artifact_sha256": sha(LEDGER),
        "reason": (
            "Direct v122 state bridge with exact feature parity; only the reviewed "
            "BCF replay ledger, its mapper/validator, features.py receipt, and "
            "bounded deterministic 15-minute coverage I/O plus source-support "
            "adapters differ. The current decision-open adapter reproduced all 170 "
            "v122 opens exactly; the current panel updater reproduced every field "
            "at the appended v122 07:00 row while preserving prior source revisions."
        ),
    }
    write_new(OUT_OVERLAY, overlay)

    authorization = copy.deepcopy(json.loads(SOURCE_AUTHORIZATION.read_text()))
    authorization.update({
        "authorization_source": (
            "User-approved live continuation after exact stateful feature parity and "
            "byte-identical operator-payload reseal; no economic contract change."
        ),
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
    })
    write_new(OUT_AUTHORIZATION, authorization)

    execution = copy.deepcopy(json.loads(SOURCE_EXECUTION.read_text()))
    execution.update({
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "activation_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "activation_authorization_sha256": sha(OUT_AUTHORIZATION),
        "version_note": (
            "v153: exact-parity features.py implementation receipt, bounded parallel "
            "15-minute coverage reads, and direct v122 state bridge; economic "
            "contract unchanged."
        ),
    })
    execution["runtime_code_sha256"] = {
        relative: sha(ROOT / relative)
        for relative in dict(execution.get("runtime_code_sha256") or {})
    }
    execution["runtime_reseal_predecessors"] = [{
        "successor_execution_semantics": "bcf_samebundle21d_feature_state_exact_parity_v1",
        "current_inference_bundle_sha256": sha(OUT_OVERLAY),
        "predecessor_execution": str(V122_EXECUTION.relative_to(ROOT)),
        "predecessor_execution_sha256": sha(V122_EXECUTION),
        "predecessor_inference_bundle": str(V122_OVERLAY.relative_to(ROOT)),
        "predecessor_inference_bundle_sha256": sha(V122_OVERLAY),
        "allowed_runtime_code_paths": sorted(changed),
        "added_runtime_code_paths": [],
        "approved_calibration_artifact": str(LEDGER.relative_to(ROOT)),
        "approved_calibration_artifact_sha256": sha(LEDGER),
        "reviewed_current_runtime": True,
        "operator_state_payload_preserved_exact": True,
        "reason": (
            "Direct v122 bridge: BCF replay ledger/calibrator repair and a "
            "byte-for-byte feature-state implementation receipt plus bounded "
            "parallel 15-minute coverage I/O plus verified source-support "
            "adapters only."
        ),
    }]
    write_new(OUT_EXECUTION, execution)

    review = {
        "schema": "strict_r3_bcf_feature_state_reseal_runtime_review_v1",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "source_overlay": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "source_overlay_sha256": sha(SOURCE_OVERLAY),
        "v122_overlay": str(V122_OVERLAY.relative_to(ROOT)),
        "v122_overlay_sha256": sha(V122_OVERLAY),
        "changed_runtime_paths": sorted(changed),
        "feature_state_source": str(OLD_STATE.relative_to(ROOT)),
        "feature_state_successor": str(NEW_STATE.relative_to(ROOT)),
        "operator_state_payload_sha256": old_payload,
        "operator_state_payload_preserved_exact": True,
        "feature_parity_receipt": str(PARITY_RECEIPT.relative_to(ROOT)),
        "feature_parity_receipt_sha256": sha(PARITY_RECEIPT),
        "feature_parity_status": parity.get("status"),
        "successor_overlay": str(OUT_OVERLAY.relative_to(ROOT)),
        "successor_overlay_sha256": sha(OUT_OVERLAY),
        "successor_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "successor_authorization_sha256": sha(OUT_AUTHORIZATION),
        "successor_execution": str(OUT_EXECUTION.relative_to(ROOT)),
        "successor_execution_sha256": sha(OUT_EXECUTION),
        "economic_contract_changed": False,
    }
    write_new(OUT_REVIEW, review)
    print(json.dumps(review, sort_keys=True))


if __name__ == "__main__":
    main()
