#!/usr/bin/env python3
"""Seal the exact-parity ``features.py`` implementation successor.

The current source file changed outside the live contract.  This sealer permits
that new hash only after it verifies the complete frozen 120-field matrix and
the persisted feature-state payload against the successful 18:00 UTC receipt.
It changes no model, Geometry/K9 state, admission, portfolio, execution, or
exit economics.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v133_"
    "bcf_current_dual_samebundle21d_direct_15m_reference_"
    "lockstep_geometry_predecessor.json"
)
SOURCE_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260822_v60_"
    "v133_direct_15m_reference_lockstep_geometry_predecessor_live.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v134_v158_direct_15m_reference_"
    "lockstep_geometry_predecessor_live.json"
)
OLD_RUN = ROOT / "data_perp/artifacts/strict_r3_successor_v133_live_20260822T180000Z_v1"
OLD_STATE = OLD_RUN / "feature_state/bundle"
RECEIPT_ROOT = ROOT / (
    "data_perp/artifacts/strict_r3_feature_runtime_equivalence_"
    "20260822T180000Z_v1"
)
NEW_MATRIX = RECEIPT_ROOT / "features/canonical120_features.parquet"
NEW_STATE = RECEIPT_ROOT / "state_bundle"
OUT_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v134_"
    "bcf_current_dual_samebundle21d_direct_15m_reference_lockstep_"
    "geometry_feature_parity_rebind.json"
)
OUT_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260822_v61_"
    "v134_direct_15m_reference_lockstep_geometry_feature_parity_rebind_live.json"
)
OUT_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v135_v159_direct_15m_reference_"
    "lockstep_geometry_feature_parity_rebind_live.json"
)
OUT_REVIEW = RECEIPT_ROOT / "runtime_review.json"
RUNTIME_PATH = "extreme_price_movements/features.py"
DECISION = pd.Timestamp("2026-08-22T18:00:00Z")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_object(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def write_new(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def resolved_runtime_hashes(overlay: dict) -> dict[str, str]:
    base = read_object(ROOT / str(overlay["base_bundle"]))
    hashes = dict(base.get("runtime_code_sha256") or {})
    hashes.update(dict((overlay.get("overrides") or {}).get("runtime_code_sha256") or {}))
    return hashes


def state_payload_digest(bundle: Path) -> str:
    inventory = pd.read_parquet(bundle / "operator_state_inventory.parquet")
    required = {"relative_path", "sha256"}
    if not required.issubset(inventory.columns):
        raise ValueError("operator state inventory lacks content hashes")
    digest = hashlib.sha256()
    rows = inventory.loc[:, ["relative_path", "sha256"]].sort_values(
        "relative_path", kind="stable"
    )
    for row in rows.itertuples(index=False):
        digest.update(f"{row.relative_path}\0{row.sha256}\n".encode())
    return digest.hexdigest()


def assert_exact_feature_matrix() -> dict:
    base = read_object(ROOT / str(read_object(SOURCE_OVERLAY)["base_bundle"]))
    contract = read_object(ROOT / str(base["paths"]["feature_contract"]))
    fields = list(contract["base_fields_by_side"]["long"])
    old = pd.read_parquet(OLD_RUN / "features/canonical120_features.parquet")
    old = old.loc[
        pd.to_datetime(old["__decision_ts__"], utc=True).eq(DECISION),
        ["candidate_id", *fields],
    ].sort_values("candidate_id", kind="stable").reset_index(drop=True)
    new = pd.read_parquet(NEW_MATRIX)[["candidate_id", *fields]]
    new = new.sort_values("candidate_id", kind="stable").reset_index(drop=True)
    if not old["candidate_id"].equals(new["candidate_id"]):
        raise ValueError("feature parity candidate identities differ")
    changed: list[str] = []
    for field in fields:
        left = pd.to_numeric(old[field], errors="coerce").to_numpy(float)
        right = pd.to_numeric(new[field], errors="coerce").to_numpy(float)
        if not np.isclose(left, right, rtol=0.0, atol=0.0, equal_nan=True).all():
            changed.append(field)
    if changed:
        raise ValueError(f"frozen feature parity failed: {changed[:12]}")
    return {
        "decision_ts": DECISION.isoformat(),
        "rows": int(len(old)),
        "fields": int(len(fields)),
        "cells": int(len(old) * len(fields)),
        "candidate_identity_exact": True,
        "feature_values_exact": True,
        "changed_fields": [],
        "old_current_matrix_sha256": sha(
            OLD_RUN / "current_hour_inputs/canonical120_features.parquet"
        ),
        "new_matrix_sha256": sha(NEW_MATRIX),
    }


def main() -> None:
    required = (
        SOURCE_OVERLAY, SOURCE_AUTHORIZATION, SOURCE_EXECUTION,
        OLD_STATE / "state_bundle_manifest.json", NEW_MATRIX,
        NEW_STATE / "state_bundle_manifest.json",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    source = read_object(SOURCE_OVERLAY)
    expected_hashes = resolved_runtime_hashes(source)
    observed_hashes = {relative: sha(ROOT / relative) for relative in expected_hashes}
    changed_runtime = sorted(
        relative for relative in expected_hashes
        if observed_hashes[relative] != expected_hashes[relative]
    )
    if changed_runtime != [RUNTIME_PATH]:
        raise ValueError(f"unexpected runtime delta: {changed_runtime}")

    feature_parity = assert_exact_feature_matrix()
    old_payload = state_payload_digest(OLD_STATE)
    new_payload = state_payload_digest(NEW_STATE)
    if old_payload != new_payload:
        raise ValueError("feature-state payload changed during runtime re-receipt")

    overlay = copy.deepcopy(source)
    runtime_hashes = dict(overlay["overrides"].get("runtime_code_sha256") or {})
    runtime_hashes[RUNTIME_PATH] = observed_hashes[RUNTIME_PATH]
    overlay["overrides"]["runtime_code_sha256"] = runtime_hashes
    feature_state = overlay["overrides"]["runtime"]["feature_state"]
    feature_state["one_time_state_reseal"] = {
        "superseded_bundle": str(OLD_STATE.relative_to(ROOT)),
        "resealed_bundle": str(NEW_STATE.relative_to(ROOT)),
        "superseded_manifest_sha256": sha(OLD_STATE / "state_bundle_manifest.json"),
        "resealed_manifest_sha256": sha(NEW_STATE / "state_bundle_manifest.json"),
        "operator_state_payload_sha256": old_payload,
        "feature_parity_receipt": str(OUT_REVIEW.relative_to(ROOT)),
        "reason": (
            "The current features.py materialized the complete 18:00 UTC "
            "frozen 120-field matrix exactly, and its resulting persisted "
            "operator-state payload is byte-identical to the sealed state."
        ),
    }
    overlay["purpose"] = (
        "v134: exact-parity features.py runtime rebind. Models, frozen 120 "
        "fields, Geometry/K9, admission, portfolio, entry economics, and "
        "exit policy are unchanged."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": [RUNTIME_PATH],
        "economic_contract_changed": False,
        "reason": "Exact frozen feature and operator-state parity at 18:00 UTC.",
    }
    write_new(OUT_OVERLAY, overlay)

    authorization = copy.deepcopy(read_object(SOURCE_AUTHORIZATION))
    authorization.update({
        "authorization_source": (
            "User-authorized live continuation after an exact 120-field and "
            "operator-state runtime-equivalence proof; fresh decisions only."
        ),
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
    })
    write_new(OUT_AUTHORIZATION, authorization)

    execution = copy.deepcopy(read_object(SOURCE_EXECUTION))
    execution.update({
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "activation_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "activation_authorization_sha256": sha(OUT_AUTHORIZATION),
        "version_note": "v159: exact-parity features.py runtime rebind; economics unchanged.",
    })
    execution_hashes = dict(execution.get("runtime_code_sha256") or {})
    for relative, expected in execution_hashes.items():
        actual = sha(ROOT / relative)
        if relative != RUNTIME_PATH and actual != expected:
            raise ValueError(f"unexpected execution runtime delta: {relative}")
    execution_hashes[RUNTIME_PATH] = observed_hashes[RUNTIME_PATH]
    execution["runtime_code_sha256"] = execution_hashes
    execution["runtime_reseal_predecessors"] = list(
        execution.get("runtime_reseal_predecessors") or []
    ) + [{
        "successor_execution_semantics": "exact_frozen_feature_runtime_rebind_v1",
        "predecessor_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "predecessor_execution_sha256": sha(SOURCE_EXECUTION),
        "allowed_runtime_code_paths": [RUNTIME_PATH],
        "economic_contract_changed": False,
        "feature_state_payload_preserved_exact": True,
    }]
    write_new(OUT_EXECUTION, execution)

    review = {
        "schema": "strict_r3_exact_feature_runtime_rebind_review_v1",
        "status": "pass",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "source_overlay": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "source_overlay_sha256": sha(SOURCE_OVERLAY),
        "successor_overlay": str(OUT_OVERLAY.relative_to(ROOT)),
        "successor_overlay_sha256": sha(OUT_OVERLAY),
        "successor_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "successor_authorization_sha256": sha(OUT_AUTHORIZATION),
        "successor_execution": str(OUT_EXECUTION.relative_to(ROOT)),
        "successor_execution_sha256": sha(OUT_EXECUTION),
        "changed_runtime_paths": changed_runtime,
        "feature_parity": feature_parity,
        "feature_state": {
            "superseded_bundle": str(OLD_STATE.relative_to(ROOT)),
            "resealed_bundle": str(NEW_STATE.relative_to(ROOT)),
            "operator_state_payload_sha256": old_payload,
            "payload_exact": True,
        },
        "economic_contract_changed": False,
        "fresh_decisions_only": True,
    }
    write_new(OUT_REVIEW, review)
    print(json.dumps(review, sort_keys=True))


if __name__ == "__main__":
    main()
