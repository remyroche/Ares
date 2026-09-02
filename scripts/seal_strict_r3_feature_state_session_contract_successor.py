#!/usr/bin/env python3
"""Seal the parity-proven feature-state session-contract runtime successor."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v109_"
    "bcf_current_dual_authoritative_kraken_fills.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v110_v134_"
    "authoritative_kraken_fills.json"
)
SOURCE_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260820_v37_"
    "v109_authoritative_kraken_fills.json"
)
OUT_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v111_"
    "bcf_current_dual_feature_state_session_output_schema_fix.json"
)
OUT_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v112_v136_"
    "feature_state_session_output_schema_fix.json"
)
OUT_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260820_v39_"
    "v111_feature_state_session_output_schema_fix.json"
)
OLD_STATE = ROOT / (
    "data_perp/artifacts/strict_r3_successor_v104_live_20260820T070000Z_v1/"
    "feature_state/bundle"
)
NEW_STATE = ROOT / (
    "data_perp/artifacts/strict_r3_feature_state_session_contract_fix_20260820_v3/"
    "validated_state_0700/bundle"
)
PARITY_RECEIPT = ROOT / (
    "data_perp/artifacts/strict_r3_feature_state_session_contract_fix_20260820_v3/"
    "feature_parity_receipt.json"
)
REVIEW = ROOT / (
    "data_perp/artifacts/strict_r3_feature_state_session_output_schema_live_reseal_"
    "20260820_v1/runtime_review.json"
)
EXPECTED_RUNTIME_DELTA = {
    "extreme_price_movements/features.py",
    "scripts/materialize_strict_r3_forward_features_incremental_v13.py",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def runtime_hashes(overlay: dict) -> dict[str, str]:
    base = ROOT / str(overlay["base_bundle"])
    payload = json.loads(base.read_text())
    result = dict(payload.get("runtime_code_sha256") or {})
    result.update(dict(overlay["overrides"].get("runtime_code_sha256") or {}))
    return result


def normalized_static(overlay: dict) -> dict:
    result = copy.deepcopy(overlay)
    result.pop("purpose", None)
    result.pop("runtime_reseal", None)
    result["overrides"].pop("runtime_code_sha256", None)
    feature_state = result["overrides"]["runtime"]["feature_state"]
    feature_state["one_time_state_reseal"] = "__parity_proven_state_bridge__"
    return result


def main() -> None:
    for path in (OLD_STATE, NEW_STATE, PARITY_RECEIPT):
        if not path.exists():
            raise FileNotFoundError(path)
    parity = json.loads(PARITY_RECEIPT.read_text())
    if parity.get("status") != "pass":
        raise ValueError("cannot seal a feature-state successor without parity pass")
    old_manifest = OLD_STATE / "state_bundle_manifest.json"
    new_manifest = NEW_STATE / "state_bundle_manifest.json"
    old_payload = payload_digest(OLD_STATE)
    new_payload = payload_digest(NEW_STATE)
    if old_payload != new_payload:
        raise ValueError("validated successor changed an operator-state payload")

    source = json.loads(SOURCE_OVERLAY.read_text())
    expected = runtime_hashes(source)
    actual = {relative: sha(ROOT / relative) for relative in expected}
    changed = {relative for relative in expected if expected[relative] != actual[relative]}
    if changed != EXPECTED_RUNTIME_DELTA:
        raise ValueError(f"unexpected runtime changes: {sorted(changed)}")

    overlay = copy.deepcopy(source)
    overlay["overrides"]["runtime_code_sha256"].update(
        {relative: actual[relative] for relative in changed}
    )
    overlay["overrides"]["runtime"]["feature_state"]["one_time_state_reseal"] = {
        "superseded_bundle": str(OLD_STATE.relative_to(ROOT)),
        "resealed_bundle": str(NEW_STATE.relative_to(ROOT)),
        "superseded_manifest_sha256": sha(old_manifest),
        "resealed_manifest_sha256": sha(new_manifest),
        "operator_state_payload_sha256": old_payload,
        "reason": (
            "Session-calendar fields are deterministic compute helpers and are "
            "excluded from recursive state and the frozen persisted model matrix. "
            "The current implementation matched the complete 136-column "
            "2026-08-20 07:00 receipt exactly."
        ),
        "feature_parity_receipt": str(PARITY_RECEIPT.relative_to(ROOT)),
        "feature_parity_receipt_sha256": sha(PARITY_RECEIPT),
    }
    overlay["purpose"] = (
        "v111: parity-proven feature-state session/output-schema repair. The "
        "frozen model matrix and recursive state semantics are unchanged."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": sorted(changed),
        "reason": "Feature-state/output-schema repair with exact same-hour parity.",
    }
    if normalized_static(source) != normalized_static(overlay):
        raise AssertionError("feature-state successor changed a static strategy contract")
    write_new(OUT_OVERLAY, overlay)

    authorization = copy.deepcopy(json.loads(SOURCE_AUTHORIZATION.read_text()))
    authorization.update({
        "authorization_source": (
            "User-authorized 2026-08-20 live runtime recovery after an exact "
            "frozen-field parity test. No model, EV map, admission, auction, or "
            "exit-policy parameter changed."
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
            "v136: parity-proven feature-state session/output-schema repair. No model, "
            "admission, portfolio, entry, execution-friction or parent-policy change."
        ),
    })
    execution["runtime_reseal_predecessors"] = list(
        execution.get("runtime_reseal_predecessors") or []
    ) + [{
        "successor_execution_semantics": "feature_state_session_contract_parity_reseal_v1",
        "current_inference_bundle_sha256": sha(OUT_OVERLAY),
        "predecessor_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "predecessor_execution_sha256": sha(SOURCE_EXECUTION),
        "predecessor_inference_bundle": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "predecessor_inference_bundle_sha256": sha(SOURCE_OVERLAY),
        "allowed_runtime_code_paths": sorted(changed),
        "added_runtime_code_paths": [],
        "changed_execution_fields": [],
        "reviewed_current_runtime": True,
        "operator_state_payload_preserved_exact": True,
    }]
    write_new(OUT_EXECUTION, execution)

    review = {
        "schema": "strict_r3_feature_state_session_contract_runtime_review_v1",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "source_overlay": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "source_overlay_sha256": sha(SOURCE_OVERLAY),
        "changed_runtime_paths": sorted(changed),
        "old_state_bundle": str(OLD_STATE.relative_to(ROOT)),
        "new_state_bundle": str(NEW_STATE.relative_to(ROOT)),
        "operator_state_payload_sha256": old_payload,
        "operator_state_payload_preserved_exact": True,
        "feature_parity_receipt": str(PARITY_RECEIPT.relative_to(ROOT)),
        "feature_parity_receipt_sha256": sha(PARITY_RECEIPT),
        "non_runtime_contract_exact": normalized_static(source) == normalized_static(overlay),
        "successor_overlay": str(OUT_OVERLAY.relative_to(ROOT)),
        "successor_overlay_sha256": sha(OUT_OVERLAY),
        "successor_execution": str(OUT_EXECUTION.relative_to(ROOT)),
        "successor_execution_sha256": sha(OUT_EXECUTION),
    }
    write_new(REVIEW, review)
    print(json.dumps(review, sort_keys=True))


if __name__ == "__main__":
    main()
