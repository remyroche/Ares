#!/usr/bin/env python3
"""Seal the producer-only direct-predecessor roll-forward successor.

The live hourly producer must advance its in-memory direct predecessor after a
successful cycle.  Without this, the next hour can fall back to a broad
historical manifest scan.  This creates an immutable successor with exactly
that one reviewed runtime-source change.  Models, features, Geometry/K9,
calibration, admission, portfolio policy, and exit policy remain byte
identical.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD_OVERLAY = Path(
    "config/strict_r3_inference_overlay_long_20260801_v134_bcf_current_dual_samebundle21d_direct_15m_reference_lockstep_geometry_feature_parity_rebind.json"
)
NEW_OVERLAY = Path(
    "config/strict_r3_inference_overlay_long_20260801_v135_bcf_current_dual_samebundle21d_direct_15m_reference_lockstep_geometry_feature_parity_direct_bootstrap_rollforward.json"
)
OLD_EXECUTION = Path(
    "config/strict_r3_kraken_live_execution_v135_v159_direct_15m_reference_lockstep_geometry_feature_parity_rebind_live.json"
)
NEW_EXECUTION = Path(
    "config/strict_r3_kraken_live_execution_v136_v160_direct_bootstrap_rollforward_live.json"
)
OLD_AUTHORIZATION = Path(
    "config/strict_r3_kraken_live_activation_authorization_20260822_v61_v134_direct_15m_reference_lockstep_geometry_feature_parity_rebind_live.json"
)
NEW_AUTHORIZATION = Path(
    "config/strict_r3_kraken_live_activation_authorization_20260822_v62_v135_direct_bootstrap_rollforward_live.json"
)
OLD_STATE = Path(
    "data_perp/live/strict_r3_kraken_live_state_v92_v159_direct_15m_reference_lockstep_geometry_feature_parity_rebind_live.json"
)
NEW_STATE = Path(
    "data_perp/live/strict_r3_kraken_live_state_v93_v160_direct_bootstrap_rollforward_live.json"
)
RECEIPT = Path(
    "data_perp/artifacts/strict_r3_direct_bootstrap_rollforward_reseal_20260822_v1/run_manifest.json"
)
CHANGED_PATH = "scripts/run_strict_r3_live_hourly_entry_producer.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(relative: Path) -> dict:
    return json.loads((ROOT / relative).read_text())


def _write_once(relative: Path, payload: dict) -> str:
    path = ROOT / relative
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != rendered:
            raise RuntimeError(f"immutable successor exists with different content: {relative}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
    return _sha(path)


def main() -> None:
    for required in (OLD_OVERLAY, OLD_EXECUTION, OLD_AUTHORIZATION, OLD_STATE):
        if not (ROOT / required).is_file():
            raise FileNotFoundError(required)

    old_overlay = _load(OLD_OVERLAY)
    old_execution = _load(OLD_EXECUTION)
    old_authorization = _load(OLD_AUTHORIZATION)
    old_state = _load(OLD_STATE)
    producer_hash = _sha(ROOT / CHANGED_PATH)

    old_runtime = dict(old_overlay.get("overrides", {}).get("runtime_code_sha256") or {})
    changed_disk_paths = sorted(
        relative
        for relative, expected in old_runtime.items()
        if not (ROOT / relative).is_file() or _sha(ROOT / relative) != str(expected)
    )
    if changed_disk_paths != [CHANGED_PATH]:
        raise RuntimeError(
            "refuse a mixed reseal; only the reviewed producer may differ: "
            f"{changed_disk_paths}"
        )
    if str(dict(old_execution.get("runtime_code_sha256") or {}).get(CHANGED_PATH)) != str(
        old_runtime.get(CHANGED_PATH)
    ):
        raise RuntimeError("active execution and overlay disagree on producer source hash")

    new_overlay = copy.deepcopy(old_overlay)
    new_overlay["overrides"]["runtime_code_sha256"][CHANGED_PATH] = producer_hash
    new_overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "changed_runtime_paths": [CHANGED_PATH],
        "economic_contract_changed": False,
        "reason": "After every successful decision the persistent producer advances its direct, hash-bound predecessor in memory; this prevents a later full historical-manifest scan.",
        "supersedes": str(OLD_OVERLAY),
    }
    overlay_hash = _write_once(NEW_OVERLAY, new_overlay)

    new_authorization = copy.deepcopy(old_authorization)
    new_authorization["inference_bundle"] = str(NEW_OVERLAY)
    new_authorization["inference_bundle_sha256"] = overlay_hash
    new_authorization["authorization_source"] = (
        "User-approved producer-only direct-predecessor roll-forward reseal; "
        "economic/model/policy artifacts unchanged."
    )
    authorization_hash = _write_once(NEW_AUTHORIZATION, new_authorization)

    new_execution = copy.deepcopy(old_execution)
    new_execution["inference_bundle"] = str(NEW_OVERLAY)
    new_execution["inference_bundle_sha256"] = overlay_hash
    new_execution["activation_authorization"] = str(NEW_AUTHORIZATION)
    new_execution["activation_authorization_sha256"] = authorization_hash
    new_execution["runtime_code_sha256"][CHANGED_PATH] = producer_hash
    bridges = list(new_execution.get("runtime_reseal_predecessors") or [])
    bridges.append(
        {
            "current_inference_bundle_sha256": overlay_hash,
            "predecessor_inference_bundle": str(OLD_OVERLAY),
            "predecessor_inference_bundle_sha256": _sha(ROOT / OLD_OVERLAY),
            "predecessor_execution": str(OLD_EXECUTION),
            "predecessor_execution_sha256": _sha(ROOT / OLD_EXECUTION),
            "allowed_runtime_code_paths": [CHANGED_PATH],
            "added_runtime_code_paths": [],
            "economic_contract_changed": False,
            "reviewed_current_runtime": True,
            "reason": "Producer-only direct predecessor advancement after a successful hour; prevents broad historical-manifest scan.",
            "successor_execution_semantics": "direct_predecessor_rollforward_fastpath_v1",
        }
    )
    new_execution["runtime_reseal_predecessors"] = bridges
    new_execution["version_note"] = (
        "v160 direct-bootstrap roll-forward: only scheduler predecessor handoff "
        "changes; all economic artifacts and entry/exit policy remain frozen."
    )
    execution_hash = _write_once(NEW_EXECUTION, new_execution)

    new_state = copy.deepcopy(old_state)
    new_state["inference_bundle_sha256"] = overlay_hash
    new_state["activation_authorization_sha256"] = authorization_hash
    new_state["contract_migration"] = {
        "schema": "strict_r3_live_state_contract_migration_v1",
        "prior_state": str(OLD_STATE),
        "prior_state_sha256": _sha(ROOT / OLD_STATE),
        "prior_inference_bundle_sha256": str(old_state["inference_bundle_sha256"]),
        "inference_bundle_sha256": overlay_hash,
        "prior_activation_authorization_sha256": str(
            old_state["activation_authorization_sha256"]
        ),
        "activation_authorization_sha256": authorization_hash,
        "positions_preserved_exact": True,
        "processed_decisions_preserved_exact": True,
        "reason": "Producer-only direct predecessor roll-forward runtime reseal.",
    }
    history = list(new_state.get("runtime_reseal_history") or [])
    history.append(
        {
            "changed_runtime_paths": [CHANGED_PATH],
            "economic_contract_changed": False,
            "execution_bundle": str(NEW_EXECUTION),
            "execution_bundle_sha256": execution_hash,
            "inference_bundle": str(NEW_OVERLAY),
            "inference_bundle_sha256": overlay_hash,
        }
    )
    new_state["runtime_reseal_history"] = history
    state_hash = _write_once(NEW_STATE, new_state)

    # Explicitly prove that all non-runtime economic inputs remained identical.
    old_paths = dict(old_overlay.get("overrides", {}).get("paths") or {})
    new_paths = dict(new_overlay.get("overrides", {}).get("paths") or {})
    old_artifacts = dict(old_overlay.get("overrides", {}).get("sha256") or {})
    new_artifacts = dict(new_overlay.get("overrides", {}).get("sha256") or {})
    if old_paths != new_paths or old_artifacts != new_artifacts:
        raise AssertionError("model/feature/policy artifact contract changed")
    if old_execution["exit_policy_sha256"] != new_execution["exit_policy_sha256"]:
        raise AssertionError("exit policy changed")

    receipt = {
        "schema": "strict_r3_direct_bootstrap_rollforward_reseal_v1",
        "status": "pass",
        "economic_contract_changed": False,
        "changed_runtime_paths": [CHANGED_PATH],
        "prior": {
            "overlay": str(OLD_OVERLAY),
            "overlay_sha256": _sha(ROOT / OLD_OVERLAY),
            "execution": str(OLD_EXECUTION),
            "execution_sha256": _sha(ROOT / OLD_EXECUTION),
            "state": str(OLD_STATE),
            "state_sha256": _sha(ROOT / OLD_STATE),
        },
        "successor": {
            "overlay": str(NEW_OVERLAY),
            "overlay_sha256": overlay_hash,
            "execution": str(NEW_EXECUTION),
            "execution_sha256": execution_hash,
            "authorization": str(NEW_AUTHORIZATION),
            "authorization_sha256": authorization_hash,
            "state": str(NEW_STATE),
            "state_sha256": state_hash,
        },
        "artifact_contract_preserved_exact": True,
        "exit_policy_preserved_exact": True,
        "producer_sha256": producer_hash,
    }
    _write_once(RECEIPT, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
