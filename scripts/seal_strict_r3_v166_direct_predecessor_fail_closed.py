#!/usr/bin/env python3
"""Seal v166: direct predecessor lineage fails closed, never archival-scans."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD_OVERLAY = Path("config/strict_r3_inference_overlay_long_20260801_v140_bcf_current_dual_samebundle21d_direct_15m_reference_lockstep_geometry_feature_parity_warm_execution_parallel_source_io.json")
NEW_OVERLAY = Path("config/strict_r3_inference_overlay_long_20260801_v141_bcf_current_dual_samebundle21d_direct_15m_reference_lockstep_geometry_feature_parity_warm_execution_parallel_source_io_lockstep_failclosed.json")
OLD_EXECUTION = Path("config/strict_r3_kraken_live_execution_v141_v165_parallel_source_io_live.json")
NEW_EXECUTION = Path("config/strict_r3_kraken_live_execution_v142_v166_lockstep_failclosed_live.json")
OLD_AUTH = Path("config/strict_r3_kraken_live_activation_authorization_20260822_v67_v140_parallel_source_io_live.json")
NEW_AUTH = Path("config/strict_r3_kraken_live_activation_authorization_20260822_v68_v141_lockstep_failclosed_live.json")
OLD_STATE = Path("data_perp/live/strict_r3_kraken_live_state_v98_v165_parallel_source_io_live.json")
NEW_STATE = Path("data_perp/live/strict_r3_kraken_live_state_v99_v166_lockstep_failclosed_live.json")
RECEIPT = Path("data_perp/artifacts/strict_r3_lockstep_failclosed_reseal_20260822_v1/run_manifest.json")
CHANGED = ["scripts/run_strict_r3_live_hourly_entry_producer.py"]
VERSION_LABEL = "v166 direct lock-step predecessor fail-closed"
RESEAL_REASON = (
    "A configured direct predecessor must advance frozen Geometry/K9 into the "
    "exact next decision; otherwise live processing fails closed without "
    "archival scanning."
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads((ROOT / path).read_text())


def write_once(path: Path, payload: dict) -> str:
    target = ROOT / path
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if target.exists():
        if target.read_text() != rendered:
            raise RuntimeError(f"immutable successor differs: {path}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered)
    return sha(target)


def main() -> None:
    overlay0, execution0, auth0, state0 = map(read, (OLD_OVERLAY, OLD_EXECUTION, OLD_AUTH, OLD_STATE))
    expected = dict(overlay0["overrides"]["runtime_code_sha256"])
    mismatches = sorted(
        name for name, digest in expected.items()
        if not (ROOT / name).is_file() or sha(ROOT / name) != str(digest)
    )
    if mismatches != CHANGED:
        raise RuntimeError(f"refuse mixed runtime reseal: {mismatches}")

    overlay = copy.deepcopy(overlay0)
    overlay["overrides"]["runtime_code_sha256"][CHANGED[0]] = sha(ROOT / CHANGED[0])
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "changed_runtime_paths": CHANGED,
        "economic_contract_changed": False,
        "reason": RESEAL_REASON,
        "supersedes": str(OLD_OVERLAY),
    }
    overlay_hash = write_once(NEW_OVERLAY, overlay)

    auth = copy.deepcopy(auth0)
    auth.update({
        "inference_bundle": str(NEW_OVERLAY),
        "inference_bundle_sha256": overlay_hash,
        "authorization_source": f"User-approved {VERSION_LABEL} runtime reseal; models, calibration, admission, portfolio and exits unchanged.",
    })
    auth_hash = write_once(NEW_AUTH, auth)

    execution = copy.deepcopy(execution0)
    execution.update({
        "inference_bundle": str(NEW_OVERLAY),
        "inference_bundle_sha256": overlay_hash,
        "activation_authorization": str(NEW_AUTH),
        "activation_authorization_sha256": auth_hash,
        "version_note": f"{VERSION_LABEL}; unchanged economics and exit policy.",
    })
    execution["runtime_code_sha256"][CHANGED[0]] = sha(ROOT / CHANGED[0])
    execution.setdefault("runtime_reseal_predecessors", []).append({
        "current_inference_bundle_sha256": overlay_hash,
        "predecessor_inference_bundle": str(OLD_OVERLAY),
        "predecessor_inference_bundle_sha256": sha(ROOT / OLD_OVERLAY),
        "predecessor_execution": str(OLD_EXECUTION),
        "predecessor_execution_sha256": sha(ROOT / OLD_EXECUTION),
        "allowed_runtime_code_paths": CHANGED,
        "economic_contract_changed": False,
        "reason": RESEAL_REASON,
    })
    execution_hash = write_once(NEW_EXECUTION, execution)

    state = copy.deepcopy(state0)
    state["inference_bundle_sha256"] = overlay_hash
    state["activation_authorization_sha256"] = auth_hash
    state["contract_migration"] = {
        "schema": "strict_r3_live_state_contract_migration_v1",
        "prior_state": str(OLD_STATE), "prior_state_sha256": sha(ROOT / OLD_STATE),
        "inference_bundle_sha256": overlay_hash,
        "activation_authorization_sha256": auth_hash,
        "positions_preserved_exact": True,
        "processed_decisions_preserved_exact": True,
        "reason": f"{VERSION_LABEL} runtime behavior.",
    }
    state.setdefault("runtime_reseal_history", []).append({
        "changed_runtime_paths": CHANGED,
        "economic_contract_changed": False,
        "execution_bundle": str(NEW_EXECUTION), "execution_bundle_sha256": execution_hash,
        "inference_bundle": str(NEW_OVERLAY), "inference_bundle_sha256": overlay_hash,
    })
    state_hash = write_once(NEW_STATE, state)

    assert overlay0["overrides"].get("paths") == overlay["overrides"].get("paths")
    assert overlay0["overrides"].get("sha256") == overlay["overrides"].get("sha256")
    assert execution0.get("exit_policy_sha256") == execution.get("exit_policy_sha256")
    receipt = {
        "schema": "strict_r3_lockstep_failclosed_reseal_v1", "status": "pass",
        "economic_contract_changed": False, "changed_runtime_paths": CHANGED,
        "artifact_contract_preserved_exact": True, "exit_policy_preserved_exact": True,
        "successor": {"overlay": str(NEW_OVERLAY), "overlay_sha256": overlay_hash, "execution": str(NEW_EXECUTION), "execution_sha256": execution_hash, "authorization": str(NEW_AUTH), "authorization_sha256": auth_hash, "state": str(NEW_STATE), "state_sha256": state_hash},
    }
    write_once(RECEIPT, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
