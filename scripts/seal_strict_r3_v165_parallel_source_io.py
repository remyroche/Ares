#!/usr/bin/env python3
"""Seal the v165 deterministic parallel source-I/O successor."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD_OVERLAY = Path("config/strict_r3_inference_overlay_long_20260801_v139_bcf_current_dual_samebundle21d_direct_15m_reference_lockstep_geometry_feature_parity_warm_execution_market_cache_parallel_grid.json")
NEW_OVERLAY = Path("config/strict_r3_inference_overlay_long_20260801_v140_bcf_current_dual_samebundle21d_direct_15m_reference_lockstep_geometry_feature_parity_warm_execution_parallel_source_io.json")
OLD_EXECUTION = Path("config/strict_r3_kraken_live_execution_v140_v164_parallel_grid_io_live.json")
NEW_EXECUTION = Path("config/strict_r3_kraken_live_execution_v141_v165_parallel_source_io_live.json")
OLD_AUTH = Path("config/strict_r3_kraken_live_activation_authorization_20260822_v66_v139_parallel_grid_io_live.json")
NEW_AUTH = Path("config/strict_r3_kraken_live_activation_authorization_20260822_v67_v140_parallel_source_io_live.json")
OLD_STATE = Path("data_perp/live/strict_r3_kraken_live_state_v97_v164_parallel_grid_io_live.json")
NEW_STATE = Path("data_perp/live/strict_r3_kraken_live_state_v98_v165_parallel_source_io_live.json")
RECEIPT = Path("data_perp/artifacts/strict_r3_parallel_source_io_reseal_20260822_v1/run_manifest.json")
CHANGED = [
    "scripts/materialize_strict_r3_target_free_hourly_grid_v2.py",
    "scripts/run_tp6_sl4_exact170_canonical_consensus.py",
]


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
    for path in (OLD_OVERLAY, OLD_EXECUTION, OLD_AUTH, OLD_STATE):
        if not (ROOT / path).is_file():
            raise FileNotFoundError(path)
    overlay0, execution0, auth0, state0 = map(read, (OLD_OVERLAY, OLD_EXECUTION, OLD_AUTH, OLD_STATE))
    expected = dict(overlay0["overrides"]["runtime_code_sha256"])
    mismatches = sorted(
        name for name, digest in expected.items()
        if not (ROOT / name).is_file() or sha(ROOT / name) != str(digest)
    )
    if mismatches != CHANGED:
        raise RuntimeError(f"refuse mixed runtime reseal: {mismatches}")

    overlay = copy.deepcopy(overlay0)
    for name in CHANGED:
        overlay["overrides"]["runtime_code_sha256"][name] = sha(ROOT / name)
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "changed_runtime_paths": CHANGED,
        "economic_contract_changed": False,
        "reason": (
            "Independent target-free-grid and source-panel Parquet reads are "
            "bounded-parallel. Source precedence, values, universe order, "
            "eligibility and all economic artifacts are unchanged."
        ),
        "supersedes": str(OLD_OVERLAY),
    }
    overlay_hash = write_once(NEW_OVERLAY, overlay)

    auth = copy.deepcopy(auth0)
    auth.update({
        "inference_bundle": str(NEW_OVERLAY),
        "inference_bundle_sha256": overlay_hash,
        "authorization_source": "User-approved v165 deterministic parallel source I/O runtime reseal; economic and exit artifacts unchanged.",
    })
    auth_hash = write_once(NEW_AUTH, auth)

    execution = copy.deepcopy(execution0)
    execution.update({
        "inference_bundle": str(NEW_OVERLAY),
        "inference_bundle_sha256": overlay_hash,
        "activation_authorization": str(NEW_AUTH),
        "activation_authorization_sha256": auth_hash,
        "version_note": "v165 deterministic parallel source I/O; unchanged model, calibration, admission, portfolio and rich exit-policy contract.",
    })
    for name in CHANGED:
        execution["runtime_code_sha256"][name] = sha(ROOT / name)
    execution.setdefault("runtime_reseal_predecessors", []).append({
        "current_inference_bundle_sha256": overlay_hash,
        "predecessor_inference_bundle": str(OLD_OVERLAY),
        "predecessor_inference_bundle_sha256": sha(ROOT / OLD_OVERLAY),
        "predecessor_execution": str(OLD_EXECUTION),
        "predecessor_execution_sha256": sha(ROOT / OLD_EXECUTION),
        "allowed_runtime_code_paths": CHANGED,
        "economic_contract_changed": False,
        "reason": "Bounded deterministic parallel reads of independent point-in-time source files.",
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
        "reason": "v165 deterministic parallel point-in-time source reads.",
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
        "schema": "strict_r3_parallel_source_io_reseal_v1", "status": "pass",
        "economic_contract_changed": False, "changed_runtime_paths": CHANGED,
        "artifact_contract_preserved_exact": True, "exit_policy_preserved_exact": True,
        "successor": {"overlay": str(NEW_OVERLAY), "overlay_sha256": overlay_hash, "execution": str(NEW_EXECUTION), "execution_sha256": execution_hash, "authorization": str(NEW_AUTH), "authorization_sha256": auth_hash, "state": str(NEW_STATE), "state_sha256": state_hash},
    }
    write_once(RECEIPT, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
