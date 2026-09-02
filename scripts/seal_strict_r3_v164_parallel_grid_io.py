#!/usr/bin/env python3
"""Seal the v164 target-free-grid parallel-read runtime successor.

This successor changes only the bounded concurrency used to load independent
point-in-time source files.  It preserves the candidate universe, field
definitions, eligibility rules, frozen artifacts and exit policy exactly.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD_OVERLAY = Path(
    "config/strict_r3_inference_overlay_long_20260801_v138_bcf_current_dual_samebundle21d_direct_15m_reference_lockstep_geometry_feature_parity_warm_execution_market_cache.json"
)
NEW_OVERLAY = Path(
    "config/strict_r3_inference_overlay_long_20260801_v139_bcf_current_dual_samebundle21d_direct_15m_reference_lockstep_geometry_feature_parity_warm_execution_market_cache_parallel_grid.json"
)
OLD_EXECUTION = Path(
    "config/strict_r3_kraken_live_execution_v139_v163_warm_execution_market_cache_live.json"
)
NEW_EXECUTION = Path(
    "config/strict_r3_kraken_live_execution_v140_v164_parallel_grid_io_live.json"
)
OLD_AUTHORIZATION = Path(
    "config/strict_r3_kraken_live_activation_authorization_20260822_v65_v138_warm_execution_market_cache_live.json"
)
NEW_AUTHORIZATION = Path(
    "config/strict_r3_kraken_live_activation_authorization_20260822_v66_v139_parallel_grid_io_live.json"
)
OLD_STATE = Path(
    "data_perp/live/strict_r3_kraken_live_state_v96_v163_warm_execution_market_cache_live.json"
)
NEW_STATE = Path(
    "data_perp/live/strict_r3_kraken_live_state_v97_v164_parallel_grid_io_live.json"
)
RECEIPT = Path(
    "data_perp/artifacts/strict_r3_parallel_grid_io_reseal_20260822_v1/run_manifest.json"
)
CHANGED_PATH = "scripts/materialize_strict_r3_target_free_hourly_grid_v2.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(relative: Path) -> dict:
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
    for path in (OLD_OVERLAY, OLD_EXECUTION, OLD_AUTHORIZATION, OLD_STATE):
        if not (ROOT / path).is_file():
            raise FileNotFoundError(path)
    old_overlay = _read(OLD_OVERLAY)
    old_execution = _read(OLD_EXECUTION)
    old_auth = _read(OLD_AUTHORIZATION)
    old_state = _read(OLD_STATE)
    old_runtime = dict(old_overlay["overrides"]["runtime_code_sha256"])
    changed = sorted(
        name for name, expected in old_runtime.items()
        if not (ROOT / name).is_file() or _sha(ROOT / name) != str(expected)
    )
    if changed != [CHANGED_PATH]:
        raise RuntimeError(f"refuse mixed runtime reseal: {changed}")

    overlay = copy.deepcopy(old_overlay)
    overlay["overrides"]["runtime_code_sha256"][CHANGED_PATH] = _sha(ROOT / CHANGED_PATH)
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "changed_runtime_paths": [CHANGED_PATH],
        "economic_contract_changed": False,
        "reason": (
            "Independent symbol-local Parquet reads use a deterministic bounded "
            "pool. Universe order, point-in-time source values and all eligibility "
            "rules are unchanged."
        ),
        "supersedes": str(OLD_OVERLAY),
    }
    overlay_hash = _write_once(NEW_OVERLAY, overlay)

    auth = copy.deepcopy(old_auth)
    auth["inference_bundle"] = str(NEW_OVERLAY)
    auth["inference_bundle_sha256"] = overlay_hash
    auth["authorization_source"] = (
        "User-approved v164 deterministic parallel target-free-grid source-read "
        "runtime reseal; economic and exit-policy artifacts unchanged."
    )
    auth_hash = _write_once(NEW_AUTHORIZATION, auth)

    execution = copy.deepcopy(old_execution)
    execution["inference_bundle"] = str(NEW_OVERLAY)
    execution["inference_bundle_sha256"] = overlay_hash
    execution["activation_authorization"] = str(NEW_AUTHORIZATION)
    execution["activation_authorization_sha256"] = auth_hash
    execution["runtime_code_sha256"][CHANGED_PATH] = _sha(ROOT / CHANGED_PATH)
    execution.setdefault("runtime_reseal_predecessors", []).append({
        "current_inference_bundle_sha256": overlay_hash,
        "predecessor_inference_bundle": str(OLD_OVERLAY),
        "predecessor_inference_bundle_sha256": _sha(ROOT / OLD_OVERLAY),
        "predecessor_execution": str(OLD_EXECUTION),
        "predecessor_execution_sha256": _sha(ROOT / OLD_EXECUTION),
        "allowed_runtime_code_paths": [CHANGED_PATH],
        "economic_contract_changed": False,
        "reason": "Bounded deterministic parallel reads of independent target-free grid source files.",
    })
    execution["version_note"] = (
        "v164: deterministic parallel target-free-grid I/O; unchanged models, "
        "calibration, admission, portfolio and rich exit-policy contract."
    )
    execution_hash = _write_once(NEW_EXECUTION, execution)

    state = copy.deepcopy(old_state)
    state["inference_bundle_sha256"] = overlay_hash
    state["activation_authorization_sha256"] = auth_hash
    state["contract_migration"] = {
        "schema": "strict_r3_live_state_contract_migration_v1",
        "prior_state": str(OLD_STATE),
        "prior_state_sha256": _sha(ROOT / OLD_STATE),
        "inference_bundle_sha256": overlay_hash,
        "activation_authorization_sha256": auth_hash,
        "positions_preserved_exact": True,
        "processed_decisions_preserved_exact": True,
        "reason": "v164 deterministic parallel target-free-grid I/O runtime reseal.",
    }
    state.setdefault("runtime_reseal_history", []).append({
        "changed_runtime_paths": [CHANGED_PATH],
        "economic_contract_changed": False,
        "execution_bundle": str(NEW_EXECUTION),
        "execution_bundle_sha256": execution_hash,
        "inference_bundle": str(NEW_OVERLAY),
        "inference_bundle_sha256": overlay_hash,
    })
    state_hash = _write_once(NEW_STATE, state)

    if old_overlay["overrides"].get("paths") != overlay["overrides"].get("paths"):
        raise AssertionError("model/feature/policy paths changed")
    if old_overlay["overrides"].get("sha256") != overlay["overrides"].get("sha256"):
        raise AssertionError("model/feature/policy hashes changed")
    if old_execution.get("exit_policy_sha256") != execution.get("exit_policy_sha256"):
        raise AssertionError("exit policy changed")

    receipt = {
        "schema": "strict_r3_parallel_grid_io_reseal_v1",
        "status": "pass",
        "economic_contract_changed": False,
        "changed_runtime_paths": [CHANGED_PATH],
        "prior": {"overlay": str(OLD_OVERLAY), "execution": str(OLD_EXECUTION), "state": str(OLD_STATE)},
        "successor": {
            "overlay": str(NEW_OVERLAY), "overlay_sha256": overlay_hash,
            "execution": str(NEW_EXECUTION), "execution_sha256": execution_hash,
            "authorization": str(NEW_AUTHORIZATION), "authorization_sha256": auth_hash,
            "state": str(NEW_STATE), "state_sha256": state_hash,
        },
        "artifact_contract_preserved_exact": True,
        "exit_policy_preserved_exact": True,
    }
    _write_once(RECEIPT, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
