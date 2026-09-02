#!/usr/bin/env python3
"""Seal the bounded Kraken market-cache successor for the warmed executor.

Only the persistent hourly producer changes.  Models, features, frozen
Geometry/K9, calibration, admission, portfolio rules and the rich exit policy
remain byte-identical to the v160 successor.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD_OVERLAY = Path(
    "config/strict_r3_inference_overlay_long_20260801_v137_bcf_current_dual_samebundle21d_direct_15m_reference_lockstep_geometry_feature_parity_warm_execution_fresh_boundary.json"
)
NEW_OVERLAY = Path(
    "config/strict_r3_inference_overlay_long_20260801_v138_bcf_current_dual_samebundle21d_direct_15m_reference_lockstep_geometry_feature_parity_warm_execution_market_cache.json"
)
OLD_EXECUTION = Path(
    "config/strict_r3_kraken_live_execution_v138_v162_warm_execution_fresh_boundary_live.json"
)
NEW_EXECUTION = Path(
    "config/strict_r3_kraken_live_execution_v139_v163_warm_execution_market_cache_live.json"
)
OLD_AUTHORIZATION = Path(
    "config/strict_r3_kraken_live_activation_authorization_20260822_v64_v137_warm_execution_fresh_boundary_live.json"
)
NEW_AUTHORIZATION = Path(
    "config/strict_r3_kraken_live_activation_authorization_20260822_v65_v138_warm_execution_market_cache_live.json"
)
OLD_STATE = Path(
    "data_perp/live/strict_r3_kraken_live_state_v95_v162_warm_execution_fresh_boundary_live.json"
)
NEW_STATE = Path(
    "data_perp/live/strict_r3_kraken_live_state_v96_v163_warm_execution_market_cache_live.json"
)
RECEIPT = Path(
    "data_perp/artifacts/strict_r3_warm_execution_market_cache_reseal_20260822_v1/run_manifest.json"
)
CHANGED_PATH = "extreme_price_movements/data_store.py"


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
    changed = sorted(
        path for path, expected in old_runtime.items()
        if not (ROOT / path).is_file() or _sha(ROOT / path) != str(expected)
    )
    if changed != [CHANGED_PATH]:
        raise RuntimeError(f"refuse mixed reseal: {changed}")
    if old_execution.get("runtime_code_sha256", {}).get(CHANGED_PATH) != old_runtime.get(CHANGED_PATH):
        raise RuntimeError("active overlay/execution producer hashes differ")

    new_overlay = copy.deepcopy(old_overlay)
    new_overlay["overrides"]["runtime_code_sha256"][CHANGED_PATH] = producer_hash
    new_overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "changed_runtime_paths": [CHANGED_PATH],
        "economic_contract_changed": False,
        "reason": "The warmed executor may restore an atomic, freshness-bounded local cache of Kraken market definitions before a decision instead of blocking on a full remote market reload. It contains no prices, positions, features or model inputs.",
        "supersedes": str(OLD_OVERLAY),
    }
    overlay_hash = _write_once(NEW_OVERLAY, new_overlay)

    new_auth = copy.deepcopy(old_authorization)
    new_auth["inference_bundle"] = str(NEW_OVERLAY)
    new_auth["inference_bundle_sha256"] = overlay_hash
    new_auth["authorization_source"] = "User-approved bounded market-cache reseal; economic and exit-policy artifacts unchanged."
    auth_hash = _write_once(NEW_AUTHORIZATION, new_auth)

    new_execution = copy.deepcopy(old_execution)
    new_execution["inference_bundle"] = str(NEW_OVERLAY)
    new_execution["inference_bundle_sha256"] = overlay_hash
    new_execution["activation_authorization"] = str(NEW_AUTHORIZATION)
    new_execution["activation_authorization_sha256"] = auth_hash
    new_execution["runtime_code_sha256"][CHANGED_PATH] = producer_hash
    new_execution.setdefault("runtime_reseal_predecessors", []).append({
        "current_inference_bundle_sha256": overlay_hash,
        "predecessor_inference_bundle": str(OLD_OVERLAY),
        "predecessor_inference_bundle_sha256": _sha(ROOT / OLD_OVERLAY),
        "predecessor_execution": str(OLD_EXECUTION),
        "predecessor_execution_sha256": _sha(ROOT / OLD_EXECUTION),
        "allowed_runtime_code_paths": [CHANGED_PATH],
        "added_runtime_code_paths": [],
        "economic_contract_changed": False,
        "reviewed_current_runtime": True,
        "reason": "A bounded local cache restores only previously fetched Kraken market definitions for warm execution startup; all live price/book/account calls and entry/exit safeguards are unchanged.",
        "successor_execution_semantics": "warm_in_process_execution_market_cache_v1",
    })
    new_execution["version_note"] = "v163 warmed execution boundary with bounded cached Kraken market definitions; unchanged economic/artifact contract."
    execution_hash = _write_once(NEW_EXECUTION, new_execution)

    new_state = copy.deepcopy(old_state)
    new_state["inference_bundle_sha256"] = overlay_hash
    new_state["activation_authorization_sha256"] = auth_hash
    new_state["contract_migration"] = {
        "schema": "strict_r3_live_state_contract_migration_v1",
        "prior_state": str(OLD_STATE),
        "prior_state_sha256": _sha(ROOT / OLD_STATE),
        "inference_bundle_sha256": overlay_hash,
        "activation_authorization_sha256": auth_hash,
        "positions_preserved_exact": True,
        "processed_decisions_preserved_exact": True,
        "reason": "Warm in-process execution bounded market-cache runtime reseal.",
    }
    new_state.setdefault("runtime_reseal_history", []).append({
        "changed_runtime_paths": [CHANGED_PATH],
        "economic_contract_changed": False,
        "execution_bundle": str(NEW_EXECUTION),
        "execution_bundle_sha256": execution_hash,
        "inference_bundle": str(NEW_OVERLAY),
        "inference_bundle_sha256": overlay_hash,
    })
    state_hash = _write_once(NEW_STATE, new_state)

    if old_overlay.get("overrides", {}).get("paths") != new_overlay.get("overrides", {}).get("paths"):
        raise AssertionError("model/feature/policy paths changed")
    if old_overlay.get("overrides", {}).get("sha256") != new_overlay.get("overrides", {}).get("sha256"):
        raise AssertionError("model/feature/policy hashes changed")
    if old_execution.get("exit_policy_sha256") != new_execution.get("exit_policy_sha256"):
        raise AssertionError("exit policy changed")

    receipt = {
        "schema": "strict_r3_warm_execution_market_cache_reseal_v1",
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
        "producer_sha256": producer_hash,
    }
    _write_once(RECEIPT, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
