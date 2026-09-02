#!/usr/bin/env python3
"""Seal the v60-derived completed recovery terminal as the only live seed.

This is a lineage-only successor.  It verifies the completed no-order recovery
and its independent persisted-state replay receipt before binding the final
recovered feature/K9 bundle into new inference, authorization and execution
contracts.  Models, feature fields, dual admission, portfolio rules and the
rich exit policy are copied byte-for-byte from v82/v102.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECOVERY_ROOT = ROOT / "data_perp/artifacts/strict_r3_stateful_recovery_v60_20260817T200000Z_20260818T090000Z_v4"
FINAL_RUN = RECOVERY_ROOT / "hour_20260818T090000Z/run"
FINAL_BUNDLE = FINAL_RUN / "feature_state/bundle"
PARITY = RECOVERY_ROOT / "terminal_state_replay_parity_manifest.json"
SOURCE_OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v82_bcf_current_dual_v60_stateful_recovery_bridge.json"
SOURCE_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v80_v102_bcf_current_dual_v60_stateful_recovery_bridge.json"
SOURCE_AUTH = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260818_v10_bcf_current_dual_v60_stateful_recovery_bridge.json"
OUT_OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v83_bcf_current_dual_recovered_terminal_state.json"
OUT_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v81_v103_bcf_current_dual_recovered_terminal_state.json"
OUT_AUTH = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260818_v11_bcf_current_dual_recovered_terminal_state.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def main() -> None:
    recovery = load(RECOVERY_ROOT / "run_manifest.json")
    parity = load(PARITY)
    final = load(FINAL_RUN / "run_manifest.json")
    state = load(FINAL_BUNDLE / "state_bundle_manifest.json")
    if not (
        recovery.get("schema") == "strict_r3_stateful_recovery_v1"
        and recovery.get("status") == "complete"
        and recovery.get("final_run") == relative(FINAL_RUN)
        and int(recovery.get("exchange_calls", -1)) == 0
        and recovery.get("order_submission_enabled") is False
    ):
        raise AssertionError("recovery root is not a completed zero-order chain")
    if not (
        parity.get("schema") == "strict_r3_stateful_recovery_terminal_replay_parity_v1"
        and parity.get("status") == "pass"
    ):
        raise AssertionError("terminal persisted-state parity receipt did not pass")
    if not (
        final.get("decision_ts") == "2026-08-18T09:00:00+00:00"
        and final.get("stateful_feature_contract_hash") == state.get("feature_contract_sha256")
    ):
        raise AssertionError("terminal run and feature bundle do not agree")

    overlay = load(SOURCE_OVERLAY)
    current_hashes = {}
    for source, expected in dict(overlay["overrides"]["runtime_code_sha256"]).items():
        observed = sha(ROOT / source)
        if observed != expected:
            raise AssertionError(f"unreviewed inference runtime source change: {source}")
        current_hashes[source] = observed
    feature_state = overlay["overrides"]["runtime"]["feature_state"]
    feature_state["initial_seed_bundle"] = relative(FINAL_BUNDLE)
    feature_state["initial_seed_expected_state_timestamp"] = str(state["expected_state_timestamp"])
    feature_state["initial_seed_manifest_sha256"] = sha(FINAL_BUNDLE / "state_bundle_manifest.json")
    overlay["overrides"]["runtime_code_sha256"] = current_hashes
    overlay["purpose"] = (
        "v83: canonical dual-admission runtime seeded only from the completed "
        "v60-derived terminal recovery state. Frozen models, feature contract, "
        "Geometry/K9, admission, portfolio policy and rich exit are unchanged."
    )
    overlay["stateful_recovery_successor"] = {
        "schema": "strict_r3_stateful_recovery_successor_v1",
        "source_bootstrap": "v60_only",
        "recovery_root": relative(RECOVERY_ROOT),
        "recovery_manifest_sha256": sha(RECOVERY_ROOT / "run_manifest.json"),
        "terminal_run": relative(FINAL_RUN),
        "terminal_run_manifest_sha256": sha(FINAL_RUN / "run_manifest.json"),
        "terminal_feature_state_bundle": relative(FINAL_BUNDLE),
        "terminal_feature_state_manifest_sha256": sha(FINAL_BUNDLE / "state_bundle_manifest.json"),
        "terminal_stateful_replay_parity": relative(PARITY),
        "terminal_stateful_replay_parity_sha256": sha(PARITY),
        "recovered_hours": int(recovery.get("recovered_hours", 0)),
        "orders_during_recovery": 0,
        "allowed_live_predecessor": relative(FINAL_RUN),
    }
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": relative(SOURCE_OVERLAY),
        "changed_runtime_paths": [],
        "reason": "Bind the validated completed v60-derived recovery terminal as the sole live state seed.",
    }
    write_new(OUT_OVERLAY, overlay)

    authorization = load(SOURCE_AUTH)
    authorization.update({
        "authorization_source": "User-approved live dual-admission stack, restarted only after completed v60-derived no-order recovery and terminal parity pass.",
        "inference_bundle": relative(OUT_OVERLAY),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "recovery_successor_manifest": relative(RECOVERY_ROOT / "run_manifest.json"),
        "recovery_successor_manifest_sha256": sha(RECOVERY_ROOT / "run_manifest.json"),
    })
    write_new(OUT_AUTH, authorization)

    execution = load(SOURCE_EXECUTION)
    execution_hashes = {}
    for source in dict(execution["runtime_code_sha256"]):
        execution_hashes[source] = sha(ROOT / source)
    execution.update({
        "version_note": "v103: only the validated v60-derived terminal recovery state may seed a live producer; strategy and policy unchanged.",
        "inference_bundle": relative(OUT_OVERLAY),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "activation_authorization": relative(OUT_AUTH),
        "activation_authorization_sha256": sha(OUT_AUTH),
        "runtime_code_sha256": execution_hashes,
        "runtime_reseal_predecessors": [],
        "stateful_recovery_successor": overlay["stateful_recovery_successor"],
    })
    write_new(OUT_EXECUTION, execution)
    print(json.dumps({
        "overlay": relative(OUT_OVERLAY), "overlay_sha256": sha(OUT_OVERLAY),
        "authorization": relative(OUT_AUTH), "authorization_sha256": sha(OUT_AUTH),
        "execution": relative(OUT_EXECUTION), "execution_sha256": sha(OUT_EXECUTION),
        "recovery_terminal": relative(FINAL_RUN),
        "parity": relative(PARITY),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
