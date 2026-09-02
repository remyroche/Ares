#!/usr/bin/env python3
"""Advance the sealed recovery successor through the recovered 10:00 hour.

The v83 successor deliberately permits no generic historical predecessor.  It
is superseded here only after the appended v60-derived 10:00 recovery and its
same-predecessor stateful parity replay both pass.  No model or policy changes
are made.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_RECOVERY = ROOT / "data_perp/artifacts/strict_r3_stateful_recovery_v60_20260817T200000Z_20260818T090000Z_v4"
RECOVERY_ROOT = ROOT / "data_perp/artifacts/strict_r3_stateful_recovery_v60_extension_20260818T100000Z_v5"
FINAL_RUN = RECOVERY_ROOT / "hour_20260818T100000Z/run"
FINAL_BUNDLE = FINAL_RUN / "feature_state/bundle"
PARITY = RECOVERY_ROOT / "terminal_state_replay_parity_manifest.json"
SOURCE_OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v83_bcf_current_dual_recovered_terminal_state.json"
SOURCE_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v81_v103_bcf_current_dual_recovered_terminal_state.json"
SOURCE_AUTH = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260818_v11_bcf_current_dual_recovered_terminal_state.json"
OUT_OVERLAY = ROOT / "config/strict_r3_inference_overlay_long_20260801_v84_bcf_current_dual_recovered_terminal_1000.json"
OUT_EXECUTION = ROOT / "config/strict_r3_kraken_live_execution_v82_v104_bcf_current_dual_recovered_terminal_1000.json"
OUT_AUTH = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260818_v12_bcf_current_dual_recovered_terminal_1000.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _assert_completed(root: Path, final_run: Path, parity: Path) -> None:
    recovery = load(root / "run_manifest.json")
    terminal = load(final_run / "run_manifest.json")
    state = load(final_run / "feature_state/bundle/state_bundle_manifest.json")
    replay = load(parity)
    if not (
        recovery.get("schema") == "strict_r3_stateful_recovery_v1"
        and recovery.get("status") == "complete"
        and recovery.get("final_run") == relative(final_run)
        and int(recovery.get("exchange_calls", -1)) == 0
        and recovery.get("order_submission_enabled") is False
        and replay.get("status") == "pass"
        and replay.get("schema") == "strict_r3_stateful_recovery_terminal_replay_parity_v1"
        and terminal.get("stateful_feature_contract_hash") == state.get("feature_contract_sha256")
    ):
        raise AssertionError(f"incomplete or unvalidated recovery: {root}")


def main() -> None:
    _assert_completed(
        BASE_RECOVERY,
        BASE_RECOVERY / "hour_20260818T090000Z/run",
        BASE_RECOVERY / "terminal_state_replay_parity_manifest.json",
    )
    _assert_completed(RECOVERY_ROOT, FINAL_RUN, PARITY)
    extension = load(RECOVERY_ROOT / "run_manifest.json")
    if extension.get("bootstrap_run") != relative(BASE_RECOVERY / "hour_20260818T090000Z/run"):
        raise AssertionError("10:00 extension does not descend from the completed v60-derived 09:00 terminal")

    overlay = load(SOURCE_OVERLAY)
    for source, expected in dict(overlay["overrides"]["runtime_code_sha256"]).items():
        if sha(ROOT / source) != expected:
            raise AssertionError(f"unreviewed inference runtime source change: {source}")
    state = load(FINAL_BUNDLE / "state_bundle_manifest.json")
    feature_state = overlay["overrides"]["runtime"]["feature_state"]
    feature_state["initial_seed_bundle"] = relative(FINAL_BUNDLE)
    feature_state["initial_seed_expected_state_timestamp"] = str(state["expected_state_timestamp"])
    feature_state["initial_seed_manifest_sha256"] = sha(FINAL_BUNDLE / "state_bundle_manifest.json")
    overlay["purpose"] = (
        "v84: canonical dual-admission runtime seeded only from the completed "
        "v60-derived recovery chain through 10:00 UTC. Frozen models, feature "
        "contract, Geometry/K9, admission, portfolio policy and rich exit are unchanged."
    )
    overlay["stateful_recovery_successor"] = {
        "schema": "strict_r3_stateful_recovery_successor_v1",
        "source_bootstrap": "v60_only",
        "recovery_chain": [relative(BASE_RECOVERY), relative(RECOVERY_ROOT)],
        "terminal_run": relative(FINAL_RUN),
        "terminal_run_manifest_sha256": sha(FINAL_RUN / "run_manifest.json"),
        "terminal_feature_state_bundle": relative(FINAL_BUNDLE),
        "terminal_feature_state_manifest_sha256": sha(FINAL_BUNDLE / "state_bundle_manifest.json"),
        "terminal_stateful_replay_parity": relative(PARITY),
        "terminal_stateful_replay_parity_sha256": sha(PARITY),
        "orders_during_recovery": 0,
        "allowed_live_predecessor": relative(FINAL_RUN),
    }
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": relative(SOURCE_OVERLAY),
        "changed_runtime_paths": [],
        "reason": "Advance the sealed v60-derived stateful recovery successor through its validated 10:00 terminal.",
    }
    write_new(OUT_OVERLAY, overlay)

    authorization = load(SOURCE_AUTH)
    authorization.update({
        "authorization_source": "User-approved live dual-admission stack, restarted only after the v60-derived recovery chain through 10:00 UTC and terminal parity passes.",
        "inference_bundle": relative(OUT_OVERLAY),
        "inference_bundle_sha256": sha(OUT_OVERLAY),
        "recovery_successor_manifest": relative(RECOVERY_ROOT / "run_manifest.json"),
        "recovery_successor_manifest_sha256": sha(RECOVERY_ROOT / "run_manifest.json"),
    })
    write_new(OUT_AUTH, authorization)

    execution = load(SOURCE_EXECUTION)
    execution_hashes = {source: sha(ROOT / source) for source in execution["runtime_code_sha256"]}
    execution.update({
        "version_note": "v104: sealed only to the validated v60-derived terminal recovery through 10:00 UTC; strategy and policy unchanged.",
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
        "recovery_terminal": relative(FINAL_RUN), "parity": relative(PARITY),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
