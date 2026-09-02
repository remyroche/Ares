#!/usr/bin/env python3
"""Seal a live successor from the verified strict-R3 stateful recovery.

The recovery itself is shadow-only.  This utility is the explicit boundary
between its immutable terminal state and a future live producer: it requires a
complete no-order chain plus a same-predecessor terminal parity pass, points the
persisted-state contract at that terminal bundle, and records the one reviewed
Adaptive Exit runtime-dependency change.  It never migrates a live state or
starts a process.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v128_"
    "bcf_current_dual_samebundle21d_feature_state_reseal_recovery_rebind.json"
)
SOURCE_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260822_v56_"
    "v128_bcf_current_dual_samebundle21d_feature_state_reseal_recovery_rebind.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v130_v154_"
    "bcf_current_dual_samebundle21d_feature_state_reseal_recovery_rebind.json"
)
RECOVERY_ROOT = ROOT / (
    "data_perp/artifacts/strict_r3_stateful_recovery_v155_"
    "20260822T110000Z_130000Z_v2"
)
TERMINAL_RUN = RECOVERY_ROOT / "hour_20260822T130000Z" / "run"
TERMINAL_STATE = TERMINAL_RUN / "feature_state" / "bundle"
TERMINAL_PARITY = RECOVERY_ROOT / "terminal_state_replay_parity_manifest.json"

OUT_OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v129_"
    "bcf_current_dual_samebundle21d_recovered_terminal.json"
)
OUT_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260822_v57_"
    "v129_bcf_current_dual_samebundle21d_recovered_terminal.json"
)
OUT_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v131_v155_"
    "bcf_current_dual_samebundle21d_recovered_terminal.json"
)
OUT_REVIEW = RECOVERY_ROOT / "live_successor_review.json"

RUNTIME_PATH = "extreme_price_movements/path_based_exit_optimisation.py"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"immutable output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _resolved_runtime_hashes(overlay: dict) -> dict[str, str]:
    base = _read(ROOT / str(overlay["base_bundle"]))
    hashes = dict(base.get("runtime_code_sha256") or {})
    hashes.update(dict(overlay.get("overrides", {}).get("runtime_code_sha256") or {}))
    return hashes


def main() -> None:
    required = (
        SOURCE_OVERLAY, SOURCE_AUTHORIZATION, SOURCE_EXECUTION,
        RECOVERY_ROOT / "run_manifest.json", TERMINAL_RUN / "run_manifest.json",
        TERMINAL_STATE / "state_bundle_manifest.json", TERMINAL_PARITY,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if any(path.exists() for path in (
        OUT_OVERLAY, OUT_AUTHORIZATION, OUT_EXECUTION, OUT_REVIEW,
    )):
        raise FileExistsError("recovered-terminal successor outputs are immutable")

    recovery = _read(RECOVERY_ROOT / "run_manifest.json")
    parity = _read(TERMINAL_PARITY)
    terminal = _read(TERMINAL_RUN / "run_manifest.json")
    state = _read(TERMINAL_STATE / "state_bundle_manifest.json")
    if recovery.get("status") != "complete" or parity.get("status") != "pass":
        raise ValueError("requires complete recovery and passing terminal parity")
    if terminal.get("exchange_calls") != 0 or terminal.get("order_submission_enabled") is not False:
        raise ValueError("terminal recovery must be strictly no-order")
    if parity.get("full_raw_reconstruction_used") is not False:
        raise ValueError("terminal parity must use the persisted-state contract")

    source_overlay = _read(SOURCE_OVERLAY)
    source_hashes = _resolved_runtime_hashes(source_overlay)
    changed = [
        relative for relative, expected in source_hashes.items()
        if _sha(ROOT / relative) != expected
    ]
    if changed:
        raise ValueError(f"unreviewed overlay runtime deltas: {sorted(changed)}")
    if RUNTIME_PATH in source_hashes:
        raise ValueError("recovery runtime dependency unexpectedly already sealed")

    overlay = copy.deepcopy(source_overlay)
    runtime = overlay["overrides"]["runtime"]
    feature_state = runtime["feature_state"]
    expected_ts = str(state["expected_state_timestamp"])
    feature_state.update({
        "initial_seed_bundle": str(TERMINAL_STATE.relative_to(ROOT)),
        "initial_seed_expected_state_timestamp": expected_ts,
        "initial_seed_manifest_sha256": _sha(TERMINAL_STATE / "state_bundle_manifest.json"),
        "recovered_terminal_state": {
            "recovery_root": str(RECOVERY_ROOT.relative_to(ROOT)),
            "recovery_manifest_sha256": _sha(RECOVERY_ROOT / "run_manifest.json"),
            "terminal_run": str(TERMINAL_RUN.relative_to(ROOT)),
            "terminal_run_manifest_sha256": _sha(TERMINAL_RUN / "run_manifest.json"),
            "same_predecessor_parity": str(TERMINAL_PARITY.relative_to(ROOT)),
            "same_predecessor_parity_sha256": _sha(TERMINAL_PARITY),
            "geometry_bundle_sha256": terminal["inference_bundle_audit"]["geometry_bundle_sha256"],
            "counterfactual_recovery_only": True,
        },
    })
    overlay_hashes = dict(overlay["overrides"].get("runtime_code_sha256") or {})
    overlay_hashes[RUNTIME_PATH] = _sha(ROOT / RUNTIME_PATH)
    overlay["overrides"]["runtime_code_sha256"] = overlay_hashes
    overlay["purpose"] = (
        "v129: live successor from the verified no-order 09:00–13:00 stateful "
        "recovery. The producer may begin only at a fresh future decision; "
        "recovered decisions remain simulation-only."
    )
    overlay["runtime_reseal"] = {
        "schema": "strict_r3_runtime_reseal_v1",
        "supersedes": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "changed_runtime_paths": [RUNTIME_PATH],
        "economic_contract_changed": False,
        "reason": (
            "Adaptive Exit state construction now embeds an exact-tested copy "
            "of the deployable SimplePolicy barrier primitive instead of "
            "importing the heavyweight research optimiser for open positions."
        ),
    }
    _write_new(OUT_OVERLAY, overlay)

    authorization = copy.deepcopy(_read(SOURCE_AUTHORIZATION))
    authorization.update({
        "authorization_source": (
            "User-approved live continuation after the verified no-order "
            "stateful recovery and same-predecessor terminal parity pass."
        ),
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": _sha(OUT_OVERLAY),
    })
    _write_new(OUT_AUTHORIZATION, authorization)

    execution = copy.deepcopy(_read(SOURCE_EXECUTION))
    execution_hashes = dict(execution.get("runtime_code_sha256") or {})
    execution_changed = [
        relative for relative, expected in execution_hashes.items()
        if _sha(ROOT / relative) != expected
    ]
    if execution_changed:
        raise ValueError(f"unreviewed execution runtime deltas: {sorted(execution_changed)}")
    execution.update({
        "inference_bundle": str(OUT_OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": _sha(OUT_OVERLAY),
        "activation_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "activation_authorization_sha256": _sha(OUT_AUTHORIZATION),
        "version_note": (
            "v155: v129 recovered-terminal state successor. Models, geometry, "
            "dual BCF/current MC1 admission, portfolio, and rich exit economics "
            "remain unchanged."
        ),
    })
    predecessors = list(execution.get("runtime_reseal_predecessors") or [])
    predecessors.append({
        "successor_execution_semantics": "bcf_current_dual_recovered_terminal_v1",
        "predecessor_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "predecessor_execution_sha256": _sha(SOURCE_EXECUTION),
        "allowed_runtime_code_paths": [RUNTIME_PATH],
        "economic_contract_changed": False,
        "reason": "Verified recovered terminal state and exact local policy primitive.",
    })
    execution["runtime_reseal_predecessors"] = predecessors
    _write_new(OUT_EXECUTION, execution)

    review = {
        "schema": "strict_r3_recovered_terminal_live_successor_review_v1",
        "status": "pass",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "source_overlay": str(SOURCE_OVERLAY.relative_to(ROOT)),
        "source_overlay_sha256": _sha(SOURCE_OVERLAY),
        "successor_overlay": str(OUT_OVERLAY.relative_to(ROOT)),
        "successor_overlay_sha256": _sha(OUT_OVERLAY),
        "source_authorization": str(SOURCE_AUTHORIZATION.relative_to(ROOT)),
        "successor_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "successor_authorization_sha256": _sha(OUT_AUTHORIZATION),
        "source_execution": str(SOURCE_EXECUTION.relative_to(ROOT)),
        "successor_execution": str(OUT_EXECUTION.relative_to(ROOT)),
        "successor_execution_sha256": _sha(OUT_EXECUTION),
        "recovery_root": str(RECOVERY_ROOT.relative_to(ROOT)),
        "terminal_state": str(TERMINAL_STATE.relative_to(ROOT)),
        "terminal_state_manifest_sha256": _sha(TERMINAL_STATE / "state_bundle_manifest.json"),
        "terminal_parity": str(TERMINAL_PARITY.relative_to(ROOT)),
        "terminal_parity_sha256": _sha(TERMINAL_PARITY),
        "changed_runtime_paths": [RUNTIME_PATH],
        "economic_contract_changed": False,
        "validation": {
            "recovery_complete": True,
            "terminal_same_predecessor_parity": True,
            "terminal_no_order": True,
            "frozen_geometry_unchanged": True,
            "fresh_future_decision_required": True,
        },
    }
    _write_new(OUT_REVIEW, review)
    print(json.dumps({"status": "pass", **review}, sort_keys=True))


if __name__ == "__main__":
    main()
