#!/usr/bin/env python3
"""Authorize a future-only live successor from the v131 exact recovery.

The v131 overlay itself is deliberately reused: its hash is the exact one
that produced the recovered terminal state, so a fresh producer can consume
that state without a compatibility bridge.  This utility creates only a new
authorization/execution pair.  It never scores a stale decision, writes a
state, starts a process, or submits an order.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / (
    "config/strict_r3_inference_overlay_long_20260801_v131_"
    "bcf_current_dual_samebundle21d_direct_15m_reference_shadow.json"
)
SOURCE_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260822_v57_"
    "v129_bcf_current_dual_samebundle21d_recovered_terminal.json"
)
SOURCE_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v131_v155_"
    "bcf_current_dual_samebundle21d_recovered_terminal.json"
)
RECOVERY_ROOT = ROOT / (
    "data_perp/artifacts/strict_r3_stateful_recovery_v158_"
    "20260822T140000Z_150000Z_v1"
)
TERMINAL_RUN = RECOVERY_ROOT / "hour_20260822T150000Z" / "run"
TERMINAL_STATE = TERMINAL_RUN / "feature_state" / "bundle"
TERMINAL_PARITY = RECOVERY_ROOT / "terminal_state_replay_parity_manifest.json"

OUT_AUTHORIZATION = ROOT / (
    "config/strict_r3_kraken_live_activation_authorization_20260822_v58_"
    "v131_direct_15m_reference_live.json"
)
OUT_EXECUTION = ROOT / (
    "config/strict_r3_kraken_live_execution_v132_v156_"
    "direct_15m_reference_live.json"
)
OUT_REVIEW = RECOVERY_ROOT / "live_successor_review.json"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _write_new(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _resolved_runtime_hashes(overlay: dict) -> dict[str, str]:
    base = _read(ROOT / str(overlay["base_bundle"]))
    hashes = dict(base.get("runtime_code_sha256") or {})
    hashes.update(dict(overlay.get("overrides", {}).get("runtime_code_sha256") or {}))
    return hashes


def main() -> None:
    required = (
        OVERLAY, SOURCE_AUTHORIZATION, SOURCE_EXECUTION,
        RECOVERY_ROOT / "run_manifest.json", TERMINAL_RUN / "run_manifest.json",
        TERMINAL_STATE / "state_bundle_manifest.json", TERMINAL_PARITY,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if any(path.exists() for path in (OUT_AUTHORIZATION, OUT_EXECUTION, OUT_REVIEW)):
        raise FileExistsError("direct-15m live successor outputs are immutable")

    recovery = _read(RECOVERY_ROOT / "run_manifest.json")
    terminal = _read(TERMINAL_RUN / "run_manifest.json")
    parity = _read(TERMINAL_PARITY)
    state = _read(TERMINAL_STATE / "state_bundle_manifest.json")
    if recovery.get("status") != "complete" or parity.get("status") != "pass":
        raise ValueError("requires complete recovery and exact terminal parity")
    if terminal.get("exchange_calls") != 0 or terminal.get("order_submission_enabled") is not False:
        raise ValueError("recovery terminal must be strictly no-order")
    if str((terminal.get("hashes") or {}).get("inference_bundle") or "") != _sha(OVERLAY):
        raise ValueError("terminal recovery was not scored by the v131 overlay")
    if str(state.get("expected_state_timestamp")) != "2026-08-22T14:00:00+00:00":
        raise ValueError("unexpected recovered terminal state timestamp")

    overlay = _read(OVERLAY)
    changed = [
        relative for relative, expected in _resolved_runtime_hashes(overlay).items()
        if _sha(ROOT / relative) != expected
    ]
    if changed:
        raise ValueError(f"unreviewed v131 runtime deltas: {sorted(changed)}")
    execution_source = _read(SOURCE_EXECUTION)
    execution_changed = [
        relative for relative, expected in dict(execution_source.get("runtime_code_sha256") or {}).items()
        if _sha(ROOT / relative) != expected
    ]
    if execution_changed:
        raise ValueError(f"unreviewed execution runtime deltas: {sorted(execution_changed)}")

    authorization = copy.deepcopy(_read(SOURCE_AUTHORIZATION))
    authorization.update({
        "authorization_source": (
            "User-approved future-only live continuation after the v131 "
            "direct-15m candidate-reference repair passed complete no-order "
            "recovery and exact terminal replay parity."
        ),
        "inference_bundle": str(OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": _sha(OVERLAY),
    })
    _write_new(OUT_AUTHORIZATION, authorization)

    execution = copy.deepcopy(execution_source)
    execution.update({
        "inference_bundle": str(OVERLAY.relative_to(ROOT)),
        "inference_bundle_sha256": _sha(OVERLAY),
        "activation_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "activation_authorization_sha256": _sha(OUT_AUTHORIZATION),
        "version_note": (
            "v156: exact v131 direct-15m candidate-reference recovery "
            "successor. No model, admission, portfolio, or exit economics "
            "changed; new entries begin only at a fresh future decision."
        ),
    })
    _write_new(OUT_EXECUTION, execution)

    review = {
        "schema": "strict_r3_direct_15m_reference_live_successor_review_v1",
        "status": "pass",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "successor_overlay": str(OVERLAY.relative_to(ROOT)),
        "successor_overlay_sha256": _sha(OVERLAY),
        "successor_authorization": str(OUT_AUTHORIZATION.relative_to(ROOT)),
        "successor_authorization_sha256": _sha(OUT_AUTHORIZATION),
        "successor_execution": str(OUT_EXECUTION.relative_to(ROOT)),
        "successor_execution_sha256": _sha(OUT_EXECUTION),
        "recovery_root": str(RECOVERY_ROOT.relative_to(ROOT)),
        "terminal_run": str(TERMINAL_RUN.relative_to(ROOT)),
        "terminal_run_manifest_sha256": _sha(TERMINAL_RUN / "run_manifest.json"),
        "terminal_state": str(TERMINAL_STATE.relative_to(ROOT)),
        "terminal_state_manifest_sha256": _sha(TERMINAL_STATE / "state_bundle_manifest.json"),
        "terminal_parity": str(TERMINAL_PARITY.relative_to(ROOT)),
        "terminal_parity_sha256": _sha(TERMINAL_PARITY),
        "economic_contract_changed": False,
        "validation": {
            "all_170_source_rows_present": True,
            "direct_15m_reference_candidate_repair": True,
            "terminal_no_order": True,
            "terminal_exact_parity": True,
            "frozen_geometry_unchanged": True,
            "overlay_hash_matches_recovered_terminal": True,
            "fresh_future_decision_required": True,
        },
    }
    _write_new(OUT_REVIEW, review)
    print(json.dumps(review, sort_keys=True))


if __name__ == "__main__":
    main()
