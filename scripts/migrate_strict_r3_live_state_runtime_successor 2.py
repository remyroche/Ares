#!/usr/bin/env python3
"""Rebind a non-flat strict-R3 live ledger to a reviewed runtime successor.

This migration is narrower than a model migration. Frozen artifacts, feature
contract, model/calibration/geometry inputs, admission, policy, economics and
portfolio semantics must remain identical. The only permitted inference
change is runtime implementation/state lineage, including the reviewed causal
no-trade 15-minute representation. Actual positions and processed decisions
are copied byte-for-byte as JSON values into a new immutable state file.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.strict_r3_live_execution import (  # noqa: E402
    StrictR3ExecutionContract,
    atomic_json,
)


MIGRATION_KIND = "causal_no_trade_15m_runtime_successor_v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _execution_economics(payload: dict) -> dict:
    excluded = {
        "inference_bundle", "inference_bundle_sha256",
        "activation_authorization", "activation_authorization_sha256",
        "runtime_code_sha256", "live_shadow_bridge_contract",
        "exit_replay_contract", "runtime_checkpoint_required_before_order_submission",
    }
    return {key: value for key, value in payload.items() if key not in excluded}


def _inference_semantics(payload: dict) -> dict:
    runtime = dict(payload.get("runtime") or {})
    runtime_fields = (
        "adaptive_exit", "admission", "base_route",
        "candidate_feature_population", "candidate_materializer",
        "current_spread_gate", "entry_open_contract", "entry_price_lineage",
        "feature_edge_contract", "feature_history_start", "late_source_policy",
        "oi_refresh_contract", "policy_bar_root", "resolved_calibration_update",
        "score_chunk_hours", "shadow_cycle",
    )
    return {
        "schema": payload.get("schema"),
        "scope": payload.get("scope"),
        "activation_ts": payload.get("activation_ts"),
        "end_exclusive_ts": payload.get("end_exclusive_ts"),
        "outside_window": payload.get("outside_window"),
        "live_decision_freshness_seconds": payload.get(
            "live_decision_freshness_seconds"
        ),
        "missing_entry_data_contract": payload.get("missing_entry_data_contract"),
        "reference_window_days": payload.get("reference_window_days"),
        "ev_bridge_role": payload.get("ev_bridge_role"),
        "admission_contract": payload.get("admission_contract"),
        "trust_overlay_contract": payload.get("trust_overlay_contract"),
        "resolved_outcome_contract": payload.get("resolved_outcome_contract"),
        "adaptive_exit_contract": payload.get("adaptive_exit_contract"),
        "adaptive_exit_role": payload.get("adaptive_exit_role"),
        "paths": payload.get("paths"),
        "sha256": payload.get("sha256"),
        "producer": payload.get("producer"),
        "feature_parity": payload.get("feature_parity"),
        "runtime_semantics": {key: runtime.get(key) for key in runtime_fields},
    }


def migrate_runtime_successor_state(
    *,
    state: dict,
    old_execution: dict,
    new_execution: dict,
    old_inference: dict,
    new_inference: dict,
) -> dict:
    expected_old = {
        "inference_bundle_sha256": old_execution["inference_bundle_sha256"],
        "exit_policy_sha256": old_execution["exit_policy_sha256"],
        "activation_authorization_sha256": old_execution[
            "activation_authorization_sha256"
        ],
    }
    for field, expected in expected_old.items():
        if state.get(field) != expected:
            raise ValueError(f"source live state does not match old contract: {field}")
    if bool(old_execution.get(
        "runtime_checkpoint_required_before_order_submission", False
    )) and not bool(new_execution.get(
        "runtime_checkpoint_required_before_order_submission", False
    )):
        raise ValueError("successor may not weaken the runtime-checkpoint gate")
    if not bool(new_execution.get(
        "runtime_checkpoint_required_before_order_submission", False
    )):
        raise ValueError("successor requires a pre-order runtime checkpoint")
    if _execution_economics(old_execution) != _execution_economics(new_execution):
        raise ValueError("execution economics or policy changed during successor migration")
    if _inference_semantics(old_inference) != _inference_semantics(new_inference):
        raise ValueError("frozen inference/model/admission semantics changed")

    output = copy.deepcopy(state)
    positions_before = copy.deepcopy(output.get("positions") or [])
    decisions_before = copy.deepcopy(output.get("processed_decision_ids") or [])
    output.update({
        "inference_bundle_sha256": new_execution["inference_bundle_sha256"],
        "exit_policy_sha256": new_execution["exit_policy_sha256"],
        "activation_authorization_sha256": new_execution[
            "activation_authorization_sha256"
        ],
        "contract_migration": {
            "kind": MIGRATION_KIND,
            "previous_inference_bundle_sha256": old_execution[
                "inference_bundle_sha256"
            ],
            "new_inference_bundle_sha256": new_execution[
                "inference_bundle_sha256"
            ],
            "positions_preserved_exact": True,
            "processed_decisions_preserved_exact": True,
        },
    })
    if output.get("positions") != positions_before:
        raise AssertionError("successor migration changed live positions")
    if output.get("processed_decision_ids") != decisions_before:
        raise AssertionError("successor migration changed processed decisions")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--old-execution", type=Path, required=True)
    parser.add_argument("--new-execution", type=Path, required=True)
    parser.add_argument("--out-state", type=Path, required=True)
    parser.add_argument("--out-receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.out_state.exists() or args.out_receipt.exists():
        raise FileExistsError("successor migration outputs are immutable")

    state = json.loads(args.state.read_text())
    old_execution = json.loads(args.old_execution.read_text())
    new_execution = json.loads(args.new_execution.read_text())
    old_inference_path = ROOT / old_execution["inference_bundle"]
    new_inference_path = ROOT / new_execution["inference_bundle"]
    if _sha(old_inference_path) != old_execution["inference_bundle_sha256"]:
        raise ValueError("old inference bundle file hash mismatch")
    new_contract = StrictR3ExecutionContract.load(args.new_execution, root=ROOT)
    if not new_contract.order_submission_authorized:
        raise ValueError("successor execution contract is not live-authorized")
    output = migrate_runtime_successor_state(
        state=state,
        old_execution=old_execution,
        new_execution=new_execution,
        old_inference=json.loads(old_inference_path.read_text()),
        new_inference=json.loads(new_inference_path.read_text()),
    )
    atomic_json(args.out_state, output)
    receipt = {
        "schema": "strict_r3_live_state_runtime_successor_migration_v1",
        "migration_kind": MIGRATION_KIND,
        "source_state": str(args.state),
        "source_state_sha256": _sha(args.state),
        "output_state": str(args.out_state),
        "output_state_sha256": _sha(args.out_state),
        "old_execution": str(args.old_execution),
        "old_execution_sha256": _sha(args.old_execution),
        "new_execution": str(args.new_execution),
        "new_execution_sha256": _sha(args.new_execution),
        "positions_preserved_exact": output.get("positions") == state.get("positions"),
        "processed_decisions_preserved_exact": (
            output.get("processed_decision_ids")
            == state.get("processed_decision_ids")
        ),
        "position_count": len(output.get("positions") or []),
    }
    atomic_json(args.out_receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
