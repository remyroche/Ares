#!/usr/bin/env python3
"""Migrate a strict-R3 live ledger to a runtime-only successor contract.

The migration is deliberately narrow: model/artifact/policy/economic fields
must be identical.  Only the inference bundle identity, authorization identity,
and declared runtime hashes may change.  Positions and processed decisions are
preserved exactly in a new immutable state file.
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

from extreme_price_movements.inference.strict_r3_live_execution import (
    StrictR3ExecutionContract,
    atomic_json,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_inference(payload: dict) -> dict:
    out = copy.deepcopy(payload)
    out.pop("live_state_reconciliation_contract", None)
    out.get("runtime_code_sha256", {}).pop(
        "scripts/run_strict_r3_hourly_shadow.py", None,
    )
    return out


def _invariant_execution(payload: dict) -> dict:
    fields = (
        "schema", "exchange_id", "side", "exit_policy_sha256", "leverage",
        "maximum_decision_age_seconds", "maximum_exit_slippage_bps",
        "order_submission_authorized", "entry_order", "protective_stop",
        "trailing_contract", "adaptive_exit_role", "failure_policy",
    )
    return {field: payload.get(field) for field in fields}


def migrate_state(
    *, state: dict, old_execution: dict, new_execution: dict,
    old_inference: dict, new_inference: dict,
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
    if _invariant_execution(old_execution) != _invariant_execution(new_execution):
        raise ValueError("execution economics or policy changed during state migration")
    if _normalized_inference(old_inference) != _normalized_inference(new_inference):
        raise ValueError("inference artifacts or model semantics changed during state migration")

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
            "kind": "runtime_only_actual_fill_reconciliation",
            "previous_inference_bundle_sha256": old_execution[
                "inference_bundle_sha256"
            ],
            "new_inference_bundle_sha256": new_execution[
                "inference_bundle_sha256"
            ],
        },
    })
    if output.get("positions") != positions_before:
        raise AssertionError("state migration changed positions")
    if output.get("processed_decision_ids") != decisions_before:
        raise AssertionError("state migration changed processed decisions")
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
        raise FileExistsError("live-state migration outputs are immutable")

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
    output = migrate_state(
        state=state,
        old_execution=old_execution,
        new_execution=new_execution,
        old_inference=json.loads(old_inference_path.read_text()),
        new_inference=json.loads(new_inference_path.read_text()),
    )
    atomic_json(args.out_state, output)
    receipt = {
        "schema": "strict_r3_live_state_contract_migration_v1",
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
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
