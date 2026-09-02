#!/usr/bin/env python3
"""Create an immutable state successor for the v141 leverage contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.strict_r3_live_execution import (
    StrictR3ExecutionContract,
    atomic_json,
    live_state_lock,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-execution", type=Path, required=True)
    parser.add_argument("--successor-execution", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--migration-kind",
        default="approved_inverse_policy_stop_absolute_pct_leverage_v1",
    )
    parser.add_argument(
        "--new-entry-formula",
        default="min(10, 66 / policy_stop_absolute_pct)",
    )
    args = parser.parse_args()
    if args.destination.exists() or args.receipt.exists():
        raise FileExistsError("state successor or receipt already exists")
    with live_state_lock(args.source):
        payload = json.loads(args.source.read_text())
        source_hash = sha(args.source)
    old_execution = json.loads(args.source_execution.read_text())
    new_contract = StrictR3ExecutionContract.load(
        args.successor_execution, root=ROOT,
    )
    new_execution = json.loads(args.successor_execution.read_text())
    allowed_execution_delta = {
        "inference_bundle", "inference_bundle_sha256",
        "activation_authorization", "activation_authorization_sha256",
        "runtime_code_sha256", "runtime_reseal_predecessors", "version_note",
        "leverage", "leverage_sizing",
    }
    old_static = {
        key: value for key, value in old_execution.items()
        if key not in allowed_execution_delta
    }
    new_static = {
        key: value for key, value in new_execution.items()
        if key not in allowed_execution_delta
    }
    if old_static != new_static:
        raise ValueError("dynamic leverage migration changed another execution rule")
    if (
        str(payload.get("exit_policy_sha256"))
        != str(new_contract.exit_policy_sha256)
    ):
        raise ValueError("dynamic leverage migration would change the exit policy")
    positions = payload.get("positions")
    if not isinstance(positions, list):
        raise ValueError("source state has no position list")
    output = dict(payload)
    output.update({
        "inference_bundle_sha256": new_contract.inference_bundle_sha256,
        "exit_policy_sha256": new_contract.exit_policy_sha256,
        "activation_authorization_sha256": new_contract.activation_authorization_sha256,
        "contract_migration": {
            "kind": str(args.migration_kind),
            "previous_inference_bundle_sha256": old_execution["inference_bundle_sha256"],
            "new_inference_bundle_sha256": new_contract.inference_bundle_sha256,
            "previous_leverage": old_execution["leverage"],
            "new_leverage_formula": str(args.new_entry_formula),
            "positions_preserved_exact": True,
            "processed_decisions_preserved_exact": True,
        },
    })
    if output.get("positions") != positions:
        raise AssertionError("state migration changed an open position")
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.destination, output)
    receipt = {
        "schema": "strict_r3_dynamic_stop_leverage_state_migration_v1",
        "source_state": str(args.source),
        "source_state_sha256": source_hash,
        "source_execution": str(args.source_execution),
        "source_execution_sha256": sha(args.source_execution),
        "successor_execution": str(args.successor_execution),
        "successor_execution_sha256": sha(args.successor_execution),
        "successor_state": str(args.destination),
        "successor_state_sha256": sha(args.destination),
        "as_of_ts": payload.get("as_of_ts"),
        "position_count": len(positions),
        "position_identity": [
            {
                "candidate_id": row.get("candidate_id"),
                "symbol": row.get("symbol"),
                "effective_leverage": row.get("effective_leverage"),
            }
            for row in positions
        ],
        "migration_semantics": (
            "new-entry leverage successor only; existing positions retain their "
            "Kraken-confirmed fill leverage"
        ),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
