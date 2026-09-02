#!/usr/bin/env python3
"""Migrate a flat strict-R3 live state across a reviewed runtime reseal.

This is intentionally narrower than the general successor migrator.  It is
usable only when there are no open positions and the successor review declares
an unchanged economic contract.  It avoids importing the full research graph
solely to rewrite three hashes, while the live producer still validates the
complete sealed contract before it may submit an entry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--old-execution", type=Path, required=True)
    parser.add_argument("--new-execution", type=Path, required=True)
    parser.add_argument("--runtime-review", type=Path, required=True)
    parser.add_argument("--out-state", type=Path, required=True)
    parser.add_argument("--out-receipt", type=Path, required=True)
    args = parser.parse_args()

    if args.out_state.exists() or args.out_receipt.exists():
        raise FileExistsError("flat runtime-reseal outputs are immutable")
    state = _read(args.state)
    old_execution = _read(args.old_execution)
    new_execution = _read(args.new_execution)
    review = _read(args.runtime_review)
    positions = state.get("positions")
    if positions != []:
        raise ValueError("flat runtime reseal requires exactly zero open positions")
    if state.get("inference_bundle_sha256") != old_execution.get("inference_bundle_sha256"):
        raise ValueError("state does not match its declared old inference bundle")
    if state.get("exit_policy_sha256") != old_execution.get("exit_policy_sha256"):
        raise ValueError("state does not match its declared old exit policy")
    if state.get("activation_authorization_sha256") != old_execution.get(
        "activation_authorization_sha256"
    ):
        raise ValueError("state does not match its declared old authorization")
    if review.get("economic_contract_changed") is not False:
        raise ValueError("runtime review does not prove unchanged economics")
    if review.get("successor_execution_sha256") != _sha(args.new_execution):
        raise ValueError("runtime review does not bind the new execution contract")
    if review.get("successor_overlay_sha256") != new_execution.get(
        "inference_bundle_sha256"
    ):
        raise ValueError("runtime review does not bind the new inference bundle")
    if review.get("successor_authorization_sha256") != new_execution.get(
        "activation_authorization_sha256"
    ):
        raise ValueError("runtime review does not bind the new authorization")
    if old_execution.get("exit_policy_sha256") != new_execution.get("exit_policy_sha256"):
        raise ValueError("flat runtime reseal may not alter the exit policy")
    if old_execution.get("side") != new_execution.get("side"):
        raise ValueError("flat runtime reseal may not alter trading side")
    if new_execution.get("order_submission_authorized") is not True:
        raise ValueError("new execution contract is not live-authorized")

    output = json.loads(json.dumps(state))
    decisions_before = output.get("processed_decision_ids")
    output.update({
        "inference_bundle_sha256": new_execution["inference_bundle_sha256"],
        "exit_policy_sha256": new_execution["exit_policy_sha256"],
        "activation_authorization_sha256": new_execution[
            "activation_authorization_sha256"
        ],
        "contract_migration": {
            "kind": "strict_r3_flat_runtime_reseal_v1",
            "previous_inference_bundle_sha256": old_execution[
                "inference_bundle_sha256"
            ],
            "new_inference_bundle_sha256": new_execution[
                "inference_bundle_sha256"
            ],
            "positions_preserved_exact": True,
            "processed_decisions_preserved_exact": True,
            "runtime_review_sha256": _sha(args.runtime_review),
        },
    })
    if output.get("positions") != [] or output.get("processed_decision_ids") != decisions_before:
        raise AssertionError("flat runtime reseal attempted to mutate state history")
    _atomic_json(args.out_state, output)
    receipt = {
        "schema": "strict_r3_flat_runtime_reseal_migration_v1",
        "status": "pass",
        "source_state": str(args.state),
        "source_state_sha256": _sha(args.state),
        "output_state": str(args.out_state),
        "output_state_sha256": _sha(args.out_state),
        "old_execution": str(args.old_execution),
        "old_execution_sha256": _sha(args.old_execution),
        "new_execution": str(args.new_execution),
        "new_execution_sha256": _sha(args.new_execution),
        "runtime_review": str(args.runtime_review),
        "runtime_review_sha256": _sha(args.runtime_review),
        "position_count": 0,
        "positions_preserved_exact": True,
        "processed_decisions_preserved_exact": True,
    }
    _atomic_json(args.out_receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
