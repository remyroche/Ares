#!/usr/bin/env python3
"""Create an append-only strict-R3 live-state contract migration.

This is deliberately narrow: it never changes positions, processed decision
identities, policy state, or score lineage.  It only binds an existing state
to a newly sealed execution authorization after a reviewed runtime/policy
successor is introduced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "strict_r3_kraken_live_state_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--inference-bundle-sha256", required=True)
    parser.add_argument("--exit-policy-sha256", required=True)
    parser.add_argument("--activation-authorization-sha256", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(destination)
    payload: dict[str, Any] = json.loads(source.read_text())
    if payload.get("schema") != SCHEMA:
        raise ValueError("source has an unknown strict-R3 live-state schema")
    if not isinstance(payload.get("positions"), list):
        raise ValueError("source live state has invalid positions")
    if not isinstance(payload.get("processed_decision_ids"), list):
        raise ValueError("source live state has invalid processed decisions")

    prior_positions = json.dumps(payload["positions"], sort_keys=True, separators=(",", ":"))
    prior_processed = json.dumps(
        payload["processed_decision_ids"], sort_keys=True, separators=(",", ":")
    )
    payload["inference_bundle_sha256"] = str(args.inference_bundle_sha256)
    payload["exit_policy_sha256"] = str(args.exit_policy_sha256)
    payload["activation_authorization_sha256"] = str(
        args.activation_authorization_sha256
    )
    payload["contract_migration"] = {
        "schema": "strict_r3_live_state_contract_migration_v1",
        "prior_state": str(source),
        "prior_state_sha256": sha256(source),
        "inference_bundle_sha256": str(args.inference_bundle_sha256),
        "activation_authorization_sha256": str(args.activation_authorization_sha256),
        "positions_preserved_exact": prior_positions
        == json.dumps(payload["positions"], sort_keys=True, separators=(",", ":")),
        "processed_decisions_preserved_exact": prior_processed
        == json.dumps(
            payload["processed_decision_ids"], sort_keys=True, separators=(",", ":")
        ),
        "reason": str(args.reason),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    temporary.replace(destination)


if __name__ == "__main__":
    main()
