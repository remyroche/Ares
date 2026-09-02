#!/usr/bin/env python3
"""Advance saved P8U regular feature state, one source hour at a time.

Research/preproduction only. This process accepts target-free primitive state,
emits target-free regular snapshots, and cannot score models, read outcomes,
call policy/portfolio code, communicate with an exchange, or submit orders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_regular_state_forward import (  # noqa: E402
    SCHEMA,
    forward_regular_state,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bootstrap-state-root", type=Path, required=True)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--state-scope", required=True)
    parser.add_argument("--state-components", nargs="+", required=True)
    parser.add_argument("--through", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    if not args.source_state.is_file():
        raise FileNotFoundError(args.source_state)
    source = joblib.load(args.source_state)
    if not isinstance(source, Mapping):
        raise ValueError("regular-state source must be a mapping")
    result = forward_regular_state(
        bundle=args.bundle,
        bootstrap_state_root=args.bootstrap_state_root,
        output_root=args.out_root,
        source=source,
        state_scope=args.state_scope,
        state_components=tuple(map(str, args.state_components)),
        through=args.through,
    )
    receipt = {
        "schema": SCHEMA,
        "status": "pass_target_free_regular_state_forward",
        "bootstrap_state_root": str(args.bootstrap_state_root.resolve()),
        "source_state": str(args.source_state.resolve()),
        "source_state_sha256": _sha256(args.source_state),
        "source_start": result.source_start.isoformat(),
        "source_end": result.source_end.isoformat(),
        "committed_hours": len(result.committed_timestamps),
        "first_committed_timestamp": (
            None if not result.committed_timestamps else result.committed_timestamps[0].isoformat()
        ),
        "last_committed_timestamp": (
            None if not result.committed_timestamps else result.committed_timestamps[-1].isoformat()
        ),
        "regular_feature_count": result.regular_feature_count,
        "state_contract_feature_count": result.state_contract_feature_count,
        "state_contract_source": result.state_contract_source,
        "state_contract_sha256": result.state_contract_sha256,
        "clone_mode": result.clone_mode,
        "source_rows_fed_per_commit": 1,
        "broad_retained_tail_feature_graph_called": False,
        "outcome_columns_consumed": [],
        "policy_or_portfolio_called": False,
        "exchange_or_order_submission_called": False,
    }
    _atomic_json(result.output_root / "forward_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
