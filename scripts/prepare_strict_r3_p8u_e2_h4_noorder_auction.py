#!/usr/bin/env python3
"""Build an immutable no-order E2-before-auction portfolio proposal.

This is the successor parity stage only.  It reads the target-free E2
selection, applies the unchanged fixed-slot constraints against a supplied
reconciled state snapshot, and writes a proposal.  It cannot connect to an
exchange or submit an order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_e2_h4_auction import (
    E2_AUCTION_SCHEMA,
    E2AuctionSpec,
    apply_e2_before_auction,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2-selection", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection_root = args.e2_selection.resolve()
    selection_path = selection_root / "e2_candidate_selection_target_free.parquet"
    receipt_path = selection_root / "receipt.json"
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("E2 no-order auction output must be immutable")
    selection_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if selection_receipt.get("status") != "pass_target_free_no_order_e2_selection":
        raise ValueError("E2 selection receipt is not target-free and successful")
    if selection_receipt.get("order_submission") is not False:
        raise ValueError("E2 selection receipt has unexpected order authority")
    state = json.loads(args.state.resolve().read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError("E2 auction state must be a JSON object")
    wallet = float(state["wallet_equity_quote"])
    auction = apply_e2_before_auction(
        pd.read_parquet(selection_path), state=state, wallet_equity_quote=wallet, spec=E2AuctionSpec()
    )
    output.mkdir(parents=True, exist_ok=False)
    auction.to_parquet(output / "e2_auction_target_free.parquet", index=False, compression="zstd")
    audit = {
        "schema": E2_AUCTION_SCHEMA,
        "status": "prepared_no_order_e2_before_auction",
        "order_submission": False,
        "selection_root": str(selection_root),
        "selection_receipt_sha256": _sha256(receipt_path),
        "selection_table_sha256": _sha256(selection_path),
        "bundle_manifest_sha256": selection_receipt["bundle_manifest_sha256"],
        "state_path": str(args.state.resolve()),
        "state_sha256": _sha256(args.state.resolve()),
        "candidate_rows": int(len(auction)),
        "e2_selected_rows": int(auction["e2_entry_selected"].sum()),
        "auction_proposed_rows": int(auction["execution_action"].eq("propose").sum()),
        "priority": "bcf_mc1_expected_bps",
        "portfolio_constraints": {
            "max_concurrent_positions": 8,
            "max_new_entries_per_decision": 2,
            "margin_budget_fraction": 0.80,
            "margin_slot_fraction": 0.10,
            "leverage": 7.0,
        },
        "outcome_columns_consumed": [],
        "required_pre_submit_checks": [
            "current account reconciliation",
            "fresh executable-book and fill-impact validation",
            "entry freshness and price-gap guard",
            "separately authorized exchange gateway",
        ],
    }
    (output / "receipt.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
