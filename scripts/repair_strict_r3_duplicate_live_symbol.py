#!/usr/bin/env python3
"""Repair one verified accidental same-symbol live entry.

This is an emergency recovery tool, not a trading path.  It will only act when
the canonical ledger contains exactly two long rows for one symbol and Kraken's
net position exactly equals their combined contract amounts.  It reduces the
explicitly named accidental row, verifies the residual exchange quantity, then
removes only that row from the ledger.  All ordinary entries remain forbidden
until the normal, hash-sealed producer is restarted.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.data_fetcher import make_exchange
from extreme_price_movements.inference.strict_r3_live_execution import (
    STATE_SCHEMA,
    _cancel_stop,
    _create_reduce_market_order,
    _fetch_exchange_positions,
    atomic_json,
    live_state_lock,
    StrictR3ExecutionContract,
)


def duplicate_symbol_repair_plan(
    *,
    state: dict[str, Any],
    symbol: str,
    retain_candidate_id: str,
    retire_candidate_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the exact two-row state and return retained/retired records."""
    if state.get("schema") != STATE_SCHEMA:
        raise ValueError("unexpected strict-R3 live-state schema")
    rows = [
        copy.deepcopy(row)
        for row in list(state.get("positions") or [])
        if str(row.get("symbol")) == str(symbol)
    ]
    if len(rows) != 2:
        raise ValueError("duplicate repair requires exactly two same-symbol rows")
    by_id = {str(row.get("candidate_id")): row for row in rows}
    if len(by_id) != 2:
        raise ValueError("duplicate repair rows need two distinct candidate IDs")
    if set(by_id) != {str(retain_candidate_id), str(retire_candidate_id)}:
        raise ValueError("duplicate repair candidate IDs do not match state")
    retained = by_id[str(retain_candidate_id)]
    retired = by_id[str(retire_candidate_id)]
    for row in (retained, retired):
        if str(row.get("side")).lower() != "long":
            raise ValueError("strict-R3 duplicate repair is long-only")
        amount = float(row.get("amount") or 0.0)
        if not math.isfinite(amount) or amount <= 0.0:
            raise ValueError("duplicate repair requires positive contract amounts")
        if str(row.get("exchange_symbol")) != str(symbol):
            raise ValueError("ledger symbol and exchange symbol differ")
    return retained, retired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-bundle", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--retain-candidate-id", required=True)
    parser.add_argument("--retire-candidate-id", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.state.is_file():
        raise ValueError("canonical live state is absent")
    contract = StrictR3ExecutionContract.load(args.execution_bundle, root=ROOT)
    if not contract.order_submission_authorized:
        raise ValueError("duplicate-symbol repair requires an authorized live contract")

    with live_state_lock(args.state):
        state = json.loads(args.state.read_text())
        retained, retired = duplicate_symbol_repair_plan(
            state=state,
            symbol=args.symbol,
            retain_candidate_id=args.retain_candidate_id,
            retire_candidate_id=args.retire_candidate_id,
        )
        planned = {
            "symbol": str(args.symbol),
            "retain_candidate_id": str(args.retain_candidate_id),
            "retire_candidate_id": str(args.retire_candidate_id),
            "retain_amount": float(retained["amount"]),
            "retire_amount": float(retired["amount"]),
            "retire_stop_order_id": retired.get("stop_order_id"),
        }
        if not args.apply:
            print(json.dumps({"mode": "dry_run", **planned}, sort_keys=True))
            return

        exchange = make_exchange("perps")
        positions = _fetch_exchange_positions(exchange)
        exchange_position = positions.get(str(args.symbol))
        if exchange_position is None:
            raise ValueError("exchange lacks the duplicate-symbol position")
        exchange_amount = float(exchange_position.get("contracts") or 0.0)
        expected = float(retained["amount"]) + float(retired["amount"])
        if not math.isclose(exchange_amount, expected, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"exchange amount {exchange_amount} does not equal duplicate ledger sum {expected}"
            )

        # Both native stops cover the aggregate until the explicit reduction is
        # filled.  Immediately cancel only the retired row's stop afterwards;
        # the retained row's stop continuously protects the intended residual.
        order = _create_reduce_market_order(
            exchange,
            symbol=str(args.symbol),
            side="sell",
            amount=float(retired["amount"]),
            config={"execution_account": "perps", "exchange": "krakenfutures"},
        )
        _cancel_stop(
            exchange,
            symbol=str(args.symbol),
            order_id=retired.get("stop_order_id"),
        )
        refreshed = _fetch_exchange_positions(exchange).get(str(args.symbol))
        if refreshed is None:
            raise ValueError("repair reduction unexpectedly flattened retained position")
        residual = float(refreshed.get("contracts") or 0.0)
        if not math.isclose(residual, float(retained["amount"]), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"repair residual {residual} does not equal retained amount {retained['amount']}"
            )
        actual_entry = float(refreshed.get("entryPrice") or 0.0)
        if not math.isfinite(actual_entry) or actual_entry <= 0.0:
            raise ValueError("repair exchange position lacks a positive entry price")
        retained["entry_price"] = actual_entry
        retained["gross_notional"] = actual_entry * residual * float(retained["contract_size"])
        retained["duplicate_symbol_repair"] = {
            "schema": "strict_r3_duplicate_symbol_repair_v1",
            "repaired_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "retired_candidate_id": str(args.retire_candidate_id),
            "retired_amount": float(retired["amount"]),
            "retired_stop_order_id": retired.get("stop_order_id"),
            "reduce_order_id": order.get("id"),
            "post_repair_exchange_entry_price": actual_entry,
            "post_repair_exchange_contracts": residual,
        }
        state["positions"] = [
            retained if str(row.get("candidate_id")) == str(args.retain_candidate_id)
            else row
            for row in state["positions"]
            if str(row.get("candidate_id")) != str(args.retire_candidate_id)
        ]
        state["as_of_ts"] = pd.Timestamp.now(tz="UTC").isoformat()
        atomic_json(args.state, state)
        print(json.dumps({"mode": "applied", **planned, "order_id": order.get("id"), "residual": residual}, sort_keys=True))


if __name__ == "__main__":
    main()
