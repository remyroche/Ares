#!/usr/bin/env python3
"""Reconcile canonical strict-R3 state with confirmed Kraken protective fills.

This utility is deliberately exchange-read-only. It removes a locally tracked
position only when Kraken reports no open position and the exact persisted
protective order is present in closed order history with a complete opposite-
side fill. Untracked exchange positions, missing evidence, or ambiguous fills
fail closed. ``--apply`` changes only the local atomic state file and emits an
immutable receipt containing before/after hashes and fill evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.data_fetcher import make_exchange  # noqa: E402
from extreme_price_movements.inference.strict_r3_live_execution import (  # noqa: E402
    _fetch_exchange_positions,
    atomic_json,
    load_state,
    sha256_file,
)


SCHEMA = "strict_r3_kraken_live_state_reconciliation_v1"


def _confirmed_protective_fill(exchange: Any, position: Mapping[str, Any]) -> dict[str, Any]:
    symbol = str(position["exchange_symbol"])
    stop_order_id = str(position["stop_order_id"])
    entry_ts = pd.Timestamp(position["entry_ts"])
    if entry_ts.tzinfo is None:
        entry_ts = entry_ts.tz_localize("UTC")
    else:
        entry_ts = entry_ts.tz_convert("UTC")
    rows = exchange.fetch_closed_orders(
        symbol,
        since=max(0, int(entry_ts.timestamp() * 1000) - 60_000),
    )
    matches = [row for row in rows if str(row.get("id")) == stop_order_id]
    if len(matches) != 1:
        raise ValueError(
            f"missing or ambiguous confirmed protective fill for {symbol}: {stop_order_id}"
        )
    order = matches[0]
    amount = float(order.get("amount") or 0.0)
    filled = float(order.get("filled") or 0.0)
    expected = float(position["amount"])
    if str(order.get("status")).lower() != "closed":
        raise ValueError(f"protective order is not closed: {stop_order_id}")
    if str(order.get("side")).lower() != "sell":
        raise ValueError(f"protective fill is not a long-position sell: {stop_order_id}")
    if filled <= 0.0 or abs(filled - expected) > max(1e-9, 1e-6 * expected):
        raise ValueError(f"protective fill amount mismatch: {stop_order_id}")
    return {
        "candidate_id": str(position["candidate_id"]),
        "symbol": symbol,
        "stop_order_id": stop_order_id,
        "status": str(order.get("status")),
        "amount": amount,
        "filled": filled,
        "average": order.get("average"),
        "price": order.get("price"),
        "timestamp": order.get("timestamp"),
        "datetime": order.get("datetime"),
    }


def reconcile_state(*, exchange: Any, state_path: Path, apply: bool) -> dict[str, Any]:
    before_hash = sha256_file(state_path)
    state = load_state(state_path, decision_ts=pd.Timestamp.now(tz="UTC"))
    before = copy.deepcopy(state)
    local = {str(row["exchange_symbol"]): dict(row) for row in state["positions"]}
    exchange_positions = _fetch_exchange_positions(exchange)
    untracked = sorted(set(exchange_positions).difference(local))
    if untracked:
        raise ValueError(f"exchange positions absent from canonical state: {untracked}")
    removals = []
    retained = []
    for symbol, position in local.items():
        if symbol in exchange_positions:
            retained.append(position)
        else:
            removals.append(_confirmed_protective_fill(exchange, position))
    state["positions"] = retained
    state["last_exchange_reconciliation_ts"] = pd.Timestamp.now(tz="UTC").isoformat()
    if apply and removals:
        atomic_json(state_path, state)
    after_hash = sha256_file(state_path)
    if not apply and after_hash != before_hash:
        raise AssertionError("read-only reconciliation changed local state")
    if apply and not removals and state != before:
        # Do not churn the canonical file solely to refresh telemetry.
        after_hash = before_hash
    return {
        "schema": SCHEMA,
        "mode": "apply" if apply else "read_only",
        "state_path": str(state_path),
        "state_sha256_before": before_hash,
        "state_sha256_after": after_hash,
        "exchange_position_symbols": sorted(exchange_positions),
        "local_positions_before": len(local),
        "local_positions_after": len(retained),
        "confirmed_protective_fills": removals,
        "changed": bool(apply and removals),
        "exchange_write_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"immutable reconciliation receipt exists: {args.out}")
    exchange = make_exchange("perps")
    if str(getattr(exchange, "id", "")) != "krakenfutures":
        raise ValueError("canonical reconciliation requires Kraken Futures")
    result = reconcile_state(exchange=exchange, state_path=args.state, apply=args.apply)
    atomic_json(args.out, result)
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
