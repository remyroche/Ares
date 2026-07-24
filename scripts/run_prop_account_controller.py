#!/usr/bin/env python3
"""Evaluate current inference outputs through the prop-account risk overlay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.data_fetcher import make_exchange
from extreme_price_movements.inference.liquidity_precheck import (
    evaluate_orderbook_liquidity,
    fetch_ticker_snapshot,
)
from extreme_price_movements.inference.portfolio_policy import PortfolioPolicyConfig
from extreme_price_movements.inference.prop_account_controller import (
    AccountSnapshot,
    ControllerState,
    L2Capacity,
    MarkedPosition,
    PropAccountController,
    PropAccountPolicy,
    load_whitelist,
)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--account-snapshot", type=Path, required=True)
    p.add_argument(
        "--policy", type=Path, default=Path("config/prop_account_policy_v1.json")
    )
    p.add_argument(
        "--whitelist", type=Path, default=Path("config/prop_account_whitelist.json")
    )
    p.add_argument(
        "--state",
        type=Path,
        default=Path(
            "data_perp/exchanges/krakenfutures/live_state/prop_account_controller_state.json"
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data_perp/exchanges/krakenfutures/live_state/prop_account_actions.jsonl"
        ),
    )
    p.add_argument(
        "--live-l2",
        action="store_true",
        help="Fetch Kraken Futures ticker/book; this controller never places orders",
    )
    p.add_argument("--max-candidates", type=int, default=10)
    return p.parse_args()


def _l2(
    exchange: Any, row: dict[str, Any], requested: float, policy: PropAccountPolicy
) -> L2Capacity:
    portfolio = PortfolioPolicyConfig(
        max_orderbook_slippage_bps=policy.max_l2_slippage_bps,
        min_liquidity_capacity_weight=policy.min_l2_capacity_weight,
    )
    ticker = fetch_ticker_snapshot(
        exchange=exchange,
        symbol=row["symbol"],
        side=row["side"],
        policy=portfolio,
        mode="live",
    )
    snap = evaluate_orderbook_liquidity(
        exchange=exchange,
        symbol=row["symbol"],
        side=row["side"],
        intended_quote_size=requested,
        ticker_snapshot=ticker,
        policy=portfolio,
        mode="live",
    )
    return L2Capacity(
        capacity_quote=float(snap.orderbook_capacity_quote_within_slippage or 0.0),
        capacity_weight=float(snap.liquidity_capacity_weight),
        expected_slippage_bps=snap.expected_fill_slippage_bps,
        reject_reason=snap.reject_reason if snap.hard_reject else None,
    )


def main() -> int:
    args = _args()
    policy = PropAccountPolicy.from_json(args.policy)
    account = AccountSnapshot.from_mapping(
        json.loads(args.account_snapshot.read_text())
    )
    state = ControllerState.load(args.state, account)
    controller = PropAccountController(policy, load_whitelist(args.whitelist))
    account_decision = controller.account_action(account, state)
    decisions = [account_decision.to_dict()]
    if account_decision.action == "flatten":
        decisions.extend(
            {
                "action": "flatten_position",
                "reason": account_decision.reason,
                "symbol": p.symbol,
                "side": p.side,
                "marked_notional": p.marked_notional,
            }
            for p in account.positions
        )
    elif account_decision.action == "allow":
        ledger = pd.read_parquet(args.ledger)
        ledger_ts = pd.to_datetime(ledger["timestamp"], utc=True)
        cutoff = account.timestamp - pd.Timedelta(minutes=policy.max_signal_age_minutes)
        batch = ledger.loc[
            ledger_ts.between(cutoff, account.timestamp, inclusive="both")
        ].copy()
        if "portfolio_decision" in batch:
            batch = batch.loc[
                batch["portfolio_decision"].isin(("traded", "order_rejected"))
            ]
        elif "passed_rank_gate" in batch:
            batch = batch.loc[batch["passed_rank_gate"].fillna(False)]
        rank_col = "threshold_basis_corrected_expected_ev_rank"
        batch = batch.sort_values(
            ["timestamp", rank_col], ascending=[True, False]
        ).head(args.max_candidates)
        exchange = make_exchange("perps") if args.live_l2 else None
        rows = batch.to_dict("records")
        risk_shares = controller.opportunity_risk_shares(rows)
        for row, risk_share in zip(rows, risk_shares):
            row["portfolio_risk_budget_share"] = risk_share
        # Evaluate larger diversified allocations first. The share calculation
        # itself is independent of arrival order and may deliberately leave
        # budget unused when a batch lacks side/archetype diversity.
        rows.sort(
            key=lambda row: (
                float(row.get("portfolio_risk_budget_share", 0.0)),
                float(row.get(rank_col, 0.0)),
            ),
            reverse=True,
        )
        working_account = account
        for row in rows:
            requested = controller.requested_notional(row, working_account)
            l2 = _l2(exchange, row, requested, policy) if exchange is not None else None
            decision = controller.evaluate_entry(row, working_account, state, l2)
            decisions.append(decision.to_dict())
            if decision.action == "enter":
                price = controller._first(
                    row, ("expected_entry_price", "signal_price", "policy_entry_price")
                )
                stop = controller._first(row, ("policy_stop_price", "stop_price"))
                working_account = AccountSnapshot(
                    timestamp=working_account.timestamp,
                    equity=working_account.equity,
                    positions=working_account.positions
                    + (
                        MarkedPosition(
                            symbol=str(row["symbol"]),
                            side=str(row["side"]),
                            marked_notional=decision.approved_notional,
                            mark_price=price,
                            stop_price=stop,
                            policy_archetype=str(
                                row.get("policy_archetype")
                                or row.get("archetype_policy_key")
                                or ""
                            ),
                            leverage=controller._first(
                                row,
                                (
                                    "requested_entry_leverage",
                                    "configured_entry_leverage",
                                ),
                                1.0,
                            ),
                        ),
                    ),
                    day_start_equity=working_account.day_start_equity,
                    high_water_equity=working_account.high_water_equity,
                    starting_equity=working_account.starting_equity,
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as fh:
        for decision in decisions:
            fh.write(json.dumps(decision, sort_keys=True) + "\n")
    state.save(args.state)
    print(
        json.dumps(
            {
                "decisions": decisions,
                "state": str(args.state),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
