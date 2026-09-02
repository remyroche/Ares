#!/usr/bin/env python3
"""Initialize a clean, explicitly authorized strict-R3 Kraken live state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.data_fetcher import make_exchange
from extreme_price_movements.inference.strict_r3_live_execution import (
    StrictR3ExecutionContract,
    atomic_json,
    initial_state,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-bundle", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--activation-ts", required=True)
    args = parser.parse_args()
    if args.state.exists():
        raise FileExistsError(f"live state already exists: {args.state}")
    contract = StrictR3ExecutionContract.load(args.execution_bundle, root=ROOT)
    if not contract.order_submission_authorized:
        raise ValueError("execution contract has not been explicitly authorized")
    activation_ts = pd.Timestamp(args.activation_ts)
    activation_ts = (
        activation_ts.tz_localize("UTC")
        if activation_ts.tzinfo is None
        else activation_ts.tz_convert("UTC")
    )
    if contract.authorized_after is None or activation_ts < contract.authorized_after:
        raise ValueError("activation timestamp predates authorization")
    exchange = make_exchange("perps")
    if str(getattr(exchange, "id", "")) != "krakenfutures":
        raise ValueError("canonical live execution requires Kraken Futures")
    positions = [
        row for row in (exchange.fetch_positions() or [])
        if abs(float(row.get("contracts") or 0.0)) > 0.0
    ]
    if positions:
        raise ValueError("Kraken account is not flat; activation fails closed")
    state = initial_state(as_of_ts=activation_ts)
    state.update({
        "activation_ts": activation_ts.isoformat(),
        "inference_bundle_sha256": contract.inference_bundle_sha256,
        "exit_policy_sha256": contract.exit_policy_sha256,
        "activation_authorization_sha256": (
            contract.activation_authorization_sha256
        ),
    })
    atomic_json(args.state, state)
    print(json.dumps(state, default=str))


if __name__ == "__main__":
    main()
