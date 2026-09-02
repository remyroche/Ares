#!/usr/bin/env python3
"""Execute one verified canonical strict-R3 hour on Kraken Futures."""

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
    execute_verified_hour,
    live_state_lock,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-bundle", type=Path, required=True)
    parser.add_argument("--hourly-run", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--submit-orders", action="store_true")
    parser.add_argument("--live-hour-audit", type=Path, default=None)
    parser.add_argument("--current-replay-audit", type=Path, default=None)
    parser.add_argument("--exit-replay-audit", type=Path, default=None)
    parser.add_argument(
        "--runtime-checkpoint",
        type=Path,
        default=None,
        help=(
            "Immutable checkpoint directory binding feature, score, "
            "calibration, portfolio and exit state. Required by successor "
            "execution contracts before --submit-orders."
        ),
    )
    parser.add_argument("--now", default=None)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"immutable execution receipt exists: {args.out}")
    contract = StrictR3ExecutionContract.load(args.execution_bundle, root=ROOT)
    exchange = make_exchange("perps")
    if str(getattr(exchange, "id", "")) != "krakenfutures":
        raise ValueError("canonical live execution requires Kraken Futures")
    now = pd.Timestamp(args.now) if args.now else pd.Timestamp.now(tz="UTC")
    with live_state_lock(args.state):
        result = execute_verified_hour(
            exchange=exchange, contract=contract, hourly_run=args.hourly_run,
            state_path=args.state, now=now, submit_orders=args.submit_orders,
            live_hour_audit=args.live_hour_audit,
            current_replay_audit=args.current_replay_audit,
            exit_replay_audit=args.exit_replay_audit,
            runtime_checkpoint=args.runtime_checkpoint,
        )
    atomic_json(args.out, result)
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
