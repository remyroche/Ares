#!/usr/bin/env python3
"""Poll canonical strict-R3 Kraken positions once; never open new trades."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.data_fetcher import make_exchange
from extreme_price_movements.inference.strict_r3_live_execution import (
    StrictR3ExecutionContract,
    atomic_json,
    monitor_live_positions_once,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-bundle", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--submit-orders", action="store_true")
    parser.add_argument("--now", default=None)
    parser.add_argument(
        "--interval-seconds", type=float, default=None,
        help="Keep one initialized process alive and poll at this cadence.",
    )
    args = parser.parse_args()
    contract = StrictR3ExecutionContract.load(args.execution_bundle, root=ROOT)
    exchange = make_exchange("perps")
    if str(getattr(exchange, "id", "")) != "krakenfutures":
        raise ValueError("canonical minute monitor requires Kraken Futures")
    interval = None if args.interval_seconds is None else max(1.0, float(args.interval_seconds))
    next_run = time.monotonic()
    while True:
        now = pd.Timestamp(args.now) if args.now else pd.Timestamp.now(tz="UTC")
        now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
        run_dir = args.out_root / f"monitor_{now.strftime('%Y%m%dT%H%M%S%fZ')}"
        if run_dir.exists():
            raise FileExistsError(f"immutable minute-monitor receipt exists: {run_dir}")
        run_dir.mkdir(parents=True)
        try:
            result = monitor_live_positions_once(
                exchange=exchange,
                contract=contract,
                state_path=args.state,
                now=now,
                submit_orders=args.submit_orders,
            )
        except Exception as exc:
            result = {
                "schema": "strict_r3_kraken_live_position_monitor_v1",
                "mode": "live" if args.submit_orders else "no-order-rehearsal",
                "observed_at": now.isoformat(),
                "status": "failed_closed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "exchange_write_calls": 0,
            }
        atomic_json(run_dir / "run_manifest.json", result)
        print(json.dumps(result, default=str), flush=True)
        if interval is None or args.now:
            if result.get("status") == "failed_closed":
                raise RuntimeError(str(result["error"]))
            break
        next_run += interval
        delay = next_run - time.monotonic()
        if delay < 0.0:
            next_run = time.monotonic()
            delay = 0.0
        time.sleep(delay)


if __name__ == "__main__":
    main()
