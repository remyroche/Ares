#!/usr/bin/env python3
"""Persistent minute-resolution exit monitor for positions opened by P8U.

It deliberately reuses the proven strict-R3 rich-policy monitor.  It has no
entry or model authority.  Before the first P8U fill it performs no private
account calls, so it cannot interfere with an independently running legacy
monitor.  Once P8U has a tracked position it advances stop, smooth protection,
trailing and timeout on completed one-minute bars; Adaptive Exit remains a
15-minute-only input carried at entry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_strict_r3_p8u_live_gateway import _generic_execution_contract, _load_gateway_contract
from extreme_price_movements.inference.data_fetcher import make_exchange
from extreme_price_movements.inference.strict_r3_live_execution import monitor_live_positions_once


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_monitor_authority(contract: dict[str, object]) -> None:
    activation = json.loads(Path(str(contract["__activation_path__"])).read_text(encoding="utf-8"))
    runtime = dict(activation.get("p8u_position_monitor_runtime") or {})
    current = Path(__file__).resolve()
    if runtime.get("path") != str(current.relative_to(ROOT)) or runtime.get("sha256") != _sha256(current):
        raise ValueError("P8U position-monitor runtime is not sealed in activation")


def _has_tracked_positions(state_path: Path) -> bool:
    if not state_path.exists():
        return False
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    return bool(payload.get("positions"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-contract", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    if bool(args.once) == bool(args.loop):
        parser.error("select exactly one of --once or --loop")
    contract, _ = _load_gateway_contract(args.gateway_contract)
    _check_monitor_authority(contract)
    generic = _generic_execution_contract(contract)
    state_path = args.state.resolve()
    while True:
        if _has_tracked_positions(state_path):
            exchange = make_exchange("perps")
            result = monitor_live_positions_once(
                exchange=exchange,
                contract=generic,
                state_path=state_path,
                now=pd.Timestamp.now(tz="UTC"),
                submit_orders=True,
            )
            print(json.dumps(result, sort_keys=True, default=str), flush=True)
        else:
            print(json.dumps({"status": "idle_no_tracked_p8u_positions"}), flush=True)
        if args.once:
            return
        time.sleep(max(5.0, float(args.interval_seconds)))


if __name__ == "__main__":
    main()
