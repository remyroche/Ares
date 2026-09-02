#!/usr/bin/env python3
"""Minute exit monitor for the sealed P8U E2 + H4 successor.

It keeps every parent rich-policy safety mechanism: immediate native stop,
completed-minute hard-stop / trailing / smooth-protection / timeout handling,
and exchange reconciliation.  At each completed 15-minute boundary, it first
persists one target-free H4 decision for the *next* interval.  If that state or
its source is unavailable, the monitor explicitly retains the parent policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.data_fetcher import make_exchange
from extreme_price_movements.inference.p8u_e2_h4_continuation import (
    persist_h4_next_interval_decision,
)
from extreme_price_movements.inference.p8u_e2_h4_live_parity import P8UE2H4LiveParityBundle
from extreme_price_movements.inference.p8u_e2_h4_rich_policy import (
    advance_h4_aware_rich_policy_position,
)
from extreme_price_movements.inference.strict_r3_live_execution import (
    atomic_json,
    live_state_lock,
    load_state,
    monitor_live_positions_once,
    utc,
)
from scripts.run_strict_r3_p8u_e2_h4_live_gateway import (
    _generic_contract,
    _load_contract,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _root_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_reference(contract: Mapping[str, Any]) -> pd.DataFrame:
    descriptor = dict(contract.get("h4_expectation_reference") or {})
    path_value, expected = descriptor.get("path"), descriptor.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected, str):
        raise ValueError("successor monitor lacks a hash-bound H4 expectation reference")
    path = _root_path(path_value).resolve()
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError("H4 expectation reference hash mismatch")
    reference = pd.read_parquet(path)
    forbidden = {"policy_net_bps", "policy_gross_bps", "activation50_advantage_bps"}
    if forbidden.intersection(reference.columns):
        raise ValueError("H4 expectation reference is not target-free")
    reference["state_decision_ts"] = pd.to_datetime(reference["state_decision_ts"], utc=True, errors="raise")
    return reference


def _check_monitor_authority(contract: Mapping[str, Any]) -> None:
    activation = json.loads(Path(str(contract["__activation_path__"])).read_text())
    runtime = dict(activation.get("p8u_e2_h4_position_monitor_runtime") or {})
    current = Path(__file__).resolve()
    if runtime.get("path") != str(current.relative_to(ROOT)) or runtime.get("sha256") != _sha256(current):
        raise ValueError("successor position monitor is not sealed in activation")
    if activation.get("h4_expectation_reference_sha256") != str(contract["h4_expectation_reference"]["sha256"]):
        raise ValueError("successor activation does not bind its H4 reference")


def _has_positions(state_path: Path) -> bool:
    return state_path.is_file() and bool(json.loads(state_path.read_text()).get("positions"))


def _refresh_h4_state(*, exchange: Any, state_path: Path, bundle: P8UE2H4LiveParityBundle, reference: pd.DataFrame, timestamp: pd.Timestamp) -> list[dict[str, object]]:
    """Persist one next-interval H4 decision per open position when due."""
    # Import only in the actual position-monitor process.  This loader is the
    # established completed-bar source for live rich-policy monitoring.
    from extreme_price_movements.inference.run_inference import _load_live_policy_bars

    boundary = utc(timestamp).floor("15min")
    actions: list[dict[str, object]] = []
    with live_state_lock(state_path):
        state = load_state(state_path, decision_ts=timestamp)
        changed = False
        positions = list(state.get("positions") or [])
        for index, position_raw in enumerate(positions):
            if not isinstance(position_raw, Mapping):
                raise ValueError("successor state contains malformed position")
            position = dict(position_raw)
            entry_ts = utc(position.get("entry_fill_ts") or position["entry_ts"])
            previous = position.get("p8u_h4_last_state_decision_ts")
            previous_ts = utc(previous) if previous else None
            if boundary <= entry_ts or (previous_ts is not None and previous_ts >= boundary):
                continue
            symbol = str(position["exchange_symbol"])
            try:
                # 30h includes the 104 completed 15m bars required by the
                # exact state/VWAP contract, without touching any future bar.
                bars = _load_live_policy_bars(
                    cfg={"execution_account": "perps", "exchange": "krakenfutures"},
                    exchange=exchange, symbol=symbol, timeframe="15m",
                    start=boundary - pd.Timedelta(hours=30), end=boundary,
                )
                updated, scored = persist_h4_next_interval_decision(
                    position=position, bars_15m=bars, state_decision_ts=boundary,
                    expectation_reference=reference, bundle=bundle,
                )
                updated["p8u_h4_state_status"] = "active_or_parent_next_interval"
                positions[index] = updated
                actions.append({
                    "action": "h4_next_interval_decision",
                    "candidate_id": str(position["candidate_id"]),
                    "state_decision_ts": boundary.isoformat(),
                    "prediction_bps": float(updated["p8u_h4_prediction_bps"]),
                    "active": bool(updated["p8u_h4_active"]),
                    "state_hash": str(updated["p8u_h4_last_state_hash"]),
                })
            except Exception as exc:
                # H4 has no authority to weaken the already active parent
                # policy.  An unavailable/invalid state therefore becomes an
                # explicit next-interval parent fallback, never a guessed H4
                # value and never a monitor crash that would blind exits.
                position.update({
                    "p8u_h4_last_state_decision_ts": boundary.isoformat(),
                    "p8u_h4_active": False,
                    "p8u_h4_effective_from": boundary.isoformat(),
                    "p8u_h4_effective_until": (boundary + pd.Timedelta(minutes=15)).isoformat(),
                    "p8u_h4_state_status": f"parent_fallback:{type(exc).__name__}",
                })
                positions[index] = position
                actions.append({"action": "h4_parent_fallback", "candidate_id": str(position["candidate_id"]), "state_decision_ts": boundary.isoformat(), "reason": f"{type(exc).__name__}:{exc}"})
            changed = True
        if changed:
            state["positions"] = positions
            state["p8u_e2_h4_last_state_refresh"] = boundary.isoformat()
            atomic_json(state_path, state)
    return actions


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
    contract, _ = _load_contract(args.gateway_contract)
    _check_monitor_authority(contract)
    generic = _generic_contract(contract)
    bundle = P8UE2H4LiveParityBundle.load(Path(str(contract["__e2_h4_root__"])))
    reference = _load_reference(contract)
    state_path = args.state.resolve()
    while True:
        now = pd.Timestamp.now(tz="UTC")
        if _has_positions(state_path):
            exchange = make_exchange("perps")
            h4_actions = _refresh_h4_state(exchange=exchange, state_path=state_path, bundle=bundle, reference=reference, timestamp=now)
            result = monitor_live_positions_once(
                exchange=exchange, contract=generic, state_path=state_path, now=now,
                submit_orders=True, rich_policy_advance=advance_h4_aware_rich_policy_position,
            )
            result["h4_state_actions"] = h4_actions
            print(json.dumps(result, sort_keys=True, default=str), flush=True)
        else:
            print(json.dumps({"status": "idle_no_tracked_e2_h4_positions"}), flush=True)
        if args.once:
            return
        time.sleep(max(5.0, float(args.interval_seconds)))


if __name__ == "__main__":
    main()
