#!/usr/bin/env python3
"""Correct one live position's policy clock to its verified exchange fill.

This is an immutable state-successor migration for the exceptional case where
an older execution contract stored the decision timestamp rather than the
actual fill timestamp.  It never changes a model, score, admission decision,
order, position amount, stop, or policy geometry.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.strict_r3_live_execution import atomic_json


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--actual-fill-ts", required=True)
    parser.add_argument("--out-state", type=Path, required=True)
    parser.add_argument("--out-receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.out_state.exists() or args.out_receipt.exists():
        raise FileExistsError("policy-clock repair outputs are immutable")

    source = json.loads(args.state.read_text())
    positions = list(source.get("positions") or [])
    matches = [
        index for index, position in enumerate(positions)
        if str(position.get("candidate_id")) == str(args.candidate_id)
    ]
    if len(matches) != 1:
        raise ValueError("candidate id must identify exactly one open position")
    index = matches[0]
    original = copy.deepcopy(positions[index])
    actual_fill = _utc(args.actual_fill_ts)
    recorded_entry = _utc(original["entry_ts"])
    if actual_fill < recorded_entry:
        raise ValueError("verified actual fill cannot predate the recorded entry")
    timeout_hours = int(
        dict(original.get("entry_reporting_context") or {}).get(
            "policy_timeout_hours", 12,
        )
    )
    if timeout_hours <= 0:
        raise ValueError("position lacks a positive canonical policy timeout")
    repaired = copy.deepcopy(source)
    position = dict(repaired["positions"][index])
    position.update({
        "entry_ts": actual_fill.isoformat(),
        "entry_fill_ts": actual_fill.isoformat(),
        "next_bar_ts": (actual_fill.floor("min") + pd.Timedelta(minutes=1)).isoformat(),
        "timeout_ts": (actual_fill + pd.Timedelta(hours=timeout_hours)).isoformat(),
        "late_entry_policy_clock": True,
        "actual_fill_policy_clock": True,
        "policy_clock_repair": {
            "schema": "strict_r3_live_actual_fill_policy_clock_repair_v1",
            "recorded_entry_ts": recorded_entry.isoformat(),
            "verified_exchange_fill_ts": actual_fill.isoformat(),
            "timeout_hours": timeout_hours,
            "reason": "bounded-current-hour execution must not consume pre-fill bars",
        },
    })
    repaired["positions"][index] = position
    if any(
        repaired["positions"][index].get(field) != original.get(field)
        for field in ("candidate_id", "symbol", "exchange_symbol", "amount", "entry_price", "stop_order_id", "stop_price", "policy_sha256")
    ):
        raise AssertionError("policy-clock repair changed protected position fields")
    atomic_json(args.out_state, repaired)
    receipt = {
        "schema": "strict_r3_live_actual_fill_policy_clock_repair_v1",
        "source_state": str(args.state),
        "source_state_sha256": _sha(args.state),
        "output_state": str(args.out_state),
        "output_state_sha256": _sha(args.out_state),
        "candidate_id": str(args.candidate_id),
        "recorded_entry_ts": recorded_entry.isoformat(),
        "verified_exchange_fill_ts": actual_fill.isoformat(),
        "next_complete_minute": position["next_bar_ts"],
        "timeout_ts": position["timeout_ts"],
        "protected_position_fields_preserved": True,
    }
    atomic_json(args.out_receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
