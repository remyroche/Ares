#!/usr/bin/env python3
"""Advance one P8U E2/H4 successor position without exchange authority.

This is the separately named, no-order monitor core required before a future
exchange-writing monitor may be authorized.  It atomically materialises the
completed 15-minute H4 decision into successor position state, then optionally
advances already-completed one-minute bars under that persisted next-interval
authority.  It writes a new immutable snapshot; it never edits its input state
and cannot call an exchange or submit an order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_e2_h4_continuation import persist_h4_next_interval_decision
from extreme_price_movements.inference.p8u_e2_h4_live_parity import P8UE2H4LiveParityBundle
from extreme_price_movements.inference.p8u_e2_h4_rich_policy import advance_h4_aware_rich_policy_position


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_bars(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "timestamp" in frame:
        frame = frame.set_index("timestamp")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True, errors="raise"))
    if frame.index.duplicated().any():
        raise ValueError("monitor bar input duplicates timestamps")
    return frame.sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--position", type=Path, required=True, help="JSON object for one already-open position")
    parser.add_argument("--state-decision", required=True, help="completed 15-minute UTC boundary")
    parser.add_argument("--expectation-reference", type=Path, required=True)
    parser.add_argument("--bars-15m", type=Path, required=True)
    parser.add_argument("--parent-policy", type=Path, required=True)
    parser.add_argument("--bars-1m", type=Path, help="optional completed minute bars after state boundary")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("successor monitor output must be immutable")
    position = json.loads(args.position.resolve().read_text(encoding="utf-8"))
    if not isinstance(position, dict):
        raise ValueError("position snapshot must be a JSON object")
    bundle = P8UE2H4LiveParityBundle.load(args.bundle.resolve())
    reference = pd.read_parquet(args.expectation_reference.resolve())
    bars_15m = _load_bars(args.bars_15m.resolve())
    successor_state, h4_scored = persist_h4_next_interval_decision(
        position=position,
        bars_15m=bars_15m,
        state_decision_ts=args.state_decision,
        expectation_reference=reference,
        bundle=bundle,
    )
    h4_state_ts = pd.Timestamp(successor_state["p8u_h4_last_state_decision_ts"])
    minute_trace: list[dict[str, Any]] = []
    exit_proposal: dict[str, Any] | None = None
    if args.bars_1m is not None:
        from extreme_price_movements.inference.strict_r3_live_execution import _policy_payload, _rich_policy_params

        minute = _load_bars(args.bars_1m.resolve())
        # The minute ending exactly at the H4 decision boundary belongs to the
        # already-completed state bar.  It is intentionally excluded.
        minute = minute.loc[(minute.index + pd.Timedelta(minutes=1)).gt(h4_state_ts)]
        policy_payload = _policy_payload(args.parent_policy.resolve())
        params, median = _rich_policy_params(policy_payload)
        advanced = advance_h4_aware_rich_policy_position(
            position=successor_state,
            bars=minute,
            parent_params=params,
            median_atr_fraction=median,
        )
        successor_state = advanced["state"]
        minute_trace = advanced["h4_minute_trace"]
        exit_proposal = advanced["exit"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "next_position_state.json").write_text(
        json.dumps(successor_state, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    h4_scored.to_parquet(output / "h4_completed_state_prediction.parquet", index=False, compression="zstd")
    pd.DataFrame(minute_trace).to_parquet(output / "h4_completed_minute_trace.parquet", index=False, compression="zstd")
    receipt = {
        "schema": "strict_r3_p8u_e2_h4_noorder_position_monitor_v1",
        "status": "persisted_h4_next_interval_no_order",
        "order_submission": False,
        "exchange_or_order_submission_called": False,
        "bundle_manifest_sha256": bundle.manifest_sha256,
        "position_snapshot_sha256": _sha256(args.position.resolve()),
        "parent_policy_sha256": _sha256(args.parent_policy.resolve()),
        "state_decision_ts": str(h4_state_ts),
        "h4_state_hash": successor_state["p8u_h4_last_state_hash"],
        "h4_prediction_bps": successor_state["p8u_h4_prediction_bps"],
        "h4_active": successor_state["p8u_h4_active"],
        "effective_from": successor_state["p8u_h4_effective_from"],
        "effective_until": successor_state["p8u_h4_effective_until"],
        "completed_minute_bars_advanced": int(len(minute_trace)),
        "exit_proposal": exit_proposal,
        "outcome_columns_consumed": [],
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
