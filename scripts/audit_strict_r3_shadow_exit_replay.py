#!/usr/bin/env python3
"""Independently replay one strict-R3 hourly exit transition.

This auditor deliberately starts from the immutable predecessor state and the
frozen 15-minute cache.  It does not consume the run's next state when deriving
expected exits, so it can detect execution-path or policy drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_inference_bundle import StrictR3InferenceBundle
from extreme_price_movements.strict_r3_shadow_portfolio import (
    ShadowPortfolioState,
    advance_shadow_state,
)
from scripts.run_strict_r3_shadow_cycle import _load_state_bars


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _canonical(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate_id", "symbol", "side", "entry_ts", "exit_ts",
        "entry_price", "exit_price", "gross_bps", "cost_bps", "net_bps",
        "gross_notional", "wallet_pnl", "exit_reason",
    ]
    out = frame.reindex(columns=columns).copy()
    for field in ("entry_ts", "exit_ts"):
        out[field] = pd.to_datetime(out[field], utc=True, errors="coerce")
    return out.sort_values(["candidate_id", "exit_ts"], kind="stable").reset_index(drop=True)


def _compare(expected: pd.DataFrame, actual: pd.DataFrame) -> dict[str, object]:
    expected, actual = _canonical(expected), _canonical(actual)
    if expected[["candidate_id", "exit_ts"]].astype(str).to_dict("records") != actual[["candidate_id", "exit_ts"]].astype(str).to_dict("records"):
        raise AssertionError("exit identity or timestamp differs from independent replay")
    exact_fields = ["symbol", "side", "exit_reason"]
    for field in exact_fields:
        if not expected[field].astype(str).equals(actual[field].astype(str)):
            raise AssertionError(f"exit field differs: {field}")
    numeric_fields = [
        "entry_price", "exit_price", "gross_bps", "cost_bps", "net_bps",
        "gross_notional", "wallet_pnl",
    ]
    maximum_delta = 0.0
    for field in numeric_fields:
        lhs = pd.to_numeric(expected[field], errors="coerce").to_numpy(float)
        rhs = pd.to_numeric(actual[field], errors="coerce").to_numpy(float)
        if not np.allclose(lhs, rhs, rtol=0.0, atol=1e-10, equal_nan=True):
            raise AssertionError(f"exit numeric field differs: {field}")
        if len(lhs):
            maximum_delta = max(maximum_delta, float(np.nanmax(np.abs(lhs - rhs))))
    return {
        "expected_exit_rows": int(len(expected)),
        "actual_exit_rows": int(len(actual)),
        "maximum_numeric_delta": maximum_delta,
        "exit_identities": expected[["candidate_id", "symbol", "exit_ts", "exit_reason"]].astype(str).to_dict("records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-bundle", type=Path, required=True)
    parser.add_argument("--previous-run", type=Path, required=True)
    parser.add_argument(
        "--entry-state", type=Path, required=True,
        help=(
            "Immutable actual-live-fill bridge state used by the hourly run. "
            "The auditor rejects a simulated predecessor state."
        ),
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    bundle = StrictR3InferenceBundle.load(args.inference_bundle, root=ROOT)
    run_manifest = json.loads((args.run / "run_manifest.json").read_text())
    decision = _utc(run_manifest["decision_ts"])
    bundle_audit = bundle.validate(decision_ts=decision)
    previous_state_path = args.previous_run / "cycle" / "next_portfolio_state.json"
    entry_payload = json.loads(args.entry_state.read_text())
    provenance = dict(entry_payload.get("bridge_provenance") or {})
    if str(provenance.get("shadow_reference_sha256")) != _sha(previous_state_path):
        raise ValueError("exit entry-state bridge does not descend from predecessor")
    if int(provenance.get("live_execution_state_overlays", -1)) != int(
        provenance.get("matched_positions", -2)
    ):
        raise ValueError("exit entry-state bridge lacks actual-fill overlays")
    if _sha(args.entry_state) != str(run_manifest["hashes"]["portfolio_state"]):
        raise ValueError("exit replay entry state differs from hourly run input")
    state = ShadowPortfolioState.from_payload(entry_payload, expected_as_of_ts=decision)
    policy_path = bundle.path("exit_policy")
    policy_payload = json.loads(policy_path.read_text())
    policy = policy_payload.get("winner", policy_payload)
    symbols = {position.symbol for position in state.open_positions}
    starts = [state.as_of_ts, *[
        position.entry_ts for position in state.open_positions
        if position.entry_ts is not None
    ]]
    bars = _load_state_bars(
        ROOT / str(bundle.payload["runtime"]["policy_bar_root"]),
        symbols,
        start=min(starts),
        end=decision,
    )
    replayed_state, expected_exits = advance_shadow_state(
        state,
        decision_ts=decision,
        bars_by_symbol=bars,
        stop_loss_atr=float(policy["sl_mult"]),
        trailing_activation_atr=float(policy["trailing_activation_mult"]),
        trailing_giveback_atr=float(policy["fixed_trailing_gap_mult"]),
        cost_bps=100.0,
        defer_incomplete_paths=True,
    )
    actual_exits = pd.read_parquet(args.run / "cycle" / "shadow_exits.parquet")
    comparison = _compare(expected_exits, actual_exits)
    deferred = sorted(
        position.symbol for position in replayed_state.open_positions
        if position.next_bar_ts is not None and position.next_bar_ts < decision
    )
    receipt = {
        "schema": "strict_r3_shadow_exit_independent_replay_audit_v1",
        "decision_ts": decision.isoformat(),
        "previous_run": str(args.previous_run),
        "run": str(args.run),
        "inference_bundle_sha256": _sha(args.inference_bundle),
        "exit_policy_sha256": _sha(policy_path),
        "bundle_audit": bundle_audit,
        "predecessor_state_sha256": _sha(previous_state_path),
        "entry_state": str(args.entry_state),
        "entry_state_sha256": _sha(args.entry_state),
        "live_state_sha256": provenance.get("live_state_sha256"),
        "live_execution_state_overlays": int(
            provenance["live_execution_state_overlays"]
        ),
        "bar_symbols": sorted(symbols),
        "deferred_symbols": deferred,
        "cost_bps_once": 100.0,
        "comparison": comparison,
        "checks": {
            "identity_exact": True,
            "timestamps_exact": True,
            "policy_exact": True,
            "numeric_exact_within_1e_10": True,
            "future_bars_absent": True,
            "actual_fill_entry_state": True,
            "entry_state_matches_hourly_run": True,
        },
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "exit_replay_audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
