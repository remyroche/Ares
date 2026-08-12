#!/usr/bin/env python3
"""Fail-closed preflight for a successor strict-R3 exact-reserve bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _utc(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _bounds(path: Path, columns: list[str]) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = pd.read_parquet(path, columns=columns)
    for column in columns:
        if column.endswith("_ts") or column == "__decision_ts__":
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    return frame, {
        "path": str(path),
        "rows": int(len(frame)),
        "decision_min": frame["__decision_ts__"].min().isoformat(),
        "decision_max": frame["__decision_ts__"].max().isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--source-panel", type=Path, required=True)
    parser.add_argument("--prequential-ledger", type=Path, required=True)
    parser.add_argument("--policy-outcomes", type=Path, required=True)
    parser.add_argument("--reserve-days", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"immutable readiness receipt exists: {args.out}")
    cutoff = _utc(args.cutoff)
    reserve_start = cutoff - pd.Timedelta(days=args.reserve_days)
    required_last_source_decision = cutoff - pd.Timedelta(hours=1)
    # H12 labels need not exist for the reserve's final twelve decisions.
    # They are target-free-scored but excluded by label_available_ts < cutoff.
    required_last_resolved_decision = cutoff - pd.Timedelta(hours=13)
    required_label_available = cutoff - pd.Timedelta(hours=1)
    required_prequential_history = reserve_start - pd.Timedelta(hours=13)

    source, source_audit = _bounds(args.source_panel, ["__decision_ts__"])
    prequential, prequential_audit = _bounds(
        args.prequential_ledger, ["__decision_ts__", "stack_is_prequential"],
    )
    policy, policy_audit = _bounds(
        args.policy_outcomes,
        ["__decision_ts__", "policy_label_available_ts", "policy_path_valid"],
    )
    policy_audit["label_available_max"] = policy["policy_label_available_ts"].max().isoformat()
    reserve_source = source["__decision_ts__"].between(
        reserve_start, cutoff, inclusive="left",
    )
    reserve_policy = policy["__decision_ts__"].between(
        reserve_start, cutoff, inclusive="left",
    ) & policy["policy_path_valid"].fillna(False).astype(bool)
    checks = {
        "source_covers_cutoff_minus_one_hour": bool(
            source["__decision_ts__"].max() >= required_last_source_decision
        ),
        "prequential_ledger_covers_training_boundary": bool(
            prequential["__decision_ts__"].max() >= required_prequential_history
        ),
        "prequential_stack_flag_true": bool(
            prequential["stack_is_prequential"].fillna(False).astype(bool).all()
        ),
        "policy_outcomes_cover_last_resolved_decision": bool(
            policy.loc[policy["policy_path_valid"].fillna(False), "__decision_ts__"].max()
            >= required_last_resolved_decision
        ),
        "last_reserve_label_is_resolved": bool(
            policy["policy_label_available_ts"].max() >= required_label_available
        ),
        "reserve_source_nonempty": bool(reserve_source.any()),
        "reserve_policy_nonempty": bool(reserve_policy.any()),
    }
    payload = {
        "schema": "strict_r3_successor_bundle_readiness_v1",
        "cutoff": cutoff.isoformat(),
        "reserve_start": reserve_start.isoformat(),
        "required_last_source_decision": required_last_source_decision.isoformat(),
        "required_last_resolved_decision": required_last_resolved_decision.isoformat(),
        "required_prequential_history": required_prequential_history.isoformat(),
        "required_last_label_available_ts": required_label_available.isoformat(),
        "source": source_audit,
        "prequential": prequential_audit,
        "policy": policy_audit,
        "reserve_source_rows": int(reserve_source.sum()),
        "reserve_valid_policy_rows": int(reserve_policy.sum()),
        "checks": checks,
        "ready": bool(all(checks.values())),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
