#!/usr/bin/env python3
"""Fail-closed preflight for a successor strict-R3 exact-reserve bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd


DEFAULT_RESERVE_DAYS = 28


def _utc(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _bounds(
    paths: Sequence[Path], columns: list[str], *, identity: str = "candidate_id",
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not paths:
        raise ValueError("at least one input path is required")
    read_columns = list(dict.fromkeys([identity, *columns]))
    pieces = [pd.read_parquet(path, columns=read_columns) for path in paths]
    frame = pd.concat(pieces, ignore_index=True)
    raw_rows = len(frame)
    for column in columns:
        if column.endswith("_ts") or column == "__decision_ts__":
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if frame[identity].isna().any():
        raise ValueError(f"readiness input has null {identity}")
    conflicts = frame.loc[frame[identity].duplicated(False)].copy()
    if len(conflicts):
        comparable = [column for column in read_columns if column != identity]
        conflicting_ids = (
            conflicts.groupby(identity, sort=False)[comparable]
            .nunique(dropna=False).gt(1).any(axis=1)
        )
        if conflicting_ids.any():
            raise ValueError(
                "readiness input fragments conflict on duplicate identities: "
                f"{conflicting_ids.loc[conflicting_ids].index[0]}"
            )
    frame = frame.drop_duplicates(subset=[identity], keep="last").reset_index(drop=True)
    return frame, {
        "paths": [str(path) for path in paths],
        "rows": int(len(frame)),
        "raw_rows_before_union_deduplication": int(raw_rows),
        "unique_decision_hours": int(frame["__decision_ts__"].nunique()),
        "decision_min": frame["__decision_ts__"].min().isoformat(),
        "decision_max": frame["__decision_ts__"].max().isoformat(),
    }


def _hourly_calendar_audit(
    timestamps: pd.Series, *, start: pd.Timestamp, end: pd.Timestamp,
) -> dict[str, object]:
    expected = pd.date_range(start, end - pd.Timedelta(hours=1), freq="h", tz="UTC")
    observed = pd.DatetimeIndex(
        pd.to_datetime(timestamps, utc=True, errors="coerce").dropna().unique(),
    ).sort_values()
    observed = observed[(observed >= start) & (observed < end)]
    missing = expected.difference(observed)
    days = pd.date_range(start.normalize(), (end - pd.Timedelta(hours=1)).normalize(), freq="D", tz="UTC")
    observed_days = pd.DatetimeIndex(observed.normalize().unique())
    missing_days = days.difference(observed_days)
    coverage = float(len(expected.difference(missing)) / len(expected)) if len(expected) else 0.0
    return {
        "expected_hours": int(len(expected)),
        "observed_hours": int(len(expected) - len(missing)),
        "coverage": coverage,
        "missing_hours": int(len(missing)),
        "first_missing_hours": [value.isoformat() for value in missing[:24]],
        "expected_days": int(len(days)),
        "missing_days": int(len(missing_days)),
        "first_missing_days": [value.isoformat() for value in missing_days[:10]],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--source-panel", type=Path, action="append", required=True)
    parser.add_argument("--prequential-ledger", type=Path, required=True)
    parser.add_argument("--policy-outcomes", type=Path, action="append", required=True)
    parser.add_argument(
        "--reserve-days",
        type=int,
        default=DEFAULT_RESERVE_DAYS,
        help="Physical same-model calibration reserve; canonical strict-R3 uses 28 days.",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
        [args.prequential_ledger], ["__decision_ts__", "stack_is_prequential"],
    )
    policy, policy_audit = _bounds(
        args.policy_outcomes,
        ["__decision_ts__", "policy_label_available_ts", "policy_path_valid"],
    )
    policy_audit["label_available_max"] = policy["policy_label_available_ts"].max().isoformat()
    reserve_source = source["__decision_ts__"].between(
        reserve_start, cutoff, inclusive="left",
    )
    policy_valid_and_resolved = (
        policy["policy_path_valid"].fillna(False).astype(bool)
        & policy["policy_label_available_ts"].lt(cutoff)
    )
    reserve_policy = policy["__decision_ts__"].between(
        reserve_start, cutoff, inclusive="left",
    ) & policy_valid_and_resolved
    source_calendar = _hourly_calendar_audit(
        source.loc[source["__decision_ts__"].between(reserve_start, cutoff, inclusive="left"), "__decision_ts__"],
        start=reserve_start, end=cutoff,
    )
    resolved_calendar = _hourly_calendar_audit(
        policy.loc[
            policy_valid_and_resolved
            & policy["__decision_ts__"].between(
                reserve_start, required_last_resolved_decision + pd.Timedelta(hours=1),
                inclusive="left",
            ),
            "__decision_ts__",
        ],
        start=reserve_start,
        end=required_last_resolved_decision + pd.Timedelta(hours=1),
    )
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
            policy.loc[policy_valid_and_resolved, "__decision_ts__"].max()
            >= required_last_resolved_decision
        ),
        "last_reserve_label_is_resolved": bool(
            policy.loc[
                policy_valid_and_resolved
                & policy["__decision_ts__"].eq(required_last_resolved_decision),
                "policy_label_available_ts",
            ].max()
            >= required_label_available
        ),
        "reserve_source_nonempty": bool(reserve_source.any()),
        "reserve_policy_nonempty": bool(reserve_policy.any()),
        # One sparse hour may be absent at an immutable activation boundary,
        # but no calendar day may disappear and the reserve must otherwise be
        # effectively complete.  This prevents a short recent fragment from
        # masquerading as a physical 28-day reserve.
        "source_calendar_at_least_99_5pct": bool(source_calendar["coverage"] >= 0.995),
        "source_calendar_has_every_day": bool(source_calendar["missing_days"] == 0),
        "resolved_calendar_at_least_99_5pct": bool(resolved_calendar["coverage"] >= 0.995),
        "resolved_calendar_has_every_day": bool(resolved_calendar["missing_days"] == 0),
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
        "source_calendar": source_calendar,
        "resolved_policy_calendar": resolved_calendar,
        "checks": checks,
        "ready": bool(all(checks.values())),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
