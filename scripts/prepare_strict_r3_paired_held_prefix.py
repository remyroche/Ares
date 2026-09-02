#!/usr/bin/env python3
"""Prepare an immutable continuous held prefix for a no-order paired replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _utc(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True, help="inclusive decision timestamp")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    start, end = _utc(args.start), _utc(args.end)
    if end < start:
        raise ValueError("end precedes start")
    candidates = pd.read_parquet(args.candidates)
    candidates["__decision_ts__"] = pd.to_datetime(candidates["__decision_ts__"], utc=True)
    held = candidates.loc[
        candidates["__decision_ts__"].between(start, end, inclusive="both")
    ].copy()
    features = pd.read_parquet(args.features)
    features["candidate_id"] = features["candidate_id"].astype(str)
    held["candidate_id"] = held["candidate_id"].astype(str)
    held_features = features.loc[features["candidate_id"].isin(held["candidate_id"])].copy()
    left, right = set(held["candidate_id"]), set(held_features["candidate_id"])
    if left != right:
        raise ValueError(
            f"candidate/feature identity mismatch: candidates_only={len(left-right)} "
            f"features_only={len(right-left)}"
        )
    if held["candidate_id"].duplicated().any() or held_features["candidate_id"].duplicated().any():
        raise ValueError("candidate identity is not one-to-one")
    observed_hours = pd.DatetimeIndex(sorted(held["__decision_ts__"].unique()))
    expected_hours = pd.date_range(start, end, freq="h", tz="UTC")
    if not observed_hours.equals(expected_hours):
        raise ValueError("held prefix has missing or extra decision timestamps")
    hourly_counts = held.groupby("__decision_ts__", sort=True)["candidate_id"].nunique()
    if not hourly_counts.eq(170).all():
        raise ValueError("held prefix is not the full frozen 170-symbol universe")
    args.out_dir.mkdir(parents=True)
    held.sort_values(["__decision_ts__", "candidate_id"], kind="stable").to_parquet(
        args.out_dir / "held_candidates.parquet", index=False, compression="zstd"
    )
    held_features.sort_values("candidate_id", kind="stable").to_parquet(
        args.out_dir / "held_features.parquet", index=False, compression="zstd"
    )
    manifest = {
        "schema": "strict_r3_paired_held_prefix_v1",
        "outcome_columns_consumed": [],
        "order_submission": False,
        "exchange_calls": 0,
        "start": str(start),
        "end": str(end),
        "rows": int(len(held)),
        "hours": int(len(observed_hours)),
        "symbols_per_hour": int(hourly_counts.iloc[0]),
        "source_candidates": str(args.candidates),
        "source_features": str(args.features),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
