#!/usr/bin/env python3
"""Validate an L2 reconstruction against a normalized Tardis snapshot_5 pilot.

This is intentionally a research receipt, not an input producer.  Both
inputs remain immutable, and the comparison always uses the last complete L2
state known at or before each snapshot local timestamp.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections.abc import Iterator
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.execution.tardis_book import IncrementalL2Book, validate_snapshot5_top_of_book  # noqa: E402


def _normalize(frame: pd.DataFrame, *, timestamp: str, bid: str, ask: str) -> pd.DataFrame:
    required = {timestamp, bid, ask}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"input lacks declared columns: {sorted(missing)}")
    return frame.rename(columns={timestamp: "local_timestamp", bid: "best_bid", ask: "best_ask"})[
        ["local_timestamp", "best_bid", "best_ask"]
    ]


def _raw_epoch_us(value: object) -> int:
    """Validate raw Tardis microsecond timestamps without Pandas conversion."""
    value_int = int(float(value))
    if abs(value_int) < 100_000_000_000_000:
        raise ValueError(f"expected raw Tardis microseconds, got {value!r}")
    return value_int


def _iter_raw_messages(path: Path) -> Iterator[list[dict[str, str]]]:
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamp", "local_timestamp", "is_snapshot", "side", "price", "amount"}
        if reader.fieldnames is None or required.difference(reader.fieldnames):
            raise ValueError(f"{path} is not a complete raw incremental_book_L2 file")
        current: str | None = None
        rows: list[dict[str, str]] = []
        for row in reader:
            local = row["local_timestamp"]
            if current is None:
                current = local
            elif local != current:
                yield rows
                rows = []
                current = local
            rows.append(row)
        if rows:
            yield rows


def _validate_raw_streams(
    *,
    incremental: Path,
    snapshots: Path,
    sample_stride: int,
    tolerance_bps: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Compare raw book states with raw ``book_snapshot_5`` without a tick dump.

    For each snapshot this consumes only complete L2 messages with a
    ``local_timestamp`` no later than that snapshot.  It is therefore both a
    memory-bounded validation and a direct test of the causal reconstruction
    contract.  Sampling changes only what is persisted, never the book walk.
    """
    if sample_stride < 1:
        raise ValueError("sample_stride must be positive")
    book = IncrementalL2Book()
    messages = _iter_raw_messages(incremental)
    next_message = next(messages, None)
    last_update = None
    last_state_local_us: int | None = None
    output: list[dict[str, object]] = []
    total_snapshots = 0
    matched_snapshots = 0
    with gzip.open(snapshots, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"local_timestamp", "bids[0].price", "asks[0].price"}
        if reader.fieldnames is None or required.difference(reader.fieldnames):
            raise ValueError(f"{snapshots} is not a raw book_snapshot_5 file")
        for row in reader:
            total_snapshots += 1
            snapshot_us = _raw_epoch_us(row["local_timestamp"])
            while next_message is not None and _raw_epoch_us(next_message[0]["local_timestamp"]) <= snapshot_us:
                update = book.apply_records(next_message, materialize=False, preserve_timestamps=False)
                if update is not None:
                    last_update = update
                    last_state_local_us = _raw_epoch_us(next_message[0]["local_timestamp"])
                next_message = next(messages, None)
            if total_snapshots % sample_stride:
                continue
            try:
                snapshot_bid = float(row["bids[0].price"])
                snapshot_ask = float(row["asks[0].price"])
            except (TypeError, ValueError):
                continue
            if last_update is None or not last_update.valid:
                output.append({
                    "snapshot_local_timestamp_us": snapshot_us,
                    "matched": False,
                    "within_tolerance": False,
                })
                continue
            matched_snapshots += 1
            bid_delta_bps = (last_update.best_bid / snapshot_bid - 1.0) * 10_000.0
            ask_delta_bps = (last_update.best_ask / snapshot_ask - 1.0) * 10_000.0
            output.append({
                "snapshot_local_timestamp_us": snapshot_us,
                "state_local_timestamp_us": last_state_local_us,
                "matched": True,
                "state_best_bid": last_update.best_bid,
                "state_best_ask": last_update.best_ask,
                "snapshot_best_bid": snapshot_bid,
                "snapshot_best_ask": snapshot_ask,
                "bid_delta_bps": bid_delta_bps,
                "ask_delta_bps": ask_delta_bps,
                "within_tolerance": abs(bid_delta_bps) <= tolerance_bps and abs(ask_delta_bps) <= tolerance_bps,
            })
    return pd.DataFrame.from_records(output), {
        "total_snapshots": total_snapshots,
        "matched_snapshots": matched_snapshots,
        "persisted_sample_rows": len(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstructed", type=Path, help="Detailed execution states parquet")
    parser.add_argument("--snapshots", type=Path, help="Normalized snapshot_5 parquet or CSV")
    parser.add_argument("--raw-incremental", type=Path, help="Immutable raw incremental_book_L2 gzip")
    parser.add_argument("--raw-snapshot5", type=Path, help="Immutable raw book_snapshot_5 gzip")
    parser.add_argument("--sample-stride", type=int, default=100, help="Persist every Nth raw snapshot after a full causal walk")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reconstructed-timestamp", default="available_ts")
    parser.add_argument("--snapshot-timestamp", default="local_timestamp")
    parser.add_argument("--bid-column", default="best_bid")
    parser.add_argument("--ask-column", default="best_ask")
    parser.add_argument("--tolerance-bps", type=float, default=.5)
    args = parser.parse_args()

    raw_mode = args.raw_incremental is not None or args.raw_snapshot5 is not None
    if raw_mode:
        if args.raw_incremental is None or args.raw_snapshot5 is None:
            raise ValueError("raw validation requires both --raw-incremental and --raw-snapshot5")
        result, raw_audit = _validate_raw_streams(
            incremental=args.raw_incremental,
            snapshots=args.raw_snapshot5,
            sample_stride=int(args.sample_stride),
            tolerance_bps=float(args.tolerance_bps),
        )
    else:
        if args.reconstructed is None or args.snapshots is None:
            raise ValueError("normalized validation requires --reconstructed and --snapshots")
        reconstructed = pd.read_parquet(args.reconstructed)
        reader = pd.read_parquet if args.snapshots.suffix == ".parquet" else pd.read_csv
        snapshots = reader(args.snapshots)
        result = validate_snapshot5_top_of_book(
            _normalize(reconstructed, timestamp=args.reconstructed_timestamp, bid=args.bid_column, ask=args.ask_column),
            _normalize(snapshots, timestamp=args.snapshot_timestamp, bid=args.bid_column, ask=args.ask_column),
            tolerance_bps=float(args.tolerance_bps),
        )
        raw_audit = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.out, index=False)
    if "matched" in result.columns:
        matched_count = int(result["matched"].fillna(False).sum())
    else:
        matched_count = int(result["matched_state_ts"].notna().sum()) if not result.empty else 0
    receipt = {
        "schema": "ares.tardis_snapshot5_reconstruction_audit.v1",
        "rows": int(len(result)),
        "matched": matched_count,
        "within_tolerance": int(result["within_tolerance"].fillna(False).sum()) if not result.empty else 0,
        "tolerance_bps": float(args.tolerance_bps),
        "raw_mode": raw_mode,
        **raw_audit,
        "causality": "each snapshot is matched only to a complete reconstructed state at or before snapshot local timestamp",
    }
    args.out.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
