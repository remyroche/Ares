#!/usr/bin/env python3
"""Materialise the complete target-free P8U long candidate universe.

The P8U Router must evaluate every frozen-universe symbol at each completed
source hour.  This utility deliberately does *not* qualify a row using an H12
path, label validity, future bars, or post-decision execution result.  Later
source/feature checks may reject a row with an explicit reason, but that must
never alter the candidate identities used for contemporaneous calculations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _timestamp_sha256(timestamps: tuple[pd.Timestamp, ...]) -> str:
    payload = "\n".join(stamp.isoformat() for stamp in sorted(timestamps)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--timestamps",
        nargs="+",
        help="Explicit completed source-hour timestamps (UTC).",
    )
    selection.add_argument(
        "--start",
        help="Inclusive UTC start of a completed source-hour range.",
    )
    parser.add_argument(
        "--end-exclusive",
        help="Exclusive UTC end for --start source-index selection.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.timestamps is not None and args.end_exclusive is not None:
        parser.error("--end-exclusive is valid only with --start")
    if args.start is not None and args.end_exclusive is None:
        parser.error("--start requires --end-exclusive")
    if args.out_dir.exists():
        raise FileExistsError(f"immutable candidate output already exists: {args.out_dir}")
    source = joblib.load(args.source_state)
    if not isinstance(source, Mapping) or not isinstance(source.get("panel"), Mapping):
        raise ValueError("P8U source state has no target-free primitive panel")
    symbols = tuple(map(str, source.get("symbols") or ()))
    close = source["panel"].get("close")
    if len(symbols) != 160 or len(set(symbols)) != len(symbols) or not isinstance(close, pd.DataFrame):
        raise ValueError("P8U candidate builder requires the frozen 160-symbol source universe")
    source_index = pd.DatetimeIndex(close.index)
    if not source_index.is_unique or not source_index.is_monotonic_increasing:
        raise ValueError("P8U candidate source index must be unique and increasing")
    if args.timestamps is not None:
        timestamps = tuple(_utc(value) for value in args.timestamps)
        selection_mode = "explicit_timestamps"
        selection_start = None
        selection_end_exclusive = None
    else:
        selection_start = _utc(args.start)
        selection_end_exclusive = _utc(args.end_exclusive)
        if selection_start >= selection_end_exclusive:
            raise ValueError("--start must be earlier than --end-exclusive")
        timestamps = tuple(source_index[(source_index >= selection_start) & (source_index < selection_end_exclusive)])
        if not timestamps:
            raise ValueError("source-index range selected no completed candidate hours")
        selection_mode = "source_index_range"
    if len(timestamps) != len(set(timestamps)):
        raise ValueError("candidate timestamps must be unique")
    missing = [stamp.isoformat() for stamp in timestamps if stamp not in source_index]
    if missing:
        raise ValueError(f"source state lacks completed candidate hour(s): {missing[:3]}")
    frames: list[pd.DataFrame] = []
    for stamp in sorted(timestamps):
        # Candidate identity is intentionally independent of the current
        # primitive value.  Even a source-incomplete symbol remains present
        # and can later be fail-closed with a causal rejection reason.
        frame = pd.DataFrame({"__symbol__": symbols})
        frame["__ts__"] = stamp
        frame["__decision_ts__"] = stamp + pd.Timedelta(hours=1)
        frame["side_name"] = "long"
        stamp_id = stamp.strftime("%Y-%m-%dT%H:%M:%SZ")
        frame["candidate_id"] = frame["__symbol__"] + "|long|" + stamp_id
        frames.append(frame.loc[:, ["candidate_id", "__decision_ts__", "side_name", "__ts__", "__symbol__"]])
    candidates = pd.concat(frames, ignore_index=True)
    if candidates["candidate_id"].duplicated().any() or len(candidates) != len(symbols) * len(timestamps):
        raise AssertionError("P8U full-universe candidate materialisation changed identity cardinality")
    args.out_dir.mkdir(parents=True)
    candidates.to_parquet(args.out_dir / "candidates.parquet", index=False, compression="zstd")
    receipt = {
        "schema": "strict_r3_p8u_target_free_candidates_v2",
        "status": "pass_complete_frozen_universe",
        "source_state": str(args.source_state.resolve()),
        "source_state_sha256": _sha256(args.source_state),
        "selection_mode": selection_mode,
        "selection_start": None if selection_start is None else selection_start.isoformat(),
        "selection_end_exclusive": None if selection_end_exclusive is None else selection_end_exclusive.isoformat(),
        "selected_timestamps": [stamp.isoformat() for stamp in sorted(timestamps)] if selection_mode == "explicit_timestamps" else None,
        "selected_timestamps_count": len(timestamps),
        "selected_timestamps_sha256": _timestamp_sha256(timestamps),
        "first_selected_timestamp": min(timestamps).isoformat(),
        "last_selected_timestamp": max(timestamps).isoformat(),
        "symbols": len(symbols),
        "candidate_rows": len(candidates),
        "candidate_population": "all frozen-universe symbol x source timestamp x long",
        "future_path_or_outcome_filter_applied": False,
        "outcome_columns_consumed": [],
    }
    _atomic_json(args.out_dir / "receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
