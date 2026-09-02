#!/usr/bin/env python3
"""Audit causal sequence availability for a temporal meta-model contract.

The report identifies exact symbol/day/channel causes of missing windows before
any TCN/CNN fit.  It never fills gaps and is safe to run against an OOS ledger.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.data_store import (  # noqa: E402
    _feature_schema_names,
    read_symbol_features,
)


def _path(root: Path, symbol: str) -> Path:
    return root / f"symbol={str(symbol).replace('/', '_')}.parquet"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--channels-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lookback-bars", type=int, default=16)
    parser.add_argument("--bar-minutes", type=int, default=60)
    parser.add_argument("--max-stale-bars", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.channels_json.read_text())
    channels = sorted(set(sum((list(value.get("channels", [])) for value in source["contracts"].values()), [])))
    frame = pd.read_parquet(args.candidates, columns=["__ts__", "__symbol__"])
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["__ts__"])
    rows: list[dict[str, object]] = []
    step = np.timedelta64(args.bar_minutes, "m")
    for symbol, group in frame.groupby("__symbol__", observed=True):
        path = _path(args.feature_root, str(symbol))
        base = {"symbol": str(symbol), "candidate_rows": int(len(group)), "feature_file": str(path)}
        if not path.exists():
            rows.append({**base, "status": "missing_file", "complete_windows": 0, "coverage": 0.0})
            continue
        names = _feature_schema_names(str(path))
        missing_columns = [name for name in channels if name not in names]
        if missing_columns:
            rows.append({**base, "status": "missing_columns", "missing_columns": ",".join(missing_columns), "complete_windows": 0, "coverage": 0.0})
            continue
        target = group["__ts__"].to_numpy(dtype="datetime64[ns]")
        start = pd.Timestamp(target.min()).tz_localize("UTC") - pd.Timedelta(
            minutes=args.bar_minutes * (args.lookback_bars - 1 + args.max_stale_bars)
        )
        end = pd.Timestamp(target.max()).tz_localize("UTC")
        store = read_symbol_features(
            str(path), columns=channels, start_ts=start, end_ts=end
        )
        if store.empty or any(name not in store.columns for name in channels):
            rows.append({**base, "status": "read_missing_columns", "complete_windows": 0, "coverage": 0.0})
            continue
        timestamps = pd.to_datetime(store.index, utc=True, errors="coerce").to_numpy(dtype="datetime64[ns]")
        ends = pd.Index(timestamps).get_indexer(target)
        offsets = np.arange(args.lookback_bars - 1, -1, -1, dtype=np.int64)
        valid = ends >= args.lookback_bars - 1
        complete = np.zeros(len(group), dtype=bool)
        if valid.any():
            idx = ends[valid, None] - offsets[None, :]
            expected = target[valid, None] - offsets[None, :] * step
            complete[valid] = np.all(timestamps[idx] == expected, axis=1)
        expected_all = target[:, None] - offsets[None, :] * step
        observed_at = np.searchsorted(timestamps, expected_all, side="right") - 1
        bounded = np.all(observed_at >= 0, axis=1)
        bounded_rows = np.flatnonzero(bounded)
        if len(bounded_rows):
            previous = timestamps[observed_at[bounded_rows]]
            age = (expected_all[bounded_rows] - previous) / step
            bounded[bounded_rows] = np.all(age <= args.max_stale_bars, axis=1)
        rows.append({**base, "status": "ok", "complete_windows": int(complete.sum()), "coverage": float(complete.mean()), "bounded_stale_windows": int(bounded.sum()), "bounded_stale_coverage": float(bounded.mean()), "first_ts": str(group["__ts__"].min()), "last_ts": str(group["__ts__"].max())})
    report = pd.DataFrame(rows).sort_values(["coverage", "candidate_rows"], ascending=[True, False])
    report.to_csv(args.output_dir / "sequence_coverage_by_symbol.csv", index=False)
    report["month"] = pd.to_datetime(report.get("last_ts"), utc=True, errors="coerce").dt.strftime("%Y-%m")
    summary = {
        "candidate_rows": int(len(frame)), "symbols": int(frame["__symbol__"].nunique()), "channels": channels,
        "weighted_complete_coverage": float(report["complete_windows"].sum() / max(report["candidate_rows"].sum(), 1)),
        "weighted_bounded_stale_coverage": float(report.get("bounded_stale_windows", pd.Series(dtype=float)).fillna(0).sum() / max(report["candidate_rows"].sum(), 1)),
        "missing_file_symbols": int((report["status"] == "missing_file").sum()),
        "missing_column_symbols": int((report["status"] == "missing_columns").sum()),
        "low_coverage_symbols": int((report["coverage"] < 0.95).sum()),
        "low_bounded_stale_coverage_symbols": int((report.get("bounded_stale_coverage", 0.0).fillna(0.0) < 0.90).sum()),
    }
    (args.output_dir / "sequence_coverage_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
