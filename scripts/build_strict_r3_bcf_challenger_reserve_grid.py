#!/usr/bin/env python3
"""Create the immutable target-free hourly grid for an August BCF reserve.

This helper deliberately has no market-data, label, scoring, or exchange
dependencies.  It fixes candidate identity to the preserved 170-symbol
universe before any feature availability or resolved-outcome filtering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _utc_hour(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    if stamp != stamp.floor("h"):
        raise ValueError("start and end must be exact UTC hourly decision timestamps")
    return stamp


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe-manifest", type=Path, required=True)
    parser.add_argument("--start-decision", required=True)
    parser.add_argument("--end-decision", required=True,
                        help="Inclusive final decision hour.")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.out_dir.exists():
        raise FileExistsError(f"immutable target-free reserve grid exists: {args.out_dir}")
    manifest = json.loads(args.universe_manifest.read_text())
    source_map = dict(manifest.get("source_map") or {})
    symbols = [str(symbol) for symbol in source_map]
    if len(symbols) != 170 or len(set(symbols)) != len(symbols):
        raise ValueError("challenger reserve requires the preserved unique 170-symbol universe")
    start = _utc_hour(args.start_decision)
    end = _utc_hour(args.end_decision)
    if end < start:
        raise ValueError("end precedes start")

    rows: list[dict[str, object]] = []
    for decision in pd.date_range(start, end, freq="h", tz="UTC"):
        signal = decision - pd.Timedelta(hours=1)
        signal_label = signal.strftime("%Y-%m-%dT%H:%M:%SZ")
        for symbol in symbols:
            rows.append({
                "candidate_id": f"{symbol}|long|{signal_label}",
                "__ts__": signal,
                "__decision_ts__": decision,
                "__symbol__": symbol,
                "side_name": "long",
            })
    grid = pd.DataFrame(rows)
    if grid["candidate_id"].duplicated().any():
        raise AssertionError("target-free grid created duplicate candidate identities")
    args.out_dir.mkdir(parents=True)
    path = args.out_dir / "candidates.parquet"
    grid.to_parquet(path, index=False, compression="zstd")
    receipt = {
        "schema": "strict_r3_bcf_challenger_samebundle_reserve_grid_v1",
        "status": "complete",
        "mode": "target_free_before_features_and_outcomes",
        "side": "long",
        "start_decision": start.isoformat(),
        "end_decision": end.isoformat(),
        "hours": int(grid["__decision_ts__"].nunique()),
        "universe_rows": len(symbols),
        "candidate_rows": len(grid),
        "candidate_id_pattern": "{symbol}|long|{signal_timestamp_utc}",
        "universe_manifest": str(args.universe_manifest),
        "universe_manifest_sha256": _sha(args.universe_manifest),
        "source_map_sha256": str(manifest.get("source_map_sha256") or ""),
        "outcomes_consumed": [],
        "exchange_calls": 0,
        "order_submission_enabled": False,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
