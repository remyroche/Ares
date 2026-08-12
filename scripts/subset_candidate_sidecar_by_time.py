#!/usr/bin/env python3
"""Create an immutable, identity-preserving time slice of a parquet sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="sidecar.parquet")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    if end <= start:
        raise ValueError("require start < end")
    frame = pd.read_parquet(args.source)
    required = {"candidate_id", "__decision_ts__"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"source lacks identity fields: {missing}")
    timestamp = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    selected = frame.loc[timestamp.ge(start) & timestamp.lt(end)].copy()
    selected["__decision_ts__"] = timestamp.loc[selected.index]
    selected = selected.sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    ).reset_index(drop=True)
    if selected.empty or selected["candidate_id"].duplicated().any():
        raise ValueError("time slice is empty or has duplicate candidate IDs")
    args.out_dir.mkdir(parents=True)
    output = args.out_dir / args.output_name
    selected.to_parquet(output, index=False, compression="zstd")
    manifest = {
        "schema": "immutable_candidate_sidecar_time_slice_v1",
        "source": str(args.source),
        "source_sha256": _sha(args.source),
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "rows": int(len(selected)),
        "columns": list(selected.columns),
        "identity_preserved": True,
        "output": str(output),
        "output_sha256": _sha(output),
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
    )
    print(json.dumps({"event": "complete", **manifest}))


if __name__ == "__main__":
    main()
