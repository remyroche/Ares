#!/usr/bin/env python3
"""Distill compact per-minute Kraken trade aggregates into a small recap.

The input consists of already-materialised per-minute state surfaces, not raw
trade prints.  This producer retains only aggregate quote volume, trade count,
and signed order-flow imbalance plus their causal trailing changes.  It refuses
to persist an individual timestamp, price, size, or trade identifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.execution.liquidity_transition import TRADE_ACTIVITY_COLUMNS  # noqa: E402


KEY_COLUMNS = ("symbol", "state_minute", "available_ts")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.surface_root.rglob("surface.parquet"))
    if not paths:
        raise FileNotFoundError(f"no compact surfaces below {args.surface_root}")
    pieces: list[pd.DataFrame] = []
    sources: list[dict[str, object]] = []
    fields = list(TRADE_ACTIVITY_COLUMNS)
    for path in paths:
        frame = pd.read_parquet(path)
        required = set(KEY_COLUMNS).union(fields)
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} lacks compact activity fields: {sorted(missing)}")
        recap = frame.loc[:, [*KEY_COLUMNS, *fields]].copy()
        recap["state_minute"] = pd.to_datetime(recap["state_minute"], utc=True, errors="coerce")
        recap["activity_available_ts"] = pd.to_datetime(recap.pop("available_ts"), utc=True, errors="coerce")
        if recap["state_minute"].isna().any() or recap["activity_available_ts"].isna().any():
            raise ValueError(f"{path} has invalid activity timestamps")
        pieces.append(recap)
        sources.append({"path": str(path), "rows": int(len(recap)), "sha256": _sha256(path)})
    output = pd.concat(pieces, ignore_index=True, copy=False).sort_values(["symbol", "state_minute"], kind="stable")
    if output.duplicated(["symbol", "state_minute"]).any():
        raise ValueError("input has duplicate compact activity symbol/minute identities")
    # The source state is formed over this minute and its availability must be
    # at or before the following whole-minute decision boundary.  This guards
    # the exact contract relied on by the panel builder.
    deadline = output["state_minute"] + pd.Timedelta(minutes=1)
    if output["activity_available_ts"].gt(deadline).any():
        raise ValueError("an aggregate activity row arrives after its next-minute decision boundary")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    staged = args.out.with_name(f".{args.out.name}.partial")
    output.to_parquet(staged, index=False)
    os.replace(staged, args.out)
    manifest = {
        "schema": "ares.execution_activity_recap.v1",
        "source_root": str(args.surface_root),
        "retention": "per-minute aggregate quote volume, count, and signed flow only; no individual trade prints retained",
        "fields": fields,
        "rows": int(len(output)),
        "symbols": int(output["symbol"].nunique()),
        "dates": sorted(output["state_minute"].dt.date.astype(str).unique().tolist()),
        "availability": "activity_available_ts must be at or before state_minute plus one minute",
        "sources": sources,
        "output": str(args.out),
    }
    args.out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("rows", "symbols", "dates", "output")}, indent=2))


if __name__ == "__main__":
    main()
