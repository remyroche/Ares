#!/usr/bin/env python3
"""Normalize immutable Tardis ``book_snapshot_5`` CSVs for reconstruction QA.

This creates only a compact top-of-book comparison table. It never replaces or
modifies the raw snapshot archive and is not an inference input.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.execution.tardis_book import to_utc_timestamp  # noqa: E402


REQUIRED = {"local_timestamp", "bids[0].price", "asks[0].price"}


def normalize_snapshot5(path: Path, *, symbol: str | None = None, chunksize: int = 500_000) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, compression="gzip", chunksize=int(chunksize)):
        missing = REQUIRED.difference(chunk.columns)
        if missing:
            raise ValueError(f"{path} lacks book_snapshot_5 fields: {sorted(missing)}")
        out = pd.DataFrame({
            "local_timestamp": to_utc_timestamp(chunk["local_timestamp"]),
            "best_bid": pd.to_numeric(chunk["bids[0].price"], errors="coerce"),
            "best_ask": pd.to_numeric(chunk["asks[0].price"], errors="coerce"),
        })
        out["symbol"] = str(symbol) if symbol else chunk.get("symbol", "")
        frames.append(out.dropna(subset=["local_timestamp", "best_bid", "best_ask"]))
    return pd.concat(frames, ignore_index=True, copy=False) if frames else pd.DataFrame(columns=["symbol", "local_timestamp", "best_bid", "best_ask"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--symbol")
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()
    frame = normalize_snapshot5(args.raw, symbol=args.symbol, chunksize=int(args.chunksize))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    print(f"rows={len(frame):,} out={args.out}")


if __name__ == "__main__":
    main()
