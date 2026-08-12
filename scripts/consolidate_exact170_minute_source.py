#!/usr/bin/env python3
"""Consolidate the fragmented Kraken minute source for exact 170 replay."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def _one_symbol(args: tuple[str, str, str]) -> tuple[str, int]:
    symbol, source_root_s, output_root_s = args
    source_root = Path(source_root_s)
    output_root = Path(output_root_s)
    start = pd.Timestamp("2025-12-01", tz="UTC")
    end = pd.Timestamp("2026-08-02", tz="UTC")
    written = 0
    source_symbol = source_root / f"symbol={symbol}"
    if not source_symbol.exists():
        return symbol, 0
    for year in (2025, 2026):
            lo = max(start, pd.Timestamp(f"{year}-01-01", tz="UTC"))
            hi = min(end, pd.Timestamp(f"{year + 1}-01-01", tz="UTC"))
            if lo >= hi:
                continue
            source_year = source_symbol / f"year={year}"
            if not source_year.exists():
                continue
            table = ds.dataset(source_year, format="parquet").to_table(columns=["ts", "open", "high", "low", "close"], use_threads=True)
            frame = table.to_pandas()
            frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
            frame = frame.loc[frame["ts"].ge(lo) & frame["ts"].lt(hi)].drop_duplicates("ts", keep="last").sort_values("ts")
            if frame.empty:
                continue
            # The label loader uses the final two hyphen-delimited stem tokens
            # as immutable epoch bounds for pruning.
            first = int(frame["ts"].iloc[0].timestamp())
            last = int(frame["ts"].iloc[-1].timestamp()) + 60
            target = output_root / f"symbol={symbol}" / f"year={year}" / f"part-consolidated-{first}-{last}.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), target, compression="zstd")
            written += 1
    return symbol, written


def run(universe_file: Path, source_root: Path, output_root: Path, workers: int) -> None:
    symbols = pd.read_csv(universe_file)["symbol"].dropna().astype(str).str.replace("/", "_", regex=False).tolist()
    output_root.mkdir(parents=True, exist_ok=True)
    jobs = [(s, str(source_root), str(output_root)) for s in symbols]
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, (symbol, written) in enumerate(pool.map(_one_symbol, jobs), 1):
            if i % 10 == 0 or i == len(jobs):
                print({"symbols_done": i, "symbols_total": len(symbols), "last_symbol": symbol, "files_written": written}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--universe-file", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    run(args.universe_file, args.source_root, args.output_root, args.workers)
