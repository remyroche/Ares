#!/usr/bin/env python3
"""
Standalone script to bulk-download / update 15m OHLCV data for the full
symbol universe, covering USDT, USDC, and BUSD quote variants.

Usage:
    python3 download_15m_universe.py
    python3 download_15m_universe.py --since 2024-01-01 --quotes USDT USDC
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from extreme_price_movements.universe import refresh_margin_universe_daily
from extreme_price_movements.hf_data_loader import bulk_sync_15m_universe
from extreme_price_movements.utils import tprint


def main():
    parser = argparse.ArgumentParser(description="Bulk-sync 15m OHLCV universe")
    parser.add_argument("--since", default="2023-01-01", help="Start date for backfill (YYYY-MM-DD)")
    parser.add_argument("--quotes", nargs="+", default=["USDT", "USDC", "BUSD"], help="Quote currencies to cover")
    parser.add_argument("--no-skip", action="store_true", help="Re-download even if already cached")
    parser.add_argument("--symbol-order", choices=["alpha_asc", "alpha_desc"], default="alpha_asc", help="Symbol traversal order")
    parser.add_argument("--partition-count", type=int, default=1, help="Total number of parallel partitions")
    parser.add_argument("--partition-id", type=int, default=0, help="Zero-based partition id")
    args = parser.parse_args()

    since_ts = pd.Timestamp(args.since, tz="UTC")
    until_ts = pd.Timestamp.now(tz="UTC")

    tprint(f"Loading margin universe...")
    try:
        cache = refresh_margin_universe_daily(None, quotes=tuple(args.quotes))
        symbols = list(cache.symbols) if cache else []
    except Exception as e:
        tprint(f"WARNING: Could not refresh margin universe ({e}), using local OHLCV store symbols")
        from extreme_price_movements.data_store import PartitionedOHLCVStore
        from extreme_price_movements.config import CFG
        store = PartitionedOHLCVStore(root_dir=CFG["data_root"], timeframe=CFG["timeframe"])
        symbols = store.list_symbols()

    if not symbols:
        tprint("ERROR: No symbols found. Exiting.")
        sys.exit(1)

    tprint(f"Universe: {len(symbols)} symbols. Syncing 15m data ({args.since} → now) for quotes: {args.quotes}")
    tprint(f"Order={args.symbol_order}, partition={args.partition_id}/{args.partition_count}")

    results = bulk_sync_15m_universe(
        symbols=symbols,
        since_ts=since_ts,
        until_ts=until_ts,
        quotes=tuple(args.quotes),
        skip_existing=not args.no_skip,
        symbol_order=args.symbol_order,
        partition_count=args.partition_count,
        partition_id=args.partition_id,
    )

    # Summary
    from collections import Counter
    counts = Counter(results.values())
    tprint(f"\n=== Download Summary ===")
    for status, count in sorted(counts.items()):
        tprint(f"  {status}: {count}")


if __name__ == "__main__":
    main()
