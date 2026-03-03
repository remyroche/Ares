#!/usr/bin/env python3
"""
Simple 15m universe sync - only USDT since we already have it.
"""

import sys
import pandas as pd
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from extreme_price_movements.universe import refresh_margin_universe_daily
from extreme_price_movements.hf_data_loader import bulk_sync_15m_universe
from extreme_price_movements.utils import tprint


def main():
    tprint("Loading margin universe...")
    try:
        cache = refresh_margin_universe_daily(None, quotes=("USDT",))
        symbols = list(cache.symbols) if cache else []
    except Exception as e:
        tprint(f"WARNING: Could not refresh margin universe ({e})")
        symbols = []

    if not symbols:
        tprint("ERROR: No symbols found. Exiting.")
        sys.exit(1)

    tprint(f"Found {len(symbols)} USDT symbols")
    tprint("Syncing only missing USDT data (skip_existing=True)...")
    
    # Run bulk sync for USDT only
    since_ts = pd.Timestamp("2023-01-01", tz="UTC")
    until_ts = pd.Timestamp.now(tz="UTC")
    
    results = bulk_sync_15m_universe(
        symbols=symbols,
        since_ts=since_ts,
        until_ts=until_ts,
        quotes=("USDT",),  # Only USDT
        skip_existing=True,
    )

    # Summary
    from collections import Counter
    counts = Counter(results.values())
    tprint(f"\n=== Download Summary ===")
    for status, count in sorted(counts.items()):
        tprint(f"  {status}: {count}")


if __name__ == "__main__":
    main()
