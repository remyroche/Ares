#!/usr/bin/env python3
"""
Direct 15m universe sync using hf_data_loader functions.
Checks both main data directory and HF data directory to avoid redundant downloads.
"""

import sys
import pandas as pd
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from extreme_price_movements.universe import refresh_margin_universe_daily
from extreme_price_movements.hf_data_loader import bulk_sync_15m_universe
from extreme_price_movements.utils import tprint
from extreme_price_movements.config import CFG


def check_existing_data(symbol: str, quotes: tuple = ("USDT", "USDC", "BUSD")) -> dict:
    """Check if data exists in either main data directory or HF directory."""
    existing = {}
    
    # Check main data directory
    from extreme_price_movements.config import CFG
    ohlcv_dir = Path(CFG["data_root"]) / "ohlcv"
    
    for quote in quotes:
        # Main directory check (symbol=BASE_QUOTE format)
        main_symbol_dir = ohlcv_dir / f"symbol={symbol}_{quote}"
        existing[quote] = main_symbol_dir.exists() and any(main_symbol_dir.iterdir())
        
        # HF directory check (basequote_15m.parquet format)
        if not existing[quote]:
            from extreme_price_movements.hf_data_loader import HF_DATA_DIR
            hf_path = HF_DATA_DIR / f"{symbol.lower()}{quote.lower()}_15m.parquet"
            existing[quote] = hf_path.exists()
            
    return existing


def main():
    tprint("Loading margin universe...")
    try:
        cache = refresh_margin_universe_daily(None, quotes=("USDT", "USDC", "BUSD"))
        symbols = list(cache.symbols) if cache else []
    except Exception as e:
        tprint(f"WARNING: Could not refresh margin universe ({e}), using local OHLCV store symbols")
        # Extract base assets from filesystem
        ohlcv_dir = Path(CFG["data_root"]) / "ohlcv"
        symbol_dirs = [d for d in ohlcv_dir.iterdir() if d.is_dir() and d.name.startswith("symbol=")]
        
        base_assets = set()
        for symbol_dir in symbol_dirs:
            symbol_name = symbol_dir.name.replace("symbol=", "")
            if "_" in symbol_name:
                base = symbol_name.split("_")[0]
                if base not in ["USDT", "USDC", "BUSD"]:
                    base_assets.add(base)
        symbols = list(base_assets)

    if not symbols:
        tprint("ERROR: No symbols found. Exiting.")
        sys.exit(1)

    tprint(f"Found {len(symbols)} base assets")
    
    # Check existing data for a sample to avoid redundant downloads
    quotes = ("USDT", "USDC", "BUSD")
    sample_symbols = symbols[:5]  # Check first 5 as sample
    
    tprint("Checking existing data coverage...")
    has_usdt = has_usdc = has_busd = False
    
    for symbol in sample_symbols:
        existing = check_existing_data(symbol, quotes)
        if existing["USDT"]:
            has_usdt = True
        if existing["USDC"]:
            has_usdc = True
        if existing["BUSD"]:
            has_busd = True
            
    # Determine which quotes to download
    quotes_to_sync = []
    if has_usdt:
        quotes_to_sync.append("USDT")
    if has_usdc:
        quotes_to_sync.append("USDC") 
    if has_busd:
        quotes_to_sync.append("BUSD")
        
    if not quotes_to_sync:
        quotes_to_sync = ["USDT"]  # Default to USDT
        
    tprint(f"Syncing quotes: {quotes_to_sync}")
    
    # Run bulk sync
    since_ts = pd.Timestamp("2023-01-01", tz="UTC")
    until_ts = pd.Timestamp.now(tz="UTC")
    
    results = bulk_sync_15m_universe(
        symbols=symbols,
        since_ts=since_ts,
        until_ts=until_ts,
        quotes=tuple(quotes_to_sync),
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
