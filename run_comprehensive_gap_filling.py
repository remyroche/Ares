#!/usr/bin/env python3
"""
Comprehensive Gap Filling Script

Runs gap filling on all available klines, futures, and aggtrades data files.
"""

import asyncio
import sys
import os
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, '/Users/remyroche/Documents/Ares')

from src.utils.data.quality.data_cleaning import DataCleaner
from src.utils.logger import system_logger

logger = system_logger.getChild("ComprehensiveGapFilling")

def find_data_files(data_cache_path: str = "data_cache"):
    """Find all data files that might need gap filling."""
    data_cache = Path(data_cache_path)
    data_files = []

    if not data_cache.exists():
        logger.error(f"Data cache path does not exist: {data_cache}")
        return data_files

    # Find all parquet files
    for parquet_file in data_cache.rglob("*.parquet"):
        filename = parquet_file.name.lower()

        # Identify data type from filename
        if filename.startswith("klines_"):
            data_type = "klines"
        elif filename.startswith("futures_"):
            data_type = "futures"
        elif filename.startswith("aggtrades_"):
            data_type = "aggtrades"
        else:
            continue

        # Extract symbol and exchange from filename
        try:
            parts = filename.split("_")
            if len(parts) >= 3:
                exchange = parts[1]
                symbol = parts[2]
                data_files.append({
                    'path': parquet_file,
                    'data_type': data_type,
                    'exchange': exchange.upper(),
                    'symbol': symbol.upper(),
                    'filename': filename
                })
        except Exception as e:
            logger.warning(f"Could not parse filename {filename}: {e}")
            continue

    return data_files

def extract_timeframe_from_filename(filename: str) -> str:
    """Extract timeframe from filename."""
    if "_1m" in filename:
        return "1m"
    elif "_5m" in filename:
        return "5m"
    elif "_15m" in filename:
        return "15m"
    elif "_1h" in filename:
        return "1h"
    elif "_4h" in filename:
        return "4h"
    elif "_1d" in filename:
        return "1d"
    else:
        return "1m"  # default

async def run_gap_filling_on_file(file_info: dict) -> bool:
    """Run gap filling on a single data file."""
    file_path = file_info['path']
    data_type = file_info['data_type']
    exchange = file_info['exchange']
    symbol = file_info['symbol']
    filename = file_info['filename']

    logger.info(f"🔍 Processing {filename}")
    logger.info(f"   Type: {data_type}, Exchange: {exchange}, Symbol: {symbol}")

    try:
        # Read the data
        if data_type == "klines":
            # For klines, we need to determine timeframe
            timeframe = extract_timeframe_from_filename(filename)
            logger.info(f"   Timeframe: {timeframe}")
        else:
            timeframe = None

        # Read data using safe operations
        from src.utils.common_operations import safe_read_parquet
        data = safe_read_parquet(file_path)

        if data is None or data.empty:
            logger.warning(f"⚠️ No data in {filename}")
            return False

        logger.info(f"   Loaded {len(data)} rows")

        # Initialize DataCleaner with appropriate data type
        cleaner = DataCleaner(data_type=data_type)

        # Run data cleaning with gap filling
        cleaned_data = await cleaner.clean_dataframe(
            data,
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe if timeframe else "1m",
            handle_missing_values=True
        )

        if cleaned_data is None:
            logger.error(f"❌ Failed to clean data for {filename}")
            return False

        # Check results
        original_rows = len(data)
        cleaned_rows = len(cleaned_data)

        logger.info(f"✅ Gap filling completed for {filename}")
        logger.info(f"   Original rows: {original_rows}")
        logger.info(f"   Final rows: {cleaned_rows}")
        logger.info(f"   Rows added: {cleaned_rows - original_rows}")

        # Get gap report
        gap_report = cleaner.get_gap_report(cleaned_data, 'timestamp')
        logger.info(f"   Final gaps: {gap_report['total_gaps']}")

        if gap_report['total_gaps'] == 0:
            logger.info("   🎉 All gaps filled!")
        else:
            logger.warning(f"   ⚠️ {gap_report['total_gaps']} gaps remaining")

        return True

    except Exception as e:
        logger.error(f"❌ Error processing {filename}: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main function to run comprehensive gap filling."""
    logger.info("🚀 Starting comprehensive gap filling...")

    # Find all data files
    data_files = find_data_files()

    if not data_files:
        logger.warning("⚠️ No data files found for gap filling")
        return

    logger.info(f"📁 Found {len(data_files)} data files to process:")

    for i, file_info in enumerate(data_files, 1):
        logger.info(f"   {i}. {file_info['filename']}")

    # Process each file
    success_count = 0
    total_files = len(data_files)

    for i, file_info in enumerate(data_files, 1):
        logger.info(f"\n🔧 Processing file {i}/{total_files}: {file_info['filename']}")

        try:
            if await run_gap_filling_on_file(file_info):
                success_count += 1
            else:
                logger.error(f"❌ Failed to process {file_info['filename']}")
        except Exception as e:
            logger.error(f"❌ Exception processing {file_info['filename']}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    logger.info(f"\n📊 Gap filling summary:")
    logger.info(f"   Total files: {total_files}")
    logger.info(f"   Successfully processed: {success_count}")
    logger.info(f"   Failed: {total_files - success_count}")

    if success_count == total_files:
        logger.info("🎉 All data files processed successfully!")
    else:
        logger.warning(f"⚠️ {total_files - success_count} files had issues")

if __name__ == "__main__":
    asyncio.run(main())
