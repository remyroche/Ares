#!/usr/bin/env python3
"""
Download aggregated trades from MEXC with the same format as Binance.
This script ensures compatibility with existing data processing pipelines.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from exchange.factory import ExchangeFactory
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    warning,
    critical,
    problem,
    failed,
    invalid,
    missing,
    timeout,
    connection_error,
    validation_error,
    initialization_error,
    execution_error,
)

logger = system_logger.getChild("MEXCAggTradesDownloader")


@handle_errors(
    exceptions=(Exception,),
    default_return=False,
    context="download_mexc_agg_trades",
)
async def download_mexc_agg_trades(
    symbol: str = "BTCUSDT",
    lookback_days: int = 30,
    output_dir: str = "data",
) -> bool:
    """
    Download aggregated trades from MEXC with Binance-compatible format.

    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        lookback_days: Number of days to look back
        output_dir: Output directory for CSV files

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"🚀 Starting MEXC aggregated trades download for {symbol}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize MEXC exchange
        exchange = ExchangeFactory.get_exchange("mexc")

        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)

        start_time_ms = int(start_time.timestamp() * 1000)
        end_time_ms = int(end_time.timestamp() * 1000)

        logger.info(f"📅 Time range: {start_time} to {end_time}")
        logger.info(f"🔢 Timestamps: {start_time_ms} to {end_time_ms}")

        # Download aggregated trades
        logger.info("📥 Downloading aggregated trades from MEXC...")

        trades = await exchange.get_historical_agg_trades(
            symbol=symbol,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            limit=1000,
        )

        if not trades:
            print(warning("⚠️ No aggregated trades received from MEXC")))
            return False

        logger.info(f"✅ Downloaded {len(trades)} aggregated trades from MEXC")

        # Convert to DataFrame with Binance-compatible format
        df = pd.DataFrame(trades)

        # Ensure we have the correct columns (Binance format: a, p, q, T, m, f, l)
        expected_columns = ["a", "p", "q", "T", "m", "f", "l"]
        missing_columns = [col for col in expected_columns if col not in df.columns]

        if missing_columns:
            print(missing("⚠️ Missing columns in MEXC data: {missing_columns}")))
            # Add missing columns with default values
            for col in missing_columns:
                df[col] = 0

        # Reorder columns to match Binance format
        df = df[expected_columns]

        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["T"], unit="ms")
        df.set_index("timestamp", inplace=True)

        # Convert numeric columns
        numeric_cols = ["p", "q", "a", "f", "l"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

        # Convert boolean column
        df["m"] = df["m"].astype(bool)

        logger.info(f"📊 DataFrame created with {len(df)} rows")
        logger.info(f"📋 Columns: {list(df.columns)}")
        logger.info(f"📈 Data range: {df.index.min()} to {df.index.max()}")

        # Save to CSV file
        filename = f"{output_dir}/agg_trades_{symbol}_mexc.csv"
        df.to_csv(filename)

        file_size = os.path.getsize(filename)
        logger.info(f"💾 Saved to '{filename}'")
        logger.info(f"📁 File size: {file_size:,} bytes")
        logger.info(f"📊 Total records: {len(df)}")

        # Display sample data
        logger.info("📋 Sample data:")
        logger.info(df.head().to_string())

        # Verify format compatibility
        logger.info("🔍 Verifying Binance format compatibility...")

        # Check if all required columns are present
        if all(col in df.columns for col in expected_columns):
            logger.info("✅ All required columns present")
        else:
            print(missing("❌ Missing required columns")))
            return False

        # Check data types
        if df["p"].dtype in ["float64", "float32"]:
            logger.info("✅ Price column is numeric")
        else:
            print(warning("⚠️ Price column is not numeric")))

        if df["q"].dtype in ["float64", "float32"]:
            logger.info("✅ Quantity column is numeric")
        else:
            print(warning("⚠️ Quantity column is not numeric")))

        if df["m"].dtype == "bool":
            logger.info("✅ Maker flag column is boolean")
        else:
            print(warning("⚠️ Maker flag column is not boolean")))

        logger.info("🎉 MEXC aggregated trades download completed successfully!")
        return True

    except Exception as e:
        print(error("❌ Error downloading MEXC aggregated trades: {e}")))
        return False


async def main():
    """Main function to run the download script."""
    import argparse

    parser = argparse.ArgumentParser(description="Download MEXC aggregated trades")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to look back",
    )
    parser.add_argument("--output", default="data", help="Output directory")

    args = parser.parse_args()

    success = await download_mexc_agg_trades(
        symbol=args.symbol,
        lookback_days=args.days,
        output_dir=args.output,
    )

    if success:
        logger.info("✅ MEXC aggregated trades download completed successfully!")
        sys.exit(0)
    else:
        print(failed("❌ MEXC aggregated trades download failed!")))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
