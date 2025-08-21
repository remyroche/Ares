#!/usr/bin/env python3
"""
Compare aggregated trades formats between MEXC and Binance to ensure compatibility.
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from src.utils.logger import system_logger
import asyncio
import sys
import pandas as pd

from exchange.factory import ExchangeFactory
from src.utils.error_handler import handle_errors
from src.utils.warning_symbols import error, failed, missing, warning

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = system_logger.getChild("AggTradesFormatComparator")

@handle_errors(
    exceptions=(Exception,),
    default_return=False,
    context="compare_agg_trades_formats",
)
async def compare_agg_trades_formats(
    symbol: str = "BTCUSDT",
    lookback_hours: int = 24
) -> bool:
    """
    Compare aggregated trades formats between MEXC and Binance.

    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        lookback_hours: Number of hours to look back

    Returns:
        bool: True if formats match, False otherwise
    """
    try:
        logger.info(f"🔍 Comparing aggregated trades formats for {symbol}")

        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=lookback_hours)

        start_time_ms = int(start_time.timestamp() * 1000)
        end_time_ms = int(end_time.timestamp() * 1000)

        logger.info(f"📅 Time range: {start_time} to {end_time}")

        # Download from both exchanges
        exchanges = {
            "binance": ExchangeFactory.get_exchange("binance"),
            "mexc": ExchangeFactory.get_exchange("mexc"),
        }

        results = {}

        for exchange_name, exchange in exchanges.items():
            logger.info(f"📥 Downloading from {exchange_name.upper()}...")

            try:
                trades = await exchange.get_historical_agg_trades(
                    symbol, start_time_ms=start_time_ms,
                    end_time_ms=end_time_ms, limit=100,
                )

                if trades:
                    df = pd.DataFrame(trades)
                    results[exchange_name] = df
                    logger.info(
                        f"✅ Downloaded {len(trades)} trades from {exchange_name.upper()}",
                    )
                else:
                    print(warning(f"⚠️ No trades received from {exchange_name.upper()}"))
                    results[exchange_name] = pd.DataFrame()

            except Exception as e:
                logger.exception(
                    f"❌ Error downloading from {exchange_name.upper()}: {e}",
                )
                results[exchange_name] = pd.DataFrame()

        # Compare formats
        logger.info("🔍 Comparing data formats...")

        # Check if we have data from both exchanges
        if results["binance"].empty or results["mexc"].empty:
            print(missing("❌ Missing data from one or both exchanges"))
            return False

        # Get column information
        binance_cols = list(results["binance"].columns)
        mexc_cols = list(results["mexc"].columns)

        logger.info(f"📋 Binance columns: {binance_cols}")
        logger.info(f"📋 MEXC columns: {mexc_cols}")

        # Check if columns match
        if set(binance_cols) == set(mexc_cols):
            logger.info("✅ Column names match between exchanges")
        else:
            print(error("❌ Column names don't match"))
            print(error("   Binance only: {set(binance_cols) - set(mexc_cols)}"))
            print(error("   MEXC only: {set(mexc_cols) - set(binance_cols)}"))
            return False

        # Check data types
        logger.info("🔍 Comparing data types...")

        for col in binance_cols:
            if col in mexc_cols:
                binance_dtype = results["binance"][col].dtype
                mexc_dtype = results["mexc"][col].dtype

                if binance_dtype == mexc_dtype:
                    logger.info(f"✅ Column '{col}': {binance_dtype} (both exchanges)")
                else:
                    logger.warning(
                        f"⚠️ Column '{col}': Binance={binance_dtype}, MEXC={mexc_dtype}",
                    )

        # Check data ranges
        logger.info("🔍 Comparing data ranges...")

        for col in ["p", "q", "T"]:  # price = quantity, timestamp
            if col in binance_cols and col in mexc_cols:
                binance_min = results["binance"][col].min()
                binance_max = results["binance"][col].max()
                mexc_min = results["mexc"][col].min()
                mexc_max = results["mexc"][col].max()

                logger.info(f"📊 Column '{col}' ranges:")
                logger.info(f"   Binance: {binance_min} to {binance_max}")
                logger.info(f"   MEXC: {mexc_min} to {mexc_max}")

        # Display sample data
        logger.info("📋 Sample data comparison:")

        for exchange_name, df in results.items():
            if not df.empty:
                logger.info(f"\n{exchange_name.upper()} sample:")
                logger.info(df.head(3).to_string())

        # Verify specific format requirements
        logger.info("🔍 Verifying Binance format compatibility...")

        # Check required columns
        required_cols = ["a", "p", "q", "T", "m", "f", "l"]
        missing_in_binance = [col for col in required_cols if col not in binance_cols]
        missing_in_mexc = [col for col in required_cols if col not in mexc_cols]

        if missing_in_binance:
            print(missing("❌ Binance missing columns: {missing_in_binance}"))
            return False

        if missing_in_mexc:
            print(missing("❌ MEXC missing columns: {missing_in_mexc}"))
            return False

        logger.info("✅ All required columns present in both exchanges")

        # Check data quality
        logger.info("🔍 Checking data quality...")

        for exchange_name, df in results.items():
            if not df.empty:
                null_counts = df.isnull().sum()
                logger.info(f"\n{exchange_name.upper()} null counts:")
                for col, count in null_counts.items():
                    if count > 0:
                        print(warning("   {col}: {count} null values"))
                    else:
                        logger.info(f"   {col}: {count} null values")

        logger.info("🎉 Format comparison completed successfully!")
        logger.info(
            "✅ MEXC aggregated trades format is compatible with Binance format",
        )

        return True

    except Exception:
        print(error("❌ Error comparing formats: {e}"))
        return False

async def main():
    """Main function to run the comparison script."""

    parser = argparse.ArgumentParser(description="Compare aggregated trades formats")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Number of hours to look back",
    )

    args = parser.parse_args()

    success = await compare_agg_trades_formats(
        symbol=args.symbol, lookback_hours = args.hours,
    )

    if success:
        logger.info("✅ Format comparison completed successfully!")
        sys.exit(0)
    else:
        print(failed("❌ Format comparison failed!"))
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
