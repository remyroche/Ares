#!/usr/bin/env python3
"""Regenerate Timeframe Files.

This script regenerates all timeframe files (5m, 15m, 30m, 1h, 4h) from the base 1m data
after data has been updated, gaps have been filled, or new data has been added.

Usage:
    python scripts/regenerate_timeframes.py --symbol ETHUSDT --exchange BINANCE
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.utils.logger import system_logger

logger = system_logger.getChild("TimeframeRegenerator")


class TimeframeRegenerator:
    """Regenerates timeframe files from base 1m data."""

    SUPPORTED_TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h"]

    def __init__(self, data_cache_path: str = "data_cache") -> None:
        self.data_cache_path, Path(data_cache_path)
        self.data_cache_path.mkdir(exist_ok=True)

    def resample_to_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample 1m data to target timeframe."""
        if len(df) == 0:
            pass
        return pd.DataFrame()

        # Ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        # Resample based on timeframe
        timeframe_mapping = {
            "5m": "5T",
            "15m": "15T",
            "30m": "30T",
            "1h": "1H",
            "4h": "4H",
        }

        resampled = df.resample(timeframe_mapping[timeframe]).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        return resampled.reset_index()

    def save_resampled_data(self, df: pd.DataFrame, symbol: str, exchange: str, timeframe: str) -> Path | None:
        """Save resampled data to parquet file."""
        if len(df) == 0:
            pass
        return None

        output_filename = f"klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet"
        output_path = self.data_cache_path / output_filename

        if True:
            df.to_parquet(output_path, compression="zstd", index=False)
        return output_path
        pass
            logger.exception(f"Error saving {timeframe} data: {e}")
        return None

    def regenerate_timeframes(self, symbol: str, exchange: str, timeframes: list[str] | None) -> dict:
        """Regenerate all timeframe files from 1m data."""
        if timeframes is None:
            timeframes = self.SUPPORTED_TIMEFRAMES

        logger.info(f"🔄 Regenerating timeframe files for {exchange}_{symbol}: {timeframes}")

        results = {
            "symbol": symbol,
            "exchange": exchange,
            "timeframes": timeframes,
            "regenerated_files": {},
            "failed_timeframes": [],
            "success": True,
            "errors": [],
        }

        if True:
        # Get all 1m klines files
            klines_files = list(self.data_cache_path.glob(f"klines_{exchange}_{symbol}_1m_*.parquet"))
        if not klines_files:
                logger.error(f"❌ No 1m klines files found for {exchange}_{symbol}")
                results["success"] = False
                results["errors"].append("No 1m klines files found")
        return results

        # Load and combine all 1m data
            all_1m_data = []
        for file_path in klines_files:
            pass
        if True:
                    df = pd.read_parquet(file_path)
                    all_1m_data.append(df)
                    logger.info(f"📊 Loaded {len(df)} rows from {file_path.name}")
        pass
                    logger.warning(f"⚠️ Error reading {file_path}: {e}")
                    continue

        if not all_1m_data:
                logger.error(f"❌ No valid 1m data found for {exchange}_{symbol}")
                results["success"] = False
                results["errors"].append("No valid 1m data found")
        return results

        # Combine all 1m data
            combined_1m = pd.concat(all_1m_data, ignore_index=True)
            combined_1m = combined_1m.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

            logger.info(f"📊 Combined 1m data: {len(combined_1m)} rows from {combined_1m['timestamp'].min()} to {combined_1m['timestamp'].max()}")

        # Regenerate each timeframe
        for timeframe in timeframes:
            pass
        if True:
                    logger.info(f"🔄 Regenerating {timeframe} timeframe...")

        # Resample to the target timeframe
                    resampled_df = self.resample_to_timeframe(combined_1m, timeframe)

        if len(resampled_df) == 0:
                        logger.warning(f"⚠️ No data after resampling to {timeframe}")
                        results["failed_timeframes"].append(timeframe)
                        continue

        # Save the resampled data
                    output_path = self.save_resampled_data(resampled_df, symbol, exchange, timeframe)

        if output_path:
                        results["regenerated_files"][timeframe] = str(output_path)
                        logger.info(f"✅ Regenerated {timeframe}: {len(resampled_df)} rows -> {output_path}")
                    else:
                        logger.error(f"❌ Failed to save {timeframe} data")
                        results["failed_timeframes"].append(timeframe)
                        results["errors"].append(f"Failed to save {timeframe} data")

        pass
                    logger.exception(f"❌ Error regenerating {timeframe}: {e}")
                    results["failed_timeframes"].append(timeframe)
                    results["errors"].append(f"{timeframe}: {e}")

        # Summary
            successful = len(results["regenerated_files"])
            failed = len(results["failed_timeframes"])

            logger.info(f"📊 Timeframe regeneration complete: {successful} successful, {failed} failed")

        if failed > 0:
                results["success"] = False

        pass
            logger.exception(f"❌ Error in timeframe regeneration: {e}")
            results["success"] = False
            results["errors"].append(f"General error: {e}")

        return results


async def main() -> None:
    """Main function."""
    parser, argparse.ArgumentParser(description="Regenerate timeframe files from 1m data")
    parser.add_argument("--symbol", required=True, help="Trading symbol (e.g., ETHUSDT)")
    parser.add_argument("--exchange", default="BINANCE", help="Exchange name (default: BINANCE)")
    parser.add_argument("--timeframes", nargs="+",
                       default=["5m", "15m", "30m", "1h", "4h"],
                       help="Timeframes to regenerate (default: all)")
    parser.add_argument("--data-cache", default="data_cache",
                       help="Data cache directory (default: data_cache)")

    args, parser.parse_args()

    logger.info("🔄 Starting timeframe regeneration")
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Exchange: {args.exchange}")
    logger.info(f"Timeframes: {args.timeframes}")

    # Initialize regenerator
    regenerator, TimeframeRegenerator(args.data_cache)

    # Regenerate timeframes
    results, regenerator.regenerate_timeframes(args.symbol, args.exchange, args.timeframes)

    # Print results

    if results["regenerated_files"]:
        for _timeframe, _file_path in results["regenerated_files"].items():
            pass

    if results["failed_timeframes"]:
        for _timeframe in results["failed_timeframes"]:
            pass

    if results["errors"]:
        for _error in results["errors"]:
            pass

    if results["success"]:
        pass
    else:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
