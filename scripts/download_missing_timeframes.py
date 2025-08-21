#!/usr/bin/env python3
"""
Download Missing Timeframes Script

This script downloads the missing timeframe data (5m, 15m, 30m) for the
multi-timeframe HMM ensemble system.
"""

from pathlib import Path
from src.training.steps.data_downloader import download_all_data_with_consolidation
from src.utils.logger import system_logger
from typing import Any
import argparse
import asyncio
import sys

from src.config import CONFIG

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = system_logger.getChild("MissingTimeframesDownloader")


class MissingTimeframesDownloader:
    """Downloads missing timeframe data for multi-timeframe HMM ensemble."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.required_timeframes = ["5m", "15m", "30m"]
        self.symbol = "ETHUSDT"
        self.exchange = "BINANCE"
        self.data_dir = Path("data")

    async def check_existing_data(self) -> dict[str, bool]:
        """Check which timeframes already have data."""
        logger.info("🔍 Checking existing timeframe data...")

        existing_data = {}
        for timeframe in self.required_timeframes:
            csv_file = self.data_dir / f"ETHUSDT_{timeframe}.csv"
            existing_data[timeframe] = csv_file.exists()

            status = "✅" if existing_data[timeframe] else "❌"
            logger.info(
                f"  {timeframe}: {status} {'Available' if existing_data[timeframe] else 'Missing'}",
            )

        return existing_data

    async def download_timeframe(self, timeframe: str) -> bool:
        """Download data for a specific timeframe."""
        logger.info(f"📥 Downloading {timeframe} data for {self.symbol}...")

        try:
            success = await download_all_data_with_consolidation(
                symbol=self.symbol, exchange_name=self.exchange,
                interval=timeframe
            )

            if success:
                logger.info(f"✅ Successfully downloaded {timeframe} data")
                return True
            logger.error(f"❌ Failed to download {timeframe} data")
            return False

        except Exception as e:
            logger.exception(f"💥 Error downloading {timeframe} data: {e}")
            return False

    async def download_missing_timeframes(self) -> dict[str, bool]:
        """Download all missing timeframe data."""
        logger.info("🚀 Starting download of missing timeframe data...")

        # Check existing data
        existing_data = await self.check_existing_data()

        # Identify missing timeframes
        missing_timeframes = [tf for tf, exists in existing_data.items() if not exists]

        if not missing_timeframes:
            logger.info("✅ All required timeframes already have data!")
            return {tf: True for tf in self.required_timeframes}

        logger.info(f"📋 Missing timeframes: {', '.join(missing_timeframes)}")

        # Download missing timeframes
        download_results = {}
        for timeframe in missing_timeframes:
            success = await self.download_timeframe(timeframe)
            download_results[timeframe] = success

            # Add small delay between downloads to respect rate limits
            if timeframe != missing_timeframes[-1]:  # Not the last one
                await asyncio.sleep(2)

        # Combine results
        return {**existing_data, **download_results}

    def verify_downloads(self, results: dict[str, bool]) -> bool:
        """Verify that all downloads were successful."""
        logger.info("🔍 Verifying downloads...")

        all_successful = True
        for timeframe , success in results.items():
            status = "✅" if success else "❌"
            logger.info(f"  {timeframe}: {status} {'Success' if success else 'Failed'}")

            if not success:
                all_successful = False

        if all_successful:
            logger.info("✅ All timeframes successfully downloaded!")
        else:
            logger.warning("⚠️  Some timeframes failed to download")

        return all_successful

    def print_summary(self, results: dict[str, bool]) -> None:
        """Print a summary of the download results."""
        print("\n" + "=" * 80)
        print("📊 TIMEFRAME DATA DOWNLOAD SUMMARY")
        print("=" * 80)

        print(f"\n🎯 Target Timeframes: {', '.join(self.required_timeframes)}")
        print(f"📈 Symbol: {self.symbol}")
        print(f"🏢 Exchange: {self.exchange}")

        successful = [tf for tf, success in results.items() if success]
        failed = [tf for tf, success in results.items() if not success]

        print(f"\n✅ Successful Downloads: {len(successful)}")
        if successful:
            print(f"   {', '.join(successful)}")

        if failed:
            print(f"\n❌ Failed Downloads: {len(failed)}")
            print(f"   {', '.join(failed)}")

        print(f"\n📁 Data Directory: {self.data_dir.absolute()}")

        if not failed:
            print("\n🎉 ALL TIMEFRAMES READY FOR MULTI-TIMEFRAME HMM ENSEMBLE!")
        else:
            print("\n⚠️  SOME TIMEFRAMES NEED ATTENTION")

        print("=" * 80)


async def main():
    """Main function to run the download process."""
    parser = argparse.ArgumentParser(
        description="Download missing timeframe data for multi-timeframe HMM ensemble",
    )
    parser.add_argument("--symbol", type=str, default="ETHUSDT", help="Trading symbol")
    parser.add_argument("--exchange", type=str, default="BINANCE", help="Exchange name")
    parser.add_argument(
        "--timeframes",
        type=str,
        nargs="+",
        default=["5m", "15m", "30m"],
        help="Timeframes to download",
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = CONFIG if hasattr(CONFIG, "get") else {}

        # Create downloader
        downloader = MissingTimeframesDownloader(config)
        downloader.symbol = args.symbol
        downloader.exchange = args.exchange
        downloader.required_timeframes = args.timeframes

        # Download missing timeframes
        results = await downloader.download_missing_timeframes()

        # Verify downloads
        all_successful = downloader.verify_downloads(results)

        # Print summary
        downloader.print_summary(results)

        return all_successful

    except Exception as e:
        logger.exception(f"💥 Download process failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
