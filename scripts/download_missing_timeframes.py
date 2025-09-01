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
import project_root = Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = system_logger.getChild("MissingTimeframesDownloader")


class MissingTimeframesDownloader:
    """Downloads missing timeframe data for multi-timeframe HMM ensemble."""

    def __init__(self, config: dict[str, Any]):
    pass
    pass
        self.config = config
        self.required_timeframes = ["5m", "15m", "30m"]
        self.symbol = "ETHUSDT"
        self.exchange = "BINANCE"
        self.data_dir = Path("data")

    async def check_existing_data(self) -> dict[str, bool]:
        """Check which timeframes already have data."""
        logger.info("🔍 Checking existing timeframe data...")

        existing_data: dict[str, bool] = {}
        for timeframe in self.required_timeframes:
    pass
    pass
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
                symbol=self.symbol,
                exchange_name=self.exchange,
                interval=timeframe,
    except Exception as e:
        pass
    except Exception as e:
        pass
            )

            if success:
    pass
    pass
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
    pass
    pass
            logger.info("✅ All required timeframes already have data!")
            return {tf: True for tf in self.required_timeframes}

        logger.info(f"📋 Missing timeframes: {', '.join(missing_timeframes)}")

        # Download missing timeframes
        download_results: dict[str, bool] = {}
        for timeframe in missing_timeframes:
    pass
    pass
            success = await self.download_timeframe(timeframe)
            download_results[timeframe] = success

            # Add small delay between downloads to respect rate limits
            if timeframe != missing_timeframes[-1]:  # Not the last one
                await asyncio.sleep(2)

        # Combine results
        return {**existing_data, **download_results}

    def verify_downloads(self, results: dict[str, bool]) -> bool:
    pass
    pass
        """Verify that all downloads were successful."""
        logger.info("🔍 Verifying downloads...")

        all_successful = True
        for timeframe, success in results.items():
    pass
    pass
            status = "✅" if success else "❌"
            logger.info(f"  {timeframe}: {status} {'Success' if success else 'Failed'}")
            if not success:
    pass
    pass
                all_successful = False

        if all_successful:
    pass
    pass
            logger.info("✅ All timeframes successfully downloaded!")
        else:
            logger.warning("⚠️  Some timeframes failed to download")

        return all_successful

    def print_summary(self, results: dict[str, bool]) -> None:
    pass
    pass
        """Print a summary of the download results."""
        print("\\\n=== Download Summary ===")
        for timeframe, success in sorted(results.items()):
    pass
    pass
            status = "SUCCESS" if success else "FAILED"
            print(f"{timeframe}: {status}")


def main() -> None:
    pass
    pass
    parser = argparse.ArgumentParser(description="Download missing timeframes")
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--exchange", default="BINANCE")
    args = parser.parse_args()

    cfg = dict(CONFIG)
    cfg["symbol"] = args.symbol
    cfg["exchange"] = args.exchange

    downloader = MissingTimeframesDownloader(cfg)

    results = asyncio.run(downloader.download_missing_timeframes())
    downloader.print_summary(results)
    downloader.verify_downloads(results)


if __name__ == "__main__":
    pass
    pass
    main()
